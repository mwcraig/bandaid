"""
Unit tests for the cached Gaia catalog query in :mod:`bandaid.catalog`.

These tests never touch the network: they patch ``bandaid.catalog.Vizier`` with a
``unittest.mock`` stand-in whose ``query_region_async`` answers with a VOTable
of a small in-memory table shaped like the real VizieR ``I/355/gaiadr3`` result
(columns ``Gmag, RA_ICRS, DE_ICRS, pmRA, pmDE``) -- the shared ``gaia_table`` /
``fake_vizier`` fixtures from ``tests/conftest.py``. They check the reshaping
contract that the notebook relies on (matching ``twirl.gaia_radecs``), the
optional proper-motion propagation via ``SkyCoord.apply_space_motion``, the
empty-result path, that the query is issued against the right catalog with the
requested row limit and sort, and the fallback from one server to the next when
a server fails. The GAVO TAP last resort is faked by the ``fake_gavo`` fixture
here, which answers with a trimmed recorded GAVO VOTable. Recorded calls are
inspected via the mock's ``call_args`` rather than any side-effect state.
"""

import json
import logging
import math
import warnings
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
import requests
from astropy.coordinates import SkyCoord
from astropy.time import Time

from bandaid import catalog
from bandaid.catalog import (
    GAVO_TAP_SERVER,
    GAVO_TAP_SYNC_URL,
    SERVER_TIMEOUT,
    cached_gaia_radecs,
    query_field_catalog,
)
from bandaid.exceptions import (
    BatchPrepError,
    CatalogTruncationError,
    GaiaQueryError,
)

# Gaia DR3 in VizieR and the reference epoch (Julian year) of its positions.
GAIA_DR3_CATALOG = "I/355/gaiadr3"
GAIA_DR3_EPOCH = 2016.0
# What VizieR sends for a cone with no matching sources: no TABLE and no error.
EMPTY_FIELD_VOTABLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<VOTABLE version="1.4" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
<INFO name="server_software" value="VizieR/7.6.1"/>
<RESOURCE>
</RESOURCE>
<RESOURCE>
</RESOURCE>
</VOTABLE>
"""
# What CDS sent during its 2026-10-20 outage, with HTTP status 200.
OUTAGE_MESSAGE = "The database is not currently reachable"
OUTAGE_VOTABLE = f"""<?xml version="1.0" encoding="UTF-8"?>
<VOTABLE version="1.4" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
<RESOURCE>
<INFO name="Error">{OUTAGE_MESSAGE}</INFO>
</RESOURCE>
</VOTABLE>
""".encode()
# An error INFO (as a value attribute) alongside an empty TABLE.
ERROR_WITH_EMPTY_TABLE_VOTABLE = f"""<?xml version="1.0" encoding="UTF-8"?>
<VOTABLE version="1.4" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
<RESOURCE>
<INFO name="Error" value="{OUTAGE_MESSAGE}"/>
<TABLE><FIELD name="Gmag" datatype="double"/><DATA><TABLEDATA/></DATA></TABLE>
</RESOURCE>
</VOTABLE>
""".encode()
# What the xeus-python browser kernel (pyjs 4.0.7) raises for a failed
# XMLHttpRequest, here one to IDIA: str() of the exception is JSON of the
# JavaScript error's own properties, which for a DOMException is only "stack".
BROWSER_ERROR_TEXT = (
    "NetworkError: Failed to execute 'send' on 'XMLHttpRequest': "
    "Failed to load 'https://vizier.idia.ac.za/viz-bin/votable'."
)
_XPYTHON = "http://localhost:8012/xeus/browser-photom/bin/xpython"
BROWSER_ERROR_JSON = json.dumps(
    {
        "stack": "\n".join(
            [
                BROWSER_ERROR_TEXT,
                f" at Module._apply_try_catch ({_XPYTHON}.js:7:1950)",
                " at methodCaller<(emscripten::val, emscripten::val, "
                "emscripten::val) => emscripten::val> (eval at "
                f"__emval_get_method_caller ({_XPYTHON}.js:7:222664), "
                "<anonymous>:7:17)",
                f" at __emval_call ({_XPYTHON}.js:7:220678)",
                f" at {_XPYTHON}.wasm:wasm-function[17014]:0x6e388a",
                f" at {_XPYTHON}.wasm:wasm-function[17013]:0x6e3765",
                f" at {_XPYTHON}.wasm:wasm-function[16940]:0x6dd416",
                f" at {_XPYTHON}.wasm:wasm-function[270]:0x21eb0f",
                f" at _PyEM_TrampolineCall_JavaScript ({_XPYTHON}.js:7:628185)",
                f" at {_XPYTHON}.wasm:wasm-function[18806]:0x78f748",
                f" at _PyEM_TrampolineCall_JavaScript ({_XPYTHON}.js:7:628185)",
            ]
        )
    },
    separators=(",", ":"),
)
# The documented VizieR server order: CDS first, then IDIA, then CfA. The GAVO
# TAP service is tried after all three (see ALL_SERVERS_TRIED).
SERVER_ORDER = ["vizier.cds.unistra.fr", "vizier.idia.ac.za", "vizier.cfa.harvard.edu"]
# How a TAP service reports a failed query.
TAP_ERROR_MESSAGE = "Field query: No such field known"
TAP_ERROR_VOTABLE = f"""<?xml version="1.0" encoding="UTF-8"?>
<VOTABLE version="1.5" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
<RESOURCE type="results">
<INFO name="QUERY_STATUS" value="ERROR">{TAP_ERROR_MESSAGE}</INFO>
</RESOURCE>
</VOTABLE>
""".encode()
# How a TAP service reports a failure after it has started sending rows: a
# QUERY_STATUS ERROR after the table, which overrides the OK before it.
MID_STREAM_ERROR_MESSAGE = "Query aborted while sending rows"
# Trimmed from a recorded dc.g-vo.org answer (2026-10-29) to the GAVO query
# for the three brightest stars within 0.1 deg of (239.9, 25.9). Only the
# FIELD declarations, the QUERY_STATUS and the rows are kept, and the rows
# are written as TABLEDATA; the service itself sends BINARY.
GAVO_VOTABLE = b"""<?xml version='1.0' encoding='utf-8'?>
<VOTABLE version="1.5" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
<RESOURCE type="results">
<INFO name="QUERY_STATUS" value="OK">Query successful</INFO>
<COOSYS ID="system" epoch="J2016.0" refposition="BARYCENTER" system="ICRS"/>
<TABLE name="dr3lite">
<FIELD ID="Gmag" datatype="float" name="Gmag"
  ucd="phot.mag;em.opt;stat.mean" unit="mag"/>
<FIELD ID="RA_ICRS" datatype="double" name="RA_ICRS"
  ucd="pos.eq.ra;meta.main" unit="deg"/>
<FIELD ID="DE_ICRS" datatype="double" name="DE_ICRS"
  ucd="pos.eq.dec;meta.main" unit="deg"/>
<FIELD ID="pmRA" datatype="float" name="pmRA"
  ucd="pos.pm;pos.eq.ra" unit="mas/yr"/>
<FIELD ID="pmDE" datatype="float" name="pmDE"
  ucd="pos.pm;pos.eq.dec" unit="mas/yr"/>
<DATA><TABLEDATA>
<TR><TD>8.724516</TD><TD>239.87565390231384</TD><TD>25.92022378794317</TD>
  <TD>-4.460514</TD><TD>12.015859</TD></TR>
<TR><TD>13.505642</TD><TD>239.921962110703</TD><TD>25.893150190427217</TD>
  <TD>-0.91251844</TD><TD>-1.9135457</TD></TR>
<TR><TD>14.075624</TD><TD>239.86838050558544</TD><TD>25.869797174834083</TD>
  <TD>2.8270888</TD><TD>-7.088754</TD></TR>
</TABLEDATA></DATA>
</TABLE>
</RESOURCE>
</VOTABLE>
"""
# The columns of GAVO_VOTABLE.
GAVO_GMAG = [8.724516, 13.505642, 14.075624]
GAVO_RA = [239.87565390231384, 239.921962110703, 239.86838050558544]
GAVO_DEC = [25.92022378794317, 25.893150190427217, 25.869797174834083]
GAVO_PMRA = [-4.460514, -0.91251844, 2.8270888]
GAVO_PMDE = [12.015859, -1.9135457, -7.088754]
# GAVO's answer for an empty field: the same table with no rows.
GAVO_EMPTY_VOTABLE = (
    GAVO_VOTABLE.split(b"<TR>")[0]
    + b"</TABLEDATA>"
    + (GAVO_VOTABLE.split(b"</TABLEDATA>")[1])
)
# The GAVO rows followed by a mid-stream failure.
GAVO_MID_STREAM_ERROR_VOTABLE = GAVO_VOTABLE.replace(
    b"</TABLE>",
    b'</TABLE>\n<INFO name="QUERY_STATUS" value="ERROR">'
    + MID_STREAM_ERROR_MESSAGE.encode()
    + b"</INFO>",
)
# Every server in the order tried, as the all-servers-failed message lists them:
# the three VizieR servers, then GAVO.
ALL_SERVERS_TRIED = (
    "tried vizier.cds.unistra.fr, vizier.idia.ac.za, vizier.cfa.harvard.edu, "
    "dc.g-vo.org;"
)
# Arbitrary row limit used to assert it is forwarded to Vizier unchanged.
ROW_LIMIT_PROBE = 1234
# Floor (deg) the proper-motion shift must exceed; nine years of ~10 mas/yr is
# ~2.5e-5 deg, comfortably above this.
PM_SHIFT_FLOOR_DEG = 1e-6


@pytest.fixture
def center():
    """A nominal field center near the fixture stars."""
    return SkyCoord(ra=239.9 * u.deg, dec=25.9 * u.deg)


@pytest.fixture
def fake_gavo(mocker, tmp_path, votable_response):
    """
    Patch the GAVO TAP client to answer with ``GAVO_VOTABLE``; return its mock.

    Notes
    -----
    The stand-in is a plain mock set on the class, which does not bind like a
    method, so ``call_args`` holds ``("POST", url)`` and the keyword arguments
    without ``self``. Set ``side_effect`` to a list to script other answers.
    ``_GavoTap._last_query`` is a mock too, so `_gavo_cache_removal` shows
    whether a failed answer was evicted, and ``cache_location`` is
    ``tmp_path``, so nothing touches the user's astroquery cache.
    """
    mocker.patch.object(
        catalog._GavoTap,  # noqa: SLF001
        "cache_location",
        new_callable=mocker.PropertyMock,
        return_value=tmp_path,
    )
    mocker.patch.object(catalog._GavoTap, "_last_query", create=True)  # noqa: SLF001
    request = mocker.patch.object(catalog._GavoTap, "_request")  # noqa: SLF001
    request.side_effect = lambda *args, **kwargs: votable_response(GAVO_VOTABLE)
    return request


def test_returns_radecs_and_mags_with_correct_shapes(fake_vizier, center, gaia_table):
    """magnitude=True returns (radecs, mags) with twirl's shapes and order."""
    radecs, mags = cached_gaia_radecs(center, 0.2, magnitude=True)

    # A healthy answer from CDS means no mirror is asked.
    assert _servers_tried(fake_vizier) == SERVER_ORDER[:1]
    assert radecs.shape == (len(gaia_table), 2)
    assert mags.shape == (len(gaia_table),)
    # Order is preserved (brightest-first, as returned by VizieR's "+Gmag").
    np.testing.assert_allclose(radecs[:, 0], gaia_table["RA_ICRS"].value)
    np.testing.assert_allclose(radecs[:, 1], gaia_table["DE_ICRS"].value)
    np.testing.assert_allclose(mags, gaia_table["Gmag"].value)


def test_magnitude_false_returns_only_radecs(fake_vizier, center, gaia_table):
    """magnitude=False returns just the (n, 2) radecs array."""
    result = cached_gaia_radecs(center, 0.2, magnitude=False)

    fake_vizier.return_value.query_region_async.assert_called_once()
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(gaia_table), 2)


def test_query_uses_catalog_row_limit_and_brightness_sort(fake_vizier, center):
    """The query targets I/355/gaiadr3 with the given row limit and a Gmag sort."""
    cached_gaia_radecs(center, 0.2, limit=ROW_LIMIT_PROBE)

    init_kwargs = fake_vizier.call_args.kwargs
    query_kwargs = fake_vizier.return_value.query_region_async.call_args.kwargs
    assert query_kwargs["catalog"] == GAIA_DR3_CATALOG
    assert init_kwargs["row_limit"] == ROW_LIMIT_PROBE
    # Fail over fast from a host that is down, but give a slow, healthy query
    # astroquery's default 60 s to answer: CDS sends nothing until it is done.
    assert init_kwargs["timeout"] == (10, 60)
    # "+Gmag" requests an ascending (brightest-first) sort from VizieR.
    assert "+Gmag" in init_kwargs["columns"]


def test_radius_is_half_the_min_fov(fake_vizier, center):
    """The cone radius is min(fov)/2, matching twirl (notebook passes 2*fov)."""
    cached_gaia_radecs(center, 0.4)

    radius = fake_vizier.return_value.query_region_async.call_args.kwargs["radius"]
    assert u.Quantity(radius).to_value(u.deg) == pytest.approx(0.2)


def test_no_proper_motion_by_default(fake_vizier, center, gaia_table):
    """With obs_epoch=None, positions are returned at the DR3 epoch unchanged."""
    radecs, _ = cached_gaia_radecs(center, 0.2, magnitude=True)

    fake_vizier.return_value.query_region_async.assert_called_once()
    np.testing.assert_allclose(radecs[:, 0], gaia_table["RA_ICRS"].value)
    np.testing.assert_allclose(radecs[:, 1], gaia_table["DE_ICRS"].value)


def test_proper_motion_applied_matches_apply_space_motion(
    fake_vizier, center, gaia_table
):
    """obs_epoch propagates positions exactly like SkyCoord.apply_space_motion."""
    obs_epoch = Time("2024-06-01")
    radecs, _ = cached_gaia_radecs(center, 0.2, obs_epoch=obs_epoch)

    fake_vizier.return_value.query_region_async.assert_called_once()
    # The masked (no-PM) fixture star is propagated with zero proper motion.
    expected = SkyCoord(
        ra=gaia_table["RA_ICRS"],
        dec=gaia_table["DE_ICRS"],
        pm_ra_cosdec=gaia_table["pmRA"].filled(0).quantity,
        pm_dec=gaia_table["pmDE"].filled(0).quantity,
        obstime=Time(GAIA_DR3_EPOCH, format="jyear"),
    ).apply_space_motion(new_obstime=obs_epoch)

    np.testing.assert_allclose(radecs[:, 0], expected.ra.deg, rtol=0, atol=1e-9)
    np.testing.assert_allclose(radecs[:, 1], expected.dec.deg, rtol=0, atol=1e-9)
    # And the propagated positions actually moved off the catalog epoch.
    shift = np.abs(radecs[:, 1] - gaia_table["DE_ICRS"].value).max()
    assert shift > PM_SHIFT_FLOOR_DEG


@pytest.mark.usefixtures("fake_vizier")
def test_proper_motion_propagation_emits_no_erfa_warning(center):
    """
    Propagating proper motions without distances does not warn the caller.

    ERFA substitutes a default distance, which is expected for Gaia rows that
    lack parallaxes. The project's pytest configuration filters that warning, so
    escalate every warning to an error here to see what a command-line user sees.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cached_gaia_radecs(center, 0.2, obs_epoch=Time("2024-06-01"))


def test_masked_pm_yields_finite_catalog_epoch_positions(
    fake_vizier, center, gaia_table
):
    """
    Masked-PM rows propagate to finite positions at their catalog coordinates.

    On a real VizieR result, ``MaskedColumn.quantity`` converts masked proper
    motions to NaN in a *plain* ``Quantity`` -- ``np.ma.filled`` on it is a
    no-op -- so Gaia sources without PM solutions used to come back with NaN
    positions and crash ``search_around_sky`` downstream in ``prepare_batch``.
    Fixes https://github.com/mwcraig/bandaid/issues/80.
    """
    # Guard the fixture's realism first: it must mimic the astroquery
    # round-trip that triggered the bug (plain NaN-bearing Quantity), and the
    # buggy fill must demonstrably be a no-op on it.
    pm_quantity = gaia_table["pmRA"].quantity
    assert not isinstance(pm_quantity, np.ma.MaskedArray)
    assert np.isnan(pm_quantity.value[2])
    assert np.isnan(np.ma.filled(pm_quantity, 0 * u.mas / u.yr).value[2])

    radecs, mags = cached_gaia_radecs(
        center, 0.2, obs_epoch=Time("2026-04-18T02:39:48")
    )

    fake_vizier.return_value.query_region_async.assert_called_once()
    assert np.isfinite(radecs).all()
    assert mags.shape == (len(gaia_table),)
    # Zero PM means the masked star stays at its J2016.0 catalog position.
    np.testing.assert_allclose(
        radecs[2],
        [gaia_table["RA_ICRS"][2], gaia_table["DE_ICRS"][2]],
        rtol=0,
        atol=1e-9,
    )


def test_literal_nonfinite_pm_treated_as_zero(fake_vizier, center, gaia_table):
    """
    An unmasked non-finite proper-motion value is neutralized to zero.

    Belt and suspenders for https://github.com/mwcraig/bandaid/issues/80: a
    literal NaN/inf PM (no mask at all) means "no proper motion", so the star
    is propagated with zero PM instead of yielding a NaN position.
    """
    gaia_table["pmRA"] = np.array([np.inf, -0.957, 2.839]) * (u.mas / u.yr)
    gaia_table["pmDE"] = np.array([np.nan, -1.993, -7.168]) * (u.mas / u.yr)

    radecs, _ = cached_gaia_radecs(center, 0.2, obs_epoch=Time("2026-04-18T02:39:48"))

    fake_vizier.return_value.query_region_async.assert_called_once()
    assert np.isfinite(radecs).all()
    np.testing.assert_allclose(
        radecs[0],
        [gaia_table["RA_ICRS"][0], gaia_table["DE_ICRS"][0]],
        rtol=0,
        atol=1e-9,
    )


def test_empty_result_returns_shaped_empties(fake_vizier, center, votable_response):
    """
    A field with no Gaia sources yields (0, 2) / (0,) arrays, not an IndexError.

    The healthy empty answer is final: no other server is tried, since every
    mirror would return the same empty field, and its cache entry is kept.
    """
    fake_vizier.return_value.query_region_async.side_effect = None
    fake_vizier.return_value.query_region_async.return_value = votable_response(
        EMPTY_FIELD_VOTABLE
    )

    radecs, mags = cached_gaia_radecs(center, 0.2, magnitude=True)
    assert radecs.shape == (0, 2)
    assert mags.shape == (0,)
    assert _servers_tried(fake_vizier) == SERVER_ORDER[:1]
    _cache_removal(fake_vizier).assert_not_called()
    assert cached_gaia_radecs(center, 0.2, magnitude=False).shape == (0, 2)


def test_mag_limit_adds_vizier_column_filter(fake_vizier, center):
    """A mag_limit is forwarded to Vizier as a Gmag column filter."""
    cached_gaia_radecs(center, 0.2, mag_limit=17.5)

    assert fake_vizier.call_args.kwargs["column_filters"] == {"Gmag": "<=17.5"}


def test_no_mag_limit_passes_no_column_filter(fake_vizier, center):
    """Without mag_limit the Vizier call has no column_filters kwarg at all."""
    cached_gaia_radecs(center, 0.2)

    assert "column_filters" not in fake_vizier.call_args.kwargs


class DOMException:
    """
    Stand-in for the JavaScript error of a failed ``XMLHttpRequest``.

    A ``DOMException``'s ``name`` and ``message`` are getters on its
    prototype, not properties of its own, which is why pyjs's JSON text of
    the error holds only ``stack``. Class-level properties model that.
    """

    name = property(lambda _self: "NetworkError")
    message = property(lambda _self: BROWSER_ERROR_TEXT.removeprefix("NetworkError: "))


class JsGenericError(catalog.JsException):
    """
    Stand-in for pyjs's exception for a JavaScript error that is no ``Error``.

    Parameters
    ----------
    text : str
        What ``str()`` of the exception gives.
    js_error : object, optional
        What ``get_js_proxy`` hands out, as pyjs's ``JsHolder`` does. ``None``
        has no ``name``, so the JSON text is used.
    """

    def __init__(self, text, js_error=None) -> None:
        super().__init__(text)
        self._js_error = js_error

    def get_js_proxy(self):
        """
        Return the JavaScript error.

        Returns
        -------
        object
            The stand-in JavaScript error.
        """
        return self._js_error


class BrokenProxyError(JsGenericError):
    """A browser kernel exception whose JavaScript error cannot be reached."""

    def get_js_proxy(self):
        """
        Fail to return the JavaScript error.

        Raises
        ------
        RuntimeError
            Always.
        """
        msg = "proxy is gone"
        raise RuntimeError(msg)


class UnprintableError(catalog.JsException):
    """A browser kernel exception whose text cannot even be formed."""

    def __str__(self) -> str:
        """
        Fail to give the text.

        Raises
        ------
        RuntimeError
            Always.
        """  # noqa: DOC201 (never returns)
        msg = "no text"
        raise RuntimeError(msg)


def _servers_tried(fake_vizier):
    return [c.kwargs["vizier_server"] for c in fake_vizier.call_args_list]


def _cache_removal(fake_vizier):
    # astroquery offers cache-entry removal only on the private _last_query.
    return fake_vizier.return_value._last_query.remove_cache_file  # noqa: SLF001


def _gavo_cache_removal():
    # The class-level _last_query mock that fake_gavo installs.
    return catalog._GavoTap._last_query.remove_cache_file  # noqa: SLF001


class TestServerFallback:
    """A failing server is skipped, and an error answer is kept out of the cache."""

    @pytest.mark.parametrize(
        ("first_answer", "expect_eviction", "logged"),
        [
            ((OUTAGE_VOTABLE, 200), True, OUTAGE_MESSAGE),
            ((ERROR_WITH_EMPTY_TABLE_VOTABLE, 200), True, OUTAGE_MESSAGE),
            ((b"Service Unavailable", 503), True, "503"),
            # IDIA's error document for a bad catalog arrives as HTTP 200 with a
            # BEL control character in it, which the VOTable parser rejects.
            ((b"<VOTABLE>\x07", 200), True, "other than a VOTable"),
            # The browser kernel's timeout, which is not a requests exception.
            (catalog.JsException("XMLHttpRequest timed out"), False, "timed out"),
        ],
        ids=[
            "error-no-table",
            "error-empty-table",
            "http-503",
            "not-a-votable",
            "exception",
        ],
    )
    def test_failed_server_is_skipped(
        self,
        fake_vizier,
        center,
        gaia_table,
        votable_response,
        caplog,
        first_answer,
        expect_eviction,
        logged,
    ):
        """
        A failed first server is logged and skipped; a failed answer is uncached.

        An error-only VOTable, a failure status, an unparsable body, or the
        browser kernel's timeout (which is not a ``requests`` exception) all
        move on to the next server.
        """
        if isinstance(first_answer, tuple):
            first_answer = votable_response(*first_answer)
        fake_vizier.return_value.query_region_async.side_effect = [
            first_answer,
            votable_response(gaia_table),
        ]

        with caplog.at_level(logging.WARNING, logger="bandaid.catalog"):
            radecs, mags = cached_gaia_radecs(center, 0.2)

        assert _servers_tried(fake_vizier) == SERVER_ORDER[:2]
        if expect_eviction:
            _cache_removal(fake_vizier).assert_called_once_with(
                fake_vizier.return_value.cache_location
            )
        else:
            _cache_removal(fake_vizier).assert_not_called()
        np.testing.assert_allclose(mags, gaia_table["Gmag"].value)
        assert radecs.shape == (len(gaia_table), 2)
        assert any(
            SERVER_ORDER[0] in r.getMessage() and logged in r.getMessage()
            for r in caplog.records
        )

    def test_local_error_propagates_instead_of_trying_the_next_server(
        self, fake_vizier, center
    ):
        """A fault in our own code is not a server failure: it propagates as is."""
        fake_vizier.return_value.query_region_async.side_effect = TypeError(
            "got an unexpected keyword argument"
        )

        with pytest.raises(TypeError, match="unexpected keyword"):
            cached_gaia_radecs(center, 0.2)

        assert _servers_tried(fake_vizier) == SERVER_ORDER[:1]

    def test_missing_cache_file_does_not_mask_the_failure(
        self, fake_vizier, center, gaia_table, votable_response, caplog
    ):
        """An already-missing cache entry leaves the server's error as the reason."""
        _cache_removal(fake_vizier).side_effect = FileNotFoundError("no pickle")
        fake_vizier.return_value.query_region_async.side_effect = [
            votable_response(OUTAGE_VOTABLE),
            votable_response(gaia_table),
        ]

        with caplog.at_level(logging.WARNING, logger="bandaid.catalog"):
            cached_gaia_radecs(center, 0.2)

        messages = [r.getMessage() for r in caplog.records]
        assert any(OUTAGE_MESSAGE in m for m in messages)
        assert not any("FileNotFoundError" in m for m in messages)

    @pytest.mark.parametrize(
        ("mag_limit", "mag_cut"),
        [(17.5, " AND phot_g_mean_mag <= 17.5"), (None, "")],
        ids=["mag-limit", "no-mag-limit"],
    )
    def test_gavo_tap_is_the_last_resort(
        self, fake_vizier, fake_gavo, center, mag_limit, mag_cut
    ):
        """With every VizieR server down, GAVO's TAP service answers the query."""
        fake_vizier.return_value.query_region_async.side_effect = TimeoutError

        radecs, mags = cached_gaia_radecs(
            center, 0.2, limit=ROW_LIMIT_PROBE, mag_limit=mag_limit
        )

        # GAVO otherwise caps the answer at its own 20000-row default, so a
        # truncated dense field would slip past the row-limit check: MAXREC.
        fake_gavo.assert_called_once_with(
            "POST",
            GAVO_TAP_SYNC_URL,
            data={
                "REQUEST": "doQuery",
                "LANG": "ADQL",
                "FORMAT": "votable",
                "MAXREC": ROW_LIMIT_PROBE,
                "QUERY": (
                    f'SELECT TOP {ROW_LIMIT_PROBE} phot_g_mean_mag AS "Gmag", '  # noqa: S608
                    'ra AS "RA_ICRS", dec AS "DE_ICRS", '
                    'pmra AS "pmRA", pmdec AS "pmDE" FROM gaia.dr3lite '
                    "WHERE DISTANCE(POINT('ICRS', ra, dec), "
                    "POINT('ICRS', 239.9, 25.9)) <= 0.1"
                    f"{mag_cut} ORDER BY phot_g_mean_mag"
                ),
            },
            timeout=SERVER_TIMEOUT,
        )
        # Gmag is float32 in GAVO's answer.
        np.testing.assert_allclose(mags, GAVO_GMAG, rtol=1e-6)
        np.testing.assert_array_equal(radecs, np.column_stack([GAVO_RA, GAVO_DEC]))
        _gavo_cache_removal().assert_not_called()

    @pytest.mark.usefixtures("fake_gavo")
    def test_gavo_tap_positions_are_propagated(self, fake_vizier, center):
        """obs_epoch moves GAVO's positions with its proper-motion columns."""
        fake_vizier.return_value.query_region_async.side_effect = TimeoutError
        obs_epoch = Time("2024-06-01")

        radecs, _ = cached_gaia_radecs(center, 0.2, obs_epoch=obs_epoch)

        expected = SkyCoord(
            ra=GAVO_RA * u.deg,
            dec=GAVO_DEC * u.deg,
            pm_ra_cosdec=GAVO_PMRA * u.mas / u.yr,
            pm_dec=GAVO_PMDE * u.mas / u.yr,
            obstime=Time(GAIA_DR3_EPOCH, format="jyear"),
        ).apply_space_motion(new_obstime=obs_epoch)
        np.testing.assert_allclose(radecs[:, 0], expected.ra.deg, rtol=0, atol=1e-9)
        np.testing.assert_allclose(radecs[:, 1], expected.dec.deg, rtol=0, atol=1e-9)
        shift = np.abs(radecs[:, 1] - GAVO_DEC).max()
        assert shift > PM_SHIFT_FLOOR_DEG

    def test_gavo_tap_empty_field_returns_shaped_empties(
        self, fake_vizier, fake_gavo, center, votable_response
    ):
        """GAVO's healthy answer for an empty field is final and stays cached."""
        fake_vizier.return_value.query_region_async.side_effect = TimeoutError
        fake_gavo.side_effect = [votable_response(GAVO_EMPTY_VOTABLE)]

        radecs, mags = cached_gaia_radecs(center, 0.2)

        assert radecs.shape == (0, 2)
        assert mags.shape == (0,)
        fake_gavo.assert_called_once()
        _gavo_cache_removal().assert_not_called()

    @pytest.mark.parametrize(
        ("answer", "status_code", "error_text"),
        [
            (TAP_ERROR_VOTABLE, 500, TAP_ERROR_MESSAGE),
            # The rows arrived, but the status after them says they are not
            # the whole answer.
            (GAVO_MID_STREAM_ERROR_VOTABLE, 200, MID_STREAM_ERROR_MESSAGE),
        ],
        ids=["tap-error", "error-after-rows"],
    )
    def test_failed_gavo_answer_is_evicted(
        self,
        fake_vizier,
        fake_gavo,
        center,
        tmp_path,
        votable_response,
        answer,
        status_code,
        error_text,
    ):
        """A GAVO error is the final failure, and its answer leaves the cache."""
        fake_vizier.return_value.query_region_async.side_effect = TimeoutError
        fake_gavo.side_effect = [votable_response(answer, status_code=status_code)]

        with pytest.raises(GaiaQueryError) as excinfo:
            cached_gaia_radecs(center, 0.2)

        message = str(excinfo.value)
        assert ALL_SERVERS_TRIED in message
        assert f"last error: GaiaQueryError: {GAVO_TAP_SERVER}" in message
        assert error_text in message
        _gavo_cache_removal().assert_called_once_with(tmp_path)

    def test_all_servers_failing_raises_with_last_cause(
        self, fake_vizier, fake_gavo, center, votable_response
    ):
        """
        When every server fails, one error names them and carries the last cause.

        The last server, GAVO, answers with a TAP error document and HTTP 500.
        Its text must be in the message itself, since a CLI or browser user
        sees only ``str(exc)``, and in the chained cause.
        """
        fake_vizier.return_value.query_region_async.side_effect = [
            TimeoutError("connect timed out"),
            TimeoutError("connect timed out"),
            votable_response(OUTAGE_VOTABLE),
        ]
        fake_gavo.side_effect = [votable_response(TAP_ERROR_VOTABLE, status_code=500)]

        with pytest.raises(GaiaQueryError) as excinfo:
            cached_gaia_radecs(center, 0.2)

        assert _servers_tried(fake_vizier) == SERVER_ORDER
        message = str(excinfo.value)
        assert ALL_SERVERS_TRIED in message
        assert f"last error: GaiaQueryError: {GAVO_TAP_SERVER}" in message
        assert TAP_ERROR_MESSAGE in message
        cause = excinfo.value.__cause__
        assert isinstance(cause, GaiaQueryError)
        assert GAVO_TAP_SERVER in str(cause)
        assert TAP_ERROR_MESSAGE in str(cause)

    @pytest.mark.parametrize(
        "js_error", [DOMException(), None], ids=["js-error-object", "json-stack"]
    )
    def test_browser_failure_is_reported_by_its_first_line(
        self, fake_vizier, fake_gavo, center, caplog, js_error
    ):
        """
        A browser request failure is logged and reported without its JS stack.

        The browser kernel's exception text is a kilobyte of JSON holding the
        JavaScript stack; only its first line says what failed. The line comes
        from the JavaScript error object when the exception hands it out, as
        pyjs's does, and from the JSON otherwise.
        """
        fake_vizier.return_value.query_region_async.side_effect = [
            JsGenericError(BROWSER_ERROR_JSON, js_error) for _ in SERVER_ORDER
        ]
        fake_gavo.side_effect = [JsGenericError(BROWSER_ERROR_JSON, js_error)]

        with (
            caplog.at_level(logging.WARNING, logger="bandaid.catalog"),
            pytest.raises(GaiaQueryError) as excinfo,
        ):
            cached_gaia_radecs(center, 0.2)

        expected = f"JsGenericError: {BROWSER_ERROR_TEXT}"
        message = str(excinfo.value)
        assert message.endswith(f"last error: {expected}")
        assert "at Module" not in message
        warnings_logged = [r.getMessage() for r in caplog.records]
        # One warning per VizieR server, then one for GAVO.
        assert len(warnings_logged) == len(SERVER_ORDER) + 1
        for logged in warnings_logged:
            assert logged.endswith(f"failed, skipping it: {expected}")
            assert "at Module" not in logged
        # The chained cause keeps the whole text for anyone who wants the stack.
        assert str(excinfo.value.__cause__) == BROWSER_ERROR_JSON


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (
            JsGenericError(BROWSER_ERROR_JSON, DOMException()),
            f"JsGenericError: {BROWSER_ERROR_TEXT}",
        ),
        (
            JsGenericError(BROWSER_ERROR_JSON),
            f"JsGenericError: {BROWSER_ERROR_TEXT}",
        ),
        (
            JsGenericError(
                BROWSER_ERROR_JSON, SimpleNamespace(name=None, message=None)
            ),
            f"JsGenericError: {BROWSER_ERROR_TEXT}",
        ),
        (
            JsGenericError(BROWSER_ERROR_JSON, SimpleNamespace(name="", message="m")),
            f"JsGenericError: {BROWSER_ERROR_TEXT}",
        ),
        (
            JsGenericError(BROWSER_ERROR_JSON, SimpleNamespace(name=1, message=b"m")),
            f"JsGenericError: {BROWSER_ERROR_TEXT}",
        ),
        (
            BrokenProxyError(BROWSER_ERROR_JSON),
            f"BrokenProxyError: {BROWSER_ERROR_TEXT}",
        ),
        (
            catalog.JsException("XMLHttpRequest timed out\n at send (x.js:1:2)"),
            "JsException: XMLHttpRequest timed out",
        ),
        (
            JsGenericError('{"name": "NetworkError"}'),
            'JsGenericError: {"name": "NetworkError"}',
        ),
        (JsGenericError('{"stack": 5}'), 'JsGenericError: {"stack": 5}'),
        (JsGenericError('{"stack": "  "}'), 'JsGenericError: {"stack": "  "}'),
        (JsGenericError(""), "JsGenericError: "),
        (UnprintableError(), "UnprintableError"),
        (
            requests.ConnectTimeout("connect timed out\nafter 10 s"),
            "ConnectTimeout: connect timed out\nafter 10 s",
        ),
    ],
    ids=[
        "js-error-object",
        "json-stack",
        "js-error-object-without-message",
        "js-error-object-with-empty-name",
        "js-error-object-with-non-str",
        "get-js-proxy-raises",
        "plain-text",
        "json-without-stack",
        "json-stack-not-str",
        "json-stack-blank",
        "empty",
        "unprintable",
        "requests-unchanged",
    ],
)
def test_describe_failure(exc, expected):
    """A browser failure is cut to its first line; any other text is kept whole."""
    assert catalog._describe_failure(exc) == expected  # noqa: SLF001


# astroquery formats the cone center with astropy's vectorized Angle.to_string,
# which on the x86_64 Linux CPython builds that uv installs reports a
# floating-point flag that plain float formatting leaves set, although nothing
# is wrong (https://github.com/astropy/astropy/issues/18989). Only this test
# reaches that code; the others replace Vizier entirely.
@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in do_format:RuntimeWarning"
)
@pytest.mark.parametrize(
    ("query", "payload", "cached"),
    [
        (partial(catalog._query_vizier, SERVER_ORDER[0]), OUTAGE_VOTABLE, False),  # noqa: SLF001
        (partial(catalog._query_vizier, SERVER_ORDER[0]), EMPTY_FIELD_VOTABLE, True),  # noqa: SLF001
        (catalog._query_gavo_tap, TAP_ERROR_VOTABLE, False),  # noqa: SLF001
        (catalog._query_gavo_tap, GAVO_EMPTY_VOTABLE, True),  # noqa: SLF001
    ],
    ids=[
        "vizier-error-evicted",
        "vizier-empty-field-kept",
        "gavo-error-evicted",
        "gavo-empty-field-kept",
    ],
)
def test_real_astroquery_cache_eviction(
    mocker, tmp_path, center, votable_response, query, payload, cached
):
    """With astroquery's real cache, an error answer leaves no pickle behind."""
    mocker.patch(
        "astroquery.query.BaseQuery.cache_location",
        new_callable=mocker.PropertyMock,
        return_value=tmp_path,
    )
    response = votable_response(payload)
    response.request = requests.PreparedRequest()  # astroquery strips its hooks
    mocker.patch("requests.Session.request", return_value=response)

    with nullcontext() if cached else pytest.raises(GaiaQueryError):
        query(center, 0.1, limit=10, mag_limit=None)

    assert bool(list(tmp_path.glob("*.pickle"))) is cached


# Field geometry and magnitude limits used by the field-catalog tests.
FOV_RAD = 0.74
CONE_MARGIN = 0.4
GAIA_LIMIT = 15.0
CONTAMINANT_LIMIT = 18.0
ROW_LIMIT = 10000
# Equals math.ceil(10000 * ((0.74 + 0.4) / 0.74) ** 2).
SCALED_ROW_LIMIT = 23733


class TestQueryFieldCatalog:
    """``query_field_catalog`` scales the row limit and checks for truncation."""

    @staticmethod
    def _patch(mocker, mags) -> object:
        mags = np.asarray(mags, dtype=float)
        radecs = np.zeros((len(mags), 2))
        return mocker.patch(
            "bandaid.catalog.cached_gaia_radecs", return_value=(radecs, mags)
        )

    @staticmethod
    def _query(center, **kwargs: object) -> tuple:
        defaults = {
            "gaia_mag_limit": GAIA_LIMIT,
            "contaminant_mag_limit": CONTAMINANT_LIMIT,
            "row_limit": ROW_LIMIT,
        }
        defaults.update(kwargs)
        return query_field_catalog(center, FOV_RAD, **defaults)

    def test_zero_margin_keeps_row_limit(self, mocker, center):
        """With no margin the row limit and cone are unchanged."""
        mock = self._patch(mocker, [10.0])
        self._query(center)

        args, kwargs = mock.call_args
        assert args == (center, 2 * FOV_RAD)
        assert kwargs["limit"] == ROW_LIMIT
        assert kwargs["mag_limit"] == CONTAMINANT_LIMIT

    def test_margin_scales_limit_with_cone_area(self, mocker, center):
        """The row limit grows with the cone area; the cone is widened."""
        mock = self._patch(mocker, [10.0])
        self._query(center, cone_margin=CONE_MARGIN)

        args, kwargs = mock.call_args
        assert args[1] == pytest.approx(2 * (FOV_RAD + CONE_MARGIN))
        assert kwargs["limit"] == SCALED_ROW_LIMIT
        assert kwargs["limit"] == math.ceil(ROW_LIMIT * (1.14 / 0.74) ** 2)

    def test_obs_epoch_forwarded(self, mocker, center):
        """obs_epoch is passed through to the cached query."""
        mock = self._patch(mocker, [10.0])
        epoch = Time("2026-01-01")
        self._query(center, obs_epoch=epoch)

        assert mock.call_args.kwargs["obs_epoch"] == epoch

    def test_returns_radecs_and_mags(self, mocker, center):
        """The query result is returned unchanged."""
        self._patch(mocker, [10.0, 11.0])
        radecs, mags = self._query(center)

        assert radecs.shape == (2, 2)
        np.testing.assert_allclose(mags, [10.0, 11.0])

    def test_nonpositive_fov_raises(self, center):
        """fov_rad must be positive."""
        with pytest.raises(ValueError, match="fov_rad"):
            query_field_catalog(
                center,
                0.0,
                gaia_mag_limit=GAIA_LIMIT,
                contaminant_mag_limit=CONTAMINANT_LIMIT,
            )

    def test_truncation_inside_target_range_raises(self, mocker, center):
        """Truncation brighter than the target limit loses targets and is fatal."""
        self._patch(mocker, np.linspace(9.0, GAIA_LIMIT, ROW_LIMIT))
        with pytest.raises(CatalogTruncationError, match="targets") as excinfo:
            self._query(center)

        assert isinstance(excinfo.value, BatchPrepError)

    def test_truncation_in_contaminant_range_warns(self, mocker, center, caplog):
        """Truncation between the two limits only warns."""
        self._patch(mocker, np.linspace(9.0, 16.5, ROW_LIMIT))
        with caplog.at_level(logging.WARNING, logger="bandaid.catalog"):
            self._query(center)

        assert any("incomplete" in r.getMessage() for r in caplog.records)

    def test_truncation_at_contaminant_limit_is_silent(self, mocker, center, caplog):
        """Reaching the contaminant limit exactly at the row limit is fine."""
        self._patch(mocker, np.linspace(9.0, CONTAMINANT_LIMIT, ROW_LIMIT))
        with caplog.at_level(logging.WARNING, logger="bandaid.catalog"):
            self._query(center)

        assert not caplog.records

    def test_below_row_limit_is_silent(self, mocker, center, caplog):
        """No truncation check when fewer rows than the limit came back."""
        self._patch(mocker, np.linspace(9.0, 12.0, ROW_LIMIT - 1))
        with caplog.at_level(logging.WARNING, logger="bandaid.catalog"):
            self._query(center)

        assert not caplog.records

    def test_empty_result_is_silent(self, mocker, center, caplog):
        """An empty catalog neither raises nor warns."""
        self._patch(mocker, [])
        with caplog.at_level(logging.WARNING, logger="bandaid.catalog"):
            _, mags = self._query(center)

        assert len(mags) == 0
        assert not caplog.records
