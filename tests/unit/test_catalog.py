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
a server fails. Recorded calls are inspected via the mock's ``call_args`` rather
than any side-effect state.
"""

import logging
import math
import warnings

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from bandaid.catalog import cached_gaia_radecs, query_field_catalog
from bandaid.exceptions import (
    BatchPrepError,
    CatalogTruncationError,
    GaiaQueryError,
    GaiaServiceError,
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
CDS = "vizier.cds.unistra.fr"
IDIA = "vizier.idia.ac.za"
GAVO = "dc.g-vo.org"
GAVO_TAP_SYNC_URL = "https://dc.g-vo.org/tap/sync"
# How a TAP service reports a failed query.
TAP_ERROR_MESSAGE = "Field query: No such field known"
TAP_ERROR_VOTABLE = f"""<?xml version="1.0" encoding="UTF-8"?>
<VOTABLE version="1.5" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
<RESOURCE type="results">
<INFO name="QUERY_STATUS" value="ERROR">{TAP_ERROR_MESSAGE}</INFO>
</RESOURCE>
</VOTABLE>
""".encode()
# Per-server timeout in seconds.
SERVER_TIMEOUT = 30
# Arbitrary row limit used to assert it is forwarded to Vizier unchanged.
ROW_LIMIT_PROBE = 1234
# Floor (deg) the proper-motion shift must exceed; nine years of ~10 mas/yr is
# ~2.5e-5 deg, comfortably above this.
PM_SHIFT_FLOOR_DEG = 1e-6


@pytest.fixture
def center():
    """A nominal field center near the fixture stars."""
    return SkyCoord(ra=239.9 * u.deg, dec=25.9 * u.deg)


def test_returns_radecs_and_mags_with_correct_shapes(fake_vizier, center, gaia_table):
    """magnitude=True returns (radecs, mags) with twirl's shapes and order."""
    radecs, mags = cached_gaia_radecs(center, 0.2, magnitude=True)

    fake_vizier.return_value.query_region_async.assert_called_once()
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
    """A field with no Gaia sources yields (0, 2) / (0,) arrays, not an IndexError."""
    fake_vizier.return_value.query_region_async.side_effect = lambda *args, **kwargs: (
        votable_response(EMPTY_FIELD_VOTABLE)
    )

    radecs, mags = cached_gaia_radecs(center, 0.2, magnitude=True)
    assert radecs.shape == (0, 2)
    assert mags.shape == (0,)
    assert cached_gaia_radecs(center, 0.2, magnitude=False).shape == (0, 2)


def test_mag_limit_adds_vizier_column_filter(fake_vizier, center):
    """A mag_limit is forwarded to Vizier as a Gmag column filter."""
    cached_gaia_radecs(center, 0.2, mag_limit=17.5)

    assert fake_vizier.call_args.kwargs["column_filters"] == {"Gmag": "<=17.5"}


def test_no_mag_limit_passes_no_column_filter(fake_vizier, center):
    """Without mag_limit the Vizier call has no column_filters kwarg at all."""
    cached_gaia_radecs(center, 0.2)

    assert "column_filters" not in fake_vizier.call_args.kwargs


class BrowserTimeoutError(Exception):
    """Stands in for ``pyodide.ffi.JsException``, a browser-kernel timeout."""


def _servers_tried(fake_vizier):
    return [c.kwargs["vizier_server"] for c in fake_vizier.call_args_list]


def _cache_removal(fake_vizier):
    # astroquery offers cache-entry removal only on the private _last_query.
    return fake_vizier.return_value._last_query.remove_cache_file  # noqa: SLF001


class TestServerFallback:
    """A failing server is skipped, and an error answer is kept out of the cache."""

    def test_first_server_is_cds_with_timeout_and_cache(
        self, fake_vizier, fake_gavo, center
    ):
        """A healthy CDS answers alone, with a timeout and astroquery's cache."""
        cached_gaia_radecs(center, 0.2)

        assert _servers_tried(fake_vizier) == [CDS]
        fake_gavo.assert_not_called()
        assert fake_vizier.call_args.kwargs["timeout"] == SERVER_TIMEOUT
        query_kwargs = fake_vizier.return_value.query_region_async.call_args.kwargs
        assert query_kwargs["cache"] is True

    @pytest.mark.parametrize(
        "payload",
        [OUTAGE_VOTABLE, ERROR_WITH_EMPTY_TABLE_VOTABLE],
        ids=["no-table", "empty-table"],
    )
    def test_error_votable_evicts_cache_and_falls_back(
        self, fake_vizier, center, gaia_table, votable_response, caplog, payload
    ):
        """An error-only VOTable is a failed server: uncached, logged, skipped."""
        instance = fake_vizier.return_value
        instance.query_region_async.side_effect = [
            votable_response(payload),
            votable_response(gaia_table),
        ]

        with caplog.at_level(logging.WARNING, logger="bandaid.catalog"):
            radecs, mags = cached_gaia_radecs(center, 0.2)

        assert _servers_tried(fake_vizier) == [CDS, IDIA]
        _cache_removal(fake_vizier).assert_called_once_with(instance.cache_location)
        np.testing.assert_allclose(mags, gaia_table["Gmag"].value)
        assert radecs.shape == (len(gaia_table), 2)
        assert any(
            CDS in r.getMessage() and OUTAGE_MESSAGE in r.getMessage()
            for r in caplog.records
        )

    def test_empty_field_is_cached_and_not_a_failure(
        self, fake_vizier, center, votable_response
    ):
        """A healthy answer with no sources is final: no fallback, cache kept."""
        instance = fake_vizier.return_value
        instance.query_region_async.side_effect = [
            votable_response(EMPTY_FIELD_VOTABLE)
        ]

        radecs, mags = cached_gaia_radecs(center, 0.2)

        assert radecs.shape == (0, 2)
        assert mags.shape == (0,)
        assert _servers_tried(fake_vizier) == [CDS]
        _cache_removal(fake_vizier).assert_not_called()

    def test_http_error_status_evicts_cache_and_falls_back(
        self, fake_vizier, center, gaia_table, votable_response
    ):
        """A non-success HTTP status is a failed server and is not cached."""
        instance = fake_vizier.return_value
        instance.query_region_async.side_effect = [
            votable_response(b"Service Unavailable", status_code=503),
            votable_response(gaia_table),
        ]

        _, mags = cached_gaia_radecs(center, 0.2)

        assert _servers_tried(fake_vizier) == [CDS, IDIA]
        _cache_removal(fake_vizier).assert_called_once_with(instance.cache_location)
        assert len(mags) == len(gaia_table)

    def test_any_exception_falls_back(
        self, fake_vizier, center, gaia_table, votable_response, caplog
    ):
        """A non-requests exception (the browser's timeout) also moves on."""
        fake_vizier.return_value.query_region_async.side_effect = [
            BrowserTimeoutError("XMLHttpRequest timed out"),
            votable_response(gaia_table),
        ]

        with caplog.at_level(logging.WARNING, logger="bandaid.catalog"):
            _, mags = cached_gaia_radecs(center, 0.2)

        assert _servers_tried(fake_vizier) == [CDS, IDIA]
        assert len(mags) == len(gaia_table)
        assert any("timed out" in r.getMessage() for r in caplog.records)

    def test_gavo_tap_is_the_last_resort(
        self, fake_vizier, fake_gavo, center, gaia_table, votable_response
    ):
        """With both VizieR servers down, GAVO's TAP service answers the query."""
        fake_vizier.return_value.query_region_async.side_effect = [
            votable_response(OUTAGE_VOTABLE),
            TimeoutError("read timed out"),
        ]

        radecs, mags = cached_gaia_radecs(
            center, 0.2, limit=ROW_LIMIT_PROBE, mag_limit=17.5
        )

        np.testing.assert_allclose(mags, gaia_table["Gmag"].value)
        np.testing.assert_allclose(radecs[:, 0], gaia_table["RA_ICRS"].value)
        fake_gavo.assert_called_once()
        assert fake_gavo.call_args.args[0] == GAVO_TAP_SYNC_URL
        assert fake_gavo.call_args.kwargs["timeout"] == SERVER_TIMEOUT
        query = fake_gavo.call_args.kwargs["data"]["QUERY"]
        assert "gaia.dr3lite" in query
        assert f"TOP {ROW_LIMIT_PROBE}" in query
        assert "phot_g_mean_mag <= 17.5" in query
        assert query.rstrip().endswith("ORDER BY phot_g_mean_mag")

    def test_gavo_query_has_no_mag_cut_without_mag_limit(
        self, fake_vizier, fake_gavo, center
    ):
        """Without mag_limit the ADQL has no magnitude condition."""
        fake_vizier.return_value.query_region_async.side_effect = TimeoutError

        cached_gaia_radecs(center, 0.2)

        query = fake_gavo.call_args.kwargs["data"]["QUERY"]
        assert "DISTANCE" in query
        assert "phot_g_mean_mag <=" not in query

    def test_all_servers_failing_raises_with_last_cause(
        self, fake_vizier, fake_gavo, center, votable_response
    ):
        """When every server fails, one error names them and chains the last cause."""
        last = TimeoutError("read timed out")
        fake_vizier.return_value.query_region_async.side_effect = [
            votable_response(OUTAGE_VOTABLE),
            TimeoutError("connect timed out"),
        ]
        fake_gavo.side_effect = last

        with pytest.raises(GaiaQueryError) as excinfo:
            cached_gaia_radecs(center, 0.2)

        for server in (CDS, IDIA, GAVO):
            assert server in str(excinfo.value)
        assert excinfo.value.__cause__ is last

    def test_error_answer_cause_carries_server_message(
        self, fake_vizier, fake_gavo, center, votable_response
    ):
        """A TAP error answer surfaces as a GaiaServiceError with the server's text."""
        fake_vizier.return_value.query_region_async.side_effect = (
            lambda *args, **kwargs: votable_response(OUTAGE_VOTABLE)
        )
        fake_gavo.side_effect = lambda *args, **kwargs: votable_response(
            TAP_ERROR_VOTABLE
        )

        with pytest.raises(GaiaQueryError) as excinfo:
            cached_gaia_radecs(center, 0.2)

        cause = excinfo.value.__cause__
        assert isinstance(cause, GaiaServiceError)
        assert cause.server == GAVO
        assert cause.server_message == TAP_ERROR_MESSAGE


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
