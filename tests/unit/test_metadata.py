"""Unit tests for header/metadata parsing, airmass, good-star mask, and starlist."""

import astropy.units as u
import numpy as np
import pytest
from _helpers import _seestar_header
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.table import Table

from bandaid.config import InstrumentProfile
from bandaid.exceptions import (
    FrameMetadataError,
    NoUsableStarsError,
)
from bandaid.instruments import load_instrument
from bandaid.photometry import (
    _airmass_from_metadata,
    eloy_to_starlist,
    good_star_mask,
    metadata_from_header,
)


class TestAirmassFromMetadata:
    """Airmass comes from the resolved metadata, derived when absent (#29, #59)."""

    @staticmethod
    def _metadata(**overrides: object) -> dict:
        """Resolved-metadata dict carrying the keys airmass derivation needs."""
        metadata = {
            "airmass": None,
            "site_lat": 40.0,
            "site_lon": -105.0,
            "site_elev": None,
            "ra": 10.0,
            "dec": 20.0,
            "obs_time": "2024-06-01T07:00:00",
        }
        metadata.update(overrides)
        return metadata

    def test_uses_resolved_airmass_when_present(self):
        """A resolved ``airmass`` value is returned verbatim, pointing ignored."""
        assert _airmass_from_metadata(self._metadata(airmass=1.37)) == pytest.approx(
            1.37
        )

    def test_seestar_header_airmass_resolves_identically(self):
        """Seestar identity: a header AIRMASS still short-circuits the derivation."""
        header = _seestar_header()
        header["AIRMASS"] = 1.37
        metadata = metadata_from_header(header)
        assert _airmass_from_metadata(metadata) == pytest.approx(1.37)

    @staticmethod
    def _pointing_at_altitude(alt_deg) -> tuple:
        """RA/Dec (deg) that put the field at ``alt_deg`` altitude for the site."""
        location = EarthLocation(lat=40.0 * u.deg, lon=-105.0 * u.deg, height=0.0 * u.m)
        obstime = "2024-06-01T07:00:00"
        target = SkyCoord(
            AltAz(
                alt=alt_deg * u.deg,
                az=0.0 * u.deg,
                obstime=obstime,
                location=location,
            )
        ).icrs
        return target.ra.degree, target.dec.degree

    def _metadata_pointing_at_altitude(self, alt_deg):
        """Build metadata whose ra/dec put the field at ``alt_deg`` altitude."""
        ra, dec = self._pointing_at_altitude(alt_deg)
        return self._metadata(ra=ra, dec=dec)

    @staticmethod
    def _kasten_young(alt_deg) -> float:
        """Kasten & Young (1989) relative optical airmass for a given altitude."""
        return 1.0 / (
            np.sin(np.deg2rad(alt_deg)) + 0.50572 * (alt_deg + 6.07995) ** -1.6364
        )

    def test_derives_from_pointing_when_absent(self):
        """With no airmass, derive it from the ra/dec/site/obs_time metadata."""
        # Point ~1 deg from the zenith: a finite, physical (~1) airmass where
        # Kasten-Young and sec(z) coincide.
        metadata = self._metadata_pointing_at_altitude(89.0)

        airmass = _airmass_from_metadata(metadata)

        assert np.isfinite(airmass)
        assert airmass == pytest.approx(self._kasten_young(89.0), rel=1e-6)

    def test_seestar_header_derivation_unchanged(self):
        """Seestar identity: raw-header keys resolve to the same derived airmass."""
        ra, dec = self._pointing_at_altitude(89.0)
        header = _seestar_header()
        header["RA"] = ra
        header["DEC"] = dec
        header["SITELAT"] = 40.0
        header["SITELONG"] = -105.0
        header["SITEELEV"] = 0.0
        header["DATE-OBS"] = "2024-06-01T07:00:00"

        airmass = _airmass_from_metadata(metadata_from_header(header))

        assert airmass == pytest.approx(self._kasten_young(89.0), rel=1e-6)

    def test_header_map_renames_resolve(self):
        """A dialect with renamed site/pointing/time keywords derives airmass (#59)."""
        # None of the Seestar keyword names appear in this header; only the
        # header_map knows where the values live.
        custom_map = {
            "obs_time": "@DATE-LOC",
            "site_lat": "@OBSLAT",
            "site_lon": "@OBSLON",
            "site_elev": "@OBSELEV",
            "ra": "@OBJCTRA",
            "dec": "@OBJCTDEC",
            "egain": 1.0,
        }
        profile = InstrumentProfile(name="Renamed", header_map=custom_map)
        ra, dec = self._pointing_at_altitude(89.0)
        header = fits.Header()
        header["NAXIS1"] = 50
        header["NAXIS2"] = 50
        header["DATE-LOC"] = "2024-06-01T07:00:00"
        header["OBSLAT"] = 40.0
        header["OBSLON"] = -105.0
        header["OBSELEV"] = 0.0
        header["OBJCTRA"] = ra
        header["OBJCTDEC"] = dec

        metadata = metadata_from_header(header, profile=profile)
        airmass = _airmass_from_metadata(metadata)

        assert airmass == pytest.approx(self._kasten_young(89.0), rel=1e-6)

    def test_high_airmass_uses_kasten_young_not_secz(self):
        """At low altitude the result follows Kasten-Young, not sec(z)."""
        # Altitude 15 deg (zenith angle 75 deg) is in the range these frames
        # reach, where sec(z) overestimates by ~1.3%. The result must match KY1989
        # and be distinguishable from sec(z).
        metadata = self._metadata_pointing_at_altitude(15.0)

        airmass = _airmass_from_metadata(metadata)

        secz = 1.0 / np.cos(np.deg2rad(90.0 - 15.0))
        assert airmass == pytest.approx(self._kasten_young(15.0), rel=1e-4)
        assert airmass < secz
        assert airmass == pytest.approx(secz, rel=2e-2)

    def test_raises_when_inputs_missing(self):
        """No airmass and no pointing/site/time -> skip the frame, not a NaN."""
        with pytest.raises(FrameMetadataError):
            _airmass_from_metadata({"obs_time": "2024-06-01T07:00:00"})

    def test_raises_when_lookups_resolved_to_none(self):
        """Metadata keys present but unresolved (None) skip the frame cleanly."""
        # An "@KEY" directive whose keyword is absent resolves to None; that must
        # map to the same FrameMetadataError as a missing key.
        with pytest.raises(FrameMetadataError):
            _airmass_from_metadata(self._metadata(site_lat=None))

    def test_raises_on_malformed_airmass(self):
        """A present-but-unparseable airmass skips the frame rather than crashing."""
        with pytest.raises(FrameMetadataError):
            _airmass_from_metadata(self._metadata(airmass="not-a-number"))

    def test_raises_on_overflowing_obs_time(self):
        """
        An all-digit garbage obs_time skips the frame rather than crashing.

        dateutil raises OverflowError (not ValueError) for all-digit strings
        too large for a C long (PR #71/#74 review).
        """
        with pytest.raises(FrameMetadataError):
            _airmass_from_metadata(self._metadata(obs_time="999999999999999999"))


class TestMetadataFromHeader:
    """Unit tests for ``metadata_from_header`` against the Seestar template."""

    def test_resolves_template_directives(self):
        """@/!/literal directives and NAXIS sizing all resolve as documented."""
        # These literals live in basic.json (not the header), so they are pinned
        # here rather than read back from the input.
        expected_adc_depth = 12
        expected_max_adu = 50000

        header = _seestar_header()
        metadata = metadata_from_header(header)

        # "@KEY" -> header lookup.
        assert metadata["obs_time"] == "2024-01-01T00:00:00"
        assert metadata["block_filter"] == "L"
        # "!CREATOR index 0" -> first whitespace token of CREATOR.
        assert metadata["tel_manufac"] == "ZWO"
        # Plain literals pass through untouched.
        assert metadata["adc_depth"] == expected_adc_depth
        assert metadata["largest_usable_adu_value"] == expected_max_adu
        assert metadata["egain"] == pytest.approx(0.3116)
        assert metadata["roworder"] == "top-down"
        assert metadata["refframe"] == "ICRS"
        # width/height come straight from NAXIS1/NAXIS2.
        assert metadata["width"] == header["NAXIS1"]
        assert metadata["height"] == header["NAXIS2"]
        # Comment/internal keys are dropped, not surfaced.
        assert "_note" not in metadata
        assert "_filter" not in metadata
        assert "#stack" not in metadata

    def test_present_stackcnt_used(self):
        """When STACKCNT is present its value flows into ``stack``."""
        header = _seestar_header(with_stackcnt=True)
        metadata = metadata_from_header(header)
        assert metadata["stack"] == header["STACKCNT"]

    def test_missing_stackcnt_falls_back_to_default(self):
        """A missing STACKCNT falls back to the ``#stack`` default of 1."""
        default_stack = 1
        metadata = metadata_from_header(_seestar_header(with_stackcnt=False))
        assert metadata["stack"] == default_stack

    def test_explicit_profile_header_map_is_used(self):
        """An explicit ``profile=`` header_map overrides the default dialect."""
        expected_exposure = 42.0
        base = load_instrument("Seestar50")
        # Re-point the exposure binding at a different header keyword.
        custom_map = {**base.header_map, "exposure": "@MYEXP"}
        profile = InstrumentProfile(name="Custom", header_map=custom_map)

        header = _seestar_header()
        header["MYEXP"] = expected_exposure
        metadata = metadata_from_header(header, profile=profile)
        assert metadata["exposure"] == expected_exposure


class TestGoodStarMask:
    """
    In-bounds tests for ``good_star_mask`` pixel-center bounds (issue #57).

    Coordinates are pixel centers: pixel 0 spans [-0.5, 0.5), so a star is
    on-frame when its center lies in [-0.5, width - 0.5) (and likewise in y).
    The old ``> 0`` / ``< width`` test dropped a star centered on column 0 and
    admitted a center up to a pixel past the last column.
    """

    def _mask_for_xy(self, eloy_table, starlist_metadata, x, y):
        """Return the good_star_mask for a single otherwise-good star at x, y."""
        row = {
            "x": x,
            "y": y,
            "ra": 10.0,
            "dec": 20.0,
            "tot_count": 100.0,
            "count_err": 5.0,
            "bkgd_count": 1.0,
            "peak_count": 200.0,
        }
        return good_star_mask(eloy_table([row]), starlist_metadata)

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            (0.0, 30.0),  # centered on column 0 -- dropped by the old x > 0
            (20.0, 0.0),  # centered on row 0 -- dropped by the old y > 0
            (-0.5, 30.0),  # left edge of pixel 0: still on-frame
            (99.4, 30.0),  # inside the last column (width 100)
        ],
    )
    def test_on_frame_centers_survive(self, eloy_table, starlist_metadata, x, y):
        """Stars whose centers lie on the frame pass the bounds test."""
        assert self._mask_for_xy(eloy_table, starlist_metadata, x, y)[0]

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            (99.5, 30.0),  # at/past the right edge of the last column
            (20.0, 99.6),  # past the bottom edge -- admitted by the old y < height
            (-0.6, 30.0),  # past the left edge of pixel 0
            (20.0, -0.6),
        ],
    )
    def test_off_frame_centers_dropped(self, eloy_table, starlist_metadata, x, y):
        """Stars whose centers fall off the frame fail the bounds test."""
        assert not self._mask_for_xy(eloy_table, starlist_metadata, x, y)[0]


class TestEloyToStarlist:
    """
    Unit tests for the table->StarList conversion ``eloy_to_starlist``.

    The ``starlist_metadata`` and ``eloy_table`` fixtures live in ``conftest.py``
    so they can be shared with ``test_scripts.py``.
    """

    def test_filters_bad_rows(self, eloy_table, starlist_metadata):
        """Only finite, positive, in-bounds rows survive into the StarList."""
        good_a = {
            "x": 20.0,
            "y": 30.0,
            "ra": 10.0,
            "dec": 20.0,
            "tot_count": 100.0,
            "count_err": 5.0,
            "bkgd_count": 1.0,
            "peak_count": 200.0,
        }
        good_b = {
            "x": 70.0,
            "y": 60.0,
            "ra": 11.0,
            "dec": 21.0,
            "tot_count": 300.0,
            "count_err": 7.0,
            "bkgd_count": 1.0,
            "peak_count": 400.0,
        }
        # Each bad row trips exactly one filter condition.
        bad_nan_count = {**good_a, "tot_count": np.nan}
        bad_zero_count = {**good_a, "tot_count": 0.0}
        bad_inf_err = {**good_a, "count_err": np.inf}
        bad_zero_err = {**good_a, "count_err": 0.0}
        bad_x_out = {**good_a, "x": 150.0}
        bad_y_out = {**good_a, "y": -5.0}

        table = eloy_table(
            [
                good_a,
                good_b,
                bad_nan_count,
                bad_zero_count,
                bad_inf_err,
                bad_zero_err,
                bad_x_out,
                bad_y_out,
            ],
        )
        starlist = eloy_to_starlist(table, starlist_metadata)

        kept_x = sorted(item.x for item in starlist.staritems)
        assert kept_x == [20.0, 70.0]

    def test_contaminated_rows_excluded(self, eloy_table, starlist_metadata):
        """A ``contaminated`` column drops flagged rows even when otherwise good."""
        good = {
            "x": 20.0,
            "y": 30.0,
            "ra": 10.0,
            "dec": 20.0,
            "tot_count": 100.0,
            "count_err": 5.0,
            "bkgd_count": 1.0,
            "peak_count": 200.0,
        }
        contaminated_good = {**good, "x": 70.0}
        table = eloy_table([good, contaminated_good], contaminated=[False, True])

        starlist = eloy_to_starlist(table, starlist_metadata)

        kept_x = [item.x for item in starlist.staritems]
        assert kept_x == [20.0]


class TestEloyToStarlistGuard:
    """``eloy_to_starlist`` rejects frames with no usable stars."""

    def test_raises_when_no_good_stars(self):
        """A table whose rows all fail filtering raises NoUsableStarsError."""
        table = Table(
            {
                "tot_count": [-1.0, np.nan],
                "count_err": [1.0, 1.0],
                "x": [10.0, 20.0],
                "y": [10.0, 20.0],
            },
        )
        with pytest.raises(NoUsableStarsError):
            eloy_to_starlist(table, {"width": 100, "height": 100})


class TestMetadataFromHeaderGuard:
    """``metadata_from_header`` turns missing keywords into FrameMetadataError."""

    def test_incomplete_header_raises(self):
        """A header missing required keywords is a frame metadata error."""
        with pytest.raises(FrameMetadataError):
            metadata_from_header(fits.Header())
