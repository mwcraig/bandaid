"""Unit tests for once-per-batch preparation and frame-consistency checks."""

import astropy.units as u
import numpy as np
import pytest
from _helpers import _consistency_header
from astropy.coordinates import SkyCoord
from astropy.table import MaskedColumn, Table
from astropy.time import Time
from dateutil import parser

from bandaid import scripts
from bandaid.catalog import GAIA_DR2_EPOCH
from bandaid.config import (
    ApertureConfig,
    InstrumentProfile,
    PhotometryConfig,
    SourceSelectionConfig,
)
from bandaid.exceptions import (
    BatchPrepError,
    FrameError,
    FrameMetadataError,
    TooFewStarsError,
)
from bandaid.photometry import min_separation_fwhm, neighbor_contamination_flag_sky


def _batch_metadata():
    """Return a metadata dict like the one ``calibration_sequence`` produces."""
    return {
        "ra": 10.0,
        "dec": 0.0,
        "obs_time": "2026-04-28T03:03:43.270038",
        "fov_rad": 0.74,
        "pixscale": 2.4,
        "width": 1080,
        "height": 1920,
        "bayerpat": "GRBG",
        "roworder": "top-down",
        "ybayroff": 0,
        "egain": 0.3116,
    }


def _batch_radecs_mags():
    """
    Sky positions + mags with one tight equal-brightness pair to be dropped.

    The first two stars sit ~1 arcsec apart at equal magnitude, so both are
    contaminated; the remaining two are degrees away and survive.
    """
    radecs = np.array(
        [
            [10.0, 0.0],
            [10.0 + 1.0 / 3600.0, 0.0],
            [10.1, 0.0],
            [10.2, 0.0],
        ],
    )
    mags = np.array([12.0, 12.0, 10.0, 11.0])
    return radecs, mags


def _patch_prep(monkeypatch, *, metadata=None, radecs_mags=None, fwhm_pix=2.0):
    """Monkeypatch the heavy prep dependencies and return the spied call args."""
    metadata = metadata if metadata is not None else _batch_metadata()
    radecs, mags = radecs_mags if radecs_mags is not None else _batch_radecs_mags()

    # These tests exercise the mag-cut/contamination plumbing with deliberately
    # tiny synthetic catalogs, so relax the "enough Gaia stars to solve a WCS"
    # floor; the floor itself is covered by TestPrepareBatch's guard tests.
    monkeypatch.setattr(scripts, "N_GAIA_STARS_ALIGN_RETRY", 1)

    calls = {}

    def fake_calibration_sequence(file, *, cnn=None, profile=None, **_kwargs: object):
        calls["calibration_file"] = file
        calls["calibration_cnn"] = cnn
        calls["calibration_profile"] = profile
        return np.zeros((4, 4)), metadata, np.zeros((3, 2)), fwhm_pix, object()

    def fake_cached_gaia_radecs(center, fov, *, obs_epoch=None):
        calls["center"] = center
        calls["fov"] = fov
        calls["obs_epoch"] = obs_epoch
        return radecs, mags

    monkeypatch.setattr(scripts, "calibration_sequence", fake_calibration_sequence)
    monkeypatch.setattr(scripts, "cached_gaia_radecs", fake_cached_gaia_radecs)
    return calls, metadata, radecs, mags, fwhm_pix


class TestEstimateCenterFromHeader:
    """Unit tests for ``estimate_center_from_header``."""

    def test_walks_header_pointing_by_profile_offset(self):
        """The header pointing is shifted to the field center by the profile vector."""
        # The bare Seestar defaults carry the framing offset (the default pipeline
        # is the Seestar), so the estimate walks the header away from the corner.
        profile = InstrumentProfile()
        d_ra_cosdec, d_dec = profile.header_center_offset
        metadata = {"ra": 10.0, "dec": 30.0}

        ra, dec = scripts.estimate_center_from_header(metadata, profile)

        assert dec == pytest.approx(30.0 + d_dec)
        # The RA offset is stored as Delta(RA*cos(dec)); dividing by cos(dec)
        # recovers the RA shift itself.
        assert ra == pytest.approx(10.0 + d_ra_cosdec / np.cos(np.radians(30.0)))

    def test_no_offset_returns_header_unchanged(self):
        """A profile without framing constants returns the raw header pointing."""
        profile = InstrumentProfile(header_center_offset=None)
        metadata = {"ra": 10.0, "dec": 30.0}

        assert scripts.estimate_center_from_header(metadata, profile) == (10.0, 30.0)

    def test_string_pointing_is_coerced_to_float(self):
        """A numeric-string header pointing (the raw @RA/@DEC form) is coerced."""
        # @RA/@DEC pass the header value through untouched, so it often arrives
        # as a numeric string; the estimate must do arithmetic on floats.
        profile = InstrumentProfile()
        metadata = {"ra": "10.0", "dec": "0.0"}

        ra, dec = scripts.estimate_center_from_header(metadata, profile)

        d_ra_cosdec, d_dec = profile.header_center_offset
        assert ra == pytest.approx(10.0 + d_ra_cosdec)  # cos(0) == 1
        assert dec == pytest.approx(d_dec)


class TestPrepareBatch:
    """Unit tests for ``prepare_batch``."""

    def test_returns_batchprep_with_expected_fields(self, monkeypatch):
        """The bundle carries the Gaia list, the cnn, and the three CFA masks."""
        _, _, radecs, _, _ = _patch_prep(monkeypatch)
        cnn = object()

        # append_l4 defaults to True (issue #61); pin it False here so this
        # test's "three CFA masks" stays decoupled from that default.
        prep = scripts.prepare_batch("frame1.fits", cnn=cnn, append_l4=False)

        assert isinstance(prep, scripts.BatchPrep)
        np.testing.assert_array_equal(prep.radecs, radecs)
        assert prep.cnn is cnn
        assert set(prep.bayer_masks) == {"TR", "TB", "TG"}

    def test_append_l4_true_by_default(self, monkeypatch):
        """Omitting ``append_l4`` adds the L4 channel (issue #61)."""
        _patch_prep(monkeypatch)

        prep = scripts.prepare_batch("frame1.fits", cnn=object())

        assert set(prep.bayer_masks) == {"TR", "TB", "TG", "L4"}
        assert prep.bayer_masks["L4"] is None

    def test_first_frame_resolved_with_config_instrument_profile(self, monkeypatch):
        """
        The config's instrument is threaded into the first-frame calibration.

        Without this, ``prepare_batch`` would resolve the first frame's metadata
        with the bundled-Seestar50 fallback rather than ``config.instrument`` --
        wrong for any other telescope.
        """
        calls, *_ = _patch_prep(monkeypatch)
        instrument = InstrumentProfile(name="MyScope")
        scripts.prepare_batch(
            "frame1.fits",
            cnn=object(),
            config=PhotometryConfig(instrument=instrument),
        )
        assert calls["calibration_profile"] is instrument

    def test_first_frame_fwhm_uses_the_per_frame_detection_settings(self, monkeypatch):
        """
        The batch-gating FWHM is measured with the per-frame detection settings.

        ``prepare_batch`` must hand ``calibration_sequence`` the same
        ``detect_on_bayer_balanced=True`` and ``fwhm_n_stars`` that every
        per-frame call uses (``photometry.process_one_image``); otherwise the
        FWHM that sizes the contamination radii is measured in a different
        detection regime than the photometry it protects. Fixes
        https://github.com/mwcraig/bandaid/issues/55.
        """
        captured = {}
        fwhm_n_stars = 7

        def fake_calibration_sequence(_file, **kwargs: object):
            captured.update(kwargs)
            msg = "stop after capturing the call"
            raise TooFewStarsError(msg)

        monkeypatch.setattr(scripts, "calibration_sequence", fake_calibration_sequence)

        config = PhotometryConfig(
            instrument=InstrumentProfile(name="Seestar50", fwhm_n_stars=fwhm_n_stars)
        )
        with pytest.raises(BatchPrepError):
            scripts.prepare_batch("first.fits", cnn=object(), config=config)

        assert captured.get("detect_on_bayer_balanced") is True
        assert captured.get("fwhm_n_stars") == fwhm_n_stars

    def test_gaia_queried_at_resolved_center_and_widened_fov(self, monkeypatch):
        """
        Gaia is queried at the resolved field center over the widened field.

        The Seestar header points ~0.35 deg off the field center, so the cone is
        centered on the header-estimate field center (not the raw header) and
        widened by the profile margin (issue #83).
        """
        calls, metadata, _, _, _ = _patch_prep(monkeypatch)
        scripts.prepare_batch("frame1.fits", cnn=object())

        instrument = InstrumentProfile()
        expected_center = scripts.estimate_center_from_header(metadata, instrument)
        assert calls["center"] == pytest.approx(expected_center)
        # fov_rad is a field *radius*; the query takes the full field (2 * radius)
        # widened by the profile's cone margin.
        assert calls["fov"] == pytest.approx(
            2 * (metadata["fov_rad"] + instrument.cone_radius_margin)
        )

    def test_batchprep_center_is_resolved_field_center(self, monkeypatch):
        """``BatchPrep.center`` stores the resolved true center, not the header."""
        _, metadata, *_ = _patch_prep(monkeypatch)

        prep = scripts.prepare_batch("frame1.fits", cnn=object())

        expected = scripts.estimate_center_from_header(metadata, InstrumentProfile())
        assert prep.center == pytest.approx(expected)

    def test_falls_back_to_from_name_without_framing_constants(self, monkeypatch):
        """A profile with no framing offset resolves the center by object name."""
        metadata = _batch_metadata()
        metadata["object"] = "SS Leo"
        calls, *_ = _patch_prep(monkeypatch, metadata=metadata)
        resolved = SkyCoord(168.0, 11.0, unit="deg")
        monkeypatch.setattr(scripts.SkyCoord, "from_name", lambda name: resolved)
        instrument = InstrumentProfile(name="NoFraming", header_center_offset=None)

        scripts.prepare_batch(
            "frame1.fits",
            cnn=object(),
            config=PhotometryConfig(instrument=instrument),
        )

        assert calls["center"] == pytest.approx((168.0, 11.0))

    def test_from_name_failure_falls_back_to_raw_header(self, monkeypatch):
        """When object resolution fails the center degrades to the raw header."""
        metadata = _batch_metadata()
        metadata["object"] = "Unresolvable"
        calls, *_ = _patch_prep(monkeypatch, metadata=metadata)

        def _boom(_name):
            msg = "name not resolved"
            raise ValueError(msg)

        monkeypatch.setattr(scripts.SkyCoord, "from_name", _boom)
        instrument = InstrumentProfile(name="NoFraming", header_center_offset=None)

        scripts.prepare_batch(
            "frame1.fits",
            cnn=object(),
            config=PhotometryConfig(instrument=instrument),
        )

        assert calls["center"] == pytest.approx((metadata["ra"], metadata["dec"]))

    def test_absent_object_falls_back_to_raw_header(self, monkeypatch):
        """No ``object`` metadata and no framing constants uses the raw header."""
        # _batch_metadata() carries no "object", so from_name is never attempted.
        calls, metadata, *_ = _patch_prep(monkeypatch)
        instrument = InstrumentProfile(name="NoFraming", header_center_offset=None)

        scripts.prepare_batch(
            "frame1.fits",
            cnn=object(),
            config=PhotometryConfig(instrument=instrument),
        )

        assert calls["center"] == pytest.approx((metadata["ra"], metadata["dec"]))

    def test_obs_epoch_forwarded_to_gaia_query(self, monkeypatch):
        """
        The first frame's ``obs_time`` is forwarded to Gaia as ``obs_epoch``.

        Gaia DR2 positions are J2015.5; without the epoch, ``cached_gaia_radecs``
        returns catalog-epoch positions and every high-proper-motion star is
        mis-placed in the forced-photometry target list. Fixes
        https://github.com/mwcraig/bandaid/issues/56.
        """
        calls, metadata, *_ = _patch_prep(monkeypatch)

        scripts.prepare_batch("frame1.fits", cnn=object())

        assert calls["obs_epoch"] == Time(parser.parse(metadata["obs_time"]))

    def test_non_iso_obs_time_parsed_with_dateutil(self, monkeypatch):
        """
        A dateutil-parseable but non-ISO ``obs_time`` still yields the epoch.

        ``Time()`` alone rejects sloppy header dates like ``2026/04/28``;
        mirroring ``build_photometry_table``'s dateutil parsing keeps the Gaia
        epoch exactly as tolerant as the rest of the pipeline.
        """
        metadata = _batch_metadata()
        metadata["obs_time"] = "2026/04/28 03:03:43"
        calls, *_ = _patch_prep(monkeypatch, metadata=metadata)

        scripts.prepare_batch("frame1.fits", cnn=object())

        assert calls["obs_epoch"] == Time("2026-04-28T03:03:43")

    @pytest.mark.parametrize(
        "bad_obs_time",
        [
            # None: the Seestar profile has no default for obs_time, so a frame
            # missing DATE-OBS resolves it to None. Feeding that to the Gaia query
            # would fail inside the query's "except Exception" wrapper and surface
            # as a misleading "could not query Gaia" BatchPrepError; the metadata
            # must be validated first instead.
            None,
            "not a date at all",
            # dateutil raises OverflowError (not ValueError) for all-digit
            # strings too large for a C long -- e.g. a corrupted numeric
            # DATE-OBS (PR #71 review).
            "999999999999999999",
        ],
        ids=["missing-None", "not-a-date", "overflowing-digits"],
    )
    def test_unparseable_obs_time_raises_clear_metadata_error(
        self, monkeypatch, bad_obs_time
    ):
        """A missing/unparseable ``obs_time`` fails as a metadata error, not Gaia."""
        metadata = _batch_metadata()
        metadata["obs_time"] = bad_obs_time
        _patch_prep(monkeypatch, metadata=metadata)

        with pytest.raises(FrameMetadataError, match="obs_time"):
            scripts.prepare_batch("frame1.fits", cnn=object())

    def test_high_pm_star_propagated_to_obs_epoch(
        self, monkeypatch, gaia_table, fake_vizier
    ):
        """
        End to end, a high-PM star lands at its observation-epoch position.

        Runs the *real* ``cached_gaia_radecs`` with only ``catalog.Vizier``
        patched (network-free). The brightest fixture star is given an extreme
        proper motion (1000 mas/yr, ~10.9 arcsec over 2015.5 -> 2026.3) and the
        faintest a *masked* one. The prep's positions must match an independent
        ``SkyCoord.apply_space_motion`` computation -- so the high-PM star has
        moved well off its raw J2015.5 catalog position and the masked-PM star
        is propagated with zero proper motion. Fixes
        https://github.com/mwcraig/bandaid/issues/56.
        """
        assert fake_vizier is not None  # patching Vizier is the fixture's job
        # An extreme-PM bright star and a masked-PM faint star. Fixture mags
        # (8.8, 13.5, 14.1) are within gaia_mag_limit and the stars are ~arcmin
        # apart, so none are dropped by the mag cut or contamination flagging.
        # NaN sits beneath the mask, as in the real astroquery round-trip
        # (https://github.com/mwcraig/bandaid/issues/80).
        gaia_table["pmRA"] = MaskedColumn(
            [1000.0, -0.957, np.nan], unit=u.mas / u.yr, mask=[False, False, True]
        )
        gaia_table["pmDE"] = MaskedColumn(
            [12.364, -1.993, np.nan], unit=u.mas / u.yr, mask=[False, False, True]
        )

        metadata = _batch_metadata()
        # Point the fake frame at the fixture stars.
        metadata["ra"], metadata["dec"] = 239.9, 25.9
        monkeypatch.setattr(scripts, "N_GAIA_STARS_ALIGN_RETRY", 1)
        monkeypatch.setattr(
            scripts,
            "calibration_sequence",
            lambda file, **_kwargs: (
                np.zeros((4, 4)),
                metadata,
                np.zeros((3, 2)),
                2.0,
                object(),
            ),
        )

        prep = scripts.prepare_batch("frame1.fits", cnn=object())

        # Independent cross-check, masked proper motions treated as zero.
        expected = SkyCoord(
            ra=gaia_table["RA_ICRS"],
            dec=gaia_table["DE_ICRS"],
            pm_ra_cosdec=[1000.0, -0.957, 0.0] * (u.mas / u.yr),
            pm_dec=[12.364, -1.993, 0.0] * (u.mas / u.yr),
            obstime=Time(GAIA_DR2_EPOCH, format="jyear"),
        ).apply_space_motion(new_obstime=Time(parser.parse(metadata["obs_time"])))
        np.testing.assert_allclose(
            prep.radecs[:, 0], expected.ra.deg, rtol=0, atol=1e-9
        )
        np.testing.assert_allclose(
            prep.radecs[:, 1], expected.dec.deg, rtol=0, atol=1e-9
        )

        # The regression guard: the high-PM star is NOT at its raw catalog
        # position -- 1000 mas/yr over ~10.8 yr accumulates ~10.9 arcsec.
        raw = SkyCoord(gaia_table["RA_ICRS"][0], gaia_table["DE_ICRS"][0], unit="deg")
        propagated = SkyCoord(prep.radecs[0, 0], prep.radecs[0, 1], unit="deg")
        assert raw.separation(propagated) > 9 * u.arcsec

        # The masked-PM star stays at its catalog position (zero PM applied).
        np.testing.assert_allclose(
            prep.radecs[2],
            [gaia_table["RA_ICRS"][2], gaia_table["DE_ICRS"][2]],
            rtol=0,
            atol=1e-9,
        )

    def test_contaminated_stars_dropped_from_photometry_coords(self, monkeypatch):
        """The contaminated pair is removed from ``photometry_coords``."""
        _, metadata, radecs, mags, fwhm_pix = _patch_prep(monkeypatch)

        prep = scripts.prepare_batch("frame1.fits", cnn=object())

        fwhm_arcsec = fwhm_pix * metadata["pixscale"]
        flagged = neighbor_contamination_flag_sky(radecs, mags, fwhm_arcsec)
        expected = SkyCoord(radecs[~flagged], unit="deg")

        # The tight equal-mag pair is dropped; the two isolated stars remain.
        assert flagged.tolist() == [True, True, False, False]
        np.testing.assert_allclose(prep.photometry_coords.ra.deg, expected.ra.deg)
        np.testing.assert_allclose(prep.photometry_coords.dec.deg, expected.dec.deg)

    def test_configured_aperture_radius_reaches_contamination_flagging(
        self, monkeypatch
    ):
        """
        The contamination flag is evaluated at ``max(config.apertures.radii)``.

        An equal-magnitude pair placed between the 1-FWHM and 2-FWHM aperture
        contamination thresholds is kept with the default ``radii=(1.0,)`` but
        dropped when the run is configured with ``radii=(2.0,)``: spillover into
        an aperture scales with its area, so a larger aperture needs a larger
        clean separation. The seeing margin is pinned to 1.0 in both runs to
        isolate the radius effect. Fixes
        https://github.com/mwcraig/bandaid/issues/53.
        """
        fwhm_pix = 2.0
        fwhm_arcsec = fwhm_pix * _batch_metadata()["pixscale"]
        sep_r1 = float(min_separation_fwhm(0.0)) * fwhm_arcsec
        sep_r2 = float(min_separation_fwhm(0.0, aperture_radius_fwhm=2.0)) * fwhm_arcsec
        sep_arcsec = 0.5 * (sep_r1 + sep_r2)
        radecs = np.array([[10.0, 0.0], [10.0 + sep_arcsec / 3600.0, 0.0], [10.2, 0.0]])
        mags = np.array([12.0, 12.0, 10.0])
        _patch_prep(monkeypatch, radecs_mags=(radecs, mags), fwhm_pix=fwhm_pix)
        no_margin = InstrumentProfile(contamination_seeing_margin=1.0)

        kept = scripts.prepare_batch(
            "frame1.fits",
            cnn=object(),
            config=PhotometryConfig(instrument=no_margin),
        )
        np.testing.assert_allclose(kept.photometry_coords.ra.deg, radecs[:, 0])

        dropped = scripts.prepare_batch(
            "frame1.fits",
            cnn=object(),
            config=PhotometryConfig(
                apertures=ApertureConfig(radii=(2.0,)), instrument=no_margin
            ),
        )
        np.testing.assert_allclose(dropped.photometry_coords.ra.deg, radecs[[2], 0])

    def test_contamination_seeing_margin_flags_pessimistically(self, monkeypatch):
        """
        The batch flag is evaluated at ``first_frame_fwhm * seeing margin``.

        The flag is computed once, from the first frame's FWHM, and applied all
        night. An equal-magnitude pair placed 15% outside its contamination
        threshold at that FWHM is clean with ``contamination_seeing_margin=1.0``
        but is flagged (and dropped for the whole batch) with a margin of 1.3,
        because seeing only 15% softer than the first frame would contaminate
        it. Fixes https://github.com/mwcraig/bandaid/issues/64.
        """
        fwhm_pix = 2.0
        fwhm_arcsec = fwhm_pix * _batch_metadata()["pixscale"]
        sep_arcsec = 1.15 * float(min_separation_fwhm(0.0)) * fwhm_arcsec
        radecs = np.array([[10.0, 0.0], [10.0 + sep_arcsec / 3600.0, 0.0], [10.2, 0.0]])
        mags = np.array([12.0, 12.0, 10.0])
        _patch_prep(monkeypatch, radecs_mags=(radecs, mags), fwhm_pix=fwhm_pix)

        kept = scripts.prepare_batch(
            "frame1.fits",
            cnn=object(),
            config=PhotometryConfig(
                instrument=InstrumentProfile(contamination_seeing_margin=1.0)
            ),
        )
        np.testing.assert_allclose(kept.photometry_coords.ra.deg, radecs[:, 0])

        flagged = scripts.prepare_batch(
            "frame1.fits",
            cnn=object(),
            config=PhotometryConfig(
                instrument=InstrumentProfile(contamination_seeing_margin=1.3)
            ),
        )
        np.testing.assert_allclose(flagged.photometry_coords.ra.deg, radecs[[2], 0])

    @pytest.mark.parametrize(
        ("gaia_mag_limit", "n_kept"),
        [
            # Default limit of 15 cuts 15.1/16.0 but keeps 15.0 itself.
            (None, 2),
            # An explicit limit cuts at that magnitude instead.
            (12.0, 1),
        ],
        ids=["default-limit-15", "custom-limit-12"],
    )
    def test_gaia_mag_limit_drops_faint_stars(
        self, monkeypatch, gaia_mag_limit, n_kept
    ):
        """Stars fainter than the (default or explicit) Gaia mag limit are cut."""
        radecs = np.array([[10.0, 0.0], [10.1, 0.0], [10.2, 0.0], [10.3, 0.0]])
        mags = np.array([12.0, 15.0, 15.1, 16.0])
        _patch_prep(monkeypatch, radecs_mags=(radecs, mags))

        config_kwargs = (
            {}
            if gaia_mag_limit is None
            else {
                "config": PhotometryConfig(
                    source_selection=SourceSelectionConfig(
                        gaia_mag_limit=gaia_mag_limit
                    )
                )
            }
        )
        prep = scripts.prepare_batch("frame1.fits", cnn=object(), **config_kwargs)

        np.testing.assert_array_equal(prep.radecs, radecs[:n_kept])
        # The kept stars are degrees apart, so none are contamination-flagged.
        np.testing.assert_allclose(prep.photometry_coords.ra.deg, radecs[:n_kept, 0])

    def test_faint_real_star_contaminates_brighter_target(self, monkeypatch):
        """
        A real star fainter than the photometry limit still flags a brighter target.

        The mag-16 star sits ~1 arcsec from the mag-14 star -- well inside the
        ~7.5 arcsec the contamination model requires for that pair at this FWHM. It
        is fainter than the photometry limit of 15, so it is *not* a photometry
        target, but it is within the default contaminant limit (gaia_mag_limit + 3
        = 18), so it still contaminates the mag-14 target. The mag-14 star is
        therefore flagged and dropped from ``photometry_coords``; only the
        isolated mag-10 star survives. ``radecs`` (the alignment catalog) keeps
        both targets regardless of contamination. Fixes
        https://github.com/mwcraig/bandaid/issues/24.
        """
        radecs = np.array([[10.0, 0.0], [10.0 + 1.0 / 3600.0, 0.0], [10.2, 0.0]])
        mags = np.array([14.0, 16.0, 10.0])
        _patch_prep(monkeypatch, radecs_mags=(radecs, mags))

        prep = scripts.prepare_batch("frame1.fits", cnn=object())

        # Targets (mag <= 15) are the mag-14 and mag-10 stars; both stay in radecs.
        np.testing.assert_array_equal(prep.radecs, radecs[[0, 2]])
        # The mag-14 target is now flagged by the faint mag-16 neighbor, leaving
        # only the far mag-10 star.
        np.testing.assert_allclose(prep.photometry_coords.ra.deg, radecs[[2], 0])

    def test_contaminant_mag_offset_bounds_the_flagging_catalog(self, monkeypatch):
        """
        ``contaminant_mag_offset`` caps which faint stars can flag a target.

        Same close pair as ``test_faint_real_star_contaminates_brighter_target``,
        but a small ``contaminant_mag_offset=0.5`` shrinks the contaminant limit to
        ``gaia_mag_limit + 0.5 = 15.5``, which excludes the mag-16 neighbor from the
        contaminant catalog entirely, so the mag-14 target is no longer flagged and
        survives into ``photometry_coords``.
        """
        radecs = np.array([[10.0, 0.0], [10.0 + 1.0 / 3600.0, 0.0], [10.2, 0.0]])
        mags = np.array([14.0, 16.0, 10.0])
        _patch_prep(monkeypatch, radecs_mags=(radecs, mags))

        prep = scripts.prepare_batch(
            "frame1.fits",
            cnn=object(),
            config=PhotometryConfig(
                source_selection=SourceSelectionConfig(contaminant_mag_offset=0.5),
            ),
        )

        np.testing.assert_array_equal(prep.radecs, radecs[[0, 2]])
        np.testing.assert_allclose(prep.photometry_coords.ra.deg, radecs[[0, 2], 0])

    def test_nan_magnitude_dropped_by_mag_limit(self, monkeypatch):
        """A star with no Gaia magnitude fails the cut and is dropped entirely."""
        radecs = np.array([[10.0, 0.0], [10.1, 0.0], [10.2, 0.0]])
        mags = np.array([12.0, np.nan, 10.0])
        _patch_prep(monkeypatch, radecs_mags=(radecs, mags))

        prep = scripts.prepare_batch("frame1.fits", cnn=object())

        np.testing.assert_array_equal(prep.radecs, radecs[[0, 2]])

    def test_raises_when_too_few_stars_detected(self, monkeypatch):
        """A first-frame TooFewStarsError becomes a fatal BatchPrepError."""

        def _too_few(file, **_kwargs: object):
            msg = "only 1 stars detected"
            raise TooFewStarsError(msg, file=file)

        monkeypatch.setattr(scripts, "calibration_sequence", _too_few)
        with pytest.raises(BatchPrepError, match="too few stars"):
            scripts.prepare_batch("frame1.fits", cnn=object())

    def test_empty_gaia_field_raises_batchpreperror(self, monkeypatch):
        """An empty Gaia cone is fatal -- no reference stars to solve any WCS."""
        _patch_prep(monkeypatch, radecs_mags=(np.empty((0, 2)), np.empty(0)))
        # Use the real floor, not _patch_prep's relaxed one, for the guard.
        monkeypatch.setattr(scripts, "N_GAIA_STARS_ALIGN_RETRY", 20)
        with pytest.raises(BatchPrepError, match="Gaia returned only 0"):
            scripts.prepare_batch("frame1.fits", cnn=object())

    def test_sparse_gaia_field_raises_batchpreperror(self, monkeypatch):
        """Fewer than N_GAIA_STARS_ALIGN_RETRY references is fatal for the batch."""
        radecs = np.column_stack([np.linspace(9.0, 11.0, 5), np.zeros(5)])
        _patch_prep(monkeypatch, radecs_mags=(radecs, np.full(5, 12.0)))
        monkeypatch.setattr(scripts, "N_GAIA_STARS_ALIGN_RETRY", 20)
        with pytest.raises(BatchPrepError, match="Gaia returned only 5"):
            scripts.prepare_batch("frame1.fits", cnn=object())

    def test_gaia_network_error_raises_batchpreperror(self, monkeypatch):
        """A Gaia query failure is surfaced as a fatal BatchPrepError."""
        monkeypatch.setattr(
            scripts,
            "calibration_sequence",
            lambda file, *, cnn=None, **_kwargs: (
                np.zeros((4, 4)),
                _batch_metadata(),
                None,
                2.0,
                object(),
            ),
        )

        def _boom(*_args: object, **_kwargs: object):
            msg = "no network"
            raise ConnectionError(msg)

        monkeypatch.setattr(scripts, "cached_gaia_radecs", _boom)
        with pytest.raises(BatchPrepError, match="could not query Gaia"):
            scripts.prepare_batch("frame1.fits", cnn=object())


class TestCheckFrameConsistency:
    """Unit tests for the per-frame pointing/shape guard."""

    # The true field center for a Seestar frame pointing at RA=10/DEC=0: the raw
    # header walked by the header_center_offset (-0.32, +0.15). prep.center now
    # holds this resolved center, so a stable frame reads ~0 offset against it.
    STABLE_CENTER = (9.68, 0.15)

    @staticmethod
    def _prep(**overrides: object) -> scripts.BatchPrep:
        """A BatchPrep carrying consistency fields, overridable per test."""
        fields = {
            "center": TestCheckFrameConsistency.STABLE_CENTER,
            "fov_rad": 0.74,
            "shape": (1920, 1080),
        }
        fields.update(overrides)
        return scripts.BatchPrep(
            radecs=np.zeros((1, 2)),
            photometry_coords=SkyCoord([0.0], [0.0], unit="deg"),
            cnn=object(),
            bayer_masks={},
            **fields,
        )

    def test_consistent_frame_passes(self):
        """A frame matching the prep's shape and pointing is accepted."""
        header = _consistency_header()
        scripts.check_frame_consistency("ok.fits", header, self._prep())

    def test_shape_mismatch_raises_frameerror(self):
        """A different image shape is rejected."""
        header = _consistency_header(NAXIS1=1000)
        with pytest.raises(FrameError, match="shape"):
            scripts.check_frame_consistency("bad.fits", header, self._prep())

    def test_offfield_pointing_raises_frameerror(self):
        """A frame pointing beyond the field radius is rejected."""
        header = _consistency_header(RA=12.0)
        with pytest.raises(FrameError, match="pointing"):
            scripts.check_frame_consistency("bad.fits", header, self._prep())

    def test_drifted_frame_within_radius_accepted(self):
        """
        A frame drifted <1 field radius from the true center is accepted.

        The check compares the frame's *estimated* field center to the prep's
        true center, so the ~0.35 deg header-to-center baseline no longer eats
        into the drift margin. A frame whose header moved 0.5 deg (well inside
        the 0.74 deg radius) is kept, where comparing the raw header against the
        true center would have falsely rejected it.
        """
        # RA=10.5 -> estimated center (10.18, 0.15), 0.5 deg from STABLE_CENTER.
        header = _consistency_header(RA=10.5)
        scripts.check_frame_consistency("ok.fits", header, self._prep())

    def test_missing_keyword_raises_metadata_error(self):
        """A header missing a needed keyword is a metadata error."""
        header = _consistency_header()
        del header["NAXIS2"]
        with pytest.raises(FrameMetadataError):
            scripts.check_frame_consistency("bad.fits", header, self._prep())

    def test_missing_pointing_raises_metadata_error(self):
        """A header whose dialect resolves no pointing is a metadata error."""
        # "@RA"/"@DEC" lookups on a header without those keywords resolve to
        # None rather than raising, so the guard must catch the None itself.
        header = _consistency_header()
        del header["RA"]
        del header["DEC"]
        with pytest.raises(FrameMetadataError, match="pointing"):
            scripts.check_frame_consistency("bad.fits", header, self._prep())

    def test_header_map_routes_pointing_keys(self):
        """Pointing under renamed keywords resolves through the profile (#59)."""
        # A dialect whose pointing lives under OBJCTRA/OBJCTDEC, with no
        # RA/DEC in the header at all: the check must consult the header_map,
        # not the raw Seestar keywords.
        custom_map = {
            **dict(InstrumentProfile().header_map),
            "ra": "@OBJCTRA",
            "dec": "@OBJCTDEC",
        }
        profile = InstrumentProfile(name="Renamed", header_map=custom_map)
        prep = self._prep(config=PhotometryConfig(instrument=profile))
        header = _consistency_header(OBJCTRA=10.0, OBJCTDEC=0.0)
        del header["RA"]
        del header["DEC"]

        # In-field frame passes...
        scripts.check_frame_consistency("ok.fits", header, prep)

        # ...and the off-field rejection still fires on the mapped keywords.
        header["OBJCTRA"] = 50.0
        with pytest.raises(FrameError, match="pointing"):
            scripts.check_frame_consistency("bad.fits", header, prep)

    def test_inconsistent_frame_is_skipped_by_batch(self, monkeypatch):
        """process_batch skips an off-field frame and keeps the good one."""
        prep = self._prep()

        def _header(file):
            ra = 10.0 if file == "good.fits" else 50.0
            return _consistency_header(RA=ra)

        monkeypatch.setattr(scripts.fits, "getheader", _header)
        monkeypatch.setattr(
            scripts,
            "process_one_image",
            lambda *a, **k: {"TR": Table({"tot_count": [1.0]})},
        )
        results = scripts.process_batch(
            ["good.fits", "bad.fits"],
            prep,
            user_specific_metadata={},
        )
        assert list(results) == ["good.fits"]
