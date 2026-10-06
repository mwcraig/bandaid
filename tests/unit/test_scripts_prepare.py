"""Unit tests for once-per-batch preparation and frame-consistency checks."""

import csv

import astropy.units as u
import numpy as np
import pytest
from _helpers import (
    SEESTAR_RULE,
    _batch_metadata,
    _batch_radecs_mags,
    _consistency_header,
    _patch_prep,
    _stub_load_frame,
)
from astropy.coordinates import SkyCoord
from astropy.table import MaskedColumn, Table
from astropy.time import Time
from dateutil import parser

from bandaid import instruments, scripts
from bandaid.catalog import GAIA_DR2_EPOCH
from bandaid.config import (
    ApertureConfig,
    HeaderMatchRule,
    InstrumentProfile,
    PhotometryConfig,
    SourceSelectionConfig,
)
from bandaid.exceptions import (
    BatchPrepError,
    CatalogTruncationError,
    FrameError,
    FrameMetadataError,
    InstrumentDetectionError,
    TooFewStarsError,
)
from bandaid.instruments import load_instrument, register_instrument
from bandaid.photometry import (
    min_separation_fwhm,
    neighbor_contamination_flag_sky,
)

# FK5(J2000) differs from ICRS by well under a tenth of an arcsecond; 25 years of
# precession moves a pointing by about a third of a degree.
FRAME_TIE_ARCSEC = 0.1
PRECESSION_MIN_DEG = 0.3
ROW_LIMIT = 1234
N_POOL_STARS = 20
N_EDGE_STARS = 5


class TestEstimateCenterFromHeader:
    """Unit tests for ``estimate_center_from_header``."""

    # A profile whose header pointing is in the equinox of the observation date,
    # like the bundled Seestar50.
    OF_DATE = InstrumentProfile(header_frame="fk5", header_equinox="date")

    def test_icrs_header_is_returned_unchanged(self):
        """An ICRS header pointing passes through unconverted."""
        profile = InstrumentProfile(header_frame="icrs")
        metadata = {"ra": 10.0, "dec": 30.0, "obs_time": "2026-04-28T03:03:43"}

        assert scripts.estimate_center_from_header(metadata, profile) == (10.0, 30.0)

    def test_default_profile_is_icrs(self):
        """A bare profile treats the header as ICRS, so nothing is converted."""
        metadata = {"ra": 10.0, "dec": 30.0}

        center = scripts.estimate_center_from_header(metadata, InstrumentProfile())

        assert center == (10.0, 30.0)

    def test_icrs_ra_is_wrapped_into_range(self):
        """An out-of-range ICRS header RA is wrapped into [0, 360)."""
        ra, dec = scripts.estimate_center_from_header(
            {"ra": -0.22, "dec": 0.0}, InstrumentProfile()
        )

        assert ra == pytest.approx(359.78)
        assert dec == 0.0

    @pytest.mark.parametrize(
        ("ra", "dec", "obs_time", "expected_shift"),
        [
            # (Delta(RA*cos(dec)), Delta(dec)) in degrees from the header pointing
            # to the plate-solved field center, as predicted by precession for
            # fields spread around the sky.
            (26.6833, 13.0547, "2025-09-09T03:00:00", (-0.335, -0.128)),
            (172.45, 30.07, "2026-03-15T05:00:00", (-0.302, 0.145)),
            (157.6, 70.5, "2026-03-15T05:00:00", (-0.164, 0.134)),
        ],
        ids=["LS Psc", "TU UMa", "Qatar-8"],
    )
    def test_of_date_header_is_precessed_to_icrs(
        self, ra, dec, obs_time, expected_shift
    ):
        """A header in the equinox of date is converted back to ICRS."""
        metadata = {"ra": ra, "dec": dec, "obs_time": obs_time}

        center_ra, center_dec = scripts.estimate_center_from_header(
            metadata, self.OF_DATE
        )

        d_ra_cosdec = (center_ra - ra) * np.cos(np.radians(dec))
        assert d_ra_cosdec == pytest.approx(expected_shift[0], abs=0.01)
        assert center_dec - dec == pytest.approx(expected_shift[1], abs=0.01)

    def test_fk5_j2000_header_moves_only_by_the_frame_tie(self):
        """An FK5 J2000 header differs from ICRS by well under an arcsecond."""
        profile = InstrumentProfile(header_frame="fk5", header_equinox="J2000")

        ra, dec = scripts.estimate_center_from_header(
            {"ra": 10.0, "dec": 30.0}, profile
        )

        moved = SkyCoord(ra, dec, unit="deg").separation(
            SkyCoord(10.0, 30.0, unit="deg")
        )
        assert moved.arcsec < FRAME_TIE_ARCSEC

    def test_fixed_equinox_ignores_the_observation_time(self):
        """A fixed header equinox gives one answer whatever (or no) obs_time."""
        profile = InstrumentProfile(header_frame="fk5", header_equinox="J2025.5")
        pointing = {"ra": 10.0, "dec": 30.0}

        without_time = scripts.estimate_center_from_header(pointing, profile)
        with_time = scripts.estimate_center_from_header(
            {**pointing, "obs_time": "2010-01-01T00:00:00"}, profile
        )

        assert with_time == pytest.approx(without_time)
        # 25.5 years of precession is a third of a degree, so the conversion
        # was applied.
        moved = SkyCoord(*without_time, unit="deg").separation(
            SkyCoord(10.0, 30.0, unit="deg")
        )
        assert moved.deg > PRECESSION_MIN_DEG

    @pytest.mark.parametrize(
        "time_fields",
        [{}, {"obs_time": None}, {"obs_time": "not-a-date"}],
        ids=["obs_time-missing", "obs_time-None", "obs_time-unparsable"],
    )
    def test_of_date_header_needs_an_observation_time(self, time_fields):
        """Without a usable observation time an of-date header cannot be converted."""
        metadata = {"ra": 10.0, "dec": 30.0, **time_fields}

        with pytest.raises(FrameMetadataError, match="obs_time"):
            scripts.estimate_center_from_header(metadata, self.OF_DATE)

    def test_string_pointing_is_coerced_to_float(self):
        """A numeric-string header pointing (the raw @RA/@DEC form) is coerced."""
        # @RA/@DEC pass the header value through untouched, so it often arrives
        # as a numeric string.
        metadata = {"ra": "10.0", "dec": "0.0"}

        center = scripts.estimate_center_from_header(metadata, InstrumentProfile())

        assert center == (10.0, 0.0)

    @pytest.mark.parametrize(
        "metadata",
        [
            {"dec": 0.0},  # ra key absent -> KeyError
            {"ra": None, "dec": 0.0},  # @RA absent resolves to None -> TypeError
            {"ra": "not-a-number", "dec": 0.0},  # present but junk -> ValueError
        ],
        ids=["ra-missing", "ra-None", "ra-non-numeric"],
    )
    def test_bad_pointing_raises_metadata_error(self, metadata):
        """
        A missing/non-numeric pointing fails as a metadata error, not a bare one.

        The coercion is the single choke point every center path funnels through
        (``prepare_batch`` and ``check_frame_consistency``), so translating the
        failure here keeps the recoverable/fatal error semantics and per-frame
        labelling consistent (#58/#78) instead of leaking a bare
        ``KeyError``/``TypeError``/``ValueError``.
        """
        with pytest.raises(FrameMetadataError, match="pointing"):
            scripts.estimate_center_from_header(metadata, InstrumentProfile())

    @pytest.mark.parametrize(
        "profile",
        [InstrumentProfile(), OF_DATE],
        ids=["icrs", "fk5-of-date"],
    )
    def test_out_of_range_dec_raises_metadata_error(self, profile):
        """A declination outside [-90, 90] is a metadata error in either frame."""
        metadata = {"ra": 10.0, "dec": 91.0, "obs_time": "2026-04-28T03:03:43"}

        with pytest.raises(FrameMetadataError, match="pointing"):
            scripts.estimate_center_from_header(metadata, profile)

    @pytest.mark.parametrize(
        ("ra", "dec"),
        [
            (np.nan, 20.0),
            (np.inf, 20.0),
            (-np.inf, 20.0),
            (10.0, np.nan),
            ("nan", "20.0"),
        ],
        ids=["ra-nan", "ra-inf", "ra-minus-inf", "dec-nan", "ra-nan-string"],
    )
    @pytest.mark.parametrize(
        "profile",
        [InstrumentProfile(), OF_DATE],
        ids=["icrs", "fk5-of-date"],
    )
    def test_non_finite_pointing_raises_metadata_error(self, profile, ra, dec):
        """A NaN or infinite pointing is a metadata error in either frame."""
        metadata = {"ra": ra, "dec": dec, "obs_time": "2026-04-28T03:03:43"}

        with pytest.raises(FrameMetadataError, match="pointing"):
            scripts.estimate_center_from_header(metadata, profile)

    def test_precession_across_ra_zero_stays_in_range(self):
        """A field near RA 0h that crosses 0 under precession wraps into [0, 360)."""
        metadata = {"ra": 0.1, "dec": 0.0, "obs_time": "2026-04-28T03:03:43"}

        ra, _dec = scripts.estimate_center_from_header(metadata, self.OF_DATE)

        # 0.1 deg minus ~0.34 deg of precession would be negative unwrapped.
        assert ra == pytest.approx(359.76, abs=0.01)


class TestPrepareBatch:
    """Unit tests for ``prepare_batch``."""

    def test_returns_batchprep_with_expected_fields(self, mocker):
        """The bundle carries the Gaia list, the cnn, and the three CFA masks."""
        prep_data = _patch_prep(mocker)
        cnn = object()

        # append_l4 defaults to True (issue #61); pin it False here so this
        # test's "three CFA masks" stays decoupled from that default.
        prep = scripts.prepare_batch("frame1.fits", cnn=cnn, append_l4=False)

        assert isinstance(prep, scripts.BatchPrep)
        np.testing.assert_array_equal(prep.radecs, prep_data.radecs)
        assert prep.cnn is cnn
        assert set(prep.bayer_masks) == {"TR", "TB", "TG"}

    def test_append_l4_true_by_default(self, mocker):
        """Omitting ``append_l4`` adds the L4 channel (issue #61)."""
        _patch_prep(mocker)

        prep = scripts.prepare_batch("frame1.fits", cnn=object())

        assert set(prep.bayer_masks) == {"TR", "TB", "TG", "L4"}
        assert prep.bayer_masks["L4"] is None

    def test_loads_first_frame_exactly_once(self, mocker):
        """Without a caller-provided frame, the first frame is loaded once (#44)."""
        _patch_prep(mocker)
        # A default config resolves its instrument by detection, so the frame
        # needs a detectable header (not this test's concern -- it only pins
        # that _load_frame is called exactly once).
        frame = scripts.LoadedFrame(np.zeros((4, 4)), {"INSTRUME": "Seestar S50"})
        load_frame = mocker.patch("bandaid.scripts._load_frame", return_value=frame)

        scripts.prepare_batch("frame1.fits", cnn=object())

        load_frame.assert_called_once_with("frame1.fits")

    def test_provided_frame_is_used_without_loading(self, mocker):
        """
        A caller-provided ``frame=`` skips the load entirely (#44).

        ``photometer_frames`` opens the first frame once and hands the load to
        both ``prepare_batch`` and ``process_batch``, so a provided frame must
        reach the calibration step without ``_load_frame`` being called.
        """
        prep_data = _patch_prep(mocker)
        # A default config resolves its instrument by detection, so the frame
        # needs a detectable header (not this test's concern -- it only pins
        # that the provided frame skips _load_frame).
        frame = scripts.LoadedFrame(np.zeros((4, 4)), {"INSTRUME": "Seestar S50"})

        def fail_load_frame(file):
            msg = f"unexpected _load_frame({file!r}) with frame= provided"
            raise AssertionError(msg)

        mocker.patch("bandaid.scripts._load_frame", side_effect=fail_load_frame)

        scripts.prepare_batch("frame1.fits", cnn=object(), frame=frame)

        assert prep_data.calibration_sequence.call_args.kwargs["frame"] is frame

    def test_auto_detects_instrument_from_first_frame_header(self, mocker):
        """
        A default (``instrument=None``) config resolves by detecting the header.

        ``prepare_batch`` is the first place a header is in hand, so it must
        call ``detect_instrument`` and carry the resolved profile forward on
        the config it hands to ``calibration_sequence`` and stores on the
        returned ``BatchPrep``. ``instrument_auto_detected`` records that the
        resolution came from detection, not an explicit choice --
        ``check_frame_consistency``'s batch-mixing guard reads this flag.
        """
        prep_data = _patch_prep(mocker)
        frame = scripts.LoadedFrame(np.zeros((4, 4)), {"INSTRUME": "Seestar S50"})

        prep = scripts.prepare_batch(
            "frame1.fits", cnn=object(), config=PhotometryConfig(), frame=frame
        )

        resolved = prep_data.calibration_sequence.call_args.kwargs["profile"]
        assert resolved.name == "Seestar50"
        assert prep.config.instrument.name == "Seestar50"
        assert prep.instrument_auto_detected is True

    def test_explicit_instrument_is_not_marked_auto_detected(self, mocker):
        """An explicit instrument records ``instrument_auto_detected=False``."""
        _patch_prep(mocker)
        instrument = InstrumentProfile(name="MyScope")

        prep = scripts.prepare_batch(
            "frame1.fits",
            cnn=object(),
            config=PhotometryConfig(instrument=instrument),
        )

        assert prep.instrument_auto_detected is False

    def test_unmatched_first_frame_header_raises_batch_prep_error(self, mocker):
        """
        A first frame whose header matches no profile fails the whole batch.

        ``InstrumentDetectionError`` is now a `FrameMetadataError` (recoverable
        per-frame) by declaration, so `prepare_batch` --
        the one caller that actually wants the fatal behavior, since an
        unresolvable first frame leaves no detection/PSF settings to prepare
        the batch with -- wraps it into the batch-fatal `BatchPrepError`
        itself instead of relying on inheritance.
        """
        _patch_prep(mocker)
        frame = scripts.LoadedFrame(np.zeros((4, 4)), {})

        with pytest.raises(BatchPrepError) as exc_info:
            scripts.prepare_batch(
                "frame1.fits", cnn=object(), config=PhotometryConfig(), frame=frame
            )

        assert isinstance(exc_info.value.__cause__, InstrumentDetectionError)

    def test_first_frame_resolved_with_config_instrument_profile(self, mocker):
        """
        The config's instrument is threaded into the first-frame calibration.

        Without this, ``prepare_batch`` would resolve the first frame's metadata
        with the bundled-Seestar50 fallback rather than ``config.instrument`` --
        wrong for any other telescope.
        """
        prep_data = _patch_prep(mocker)
        instrument = InstrumentProfile(name="MyScope")
        scripts.prepare_batch(
            "frame1.fits",
            cnn=object(),
            config=PhotometryConfig(instrument=instrument),
        )
        assert prep_data.calibration_sequence.call_args.kwargs["profile"] is instrument

    def test_first_frame_fwhm_uses_the_per_frame_detection_settings(self, mocker):
        """
        The batch-gating FWHM is measured with the per-frame detection settings.

        ``prepare_batch`` must hand ``calibration_sequence`` the same
        ``detect_on_bayer_balanced=True`` and ``fwhm_n_stars`` that every
        per-frame call uses (``photometry.process_one_image``); otherwise the
        FWHM that sizes the contamination radii is measured in a different
        detection regime than the photometry it protects. Fixes
        https://github.com/mwcraig/bandaid/issues/55.
        """
        fwhm_n_stars = 7

        def _stop_after_capturing(_file, **_kwargs: object):
            msg = "stop after capturing the call"
            raise TooFewStarsError(msg)

        calibration_sequence = mocker.patch(
            "bandaid.scripts.calibration_sequence", side_effect=_stop_after_capturing
        )
        _stub_load_frame(mocker)

        config = PhotometryConfig(
            instrument=InstrumentProfile(name="Seestar50", fwhm_n_stars=fwhm_n_stars)
        )
        with pytest.raises(BatchPrepError):
            scripts.prepare_batch("first.fits", cnn=object(), config=config)

        captured = calibration_sequence.call_args.kwargs
        assert captured.get("detect_on_bayer_balanced") is True
        # fwhm_n_stars is left unset so calibration_sequence resolves it from
        # the profile, exactly as the per-frame call does.
        assert captured.get("fwhm_n_stars") is None
        assert captured["profile"].fwhm_n_stars == fwhm_n_stars

    def test_gaia_queried_at_resolved_center_with_cone_margin(self, mocker):
        """
        Gaia is queried at the ICRS field center with the instrument's margin.

        The Seestar header pointing is in the equinox of date, so the cone is
        centered on that pointing converted to ICRS (not the raw header). The
        field radius, cone margin, magnitude limits and row limit all come from
        the instrument and config.
        """
        prep_data = _patch_prep(mocker)
        instrument = load_instrument("Seestar50").model_copy(
            update={"cone_radius_margin": 0.25}
        )
        config = PhotometryConfig(
            instrument=instrument,
            source_selection=SourceSelectionConfig(
                gaia_mag_limit=14.5,
                contaminant_mag_offset=2.0,
                gaia_row_limit=ROW_LIMIT,
            ),
        )
        scripts.prepare_batch("frame1.fits", cnn=object(), config=config)

        expected_center = scripts.estimate_center_from_header(
            prep_data.metadata, instrument
        )
        call = prep_data.query_field_catalog.call_args
        center, fov_rad = call.args
        assert center == pytest.approx(expected_center)
        # The precession since J2000 is about a third of a degree, so the raw
        # header pointing would not satisfy the assertion above.
        raw = SkyCoord(prep_data.metadata["ra"], prep_data.metadata["dec"], unit="deg")
        assert SkyCoord(*center, unit="deg").separation(raw).deg > PRECESSION_MIN_DEG
        assert fov_rad == pytest.approx(prep_data.metadata["fov_rad"])
        assert call.kwargs["cone_margin"] == pytest.approx(0.25)
        assert call.kwargs["gaia_mag_limit"] == pytest.approx(14.5)
        assert call.kwargs["contaminant_mag_limit"] == pytest.approx(16.5)
        assert call.kwargs["row_limit"] == ROW_LIMIT
        assert call.kwargs["obs_epoch"] is not None

    def test_batchprep_center_is_resolved_field_center(self, mocker):
        """``BatchPrep.center`` stores the ICRS field center, not the header."""
        prep_data = _patch_prep(mocker)

        prep = scripts.prepare_batch("frame1.fits", cnn=object())

        expected = scripts.estimate_center_from_header(
            prep_data.metadata, load_instrument("Seestar50")
        )
        assert prep.center == pytest.approx(expected)

    def test_icrs_profile_centers_on_raw_header(self, mocker):
        """A profile with an ICRS header queries Gaia at the header pointing."""
        prep_data = _patch_prep(mocker)
        instrument = InstrumentProfile(name="IcrsHeader")

        scripts.prepare_batch(
            "frame1.fits",
            cnn=object(),
            config=PhotometryConfig(instrument=instrument),
        )

        center, _fov = prep_data.query_field_catalog.call_args.args
        assert center == pytest.approx(
            (prep_data.metadata["ra"], prep_data.metadata["dec"])
        )

    def test_center_never_resolves_the_object_name(self, mocker):
        """The field center comes from the header; the object name is not looked up."""
        metadata = _batch_metadata()
        metadata["object"] = "SS Leo"
        prep_data = _patch_prep(mocker, metadata=metadata)
        from_name = mocker.patch.object(scripts.SkyCoord, "from_name")

        scripts.prepare_batch("frame1.fits", cnn=object())

        from_name.assert_not_called()
        center, _fov = prep_data.query_field_catalog.call_args.args
        assert center == pytest.approx(
            scripts.estimate_center_from_header(metadata, load_instrument("Seestar50"))
        )

    def test_obs_epoch_forwarded_to_gaia_query(self, mocker):
        """
        The first frame's ``obs_time`` is forwarded to Gaia as ``obs_epoch``.

        Gaia DR2 positions are J2015.5; without the epoch, ``query_field_catalog``
        returns catalog-epoch positions and every high-proper-motion star is
        mis-placed in the forced-photometry target list. Fixes
        https://github.com/mwcraig/bandaid/issues/56.
        """
        prep_data = _patch_prep(mocker)

        scripts.prepare_batch("frame1.fits", cnn=object())

        obs_epoch = prep_data.query_field_catalog.call_args.kwargs["obs_epoch"]
        assert obs_epoch == Time(parser.parse(prep_data.metadata["obs_time"]))

    def test_non_iso_obs_time_parsed_with_dateutil(self, mocker):
        """
        A dateutil-parseable but non-ISO ``obs_time`` still yields the epoch.

        ``Time()`` alone rejects sloppy header dates like ``2026/04/28``;
        mirroring ``build_photometry_table``'s dateutil parsing keeps the Gaia
        epoch exactly as tolerant as the rest of the pipeline.
        """
        metadata = _batch_metadata()
        metadata["obs_time"] = "2026/04/28 03:03:43"
        prep_data = _patch_prep(mocker, metadata=metadata)

        scripts.prepare_batch("frame1.fits", cnn=object())

        obs_epoch = prep_data.query_field_catalog.call_args.kwargs["obs_epoch"]
        assert obs_epoch == Time("2026-04-28T03:03:43")

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
    def test_unparsable_obs_time_raises_clear_metadata_error(
        self, mocker, bad_obs_time
    ):
        """A missing/unparsable ``obs_time`` fails as a metadata error, not Gaia."""
        metadata = _batch_metadata()
        metadata["obs_time"] = bad_obs_time
        _patch_prep(mocker, metadata=metadata)

        with pytest.raises(FrameMetadataError, match="obs_time"):
            scripts.prepare_batch("frame1.fits", cnn=object())

    def test_bad_pointing_labeled_with_first_file(self, mocker):
        """
        A bad first-frame pointing fails as a metadata error naming the frame.

        ``estimate_center_from_header`` knows the pointing is bad but not which file
        it came from; ``prepare_batch`` attaches ``first_file`` before re-raising,
        matching the ``metadata_from_header``/``obs_time`` labelling right above
        the call so the failure is actionable.
        """
        metadata = _batch_metadata()
        metadata["ra"] = "not-a-number"
        _patch_prep(mocker, metadata=metadata)

        with pytest.raises(FrameMetadataError, match="pointing") as excinfo:
            scripts.prepare_batch("frame1.fits", cnn=object())
        assert excinfo.value.file == "frame1.fits"

    def test_high_pm_star_propagated_to_obs_epoch(
        self, mocker, gaia_table, fake_vizier
    ):
        """
        End to end, a high-PM star lands at its observation-epoch position.

        Runs the *real* ``query_field_catalog`` with only ``catalog.Vizier``
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
        mocker.patch("bandaid.scripts.N_GAIA_STARS_ALIGN_RETRY", 1)
        mocker.patch(
            "bandaid.scripts.calibration_sequence",
            return_value=(
                np.zeros((4, 4)),
                metadata,
                np.zeros((3, 2)),
                2.0,
                object(),
            ),
        )
        _stub_load_frame(mocker)

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

    def test_contaminated_stars_dropped_from_photometry_coords(self, mocker):
        """The contaminated pair is removed from ``photometry_coords``."""
        prep_data = _patch_prep(mocker)

        prep = scripts.prepare_batch("frame1.fits", cnn=object())

        fwhm_arcsec = prep_data.fwhm_pix * prep_data.metadata["pixscale"]
        flagged = neighbor_contamination_flag_sky(
            prep_data.radecs, prep_data.mags, fwhm_arcsec
        )
        expected = SkyCoord(prep_data.radecs[~flagged], unit="deg")

        # The tight equal-mag pair is dropped; the two isolated stars remain.
        assert flagged.tolist() == [True, True, False, False]
        np.testing.assert_allclose(prep.photometry_coords.ra.deg, expected.ra.deg)
        np.testing.assert_allclose(prep.photometry_coords.dec.deg, expected.dec.deg)

    def test_configured_aperture_radius_reaches_contamination_flagging(self, mocker):
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
        _patch_prep(mocker, radecs_mags=(radecs, mags), fwhm_pix=fwhm_pix)
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

    def test_contamination_seeing_margin_flags_pessimistically(self, mocker):
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
        _patch_prep(mocker, radecs_mags=(radecs, mags), fwhm_pix=fwhm_pix)

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
    def test_gaia_mag_limit_drops_faint_stars(self, mocker, gaia_mag_limit, n_kept):
        """Stars fainter than the (default or explicit) Gaia mag limit are cut."""
        radecs = np.array([[10.0, 0.0], [10.1, 0.0], [10.2, 0.0], [10.3, 0.0]])
        mags = np.array([12.0, 15.0, 15.1, 16.0])
        _patch_prep(mocker, radecs_mags=(radecs, mags))

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

    def test_faint_real_star_contaminates_brighter_target(self, mocker):
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
        _patch_prep(mocker, radecs_mags=(radecs, mags))

        prep = scripts.prepare_batch("frame1.fits", cnn=object())

        # Targets (mag <= 15) are the mag-14 and mag-10 stars; both stay in radecs.
        np.testing.assert_array_equal(prep.radecs, radecs[[0, 2]])
        # The mag-14 target is now flagged by the faint mag-16 neighbor, leaving
        # only the far mag-10 star.
        np.testing.assert_allclose(prep.photometry_coords.ra.deg, radecs[[2], 0])

    def test_contaminant_mag_offset_bounds_the_flagging_catalog(self, mocker):
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
        _patch_prep(mocker, radecs_mags=(radecs, mags))

        prep = scripts.prepare_batch(
            "frame1.fits",
            cnn=object(),
            config=PhotometryConfig(
                source_selection=SourceSelectionConfig(contaminant_mag_offset=0.5),
            ),
        )

        np.testing.assert_array_equal(prep.radecs, radecs[[0, 2]])
        np.testing.assert_allclose(prep.photometry_coords.ra.deg, radecs[[0, 2], 0])

    def test_nan_magnitude_dropped_by_mag_limit(self, mocker):
        """A star with no Gaia magnitude fails the cut and is dropped entirely."""
        radecs = np.array([[10.0, 0.0], [10.1, 0.0], [10.2, 0.0]])
        mags = np.array([12.0, np.nan, 10.0])
        _patch_prep(mocker, radecs_mags=(radecs, mags))

        prep = scripts.prepare_batch("frame1.fits", cnn=object())

        np.testing.assert_array_equal(prep.radecs, radecs[[0, 2]])

    def test_raises_when_too_few_stars_detected(self, mocker):
        """A first-frame TooFewStarsError becomes a fatal BatchPrepError."""

        def _too_few(file, **_kwargs: object):
            msg = "only 1 stars detected"
            raise TooFewStarsError(msg, file=file)

        mocker.patch("bandaid.scripts.calibration_sequence", side_effect=_too_few)
        _stub_load_frame(mocker)
        with pytest.raises(BatchPrepError, match="too few stars"):
            scripts.prepare_batch("frame1.fits", cnn=object())

    def test_empty_gaia_field_raises_batchpreperror(self, mocker):
        """An empty Gaia cone is fatal -- no reference stars to solve any WCS."""
        _patch_prep(mocker, radecs_mags=(np.empty((0, 2)), np.empty(0)))
        # Use the real floor, not _patch_prep's relaxed one, for the guard.
        mocker.patch("bandaid.scripts.N_GAIA_STARS_ALIGN_RETRY", 20)
        with pytest.raises(BatchPrepError, match="Gaia returned only 0"):
            scripts.prepare_batch("frame1.fits", cnn=object())

    def test_sparse_gaia_field_raises_batchpreperror(self, mocker):
        """Fewer than N_GAIA_STARS_ALIGN_RETRY references is fatal for the batch."""
        radecs = np.column_stack([np.linspace(9.0, 11.0, 5), np.zeros(5)])
        _patch_prep(mocker, radecs_mags=(radecs, np.full(5, 12.0)))
        mocker.patch("bandaid.scripts.N_GAIA_STARS_ALIGN_RETRY", 20)
        with pytest.raises(BatchPrepError, match="Gaia returned only 2"):
            scripts.prepare_batch("frame1.fits", cnn=object())

    @staticmethod
    def _stars_in_and_out_of_pool(mocker, n_in, n_out) -> None:
        """
        Patch a prep whose catalog has ``n_in`` stars in the pool, ``n_out`` outside.

        The stars sit due north of the field center, inside or just beyond
        ``solve_pool_radius_scale`` times the field radius, all inside the
        field radius itself (as with a cone widened by the margin).
        """
        metadata = _batch_metadata()
        instrument = load_instrument("Seestar50")
        ra0, dec0 = scripts.estimate_center_from_header(metadata, instrument)
        pool = metadata["fov_rad"] * instrument.solve_pool_radius_scale
        offsets = np.concatenate(
            [np.linspace(0.0, 0.8 * pool, n_in), np.full(n_out, 1.05 * pool)]
        )
        radecs = np.column_stack([np.full(n_in + n_out, ra0), dec0 + offsets])
        _patch_prep(
            mocker,
            metadata=metadata,
            radecs_mags=(radecs, np.full(n_in + n_out, 12.0)),
        )
        # Use the real floor, not _patch_prep's relaxed one, for the guard.
        mocker.patch("bandaid.scripts.N_GAIA_STARS_ALIGN_RETRY", 20)

    def test_floor_counts_only_stars_inside_solve_pool(self, mocker):
        """Stars in the widened cone but outside the solve pool do not count."""
        self._stars_in_and_out_of_pool(mocker, n_in=12, n_out=13)
        with pytest.raises(BatchPrepError, match="solve pool"):
            scripts.prepare_batch("frame1.fits", cnn=object())

    def test_floor_met_when_stars_inside_solve_pool(self, mocker):
        """Enough stars inside the solve pool passes; the full list is kept."""
        self._stars_in_and_out_of_pool(mocker, n_in=N_POOL_STARS, n_out=N_EDGE_STARS)
        prep = scripts.prepare_batch("frame1.fits", cnn=object())
        assert len(prep.radecs) == N_POOL_STARS + N_EDGE_STARS

    def test_catalog_truncation_error_propagates_unwrapped(self, mocker):
        """A CatalogTruncationError is not re-wrapped as a failed Gaia query."""
        _patch_prep(mocker)
        mocker.patch(
            "bandaid.scripts.query_field_catalog",
            side_effect=CatalogTruncationError("row limit hit"),
        )
        with pytest.raises(CatalogTruncationError, match="row limit hit"):
            scripts.prepare_batch("frame1.fits", cnn=object())

    def test_gaia_network_error_raises_batchpreperror(self, mocker):
        """A Gaia query failure is surfaced as a fatal BatchPrepError."""
        mocker.patch(
            "bandaid.scripts.calibration_sequence",
            return_value=(
                np.zeros((4, 4)),
                _batch_metadata(),
                None,
                2.0,
                object(),
            ),
        )
        _stub_load_frame(mocker)

        def _boom(*_args: object, **_kwargs: object):
            msg = "no network"
            raise ConnectionError(msg)

        mocker.patch("bandaid.scripts.query_field_catalog", side_effect=_boom)
        with pytest.raises(BatchPrepError, match="could not query Gaia"):
            scripts.prepare_batch("frame1.fits", cnn=object())

    def test_forced_targets_appended_to_photometry_coords_only(self, mocker):
        """A forced target lands in ``photometry_coords`` but not ``radecs``."""
        radecs, mags = _batch_radecs_mags()
        _patch_prep(mocker, radecs_mags=(radecs, mags))
        forced = SkyCoord([20.0] * u.deg, [5.0] * u.deg)

        prep = scripts.prepare_batch("frame1.fits", cnn=object(), forced_targets=forced)

        # radecs is the Gaia-only alignment catalog -- the forced target never
        # appears there.
        np.testing.assert_array_equal(prep.radecs, radecs)
        # photometry_coords is the contamination-filtered Gaia targets plus the
        # forced target appended at the end.
        expected_ra = np.concatenate([radecs[[2, 3], 0], [20.0]])
        expected_dec = np.concatenate([radecs[[2, 3], 1], [5.0]])
        np.testing.assert_allclose(prep.photometry_coords.ra.deg, expected_ra)
        np.testing.assert_allclose(prep.photometry_coords.dec.deg, expected_dec)

    def test_forced_targets_bypass_contamination_flagging(self, mocker):
        """A forced target near a bright star still reaches ``photometry_coords``."""
        # A mag-8 star and a forced target ~1 arcsec away -- well inside the
        # contamination model's separation for a bright star -- but the forced
        # target has no Gaia magnitude to size that model against, so it is
        # never evaluated for contamination and always survives.
        radecs = np.array([[10.0, 0.0], [10.2, 0.0]])
        mags = np.array([8.0, 10.0])
        _patch_prep(mocker, radecs_mags=(radecs, mags))
        forced = SkyCoord(
            [10.0 + 1.0 / 3600.0] * u.deg,
            [0.0] * u.deg,
        )

        prep = scripts.prepare_batch("frame1.fits", cnn=object(), forced_targets=forced)

        assert forced.ra.deg[0] in prep.photometry_coords.ra.deg
        assert forced.dec.deg[0] in prep.photometry_coords.dec.deg

    def test_forced_targets_scalar_skycoord_appended_as_one_target(self, mocker):
        """A scalar ``forced_targets`` SkyCoord (e.g. ``from_name``) is accepted."""
        # A scalar SkyCoord (no list/array) has no len(); prepare_batch must
        # reshape it to a 1-element array before concatenating, not crash.
        radecs, mags = _batch_radecs_mags()
        _patch_prep(mocker, radecs_mags=(radecs, mags))
        forced = SkyCoord(20.0 * u.deg, 5.0 * u.deg)
        assert forced.isscalar

        prep = scripts.prepare_batch("frame1.fits", cnn=object(), forced_targets=forced)

        expected_ra = np.concatenate([radecs[[2, 3], 0], [20.0]])
        expected_dec = np.concatenate([radecs[[2, 3], 1], [5.0]])
        np.testing.assert_allclose(prep.photometry_coords.ra.deg, expected_ra)
        np.testing.assert_allclose(prep.photometry_coords.dec.deg, expected_dec)

    def test_forced_targets_non_icrs_frame_lands_as_icrs(self, mocker):
        """An FK5 ``forced_targets`` is transformed to ICRS before concatenating."""
        # photometry_coords is built as plain ICRS; concatenating a differently
        # framed SkyCoord without transforming first raises a confusing
        # TypeError, so prepare_batch must convert to ICRS itself.
        radecs, mags = _batch_radecs_mags()
        _patch_prep(mocker, radecs_mags=(radecs, mags))
        forced_icrs = SkyCoord([20.0] * u.deg, [5.0] * u.deg)
        forced_fk5 = forced_icrs.transform_to("fk5")

        prep = scripts.prepare_batch(
            "frame1.fits", cnn=object(), forced_targets=forced_fk5
        )

        assert prep.photometry_coords.frame.name == "icrs"
        # Same sky position, round-tripped through FK5; arcsec-level agreement
        # confirms the frame conversion, not just that concatenation ran.
        max_separation_arcsec = 1e-3
        appended = prep.photometry_coords[-1]
        assert appended.separation(forced_icrs).arcsec[0] < max_separation_arcsec

    def test_forced_targets_stored_on_batchprep(self, mocker):
        """``BatchPrep.forced_targets`` carries the (reshaped, ICRS) forced targets."""
        radecs, mags = _batch_radecs_mags()
        _patch_prep(mocker, radecs_mags=(radecs, mags))
        forced = SkyCoord([20.0] * u.deg, [5.0] * u.deg)

        prep = scripts.prepare_batch("frame1.fits", cnn=object(), forced_targets=forced)

        assert prep.forced_targets is not None
        np.testing.assert_allclose(prep.forced_targets.ra.deg, [20.0])
        np.testing.assert_allclose(prep.forced_targets.dec.deg, [5.0])

    def test_forced_targets_none_by_default_on_batchprep(self, mocker):
        """``BatchPrep.forced_targets`` is None when no forced targets are given."""
        _patch_prep(mocker)

        prep = scripts.prepare_batch("frame1.fits", cnn=object())

        assert prep.forced_targets is None


class TestBatchPrep:
    """Unit tests for the ``BatchPrep`` dataclass's own construction invariant."""

    @staticmethod
    def _kwargs(**overrides: object) -> dict:
        """Minimal BatchPrep constructor kwargs, overridable per test."""
        fields = {
            "radecs": np.zeros((1, 2)),
            "photometry_coords": SkyCoord([0.0], [0.0], unit="deg"),
            "cnn": object(),
            "bayer_masks": {},
            "center": (10.0, 0.0),
            "fov_rad": 0.74,
            "shape": (1920, 1080),
            "config": PhotometryConfig(instrument=InstrumentProfile()),
        }
        fields.update(overrides)
        return fields

    def test_unresolved_instrument_raises_valueerror(self):
        """
        A ``config`` whose ``instrument`` is still None (unresolved) is rejected.

        `check_frame_consistency` and `process_batch` both trust
        ``config.instrument`` to already be resolved by the time a ``BatchPrep``
        reaches them -- `prepare_batch` is the only place that is supposed to
        build one, and it always resolves the instrument first. Constructing
        one directly with a bare, unresolved
        ``PhotometryConfig()`` must fail loudly instead of silently letting
        per-frame detection paper over it.
        """
        with pytest.raises(ValueError, match="instrument"):
            scripts.BatchPrep(**self._kwargs(config=PhotometryConfig()))

    def test_resolved_instrument_constructs_fine(self):
        """A ``config`` with an explicit, resolved instrument constructs fine."""
        prep = scripts.BatchPrep(
            **self._kwargs(config=PhotometryConfig(instrument=InstrumentProfile()))
        )
        assert prep.config.instrument is not None


class TestCheckFrameConsistency:
    """Unit tests for the per-frame pointing/shape guard."""

    # The field center for a frame pointing at RA=10/DEC=0 under the bare
    # (ICRS-header) profile these tests use: the header pointing itself.
    # prep.center holds this, so a stable frame reads ~0 offset against it.
    STABLE_CENTER = (10.0, 0.0)

    @pytest.fixture(autouse=True)
    def _isolate_registry(self, isolate_registry):
        """
        Restore the in-process profile registry after each test in this class.

        A couple of tests below register an extra profile to exercise the
        batch-mixing guard's ambiguity handling; without this, that
        registration would leak into later tests.
        """
        with isolate_registry(instruments, "_REGISTERED"):
            yield

    @staticmethod
    def _prep(**overrides: object) -> scripts.BatchPrep:
        """A BatchPrep carrying consistency fields, overridable per test."""
        fields = {
            "center": TestCheckFrameConsistency.STABLE_CENTER,
            "fov_rad": 0.74,
            "shape": (1920, 1080),
            # A resolved (non-None) instrument, matching what prepare_batch
            # would have already resolved by the time check_frame_consistency
            # runs; a bare InstrumentProfile() carries no header_match, so the
            # batch-mixing guard (only enforced when header_match is
            # non-empty) is a no-op here unless a test overrides this.
            "config": PhotometryConfig(instrument=InstrumentProfile()),
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
        """A frame pointing beyond the field radius is rejected, carrying its offset."""
        header = _consistency_header(RA=12.0)
        with pytest.raises(FrameError, match="pointing") as exc_info:
            scripts.check_frame_consistency("bad.fits", header, self._prep())
        assert exc_info.value.pointing_offset == pytest.approx(2.0)

    def test_drifted_frame_within_radius_accepted(self):
        """
        A frame drifted <1 field radius from the batch center is accepted.

        A frame whose header moved 0.5 deg (well inside the 0.74 deg radius) is
        kept; only drift beyond the field radius is rejected.
        """
        # RA=10.5 is 0.5 deg from STABLE_CENTER under the ICRS-header profile.
        header = _consistency_header(RA=10.5)
        scripts.check_frame_consistency("ok.fits", header, self._prep())

    @staticmethod
    def _margin_prep(margin) -> scripts.BatchPrep:
        """A prep whose instrument carries the given cone margin."""
        return TestCheckFrameConsistency._prep(
            config=PhotometryConfig(
                instrument=InstrumentProfile(cone_radius_margin=margin)
            )
        )

    def test_returns_offset_from_batch_center(self):
        """The offset (degrees) of the frame's center is returned."""
        header = _consistency_header(RA=10.0)
        offset = scripts.check_frame_consistency("ok.fits", header, self._prep())
        assert offset == pytest.approx(0.0, abs=1e-9)

    def test_offset_within_margin_is_silent(self, caplog):
        """A drift fully covered by the cone margin logs nothing."""
        header = _consistency_header(RA=10.2)
        with caplog.at_level("WARNING", logger="bandaid.scripts"):
            offset = scripts.check_frame_consistency(
                "ok.fits", header, self._margin_prep(0.4)
            )
        assert offset == pytest.approx(0.2)
        assert not caplog.records

    def test_offset_between_margin_and_field_radius_warns(self, caplog):
        """A partly covered drift warns with offset, margin and file; no raise."""
        header = _consistency_header(RA=10.5)
        with caplog.at_level("WARNING", logger="bandaid.scripts"):
            offset = scripts.check_frame_consistency(
                "drift.fits", header, self._margin_prep(0.4)
            )
        assert offset == pytest.approx(0.5)
        message = " ".join(r.getMessage() for r in caplog.records)
        assert "0.500" in message
        assert "0.400" in message
        assert "drift.fits" in message

    def test_zero_margin_offset_warns_instead_of_raising(self, caplog):
        """With a zero margin any drift inside the radius warns."""
        header = _consistency_header(RA=10.3)
        with caplog.at_level("WARNING", logger="bandaid.scripts"):
            offset = scripts.check_frame_consistency(
                "d.fits", header, self._margin_prep(0.0)
            )
        assert offset == pytest.approx(0.3)
        assert len(caplog.records) == 1

    def test_default_margin_covers_pointing_jitter(self, caplog):
        """The class-default margin keeps a small frame-to-frame offset silent."""
        header = _consistency_header(RA=10.06)
        with caplog.at_level("WARNING", logger="bandaid.scripts"):
            offset = scripts.check_frame_consistency("d.fits", header, self._prep())
        assert offset == pytest.approx(0.06)
        assert not caplog.records

    def test_offset_beyond_field_radius_still_raises_with_margin(self):
        """The margin does not extend the hard rejection radius."""
        header = _consistency_header(RA=12.0)
        with pytest.raises(FrameError, match="pointing"):
            scripts.check_frame_consistency("bad.fits", header, self._margin_prep(0.4))

    def test_fk5_of_date_frame_compared_in_icrs(self):
        """With an fk5/"date" profile the frame's converted center is compared."""
        profile = InstrumentProfile(header_frame="fk5", header_equinox="date")
        obs_time = "2026-04-28T03:03:43"
        header = _consistency_header(**{"DATE-OBS": obs_time})
        center = scripts.estimate_center_from_header(
            {"ra": 10.0, "dec": 0.0, "obs_time": obs_time}, profile
        )
        # The precession shift (~0.3 deg) is real: the raw header would not match.
        assert center != (10.0, 0.0)
        prep = self._prep(center=center, config=PhotometryConfig(instrument=profile))
        scripts.check_frame_consistency("ok.fits", header, prep)

    def test_fk5_of_date_frame_without_obs_time_labeled_with_file(self):
        """An fk5/"date" frame lacking DATE-OBS is a metadata error naming the file."""
        profile = InstrumentProfile(header_frame="fk5", header_equinox="date")
        prep = self._prep(config=PhotometryConfig(instrument=profile))
        header = _consistency_header()
        del header["DATE-OBS"]
        with pytest.raises(FrameMetadataError, match="obs_time") as excinfo:
            scripts.check_frame_consistency("bad.fits", header, prep)
        assert excinfo.value.file == "bad.fits"

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

    def test_non_numeric_pointing_labeled_with_file(self):
        """
        A present-but-non-numeric pointing fails as a metadata error naming the file.

        The ``None`` case is caught by the explicit guard above; the residual is
        a pointing that resolves to a non-numeric value, which surfaces from
        ``estimate_center_from_header``. That call sits outside the
        ``metadata_from_header`` try/except, so its error must be labelled with
        the frame path here to stay consistent with the other rejections.
        """
        header = _consistency_header(RA="garbage")
        with pytest.raises(FrameMetadataError, match="pointing") as excinfo:
            scripts.check_frame_consistency("bad.fits", header, self._prep())
        assert excinfo.value.file == "bad.fits"

    def test_nan_pointing_is_rejected_not_passed_as_undrifted(self):
        """A NaN pointing is a metadata error; it must not pass the drift check."""
        header = _consistency_header(RA="nan")
        with pytest.raises(FrameMetadataError, match="pointing") as excinfo:
            scripts.check_frame_consistency("bad.fits", header, self._prep())
        assert excinfo.value.file == "bad.fits"

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

    def test_different_instrument_header_rejected_when_auto_detected(self):
        """
        An AUTO-DETECTED frame whose header matches none of the rules is rejected.

        The bundled Seestar50 profile (unlike the bare-class default) carries a
        non-empty ``header_match``, so a batch prepared against it *by
        auto-detection* must reject a later frame whose header identifies a
        different instrument -- a batch-mixing guard against e.g. an
        accidentally interleaved night from a different telescope riding along
        on the auto-detected profile.
        """
        prep = self._prep(
            config=PhotometryConfig(instrument=load_instrument("Seestar50")),
            instrument_auto_detected=True,
        )
        header = _consistency_header(INSTRUME="Some Other Scope")

        with pytest.raises(FrameError, match="instrument"):
            scripts.check_frame_consistency("bad.fits", header, prep)

    def test_matching_instrument_header_passes(self):
        """A frame whose header matches the batch instrument's rule is accepted."""
        prep = self._prep(
            config=PhotometryConfig(instrument=load_instrument("Seestar50")),
            instrument_auto_detected=True,
        )
        header = _consistency_header(INSTRUME="Seestar S50")

        scripts.check_frame_consistency("ok.fits", header, prep)

    def test_explicitly_chosen_instrument_unmatched_header_not_rejected(self):
        """
        An EXPLICITLY-chosen instrument tolerates a header that matches nothing.

        ``--instrument``/``--profile``/``--config`` (or
        ``config=PhotometryConfig(instrument=...)``) is an explicit,
        deliberate choice, so a header that resolves to no registered
        instrument at all (missing/malformed, matching neither the batch
        instrument nor any other) is trusted and photometered rather than
        rejected -- the escape hatch this guard grants an explicit selection
        (``instrument_auto_detected=False`` is the ``_prep()`` default, as it
        would be for an explicit selection reaching `prepare_batch``). This is
        narrower than "the guard is skipped entirely": see
        `test_explicitly_chosen_instrument_different_registered_instrument_rejected`
        for the case that is still policed.
        """
        prep = self._prep(
            config=PhotometryConfig(instrument=load_instrument("Seestar50"))
        )
        header = _consistency_header(INSTRUME="Some Other Scope")

        scripts.check_frame_consistency("ok.fits", header, prep)

    def test_explicitly_chosen_instrument_different_registered_instrument_rejected(
        self,
    ):
        """
        An EXPLICITLY-chosen instrument still rejects a header naming another one.

        Before this, ``instrument_auto_detected=False`` disabled the guard
        entirely, so a scripted workflow that always passes
        ``--instrument Seestar50`` got zero mixing protection: frames from a
        second registered telescope riding along on the same batch were
        photometered under the wrong profile's tuning -- exactly the
        silently-wrong-results class this guard exists to prevent. The guard
        is narrowed, not disabled, for an explicit
        selection: it still fires when the header positively identifies a
        *different, registered* instrument; only a header matching nothing at
        all is exempt (see the sibling test above).
        """
        other = InstrumentProfile(
            name="OtherScope",
            header_match=(HeaderMatchRule(keyword="INSTRUME", pattern="Other Scope"),),
        )
        register_instrument(other)
        prep = self._prep(
            config=PhotometryConfig(instrument=load_instrument("Seestar50"))
        )
        header = _consistency_header(INSTRUME="Other Scope")

        with pytest.raises(FrameError, match="instrument"):
            scripts.check_frame_consistency("bad.fits", header, prep)

    def test_mixing_guard_fires_before_metadata_error(self):
        """
        The batch-mixing guard is checked before ``metadata_from_header`` runs.

        A frame from a genuinely different instrument is likely to also be
        missing a keyword the *batch* instrument's ``header_map`` needs (here,
        ``egain``, resolved via ``@EGAIN`` rather than the Seestar's literal
        default). Before this fix, ``metadata_from_header`` ran first and
        raised the less diagnostic ``FrameMetadataError`` before the guard ever
        got a chance to name the more likely cause; the guard now only needs
        ``header`` and the batch instrument, so it runs first.
        """
        custom_header_map = {
            **dict(InstrumentProfile().header_map),
            "egain": "@EGAIN",
        }
        custom = InstrumentProfile(
            name="CustomScope",
            header_match=(HeaderMatchRule(keyword="INSTRUME", pattern="CustomScope"),),
            header_map=custom_header_map,
        )
        register_instrument(custom)
        prep = self._prep(
            config=PhotometryConfig(instrument=custom),
            instrument_auto_detected=True,
        )
        # No INSTRUME (fails the batch-mixing guard) and no EGAIN (would also
        # fail metadata_from_header's egain check, if it ran).
        header = _consistency_header()

        with pytest.raises(FrameError) as excinfo:
            scripts.check_frame_consistency("bad.fits", header, prep)

        assert type(excinfo.value) is FrameError
        assert excinfo.value.file == "bad.fits"

    def test_missing_keyword_guard_message(self):
        """
        The guard names the missing keyword when it's absent from the header.

        Distinguishes "the header never carried the identifying keyword at
        all" from "it carried the keyword with a different value": the former
        is worded as a missing-keyword problem rather than the generic
        "different instrument mixed in" phrasing.
        """
        prep = self._prep(
            config=PhotometryConfig(instrument=load_instrument("Seestar50")),
            instrument_auto_detected=True,
        )
        # _consistency_header() carries no INSTRUME key at all.
        header = _consistency_header()

        with pytest.raises(FrameError, match="missing") as excinfo:
            scripts.check_frame_consistency("bad.fits", header, prep)
        assert "INSTRUME" in str(excinfo.value)

    def test_different_value_guard_message(self):
        """The guard keeps the "different instrument" wording for a present value."""
        prep = self._prep(
            config=PhotometryConfig(instrument=load_instrument("Seestar50")),
            instrument_auto_detected=True,
        )
        header = _consistency_header(INSTRUME="Some Other Scope")

        with pytest.raises(FrameError, match="does not match") as excinfo:
            scripts.check_frame_consistency("bad.fits", header, prep)
        assert (
            "possibly a frame from a different instrument mixed into this batch"
            in str(excinfo.value)
        )

    def test_multi_rule_present_wrong_value_is_not_reported_as_missing(self):
        """
        One absent OR-rule keyword does not mean that keyword is required.

        ``header_match`` is OR semantics: a profile matches if *any* rule
        matches. For a profile with ``INSTRUME=Foo OR TELESCOP=Bar``, a header
        carrying ``INSTRUME=Other`` (present, wrong value) and no ``TELESCOP``
        at all used to report only that ``TELESCOP`` is missing -- because the
        old check fired the "missing" branch whenever *any* rule's keyword was
        absent, not only when *all* of them were (a Copilot-flagged bug). The
        more accurate "does not match" diagnostic is
        correct here: the header did carry an identifying keyword, just with
        the wrong value.
        """
        custom = InstrumentProfile(
            name="MultiRuleScope",
            header_match=(
                HeaderMatchRule(keyword="INSTRUME", pattern="Foo"),
                HeaderMatchRule(keyword="TELESCOP", pattern="Bar"),
            ),
        )
        register_instrument(custom)
        prep = self._prep(
            config=PhotometryConfig(instrument=custom),
            instrument_auto_detected=True,
        )
        # _consistency_header() carries no TELESCOP key at all.
        header = _consistency_header(INSTRUME="Other")

        with pytest.raises(FrameError, match="does not match") as excinfo:
            scripts.check_frame_consistency("bad.fits", header, prep)
        assert "missing" not in str(excinfo.value)

    def test_ambiguous_instrument_rejected_by_guard(self):
        """
        An auto-detected batch rejects a frame whose header is ambiguous.

        ``check_frame_consistency`` now uses the same predicate as first-frame
        detection: a later frame whose header matches more than
        one registered profile is rejected even though it matches the batch
        instrument's own rule, because detection itself could not have picked
        the batch instrument unambiguously from this header.
        """
        clone = InstrumentProfile(
            name="Clone",
            header_match=(SEESTAR_RULE,),
        )
        # register_instrument now eagerly rejects a colliding rule, so this
        # deliberately-ambiguous fixture is inserted directly into the
        # isolated registry, bypassing that check, to exercise the
        # detection-time ambiguity error the guard must still catch.
        instruments._REGISTERED["Clone"] = clone  # noqa: SLF001
        prep = self._prep(
            config=PhotometryConfig(instrument=load_instrument("Seestar50")),
            instrument_auto_detected=True,
        )
        header = _consistency_header(INSTRUME="Seestar S50")

        with pytest.raises(FrameError, match="Clone"):
            scripts.check_frame_consistency("bad.fits", header, prep)

        # With the duplicate registration gone, a plain matching frame passes.
        del instruments._REGISTERED["Clone"]  # noqa: SLF001
        scripts.check_frame_consistency("ok.fits", header, prep)

    def test_explicit_selection_rejects_ambiguous_header(self):
        """
        An explicit selection does not exempt an ambiguous header.

        A header matching two registered profiles is rejected under
        ``instrument_auto_detected=False`` just as under auto-detection: the
        escape hatch is for a header that identifies nothing, not one that
        positively identifies more than one instrument.
        """
        clone = InstrumentProfile(
            name="Clone",
            header_match=(SEESTAR_RULE,),
        )
        # Inserted directly: register_instrument would reject the colliding rule.
        instruments._REGISTERED["Clone"] = clone  # noqa: SLF001
        prep = self._prep(
            config=PhotometryConfig(instrument=load_instrument("Seestar50")),
            instrument_auto_detected=False,
        )
        header = _consistency_header(INSTRUME="Seestar S50")

        with pytest.raises(FrameError, match="Clone"):
            scripts.check_frame_consistency("bad.fits", header, prep)

    @pytest.mark.parametrize(
        ("auto_detected", "header_kwargs", "expected"),
        [
            (True, {}, "auto-detected batch instrument Seestar50's header_match"),
            (
                True,
                {"INSTRUME": "Other Scope"},
                "auto-detected batch instrument Seestar50's header_match",
            ),
            (False, {"INSTRUME": "Other Scope"}, "batch instrument Seestar50's"),
        ],
    )
    def test_guard_message_wording(self, auto_detected, header_kwargs, expected):
        """
        Guard messages read cleanly and say auto-detected only when true.

        Covers the missing-keyword and the different-value messages, and the
        explicit-selection path (a different registered instrument), which
        must not send the user hunting for a detection problem.
        """
        register_instrument(
            InstrumentProfile(
                name="OtherScope",
                header_match=(
                    HeaderMatchRule(keyword="INSTRUME", pattern="Other Scope"),
                ),
            )
        )
        prep = self._prep(
            config=PhotometryConfig(instrument=load_instrument("Seestar50")),
            instrument_auto_detected=auto_detected,
        )
        header = _consistency_header(**header_kwargs)

        with pytest.raises(FrameError) as excinfo:
            scripts.check_frame_consistency("bad.fits", header, prep)

        message = str(excinfo.value)
        assert expected in message
        assert "''s" not in message
        assert ("auto-detected" in message) is auto_detected

    def test_inconsistent_frame_is_skipped_by_batch(self, mocker):
        """process_batch skips an off-field frame and keeps the good one."""
        prep = self._prep()

        def _fake_load_frame(file):
            ra = 10.0 if file == "good.fits" else 50.0
            return scripts.LoadedFrame(np.zeros((2, 2)), _consistency_header(RA=ra))

        mocker.patch("bandaid.scripts._load_frame", side_effect=_fake_load_frame)
        mocker.patch(
            "bandaid.scripts.process_one_image",
            return_value={"TR": Table({"tot_count": [1.0]})},
        )
        results = scripts.process_batch(
            ["good.fits", "bad.fits"],
            prep,
            user_specific_metadata={},
        )
        assert list(results) == ["good.fits"]

    def test_mixing_guard_skip_is_recorded_and_batch_continues(self, mocker, tmp_path):
        """
        process_batch skips a guard-rejected frame and records it in the manifest.

        Exercises the whole path for an auto-detected batch: the guard's
        `FrameError` is caught by the per-frame loop, the frame is absent from
        the results, its QA manifest row reads ``skipped: FrameError``, and
        the good frame after it is still processed.
        """
        prep = self._prep(
            config=PhotometryConfig(instrument=load_instrument("Seestar50")),
            instrument_auto_detected=True,
        )

        def _fake_load_frame(file):
            instrume = "Some Other Scope" if file == "bad.fits" else "Seestar S50"
            return scripts.LoadedFrame(
                np.zeros((2, 2)), _consistency_header(INSTRUME=instrume)
            )

        mocker.patch("bandaid.scripts._load_frame", side_effect=_fake_load_frame)
        mocker.patch(
            "bandaid.scripts.process_one_image",
            return_value={"TR": Table({"tot_count": [1.0]})},
        )
        results = scripts.process_batch(
            ["bad.fits", "good.fits"],
            prep,
            user_specific_metadata={},
            output_dir=tmp_path,
            write_frame=lambda _result, path: path,
        )

        assert list(results) == ["good.fits"]
        with (tmp_path / scripts.QA_MANIFEST_FILENAME).open(newline="") as f:
            rows = {row["file"]: row for row in csv.DictReader(f)}
        assert rows["bad.fits"]["status"] == "skipped: FrameError"
        assert rows["good.fits"]["status"] == "ok"


class TestQaRecordOkForcedTargets:
    """Unit tests for ``_qa_record_ok``'s ``n_forced_measured`` QA column."""

    def test_blank_without_forced_targets(self, by_filter):
        """``n_forced_measured`` is None when the batch has no forced targets."""
        record = scripts._qa_record_ok(  # noqa: SLF001
            "a.fits", by_filter(), forced_targets=None
        )

        assert record["n_forced_measured"] is None

    def test_counts_forced_targets_matching_good_rows(self, by_filter):
        """A forced target at a good row's exact position is counted."""
        # by_filter's two rows are both good (finite/positive/in-bounds) and
        # sit at ra/dec (10.0, 20.0) and (11.0, 21.0); photometry runs at the
        # catalog positions, so a real match lands there essentially exactly.
        forced = SkyCoord([10.0, 11.0], [20.0, 21.0], unit="deg")
        expected_matches = 2

        record = scripts._qa_record_ok(  # noqa: SLF001
            "a.fits", by_filter(), forced_targets=forced
        )

        assert record["n_forced_measured"] == expected_matches

    def test_zero_when_forced_target_has_no_good_row_match(self, by_filter):
        """A forced target far from every good row counts as 0, not None."""
        forced = SkyCoord([200.0], [-50.0], unit="deg")

        record = scripts._qa_record_ok(  # noqa: SLF001
            "a.fits", by_filter(), forced_targets=forced
        )

        assert record["n_forced_measured"] == 0

    def test_none_when_needed_columns_unavailable(self, by_filter):
        """Forced targets configured but no evaluable photometry columns -> None."""
        # Strip the columns good_star_mask needs so `good` stays None; the
        # missing-data blank must win over a misleadingly precise 0.
        result = by_filter()
        for table in result.values():
            table.remove_columns(["tot_count", "count_err"])
        forced = SkyCoord([10.0], [20.0], unit="deg")

        record = scripts._qa_record_ok(  # noqa: SLF001
            "a.fits", result, forced_targets=forced
        )

        assert record["n_forced_measured"] is None
