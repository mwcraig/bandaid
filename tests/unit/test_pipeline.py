"""Unit tests for the detect/align/centroid pipeline (prepare_image, process)."""

from pathlib import Path

import numpy as np
import pytest
from _helpers import SEED, _make_tan_wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.table import Table
from eloy import detection
from eloy.ballet.model import Ballet

from bandaid.config import InstrumentProfile, PhotometryConfig
from bandaid.exceptions import (
    DegenerateBayerChannelError,
    FrameMetadataError,
    TooFewStarsError,
)
from bandaid.image2sl_qt import generate_bayer_masks
from bandaid.photometry import (
    DETECTION_OPENING,
    MIN_DETECTED_STARS,
    N_GAIA_STARS_ALIGN,
    THRESH,
    _brightest_unsaturated,
    _fwhm_from_coords,
    calibration_sequence,
    metadata_from_header,
    prepare_image,
    process_one_image,
)


class TestPrepareImage:
    def test_no_photometry_coord_input(self, make_test_image, tmp_path, monkeypatch):
        """Aligned coords fall back to detected coords when none are provided."""
        # This test only checks the alignment fallback, not centroiding, so stub
        # centroid_stars to avoid constructing the real Ballet CNN (which would pull
        # model weights from HuggingFace). The stub returns the aligned coords
        # unchanged.
        monkeypatch.setattr(
            "bandaid.photometry.centroid_stars",
            lambda data, coords, cnn: coords,
        )
        image_size = (500, 500)

        source_properties = Table(
            {
                "amplitude": [100, 200, 300, 400],
                "x_mean": [50, 100, 150, 200],
                "y_mean": [50, 100, 150, 400],
                "x_stddev": [3, 3, 3, 3],
                "y_stddev": [3, 3, 3, 3],
            },
        )
        test_image = make_test_image(
            image_size=image_size,
            source_properties=source_properties,
            include_noise=False,
            noise_mean=0,
            noise_stddev=0,
            seed=SEED,
        )
        coords_xy = np.array(
            [[row["x_mean"], row["y_mean"]] for row in source_properties],
        )
        wcs = _make_tan_wcs(image_size, crval=(0.0, 0.0))

        radecs = np.array(wcs.pixel_to_world_values(coords_xy[:, 0], coords_xy[:, 1])).T
        radecs = radecs + np.array(
            [[0.01, 0.01]]
        )  # Add a small offset to ensure coords are not exactly on the sources
        ccd = CCDData(test_image, wcs=wcs, unit="adu")
        ccd.header["creator"] = "test_prepare_image"
        path = tmp_path / "test_image.fits"
        ccd.write(path)
        img = prepare_image(
            path,
            radecs,
            None,
            photometry_coords=None,
            wcs=wcs,
        )

        assert np.array_equal(img.coords, img.aligned_coords)

    def test_instrument_config_reaches_detection(self, monkeypatch):
        """
        A non-default instrument config sets the detection threshold/opening.

        ``prepare_image`` historically hardcoded ``threshold=THRESH`` and never
        forwarded ``opening`` to ``calibration_sequence``, so detection settings
        passed in via the config never reached detection. Spy on
        ``calibration_sequence`` and assert the configured values arrive.
        """
        expected_thresh = 0.9
        expected_opening = 7
        expected_fwhm_n_stars = 33
        captured = {}

        def _spy_calibration_sequence(_file, *_args: object, **kwargs: object):
            captured["threshold"] = kwargs.get("threshold")
            captured["opening"] = kwargs.get("opening")
            captured["fwhm_n_stars"] = kwargs.get("fwhm_n_stars")
            calibrated = np.zeros((10, 10))
            coords = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
            return calibrated, {"creator": "spy", "pixscale": 2.4}, coords, 2.0, None

        monkeypatch.setattr(
            "bandaid.photometry.calibration_sequence",
            _spy_calibration_sequence,
        )
        monkeypatch.setattr(
            "bandaid.photometry.align",
            lambda coords, radecs, **kwargs: (coords, _make_tan_wcs()),
        )
        monkeypatch.setattr(
            "bandaid.photometry.centroid_stars",
            lambda data, coords, cnn: coords,
        )
        monkeypatch.setattr(
            "bandaid.photometry.fits.getheader",
            lambda file: {"creator": "spy"},
        )

        config = PhotometryConfig(
            instrument=InstrumentProfile(
                thresh=expected_thresh,
                detection_opening=expected_opening,
                fwhm_n_stars=expected_fwhm_n_stars,
            ),
        )
        prepare_image(
            "unused.fits",
            np.zeros((5, 2)),
            None,
            config=config,
        )

        assert captured["threshold"] == expected_thresh
        assert captured["opening"] == expected_opening
        assert captured["fwhm_n_stars"] == expected_fwhm_n_stars

    def test_instrument_wcs_scale_tolerance_reaches_alignment(self, monkeypatch):
        """
        The instrument's ``wcs_scale_tolerance`` is forwarded to ``align``.

        A non-default profile tolerance must reach the plate-scale check, so spy
        on ``align`` and assert the configured value arrives as ``scale_tolerance``.
        """
        expected_tolerance = 0.07
        captured = {}

        def _spy_calibration_sequence(_file, *_args: object, **_kwargs: object):
            calibrated = np.zeros((10, 10))
            coords = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
            return calibrated, {"creator": "spy", "pixscale": 2.4}, coords, 2.0, None

        def _spy_align(coords, _radecs, **kwargs: object):
            captured["scale_tolerance"] = kwargs.get("scale_tolerance")
            return coords, _make_tan_wcs()

        monkeypatch.setattr(
            "bandaid.photometry.calibration_sequence", _spy_calibration_sequence
        )
        monkeypatch.setattr("bandaid.photometry.align", _spy_align)
        monkeypatch.setattr(
            "bandaid.photometry.centroid_stars", lambda data, coords, cnn: coords
        )
        monkeypatch.setattr(
            "bandaid.photometry.fits.getheader", lambda file: {"creator": "spy"}
        )

        config = PhotometryConfig(
            instrument=InstrumentProfile(wcs_scale_tolerance=expected_tolerance),
        )
        prepare_image("unused.fits", np.zeros((5, 2)), None, config=config)

        assert captured["scale_tolerance"] == expected_tolerance

    def test_missing_pixscale_raises_when_solving(self, monkeypatch):
        """
        A missing/non-numeric ``pixscale`` fails loud instead of silent-skipping.

        ``pixscale`` comes from the instrument profile via
        ``metadata_from_header`` and is required to scale-check a solved WCS, so
        when it is absent and no WCS is supplied ``prepare_image`` raises
        ``FrameMetadataError`` rather than passing ``expected_pixscale=None``
        (which would quietly disable the check).
        """

        def _spy_calibration_sequence(_file, *_args: object, **_kwargs: object):
            calibrated = np.zeros((10, 10))
            coords = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
            # metadata deliberately omits "pixscale".
            return calibrated, {"creator": "spy"}, coords, 2.0, None

        monkeypatch.setattr(
            "bandaid.photometry.calibration_sequence", _spy_calibration_sequence
        )
        monkeypatch.setattr(
            "bandaid.photometry.align",
            lambda *a, **k: pytest.fail("align must not run without a pixscale"),
        )
        monkeypatch.setattr(
            "bandaid.photometry.centroid_stars", lambda data, coords, cnn: coords
        )
        monkeypatch.setattr(
            "bandaid.photometry.fits.getheader", lambda file: {"creator": "spy"}
        )

        with pytest.raises(FrameMetadataError, match="pixscale"):
            prepare_image("unused.fits", np.zeros((5, 2)), None)

    def test_missing_pixscale_ok_when_wcs_supplied(self, monkeypatch):
        """
        A supplied WCS is trusted, so a missing ``pixscale`` is not required.

        ``align`` skips the scale check for a caller-supplied WCS, so
        ``prepare_image`` must not demand ``pixscale`` in that case; it forwards
        ``expected_pixscale=None`` without raising.
        """
        supplied_wcs = _make_tan_wcs()
        captured = {}

        def _spy_calibration_sequence(_file, *_args: object, **_kwargs: object):
            calibrated = np.zeros((10, 10))
            coords = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
            return calibrated, {"creator": "spy"}, coords, 2.0, None

        def _spy_align(coords, _radecs, **kwargs: object):
            captured["expected_pixscale"] = kwargs.get("expected_pixscale")
            return coords, supplied_wcs

        monkeypatch.setattr(
            "bandaid.photometry.calibration_sequence", _spy_calibration_sequence
        )
        monkeypatch.setattr("bandaid.photometry.align", _spy_align)
        monkeypatch.setattr(
            "bandaid.photometry.centroid_stars", lambda data, coords, cnn: coords
        )
        monkeypatch.setattr(
            "bandaid.photometry.fits.getheader", lambda file: {"creator": "spy"}
        )

        prepare_image("unused.fits", np.zeros((5, 2)), None, wcs=supplied_wcs)

        assert captured["expected_pixscale"] is None

    def test_header_center_and_shape_reach_alignment(self, monkeypatch):
        """
        The header pointing and image shape are forwarded to ``align``.

        The Gaia catalog is queried at the header ra/dec, so ``prepare_image``
        must pass that location (as ``expected_center``) plus the frame shape to
        ``align`` for the solved-WCS in-frame check. Spy on ``align`` and assert
        both arrive.
        """
        captured = {}

        def _spy_calibration_sequence(_file, *_args: object, **_kwargs: object):
            calibrated = np.zeros((10, 12))
            coords = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
            metadata = {"creator": "spy", "pixscale": 2.4, "ra": 10.0, "dec": 20.0}
            return calibrated, metadata, coords, 2.0, None

        def _spy_align(coords, _radecs, **kwargs: object):
            captured["expected_center"] = kwargs.get("expected_center")
            captured["shape"] = kwargs.get("shape")
            return coords, _make_tan_wcs()

        monkeypatch.setattr(
            "bandaid.photometry.calibration_sequence", _spy_calibration_sequence
        )
        monkeypatch.setattr("bandaid.photometry.align", _spy_align)
        monkeypatch.setattr(
            "bandaid.photometry.centroid_stars", lambda data, coords, cnn: coords
        )
        monkeypatch.setattr(
            "bandaid.photometry.fits.getheader", lambda file: {"creator": "spy"}
        )

        prepare_image("unused.fits", np.zeros((5, 2)), None)

        center = captured["expected_center"]
        assert isinstance(center, SkyCoord)
        assert center.ra.deg == pytest.approx(10.0)
        assert center.dec.deg == pytest.approx(20.0)
        assert captured["shape"] == (10, 12)

    def test_missing_header_radec_skips_center_check(self, monkeypatch):
        """
        A frame without usable header ra/dec skips the center check, not fails.

        Unlike ``pixscale`` (instrument-profile-sourced, so missing means a
        malformed profile), the pointing comes from the frame header; a frame
        without it should still solve, just without the in-frame check.
        """
        captured = {}

        def _spy_calibration_sequence(_file, *_args: object, **_kwargs: object):
            calibrated = np.zeros((10, 10))
            coords = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
            # metadata deliberately omits "ra"/"dec".
            return calibrated, {"creator": "spy", "pixscale": 2.4}, coords, 2.0, None

        def _spy_align(coords, _radecs, **kwargs: object):
            captured["expected_center"] = kwargs.get("expected_center")
            return coords, _make_tan_wcs()

        monkeypatch.setattr(
            "bandaid.photometry.calibration_sequence", _spy_calibration_sequence
        )
        monkeypatch.setattr("bandaid.photometry.align", _spy_align)
        monkeypatch.setattr(
            "bandaid.photometry.centroid_stars", lambda data, coords, cnn: coords
        )
        monkeypatch.setattr(
            "bandaid.photometry.fits.getheader", lambda file: {"creator": "spy"}
        )

        prepare_image("unused.fits", np.zeros((5, 2)), None)

        assert captured["expected_center"] is None


# --- Synthetic-FITS helpers for the detect/align/centroid pipeline tests ---


# Well-separated source positions (x, y) for a 480x480 frame; the first two also
# serve the small "too few stars" frames.
_SOURCE_POSITIONS = [(60, 60), (160, 160), (260, 260), (360, 360), (200, 400)]


def _detectable_image(
    make_test_image,
    *,
    n_sources=5,
    fwhm=4.0,
    amplitude=600.0,
    image_size=(480, 480),
    noise_mean=100.0,
    noise_stddev=2.0,
    include_noise=True,
):
    """
    Build a noisy multi-Gaussian frame that eloy's detection can resolve.

    ``amplitude`` far above ``noise_stddev`` keeps detection reliable; an
    ``amplitude`` above the 50000 ADU saturation cap exercises the saturated
    path in ``calibration_sequence``. Pass ``include_noise=False`` for the
    "too few stars" frames so detection returns exactly ``n_sources`` regardless
    of the threshold/opening (flat Gaussian noise at the low production threshold
    spawns spurious blobs that would otherwise pad the count past the floor).
    """
    sigma = fwhm * gaussian_fwhm_to_sigma
    positions = _SOURCE_POSITIONS[:n_sources]
    source_properties = Table(
        {
            "amplitude": [amplitude] * n_sources,
            "x_mean": [x for x, _ in positions],
            "y_mean": [y for _, y in positions],
            "x_stddev": [sigma] * n_sources,
            "y_stddev": [sigma] * n_sources,
        },
    )
    return make_test_image(
        image_size=image_size,
        source_properties=source_properties,
        include_noise=include_noise,
        noise_mean=noise_mean,
        noise_stddev=noise_stddev,
        seed=SEED,
    )


def _write_seestar_fits(path, image):
    """Write ``image`` to ``path`` with the header keys the pipeline reads."""
    ccd = CCDData(image, unit="adu")
    # metadata_from_header indexes CREATOR directly ("!CREATOR index 0"), so it
    # must be present; the others feed "@KEY" lookups used downstream.
    ccd.header["CREATOR"] = "ZWO Seestar S50"
    ccd.header["DATE-OBS"] = "2024-01-01T00:00:00"
    ccd.header["BAYERPAT"] = "RGGB"
    # Real Seestar frames carry pointing and site so airmass derives (issue #29);
    # without them build_photometry_table now skips the frame.
    ccd.header["RA"] = 10.0
    ccd.header["DEC"] = 20.0
    ccd.header["SITELAT"] = 40.0
    ccd.header["SITELONG"] = -105.0
    ccd.write(path)
    return path


# A few reference RA/Decs; align is always stubbed in these tests so the exact
# values only need to be a plausibly shaped array.
_REF_RADECS = np.array(
    [[10.0, 20.0], [10.01, 20.0], [10.0, 20.01], [10.02, 20.02], [10.03, 20.0]],
)


def _stub_wcs_and_centroid(
    monkeypatch,
    *,
    record_centroid_data=None,
    wcs_image_size=(500, 500),
    wcs_crval=(10.0, 20.0),
):
    """
    Stub the slow/networked externals reached via ``prepare_image``.

    ``compute_wcs`` (twirl's stochastic asterism solver) returns a fixed TAN WCS
    and ``centroid_stars`` (the HuggingFace-backed Ballet CNN) returns its input
    coordinates unchanged. If ``record_centroid_data`` is a list, the image
    actually handed to centroiding is appended to it so tests can inspect it.

    ``wcs_image_size``/``wcs_crval`` size and center the stubbed TAN WCS; the
    defaults match the synthetic-FITS callers, while the real-frame smoke test
    passes the actual frame shape and field center so the cosmetic RA/Dec columns
    land near the real field.
    """
    monkeypatch.setattr(
        "bandaid.photometry.compute_wcs",
        lambda coords, radecs, tolerance: _make_tan_wcs(wcs_image_size, wcs_crval),
    )

    def fake_centroid_stars(data, coords, _cnn):
        if record_centroid_data is not None:
            record_centroid_data.append(data)
        return coords

    monkeypatch.setattr("bandaid.photometry.centroid_stars", fake_centroid_stars)


class TestCalibrationSequence:
    """Unit tests for detection + FWHM estimation in ``calibration_sequence``."""

    def test_main_path_recovers_fwhm_and_sources(self, make_test_image, tmp_path):
        """A clean multi-source frame yields the sources and the injected FWHM."""
        fwhm = 4.0
        n_sources = 5
        expected_max_adu = 50000
        image = _detectable_image(make_test_image, n_sources=n_sources, fwhm=fwhm)
        path = _write_seestar_fits(tmp_path / "calib.fits", image)

        calibrated, metadata, coords, measured_fwhm, regions = calibration_sequence(
            path,
            threshold=1,
        )

        assert calibrated is not None
        assert len(regions) == n_sources
        assert coords.shape == (n_sources, 2)
        # The PSF fit recovers the injected FWHM to within ~5%.
        assert measured_fwhm == pytest.approx(fwhm, rel=0.05)
        assert metadata["largest_usable_adu_value"] == expected_max_adu

    def test_too_few_stars_raises(self, make_test_image, tmp_path):
        """Fewer than MIN_DETECTED_STARS detections raises TooFewStarsError."""
        image = _detectable_image(
            make_test_image,
            n_sources=2,
            image_size=(200, 200),
        )
        path = _write_seestar_fits(tmp_path / "few.fits", image)

        with pytest.raises(TooFewStarsError, match="stars detected"):
            calibration_sequence(path, threshold=1)

    def test_all_saturated_raises(self, make_test_image, tmp_path):
        """When every source saturates, no PSF can be fit, so it raises."""
        # Amplitude above the 50000 ADU cap means every cutout is dropped as
        # saturated, leaving nothing to fit.
        image = _detectable_image(make_test_image, n_sources=5, amplitude=60000.0)
        path = _write_seestar_fits(tmp_path / "sat.fits", image)

        with pytest.raises(TooFewStarsError, match="saturated"):
            calibration_sequence(path, threshold=1)

    def test_forwards_opening_to_detection(
        self, make_test_image, tmp_path, monkeypatch
    ):
        """
        calibration_sequence passes the opening kernel size through to detection.

        The morphological opening (not the threshold) is what gates faint-star
        detection, so the pipeline default must reach eloy's stars_detection. The
        detector is stubbed to capture its kwargs and return no regions; the
        resulting TooFewStarsError is incidental -- the assertion is the forwarded
        opening.
        """
        image = _detectable_image(make_test_image, n_sources=5)
        path = _write_seestar_fits(tmp_path / "open.fits", image)
        captured = {}

        def fake_stars_detection(data, threshold=5, opening=5):  # noqa: ARG001
            captured["opening"] = opening
            return []

        monkeypatch.setattr(
            "bandaid.photometry.detection.stars_detection", fake_stars_detection
        )

        # Default: the pipeline's DETECTION_OPENING reaches the detector.
        with pytest.raises(TooFewStarsError):
            calibration_sequence(path, threshold=1)
        assert captured["opening"] == DETECTION_OPENING

        # And an explicit override is honored.
        custom_opening = 7
        with pytest.raises(TooFewStarsError):
            calibration_sequence(path, threshold=1, opening=custom_opening)
        assert captured["opening"] == custom_opening

    def test_detects_on_balanced_copy_when_flagged(
        self, make_test_image, tmp_path, monkeypatch
    ):
        """
        detect_on_bayer_balanced runs detection/FWHM on a balanced copy (#22).

        The flag is meant to reach source detection, not just centroiding, while
        photometry must still see the original unbalanced counts. The detector is
        wrapped to capture the array it receives and ``bayer_balance_image`` is
        replaced with an in-place marker, so we can assert detection saw the
        balanced image while the returned ``calibrated_data`` is left unbalanced.
        """
        marker = 1000.0

        def fake_balance(arr):
            # Stand in for the real channel balancing with an obvious in-place
            # transform so a balanced array is trivially distinguishable.
            arr += marker

        monkeypatch.setattr("bandaid.photometry.bayer_balance_image", fake_balance)

        seen = {}
        real_detection = detection.stars_detection

        def capturing_detection(data, threshold=5, opening=5):
            seen["data"] = np.array(data, copy=True)
            return real_detection(data, threshold=threshold, opening=opening)

        monkeypatch.setattr(
            "bandaid.photometry.detection.stars_detection", capturing_detection
        )

        n_sources = 5
        image = _detectable_image(make_test_image, n_sources=n_sources)
        path = _write_seestar_fits(tmp_path / "bayer_detect.fits", image)

        calibrated, _, coords, _, regions = calibration_sequence(
            path,
            threshold=1,
            detect_on_bayer_balanced=True,
        )

        # Detection saw the balanced (marked) image...
        np.testing.assert_allclose(seen["data"], image + marker)
        # ...while the returned calibrated_data is the original, unbalanced counts
        # that downstream photometry relies on.
        np.testing.assert_allclose(calibrated, image)
        # Check that the balanced detection still recovers the injected sources.
        assert len(regions) == n_sources
        assert coords.shape == (n_sources, 2)

    def test_attaches_file_when_bayer_balance_is_degenerate(
        self, make_test_image, tmp_path, monkeypatch
    ):
        """A degenerate-channel error from bayer_balance_image gets the file (#61)."""

        def raising_balance(_arr):
            msg = "zero variance"
            raise DegenerateBayerChannelError(msg)

        monkeypatch.setattr("bandaid.photometry.bayer_balance_image", raising_balance)

        image = _detectable_image(make_test_image)
        path = _write_seestar_fits(tmp_path / "degenerate.fits", image)

        with pytest.raises(DegenerateBayerChannelError) as exc_info:
            calibration_sequence(path, threshold=1, detect_on_bayer_balanced=True)
        assert exc_info.value.file == path


class TestPrepareImageBranches:
    """Branch coverage for ``prepare_image`` beyond the alignment fallback."""

    def test_raises_when_too_few_stars(self, make_test_image, tmp_path):
        """prepare_image propagates calibration_sequence's TooFewStarsError."""
        image = _detectable_image(
            make_test_image,
            n_sources=2,
            image_size=(200, 200),
            include_noise=False,
        )
        path = _write_seestar_fits(tmp_path / "few.fits", image)

        # No external stubbing needed: it raises before align/centroid.
        with pytest.raises(TooFewStarsError, match="stars detected"):
            prepare_image(path, _REF_RADECS, None)

    def test_merges_user_specific_metadata(
        self, make_test_image, tmp_path, monkeypatch
    ):
        """user_specific_metadata overrides values pulled from the header."""
        _stub_wcs_and_centroid(monkeypatch)
        image = _detectable_image(make_test_image)
        path = _write_seestar_fits(tmp_path / "meta.fits", image)

        override_egain = 1.23
        img = prepare_image(
            path,
            _REF_RADECS,
            None,
            user_specific_metadata={"observer": "XYZ", "egain": override_egain},
        )

        assert img.metadata["observer"] == "XYZ"
        assert img.metadata["egain"] == override_egain

    def test_detect_on_bayer_balanced_uses_working_copy(
        self, make_test_image, tmp_path, monkeypatch
    ):
        """Bayer balancing feeds a balanced copy to centroiding, not the original."""
        centroid_inputs = []
        _stub_wcs_and_centroid(monkeypatch, record_centroid_data=centroid_inputs)
        image = _detectable_image(make_test_image)
        path = _write_seestar_fits(tmp_path / "bayer.fits", image)

        img = prepare_image(
            path,
            _REF_RADECS,
            None,
            detect_on_bayer_balanced=True,
        )

        # calibrated_data is left untouched...
        np.testing.assert_allclose(img.calibrated_data, image)
        # ...while the image handed to centroiding was balanced in place (so it
        # differs from the untouched calibrated frame).
        assert len(centroid_inputs) == 1
        assert not np.allclose(centroid_inputs[0], img.calibrated_data)

    def test_attaches_file_when_centroiding_bayer_balance_is_degenerate(
        self, make_test_image, tmp_path, monkeypatch
    ):
        """
        A degenerate-channel error from the centroiding balance pass gets the file.

        ``prepare_image`` calls ``bayer_balance_image`` a second time (for
        centroiding) after ``calibration_sequence``'s own detection-time call.
        The fake lets the first (detection) call succeed and only the second
        (centroiding) call raise, isolating that call site's own file-attaching
        ``try``/``except`` (issue #61).
        """
        _stub_wcs_and_centroid(monkeypatch)
        image = _detectable_image(make_test_image)
        path = _write_seestar_fits(tmp_path / "degenerate2.fits", image)

        calls = {"n": 0}

        def flaky_balance(_arr):
            calls["n"] += 1
            if calls["n"] == 1:
                return
            msg = "zero variance"
            raise DegenerateBayerChannelError(msg)

        monkeypatch.setattr("bandaid.photometry.bayer_balance_image", flaky_balance)
        expected_call_count = 2  # detection (succeeds), then centroiding (raises)

        with pytest.raises(DegenerateBayerChannelError) as exc_info:
            prepare_image(
                path,
                _REF_RADECS,
                None,
                detect_on_bayer_balanced=True,
            )
        assert exc_info.value.file == path
        assert calls["n"] == expected_call_count


class TestProcessOneImage:
    """End-to-end (stubbed-externals) coverage for ``process_one_image``."""

    def test_raises_when_image_rejected(self, make_test_image, tmp_path):
        """A frame with too few stars raises TooFewStarsError."""
        image = _detectable_image(
            make_test_image,
            n_sources=2,
            image_size=(200, 200),
            include_noise=False,
        )
        path = _write_seestar_fits(tmp_path / "few.fits", image)
        masks = generate_bayer_masks(
            image.shape,
            {"bayerpat": "RGGB", "roworder": "top-down", "ybayroff": 0},
            append_l4=True,
        )

        with pytest.raises(TooFewStarsError, match="stars detected"):
            process_one_image(path, {}, _REF_RADECS, None, masks)

    def test_full_path_builds_per_filter_tables_with_l4(
        self, make_test_image, tmp_path, monkeypatch
    ):
        """Every filter gets a table and the L4 channel sums the RGB counts."""
        _stub_wcs_and_centroid(monkeypatch)
        image = _detectable_image(make_test_image)
        path = _write_seestar_fits(tmp_path / "proc.fits", image)
        masks = generate_bayer_masks(
            image.shape,
            {"bayerpat": "RGGB", "roworder": "top-down", "ybayroff": 0},
            append_l4=True,
        )

        result = process_one_image(path, {}, _REF_RADECS, None, masks)

        assert set(result) == {"TR", "TG", "TB", "L4"}
        rgb_sum = (
            result["TR"]["tot_count"]
            + result["TG"]["tot_count"]
            + result["TB"]["tot_count"]
        )
        np.testing.assert_allclose(result["L4"]["tot_count"], rgb_sum)


# --- Real-frame smoke test -------------------------------------------------

# A genuine (full-size, uncropped) Seestar S50 frame committed under tests/data/
# as a bzip2-compressed FITS. astropy reads ``.fits.bz2``/``.fit.bz2``
# transparently, so the pipeline loads it with no special handling. Discover it
# by glob rather than a fixed name so whatever the user commits is picked up; the
# suite stays green (skipped) until the fixture lands.
_DATA_DIR = Path(__file__).parent.parent / "data"
_REAL_FRAMES = sorted(_DATA_DIR.glob("*.fits.bz2")) + sorted(
    _DATA_DIR.glob("*.fit.bz2"),
)
_REAL_FRAME = _REAL_FRAMES[0] if _REAL_FRAMES else None

_real_frame_required = pytest.mark.skipif(
    _REAL_FRAME is None,
    reason=f"no real Seestar fixture (*.fits.bz2) in {_DATA_DIR}",
)


@_real_frame_required
class TestSmokeRealFrame:
    """
    Smoke test: drive the real pipeline on a genuine Seestar frame.

    The two heavy externals (twirl's WCS solve, the Ballet CNN) are stubbed so
    the test is offline and deterministic; everything else -- real header parse,
    source detection, the median-PSF FWHM fit on real cutouts, the saturation
    cap, Bayer masks, and aperture photometry -- runs against genuine pixels.
    This is the realistic counterpart to the synthetic-FITS tests above and
    catches integration breakage they cannot.
    """

    def test_calibration_sequence_recovers_real_sources(self):
        """Detection + FWHM fit succeed and the real header resolves the template."""
        expected_max_adu = 50000  # from basic.json, keyed off the real header

        # calibration_sequence reaches neither twirl nor the Ballet CNN, so this
        # path needs no stubbing.
        calibrated, metadata, coords, fwhm, regions = calibration_sequence(
            str(_REAL_FRAME),
            threshold=THRESH,
        )

        assert calibrated is not None
        assert len(regions) >= MIN_DETECTED_STARS
        assert coords.shape == (len(regions), 2)
        assert np.isfinite(fwhm)
        assert fwhm > 0
        assert metadata["largest_usable_adu_value"] == expected_max_adu
        assert metadata["width"] == calibrated.shape[1]
        assert metadata["height"] == calibrated.shape[0]

    def test_process_one_image_builds_per_filter_tables(self, monkeypatch):
        """Every Bayer filter gets a non-empty table and L4 sums the RGB counts."""
        header = fits.getheader(str(_REAL_FRAME))
        data = fits.getdata(str(_REAL_FRAME))
        metadata = metadata_from_header(header)

        # Center the stubbed WCS on the real field so the cosmetic ra/dec columns
        # are plausible in a failure dump.
        _stub_wcs_and_centroid(
            monkeypatch,
            wcs_image_size=data.shape,
            wcs_crval=(header["RA"], header["DEC"]),
        )

        masks = generate_bayer_masks(
            data.shape,
            {
                "bayerpat": metadata["bayerpat"],
                "roworder": metadata["roworder"],
                "ybayroff": metadata["ybayroff"],
            },
            append_l4=True,
        )

        # twirl is stubbed, so radecs is never matched; it only needs >=
        # N_GAIA_STARS_ALIGN plausibly shaped rows (align slices the first
        # N_GAIA_STARS_ALIGN refs). photometry_coords=None means aligned ==
        # detections.
        radecs = np.column_stack(
            [
                np.full(N_GAIA_STARS_ALIGN, header["RA"]),
                np.full(N_GAIA_STARS_ALIGN, header["DEC"]),
            ],
        )

        result = process_one_image(str(_REAL_FRAME), {}, radecs, None, masks)

        assert set(result) == {"TR", "TG", "TB", "L4"}
        for table in result.values():
            assert len(table) > 0
            assert np.isfinite(table.meta["fwhm"])

        # L4 total count is the per-row RGB sum (same invariant as the synthetic
        # test; equal_nan handles any edge apertures that come back non-finite).
        rgb_sum = (
            result["TR"]["tot_count"]
            + result["TG"]["tot_count"]
            + result["TB"]["tot_count"]
        )
        np.testing.assert_allclose(result["L4"]["tot_count"], rgb_sum, equal_nan=True)

    def test_fwhm_cap_keeps_real_frame_fwhm_small(self):
        """
        The brightest-N cap bounds the real-frame FWHM fit and keeps it small.

        On genuine pixels the fit must (a) feed at most ``fwhm_n_stars`` of the
        detections to the PSF stack and (b) recover a small FWHM near the
        true PSF (~2.8 px) -- a regression guard against the re-inflation an
        uncapped fit over thousands of faint detections would smear back in.
        """
        calibrated, metadata, coords, fwhm, _ = calibration_sequence(
            str(_REAL_FRAME),
            threshold=THRESH,
        )
        max_adu = metadata["largest_usable_adu_value"]
        n_cap = InstrumentProfile().fwhm_n_stars

        # The cap selects at most n_cap unsaturated detections (fewer than the
        # full detection list) to build the PSF the FWHM is fit from.
        kept = _brightest_unsaturated(calibrated, coords, max_adu, n_cap)
        assert 0 < len(kept) <= n_cap
        assert len(kept) <= len(coords)

        # The fit calibration_sequence already ran (default cap) lands near the
        # true PSF, not the inflated ~8 px an uncapped CNN fit produced.
        fwhm_ceiling = 6.0  # true PSF ~2.8 px; well clear of the ~8 px inflation
        assert 0 < fwhm < fwhm_ceiling

    @pytest.mark.remote_data
    def test_real_ballet_cnn_fwhm_smoke(self):
        """
        Drive the *live* Ballet CNN on a real frame end-to-end.

        Every other test stubs the CNN, so this is the only coverage of the real
        ``Ballet`` path the FWHM cap exists to protect: weights download ->
        JAX inference -> ``ballet_centroid`` -> ePSF registration -> brightest-N
        cap -> a small FWHM. It guards against eloy API / weight-format drift the
        stubbed tests are blind to. ``Ballet()`` downloads ``centroid_15x15.npz``
        from the public ``lgrcia/ballet`` HuggingFace repo (no auth) on first run.
        """
        calibrated, metadata, coords, _, _ = calibration_sequence(
            str(_REAL_FRAME),
            threshold=THRESH,
        )
        max_adu = metadata["largest_usable_adu_value"]
        cnn = Ballet()

        n_cap = InstrumentProfile().fwhm_n_stars
        fwhm = _fwhm_from_coords(
            calibrated, coords, max_adu=max_adu, cnn=cnn, n_stars=n_cap
        )

        assert fwhm is not None
        fwhm_ceiling = 6.0  # true PSF ~2.8 px; well clear of the ~8 px inflation
        assert 0 < fwhm < fwhm_ceiling
