"""Unit tests for the detect/align/centroid pipeline (prepare_image, process)."""

import gzip
import warnings
from pathlib import Path

import numpy as np
import pytest
from _helpers import (
    SEED,
    SEESTAR_PIXSCALE,
    _make_tan_wcs,
    _seestar_header,
    five_diagonal_regions,
)
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.table import Table
from eloy import detection
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening

from bandaid import instruments, photometry
from bandaid.ballet import NumpyBallet
from bandaid.config import HeaderMatchRule, InstrumentProfile, PhotometryConfig
from bandaid.exceptions import (
    DegenerateBayerChannelError,
    FrameMetadataError,
    InstrumentDetectionError,
    NoUsableStarsError,
    TooFewStarsError,
    WCSSolveError,
)
from bandaid.image2sl_qt import bayer_balance_image, generate_bayer_masks
from bandaid.instruments import register_instrument
from bandaid.photometry import (
    DETECTION_OPENING,
    MIN_DETECTED_STARS,
    N_GAIA_STARS_ALIGN,
    THRESH,
    _MASK_INDEPENDENT_COLUMNS,
    ImageData,
    LoadedFrame,
    _box_opening,
    _brightest_unsaturated,
    _centroid_prior_summary,
    _detect_stars,
    _drop_edge_catalog_stars,
    _fwhm_from_coords,
    build_photometry_table,
    calibration_sequence,
    measure_photometry,
    metadata_from_header,
    prepare_image,
    process_one_image,
)
from bandaid.photometry import (
    centroid_stars as real_centroid_stars,
)
from bandaid.scripts import estimate_center_from_header


class TestPrepareImage:
    def test_no_photometry_coord_input(self, make_test_image, tmp_path, mocker):
        """Aligned coords fall back to detected coords when none are provided."""
        # This test only checks the alignment fallback, not centroiding, so stub
        # centroid_stars to avoid constructing the real Ballet CNN (which would pull
        # model weights from HuggingFace). The stub returns the aligned coords
        # unchanged.
        mocker.patch(
            "bandaid.photometry.centroid_stars",
            side_effect=lambda data, coords, cnn: coords,
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
            # This test only exercises the alignment fallback, not instrument
            # detection, and the written header carries no INSTRUME/TELESCOP.
            config=PhotometryConfig(instrument=InstrumentProfile()),
        )

        assert np.array_equal(img.coords, img.aligned_coords)

    def test_instrument_config_reaches_detection(self, stub_prepare_image_externals):
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
        externals = stub_prepare_image_externals()

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

        # The values travel via the profile: calibration_sequence resolves
        # the unset threshold/opening/fwhm_n_stars from it.
        profile = externals.calibration_sequence.call_args.kwargs["profile"]
        assert profile.thresh == expected_thresh
        assert profile.detection_opening == expected_opening
        assert profile.fwhm_n_stars == expected_fwhm_n_stars

    def test_stubbed_calibration_sequence_feeds_centroiding_when_balanced(
        self, stub_prepare_image_externals
    ):
        """
        The shared stub's detection image is what ``prepare_image`` centroids.

        ``prepare_image`` centroids the ``detection_image`` of the result
        ``calibration_sequence`` returns. Assert the stub's calibrated array
        reaches ``centroid_stars``.
        """
        calibrated = np.full((10, 10), 7.0)
        externals = stub_prepare_image_externals(calibrated=calibrated)

        prepare_image(
            "unused.fits",
            np.zeros((5, 2)),
            None,
            detect_on_bayer_balanced=True,
        )

        externals.centroid_stars.assert_called_once()
        assert externals.centroid_stars.call_args.args[0] is calibrated

    def test_auto_detects_instrument_from_frame_header(
        self, stub_prepare_image_externals
    ):
        """
        A default (``instrument=None``) config resolves by detecting the header.

        ``prepare_image`` is the other resolution point (besides
        ``prepare_batch``): a direct caller with a default config must get the
        same auto-detection, so ``calibration_sequence`` still sees a
        concrete profile.
        """
        externals = stub_prepare_image_externals()
        mocker_load_frame = externals.load_frame
        mocker_load_frame.side_effect = lambda _file: LoadedFrame(
            np.zeros((10, 10)), {"INSTRUME": "Seestar S50"}
        )

        prepare_image("unused.fits", np.zeros((5, 2)), None, config=PhotometryConfig())

        resolved = externals.calibration_sequence.call_args.kwargs["profile"]
        assert resolved.name == "Seestar50"

    def test_unmatched_header_raises_frame_metadata_error(
        self, stub_prepare_image_externals
    ):
        """
        A frame whose header matches no profile raises a per-frame error.

        ``prepare_image`` is a legitimate direct/single-frame entry point, so
        an unresolvable header is that frame's problem, not a batch-fatal one:
        ``InstrumentDetectionError`` is itself a `FrameMetadataError`, so it
        propagates with the file attached and is
        caught by the same ``except FrameError`` skip loop other frame errors
        are.
        """
        externals = stub_prepare_image_externals()
        externals.load_frame.side_effect = lambda _file: LoadedFrame(
            np.zeros((10, 10)), {}
        )

        with pytest.raises(FrameMetadataError) as exc_info:
            prepare_image(
                "unused.fits", np.zeros((5, 2)), None, config=PhotometryConfig()
            )

        assert exc_info.value.file == "unused.fits"
        assert isinstance(exc_info.value, InstrumentDetectionError)

    def test_instrument_wcs_scale_tolerance_reaches_alignment(
        self, stub_prepare_image_externals
    ):
        """
        The instrument's ``wcs_scale_tolerance`` is forwarded to ``align``.

        A non-default profile tolerance must reach the plate-scale check, so spy
        on ``align`` and assert the configured value arrives as ``scale_tolerance``.
        """
        expected_tolerance = 0.07
        externals = stub_prepare_image_externals()

        config = PhotometryConfig(
            instrument=InstrumentProfile(wcs_scale_tolerance=expected_tolerance),
        )
        prepare_image("unused.fits", np.zeros((5, 2)), None, config=config)

        assert externals.align.call_args.kwargs["scale_tolerance"] == expected_tolerance

    def test_instrument_wcs_pointing_tolerance_reaches_alignment(
        self, stub_prepare_image_externals
    ):
        """
        The instrument's ``wcs_pointing_tolerance`` is forwarded to ``align``.

        Spy on ``align`` and assert a non-default profile tolerance arrives as
        ``pointing_tolerance``.
        """
        expected_tolerance = 0.13
        externals = stub_prepare_image_externals()

        config = PhotometryConfig(
            instrument=InstrumentProfile(wcs_pointing_tolerance=expected_tolerance),
        )
        prepare_image("unused.fits", np.zeros((5, 2)), None, config=config)

        assert (
            externals.align.call_args.kwargs["pointing_tolerance"] == expected_tolerance
        )

    def test_missing_pixscale_raises_when_solving(self, stub_prepare_image_externals):
        """
        A missing/non-numeric ``pixscale`` fails loud instead of silent-skipping.

        ``pixscale`` comes from the instrument profile via
        ``metadata_from_header`` and is required to scale-check a solved WCS, so
        when it is absent and no WCS is supplied ``prepare_image`` raises
        ``FrameMetadataError`` rather than passing ``expected_pixscale=None``
        (which would quietly disable the check).
        """
        externals = stub_prepare_image_externals(metadata={"creator": "spy"})

        with pytest.raises(FrameMetadataError, match="pixscale"):
            prepare_image("unused.fits", np.zeros((5, 2)), None)

        # align must never run when the scale check cannot be performed.
        externals.align.assert_not_called()

    def test_missing_pixscale_ok_when_wcs_supplied(self, stub_prepare_image_externals):
        """
        A supplied WCS is trusted, so a missing ``pixscale`` is not required.

        ``align`` skips the scale check for a caller-supplied WCS, so
        ``prepare_image`` must not demand ``pixscale`` in that case; it forwards
        ``expected_pixscale=None`` without raising.
        """
        supplied_wcs = _make_tan_wcs()
        externals = stub_prepare_image_externals(metadata={"creator": "spy"})

        prepare_image("unused.fits", np.zeros((5, 2)), None, wcs=supplied_wcs)

        assert externals.align.call_args.kwargs["expected_pixscale"] is None

    def test_header_center_and_shape_reach_alignment(
        self, stub_prepare_image_externals
    ):
        """
        The header pointing and image shape are forwarded to ``align``.

        The Gaia catalog is queried at the header ra/dec, so ``prepare_image``
        must pass that location (as ``expected_center``) plus the frame shape to
        ``align`` for the solved-WCS in-frame check. Spy on ``align`` and assert
        both arrive.
        """
        externals = stub_prepare_image_externals(
            metadata={
                "creator": "spy",
                "pixscale": 2.4,
                "ra": 10.0,
                "dec": 20.0,
                "fov_rad": 1.0,
            },
            calibrated=np.zeros((10, 12)),
        )

        prepare_image(
            "unused.fits",
            np.zeros((5, 2)),
            None,
            config=PhotometryConfig(instrument=InstrumentProfile()),
        )

        kwargs = externals.align.call_args.kwargs
        center = kwargs["expected_center"]
        assert isinstance(center, SkyCoord)
        assert center.ra.deg == pytest.approx(10.0)
        assert center.dec.deg == pytest.approx(20.0)
        assert kwargs["shape"] == (10, 12)

    def test_missing_header_radec_raises(self, stub_prepare_image_externals):
        """
        A frame without usable header ra/dec cannot be solved.

        The per-frame solve pool is cut around the header pointing; without one
        the only alternative is the whole batch catalog, so the frame is
        rejected with the file named instead.
        """
        # metadata deliberately omits "ra"/"dec".
        externals = stub_prepare_image_externals(
            metadata={"creator": "spy", "pixscale": 2.4, "fov_rad": 1.0}
        )

        with pytest.raises(FrameMetadataError, match="pointing") as exc_info:
            prepare_image("unused.fits", np.zeros((5, 2)), None)

        assert exc_info.value.file == "unused.fits"
        externals.align.assert_not_called()

    def test_string_header_radec_still_reaches_alignment(
        self, stub_prepare_image_externals
    ):
        """
        String header ra/dec are coerced, not treated as missing.

        The ``@RA``/``@DEC`` directives return the raw header value, which the
        FITS header often stores as a numeric string; the airmass path already
        ``float(...)``-coerces them. ``prepare_image`` must do the same so the
        pointing check runs on real frames instead of silently skipping.
        """
        externals = stub_prepare_image_externals(
            # ra/dec as numeric strings, as they arrive from the FITS header.
            metadata={
                "creator": "spy",
                "pixscale": 2.4,
                "ra": "10.0",
                "dec": "20.0",
                "fov_rad": 1.0,
            },
        )

        prepare_image(
            "unused.fits",
            np.zeros((5, 2)),
            None,
            config=PhotometryConfig(instrument=InstrumentProfile()),
        )

        center = externals.align.call_args.kwargs["expected_center"]
        assert isinstance(center, SkyCoord)
        assert center.ra.deg == pytest.approx(10.0)
        assert center.dec.deg == pytest.approx(20.0)

    def test_header_center_converted_to_icrs_for_alignment(
        self, stub_prepare_image_externals
    ):
        """
        An equinox-of-date header pointing is converted before the center check.

        The Gaia cone is centered on the header pointing converted to ICRS, so
        the in-frame check must compare against that same converted location,
        not the raw header value.
        """
        metadata = {
            "creator": "spy",
            "pixscale": 2.4,
            "ra": 10.0,
            "dec": 20.0,
            "fov_rad": 1.0,
            "obs_time": "2025-09-09T05:00:00",
        }
        profile = InstrumentProfile(header_frame="fk5", header_equinox="date")
        externals = stub_prepare_image_externals(metadata=metadata)

        prepare_image(
            "unused.fits",
            np.zeros((5, 2)),
            None,
            config=PhotometryConfig(instrument=profile),
        )

        expected_ra, expected_dec = estimate_center_from_header(metadata, profile)
        center = externals.align.call_args.kwargs["expected_center"]
        assert center.ra.deg == pytest.approx(expected_ra)
        assert center.dec.deg == pytest.approx(expected_dec)
        # Precession over ~25 years moves the pointing by far more than this.
        assert center.ra.deg != pytest.approx(10.0, abs=0.1)

    def test_equinox_of_date_without_obs_time_raises(
        self, stub_prepare_image_externals
    ):
        """A pointing that cannot be converted to ICRS rejects the frame."""
        externals = stub_prepare_image_externals(
            metadata={
                "creator": "spy",
                "pixscale": 2.4,
                "ra": 10.0,
                "dec": 20.0,
                "fov_rad": 1.0,
            }
        )

        with pytest.raises(FrameMetadataError, match="obs_time") as exc_info:
            prepare_image(
                "unused.fits",
                np.zeros((5, 2)),
                None,
                config=PhotometryConfig(
                    instrument=InstrumentProfile(
                        header_frame="fk5", header_equinox="date"
                    )
                ),
            )

        assert exc_info.value.file == "unused.fits"
        externals.align.assert_not_called()

    @pytest.mark.parametrize(
        ("ra", "dec"),
        [
            ("N/A", "N/A"),
            (True, True),
            (True, 20.0),
            (10.0, True),
            (None, None),
            (10.0, 91.0),
        ],
        ids=[
            "non-numeric",
            "bool",
            "ra-bool",
            "dec-bool",
            "missing",
            "dec-91",
        ],
    )
    def test_unusable_header_radec_raises(self, stub_prepare_image_externals, ra, dec):
        """Header ra/dec that are not a usable sky position reject the frame."""
        externals = stub_prepare_image_externals(
            metadata={
                "creator": "spy",
                "pixscale": 2.4,
                "ra": ra,
                "dec": dec,
                "fov_rad": 1.0,
            }
        )

        with pytest.raises(FrameMetadataError, match="pointing"):
            prepare_image("unused.fits", np.zeros((5, 2)), None)

        externals.align.assert_not_called()

    @staticmethod
    def _pool_catalog(center, offsets_deg) -> np.ndarray:
        """Catalog stars at the given dec offsets (deg) from ``center``, in order."""
        ra, dec = center
        return np.array([[ra, dec + off] for off in offsets_deg])

    def test_solve_pool_cut_to_stars_near_header_center(
        self, stub_prepare_image_externals
    ):
        """
        ``align`` gets only catalog stars near the header pointing, in order.

        The brightest (first) stars are far off-field; the cut must drop them
        and keep the fainter near stars in their original brightest-first order.
        """
        fov_rad = 1.0
        externals = stub_prepare_image_externals(
            metadata={
                "creator": "spy",
                "pixscale": 2.4,
                "ra": 10.0,
                "dec": 20.0,
                "fov_rad": fov_rad,
            }
        )
        center = (10.0, 20.0)
        radecs = self._pool_catalog(center, [5.0, -4.0, 0.5, -0.2, 0.0, 3.0])

        prepare_image(
            "unused.fits",
            radecs,
            None,
            config=PhotometryConfig(instrument=InstrumentProfile()),
        )

        received = externals.align.call_args.args[1]
        np.testing.assert_array_equal(received, radecs[[2, 3, 4]])

    def test_solve_pool_radius_scales_with_config(self, stub_prepare_image_externals):
        """A star at 0.8 fov_rad is kept at scale 0.9 and dropped at scale 0.5."""
        fov_rad = 1.0
        metadata = {
            "creator": "spy",
            "pixscale": 2.4,
            "ra": 10.0,
            "dec": 20.0,
            "fov_rad": fov_rad,
        }
        radecs = self._pool_catalog((10.0, 20.0), [0.0, 0.8 * fov_rad])

        externals = stub_prepare_image_externals(metadata=metadata)
        prepare_image(
            "unused.fits",
            radecs,
            None,
            config=PhotometryConfig(
                instrument=InstrumentProfile(solve_pool_radius_scale=0.9)
            ),
        )
        assert len(externals.align.call_args.args[1]) == len(radecs)

        externals = stub_prepare_image_externals(metadata=metadata)
        prepare_image(
            "unused.fits",
            radecs,
            None,
            config=PhotometryConfig(
                instrument=InstrumentProfile(solve_pool_radius_scale=0.5)
            ),
        )
        assert len(externals.align.call_args.args[1]) == len(radecs) - 1

    @pytest.mark.parametrize(
        "metadata_update",
        [
            {},
            {"fov_rad": None},
            {"fov_rad": "wide"},
            {"fov_rad": True},
            {"fov_rad": 0.0},
            {"fov_rad": -1.0},
            {"fov_rad": float("nan")},
            {"fov_rad": float("inf")},
        ],
    )
    def test_solve_pool_without_usable_fov_rad_raises(
        self, stub_prepare_image_externals, metadata_update
    ):
        """A missing or unusable ``fov_rad`` rejects the frame."""
        externals = stub_prepare_image_externals(
            metadata={
                "creator": "spy",
                "pixscale": 2.4,
                "ra": 10.0,
                "dec": 20.0,
                **metadata_update,
            }
        )
        radecs = self._pool_catalog((10.0, 20.0), [5.0, 0.0, -4.0])

        with pytest.raises(FrameMetadataError, match="fov_rad") as exc_info:
            prepare_image(
                "unused.fits",
                radecs,
                None,
                config=PhotometryConfig(instrument=InstrumentProfile()),
            )

        assert exc_info.value.file == "unused.fits"
        externals.align.assert_not_called()

    @pytest.mark.parametrize("fov_rad", [np.float32(1.0), "1.0", 1])
    def test_solve_pool_coerces_fov_rad(self, stub_prepare_image_externals, fov_rad):
        """A numpy scalar, numeric string or int ``fov_rad`` still cuts the pool."""
        externals = stub_prepare_image_externals(
            metadata={
                "creator": "spy",
                "pixscale": 2.4,
                "ra": 10.0,
                "dec": 20.0,
                "fov_rad": fov_rad,
            }
        )
        radecs = self._pool_catalog((10.0, 20.0), [5.0, 0.0, -4.0])

        prepare_image(
            "unused.fits",
            radecs,
            None,
            config=PhotometryConfig(instrument=InstrumentProfile()),
        )

        np.testing.assert_array_equal(externals.align.call_args.args[1], radecs[[1]])

    def test_solve_pool_full_catalog_with_supplied_wcs(
        self, stub_prepare_image_externals
    ):
        """A caller-supplied WCS skips the cut; the full catalog reaches ``align``."""
        externals = stub_prepare_image_externals(
            metadata={
                "creator": "spy",
                "pixscale": 2.4,
                "ra": 10.0,
                "dec": 20.0,
                "fov_rad": 1.0,
            }
        )
        radecs = self._pool_catalog((10.0, 20.0), [5.0, 0.0, -4.0])

        prepare_image("unused.fits", radecs, None, wcs=_make_tan_wcs())

        assert externals.align.call_args.args[1] is radecs

    def test_wcs_solve_error_reports_pool_size(self, stub_prepare_image_externals):
        """A failed solve says how many catalog stars were in the frame's pool."""
        externals = stub_prepare_image_externals(
            metadata={
                "creator": "spy",
                "pixscale": 2.4,
                "ra": 10.0,
                "dec": 20.0,
                "fov_rad": 1.0,
            }
        )
        externals.align.side_effect = WCSSolveError("no match")
        radecs = self._pool_catalog((10.0, 20.0), [5.0, 0.0, 0.1, -4.0])

        with pytest.raises(WCSSolveError) as exc_info:
            prepare_image(
                "unused.fits",
                radecs,
                None,
                config=PhotometryConfig(instrument=InstrumentProfile()),
            )

        text = str(exc_info.value)
        assert "no match" in text
        assert "unused.fits" in text
        assert "solve pool: 2 catalog stars" in text

    def test_off_frame_catalog_stars_dropped_before_centroiding(
        self, stub_prepare_image_externals
    ):
        """Off-frame catalog stars never reach centroiding, aligned or input."""
        aligned = np.array(
            [[50.0, 50.0], [-50.0, 50.0], [60.0, 70.0], [200.0, 50.0]],
        )
        externals = stub_prepare_image_externals(
            coords=aligned, calibrated=np.zeros((100, 100))
        )
        photometry_coords = SkyCoord(
            ra=[1.0, 2.0, 3.0, 4.0], dec=[0.0, 0.0, 0.0, 0.0], unit="deg"
        )

        img = prepare_image(
            "unused.fits",
            np.zeros((5, 2)),
            None,
            photometry_coords=photometry_coords,
        )

        kept = [0, 2]
        assert np.array_equal(externals.centroid_stars.call_args[0][1], aligned[kept])
        assert len(img.input_photometry_coords) == len(kept)
        assert len(img.aligned_coords) == len(kept)
        assert np.array_equal(
            img.input_photometry_coords.ra.deg, photometry_coords.ra.deg[kept]
        )
        assert np.array_equal(
            img.input_photometry_coords.dec.deg, photometry_coords.dec.deg[kept]
        )

    def test_edge_cut_uses_width_and_height_separately(
        self, stub_prepare_image_externals
    ):
        """The edge cut checks x against width and y against height, not swapped."""
        aligned = np.array([[150.0, 30.0], [30.0, 150.0]])
        externals = stub_prepare_image_externals(
            calibrated=np.zeros((60, 200)), coords=aligned
        )
        photometry_coords = SkyCoord(ra=[1.0, 2.0], dec=[0.0, 0.0], unit="deg")

        prepare_image(
            "unused.fits",
            np.zeros((5, 2)),
            None,
            photometry_coords=photometry_coords,
        )

        assert np.array_equal(externals.centroid_stars.call_args[0][1], aligned[[0]])

    def test_edge_cut_keeps_stars_exactly_one_margin_from_each_edge(
        self, stub_prepare_image_externals
    ):
        """The span is half-open like ``good_star_mask``: high edge dropped."""
        margin = 10.0
        height, width = 80, 120
        x_hi, y_hi = width - 0.5 - margin, height - 0.5 - margin
        eps = 0.1
        aligned = np.array(
            [
                [margin, 40.0],
                [margin - eps, 40.0],
                [x_hi - eps, 40.0],
                [x_hi, 40.0],
                [60.0, margin],
                [60.0, margin - eps],
                [60.0, y_hi - eps],
                [60.0, y_hi],
            ],
        )
        externals = stub_prepare_image_externals(
            coords=aligned, calibrated=np.zeros((height, width))
        )
        photometry_coords = SkyCoord(
            ra=np.arange(8, dtype=float), dec=np.zeros(8), unit="deg"
        )

        prepare_image(
            "unused.fits",
            np.zeros((5, 2)),
            None,
            photometry_coords=photometry_coords,
        )

        kept = [0, 2, 4, 6]
        assert np.array_equal(externals.centroid_stars.call_args[0][1], aligned[kept])

    def test_edge_cut_follows_configured_margin(self, stub_prepare_image_externals):
        """A star 5 px from an edge is dropped at the default margin, kept at 4 px."""
        aligned = np.array([[5.0, 50.0], [50.0, 50.0]])
        photometry_coords = SkyCoord(ra=[1.0, 2.0], dec=[0.0, 0.0], unit="deg")
        externals = stub_prepare_image_externals(
            coords=aligned, calibrated=np.zeros((100, 100))
        )

        prepare_image(
            "unused.fits", np.zeros((5, 2)), None, photometry_coords=photometry_coords
        )
        default_kept = externals.centroid_stars.call_args[0][1]
        prepare_image(
            "unused.fits",
            np.zeros((5, 2)),
            None,
            config=PhotometryConfig(edge_margin_px=4.0),
            photometry_coords=photometry_coords,
        )
        narrow_kept = externals.centroid_stars.call_args[0][1]

        assert np.array_equal(default_kept, aligned[[1]])
        assert np.array_equal(narrow_kept, aligned)

    def test_edge_star_never_reaches_the_cnn(
        self, stub_prepare_image_externals, mocker
    ):
        """A catalog star projected 3 px inside an edge is not sent to the CNN."""
        aligned = np.array([[50.0, 50.0], [3.0, 50.0]])
        stub_prepare_image_externals(coords=aligned, calibrated=np.zeros((100, 100)))
        mocker.patch("bandaid.photometry.centroid_stars", new=real_centroid_stars)
        ballet_centroid = mocker.patch(
            "bandaid.photometry.centroid.ballet_centroid",
            side_effect=lambda _data, coords, _cnn: coords,
        )
        photometry_coords = SkyCoord(ra=[1.0, 2.0], dec=[0.0, 0.0], unit="deg")

        img = prepare_image(
            "unused.fits", np.zeros((5, 2)), None, photometry_coords=photometry_coords
        )

        assert np.array_equal(ballet_centroid.call_args[0][1], aligned[[0]])
        assert np.array_equal(img.centroid_coords, aligned[[0]])
        assert len(img.input_photometry_coords) == 1

    def test_gaia_g_is_cut_with_the_coordinates(self):
        """Gaia G is dropped with its star so each G still matches its row."""
        aligned = np.array([[50.0, 50.0], [3.0, 50.0], [60.0, 70.0], [-50.0, 5.0]])
        coords = SkyCoord(ra=np.arange(4, dtype=float), dec=np.zeros(4), unit="deg")
        gaia_g = np.array([8.0, 9.0, 10.0, 11.0])

        _out_aligned, _out_coords, out_g, _n_dropped = _drop_edge_catalog_stars(
            aligned,
            coords,
            (100, 100),
            "unused.fits",
            edge_margin_px=10.0,
            gaia_g=gaia_g,
        )

        np.testing.assert_array_equal(out_g, gaia_g[[0, 2]])

    def test_gaia_g_is_none_without_a_catalog(self):
        """Without catalog coordinates there is no G to return."""
        aligned = np.array([[50.0, 50.0]])

        _out_aligned, out_coords, out_g, n_dropped = _drop_edge_catalog_stars(
            aligned, None, (100, 100), "unused.fits", edge_margin_px=10.0
        )

        assert out_coords is None
        assert out_g is None
        assert n_dropped == 0

    def test_policy_receives_cut_gaia_g_cut_and_config(
        self, stub_prepare_image_externals, mocker
    ):
        """The centroid policy gets the cut G, the magnitude cut and the config."""
        aligned = np.array(
            [[50.0, 50.0], [-50.0, 5.0], [60.0, 70.0], [200.0, 5.0], [3.0, 50.0]]
        )
        stub_prepare_image_externals(coords=aligned, calibrated=np.zeros((100, 100)))
        spy = mocker.spy(photometry, "centroid_with_prior")
        photometry_coords = SkyCoord(ra=np.arange(5.0), dec=np.zeros(5), unit="deg")
        gaia_g = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
        config = PhotometryConfig(instrument=InstrumentProfile())
        g_cut = 9.5

        prepare_image(
            "unused.fits",
            np.zeros((5, 2)),
            None,
            config=config,
            photometry_coords=photometry_coords,
            gaia_g=gaia_g,
            g_cut=g_cut,
        )

        # Rows 1 and 3 are off-frame and row 4 is inside the edge margin.
        np.testing.assert_array_equal(spy.call_args.kwargs["gaia_g"], gaia_g[[0, 2]])
        assert spy.call_args.kwargs["g_cut"] == g_cut
        assert spy.call_args.kwargs["config"] is config.centroid

    def test_image_data_records_how_each_star_was_centroided(
        self, stub_prepare_image_externals, mocker
    ):
        """``ImageData`` carries the per-row method, expected positions and summary."""
        aligned = np.array([[50.0, 50.0], [60.0, 70.0]])
        stub_prepare_image_externals(coords=aligned, calibrated=np.zeros((100, 100)))
        method = np.array(["cnn", "plane"])
        expected = aligned + 0.25
        g_cut = 9.5
        mocker.patch(
            "bandaid.photometry.centroid_with_prior",
            return_value=photometry.CentroidResult(
                coords=aligned,
                method=method,
                expected=expected,
                plane=None,
                fallback=True,
                active=True,
            ),
        )

        img = prepare_image(
            "unused.fits",
            np.zeros((5, 2)),
            None,
            photometry_coords=SkyCoord(ra=[1.0, 2.0], dec=[0.0, 0.0], unit="deg"),
            gaia_g=np.array([8.0, 9.0]),
            g_cut=g_cut,
        )

        np.testing.assert_array_equal(img.centroid_method, method)
        np.testing.assert_array_equal(img.centroid_expected, expected)
        assert img.centroid_prior["n_cnn_class"] == 1
        assert img.centroid_prior["g_cut"] == g_cut
        assert img.centroid_prior["plane_fallback"] is True
        assert img.centroid_prior["plane_n_used"] == 0

    def test_centroid_prior_summary_reports_the_fitted_plane(self):
        """The summary gives the plane's counts, rms, centre offset and slopes."""
        plane = photometry.OffsetPlane(
            coeffs_x=np.array([0.3, 0.1, -0.05]),
            coeffs_y=np.array([-0.2, 0.04, 0.08]),
            shape=(200, 300),
            n_used=28,
            n_clipped=2,
            rms=0.12,
        )
        result = photometry.CentroidResult(
            coords=np.zeros((3, 2)),
            method=np.array(["cnn", "plane", "plane"]),
            expected=np.zeros((3, 2)),
            plane=plane,
            active=True,
        )

        summary = _centroid_prior_summary(result, 9.5)

        assert summary == {
            "g_cut": 9.5,
            "n_cnn_class": 1,
            "plane_fallback": False,
            "plane_n_used": 28,
            "plane_n_clipped": 2,
            "plane_rms": 0.12,
            "plane_dx_center": 0.3,
            "plane_dx_slope_x": 0.1,
            "plane_dx_slope_y": -0.05,
            "plane_dy_center": -0.2,
            "plane_dy_slope_x": 0.04,
            "plane_dy_slope_y": 0.08,
        }

    def test_centroid_prior_summary_is_none_when_the_policy_did_not_run(self):
        """A frame centroided without the policy has no summary."""
        result = photometry.CentroidResult(
            coords=np.zeros((1, 2)),
            method=np.array(["cnn"]),
            expected=np.zeros((1, 2)),
        )

        assert _centroid_prior_summary(result, None) is None

    def test_policy_gets_no_gaia_g_without_a_catalog(
        self, stub_prepare_image_externals, mocker
    ):
        """Detected coordinates are not catalog projections: no G reaches the policy."""
        stub_prepare_image_externals()
        spy = mocker.spy(photometry, "centroid_with_prior")

        prepare_image(
            "unused.fits",
            np.zeros((5, 2)),
            None,
            gaia_g=np.array([8.0, 9.0, 10.0]),
            g_cut=9.5,
        )

        assert spy.call_args.kwargs["gaia_g"] is None

    def test_appended_forced_target_inside_the_margin_is_dropped(
        self, stub_prepare_image_externals, mocker
    ):
        """The last (forced-target) row gets no exemption from the edge margin."""
        aligned = np.array([[50.0, 50.0], [60.0, 60.0], [3.0, 60.0]])
        stub_prepare_image_externals(coords=aligned, calibrated=np.zeros((100, 100)))
        mocker.patch("bandaid.photometry.centroid_stars", new=real_centroid_stars)
        ballet_centroid = mocker.patch(
            "bandaid.photometry.centroid.ballet_centroid",
            side_effect=lambda _data, coords, _cnn: coords,
        )
        photometry_coords = SkyCoord(ra=[1.0, 2.0, 3.0], dec=[0.0] * 3, unit="deg")

        img = prepare_image(
            "unused.fits", np.zeros((5, 2)), None, photometry_coords=photometry_coords
        )

        assert np.array_equal(ballet_centroid.call_args[0][1], aligned[[0, 1]])
        assert np.array_equal(img.input_photometry_coords.ra.deg, [1.0, 2.0])

    def test_image_data_records_the_edge_drop_count(self, stub_prepare_image_externals):
        """``ImageData.n_edge_dropped`` is the number of stars the margin removed."""
        aligned = np.array([[50.0, 50.0], [3.0, 50.0], [60.0, 97.0], [-50.0, 5.0]])
        stub_prepare_image_externals(coords=aligned, calibrated=np.zeros((100, 100)))
        photometry_coords = SkyCoord(ra=[1.0, 2.0, 3.0, 4.0], dec=[0.0] * 4, unit="deg")

        img = prepare_image(
            "unused.fits", np.zeros((5, 2)), None, photometry_coords=photometry_coords
        )

        # Two stars sit inside the margin; the far off-frame one is not an edge drop.
        assert img.n_edge_dropped == len([1, 2])

    def test_edge_drop_count_covers_the_margin_on_both_sides(
        self, stub_prepare_image_externals
    ):
        """Dropped stars within the margin outside the frame count, farther ones not."""
        aligned = np.array(
            [[50.0, 50.0], [-9.0, 50.0], [-11.0, 50.0], [109.0, 50.0], [111.0, 50.0]]
        )
        stub_prepare_image_externals(coords=aligned, calibrated=np.zeros((100, 100)))
        photometry_coords = SkyCoord(ra=np.arange(5.0), dec=np.zeros(5), unit="deg")

        img = prepare_image(
            "unused.fits", np.zeros((5, 2)), None, photometry_coords=photometry_coords
        )

        assert img.n_edge_dropped == len([1, 3])

    def test_all_stars_inside_the_margin_raises(self, stub_prepare_image_externals):
        """With every star inside the margin, NoUsableStarsError names the file."""
        aligned = np.array([[3.0, 50.0], [50.0, 97.0]])
        stub_prepare_image_externals(coords=aligned, calibrated=np.zeros((100, 100)))
        photometry_coords = SkyCoord(ra=[1.0, 2.0], dec=[0.0, 0.0], unit="deg")

        with pytest.raises(NoUsableStarsError) as exc_info:
            prepare_image(
                "unused.fits",
                np.zeros((5, 2)),
                None,
                photometry_coords=photometry_coords,
            )

        assert exc_info.value.file == "unused.fits"

    def test_no_catalog_leaves_aligned_coords_untouched(
        self, stub_prepare_image_externals
    ):
        """With no catalog, the in-frame cut is skipped; aligned coords pass whole."""
        aligned = np.array([[5.0, 5.0], [-50.0, 5.0]])
        externals = stub_prepare_image_externals(coords=aligned)

        img = prepare_image(
            "unused.fits", np.zeros((5, 2)), None, photometry_coords=None
        )

        assert np.array_equal(img.aligned_coords, aligned)
        assert np.array_equal(externals.centroid_stars.call_args[0][1], aligned)

    def test_no_catalog_star_in_frame_raises(self, stub_prepare_image_externals):
        """When every catalog star is off-frame, NoUsableStarsError names the file."""
        aligned = np.array([[-50.0, 5.0], [200.0, 5.0]])
        stub_prepare_image_externals(coords=aligned)
        photometry_coords = SkyCoord(ra=[1.0, 2.0], dec=[0.0, 0.0], unit="deg")
        path = "unused.fits"

        with pytest.raises(NoUsableStarsError) as exc_info:
            prepare_image(
                path, np.zeros((5, 2)), None, photometry_coords=photometry_coords
            )

        assert exc_info.value.file == path


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
    positions=None,
):
    """
    Build a noisy multi-Gaussian frame that eloy's detection can resolve.

    Parameters
    ----------
    make_test_image : callable
        The ``make_test_image`` factory fixture.
    n_sources : int, optional
        Number of sources; positions default to ``_SOURCE_POSITIONS[:n_sources]``.
        By default 5.
    fwhm : float, optional
        FWHM of every source in pixels. By default 4.0.
    amplitude : float or sequence of float, optional
        Peak amplitude shared by all sources, or one value per source. Keep it
        far above ``noise_stddev`` so detection is reliable; a value above the
        50000 ADU saturation cap exercises the saturated path in
        ``calibration_sequence``. By default 600.0.
    image_size : tuple of int, optional
        Frame shape ``(ny, nx)``. By default ``(480, 480)``.
    noise_mean : float, optional
        Mean of the Gaussian sky noise. By default 100.0.
    noise_stddev : float, optional
        Standard deviation of the Gaussian sky noise. By default 2.0.
    include_noise : bool, optional
        Pass False for the "too few stars" frames so detection returns exactly
        ``n_sources`` regardless of the threshold/opening (flat Gaussian noise
        at the low production threshold spawns spurious blobs that would
        otherwise pad the count past the floor). By default True.
    positions : sequence of tuple of float, optional
        ``(x, y)`` source positions, overriding ``_SOURCE_POSITIONS[:n_sources]``;
        length must equal ``n_sources``. A position can sit outside
        ``image_size`` to exercise an edge-clipped source. By default None.

    Returns
    -------
    numpy.ndarray
        The synthesized frame.
    """
    sigma = fwhm * gaussian_fwhm_to_sigma
    positions = _SOURCE_POSITIONS[:n_sources] if positions is None else positions
    amplitudes = list(amplitude) if np.ndim(amplitude) else [amplitude] * n_sources
    source_properties = Table(
        {
            "amplitude": amplitudes,
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
    # must be present; the others feed "@KEY" lookups used downstream. INSTRUME
    # is what a default (instrument=None) PhotometryConfig auto-detects on.
    ccd.header["CREATOR"] = "ZWO Seestar S50"
    ccd.header["INSTRUME"] = "Seestar S50"
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


# ICRS field center of the frames _write_seestar_fits writes: the Seestar50
# profile reads their header RA/DEC as equinox-of-date, so the solved WCS must
# sit at the converted pointing to pass the in-frame check.
_SYNTHETIC_FIELD_CENTER = estimate_center_from_header(
    {"ra": 10.0, "dec": 20.0, "obs_time": "2024-01-01T00:00:00"},
    InstrumentProfile(header_frame="fk5", header_equinox="date"),
)


def _stub_wcs_and_centroid(
    mocker,
    *,
    wcs_image_size=(500, 500),
    wcs_crval=_SYNTHETIC_FIELD_CENTER,
):
    """
    Stub the slow/networked externals reached via ``prepare_image``.

    ``compute_wcs`` (twirl's stochastic asterism solver) returns a fixed TAN WCS
    and ``centroid_stars`` (the HuggingFace-backed Ballet CNN) returns its input
    coordinates unchanged. A test that needs to inspect the image actually
    handed to centroiding can read it off the returned centroid mock's
    ``.call_args``.

    ``wcs_image_size``/``wcs_crval`` size and center the stubbed TAN WCS; the
    defaults match the synthetic-FITS callers, while the real-frame smoke test
    passes the actual frame shape and field center so the cosmetic RA/Dec columns
    land near the real field.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        The pytest-mock fixture used to install the stubs.
    wcs_image_size : tuple[int, int], optional
        Pixel shape (ny, nx) used to build the stubbed TAN WCS.
    wcs_crval : tuple[float, float], optional
        Field center (ra, dec) in degrees used to build the stubbed TAN WCS.

    Returns
    -------
    unittest.mock.MagicMock
        The mock installed in place of ``compute_wcs``.
    unittest.mock.MagicMock
        The mock installed in place of ``centroid_stars``; call
        ``.call_args`` on it to inspect the coordinates/image a test handed
        to centroiding.
    """
    wcs_mock = mocker.patch(
        "bandaid.photometry.compute_wcs",
        return_value=_make_tan_wcs(wcs_image_size, wcs_crval),
    )

    centroid_mock = mocker.patch(
        "bandaid.photometry.centroid_stars",
        side_effect=lambda data, coords, _cnn: coords,
    )

    return wcs_mock, centroid_mock


@pytest.fixture
def fromfile_spy(mocker):
    """
    Factory installing a spy on ``fits.HDUList.fromfile`` to count file opens.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        The pytest-mock fixture used to install the spy.

    Returns
    -------
    collections.abc.Callable
        Zero-argument callable installing and returning the spy (a
        `unittest.mock.MagicMock` wrapping ``fits.HDUList.fromfile``); assert
        on the spy's ``call_count``.

    Notes
    -----
    Spying at the HDUList level rather than at ``fits.open`` (or ``getdata``,
    ``getheader``) is deliberate: every one of those convenience functions
    funnels through ``fromfile``, so a reintroduced extra read is counted no
    matter which of them performs it. This does couple the test to an
    astropy-internal classmethod, but only in this one place, so an astropy
    release that reroutes how ``fits.open`` constructs an ``HDUList`` gives a
    single spot to re-verify; the failure mode if that ever happens is a
    silent under-count that keeps the test green while counting nothing.

    A factory rather than the spy itself because ``writeto`` also funnels
    through ``fromfile``: the spy must be installed *after* the test writes
    its FITS fixture file, or the write is counted as an open.
    """
    return lambda: mocker.spy(fits.HDUList, "fromfile")


class TestDetectStars:
    """
    `_detect_stars` is a drop-in for eloy's `stars_detection`, only faster.

    The algorithm (global threshold, square-kernel binary opening, labelling,
    brightest-first ordering) is unchanged; these tests pin the pieces that must
    stay bit-identical to what the plate solve and FWHM fit were tuned on.
    """

    @pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 7, 32, 33])
    def test_box_opening_matches_skimage(self, size):
        """
        The box-filter opening reproduces skimage's binary_opening exactly.

        Border handling is the subtle part (skimage pads True for the erosion
        and False for the dilation), so alongside random masks the sample holds
        blobs that touch every edge and every corner. Even kernel sizes are
        included because ``detection_opening`` only requires ``>= 1``, and so
        are sizes >= 32, where a fixed erosion threshold of 0.999 would let a
        window holding a single False pixel pass as all-True (1 - 1/32**2 >
        0.999); the solid mask with one hole is what catches that.
        """
        rng = np.random.default_rng(0)
        masks = [rng.random((40, 41)) < p for p in (0.3, 0.6)]

        one_hole = np.ones((40, 41), dtype=bool)
        one_hole[20, 20] = False
        masks.append(one_hole)

        blobs = np.zeros((40, 41), dtype=bool)
        blobs[0:6, 0:6] = True  # each corner
        blobs[0:6, -6:] = True
        blobs[-6:, 0:6] = True
        blobs[-6:, -6:] = True
        blobs[0:3, 15:25] = True  # thin strips along each edge
        blobs[-3:, 15:25] = True
        blobs[15:25, 0:3] = True
        blobs[15:25, -3:] = True
        blobs[18:24, 18:26] = True  # interior block
        masks.append(blobs)

        for mask in masks:
            expected = binary_opening(mask, np.ones((size, size)))
            np.testing.assert_array_equal(_box_opening(mask, size), expected)

    def test_regions_are_ordered_brightest_first(self, make_test_image):
        """
        Regions come back sorted by peak intensity, descending.

        ``align`` slices the brightest detections by list position, so the
        order is load-bearing: four sources of distinct amplitude must come
        back in amplitude order regardless of where they sit on the frame.

        The fixture is built to catch three wrong implementations that a
        simpler brightest-first test lets through: sorting by ``area`` instead
        of ``intensity_max`` (edge-clipping the dimmest source knocks its area
        out of amplitude order, since area and peak both grow with amplitude
        for an unclipped Gaussian); passing a transposed intensity image to
        ``regionprops`` (only visible on a non-square frame, where the
        mismatched shape raises); and substituting
        ``scipy.ndimage.binary_opening`` for ``_box_opening`` (its border rule
        drops the edge-clipped source outright at ``opening=5``, where
        ``_box_opening``/skimage keeps it). All four sources -- including the
        edge-clipped one -- are detected at ``opening=5``.
        """
        fwhm = 4.0
        sigma = fwhm * gaussian_fwhm_to_sigma
        noise_mean = 100.0
        amplitudes = [700.0, 300.0, 500.0, 400.0]
        # Off-diagonal, non-square-frame positions; the last source sits 1 px
        # past the left edge, so only a thin crescent of its above-threshold
        # core is on-frame -- thinner than the opening=5 kernel.
        positions = [(100, 300), (300, 100), (400, 380), (-1, 240)]
        image = _detectable_image(
            make_test_image,
            n_sources=4,
            fwhm=fwhm,
            amplitude=amplitudes,
            positions=positions,
            image_size=(480, 500),
            noise_mean=noise_mean,
        )

        regions = _detect_stars(image, threshold=5, opening=5)

        assert len(regions) == len(amplitudes)
        peaks = [r.intensity_max for r in regions]
        assert peaks == sorted(peaks, reverse=True)

        # The on-frame sources peak at their nominal amplitude plus the noise
        # floor; the clipped source's on-frame peak is attenuated by its 1 px
        # offset past the edge.
        clip_offset = 1.0
        attenuation = np.exp(-(clip_offset**2) / (2 * sigma**2))
        expected_peaks = [
            amplitude + noise_mean if x >= 0 else amplitude * attenuation + noise_mean
            for amplitude, (x, _y) in zip(amplitudes, positions, strict=True)
        ]
        expected_order = np.argsort(expected_peaks)[::-1]
        for region, source_index in zip(regions, expected_order, strict=True):
            np.testing.assert_allclose(
                region.intensity_max, expected_peaks[source_index], atol=30
            )

        # Centroids pin identity for the three on-frame sources directly; the
        # edge-clipped source's centroid sits near the frame boundary, not at
        # its (off-frame) nominal position.
        edge_hug_px = 3  # well inside the opening=5 kernel width
        for region, source_index in zip(regions, expected_order, strict=True):
            x_expected, y_expected = positions[source_index]
            if x_expected < 0:
                assert region.centroid[1] < edge_hug_px
                continue
            np.testing.assert_allclose(
                region.centroid, (y_expected, x_expected), atol=0.5
            )

    def test_nan_in_background_does_not_change_detections(self, make_test_image):
        """
        A NaN pixel in empty sky leaves the region count and centroids unchanged.

        NaNs reach detection only via calibration/masking, but an unguarded
        median would turn a single NaN into a NaN threshold and, silently, zero
        detections.
        """
        image = _detectable_image(make_test_image, n_sources=5)
        clean = _detect_stars(image, threshold=5, opening=3)

        with_nan = image.copy()
        with_nan[5, 5] = np.nan
        regions = _detect_stars(with_nan, threshold=5, opening=3)

        assert len(regions) == len(clean)
        np.testing.assert_array_equal(
            [r.centroid for r in regions], [r.centroid for r in clean]
        )

    def test_nan_inside_star_is_filled_with_the_median(self, make_test_image):
        """
        A NaN on a star's peak is treated as sky, and the star still detects.

        Pins the fill rule: NaN pixels are replaced by the median of the
        finite pixels before thresholding. A single NaN barely moves the
        threshold estimator (also computed over finite pixels only), so the
        result equals detection on the explicitly filled image here even
        though the two are computed from different intermediate arrays. With
        a 3 px opening the ring around the hole survives erosion and the
        dilation closes the hole, so the region is still found.
        """
        image = _detectable_image(make_test_image, n_sources=5)
        n_clean = len(_detect_stars(image, threshold=5, opening=3))

        with_nan = image.copy()
        x_peak, y_peak = _SOURCE_POSITIONS[0]
        with_nan[y_peak, x_peak] = np.nan
        filled = np.where(np.isnan(with_nan), np.nanmedian(with_nan), with_nan)

        regions = _detect_stars(with_nan, threshold=5, opening=3)
        expected = _detect_stars(filled, threshold=5, opening=3)

        assert len(regions) == n_clean
        np.testing.assert_array_equal(
            [r.centroid for r in regions], [r.centroid for r in expected]
        )
        assert any(
            np.hypot(r.centroid[0] - y_peak, r.centroid[1] - x_peak) < 1.0
            for r in regions
        )

    def test_estimator_matches_eloy_on_a_large_nan_block(self, make_test_image):
        """
        A large NaN block reproduces eloy's finite-pixels-only estimator.

        Eloy's ``stars_detection`` computes its median/sigma with
        ``nanmedian``/``nanstd``, which exclude NaN. Filling NaN with the
        median *before* computing the estimator (rather than after) shrinks
        the apparent sky sigma once the NaN block is large enough, shifting
        which pixels clear threshold. The reference below reimplements
        eloy's finite-pixels-only estimator directly and applies it to the
        median-filled image (matching the intensity image regionprops sees),
        pinning that ``_detect_stars`` reproduces it exactly.
        """
        image = _detectable_image(make_test_image, n_sources=1)
        with_nan = image.copy()
        # ~34% of the frame, well clear of the source at (60, 60).
        with_nan[200:480, 200:480] = np.nan

        regions = _detect_stars(with_nan, threshold=5, opening=3)

        finite = with_nan[np.isfinite(with_nan)]
        median = np.median(finite)
        core = finite[np.abs(finite - median) < np.std(finite)]
        core_std = np.std(core)
        filled = np.where(np.isnan(with_nan), median, with_nan)
        mask = _box_opening(filled > 5 * core_std + median, 3)
        expected = sorted(
            regionprops(label(mask), filled),
            key=lambda r: r.intensity_max,
            reverse=True,
        )

        assert len(regions) == len(expected) > 0
        np.testing.assert_array_equal(
            [r.centroid for r in regions], [r.centroid for r in expected]
        )

    @pytest.mark.parametrize("fill", [np.inf, -np.inf], ids=["+inf", "-inf"])
    def test_non_finite_block_in_background_is_treated_as_sky(
        self, make_test_image, fill
    ):
        """
        A block of +/-inf pixels in empty sky is filled and detected like NaN.

        ``np.isnan`` alone lets an inf pixel through: ``np.std`` on a flat
        containing it returns NaN, the core selection comes back empty, and
        the empty-core guard silently sets ``core_std = 0``, returning a
        normal-looking (but poisoned) detection list instead of raising or
        excluding the pixel. The guard must catch non-finite pixels
        generally, and the fill median must be computed over the finite
        pixels only -- ``np.nanmedian`` passes +/-inf straight through.
        """
        image = _detectable_image(make_test_image, n_sources=5)
        block = image.copy()
        block[5:15, 5:15] = fill
        finite = block[np.isfinite(block)]
        filled = np.where(np.isfinite(block), block, np.median(finite))

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            regions = _detect_stars(block, threshold=5, opening=3)
        expected = _detect_stars(filled, threshold=5, opening=3)

        assert len(regions) == len(expected)
        np.testing.assert_array_equal(
            [r.centroid for r in regions], [r.centroid for r in expected]
        )

    @pytest.mark.parametrize(
        "image",
        [np.full((50, 60), np.nan), np.full((50, 60), 100.0)],
        ids=["all-nan", "constant"],
    )
    def test_degenerate_images_yield_no_regions_quietly(self, image):
        """An all-NaN or constant frame gives no regions and no RuntimeWarning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            regions = _detect_stars(image, threshold=THRESH, opening=3)

        assert regions == []

    def test_threshold_gates_faint_sources(self):
        """
        ``threshold`` is in units of the background sigma above the median.

        A broad source peaking ~6 sigma above the sky is found at the production
        0.5 sigma threshold and lost at a 10 sigma one.
        """
        rng = np.random.default_rng(SEED)
        noise_stddev = 2.0
        image = rng.normal(100.0, noise_stddev, (200, 200))
        yy, xx = np.mgrid[0:200, 0:200]
        x_peak, y_peak = 120.0, 80.0
        sigma = 8.0 * gaussian_fwhm_to_sigma
        image += (
            6
            * noise_stddev
            * np.exp(-((xx - x_peak) ** 2 + (yy - y_peak) ** 2) / (2 * sigma**2))
        )

        match_radius = 2.0

        def found(threshold):
            regions = _detect_stars(image, threshold=threshold, opening=5)
            return any(
                np.hypot(r.centroid[0] - y_peak, r.centroid[1] - x_peak) < match_radius
                for r in regions
            )

        assert found(0.5)
        assert not found(10)

    def test_opening_gates_small_blobs(self):
        """
        ``opening`` is the kernel size: a 3 px blob survives 3 and not 5.

        A source must hold a solid ``opening x opening`` core above threshold, so
        this -- not the threshold -- is what gates faint-star detection.
        """
        rng = np.random.default_rng(SEED)
        image = rng.normal(100.0, 2.0, (200, 200))
        image[50:53, 50:53] += 200.0

        assert len(_detect_stars(image, threshold=5, opening=3)) == 1
        assert _detect_stars(image, threshold=5, opening=5) == []

    def test_core_sigma_clip_finds_a_faint_source_beside_a_bright_one(
        self, make_test_image
    ):
        """
        The sigma-clipped core, not the raw sky std, sets the threshold.

        A bright source inflates ``np.std`` of the whole frame far above the
        true sky sigma; clipping the estimator to pixels within one (raw) std
        of the median excludes the bright source's wings before measuring
        sigma, so a much fainter second source still clears threshold. Without
        the clip, the raw std (~376 here, against a clipped ~2.6) pushes the
        threshold well above the faint source's peak and it is lost.
        """
        amplitudes = [60000.0, 30.0]
        image = _detectable_image(
            make_test_image, n_sources=len(amplitudes), amplitude=amplitudes
        )

        regions = _detect_stars(image, threshold=5, opening=3)

        assert len(regions) == len(amplitudes)
        x_faint, y_faint = _SOURCE_POSITIONS[1]
        assert any(
            np.hypot(r.centroid[0] - y_faint, r.centroid[1] - x_faint) < 1.0
            for r in regions
        )


class TestCalibrationSequence:
    """Unit tests for detection + FWHM estimation in ``calibration_sequence``."""

    def test_main_path_recovers_fwhm_and_sources(self, make_test_image, tmp_path):
        """A clean multi-source frame yields the sources and the injected FWHM."""
        fwhm = 4.0
        n_sources = 5
        expected_max_adu = 50000
        image = _detectable_image(make_test_image, n_sources=n_sources, fwhm=fwhm)
        path = _write_seestar_fits(tmp_path / "calib.fits", image)

        result = calibration_sequence(
            path,
            threshold=1,
        )

        assert result.calibrated_data is not None
        assert len(result.regions) == n_sources
        assert result.coords.shape == (n_sources, 2)
        # The PSF fit recovers the injected FWHM to within ~5%.
        assert result.fwhm == pytest.approx(fwhm, rel=0.05)
        assert result.metadata["largest_usable_adu_value"] == expected_max_adu
        # Without Bayer balancing, detection ran on the calibrated array itself.
        assert result.detection_image is result.calibrated_data

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

    def test_forwards_opening_to_detection(self, make_test_image, tmp_path, mocker):
        """
        calibration_sequence passes the opening kernel size through to detection.

        The morphological opening (not the threshold) is what gates faint-star
        detection, so the pipeline default must reach ``_detect_stars``. The
        detector is stubbed to return no regions; the resulting TooFewStarsError
        is incidental -- the assertion is the forwarded opening, read back off
        the mock's ``call_args``.
        """
        image = _detectable_image(make_test_image, n_sources=5)
        path = _write_seestar_fits(tmp_path / "open.fits", image)

        stars_detection_mock = mocker.patch(
            "bandaid.photometry._detect_stars", return_value=[]
        )

        # Default: the pipeline's DETECTION_OPENING reaches the detector.
        with pytest.raises(TooFewStarsError):
            calibration_sequence(path, threshold=1)
        assert stars_detection_mock.call_args.kwargs["opening"] == DETECTION_OPENING

        # And an explicit override is honored.
        custom_opening = 7
        with pytest.raises(TooFewStarsError):
            calibration_sequence(path, threshold=1, opening=custom_opening)
        assert stars_detection_mock.call_args.kwargs["opening"] == custom_opening

    def test_unset_defaults_follow_the_resolved_profile_not_seestar50(
        self, tmp_path, mocker, isolate_registry
    ):
        """
        Unset detection/FWHM parameters follow the *resolved* profile.

        Before this, ``threshold``/``opening``/``fwhm_cutout_half``/
        ``fwhm_n_stars`` defaulted to module-level constants derived from a
        bare ``InstrumentProfile()`` (Seestar50's tuning) regardless of which
        profile ``calibration_sequence`` actually resolved, so a direct call
        on a future non-Seestar profile detected and fit the FWHM at
        Seestar50's settings even though ``profile`` auto-detected correctly.
        Now the defaults are pulled from the resolved profile itself.
        """
        custom = InstrumentProfile(
            name="CustomScope",
            header_match=(HeaderMatchRule(keyword="INSTRUME", pattern="Custom Scope"),),
            thresh=3.0,
            detection_opening=9,
            fwhm_cutout_half=11,
            fwhm_n_stars=13,
        )
        with isolate_registry(instruments, "_REGISTERED"):
            register_instrument(custom)

            header = _seestar_header()
            header["INSTRUME"] = "Custom Scope"
            path = tmp_path / "custom.fits"
            fits.PrimaryHDU(np.zeros((200, 200)), header=header).writeto(
                path, output_verify="silentfix"
            )

            detect = mocker.patch(
                "bandaid.photometry._detect_stars", side_effect=five_diagonal_regions
            )
            fwhm_helper = mocker.patch(
                "bandaid.photometry._fwhm_from_coords", return_value=2.5
            )

            calibration_sequence(path)

        assert detect.call_args.kwargs["threshold"] == custom.thresh
        assert detect.call_args.kwargs["opening"] == custom.detection_opening
        assert (
            fwhm_helper.call_args.kwargs["fwhm_cutout_half"] == custom.fwhm_cutout_half
        )
        assert fwhm_helper.call_args.kwargs["n_stars"] == custom.fwhm_n_stars

    def test_detects_on_balanced_copy_when_flagged(
        self, make_test_image, tmp_path, mocker
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

        mocker.patch("bandaid.photometry.bayer_balance_image", side_effect=fake_balance)

        seen = {}
        real_detection = _detect_stars

        def capturing_detection(data, threshold=5, opening=5):
            # Copy defensively: the array is reused (and, further downstream,
            # could be mutated) after this call, so a bare reference would not
            # reliably reflect what detection actually saw.
            seen["data"] = np.array(data, copy=True)
            return real_detection(data, threshold=threshold, opening=opening)

        mocker.patch(
            "bandaid.photometry._detect_stars",
            side_effect=capturing_detection,
        )

        n_sources = 5
        image = _detectable_image(make_test_image, n_sources=n_sources)
        path = _write_seestar_fits(tmp_path / "bayer_detect.fits", image)

        result = calibration_sequence(
            path,
            threshold=1,
            detect_on_bayer_balanced=True,
        )

        # Detection saw the balanced (marked) image...
        np.testing.assert_allclose(seen["data"], image + marker)
        # ...while the returned calibrated_data is the original, unbalanced counts
        # that downstream photometry relies on.
        np.testing.assert_allclose(result.calibrated_data, image)
        # The result carries the balanced array as a distinct object.
        assert result.detection_image is not result.calibrated_data
        np.testing.assert_allclose(result.detection_image, image + marker)
        # Check that the balanced detection still recovers the injected sources.
        assert len(result.regions) == n_sources
        assert result.coords.shape == (n_sources, 2)

    def test_attaches_file_when_bayer_balance_is_degenerate(
        self, make_test_image, tmp_path, mocker
    ):
        """A degenerate-channel error from bayer_balance_image gets the file (#61)."""

        def raising_balance(_arr):
            msg = "zero variance"
            raise DegenerateBayerChannelError(msg)

        mocker.patch(
            "bandaid.photometry.bayer_balance_image", side_effect=raising_balance
        )

        image = _detectable_image(make_test_image)
        path = _write_seestar_fits(tmp_path / "degenerate.fits", image)

        with pytest.raises(DegenerateBayerChannelError) as exc_info:
            calibration_sequence(path, threshold=1, detect_on_bayer_balanced=True)
        assert exc_info.value.file == path

    def test_unmatched_header_raises_frame_metadata_error(self):
        """
        An unresolvable header raises FrameMetadataError, via InstrumentDetectionError.

        ``calibration_sequence`` is one of the per-frame entry points: when
        ``profile`` is None, ``metadata_from_header`` detects
        the instrument and can raise ``InstrumentDetectionError``, itself a
        `FrameMetadataError`. It is labelled with the file here, the same as
        the metadata errors `metadata_from_header` raises directly, so it
        stays inside the ``FrameError`` family a per-frame skip loop already
        handles. A frame with an empty header is handed in directly, so the
        failure happens before any detection would run.
        """
        frame = LoadedFrame(np.zeros((10, 10)), {})

        with pytest.raises(FrameMetadataError) as exc_info:
            calibration_sequence(
                "fake_file.fits", threshold=1, profile=None, frame=frame
            )

        assert exc_info.value.file == "fake_file.fits"
        assert isinstance(exc_info.value, InstrumentDetectionError)

    @pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gz"])
    def test_opens_the_file_exactly_once(
        self, make_test_image, tmp_path, fromfile_spy, compressed
    ):
        """
        calibration_sequence opens the file exactly once, not per field (#44).

        Covers both a plain FITS file and a gzip-compressed one, since
        compressed frames go through an extra decompression step that could
        (re)introduce a second open.
        """
        image = _detectable_image(make_test_image, n_sources=5)
        path = _write_seestar_fits(tmp_path / "open_once.fits", image)
        if compressed:
            gz_path = tmp_path / "open_once.fits.gz"
            with path.open("rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                f_out.write(f_in.read())
            path = str(gz_path)
        spy = fromfile_spy()

        calibration_sequence(path, threshold=1)

        assert spy.call_count == 1


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

    def test_merges_user_specific_metadata(self, make_test_image, tmp_path, mocker):
        """user_specific_metadata overrides values pulled from the header."""
        _stub_wcs_and_centroid(mocker)
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
        self, make_test_image, tmp_path, mocker
    ):
        """Bayer balancing feeds a balanced copy to centroiding, not the original."""
        # centroid_stars's argument is not mutated after the call (prepare_image
        # does nothing further with ``working_image`` once centroiding returns),
        # so reading it back off .call_args is safe -- no defensive copy needed.
        _, centroid_stars_mock = _stub_wcs_and_centroid(mocker)
        image = _detectable_image(make_test_image)
        path = _write_seestar_fits(tmp_path / "bayer.fits", image)
        # Spy (not replace) the real balancer: the identity check below needs the
        # genuine in-place-balanced array, not a stand-in.
        balance_spy = mocker.patch(
            "bandaid.photometry.bayer_balance_image", wraps=bayer_balance_image
        )

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
        centroid_stars_mock.assert_called_once()
        centroid_data = centroid_stars_mock.call_args.args[0]
        assert not np.allclose(centroid_data, img.calibrated_data)

        # PR #119: a single balance call now covers both detection and
        # centroiding -- the array centroid_stars receives is the literal same
        # object calibration_sequence balanced for detection, not a second
        # fresh copy balanced again.
        balance_spy.assert_called_once()
        assert centroid_data is balance_spy.call_args.args[0]

    def test_degenerate_bayer_balance_still_attaches_file_with_one_call(
        self, make_test_image, tmp_path, mocker
    ):
        """
        A degenerate-channel failure still gets ``exc.file`` attached.

        The failure comes from the (now sole) balance call; ``prepare_image``
        makes no balancing attempt of its own (issue #61's contract, now served
        entirely by ``calibration_sequence``'s own try/except since PR #119
        removed the second call site it used to protect).
        """
        _stub_wcs_and_centroid(mocker)
        image = _detectable_image(make_test_image)
        path = _write_seestar_fits(tmp_path / "degenerate2.fits", image)

        def raising_balance(_arr):
            msg = "zero variance"
            raise DegenerateBayerChannelError(msg)

        bayer_balance_mock = mocker.patch(
            "bandaid.photometry.bayer_balance_image", side_effect=raising_balance
        )

        with pytest.raises(DegenerateBayerChannelError) as exc_info:
            prepare_image(
                path,
                _REF_RADECS,
                None,
                detect_on_bayer_balanced=True,
            )
        assert exc_info.value.file == path
        assert bayer_balance_mock.call_count == 1


class TestProcessOneImage:
    """End-to-end (stubbed-externals) coverage for ``process_one_image``."""

    @pytest.fixture
    def l4_frame(self, make_test_image, tmp_path, mocker, bayer_masks_rggb):
        """
        A processable FITS path plus its RGGB+L4 masks, externals stubbed.

        Parameters
        ----------
        make_test_image : callable
            The ``make_test_image`` factory fixture.
        tmp_path : pathlib.Path
            pytest's per-test temporary directory.
        mocker : pytest_mock.MockerFixture
            Used to stub the WCS solve and CNN centroiding.
        bayer_masks_rggb : callable
            The RGGB mask factory fixture.

        Returns
        -------
        tuple
            ``(path, masks)`` ready to hand to ``process_one_image``.
        """
        _stub_wcs_and_centroid(mocker)
        image = _detectable_image(make_test_image)
        path = _write_seestar_fits(tmp_path / "frame.fits", image)
        return path, bayer_masks_rggb(image.shape)

    def test_raises_when_image_rejected(
        self, make_test_image, tmp_path, bayer_masks_rggb
    ):
        """A frame with too few stars raises TooFewStarsError."""
        image = _detectable_image(
            make_test_image,
            n_sources=2,
            image_size=(200, 200),
            include_noise=False,
        )
        path = _write_seestar_fits(tmp_path / "few.fits", image)
        masks = bayer_masks_rggb(image.shape)

        with pytest.raises(TooFewStarsError, match="stars detected"):
            process_one_image(path, {}, _REF_RADECS, None, masks)

    def test_full_path_builds_per_filter_tables_with_l4(self, l4_frame):
        """Every filter gets a table and the L4 channel sums the RGB counts."""
        path, masks = l4_frame

        result = process_one_image(path, {}, _REF_RADECS, None, masks)

        assert set(result) == {"TR", "TG", "TB", "L4"}
        rgb_sum = sum(result[name]["tot_count"] for name in ("TR", "TG", "TB"))
        np.testing.assert_allclose(result["L4"]["tot_count"], rgb_sum)

    def test_tables_carry_the_solved_scale_and_offset(self, l4_frame):
        """
        Every table's meta carries the solved plate scale and solve offset.

        ``_qa_record_ok`` only sees the tables, so ``process_one_image`` stamps
        the solved scale (arcsec/px) and the solved-centre-to-header-centre
        separation (deg) on each one.
        """
        path, masks = l4_frame

        result = process_one_image(path, {}, _REF_RADECS, None, masks)

        for table in result.values():
            # The stubbed TAN WCS is built at the Seestar50 plate scale.
            assert table.meta["wcs_pixscale"] == pytest.approx(
                SEESTAR_PIXSCALE, rel=1e-3
            )
            assert 0 <= table.meta["solve_offset_deg"] < 1

    def test_centroid_prior_summary_is_stamped_on_every_table(self, l4_frame, mocker):
        """The frame's plane summary rides along in each table's meta, L4 included."""
        path, masks = l4_frame
        real_prepare_image = photometry.prepare_image
        summary = {"plane_n_used": 7, "plane_fallback": False}

        def _prepare_with_summary(file, radecs, cnn, **kwargs: object):
            img = real_prepare_image(file, radecs, cnn, **kwargs)
            img.centroid_prior = summary
            return img

        mocker.patch(
            "bandaid.photometry.prepare_image", side_effect=_prepare_with_summary
        )

        result = process_one_image(path, {}, _REF_RADECS, None, masks)

        assert set(result) == {"TR", "TG", "TB", "L4"}
        for table in result.values():
            assert table.meta["centroid_prior"] == summary

    def test_gaia_g_and_cut_reach_prepare_image(self, l4_frame, mocker):
        """``process_one_image`` hands ``input_gaia_g`` and ``g_cut`` on."""
        path, masks = l4_frame
        spy = mocker.patch(
            "bandaid.photometry.prepare_image", side_effect=NoUsableStarsError("x")
        )
        gaia_g = np.array([9.0, 10.0])
        g_cut = 9.5

        with pytest.raises(NoUsableStarsError):
            process_one_image(
                path, {}, _REF_RADECS, None, masks, input_gaia_g=gaia_g, g_cut=g_cut
            )

        assert spy.call_args.kwargs["gaia_g"] is gaia_g
        assert spy.call_args.kwargs["g_cut"] == g_cut

    def test_edge_drop_count_is_stamped_on_every_table(self, l4_frame, mocker):
        """Every table, L4 included, carries the frame's ``n_edge_dropped``."""
        path, masks = l4_frame
        real_prepare = prepare_image
        edge_dropped = 11

        def _with_count(*args: object, **kwargs: object):
            img = real_prepare(*args, **kwargs)
            img.n_edge_dropped = edge_dropped
            return img

        mocker.patch("bandaid.photometry.prepare_image", side_effect=_with_count)

        result = process_one_image(path, {}, _REF_RADECS, None, masks)

        assert set(result) >= {"TR", "TG", "TB", "L4"}
        for table in result.values():
            assert table.meta["n_edge_dropped"] == edge_dropped

    def test_l4_channel_skips_the_full_frame_photometry_pass(self, l4_frame, mocker):
        """
        L4's own full-frame ``measure_photometry`` pass is skipped (PR #120).

        Every phot-derived L4 column is the TR/TG/TB recombination
        ``calculate_l4_quantities`` computes (issue #21), so a full-frame pass
        would be pure waste. Only the 3 RGB channels (TR/TG/TB) should reach
        ``measure_photometry``.
        """
        path, masks = l4_frame
        mp_spy = mocker.patch(
            "bandaid.photometry.measure_photometry", wraps=measure_photometry
        )

        process_one_image(path, {}, _REF_RADECS, None, masks)

        n_rgb_channels = 3  # TR, TG, TB -- L4 must not reach measure_photometry.
        assert mp_spy.call_count == n_rgb_channels

    def test_l4_copied_columns_match_a_full_frame_build(self, l4_frame, mocker):
        """
        The columns L4 copies from TR equal a genuine full-frame build's.

        Runs ``build_photometry_table(img, None)`` on the very ``ImageData``
        ``process_one_image`` used and checks the mask-independent columns and
        meta the L4 table took from TR are bit-identical to it. (The L4 column
        set itself is pinned in test_build_table.py.)
        """
        path, masks = l4_frame
        # Spy (not wraps-patch) so spy_return hands back the very ImageData
        # process_one_image built.
        prepare_spy = mocker.spy(photometry, "prepare_image")

        result = process_one_image(path, {}, _REF_RADECS, None, masks)

        reference = build_photometry_table(prepare_spy.spy_return, None)

        l4 = result["L4"]
        for col in _MASK_INDEPENDENT_COLUMNS:
            np.testing.assert_array_equal(
                np.asarray(l4[col]), np.asarray(reference[col])
            )
        # Only the keys build_photometry_table itself stamps; filter and
        # full_image_meta are added by process_one_image and differ by design.
        assert {k: l4.meta[k] for k in reference.meta} == reference.meta

    @pytest.mark.parametrize(
        ("mutate", "match"),
        [
            (lambda m: m.pop("TB"), r"\['TB'\]"),
            (lambda m: m.pop("TR"), r"\['TR'\]"),
        ],
        ids=["missing-TB", "missing-TR"],
    )
    def test_l4_malformed_mask_dict_raises(self, l4_frame, mocker, mutate, match):
        """
        With L4 requested, a mask dict missing an RGB channel raises ValueError.

        TR doubles as the source of L4's copied columns, so the missing-channel
        check must run before any channel lookup or a caller missing TR gets a
        bare ``KeyError`` instead of the documented ``ValueError``.

        The check runs before the RGB loop: the mask dict is shared across the
        batch and the ``ValueError`` is not a ``FrameError``, so with
        ``fail_fast=False`` a malformed dict would otherwise photometer every
        frame in full before failing it.
        """
        path, masks = l4_frame
        mutate(masks)
        build_spy = mocker.spy(photometry, "build_photometry_table")

        with pytest.raises(ValueError, match=match):
            process_one_image(path, {}, _REF_RADECS, None, masks)

        assert build_spy.call_count == 0

    def test_build_l4_false_returns_only_the_given_masks(self, l4_frame):
        """With ``build_l4=False`` no L4 is built and TR/TG/TB are not required."""
        path, masks = l4_frame
        masks.pop("TB")

        result = process_one_image(path, {}, _REF_RADECS, None, masks, build_l4=False)

        assert set(result) == {"TR", "TG"}

    def test_opens_the_file_exactly_once(self, l4_frame, fromfile_spy):
        """process_one_image opens the file exactly once end-to-end (#44)."""
        path, masks = l4_frame
        spy = fromfile_spy()

        process_one_image(path, {}, _REF_RADECS, None, masks)

        assert spy.call_count == 1

    def test_resolves_instrument_once_and_reuses_it(self, mocker):
        """
        A default config's instrument is resolved once and passed downstream.

        ``process_one_image`` used to reuse its own unresolved ``config``
        after calling ``prepare_image`` (which resolves internally), so a
        default (``instrument=None``) config reached ``build_photometry_table``
        still unresolved. Resolve once, from the loaded frame's header, before
        calling ``prepare_image``, and pass the resolved config
        down. ``prepare_image`` and ``build_photometry_table`` are stubbed so
        this only exercises the resolution/reuse, and ``_load_frame`` is
        spied on to confirm the file is still opened exactly once.
        """
        header = {"INSTRUME": "Seestar S50"}
        load_frame = mocker.patch(
            "bandaid.photometry._load_frame",
            side_effect=lambda _file: LoadedFrame(np.zeros((10, 10)), header),
        )
        centroid_coords = np.array([[10.0, 10.0], [20.0, 20.0]])
        img = ImageData(
            calibrated_data=np.zeros((50, 50)),
            coords=centroid_coords,
            fwhm=3.0,
            centroid_coords=centroid_coords,
            aligned_coords=centroid_coords,
            wcs=None,
            header=header,
            metadata={"egain": 1.0},
        )
        prepare_image_mock = mocker.patch(
            "bandaid.photometry.prepare_image", return_value=img
        )
        table = Table({"tot_count": [1.0]})
        build_table_mock = mocker.patch(
            "bandaid.photometry.build_photometry_table", return_value=table
        )

        process_one_image(
            "unused.fits",
            {},
            _REF_RADECS,
            None,
            {"TR": None},
            config=PhotometryConfig(),
            build_l4=False,
        )

        resolved_config = build_table_mock.call_args.kwargs["config"]
        assert resolved_config.instrument.name == "Seestar50"
        assert prepare_image_mock.call_args.kwargs["config"] is resolved_config
        load_frame.assert_called_once()


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
        result = calibration_sequence(
            str(_REAL_FRAME),
            threshold=THRESH,
        )

        assert result.calibrated_data is not None
        assert len(result.regions) >= MIN_DETECTED_STARS
        assert result.coords.shape == (len(result.regions), 2)
        assert np.isfinite(result.fwhm)
        assert result.fwhm > 0
        assert result.metadata["largest_usable_adu_value"] == expected_max_adu
        assert result.metadata["width"] == result.calibrated_data.shape[1]
        assert result.metadata["height"] == result.calibrated_data.shape[0]

    def test_detect_stars_matches_eloy_on_real_frame(self):
        """
        `_detect_stars` reproduces eloy's detection on genuine pixels, exactly.

        Balance the frame the way ``calibration_sequence`` does, run both
        detectors at the production threshold/opening, and require the same
        regions in the same order. This is the in-repo real-data identity guard
        for the bandaid-side detector.
        """
        data = 1.0 * fits.getdata(str(_REAL_FRAME))
        bayer_balance_image(data)

        expected = detection.stars_detection(
            data, threshold=THRESH, opening=DETECTION_OPENING
        )
        regions = _detect_stars(data, threshold=THRESH, opening=DETECTION_OPENING)

        assert len(regions) == len(expected)
        np.testing.assert_array_equal(
            [r.centroid for r in regions], [r.centroid for r in expected]
        )

    def test_process_one_image_builds_per_filter_tables(self, mocker):
        """Every Bayer filter gets a non-empty table and L4 sums the RGB counts."""
        header = fits.getheader(str(_REAL_FRAME))
        data = fits.getdata(str(_REAL_FRAME))
        metadata = metadata_from_header(header)

        # Center the stubbed WCS on the frame's converted header pointing (the
        # header RA/DEC is equinox-of-date), so it passes the pointing check and
        # the cosmetic ra/dec columns are plausible in a failure dump.
        _stub_wcs_and_centroid(
            mocker,
            wcs_image_size=data.shape,
            wcs_crval=estimate_center_from_header(
                metadata, InstrumentProfile(header_frame="fk5", header_equinox="date")
            ),
        )

        masks = generate_bayer_masks(
            data.shape,
            {
                "bayerpat": metadata["bayerpat"],
                "roworder": metadata["roworder"],
                "ybayroff": metadata["ybayroff"],
            },
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
        result = calibration_sequence(
            str(_REAL_FRAME),
            threshold=THRESH,
        )
        calibrated, coords = result.calibrated_data, result.coords
        max_adu = result.metadata["largest_usable_adu_value"]
        n_cap = InstrumentProfile().fwhm_n_stars

        # The cap selects at most n_cap unsaturated detections (fewer than the
        # full detection list) to build the PSF the FWHM is fit from.
        kept = _brightest_unsaturated(calibrated, coords, max_adu, n_cap)
        assert 0 < len(kept) <= n_cap
        assert len(kept) <= len(coords)

        # The fit calibration_sequence already ran (default cap) lands near the
        # true PSF, not the inflated ~8 px an uncapped CNN fit produced.
        fwhm_ceiling = 6.0  # true PSF ~2.8 px; well clear of the ~8 px inflation
        assert 0 < result.fwhm < fwhm_ceiling

    @pytest.mark.remote_data
    def test_real_ballet_cnn_fwhm_smoke(self):
        """
        Drive the *live* Ballet CNN on a real frame end-to-end.

        Every other test stubs the CNN, so this is the only coverage of the real
        centroider path the FWHM cap exists to protect: weights download ->
        numpy CNN inference -> ``ballet_centroid`` -> ePSF registration ->
        brightest-N cap -> a small FWHM. It runs the real ``NumpyBallet`` and
        needs only network access (no jax): ``NumpyBallet()`` downloads
        ``centroid_15x15.npz`` from the public ``lgrcia/ballet`` HuggingFace
        repo (no auth) on first run.
        """
        result = calibration_sequence(
            str(_REAL_FRAME),
            threshold=THRESH,
        )
        calibrated, coords = result.calibrated_data, result.coords
        max_adu = result.metadata["largest_usable_adu_value"]
        cnn = NumpyBallet()

        n_cap = InstrumentProfile().fwhm_n_stars
        fwhm = _fwhm_from_coords(
            calibrated, coords, max_adu=max_adu, cnn=cnn, n_stars=n_cap
        )

        assert fwhm is not None
        fwhm_ceiling = 6.0  # true PSF ~2.8 px; well clear of the ~8 px inflation
        assert 0 < fwhm < fwhm_ceiling
