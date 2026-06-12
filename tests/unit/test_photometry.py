"""
Unit tests for the photometry pipeline in :mod:`bandaid.photometry`.

Covers aperture photometry on synthetic single-source images
(``measure_photometry``), the bright-neighbor minimum-separation model
(``min_separation_fwhm``), and the detect/align/centroid path
(``prepare_image``), using the synthetic-image fixtures from ``conftest.py``.
"""

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS

from bandaid import measure_photometry
from bandaid.exceptions import (
    FrameMetadataError,
    NoUsableStarsError,
    TooFewStarsError,
    WCSSolveError,
)
from bandaid.image2sl_qt import generate_bayer_masks
from bandaid.photometry import (
    ANNULUS,
    MIN_DETECTED_STARS,
    N_STARS_ALIGN,
    RELATIVE_RADII,
    THRESH,
    ImageData,
    align,
    build_photometry_table,
    calculate_l4_quantities,
    calibration_sequence,
    centroid_drift_flag,
    centroid_stars,
    eloy_to_starlist,
    flag_partial_obstruction,
    metadata_from_header,
    min_separation_fwhm,
    neighbor_contamination_flag,
    neighbor_contamination_flag_sky,
    prepare_image,
    process_one_image,
)

# Make the tests reproducible by using a fixed random seed for noise generation in
# the test images.
SEED = 843032


def _single_source_photometry_inputs(make_test_image, fwhm=2.3, annulus=ANNULUS):
    """
    Build a noiseless single-source image and the ``measure_photometry`` inputs.

    Parameters
    ----------
    make_test_image : callable
        The ``make_test_image`` fixture factory.
    fwhm : float, optional
        FWHM (pixels) of the Gaussian source.
    annulus : tuple, optional
        Annulus (in FWHM) the image must be large enough to contain.

    Returns
    -------
    tuple
        ``(image, coords, fwhm, mask)`` ready to pass to ``measure_photometry``.
    """
    image_side = max(max(annulus) * fwhm * 2, 100)
    image_size = (image_side, image_side)
    source_x = image_size[1] / 2
    source_y = image_size[0] / 2
    source_properties = Table(
        {
            "amplitude": [100],
            "x_mean": [source_x],
            "y_mean": [source_y],
            "x_stddev": [fwhm * gaussian_fwhm_to_sigma],
            "y_stddev": [fwhm * gaussian_fwhm_to_sigma],
        },
    )
    image = make_test_image(
        image_size=image_size,
        source_properties=source_properties,
        include_noise=False,
        noise_mean=0,
        noise_stddev=0,
        seed=SEED,
    )
    coords = np.array([[source_x, source_y]])
    mask = np.zeros(image_size, dtype=bool)
    return image, coords, fwhm, mask


@pytest.mark.parametrize(
    ("include_noise", "noise_stddev"), [(True, 10), (True, 5), (False, 0)]
)
def test_measure_photometry_single_source(make_test_image, include_noise, noise_stddev):
    """Ensure measure_photometry works as expected when we pass in a single source."""
    fwhm = 2.3
    # Make the image at least 100 pixels and big enough for twice
    # the outer annulus radius
    image_side = max(max(ANNULUS) * fwhm * 2, 100)
    image_size = (image_side, image_side)
    amplitude = 100
    source_x = image_size[1] / 2
    source_y = image_size[0] / 2
    noise_mean = 10

    source_properties = Table(
        {
            "amplitude": [amplitude],
            "x_mean": [source_x],
            "y_mean": [source_y],
            "x_stddev": [fwhm * gaussian_fwhm_to_sigma],
            "y_stddev": [fwhm * gaussian_fwhm_to_sigma],
        },
    )
    source_image = make_test_image(
        image_size=image_size,
        source_properties=source_properties,
        include_noise=include_noise,
        noise_mean=noise_mean,
        noise_stddev=noise_stddev,
        seed=SEED,
    )

    egain = 0.3
    mask = np.zeros(image_size, dtype=bool)

    photom = measure_photometry(
        source_image,
        np.array([[source_x, source_y]]),
        np.array([[source_x, source_y]]),
        fwhm,
        egain,
        mask,
    )
    aperture_radius = fwhm * RELATIVE_RADII[0]
    expected_counts = (
        source_properties["amplitude"][0]
        * 2
        * np.pi
        * source_properties["x_stddev"][0] ** 2
        * (
            1
            - np.exp(
                -(aperture_radius**2) / (2 * source_properties["x_stddev"][0] ** 2)
            )
        )
    )
    # Yes, this could be simplified but this is more explicit
    poisson_error_source = np.sqrt(expected_counts * egain) / egain

    # Do a sigma-clipped standard deviation of the image to get the noise error
    _, _, clip_std = sigma_clipped_stats(source_image)
    noise_error = clip_std * np.sqrt(np.pi * aperture_radius**2)
    # There is no Poisson distributed sky background in this case, just a constant
    # offset from zero, so the sky background error is zero.
    sky_background_error = 0
    expected_error = np.sqrt(
        poisson_error_source**2 + noise_error**2 + sky_background_error**2
    )

    # The uncertainties are fairly large because the aperture is the size of the
    # star FWHM, so pixelation matters. Allow 2 sigma: with a single fixed-seed noise
    # realization the measured count can sit ~1.5 sigma from the analytic expectation.
    assert photom["tot_count"][0] == pytest.approx(
        expected_counts, abs=2 * expected_error
    )

    # The background std estimate from the annulus has statistical scatter
    # proportional to noise_stddev, so absolute tolerances scale with
    # noise_error (= noise_stddev * sqrt(aperture_area)).
    # The factor 0.06 accounts for the finite annulus size and sigma-clipping.
    count_err_tol = 0.06 * noise_error
    assert photom["count_err"][0] == pytest.approx(
        expected_error,
        rel=0.04,
        abs=count_err_tol,
    )
    snr = photom["tot_count"][0] / photom["count_err"][0]
    expected_snr = expected_counts / expected_error
    # SNR tolerance: noise in the aperture affects both measured counts and
    # the error estimate. The fractional noise contribution to the error
    # (noise_error / expected_error) sets the scale of SNR scatter; the
    # factor of 2 accounts for the correlated effect on both numerator
    # and denominator.
    snr_tol = 2 * noise_error / expected_error
    assert snr == pytest.approx(
        expected_snr,
        rel=0.06,
        abs=snr_tol,
    )


def test_measure_photometry_default_matches_explicit_constants(make_test_image):
    """Omitting relative_radii/annulus matches passing the module constants."""
    image, coords, fwhm, mask = _single_source_photometry_inputs(make_test_image)
    egain = 0.3

    default = measure_photometry(image, coords, coords, fwhm, egain, mask)
    explicit = measure_photometry(
        image,
        coords,
        coords,
        fwhm,
        egain,
        mask,
        relative_radii=RELATIVE_RADII,
        annulus=ANNULUS,
    )

    np.testing.assert_array_equal(default["tot_count"], explicit["tot_count"])
    np.testing.assert_array_equal(default["count_err"], explicit["count_err"])
    assert default["aperture_radii"] == explicit["aperture_radii"]
    assert default["annulus_radii"] == explicit["annulus_radii"]


def test_measure_photometry_custom_relative_radii(make_test_image):
    """A custom relative_radii drives the output shapes and aperture radius."""
    image, coords, fwhm, mask = _single_source_photometry_inputs(make_test_image)
    egain = 0.3
    relative_radii = [1.0, 2.0]

    photom = measure_photometry(
        image,
        coords,
        coords,
        fwhm,
        egain,
        mask,
        relative_radii=relative_radii,
    )

    # One column per requested radius.
    assert photom["fluxes"].shape == (1, len(relative_radii))
    assert photom["total_bkg"].shape == (1, len(relative_radii))
    # A larger aperture captures more flux for a Gaussian source.
    assert photom["fluxes"][0, 1] > photom["fluxes"][0, 0]
    # The reported aperture radius is the first requested radius times the FWHM.
    assert photom["aperture_radii"] == pytest.approx(relative_radii[0] * fwhm)


def test_measure_photometry_custom_annulus(make_test_image):
    """A custom annulus is honored in the returned annulus_radii."""
    annulus = (6, 10)
    image, coords, fwhm, mask = _single_source_photometry_inputs(
        make_test_image,
        annulus=annulus,
    )
    egain = 0.3

    photom = measure_photometry(
        image,
        coords,
        coords,
        fwhm,
        egain,
        mask,
        annulus=annulus,
    )

    max_aper = np.max(RELATIVE_RADII) * fwhm
    expected = (max(max_aper, annulus[0] * fwhm), annulus[1] * fwhm)
    assert photom["annulus_radii"] == pytest.approx(expected)


def test_measure_photometry_accepts_list_relative_radii(make_test_image):
    """A plain list (not just ndarray) works for relative_radii."""
    image, coords, fwhm, mask = _single_source_photometry_inputs(make_test_image)
    egain = 0.3

    photom = measure_photometry(
        image,
        coords,
        coords,
        fwhm,
        egain,
        mask,
        relative_radii=[1.0],
    )

    assert photom["aperture_radii"] == pytest.approx(1.0 * fwhm)


def test_measure_photometry_accepts_scalar_relative_radii(make_test_image):
    """A bare scalar relative_radii is coerced to 1D and works as one aperture."""
    image, coords, fwhm, mask = _single_source_photometry_inputs(make_test_image)
    egain = 0.3

    photom = measure_photometry(
        image,
        coords,
        coords,
        fwhm,
        egain,
        mask,
        relative_radii=1.0,
    )

    # A scalar behaves like a single-element radius list.
    assert photom["fluxes"].shape == (len(coords), 1)
    assert photom["aperture_radii"] == pytest.approx(1.0 * fwhm)


@pytest.mark.parametrize(
    "bad_annulus",
    [
        5.0,  # not a sequence
        (5,),  # too few elements
        (5, 8, 10),  # too many elements
        (8, 5),  # outer smaller than inner
        (5, 5),  # outer equal to inner
    ],
)
def test_measure_photometry_rejects_invalid_annulus(make_test_image, bad_annulus):
    """A malformed annulus raises a clear ValueError up front."""
    image, coords, fwhm, mask = _single_source_photometry_inputs(make_test_image)
    egain = 0.3

    with pytest.raises(ValueError, match="annulus"):
        measure_photometry(
            image,
            coords,
            coords,
            fwhm,
            egain,
            mask,
            annulus=bad_annulus,
        )


def test_measure_photometry_rejects_aperture_larger_than_annulus(make_test_image):
    """A degenerate annulus (outer <= forced inner) raises a clear ValueError."""
    image, coords, fwhm, mask = _single_source_photometry_inputs(make_test_image)
    egain = 0.3

    # The largest aperture (20 FWHM) exceeds the outer annulus radius (8 FWHM),
    # so after forcing the inner radius up to the largest aperture there is no
    # usable background annulus left.
    with pytest.raises(ValueError, match="annulus"):
        measure_photometry(
            image,
            coords,
            coords,
            fwhm,
            egain,
            mask,
            relative_radii=[20.0],
            annulus=(5, 8),
        )


def test_measure_photometry_accepts_non_subscriptable_annulus(make_test_image):
    """A 2-element iterable that unpacks but is not subscriptable still works."""
    image, coords, fwhm, mask = _single_source_photometry_inputs(make_test_image)
    egain = 0.3

    # An iterator unpacks into (inner, outer) but cannot be indexed; the function
    # must use the unpacked values rather than re-indexing the original.
    photom = measure_photometry(
        image,
        coords,
        coords,
        fwhm,
        egain,
        mask,
        annulus=iter((6, 10)),
    )

    expected = (max(np.max(RELATIVE_RADII) * fwhm, 6 * fwhm), 10 * fwhm)
    assert photom["annulus_radii"] == pytest.approx(expected)


def test_min_separation_fwhm():
    """Check a few extreme cases for a reasonable minimum separation between sources."""
    tenk_flux_ratio = 10
    # first check for a target with a much, much dimmer companion.
    # In that case the minimum separation should be roughly zero.
    assert min_separation_fwhm(-tenk_flux_ratio, tolerance=0.01) == pytest.approx(0)

    # Now assume the neighbor is much brighter than the target. Then the minimum
    # separation should be large.
    assert min_separation_fwhm(tenk_flux_ratio, tolerance=0.01) == pytest.approx(
        11.036,
        rel=0.01,
    )

    # Now a case where the neighbor is the same brightness as the target.
    assert min_separation_fwhm(0, tolerance=0.01) == pytest.approx(2.176, rel=0.01)


class TestCentroidDriftFlag:
    """Unit tests for the centroid-drift consistency check ``centroid_drift_flag``."""

    def test_zero_drift_not_flagged(self):
        """A centroid sitting exactly on its aligned position is not flagged."""
        coords = np.array([[100.0, 100.0], [200.0, 250.0], [10.0, 400.0]])
        flag = centroid_drift_flag(coords, coords, fwhm=2.3)
        assert not flag.any()
        assert flag.dtype == bool

    def test_drift_just_over_and_under_fwhm_tolerance(self):
        """Drift just past ``tolerance * fwhm`` flags; just under does not."""
        fwhm = 2.0
        # max_allowed = min(1.0 * 2.0, cap=4.0) = 2.0 pixels
        aligned = np.array([[100.0, 100.0], [100.0, 100.0]])
        centroid = np.array(
            [
                [100.0 + 2.0 + 1e-6, 100.0],  # just over -> flagged
                [100.0 + 2.0 - 1e-6, 100.0],  # just under -> not flagged
            ],
        )
        flag = centroid_drift_flag(centroid, aligned, fwhm=fwhm)
        assert flag[0]
        assert not flag[1]

    def test_pixel_cap_binds_for_large_fwhm(self):
        """When ``tolerance * fwhm`` exceeds the cap, the cap governs the flag."""
        # tolerance * fwhm = 1.0 * 100 = 100 px, but cap is 4 px, so anything
        # beyond 4 px should flag even though it is well under 100 px.
        fwhm = 100.0
        aligned = np.array([[0.0, 0.0], [0.0, 0.0]])
        centroid = np.array(
            [
                [4.0 + 1e-6, 0.0],  # just past the cap -> flagged
                [4.0 - 1e-6, 0.0],  # just under the cap -> not flagged
            ],
        )
        flag = centroid_drift_flag(centroid, aligned, fwhm=fwhm)
        assert flag[0]
        assert not flag[1]

    def test_custom_tolerance_and_cap_respected(self):
        """Explicit ``tolerance`` and ``cap`` override the module defaults."""
        aligned = np.array([[0.0, 0.0]])
        centroid = np.array([[3.0, 0.0]])  # 3 px drift
        # Default (tol=1.0, fwhm=2.0 -> 2.0 px allowed): flagged.
        assert centroid_drift_flag(centroid, aligned, fwhm=2.0)[0]
        # Loosened tolerance (4.0 * 2.0 = 8 px allowed, cap 10): not flagged.
        assert not centroid_drift_flag(
            centroid,
            aligned,
            fwhm=2.0,
            tolerance=4.0,
            cap=10.0,
        )[0]
        # Tight cap (1 px) overrides a generous tolerance: flagged.
        assert centroid_drift_flag(
            centroid,
            aligned,
            fwhm=2.0,
            tolerance=4.0,
            cap=1.0,
        )[0]

    def test_nan_centroid_flagged_as_drifted(self):
        """A non-finite centroid is treated as drifted (flagged True)."""
        aligned = np.array([[100.0, 100.0], [200.0, 200.0]])
        centroid = np.array([[np.nan, 100.0], [200.0, 200.0]])
        flag = centroid_drift_flag(centroid, aligned, fwhm=2.3)
        assert flag[0]
        assert not flag[1]


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


def _make_tan_wcs(image_size=(500, 500), crval=(10.0, 20.0)):
    """Build a simple TAN WCS centred at ``crval`` for the given image size."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [image_size[1] / 2, image_size[0] / 2]
    wcs.wcs.crval = list(crval)
    wcs.wcs.cdelt = [-2.4 / 3600, 2.4 / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def _fake_phot_factory(n_stars):
    """
    Return a stub for ``measure_photometry`` sized for ``n_stars`` sources.

    The stub bypasses the real aperture photometry (which needs realistic image
    data) so the tests can exercise only the RA/Dec column logic in
    ``build_photometry_table``. Shapes mirror the real return value: scalar-per-
    star arrays plus ``(n_stars, len(RELATIVE_RADII))`` arrays for the
    aperture-resolved quantities.
    """
    n_radii = len(RELATIVE_RADII)

    def _fake_measure_photometry(*_args: object, **_kwargs: object) -> dict:
        return {
            "tot_count": np.arange(n_stars, dtype=float),
            "count_err": np.ones(n_stars),
            "bkgd_count": np.ones(n_stars),
            "bkgd_std": np.ones(n_stars),
            "peak_count": np.ones(n_stars),
            "snr": np.ones(n_stars),
            "total_bkg": np.ones((n_stars, n_radii)),
            "fluxes": np.ones((n_stars, n_radii)),
            "aperture_radii": 1.0,
            "annulus_radii": (5.0, 8.0),
            "aperture_area": np.ones(n_stars),
        }

    return _fake_measure_photometry


def _make_image_data(
    wcs, centroid_coords, input_photometry_coords, aligned_coords=None
):
    """
    Build an ImageData with just enough fields for build_photometry_table.

    ``aligned_coords`` defaults to ``centroid_coords`` (zero drift) but can be
    supplied to exercise the centroid-drift flag.
    """
    if aligned_coords is None:
        aligned_coords = centroid_coords
    header = fits.Header()
    header["DATE-OBS"] = "2020-01-01T00:00:00"
    header["AIRMASS"] = 1.2
    return ImageData(
        calibrated_data=np.zeros((50, 50)),
        coords=centroid_coords,
        fwhm=2.3,
        centroid_coords=centroid_coords,
        aligned_coords=aligned_coords,
        wcs=wcs,
        header=header,
        input_photometry_coords=input_photometry_coords,
        metadata={"egain": 1.0},
    )


class TestBuildPhotometryTable:
    def test_uses_photometry_coords_when_provided(self, monkeypatch):
        """RA/Dec come straight from photometry_coords, not the WCS round-trip."""
        n_stars = 3
        monkeypatch.setattr(
            "bandaid.photometry.measure_photometry",
            _fake_phot_factory(n_stars),
        )
        wcs = _make_tan_wcs()
        # Centroids near the image centre; their WCS sky positions are close to
        # the WCS crval (10, 20) -- deliberately NOT the photometry_coords below.
        centroid_coords = np.array([[245.0, 250.0], [255.0, 260.0], [250.0, 240.0]])
        photometry_coords = SkyCoord(
            ra=[100.0, 150.0, 200.0] * u.deg,
            dec=[-30.0, -10.0, 5.0] * u.deg,
        )
        img = _make_image_data(wcs, centroid_coords, photometry_coords)

        table = build_photometry_table(img, mask=None)

        # ra/dec equal the supplied sky coordinates exactly...
        np.testing.assert_allclose(table["ra"], photometry_coords.ra.degree)
        np.testing.assert_allclose(table["dec"], photometry_coords.dec.degree)

        # ...and are NOT what the WCS round-trip would have produced.
        wcs_radec = wcs.pixel_to_world(centroid_coords[..., 0], centroid_coords[..., 1])
        assert not np.allclose(table["ra"], wcs_radec.ra.degree)
        assert not np.allclose(table["dec"], wcs_radec.dec.degree)

        # Rows line up with the per-star x/y (centroid) columns.
        assert len(table) == n_stars
        assert len(table["ra"]) == len(table["x"])

    def test_falls_back_to_wcs_when_no_photometry_coords(self, monkeypatch):
        """With no photometry_coords, RA/Dec are derived from the image WCS."""
        n_stars = 3
        monkeypatch.setattr(
            "bandaid.photometry.measure_photometry",
            _fake_phot_factory(n_stars),
        )
        wcs = _make_tan_wcs()
        centroid_coords = np.array([[245.0, 250.0], [255.0, 260.0], [250.0, 240.0]])
        img = _make_image_data(wcs, centroid_coords, input_photometry_coords=None)

        table = build_photometry_table(img, mask=None)

        wcs_radec = wcs.pixel_to_world(centroid_coords[..., 0], centroid_coords[..., 1])
        np.testing.assert_allclose(table["ra"], wcs_radec.ra.degree)
        np.testing.assert_allclose(table["dec"], wcs_radec.dec.degree)

    def test_overrides_reach_photometry_and_show_in_output(self, make_test_image):
        """Custom relative_radii/annulus flow through to the real photometry output."""
        # Drive build_photometry_table end-to-end on a real single-source image
        # (no monkeypatching): the overrides only reach the output table meta if
        # build_photometry_table actually forwarded them to measure_photometry.
        relative_radii = [1.0, 2.0]
        annulus = (6, 10)
        image, coords, fwhm, _ = _single_source_photometry_inputs(
            make_test_image,
            annulus=annulus,
        )
        img = _make_image_data(_make_tan_wcs(image.shape), coords, None)
        img.calibrated_data = image

        table = build_photometry_table(
            img,
            mask=None,
            relative_radii=relative_radii,
            annulus=annulus,
        )

        # One flux column per requested radius, and the meta echoes the overrides.
        assert table["fluxes"].shape == (len(coords), len(relative_radii))
        assert table.meta["aperture_radii"] == pytest.approx(relative_radii[0] * fwhm)
        max_aper = max(relative_radii) * fwhm
        expected_annulus = (max(max_aper, annulus[0] * fwhm), annulus[1] * fwhm)
        assert table.meta["annulus_radii"] == pytest.approx(expected_annulus)

    def test_defaults_show_module_constants_in_output(self, make_test_image):
        """With no overrides, the output reflects the module-level constants."""
        image, coords, fwhm, _ = _single_source_photometry_inputs(make_test_image)
        img = _make_image_data(_make_tan_wcs(image.shape), coords, None)
        img.calibrated_data = image

        table = build_photometry_table(img, mask=None)

        assert table.meta["aperture_radii"] == pytest.approx(RELATIVE_RADII[0] * fwhm)
        max_aper = np.max(RELATIVE_RADII) * fwhm
        expected_annulus = (max(max_aper, ANNULUS[0] * fwhm), ANNULUS[1] * fwhm)
        assert table.meta["annulus_radii"] == pytest.approx(expected_annulus)

    def test_centroid_drift_column(self, monkeypatch):
        """Output has a bool ``centroid_drift`` column reflecting per-star drift."""
        n_stars = 3
        monkeypatch.setattr(
            "bandaid.photometry.measure_photometry",
            _fake_phot_factory(n_stars),
        )
        wcs = _make_tan_wcs()
        # fwhm in _make_image_data is 2.3, so max_allowed = min(1.0 * 2.3, 4.0)
        # = 2.3 px. Build aligned vs centroid pairs straddling that threshold.
        aligned_coords = np.array([[250.0, 250.0], [250.0, 250.0], [250.0, 250.0]])
        centroid_coords = np.array(
            [
                [250.0, 250.0],  # zero drift -> not flagged
                [250.0 + 1.0, 250.0],  # 1 px drift -> not flagged
                [250.0 + 5.0, 250.0],  # 5 px drift (> 2.3 and > cap) -> flagged
            ],
        )
        img = _make_image_data(
            wcs,
            centroid_coords,
            input_photometry_coords=None,
            aligned_coords=aligned_coords,
        )

        table = build_photometry_table(img, mask=None)

        assert "centroid_drift" in table.colnames
        assert table["centroid_drift"].dtype == bool
        assert len(table["centroid_drift"]) == n_stars
        np.testing.assert_array_equal(
            table["centroid_drift"],
            [False, False, True],
        )


class TestNeighborContaminationFlag:
    """Unit tests for the bright-neighbor flag ``neighbor_contamination_flag``."""

    @pytest.mark.parametrize("n", [0, 1])
    def test_fewer_than_two_stars_never_flagged(self, n):
        """With <2 stars no pair can exist, so the early return is all-False."""
        coords = np.zeros((n, 2))
        mags = np.zeros(n)
        flag = neighbor_contamination_flag(coords, mags, fwhm=2.0)
        assert flag.shape == (n,)
        assert flag.dtype == bool
        assert not flag.any()

    def test_equal_brightness_pair_flagged_inside_threshold(self):
        """Equal-mag neighbors flag both stars inside ~2.18 FWHM, neither outside."""
        fwhm = 2.0
        # min_separation_fwhm(0) ~ 2.176 FWHM -> ~4.35 px at fwhm=2.
        threshold_px = min_separation_fwhm(0.0) * fwhm

        close = np.array([[0.0, 0.0], [threshold_px - 0.5, 0.0]])
        far = np.array([[0.0, 0.0], [threshold_px + 0.5, 0.0]])
        mags = np.array([12.0, 12.0])

        np.testing.assert_array_equal(
            neighbor_contamination_flag(close, mags, fwhm=fwhm),
            [True, True],
        )
        assert not neighbor_contamination_flag(far, mags, fwhm=fwhm).any()

    def test_flag_is_asymmetric_for_unequal_brightness(self):
        """A faint star is flagged by a bright neighbor that it does not flag back."""
        fwhm = 2.0
        # delta_mag=6 needs ~5.9 FWHM (~11.8 px); delta_mag=-6 needs 0. At 6 px the
        # faint star (bright neighbor) is flagged; the bright star is not.
        mags = np.array([8.0, 14.0])  # star 0 bright, star 1 faint
        coords = np.array([[0.0, 0.0], [6.0, 0.0]])

        flag = neighbor_contamination_flag(coords, mags, fwhm=fwhm)
        assert not flag[0]  # bright star: faint neighbor spills negligibly
        assert flag[1]  # faint star: bright neighbor contaminates it

    def test_non_finite_magnitude_contributes_no_contamination(self):
        """A NaN-magnitude star neither flags nor is flagged."""
        fwhm = 2.0
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])  # essentially on top of one another
        mags = np.array([10.0, np.nan])
        flag = neighbor_contamination_flag(coords, mags, fwhm=fwhm)
        assert not flag.any()


class TestNeighborContaminationFlagSky:
    """Unit tests for the sky-space flag ``neighbor_contamination_flag_sky``."""

    @pytest.mark.parametrize("n", [0, 1])
    def test_fewer_than_two_stars_never_flagged(self, n):
        """With <2 stars no pair can exist, so the early return is all-False."""
        radecs = np.zeros((n, 2))
        mags = np.zeros(n)
        flag = neighbor_contamination_flag_sky(radecs, mags, fwhm_arcsec=2.0)
        assert flag.shape == (n,)
        assert flag.dtype == bool
        assert not flag.any()

    def test_non_finite_magnitude_contributes_no_contamination(self):
        """A NaN-magnitude star neither flags nor is flagged."""
        # The two stars are ~0.5 arcsec apart -- essentially on top of one another.
        radecs = np.array([[10.0, 0.0], [10.0 + 0.5 / 3600.0, 0.0]])
        mags = np.array([10.0, np.nan])
        flag = neighbor_contamination_flag_sky(radecs, mags, fwhm_arcsec=2.0)
        assert not flag.any()

    def test_matches_pixel_front_end_on_equator(self):
        """
        Sky and pixel front ends agree star-for-star.

        Stars are placed along the celestial equator, where the great-circle
        separation between ``(ra, 0)`` points is exactly the RA difference. The
        equivalent pixel layout is those angular separations divided by the plate
        scale, and ``fwhm_arcsec == fwhm_pix * pixscale``; the two front ends scale
        identically, so they must flag the same stars.
        """
        pixscale = 2.4  # arcsec / pixel
        fwhm_pix = 2.0
        fwhm_arcsec = fwhm_pix * pixscale

        ra0 = 10.0
        offsets_arcsec = np.array([0.0, 3.0, 7.0, 30.0])
        ras = ra0 + offsets_arcsec / 3600.0
        decs = np.zeros_like(ras)
        radecs = np.column_stack([ras, decs])
        mags = np.array([12.0, 12.0, 9.0, 13.0])

        coords_pix = np.column_stack(
            [offsets_arcsec / pixscale, np.zeros_like(ras)],
        )

        sky_flag = neighbor_contamination_flag_sky(radecs, mags, fwhm_arcsec)
        pix_flag = neighbor_contamination_flag(coords_pix, mags, fwhm_pix)
        np.testing.assert_array_equal(sky_flag, pix_flag)
        # Guard: the case is non-trivial -- some flagged, some not.
        assert sky_flag.any()
        assert not sky_flag.all()

    def test_all_non_finite_magnitudes_never_flagged(self):
        """With no finite magnitude there is no valid pair, so nothing is flagged."""
        radecs = np.array([[10.0, 0.0], [10.0 + 0.5 / 3600.0, 0.0]])
        mags = np.array([np.nan, np.nan])
        flag = neighbor_contamination_flag_sky(radecs, mags, fwhm_arcsec=2.0)
        assert flag.shape == (2,)
        assert not flag.any()

    def test_zero_fwhm_never_flags(self):
        """A zero FWHM makes every required separation zero, so nothing is flagged."""
        radecs = np.array([[10.0, 0.0], [10.0 + 0.5 / 3600.0, 0.0]])
        mags = np.array([10.0, 12.0])
        flag = neighbor_contamination_flag_sky(radecs, mags, fwhm_arcsec=0.0)
        assert not flag.any()

    def test_matches_dense_all_pairs_reference_on_random_field(self):
        """
        The sky flag reproduces a brute-force all-pairs reference.

        The reference applies the documented contamination rule directly to an
        explicit N x N great-circle separation matrix: target i is flagged when
        any other star j with finite magnitudes sits closer than
        ``min_separation_fwhm(mag_i - mag_j) * fwhm``. This pins the sky front
        end (whatever its internal pair search) to the dense model on a
        realistic random field, including NaN magnitudes and an exact-duplicate
        position (a zero-separation pair, which the rule flags).
        """
        # A dense random field in a small RA/Dec patch, with some NaN
        # magnitudes and one exact-duplicate position to exercise edge cases.
        rng = np.random.default_rng(SEED)
        n = 800
        radecs = np.column_stack(
            [rng.uniform(10.0, 10.5, n), rng.uniform(19.75, 20.25, n)],
        )
        mags = rng.uniform(8.0, 18.0, n)
        mags[rng.choice(n, 20, replace=False)] = np.nan
        radecs[5] = radecs[4]
        fwhm_arcsec = 5.0

        # Brute-force reference: full N x N great-circle separations.
        coords = SkyCoord(radecs[:, 0], radecs[:, 1], unit="deg")
        sep_arcsec = coords[:, None].separation(coords[None, :]).arcsec
        # Minimum allowed separation for each pair from the magnitude difference.
        required = min_separation_fwhm(mags[:, None] - mags[None, :]) * fwhm_arcsec
        # Only pairs with two finite magnitudes count, and never compare a star
        # with itself (the diagonal).
        finite = np.isfinite(mags)
        valid = finite[:, None] & finite[None, :]
        np.fill_diagonal(valid, val=False)
        # A target is flagged if any valid neighbor is closer than required.
        expected = (valid & (sep_arcsec < required)).any(axis=1)

        flag = neighbor_contamination_flag_sky(radecs, mags, fwhm_arcsec)

        np.testing.assert_array_equal(flag, expected)
        # Guard: the case is non-trivial -- some flagged, some not.
        assert expected.any()
        assert not expected.all()


def _seestar_header(*, with_stackcnt=True):
    """Build a FITS header carrying the keys referenced by ``basic.json``."""
    header = fits.Header()
    header["DATE-OBS"] = "2024-01-01T00:00:00"
    header["SITELAT"] = 40.0
    header["SITELONG"] = -105.0
    header["SITEELEV"] = 1600.0
    header["obscode"] = "ABC"
    header["FILTER"] = "L"
    header["EXPTIME"] = 10.0
    header["CREATOR"] = "ZWO Seestar S50"
    header["INSTRUME"] = "Seestar S50"
    header["PROGRAM"] = "SeestarApp"
    header["BAYERPAT"] = "RGGB"
    header["DEC"] = 20.0
    header["RA"] = 10.0
    header["OBJECT"] = "WASP-12"
    header["TELESCOP"] = "Seestar"
    header["NAXIS1"] = 1080
    header["NAXIS2"] = 1920
    if with_stackcnt:
        header["STACKCNT"] = 7
    return header


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


class TestAlign:
    """Unit tests for the WCS-solve/projection helper ``align``."""

    def test_projects_photometry_coords_through_supplied_wcs(self):
        """photometry_coords are projected to pixels via the provided WCS."""
        wcs = _make_tan_wcs(crval=(10.0, 20.0))
        sky = SkyCoord(ra=[10.0, 10.01] * u.deg, dec=[20.0, 20.01] * u.deg)
        coords = np.array([[250.0, 250.0], [260.0, 260.0]])

        aligned, returned_wcs = align(
            coords, radecs=None, photometry_coords=sky, wcs=wcs
        )

        assert returned_wcs is wcs
        expected = np.array(wcs.world_to_pixel(sky)).T
        np.testing.assert_allclose(aligned, expected)
        assert aligned.shape == (2, 2)

    def test_solves_wcs_from_detections_when_none_supplied(self, monkeypatch):
        """
        With wcs=None, align calls compute_wcs on the first N_STARS_ALIGN stars.

        twirl.compute_wcs is a slow, stochastic asterism solver and the unit
        under test is align's orchestration (input slicing + branch selection),
        not twirl's matching -- so it is stubbed with a sentinel WCS.
        """
        sentinel_wcs = _make_tan_wcs()
        calls = {}

        def fake_compute_wcs(coords, radecs, tolerance):
            calls["coords"] = coords
            calls["radecs"] = radecs
            calls["tolerance"] = tolerance
            return sentinel_wcs

        monkeypatch.setattr("bandaid.photometry.compute_wcs", fake_compute_wcs)

        n_detected = N_STARS_ALIGN + 5
        coords = np.arange(n_detected * 2, dtype=float).reshape(n_detected, 2)
        radecs = np.arange(n_detected * 2, dtype=float).reshape(n_detected, 2)

        aligned, returned_wcs = align(coords, radecs, photometry_coords=None)

        assert returned_wcs is sentinel_wcs
        # Only the brightest N_STARS_ALIGN detections/refs reach the solver.
        assert len(calls["coords"]) == N_STARS_ALIGN
        assert len(calls["radecs"]) == N_STARS_ALIGN
        # With no photometry_coords, aligned coords are the detections themselves.
        np.testing.assert_array_equal(aligned, coords)

    def test_suppresses_compute_wcs_stdout(self, monkeypatch, capsys):
        """
        Swallow the stdout twirl's asterism matcher prints.

        The matcher prints diagnostics (e.g. "Match took ... us") straight to
        stdout; align must swallow that noise so callers/notebooks stay clean.
        The WCS return value is unaffected.
        """
        sentinel_wcs = _make_tan_wcs()

        def noisy_compute_wcs(*args: object, **kwargs: object):  # noqa: ARG001
            print("Match took 12345.000 us")  # noqa: T201
            print(7)  # noqa: T201
            return sentinel_wcs

        monkeypatch.setattr("bandaid.photometry.compute_wcs", noisy_compute_wcs)

        coords = np.arange(N_STARS_ALIGN * 2, dtype=float).reshape(N_STARS_ALIGN, 2)
        radecs = coords.copy()

        _, returned_wcs = align(coords, radecs, photometry_coords=None)

        assert returned_wcs is sentinel_wcs
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize(
        "twirl_error",
        [
            # The original SS Leo failure: too few matched points reach
            # fit_wcs_from_points, so scipy's least-squares fitter raises.
            ValueError("Initial guess is outside of provided bounds"),
            # The shallower exit: cross_match finds zero pairs and the empty
            # float index array fails when used to index.
            IndexError("arrays used as indices must be of integer type"),
        ],
        ids=["fit_wcs_from_points-ValueError", "cross_match-IndexError"],
    )
    def test_twirl_raising_becomes_wcs_solve_error(self, monkeypatch, twirl_error):
        """A too-few-stars raise from twirl surfaces as a recoverable WCSSolveError."""

        def failing_compute_wcs(coords, radecs, tolerance):  # noqa: ARG001
            raise twirl_error

        monkeypatch.setattr("bandaid.photometry.compute_wcs", failing_compute_wcs)

        coords = np.arange(N_STARS_ALIGN * 2, dtype=float).reshape(N_STARS_ALIGN, 2)

        with pytest.raises(WCSSolveError, match="twirl raised") as excinfo:
            align(coords, coords.copy(), photometry_coords=None)
        # The original twirl error is preserved on the chain for the log.
        assert excinfo.value.__cause__ is twirl_error

    def test_twirl_returning_none_becomes_wcs_solve_error(self, monkeypatch):
        """compute_wcs returning None (no match) surfaces as WCSSolveError."""

        def none_compute_wcs(coords, radecs, tolerance):  # noqa: ARG001
            return None

        monkeypatch.setattr("bandaid.photometry.compute_wcs", none_compute_wcs)

        coords = np.arange(N_STARS_ALIGN * 2, dtype=float).reshape(N_STARS_ALIGN, 2)

        with pytest.raises(WCSSolveError, match="no WCS"):
            align(coords, coords.copy(), photometry_coords=None)

    def test_unexpected_twirl_error_propagates(self, monkeypatch):
        """A non too-few-stars error is a bug and is left to propagate, not masked."""
        bug = TypeError("genuine bug, not a bad frame")

        def buggy_compute_wcs(coords, radecs, tolerance):  # noqa: ARG001
            raise bug

        monkeypatch.setattr("bandaid.photometry.compute_wcs", buggy_compute_wcs)

        coords = np.arange(N_STARS_ALIGN * 2, dtype=float).reshape(N_STARS_ALIGN, 2)

        with pytest.raises(TypeError, match="genuine bug"):
            align(coords, coords.copy(), photometry_coords=None)


def test_centroid_stars_delegates_to_ballet(monkeypatch):
    """
    centroid_stars forwards (data, coords, cnn) to centroid.ballet_centroid.

    The wrapper has no logic of its own and the real call loads a Ballet CNN
    from HuggingFace, so ballet_centroid is stubbed and call-through verified.
    """
    recorded = {}
    result_sentinel = np.array([[1.0, 2.0]])

    def fake_ballet_centroid(data, coords, cnn):
        recorded["args"] = (data, coords, cnn)
        return result_sentinel

    monkeypatch.setattr(
        "bandaid.photometry.centroid.ballet_centroid",
        fake_ballet_centroid,
    )

    data = np.zeros((10, 10))
    coords = np.array([[5.0, 5.0]])
    cnn = object()
    out = centroid_stars(data, coords, cnn)

    assert out is result_sentinel
    assert recorded["args"][0] is data
    assert recorded["args"][1] is coords
    assert recorded["args"][2] is cnn


class TestCalculateL4Quantities:
    """Unit tests for the RGB->L4 combination ``calculate_l4_quantities``."""

    def test_missing_channel_raises(self):
        """A missing TR/TG/TB channel raises an actionable ValueError."""
        by_filter = {"TR": Table(), "TG": Table()}
        with pytest.raises(ValueError, match=r"\['TB'\]"):
            calculate_l4_quantities(Table(), by_filter, 0.5)

    def test_combines_rgb_filters(self):
        """L4 columns are the documented combinations of the TR/TG/TB tables."""
        egain = 0.5

        def _filter_table(tot, area, bkgd, bkgd_std, peak):
            t = Table()
            t["tot_count"] = np.array(tot, dtype=float)
            t["aperture_area"] = np.array(area, dtype=float)
            t["bkgd_count"] = np.array(bkgd, dtype=float)
            t["bkgd_std"] = np.array(bkgd_std, dtype=float)
            t["peak_count"] = np.array(peak, dtype=float)
            return t

        by_filter = {
            "TR": _filter_table([100, 200], [10, 12], [5, 6], [2, 3], [50, 90]),
            "TG": _filter_table([110, 210], [11, 13], [4, 7], [1, 2], [70, 80]),
            "TB": _filter_table([120, 220], [9, 14], [6, 5], [3, 1], [60, 95]),
        }
        final_data = Table()

        calculate_l4_quantities(final_data, by_filter, egain)

        tr, tg, tb = by_filter["TR"], by_filter["TG"], by_filter["TB"]
        expected_tot = tr["tot_count"] + tg["tot_count"] + tb["tot_count"]
        expected_area = tr["aperture_area"] + tg["aperture_area"] + tb["aperture_area"]
        expected_bkgd = (
            tr["bkgd_count"] * tr["aperture_area"]
            + tg["bkgd_count"] * tg["aperture_area"]
            + tb["bkgd_count"] * tb["aperture_area"]
        ) / expected_area
        expected_peak = np.max(
            [tr["peak_count"], tg["peak_count"], tb["peak_count"]],
            axis=0,
        )
        expected_err = np.sqrt(
            (
                tr["bkgd_std"] ** 2 * tr["aperture_area"]
                + tg["bkgd_std"] ** 2 * tg["aperture_area"]
                + tb["bkgd_std"] ** 2 * tb["aperture_area"]
            )
            + expected_tot / egain
        )
        expected_snr = expected_tot / expected_err

        np.testing.assert_allclose(final_data["tot_count"], expected_tot)
        np.testing.assert_allclose(final_data["aperture_area"], expected_area)
        np.testing.assert_allclose(final_data["bkgd_count"], expected_bkgd)
        np.testing.assert_allclose(final_data["peak_count"], expected_peak)
        np.testing.assert_allclose(final_data["count_err"], expected_err)
        np.testing.assert_allclose(final_data["snr"], expected_snr)


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
):
    """
    Build a noisy multi-Gaussian frame that eloy's detection can resolve.

    ``amplitude`` far above ``noise_stddev`` keeps detection reliable; an
    ``amplitude`` above the 50000 ADU saturation cap exercises the saturated
    path in ``calibration_sequence``.
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
        include_noise=True,
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


class TestPrepareImageBranches:
    """Branch coverage for ``prepare_image`` beyond the alignment fallback."""

    def test_raises_when_too_few_stars(self, make_test_image, tmp_path):
        """prepare_image propagates calibration_sequence's TooFewStarsError."""
        image = _detectable_image(
            make_test_image,
            n_sources=2,
            image_size=(200, 200),
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


class TestProcessOneImage:
    """End-to-end (stubbed-externals) coverage for ``process_one_image``."""

    def test_raises_when_image_rejected(self, make_test_image, tmp_path):
        """A frame with too few stars raises TooFewStarsError."""
        image = _detectable_image(
            make_test_image,
            n_sources=2,
            image_size=(200, 200),
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
        # N_STARS_ALIGN plausibly shaped rows (align slices the first
        # N_STARS_ALIGN). photometry_coords=None means aligned == detections.
        radecs = np.column_stack(
            [
                np.full(N_STARS_ALIGN, header["RA"]),
                np.full(N_STARS_ALIGN, header["DEC"]),
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


class TestFlagPartialObstruction:
    """A localized SNR drop is flagged; a uniform one is not."""

    @staticmethod
    def _grid_table(snr_grid, shape=(400, 400)) -> Table:
        """Build a table with five stars per cell of a 4x4 SNR grid."""
        height, width = shape
        xs, ys, snrs = [], [], []
        for r in range(4):
            for c in range(4):
                xs.extend([(c + 0.5) / 4 * width] * 5)
                ys.extend([(r + 0.5) / 4 * height] * 5)
                snrs.extend([snr_grid[r, c]] * 5)
        return Table({"x": xs, "y": ys, "snr": snrs})

    def test_localized_low_snr_is_flagged(self):
        """One quadrant of heavily depressed SNR trips the flag."""
        snr_grid = np.full((4, 4), 50.0)
        snr_grid[0, 0] = 2.0
        assert flag_partial_obstruction(self._grid_table(snr_grid), (400, 400))

    def test_uniform_low_snr_is_not_flagged(self):
        """A uniform SNR drop (transparency loss) does not trip the flag."""
        table = self._grid_table(np.full((4, 4), 5.0))
        assert not flag_partial_obstruction(table, (400, 400))
