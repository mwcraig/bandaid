"""
Unit tests for the photometry pipeline in :mod:`bandaid.photometry`.

Covers aperture photometry on synthetic single-source images
(``measure_photometry``), the bright-neighbor minimum-separation model
(``min_separation_fwhm``), and the detect/align/centroid path
(``prepare_image``), using the synthetic-image fixtures from ``conftest.py``.
"""

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
from bandaid.photometry import (
    ANNULUS,
    RELATIVE_RADII,
    ImageData,
    build_photometry_table,
    centroid_drift_flag,
    min_separation_fwhm,
    prepare_image,
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
    """Unit tests for the centroid-drift sanity check ``centroid_drift_flag``."""

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
