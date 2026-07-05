"""Unit tests for ``measure_photometry`` and peak-count semantics."""

import warnings

import numpy as np
import pytest
from _helpers import (
    SEED,
    _bright_neighbor_scene,
    _peak_scene_photometry,
    _single_source_photometry_inputs,
)
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.table import Table

from bandaid import measure_photometry
from bandaid.image2sl_qt import generate_bayer_masks
from bandaid.photometry import (
    ANNULUS,
    RELATIVE_RADII,
)


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
    """Omitting radii/annulus matches passing the module constants."""
    image, coords, fwhm, mask = _single_source_photometry_inputs(make_test_image)
    egain = 0.3

    default = measure_photometry(image, coords, fwhm, egain, mask)
    explicit = measure_photometry(
        image,
        coords,
        fwhm,
        egain,
        mask,
        radii=RELATIVE_RADII,
        annulus=ANNULUS,
    )

    np.testing.assert_array_equal(default["tot_count"], explicit["tot_count"])
    np.testing.assert_array_equal(default["count_err"], explicit["count_err"])
    assert default["aperture_radii"] == explicit["aperture_radii"]
    assert default["annulus_radii"] == explicit["annulus_radii"]


def test_measure_photometry_custom_radii(make_test_image):
    """A custom radii drives the output shapes and aperture radius."""
    image, coords, fwhm, mask = _single_source_photometry_inputs(make_test_image)
    egain = 0.3
    radii = [1.0, 2.0]

    photom = measure_photometry(
        image,
        coords,
        fwhm,
        egain,
        mask,
        radii=radii,
    )

    # One column per requested radius.
    assert photom["fluxes"].shape == (1, len(radii))
    assert photom["total_bkg"].shape == (1, len(radii))
    # A larger aperture captures more flux for a Gaussian source.
    assert photom["fluxes"][0, 1] > photom["fluxes"][0, 0]
    # The reported aperture radius is the first requested radius times the FWHM.
    assert photom["aperture_radii"] == pytest.approx(radii[0] * fwhm)


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
        fwhm,
        egain,
        mask,
        annulus=annulus,
    )

    max_aper = np.max(RELATIVE_RADII) * fwhm
    expected = (max(max_aper, annulus[0] * fwhm), annulus[1] * fwhm)
    assert photom["annulus_radii"] == pytest.approx(expected)


def test_measure_photometry_accepts_list_radii(make_test_image):
    """A plain list (not just ndarray) works for radii."""
    image, coords, fwhm, mask = _single_source_photometry_inputs(make_test_image)
    egain = 0.3

    photom = measure_photometry(
        image,
        coords,
        fwhm,
        egain,
        mask,
        radii=[1.0],
    )

    assert photom["aperture_radii"] == pytest.approx(1.0 * fwhm)


def test_measure_photometry_accepts_scalar_radii(make_test_image):
    """A bare scalar radii is coerced to 1D and works as one aperture."""
    image, coords, fwhm, mask = _single_source_photometry_inputs(make_test_image)
    egain = 0.3

    photom = measure_photometry(
        image,
        coords,
        fwhm,
        egain,
        mask,
        radii=1.0,
    )

    # A scalar behaves like a single-element radius list.
    assert photom["fluxes"].shape == (len(coords), 1)
    assert photom["aperture_radii"] == pytest.approx(1.0 * fwhm)


def test_measure_photometry_negative_net_count_no_warning():
    """
    Negative net counts emit no RuntimeWarning but still yield NaN SNR (#30).

    A faint star whose annulus background over-subtracts gives ``net_count < 0``,
    so ``sqrt(egain * net_count)`` is the root of a negative number. The NaN is an
    expected intermediate that ``eloy_to_starlist`` filters downstream, so the
    function must not spray ``RuntimeWarning: invalid value encountered in sqrt``.
    """
    fwhm = 2.3
    egain = 0.3
    side = int(max(ANNULUS) * fwhm * 2) + 10
    # Bright, uniform background with a dark hole at the aperture: the 1*FWHM
    # aperture sums to ~0 while the distant annulus reads the bright background,
    # so net_count = flux - total_bkg is strongly negative.
    background = 100.0
    image = np.full((side, side), background, dtype=float)
    center = side / 2.0
    yy, xx = np.mgrid[0:side, 0:side]
    radius = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    image[radius <= 1.5 * fwhm] = 0.0
    coords = np.array([[center, center]])
    mask = np.zeros_like(image, dtype=bool)

    with warnings.catch_warnings():
        # Promote RuntimeWarning specifically to an error so the test fails if
        # the function emits it; other warning categories are left untouched.
        warnings.simplefilter("error", RuntimeWarning)
        photom = measure_photometry(image, coords, fwhm, egain, mask)

    # The NaN contract: negative count survives, SNR is NaN, no warning was raised.
    assert photom["tot_count"][0] < 0
    assert np.isnan(photom["snr"][0])


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
            fwhm,
            egain,
            mask,
            radii=[20.0],
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
        fwhm,
        egain,
        mask,
        annulus=iter((6, 10)),
    )

    expected = (max(np.max(RELATIVE_RADII) * fwhm, 6 * fwhm), 10 * fwhm)
    assert photom["annulus_radii"] == pytest.approx(expected)


# --- Regression tests for issue #54: peak_count semantics ---


# Any peak_count above this reveals the ~5000-count neighbor leaking into the
# box; the measured stars themselves peak at ~110 (amplitude 100 + sky 10).
_PEAK_LEAK_CEILING = 200.0


def test_peak_count_is_the_targets_own_peak(make_test_image):
    """
    ``peak_count`` reports the star's own peak, not a bright neighbor's (#54).

    The 5000-count neighbor 10 px away sat inside the old fixed 25x25 box, so
    the faint target reported peak_count ~5010 instead of its own ~110. The
    ~2*FWHM box anchored at the measured centroid must not reach it.
    """
    image, coords = _bright_neighbor_scene(make_test_image)

    photom = _peak_scene_photometry(image, coords, None)

    peak_target, peak_control = photom["peak_count"]
    # Both stars peak at ~110 (amplitude 100 + sky 10); anything approaching
    # the neighbor's ~5000 means the peak box leaked onto the neighbor.
    assert peak_control < _PEAK_LEAK_CEILING
    assert peak_target < _PEAK_LEAK_CEILING


def test_peak_count_respects_the_channel_mask(make_test_image):
    """
    Per-channel ``peak_count`` is the max over that channel's pixels only (#54).

    Both measured stars are centered on R pixels of the RGGB mosaic, so the TR
    peak reads the central pixel while TG (nearest unmasked pixel 1 px away)
    and TB (sqrt(2) px away) sample progressively farther down the profile:
    the channels must come out in that strict order. Before the fix the mask
    was never applied and the TR/TG/TB columns were bit-identical.
    """
    image, coords = _bright_neighbor_scene(make_test_image)
    masks = generate_bayer_masks(
        image.shape,
        {"bayerpat": "RGGB", "roworder": "top-down", "ybayroff": 0},
        append_l4=False,
    )

    peaks = {
        name: _peak_scene_photometry(image, coords, mask)["peak_count"]
        for name, mask in masks.items()
    }

    for star in range(len(coords)):
        assert peaks["TR"][star] > peaks["TG"][star] > peaks["TB"][star]
    # The mask must not reintroduce the neighbor: every channel still reports
    # the target's own (faint) peak.
    for channel_peaks in peaks.values():
        assert np.all(channel_peaks < _PEAK_LEAK_CEILING)


@pytest.mark.parametrize("channel", ["TR", "TG", "TB"])
def test_peak_count_masked_star_at_frame_edge(make_test_image, channel):
    """
    A masked peak box hanging off the frame edge reads only in-frame pixels.

    At the (0, 0) corner most of the ~2*FWHM box is out-of-frame padding,
    which must count as excluded -- never as usable data -- while the
    surviving in-frame unmasked pixels still supply a finite peak. The corner
    is far from every source, so each channel must read exactly the sky
    pedestal.
    """
    sky = 10.0
    image, _ = _bright_neighbor_scene(make_test_image, sky=sky)
    masks = generate_bayer_masks(
        image.shape,
        {"bayerpat": "RGGB", "roworder": "top-down", "ybayroff": 0},
        append_l4=False,
    )
    coords = np.array([[0.0, 0.0]])

    photom = _peak_scene_photometry(image, coords, masks[channel])

    assert photom["peak_count"][0] == sky


def test_peak_count_nan_for_non_finite_centroid(make_test_image):
    """
    A non-finite centroid yields NaN ``peak_count``; other rows unaffected (#54).

    A failed centroid can be NaN, and integer-indexing a cutout there would
    raise. Per the NaN contract in ``measure_photometry`` the row must instead
    degrade to NaN (it is dropped downstream by the ``tot_count``/``count_err``
    filters) while every other row keeps its all-finite-run values.
    """
    image, coords = _bright_neighbor_scene(make_test_image)
    baseline = _peak_scene_photometry(image, coords, None)

    bad_centroids = coords.copy()
    bad_centroids[0] = np.nan
    photom = _peak_scene_photometry(image, bad_centroids, None)

    assert np.isnan(photom["peak_count"][0])
    # The finite row is bit-identical to the all-finite run, for the peak and
    # for the rest of the per-star outputs.
    for key in ("peak_count", "tot_count", "count_err", "bkgd_count", "snr"):
        assert photom[key][1] == baseline[key][1]
