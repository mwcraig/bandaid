"""Unit tests for FWHM estimation, CNN centroiding, and centroid-drift flags."""

import numpy as np
import pytest
from _helpers import SEED, _seestar_header, five_diagonal_regions
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.table import Table

from bandaid.config import CentroidConfig
from bandaid.photometry import (
    _brightest_unsaturated,
    _centroid_prior_diagnostics,
    _fit_offset_plane,
    _fwhm_from_coords,
    calibration_sequence,
    centroid_drift_flag,
    centroid_stars,
    centroid_with_prior,
)


def _grid_star_image(make_test_image, fwhm, *, jitter=1.0, seed=SEED):
    """
    Build a noisy image of a grid of identical sub-pixel Gaussian stars.

    Unlike the other image-generation helpers, this hands back ground-truth
    coordinates alongside the frame:

    - ``make_test_image`` (conftest) is the base factory all of these wrap.
    - ``_detectable_image`` returns only an image of a few well-separated
      sources at fixed positions, tuned so eloy's detection resolves them.
    - ``_single_source_photometry_inputs`` builds a single noiseless source
      plus its ``measure_photometry`` inputs.
    - this lays down a dense grid of identical stars at deliberate sub-pixel
      offsets so a stable ePSF can be median-stacked.

    Returns ``(image, true_coords_xy, jittered_coords_xy)``. ``true_coords_xy`` are
    the exact (sub-pixel) star centers; ``jittered_coords_xy`` are those centers
    displaced by up to ``jitter`` px, standing in for an imperfect detection centroid
    that smears a position-stacked PSF. Together they let the registration test prove
    the effective PSF survives centroid error.
    """
    rng = np.random.default_rng(seed)
    img_size = (300, 300)
    gx, gy = np.meshgrid(np.arange(40, 280, 48.0), np.arange(40, 280, 48.0))
    xs = gx.ravel() + rng.uniform(-0.5, 0.5, gx.size)
    ys = gy.ravel() + rng.uniform(-0.5, 0.5, gy.size)
    n = xs.size
    sigma = fwhm * gaussian_fwhm_to_sigma
    src = Table(
        {
            "amplitude": [500.0] * n,
            "x_mean": xs,
            "y_mean": ys,
            "x_stddev": [sigma] * n,
            "y_stddev": [sigma] * n,
        },
    )
    image = make_test_image(
        image_size=img_size,
        source_properties=src,
        include_noise=True,
        noise_mean=100.0,
        noise_stddev=2.0,
        seed=seed,
    )
    true_coords = np.column_stack([xs, ys])
    jit = rng.uniform(-jitter, jitter, true_coords.shape)
    return image, true_coords, true_coords + jit


class TestFwhmFromCoords:
    """The FWHM-from-cutouts helper, with and without CNN re-centroiding."""

    def test_cnn_registration_recovers_injected_fwhm(self, make_test_image, mocker):
        """
        A perfect CNN recovers the injected FWHM despite misregistered input.

        ``ballet_centroid`` returns the exact centers, so the registered stack
        recovers the true PSF even though the detection coordinates are jittered.
        """
        inject_fwhm = 3.0
        image, true_coords, jittered = _grid_star_image(make_test_image, inject_fwhm)

        # Perfect CNN: ballet_centroid hands back the exact star centers.
        mocker.patch(
            "bandaid.photometry.ballet_centroid",
            return_value=true_coords,
        )
        fwhm_cnn = _fwhm_from_coords(image, jittered, max_adu=50000, cnn=object())
        assert fwhm_cnn == pytest.approx(inject_fwhm, rel=0.05)

    def test_legacy_path_recovers_fwhm_with_accurate_coords(self, make_test_image):
        """
        The legacy (cnn=None) stack recovers the injected FWHM on accurate coords.

        Guards backward compatibility of the un-centroided path.
        """
        inject_fwhm = 3.0
        image, true_coords, _ = _grid_star_image(make_test_image, inject_fwhm)
        fwhm = _fwhm_from_coords(image, true_coords, max_adu=50000, cnn=None)
        assert fwhm == pytest.approx(inject_fwhm, rel=0.05)

    def test_cap_prevents_faint_junk_from_inflating_fwhm(self, make_test_image, mocker):
        """
        Capping the fit to the brightest sources recovers the true FWHM.

        With every detection fed to the fit, a realistic CNN mis-centroids the
        many faint junk sources (large random shifts) and the misregistered
        cutouts smear the stack, inflating the FWHM. Restricting the fit to the
        brightest ``n_stars`` -- selected *before* centroiding -- drops the junk
        and recovers the injected ~3 px.
        """
        inject_fwhm = 3.0
        sigma = inject_fwhm * gaussian_fwhm_to_sigma
        rng = np.random.default_rng(SEED)
        # 5x5 grid (25 bright stars), plus a large block of faint junk sources.
        gx, gy = np.meshgrid(np.arange(40, 280, 48.0), np.arange(40, 280, 48.0))
        bx, by = gx.ravel(), gy.ravel()
        n_faint = 200
        fx = rng.uniform(20.0, 280.0, n_faint)
        fy = rng.uniform(20.0, 280.0, n_faint)
        xs = np.concatenate([bx, fx])
        ys = np.concatenate([by, fy])
        src = Table(
            {
                "amplitude": [500.0] * bx.size + [30.0] * n_faint,
                "x_mean": xs,
                "y_mean": ys,
                "x_stddev": [sigma] * xs.size,
                "y_stddev": [sigma] * ys.size,
            },
        )
        image = make_test_image(
            image_size=(300, 300),
            source_properties=src,
            include_noise=True,
            noise_mean=100.0,
            noise_stddev=2.0,
            seed=SEED,
        )
        all_coords = np.column_stack([xs, ys])

        # Realistic CNN: accurate on bright sources, large random shifts on the
        # faint ones, keyed off the local peak so it is order-independent. The
        # floor sits between the bright (~600) and faint (~130) peak levels.
        bright_peak_floor = 300.0
        shift_rng = np.random.default_rng(SEED + 1)

        def realistic_cnn(data, coords, _cnn):
            coords = np.asarray(coords, dtype=float)
            iy = np.clip(np.round(coords[:, 1]).astype(int), 0, data.shape[0] - 1)
            ix = np.clip(np.round(coords[:, 0]).astype(int), 0, data.shape[1] - 1)
            faint = data[iy, ix] < bright_peak_floor
            out = coords.copy()
            out[faint] += shift_rng.uniform(-3.0, 3.0, size=(int(faint.sum()), 2))
            return out

        mocker.patch("bandaid.photometry.ballet_centroid", side_effect=realistic_cnn)
        fwhm = _fwhm_from_coords(
            image, all_coords, max_adu=50000, cnn=object(), n_stars=bx.size
        )
        assert fwhm == pytest.approx(inject_fwhm, rel=0.1)

    def test_cap_applied_before_centroiding(self, make_test_image, mocker):
        """
        The brightest-N cut runs *before* the expensive ``ballet_centroid`` call.

        That ordering is what realizes the speed win, so spy on the CNN and assert
        it never sees more than ``n_stars`` coordinates even when handed ~1000.
        """
        n_stars = 50
        image, _, _ = _grid_star_image(make_test_image, 3.0)
        rng = np.random.default_rng(SEED)
        coords = np.column_stack(
            [rng.uniform(30.0, 270.0, 1000), rng.uniform(30.0, 270.0, 1000)]
        )
        cnn = mocker.patch(
            "bandaid.photometry.ballet_centroid",
            side_effect=lambda _data, received, _cnn: np.asarray(received, dtype=float),
        )
        _fwhm_from_coords(image, coords, max_adu=50000, cnn=object(), n_stars=n_stars)
        assert len(cnn.call_args.args[1]) <= n_stars

    def test_brightest_unsaturated_keeps_high_peak_drops_saturated(self):
        """The helper returns the highest-peak coords and drops saturated/empty."""
        data = np.full((60, 60), 5.0)
        # coords are (x, y); the peak is read at data[y, x].
        data[10, 15] = 100.0  # (x=15, y=10) brightest
        data[20, 25] = 60.0  # (x=25, y=20) second
        data[30, 35] = 30.0  # (x=35, y=30) third
        data[40, 45] = 70000.0  # (x=45, y=40) saturated -> dropped
        data[50, 55] = -3.0  # (x=55, y=50) negative -> dropped
        coords = np.array(
            [[15, 10], [25, 20], [35, 30], [45, 40], [55, 50]], dtype=float
        )

        top2 = _brightest_unsaturated(data, coords, max_adu=50000, n=2)
        assert {tuple(c) for c in top2} == {(15.0, 10.0), (25.0, 20.0)}

        # n exceeds the count: keep all unsaturated, still drop saturated/empty.
        keep_all = _brightest_unsaturated(data, coords, max_adu=50000, n=10)
        assert {tuple(c) for c in keep_all} == {
            (15.0, 10.0),
            (25.0, 20.0),
            (35.0, 30.0),
        }


class TestCalibrationSequenceCnn:
    """`calibration_sequence` threads its optional ``cnn`` to the FWHM helper."""

    def test_cnn_is_passed_to_fwhm_helper(self, tmp_path, mocker):
        """The ``cnn`` given to ``calibration_sequence`` reaches the FWHM helper."""
        mocker.patch(
            "bandaid.photometry._detect_stars",
            five_diagonal_regions,
        )
        stub_fwhm = 2.5
        fwhm_helper = mocker.patch(
            "bandaid.photometry._fwhm_from_coords", return_value=stub_fwhm
        )

        path = tmp_path / "frame.fits"
        fits.PrimaryHDU(np.zeros((200, 200)), header=_seestar_header()).writeto(
            path, output_verify="silentfix"
        )
        sentinel = object()
        fwhm = calibration_sequence(path, cnn=sentinel).fwhm

        assert fwhm_helper.call_args.kwargs["cnn"] is sentinel
        assert fwhm == stub_fwhm

    def test_fwhm_n_stars_is_passed_to_fwhm_helper(self, tmp_path, mocker):
        """``fwhm_n_stars`` reaches the FWHM helper as its ``n_stars`` cap."""
        mocker.patch(
            "bandaid.photometry._detect_stars",
            five_diagonal_regions,
        )
        fwhm_helper = mocker.patch(
            "bandaid.photometry._fwhm_from_coords", return_value=2.5
        )
        requested_n_stars = 7

        path = tmp_path / "frame.fits"
        fits.PrimaryHDU(np.zeros((200, 200)), header=_seestar_header()).writeto(
            path, output_verify="silentfix"
        )
        calibration_sequence(path, fwhm_n_stars=requested_n_stars)

        assert fwhm_helper.call_args.kwargs["n_stars"] == requested_n_stars


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


FRAME_SHAPE = (200, 300)  # (height, width)
MIN_FIT_STARS = 12
N_BRIGHT = 30
FIT_NOISE_PIX = 0.05


def _fit_stars(n, *, seed=SEED):
    """Scatter ``n`` projected star positions well inside ``FRAME_SHAPE``."""
    rng = np.random.default_rng(seed)
    height, width = FRAME_SHAPE
    return np.column_stack(
        [rng.uniform(20, width - 20, n), rng.uniform(20, height - 20, n)]
    )


def _affine_offset(xy):
    """Return a known per-star CNN-minus-projected offset, linear in position."""
    height, width = FRAME_SHAPE
    nx = (xy[:, 0] - width / 2) / (width / 2)
    ny = (xy[:, 1] - height / 2) / (height / 2)
    return np.column_stack(
        [0.30 + 0.10 * nx - 0.05 * ny, -0.20 + 0.04 * nx + 0.08 * ny]
    )


def _noisy_measurements(projected, *, seed=SEED):
    """Return CNN-like positions: projected plus the affine offset plus noise."""
    rng = np.random.default_rng(seed + 1)
    return (
        projected
        + _affine_offset(projected)
        + rng.normal(0, FIT_NOISE_PIX, projected.shape)
    )


class TestFitOffsetPlane:
    """The per-frame plane fitted to (CNN - projected) offsets of bright stars."""

    def test_recovers_a_known_affine_offset(self):
        """A linear offset field is recovered across the whole frame to < 0.05 px."""
        projected = _fit_stars(40)
        measured = _noisy_measurements(projected)

        plane = _fit_offset_plane(projected, measured, FRAME_SHAPE)

        height, width = FRAME_SHAPE
        grid = np.array(
            [[2.0, 2.0], [width - 2.0, 2.0], [2.0, height - 2.0], [150.0, 100.0]]
        )
        max_error = 0.05
        assert np.abs(plane.offsets(grid) - _affine_offset(grid)).max() < max_error

    def test_reports_the_fit_diagnostics(self):
        """Centre offset, rms and star counts describe the fit."""
        projected = _fit_stars(40)
        measured = _noisy_measurements(projected)

        plane = _fit_offset_plane(projected, measured, FRAME_SHAPE)

        centre = _affine_offset(np.array([[150.0, 100.0]]))[0]
        np.testing.assert_allclose(
            [plane.coeffs_x[0], plane.coeffs_y[0]], centre, atol=0.05
        )
        assert plane.n_used + plane.n_clipped == len(projected)
        # Radial rms of two independent axes with 0.05 px noise each.
        assert plane.rms == pytest.approx(FIT_NOISE_PIX * np.sqrt(2), rel=0.3)

    def test_one_gross_outlier_is_clipped(self):
        """A star the CNN got badly wrong is clipped and does not tilt the plane."""
        projected = _fit_stars(30)
        measured = _noisy_measurements(projected)
        measured[3] += [5.0, -5.0]

        plane = _fit_offset_plane(projected, measured, FRAME_SHAPE)

        assert plane.n_clipped == 1
        assert plane.n_used == len(projected) - 1
        grid = np.array([[2.0, 2.0], [297.0, 197.0]])
        max_error = 0.1
        assert np.abs(plane.offsets(grid) - _affine_offset(grid)).max() < max_error

    @pytest.mark.parametrize(
        "shift", [[6.0, 0.0], [0.0, 6.0]], ids=["x-only", "y-only"]
    )
    def test_an_outlier_on_one_axis_is_clipped(self, shift):
        """A star that is off on either axis alone is clipped."""
        projected = _fit_stars(30)
        measured = _noisy_measurements(projected)
        measured[3] += shift

        plane = _fit_offset_plane(projected, measured, FRAME_SHAPE)

        assert plane.n_clipped == 1

    def test_rows_returned_exactly_at_the_input_are_excluded(self):
        """A CNN result equal to its input is a failed fallback, not a measurement."""
        projected = _fit_stars(40)
        measured = _noisy_measurements(projected)
        failed = np.arange(8)
        measured[failed] = projected[failed]

        plane = _fit_offset_plane(projected, measured, FRAME_SHAPE)

        # Counting the failed rows would drag the fitted offset toward zero.
        assert plane.n_used + plane.n_clipped == len(projected) - len(failed)
        centre = _affine_offset(np.array([[150.0, 100.0]]))[0]
        np.testing.assert_allclose(
            [plane.coeffs_x[0], plane.coeffs_y[0]], centre, atol=0.05
        )

    def test_non_finite_rows_are_excluded(self):
        """A NaN CNN result does not poison the fit."""
        projected = _fit_stars(40)
        measured = _noisy_measurements(projected)
        measured[[0, 1]] = np.nan

        plane = _fit_offset_plane(projected, measured, FRAME_SHAPE)

        assert plane.n_used + plane.n_clipped == len(projected) - 2
        assert np.isfinite(plane.coeffs_x).all()

    def test_fewer_than_the_minimum_gives_no_plane(self):
        """Eleven usable stars give no plane; twelve do."""
        projected = _fit_stars(MIN_FIT_STARS)
        measured = _noisy_measurements(projected)

        assert _fit_offset_plane(projected[:-1], measured[:-1], FRAME_SHAPE) is None
        assert _fit_offset_plane(projected, measured, FRAME_SHAPE) is not None

    def test_minimum_applies_after_dropping_failed_rows(self):
        """Rows excluded as failed do not count toward the minimum."""
        projected = _fit_stars(MIN_FIT_STARS + 2)
        measured = _noisy_measurements(projected)
        measured[[0, 1, 2]] = projected[[0, 1, 2]]

        assert _fit_offset_plane(projected, measured, FRAME_SHAPE) is None

    def test_minimum_applies_after_clipping(self):
        """A fit that clipping reduces below the minimum gives no plane."""
        n_stars = 30
        projected = _fit_stars(n_stars)
        measured = _noisy_measurements(projected)
        measured[0] += [8.0, 8.0]

        plane = _fit_offset_plane(
            projected, measured, FRAME_SHAPE, min_fit_stars=n_stars
        )

        assert plane is None


GARBAGE_Y = 150.0  # the fake CNN fails on stars below this row of the frame
GARBAGE_SHIFT = 3.0


def _prior_frame(make_test_image, *, n_bright=N_BRIGHT, n_faint=10, seed=SEED):
    """
    Build a frame of bright stars, faint stars and three edge-band stars.

    The bright stars sit in the lower part of the frame and the faint ones in
    the upper part (``y > GARBAGE_Y``), where the fake CNN misplaces them. The
    rows are shuffled so Gaia G, not row order, has to pick the fit stars.

    Parameters
    ----------
    make_test_image : callable
        The ``make_test_image`` factory fixture.
    n_bright : int, optional
        Number of bright stars (G from 8).
    n_faint : int, optional
        Number of faint stars (G from 14).
    seed : int, optional
        Random seed for positions, row order and noise.

    Returns
    -------
    tuple
        ``(image, projected, gaia_g, edge_rows)`` where ``edge_rows`` is a boolean
        mask of the three stars within the edge band.
    """
    rng = np.random.default_rng(seed)
    height, width = FRAME_SHAPE
    bright = np.column_stack(
        [rng.uniform(25, width - 25, n_bright), rng.uniform(25, 140, n_bright)]
    )
    faint = np.column_stack(
        [rng.uniform(25, width - 25, n_faint), rng.uniform(155, 185, n_faint)]
    )
    edge = np.array(
        [[3.0, 100.0], [150.0, height - 0.5 - 3.5], [width - 0.5 - 2.5, 60.0]]
    )
    projected = np.vstack([bright, faint, edge])
    gaia_g = np.concatenate(
        [
            8.0 + 0.1 * np.arange(n_bright),
            14.0 + 0.1 * np.arange(n_faint),
            [10.0, 11.0, 12.0],
        ]
    )
    is_edge = np.arange(len(projected)) >= n_bright + n_faint
    order = rng.permutation(len(projected))
    sigma = 2.0 * gaussian_fwhm_to_sigma
    sources = Table(
        {
            "amplitude": [500.0] * len(projected),
            "x_mean": projected[:, 0],
            "y_mean": projected[:, 1],
            "x_stddev": [sigma] * len(projected),
            "y_stddev": [sigma] * len(projected),
        }
    )
    image = make_test_image(
        image_size=FRAME_SHAPE,
        source_properties=sources,
        noise_mean=100.0,
        noise_stddev=2.0,
        seed=seed,
    )
    return image, projected[order], gaia_g[order], is_edge[order]


def _fake_cnn(mocker):
    """
    Stand in for the Ballet CNN with a known offset field and a failure region.

    Every star is returned at its input plus the affine offset and a little
    noise, except stars in the upper part of the frame, which come back
    `GARBAGE_SHIFT` px off. Returns the mock; its ``call_args_list`` holds every
    coordinate array the CNN was asked to centroid.
    """
    rng = np.random.default_rng(SEED)

    def _centroid(_data, coords, _cnn):
        out = coords + _affine_offset(coords)
        out = out + rng.normal(0, 0.02, out.shape)
        out[coords[:, 1] > GARBAGE_Y] += GARBAGE_SHIFT
        return out

    return mocker.patch(
        "bandaid.photometry.centroid.ballet_centroid", side_effect=_centroid
    )


def _cnn_inputs(mock):
    """Return every coordinate row the mocked CNN was asked about."""
    if not mock.call_args_list:
        return np.zeros((0, 2))
    return np.vstack([call.args[1] for call in mock.call_args_list])


def _edge_expected(projected):
    """Return where a perfect plane puts the given projected positions."""
    return projected + _affine_offset(projected)


class TestCentroidWithPrior:
    """The edge-band rule: projected position plus plane instead of the CNN."""

    def test_edge_star_is_never_sent_to_the_cnn_and_lands_on_the_prior(
        self, make_test_image, mocker
    ):
        """A star a few px from an edge keeps its prior position, not a CNN guess."""
        image, projected, gaia_g, is_edge = _prior_frame(make_test_image)
        cnn_mock = _fake_cnn(mocker)

        result = centroid_with_prior(image, projected, object(), gaia_g=gaia_g)

        sent = _cnn_inputs(cnn_mock)
        for row in projected[is_edge]:
            assert not (sent == row).all(axis=1).any()
        np.testing.assert_allclose(
            result.coords[is_edge], _edge_expected(projected[is_edge]), atol=0.05
        )
        assert (result.method[is_edge] == "edge_plane").all()

    def test_stars_outside_the_band_keep_their_cnn_centroid(
        self, make_test_image, mocker
    ):
        """Interior rows are exactly what the CNN returned, in input row order."""
        image, projected, gaia_g, is_edge = _prior_frame(make_test_image)
        _fake_cnn(mocker)

        result = centroid_with_prior(image, projected, object(), gaia_g=gaia_g)

        interior = ~is_edge
        assert result.coords.shape == projected.shape
        assert (result.method[interior] == "cnn").all()
        # Same noisy, shifted output the fake CNN gives (to its 0.02 px noise).
        expected = _edge_expected(projected[interior])
        garbage = projected[interior][:, 1] > GARBAGE_Y
        expected[garbage] += GARBAGE_SHIFT
        np.testing.assert_allclose(result.coords[interior], expected, atol=0.1)

    def test_fit_stars_are_the_brightest_by_gaia_g(self, make_test_image, mocker):
        """The plane ignores faint stars the CNN misplaced, whatever their row."""
        image, projected, gaia_g, is_edge = _prior_frame(make_test_image)
        _fake_cnn(mocker)

        result = centroid_with_prior(image, projected, object(), gaia_g=gaia_g)

        # Using the 10 misplaced faint stars would move the plane by about a
        # pixel; the 30 brightest alone reproduce the true offset field.
        np.testing.assert_allclose(
            result.coords[is_edge], _edge_expected(projected[is_edge]), atol=0.05
        )
        assert (
            result.plane.n_used + result.plane.n_clipped == CentroidConfig().fit_n_stars
        )

    def test_fit_set_size_follows_the_config(self, make_test_image, mocker):
        """``fit_n_stars`` sets how many of the brightest stars define the plane."""
        image, projected, gaia_g, _ = _prior_frame(make_test_image)
        _fake_cnn(mocker)
        n_fit = 15

        result = centroid_with_prior(
            image,
            projected,
            object(),
            gaia_g=gaia_g,
            config=CentroidConfig(fit_n_stars=n_fit),
        )

        assert result.plane.n_used + result.plane.n_clipped == n_fit

    def test_stars_without_gaia_g_are_not_fit_stars(self, make_test_image, mocker):
        """A forced target (NaN G) is centroided but never defines the plane."""
        image, projected, gaia_g, is_edge = _prior_frame(make_test_image)
        _fake_cnn(mocker)
        # Make every faint (misplaced) star look like a forced target. The fit
        # set is allowed to be large enough that they would otherwise be in it.
        gaia_g[(projected[:, 1] > GARBAGE_Y) & ~is_edge] = np.nan

        result = centroid_with_prior(
            image,
            projected,
            object(),
            gaia_g=gaia_g,
            config=CentroidConfig(fit_n_stars=40),
        )

        assert result.plane.n_used + result.plane.n_clipped == N_BRIGHT
        np.testing.assert_allclose(
            result.coords[is_edge], _edge_expected(projected[is_edge]), atol=0.05
        )

    def test_no_plane_leaves_edge_stars_at_the_projected_position(
        self, make_test_image, mocker
    ):
        """With too few fit stars, edge rows take the bare projected position."""
        image, projected, gaia_g, is_edge = _prior_frame(
            make_test_image, n_bright=5, n_faint=0
        )
        cnn_mock = _fake_cnn(mocker)

        result = centroid_with_prior(image, projected, object(), gaia_g=gaia_g)

        assert result.plane is None
        assert result.fallback
        np.testing.assert_array_equal(result.coords[is_edge], projected[is_edge])
        assert (result.method[is_edge] == "edge_projected").all()
        # The interior stars are still measured, the edge stars still not.
        assert len(_cnn_inputs(cnn_mock)) == (~is_edge).sum()

    def test_all_stars_in_the_band_never_reach_the_cnn(self, make_test_image, mocker):
        """A frame with every star in the band does not call the CNN at all."""
        image, projected, gaia_g, _ = _prior_frame(
            make_test_image, n_bright=0, n_faint=0
        )
        cnn_mock = _fake_cnn(mocker)

        result = centroid_with_prior(image, projected, object(), gaia_g=gaia_g)

        cnn_mock.assert_not_called()
        np.testing.assert_array_equal(result.coords, projected)
        assert result.fallback

    def test_fallback_is_not_reported_when_nothing_was_in_the_band(self, mocker):
        """No plane on a frame with no edge-band stars changes nothing: no flag."""
        _fake_cnn(mocker)
        projected = _fit_stars(5)

        result = centroid_with_prior(
            np.zeros(FRAME_SHAPE), projected, object(), gaia_g=np.arange(5.0)
        )

        assert result.plane is None
        assert not result.fallback

    @pytest.mark.parametrize(
        ("xy", "in_band"),
        [
            ([8.0, 100.0], False),
            ([7.9, 100.0], True),
            ([299.5 - 8.0, 100.0], False),
            ([299.5 - 7.9, 100.0], True),
            ([150.0, 8.0], False),
            ([150.0, 7.9], True),
            ([150.0, 199.5 - 8.0], False),
            ([150.0, 199.5 - 7.9], True),
            ([-4.0, 100.0], True),
            ([150.0, 204.0], True),
        ],
    )
    def test_edge_band_boundaries(self, make_test_image, mocker, xy, in_band):
        """A star is in the band when closer than the margin to any edge or past it."""
        image, projected, gaia_g, _ = _prior_frame(make_test_image)
        cnn_mock = _fake_cnn(mocker)
        projected = np.vstack([projected, xy])
        gaia_g = np.append(gaia_g, 9.5)

        result = centroid_with_prior(image, projected, object(), gaia_g=gaia_g)

        sent = (_cnn_inputs(cnn_mock) == np.array(xy)).all(axis=1).any()
        assert sent != in_band
        assert (result.method[-1] == "edge_plane") == in_band

    def test_margin_follows_the_config(self, make_test_image, mocker):
        """``edge_margin_px`` widens or narrows the band."""
        image, projected, gaia_g, _ = _prior_frame(make_test_image)
        _fake_cnn(mocker)
        projected = np.vstack([projected, [15.0, 100.0]])
        gaia_g = np.append(gaia_g, 9.5)

        default = centroid_with_prior(image, projected, object(), gaia_g=gaia_g)
        wide = centroid_with_prior(
            image,
            projected,
            object(),
            gaia_g=gaia_g,
            config=CentroidConfig(edge_margin_px=20.0),
        )

        assert default.method[-1] == "cnn"
        assert wide.method[-1] == "edge_plane"

    def test_switched_off_is_a_plain_cnn_call(self, make_test_image, mocker):
        """With the rule off every star goes to the CNN, in one call, unchanged."""
        image, projected, gaia_g, _ = _prior_frame(make_test_image)
        cnn_mock = _fake_cnn(mocker)

        result = centroid_with_prior(
            image,
            projected,
            object(),
            gaia_g=gaia_g,
            config=CentroidConfig(edge_band_prior=False),
        )

        cnn_mock.assert_called_once()
        np.testing.assert_array_equal(cnn_mock.call_args.args[1], projected)
        assert (result.method == "cnn").all()
        assert result.plane is None
        assert not result.fallback

    def test_without_gaia_g_is_a_plain_cnn_call(self, make_test_image, mocker):
        """Without magnitudes no fit set can be chosen, so the CNN gets every star."""
        image, projected, _, _ = _prior_frame(make_test_image)
        cnn_mock = _fake_cnn(mocker)

        result = centroid_with_prior(image, projected, object())

        cnn_mock.assert_called_once()
        np.testing.assert_array_equal(cnn_mock.call_args.args[1], projected)
        assert (result.method == "cnn").all()


class TestCentroidPriorDiagnostics:
    """The per-frame plane summary recorded in the QA manifest."""

    def test_summarises_the_plane_and_the_band(self, make_test_image, mocker):
        """Centre offset, slopes, rms, counts and the edge-star count are reported."""
        image, projected, gaia_g, is_edge = _prior_frame(make_test_image)
        _fake_cnn(mocker)
        result = centroid_with_prior(image, projected, object(), gaia_g=gaia_g)

        diagnostics = _centroid_prior_diagnostics(result)

        plane = result.plane
        assert diagnostics["n_edge_prior"] == is_edge.sum()
        assert diagnostics["plane_fallback"] is False
        assert diagnostics["plane_n_used"] == plane.n_used
        assert diagnostics["plane_n_clipped"] == plane.n_clipped
        assert diagnostics["plane_rms"] == pytest.approx(plane.rms)
        assert diagnostics["plane_dx_center"] == pytest.approx(plane.coeffs_x[0])
        assert diagnostics["plane_dy_center"] == pytest.approx(plane.coeffs_y[0])
        assert diagnostics["plane_dx_slope_x"] == pytest.approx(plane.coeffs_x[1])
        assert diagnostics["plane_dx_slope_y"] == pytest.approx(plane.coeffs_x[2])
        assert diagnostics["plane_dy_slope_x"] == pytest.approx(plane.coeffs_y[1])
        assert diagnostics["plane_dy_slope_y"] == pytest.approx(plane.coeffs_y[2])

    def test_a_frame_without_a_plane_reports_the_fallback(
        self, make_test_image, mocker
    ):
        """No plane: zero stars used, no plane numbers, and the fallback flag set."""
        image, projected, gaia_g, is_edge = _prior_frame(
            make_test_image, n_bright=5, n_faint=0
        )
        _fake_cnn(mocker)
        result = centroid_with_prior(image, projected, object(), gaia_g=gaia_g)

        diagnostics = _centroid_prior_diagnostics(result)

        assert diagnostics["plane_fallback"] is True
        assert diagnostics["plane_n_used"] == 0
        assert diagnostics["n_edge_prior"] == is_edge.sum()
        assert diagnostics["plane_rms"] is None
        assert diagnostics["plane_dx_center"] is None

    def test_nothing_is_reported_when_the_rule_did_not_run(
        self, make_test_image, mocker
    ):
        """With the rule off there is no plane to describe."""
        image, projected, gaia_g, _ = _prior_frame(make_test_image)
        _fake_cnn(mocker)
        result = centroid_with_prior(
            image,
            projected,
            object(),
            gaia_g=gaia_g,
            config=CentroidConfig(edge_band_prior=False),
        )

        assert _centroid_prior_diagnostics(result) is None


def test_centroid_stars_delegates_to_ballet(mocker):
    """
    centroid_stars forwards (data, coords, cnn) to centroid.ballet_centroid.

    The wrapper has no logic of its own and the real call loads a Ballet CNN
    from HuggingFace, so ballet_centroid is stubbed and call-through verified.
    """
    result_sentinel = np.array([[1.0, 2.0]])

    ballet_centroid = mocker.patch(
        "bandaid.photometry.centroid.ballet_centroid",
        return_value=result_sentinel,
    )

    data = np.zeros((10, 10))
    coords = np.array([[5.0, 5.0]])
    cnn = object()
    out = centroid_stars(data, coords, cnn)

    assert out is result_sentinel
    ballet_centroid.assert_called_once_with(data, coords, cnn)
