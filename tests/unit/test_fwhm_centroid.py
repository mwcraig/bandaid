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


N_CATALOG = 60
N_CNN_CLASS = 10
# Gaia G of catalog row i is 8 + i, so the ten brightest rows are G <= 17.
G_CUT = 17.0
PLANE_TOLERANCE_PIX = 0.15


def _catalog_g(n=N_CATALOG):
    """Return Gaia G values that increase with row number, one magnitude apart."""
    return 8.0 + np.arange(n)


def _measured(coords):
    """Return CNN-like positions: input plus the affine offset plus a small ripple."""
    ripple = 0.05 * np.column_stack(
        [np.sin(7.3 * coords[:, 0]), np.cos(5.1 * coords[:, 1])]
    )
    return coords + _affine_offset(coords) + ripple


@pytest.fixture
def cnn_calls(mocker):
    """Patch ``centroid_stars`` with `_measured`; return the list of coordinate sets."""
    calls = []

    def fake(_data, coords, _cnn):
        calls.append(np.array(coords))
        return _measured(coords)

    mocker.patch("bandaid.photometry.centroid_stars", side_effect=fake)
    return calls


def _run_policy(projected, gaia_g, *, config=None, g_cut=G_CUT):
    """Run `centroid_with_prior` on a blank frame of ``FRAME_SHAPE``."""
    return centroid_with_prior(
        np.zeros(FRAME_SHAPE),
        projected,
        None,
        gaia_g=gaia_g,
        g_cut=g_cut,
        config=config,
    )


class TestCentroidWithPrior:
    """The batch-fixed CNN class keeps its centroid; the rest take the plane."""

    def test_without_gaia_g_every_star_goes_to_the_cnn(self, cnn_calls):
        """With no magnitudes there is no policy: one CNN call on every star."""
        projected = _fit_stars(N_CATALOG)

        result = centroid_with_prior(np.zeros(FRAME_SHAPE), projected, None)

        assert len(cnn_calls) == 1
        np.testing.assert_array_equal(cnn_calls[0], projected)
        np.testing.assert_array_equal(result.coords, _measured(projected))
        assert (result.method == "cnn").all()
        assert not result.active

    def test_without_a_cut_every_star_goes_to_the_cnn(self, cnn_calls):
        """Magnitudes without a batch cut give the plain all-CNN result."""
        projected = _fit_stars(N_CATALOG)

        result = _run_policy(projected, _catalog_g(), g_cut=None)

        assert len(cnn_calls) == 1
        assert (result.method == "cnn").all()
        assert not result.active

    def test_switched_off_every_star_goes_to_the_cnn(self, cnn_calls):
        """``gaia_prior=False`` gives the plain all-CNN result."""
        projected = _fit_stars(N_CATALOG)

        result = _run_policy(
            projected, _catalog_g(), config=CentroidConfig(gaia_prior=False)
        )

        assert len(cnn_calls) == 1
        np.testing.assert_array_equal(result.coords, _measured(projected))
        assert not result.active

    def test_class_stars_keep_their_cnn_centroid(self, cnn_calls):
        """A star at or brighter than the cut keeps exactly its CNN position."""
        projected = _fit_stars(N_CATALOG)

        result = _run_policy(projected, _catalog_g())

        cnn_class = slice(0, N_CNN_CLASS)
        np.testing.assert_array_equal(
            result.coords[cnn_class], _measured(projected[cnn_class])
        )
        assert (result.method[cnn_class] == "cnn").all()

    def test_a_star_at_the_cut_is_in_the_class_and_one_fainter_is_not(self, cnn_calls):
        """The class is G <= cut: row 9 (G = 17) is in, row 10 (G = 18) is out."""
        projected = _fit_stars(N_CATALOG)

        result = _run_policy(projected, _catalog_g())

        assert result.method[N_CNN_CLASS - 1] == "cnn"
        assert result.method[N_CNN_CLASS] == "plane"

    def test_other_stars_take_the_projected_position_plus_the_plane(self, cnn_calls):
        """Stars outside the class sit on projected plus the fitted plane."""
        projected = _fit_stars(N_CATALOG)

        result = _run_policy(projected, _catalog_g())

        rest = slice(N_CNN_CLASS, None)
        expected = projected[rest] + _affine_offset(projected[rest])
        assert np.abs(result.coords[rest] - expected).max() < PLANE_TOLERANCE_PIX
        assert (result.method[rest] == "plane").all()
        assert result.plane is not None
        assert not result.fallback
        assert result.active

    def test_the_cnn_sees_only_the_class_and_the_fit_set(self, cnn_calls):
        """The 30 brightest are centroided once; every fainter star never is."""
        projected = _fit_stars(N_CATALOG)

        _run_policy(projected, _catalog_g())

        assert len(cnn_calls) == 1
        fit_n_stars = CentroidConfig().fit_n_stars
        np.testing.assert_array_equal(cnn_calls[0], projected[:fit_n_stars])

    def test_fit_stars_outside_the_class_are_output_at_the_plane(self, cnn_calls):
        """A fit-set star that is not CNN-class is measured but not output."""
        projected = _fit_stars(N_CATALOG)

        result = _run_policy(projected, _catalog_g())

        fit_only = slice(N_CNN_CLASS, CentroidConfig().fit_n_stars)
        measured = _measured(projected[fit_only])
        assert not np.allclose(result.coords[fit_only], measured, atol=1e-6)
        assert (result.method[fit_only] == "plane").all()

    def test_class_larger_than_the_fit_set_is_still_measured(self, cnn_calls):
        """Class stars outside the fit set are centroided too, but do not fit."""
        projected = _fit_stars(N_CATALOG)
        config = CentroidConfig(fit_n_stars=12, min_fit_stars=12)
        big_class_cut = 8.0 + 19  # twenty class stars

        result = _run_policy(projected, _catalog_g(), config=config, g_cut=big_class_cut)

        n_class = 20
        np.testing.assert_array_equal(cnn_calls[0], projected[:n_class])
        np.testing.assert_array_equal(
            result.coords[:n_class], _measured(projected[:n_class])
        )
        assert (result.method[:n_class] == "cnn").all()
        assert (result.method[n_class:] == "plane").all()

    def test_a_star_without_gaia_g_is_in_the_class_and_not_in_the_fit(self, cnn_calls):
        """A forced target has no G: it keeps its CNN centroid and never fits."""
        projected = _fit_stars(N_CATALOG + 1)
        gaia_g = np.append(_catalog_g(), np.nan)

        result = _run_policy(projected, gaia_g)

        assert result.method[-1] == "cnn"
        np.testing.assert_array_equal(result.coords[-1], _measured(projected[-1:])[0])
        fit_n_stars = CentroidConfig().fit_n_stars
        np.testing.assert_array_equal(
            cnn_calls[0], projected[[*range(fit_n_stars), N_CATALOG]]
        )

    def test_the_fit_set_is_the_brightest_whatever_the_row_order(self, cnn_calls):
        """The fit set is chosen by G, not by position in the catalog."""
        projected = _fit_stars(N_CATALOG)
        order = np.random.default_rng(SEED).permutation(N_CATALOG)

        _run_policy(projected[order], _catalog_g()[order])

        fit_n_stars = CentroidConfig().fit_n_stars
        np.testing.assert_array_equal(
            np.sort(cnn_calls[0][:, 0]), np.sort(projected[:fit_n_stars, 0])
        )

    def test_row_order_and_shape_are_preserved(self, cnn_calls):
        """The result is row-aligned with the input."""
        projected = _fit_stars(N_CATALOG)

        result = _run_policy(projected, _catalog_g())

        assert result.coords.shape == projected.shape
        assert result.method.shape == (N_CATALOG,)
        assert result.expected.shape == projected.shape

    def test_expected_positions_are_projected_plus_the_plane(self, cnn_calls):
        """The drift reference is the plane position for every row."""
        projected = _fit_stars(N_CATALOG)

        result = _run_policy(projected, _catalog_g())

        np.testing.assert_allclose(
            result.expected, projected + result.plane.offsets(projected)
        )

    def test_too_few_fit_stars_gives_the_all_cnn_frame(self, cnn_calls):
        """Under 12 fit stars the whole frame is centroided by the CNN."""
        n_stars = 11
        projected = _fit_stars(n_stars)

        result = _run_policy(projected, _catalog_g(n_stars), g_cut=8.0 + 4)

        np.testing.assert_array_equal(result.coords, _measured(projected))
        assert result.plane is None
        assert result.fallback
        n_class = 5
        assert (result.method[:n_class] == "cnn").all()
        assert (result.method[n_class:] == "fallback_cnn").all()
        np.testing.assert_array_equal(result.expected, projected)

    def test_the_cnn_is_never_called_with_an_empty_array(self, cnn_calls):
        """A frame whose stars are all fit-set stars still calls the CNN once."""
        projected = _fit_stars(N_CATALOG)

        _run_policy(projected, _catalog_g(), g_cut=-np.inf)

        assert all(len(call) for call in cnn_calls)


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
