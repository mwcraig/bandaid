"""Unit tests for FWHM estimation, CNN centroiding, and centroid-drift flags."""

import numpy as np
import pytest
from _helpers import SEED, _seestar_header, five_diagonal_regions
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.table import Table

from bandaid.photometry import (
    _brightest_unsaturated,
    _fwhm_from_coords,
    calibration_sequence,
    centroid_drift_flag,
    centroid_stars,
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

    def test_cnn_registration_recovers_injected_fwhm(
        self, make_test_image, monkeypatch
    ):
        """
        A perfect CNN recovers the injected FWHM despite misregistered input.

        ``ballet_centroid`` returns the exact centers, so the registered stack
        recovers the true PSF even though the detection coordinates are jittered.
        """
        inject_fwhm = 3.0
        image, true_coords, jittered = _grid_star_image(make_test_image, inject_fwhm)

        # Perfect CNN: ballet_centroid hands back the exact star centers.
        monkeypatch.setattr(
            "bandaid.photometry.ballet_centroid",
            lambda data, coords, cnn: true_coords,
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

    def test_cap_prevents_faint_junk_from_inflating_fwhm(
        self, make_test_image, monkeypatch
    ):
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

        monkeypatch.setattr("bandaid.photometry.ballet_centroid", realistic_cnn)
        fwhm = _fwhm_from_coords(
            image, all_coords, max_adu=50000, cnn=object(), n_stars=bx.size
        )
        assert fwhm == pytest.approx(inject_fwhm, rel=0.1)

    def test_cap_applied_before_centroiding(self, make_test_image, monkeypatch):
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
        seen = {}

        def spy(_data, received, _cnn):
            seen["n"] = len(received)
            return np.asarray(received, dtype=float)

        monkeypatch.setattr("bandaid.photometry.ballet_centroid", spy)
        _fwhm_from_coords(image, coords, max_adu=50000, cnn=object(), n_stars=n_stars)
        assert seen["n"] <= n_stars

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

    def test_cnn_is_passed_to_fwhm_helper(self, tmp_path, monkeypatch):
        """The ``cnn`` given to ``calibration_sequence`` reaches the FWHM helper."""
        monkeypatch.setattr(
            "bandaid.photometry.detection.stars_detection",
            five_diagonal_regions,
        )
        captured = {}
        stub_fwhm = 2.5

        def _spy(_data, _coords, _max_adu, *, cnn=None, **_kwargs: object):
            captured["cnn"] = cnn
            return stub_fwhm

        monkeypatch.setattr("bandaid.photometry._fwhm_from_coords", _spy)

        path = tmp_path / "frame.fits"
        fits.PrimaryHDU(np.zeros((200, 200)), header=_seestar_header()).writeto(
            path, output_verify="silentfix"
        )
        sentinel = object()
        _, _, _, fwhm, _ = calibration_sequence(path, cnn=sentinel)

        assert captured["cnn"] is sentinel
        assert fwhm == stub_fwhm

    def test_fwhm_n_stars_is_passed_to_fwhm_helper(self, tmp_path, monkeypatch):
        """``fwhm_n_stars`` reaches the FWHM helper as its ``n_stars`` cap."""
        monkeypatch.setattr(
            "bandaid.photometry.detection.stars_detection",
            five_diagonal_regions,
        )
        captured = {}
        requested_n_stars = 7

        def _spy(_data, _coords, _max_adu, *, n_stars=None, **_kwargs: object):
            captured["n_stars"] = n_stars
            return 2.5

        monkeypatch.setattr("bandaid.photometry._fwhm_from_coords", _spy)

        path = tmp_path / "frame.fits"
        fits.PrimaryHDU(np.zeros((200, 200)), header=_seestar_header()).writeto(
            path, output_verify="silentfix"
        )
        calibration_sequence(path, fwhm_n_stars=requested_n_stars)

        assert captured["n_stars"] == requested_n_stars


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
