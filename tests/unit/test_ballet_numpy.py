"""Unit tests for the pure-numpy Ballet centroid CNN."""

import importlib.util
import os
import warnings

import numpy as np
import pytest

from bandaid import ballet_numpy
from bandaid.ballet_numpy import (
    BALLET_HF_REPO_ID,
    BALLET_WEIGHTS_FILENAME,
    NumpyBallet,
    download_weights,
)

# Exact key names and shapes of the pretrained centroid_15x15.npz, used to
# build random-init stand-in weights for the offline tests.
_WEIGHT_SHAPES = {
    "Conv_0_kernel": (3, 3, 1, 64),
    "Conv_0_bias": (64,),
    "Conv_1_kernel": (3, 3, 64, 128),
    "Conv_1_bias": (128,),
    "Conv_2_kernel": (3, 3, 128, 256),
    "Conv_2_bias": (256,),
    "Dense_0_kernel": (4096, 2048),
    "Dense_0_bias": (2048,),
    "Dense_1_kernel": (2048, 512),
    "Dense_1_bias": (512,),
    "Dense_2_kernel": (512, 2),
    "Dense_2_bias": (2,),
}


def _random_weights_npz(tmp_path):
    """
    Write a random-init weights file with the real key names/shapes.

    The small scale keeps the sigmoids away from saturation so the offline
    tests exercise a numerically ordinary forward pass.
    """
    rng = np.random.default_rng(0)
    path = tmp_path / "random_ballet_weights.npz"
    np.savez(
        path,
        **{
            key: rng.normal(scale=0.05, size=shape).astype(np.float32)
            for key, shape in _WEIGHT_SHAPES.items()
        },
    )
    return str(path)


def _make_synthetic_cutouts():
    """
    Build deterministic star cutouts with known subpixel centers.

    Returns
    -------
    cutouts : numpy.ndarray
        ``(16, 15, 15)`` float64 (float64 on purpose: exercises the model's
        cast to float32) Gaussian stars on a sky background with Poisson-ish
        noise.
    centers : numpy.ndarray
        ``(16, 2)`` true centers as (row, col) array indices, within +/-2 px
        of the cutout center (7, 7).
    """
    rng = np.random.default_rng(42)
    n_stars = 16
    yy, xx = np.mgrid[0:15, 0:15].astype(np.float64)
    centers = 7.0 + rng.uniform(-2.0, 2.0, size=(n_stars, 2))
    sigmas = rng.uniform(1.2, 2.5, size=n_stars)
    amplitudes = rng.uniform(500.0, 5000.0, size=n_stars)
    sky = 100.0

    cutouts = np.empty((n_stars, 15, 15), dtype=np.float64)
    for i in range(n_stars):
        cy, cx = centers[i]
        star = amplitudes[i] * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigmas[i] ** 2)
        )
        star += sky
        # Gaussian approximation to Poisson counting noise.
        star += rng.normal(scale=np.sqrt(star))
        cutouts[i] = star
    return cutouts, centers


@pytest.fixture(scope="module")
def synthetic_cutouts():
    """Deterministic ``(cutouts, true_centers)`` pair shared across tests."""
    return _make_synthetic_cutouts()


# NumpyBallet output for the synthetic cutouts with the real pretrained
# weights, captured from a run that passed test_numpy_matches_jax_ballet.
# Guards against silent drift in the forward pass or the published weights.
_GOLDEN_CENTROIDS = np.array(
    [
        [6.778319, 8.089869],
        [7.812648, 8.421933],
        [8.922833, 5.409710],
        [8.164245, 8.059281],
        [6.790412, 5.478280],
        [8.705327, 6.464697],
        [8.297937, 7.560709],
        [5.932405, 6.778769],
        [5.242603, 7.234195],
        [7.497408, 8.319678],
        [6.430659, 8.063524],
        [8.566591, 8.927375],
        [5.791152, 8.134964],
        [5.177485, 6.906373],
        [7.751380, 5.608217],
        [8.868269, 7.985824],
    ],
    dtype=np.float32,
)


@pytest.mark.remote_data
@pytest.mark.skipif(
    importlib.util.find_spec("flax") is None, reason="jax/flax not installed"
)
def test_numpy_matches_jax_ballet(synthetic_cutouts):
    """NumpyBallet reproduces the jax/flax Ballet to float32 round-off."""
    from eloy.ballet.model import Ballet

    cutouts, _ = synthetic_cutouts
    weights = download_weights()
    jax_out = np.asarray(Ballet(model_file=weights).centroid(cutouts))
    numpy_out = NumpyBallet(model_file=weights).centroid(cutouts)
    np.testing.assert_allclose(numpy_out, jax_out, atol=1e-4, rtol=0)


@pytest.mark.remote_data
def test_golden_centroids_real_weights(synthetic_cutouts):
    """Pretrained-weights output matches the baked golden values (no jax)."""
    cutouts, centers = synthetic_cutouts
    model = NumpyBallet(model_file=download_weights())
    out = model.centroid(cutouts)

    np.testing.assert_allclose(out, _GOLDEN_CENTROIDS, atol=1e-3, rtol=0)
    # Sanity: the CNN actually finds the stars. The output is (x, y), i.e.
    # (col, row); compare against the true centers flipped to match.
    np.testing.assert_array_less(np.abs(out - centers[:, ::-1]), 0.3)


class TestNumpyBalletOffline:
    """Offline behaviour with random-init weights (no network, no jax)."""

    def test_centroid_shape_dtype_and_flip(self, tmp_path):
        """Output is (N, 2) float32 and is ``_forward`` with columns flipped."""
        model = NumpyBallet(model_file=_random_weights_npz(tmp_path))
        cutouts, _ = _make_synthetic_cutouts()

        out = model.centroid(cutouts)

        assert out.shape == (len(cutouts), 2)
        assert out.dtype == np.float32
        forward = model._forward(np.asarray(cutouts, dtype=np.float32)[..., None])
        np.testing.assert_array_equal(out, forward[:, ::-1])

    def test_empty_batch_returns_0x2(self, tmp_path):
        """An empty batch yields an empty (0, 2) float32 result."""
        model = NumpyBallet(model_file=_random_weights_npz(tmp_path))
        out = model.centroid(np.empty((0, 15, 15)))
        assert out.shape == (0, 2)
        assert out.dtype == np.float32

    def test_constant_cutout_yields_nan_silently(self, tmp_path):
        """A flat cutout normalizes to 0/0 -> NaN without a RuntimeWarning."""
        model = NumpyBallet(model_file=_random_weights_npz(tmp_path))
        flat = np.full((1, 15, 15), 7.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = model.centroid(flat)

        assert np.isnan(out).all()
        assert not any(
            issubclass(w.category, RuntimeWarning) for w in caught
        ), [str(w.message) for w in caught]

    def test_default_download_when_no_model_file(self, tmp_path, monkeypatch):
        """With no model_file, the weights come from ``download_weights``."""
        npz = _random_weights_npz(tmp_path)
        calls = []
        monkeypatch.setattr(
            ballet_numpy, "download_weights", lambda: calls.append(1) or npz
        )

        model = NumpyBallet()

        assert calls == [1]
        expected = np.load(npz)
        np.testing.assert_array_equal(
            model.params["Conv_0"]["kernel"], expected["Conv_0_kernel"]
        )

    def test_download_weights_targets_ballet_repo(self, monkeypatch):
        """``download_weights`` asks the hub for the Ballet npz, xet disabled."""
        import huggingface_hub

        monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
        recorded = {}

        def _fake_download(repo_id, filename):
            recorded["repo_id"] = repo_id
            recorded["filename"] = filename
            return "/cached/weights.npz"

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)

        assert download_weights() == "/cached/weights.npz"
        assert recorded == {
            "repo_id": BALLET_HF_REPO_ID,
            "filename": BALLET_WEIGHTS_FILENAME,
        }
        assert os.environ["HF_HUB_DISABLE_XET"] == "1"
