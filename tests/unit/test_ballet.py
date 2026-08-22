"""Unit tests for the Ballet centroid CNN: numpy engine and backend selection."""

import importlib.util
import logging
import os
import warnings

import numpy as np
import pytest

from bandaid import ballet
from bandaid.ballet import (
    _BACKEND_ENV,
    _BALLET_HF_REPO_ID,
    _BALLET_WEIGHTS_FILENAME,
    _BALLET_WEIGHTS_REVISION,
    Ballet,
    NumpyBallet,
    _jax_available,
    _quiet_hf_xet,
    _resolve_backend,
    download_weights,
)

# Both are needed for the jax backend: eloy's model imports flax lazily and
# only fails at call time, so flax alone or jax alone is not enough.
_HAVE_JAX = all(importlib.util.find_spec(mod) is not None for mod in ("jax", "flax"))
requires_jax = pytest.mark.skipif(not _HAVE_JAX, reason="jax/flax not installed")

# Backends that can actually run in this environment.
_BACKENDS = ["numpy", *(["jax"] if _HAVE_JAX else [])]

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
@requires_jax
def test_numpy_matches_jax_ballet(synthetic_cutouts):
    """NumpyBallet reproduces the jax/flax Ballet to float32 round-off."""
    # Deferred: eloy's Ballet needs flax, which the skipif above gates on.
    from eloy.ballet.model import Ballet  # noqa: PLC0415

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
        forward = model._forward(  # noqa: SLF001
            np.asarray(cutouts, dtype=np.float32)[..., None]
        )
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
        assert not any(issubclass(w.category, RuntimeWarning) for w in caught), [
            str(w.message) for w in caught
        ]

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_chunking_matches_single_pass(self, tmp_path, mocker, backend):
        """Crossing chunk boundaries changes nothing, ragged final chunk included."""
        # Run through the selector so the jax wrapper sees the ragged final
        # chunk too -- which also proves a second compiled shape is harmless.
        model = Ballet(model_file=_random_weights_npz(tmp_path), backend=backend)
        cutouts, _ = _make_synthetic_cutouts()  # 16 cutouts, one chunk today

        expected = model.centroid(cutouts)
        mocker.patch("bandaid.ballet._CHUNK", 5)  # 16 -> chunks of 5,5,5,1

        # Round-off tolerance, not exact equality: einsum/BLAS reduction order
        # varies with batch size. A slicing or concatenate-ordering bug would
        # miss by the magnitude of the outputs, far beyond this atol.
        np.testing.assert_allclose(model.centroid(cutouts), expected, atol=1e-5, rtol=0)

    def test_max_pool_rejects_non_square_input(self):
        """The pool's odd-size padding assumes H == W; anything else raises."""
        with pytest.raises(ValueError, match="square input required"):
            ballet._max_pool_2x2_same(  # noqa: SLF001
                np.zeros((1, 4, 6, 1), dtype=np.float32)
            )

    def test_default_download_when_no_model_file(self, tmp_path, mocker):
        """With no model_file, the weights come from ``download_weights``."""
        npz = _random_weights_npz(tmp_path)
        download_weights_mock = mocker.patch(
            "bandaid.ballet.download_weights", return_value=npz
        )

        model = NumpyBallet()

        download_weights_mock.assert_called_once()
        expected = np.load(npz)
        np.testing.assert_array_equal(
            model.params["Conv_0"]["kernel"], expected["Conv_0_kernel"]
        )

    def test_download_weights_targets_ballet_repo(self, monkeypatch, mocker):
        """``download_weights`` asks the hub for the pinned Ballet npz, xet off."""
        # monkeypatch is for the env var (see below); mocker for the hub call.
        # setenv first so monkeypatch records a restore even when the var was
        # unset; a bare delenv(raising=False) of a missing var restores nothing
        # and the setdefault inside download_weights would leak past teardown.
        # monkeypatch (not mocker) because it restores env vars reliably on
        # teardown; mocker.patch has no equivalent for os.environ.
        monkeypatch.setenv("HF_HUB_DISABLE_XET", "placeholder")
        monkeypatch.delenv("HF_HUB_DISABLE_XET")

        # download_weights does `from huggingface_hub import hf_hub_download`
        # inside the function body on every call, so the name is looked up on
        # the huggingface_hub module itself, not on bandaid.ballet.
        hf_hub_download = mocker.patch(
            "huggingface_hub.hf_hub_download", return_value="/cached/weights.npz"
        )

        assert download_weights() == "/cached/weights.npz"
        hf_hub_download.assert_called_once_with(
            repo_id=_BALLET_HF_REPO_ID,
            filename=_BALLET_WEIGHTS_FILENAME,
            revision=_BALLET_WEIGHTS_REVISION,
        )
        assert os.environ["HF_HUB_DISABLE_XET"] == "1"


class TestQuietHfXet:
    """Unit tests for the best-effort ``_quiet_hf_xet`` HF-warning silencer."""

    def test_sets_disable_xet_when_unset(self, monkeypatch):
        """With no user setting, xet is disabled to avoid its stderr warning."""
        # setenv-then-delenv so teardown restores the unset state (see
        # test_download_weights_targets_ballet_repo). monkeypatch, not mocker,
        # because it restores env vars reliably; mocker has no equivalent.
        monkeypatch.setenv("HF_HUB_DISABLE_XET", "placeholder")
        monkeypatch.delenv("HF_HUB_DISABLE_XET")
        _quiet_hf_xet()
        assert os.environ["HF_HUB_DISABLE_XET"] == "1"

    def test_preserves_user_value(self, monkeypatch):
        """A user who set the var (e.g. to keep xet) is never overridden."""
        # monkeypatch (not mocker) restores env vars reliably on teardown.
        monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
        _quiet_hf_xet()
        assert os.environ["HF_HUB_DISABLE_XET"] == "0"


class TestBackendSelection:
    """Backend resolution and construction (offline, jax not required)."""

    @pytest.fixture(autouse=True)
    def _ignore_developer_env(self, monkeypatch):
        """Ignore a ``BANDAID_BALLET_BACKEND`` set in the developer's shell."""
        # monkeypatch, not mocker: it restores the env var reliably on teardown.
        monkeypatch.delenv(_BACKEND_ENV, raising=False)

    def test_numpy_backend_uses_the_numpy_engine(self, tmp_path):
        """``backend="numpy"`` reports itself and holds a ``NumpyBallet``."""
        model = Ballet(model_file=_random_weights_npz(tmp_path), backend="numpy")
        assert model.backend == "numpy"
        assert isinstance(model._engine, NumpyBallet)  # noqa: SLF001

    @pytest.mark.parametrize(
        ("installed", "expected"),
        [
            (("jax", "flax"), "jax"),
            # jax alone is not enough: eloy's model.py swallows the flax
            # ImportError and substitutes a stub that fails at *call* time, so
            # a missing flax would otherwise surface as a mid-run crash.
            (("jax",), "numpy"),
            (("flax",), "numpy"),
            ((), "numpy"),
        ],
    )
    def test_auto_requires_both_jax_and_flax(self, mocker, installed, expected):
        """Auto picks jax only when both jax and flax import-resolve."""
        real_find_spec = importlib.util.find_spec
        # bandaid.ballet calls importlib.util.find_spec via its own `import
        # importlib.util`, so the name is looked up on the shared stdlib
        # module, not on bandaid.ballet.
        mocker.patch(
            "importlib.util.find_spec",
            side_effect=lambda name: (
                (name in installed) or None
                if name in ("jax", "flax")
                else real_find_spec(name)
            ),
        )

        assert _jax_available() is (expected == "jax")
        assert _resolve_backend("auto")[0] == expected

    def test_explicit_jax_without_jax_raises(self, mocker, tmp_path):
        """``backend="jax"`` is strict: it never degrades to numpy silently."""
        mocker.patch("bandaid.ballet._jax_available", return_value=False)

        with pytest.raises(ImportError, match=r"bandaid\[jax\]"):
            _resolve_backend("jax")
        # Construction fails before any weights are fetched or loaded.
        with pytest.raises(ImportError, match=r"bandaid\[jax\]"):
            Ballet(model_file=_random_weights_npz(tmp_path), backend="jax")

    def test_env_var_is_honoured_under_auto(self, monkeypatch, mocker):
        """The env override wins over "auto", and says so in the reason."""
        mocker.patch("bandaid.ballet._jax_available", return_value=True)
        # monkeypatch: env var, restored reliably on teardown.
        monkeypatch.setenv(_BACKEND_ENV, "numpy")

        backend, reason = _resolve_backend("auto")

        assert backend == "numpy"
        assert _BACKEND_ENV in reason

    def test_explicit_backend_beats_env_var(self, monkeypatch, mocker):
        """An explicit ``backend=`` argument outranks the env override."""
        mocker.patch("bandaid.ballet._jax_available", return_value=True)
        # monkeypatch: env var, restored reliably on teardown.
        monkeypatch.setenv(_BACKEND_ENV, "jax")

        assert _resolve_backend("numpy")[0] == "numpy"

    def test_env_var_jax_without_jax_raises(self, monkeypatch, mocker):
        """Asking for jax via the env var is as strict as asking in code."""
        mocker.patch("bandaid.ballet._jax_available", return_value=False)
        # monkeypatch: env var, restored reliably on teardown.
        monkeypatch.setenv(_BACKEND_ENV, "jax")

        with pytest.raises(ImportError, match=r"bandaid\[jax\]"):
            _resolve_backend("auto")

    @pytest.mark.parametrize("bad", ["torch", "JAX", ""])
    def test_unknown_backend_raises_value_error(self, bad):
        """An unrecognized name is a mistake, not a reason to guess."""
        with pytest.raises(ValueError, match="unknown Ballet backend"):
            _resolve_backend(bad)

    def test_unknown_env_value_raises_value_error(self, monkeypatch):
        """A typo in the env override is reported, naming the env var."""
        # monkeypatch: env var, restored reliably on teardown.
        monkeypatch.setenv(_BACKEND_ENV, "torch")
        with pytest.raises(ValueError, match=f"unknown Ballet backend.*{_BACKEND_ENV}"):
            _resolve_backend("auto")

    def test_empty_env_value_is_treated_as_unset(self, monkeypatch, mocker):
        """``BANDAID_BALLET_BACKEND=`` is how a shell clears the override."""
        mocker.patch("bandaid.ballet._jax_available", return_value=False)
        # monkeypatch: env var, restored reliably on teardown.
        monkeypatch.setenv(_BACKEND_ENV, "")

        assert _resolve_backend("auto") == ("numpy", "jax/flax not installed")

    def test_construction_logs_the_backend_once(self, tmp_path, caplog):
        """Exactly one INFO line records which backend a run is using."""
        with caplog.at_level(logging.INFO, logger="bandaid.ballet"):
            Ballet(model_file=_random_weights_npz(tmp_path), backend="numpy")

        messages = [
            record.getMessage()
            for record in caplog.records
            if record.name == "bandaid.ballet"
        ]
        assert len(messages) == 1
        assert "numpy" in messages[0]

    def test_default_weights_come_from_our_pinned_download(self, tmp_path, mocker):
        """With no ``model_file``, the selector uses bandaid's pinned download."""
        npz = _random_weights_npz(tmp_path)
        download_weights_mock = mocker.patch(
            "bandaid.ballet.download_weights", return_value=npz
        )

        model = Ballet(backend="numpy")

        download_weights_mock.assert_called_once()
        expected = np.load(npz)
        np.testing.assert_array_equal(
            model._engine.params["Conv_0"]["kernel"],  # noqa: SLF001
            expected["Conv_0_kernel"],
        )


@requires_jax
class TestJaxBackend:
    """The jax backend, exercised offline with random-init weights."""

    def test_jax_matches_numpy_random_weights(self, tmp_path, synthetic_cutouts):
        """Both backends agree on the same weights, no network needed."""
        weights = _random_weights_npz(tmp_path)
        cutouts, _ = synthetic_cutouts

        numpy_out = Ballet(model_file=weights, backend="numpy").centroid(cutouts)
        model = Ballet(model_file=weights, backend="jax")
        jax_out = model.centroid(cutouts)

        assert model.backend == "jax"
        assert isinstance(jax_out, np.ndarray)
        assert jax_out.dtype == np.float32
        # Both sides are CPU float32 here; a GPU-backed jax may use TF32
        # matmuls for float32 and need a looser tolerance than this.
        np.testing.assert_allclose(jax_out, numpy_out, atol=1e-4, rtol=0)

    def test_auto_selects_jax_when_installed(self, tmp_path, monkeypatch):
        """In an environment with jax and flax, "auto" really picks jax."""
        # monkeypatch: env var, restored reliably on teardown.
        monkeypatch.delenv(_BACKEND_ENV, raising=False)
        model = Ballet(model_file=_random_weights_npz(tmp_path))
        assert model.backend == "jax"

    def test_eloy_never_downloads_its_own_weights(self, tmp_path, mocker):
        """Eloy's unpinned ``download_weights`` must never be reached."""
        import eloy.ballet.model  # noqa: PLC0415

        npz = _random_weights_npz(tmp_path)
        mocker.patch("bandaid.ballet.download_weights", return_value=npz)

        def _fail_if_called():
            pytest.fail("eloy's unpinned download_weights was used")

        mocker.patch.object(
            eloy.ballet.model, "download_weights", side_effect=_fail_if_called
        )

        assert Ballet(backend="jax").backend == "jax"

    def test_empty_batch_returns_0x2_without_entering_jax(self, tmp_path):
        """The empty-batch short circuit is shared with the numpy engine."""
        model = Ballet(model_file=_random_weights_npz(tmp_path), backend="jax")

        def _boom(_chunk):
            msg = "jax must not be called for an empty batch"
            raise AssertionError(msg)

        model._chunk_fn = _boom  # noqa: SLF001
        out = model.centroid(np.empty((0, 15, 15)))

        assert out.shape == (0, 2)
        assert out.dtype == np.float32

    def test_constant_cutout_yields_nan_silently(self, tmp_path):
        """A flat cutout gives NaN on the jax path too, warning-free."""
        model = Ballet(model_file=_random_weights_npz(tmp_path), backend="jax")
        flat = np.full((1, 15, 15), 7.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = model.centroid(flat)

        assert np.isnan(out).all()
        assert not any(issubclass(w.category, RuntimeWarning) for w in caught), [
            str(w.message) for w in caught
        ]
