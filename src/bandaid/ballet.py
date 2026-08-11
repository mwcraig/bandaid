"""
Ballet centroid CNN inference, with or without jax.

`Ballet` is the class the pipeline builds: it runs eloy's jax/flax model when
jax and flax are installed (`pip install bandaid[jax]`) and the pure-numpy
engine below otherwise. Both paths use the same pinned weights and agree to
float32 round-off, so the backend is a speed choice, not a science choice.

`NumpyBallet` is a drop-in replacement for the inference half of eloy's
jax/flax ``Ballet``: the same pretrained weights (a small ``.npz`` downloaded
once from HuggingFace), the same 15x15-cutout -> (x, y) subpixel-centroid
contract, but implemented with numpy only, so jax and flax are not needed at
runtime. The forward pass mirrors the flax model layer for layer -- three SAME
convolutions with relu, two 2x2 SAME max-pools, and three dense layers -- and
reproduces flax's exact padding and flattening conventions so the outputs
match the jax model to float32 round-off.

Vocabulary, for readers not steeped in TensorFlow/flax conventions: "SAME" is
the padding mode in which the input borders are padded just enough that the
output spatial size equals the input size divided by the stride (as opposed
to "VALID": no padding, the output shrinks by kernel-1); NHWC is the axis
order of a 4-d image batch -- batch (N), height, width, channels -- flax's
native convolution layout, which this module keeps throughout.
"""

import importlib.util
import logging
import os

import numpy as np
from eloy.ballet.model import load_weights_file
from scipy.special import expit

__all__ = [
    "Ballet",
    "NumpyBallet",
    "download_weights",
]

logger = logging.getLogger(__name__)

# Env override for the "auto" backend choice, mainly for A/B benchmarking and
# for users who have jax installed but do not want it used.
_BACKEND_ENV = "BANDAID_BALLET_BACKEND"

# HuggingFace location of the pretrained Ballet weights (public, no auth).
# The revision pins the exact weights blob the golden values in
# test_ballet.py were captured against, so an upstream re-upload can
# neither silently change production centroids nor break that test.
_BALLET_HF_REPO_ID = "lgrcia/ballet"
_BALLET_WEIGHTS_FILENAME = "centroid_15x15.npz"
_BALLET_WEIGHTS_REVISION = "cfebd20240ce3fb694f6403a244f37f971e7780b"

# Cutouts per forward-pass chunk. The convolution layers materialize
# (chunk, H, W, 256) float32 intermediates, so chunking keeps peak memory flat
# for arbitrarily large batches at no measurable speed cost.
_CHUNK = 256


def _quiet_hf_xet():
    """
    Best-effort: silence the native ``hf_xet`` unauthenticated-request warning.

    On the first weights download, ``hf_hub_download`` routes through the native
    ``hf_xet`` accelerator, which prints a "sending unauthenticated requests to
    the HF Hub ... faster downloads" line straight to stderr -- not a Python
    warning or log record, so it cannot be filtered the usual way. Disabling xet
    keeps the download working (the ``.npz`` is tiny and cached once) and avoids
    that line. ``setdefault`` so a user who set ``HF_HUB_DISABLE_XET`` (or who
    wants xet) is never overridden; setting ``HF_TOKEN`` is the fully-correct fix
    -- it both keeps xet acceleration and silences the warning. Best-effort
    because the exact behaviour depends on the installed ``hf_xet`` version.
    """
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def download_weights():
    """
    Download the pretrained Ballet weights from HuggingFace Hub.

    Returns
    -------
    str
        Local path of the cached ``.npz`` weights file.
    """
    _quiet_hf_xet()

    # Lazy import: only the no-model_file path needs the hub client, and this
    # keeps module import (and offline use with a local file) network-free.
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    return hf_hub_download(
        repo_id=_BALLET_HF_REPO_ID,
        filename=_BALLET_WEIGHTS_FILENAME,
        revision=_BALLET_WEIGHTS_REVISION,
    )


def _conv2d_same(x, kernel, bias):
    """
    3x3 stride-1 SAME convolution matching ``flax.linen.Conv``.

    Parameters
    ----------
    x : numpy.ndarray
        Input batch, NHWC ``(N, H, W, C_in)`` float32.
    kernel : numpy.ndarray
        Weights in flax HWIO layout ``(3, 3, C_in, C_out)``.
    bias : numpy.ndarray
        Per-output-channel bias, ``(C_out,)``.

    Returns
    -------
    numpy.ndarray
        Convolved batch ``(N, H, W, C_out)`` float32.
    """
    # SAME for a 3x3 stride-1 window is a symmetric one-pixel zero pad.
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    # (N, H, W, C_in, 3, 3) view of every 3x3 patch; einsum contracts the
    # patch and input-channel axes against the HWIO kernel.
    w = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(1, 2))
    return np.einsum("nhwcij,ijco->nhwo", w, kernel, optimize=True) + bias


def _max_pool_2x2_same(x):
    """
    2x2 stride-2 max-pool with flax ``padding="SAME"`` semantics.

    For an odd input size flax pads only at the high (bottom/right) edge, with
    ``-inf`` so the pad never wins the max: 15 -> 8, then 8 -> 4 unpadded.

    Parameters
    ----------
    x : numpy.ndarray
        Input batch, NHWC ``(N, H, W, C)``; H and W must be equal.

    Returns
    -------
    numpy.ndarray
        Pooled batch ``(N, ceil(H/2), ceil(W/2), C)``.

    Raises
    ------
    ValueError
        If H and W differ -- the pad-both-edges-on-odd-H logic below is only
        correct for square inputs.
    """
    n, h, w, c = x.shape
    if h != w:
        msg = f"square input required, got {h}x{w}"
        raise ValueError(msg)
    if h % 2:
        x = np.pad(
            x,
            ((0, 0), (0, 1), (0, 1), (0, 0)),
            constant_values=-np.inf,
        )
        h, w = h + 1, w + 1
    return x.reshape(n, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))


def _jax_available():
    """
    Report whether the jax backend can actually run.

    Both jax and flax must import-resolve. ``eloy/ballet/model.py`` wraps its
    flax import in ``try/except ImportError: pass`` and substitutes a stub
    ``nn.Module`` that raises ``NotImplementedError`` at *call* time, so a
    successful ``import eloy.ballet.model`` proves nothing; probe both
    explicitly instead, or a missing flax surfaces as a mid-run crash.

    Returns
    -------
    bool
        True only if both ``jax`` and ``flax`` are importable.
    """
    return all(importlib.util.find_spec(mod) is not None for mod in ("jax", "flax"))


def _resolve_backend(backend):
    """
    Turn a requested backend into the one that will be used, and why.

    Parameters
    ----------
    backend : str
        One of ``"auto"``, ``"jax"`` or ``"numpy"``. ``"auto"`` honours the
        ``BANDAID_BALLET_BACKEND`` environment variable, then falls back to
        whichever backend is installed. An explicit ``"jax"``/``"numpy"``
        outranks the environment variable.

    Returns
    -------
    name : str
        ``"jax"`` or ``"numpy"``.
    reason : str
        Short human-readable explanation of the choice, for the log line.

    Raises
    ------
    ValueError
        If ``backend`` (or the environment variable) is not a known name.
    ImportError
        If jax was asked for explicitly but jax/flax are not installed. Asking
        for it and silently getting the slow path would hide the problem.
    """
    reason = "requested"
    if backend == "auto":
        from_env = os.environ.get(_BACKEND_ENV)
        if from_env:
            backend, reason = from_env, _BACKEND_ENV

    if backend not in ("auto", "jax", "numpy"):
        source = f" (from {reason})" if reason == _BACKEND_ENV else ""
        msg = (
            f"unknown Ballet backend {backend!r}{source}; "
            'expected "auto", "jax" or "numpy"'
        )
        raise ValueError(msg)

    available = _jax_available()
    if backend == "auto":
        return (
            ("jax", "eloy/flax") if available else ("numpy", "jax/flax not installed")
        )
    if backend == "jax" and not available:
        source = f" via {reason}" if reason == _BACKEND_ENV else ""
        msg = (
            f"the jax Ballet backend was requested{source} but jax and flax are "
            "not both installed -- `pip install bandaid[jax]`, or use "
            'backend="numpy"'
        )
        raise ImportError(msg)
    return backend, reason


def _chunked_centroid(chunk_fn, x):
    """
    Apply a per-chunk forward pass over a batch of cutouts.

    Shared by both backends so the empty-batch, dtype and chunk-boundary
    behaviour is identical no matter which engine is underneath. The
    ``np.asarray`` is also what pulls a jax device array back to the host for
    the downstream ``cutout.to_original_position``.

    Parameters
    ----------
    chunk_fn : callable
        Maps one ``(n, 15, 15)`` float32 chunk to ``(n, 2)`` centroids.
    x : numpy.ndarray
        Cutouts of shape ``(N, 15, 15)``, any float dtype.

    Returns
    -------
    numpy.ndarray
        Centroids of shape ``(N, 2)`` float32.
    """
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(
        [np.asarray(chunk_fn(x[i : i + _CHUNK])) for i in range(0, len(x), _CHUNK)]
    ).astype(np.float32, copy=False)


class Ballet:
    """
    Ballet centroider using jax when available, numpy otherwise.

    Attributes
    ----------
    backend : str
        The backend actually in use, ``"jax"`` or ``"numpy"``.
    """

    def __init__(self, model_file=None, backend="auto") -> None:
        """
        Select a backend and load the CNN weights into it.

        Parameters
        ----------
        model_file : str or Path, optional
            Path to the ``.npz`` weights file. If None, the pretrained weights
            are downloaded from HuggingFace (cached after the first call).
        backend : str, optional
            ``"auto"`` (default), ``"jax"`` or ``"numpy"``. See
            `_resolve_backend`: ``"auto"`` honours ``BANDAID_BALLET_BACKEND``
            and then falls back to numpy, while ``"jax"`` is strict and raises
            rather than silently degrading.
        """
        backend, reason = _resolve_backend(backend)
        # Always pass the weights explicitly: eloy's own download_weights()
        # takes no `revision`, so letting it download would bypass the pin
        # that makes the golden centroid values meaningful.
        model_file = model_file or download_weights()

        if backend == "jax":
            # Deferred: only this branch needs jax, and _resolve_backend has
            # already established that it is installed.
            import jax  # noqa: PLC0415
            from eloy.ballet.model import Ballet as _EloyBallet  # noqa: PLC0415

            engine = _EloyBallet(model_file=model_file)
            # A run sees at most a couple of distinct batch shapes (the Gaia
            # catalog is fixed, chunking bounds the rest), so jit compiles
            # about twice per run rather than per call.
            chunk_fn = jax.jit(engine.centroid)
        else:
            engine = NumpyBallet(model_file=model_file)
            chunk_fn = lambda c: engine._forward(c[..., None])[:, ::-1]  # noqa: E731, SLF001

        self.backend = backend
        self._engine = engine
        self._chunk_fn = chunk_fn
        logger.info("Ballet centroiding backend: %s (%s)", backend, reason)

    def centroid(self, x):
        """
        Predict subpixel centroids for a batch of cutouts.

        Parameters
        ----------
        x : numpy.ndarray
            Cutouts of shape ``(N, 15, 15)``; any float dtype (cast to
            float32, matching the model's precision).

        Returns
        -------
        numpy.ndarray
            Centroids of shape ``(N, 2)`` float32, ordered ``(x, y)``.
        """
        return _chunked_centroid(self._chunk_fn, x)


class NumpyBallet:
    """
    Numpy-only Ballet centroid model, output-identical to the jax original.

    Attributes
    ----------
    params : dict
        Per-layer ``{"kernel": ..., "bias": ...}`` arrays keyed by the flax
        layer names (``Conv_0`` .. ``Dense_2``).
    """

    def __init__(self, model_file=None) -> None:
        """
        Load the CNN weights.

        Parameters
        ----------
        model_file : str or Path, optional
            Path to the ``.npz`` weights file. If None, the pretrained weights
            are downloaded from HuggingFace (cached after the first call).
        """
        model_file = model_file or download_weights()
        self.params = load_weights_file(model_file)

    def centroid(self, x):
        """
        Predict subpixel centroids for a batch of cutouts.

        Parameters
        ----------
        x : numpy.ndarray
            Cutouts of shape ``(N, 15, 15)``; any float dtype (cast to
            float32, matching the jax model's precision).

        Returns
        -------
        numpy.ndarray
            Centroids of shape ``(N, 2)`` float32, ordered ``(x, y)``. The
            network emits (y, x); the flip here matches eloy's
            ``Ballet.centroid``, which downstream
            ``cutout.to_original_position`` depends on.
        """
        return _chunked_centroid(lambda c: self._forward(c[..., None])[:, ::-1], x)

    def _forward(self, x):
        """
        Run the CNN forward pass on one NHWC float32 chunk.

        Parameters
        ----------
        x : numpy.ndarray
            Chunk of shape ``(n, 15, 15, 1)`` float32.

        Returns
        -------
        numpy.ndarray
            Raw network output ``(n, 2)`` float32, ordered (y, x).
        """
        p = self.params
        # Per-sample min-max normalization. A constant cutout normalizes to
        # 0/0 -> NaN; jax produces the NaN silently, so suppress numpy's
        # RuntimeWarning on exactly this division to match (the NaNs then
        # propagate through the layers below warning-free on their own).
        x = x - x.min(axis=(1, 2, 3), keepdims=True)
        with np.errstate(invalid="ignore"):
            x = x / x.max(axis=(1, 2, 3), keepdims=True)

        for name in ("Conv_0", "Conv_1", "Conv_2"):
            x = _conv2d_same(x, p[name]["kernel"], p[name]["bias"])
            x = np.maximum(x, 0.0)
            if name != "Conv_2":
                x = _max_pool_2x2_same(x)  # 15 -> 8, then 8 -> 4

        # Row-major NHWC flatten: matches flax's reshape((batch, -1)) and
        # therefore the (4096, 2048) layout of the Dense_0 kernel.
        x = x.reshape(len(x), -1)
        x = expit(x @ p["Dense_0"]["kernel"] + p["Dense_0"]["bias"])
        x = expit(x @ p["Dense_1"]["kernel"] + p["Dense_1"]["bias"])
        return x @ p["Dense_2"]["kernel"] + p["Dense_2"]["bias"]
