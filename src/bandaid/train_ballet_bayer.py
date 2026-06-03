"""
Phase 3: warm-start bayer-aware fine-tune of the Ballet CNN.

Continues training from the Phase-2 realistic-noise weights with bayer augmentation
turned on (a fraction of samples get a per-CFA-channel star+sky color, then the real
`bandaid.bayer_balance_image` is run on the padded frame and the center is cropped), so
the CNN learns to ignore the residual checkerboard that limits bright-star centroids.

This script uses bandaid's `ballet_training` module (the realistic + bayer generator
plus the re-exported eloy training utilities). The jitted train_step / eval_step close
over a module-global `model`, so `bind_model` sets it before training. The plain half of
each batch keeps bright sharpness; the bayer half teaches checkerboard invariance.

Env knobs: BALLET_WARM, BALLET_OUT, BALLET_EPOCHS, BALLET_TRAIN_SIZE, BALLET_TEST_SIZE,
BALLET_BAYER_FRAC.
"""

import os
import time

import jax.numpy as jnp
import numpy as np
import optax
from eloy.ballet.model import CNN, load_weights_file

from bandaid import bayer_balance_image
from bandaid.ballet_training import (
    Moffat2D,
    TrainState,
    bind_model,
    eval_step,
    get_batches,
    params_to_flat_dict,
    train_step,
)

SIZE = 15
WARM = os.environ.get("BALLET_WARM", "ballet_realistic_15x15.npz")
OUT = os.environ.get("BALLET_OUT", "ballet_realistic_bayer_15x15.npz")
EPOCHS = int(os.environ.get("BALLET_EPOCHS", "100"))
TRAIN_SIZE = int(os.environ.get("BALLET_TRAIN_SIZE", "5000"))
TEST_SIZE = int(os.environ.get("BALLET_TEST_SIZE", "5000"))
BAYER_FRAC = float(os.environ.get("BALLET_BAYER_FRAC", "0.5"))
BATCH = 100

# Bind the module-global model that the jitted train_step/eval_step close over.
model = bind_model(CNN())
gen = Moffat2D(SIZE)


def gen_fn(n: int):
    """
    Generate a mixed bayer / plain training batch.

    Thin wrapper over the module-level ``gen.random_realistic_label`` that pins
    the bayer fraction (``BAYER_FRAC``) and injects the real
    ``bandaid.bayer_balance_image`` so a fraction of samples carry the residual
    CFA checkerboard while the rest stay plain (preserving bright and faint
    sharpness).

    Parameters
    ----------
    n : int
        Number of cutouts to generate.

    Returns
    -------
    tuple of numpy.ndarray
        ``(images, labels)`` where ``images`` has shape ``(n, SIZE, SIZE, 1)``
        and ``labels`` has shape ``(n, 2)`` holding the ``(x0, y0)`` centroid
        offsets about the cutout center.
    """
    # Half bayer (checkerboard invariance), half plain (keep bright + faint sharpness).
    return gen.random_realistic_label(
        n,
        bayer_frac=BAYER_FRAC,
        balance_fn=bayer_balance_image,
    )


def main():
    """
    Run the bayer-aware warm-start fine-tune and save the result.

    Loads the Phase-2 realistic-noise weights from ``WARM``, then fine-tunes for
    ``EPOCHS`` epochs on freshly drawn ``gen_fn`` batches using AdamW with a
    single learning-rate drop at 60% of training. After the main loop it runs a
    short bias-adjustment phase, keeping the parameters with the smallest mean
    ``(x, y)`` residual, and writes them to ``OUT`` as a flat ``.npz``.

    Behavior is controlled by the module-level constants read from the
    ``BALLET_*`` environment variables (``WARM``, ``OUT``, ``EPOCHS``,
    ``TRAIN_SIZE``, ``TEST_SIZE``, ``BAYER_FRAC``). Progress and the final bias
    metric are printed to stdout.
    """
    print(f"warm-start from {WARM}")
    params = load_weights_file(WARM)

    X_test, y_test = gen_fn(TEST_SIZE)
    test_batch = (jnp.array(X_test), jnp.array(y_test))

    lr = 1e-4
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adamw(lr),
    )
    X_train, y_train = gen_fn(TRAIN_SIZE)
    lr_drops = {EPOCHS * 3 // 5: 1e-5}  # one drop at 60% through

    print(
        f"fine-tune {EPOCHS} epochs, bayer_frac={BAYER_FRAC}, "
        f"LR {lr:.0e} -> {list(lr_drops.values())}",
    )
    t0 = time.time()
    for epoch in range(EPOCHS):
        for batch in get_batches(X_train, y_train, BATCH):
            state, loss = train_step(state, batch)
        if epoch in lr_drops:
            lr = lr_drops[epoch]
            state = TrainState.create(
                apply_fn=model.apply,
                params=state.params,
                tx=optax.adamw(lr),
            )
        if epoch % 10 == 0:
            rmse = eval_step(state.params, test_batch)
            print(
                f"epoch {epoch}: loss={float(loss):.4f} test_rmse={float(rmse):.4f} "
                f"lr={lr:.0e} elapsed={time.time() - t0:.0f}s",
            )
            X_train, y_train = gen_fn(TRAIN_SIZE)  # fresh draw for variety

    # Bias-adjust: a few more passes, keep the params with smallest mean (x,y) residual.
    print("bias adjust")
    xa, ya = gen_fn(min(4 * TRAIN_SIZE, 20000))
    best, best_dev = state.params, np.inf
    for i in range(10):
        for batch in get_batches(xa, ya, BATCH):
            state, loss = train_step(state, batch)
        preds = np.asarray(model.apply({"params": state.params}, xa))
        dev = float(np.max(np.abs(np.mean(preds - ya, axis=0))))
        print(f"  {i}: max|mean (x,y) residual| = {dev:.4f}")
        if dev < best_dev:
            best_dev, best = dev, state.params

    np.savez(OUT, **params_to_flat_dict(best))
    print(f"saved {OUT}  (bias max|mean residual| = {best_dev:.4f})")


if __name__ == "__main__":
    main()
