"""
Phase 2: realistic-noise retrain of the Ballet CNN from scratch.

Trains on the ``random_realistic_label`` generator (sky pedestal + Poisson + read noise,
wide log-uniform SNR) so the CNN sees faint, noise-dominated stars the clean generator
never reaches. This produces the warm-start weights that the Phase-3 bayer fine-tune
(`train_ballet_bayer.py`) continues from.

This uses bandaid's `ballet_training` module: the realistic generator plus the
re-exported eloy training utilities. The jitted train_step / eval_step close over a
module-global `model`, so `bind_model` sets it before training. Bayer augmentation is
off here (the bright + faint sharpness baseline); the bayer half is added in Phase 3.

Env knobs: BALLET_EPOCHS, BALLET_TRAIN_SIZE, BALLET_TEST_SIZE, BALLET_OUT.
"""

import os
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import optax
from eloy.ballet.model import CNN

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
EPOCHS = int(os.environ.get("BALLET_EPOCHS", "300"))
TRAIN_SIZE = int(os.environ.get("BALLET_TRAIN_SIZE", "5000"))
TEST_SIZE = int(os.environ.get("BALLET_TEST_SIZE", "5000"))
OUT = os.environ.get("BALLET_OUT", None)
BATCH = 100
LR_MILESTONES = (EPOCHS // 3, EPOCHS // 2)  # scale schedule to total epochs

# Bind the module-global model that the jitted train_step/eval_step close over.
model = bind_model(CNN())
gen = Moffat2D(SIZE)


def gen_fn(n):
    """Draw a realistic-noise (no bayer) training batch of ``n`` cutouts."""
    return gen.random_realistic_label(n)


def main():
    """
    Train the CNN from scratch on the realistic-noise generator and save the result.

    Initializes fresh params, trains for ``EPOCHS`` with AdamW and a two-step LR drop at
    ``LR_MILESTONES``, redrawing training data each checkpoint for noise variety. After
    the main loop a short bias-adjustment phase keeps the params with the smallest mean
    ``(x, y)`` residual, which are written to ``OUT`` (or a timestamped file) as a flat
    ``.npz``.
    """
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones([1, SIZE, SIZE, 1]))["params"]

    print("Generate test/train samples")
    X_train, y_train = gen_fn(TRAIN_SIZE)
    X_test, y_test = gen_fn(TEST_SIZE)

    learning_rate = 1e-3
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adamw(learning_rate),
    )

    print(
        f"Start model training ({EPOCHS} epochs, LR drops at {LR_MILESTONES})",
    )
    t0 = time.time()
    for epoch in range(EPOCHS):
        for batch in get_batches(X_train, y_train, BATCH):
            state, loss = train_step(state, batch)
        # Drop LR at the scheduled milestones; recreate optimizer only then so Adam
        # momentum persists between regular epochs.
        if epoch == LR_MILESTONES[0]:
            learning_rate = 1e-4
            state = TrainState.create(
                apply_fn=model.apply, params=state.params, tx=optax.adamw(learning_rate),
            )
        elif epoch == LR_MILESTONES[1]:
            learning_rate = 1e-5
            state = TrainState.create(
                apply_fn=model.apply, params=state.params, tx=optax.adamw(learning_rate),
            )
        if epoch % 10 == 0:
            test_rmse = eval_step(state.params, (jnp.array(X_test), jnp.array(y_test)))
            dt = time.time() - t0
            print(
                f"Epoch {epoch}: Loss = {loss:.4f}, Test RMSE = {test_rmse:.4f}, "
                f"LR = {learning_rate:.1e}, elapsed = {dt:.0f}s",
            )
            # Fresh draw of training data each checkpoint for noise variety.
            X_train, y_train = gen_fn(TRAIN_SIZE)

    print("Adjusting model")
    X_train, y_train = gen_fn(min(4 * TRAIN_SIZE, 20000))
    adjust_params = []
    adjust_mean = []
    for i in range(10):
        for batch in get_batches(X_train, y_train, BATCH):
            state, loss = train_step(state, batch)

        predictions = model.apply({"params": state.params}, X_train)
        adjust_params.append(np.mean(predictions - y_train, 0))
        adjust_mean.append(state.params)
        print(f"{i} - (x,y) = {adjust_params[i]}")

    j = np.argmin([np.max(np.abs(d)) for d in adjust_params])
    final_model = adjust_mean[j]
    print(f"Best model: {j} - (x,y) = {adjust_params[j]}")

    print("Saving model file")
    file_name = OUT or f"{SIZE}x{SIZE}_realistic_{datetime.now().isoformat()}.npz"
    np.savez(file_name, **params_to_flat_dict(final_model))
    print(f"Model saved to {file_name}")


if __name__ == "__main__":
    main()
