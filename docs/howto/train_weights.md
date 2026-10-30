# Train your own weights

Most users should keep the default Ballet weights. Retrain only if the stock
weights centroid your frames poorly, for example for a different sensor or noise
regime. The stock weights come from HuggingFace and are fetched on first use;
training never contacts it.

## Prerequisites

Install the `train` extra (JAX, Flax, Optax) in an environment where both
`import bandaid` and `import eloy` work:

```bash
pip install -e ".[train]"
```

## The two phases

Each phase is a module that reads its settings from environment variables and
writes a new `.npz`; nothing is overwritten in place.

```bash
WDIR=/path/to/weights

# Phase 1: realistic-noise training from random weights.
BALLET_OUT=$WDIR/ballet_realistic_15x15.npz \
    python -m bandaid.train_ballet_realistic

# Phase 2: bayer-aware fine-tune, warm-started from phase 1.
BALLET_WARM=$WDIR/ballet_realistic_15x15.npz \
BALLET_OUT=$WDIR/ballet_realistic_bayer_15x15.npz \
    python -m bandaid.train_ballet_bayer
```

The phase 2 file is the endpoint. Both use a 15x15 cutout and finish with a short
bias-adjustment pass that keeps the parameters with the smallest mean (x, y)
residual.

## Environment variables

| Variable            | Used by | Default                                                          |
| ------------------- | ------- | ---------------------------------------------------------------- |
| `BALLET_WARM`       | bayer   | `ballet_realistic_15x15.npz` (relative to the working directory) |
| `BALLET_OUT`        | both    | bayer: `ballet_realistic_bayer_15x15.npz`; realistic: unset      |
| `BALLET_EPOCHS`     | both    | realistic `300`, bayer `100`                                     |
| `BALLET_TRAIN_SIZE` | both    | `5000`                                                           |
| `BALLET_TEST_SIZE`  | both    | `5000`                                                           |
| `BALLET_BAYER_FRAC` | bayer   | `0.5` (fraction of samples carrying the bayer residual)          |

When `BALLET_OUT` is unset, the realistic phase writes a timestamped
`15x15_realistic_<timestamp>.npz` in the working directory. Shrink the sizes for a
CPU smoke test: `BALLET_EPOCHS=1 BALLET_TRAIN_SIZE=200 BALLET_TEST_SIZE=200`.

## Compare against the stock weights

`eval_realistic_weights.py` at the repository root runs a synthetic SNR sweep and
prints the median recovery error per SNR bin. The stock weights are always
included as the `old` baseline (the only step that downloads them):

```bash
python eval_realistic_weights.py \
    realistic=$WDIR/ballet_realistic_15x15.npz \
    bayer=$WDIR/ballet_realistic_bayer_15x15.npz
```

## Use the result

Pass the file with `--weights` on the command line or `weights=` to
`photometer_frames`; see [troubleshooting](troubleshooting.md#weights-offline-or-custom)
for the details.
