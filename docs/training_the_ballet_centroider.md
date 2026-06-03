# Training the Ballet centroider

bandaid centroids stars with **Ballet**, a small CNN that regresses the sub-pixel
`(x, y)` offset of a star within a cutout. The stock weights were trained on clean,
bright synthetic stars, which left two weaknesses on real smart-telescope frames: faint,
sky-dominated stars (noise-limited), and bright stars on a one-shot-colour sensor (limited
by the residual bayer checkerboard left after balancing). bandaid retrains the CNN in
two warm-started phases to address these.

The synthetic-data generator and the training utilities live in
`bandaid.ballet_training`; the two phases are runnable as modules under
`bandaid.train_ballet_*`. Evaluation is a standalone script, `eval_realistic_weights.py`.

## Prerequisites

The training scripts need JAX/Flax/Optax (declared as the optional `train` extra) and a
working `eloy` install — they build on eloy's base `Moffat2D`, `CNN`, and jitted
train/eval steps:

```bash
pip install -e ".[train]"
```

The scripts also import `bandaid` itself, which pulls in the photometry stack
(`st_pipeline`, etc.). Run them in an environment where both `import bandaid` and
`import eloy` succeed.

!!! note "Output paths default to the author's machine"
    `BALLET_WARM` / `BALLET_OUT` default to absolute paths under
    `/Users/.../astronomy/eloy/`. Always set them explicitly (as below) so each phase
    reads the previous phase's output and writes where you expect — the defaults do
    **not** chain (Phase 2's default output is a *timestamped* filename, which Phase 3's
    default `BALLET_WARM` does not point at).

All phases use a `15x15` cutout, batch size 100, and finish with a short bias-adjustment
pass that keeps the parameters with the smallest mean `(x, y)` residual.

## The two-phase sequence

Pick a directory for the weights and chain the phases by hand:

```bash
WDIR=/path/to/weights

# Phase 2 — realistic-noise retrain (from scratch).
# Sky pedestal + Poisson + read noise, peak SNR log-uniform over (1.5, 200), so the CNN
# finally sees faint, noise-dominated stars. No bayer augmentation here.
BALLET_OUT=$WDIR/ballet_realistic_15x15.npz \
  python -m bandaid.train_ballet_realistic

# Phase 3 — bayer-aware fine-tune (warm-start from Phase 2).
# A fraction of samples get a per-CFA-channel star+sky colour, then the real
# bandaid.bayer_balance_image is applied, teaching the CNN to ignore the residual
# checkerboard that limits bright-star centroids.
BALLET_WARM=$WDIR/ballet_realistic_15x15.npz \
BALLET_OUT=$WDIR/ballet_realistic_bayer_15x15.npz \
  python -m bandaid.train_ballet_bayer
```

Each phase reads the previous phase's `.npz` via `BALLET_WARM` and writes a **new** file
via `BALLET_OUT`; nothing is overwritten in place. Phase 3's weights are the endpoint.

!!! note "Why there is no bright-end phase"
    A third "bright-boost" phase (oversampling high-SNR stars to lower the bright-end
    recovery floor) was tried and dropped — it didn't improve the fit enough to justify
    keeping it. The brightest stars are better served by a direct Moffat fit, which is
    deferred and not yet implemented here.

### Starting from scratch (no HuggingFace)

There is nothing extra to configure for a fully from-scratch run — **the sequence above
already is one.** Phase 2 initializes random weights with `model.init(...)` and the
training scripts never call `download_weights()`, so training has **no HuggingFace
dependency**. Phase 3 warm-starts only from Phase 2's local `.npz` (`BALLET_WARM`), never
from a downloaded model.

In other words, "train from scratch" simply means *run Phase 2*. The published stock
weights are only a baseline to compare against (see [Evaluating weights](#evaluating-weights)),
never an input to this pipeline.

### Environment knobs

All sizes can be reduced for a quick CPU smoke test (e.g. `BALLET_EPOCHS=1
BALLET_TRAIN_SIZE=200 BALLET_TEST_SIZE=200`).

| Variable | Phases | Default | Meaning |
| --- | --- | --- | --- |
| `BALLET_WARM` | 3 | author path | `.npz` weights to warm-start from |
| `BALLET_OUT` | 2, 3 | author path (2: timestamped) | where to write the trained weights |
| `BALLET_EPOCHS` | all | 300 (P2), 100 (P3) | training epochs |
| `BALLET_TRAIN_SIZE` | all | 5000 | samples drawn per training refresh |
| `BALLET_TEST_SIZE` | all | 5000 | held-out evaluation samples |
| `BALLET_BAYER_FRAC` | 3 | 0.5 | fraction of samples carrying the bayer residual |

## Evaluating weights

`eval_realistic_weights.py` runs a purely synthetic ground-truth SNR sweep (no image, no
network state) and prints the median recovery error in pixels per SNR bin, for both plain
and bayer stars. The HuggingFace stock weights are always included as the `old` baseline;
pass any number of additional `label=path` (or bare `path`) arguments:

```bash
python eval_realistic_weights.py \
  realistic=$WDIR/ballet_realistic_15x15.npz \
  bayer=$WDIR/ballet_realistic_bayer_15x15.npz
```

Compare the rows: the retrain should lower the faint-SNR error (Phase 2) and the bayer
rows (Phase 3) without regressing the others.

!!! note "The one HuggingFace touch"
    The `old` baseline makes this script the *only* step in the workflow that reaches out
    to HuggingFace (it downloads the stock weights, cached after the first run). Training
    itself never does. A fully offline run therefore needs that cache already present, or
    a small tweak to drop the `old` entry from `eval_realistic_weights.py`.

## Using new weights in the pipeline

The photometry pipeline loads centroid weights through eloy's `Ballet` wrapper
(`eloy.centroid.Ballet`), which accepts a path to an `.npz` file. Point it at your trained
weights instead of the downloaded default to use the retrained centroider.
```
