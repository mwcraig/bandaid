# Getting started

## Install

bandaid needs Python ≥ 3.12. Install it into a recent virtual environment:

```bash
$ pip install bandaid
```

or, with uv, `uv add bandaid`. Check the install:

```pycon
>>> import bandaid
>>> bandaid.available_instruments()
['Seestar50']
```

### Optional jax acceleration

The base install is numpy-only so that bandaid also runs where jax is
unavailable, such as the browser (Pyodide). On a regular machine the `jax` extra
makes centroiding faster on CPU with results that agree to float32 round-off:

```bash
$ pip install "bandaid[jax]"
```

bandaid uses jax automatically once jax and flax are importable. Set
`BANDAID_BALLET_BACKEND=numpy` to force the numpy path, or
`BANDAID_BALLET_BACKEND=jax` to make a missing jax an error instead of a silent
fallback. The backend chosen is logged once at INFO.

### Ballet weights

The Ballet centroider needs trained weights. The first photometry run downloads
the default weights from HuggingFace; the hub caches them, so later runs need no
network. `bandaid weights` pre-fetches them and prints the cached path. For
offline use see [Troubleshooting](howto/troubleshooting.md).

## Run it

Point `bandaid process` at a folder of frames and give it an output directory:

```bash
$ bandaid process night-of-2026-06-27/ -o out/ -v
```

`-v` prints one line per frame; `-vv` adds per-stage detail. The first frame takes
longest because it carries the once-per-batch preparation.

## What lands in `out/`

```text
out/
├── frame_0001.star
├── frame_0002.star
│   ...
└── qa_manifest.csv
```

Each frame that processes cleanly gets one `.star` file holding a star list per
Bayer filter — `TR`, `TG`, `TB` and a luminance channel `L4` (omit it with
`--no-build-l4`). `qa_manifest.csv` has one row per input frame; mostly
`status=ok` with steady `fwhm` and `sky_median` means a healthy night. Every
field and column is described in [Output files](reference/outputs.md).

## The same run from Python

```python
from bandaid import photometer_frames

frames, results = photometer_frames(
    ["night-of-2026-06-27/"],
    output_dir="out/",
)
# frames  -> the expanded, sorted list of input frames
# results -> {input frame: written .star path} for each frame that succeeded
```

`photometer_frames` does the file expansion, `prepare_batch` and `process_batch`
that the CLI runs. Pass `config=` to tune the run or `weights=` (for example
`weights="path/to/centroid_15x15.npz"`) to use your own Ballet weights.

## Next steps

- [Tune a run](howto/tune_a_run.md) — apertures, Gaia magnitude limit, centroid policy.
- [Add a telescope](howto/add_a_telescope.md) — use something other than the
    Seestar50.
- [Troubleshooting](howto/troubleshooting.md) — skipped and failed frames.
- [Command line](reference/command_line.md) — every option of `bandaid`.
