# Getting started

## Install

bandaid needs Python ≥ 3.12. Install it into a recent virtual environment:

```bash
$ pip install git+https://github.com/mwcraig/bandaid
```

or, with uv, `uv add git+https://github.com/mwcraig/bandaid`. Check the install:

```pycon
>>> import bandaid
>>> bandaid.available_instruments()
['Seestar50']
```

### Optional jax acceleration

The base install is numpy-only, so bandaid also runs in the browser (Pyodide). On
a regular machine `pip install "bandaid[jax] @ git+https://github.com/mwcraig/bandaid"` makes centroiding faster, and
bandaid then uses jax automatically. Set `BANDAID_BALLET_BACKEND=numpy` to force
numpy, or `=jax` to make a missing jax an error.

### Ballet weights

The Ballet centroider needs trained weights. The first photometry run downloads
the default weights from HuggingFace; the hub caches them, so later runs need no
network. `bandaid weights` pre-fetches them and prints the cached path. For
offline use see [Troubleshooting](howto/troubleshooting.md).

## Observer and site

The `.star` format records the observer and the site elevation. Seestar headers
carry the site latitude and longitude but not your AAVSO observer code or your
elevation, so you supply both in a JSON file, say `site.json`:

```json
{"observer": "ABCD", "site_elev": 300.0}
```

`observer` is your AAVSO observer code and `site_elev` is the elevation of your
site in metres above mean sea level. Any key in the file overrides the value
read from the headers, so `site_lat` and `site_lon` (decimal degrees) can go
there too if the telescope's own position is wrong.

## Run it

Point `bandaid process` at a folder of frames, give it an output directory, and
pass the metadata file:

```bash
$ bandaid process night-of-2026-06-27/ -o out/ --user-metadata site.json -v
```

Without `--user-metadata` the run stops before any frame is processed with a
`FrameMetadataError` naming `observer` and `site_elev`.

`-v` prints one line per frame; `-vv` adds per-stage detail. The first usable
frame takes longest because it carries the once-per-batch preparation.

## What lands in `out/`

One `.star` file per frame, holding a star list each for `TR`, `TG`, `TB` and the
luminance channel `L4` (omitted with `--no-build-l4`), plus `qa_manifest.csv` with
one row per input frame. Mostly `status=ok` with steady `fwhm` and `sky_median`
means a healthy night. Every field and column is described in
[Output files](reference/outputs.md).

## The same run from Python

```python
from bandaid import photometer_frames

frames, results = photometer_frames(
    ["night-of-2026-06-27/"],
    output_dir="out/",
    user_specific_metadata={"observer": "ABCD", "site_elev": 300.0},
)
# frames  -> the expanded, sorted list of input frames
# results -> {input frame: written .star path} for each frame that succeeded
```

`photometer_frames` does the file expansion, `prepare_batch` and `process_batch`
that the CLI runs; `user_specific_metadata` is the contents of `site.json`. Pass
`config=` to tune the run or `weights=` (for example
`weights="path/to/centroid_15x15.npz"`) to use your own Ballet weights.
