# Welcome to bandaid

bandaid turns a folder of Smart Telescope (Seestar) FITS frames into AAVSO-style
photometry — `.star` files you can submit or analyze — without writing any code.
It is a deliberately temporary solution: the bandaid gets ripped off on
**August 1, 2026**.

## How bandaid thinks

- **Positions come from Gaia, not from your image.** bandaid does *forced*
    photometry at the positions of a magnitude-limited Gaia catalog — you do not
    detect your own sources.
- **It works in batches.** One slow preparation pass on the first frame
    (`prepare_batch`) sets up the Gaia catalog, plate scale, field of view, Bayer
    masks, and centroiding model; a fast per-frame loop (`process_batch`) reuses
    all of it.
- **One `.star` file per frame** — each bundles a separate star list for every
    Bayer filter (red/green/blue, plus a luminance channel) — plus a single
    `qa_manifest.csv` summarizing the run.

## Get started

Install into a Python ≥ 3.12 environment:

```bash
$ pip install bandaid
```

Photometer a night of frames and inspect instruments/configuration without
writing any Python:

```bash
# Photometer every FITS frame in a directory, writing .star files + a QA manifest
$ bandaid process night-of-2026-06-27/ -o out/ -v

# Inspect instruments and configuration
$ bandaid instrument list
$ bandaid config init -o config.json

# Fetch the default Ballet centroider weights
$ bandaid weights
```

The same run from Python is one call:

```python
from bandaid import photometer_frames

frames, results = photometer_frames(["night-of-2026-06-27/"], output_dir="out/")
```

See the [documentation](https://bandaid.readthedocs.io/) for the full guide —
[installation](https://bandaid.readthedocs.io/en/latest/installation/), a
[getting-started tutorial](https://bandaid.readthedocs.io/en/latest/getting_started/),
the [command-line reference](https://bandaid.readthedocs.io/en/latest/command_line/), and
[configuration](https://bandaid.readthedocs.io/en/latest/configuration/).

## Data-quality flags

bandaid is conservative about changing your data, so it is worth knowing exactly
what each quality check does:

- `centroid_drift` — the star's measured centroid wandered too far from its
    aligned/expected position (bad WCS, too-faint star, or an obstruction). See the
    [centroid-drift check](https://bandaid.readthedocs.io/en/latest/centroid_drift_check/).
    This is **flag-only** — no rows are dropped — and it lives on the in-memory
    photometry table (run from Python with `output_dir=None`), not in the `.star`
    files.
- Contamination — a bright neighbor's PSF wings spill into the aperture.
    `prepare_batch` flags contaminated *targets* (via
    `neighbor_contamination_flag_sky`) and **drops them from the photometry list
    before any frame is measured**, so a contaminated target simply has no row in
    the output. No per-row `contaminated` column is written.

## Development

Code style is enforced with [ruff](https://docs.astral.sh/ruff/) and
[pydoclint](https://jsh9.github.io/pydoclint/), pinned in `.pre-commit-config.yaml`
and the `style` dependency group. The CI `pre-commit` job runs:

```bash
uvx pre-commit run --all-files
```

Run the same command locally before pushing, or `uvx pre-commit install` to run it
automatically on every commit. See the
[code style guide](https://bandaid.readthedocs.io/en/latest/code_style/) for the
linting policy and the individual commands.

## Copyright

- Copyright © 2026 AAVSO.
- Free software distributed under the [MIT License](./LICENSE).
