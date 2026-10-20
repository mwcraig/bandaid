# bandaid

bandaid turns a folder of Smart Telescope (Seestar) FITS frames into AAVSO-style
photometry — `.star` files you can submit or analyze — without writing any code.

## Three ideas

- **Positions come from Gaia, not from your image.** bandaid does *forced*
    photometry at the positions of a magnitude-limited Gaia catalog — you do not
    detect your own sources.
- **It works in batches.** One slow preparation pass on the first frame sets up
    the Gaia catalog, plate scale, field of view, Bayer masks, and centroiding
    model; a fast per-frame loop reuses all of it.
- **One `.star` file per frame** — each bundles a separate star list for every
    Bayer filter (red/green/blue, plus a luminance channel) — plus a single
    `qa_manifest.csv` summarizing the run.

## Use it

Install into a Python ≥ 3.12 environment and photometer a folder of frames:

```bash
$ pip install bandaid
$ bandaid process night-of-2026-06-27/ -o out/ -v
```

Full documentation: <https://bandaid.readthedocs.io/>

## Development

```bash
$ uv sync
$ uvx --python 3.12 pre-commit run --all-files
```

## Copyright

- Copyright © 2026 AAVSO.
- Free software distributed under the [MIT License](./LICENSE).
