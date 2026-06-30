# bandaid

bandaid turns a folder of Smart Telescope (Seestar) FITS frames into AAVSO-style
photometry — `.star` files you can submit or analyze — without writing any code.
It is a deliberately temporary solution: the bandaid gets ripped off on
**August 1, 2026**.

## How bandaid thinks

Three ideas explain almost everything in these docs:

- **Positions come from Gaia, not from your image.** bandaid does *forced*
    photometry at the positions of a magnitude-limited Gaia catalog — you do not
    detect your own sources. This is the most counterintuitive part for new users.
- **It works in batches.** One slow preparation pass on the first frame sets up
    the Gaia catalog, plate scale, field of view, Bayer masks, and centroiding
    model; a fast per-frame loop then reuses all of it.
- **One `.star` file per frame** — each bundles a separate star list for every
    Bayer filter (red/green/blue, plus a luminance channel) — plus a single
    `qa_manifest.csv` summarizing the run.

## Who this is for

An intermediate Python user — comfortable with pip, virtual environments, and the
REPL — who has a folder of Seestar frames and wants photometry out. You do not
need to be an expert in photometry internals; the docs focus on *using the
tools*, not on the algorithms.

## Get started

```bash
$ pip install bandaid
$ bandaid process night-of-2026-06-27/ -o out/ -v
```

- **[Installation](installation.md)** — requirements and the Ballet weights.
- **[Getting started](getting_started.md)** — the ~10-minute end-to-end run.

## Copyright

- Copyright © 2026 AAVSO.
- Free software distributed under the [MIT License](https://github.com/mwcraig/bandaid/blob/main/LICENSE).
