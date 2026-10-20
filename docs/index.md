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

## Where to go

- **[Getting started](getting_started.md)** — install and run your first batch.
- **How-to guides** — [tune a run](howto/tune_a_run.md),
    [add a telescope](howto/add_a_telescope.md),
    [troubleshoot](howto/troubleshooting.md).
- **Reference** — [command line](reference/command_line.md),
    [outputs](reference/outputs.md), [configuration](reference/configuration.md).
- **Explanation** — [star positions](explanation/star_positions.md),
    [why Gaia](explanation/why_gaia.md).
- **[Contributing](developers/contributing.md)** — developer setup.

Copyright © 2026 AAVSO. Free software distributed under the
[MIT License](https://github.com/mwcraig/bandaid/blob/main/LICENSE).
