# ~~bandaid~~ Name TBD...

~~bandaid~~ turns a folder of Smart Telescope (Seestar) FITS frames into AAVSO-style [starlist files](https://github.com/AAVSO/public-utils/tree/main/schemas/starlist).

## To use ~~bandaid~~

Unless you are comfortable working at the command line in a terminal, you should go here to just drag and drop your images: https://mwcraig.github.io/browser-photom/

## Key ideas

- **Positions come from Gaia, not from your image.** bandaid does *forced*
    photometry at the positions of a magnitude-limited Gaia catalog — you do not
    detect your own sources. You can feed in a list of coordinates at which photometry is forced.
- **WCS is done for every image using a handful of stars.**
- **It works in batches.** One slow preparation pass on the first usable frame sets up
    the Gaia catalog, plate scale, field of view, Bayer masks, and centroiding
    model; a fast per-frame loop reuses all of it.
- **Centroids are found using a convolutional neural network for bright stars and not at all for faint stars.** For faint stars the photometry position is determined by the WCS solution and a plane-fit of bright star offsets on each frame.

## Use it

Install into a Python ≥ 3.12 environment and photometer a folder of frames:

```bash
$ pip install git+https://github.com/mwcraig/bandaid
$ bandaid process night-of-2026-06-27/ -o out/ -v
```

Full documentation: <https://mwcraig.github.io/bandaid/>

## Development

```bash
$ uv sync
$ uvx --python 3.12 pre-commit run --all-files
```

## Copyright

- Copyright © 2026 AAVSO.
- Free software distributed under the [MIT License](https://github.com/mwcraig/bandaid/blob/main/LICENSE).
