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

Install into a Python ≥ 3.12 environment (git must be installed, since two
dependencies are fetched from GitHub):

```bash
$ pip install git+https://github.com/mwcraig/bandaid
```

Seestar headers do not record who you are or how high your site is, and the
`.star` format needs both, so put your AAVSO observer code and your site
elevation in metres in a small JSON file, say `site.json`:

```json
{"observer": "ABCD", "site_elev": 300.0}
```

then photometer a folder of frames:

```bash
$ bandaid process night-of-2026-06-27/ -o out/ --user-metadata site.json -v
```

The first run downloads the centroiding weights from HuggingFace. Full
documentation, including what to do offline: <https://mwcraig.github.io/bandaid/>

## Development

```bash
$ git clone https://github.com/mwcraig/bandaid
$ cd bandaid
$ uv sync          # editable install plus the dev group: tests, docs, lint
$ uv run pytest    # the test suite; no network needed
$ uv run poe lint  # pre-commit on every file, as CI runs it
```

`uv run poe docs-build` builds the documentation. The
[contributing guide](https://mwcraig.github.io/bandaid/developers/contributing/)
has the task list, the pre-commit hooks, the code style and the test conventions.

## Copyright

- Copyright © 2026 AAVSO.
- Free software distributed under the [MIT License](https://github.com/mwcraig/bandaid/blob/main/LICENSE).
