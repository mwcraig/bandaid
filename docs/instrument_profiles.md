# Instrument profiles

An **instrument profile** is bandaid's model of a single telescope. It bundles
the two telescope-specific things the pipeline needs:

- the **detection / PSF tuning** knobs (`thresh`, `detection_opening`,
    `fwhm_cutout_half`, `fwhm_n_stars`, `contamination_tolerance`, `moffat_beta`),
    and
- the **header map** — a small mapping that tells the pipeline how to read that
    telescope's per-frame FITS headers into the metadata it needs.

Both live on one `InstrumentProfile`, carried on `PhotometryConfig.instrument`
and applied to every frame in a batch. The bundled default is the Seestar50; the
field tables and the tuning knobs are described in
[Configuration](configuration.md#tier-2-instrument-per-telescope-advanced).

## The registry

Named profiles live in `bandaid.instruments`:

```python
from bandaid import load_instrument, register_instrument, available_instruments
from bandaid.config import InstrumentProfile

available_instruments()            # -> ['Seestar50']
profile = load_instrument("Seestar50")

# Share a tuned profile through a file, or register one so load_instrument
# resolves it by name (a registered name overrides a bundled one).
profile.to_file("my_scope.json")
mine = InstrumentProfile.from_file("my_scope.json")
register_instrument(mine)
available_instruments()            # -> ['Seestar50', '<mine.name>']
```

Pass a profile to a batch through the config:

```python
from bandaid import PhotometryConfig, prepare_batch

config = PhotometryConfig(instrument=load_instrument("Seestar50"))
prep = prepare_batch(first_file, cnn=cnn, config=config)
```

## The `header_map` directive language

A profile's `header_map` is a JSON object mapping each **metadata key** the
pipeline wants (lower-case, e.g. `obs_time`, `egain`, `pixscale`) to a
**directive** describing where its value comes from. `metadata_from_header`
resolves the directives against a frame's FITS header.

| Directive form   | Example                             | Meaning                                                                                      |
| ---------------- | ----------------------------------- | -------------------------------------------------------------------------------------------- |
| `"@KEY"`         | `"obs_time": "@DATE-OBS"`           | Take the value of FITS header keyword `KEY`. Falls back to a `#`-default (below) if missing. |
| `"!KEY index N"` | `"tel_manufac": "!CREATOR index 0"` | Split header keyword `KEY` on whitespace and take the `N`-th token (0-based).                |
| literal          | `"pixscale": 2.4`                   | Use the literal value as-is. Used for hardware constants the header does not carry.          |
| `"#key": value`  | `"#stack": 1`                       | A **fallback** for `key`: used only when the `@` lookup for `key` finds nothing.             |
| `"_anything"`    | `"_note": "..."`                    | A comment. Ignored entirely.                                                                 |

Notes:

- FITS keywords are upper-case; metadata keys are lower-case. The directive
    bridges the two (`"site_lat": "@SITELAT"`).
- The `!` form currently supports the `index` selector only (the function word is
    a label; the integer after it is the token index). A header keyword that is
    missing or not a string raises `FrameMetadataError` for that frame; unlike
    `@`, the `!` form does **not** consult a `#`-default.
- `metadata_from_header` always adds `width`/`height` from `NAXIS1`/`NAXIS2`, so
    you do **not** put those in the `header_map`.

## Adding a telescope

### Start from the bundled profile

Don't author a profile from a blank file — dump the Seestar50 and edit it. It is
already a complete, working example of every field and the `header_map`:

```bash
$ bandaid instrument show Seestar50 > my_scope.json
```

Change the `name`, the plate scale / FOV, and the `header_map` directives to match
your telescope's headers, leaving the detection/PSF knobs at their defaults until
you have a reason to tune them.

### Three ways to use your profile

From quickest to most permanent:

1. **Ad-hoc file** — point a single run at the JSON, no registration needed:

    ```bash
    $ bandaid process frames/ --profile my_scope.json
    ```

    ```python
    from bandaid import PhotometryConfig
    from bandaid.config import InstrumentProfile

    config = PhotometryConfig(instrument=InstrumentProfile.from_file("my_scope.json"))
    ```

1. **Register it in-process** — load it once, then refer to it by name for the
    rest of the session. Nothing is written into the installed package:

    ```python
    from bandaid import register_instrument, load_instrument
    from bandaid.config import InstrumentProfile

    register_instrument(InstrumentProfile.from_file("my_scope.json"))
    load_instrument("MyScope")          # now resolves by name
    ```

    This registration lives only in the current Python session — a separate
    `bandaid process --instrument MyScope` invocation is a new process with an
    empty registry, so it won't see it. For the CLI, use the ad-hoc
    `--profile my_scope.json` shown above, or bundle the profile (below) to
    resolve it by name everywhere.

1. **Bundle it (contributor path)** — drop
    `src/bandaid/meta_json_files/<Name>/profile.json` into the source tree and
    update the bundled-names test; it is then auto-discovered for everyone with no
    other code changes. See
    [Adding a bundled instrument profile](contributing.md#adding-a-bundled-instrument-profile).

A `my_scope.json` looks like:

```json
{
    "name": "MyScope",
    "thresh": 0.5,
    "detection_opening": 3,
    "fwhm_cutout_half": 25,
    "fwhm_n_stars": 25,
    "contamination_tolerance": 0.01,
    "moffat_beta": 3.0,
    "header_map": {
        "obs_time": "@DATE-OBS",
        "exposure": "@EXPTIME",
        "ra": "@RA",
        "dec": "@DEC",
        "bayerpat": "@BAYERPAT",
        "roworder": "top-down",
        "ybayroff": 0,
        "pixscale": 2.8,
        "fov_rad": 1.7,
        "egain": 0.31,
        "largest_usable_adu_value": 60000,
        "#stack": 1,
        "stack": "@STACKCNT"
    }
}
```

**We'd love to bundle your telescope.** If you've tuned a profile for a scope
that isn't built in yet, a pull request adding it ships it with bandaid so it
resolves by name for everyone — no runtime registration needed. See
[Adding a bundled instrument profile](contributing.md#adding-a-bundled-instrument-profile)
for the short walkthrough.

### Validate before a long run

Catch a typo before it fails deep in an overnight batch. Validate the profile (as
part of a full config) up front:

```bash
$ bandaid config validate my_config.json
```

`InstrumentProfile.from_file` raises the same Pydantic errors directly. The common
ones are out-of-range values — `thresh`, `contamination_tolerance`, and
`moffat_beta` must be `> 0`; `detection_opening`, `fwhm_cutout_half`, and
`fwhm_n_stars` must be `>= 1`.

### Verify it parses your headers

Validation checks the profile's *shape*, not that its `header_map` matches your
FITS files. Confirm the directives actually resolve by running a **single frame**
with `-vv` and checking that metadata (pixel scale, gain, RA/Dec) resolved and a
`.star` file was written:

```bash
$ bandaid process one_frame.fit --profile my_scope.json -o /tmp/check -vv
```

A `FrameMetadataError` here means a `header_map` directive points at a keyword
your headers don't carry — see [Troubleshooting](troubleshooting.md).

### Keys a profile should provide

The pipeline reads these metadata keys, so a new telescope's `header_map` must
resolve them:

- `egain` — system gain; feeds the photometry noise model. A missing/`None`
    value fails the frame, so provide it as a literal (or a header lookup that is
    always present).
- `largest_usable_adu_value` — saturation cut.
- `pixscale`, `fov_rad` — plate scale and field of view used for Gaia matching
    and FWHM-in-arcsec reporting.
- `ra`, `dec` — field center for alignment.
- `bayerpat`, `roworder`, `ybayroff` — Bayer pattern handling for debayering.

Observation-bookkeeping keys (`obs_time`, `site_lat`/`site_lon`/`site_elev`,
`observer`, `exposure`, `object`, …) flow into the output and should be mapped
when the telescope's header carries them.

## What a profile does **not** cover: observer identity

The `header_map` resolves only the *instrument* half of a frame's metadata.
Observer-specific values — site location and observer code — are a separate,
telescope-independent layer applied **last**, via the `user_specific_metadata`
dict passed to `process_batch`. Those overrides win over whatever the header
provided. See the metadata layering described on
[`process_batch`](configuration.md). Building a richer "personal.json" loader for
that layer is intentionally out of scope of the profile registry.
