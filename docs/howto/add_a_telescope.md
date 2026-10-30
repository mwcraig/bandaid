# Add a telescope

A telescope is an instrument profile: detection and PSF tuning, plate scale, gain,
and a `header_map` that translates your FITS keywords into the metadata bandaid
needs. The profile format is specified in the
[instrument profile reference](../reference/instrument_profiles.md).

## 1. Measure plate scale and gain

- `pixscale` (arcsec/px): run a night with your best guess, then take the median of
    the `wcs_pixscale` column of `qa_manifest.csv`. Frames rejected for their scale
    still record the scale they measured, so the column is filled even when the
    guess is wrong. Use the median, not the mean: a rejected solve can be a
    wrong-scale false solve.
- `wcs_scale_tolerance`: once `pixscale` is measured, tighten the default `0.05` to
    the spread of the good solves.
- `egain` (e-/ADU in the file's own units): measure it by photon transfer. A
    missing gain fails every frame. Give one number for every channel or, where
    the Bayer channels differ, a mapping of filter name to gain; the Seestar50
    profile ships `{"TR": 0.046, "TG": 0.076, "TB": 0.078}`, measured from flat
    pairs. With a mapping each channel's star list records its own gain and the
    `L4` list records the median per-star effective gain (`l4_effective_egain`).

## 2. Write a profile

Start from the bundled one and edit it:

```bash
bandaid instrument show Seestar50 > my_scope.json
```

A minimal profile for a telescope whose header is already ICRS:

```json
{
    "name": "MyScope",
    "thresh": 0.5,
    "detection_opening": 5,
    "fwhm_cutout_half": 25,
    "fwhm_n_stars": 25,
    "contamination_tolerance": 0.01,
    "moffat_beta": 3.0,
    "contamination_seeing_margin": 1.25,
    "wcs_scale_tolerance": 0.05,
    "header_match": [{"keyword": "INSTRUME", "pattern": "MyScope Model 1"}],
    "header_map": {
        "obs_time": "@DATE-OBS",
        "exposure": "@EXPTIME",
        "ra": "@RA",
        "dec": "@DEC",
        "bayerpat": "@BAYERPAT",
        "roworder": "top-down",
        "xbayroff": 0,
        "ybayroff": 0,
        "pixscale": 2.813,
        "fov_rad": 1.7,
        "egain": 0.046,
        "largest_usable_adu_value": 60000,
        "#stack": 1,
        "stack": "@STACKCNT"
    }
}
```

Replace `pixscale` and `egain` with your measured values. The `header_map`
directives, `header_match`, and the pointing-frame keys are in the
[reference](../reference/instrument_profiles.md). Leave the detection and PSF
knobs alone until you have a reason to tune them.

## 3. Use it

For one run, load the file by path (from Python, pass
`PhotometryConfig(instrument=InstrumentProfile.from_file(path))`):

```bash
bandaid process frames/ --profile my_scope.json
```

To refer to it by name for the rest of a Python session, register it:

```python
from bandaid import InstrumentProfile, load_instrument, register_instrument

register_instrument(InstrumentProfile.from_file("my_scope.json"))
load_instrument("MyScope")
```

Registration lasts only for the session; a separate `bandaid process --instrument MyScope` cannot see it. Use `--profile`, or bundle the profile (below).

## 4. Validate

Confirm the `header_map` resolves on your files with one frame before a long run:

```bash
bandaid process one_frame.fit --profile my_scope.json -o /tmp/check -vv
```

Success is a written `.star` file. A `FrameMetadataError` means a directive points
at a keyword your headers do not carry; see [troubleshooting](troubleshooting.md).

## Bundling it in the package

Bundling makes the profile resolve by name for everyone. Work in a checkout of the
repository.

1. Save the profile as `src/bandaid/meta_json_files/<Name>/profile.json`. Declare
    a `header_match` rule on a keyword that identifies the telescope model, such as
    `INSTRUME`, not a per-unit value such as a serial number.
1. Add the name to `test_lists_exactly_the_bundled_profiles` in
    `tests/unit/test_instruments.py`. That test pins the full set of bundled
    profiles.
1. Run `test_bundled_profiles_have_pairwise_disjoint_header_match_rules` in the
    same file. It fails if your `header_match` collides with another profile's.
1. Open a pull request. No registry code changes are needed: `available_instruments()`
    scans `meta_json_files/` for every `<name>/profile.json`.
