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
    pairs. How the gain is recorded per channel, including the `L4` list, is
    explained under `egain` in the
    [instrument profile reference](../reference/instrument_profiles.md#header_map-directives).
- `fov_rad` (degrees): half the frame diagonal (0.74 for the Seestar S50). It sets
    the Gaia query cone, the plate-solve star pool and the default pointing
    tolerance.

## 2. Write a profile

Start from the bundled one and edit it:

```bash
bandaid instrument show Seestar50 > my_scope.json
```

A minimal profile for a telescope whose header is already ICRS:

```json
{
    "name": "MyScope",
    "wcs_scale_tolerance": 0.05,
    "header_match": [{"keyword": "INSTRUME", "pattern": "MyScope Model 1"}],
    "header_map": {
        "obs_time": "@DATE-OBS",
        "exposure": "@EXPTIME",
        "block_filter": "@FILTER",
        "tel_manufac": "MyScope Inc",
        "tel_model": "@INSTRUME",
        "tel_firmware": "@FIRMWARE",
        "adc_depth": 12,
        "refframe": "ICRS",
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

Replace `pixscale`, `fov_rad` and `egain` with your measured values, and the
keyword names with the ones your headers use. The `header_map` directives and
`header_match` are in the [reference](../reference/instrument_profiles.md).
Omitted keys take their defaults, so leave the detection and PSF knobs out until
you have a reason to tune them.

The output format also requires the observer and site. Those belong to the
person running the telescope, so put them in a JSON file and pass it
with `--user-metadata` in steps 3 and 4:

```json
{"observer": "ABCD", "site_lat": 44.5, "site_lon": -92.1, "site_elev": 300.0}
```

A telescope whose header carries the site can map `site_lat`, `site_lon` and
`site_elev` with `@KEY` directives instead. The bundled Seestar50 profile maps
`site_lat` and `site_lon` (the Seestar writes latitude and longitude) but not
`site_elev` or `observer`, which its headers lack
(`src/bandaid/meta_json_files/Seestar50/profile.json`); supply those two with
`--user-metadata`. The observer code usually comes from user metadata; a header
that carries it can map it instead.

A required key that is missing or null on the first frame the batch is prepared
from stops the run at once with a `FrameMetadataError` that lists the keys.

If the header pointing is not ICRS, set `header_frame` and `header_equinox` to
match the header's `RADESYS` and `EQUINOX` keywords. The Seestar writes FK5 in the
equinox of date, hence `"header_frame": "fk5"` with `"header_equinox": "date"`; an
ICRS header omits both. A fixed epoch such as `"J2025.5"` is also accepted.

## 3. Use it

For one run, load the file by path (from Python, pass
`PhotometryConfig(instrument=InstrumentProfile.from_file(path))`):

```bash
bandaid process frames/ --profile my_scope.json --user-metadata site.json
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
bandaid process one_frame.fit --profile my_scope.json \
    --user-metadata site.json -o /tmp/check -vv
```

Success is a written `.star` file. A `FrameMetadataError` or
`StarListValidationError` usually means a directive points at a keyword your
headers do not carry (an absent `@KEY` resolves to null); see
[troubleshooting](troubleshooting.md).

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
