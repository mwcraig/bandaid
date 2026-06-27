# Contributing

## Adding a bundled instrument profile

bandaid ships a small set of built-in telescopes (currently just the Seestar50)
so they resolve by name with no runtime registration. We welcome pull requests
adding more.

This is a **contributor workflow**: it adds a file to the package *source*, so it
is done in a checkout of the repository — not by hand-editing an installed
`site-packages`/virtualenv copy. An end user who just wants to run their own
telescope should register a profile at runtime instead; see
[Instrument profiles](instrument_profiles.md#adding-a-telescope).

To bundle a telescope:

1. Add a profile directory and file in the source tree, named for the telescope:

    ```text
    src/bandaid/meta_json_files/<Name>/profile.json
    ```

1. Author the `profile.json` — the detection/PSF tuning knobs plus the
    `header_map` dialect. See
    [Instrument profiles](instrument_profiles.md#the-header_map-directive-language)
    for the `header_map` directive syntax and
    [Keys a profile should provide](instrument_profiles.md#keys-a-profile-should-provide)
    for the metadata keys it must resolve. A minimal example:

    ```json
    {
        "name": "MyScope",
        "thresh": 0.5,
        "detection_opening": 3,
        "fwhm_cutout_half": 25,
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

1. Update the bundled-set test in `tests/unit/test_instruments.py`
    (`test_lists_exactly_the_bundled_profiles`) to include the new name. That test
    pins the *complete* discovered set, so adding a telescope is a deliberate,
    reviewed change rather than a silent one.

1. Open a pull request. No other code changes are needed: the registry discovers
    the new directory automatically (`available_instruments()` scans
    `meta_json_files/` for any `<name>/profile.json`), so `load_instrument("MyScope")`
    and `available_instruments()` pick it up with no edits to Python.
