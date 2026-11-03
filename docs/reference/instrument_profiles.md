# Instrument profiles

An instrument profile (`InstrumentProfile`) holds one telescope's detection and
PSF tuning, its plate-solving tolerances, and the `header_map` that reads its FITS
headers. It is carried on `PhotometryConfig.instrument` and applied to every frame
of a batch. Field defaults are in [Configuration](configuration.md). To write a
profile for a new telescope, see [Add a telescope](../howto/add_a_telescope.md).

## JSON format

`InstrumentProfile.to_file(path)` and `InstrumentProfile.from_file(path)` read and
write this format; `bandaid instrument show NAME` prints it. Omitted keys take the
defaults in [Configuration](configuration.md).

Every key other than `header_match` and `header_map` (below) is a Tier 2 field of
[Configuration](configuration.md). Choose `header_frame` and `header_equinox` as described in
[Add a telescope](../howto/add_a_telescope.md#2-write-a-profile).

## `header_map` directives

Keys are lower-case metadata names; `metadata_from_header` resolves each against
a frame's header.

| Form             | Example                             | Meaning                                                                                                                                    |
| ---------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `"@KEY"`         | `"obs_time": "@DATE-OBS"`           | Header value of `KEY`; falls back to the `#` default for that key if absent.                                                               |
| `"!KEY index N"` | `"tel_manufac": "!CREATOR index 0"` | Whitespace-split header value of `KEY`, token `N` (0-based). No `#` fallback; a missing or non-string keyword raises `FrameMetadataError`. |
| literal          | `"egain": 0.046`                    | Used as given (a number, string or JSON object), for constants the header lacks.                                                           |
| `"#key": value`  | `"#stack": 1`                       | Fallback for `key` when its `@` lookup finds nothing.                                                                                      |
| `"_anything"`    | `"_note": "..."`                    | Comment; ignored.                                                                                                                          |

`width` and `height` always come from `NAXIS1`/`NAXIS2`; do not map them. The
pipeline requires these keys:

| Keys                                           | Use                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `egain`                                        | Noise model, e-/ADU: one number, or a mapping of filter name to gain when the Bayer channels differ (the Seestar50 profile ships `{"TR": 0.046, "TG": 0.076, "TB": 0.078}`). Each channel's star list records its own gain; the `L4` list records the median per-star effective gain (`l4_effective_egain`). The `L4` `count_err` is built from the per-channel error terms and does not depend on that recorded median. A missing value, or a measured channel with no entry, raises `FrameMetadataError`. |
| `largest_usable_adu_value`                     | Saturation cut.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `pixscale`, `fov_rad`                          | Plate scale (arcsec/pixel) and field radius (degrees).                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `ra`, `dec`                                    | Header pointing, for alignment and the pointing check.                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `bayerpat`, `roworder`, `xbayroff`, `ybayroff` | Debayering.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

Keys the output schema requires, so every profile (or `--user-metadata`) must
supply them: `obs_time`, `site_lat`, `site_lon`, `site_elev`, `observer`,
`block_filter`, `exposure`, `tel_manufac`, `tel_model`, `tel_firmware`,
`adc_depth`, `refframe`, `egain` and `largest_usable_adu_value`. bandaid fills
`photometry_software` itself. The optional schema keys `stack` and
`absolute_focus` may be mapped, left null or left out; the bundled profile pairs
`"stack": "@STACKCNT"` with `"#stack": 1` and maps `absolute_focus` to
`@FOCUSPOS`, the focuser position. `airmass` is derived from `ra`/`dec`, the site
and `obs_time` unless mapped. `object` and `system_id`, which the bundled profile
also maps, are not output-schema fields.

The preparation frame is checked first: a required key that is absent or null in
its metadata, after `--user-metadata` is applied, stops
the run with a `FrameMetadataError` that lists the keys. A key mapped to a header
keyword a later frame lacks resolves to null on that frame. For `obs_time` and the
site keys that is a `FrameMetadataError` (no parseable time, or no airmass), which
skips the frame; for any other key the frame is skipped with
`StarListValidationError`.

## `header_match` and detection

A `HeaderMatchRule` matches when `keyword` is present in the header and its value
equals `pattern` after stripping whitespace and casefolding. A profile matches
when any of its rules match. `detect_instrument(header)` considers every
bundled or registered profile with a non-empty `header_match` and returns the one
that matches; zero or several matches raise `InstrumentDetectionError`, naming the
header values checked and the candidate profiles. An empty `header_match` (the
default, including on a bare `InstrumentProfile()`) means a profile is never
auto-detected. The bundled `Seestar50` rule is `INSTRUME == "Seestar S50"`.

`prepare_batch` wraps a detection failure in `BatchPrepError`, chaining
`InstrumentDetectionError`. The single-frame entry points let the bare exception
propagate.

Every later frame of a batch is checked against the batch instrument's
`header_match`: a header that identifies a different or several registered
instruments is rejected. An explicitly selected profile is exempt only when the
header matches no registered instrument.

## Selection precedence

The first that applies wins:

1. `bandaid process --instrument NAME` or `--profile FILE`.
1. A `--config` file (or `PhotometryConfig`) whose `instrument` is a concrete
    profile.
1. Auto-detection from the first frame's header, when `instrument` is `None`.
1. Otherwise the batch stops with `BatchPrepError` (cause: `InstrumentDetectionError`).

## Registry

All are importable from `bandaid`; signatures are in [API reference](api.md).

- `load_instrument(name)` returns the profile; registered profiles take precedence
    over bundled ones, and an unknown name raises `ValueError`.
- `register_instrument(profile, *, replace=False)` registers a profile for the
    current process. A name that already resolves raises unless `replace=True`, and
    a `header_match` rule duplicating another profile's raises. With `replace=True`
    a profile that omits `header_match` (and `header_frame`/`header_equinox`) inherits
    the replaced profile's.
- `available_instruments()` returns the sorted bundled and registered names.

Registration lasts for the process; see [Add a telescope](../howto/add_a_telescope.md#3-use-it).

## Observer identity

The `header_map` covers only instrument metadata. The `user_specific_metadata`
dict (`--user-metadata FILE` on the command line, an argument to `process_batch`
and `process_frame`) is applied last and overrides any metadata key, typically
`observer`, `site_lat`, `site_lon`, and `site_elev`.
