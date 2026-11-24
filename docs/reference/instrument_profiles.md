# Instrument profiles

An instrument profile (`InstrumentProfile`) holds one telescope's detection and
PSF tuning, its plate-solving tolerances, and the `header_map` that reads its FITS
headers. It is carried on `PhotometryConfig.instrument` and applied to every frame
of a batch. Field defaults are in [Configuration](configuration.md). To write a
profile for a new telescope, see [Add a telescope](../howto/add_a_telescope.md).

## JSON format

`InstrumentProfile.to_file(path)` and `InstrumentProfile.from_file(path)` read and
write this format; `bandaid instrument show NAME` prints it. Omitted keys take the
defaults in [Configuration](configuration.md), except `photometry_faint_mag_limit`
and `max_plane_rms_pix`, which every profile must state.

Every key other than `header_match`, `header_map` and `star_flaggers` (below) is a Tier 2 field of
[Configuration](configuration.md). Choose `header_frame` and `header_equinox` as described in
[Add a telescope](../howto/add_a_telescope.md#2-write-a-profile).

## Observation time

bandaid works from the UTC time at the start of each exposure. Two fields say
what the header's `obs_time` (usually mapped from `DATE-OBS`) is when it is not
that:

| Field            | Values                         | Default   | Meaning                                                   |
| ---------------- | ------------------------------ | --------- | --------------------------------------------------------- |
| `obs_time_marks` | `"start"`, `"middle"`, `"end"` | `"start"` | The point of the exposure the time marks.                 |
| `obs_time_zone`  | `"utc"`, `"local"`             | `"utc"`   | The zone of a time that carries no UTC offset of its own. |

- With `"start"` and `"utc"` the header value is used as it is, so star lists
    from such a profile do not change.
- Otherwise `metadata_from_header` converts the time once to the UTC start of
    the exposure and writes it back to `obs_time` as an ISO string such as
    `2021-06-15T03:45:00` (fractional seconds kept). The star list, the
    mid-exposure `time` column, the airmass, the Gaia epoch and the equinox of
    date for `"header_equinox": "date"` all use the converted time.
- `"middle"` moves the time back by half the `exposure`, `"end"` by all of it.
    A frame whose `stack` count is above 1 is skipped with `FrameMetadataError`,
    because the stamp of a stacked frame does not say which exposure it marks.
    If it is the first frame of the batch the run stops instead, since the
    rest of the batch is likely stacked too.
- `"local"` reads a time with no offset in the zone the observer gives as
    `time_zone` in the user metadata ([Observer identity](#observer-identity)):
    an IANA zone name such as `"America/Chicago"`, or the UTC offset in hours
    as a number (`-6`). The zone is never derived from the site position. A
    run without `time_zone`, or with one that names no zone, stops before any
    frame is processed with `BatchPrepError`. From Python, the zone given to
    `prepare_batch` is the one every frame is read with.
- A time that carries its own UTC offset (`2024-01-01T05:00:00+05:00`) is read
    with that offset, whatever `obs_time_zone` says.
- With a zone name, a frame stamped in the hour the clocks repeat when they go
    back, or in the hour that does not exist when they go forward, is skipped
    with `ClockChangeError`: its time does not name one instant. The batch is
    then prepared from the next frame if that was the first. A fixed offset in
    hours has no clock changes.

The bundled `Seestar50` profile keeps the defaults.

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
| `largest_usable_adu_value`                     | Saturation cut; the Seestar core-replaced flag also uses it as its saturation ceiling.                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `pixscale`, `fov_rad`                          | Plate scale (arcsec/pixel) and field radius (degrees).                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `ra`, `dec`                                    | Header pointing, for alignment and the pointing check.                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `bayerpat`, `roworder`, `xbayroff`, `ybayroff` | Debayering.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

Keys the output schema requires, so every profile (or `--user-metadata`) must
supply them: `obs_time`, `site_lat`, `site_lon`, `site_elev`, `observer`,
`block_filter`, `exposure`, `tel_manufac`, `tel_model`, `tel_firmware`,
`adc_depth`, `refframe`, `egain`, `largest_usable_adu_value` and `stack`. The
`stack` value may be null, but the key must be present (the bundled profile pairs
`"stack": "@STACKCNT"` with `"#stack": 1`). `airmass` is derived from `ra`/`dec`,
the site and `obs_time` unless mapped. `object` and `system_id`, which the bundled
profile also maps, are not output-schema fields.

The preparation frame is checked first: a required key (other than `stack`)
that is absent or null in its metadata, after `--user-metadata` is applied, stops
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
default) means a profile is never
auto-detected. The bundled `Seestar50` rule is `INSTRUME == "Seestar S50"`.

`prepare_batch` wraps a detection failure in `BatchPrepError`, chaining
`InstrumentDetectionError`. The single-frame entry points let the bare exception
propagate.

Every later frame of a batch is checked against the batch instrument's
`header_match`: a header that identifies a different or several registered
instruments is rejected. An explicitly selected profile is exempt only when the
header matches no registered instrument.

## `star_flaggers`

`star_flaggers` lists names of *star flaggers*: instrument-specific code that
marks stars whose measurement the camera itself has spoiled. Each name must be
registered with `register_star_flagger` or built into bandaid; a profile naming an unknown one
fails in `prepare_batch` (or, from `bandaid process`, before any frame is
processed) with a `ValueError` naming it. On each frame
every named flagger runs on the camera's raw integers, the star positions and
each channel's peak and sky. A star it flags in any channel gets the flagger's
boolean column (for the bundled Seestar50 flagger, `core_replaced`) set on every
filter's table, `L4` included, and is dropped from every filter of that frame's
`.star` file; the column itself is not written there. The QA manifest counts the
`core_replaced` stars in `n_core_replaced`.

The bundled `Seestar50` profile names one flagger, the built-in
`seestar_core_replaced` (see
[Output files](outputs.md#seestar-core-replaced-stars)). The default is empty, so a custom profile, even one named
`Seestar50`, runs no instrument code unless its `star_flaggers` names a
flagger, or unless it is registered over the bundled one with
`register_instrument(..., replace=True)` without setting `star_flaggers`, in
which case it inherits the bundled flaggers (set `star_flaggers=()` to run
none). `--no-instrument-flags` (or `"instrument_star_flags": false` in a
config) switches every flagger off: no flagger is resolved or run and no frame
is read for its state, and the `.star` output and photometry tables are
identical to those of a profile without flaggers.

A flagger may build state once per batch from the first few raw frames of the
batch (`n_state_frames`); `photometer_frames` and `bandaid process` read those
frames for it. A `BatchPrep` built with `prepare_batch` without `state_frames`
carries no state, and the flagger then runs without it (for the Seestar flag, no
static list). Writing a flagger is
covered in [Add a telescope](../howto/add_a_telescope.md#instrument-star-flags).

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
    the replaced profile's, as does one that omits `star_flaggers`,
    `obs_time_marks` or `obs_time_zone`.
- `available_instruments()` returns the sorted bundled and registered names.
- `register_star_flagger(name, flagger, *, replace=False)` registers a
    `StarFlagger` under the name a profile's `star_flaggers` uses. A name already
    registered or built in raises unless `replace=True`; a registration takes
    precedence over a built-in of the same name.

Registration lasts for the process; see [Add a telescope](../howto/add_a_telescope.md#3-use-it).

## Observer identity

The `header_map` covers only instrument metadata. The `user_specific_metadata`
dict (`--user-metadata FILE` on the command line, an argument to `process_batch`
and `process_frame`) is applied last and overrides any metadata key. It carries
whatever the headers do not: for the bundled `Seestar50` profile that is
`observer` and `site_elev`, so

```json
{"observer": "ABCD", "site_elev": 300.0}
```

is the minimum ([getting started](../getting_started.md#observer-and-site)); a
profile whose headers also lack the site position adds `site_lat` and `site_lon`.

For a profile with `"obs_time_zone": "local"`:

```json
{"observer": "ABCD", "site_elev": 300.0, "time_zone": "America/Chicago"}
```
