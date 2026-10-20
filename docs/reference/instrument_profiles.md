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

| Key                                                                     | Type                         | Meaning                                                                                                                                                                                                             |
| ----------------------------------------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`                                                                  | string                       | Registry key.                                                                                                                                                                                                       |
| `thresh`, `detection_opening`, `fwhm_cutout_half`, `fwhm_n_stars`       | number / int                 | Detection and FWHM-fit tuning.                                                                                                                                                                                      |
| `contamination_tolerance`, `moffat_beta`, `contamination_seeing_margin` | number                       | Bright-neighbour contamination model.                                                                                                                                                                               |
| `wcs_scale_tolerance`                                                   | number                       | Fractional tolerance on the solved plate scale relative to `header_map["pixscale"]`.                                                                                                                                |
| `wcs_pointing_tolerance`                                                | number or null               | Degrees between a solved center and the header pointing.                                                                                                                                                            |
| `header_frame`, `header_equinox`                                        | string                       | Frame and equinox of the header RA/DEC; case-insensitive. `header_equinox` is `"date"` (each frame's observation time) or an epoch string; setting it with `header_frame` `"icrs"` is an error.                     |
| `cone_radius_margin`                                                    | number                       | Degrees added to `fov_rad` for the batch Gaia query; a frame whose pointing drifts further than this but less than `fov_rad` is processed with a coverage warning, and one drifting more than `fov_rad` is skipped. |
| `solve_pool_radius_scale`                                               | number                       | Plate-solve star-pool radius as a fraction of `fov_rad`, centered on each frame's own header pointing.                                                                                                              |
| `header_match`                                                          | list of `{keyword, pattern}` | Auto-detection rules; see below.                                                                                                                                                                                    |
| `header_map`                                                            | object                       | Metadata key to directive; see below.                                                                                                                                                                               |

The key `header_center_offset` is rejected unless it is `null`.

## `header_map` directives

Keys are lower-case metadata names; `metadata_from_header` resolves each against
a frame's header.

| Form             | Example                             | Meaning                                                                                                                                    |
| ---------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `"@KEY"`         | `"obs_time": "@DATE-OBS"`           | Header value of `KEY`; falls back to the `#` default for that key if absent.                                                               |
| `"!KEY index N"` | `"tel_manufac": "!CREATOR index 0"` | Whitespace-split header value of `KEY`, token `N` (0-based). No `#` fallback; a missing or non-string keyword raises `FrameMetadataError`. |
| literal          | `"egain": 0.046`                    | Used as given, for constants the header lacks.                                                                                             |
| `"#key": value`  | `"#stack": 1`                       | Fallback for `key` when its `@` lookup finds nothing.                                                                                      |
| `"_anything"`    | `"_note": "..."`                    | Comment; ignored.                                                                                                                          |

`width` and `height` always come from `NAXIS1`/`NAXIS2`; do not map them. The
pipeline requires these keys:

| Keys                                           | Use                                                               |
| ---------------------------------------------- | ----------------------------------------------------------------- |
| `egain`                                        | Noise model, e-/ADU. A missing value raises `FrameMetadataError`. |
| `largest_usable_adu_value`                     | Saturation cut.                                                   |
| `pixscale`, `fov_rad`                          | Plate scale (arcsec/pixel) and field radius (degrees).            |
| `ra`, `dec`                                    | Header pointing, for alignment and the pointing check.            |
| `bayerpat`, `roworder`, `xbayroff`, `ybayroff` | Debayering.                                                       |

Bookkeeping keys that flow into the output when mapped: `obs_time`, `site_lat`,
`site_lon`, `site_elev`, `observer`, `exposure`, `object`, `block_filter`,
`tel_manufac`, `tel_model`, `tel_firmware`, `adc_depth`, `stack`, `refframe`,
`system_id`. Airmass is derived from `ra`/`dec`, site and `obs_time` unless
`airmass` is mapped, in which case that value is used.

## `header_match` and detection

A `HeaderMatchRule` matches when `keyword` is present in the header and its value
equals `pattern` after stripping whitespace and casefolding. A profile matches
when any of its rules match. `detect_instrument(header)` considers every
bundled or registered profile with a non-empty `header_match` and returns the one
that matches; zero or several matches raise `InstrumentDetectionError`, naming the
header values checked and the candidate profiles. An empty `header_match` (the
default, including on a bare `InstrumentProfile()`) means a profile is never
auto-detected. The bundled `Seestar50` rule is `INSTRUME == "Seestar S50"`.

`prepare_batch` wraps a detection failure in `BatchPrepError`. The single-frame
entry points (`prepare_image`, `process_one_image`, `calibration_sequence`) let
`InstrumentDetectionError` propagate, and it is a `FrameError`.

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
1. Otherwise `InstrumentDetectionError`, which stops the batch.

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

Registration lasts for the process, so a separate `bandaid` command does not see it.

## Observer identity

The `header_map` covers only instrument metadata. The `user_specific_metadata`
dict (`--user-metadata FILE` on the command line, an argument to `process_batch`
and `process_frame`) is applied last and overrides any metadata key, typically
`observer`, `site_lat`, `site_lon`, and `site_elev`.
