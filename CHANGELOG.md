# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A `bandaid` command-line interface (`bandaid process`, `instrument`, `config`,
    `weights`) for running photometry on a night of frames and inspecting
    instruments/config without writing Python. See `docs/command_line.md`. The
    same flow is available from Python as `bandaid.photometer_frames` (with
    `expand_frame_paths` for the directory/glob/path expansion); both are
    re-exported from the package root.
    `bandaid process` expands directories, globs, and paths (de-duplicated by
    resolved path, filtered to FITS frames including `.gz` forms). Frames from a
    single directory are written flat as `<stem>.star`; frames from a mix of
    directories mirror the source tree as `<dirname>/<stem>.star`, keeping
    identically named frames distinct without mangling their names.
- A tiered, pydantic-validated `PhotometryConfig` (with `ApertureConfig`,
    `SourceSelectionConfig`, `DriftConfig`, and `InstrumentProfile`) makes the
    photometry tuning parameters configurable. `prepare_batch` accepts a
    `config=` argument carried through the batch pipeline. See
    `docs/configuration.md`.
- An instrument-profile registry (`bandaid.instruments`) unifies a telescope's
    detection tuning with its per-frame FITS-header dialect. `InstrumentProfile`
    carries both (a `header_map` plus the tuning knobs) and serialises to/from a
    file (`InstrumentProfile.from_file`/`to_file`); `load_instrument`,
    `register_instrument`, and `available_instruments` resolve profiles by name.
    The bundled Seestar50 dialect moved from `meta_json_files/Seestar50/basic.json`
    into `meta_json_files/Seestar50/profile.json`, and `metadata_from_header`
    takes an optional `profile=`.

### Changed (breaking)

- `measure_photometry` and `build_photometry_table` renamed their keyword-only
    `relative_radii=` argument to `radii=`. Both are re-exported from the package
    root, so calls using `relative_radii=` now raise `TypeError`; pass `radii=`
    instead. No deprecated alias is provided.
- `prepare_batch` dropped its `gaia_mag_limit=` and `contaminant_mag_limit=`
    keyword arguments. Set these via
    `config=PhotometryConfig(source_selection=SourceSelectionConfig(gaia_mag_limit=...))`
    instead.

## [0.1.0] - (1979-01-01)

- First release
