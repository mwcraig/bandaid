# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A tiered, pydantic-validated `PhotometryConfig` (with `ApertureConfig`,
    `DetectionConfig`, `QualityConfig`, and `InstrumentConfig`) makes the
    photometry tuning parameters configurable. `prepare_batch` accepts a
    `config=` argument carried through the batch pipeline. See
    `docs/configuration.md`.

### Changed (breaking)

- `measure_photometry` and `build_photometry_table` renamed their keyword-only
    `relative_radii=` argument to `radii=`. Both are re-exported from the package
    root, so calls using `relative_radii=` now raise `TypeError`; pass `radii=`
    instead. No deprecated alias is provided.
- `prepare_batch` dropped its `gaia_mag_limit=` and `contaminant_mag_limit=`
    keyword arguments. Set these via `config=PhotometryConfig(detection=...)`
    instead.

## [0.1.0] - (1979-01-01)

- First release
