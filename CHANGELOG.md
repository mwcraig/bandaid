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
- An optional `remote_data`-marked smoke test (`pytest-remotedata` test
    dependency) drives the real Ballet CNN on a bundled frame end-to-end,
    downloading the centroider weights from HuggingFace. It is skipped by default
    and runs only under `pytest --remote-data=any`, in a dedicated, non-blocking
    CI job. The plugin's socket-blocking also guards the rest of the suite from
    accidental network access.
- Two per-frame QA-manifest columns instrumenting the centroid-drift flag
    (#60): `n_centroid_drift` (drift-flagged stars in the frame) and
    `n_drift_rejected` (drift-flagged stars that pass the quality cuts — the
    marginal effect a future drift gate would have). Whether the flag should
    gate output will be decided from these counts on real nights.
- `InstrumentProfile.contamination_seeing_margin` (default 1.25): the
    once-per-batch bright-neighbour contamination flag is evaluated at
    `first-frame FWHM x margin`, so pairs that would become contaminated as
    seeing softens during the night are dropped up front (#64).
- The instrument `header_map` understands `xbayroff`, and `YBAYROFF`/`XBAYROFF`
    now follow the standard Siril/N.I.N.A. convention (row/column offsets into
    the Bayer pattern). The masks are pinned to Han Kleijn's public-domain
    Bayer conformance images in the test suite (#51).
- `DegenerateBayerChannelError` (a `FrameError`): a frame whose CFA channel
    sample is empty or has zero variance is now skipped cleanly instead of
    silently dividing by zero during Bayer balancing (#61).

### Changed

- Ballet CNN centroid inference now runs as a pure-numpy forward pass
    (`bandaid.ballet_numpy.NumpyBallet`), so JAX/Flax/Optax are no longer runtime
    dependencies: the `eloy[jax]` requirement became plain `eloy` (same pin) plus
    a direct `huggingface_hub` dependency for the weights download, making
    installs ~270 MB lighter. Results are identical to the JAX model within
    float32 round-off, and the `cnn=` pipeline parameter stays duck-typed (any
    object with a `centroid(cutouts) -> (N, 2)` method still works). The `train`
    extra still provides JAX for the training scripts.

### Changed (breaking)

- `measure_photometry` and `build_photometry_table` renamed their keyword-only
    `relative_radii=` argument to `radii=`. Both are re-exported from the package
    root, so calls using `relative_radii=` now raise `TypeError`; pass `radii=`
    instead. No deprecated alias is provided.
- `prepare_batch` dropped its `gaia_mag_limit=` and `contaminant_mag_limit=`
    keyword arguments. Set these via
    `config=PhotometryConfig(source_selection=SourceSelectionConfig(gaia_mag_limit=...))`
    instead.
- The per-star `sky` column is gone from the photometry tables and the custom
    writer contract (#52). Once its 2-4x scale error was fixed it was
    byte-identical to `bkgd_count`, so the duplicate was deleted; the QA
    manifest's `sky_median` is now a true median of the per-star, per-pixel
    `bkgd_count`.
- `measure_photometry` dropped its unused `aligned_coords` parameter — every
    measurement, including `peak_count`, is anchored at the measured centroids
    (#54, #61).
- The contamination-model tuning parameters of `min_separation_fwhm`,
    `neighbor_contamination_flag`, and `neighbor_contamination_flag_sky`
    (`tolerance`, `beta`, `aperture_radius_fwhm`, `target_mask`) are now
    keyword-only (#61).
- `append_l4` defaults to `True` throughout the API (`generate_bayer_masks`,
    `prepare_batch`), matching `photometer_frames` and the CLI, so composing
    the pipeline by hand yields the same channels as the CLI (#61).

### Fixed

- The per-frame FWHM fit now uses only the brightest unsaturated detections
    (`InstrumentProfile.fwhm_n_stars`, default 25) instead of every detection.
    Bayer-balanced detection yields thousands of faint sources whose CNN
    re-centroiding both dominated the per-frame runtime (~4x slower) and inflated
    the fitted FWHM (~8.6 px vs the true ~2.8 px), over-sizing every
    FWHM-scaled aperture. Capping recovers the true FWHM and the original speed
    without changing which stars are photometered.
- The July 2026 top-to-bottom code review (#63) fixed every confirmed
    calculation bug (issues #51-#62, #64; PRs #65-#76):
    - `peak_count` is now the target's own peak: measured on the star's Bayer
        channel (mask applied), in a ~2 x FWHM box anchored at the measured
        centroid, instead of an unmasked fixed 25x25 box at the catalog-aligned
        position that let bright neighbours masquerade as the target and made
        the TR/TG/TB peaks bit-identical (#54). A failed (non-finite) centroid
        now yields NaN outputs for that row instead of raising.
    - The bright-neighbour contamination model uses the largest configured
        aperture radius instead of a hard-coded 1 x FWHM, and normalises the
        tolerance by the aperture-enclosed target flux, so
        `contamination_tolerance` bounds contamination relative to the flux
        actually measured (#53). Equal-magnitude threshold moves from ~2.18 to
        ~2.30 FWHM at the defaults.
    - Gaia DR2 positions are proper-motion propagated from J2015.5 to the
        frame's observation epoch, so high-PM stars are photometered where the
        frames actually see them (~1.1 arcsec per 100 mas/yr accumulated
        drift); a frame without a parseable `DATE-OBS` now fails with a clear
        metadata error (#56).
    - `prepare_batch` measures its batch-gating first-frame FWHM with the same
        detection settings as the per-frame path (bayer-balanced detection,
        brightest-N cap), removing a systematic FWHM mismatch (#55).
    - The `time` column records mid-exposure (start + `exposure x stack / 2`)
        instead of exposure start (#57), and `good_star_mask` bounds use the
        pixel-center convention `[-0.5, dim - 0.5)` on both axes (#57).
    - `check_frame_consistency`, the airmass derivation, and the `time` column
        resolve header keywords through the instrument `header_map` instead of
        hard-coded Seestar names, so non-Seestar dialects work; Seestar50
        output is unchanged (#59).
    - A default `bandaid process` run now reports per-frame failures on stderr
        (WARNING and up) and exits non-zero when every frame fails (#58).
    - Docs sweep: removed cookiecutter boilerplate, fixed broken anchors and
        stale references, filled in `pyproject.toml` metadata, and replaced the
        placeholder package docstring (#62).

## [0.1.0] - (1979-01-01)

- First release
