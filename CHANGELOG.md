# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `bandaid process -v` now logs the frame and photometry-target counts, one outcome line
    per frame (WCS solved or the skip reason, stars measured, FWHM, elapsed time)
    and a closing total; `-vv` adds per-stage detail for each frame (plate-solve
    pool, solved scale and pointing offset, star counts after each cut, centroid
    methods and offset-plane fit, files written). Before,
    the two levels differed by about one line per frame.
- `CentroidConfig.max_plane_rms_pix`: a frame whose offset plane
    fits its own CNN-centroided stars with a larger rms is skipped with the new
    `CentroidPlaneError` (manifest status `skipped: CentroidPlaneError`, with
    `plane_rms` filled) instead of being photometered at poorly modelled positions
    (#149).
- `CentroidConfig` (on `PhotometryConfig.centroid`) holds the settings of the
    measured-versus-modelled centroid policy, on by default through
    `model_faint_positions`. See `docs/measured_vs_modelled_positions.md` (#131, #106, #147).
- The eloy table gets a per-row `centroid_method` column (`cnn`, `plane` or
    `fallback_cnn`) and the QA manifest gains `g_cut`, `n_cnn_class` and the
    `plane_*` columns (#131, #147).
- A documentation page on measured versus modelled star positions, including the
    low-SNR bias curve of the CNN centroid (#130, #147).
- `InstrumentProfile.wcs_pointing_tolerance` (degrees, default none): the largest
    separation allowed between a solved frame center and its header pointing,
    instead of one field radius.
- The QA manifest gains `wcs_pixscale` (also filled for frames rejected for their
    scale), `solve_offset_deg` and `n_snr20`, so degraded and false plate solves
    are visible without a rerun.
- `InstrumentProfile.solve_pool_radius_scale` (default `0.9`): each frame plate-solves
    against only the catalog stars within this fraction of `fov_rad` of its own
    header pointing, instead of the whole batch catalog (#133).
- `SourceSelectionConfig.gaia_row_limit` (default `10000`): the row limit of the
    batch Gaia query, scaled with cone area. The query is filtered server-side to
    `Gmag <= contaminant_mag_limit`; a result that hits the limit raises
    `CatalogTruncationError` if targets were lost and warns if only contaminants
    were (#133).
- The QA manifest gains `pointing_offset_deg`, each frame's header-center offset
    from the batch center in degrees (#133).
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
- `bandaid process --log-file PATH` also writes log records to a file, at the
    same level as the terminal (#92).
- `SourceSelectionConfig.min_snr` (default `2.0`): a minimum-SNR floor a star
    must clear to reach the output, applied by `good_star_mask` alongside its
    existing flux/error/bounds/contamination cuts. New `bandaid process`
    `--gaia-mag-limit` and `--min-snr` flags override the corresponding
    `source_selection` fields without needing a full `--config` file (#101,
    #102).
- `bandaid process --forced-targets FILE` (and `forced_targets=` on
    `prepare_batch`/`photometer_frames`): photometer extra sky positions
    absent from the Gaia catalog, e.g. a nova (#100). `FILE` is a CSV/ECSV
    table with `ra`/`dec` in ICRS degrees. Forced targets skip the
    Gaia-magnitude contamination model; all other quality cuts still apply
    (see the docs for details).
- Instrument auto-detection from the FITS header. A new `HeaderMatchRule`
    model and `InstrumentProfile.header_match` field (a tuple of
    keyword/pattern rules; empty by default -- including on a bare
    `InstrumentProfile()` -- so device identity is opt-in) drive a new
    `detect_instrument(header)` in `bandaid.instruments`: it matches the
    header against every bundled/registered profile's rules and returns the
    single match, or raises the new `InstrumentDetectionError` (a
    `FrameMetadataError`, and so a `FrameError`) naming the header values it checked and the
    available/ambiguous profile names. The bundled Seestar50 profile now
    carries the rule `INSTRUME == "Seestar S50"` (deliberately not
    `TELESCOP`, which embeds a per-device serial on real hardware, e.g.
    `S50_0e597e9b`). `prepare_batch` and `prepare_image` both resolve a
    `None` `config.instrument` this way, from the first header they have in
    hand; `check_frame_consistency` also rejects a later frame in the batch
    whose header does not match the batch instrument's rules when the batch
    instrument's `header_match` is non-empty. An explicit choice
    (`--instrument`/`--profile`/`--config`) is exempt only from the "header
    matches no registered instrument" outcome; a frame whose header
    positively identifies a different registered instrument, or matches more
    than one, is still rejected. The guard resolves the
    later frame through `detect_instrument` itself, so a header that is
    ambiguous across registered profiles is rejected too, and it runs before
    the header is resolved through the batch instrument's `header_map`, so a
    mixed-in frame is reported as a mismatch rather than as a missing header
    keyword. `InstrumentProfile.matches_header(header)` is the shared
    predicate. `BatchPrep` now requires a `config` whose `instrument` is
    resolved. See `docs/instrument_profiles.md`.
- `register_instrument` checks for conflicts before touching the registry:
    a name that already resolves (bundled or registered) is refused unless
    `replace=True` is passed, and a `header_match` rule that duplicates a
    differently named profile's rule (same keyword and value) raises
    `ValueError` at registration instead of surfacing later as an
    "ambiguous instrument" detection error on a real frame.

### Changed

- Only the brightest catalog stars keep their CNN centroid; the rest are output at
    their projected position plus a per-frame fitted plane, which moves faint-star
    positions by about 1 to 1.5 px and tightens their light curves. Set
    `CentroidConfig(model_faint_positions=False)` for the previous behavior
    (#131, #106, #147).
- `centroid_drift` is now measured from the plane position and only for
    CNN-measured stars; earlier `n_centroid_drift` counts are not comparable
    (#131, #147).
- Breaking for `.star` and table output: catalog stars projected within
    `PhotometryConfig.edge_margin_px` (default `10.0` px) of a frame edge, or off
    the frame, are no longer measured, because the centroiding CNN's fill-padded
    cutout misplaces them (#129). The QA manifest gains `n_edge_dropped`. See
    "Frame-edge margin" in `docs/configuration.md`.
- Breaking: `build_photometry_table` no longer accepts `peak_cutouts=` or
    `geometry=`; both are cached per frame on `ImageData`. `measure_photometry`
    raises `ValueError` when `geometry` is combined with an explicit `radii` or
    `annulus`. Pipeline output is unchanged (#126).
- Breaking: `align` returns a third value, a `WCSMeasurement` holding the plate
    scale and header-pointing offset that the solve validation measured, so the
    QA manifest reports the numbers the gate used.
- The Seestar50 profile now sets `wcs_scale_tolerance` to `0.005`,
    `wcs_pointing_tolerance` to `0.30` and `pixscale` to the measured `2.376`
    (was `2.4`). A profile that tightens the scale tolerance needs a measured
    `pixscale`, not the nominal one.
- `InstrumentProfile.cone_radius_margin` now defaults to `0.4` deg (was `0.0`), so
    the batch Gaia catalog covers that much pointing drift between frames.
    Widening the cone no longer disturbs plate solving because the solve pool is
    cut per frame (#133).
- `check_frame_consistency` returns the frame's pointing offset and logs a
    warning when it exceeds `cone_radius_margin` but stays within `fov_rad` (the
    frame is processed but only partly covered by the catalog); beyond `fov_rad`
    it still raises `FrameError`, which now carries the offset as
    `pointing_offset`. `prepare_batch`'s minimum-reference-star check now counts
    only the target stars inside the first frame's solve pool (#133).
- `prepare_image` raises `FrameMetadataError` when it has to solve a WCS for a
    frame with no usable header pointing or `fov_rad`, instead of solving without
    the pointing check (#133).
- Output now enforces a minimum SNR of `2.0` by default (`good_star_mask`,
    `SourceSelectionConfig.min_snr`): a star that used to reach the output at
    any SNR is now dropped if its SNR falls below `2.0`. This is a deliberate
    behavior change; pass `--min-snr 0` (CLI) or
    `SourceSelectionConfig(min_snr=0.0)` to restore the previous, unfiltered
    behavior (#101). The cut is per filter as well as per star: because SNR is
    color-dependent, a color-disadvantaged channel can keep fewer stars than
    its siblings, and a filter in which no star survives is dropped from the
    frame's `.star` output (with a warning naming the dropped filters) while
    the surviving filters still write. A frame is skipped with
    `NoUsableStarsError` only when no filter has any usable star. The QA
    manifest's `dropped_filters` column names any filters dropped this way for
    a `status='ok'` row (empty when none were), so a partial frame is visible
    without grepping the run log.
- Ballet CNN centroiding no longer needs JAX at runtime: inference is a
    pure-numpy forward pass matching the JAX model to float32 round-off, so
    jax/flax/optax drop out of the runtime dependencies (~270 MB lighter;
    `huggingface_hub` added for the weights download). The `cnn=` pipeline
    parameter stays duck-typed; the `train` extra still provides JAX.
- `bandaid.ballet` (renamed from `bandaid.ballet_numpy`) adds a `Ballet`
    selector class: `backend="auto"` (default) uses jax/flax when installed
    (~2-3x faster, identical results) and numpy otherwise, `backend="jax"`
    raises instead of silently falling back, and `BANDAID_BALLET_BACKEND`
    overrides the default. The chosen backend is logged at INFO and exposed
    as `.backend`; the base install stays numpy-only (it keeps bandaid
    running under Pyodide), and `pip install bandaid[jax]` opts in.
- Off-frame catalog stars are now dropped on every frame right after the
    WCS projection, before centroiding and aperture photometry. The Gaia cone
    has radius equal to the frame half-diagonal (so it covers every corner
    under field rotation), which is ~1.8x the frame's area: roughly half the
    catalog is off-frame on every frame, and used to be centroided and
    photometered in all filters only to be discarded at the end by
    `good_star_mask`'s bounds cut. Cutting right after projection is ~14-19%
    faster per frame; `.star` output is unchanged, since the cut is padded by
    8 px so it keeps a superset of what `good_star_mask` keeps (verified
    byte-identical on real Seestar frames, QA manifest included). The one
    visible side effect is that in-memory result tables have fewer rows --
    only ones `good_star_mask` would have dropped anyway (#115).
- Star detection now lives in bandaid (`photometry._detect_stars`) instead of
    `eloy.detection.stars_detection`: same algorithm, identical output
    (verified bit-for-bit against eloy on ~400 real frames across five fields,
    plus byte-identical `.star` output). A separable box-filter opening and a
    copy-free threshold drop detection from ~116 ms to ~57 ms per Seestar
    frame (~12% of the 0.49 s per-frame cost). Non-finite pixels are treated
    as sky (the threshold estimator uses only the finite pixels, as eloy's
    does for NaN), and an all-non-finite or constant frame gives no regions
    without a `RuntimeWarning`. `scikit-image` and `scipy` become
    direct dependencies.
- `calibration_sequence`'s `threshold`, `opening`, `fwhm_cutout_half`, and
    `fwhm_n_stars` now default to `None`, meaning "use the resolved instrument
    profile's value". For a Seestar50, `opening`, `fwhm_cutout_half`, and
    `fwhm_n_stars` resolve to the same values as before. `threshold` was a
    hard-coded `1`, which did not match the Seestar50 profile's `thresh` of
    `0.5`, so a direct call that relied on the default now detects at `0.5`.
    Pass `threshold=1` to keep the old behavior. `prepare_batch` and
    `prepare_image` already used the profile's value, so pipeline output is
    unchanged.

### Changed (breaking)

- `calibration_sequence` returns a `CalibrationResult` instead of a 5-tuple, and
    its `detection_image_out` parameter is removed; read `result.detection_image`
    for the array detection ran on. Pipeline output is unchanged (#123).
- `process_one_image` takes a `build_l4` keyword (default `True`) instead of
    an `"L4": None` entry in `bayer_masks`; `generate_bayer_masks` no longer adds
    that entry and `BatchPrep` carries the flag. Callers that planted `"L4": None`
    must pass `build_l4=True` instead. The `append_l4` argument of `prepare_batch`
    and `photometer_frames` and the CLI's `--append-l4/--no-append-l4` are renamed
    to `build_l4` / `--build-l4/--no-build-l4` (#125).
- `InstrumentProfile.header_center_offset` is replaced by `header_frame` and
    `header_equinox`, which declare the frame of the header RA/DEC; the Seestar50
    profile uses `"fk5"`/`"date"`. A profile that still sets the old key to
    anything but `null` fails validation. `scripts.resolve_field_center` and the
    object-name lookup are removed; the Gaia cone is always centered on the
    converted header pointing (#132).
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
- `calculate_l4_quantities(by_filter_data, egain)` now returns the L4 table
    instead of filling a caller-supplied one in place, and the L4 channel
    skips its own full-frame `measure_photometry` pass: every phot-derived
    column was overwritten by the TR/TG/TB recombination anyway (#21), so the
    table is built from the RGB tables alone (~10% faster per frame,
    `process_one_image` 0.386 -> 0.348 s; `.star` output byte-identical on
    real Seestar frames apart from the edge-star `peak_count` fix below). The
    columns that cannot be recombined (`fluxes`, `total_bkg`, `bkgd_std`) are
    no longer created for L4 rather than removed afterwards.
    `process_one_image` rejects a mask dict that gives "L4" a mask or lacks
    TR/TG/TB with a `ValueError` before photometering anything.
- `PhotometryConfig.instrument` now defaults to `None` ("resolve from the
    frame header") instead of hard-defaulting to an `InstrumentProfile()`
    (the Seestar50 tuning). A bare `PhotometryConfig()` -- from the CLI with
    no `--instrument`/`--profile`/`--config`, or from Python -- now
    auto-detects the instrument from the first frame's header instead of
    silently assuming a Seestar50; an unmatched or headerless frame now
    raises `InstrumentDetectionError` instead of proceeding with the wrong
    (or a guessed) instrument. This is invisible to Seestar users, whose
    headers auto-detect; pass `--instrument Seestar50` (or any explicit
    profile) to opt back out of auto-detection. `metadata_from_header`'s own
    `profile=None` default changed the same way, from a silent Seestar50
    fallback to `detect_instrument(header)`. `prepare_batch` wraps
    `InstrumentDetectionError` in a batch-fatal `BatchPrepError` (the original
    is its `__cause__`); the direct single-frame entry points `prepare_image`,
    `process_one_image`, and `calibration_sequence` re-raise the same
    `InstrumentDetectionError` with the source file attached, without wrapping
    or chaining, so a per-frame `except FrameError` skip loop keeps working.

### Fixed

- The per-frame FWHM fit is now stable: the Gaussian fit of the stacked PSF runs
    scipy's L-BFGS-B to floating-point noise (`ftol=1e-15`, `gtol=1e-12`, `maxcor=30`)
    instead of eloy's default tolerances, under which the fitted width tracked
    last-bit noise in the stack and could shift by a few percent, or stop short of
    a lower minimum, between near-identical frames. The fit now lives in bandaid
    (`_gaussian_fwhm`, same model, seed and bounds as `eloy.psf.fit_gaussian`);
    FWHMs change by < 1e-4 on all but a handful of frames (#150).
- The command line no longer prints ERFA's "distance overridden" warning when Gaia
    proper motions are propagated, and a field center below the horizon now gives a
    NaN airmass with one log message instead of numpy's "invalid value" warning (#89).
- Deriving airmass no longer attempts an IERS table download, and still works for
    recent frames when the bundled predictions are stale. Airmass can differ from a
    networked run in the last digits (#128).
- Frames that point away from the first frame no longer lose Gaia targets near
    their far edge: the batch catalog is queried over `fov_rad + cone_radius_margin`
    and each frame plate-solves against the stars near its own header pointing (#133).
- The Gaia cone is now centered correctly at every RA. The fixed Seestar header
    offset was precession from equinox-of-date to J2000 and only matched fields
    near RA ~11.5 h; precession is now applied to each frame's pointing (#132).
- The L4 `peak_count` no longer goes NaN when just one of TR/TG/TB has a NaN
    peak (a star at the frame edge whose peak box holds no unmasked pixel in
    that channel) while its counts stay finite. The channel peaks are now
    combined with a NaN-ignoring max, so `good_star_mask` keeps a star two
    channels measured fine instead of silently dropping it from the L4 table.
- Each science frame is now opened exactly once per run -- previously up to
    four times, plus a repeat of the first frame across `prepare_batch` and
    `process_batch`. The biggest win is for `.gz` frames, where every reopen
    re-decoded the entire gzip stream (#44).
- Wheel and sdist builds now include the bundled instrument profiles
    (`bandaid/meta_json_files/`): hatch's `only-packages` option excluded the
    directory (no `__init__.py`), so `import bandaid` crashed in any
    non-editable install. Editable dev installs masked the bug. Fixed by
    dropping `only-packages` — hatchling's default already ships everything
    under the package directory.
- The Ballet weights download is pinned to a specific revision of the
    HuggingFace `lgrcia/ballet` repo, so an upstream re-upload can no longer
    silently change centroid results.
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
