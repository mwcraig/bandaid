# bandaid user-documentation outline

A proposed structure for the user-facing docs (the MkDocs site under `docs/`).
This is an **outline / spec**, not the finished pages.

## Who this is for & how to write it

- **Reader:** an intermediate Python user — comfortable with pip, virtual
    environments, and the REPL — who has a folder of Seestar FITS frames and wants
    photometry out. Not assumed to be an expert in photometry internals.
- **Emphasis:** *how to use the tools*, not how the algorithms work. Keep
    algorithm explanation to a one-line "what this does for you" and link out.
- **Voice:** task-oriented. Every page opens with a runnable example.
- **Two interfaces, every time:** show the CLI command first (no Python needed),
    then the equivalent Python call.

## The mental model to establish up front

Three ideas the rest of the docs lean on:

1. **Positions come from Gaia, not source detection.** bandaid measures
    *forced* photometry at the positions of a magnitude-limited Gaia catalog — you
    do not detect your own sources. This is the most counterintuitive point for
    new users and belongs on the home page.
1. **Batch model.** One slow preparation pass on the first frame
    (`prepare_batch`) sets up the Gaia catalog, plate scale, FOV, Bayer masks, and
    centroiding model; then a fast per-frame loop (`process_batch`) reuses it.
1. **Output shape.** One `.star` file per Bayer filter per frame, plus a single
    `qa_manifest.csv` for the run.

______________________________________________________________________

## Page-by-page outline

Legend: ✅ exists and fits · ✳️ exists, needs rewrite · 🆕 new page.

### 1. Home / Overview — `docs/index.md` ✳️

Replace the cookiecutter placeholder. One paragraph on what bandaid is (Seestar →
AAVSO-style photometry, retired Aug 1 2026), the three-bullet mental model above,
"who this is for," and links to Installation and Quick start.

### 2. Installation — `docs/installation.md` 🆕

- `pip install bandaid`; Python ≥ 3.12.
- Note the git-based dependencies pulled in automatically: `eloy` (detection and
    photometry building blocks; centroid inference is pure numpy, no JAX) and
    `aavso-starlist-schema` (the `.star` schema). Both are public.
    `huggingface_hub` handles the Ballet weights download.
- Editable/dev install for contributors.
- First run downloads the default Ballet **weights** from HuggingFace and caches
    them; pre-fetch with `bandaid weights`, or point at a local `.npz` with
    `--weights`.

### 3. Quick start / Tutorial — `docs/getting_started.md` 🆕 *(highest-value gap)*

End-to-end in ~10 minutes:

1. `bandaid process night-of-2026-06-27/ -o out/ -v`
1. What lands in `out/`: the `*.star` files (per filter TR/TG/TB + optional L4)
    and `qa_manifest.csv`.
1. How to read the QA manifest to sanity-check the run (`status`, `n_detected`,
    `fwhm`, `wcs_solved`, `n_good_stars`).
1. The same run from Python with `photometer_frames(...)`.
1. "Where to go next" → Configuration, Adding your instrument.

### 4. Command-line guide — `docs/command_line.md` ✅

Keep; cross-link from the tutorial. Cover `bandaid process` (flags grouped:
inputs, output, instrument/config selection, weights, robustness `--fail-fast`,
verbosity `-v/-vv`), `bandaid instrument list|show`, `bandaid config init|validate`, `bandaid weights`.

### 5. Configuration — `docs/configuration.md` ✳️

Verify field names against the current `PhotometryConfig`. Document the
`config init → edit JSON → config validate → --config` loop and the handful of
knobs an intermediate user actually changes:

- `apertures`: `radii` (FWHM units), `gap`, `annulus_width`
- `source_selection`: `gaia_mag_limit`, `contaminant_mag_offset`
- `drift`: drift tolerances (link to the drift page)
- `instrument`: link to the instrument-profile page

Show the CLI (`--config`) and Python (`PhotometryConfig(...)`) forms side by side.

### 6. Adding a new instrument profile — `docs/instrument_profiles.md` ✳️ *(highlighted workflow)*

Expand into a full how-to:

- **What a profile is:** pure data (an `InstrumentProfile`), no code required. It
    does two jobs — (a) tune detection / PSF / contamination for your optics, and
    (b) the `header_map` translates *your* FITS header keywords into the fields
    bandaid needs.
- **Field reference**, plain language, with Seestar50 defaults as examples:
    `name`, `thresh`, `detection_opening`, `fwhm_cutout_half`, `fwhm_n_stars`,
    `contamination_tolerance`, `moffat_beta`, and `header_map` (worked
    `@KEYWORD` / index-directive example covering pixel scale, gain, ADC depth,
    Bayer pattern, obs-time, RA/Dec). Call out that `egain` and
    `largest_usable_adu_value` are required for header parsing.
- **Start from the bundled profile:**
    `bandaid instrument show Seestar50 > my_scope.json`, then edit.
- **Three ways to use your profile**, easiest → most permanent:
    1. *Ad-hoc file* — `bandaid process frames/ --profile my_scope.json`, or
        `PhotometryConfig(instrument=InstrumentProfile.from_file("my_scope.json"))`.
    1. *Register in-process* — `register_instrument(profile)`, then refer to it by
        name via `--instrument <name>` / `load_instrument("name")`.
    1. *Bundle it (contributor path)* — drop
        `src/bandaid/meta_json_files/<Name>/profile.json`, update the bundled-names
        test, open a PR; it is auto-discovered with no other code changes. Link to
        `contributing.md`.
- **Validate before a long run:** `bandaid config validate` (or let `from_file`
    raise). List the common Pydantic errors (values must be > 0, etc.).
- **Verify it parses your headers:** run a single frame with `-vv` and confirm
    metadata (pixel scale, gain, RA/Dec) resolved and a `.star` file was written.

### 7. Understanding the output — `docs/outputs.md` 🆕 (or fold into the tutorial)

- `.star` file: `StarList` JSON; the columns a user reads (`tot_count`,
    `count_err`, `sky`, `x/y`, `ra/dec`, flags); one file per Bayer filter.
- `qa_manifest.csv`: column-by-column; how to spot bad frames.
- **Data-quality flags:** `centroid_drift` (flag-only, no rows dropped) and
    `contaminated` (drops the row) — document honestly that contamination flagging
    is **not yet wired in**, so no `contaminated` column is currently written.

### 8. Data-quality / centroid-drift check — `docs/centroid_drift_check.md` ✅

Keep as the reference for what the drift flag means and its thresholds.

### 9. Advanced: training your own centroider — `docs/training_the_ballet_centroider.md` ✅

Keep; mark clearly as advanced/optional. Most users just use the default weights.

### 10. Python API reference — `docs/api.md` 🆕

Auto-generated via mkdocstrings (already configured in `mkdocs.yml`). Curate the
user-facing surface: `photometer_frames`, `prepare_batch`, `process_batch`,
`expand_frame_paths`; the config classes; `load_instrument` /
`register_instrument` / `available_instruments`; and the exception hierarchy
(`FrameError` is recoverable per-frame, `BatchPrepError` aborts the run) so users
know what is safe to catch.

### 11. Troubleshooting / FAQ — `docs/troubleshooting.md` 🆕

Symptom-first entries:

- "Too few stars" / "WCS won't solve."
- Frame skipped vs whole run aborted (`FrameError` vs `BatchPrepError`,
    `--fail-fast`).
- Weights download failed (offline → `--weights`).
- Header field missing (`FrameMetadataError` → fix the `header_map`).

### 12. Contributing & code style — `docs/contributing.md` ✅ + `docs/code_style.md` ✅

Keep; developer-facing.

______________________________________________________________________

## Things to get right while writing

- Lead with the **Gaia forced-photometry** concept — it is the biggest surprise.
- Explain the **batch model**; it shapes both the CLI behavior and the Python API.
- Document the **un-wired `contaminated` flag** honestly.
- The package may be **renamed** (the name `bandaid` is taken on PyPI), so keep
    install instructions in one place to make a rename a one-spot edit.
- Keep `docs/index.md` and `README.md` in sync — both currently carry placeholder
    cookiecutter text.
