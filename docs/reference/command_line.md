# Command line

`bandaid` wraps `photometer_frames` (file expansion, `prepare_batch`, `process_batch`).
For a walkthrough see [Getting started](../getting_started.md).

```text
bandaid [--version] [--help] COMMAND [ARGS]...
```

| Command                               | Purpose                                                              |
| ------------------------------------- | -------------------------------------------------------------------- |
| `bandaid process FILES...`            | Photometer a batch of frames into `.star` files.                     |
| `bandaid instrument list`             | List resolvable instrument-profile names.                            |
| `bandaid instrument show NAME`        | Print one profile as JSON.                                           |
| `bandaid config init [-o FILE]`       | Write a default `PhotometryConfig` (stdout without `-o`).            |
| `bandaid config validate CONFIG_FILE` | Validate a config file; non-zero exit on error.                      |
| `bandaid weights [-o FILE]`           | Print the cached default Ballet weights path; `-o` also copies them. |

## `bandaid process`

`FILES...` are directories (expanded to their FITS frames), glob patterns, or
file paths. The combined list is sorted and de-duplicated by resolved path; the
first frame seeds the once-per-batch preparation. Output naming is described in
[Output files](outputs.md).

| Option                             | Default          | Meaning                                                                                          |
| ---------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------ |
| `-o, --output-dir DIRECTORY`       | `.`              | Directory for the `.star` files and QA manifest.                                                 |
| `--instrument NAME`                | auto-detect      | Bundled or registered instrument profile.                                                        |
| `--profile FILE`                   | none             | Instrument-profile JSON file (alternative to `--instrument`).                                    |
| `--config FILE`                    | none             | Full `PhotometryConfig` JSON file.                                                               |
| `--gaia-mag-limit FLOAT`           | `15.0`           | Overrides `source_selection.gaia_mag_limit`.                                                     |
| `--min-snr FLOAT`                  | `2.0`            | Overrides `source_selection.min_snr`.                                                            |
| `--weights FILE`                   | downloaded       | Ballet centroider weights; omitted means the default weights are downloaded.                     |
| `--user-metadata FILE`             | none             | JSON object of per-frame metadata overrides (see [Instrument profiles](instrument_profiles.md)). |
| `--forced-targets FILE`            | none             | CSV/ECSV with `ra`, `dec` columns (case-insensitive; optional `name` is ignored).                |
| `--build-l4 / --no-build-l4`       | `--build-l4`     | Also produce the full-frame `L4` luminance channel.                                              |
| `--fail-fast / --no-fail-fast`     | `--no-fail-fast` | Re-raise unexpected per-frame errors instead of skipping the frame.                              |
| `--output-format TEXT`             | `starlist`       | Registered output writer.                                                                        |
| `--output-suffix TEXT`             | `.star`          | Suffix of the per-frame output files.                                                            |
| `--qa-manifest / --no-qa-manifest` | `--qa-manifest`  | Write `qa_manifest.csv`.                                                                         |
| `-v, --verbose`                    | off              | `-v` streams per-frame progress (INFO); `-vv` adds per-stage DEBUG detail.                       |
| `--log-file FILE`                  | none             | Also write log records to this file, at the terminal's level.                                    |

Rules:

- `--config` loads the full configuration. `--instrument` or `--profile` then
    replaces only its instrument; give at most one of the two.
- `--gaia-mag-limit` and `--min-snr` replace only those two `source_selection`
    fields in the loaded configuration.
- Instrument selection (explicit option, config, or header auto-detection) follows
    the precedence list in [Instrument profiles](instrument_profiles.md).
- `--forced-targets` values are ICRS degrees. An ECSV angular unit (for example
    `hourangle`) is converted; a non-angular unit, a blank or non-finite value, or
    an RA outside `[0, 360)` is an error. Forced targets skip the contamination
    check; see [Output files](outputs.md).
- Skip and error warnings go to stderr even without `-v`.
- Exit status is non-zero when every frame fails, or when the arguments expand
    to no frames, an input file fails validation, or batch preparation fails.
    A partial failure exits 0.

The default weights are downloaded on first use; see
[Troubleshooting](../howto/troubleshooting.md) for offline use.

## Other commands

- `instrument show NAME` prints JSON that can be edited and passed back with
    `bandaid process --profile`.
- `config init` writes a config whose `instrument` is `null` (auto-detect).
- `weights -o FILE` copies the cached weights to `FILE` in addition to printing
    their path, for reuse with `--weights`.
