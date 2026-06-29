# Command-line usage

Installing bandaid puts a `bandaid` command on your path. It wraps the same
batch-photometry flow you would otherwise drive from a notebook
(`prepare_batch` → `process_batch`), so you can photometer a night of frames and
inspect instruments/configuration without writing any Python.

```bash
$ bandaid --help
```

To drive the same flow from Python, call `bandaid.photometer_frames` (defined in
`bandaid.scripts`), which performs the identical file-expansion + `prepare_batch`
→ `process_batch` flow this command wraps.

## `bandaid process` — photometer a batch of frames

The main command. Point it at your frames and it builds the Ballet centroider,
prepares the batch from the first frame, photometers every frame, and writes a
`.star` file per frame plus a QA manifest.

```bash
# Photometer every FITS frame in a directory, writing results to ./out
$ bandaid process night-of-2026-06-27/ -o out/

# Or pass a glob / explicit files
$ bandaid process "night/*.fit" -o out/
```

Frame arguments may be **directories** (expanded to the FITS frames they
contain), **glob patterns** (filtered to FITS frames), or **individual file
paths**. The combined list is sorted so the batch order is deterministic, and
de-duplicated by *resolved* path: the same file reached two ways (a directory and
an explicit path, `a.fit` vs `./a.fit`) is processed once, while two distinct
files that merely share a name in different directories are both kept. The first
frame seeds the once-per-batch preparation.

Output names follow the input layout. When every frame comes from a **single
directory** — the typical one-night run — the results are written flat as
`<stem>.star` directly in the output directory. When frames come from a **mix of
directories**, the output mirrors the source tree as `<dirname>/<stem>.star`, so
identically named frames from different directories stay distinct without
mangling their names (two source directories that share a name are disambiguated
with a numeric suffix on the subdirectory).

| Option                             | Default          | Meaning                                                                    |
| ---------------------------------- | ---------------- | -------------------------------------------------------------------------- |
| `FILES...`                         | —                | Frames to photometer: directories, globs, and/or paths.                    |
| `-o, --output-dir DIR`             | `.`              | Where to write the `.star` files and QA manifest.                          |
| `--instrument NAME`                | `Seestar50`      | A bundled/registered instrument profile (see `bandaid instrument list`).   |
| `--profile FILE`                   | —                | An instrument-profile JSON file (alternative to `--instrument`).           |
| `--config FILE`                    | —                | A full `PhotometryConfig` JSON file (see `bandaid config init`).           |
| `--weights PATH`                   | downloads        | Ballet centroider weights; omit to download the defaults from HuggingFace. |
| `--user-metadata FILE`             | `{}`             | A JSON object of per-frame user-specific metadata to record.               |
| `--append-l4 / --no-append-l4`     | on               | Add a full-frame L4 luminance channel to the Bayer masks.                  |
| `--fail-fast / --no-fail-fast`     | `--no-fail-fast` | Re-raise unexpected per-frame errors instead of skipping the frame.        |
| `--output-suffix SUFFIX`           | `.star`          | Suffix for the per-frame output files.                                     |
| `--qa-manifest / --no-qa-manifest` | on               | Write a per-frame QA manifest alongside the `.star` files.                 |

`--config` loads the full configuration; an explicit `--instrument` or
`--profile` then overrides only its instrument. Use one of `--instrument` /
`--profile`, not both.

The default `--no-fail-fast` is the friendlier choice for unattended overnight
runs: a single bad frame is logged and skipped rather than aborting the batch.
Pass `--fail-fast` while debugging so unexpected errors surface immediately.

## `bandaid instrument` — inspect instrument profiles

```bash
# List the profiles the pipeline can resolve
$ bandaid instrument list
Seestar50

# Print one profile's settings as JSON
$ bandaid instrument show Seestar50
```

`instrument show` emits the profile as JSON you can save and edit, then feed back
to `bandaid process --profile`.

## `bandaid config` — create and validate configuration

```bash
# Write a default config you can edit
$ bandaid config init -o config.json

# Check that an edited config is valid before a run
$ bandaid config validate config.json
```

`config init` writes (or prints, with no `-o`) a default `PhotometryConfig` as
JSON. `config validate` parses a config file and reports any validation errors
with a non-zero exit code, so you catch a typo before it fails deep in a batch.

## `bandaid weights` — get the default Ballet weights

```bash
# Print the cached path of the default centroider weights
$ bandaid weights

# Copy them somewhere reusable
$ bandaid weights -o weights.npz
```

The default weights are downloaded from HuggingFace on first use and cached
thereafter. `bandaid weights` prints the cached `.npz` path (downloading it if
needed) so you can reuse it with `bandaid process --weights`, avoiding a fresh
download on every run. See
[Training the Ballet centroider](training_the_ballet_centroider.md) for when you
might train and supply your own weights instead.
