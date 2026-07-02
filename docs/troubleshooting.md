# Troubleshooting

Symptom-first fixes for the things that actually go wrong on a night of frames.
Most per-frame problems show up as a non-`ok` `status` in the
[`qa_manifest.csv`](outputs.md#qa_manifestcsv); start there.

## A few frames were skipped (the rest of the run finished)

This is the normal, designed behavior: a single bad frame is logged and skipped
so an overnight batch still produces results. The frame's manifest `status` names
the reason — a `FrameError` subclass:

- **`skipped: TooFewStarsError`** — too few stars were detected to work with.
    Usually a clouded-over, trailed, or badly defocused frame. Check `n_detected`
    and `fwhm` in the manifest for that row.
- **`skipped: WCSSolveError`** — the plate solve failed (`wcs_solved` is
    `False`). The field could not be matched against Gaia, typically because too
    few stars were detected or the pointing was off.
- **`skipped: FrameMetadataError`** — a required FITS header keyword was missing
    or unparseable (see *Header field missing* below).
- **`skipped: NoUsableStarsError`** — stars were detected but none survived
    photometry filtering, so the frame yields no output.

To debug an individual skipped frame, run **just that frame** with `-vv`:

```bash
$ bandaid process the_bad_frame.fit -o /tmp/debug -vv
```

The debug log shows the chained cause (e.g. the underlying twirl traceback for a
WCS failure) that the one-line skip message summarizes.

## The whole run aborted before finishing

A skipped frame is recoverable; a `BatchPrepError` is not. The once-per-batch
preparation is built from the **first** frame, and if that fails — too few stars
to measure an FWHM, or Gaia returns too few reference stars for the field — there
is nothing to process the rest of the batch against, so the whole run stops.

Fix: make sure the **first** frame in the batch is a good one (the batch is
processed in sorted order). Point `bandaid process` at a cleaner starting frame,
or prune the obviously bad frames before the run.

### `--fail-fast` vs the default

By default (`--no-fail-fast`) an *unexpected* error on a frame — a genuine bug,
not a recognized bad-frame condition — is logged and the run continues. While
debugging, add `--fail-fast` so such errors are re-raised immediately instead of
being swallowed:

```bash
$ bandaid process night/ -o out/ --fail-fast -vv
```

`FrameError` conditions (the bad-frame cases above) are always skipped regardless
of this flag; `--fail-fast` only changes how *unexpected* errors are handled.

## Weights download failed (offline / firewalled)

The first run downloads the Ballet weights from HuggingFace. If that machine is
offline or behind a firewall the download fails. Pre-fetch the weights elsewhere
and point at the local file:

```bash
# On a machine with network access:
$ bandaid weights -o ballet_weights.npz

# Then on the offline machine:
$ bandaid process night/ -o out/ --weights ballet_weights.npz
```

From Python, pass `weights="ballet_weights.npz"` to `photometer_frames`. See
[Getting started](getting_started.md#the-ballet-weights-first-run).

## Header field missing (`FrameMetadataError`)

`FrameMetadataError` means a FITS header keyword the pipeline needs was missing
or could not be parsed for that frame. The most common cause is using a telescope
whose headers use different keyword names than the bundled Seestar50 profile
expects.

The fix is a `header_map` that translates *your* telescope's header keywords into
the metadata bandaid needs. Build (or adjust) an instrument profile and verify it
parses a single frame with `-vv`:

```bash
$ bandaid process one_frame.fit --profile my_scope.json -o /tmp/check -vv
```

See [Instrument profiles](instrument_profiles.md) for the `header_map` directive
language and the full list of keys a profile must resolve.
