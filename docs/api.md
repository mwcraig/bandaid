# API reference

The Python entry points behind the `bandaid` command, auto-generated from the
source. The whole CLI flow is one call to
[`photometer_frames`](#bandaid.photometer_frames); the rest of this page is the
configuration, instrument-registry, and exception surface you need to drive or
catch it.

Every name here is importable directly from the top-level package, e.g.
`from bandaid import photometer_frames`.

## Running a batch

The high-level convenience and the two functions it is built from. See
[Getting started](getting_started.md) for a worked example and the batch model.

::: bandaid.photometer_frames

::: bandaid.prepare_batch

::: bandaid.process_batch

::: bandaid.expand_frame_paths

## Processing one frame

The per-frame step `process_batch` loops over, for a caller that gets frames
one at a time. `process_frame` reports a `FrameResult` whose `qa` record is
the frame's QA-manifest row, keyed by `QA_MANIFEST_COLUMNS`; a change to those
columns bumps `FrameResult.schema_version`. `write_qa_manifest` writes such
records as the CSV manifest, and `usable_filters` is the per-filter
keep-or-drop rule of the default writer.

::: bandaid.process_frame

::: bandaid.FrameResult

::: bandaid.FrameOutcome

::: bandaid.QA_MANIFEST_COLUMNS

::: bandaid.write_qa_manifest

::: bandaid.qa_record_ok

::: bandaid.qa_record_skipped

::: bandaid.qa_record_error

::: bandaid.usable_filters

## Configuration

The immutable, validated configuration objects. See
[Configuration](configuration.md) for which knobs to change.

::: bandaid.PhotometryConfig

::: bandaid.ApertureConfig

::: bandaid.SourceSelectionConfig

::: bandaid.DriftConfig

::: bandaid.CentroidConfig

::: bandaid.InstrumentProfile

::: bandaid.HeaderMatchRule

## Instrument registry

Resolve, register, list, and auto-detect instrument profiles. See
[Instrument profiles](instrument_profiles.md).

::: bandaid.load_instrument

::: bandaid.register_instrument

::: bandaid.available_instruments

::: bandaid.detect_instrument

## Exceptions

The two failure classes the batch driver distinguishes. A `FrameError` (and its
subclasses) is **recoverable per-frame** — the frame is skipped and the batch
continues — so it is safe to catch around a single-frame call. A `BatchPrepError`
is **fatal**: the once-per-batch preparation could not be built, so the run must
stop. See [Troubleshooting](troubleshooting.md) for what triggers each.

::: bandaid.BandaidError

::: bandaid.FrameError

::: bandaid.TooFewStarsError

::: bandaid.WCSSolveError

::: bandaid.WCSScaleError

::: bandaid.WCSPointingError

::: bandaid.FrameMetadataError

::: bandaid.NoUsableStarsError

::: bandaid.CentroidPlaneError

::: bandaid.BatchPrepError

::: bandaid.InstrumentDetectionError
