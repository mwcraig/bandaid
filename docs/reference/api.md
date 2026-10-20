# API reference

Generated from the source; every name imports from `bandaid`.

## Running a batch

`photometer_frames` does what `bandaid process` does; the other three are the
steps it is built from. See [Getting started](../getting_started.md).

::: bandaid.photometer_frames

::: bandaid.prepare_batch

::: bandaid.process_batch

::: bandaid.expand_frame_paths

## Processing one frame

`process_frame` is the per-frame step `process_batch` loops over. Its
`FrameResult.qa` record is the frame's QA-manifest row, keyed by
`QA_MANIFEST_COLUMNS`; columns are listed in [Output files](outputs.md).

::: bandaid.process_frame

::: bandaid.FrameResult

::: bandaid.FrameOutcome

::: bandaid.QA_MANIFEST_COLUMNS

::: bandaid.write_qa_manifest

::: bandaid.qa_record_ok

::: bandaid.qa_record_skipped

::: bandaid.qa_record_error

::: bandaid.usable_filters

::: bandaid.good_star_mask

## Configuration

Fields and defaults are in [Configuration](configuration.md).

::: bandaid.PhotometryConfig

::: bandaid.ApertureConfig

::: bandaid.SourceSelectionConfig

::: bandaid.DriftConfig

::: bandaid.CentroidConfig

::: bandaid.InstrumentProfile

::: bandaid.HeaderMatchRule

## Instrument registry

The profile format and selection rules are in
[Instrument profiles](instrument_profiles.md).

::: bandaid.load_instrument

::: bandaid.register_instrument

::: bandaid.available_instruments

::: bandaid.detect_instrument

## Exceptions

`FrameError` and its subclasses are per-frame failures: the batch skips the frame
and continues. `BatchPrepError` and its subclasses are fatal: the once-per-batch
preparation failed and the run stops. `InstrumentDetectionError` is a
`FrameMetadataError` and so a `FrameError`; `prepare_batch` raises it wrapped in
`BatchPrepError`. Causes and fixes are in
[Troubleshooting](../howto/troubleshooting.md).

::: bandaid.BandaidError

::: bandaid.FrameError

::: bandaid.TooFewStarsError

::: bandaid.WCSSolveError

::: bandaid.WCSScaleError

::: bandaid.WCSPointingError

::: bandaid.FrameMetadataError

::: bandaid.NoUsableStarsError

::: bandaid.CentroidPlaneError

::: bandaid.StarListValidationError

::: bandaid.DegenerateBayerChannelError

::: bandaid.InstrumentDetectionError

::: bandaid.BatchPrepError

::: bandaid.CatalogTruncationError
