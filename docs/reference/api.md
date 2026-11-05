# API reference

Generated from the source; every name imports from `bandaid`.

## Running a batch

`photometer_frames` does what `bandaid process` does; the other three are the
steps it is built from. See [Getting started](../getting_started.md).

`photometer_frames` also reads the first frames of the batch to build the
instrument's star-flag state (for the Seestar50 core-replaced flag, the static
list; see [Output files](outputs.md#seestar-core-replaced-stars)). A caller that
runs `prepare_batch` and then `process_batch` or `process_frame` gets no such
state unless it passes the raw frames to `prepare_batch` as `state_frames`:

```python
from astropy.io import fits

from bandaid import prepare_batch
from bandaid.seestar import STATIC_MAP_FRAMES

prep = prepare_batch(
    frames,
    cnn=cnn,
    state_frames=[fits.getdata(f) for f in frames[:STATIC_MAP_FRAMES]],
)
```

This matches what `photometer_frames` does on a clean batch. `prepare_batch`
leaves out a frame whose shape differs from the batch's, but unlike
`photometer_frames` the one-liner raises on a frame it cannot read, does not
leave out a frame from another instrument, and reads the frames even when
`instrument_star_flags` is off.

::: bandaid.photometer_frames

::: bandaid.prepare_batch

::: bandaid.process_batch

::: bandaid.expand_frame_paths

## Processing one frame

`process_frame` is the per-frame step `process_batch` loops over. Its
`FrameResult.qa` record is the frame's QA-manifest row, keyed by
`QA_MANIFEST_COLUMNS`; columns are listed in [Output files](outputs.md). As
with `process_batch`, a `BatchPrep` built without `state_frames` gives no
instrument star-flag state (see [Running a batch](#running-a-batch)).

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

`l4_effective_egain` is the single gain the `L4` star list records; see
[Instrument profiles](instrument_profiles.md).

::: bandaid.l4_effective_egain

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

Star flaggers are named by a profile's `star_flaggers`; see
[Instrument profiles](instrument_profiles.md#star_flaggers).

::: bandaid.register_star_flagger

::: bandaid.StarFlagger

::: bandaid.StarFlagInputs

## Exceptions

`FrameError` and its subclasses are per-frame failures: the batch skips the frame
and continues, except during `prepare_batch`, where a `FrameError` from the first
usable frame stops the run. `BatchPrepError` and its subclasses are fatal: the once-per-batch
preparation failed and the run stops. `InstrumentDetectionError` is a
`FrameMetadataError` and so a `FrameError`; `prepare_batch` raises it wrapped in
`BatchPrepError`. `GaiaQueryError` is neither: the catalog query raises it when
every Gaia server has failed, and `prepare_batch` wraps it in `BatchPrepError`.
Causes and fixes are in [Troubleshooting](../howto/troubleshooting.md).

::: bandaid.BandaidError

::: bandaid.FrameError

::: bandaid.TooFewStarsError

::: bandaid.PSFFitError

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

::: bandaid.GaiaQueryError
