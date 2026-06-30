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

## Configuration

The immutable, validated configuration objects. See
[Configuration](configuration.md) for which knobs to change.

::: bandaid.PhotometryConfig

::: bandaid.ApertureConfig

::: bandaid.SourceSelectionConfig

::: bandaid.DriftConfig

::: bandaid.InstrumentProfile

## Instrument registry

Resolve, register, and list instrument profiles. See
[Instrument profiles](instrument_profiles.md).

::: bandaid.load_instrument

::: bandaid.register_instrument

::: bandaid.available_instruments

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

::: bandaid.FrameMetadataError

::: bandaid.NoUsableStarsError

::: bandaid.BatchPrepError
