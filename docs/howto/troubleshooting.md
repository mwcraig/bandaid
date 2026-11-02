# Troubleshooting

Start with `qa_manifest.csv`: every frame has a `status` of `ok`,
`skipped: <FrameError subclass>`, or `error: <exception type>`. See
[output files](../reference/outputs.md) for the columns. To see why one frame was
skipped, run just that frame with `-vv`; the log shows the chained cause:

```bash
bandaid process the_bad_frame.fit -o /tmp/debug -vv
```

## Skipped frames

A skipped frame is the designed behavior: it is logged as a
`skipping <file>: <reason>` warning at the default verbosity and the rest of the
batch still finishes.

- `skipped: TooFewStarsError` — fewer than 3 stars detected, or every detected
    source saturated (cloud, trailing, defocus). The skipped row leaves
    `n_detected` and `fwhm` blank; the count is in the `skipping ...` warning.
    There is no knob for a bad frame.
- `skipped: PSFFitError` — no usable PSF could be fit to the frame's stars: the
    fitted FWHM exceeds the cutout (`fwhm_cutout_half`). Typically a frame with
    no real star image (clouds, an obstruction). The fitted value is in the
    `skipping ...` warning.
- `skipped: WCSSolveError` — the twirl plate solve failed or matched nothing.
    Check the header pointing; `-vv` shows the twirl traceback.
- `skipped: WCSScaleError` — every solve came back at the wrong plate scale, and
    `wcs_pixscale` holds the last one. Re-measure `pixscale` or widen
    `wcs_scale_tolerance` ([add a telescope](add_a_telescope.md)).
- `skipped: WCSPointingError` — every solve landed far from the header pointing.
    Widen `wcs_pointing_tolerance`, or check the header's RA/DEC frame and equinox.
- `skipped: FrameMetadataError` — a `header_map` directive points at a keyword the
    header lacks, or its value is unparsable. Fix the profile and verify with a
    single-frame `-vv` run ([add a telescope](add_a_telescope.md#4-validate)).
    `InstrumentDetectionError`, a subclass, means the header matched no profile or
    several; pass `--instrument` or `--profile`.
- `skipped: NoUsableStarsError` — stars were detected but none survived the quality
    cuts, or none project at least `edge_margin_px` from an edge. Lower `min_snr`
    or `edge_margin_px` ([tune a run](tune_a_run.md)).
- `skipped: CentroidPlaneError` — the per-frame offset plane fits its own stars
    worse than `max_plane_rms_pix`. Raise `centroid.max_plane_rms_pix` or set
    `centroid.model_faint_positions` false.
- `skipped: StarListValidationError` — a row passed filtering but the AAVSO
    starlist schema rejected the frame. The log names the field; report it.
- `skipped: DegenerateBayerChannelError` — a Bayer channel has no pixels or zero
    variance in the balancing window (a constant sub-frame). Discard the frame and
    check `bayerpat` and the Bayer offsets in the profile.
- `skipped: FrameError` — the frame's shape differs from the batch, or its
    pointing is beyond the field radius of the frame the batch was prepared
    from. Process it separately.

## The whole run aborted

A `BatchPrepError` is fatal: nothing can proceed without the once-per-batch
preparation. It is built from the first frame, in batch order, that the
preparation accepts: a first frame with too few stars to measure an FWHM, or no
usable PSF to fit to them, is logged and the next frame is tried, and a later
frame is passed over the same way for any per-frame failure that would make it
a `skipped:` row anyway (a file that cannot be read at all still aborts). The
passed-over frames still appear in the manifest as `skipped:` rows. Causes:

- No frame can prepare the batch ("tried all N frames"): the batch is bad
    throughout. Check the frames themselves (clouds, a closed dome, a wrong
    focus), the detection settings `thresh` and `detection_opening`, or the
    PSF-fit ceiling `fwhm_cutout_half`.
- Gaia returned too few reference stars in the prep frame's solve pool. Check the
    pointing, or raise `gaia_mag_limit`.
- The observation time or pointing of the prep frame cannot be read.
- The first frame's header resolves to no profile, or several
    (`InstrumentDetectionError`); the message names the frame. Pass
    `--instrument` or fix the header.
- `edge_margin_px` is half the smaller frame side or more.
- The Gaia query failed on every server ("could not query Gaia for the field
    at ..."). bandaid tries CDS VizieR, then the IDIA and CfA mirrors, then the
    GAVO Heidelberg TAP service, giving each 10 s to connect and 60 s to start
    answering; GAVO itself stops a query that takes more than about 20 s and
    reports that as an error. A warning is logged as each server is skipped (a
    timeout, a failure status, or the server's own error text). In the browser
    each server instead gets one 70 s limit for the whole request
    (`TimeoutError` past it), and a refused connection, a missing CORS header
    or an error status all surface as the same `NetworkError`. Usually this is
    an outage on the catalog side; a failed answer is never cached, so just run
    again.
- `CatalogTruncationError`: the Gaia query hit its row limit before reaching
    `gaia_mag_limit`. Raise `source_selection.gaia_row_limit` or lower the limit.

With `--fail-fast` an unexpected error is re-raised instead of logged and skipped;
the `FrameError` cases above are skipped either way.

## Weights: offline or custom

The first run downloads the Ballet weights from HuggingFace and the hub caches them.
On an offline or firewalled machine, fetch them elsewhere and pass the file:

```bash
# On a machine with network access:
bandaid weights -o centroid_15x15.npz

# On the offline machine:
bandaid process night/ -o out/ --weights centroid_15x15.npz
```

`bandaid weights` with no `-o` prints the cached path instead. From Python pass
`weights="centroid_15x15.npz"` to `photometer_frames`. The same options load
weights you trained ([train weights](train_weights.md)).
