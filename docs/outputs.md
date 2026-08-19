# Understanding the output

A run writes two kinds of file into your output directory:

```text
out/
├── frame_0001.star   # one file per frame; bundles every Bayer filter inside
│   ...
└── qa_manifest.csv   # one row per input frame, run-quality signals
```

This page explains what is in each, and how the data-quality flags behave.

## The `.star` files

There is **one `.star` file per frame**, named after the input frame
(`<stem>.star`). Each file is a
[`StarListSet`](https://github.com/mwcraig/aavso-starlist-schema) JSON document
that bundles **one star list per Bayer filter** — red (`TR`), green (`TG`), blue
(`TB`), and a full-frame luminance channel (`L4`) unless you pass
`--no-append-l4`. So the per-filter split lives *inside* the file, not across
several files.

The shape of one file:

```text
StarListSet
├── schema_version
└── star_lists                 # one StarList per Bayer filter
    ├── StarList (filter "TR")
    │   ├── filter, block_filter, fwhm, exposure, egain, width, height, …
    │   └── staritems           # one row per star
    │       └── x, y, ra, dec, tot_count, count_err, bkgd_count, peak_count
    ├── StarList (filter "TG") …
    ├── StarList (filter "TB") …
    └── StarList (filter "L4") …
```

The per-star fields in each `staritems` row:

| Field        | Meaning                                                                                                                                                                                                        |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tot_count`  | Background-subtracted total counts in the aperture (the flux).                                                                                                                                                 |
| `count_err`  | Uncertainty on `tot_count` from the noise model.                                                                                                                                                               |
| `bkgd_count` | Background counts under the star.                                                                                                                                                                              |
| `peak_count` | Peak pixel value in the star's own channel, within ~2 FWHM of the measured centroid — useful for spotting near-saturated stars. For the synthetic `L4` list this is the maximum of the three channels' values. |
| `x`, `y`     | Measured centroid position in pixels.                                                                                                                                                                          |
| `ra`, `dec`  | Sky position (degrees) of the measured star.                                                                                                                                                                   |

Per-frame, per-filter quantities such as the measured `fwhm` live on the
enclosing `StarList`, not on each star.

Only stars that pass photometry filtering reach the file: a row is kept only when
its `tot_count` is finite and positive, its `count_err` is finite and positive,
its centroid lands in-bounds, and its SNR meets the `source_selection.min_snr`
floor (`2.0` by default). Stars that fail (saturated, off the chip, no usable
flux, too faint) are simply absent — there is no row for them.

The cut is per star *and per filter*: because SNR is color-dependent, a star can
survive in one filter and fall below the floor in another, so a
color-disadvantaged channel (say, `TB` on a red-star field) can hold fewer
stars than its siblings. A filter in which **no** star survives is dropped from
the `.star` file entirely (a warning names the dropped filters in the run log,
and the frame's `qa_manifest.csv` row records them in `dropped_filters`); the
frame itself is skipped only when no filter has any usable star.

Targets passed on the command line with `--forced-targets` (see
[Command-line usage](command_line.md)) appear in this same table,
indistinguishable from the Gaia-catalog stars around them: `StarItem` has no
name/ID field, so a forced row is identified only by its `ra`/`dec`.

Read one back in Python with the same schema bandaid uses to write it:

```python
from aavso_starlist_schema import StarListSet

star_set = StarListSet.model_validate_json(open("out/frame_0001.star").read())
for star_list in star_set.star_lists:
    print(star_list.filter, len(star_list.staritems), "stars")
```

### Richer columns: in-memory mode

The written `.star` file holds only the schema fields above. If you want the
extra per-star diagnostics bandaid computes — `bkgd_count`, `snr`, `airmass`, and the
`centroid_drift` flag (below) — run the batch **in memory** from Python by
passing `output_dir=None`, which returns the full photometry tables instead of
writing files:

```python
from bandaid import photometer_frames

frames, results = photometer_frames(["night/"], output_dir=None)
table = results[frames[0]]["TR"]  # an astropy Table with all columns
table.colnames  # tot_count, count_err, bkgd_count, snr, centroid_drift, …
```

### Writing a different format: custom writers

The `.star` writer is just the default; the *how each frame is recorded to disk*
step is pluggable. A **frame writer** is any callable
`write(frame_result, output_path)` where:

- `frame_result` is the frame's `{filter: astropy.table.Table}` mapping — the same rich
    tables as in-memory mode (`bkgd_count`, `snr`, `airmass`, `centroid_drift`, … — a
    *superset* of the `.star` fields), each carrying `meta["full_image_meta"]` and
    `meta["fwhm"]`;
- `output_path` is the resolved per-frame path (`<stem>` + `output_suffix`); a
    writer that emits one file per filter derives per-filter names from it;
- the return value is stored as that frame's entry in the results mapping
    (usually the `Path`, or list of paths, actually written).

Pass one to `photometer_frames` (or `process_batch`) via `write_frame`. This
reuses bandaid's per-frame streaming, output-path/collision handling, and QA
manifest — you only supply the serialization. For example, one CSV of the rich
table per filter:

```python
from bandaid import photometer_frames


def write_csv(frame_result, output_path):
    written = []
    for filter_name, table in frame_result.items():
        path = output_path.with_suffix(f".{filter_name}.csv")
        table.write(path, format="ascii.csv", overwrite=True)
        written.append(path)
    return written


photometer_frames(["night/"], write_frame=write_csv, output_suffix="")
```

A writer that wants AAVSO-starlist semantics can still call
`good_star_mask` / `eloy_to_starlist` itself (see `bandaid.writers` for the
default `write_starlist_set`). Pass `min_snr=table.meta.get("min_snr")` when you
do — that is the floor the run was configured with; omitting it silently applies
the 2.0 default, overriding even an explicit `--min-snr 0`.

### Choosing a writer on the command line

`--output-format` selects among the writers bandaid registers **at import** —
today that is just `starlist` (the default). Any writer shipped inside the
package is registered the same way and is selectable by name:

```console
bandaid process night/ --output-format starlist
```

`--output-format` defaults to `starlist`; an unknown name is a clean CLI error
listing the registered writers.

A *custom* writer, though, is only reachable through the Python API
(`write_frame=`, above). `register_writer` mutates an in-process registry, and
`bandaid process` runs in its own process — so a writer you register in your own
Python session is **not** visible to a separate `bandaid` command. To record a
custom format, drive the batch from Python with `write_frame=` rather than
registering for the CLI.

## `qa_manifest.csv`

One row per input frame, written once per run. It is the fastest way to find the
bad frames in a night without opening every `.star` file.

| Column              | Meaning                                                                                                                                                                                                                                                                                                                           |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `file`              | The input frame this row describes.                                                                                                                                                                                                                                                                                               |
| `status`            | `ok`, `skipped: <FrameError type>`, or `error: <type>`.                                                                                                                                                                                                                                                                           |
| `n_detected`        | Stars detected in the frame.                                                                                                                                                                                                                                                                                                      |
| `sky_median`        | Median sky background — rises with clouds, moonlight, or haze.                                                                                                                                                                                                                                                                    |
| `fwhm`              | Measured FWHM (seeing); a spike flags a soft or trailed frame.                                                                                                                                                                                                                                                                    |
| `wcs_solved`        | `True` if a WCS solved; `False` on a plate-solve failure; blank if the frame failed earlier.                                                                                                                                                                                                                                      |
| `n_good_stars`      | Stars that survived filtering on the frame's *representative* channel: `L4` when present, else the first filter, under the same `min_snr` floor the writer applies. See below.                                                                                                                                                    |
| `dropped_filters`   | Semicolon-joined names of filters dropped from the `.star` output because no star survived filtering (e.g. `TB` or `TR;TB`), or empty for a frame where every filter kept at least one star. See below.                                                                                                                           |
| `n_centroid_drift`  | Stars with the `centroid_drift` flag set (see below) — a frame-health signal on its own.                                                                                                                                                                                                                                          |
| `n_drift_rejected`  | The subset of `n_centroid_drift` that also passes `good_star_mask` — the stars a future gate on this flag would actually remove, since most drifted stars are already dropped by the existing flux/error/bounds cuts.                                                                                                             |
| `n_forced_measured` | Forced targets (see [Command-line usage](command_line.md)) with a good, output-surviving measurement in the frame's representative channel, matched by sky position — answers "was my nova actually measured in this frame" without float-matching `ra`/`dec` across `.star` files. Blank when no forced targets were configured. |

`n_good_stars` is a single-channel count, not a per-filter tally: the SNR floor
is color-dependent, so a per-filter `.star` output can contain fewer stars than
`n_good_stars` in a color-disadvantaged channel (and `L4`, when present, can
keep a star an individual channel drops). The representative-channel number is
the right one for its QA use — cloud and transparency episodes are common-mode
across channels — but do not read it as the exact row count of every filter's
StarList. In particular, a row can show `status='ok'` with `n_good_stars=0`
when the representative channel itself was the one dropped but a sibling
filter survived and was written — `dropped_filters` is how to detect that a
frame is partial rather than reading `n_good_stars` alone.

A frame that was skipped or errored still gets a row (with its diagnostics left
blank), so the manifest accounts for **every** input frame, not just the
successful ones. `status` values other than `ok` map directly to the entries in
[Troubleshooting](troubleshooting.md).

`n_centroid_drift` and `n_drift_rejected` are diagnostic counts only, not a
filter: the flag does not remove any rows (see `centroid_drift` below), and
manifest data produced before the proper-motion fix (#56) overcounts both
columns, since the flag fired preferentially on high-proper-motion stars whose
*catalog* position was stale rather than on genuine drift.

## Data-quality flags

bandaid is conservative about *changing* your data, so it is important to know
exactly what each quality check does.

### `centroid_drift` — flagged, never dropped, not in the `.star` file

When a star's measured centroid wanders too far from its expected (aligned)
position — a bad WCS, a too-faint star, or an obstruction — bandaid sets a
`centroid_drift` flag for that star. **No rows are dropped on this flag.** It is
also **not part of the `.star` schema**, so it is not written to disk; to see it,
use the in-memory mode above (the `centroid_drift` column on the returned table).
The threshold and the reasoning are in the
[Centroid-drift check](centroid_drift_check.md).

### Contamination — dropped at batch prep, *no column written*

A bright neighbour whose PSF wings spill into a target's aperture would corrupt
that target's flux. bandaid handles this **at batch preparation**, not per row:
`prepare_batch` runs `neighbor_contamination_flag_sky` against the Gaia catalog
and **drops contaminated targets from the photometry list before any frame is
measured**. The practical consequence:

- A contaminated target is **silently absent** from the output — there is no
    `.star` row for it at all, in any frame or filter.
- There is **no `contaminated` column** anywhere in the output. (`good_star_mask`
    will honour such a column defensively if one is ever present, but the normal
    pipeline never writes one.)

So "contaminated stars are dropped" is true, but the drop happens once, up front,
to the target list — not as a per-frame, per-row column you can inspect. If you
need a star that bandaid considers contaminated, loosen
`instrument.contamination_tolerance` (see [Configuration](configuration.md)).

`--forced-targets` stars are never evaluated by this check — it needs a Gaia
magnitude to size the separation model, and a forced target (a nova or
supernova, by definition absent from Gaia) has none. A forced target that
happens to sit on top of a bright star is therefore not flagged or dropped for
contamination. The bypass runs in the other direction too: a bright forced
target (the nova itself) near a Gaia comparison star is likewise invisible to
*that* star's contamination check, so the comparison star is not flagged
either. bandaid's stance is that a user forcing a target is expected to have
already taken any contamination it causes into account; if that is not the
case, loosen `instrument.contamination_tolerance` (see
[Configuration](configuration.md)) or exclude the comparison star manually.
