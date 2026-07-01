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

| Field        | Meaning                                                        |
| ------------ | -------------------------------------------------------------- |
| `tot_count`  | Background-subtracted total counts in the aperture (the flux). |
| `count_err`  | Uncertainty on `tot_count` from the noise model.               |
| `bkgd_count` | Background counts under the star.                              |
| `peak_count` | Peak pixel value — useful for spotting near-saturated stars.   |
| `x`, `y`     | Measured centroid position in pixels.                          |
| `ra`, `dec`  | Sky position (degrees) of the measured star.                   |

Per-frame, per-filter quantities such as the measured `fwhm` live on the
enclosing `StarList`, not on each star.

Only stars that pass photometry filtering reach the file: a row is kept only when
its `tot_count` is finite and positive, its `count_err` is finite and positive,
and its centroid lands in-bounds. Stars that fail (saturated, off the chip, no
usable flux) are simply absent — there is no row for them.

Read one back in Python with the same schema bandaid uses to write it:

```python
from aavso_starlist_schema import StarListSet

star_set = StarListSet.model_validate_json(open("out/frame_0001.star").read())
for star_list in star_set.star_lists:
    print(star_list.filter, len(star_list.staritems), "stars")
```

### Richer columns: in-memory mode

The written `.star` file holds only the schema fields above. If you want the
extra per-star diagnostics bandaid computes — `sky`, `snr`, `airmass`, and the
`centroid_drift` flag (below) — run the batch **in memory** from Python by
passing `output_dir=None`, which returns the full photometry tables instead of
writing files:

```python
from bandaid import photometer_frames

frames, results = photometer_frames(["night/"], output_dir=None)
table = results[frames[0]]["TR"]   # an astropy Table with all columns
table.colnames                     # tot_count, count_err, sky, snr, centroid_drift, …
```

### Writing a different format: custom writers

The `.star` writer is just the default; the *how each frame is recorded to disk*
step is pluggable. A **frame writer** is any callable
`write(frame_result, output_path)` where:

- `frame_result` is the frame's `{filter: astropy.table.Table}` mapping — the same rich
    tables as in-memory mode (`sky`, `snr`, `airmass`, `centroid_drift`, … — a
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
default `write_starlist_set`).

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

| Column         | Meaning                                                                                      |
| -------------- | -------------------------------------------------------------------------------------------- |
| `file`         | The input frame this row describes.                                                          |
| `status`       | `ok`, `skipped: <FrameError type>`, or `error: <type>`.                                      |
| `n_detected`   | Stars detected in the frame.                                                                 |
| `sky_median`   | Median sky background — rises with clouds, moonlight, or haze.                               |
| `fwhm`         | Measured FWHM (seeing); a spike flags a soft or trailed frame.                               |
| `wcs_solved`   | `True` if a WCS solved; `False` on a plate-solve failure; blank if the frame failed earlier. |
| `n_good_stars` | Stars that survived filtering and reached the `.star` output.                                |

A frame that was skipped or errored still gets a row (with its diagnostics left
blank), so the manifest accounts for **every** input frame, not just the
successful ones. `status` values other than `ok` map directly to the entries in
[Troubleshooting](troubleshooting.md).

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
