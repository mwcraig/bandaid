# Write a custom output format

A frame writer is any callable `write(frame_result, output_path)`:

- `frame_result` is the frame's `{filter: astropy.table.Table}` mapping. The
    tables carry every measured column, a superset of the `.star` fields, plus
    `meta["full_image_meta"]` and `meta["fwhm"]`.
- `output_path` is the resolved per-frame path (stem plus `output_suffix`). A
    writer that emits one file per filter derives per-filter names from it.
- The return value (usually the path or list of paths written) is stored as the
    frame's entry in the results mapping.

Pass it as `write_frame` to `photometer_frames` or `process_batch`. bandaid keeps
its per-frame streaming, output-path handling, and QA manifest. This example
writes one CSV per filter:

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

A frame with no star passing the quality cuts in any filter is skipped before the
writer is called; `bandaid.writers.usable_filters` is that rule. A writer may
raise any `FrameError` to mark its frame as skipped.

To reuse the `.star` row filtering, call `good_star_mask` or `eloy_to_starlist`
from `bandaid.photometry` and pass `min_snr=table.meta.get("min_snr")`; omitting
it applies the default floor of 2.0 even when the run set another.

`--output-format` on the command line accepts only writers registered at import
(`starlist`). A writer registered with `register_writer` in your own session is
not visible to a separate `bandaid` process, so drive custom formats from Python.

Forced targets skip the contamination check; see
[output files](../reference/outputs.md) for where that matters.
