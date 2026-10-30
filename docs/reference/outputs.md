# Output files

```text
out/
├── frame_0001.star   # one StarListSet per input frame; all filters inside
└── qa_manifest.csv   # one row per input frame
```

When every frame comes from one directory, files are written flat as
`<stem><suffix>` (suffix `.star` by default). Frames from a mix of directories
mirror the source tree as `<dirname>/<stem><suffix>`; source directories sharing
a name get a numeric suffix. `qa_manifest.csv` is written once per run unless
`--no-qa-manifest` is given.

## `.star` files

Each file is an [`StarListSet`](https://github.com/mwcraig/aavso-starlist-schema)
JSON document with one `StarList` per filter: `TR`, `TG`, `TB`, and `L4` (omitted
with `--no-build-l4`).

```python
from aavso_starlist_schema import StarListSet

star_set = StarListSet.model_validate_json(open("out/frame_0001.star").read())
for star_list in star_set.star_lists:
    print(star_list.filter, len(star_list.staritems), "stars")
```

Per-list fields: `obs_time`, `site_lat`, `site_lon`, `site_elev`, `observer`,
`filter`, `block_filter`, `exposure`, `tel_manufac`, `tel_model`, `tel_firmware`,
`width`, `height`, `stack`, `adc_depth`, `largest_usable_adu_value`, `egain`,
`fwhm` (measured per frame and filter), `refframe`, `staritems`. Each list's
`egain` is the gain its channel was measured with; with one gain per Bayer
channel, the `L4` list records the median per-star effective gain (see
`l4_effective_egain`).

Per-star fields (`staritems` rows):

| Field        | Meaning                                                                                                                            |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `x`, `y`     | Centroid in pixels: a CNN centroid or a modelled position (see [How star positions are chosen](../explanation/star_positions.md)). |
| `ra`, `dec`  | Catalog position in degrees, as given in the target list (not the measured centroid).                                              |
| `tot_count`  | Background-subtracted counts in the aperture.                                                                                      |
| `count_err`  | Uncertainty on `tot_count` from the noise model.                                                                                   |
| `bkgd_count` | Background counts per pixel: the sigma-clipped mean of the annulus.                                                                |
| `peak_count` | Peak pixel value in the star's own channel within about 2 FWHM of the centroid; for `L4`, the maximum of the three channels.       |

A star has a row only when `tot_count` and `count_err` are finite and positive,
its centroid is in bounds, its SNR is at least `source_selection.min_snr`, and it
is not within `edge_margin_px` of a frame edge (see
[Configuration](configuration.md)). The cuts apply per filter, so a filter can
hold fewer stars than its siblings. A filter with no surviving star is omitted
from the file and listed in `dropped_filters`; a frame with no surviving star in
any filter is skipped.

Contaminated targets are dropped once at batch preparation, so they have no row
in any frame and no column records them. `centroid_drift`, `snr` and `airmass` are not `.star` fields; they are columns
of the tables `photometer_frames(..., output_dir=None)` returns instead of
writing files. Writing other formats:
[Write a custom output format](../howto/custom_writer.md).

`--output-format` selects a writer registered at import; `starlist` is the only
bundled one.

## `qa_manifest.csv`

One row per input frame, including skipped and failed frames (diagnostics blank).
Columns are `QA_MANIFEST_COLUMNS`, in this order.

| Column                | Meaning                                                                                                                                              |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `file`                | Input frame.                                                                                                                                         |
| `status`              | `ok`, `skipped: <FrameError subclass>`, or `error: <exception type>`.                                                                                |
| `n_detected`          | Stars detected in the frame.                                                                                                                         |
| `sky_median`          | Median sky background.                                                                                                                               |
| `fwhm`                | Measured FWHM.                                                                                                                                       |
| `wcs_solved`          | `True` if a WCS solved, `False` on a plate-solve failure, blank if the frame failed earlier.                                                         |
| `pointing_offset_deg` | Degrees between the frame's header center and the batch center; blank if no header pointing could be compared.                                       |
| `wcs_pixscale`        | Solved plate scale in arcsec/pixel, including the scale of a solve rejected for its scale; blank if no WCS solved.                                   |
| `solve_offset_deg`    | Degrees between the solved center and the frame's own header center; blank if no WCS solved.                                                         |
| `n_good_stars`        | Stars surviving the quality cuts in the representative channel (`L4` if present, else the first filter). Not the row count of every filter.          |
| `n_snr20`             | Subset of `n_good_stars` with `snr >= 20`; blank without an `snr` column.                                                                            |
| `dropped_filters`     | Semicolon-joined filters omitted from the `.star` file because no star survived (for example `TR;TB`); empty if none.                                |
| `n_centroid_drift`    | Stars with the `centroid_drift` flag set. Diagnostic only; no row is dropped.                                                                        |
| `n_drift_rejected`    | Subset of `n_centroid_drift` that also passes `good_star_mask`.                                                                                      |
| `n_forced_measured`   | Forced targets with a surviving measurement in the representative channel, matched by sky position; blank without forced targets.                    |
| `n_edge_dropped`      | Catalog stars dropped for lying within `edge_margin_px` of a frame edge, inside or outside the frame; blank if the frame failed before photometry.   |
| `g_cut`               | Batch Gaia G cut of the CNN class (same on every frame); blank when `model_faint_positions` is off.                                                  |
| `n_cnn_class`         | Stars keeping a CNN centroid (at or brighter than `g_cut`, plus usable forced targets); blank when the policy is off.                                |
| `plane_fallback`      | `True` when fewer than `min_fit_stars` stars were usable, so no plane was fitted and every star used its CNN centroid; blank when the policy is off. |
| `plane_n_used`        | Stars defining the offset plane after clipping; `0` on a fallback frame.                                                                             |
| `plane_n_clipped`     | Stars removed by the clip; `0` on a fallback frame.                                                                                                  |
| `plane_rms`           | Radial rms of the plane-fit residuals over the stars used, in pixels; blank without a plane.                                                         |
| `plane_dx_center`     | Plane x offset (CNN minus projected) at the frame center, in pixels.                                                                                 |
| `plane_dy_center`     | Plane y offset at the frame center, in pixels.                                                                                                       |
| `plane_dx_slope_x`    | Change in the plane's x offset from the center to the right edge, in pixels.                                                                         |
| `plane_dx_slope_y`    | Change in the plane's x offset from the center to the top edge, in pixels.                                                                           |
| `plane_dy_slope_x`    | Change in the plane's y offset from the center to the right edge, in pixels.                                                                         |
| `plane_dy_slope_y`    | Change in the plane's y offset from the center to the top edge, in pixels.                                                                           |

The `plane_*` columns are blank for a frame the centroid policy did not run on. A
row can have `status` `ok` with `n_good_stars` of `0` when only a sibling filter
survived; check `dropped_filters`.

### Status values

| Status                       | Meaning                                                                  |
| ---------------------------- | ------------------------------------------------------------------------ |
| `ok`                         | The frame was photometered and written.                                  |
| `skipped: <FrameError type>` | A per-frame failure; see [Troubleshooting](../howto/troubleshooting.md). |
| `error: <exception type>`    | An unexpected error (with `--no-fail-fast`).                             |

## Forced-target bypass

`--forced-targets` rows are not checked against the Gaia magnitude and
contamination model, since a nova or supernova has no Gaia magnitude. The bypass
runs both ways: a forced target is never dropped for a neighbour, and a bright
forced target near a Gaia comparison star does not flag that star. Every other
cut applies: a forced target too faint or too near an edge is absent from that
frame's output. Forced rows carry no name in the `.star` file and are identified
by `ra`/`dec`. A user forcing a target is expected to account for the
contamination it causes; to keep a comparison star flagged as contaminated,
adjust `instrument.contamination_tolerance` or exclude the star.
