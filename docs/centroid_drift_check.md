# Centroid-drift sanity check

Photometry in bandaid is measured at *aligned* positions: a Gaia reference
catalog is projected into each image's pixel frame, and a CNN centroider refines
those positions to the actual star centers. When that refinement goes wrong the
photometry at the reported position can no longer be trusted. A centroid can
wander far from its aligned/expected position when:

1. the WCS computed for the image is incorrect, so centroiding starts from the
    wrong positions,
1. the star is too faint to actually detect, so the centroider wanders to an
    incorrect value, or
1. the star is blocked by an obstruction or clouds.

To make these cases visible, `build_photometry_table` records a per-star boolean
`centroid_drift` column.

## How the flag is computed

The drift is the pixel-space displacement between the measured centroid and the
aligned position:

```python
drift = np.linalg.norm(centroid_coords - aligned_coords, axis=-1)
```

Both arrays are already in pixel space, so no WCS round-trip is needed and the
metric isolates centroid wander from WCS quality. A star is flagged when

```python
drift > min(drift_tolerance * fwhm, drift_cap)
```

The FWHM-relative term lets the allowance scale with seeing, while the absolute
pixel cap keeps a pathologically large FWHM from licensing an enormous shift.
Non-finite centroids (e.g. failed faint-star centroids) produce a non-finite
drift and are always flagged as drifted.

The thresholds come from module-level constants in `bandaid.photometry`:

| Constant               | Default | Meaning                                 |
| ---------------------- | ------- | --------------------------------------- |
| `DRIFT_TOLERANCE_FWHM` | `1.0`   | Max drift in units of FWHM              |
| `DRIFT_CAP_PIX`        | `4.0`   | Absolute pixel cap on the allowed drift |

These defaults are empirical starting points and are meant to be tuned against
real frames. Override them per call:

```python
table = build_photometry_table(img, mask, drift_tolerance=0.5, drift_cap=3.0)
```

or call the helper directly:

```python
from bandaid import centroid_drift_flag

flagged = centroid_drift_flag(centroid_coords, aligned_coords, fwhm)
```

## Flag, don't drop

The check is currently **flag-only**: the `centroid_drift` column is written but
no rows are removed from the resulting `StarList`. This is non-destructive, so
you can inspect the flag and tune the thresholds against real data before letting
it affect pipeline output. To start dropping flagged stars later, extend
`eloy_to_starlist` with the same one-line pattern already used for the
`contaminated` column:

```python
if "centroid_drift" in eloy_table.colnames:
    good &= ~eloy_table["centroid_drift"]
```
