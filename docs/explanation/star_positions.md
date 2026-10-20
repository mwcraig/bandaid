# How star positions are chosen

bandaid measures each star at one position: *measured* (the Ballet centroiding
CNN refines the projected Gaia position) for bright stars, or *modelled* (the
projected Gaia position plus a smooth per-frame correction fitted to the bright
stars) for the rest.

## Why not measure every star

Below an aperture SNR of about 10 the CNN centroid is off by 1.2 to 1.6 px, and
the error is a bias, not just scatter: the published weights pull toward a point
set by the noise. A synthetic star at a known position shows it:

| aperture SNR | median offset x (px) | median offset y (px) | scatter x (px) | scatter y (px) |
| ------------ | -------------------- | -------------------- | -------------- | -------------- |
| 1            | -0.32                | +0.37                | 1.17           | 1.18           |
| 2            | -0.27                | +0.35                | 1.09           | 1.07           |
| 3            | -0.21                | +0.34                | 0.99           | 0.95           |
| 5            | -0.13                | +0.33                | 0.77           | 0.72           |
| 10           | -0.07                | +0.20                | 0.40           | 0.41           |
| 20           | -0.02                | +0.05                | 0.12           | 0.12           |
| 50           | 0.00                 | 0.00                 | 0.04           | 0.04           |

(Median predicted minus true position, Moffat stars of FWHM 2.2 to 3.7 px in
real-frame white noise. The aperture SNR is sky-limited at one FWHM radius, not
the manifest's `snr` column.)

Projected positions are good to about 0.15 px once a per-frame offset is fitted
to the bright stars, so the CNN is used only where it is accurate.

## The policy

The policy is on by default; its settings are the fields of
`PhotometryConfig.centroid` ([configuration](../reference/configuration.md)).

1. **CNN class.** A catalog star is CNN-class if its Gaia G is at or brighter
    than the batch's cut `G_cut`, or if it is a forced target (no Gaia G). A
    CNN-class star keeps its CNN centroid, unless the CNN returned its input
    position exactly (its fallback for an unusable cutout); that star is output
    at the plane position.
1. **Fit set and plane.** On each frame the `fit_n_stars` (30) brightest stars
    by Gaia G that survive the edge margin are centroided by the CNN. An offset
    plane (a constant plus a linear term in each frame coordinate, per axis) is
    fitted by unweighted least squares to their CNN minus projected positions.
    Non-finite and input-equal CNN results are left out. One clip drops a star
    whose residual on either axis is over `clip_sigma` (3) robust standard
    deviations (1.4826 times the MAD) from the median; the plane is then refitted.
1. **Everything else**, including fit stars outside the CNN class, is output at
    its projected position plus the plane, so a star's position never depends on
    whether it was in a frame's fit set.
1. **Fallback.** If fewer than `min_fit_stars` (12) fit stars survive the clip,
    the frame has no plane and every star takes its CNN centroid.
1. **Skip.** If the plane's post-clip rms exceeds `max_plane_rms_pix` (1.0), the
    frame is skipped with a `CentroidPlaneError`; its manifest status is
    `skipped: CentroidPlaneError` and `plane_rms` holds the rms.

### The magnitude cut

`G_cut` is computed once per batch, from the catalog alone: the Gaia G of the
30th (`cnn_class_size`) brightest target inside a circle with the frame's area
(radius `sqrt(width * height / pi)` pixels) centred on the batch center. Because
it does not depend on the frame, a star's class is the same on every frame;
per-frame ranking would flip borderline stars between measured and modelled and
step their flux. If the circle holds fewer than `cnn_class_size` targets the cut
is infinite, every star is CNN-class, and a warning is logged.

## The centroid-drift check

The check flags stars whose centroid wandered from where it should be: a wrong
WCS, a star too faint to detect, or cloud or an obstruction. A star is flagged
when the pixel distance between its centroid and its expected position exceeds
`min(drift_tolerance_fwhm * fwhm, drift_cap_pix)`.

- **Expected position:** the projected position plus the frame's plane, or the
    projected position alone when the policy is off or the frame has no plane.
- **Thresholds:** `drift.drift_tolerance_fwhm` (1.0, in FWHM) scales the
    allowance with seeing; `drift.drift_cap_pix` (4.0 px) stops a large FWHM from
    licensing a huge shift. Both are fields of `PhotometryConfig.drift`.
- **Flagged:** only stars with `centroid_method` of `cnn` or `fallback_cnn`.
    A `plane` star sits at its expected position by construction. Non-finite
    centroids are always flagged.

The result is the boolean `centroid_drift` column. It is flag-only: no star is
dropped because of it. The manifest counts flagged stars (`n_centroid_drift`).

## Seeing the choice in the output

- `centroid_method`: `cnn` (measured), `plane` (modelled), `fallback_cnn`
    (measured because the frame had no plane).
- Manifest columns: `g_cut`, `n_cnn_class`, `plane_fallback`, `plane_n_used`,
    `plane_n_clipped`, `plane_rms`, `plane_dx_center`, `plane_dy_center`,
    `plane_dx_slope_x`, `plane_dx_slope_y`, `plane_dy_slope_x`,
    `plane_dy_slope_y`, `n_centroid_drift`, `n_drift_rejected`.

Column meanings are in [Output files](../reference/outputs.md). To centroid every
star with the CNN, see [Tune a run](../howto/tune_a_run.md):

```python
from bandaid import CentroidConfig, PhotometryConfig

config = PhotometryConfig(centroid=CentroidConfig(model_faint_positions=False))
```
