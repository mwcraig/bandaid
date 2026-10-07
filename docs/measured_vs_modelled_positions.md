# Measured versus modelled star positions

bandaid measures photometry at one position per star. For a bright star that
position is *measured*: the centroiding CNN (the Ballet network) refines the
projected catalog position to the star's actual center. For a faint star it is
*modelled*: the projected Gaia position plus a smooth correction fitted on that
frame to the bright stars. This page explains why, which stars get which, and
how to see the choice in the output.

## Why not measure every star

The CNN centroid of a faint star is noisier and more biased than the model.
Below an aperture SNR of about 10 the network is off by 1.2 to 1.6 px (the star
is a few pixels across), and it is biased: it does not return the star's
position but pulls toward a point that depends on the noise, not the star. The
bias is a property of the published weights, not of the data, and it shows up on
a synthetic star placed at a known position with a perfect catalog:

| aperture SNR | median offset x (px) | median offset y (px) | scatter x (px) | scatter y (px) |
| ------------ | -------------------- | -------------------- | -------------- | -------------- |
| 1            | -0.32                | +0.37                | 1.17           | 1.18           |
| 2            | -0.27                | +0.35                | 1.09           | 1.07           |
| 3            | -0.21                | +0.34                | 0.99           | 0.95           |
| 5            | -0.13                | +0.33                | 0.77           | 0.72           |
| 10           | -0.07                | +0.20                | 0.40           | 0.41           |
| 20           | -0.02                | +0.05                | 0.12           | 0.12           |
| 50           | 0.00                 | 0.00                 | 0.04           | 0.04           |

(Median of predicted minus true position for Moffat stars of FWHM 2.2 to 3.7 px
in white noise at a real frame's noise level; the aperture SNR here is
sky-limited, at one FWHM radius, and is not the table's `snr` column.) Pure noise
with no star at all does not rest at the middle of the cutout either: the
published weights return about (-0.4, +0.4) px from it. Weights retrained on
more realistic noise do not remove the bias below SNR 5; they move it.

Projected positions, by contrast, are good to about 0.15 px once a per-frame
smooth offset is fitted to the bright stars. The CNN is therefore used only for
the bright stars, where it is accurate, which is the SNR floor for using it:
below it, a star's position comes from the model.

## The policy

The policy is on by default (`PhotometryConfig.centroid.gaia_prior`).

1. **The CNN class.** A catalog star is *CNN-class* if its Gaia G is at or
    brighter than the batch's magnitude cut `G_cut`, or if it is a forced target
    (a forced target has no Gaia G and is always CNN-class). A CNN-class star keeps
    its CNN centroid.
1. **The fit set.** On every frame, the `fit_n_stars` (30) brightest stars by
    Gaia G that survive the [edge margin](configuration.md#frame-edge-margin) are
    centroided by the CNN, and an offset plane is fitted to their CNN minus
    projected positions. The fit set is chosen afresh on each frame from the stars
    that are on it, independently of the CNN class.
1. **Everything else** is output at its projected position plus the plane. A fit
    star that is not CNN-class is centroided only to define the plane and is also
    output at the plane position, so a star's output position never depends on
    whether it happened to be in a frame's fit set.
1. **Fallback.** If fewer than `min_fit_stars` (12) fit stars survive the
    clip, the frame has no plane and every star is centroided by the CNN, as
    before the policy existed. This is recorded in the QA manifest.

### The magnitude cut

`G_cut` is computed once for the whole batch in `prepare_batch`, from the catalog
alone: it is the Gaia G of the `cnn_class_size`-th (30th) brightest catalog
target inside a circle centred on the batch center whose area equals the frame's
area (radius `sqrt(width * height / pi)` pixels, about 0.54 degrees for a
Seestar S50). The value is logged at the start of the run. Because it depends
only on the catalog, a star's class is the same on every frame; the alternative,
ranking stars by how bright they are on each frame, makes borderline stars flip
between measured and modelled from one frame to the next, and each flip steps the
star's flux. If the circle holds fewer than `cnn_class_size` targets the cut is
infinite, every star is CNN-class and a warning is logged.

### The plane

The plane is the offset `(dx, dy)` to add to a projected position to get where
the CNN would put the star, as a constant plus a term linear in each frame
coordinate, one set of coefficients per axis, fitted by unweighted least squares
to the fit set. Stars whose CNN result is not finite, or is exactly equal to the
input position (the network's fallback for an unusable cutout), are not
measurements and are left out. One clip is applied: a star is dropped if either
axis's residual exceeds `clip_sigma` (3) standard deviations of that axis's
residuals, and the plane is refitted on the rest. Higher orders and weighting
were tried on three fields and were worse as often as they were better.

## Seeing the choice in the output

- The eloy table has a `centroid_method` column: `cnn` (measured), `plane`
    (modelled) or `fallback_cnn` (measured, because the frame had no plane). L4 carries
    the column. It cannot reach `.star` files, whose schema is fixed; use the
    in-memory mode (see [Understanding the output](outputs.md)).
- The QA manifest has `g_cut`, `n_cnn_class`, `plane_fallback`, `plane_n_used`,
    `plane_n_clipped`, `plane_rms` and the plane's centre offset and slopes
    (`plane_dx_center`, `plane_dy_center`, `plane_dx_slope_x`, `plane_dx_slope_y`,
    `plane_dy_slope_x`, `plane_dy_slope_y`). The center offset is a smooth,
    slowly varying signal that tracks the telescope; a jump in it, or in
    `plane_rms`, marks a frame whose bright-star positions are off.
- `centroid_drift` is redefined; see [Centroid-drift check](centroid_drift_check.md).

## Switching it off

```python
from bandaid import CentroidConfig, PhotometryConfig

config = PhotometryConfig(centroid=CentroidConfig(gaia_prior=False))
```

With `gaia_prior=False` every star is centroided by the CNN, and the output is
identical to a run that never had the policy. The manifest's plane columns are
blank and `centroid_method` is `cnn` throughout.

## What changes in your output

Against an all-CNN run of the same frames (155 LS Psc frames, a field with many
faint stars):

| L4 SNR   | position change, median | flux change, median | light-curve scatter ratio |
| -------- | ----------------------- | ------------------- | ------------------------- |
| 1 to 2   | about 1.5 px            | +11.6 %             | 0.79                      |
| 2 to 3   | about 1.5 px            | +3.6 %              | 0.76                      |
| 3 to 5   | about 1.2 px            | +2.4 %              | 0.80                      |
| 5 to 10  | about 0.7 px            | +0.1 %              | 0.91                      |
| 10 to 20 | small                   | about 0             | 1.00                      |
| over 20  | none                    | 0                   | 1.00                      |

The 30 brightest stars are unchanged. The faint stars move to the modelled
position, their fluxes rise (the CNN's biased positions put the aperture
slightly off the star, so the flux is under-read), and their light curves get
tighter. Because the faint-star flux rises, more stars clear a given `min_snr`:
with the default of 2, about 11 % more L4 rows on that field. Anyone comparing a
star list with one made before the policy will see this change.
