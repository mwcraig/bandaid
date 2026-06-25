# Configuration

bandaid's photometry tuning knobs live in a single immutable `PhotometryConfig`
object. You build one, pass it to `prepare_batch`, and it is carried on the
returned `BatchPrep` and applied to every frame in the batch:

```python
from bandaid import PhotometryConfig, ApertureConfig, prepare_batch

config = PhotometryConfig(apertures=ApertureConfig(gap=5, annulus_width=4))
prep = prepare_batch(first_file, cnn=cnn, config=config)
```

If you pass no `config`, a default `PhotometryConfig()` is used and the pipeline
behaves exactly as it did before the config existed: every default reproduces the
historical module-level constant.

The config is **frozen** (you cannot mutate it after construction) and
**validated** at construction, so values that would silently break the pipeline
are rejected up front with a clear error rather than failing deep in a batch.

## The three tiers

The knobs fall into three groups by how safe they are to change.

### Tier 1 — Science knobs (set these per run)

These are ordinary analysis choices and are safe to set for any run.

| Sub-config  | Field                     | Meaning                                          |
| ----------- | ------------------------- | ------------------------------------------------ |
| `apertures` | `radii`                   | Aperture radii, in units of FWHM                 |
| `apertures` | `gap`                     | FWHM gap between largest aperture and annulus    |
| `apertures` | `annulus_width`           | Radial width of the background annulus, in FWHM  |
| `detection` | `gaia_mag_limit`          | Magnitude limit for the photometry targets       |
| `detection` | `contaminant_mag_offset`  | Contaminant-catalog depth below `gaia_mag_limit` |
| `quality`   | `drift_tolerance_fwhm`    | Max centroid drift, in FWHM                      |
| `quality`   | `drift_cap_pix`           | Absolute pixel cap on centroid drift             |
| `quality`   | `contamination_tolerance` | Max neighbour spillover before flagging          |
| `quality`   | `moffat_beta`             | Moffat wing index for the contamination model    |

### Tier 2 — Instrument / per-telescope (advanced)

These depend on the plate scale and the PSF. The defaults are the Seestar50
values; change them only when pointing a **different** telescope at the sky.

| Sub-config   | Field               | Meaning                                                  |
| ------------ | ------------------- | -------------------------------------------------------- |
| `instrument` | `thresh`            | Source-detection threshold, in background sigma          |
| `instrument` | `detection_opening` | Morphological-opening kernel that gates faint detections |
| `instrument` | `fwhm_cutout_half`  | Half-width (px) of the PSF window for the FWHM fit       |

### Tier 3 — Solver internals (do not touch)

The twirl asterism-matcher star counts, the WCS match tolerance, the minimum
detected-star count, and the minimum stars for a contamination pair are **not**
exposed on the config. Mis-setting them stalls or breaks the WCS solve (the cost
of the matcher grows like `C(N, 4)`, and too-small counts leave frames unsolved),
so they remain locked module constants in `bandaid.photometry`. If one ever needs
tuning it can graduate to Tier 2 with a validator; until then, leave them alone.

## Validation

Construction enforces the invariants the pipeline relies on, for example:

- aperture radii, `gap`, and `annulus_width` must all be positive, and
- the quality cuts and `gaia_mag_limit` must be finite (the latter positive).

Several values are **derived** rather than set directly, so the invariants the
pipeline cares about hold by construction instead of needing a validator:

- the background annulus is `(max(radii) + gap, max(radii) + gap + annulus_width)`,
    exposed as `inner_annulus`, `outer_annulus`, and the `annulus` pair. Because
    `gap` and `annulus_width` are positive, the annulus always sits strictly
    outside the largest aperture and has positive width.
- `contaminant_mag_limit` is `gaia_mag_limit + contaminant_mag_offset` (offset `3`
    by default). Because the offset is positive, the contaminant list is always
    deeper than the target list. Tune the *offset* rather than an absolute limit.

```python
from bandaid import ApertureConfig

ApertureConfig(gap=-1)   # raises: gap must be greater than 0
```

## Per-function overrides

The leaf photometry functions (e.g. `measure_photometry`,
`build_photometry_table`) still
accept their individual keyword arguments. When set, those take precedence over
the config; when left at their defaults they fall back to it. This keeps the leaf
functions convenient to call directly from a notebook or a unit test without
building a full config.
