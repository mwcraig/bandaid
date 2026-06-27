# Configuration

bandaid's photometry tuning knobs live in a single immutable `PhotometryConfig`
object. You build one, pass it to `prepare_batch`, and it is carried on the
returned `BatchPrep` and applied to every frame in the batch:

```python
from bandaid import PhotometryConfig, ApertureConfig, prepare_batch

config = PhotometryConfig(apertures=ApertureConfig(gap=5, annulus_width=4))
prep = prepare_batch(first_file, cnn=cnn, config=config)
```

If you pass no `config`, a default `PhotometryConfig()` is used — every field
takes the documented default.

The config is **frozen** (you cannot mutate it after construction) and
**validated** at construction, so values that would silently break the pipeline
are rejected up front with a clear error rather than failing deep in a batch.

The leaf photometry functions (`measure_photometry`, `build_photometry_table`)
still accept their individual keyword arguments, which override the config when
set — handy for one-off calls from a notebook.

To see every default (including the derived `inner_annulus`, `outer_annulus`, and
`contaminant_mag_limit`) live from the code:

```python
>>> from bandaid import PhotometryConfig
>>> PhotometryConfig().model_dump()
```

## The three tiers

The knobs fall into three groups by how safe they are to change.

### Tier 1 — Science knobs (set these per run)

These are ordinary analysis choices and are safe to set for any run.

| Sub-config         | Field                    | Default  | Meaning                                          |
| ------------------ | ------------------------ | -------- | ------------------------------------------------ |
| `apertures`        | `radii`                  | `(1.0,)` | Aperture radii, in units of FWHM                 |
| `apertures`        | `gap`                    | `4.0`    | FWHM gap between largest aperture and annulus    |
| `apertures`        | `annulus_width`          | `3.0`    | Radial width of the background annulus, in FWHM  |
| `source_selection` | `gaia_mag_limit`         | `15.0`   | Magnitude limit for the photometry targets       |
| `source_selection` | `contaminant_mag_offset` | `3.0`    | Contaminant-catalog depth below `gaia_mag_limit` |
| `drift`            | `drift_tolerance_fwhm`   | `1.0`    | Max centroid drift, in FWHM                      |
| `drift`            | `drift_cap_pix`          | `4.0`    | Absolute pixel cap on centroid drift             |

### Tier 2 — Instrument / per-telescope (advanced)

The `instrument` field is an `InstrumentProfile`: a **named telescope** that
bundles the detection/PSF tuning below with that telescope's per-frame
FITS-header dialect (`header_map`). These depend on the plate scale, the PSF, and
the instrument's sensitivity. The defaults are the Seestar50 values; change them
only when pointing a **different** telescope at the sky.

| Sub-config   | Field                     | Default       | Meaning                                                  |
| ------------ | ------------------------- | ------------- | -------------------------------------------------------- |
| `instrument` | `name`                    | `"Seestar50"` | The telescope's name (its registry key)                  |
| `instrument` | `thresh`                  | `0.5`         | Source-detection threshold, in background sigma          |
| `instrument` | `detection_opening`       | `3`           | Morphological-opening kernel that gates faint detections |
| `instrument` | `fwhm_cutout_half`        | `25`          | Half-width (px) of the PSF window for the FWHM fit       |
| `instrument` | `contamination_tolerance` | `0.01`        | Max neighbour spillover before flagging                  |
| `instrument` | `moffat_beta`             | `3.0`         | Moffat wing index for the contamination model            |
| `instrument` | `header_map`              | Seestar50     | FITS-header dialect resolved by `metadata_from_header`   |

#### Instrument profiles registry

Named profiles live in `bandaid.instruments`. Fetch a bundled one, list what is
available, or share a user-tuned profile through a file:

```python
from bandaid import (
    PhotometryConfig, load_instrument, register_instrument, available_instruments,
)
from bandaid.config import InstrumentProfile

available_instruments()                 # -> ['Seestar50']
profile = load_instrument("Seestar50")
config = PhotometryConfig(instrument=profile)

# Save / load a tuned profile, or register one so load_instrument finds it by name.
profile.to_file("my_scope.json")
mine = InstrumentProfile.from_file("my_scope.json")
register_instrument(mine)
```

Adding a telescope is dropping a `meta_json_files/<name>/profile.json` into the
package (bundling tuning + `header_map`) or registering a profile at runtime — no
code edits. See [Instrument profiles](instrument_profiles.md) for the
`header_map` directive syntax and a worked add-a-telescope example. Note the
`header_map` resolves only the *instrument* half of a frame's metadata;
observer-identity overrides (site, observer code) are applied last via the
separate `user_specific_metadata` dict passed to `process_batch`.

### Tier 3 — Solver internals (do not touch)

The twirl asterism-matcher star counts, the WCS match tolerance, the minimum
detected-star count, and the minimum stars for a contamination pair are **not**
exposed on the config. Mis-setting them stalls or breaks the WCS solve (the cost
of the matcher grows like `C(N, 4)`, and too-small counts leave frames unsolved),
so they remain locked module constants in `bandaid.photometry`.

## Validation

Construction enforces the invariants the pipeline relies on, for example:

- aperture radii, `gap`, and `annulus_width` must all be positive, and
- the drift cuts and `gaia_mag_limit` must be finite.

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
