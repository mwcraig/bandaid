# Configuration

bandaid's photometry tuning knobs live in a single immutable `PhotometryConfig`
object. You build one, pass it to `prepare_batch`, and it is carried on the
returned `BatchPrep` and applied to every frame in the batch:

```python
from bandaid import PhotometryConfig, ApertureConfig, prepare_batch

config = PhotometryConfig(apertures=ApertureConfig(gap=5, annulus_width=4))
prep = prepare_batch(first_file, cnn=cnn, config=config)
```

If you pass no `config`, a default `PhotometryConfig()` is used. Every field
takes the documented default *except* `instrument`, which defaults to `None`
— "resolve it from the frame header" — rather than a hard-coded telescope; see
[Auto-detection from the FITS header](instrument_profiles.md#auto-detection-from-the-fits-header).
The table below documents a *resolved* `InstrumentProfile`'s own field
defaults (which a bare `InstrumentProfile()` still reproduces), not
`PhotometryConfig.instrument` itself.

The config is **frozen** (you cannot mutate it after construction) and
**validated** at construction, so values that would silently break the pipeline
are rejected up front with a clear error rather than failing deep in a batch.

The leaf photometry functions (`measure_photometry`, `build_photometry_table`)
still accept their individual keyword arguments, which override the config when
set — handy for one-off calls from a notebook.

To see every default (including the derived `inner_annulus`, `outer_annulus`, and
`contaminant_mag_limit`) live from the code:

```pycon
>>> from bandaid import PhotometryConfig
>>> PhotometryConfig().model_dump()
```

## The three tiers

The knobs fall into three groups by how safe they are to change.

### Tier 1 — Science knobs (set these per run)

These are ordinary analysis choices and are safe to set for any run.

| Sub-config         | Field                    | Default  | Meaning                                                                            |
| ------------------ | ------------------------ | -------- | ---------------------------------------------------------------------------------- |
| `apertures`        | `radii`                  | `(1.0,)` | Aperture radii, in units of FWHM                                                   |
| `apertures`        | `gap`                    | `4.0`    | FWHM gap between largest aperture and annulus                                      |
| `apertures`        | `annulus_width`          | `3.0`    | Radial width of the background annulus, in FWHM                                    |
| `source_selection` | `gaia_mag_limit`         | `15.0`   | Magnitude limit for the photometry targets                                         |
| `source_selection` | `contaminant_mag_offset` | `3.0`    | Contaminant-catalog depth below `gaia_mag_limit`                                   |
| `source_selection` | `min_snr`                | `2.0`    | Minimum SNR a star must have to reach the output                                   |
| `source_selection` | `gaia_row_limit`         | `10000`  | Base maximum rows the Gaia query may return (scaled with cone area)                |
| `drift`            | `drift_tolerance_fwhm`   | `1.0`    | Max centroid drift, in FWHM                                                        |
| `drift`            | `drift_cap_pix`          | `4.0`    | Absolute pixel cap on centroid drift                                               |
| (top level)        | `edge_margin_px`         | `10.0`   | Catalog stars projected within this many pixels of an edge are not measured        |
| `centroid`         | `model_faint_positions`  | `True`   | Faint stars take the projected position plus a per-frame plane, not a CNN centroid |
| `centroid`         | `cnn_class_size`         | `30`     | Number of brightest catalog targets that keep their CNN centroid                   |
| `centroid`         | `fit_n_stars`            | `30`     | Brightest stars per frame whose CNN centroids define the plane                     |
| `centroid`         | `min_fit_stars`          | `12`     | Fewest stars that must survive the clip for a frame to use a plane                 |
| `centroid`         | `clip_sigma`             | `3.0`    | Clip, in robust standard deviations, applied once when fitting the plane           |

### Tier 2 — Instrument / per-telescope (advanced)

The `instrument` field is an `InstrumentProfile`: a **named telescope** that
bundles the detection/PSF tuning below with that telescope's per-frame
FITS-header dialect (`header_map`). These depend on the plate scale, the PSF, and
the instrument's sensitivity. The class defaults below are the Seestar50 values
except where noted; change them only when pointing a **different** telescope at
the sky.

| Sub-config   | Field                         | Default       | Meaning                                                                                    |
| ------------ | ----------------------------- | ------------- | ------------------------------------------------------------------------------------------ |
| `instrument` | `name`                        | `"Seestar50"` | The telescope's name (its registry key)                                                    |
| `instrument` | `thresh`                      | `0.5`         | Source-detection threshold, in background sigma                                            |
| `instrument` | `detection_opening`           | `5`           | Morphological-opening kernel that gates faint detections                                   |
| `instrument` | `fwhm_cutout_half`            | `25`          | Half-width (px) of the PSF window for the FWHM fit                                         |
| `instrument` | `fwhm_n_stars`                | `25`          | Cap on the brightest detections fed to the FWHM fit                                        |
| `instrument` | `contamination_tolerance`     | `0.01`        | Max neighbour spillover before flagging                                                    |
| `instrument` | `moffat_beta`                 | `3.0`         | Moffat wing index for the contamination model                                              |
| `instrument` | `contamination_seeing_margin` | `1.25`        | Seeing-pessimism factor for the once-per-batch flag                                        |
| `instrument` | `wcs_scale_tolerance`         | `0.05`        | Max fractional plate-scale deviation before a WCS is rejected as wrong-scale               |
| `instrument` | `wcs_pointing_tolerance`      | none          | Max degrees between a solved frame center and its header pointing (none: one field radius) |
| `instrument` | `cone_radius_margin`          | `0.4`         | Degrees added to `fov_rad` for the once-per-batch Gaia query                               |
| `instrument` | `solve_pool_radius_scale`     | `0.9`         | Fraction of `fov_rad` used as the radius of each frame's plate-solve star pool             |
| `instrument` | `header_map`                  | Seestar50     | FITS-header dialect resolved by `metadata_from_header`                                     |
| `instrument` | `header_match`                | `()`          | FITS-header rules used by `detect_instrument` to auto-select this profile                  |

Three rows differ between the bare class and the bundled Seestar50 profile.
`header_match` defaults to `()` (no rules, so a bare `InstrumentProfile()` is
never auto-detected), while `load_instrument("Seestar50")` returns a profile
with one rule (`INSTRUME == "Seestar S50"`). This is deliberate; see
[Auto-detection from the FITS header](instrument_profiles.md#auto-detection-from-the-fits-header).
The Seestar50 profile also tightens `wcs_scale_tolerance` to `0.005` and sets
`wcs_pointing_tolerance` to `0.30`, values measured on Seestar S50 data that
only make sense against the profile's measured `pixscale`; see
[The Seestar50 tolerances](instrument_profiles.md#the-seestar50-tolerances).
A hand-written Seestar config that copies the class defaults from this table
loses that tightened gate.

The bright-neighbour contamination flag is computed once per batch, from the
*first* frame's FWHM, and applied to every frame of the night. Because seeing
usually changes during a night, the flag is evaluated at `first-frame FWHM × contamination_seeing_margin` (and at the largest configured aperture radius), so
pairs that would become contaminated as seeing softens are dropped up front. Set
the margin to `1.0` to evaluate the flag at exactly the first frame's seeing;
values below 1 are rejected at construction.

#### Instrument profiles registry

Named profiles live in `bandaid.instruments`. Fetch a bundled one, list what is
available, or share a user-tuned profile through a file:

```python
from bandaid import (
    PhotometryConfig,
    load_instrument,
    register_instrument,
    available_instruments,
)
from bandaid.config import InstrumentProfile

available_instruments()  # -> ['Seestar50']
profile = load_instrument("Seestar50")
config = PhotometryConfig(instrument=profile)

# Save / load a tuned profile, or register one so load_instrument finds it by
# name. The round-tripped copy is still named "Seestar50", so overriding the
# bundled profile with it must be explicit; a profile with a new name registers
# without replace=True, but its header_match rules must not duplicate another
# profile's (see Instrument profiles).
profile.to_file("my_scope.json")
mine = InstrumentProfile.from_file("my_scope.json")
register_instrument(mine, replace=True)
```

Add a telescope at runtime by registering a profile or loading one from a file —
no code edits. See [Instrument profiles](instrument_profiles.md) for the
`header_map` directive syntax and a worked add-a-telescope example; to contribute
a *bundled* instrument, see
[Adding a bundled instrument profile](contributing.md#adding-a-bundled-instrument-profile).
Note the `header_map` resolves only the *instrument* half of a frame's metadata;
observer-identity overrides (site, observer code) are applied last via the
separate `user_specific_metadata` dict passed to `process_batch`.

### Tier 3 — Solver internals (do not touch)

The twirl asterism-matcher star counts, the WCS match tolerance, the minimum
detected-star count, and the minimum stars for a contamination pair are **not**
exposed on the config. Mis-setting them stalls or breaks the WCS solve (the cost
of the matcher grows like `C(N, 4)`, and too-small counts leave frames unsolved),
so they remain locked module constants in `bandaid.photometry`.

### Frame-edge margin

A catalog star whose projected position (where the plate solution puts it,
before centroiding) is within `edge_margin_px` of any frame edge, or off the
frame, is dropped before centroiding and photometry, so it has no row in the
output. The default 10 px covers the centroiding CNN's 15x15 cutout (half-size
7 px), which is fill-padded where it overlaps an edge and then gives an
unreliable centroid. The same rule applies to every position, forced targets
included. The frame spans `[0, width - 0.5)` by `[0, height - 0.5)`, the same
span `good_star_mask` uses, shrunk by the margin on every side.

The margin does not change background-annulus handling. With the default
apertures the annulus spans about 15 to 24 px at a 3 px FWHM, so stars 10 to
24 px from an edge still have a truncated annulus, as before; photutils
measures their background from the on-frame annulus pixels.

The number of catalog stars dropped within `edge_margin_px` of a frame edge,
inside or outside it, is recorded per frame in the QA manifest column
`n_edge_dropped`. A margin of half the smaller frame side or more is rejected
when the batch is prepared.

### Centroid policy

By default only the brightest catalog stars keep their CNN centroid; every other
star is output at its projected position plus a per-frame plane fitted to the
brightest stars. See [Measured versus modelled positions](measured_vs_modelled_positions.md)
for the rule, the magnitude cut `cnn_class_size` sets, and the output columns that
record the choice. `model_faint_positions=False` switches the policy off and centroids every
star with the CNN.

## Validation

Construction enforces the invariants the pipeline relies on, for example:

- aperture radii, `gap`, and `annulus_width` must all be positive, and
- the drift cuts and `gaia_mag_limit` must be finite, and
- `edge_margin_px` must be positive and finite, and
- the centroid policy's `cnn_class_size` must be at least 1, `min_fit_stars` at
    least 3, `fit_n_stars` at least `min_fit_stars`, and `clip_sigma` positive and
    finite.

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

ApertureConfig(gap=-1)  # raises: gap must be greater than 0
```
