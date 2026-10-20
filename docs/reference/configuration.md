# Configuration reference

`PhotometryConfig` is a frozen, validated pydantic model passed to
`prepare_batch(config=...)` and carried on `BatchPrep`. Invalid values are
rejected at construction. Print every default, including derived values, with
`PhotometryConfig().model_dump()`. For which knobs to change and when, see
[Tune a run](../howto/tune_a_run.md).

| Field              | Type                        | Default  | Meaning                                                                                                                        |
| ------------------ | --------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `apertures`        | `ApertureConfig`            | defaults | Aperture and annulus geometry.                                                                                                 |
| `source_selection` | `SourceSelectionConfig`     | defaults | Gaia magnitude limits and SNR floor.                                                                                           |
| `drift`            | `DriftConfig`               | defaults | Centroid-drift cuts.                                                                                                           |
| `centroid`         | `CentroidConfig`            | defaults | Measured-versus-modelled position policy.                                                                                      |
| `edge_margin_px`   | `float`, `> 0`              | `10.0`   | Catalog stars projected within this many pixels of a frame edge, or off the frame, are not measured (forced targets included). |
| `instrument`       | `InstrumentProfile \| None` | `None`   | `None` resolves the profile from the frame header; see [Instrument profiles](instrument_profiles.md).                          |

The frame spans `[0, width - 0.5)` by `[0, height - 0.5)`, shrunk by the margin on
every side. The default covers the half-size (7 px) of the centroiding CNN's
15x15 cutout, which is fill-padded at an edge. A margin of half the smaller frame
side or more is rejected at batch preparation. Dropped stars are counted in the
`n_edge_dropped` manifest column.

## Tier 1: science knobs

Safe to set for any run.

| Sub-config         | Field                    | Type / constraint          | Default  | Meaning                                                                                                    |
| ------------------ | ------------------------ | -------------------------- | -------- | ---------------------------------------------------------------------------------------------------------- |
| `apertures`        | `radii`                  | `tuple[float]`, each `> 0` | `(1.0,)` | Aperture radii in FWHM.                                                                                    |
| `apertures`        | `gap`                    | `float`, `> 0`             | `4.0`    | Gap in FWHM between the largest aperture and the annulus.                                                  |
| `apertures`        | `annulus_width`          | `float`, `> 0`             | `3.0`    | Radial width of the background annulus in FWHM.                                                            |
| `source_selection` | `gaia_mag_limit`         | `float`, finite            | `15.0`   | Magnitude limit of the photometry targets.                                                                 |
| `source_selection` | `contaminant_mag_offset` | `float`, `> 0`             | `3.0`    | Depth of the contaminant catalog below `gaia_mag_limit`.                                                   |
| `source_selection` | `min_snr`                | `float`, `>= 0`            | `2.0`    | Minimum SNR for a star to reach the output.                                                                |
| `source_selection` | `gaia_row_limit`         | `int`, `>= 1`              | `10000`  | Base maximum rows the Gaia query returns (scaled with cone area).                                          |
| `drift`            | `drift_tolerance_fwhm`   | `float`, `> 0`             | `1.0`    | Maximum centroid drift in FWHM.                                                                            |
| `drift`            | `drift_cap_pix`          | `float`, `> 0`             | `4.0`    | Absolute cap on centroid drift in pixels.                                                                  |
| `centroid`         | `model_faint_positions`  | `bool`                     | `True`   | Faint stars take the projected position plus a per-frame plane; `False` centroids every star with the CNN. |
| `centroid`         | `cnn_class_size`         | `int`, `>= 1`              | `30`     | Number of brightest targets that keep their CNN centroid.                                                  |
| `centroid`         | `fit_n_stars`            | `int`, `>= min_fit_stars`  | `30`     | Brightest stars per frame whose CNN centroids define the plane.                                            |
| `centroid`         | `min_fit_stars`          | `int`, `>= 3`              | `12`     | Fewest stars that must survive the clip for a frame to use a plane.                                        |
| `centroid`         | `clip_sigma`             | `float`, `> 0`             | `3.0`    | Clip threshold in robust standard deviations (applied once).                                               |
| `centroid`         | `max_plane_rms_pix`      | `float`, `> 0`             | `1.0`    | A frame whose plane rms (pixels) exceeds this is skipped.                                                  |

Derived, read-only: `inner_annulus = max(radii) + gap`, `outer_annulus = inner_annulus + annulus_width`, `annulus = (inner_annulus, outer_annulus)`, and
`contaminant_mag_limit = gaia_mag_limit + contaminant_mag_offset`.

## Tier 2: instrument profile

`instrument` is an `InstrumentProfile`. The defaults below are those of a bare
`InstrumentProfile()`; the file format is in
[Instrument profiles](instrument_profiles.md).

| Field                         | Type / constraint        | Default       | Meaning                                                                                                                                      |
| ----------------------------- | ------------------------ | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`                        | `str`                    | `"Seestar50"` | Registry key.                                                                                                                                |
| `thresh`                      | `float`, `> 0`           | `0.5`         | Detection threshold in background sigma.                                                                                                     |
| `detection_opening`           | `int`, `>= 1`            | `5`           | Morphological-opening kernel gating faint detections.                                                                                        |
| `fwhm_cutout_half`            | `int`, `>= 1`            | `25`          | Half-width in pixels of the PSF cutout for the FWHM fit.                                                                                     |
| `fwhm_n_stars`                | `int`, `>= 1`            | `25`          | Cap on the brightest detections feeding the FWHM fit.                                                                                        |
| `contamination_tolerance`     | `float`, `> 0`           | `0.01`        | Maximum fractional neighbour spillover into the aperture before a star is flagged.                                                           |
| `moffat_beta`                 | `float`, `> 0`           | `3.0`         | Moffat wing index of the contamination model.                                                                                                |
| `contamination_seeing_margin` | `float`, `>= 1`          | `1.25`        | The once-per-batch contamination flag is evaluated at first-frame FWHM times this factor.                                                    |
| `wcs_scale_tolerance`         | `float`, `> 0`           | `0.05`        | Maximum fractional deviation of the solved plate scale from `pixscale` before the WCS is rejected.                                           |
| `wcs_pointing_tolerance`      | `float \| None`, `> 0`   | `None`        | Maximum degrees between a solved center and the header pointing; `None` allows one field radius.                                             |
| `header_frame`                | `"icrs"` or `"fk5"`      | `"icrs"`      | Coordinate frame of the header RA/DEC (case-insensitive).                                                                                    |
| `header_equinox`              | `str`                    | `"J2000"`     | Equinox of an `fk5` header pointing: `"date"` or an epoch such as `"J2000"`, `"B1950"`; must stay `"J2000"` when `header_frame` is `"icrs"`. |
| `cone_radius_margin`          | `float`, `>= 0`          | `0.4`         | Degrees added to `fov_rad` for the once-per-batch Gaia query.                                                                                |
| `solve_pool_radius_scale`     | `float`, `> 0`           | `0.9`         | Fraction of `fov_rad` used as the radius of each frame's plate-solve star pool.                                                              |
| `header_map`                  | mapping                  | Seestar50 map | Per-frame FITS-header dialect.                                                                                                               |
| `header_match`                | `tuple[HeaderMatchRule]` | `()`          | Rules used by `detect_instrument`; empty means never auto-detected.                                                                          |

The bundled `Seestar50` profile differs from a bare `InstrumentProfile()` in five
fields:

| Field                    | Bare class | `Seestar50`                 |
| ------------------------ | ---------- | --------------------------- |
| `header_match`           | `()`       | `INSTRUME == "Seestar S50"` |
| `wcs_scale_tolerance`    | `0.05`     | `0.005`                     |
| `wcs_pointing_tolerance` | `None`     | `0.30`                      |
| `header_frame`           | `"icrs"`   | `"fk5"`                     |
| `header_equinox`         | `"J2000"`  | `"date"`                    |

## Tier 3: solver internals

The twirl star counts, WCS match tolerance, minimum detected-star count, and
minimum stars for a contamination pair are module constants in
`bandaid.photometry`, not configuration.
