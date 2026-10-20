# Tune a run

Every tuning knob lives on one immutable, validated `PhotometryConfig`. Fields,
defaults, and valid ranges are listed in the
[configuration reference](../reference/configuration.md).

## Change a knob from the command line

Two knobs have their own options:

```bash
bandaid process night/ -o out/ --gaia-mag-limit 14 --min-snr 5
```

Everything else goes through a JSON config. Write the defaults, edit them, check
them, and pass the file:

```bash
bandaid config init -o config.json
bandaid config validate config.json
bandaid process night/ -o out/ --config config.json
```

Keys you delete from the file take their defaults, so a config can be as short as:

```json
{"edge_margin_px": 15.0, "apertures": {"gap": 5.0, "annulus_width": 4.0}}
```

`--gaia-mag-limit` and `--min-snr` override the matching fields of the file.

## Change a knob from Python

Build the config and pass it to `photometer_frames`, or to `prepare_batch` if you
drive the batch yourself:

```python
from bandaid import ApertureConfig, PhotometryConfig, photometer_frames

config = PhotometryConfig(apertures=ApertureConfig(gap=5, annulus_width=4))
photometer_frames(["night/"], config=config, output_dir="out/")
```

A config cannot be mutated. To vary one top-level field, copy it with
`config.model_copy(update={"edge_margin_px": 15.0})`; nested groups such as
`apertures` are rebuilt as above. Invalid values raise at construction.

## Which knob for which symptom

Too few stars in the output:

- Raise `source_selection.gaia_mag_limit` to target fainter stars.
- Lower `source_selection.min_snr`. The cut is per star and per filter, so a
    color-disadvantaged channel can hold fewer stars than its siblings.
- Lower `edge_margin_px` if the manifest's `n_edge_dropped` is large.

Stars missing near the frame edge: `edge_margin_px` removes every catalog star
whose projected position is within that many pixels of an edge, forced targets
included. The default covers the centroiding network's 15x15 cutout, so lower it
only if you accept unreliable edge centroids. A margin of half the smaller frame
side or more is rejected when the batch is prepared.

Noisy or biased background, or neighbours leaking into the aperture:

- `apertures.radii` sets the aperture in units of FWHM.
- `apertures.gap` and `apertures.annulus_width` place the background annulus
    outside the largest aperture.
- A star near an edge can have a truncated annulus; its background is measured
    from the on-frame annulus pixels.

A star you need is dropped as contaminated: raise
`instrument.contamination_tolerance`, which belongs to the instrument profile (see
[add a telescope](add_a_telescope.md)). Forced targets are never checked.

Too many or too few `centroid_drift` flags: `drift.drift_tolerance_fwhm` and
`drift.drift_cap_pix` set the thresholds. The flag never removes rows; see
[output files](../reference/outputs.md).

Centroid policy:

- By default the `centroid.cnn_class_size` brightest stars keep their CNN centroid
    and all others get the projected position plus a per-frame plane.
- `centroid.model_faint_positions=False` switches the policy off, so every star
    gets its CNN centroid.
- Frames whose plane fit is worse than `centroid.max_plane_rms_pix` are skipped
    with `CentroidPlaneError`.

How the policy chooses positions: [star positions](../explanation/star_positions.md).
