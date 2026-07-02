# Getting started

This is the ~10-minute end-to-end run: point bandaid at a folder of Seestar
frames and read the photometry back out. If you have not installed it yet, see
[Installation](installation.md).

## Three ideas to keep in mind

bandaid works a little differently from a "detect the stars in my image" tool.
Three ideas explain almost everything you'll see:

1. **Positions come from Gaia, not from your image.** bandaid does *forced*
    photometry: it measures flux at the positions of a magnitude-limited Gaia
    catalog rather than detecting your own sources. You do not give it a target
    list — it builds one from Gaia for wherever the telescope was pointed.
1. **It works in batches.** One slow preparation pass on the first frame
    (`prepare_batch`) sets up the Gaia catalog, plate scale, field of view, Bayer
    masks, and centroiding model; then a fast per-frame loop (`process_batch`)
    reuses all of that for every remaining frame.
1. **The output is one `.star` file per frame** — each file bundles a separate
    star list for every Bayer filter (red/green/blue, plus a luminance channel) —
    plus a single `qa_manifest.csv` summarizing the whole run.

## Run it (CLI)

Point `bandaid process` at a directory of frames and give it somewhere to write:

```bash
$ bandaid process night-of-2026-06-27/ -o out/ -v
```

`-v` streams per-frame progress so you can watch the batch move (use `-vv` for
debug detail). The first frame takes the longest — that is the once-per-batch
preparation from idea 2 — and the rest follow quickly.

## The Ballet weights (first run)

The Ballet centroider needs trained weights. The **first** time you run a
photometry batch, bandaid downloads the default weights from HuggingFace and the
HuggingFace hub caches them, so subsequent runs reuse the cached copy with no
network access.

To pre-fetch the weights (handy before going offline, or to confirm the download
works) use the `weights` command, which prints the cached path:

```bash
$ bandaid weights
/Users/you/.cache/huggingface/.../ballet_weights.npz
```

If you trained or downloaded your own weights, point any run at them with
`--weights` (CLI) or `weights=` (Python) — see
[Training the Ballet centroider](training_the_ballet_centroider.md).

## What lands in `out/`

```text
out/
├── frame_0001.star      # one file per frame; bundles all Bayer filters inside
├── frame_0002.star
│   ...
└── qa_manifest.csv      # one row per input frame
```

You get **one `.star` file per frame** that processed cleanly, plus one
`qa_manifest.csv` for the run. Each `.star` file is a JSON document holding a
separate star list for each Bayer filter — red (`TR`), green (`TG`), blue
(`TB`), and a full-frame luminance channel (`L4`) unless you pass
`--no-append-l4`. The per-star fields are covered in
[Understanding the output](outputs.md).

## Read the QA manifest

`qa_manifest.csv` has one row per input frame and is the fastest way to check a
night at a glance. The columns:

| Column             | What it tells you                                                                                                    |
| ------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `file`             | The input frame.                                                                                                     |
| `status`           | `ok`, `skipped: <reason>`, or `error: <type>`.                                                                       |
| `n_detected`       | How many stars were detected in the frame.                                                                           |
| `sky_median`       | Median sky background — climbs as clouds or moonlight roll in.                                                       |
| `fwhm`             | Measured FWHM (seeing). A spike flags a soft/trailed frame.                                                          |
| `wcs_solved`       | Whether a WCS was solved (`False` on a plate-solve failure).                                                         |
| `n_good_stars`     | Stars that survived photometry filtering and reached the output.                                                     |
| `n_centroid_drift` | Stars whose measured centroid wandered too far from its expected position (flagged, not dropped).                    |
| `n_drift_rejected` | Of those, how many also passed filtering and reached the output — the count a future gate on this flag would remove. |

A healthy night is mostly `status=ok` with steady `fwhm` and `sky_median`. Rows
with `status` other than `ok`, or a sudden jump in `fwhm`/`sky_median`, point you
straight at the bad frames — see [Troubleshooting](troubleshooting.md).

`n_centroid_drift` and `n_drift_rejected` are diagnostic only — see
[Understanding the output](outputs.md#qa_manifestcsv) for the full
interpretation, including the note that manifest data from before the
proper-motion fix (#56) overcounts both.

## The same run from Python

```python
from bandaid import photometer_frames

frames, results = photometer_frames(
    ["night-of-2026-06-27/"],
    output_dir="out/",
)
# frames  -> the expanded, sorted list of input frames
# results -> {input frame: written .star path} for each frame that succeeded
```

`photometer_frames` does the same file expansion, `prepare_batch`, and
`process_batch` the CLI runs. Pass `config=` to tune it (see
[Configuration](configuration.md)) or `weights=` to use your own Ballet weights.

## Where to go next

- **[Configuration](configuration.md)** — the knobs an intermediate user
    actually changes (aperture sizes, Gaia magnitude limit, drift cuts).
- **[Instrument profiles](instrument_profiles.md)** — point bandaid at a
    telescope other than the Seestar50.
- **[Understanding the output](outputs.md)** — what every `.star` and
    `qa_manifest.csv` column means, and how to spot untrustworthy measurements.
