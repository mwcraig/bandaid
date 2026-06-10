"""
Batch photometry driver.

Each image in a batch is photometered at fixed sky positions, and four prep
items are constant across the whole batch (same field, same SeeStar 50 camera):
the Gaia source list, the contaminant-filtered subset of it used as the
photometry/centroiding points, the CNN centroiding model, and the Bayer masks.

`prepare_batch` computes these once -- deriving the field pointing, plate scale,
FOV, Bayer pattern, and (via one cheap detection pass on the first frame) the
FWHM -- and returns them as a `BatchPrep` bundle. `process_batch` then loops the
frames through `process_one_image`, reusing that bundle. The split keeps the
once-per-batch work and the per-frame work as separate, single-trigger
functions: no shared mutable state, no "is it done yet?" bookkeeping.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from st_pipeline.schema_definition import StarListSet

from .catalog import cached_gaia_radecs
from .image2sl_qt import generate_bayer_masks
from .photometry import (
    calibration_sequence,
    eloy_to_starlist,
    neighbor_contamination_flag_sky,
    process_one_image,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchPrep:
    """
    Reusable, once-per-batch photometry inputs.

    Attributes
    ----------
    radecs : numpy.ndarray
        Full Gaia source list (``(N, 2)`` RA/Dec in degrees) used for per-frame
        WCS alignment.
    photometry_coords : astropy.coordinates.SkyCoord
        Contaminant-filtered subset of ``radecs`` used as the centroiding and
        photometry targets.
    cnn : object
        The ``eloy`` Ballet centroiding model to use for every frame.
    bayer_masks : dict
        Mapping of filter name to Bayer mask, as returned by
        `generate_bayer_masks`.
    """

    radecs: np.ndarray
    photometry_coords: SkyCoord
    cnn: object
    bayer_masks: dict


def prepare_batch(first_file, *, cnn, append_l4=False, gaia_mag_limit=15):
    """
    Compute the once-per-batch photometry inputs from the first frame.

    Runs a single detection pass on ``first_file`` (no WCS solve) to obtain the
    FWHM and image metadata, queries Gaia for the field, drops contaminated
    sources, and builds the Bayer masks.

    Parameters
    ----------
    first_file : str or Path
        Path to the first FITS frame in the batch. Used to derive the field
        pointing/FOV, plate scale, Bayer pattern, image shape, and FWHM. All
        frames in the batch are assumed to share these.
    cnn : object
        The ``eloy`` Ballet centroiding model to carry through to every frame.
    append_l4 : bool, optional
        Whether to add a full-frame "L4" luminance channel to the Bayer masks.
        Default False.
    gaia_mag_limit : float, optional
        Gaia magnitude limit for the initial source list. Default 15.

    Returns
    -------
    BatchPrep
        The reusable prep bundle for the batch.

    Raises
    ------
    ValueError
        If too few stars are detected in ``first_file`` to measure an FWHM
        (``calibration_sequence`` returns its all-None sentinel).
    """
    calibrated_data, metadata, _, fwhm_pix, _ = calibration_sequence(first_file)
    if calibrated_data is None:
        msg = f"too few stars detected in {first_file!r} to prepare the batch"
        raise ValueError(msg)

    # fov_rad is a field *radius*; cached_gaia_radecs takes the full field and
    # halves it internally (matching the established twirl.gaia_radecs usage).
    center = (metadata["ra"], metadata["dec"])
    radecs, mags = cached_gaia_radecs(center, 2 * metadata["fov_rad"])
    bright_gaia = mags <= gaia_mag_limit
    radecs = radecs[bright_gaia]
    mags = mags[bright_gaia]
    fwhm_arcsec = fwhm_pix * metadata["pixscale"]
    flagged = neighbor_contamination_flag_sky(radecs, mags, fwhm_arcsec)
    photometry_coords = SkyCoord(radecs[~flagged], unit="deg")

    bayer_masks = generate_bayer_masks(
        (metadata["height"], metadata["width"]),
        metadata,
        append_l4=append_l4,
    )

    return BatchPrep(
        radecs=radecs,
        photometry_coords=photometry_coords,
        cnn=cnn,
        bayer_masks=bayer_masks,
    )


def process_batch(
    files, prep, *, user_specific_metadata, output_dir=None, output_suffix=".star"
):
    """
    Photometer every frame in a batch using a shared `BatchPrep`.

    Parameters
    ----------
    files : iterable of str or Path
        FITS frames to process. Each is aligned and photometered independently
        (each solves its own WCS, since pointing drifts frame to frame), reusing
        the prep computed once in `prepare_batch`.
    prep : BatchPrep
        The reusable prep bundle from `prepare_batch`.
    user_specific_metadata : dict
        User-specific metadata recorded with the output for each frame.
    output_dir : str or Path or None, optional
        Directory to write the per-frame photometry results to. Default ".".

    output_suffix : str, optional
        Suffix for the output files. Default ".star".

    Returns
    -------
    dict
        Mapping of each successfully-processed input file to its result. In
        in-memory mode (``output_dir`` is None) the value is the
        ``{filter: Table}`` photometry result; in write-to-disk mode the value
        is the written output ``Path`` (the tables are not held in memory).
        Frames that fail (``process_one_image`` returns None) are skipped with a
        logged warning and omitted from the result in both modes.
    """
    results = {}
    for file in files:
        by_filter = process_one_image(
            file,
            user_specific_metadata,
            prep.radecs,
            prep.cnn,
            prep.bayer_masks,
            input_photometry_coords=prep.photometry_coords,
        )
        if by_filter is None:
            logger.warning("skipping %s: too few stars detected", file)
            continue
        if output_dir is not None:
            output_path = Path(output_dir) / (Path(file).stem + output_suffix)
            star_lists = [
                eloy_to_starlist(tab, tab.meta["full_image_meta"])
                for tab in by_filter.values()
            ]
            star_list_set = StarListSet(star_lists=star_lists)
            output_path.write_text(star_list_set.model_dump_json(indent=2))
            results[file] = output_path
        else:
            results[file] = by_filter
    return results
