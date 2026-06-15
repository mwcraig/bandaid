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
from astropy.io import fits
from st_pipeline.schema_definition import StarListSet

from .catalog import cached_gaia_radecs
from .exceptions import (
    BatchPrepError,
    FrameError,
    FrameMetadataError,
    TooFewStarsError,
)
from .image2sl_qt import generate_bayer_masks
from .photometry import (
    N_GAIA_STARS_ALIGN,
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
    center : tuple of float
        ``(ra, dec)`` in degrees of the field the Gaia catalog was queried for,
        used by `check_frame_consistency` to reject frames that drifted off it.
    fov_rad : float
        Field radius in degrees; the maximum allowed pointing offset from
        ``center``.
    shape : tuple of int
        Expected ``(height, width)`` of every frame.
    """

    radecs: np.ndarray
    photometry_coords: SkyCoord
    cnn: object
    bayer_masks: dict
    center: tuple
    fov_rad: float
    shape: tuple


def prepare_batch(
    first_file, *, cnn, append_l4=False, gaia_mag_limit=15, contaminant_mag_limit=None
):
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
        Gaia magnitude limit for the photometry *targets* -- the stars actually
        measured and used to align each frame. Default 15.
    contaminant_mag_limit : float, optional
        Gaia magnitude limit for the deeper *contaminant* catalog used only for
        contamination flagging: a real star fainter than ``gaia_mag_limit`` can
        still spill into a brighter target's aperture, so flagging runs against
        this deeper list. If ``None`` (default), it is ``gaia_mag_limit + 3``;
        values below ``gaia_mag_limit`` are clamped up to it (the contaminant
        list is never shallower than the target list). Must be finite; a
        non-finite value raises ``ValueError``.

    Returns
    -------
    BatchPrep
        The reusable prep bundle for the batch.

    Raises
    ------
    BatchPrepError
        If too few stars are detected in ``first_file`` to measure an FWHM, so
        the batch preparation cannot be built.
    ValueError
        If ``contaminant_mag_limit`` is non-finite.
    """
    # A too-few-stars failure on the *first* frame is fatal for the whole batch
    # (no FWHM/pointing to prepare from), so translate the recoverable
    # per-frame TooFewStarsError into a fatal BatchPrepError.
    try:
        _, metadata, _, fwhm_pix, _ = calibration_sequence(first_file)
    except TooFewStarsError as exc:
        msg = f"too few stars detected in {first_file!r} to prepare the batch"
        raise BatchPrepError(msg) from exc

    # fov_rad is a field *radius*; cached_gaia_radecs takes the full field and
    # halves it internally (matching the established twirl.gaia_radecs usage).
    center = (metadata["ra"], metadata["dec"])
    # A Gaia query failure (network/service error) is fatal for the whole batch;
    # surface it as a BatchPrepError instead of a raw astroquery/requests error.
    try:
        radecs, mags = cached_gaia_radecs(center, 2 * metadata["fov_rad"])
    except Exception as exc:
        msg = f"could not query Gaia for the field at {center}"
        raise BatchPrepError(msg) from exc
    # Decouple the stars we *measure* (targets, cut at gaia_mag_limit) from the
    # stars that can *contaminate* them (a deeper list down to
    # contaminant_mag_limit). A real star fainter than the photometry limit still
    # spills into a brighter target's aperture, so flagging runs against the
    # deeper list -- but only targets are ever flagged/dropped.
    if contaminant_mag_limit is None:
        contaminant_mag_limit = gaia_mag_limit + 3
    elif not np.isfinite(contaminant_mag_limit):
        # max(nan, limit) silently returns nan, which makes `contaminant`
        # all-False and later blows up as a boolean-index length mismatch.
        msg = f"contaminant_mag_limit must be finite, got {contaminant_mag_limit!r}"
        raise ValueError(msg)
    contaminant_mag_limit = max(contaminant_mag_limit, gaia_mag_limit)

    target = mags <= gaia_mag_limit
    contaminant = mags <= contaminant_mag_limit
    target_radecs = radecs[target]

    # Without enough reference stars no frame can solve a WCS, so fail the batch
    # now with a clear message rather than letting every frame fail later.
    if len(target_radecs) < N_GAIA_STARS_ALIGN:
        msg = (
            f"Gaia returned only {len(target_radecs)} stars brighter than "
            f"{gaia_mag_limit} for the field at {center}; need at least "
            f"{N_GAIA_STARS_ALIGN} to solve a WCS"
        )
        raise BatchPrepError(msg)

    fwhm_arcsec = fwhm_pix * metadata["pixscale"]
    # Asymmetric flagging: only targets can be flagged, but the deeper contaminant
    # list supplies the (possibly fainter) neighbors that can contaminate them.
    flagged = neighbor_contamination_flag_sky(
        radecs[contaminant],
        mags[contaminant],
        fwhm_arcsec,
        target_mask=target[contaminant],
    )
    flagged_target = flagged[target[contaminant]]
    photometry_coords = SkyCoord(target_radecs[~flagged_target], unit="deg")

    bayer_masks = generate_bayer_masks(
        (metadata["height"], metadata["width"]),
        metadata,
        append_l4=append_l4,
    )

    return BatchPrep(
        radecs=target_radecs,
        photometry_coords=photometry_coords,
        cnn=cnn,
        bayer_masks=bayer_masks,
        center=center,
        fov_rad=metadata["fov_rad"],
        shape=(metadata["height"], metadata["width"]),
    )


def check_frame_consistency(file, header, prep):
    """
    Reject a frame whose pointing or shape disagrees with the batch prep.

    `prepare_batch` derives the field pointing, FOV, and image shape from the
    first frame and queries Gaia once for that field. A later frame that drifted
    off the field (a slew, a meridian flip, the wrong target) or has a different
    shape would be photometered against a catalog that no longer covers it,
    producing silently wrong results -- so reject it instead.

    Parameters
    ----------
    file : str or Path
        The frame being checked (attached to any raised error).
    header : astropy.io.fits.Header
        The frame's FITS header.
    prep : BatchPrep
        The batch prep whose ``center``, ``fov_rad``, and ``shape`` the frame is
        checked against.

    Raises
    ------
    FrameError
        If the frame's shape or pointing is inconsistent with the prep.
    FrameMetadataError
        If the header lacks the keywords needed to perform the checks.
    """
    try:
        shape = (header["NAXIS2"], header["NAXIS1"])
    except KeyError as exc:
        msg = f"missing required header keyword {exc.args[0]!r}"
        raise FrameMetadataError(msg, file=file) from exc
    if shape != tuple(prep.shape):
        msg = f"frame shape {shape} does not match batch shape {tuple(prep.shape)}"
        raise FrameError(msg, file=file)

    try:
        frame_center = SkyCoord(header["RA"], header["DEC"], unit="deg")
    except KeyError as exc:
        msg = f"missing pointing header keyword {exc.args[0]!r}"
        raise FrameMetadataError(msg, file=file) from exc
    center = SkyCoord(prep.center[0], prep.center[1], unit="deg")
    offset = center.separation(frame_center).deg
    if offset > prep.fov_rad:
        msg = (
            f"frame pointing is {offset:.3f} deg from the batch center, "
            f"beyond the {prep.fov_rad:.3f} deg field radius"
        )
        raise FrameError(msg, file=file)


def process_batch(
    files,
    prep,
    *,
    user_specific_metadata,
    output_dir=None,
    output_suffix=".star",
    fail_fast=True,
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
        Directory to write the per-frame photometry results to. Default None,
        which runs in in-memory mode and returns the photometry tables instead
        of writing files. When a directory is given it is created if it does not
        already exist.

    output_suffix : str, optional
        Suffix for the output files. Default ".star".
    fail_fast : bool, optional
        How to handle an *unexpected* error (one that is not a `FrameError`)
        while processing a frame. If True (default), re-raise it so genuine
        bugs surface. If False, log it at ERROR and continue with the next
        frame -- the robust mode for unattended runs. Expected per-frame
        failures (`FrameError` and its subclasses) are always logged and
        skipped regardless of this flag.

    Returns
    -------
    dict
        Mapping of each successfully-processed input file to its result. In
        in-memory mode (``output_dir`` is None) the value is the
        ``{filter: Table}`` photometry result; in write-to-disk mode the value
        is the written output ``Path`` (the tables are not held in memory).
        Frames that raise a `FrameError` (too few stars, unsolvable WCS, ...)
        are skipped with a logged warning and omitted from the result.

    Raises
    ------
    Exception
        Any unexpected (non-`FrameError`) error raised while processing a frame
        is re-raised when ``fail_fast`` is True (the default).
    """
    results = {}
    # Create the output directory once, before the loop, so a missing parent
    # fails fast instead of partway through the batch.
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    for file in files:
        try:
            check_frame_consistency(file, fits.getheader(file), prep)
            by_filter = process_one_image(
                file,
                user_specific_metadata,
                prep.radecs,
                prep.cnn,
                prep.bayer_masks,
                input_photometry_coords=prep.photometry_coords,
            )
        except FrameError as exc:
            # Expected per-frame failure: skip the frame and keep going. exc is
            # the human-readable headline; exc_info=True captures the chained
            # __cause__ (e.g. the original twirl traceback) so no detail is lost.
            # Some raisers (build_photometry_table, eloy_to_starlist) do not know
            # the path, so label the error with the current file here.
            if exc.file is None:
                exc.file = file
            logger.warning("skipping %s: %s", file, exc.reason, exc_info=True)
            continue
        except Exception:
            # Unexpected error (a bug, not a bad frame): surface it by default;
            # only swallow-and-continue when the caller opted into robust mode.
            if fail_fast:
                raise
            logger.exception("unexpected error on %s", file)
            continue
        else:
            # The frame processed cleanly. Writing its output is deliberately
            # outside the try: a write failure (bad output_dir, permissions,
            # full disk) is systemic, not a property of this frame, so it must
            # abort the run rather than be skipped as a "bad frame".
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
