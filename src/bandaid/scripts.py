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

import csv
import glob
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from dateutil import parser
from eloy.ballet.model import Ballet

from .catalog import cached_gaia_radecs
from .config import PhotometryConfig
from .exceptions import (
    BatchPrepError,
    FrameError,
    FrameMetadataError,
    TooFewStarsError,
    WCSSolveError,
)
from .image2sl_qt import generate_bayer_masks
from .photometry import (
    N_GAIA_STARS_ALIGN_RETRY,
    calibration_sequence,
    good_star_mask,
    metadata_from_header,
    neighbor_contamination_flag_sky,
    process_one_image,
)
from .writers import write_starlist_set

# Per-frame QA manifest written alongside the starlists in write-to-disk mode.
# The columns are the run-quality signals the pipeline already computes; a
# degrading night (clouds, rising airmass) shows up at a glance and the manifest
# enables a future partial-batch resume.
QA_MANIFEST_FILENAME = "qa_manifest.csv"
QA_MANIFEST_COLUMNS = (
    "file",
    "status",
    "n_detected",
    "sky_median",
    "fwhm",
    "wcs_solved",
    "n_good_stars",
    "n_centroid_drift",
    "n_drift_rejected",
)

logger = logging.getLogger(__name__)

__all__ = [
    "BatchPrep",
    "check_frame_consistency",
    "expand_frame_paths",
    "photometer_frames",
    "prepare_batch",
    "process_batch",
]

# Filename endings treated as FITS frames when expanding directory/glob arguments.
# Seestar writes ``.fit``; the others (and their gzip-compressed forms, which
# astropy opens transparently) are accepted for telescopes that use them.
_FITS_SUFFIXES = (
    ".fit",
    ".fits",
    ".fts",
    ".fit.gz",
    ".fits.gz",
    ".fts.gz",
)


def _quiet_hf_xet():
    """
    Best-effort: silence the native ``hf_xet`` unauthenticated-request warning.

    On the first weights download, ``hf_hub_download`` routes through the native
    ``hf_xet`` accelerator, which prints a "sending unauthenticated requests to
    the HF Hub ... faster downloads" line straight to stderr -- not a Python
    warning or log record, so it cannot be filtered the usual way. Disabling xet
    keeps the download working (the ``.npz`` is tiny and cached once) and avoids
    that line. ``setdefault`` so a user who set ``HF_HUB_DISABLE_XET`` (or who
    wants xet) is never overridden; setting ``HF_TOKEN`` is the fully-correct fix
    -- it both keeps xet acceleration and silences the warning. Best-effort
    because the exact behaviour depends on the installed ``hf_xet`` version.
    """
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _is_fits(path):
    """
    Return whether ``path`` ends with a recognised FITS suffix.

    Uses the whole name (not :attr:`pathlib.Path.suffix`) so the compound
    compressed forms such as ``.fits.gz`` are matched as well.

    Parameters
    ----------
    path : str or pathlib.Path
        The path to test.

    Returns
    -------
    bool
        True if the name ends with one of `_FITS_SUFFIXES`.
    """
    return str(path).lower().endswith(_FITS_SUFFIXES)


def expand_frame_paths(paths):
    """
    Expand the raw positional path arguments into a sorted list of frame paths.

    Parameters
    ----------
    paths : collections.abc.Iterable of str
        The raw positional arguments: directories, glob patterns, and/or file
        paths.

    Returns
    -------
    list of str
        The expanded, de-duplicated, lexically sorted (resolved) frame paths.

    Raises
    ------
    FileNotFoundError
        If a literal (non-glob) path does not exist.
    ValueError
        If a literal path exists but is not a FITS frame.

    Notes
    -----
    Each argument may be a directory (expanded to the FITS frames it contains), a
    glob pattern (expanded against the filesystem, then filtered to FITS frames),
    or a literal file path. Directory and glob matches that are not FITS *files*
    are silently skipped -- including a directory or symlink whose name merely
    ends in a FITS suffix (e.g. ``bundle.fits/``), which would otherwise blow up
    later in ``fits.getheader``. A literal path is validated to exist and to be a
    FITS file, so a typo fails here with a clear error rather than as a traceback
    deep in ``prepare_batch``.

    The combined result is de-duplicated by *resolved* path -- so the same file
    reached two ways (a directory and an explicit path, ``a.fit`` vs ``./a.fit``)
    appears once, while two distinct files that merely share a basename in
    different directories are both kept -- and returned sorted so a batch is
    processed in a deterministic order.
    """
    # Map resolved path -> the path object, so duplicates collapse by identity.
    seen = {}
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            candidates = [
                child for child in path.iterdir() if child.is_file() and _is_fits(child)
            ]
        elif glob.has_magic(raw):
            candidates = [
                match
                for match in map(Path, glob.glob(raw))  # noqa: PTH207 -- need glob
                if match.is_file() and _is_fits(match)
            ]
        else:
            if not path.exists():
                msg = f"no such file: {raw}"
                raise FileNotFoundError(msg)
            if not path.is_file() or not _is_fits(path):
                msg = f"{raw} is not a FITS frame (expected one of {_FITS_SUFFIXES})"
                raise ValueError(msg)
            candidates = [path]
        for candidate in candidates:
            seen.setdefault(candidate.resolve(), candidate)
    return [str(resolved) for resolved in sorted(seen)]


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
    config : PhotometryConfig
        The photometry configuration to apply to every frame in the batch.
    """

    radecs: np.ndarray
    photometry_coords: SkyCoord
    cnn: object
    bayer_masks: dict
    center: tuple
    fov_rad: float
    shape: tuple
    config: PhotometryConfig = field(default_factory=PhotometryConfig)


def prepare_batch(
    first_file,
    *,
    cnn,
    config=None,
    append_l4=True,
):
    """
    Compute the once-per-batch photometry inputs from the first frame.

    Runs a single detection pass on ``first_file`` (no WCS solve) to obtain the
    FWHM and image metadata, queries Gaia for the field (propagating the J2015.5
    DR2 positions to the frame's observation epoch via proper motions), drops
    contaminated sources, and builds the Bayer masks.

    Parameters
    ----------
    first_file : str or Path
        Path to the first FITS frame in the batch. Used to derive the field
        pointing/FOV, plate scale, Bayer pattern, image shape, and FWHM. All
        frames in the batch are assumed to share these.
    cnn : object
        The ``eloy`` Ballet centroiding model to carry through to every frame.
    config : PhotometryConfig or None, optional
        Photometry configuration carried on the returned `BatchPrep` and applied
        to every frame. Its ``instrument`` settings drive the first-frame FWHM
        detection and the contamination flagging here (evaluated at the largest
        ``apertures`` radius, with the first-frame FWHM padded by the
        instrument's ``contamination_seeing_margin``), and its
        ``source_selection`` settings supply the Gaia target/contaminant
        magnitude limits. If None (default), a default ``PhotometryConfig`` is
        used.
    append_l4 : bool, optional
        Whether to add a full-frame "L4" luminance channel to the Bayer masks.
        Default True, matching `photometer_frames` and the CLI so composing
        `prepare_batch` + `process_batch` by hand yields the same channels as
        the CLI for the same inputs.

    Returns
    -------
    BatchPrep
        The reusable prep bundle for the batch.

    Raises
    ------
    BatchPrepError
        If too few stars are detected in ``first_file`` to measure an FWHM, so
        the batch preparation cannot be built.
    FrameMetadataError
        If the first frame's metadata has no parseable observation time
        (``obs_time``, usually mapped from ``DATE-OBS``), which is needed to
        propagate the Gaia positions to the observation epoch.
    """
    # A too-few-stars failure on the *first* frame is fatal for the whole batch
    # (no FWHM/pointing to prepare from), so translate the recoverable
    # per-frame TooFewStarsError into a fatal BatchPrepError.
    config = config or PhotometryConfig()
    instrument = config.instrument
    try:
        # Pass the CNN so the FWHM (which sizes the photometry aperture) is measured
        # by re-centroiding detections, decoupling it from the detection opening.
        # detect_on_bayer_balanced and fwhm_n_stars must match the per-frame call
        # in process_one_image (which detects on bayer-balanced data by default),
        # so the batch-gating FWHM is measured in the same detection regime as
        # the photometry it protects.
        _, metadata, _, fwhm_pix, _ = calibration_sequence(
            first_file,
            threshold=instrument.thresh,
            opening=instrument.detection_opening,
            detect_on_bayer_balanced=True,
            cnn=cnn,
            fwhm_cutout_half=instrument.fwhm_cutout_half,
            fwhm_n_stars=instrument.fwhm_n_stars,
            profile=instrument,
        )
    except TooFewStarsError as exc:
        msg = f"too few stars detected in {first_file!r} to prepare the batch"
        raise BatchPrepError(msg) from exc

    # Gaia DR2 positions are J2015.5; propagate them to the observation epoch so
    # high-proper-motion stars are placed where the frames actually see them.
    # https://github.com/mwcraig/bandaid/issues/56
    # Validate obs_time up front: a frame without DATE-OBS resolves it to None,
    # and letting that fail inside the Gaia try block below would surface a
    # metadata problem as a misleading "could not query Gaia" error.
    obs_time = metadata.get("obs_time")
    if obs_time is None:
        msg = (
            "no observation time in the first frame's metadata (obs_time, "
            "usually mapped from DATE-OBS); it is needed to propagate Gaia "
            "positions to the observation epoch"
        )
        raise FrameMetadataError(msg, file=first_file)
    # dateutil mirrors build_photometry_table's tolerant DATE-OBS parsing, so
    # the Gaia epoch accepts the same header date forms as the rest of the
    # pipeline (Time() alone is stricter than dateutil).
    try:
        obs_epoch = Time(parser.parse(obs_time))
    # OverflowError: dateutil raises it for all-digit strings too large for a
    # C long (e.g. a corrupted numeric DATE-OBS), documented in parser.parse.
    except (ValueError, TypeError, OverflowError) as exc:
        msg = f"could not parse observation time (obs_time) {obs_time!r}"
        raise FrameMetadataError(msg, file=first_file) from exc

    # fov_rad is a field *radius*; cached_gaia_radecs takes the full field and
    # halves it internally (matching the established twirl.gaia_radecs usage).
    center = (metadata["ra"], metadata["dec"])
    # A Gaia query failure (network/service error) is fatal for the whole batch;
    # surface it as a BatchPrepError instead of a raw astroquery/requests error.
    try:
        radecs, mags = cached_gaia_radecs(
            center, 2 * metadata["fov_rad"], obs_epoch=obs_epoch
        )
    except Exception as exc:
        msg = f"could not query Gaia for the field at {center}"
        raise BatchPrepError(msg) from exc
    # Decouple the stars we *measure* (targets, cut at gaia_mag_limit) from the
    # stars that can *contaminate* them (a deeper list down to
    # contaminant_mag_limit). A real star fainter than the photometry limit still
    # spills into a brighter target's aperture, so flagging runs against the
    # deeper list -- but only targets are ever flagged/dropped.
    # SourceSelectionConfig has already defaulted and finiteness-checked these.
    gaia_mag_limit = config.source_selection.gaia_mag_limit
    contaminant_mag_limit = config.source_selection.contaminant_mag_limit

    target = mags <= gaia_mag_limit
    contaminant = mags <= contaminant_mag_limit
    target_radecs = radecs[target]

    # Without enough reference stars no frame can solve a WCS, so fail the batch
    # now with a clear message rather than letting every frame fail later.
    if len(target_radecs) < N_GAIA_STARS_ALIGN_RETRY:
        msg = (
            f"Gaia returned only {len(target_radecs)} stars brighter than "
            f"{gaia_mag_limit} for the field at {center}; need at least "
            f"{N_GAIA_STARS_ALIGN_RETRY} to solve a WCS"
        )
        raise BatchPrepError(msg)

    fwhm_arcsec = fwhm_pix * metadata["pixscale"]
    # The flag is computed once, from the first frame's FWHM, but applied to
    # every frame of the batch, so evaluate it at a pessimistically softened
    # seeing (FWHM * contamination_seeing_margin): pairs that would become
    # contaminated as seeing degrades during the night are dropped up front.
    # https://github.com/mwcraig/bandaid/issues/64
    flag_fwhm_arcsec = fwhm_arcsec * instrument.contamination_seeing_margin
    # Asymmetric flagging: only targets can be flagged, but the deeper contaminant
    # list supplies the (possibly fainter) neighbors that can contaminate them.
    # The contamination model scales with the aperture area, so it is evaluated
    # at the largest configured aperture radius.
    flagged = neighbor_contamination_flag_sky(
        radecs[contaminant],
        mags[contaminant],
        flag_fwhm_arcsec,
        tolerance=instrument.contamination_tolerance,
        beta=instrument.moffat_beta,
        aperture_radius_fwhm=max(config.apertures.radii),
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
        config=config,
    )


def check_frame_consistency(file, header, prep):
    """
    Reject a frame whose pointing or shape disagrees with the batch prep.

    `prepare_batch` derives the field pointing, FOV, and image shape from the
    first frame and queries Gaia once for that field. A later frame that drifted
    off the field (a slew, a meridian flip, the wrong target) or has a different
    shape would be photometered against a catalog that no longer covers it,
    producing silently wrong results -- so reject it instead.

    The header is resolved through the batch instrument's ``header_map``
    (``prep.config.instrument``), the same dialect that resolved the prep's
    ``center``/``shape`` from the first frame -- so an instrument whose pointing
    lives under different keywords than the Seestar's is compared consistently
    (issue #59).

    Parameters
    ----------
    file : str or Path
        The frame being checked (attached to any raised error).
    header : astropy.io.fits.Header
        The frame's FITS header.
    prep : BatchPrep
        The batch prep whose ``center``, ``fov_rad``, and ``shape`` the frame is
        checked against, and whose ``config.instrument`` supplies the
        ``header_map`` dialect used to read the header.

    Raises
    ------
    FrameError
        If the frame's shape or pointing is inconsistent with the prep.
    FrameMetadataError
        If the header cannot be resolved into the metadata needed to perform
        the checks.
    """
    try:
        metadata = metadata_from_header(header, profile=prep.config.instrument)
    except FrameMetadataError as exc:
        # metadata_from_header has only the header, not the path; label it here.
        exc.file = file
        raise
    shape = (metadata["height"], metadata["width"])
    if shape != tuple(prep.shape):
        msg = f"frame shape {shape} does not match batch shape {tuple(prep.shape)}"
        raise FrameError(msg, file=file)

    # An "@KEY" directive whose keyword is absent resolves to None rather than
    # raising, so the missing-pointing case must be caught explicitly.
    ra = metadata.get("ra")
    dec = metadata.get("dec")
    if ra is None or dec is None:
        msg = "header resolved no pointing (ra/dec) through the instrument header_map"
        raise FrameMetadataError(msg, file=file)
    frame_center = SkyCoord(ra, dec, unit="deg")
    center = SkyCoord(prep.center[0], prep.center[1], unit="deg")
    offset = center.separation(frame_center).deg
    if offset > prep.fov_rad:
        msg = (
            f"frame pointing is {offset:.3f} deg from the batch center, "
            f"beyond the {prep.fov_rad:.3f} deg field radius"
        )
        raise FrameError(msg, file=file)


def _qa_record_ok(file, by_filter):
    """
    Build the QA manifest record for a frame that processed cleanly.

    Diagnostics are pulled defensively from a representative channel (L4 if
    present, else the first), so a frame missing a given column simply records a
    blank for it rather than failing the whole manifest.

    ``n_centroid_drift`` and ``n_drift_rejected`` instrument the
    `centroid_drift` flag (see `centroid_drift_flag`) without wiring it into
    filtering: ``n_centroid_drift`` is every flagged star in the frame, and
    ``n_drift_rejected`` is the subset that is also `good_star_mask`-passing --
    the marginal effect a future gate would have, since most drifted stars are
    already dropped by the flux/error/bounds cuts. Data recorded before the
    proper-motion fix (#56) overcounts both: the flag fired preferentially on
    high-proper-motion stars whose *catalog* position was stale, not on genuine
    drift.

    Parameters
    ----------
    file : str or Path
        The processed input frame.
    by_filter : dict of {str: astropy.table.Table}
        The ``process_one_image`` result for this frame.

    Returns
    -------
    dict
        One manifest row keyed by `QA_MANIFEST_COLUMNS`.
    """
    if "L4" in by_filter:
        representative = by_filter["L4"]
    else:
        representative = next(iter(by_filter.values()))
    meta = representative.meta
    full_meta = meta.get("full_image_meta", {})
    cols = set(representative.colnames)

    n_detected = (
        int(representative["stars_in_exp"][0]) if "stars_in_exp" in cols else None
    )
    # An edge-of-frame or fully-masked annulus yields a NaN bkgd_count (see the
    # NaN contract in measure_photometry); keep those rows out of the median so
    # one bad annulus cannot poison the frame's QA value.
    sky_median = None
    if "bkgd_count" in cols:
        bkgd = np.asarray(representative["bkgd_count"])
        finite = bkgd[np.isfinite(bkgd)]
        if len(finite):
            sky_median = float(np.median(finite))
    n_good_stars = None
    has_phot_cols = {"tot_count", "count_err", "x", "y"} <= cols
    has_bounds = {"width", "height"} <= set(full_meta)
    good = None
    if has_phot_cols and has_bounds:
        good = good_star_mask(representative, full_meta)
        n_good_stars = int(np.sum(good))

    # The drift flag is computed from centroid_coords/aligned_coords/fwhm before
    # channel masking, so it is identical across TR/TG/TB/L4 and the
    # representative table alone is enough -- no cross-channel bookkeeping.
    n_centroid_drift = None
    n_drift_rejected = None
    if "centroid_drift" in cols:
        drift = np.asarray(representative["centroid_drift"], dtype=bool)
        n_centroid_drift = int(np.sum(drift))
        if good is not None:
            n_drift_rejected = int(np.sum(drift & good))

    return {
        "file": str(file),
        "status": "ok",
        "n_detected": n_detected,
        "sky_median": sky_median,
        "fwhm": meta.get("fwhm"),
        "wcs_solved": True,
        "n_good_stars": n_good_stars,
        "n_centroid_drift": n_centroid_drift,
        "n_drift_rejected": n_drift_rejected,
    }


def _qa_record_failed(file, status, *, wcs_solved=None):
    """
    Build the QA manifest record for a skipped or errored frame.

    Parameters
    ----------
    file : str or Path
        The input frame.
    status : str
        Outcome label, e.g. ``"skipped: WCSSolveError"`` or ``"error: KeyError"``.
    wcs_solved : bool or None, optional
        ``False`` for a WCS solve failure, otherwise ``None`` (the frame failed
        before -- or unrelated to -- the solve, so it is left blank).

    Returns
    -------
    dict
        One manifest row keyed by `QA_MANIFEST_COLUMNS`, diagnostics blank.
    """
    record = dict.fromkeys(QA_MANIFEST_COLUMNS)
    record["file"] = str(file)
    record["status"] = status
    record["wcs_solved"] = wcs_solved
    return record


def _write_qa_manifest(path, records):
    """
    Write the per-frame QA records to a CSV manifest.

    Parameters
    ----------
    path : pathlib.Path
        Destination CSV path.
    records : list of dict
        Per-frame rows keyed by `QA_MANIFEST_COLUMNS`; ``None`` values are
        written as empty cells.
    """
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=QA_MANIFEST_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: "" if record.get(key) is None else record[key]
                    for key in QA_MANIFEST_COLUMNS
                }
            )


def _unique_output_paths(files, output_dir, suffix):
    """
    Map each input frame to a unique output path under ``output_dir``.

    Parameters
    ----------
    files : list of str or pathlib.Path
        The input frames, in the order they will be written.
    output_dir : str or pathlib.Path
        Directory the output paths live in.
    suffix : str
        Suffix for the output files (e.g. ``".star"``).

    Returns
    -------
    dict
        Mapping of each input frame to its unique output `~pathlib.Path`.

    Notes
    -----
    Output names are kept flat and clean in the common case and grow structure
    only when needed:

    * When every frame comes from a single directory -- the typical "one night,
      one folder" run -- the basenames are already unique, so each output is a
      flat ``output_dir/<stem><suffix>``.
    * When frames come from a mix of directories the source tree is mirrored as
      ``output_dir/<dirname>/<stem><suffix>``, so identically named frames from
      different directories stay distinct without munging the file name. Two
      distinct source directories that share a basename are disambiguated with a
      numeric suffix on the subdirectory.

    A residual basename collision within a single output directory (e.g. the same
    frame referenced twice, or two inputs differing only by extension) falls back
    to a numeric suffix on the file name.
    """
    output_dir = Path(output_dir)
    paths = [Path(file) for file in files]
    parents = [path.resolve().parent for path in paths]

    # A single source directory writes flat names; a mix mirrors the tree, so
    # assign each distinct directory a unique subdirectory name up front.
    subdir_for = {}
    if len(set(parents)) > 1:
        used_subdirs = set()
        for parent in sorted(set(parents), key=str):
            base = parent.name or "root"
            name = base
            index = 1
            while name in used_subdirs:
                name = f"{base}_{index}"
                index += 1
            used_subdirs.add(name)
            subdir_for[parent] = name

    mapping = {}
    used = set()
    for file, path, parent in zip(files, paths, parents, strict=True):
        target_dir = output_dir / subdir_for[parent] if subdir_for else output_dir
        stem = path.stem
        name = stem + suffix
        index = 1
        while target_dir / name in used:
            name = f"{stem}_{index}{suffix}"
            index += 1
        used.add(target_dir / name)
        mapping[file] = target_dir / name
    return mapping


def _ensure_output_dirs(output_dir, output_paths):
    """
    Create ``output_dir`` and every subdirectory the planned outputs require.

    Doing this once, up front, makes a missing or unwritable parent fail fast
    rather than partway through the batch, and creates the per-source-directory
    subdirectories the mirrored-tree layout needs (see `_unique_output_paths`).

    Parameters
    ----------
    output_dir : str or pathlib.Path
        The root output directory.
    output_paths : dict
        Mapping of input frame to planned output `~pathlib.Path`, as returned by
        `_unique_output_paths`.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for output_path in output_paths.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)


def process_batch(
    files,
    prep,
    *,
    user_specific_metadata,
    output_dir=None,
    output_suffix=".star",
    write_frame=write_starlist_set,
    fail_fast=True,
    write_qa_manifest=True,
    qa_manifest_name=QA_MANIFEST_FILENAME,
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
        already exist. Frames from a single source directory are written flat as
        ``<stem><output_suffix>``; frames from a mix of directories mirror the
        source tree as ``<dirname>/<stem><output_suffix>`` so identically named
        frames stay distinct (see `_unique_output_paths`).

    output_suffix : str, optional
        Suffix for the output files. Default ".star".
    write_frame : collections.abc.Callable, optional
        The per-frame writer used in write-to-disk mode:
        ``write(frame_result, output_path)`` records one frame's
        ``{filter: Table}`` result and returns what to store as the frame's entry
        in the result mapping (see `bandaid.writers`). Default
        `write_starlist_set` (one `StarListSet` JSON per frame). Ignored in
        in-memory mode (``output_dir`` is None). A writer exception propagates as
        it does for the default writer -- the write step stays outside the
        per-frame error handling, since a write failure is systemic.
    fail_fast : bool, optional
        How to handle an *unexpected* error (one that is not a `FrameError`)
        while processing a frame. If True (default), re-raise it so genuine
        bugs surface. If False, log it at ERROR and continue with the next
        frame -- the robust mode for unattended runs. Expected per-frame
        failures (`FrameError` and its subclasses) are always logged and
        skipped regardless of this flag.
    write_qa_manifest : bool, optional
        Whether to write the per-frame QA manifest in write-to-disk mode.
        Default True -- the manifest is cheap and makes a degrading night
        self-evident the first time it goes bad, before anyone thinks to ask
        for it. Set False to write only the `.star` files and leave the rest
        of ``output_dir`` untouched. Ignored in in-memory mode (no directory
        to write to).
    qa_manifest_name : str, optional
        Filename for the QA manifest within ``output_dir``. Default
        `QA_MANIFEST_FILENAME`.

    Returns
    -------
    dict
        Mapping of each successfully-processed input file to its result. In
        in-memory mode (``output_dir`` is None) the value is the
        ``{filter: Table}`` photometry result; in write-to-disk mode the value
        is the written output ``Path`` (the tables are not held in memory).
        Frames that raise a `FrameError` (too few stars, unsolvable WCS, ...)
        are skipped with a logged warning and omitted from the result.

        In write-to-disk mode, unless ``write_qa_manifest`` is False, a
        per-frame QA manifest (``qa_manifest_name``) is also written to
        ``output_dir``, with one row per input frame recording its status
        (``ok`` / ``skipped: <FrameError type>`` / ``error: <type>``) and the
        available run-quality signals (`QA_MANIFEST_COLUMNS`).

    Raises
    ------
    Exception
        Any unexpected (non-`FrameError`) error raised while processing a frame
        is re-raised when ``fail_fast`` is True (the default).
    """
    results = {}
    # One QA record per frame (ok/skipped/error), written to a manifest at the
    # end when in write-to-disk mode and the caller has not opted out.
    write_manifest = output_dir is not None and write_qa_manifest
    manifest_records = []
    # Materialize the frames so the output names can be planned up front: two
    # frames sharing a basename must not collide on disk (see _unique_output_paths).
    files = list(files)
    output_paths = (
        _unique_output_paths(files, output_dir, output_suffix)
        if output_dir is not None
        else {}
    )
    # Create the output directory (and the mirrored-tree subdirectories) up
    # front so a missing or unwritable parent fails fast.
    if output_dir is not None:
        _ensure_output_dirs(output_dir, output_paths)
    for idx, file in enumerate(files, 1):
        # Per-frame progress. Invisible by default (the package logger has only a
        # NullHandler); `bandaid process --verbose` routes it to the terminal via
        # configure_logging, alongside the skip/error warnings logged below.
        logger.info("processing %d/%d: %s", idx, len(files), file)
        try:
            check_frame_consistency(file, fits.getheader(file), prep)
            by_filter = process_one_image(
                file,
                user_specific_metadata,
                prep.radecs,
                prep.cnn,
                prep.bayer_masks,
                config=prep.config,
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
            manifest_records.append(
                _qa_record_failed(
                    file,
                    f"skipped: {type(exc).__name__}",
                    wcs_solved=False if isinstance(exc, WCSSolveError) else None,
                )
            )
            continue
        except Exception as exc:
            # Unexpected error (a bug, not a bad frame): surface it by default;
            # only swallow-and-continue when the caller opted into robust mode.
            if fail_fast:
                raise
            logger.exception("unexpected error on %s", file)
            manifest_records.append(
                _qa_record_failed(file, f"error: {type(exc).__name__}")
            )
            continue
        else:
            # The frame processed cleanly. Writing its output is deliberately
            # outside the try: a write failure (bad output_dir, permissions,
            # full disk) is systemic, not a property of this frame, so it must
            # abort the run rather than be skipped as a "bad frame".
            manifest_records.append(_qa_record_ok(file, by_filter))
            if output_dir is not None:
                results[file] = write_frame(by_filter, output_paths[file])
            else:
                results[file] = by_filter

    # Persist the per-frame QA manifest next to the starlists. Only written in
    # write-to-disk mode (in-memory mode has no directory to write it to) and
    # only when the caller has not opted out.
    if write_manifest:
        _write_qa_manifest(Path(output_dir) / qa_manifest_name, manifest_records)
    return results


def photometer_frames(
    files,
    *,
    config=None,
    cnn=None,
    weights=None,
    user_specific_metadata=None,
    append_l4=True,
    output_dir=".",
    output_suffix=".star",
    write_frame=write_starlist_set,
    fail_fast=False,
    write_qa_manifest=True,
):
    """
    Expand a set of file arguments and measure per-frame photometry for each.

    The high-level convenience behind ``bandaid process``: it does the file-name
    expansion (`expand_frame_paths`), builds the Ballet centroider, and runs
    `prepare_batch` (seeded from the first frame) followed by `process_batch`.
    Driving the whole flow from Python is one call to this function; the CLI is a
    thin dressing over it.

    Parameters
    ----------
    files : collections.abc.Iterable of str
        Raw positional arguments -- directories, glob patterns, and/or file paths
        -- expanded by `expand_frame_paths`.
    config : PhotometryConfig or None, optional
        Configuration carried through the batch. None (default) uses a default
        `PhotometryConfig` (Seestar50).
    cnn : object or None, optional
        A pre-built Ballet centroider. None (default) builds one from ``weights``.
    weights : str or None, optional
        Path to Ballet weights used when ``cnn`` is None; None downloads the
        defaults from HuggingFace.
    user_specific_metadata : dict or None, optional
        Per-frame user metadata recorded with each output. None (default) is an
        empty dict.
    append_l4 : bool, optional
        Whether to add a full-frame L4 luminance channel to the Bayer masks.
        Default True.
    output_dir : str or pathlib.Path or None, optional
        Directory to write the per-frame ``.star`` files (and QA manifest) into.
        Default ``"."``; None runs in in-memory mode (see `process_batch`).
    output_suffix : str, optional
        Suffix for the per-frame output files. Default ``".star"``.
    write_frame : collections.abc.Callable, optional
        Per-frame writer used in write-to-disk mode (see `process_batch` and
        `bandaid.writers`). Default `write_starlist_set` (the ``.star`` format).
    fail_fast : bool, optional
        Whether to re-raise unexpected per-frame errors instead of skipping the
        frame. Default False (the robust mode for unattended runs).
    write_qa_manifest : bool, optional
        Whether to write a per-frame QA manifest alongside the outputs. Default
        True.

    Returns
    -------
    tuple of (list of str, dict)
        The expanded frame list and the `process_batch` result mapping (each
        successfully-processed frame to its output, see `process_batch`).

    Raises
    ------
    ValueError
        If the arguments expand to no FITS frames. `expand_frame_paths` may also
        raise `ValueError`/`FileNotFoundError` for a malformed path argument.
    """
    frames = expand_frame_paths(files)
    if not frames:
        msg = "no FITS frames found in the given files/directories"
        raise ValueError(msg)

    config = config or PhotometryConfig()
    if cnn is None:
        _quiet_hf_xet()
        cnn = Ballet(model_file=weights)

    prep = prepare_batch(frames[0], cnn=cnn, config=config, append_l4=append_l4)
    results = process_batch(
        frames,
        prep,
        user_specific_metadata=user_specific_metadata or {},
        output_dir=output_dir,
        output_suffix=output_suffix,
        write_frame=write_frame,
        fail_fast=fail_fast,
        write_qa_manifest=write_qa_manifest,
    )
    return frames, results
