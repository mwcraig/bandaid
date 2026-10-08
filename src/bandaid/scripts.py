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
import time
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from .ballet import Ballet
from .catalog import query_field_catalog
from .config import PhotometryConfig
from .exceptions import (
    BatchPrepError,
    FrameError,
    FrameMetadataError,
    InstrumentDetectionError,
    TooFewStarsError,
    WCSSolveError,
)
from .image2sl_qt import generate_bayer_masks
from .instruments import detect_instrument, resolve_config_instrument
from .photometry import (
    N_GAIA_STARS_ALIGN_RETRY,
    LoadedFrame,
    _load_frame,
    _parse_obs_time,
    _solve_pool_near,
    calibration_sequence,
    estimate_center_from_header,
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
    "pointing_offset_deg",
    "wcs_pixscale",
    "solve_offset_deg",
    "n_good_stars",
    "n_snr20",
    "dropped_filters",
    "n_centroid_drift",
    "n_drift_rejected",
    "n_forced_measured",
    "n_edge_dropped",
    "g_cut",
    "n_cnn_class",
    "plane_fallback",
    "plane_n_used",
    "plane_n_clipped",
    "plane_rms",
    "plane_dx_center",
    "plane_dy_center",
    "plane_dx_slope_x",
    "plane_dx_slope_y",
    "plane_dy_slope_x",
    "plane_dy_slope_y",
)

# The QA manifest columns filled from a frame's centroid-policy summary.
_CENTROID_MODEL_COLUMNS = (
    "g_cut",
    "n_cnn_class",
    "plane_fallback",
    "plane_n_used",
    "plane_n_clipped",
    "plane_rms",
    "plane_dx_center",
    "plane_dy_center",
    "plane_dx_slope_x",
    "plane_dx_slope_y",
    "plane_dy_slope_x",
    "plane_dy_slope_y",
)

# SNR at or above which a star counts toward the manifest's ``n_snr20`` solve-quality
# proxy (separates good, degraded and false plate solves).
QA_SNR_THRESHOLD = 20

logger = logging.getLogger(__name__)

__all__ = [
    "BatchPrep",
    "LoadedFrame",
    "check_frame_consistency",
    "estimate_center_from_header",
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
    later in `~bandaid.photometry._load_frame`. A literal path is validated to
    exist and to be a FITS file, so a typo fails here with a clear error rather
    than as a traceback deep in ``prepare_batch``.

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
        The centroiding model to use for every frame: any object with a
        ``centroid(cutouts) -> (N, 2)`` method, such as
        `~bandaid.ballet.Ballet`.
    bayer_masks : dict
        Mapping of filter name to Bayer mask, as returned by
        `generate_bayer_masks`.
    center : tuple of float
        ``(ra, dec)`` in degrees (ICRS) of the field center the Gaia catalog was
        queried for (the header pointing converted by
        `estimate_center_from_header`). `check_frame_consistency`
        compares each frame's estimated center against it to reject frames that
        drifted off the field.
    fov_rad : float
        Field radius in degrees; the maximum allowed offset of a frame's
        estimated center from ``center``. The catalog covers this radius plus
        the instrument's ``cone_radius_margin``, so a frame offset by more
        than the margin is only partly covered (and warned about).
    shape : tuple of int
        Expected ``(height, width)`` of every frame.
    config : PhotometryConfig
        The photometry configuration to apply to every frame in the batch.
        ``config.instrument`` must already be resolved (not None):
        `prepare_batch` is the only place meant to build a ``BatchPrep``, and
        it always resolves the instrument (explicitly or by detection) before
        constructing one -- see `__post_init__`.
    forced_targets : astropy.coordinates.SkyCoord or None
        The ICRS forced-target sky positions appended to
        ``photometry_coords`` by `prepare_batch`. None when the batch was
        prepared without any.
    instrument_auto_detected : bool
        Whether ``config.instrument`` was resolved by
        `~bandaid.instruments.detect_instrument` (the incoming config's
        ``instrument`` was None) rather than given explicitly
        (``--instrument``/``--profile``/``--config`` with a concrete
        instrument, or ``config=PhotometryConfig(instrument=...)``).
        `check_frame_consistency`'s batch-mixing guard applies to both, with
        one difference: an explicit selection is exempt from a header that
        matches *no* registered instrument (missing or malformed), whereas a
        header that positively identifies a *different* registered
        instrument, or is ambiguous between several, is rejected either way.
        An auto-detected batch is never exempt. Default False.
    build_l4 : bool
        Whether each frame also gets the full-frame "L4" luminance channel,
        built from the TR/TG/TB tables; handed to ``process_one_image``.
        Default True.
    gaia_g : numpy.ndarray or None
        Gaia G magnitude of each row of ``photometry_coords``, in the same
        order (NaN for a forced target, which has none). None when the batch
        was built without magnitudes. Default None.
    g_cut : float or None
        The batch's CNN-class magnitude cut: a star with ``G <= g_cut`` keeps
        its CNN centroid on every frame (see
        `~bandaid.photometry.centroid_with_catalog_model`).
        ``inf`` when the field has fewer than ``cnn_class_size`` targets in the
        cut circle; None when the policy is off. Default None.
    """

    radecs: np.ndarray
    photometry_coords: SkyCoord
    cnn: object
    bayer_masks: dict
    center: tuple
    fov_rad: float
    shape: tuple
    config: PhotometryConfig
    forced_targets: SkyCoord | None = None
    instrument_auto_detected: bool = False
    build_l4: bool = True
    gaia_g: np.ndarray | None = None
    g_cut: float | None = None

    def __post_init__(self) -> None:
        """
        Verify ``config.instrument`` was resolved and ``gaia_g`` is row-aligned.

        Raises
        ------
        ValueError
            If ``config.instrument`` is None, or ``gaia_g`` is given with a
            different length from ``photometry_coords``.

        Notes
        -----
        Constructing a ``BatchPrep`` with an unresolved (``None``) instrument
        would make `check_frame_consistency` re-run detection per frame
        instead of once for the batch -- so reject it here instead.
        """
        if self.config.instrument is None:
            msg = (
                "BatchPrep.config.instrument must be resolved (not None) -- "
                "construct BatchPrep via prepare_batch"
            )
            raise ValueError(msg)
        if self.gaia_g is not None and len(self.gaia_g) != len(self.photometry_coords):
            msg = (
                f"BatchPrep.gaia_g has {len(self.gaia_g)} entries but "
                f"photometry_coords has {len(self.photometry_coords)}: "
                "they must be row-aligned"
            )
            raise ValueError(msg)


def _with_forced_target_g(gaia_g, forced_targets):
    """
    Extend the targets' Gaia G with a NaN for each forced target.

    Parameters
    ----------
    gaia_g : numpy.ndarray
        Gaia G of the contamination-filtered targets.
    forced_targets : astropy.coordinates.SkyCoord or None
        The forced targets appended after those targets, or None.

    Returns
    -------
    numpy.ndarray
        `gaia_g` followed by one NaN per forced target: a forced target is
        absent from Gaia, so it has no magnitude.
    """
    if forced_targets is None:
        return gaia_g
    return np.concatenate([gaia_g, np.full(len(forced_targets), np.nan)])


def _cnn_class_g_cut(coords, gaia_g, center, shape, pixscale, class_size):
    """
    Return the Gaia G at or brighter than which a target is CNN-class.

    Parameters
    ----------
    coords : astropy.coordinates.SkyCoord
        Sky positions of the targets.
    gaia_g : numpy.ndarray
        Gaia G of each of `coords`; NaN rows (forced targets) are ignored.
    center : tuple of float
        ``(ra, dec)`` of the batch center, in degrees.
    shape : tuple of int
        ``(height, width)`` of the frame, in pixels.
    pixscale : float
        Plate scale, in arcseconds per pixel.
    class_size : int
        Number of targets that are CNN-class.

    Returns
    -------
    float
        The G of the `class_size`-th brightest target inside the circle, or
        ``inf`` when the circle holds fewer than `class_size` targets.

    Notes
    -----
    The circle is centred on `center` and has the area of the frame, so it
    holds about as many targets as a frame does and does not depend on where
    the frame's edges fall. The cut is a property of the catalog and the
    field, never of one frame, so a star's class is the same on every frame.
    """
    height, width = shape
    radius_deg = np.sqrt(height * width / np.pi) * pixscale / 3600.0
    ra, dec = center
    inside = coords.separation(SkyCoord(ra * u.deg, dec * u.deg)).deg <= radius_deg
    g_inside = np.sort(gaia_g[inside & np.isfinite(gaia_g)])
    if len(g_inside) < class_size:
        return float("inf")
    return float(g_inside[class_size - 1])


def _batch_g_cut(centroid_config, coords, gaia_g, center, shape, pixscale):
    """
    Compute and log the batch's CNN-class magnitude cut, or None if the policy is off.

    Parameters
    ----------
    centroid_config : `~bandaid.config.CentroidConfig`
        The centroid policy settings.
    coords : astropy.coordinates.SkyCoord
        Sky positions of the catalog targets.
    gaia_g : numpy.ndarray
        Gaia G of each of `coords`.
    center : tuple of float
        ``(ra, dec)`` of the batch center, in degrees.
    shape : tuple of int
        ``(height, width)`` of the frame, in pixels.
    pixscale : float
        Plate scale, in arcseconds per pixel.

    Returns
    -------
    float or None
        The cut from `_cnn_class_g_cut`, or None when the policy is off.
    """
    if not centroid_config.model_faint_positions:
        return None
    g_cut = _cnn_class_g_cut(
        coords, gaia_g, center, shape, pixscale, centroid_config.cnn_class_size
    )
    if np.isfinite(g_cut):
        logger.info(
            "CNN-class magnitude cut: G <= %.2f (%d targets)",
            g_cut,
            centroid_config.cnn_class_size,
        )
    else:
        logger.warning(
            "fewer than %d catalog targets within the frame-area circle: "
            "every star is CNN-class and keeps its CNN centroid",
            centroid_config.cnn_class_size,
        )
    return g_cut


def _check_edge_margin_fits_frame(edge_margin_px, metadata):
    """
    Reject an edge margin that leaves no usable area on the frame.

    Parameters
    ----------
    edge_margin_px : float
        The configured ``PhotometryConfig.edge_margin_px``.
    metadata : dict
        Frame metadata with integer ``height`` and ``width`` entries.

    Raises
    ------
    BatchPrepError
        If `edge_margin_px` is at or above half the smaller frame dimension.

    Notes
    -----
    Such a margin leaves no position at least that far from every edge, so
    every frame would raise `NoUsableStarsError` and finish as a skipped
    manifest row. Failing the batch up front names the cause once.
    """
    smaller_side = min(metadata["height"], metadata["width"])
    if edge_margin_px >= smaller_side / 2:
        msg = (
            f"edge_margin_px={edge_margin_px:g} leaves no usable area on a "
            f"{metadata['width']}x{metadata['height']} px frame; it must be "
            f"below half the smaller side ({smaller_side / 2:g} px)"
        )
        raise BatchPrepError(msg)


def _resolve_batch_instrument(config, header):
    """
    Resolve ``config.instrument`` for the batch, wrapping a detection failure.

    Parameters
    ----------
    config : PhotometryConfig
        The config whose ``instrument`` may need resolving.
    header : astropy.io.fits.Header
        The first frame's header to detect from; only consulted when
        ``config.instrument`` is None.

    Returns
    -------
    config : PhotometryConfig
        ``config`` unchanged if ``instrument`` was already set, otherwise a
        copy with the detected profile.
    auto_detected : bool
        True if ``instrument`` was resolved by detection.

    Raises
    ------
    BatchPrepError
        If ``config.instrument`` is None and ``header`` matches zero or more
        than one bundled/registered instrument profile.

    Notes
    -----
    `~bandaid.exceptions.InstrumentDetectionError` is a `FrameMetadataError`
    (recoverable per-frame) by declaration, but `prepare_batch` is the one
    caller that wants the fatal behavior: an unresolvable *first* frame leaves
    the whole batch with no detection/PSF settings to prepare with. So it is
    wrapped here instead of relying on inheritance.
    """
    try:
        return resolve_config_instrument(config, header)
    except InstrumentDetectionError as exc:
        raise BatchPrepError(str(exc)) from exc


def _check_solve_pool_floor(target_radecs, center, fov_rad, instrument, mag_limit):
    """
    Require enough target stars inside the first frame's solve pool.

    Parameters
    ----------
    target_radecs : numpy.ndarray
        Target RA/Dec in degrees, shape ``(N, 2)``.
    center : tuple of float
        ``(ra, dec)`` in degrees of the field center.
    fov_rad : float
        Field radius in degrees.
    instrument : InstrumentProfile
        Supplies ``solve_pool_radius_scale``.
    mag_limit : float
        The target magnitude limit, used in the error message.

    Raises
    ------
    BatchPrepError
        If fewer than the minimum number of stars lie in the pool.

    Notes
    -----
    Only the stars inside the pool are counted: the widened cone also holds
    edge stars that no frame's per-frame solve pool can use.
    """
    pool_radius = fov_rad * instrument.solve_pool_radius_scale
    n_in_pool = int(
        np.sum(_solve_pool_near(target_radecs, center[0], center[1], pool_radius))
    )
    if n_in_pool < N_GAIA_STARS_ALIGN_RETRY:
        msg = (
            f"Gaia returned only {n_in_pool} stars brighter than "
            f"{mag_limit} within the first frame's solve pool radius "
            f"({pool_radius:.3f} deg) for the field at {center}; need at least "
            f"{N_GAIA_STARS_ALIGN_RETRY} to solve a WCS"
        )
        raise BatchPrepError(msg)


def prepare_batch(
    first_file,
    *,
    cnn,
    config=None,
    build_l4=True,
    forced_targets=None,
    frame=None,
):
    """
    Compute the once-per-batch photometry inputs from the first frame.

    Runs a single detection pass on ``first_file`` (no WCS solve) to obtain the
    FWHM and image metadata, queries Gaia for the field (propagating the J2015.5
    DR2 positions to the frame's observation epoch via proper motions), drops
    contaminated sources, and builds the Bayer masks. The query cone is the
    field radius widened by the instrument's ``cone_radius_margin``; it is
    filtered to ``contaminant_mag_limit`` and carries a row limit with a
    truncation check (`~bandaid.catalog.query_field_catalog`).

    Parameters
    ----------
    first_file : str or Path
        Path to the first FITS frame in the batch. Used to derive the field
        pointing/FOV, plate scale, Bayer pattern, image shape, and FWHM. All
        frames in the batch are assumed to share these.
    cnn : object
        The centroiding model to carry through to every frame: any object with
        a ``centroid(cutouts) -> (N, 2)`` method, such as
        `~bandaid.ballet.Ballet`.
    config : PhotometryConfig or None, optional
        Photometry configuration carried on the returned `BatchPrep` and applied
        to every frame. Its ``instrument`` settings drive the first-frame FWHM
        detection and the contamination flagging here (evaluated at the largest
        ``apertures`` radius, with the first-frame FWHM padded by the
        instrument's ``contamination_seeing_margin``), and its
        ``source_selection`` settings supply the Gaia target/contaminant
        magnitude limits. If None (default), a default ``PhotometryConfig`` is
        used.
    build_l4 : bool, optional
        Whether each frame also gets a full-frame "L4" luminance channel,
        recorded on the returned `BatchPrep`. Default True.
    forced_targets : astropy.coordinates.SkyCoord or None, optional
        Extra sky positions to photometer that are absent from the Gaia
        catalog (e.g. a nova or supernova) -- appended to
        ``photometry_coords`` only, never to ``radecs`` (the WCS-solve
        reference catalog). They bypass contamination flagging (there is no
        Gaia magnitude to size that model against) but are still subject to
        every downstream quality cut. Any frame is accepted (e.g. FK5) and
        transformed to ICRS, matching ``photometry_coords``. A scalar
        `~astropy.coordinates.SkyCoord` (e.g. from
        `~astropy.coordinates.SkyCoord.from_name`) is accepted and treated as
        one target. None (default) adds nothing.
    frame : LoadedFrame or None, optional
        The already-loaded contents of ``first_file``. None (default) loads
        ``first_file`` here.

    Returns
    -------
    BatchPrep
        The reusable prep bundle for the batch.

    Raises
    ------
    BatchPrepError
        If too few stars are detected in ``first_file`` to measure an FWHM.
        Also raised if ``config.edge_margin_px`` is at or above half the
        smaller frame dimension, which would leave no usable area on any frame,
        and if fewer than the minimum number of target stars lie within
        the first frame's solve pool radius, if the Gaia query hit its row
        limit before reaching the target magnitude limit (a
        `~bandaid.exceptions.CatalogTruncationError`, raised unchanged), if
        the Gaia query fails (original chained as ``__cause__``), and if
        ``config.instrument``
        is None and the first frame's header does not resolve to exactly one
        instrument profile: `prepare_batch` is the one caller that treats that
        as fatal.
    FrameMetadataError
        If the first frame's metadata has no parseable observation time
        (``obs_time``, usually mapped from ``DATE-OBS``), which is needed to
        propagate the Gaia positions to the observation epoch.
    """
    if frame is None:
        frame = _load_frame(first_file)

    # A too-few-stars failure on the *first* frame is fatal for the whole batch
    # (no FWHM/pointing to prepare from), so translate the recoverable
    # per-frame TooFewStarsError into a fatal BatchPrepError.
    config = config or PhotometryConfig()
    # This is the first place a header is in hand, so resolve here and carry
    # the instrument forward on the config stored on the returned BatchPrep --
    # one resolution covers every frame in the batch (see also prepare_image,
    # the other resolution point, for direct prepare_image/calibration_sequence
    # callers). instrument_auto_detected gates check_frame_consistency's
    # batch-mixing guard, which fires only when the instrument was actually
    # detected, not explicitly chosen. See _resolve_batch_instrument for why
    # an unresolvable header here becomes a fatal BatchPrepError.
    config, instrument_auto_detected = _resolve_batch_instrument(config, frame.header)
    instrument = config.instrument
    try:
        # Pass the CNN so the FWHM (which sizes the photometry aperture) is measured
        # by re-centroiding detections, decoupling it from the detection opening.
        # detect_on_bayer_balanced and fwhm_n_stars must match the per-frame call
        # in process_one_image (which detects on bayer-balanced data by default),
        # so the batch-gating FWHM is measured in the same detection regime as
        # the photometry it protects.
        calibration = calibration_sequence(
            first_file,
            detect_on_bayer_balanced=True,
            cnn=cnn,
            profile=instrument,
            frame=frame,
        )
        metadata = calibration.metadata
        fwhm_pix = calibration.fwhm
    except TooFewStarsError as exc:
        msg = f"too few stars detected in {first_file!r} to prepare the batch"
        raise BatchPrepError(msg) from exc

    _check_edge_margin_fits_frame(config.edge_margin_px, metadata)

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
    obs_epoch = _parse_obs_time(obs_time, file=first_file)

    # Center the Gaia cone on the header pointing converted to ICRS.
    # fov_rad is a field *radius*; query_field_catalog takes a radius too and
    # widens it by the instrument's cone margin, so a frame that drifts a little
    # from the first frame's pointing is still covered by the catalog.
    # estimate_center_from_header knows a bad pointing but not which file it came
    # from; attach first_file so the failure names the frame, like the obs_time/
    # metadata_from_header labelling above.
    try:
        center = estimate_center_from_header(metadata, instrument)
    except FrameMetadataError as exc:
        exc.file = first_file
        raise
    logger.info("field center from header pointing: %s", center)
    # The target/contaminant magnitude limits bound the catalog query itself
    # (see query_field_catalog) and are used again below to split the result.
    # SourceSelectionConfig has already defaulted and finiteness-checked these.
    gaia_mag_limit = config.source_selection.gaia_mag_limit
    contaminant_mag_limit = config.source_selection.contaminant_mag_limit
    # A Gaia query failure (network/service error) is fatal for the whole batch;
    # surface it as a BatchPrepError instead of a raw astroquery/requests error.
    # A BatchPrepError raised by the query itself (row-limit truncation) already
    # carries a precise message and passes through unchanged.
    try:
        radecs, mags = query_field_catalog(
            center,
            metadata["fov_rad"],
            cone_margin=instrument.cone_radius_margin,
            obs_epoch=obs_epoch,
            gaia_mag_limit=gaia_mag_limit,
            contaminant_mag_limit=contaminant_mag_limit,
            row_limit=config.source_selection.gaia_row_limit,
        )
    except BatchPrepError:
        raise
    except Exception as exc:
        msg = f"could not query Gaia for the field at {center}"
        raise BatchPrepError(msg) from exc
    # Decouple the stars we *measure* (targets, cut at gaia_mag_limit) from the
    # stars that can *contaminate* them (a deeper list down to
    # contaminant_mag_limit). A real star fainter than the photometry limit still
    # spills into a brighter target's aperture, so flagging runs against the
    # deeper list -- but only targets are ever flagged/dropped.

    target = mags <= gaia_mag_limit
    contaminant = mags <= contaminant_mag_limit
    target_radecs = radecs[target]

    # Without enough reference stars no frame can solve a WCS, so fail the batch
    # now with a clear message rather than letting every frame fail later.
    _check_solve_pool_floor(
        target_radecs, center, metadata["fov_rad"], instrument, gaia_mag_limit
    )

    # The flag is computed once, from the first frame's FWHM, but applied to
    # every frame of the batch, so evaluate it at a pessimistically softened
    # seeing (FWHM * contamination_seeing_margin): pairs that would become
    # contaminated as seeing degrades during the night are dropped up front.
    # https://github.com/mwcraig/bandaid/issues/64
    flag_fwhm_arcsec = (
        fwhm_pix * metadata["pixscale"] * instrument.contamination_seeing_margin
    )
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
    # Forced targets (novae/supernovae -- absent from Gaia) go into
    # photometry_coords only, never radecs: they aren't astrometric
    # references for the WCS solve. Two deliberate properties follow, and
    # both cut in each direction:
    # (1) contamination flagging is bypassed for them -- it needs a Gaia
    # magnitude to size the separation model, and a forced target has none,
    # so a forced target sitting on top of a bright star is not flagged. The
    # reverse direction is equally invisible: a bright forced target (the
    # nova itself) near a Gaia comparison star does not size a flag radius
    # against that comp star either, so the comp star's contamination goes
    # unflagged too. Accepted -- a user forcing a target is expected to have
    # already weighed potential contamination.
    # (2) every downstream quality cut still applies unchanged, the edge
    # margin included; an off-frame forced target is silently dropped, by
    # design, not an error.
    if forced_targets is not None:
        # A scalar SkyCoord (e.g. SkyCoord.from_name(...) for a single nova)
        # has no len(); reshape to a 1-element array so it concatenates like
        # any other forced-target list.
        forced_targets = forced_targets.reshape(-1)
        # photometry_coords is plain ICRS (SkyCoord(..., unit="deg") above);
        # transform any other frame (e.g. FK5) before concatenating, else
        # np.concatenate raises a confusing frame-mismatch error. Rebuild a
        # bare SkyCoord from the transformed ra/dec rather than keeping
        # ``.icrs`` directly: astropy's SkyCoord remembers "extra" frame
        # attributes (e.g. FK5's equinox) across a transform, and
        # np.concatenate then rejects the mismatch against photometry_coords,
        # which carries none.
        icrs = forced_targets.icrs
        forced_targets = SkyCoord(icrs.ra, icrs.dec)
        # astropy.coordinates.concatenate is pending deprecation in favor of
        # np.concatenate, which SkyCoord supports directly. An empty
        # forced_targets is a no-op here (astropy's own concatenate
        # semantics); the CLI already rejects an empty forced-targets table
        # before it reaches this function.
        photometry_coords = np.concatenate([photometry_coords, forced_targets])
        logger.info(
            "appended %d forced target(s) to the photometry coords",
            len(forced_targets),
        )

    gaia_g = _with_forced_target_g(mags[target][~flagged_target], forced_targets)

    return BatchPrep(
        radecs=target_radecs,
        photometry_coords=photometry_coords,
        cnn=cnn,
        bayer_masks=generate_bayer_masks(
            (metadata["height"], metadata["width"]),
            metadata,
        ),
        center=center,
        fov_rad=metadata["fov_rad"],
        shape=(metadata["height"], metadata["width"]),
        config=config,
        forced_targets=forced_targets,
        instrument_auto_detected=instrument_auto_detected,
        build_l4=build_l4,
        gaia_g=gaia_g,
        g_cut=_batch_g_cut(
            config.centroid,
            photometry_coords,
            gaia_g,
            center,
            (metadata["height"], metadata["width"]),
            metadata["pixscale"],
        ),
    )


def _check_instrument_mixing(file, header, prep, batch_instrument):
    """
    Reject a frame whose header does not resolve to the batch's instrument.

    Parameters
    ----------
    file : str or Path
        The frame being checked (attached to any raised error).
    header : astropy.io.fits.Header
        The frame's FITS header.
    prep : BatchPrep
        The batch prep whose ``instrument_auto_detected`` gates whether a
        header matching no registered instrument is exempted (see Notes).
    batch_instrument : InstrumentProfile or None
        ``prep.config.instrument``, supplying the ``header_match`` rules
        checked against.

    Raises
    ------
    FrameError
        If the guard is enforced and ``header`` does not resolve to
        ``batch_instrument`` (subject to the no-match exemption described in
        Notes).

    Notes
    -----
    A later frame whose header does not resolve, through
    `~bandaid.instruments.detect_instrument`, to the same profile
    `prepare_batch` resolved is rejected -- e.g. a night with a different
    telescope's frames accidentally interleaved. Using `detect_instrument`
    (rather than only checking the batch instrument's own ``header_match``
    rules) means a header that is *ambiguous* across the registered profiles
    is rejected too, not just one that matches a definite other instrument.
    Enforced whenever ``batch_instrument.header_match`` is
    non-empty (a bare/custom profile with no rules carries no device-identity
    claim to check in the first place); a bare ``batch_instrument`` (or None)
    skips the guard entirely.

    An explicitly chosen instrument (``--instrument``/``--profile``/
    ``--config``, or ``config=PhotometryConfig(instrument=...)`` --
    ``prep.instrument_auto_detected`` is False) is exempted only from the
    no-match outcome: a header that resolves to *no* registered instrument at
    all (missing/malformed) is trusted, the deliberate override's intended
    escape hatch. A header that positively identifies a *different,
    registered* instrument is rejected even under an explicit selection --
    otherwise a scripted workflow that always passes a fixed
    ``--instrument`` would get zero protection against a second telescope's
    frames riding along on the same batch. An auto-detected batch (the
    default) is never exempted: any outcome other than a match to
    ``batch_instrument`` is rejected.

    `check_frame_consistency` calls this before the header is otherwise
    resolved, so a frame from a genuinely different instrument is rejected
    with this diagnostic message rather than the less informative
    `FrameMetadataError` that resolving its header through the *batch*
    instrument's ``header_map`` would likely raise first.
    """
    guard_active = batch_instrument is not None and batch_instrument.header_match
    if guard_active:
        try:
            detected = detect_instrument(header)
        except InstrumentDetectionError as exc:
            detected = None
            detection_error = exc
        else:
            detection_error = None

        no_match = detection_error is not None and not detection_error.matched
        if no_match and not prep.instrument_auto_detected:
            # The explicit-selection escape hatch: a header that resolves to
            # no registered instrument at all is trusted, not just one that
            # merely fails to name the batch instrument. An ambiguous header
            # (two or more matches) is still rejected.
            mismatched = False
        else:
            mismatched = detected is None or detected.name != batch_instrument.name
        if mismatched:
            # Distinguish "the header never carried any of the identifying
            # keywords at all" from "it carried at least one, with the wrong
            # value" (a present-but-different value more strongly suggests a
            # genuinely different instrument's frame, rather than an
            # incomplete header). header_match is OR semantics, so this must
            # check that *none* of the rules' keywords are present -- one
            # absent keyword does not mean that keyword was required, when
            # another rule's keyword is present but simply didn't match.
            missing = [rule.keyword for rule in batch_instrument.header_match]
            none_present = all(
                header.get(rule.keyword) is None
                for rule in batch_instrument.header_match
            )
            origin = "auto-detected batch" if prep.instrument_auto_detected else "batch"
            if none_present:
                msg = (
                    f"frame header is missing {missing}, required by the "
                    f"{origin} instrument {batch_instrument.name}'s "
                    "header_match rules"
                )
            else:
                msg = (
                    f"frame header does not match the {origin} "
                    f"instrument {batch_instrument.name}'s header_match "
                    "rules -- possibly a frame from a different instrument "
                    "mixed into this batch"
                )
            if detection_error is not None:
                msg = f"{msg} ({detection_error})"
                raise FrameError(msg, file=file) from detection_error
            raise FrameError(msg, file=file)


def check_frame_consistency(file, header, prep):
    """
    Reject a frame whose pointing, shape, or instrument disagrees with the batch prep.

    `prepare_batch` derives the field pointing, FOV, and image shape from the
    first frame and queries Gaia once for that field. A later frame that drifted
    off the field (a slew, a meridian flip, the wrong target) or has a different
    shape would be photometered against a catalog that no longer covers it,
    producing silently wrong results -- so reject it instead. A frame whose
    pointing drifted only partway off is kept but warned about.

    The batch-mixing guard (`_check_instrument_mixing`) runs first, needing
    only ``header`` and the batch instrument: a later frame whose header does
    not resolve to the profile `prepare_batch` auto-detected is rejected.

    The header is then resolved through the batch instrument's ``header_map``
    (``prep.config.instrument``), the same dialect that resolved the prep's
    ``center``/``shape`` from the first frame -- so an instrument whose pointing
    lives under different keywords than the Seestar's is compared consistently
    (issue #59). The frame's header pointing is converted to ICRS with
    :func:`estimate_center_from_header` before the comparison, the same way
    ``prep.center`` was.

    Parameters
    ----------
    file : str or Path
        The frame being checked (attached to any raised error).
    header : astropy.io.fits.Header
        The frame's FITS header.
    prep : BatchPrep
        The batch prep whose ``center``, ``fov_rad``, and ``shape`` the frame is
        checked against; whose ``config.instrument`` supplies the
        ``header_map`` dialect used to read the header and the
        ``header_match`` rules used for the batch-mixing guard; and whose
        ``instrument_auto_detected`` gates whether that guard is enforced.

    Returns
    -------
    float
        The offset in degrees of the frame's estimated center from
        ``prep.center``.

    Raises
    ------
    FrameError
        If the frame's shape, pointing, or instrument is inconsistent with
        the prep. A pointing rejection carries the offset in degrees as
        ``pointing_offset``.
    FrameMetadataError
        If the header cannot be resolved into the metadata needed to perform
        the checks.

    Notes
    -----
    The Gaia catalog covers the field radius plus the instrument's
    ``cone_radius_margin``. An offset up to the margin keeps the whole frame
    inside the catalog and is silent. An offset beyond the margin but within
    the field radius leaves the frame's edge partly uncovered: the frame is
    still processed, with a logged warning naming the offset and margin, since
    the per-frame solve pool uses only the central part of the field. Beyond
    the field radius the frame is rejected.
    """
    # Batch-mixing guard: needs only header and the batch instrument, so it
    # runs before the header is otherwise resolved; see
    # _check_instrument_mixing.
    batch_instrument = prep.config.instrument
    _check_instrument_mixing(file, header, prep, batch_instrument)

    try:
        metadata = metadata_from_header(header, profile=batch_instrument)
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
    # prep.center is the first frame's header pointing converted to ICRS, so
    # convert this frame's the same way before comparing.
    # The None case is caught by the explicit guard above; a present-but-non-
    # numeric pointing surfaces from estimate_center_from_header, which sits
    # outside the metadata_from_header try/except, so label it with the file here.
    try:
        frame_ra, frame_dec = estimate_center_from_header(
            metadata, prep.config.instrument
        )
    except FrameMetadataError as exc:
        exc.file = file
        raise
    frame_center = SkyCoord(frame_ra, frame_dec, unit="deg")
    center = SkyCoord(prep.center[0], prep.center[1], unit="deg")
    offset = float(center.separation(frame_center).deg)
    if offset > prep.fov_rad:
        msg = (
            f"frame pointing drifted: its field center is {offset:.3f} deg from "
            f"the batch center, beyond the {prep.fov_rad:.3f} deg field radius"
        )
        exc = FrameError(msg, file=file)
        # Carried to the QA manifest: this is the frame whose offset matters most.
        exc.pointing_offset = offset
        raise exc
    margin = batch_instrument.cone_radius_margin
    if offset > margin:
        logger.warning(
            "%s: field center is %.3f deg from the batch center, beyond the "
            "%.3f deg catalog margin; the frame is only partly covered by the "
            "catalog",
            file,
            offset,
            margin,
        )
    return offset


def _dropped_filters(by_filter):
    """
    Report the filters `~bandaid.writers.write_starlist_set` would drop.

    Mirrors the writer's own per-filter drop criterion (#109): a filter is
    "dropped" when its table has no `good_star_mask` survivors. Runs over
    every filter in ``by_filter`` (including L4), unlike the QA manifest's
    representative-channel diagnostics, so a drop is visible even when the
    representative channel itself is the one starved but a sibling filter
    carried the frame.

    Parameters
    ----------
    by_filter : dict of {str: astropy.table.Table}
        The ``process_one_image`` result for one frame.

    Returns
    -------
    str or None
        Semicolon-joined dropped filter names in ``by_filter`` order; ``""``
        when every filter was evaluable and none was dropped; ``None`` when
        some filter's table lacks the columns needed to evaluate it (so
        "nothing was dropped" can't be claimed).
    """
    dropped = []
    all_evaluable = True
    for filter_name, table in by_filter.items():
        table_meta = table.meta
        table_full_meta = table_meta.get("full_image_meta", {})
        table_cols = set(table.colnames)
        has_phot_cols = {"tot_count", "count_err", "x", "y"} <= table_cols
        has_bounds = {"width", "height"} <= set(table_full_meta)
        if not (has_phot_cols and has_bounds):
            # Can't evaluate this filter's survivors -- don't claim it was
            # dropped, but also don't claim the frame is clean.
            all_evaluable = False
            continue
        table_good = good_star_mask(
            table, table_full_meta, min_snr=table_meta.get("min_snr")
        )
        if not np.any(table_good):
            dropped.append(filter_name)
    if dropped:
        return ";".join(dropped)
    return "" if all_evaluable else None


def _qa_record_ok(file, by_filter, *, forced_targets=None, pointing_offset=None):
    """
    Build the QA manifest record for a frame that processed cleanly.

    Diagnostics are pulled defensively from a representative channel (L4 if
    present, else the first), so a frame missing a given column simply records a
    blank for it rather than failing the whole manifest.

    ``dropped_filters`` (see `_dropped_filters`) is evaluated across every
    filter, not just the representative channel, so the frame can still be
    ``status`` ``"ok"`` (with even ``n_good_stars`` at 0) when the
    representative channel itself is the one starved but a sibling filter
    carried the frame.

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
    forced_targets : astropy.coordinates.SkyCoord or None, optional
        The batch's forced targets (`BatchPrep.forced_targets`), used only to
        compute ``n_forced_measured``. None (default) records it as blank.
    pointing_offset : float or None, optional
        The frame's header-center offset from the batch center in degrees, as
        returned by `check_frame_consistency`. Recorded rounded to 4 decimals;
        None (default) records it as blank.

    Returns
    -------
    dict
        One manifest row keyed by `QA_MANIFEST_COLUMNS`.

    Notes
    -----
    ``n_forced_measured`` counts how many of ``forced_targets`` landed on a
    `good_star_mask`-passing row of the representative channel, matched by
    sky position within a generous 1 arcsec tolerance -- photometry runs at
    the catalog positions themselves, so a real match is essentially exact
    and 1 arcsec is only slack for float precision, not a real search radius.
    It is None (blank) when the batch was prepared without forced targets,
    and also None -- rather than a misleadingly precise 0 -- when the
    representative channel lacks the columns/bounds needed to evaluate
    `good_star_mask` in the first place.

    ``wcs_pixscale`` (solved plate scale, arcsec/pixel) and ``solve_offset_deg``
    (solved frame center to the frame's own header center, in degrees) are read
    from the table ``meta`` that `process_one_image` stamps, rounded to 4
    decimals; either is blank when absent. ``n_snr20`` counts the
    representative channel's `good_star_mask`-passing rows with ``snr >= 20``,
    and is blank under the same conditions as ``n_good_stars``.

    ``n_edge_dropped`` is the number of catalog stars the frame's edge margin
    removed before measurement, read from the table ``meta`` like
    ``wcs_pixscale``; blank when absent.

    ``g_cut`` and the ``n_cnn_class`` and ``plane_*`` columns summarise the
    centroid policy (see `~bandaid.photometry.centroid_with_catalog_model`) for the
    frame: the batch's magnitude cut, the number of CNN-class stars, whether the
    no-plane fallback fired, and the fitted plane's star counts, rms, centre
    offset and slopes in pixels. They are blank when the policy did not run
    and the ``plane_*`` columns are blank when the frame had no plane.
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
    n_snr20 = None
    has_phot_cols = {"tot_count", "count_err", "x", "y"} <= cols
    has_bounds = {"width", "height"} <= set(full_meta)
    good = None
    if has_phot_cols and has_bounds:
        # min_snr is read from table meta on purpose, not threaded in from the
        # run config: the writer applies the stamped value, and reading the
        # same stamp here keeps the QA count aligned with what was actually
        # written -- even for tables produced under a different config.
        good = good_star_mask(representative, full_meta, min_snr=meta.get("min_snr"))
        n_good_stars = int(np.sum(good))
        if "snr" in cols:
            snr = np.asarray(representative["snr"])
            n_snr20 = int(np.sum(good & (snr >= QA_SNR_THRESHOLD)))

    dropped_filters_value = _dropped_filters(by_filter)

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

    n_forced_measured = None
    if forced_targets is not None and good is not None and {"ra", "dec"} <= cols:
        good_rows = representative[good]
        if len(good_rows) == 0:
            n_forced_measured = 0
        else:
            good_coords = SkyCoord(good_rows["ra"], good_rows["dec"], unit="deg")
            _idx, sep2d, _ = forced_targets.match_to_catalog_sky(good_coords)
            n_forced_measured = int(np.sum(sep2d < 1 * u.arcsec))

    return {
        "file": str(file),
        "status": "ok",
        "n_detected": n_detected,
        "sky_median": sky_median,
        "fwhm": meta.get("fwhm"),
        "wcs_solved": True,
        "pointing_offset_deg": _round_or_none(pointing_offset),
        "wcs_pixscale": _round_or_none(meta.get("wcs_pixscale")),
        "solve_offset_deg": _round_or_none(meta.get("solve_offset_deg")),
        "n_good_stars": n_good_stars,
        "n_snr20": n_snr20,
        "dropped_filters": dropped_filters_value,
        "n_centroid_drift": n_centroid_drift,
        "n_drift_rejected": n_drift_rejected,
        "n_forced_measured": n_forced_measured,
        "n_edge_dropped": meta.get("n_edge_dropped"),
        **_centroid_model_record(meta.get("centroid_model")),
    }


def _centroid_model_record(summary):
    """
    Build the centroid-policy columns of a QA manifest row.

    Parameters
    ----------
    summary : dict or None
        The frame's plane summary from `centroid_with_catalog_model`, as stamped on its
        tables, or None when the policy did not run.

    Returns
    -------
    dict
        The policy columns of the QA manifest, floats rounded to 4 decimals
        and every value None (blank) when there is no summary.
    """
    summary = summary or {}
    record = {column: summary.get(column) for column in _CENTROID_MODEL_COLUMNS}
    return {
        column: _round_or_none(value) if isinstance(value, float) else value
        for column, value in record.items()
    }


def _round_or_none(value, ndigits=4):
    """
    Round a QA manifest value, passing None through.

    Parameters
    ----------
    value : float or None
        The value to record, or None when it was not measured.
    ndigits : int, optional
        Decimal places to keep. By default 4.

    Returns
    -------
    float or None
        The rounded value, or None.
    """
    return None if value is None else round(float(value), ndigits)


def _qa_record_failed(
    file, status, *, wcs_solved=None, pointing_offset=None, wcs_pixscale=None
):
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
    pointing_offset : float or None, optional
        The frame's header-center offset from the batch center in degrees, when
        it got past the pointing comparison; None (default) leaves it blank.
    wcs_pixscale : float or None, optional
        The plate scale in arcsec/pixel measured on a solve rejected for its
        scale; None (default) leaves it blank.

    Returns
    -------
    dict
        One manifest row keyed by `QA_MANIFEST_COLUMNS`, diagnostics blank.
    """
    record = dict.fromkeys(QA_MANIFEST_COLUMNS)
    record["file"] = str(file)
    record["status"] = status
    record["wcs_solved"] = wcs_solved
    record["pointing_offset_deg"] = _round_or_none(pointing_offset)
    record["wcs_pixscale"] = _round_or_none(wcs_pixscale)
    return record


def _record_frame_skip(file, exc, *, pointing_offset=None):
    """
    Log an expected per-frame failure and build its ``skipped`` manifest row.

    Shared by `process_batch`'s two `FrameError` handlers (the per-frame
    processing handler and the write-time handler) so a skip is logged and
    recorded identically wherever the frame-quality error surfaces.

    Parameters
    ----------
    file : str or pathlib.Path
        The input frame being skipped.
    exc : FrameError
        The frame-quality error. Some raisers (``build_photometry_table``,
        ``eloy_to_starlist``) do not know the path, so it is attached here
        when missing.
    pointing_offset : float or None, optional
        The frame's header-center offset in degrees if it got past the
        pointing comparison; None (default) leaves it blank. An offset carried
        by ``exc`` itself (a pointing-drift rejection) takes precedence.

    Returns
    -------
    dict
        The frame's ``skipped: <type>`` QA manifest row.
    """
    # exc.reason is the human-readable headline; exc_info=exc captures the
    # traceback and chained __cause__ (e.g. the original twirl traceback) so
    # no detail is lost.
    if exc.file is None:
        exc.file = file
    logger.warning("skipping %s: %s", file, exc.reason, exc_info=exc)
    return _qa_record_failed(
        file,
        f"skipped: {type(exc).__name__}",
        wcs_solved=False if isinstance(exc, WCSSolveError) else None,
        pointing_offset=getattr(exc, "pointing_offset", pointing_offset),
        wcs_pixscale=getattr(exc, "measured_scale", None),
    )


def _log_frame_stages(file, record):
    """
    Log a processed frame's measured star counts and background at DEBUG.

    Parameters
    ----------
    file : str or pathlib.Path
        The processed input frame.
    record : dict
        The frame's QA manifest row from `_qa_record_ok`.

    Notes
    -----
    The counts come from the manifest row so they are the very numbers written to
    the QA manifest: stars detected for the plate solve, catalog stars dropped by
    the edge margin, stars passing the quality cuts, stars with a signal-to-noise
    of at least `QA_SNR_THRESHOLD` and the median background count.
    """
    logger.debug(
        "%s: %s detected, %s catalog stars dropped at the edge, %s pass the "
        "quality cuts (%s with SNR >= %g), median background %s",
        file,
        record.get("n_detected"),
        record.get("n_edge_dropped"),
        record.get("n_good_stars"),
        record.get("n_snr20"),
        QA_SNR_THRESHOLD,
        _round_or_none(record.get("sky_median"), 1),
    )


def _log_frame_summary(file, record, started):
    """
    Log one INFO line giving the outcome of a frame.

    Parameters
    ----------
    file : str or pathlib.Path
        The input frame.
    record : dict
        The frame's QA manifest row.
    started : float
        `time.perf_counter` reading taken when the frame began.

    Notes
    -----
    A frame that was measured reports its star count and FWHM; one that was
    skipped reports the skip reason from its ``status`` and whether its WCS
    solved.
    """
    elapsed = time.perf_counter() - started
    if record["status"] == "ok":
        fwhm = record.get("fwhm")
        logger.info(
            "%s: ok, WCS solved, %s stars measured, FWHM %s px, %.1f s",
            file,
            record.get("n_good_stars"),
            "unknown" if fwhm is None else f"{fwhm:.2f}",
            elapsed,
        )
    else:
        logger.info(
            "%s: %s, WCS solved: %s, %.1f s",
            file,
            record["status"],
            record.get("wcs_solved"),
            elapsed,
        )


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
    first_frame=None,
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
        in-memory mode (``output_dir`` is None). A `FrameError` raised by the
        writer (e.g. the default writer's `NoUsableStarsError` when no star
        survives filtering in any filter) marks the frame skipped and the batch
        continues;
        any other writer exception is treated as a systemic write failure and
        propagates, aborting the run.
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
    first_frame : LoadedFrame or None, optional
        The already-loaded contents of the first file in ``files``. None
        (default) loads the first frame like any other.

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

    Notes
    -----
    Each frame is opened exactly once: the first frame reuses ``first_frame``
    when the caller provides it, and every other frame is loaded fresh here
    (issue #44).
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
    logger.info(
        "photometering %d frames against %d catalog stars; output to %s",
        len(files),
        len(prep.photometry_coords),
        "memory" if output_dir is None else output_dir,
    )
    for idx, file in enumerate(files, 1):
        started = time.perf_counter()
        # Per-frame progress. Invisible by default (the package logger has only a
        # NullHandler); `bandaid process --verbose` routes it to the terminal via
        # configure_logging, alongside the skip/error warnings logged below.
        logger.info("processing %d/%d: %s", idx, len(files), file)
        # Stays None for a frame that fails before its pointing offset is known.
        pointing_offset = None
        try:
            # Reuse the caller's already-opened first frame when given, else
            # load it now -- exactly one open per frame for the whole run
            # (issue #44).
            frame = (
                first_frame
                if idx == 1 and first_frame is not None
                else _load_frame(file)
            )
            pointing_offset = check_frame_consistency(file, frame.header, prep)
            by_filter = process_one_image(
                file,
                user_specific_metadata,
                prep.radecs,
                prep.cnn,
                prep.bayer_masks,
                config=prep.config,
                input_photometry_coords=prep.photometry_coords,
                input_gaia_g=prep.gaia_g,
                g_cut=prep.g_cut,
                frame=frame,
                build_l4=prep.build_l4,
            )
            # The raw pixel array is not needed past this point; drop the
            # reference now so it does not stay alive through the write step
            # (matching the pre-#44 per-frame peak-memory profile).
            del frame
        except FrameError as exc:
            # Expected per-frame failure: skip the frame and keep going.
            manifest_records.append(
                _record_frame_skip(file, exc, pointing_offset=pointing_offset)
            )
            _log_frame_summary(file, manifest_records[-1], started)
            continue
        except Exception as exc:
            # Unexpected error (a bug, not a bad frame): surface it by default;
            # only swallow-and-continue when the caller opted into robust mode.
            if fail_fast:
                raise
            logger.exception("unexpected error on %s", file)
            manifest_records.append(
                _qa_record_failed(
                    file,
                    f"error: {type(exc).__name__}",
                    pointing_offset=pointing_offset,
                )
            )
            continue
        else:
            # The frame processed cleanly. Writing its output is deliberately
            # outside the try above: a write failure (bad output_dir,
            # permissions, full disk) is systemic, not a property of this
            # frame, so it must abort the run rather than be skipped as a
            # "bad frame". A writer can still raise a frame-quality error at
            # write time, though -- the default writer raises
            # NoUsableStarsError when no star survives filtering in any
            # filter (#78) -- so
            # split on exception type: a FrameError is this frame's problem
            # and is skipped like any other, everything else propagates.
            manifest_records.append(
                _qa_record_ok(
                    file,
                    by_filter,
                    forced_targets=prep.forced_targets,
                    pointing_offset=pointing_offset,
                )
            )
            _log_frame_stages(file, manifest_records[-1])
            if output_dir is not None:
                try:
                    results[file] = write_frame(by_filter, output_paths[file])
                    logger.debug("%s: wrote %s", file, results[file])
                except FrameError as exc:
                    # Replace the provisional ok record appended above with
                    # the skip, so the manifest keeps one row per frame.
                    manifest_records[-1] = _record_frame_skip(
                        file, exc, pointing_offset=pointing_offset
                    )
            else:
                results[file] = by_filter
            _log_frame_summary(file, manifest_records[-1], started)

    # Persist the per-frame QA manifest next to the starlists. Only written in
    # write-to-disk mode (in-memory mode has no directory to write it to) and
    # only when the caller has not opted out.
    if write_manifest:
        _write_qa_manifest(Path(output_dir) / qa_manifest_name, manifest_records)
        logger.debug("wrote the QA manifest to %s", Path(output_dir) / qa_manifest_name)
    logger.info("finished: %d of %d frames photometered", len(results), len(files))
    return results


def photometer_frames(
    files,
    *,
    config=None,
    cnn=None,
    weights=None,
    user_specific_metadata=None,
    build_l4=True,
    output_dir=".",
    output_suffix=".star",
    write_frame=write_starlist_set,
    fail_fast=False,
    write_qa_manifest=True,
    forced_targets=None,
):
    """
    Expand a set of file arguments and measure per-frame photometry for each.

    The high-level convenience behind ``bandaid process``: it does the file-name
    expansion (`expand_frame_paths`), builds the `Ballet` centroider, and
    runs `prepare_batch` (seeded from the first frame) followed by
    `process_batch`.
    Driving the whole flow from Python is one call to this function; the CLI is a
    thin dressing over it.

    Parameters
    ----------
    files : collections.abc.Iterable of str
        Raw positional arguments -- directories, glob patterns, and/or file paths
        -- expanded by `expand_frame_paths`.
    config : PhotometryConfig or None, optional
        Configuration carried through the batch. None (default) uses a default
        `PhotometryConfig` whose instrument is auto-detected from the first
        frame's header (see `~bandaid.instruments.detect_instrument`);
        `prepare_batch` raises `~bandaid.exceptions.BatchPrepError` (chaining
        the underlying `~bandaid.exceptions.InstrumentDetectionError` as
        ``__cause__``) if the header does not resolve to exactly one
        registered profile.
    cnn : object or None, optional
        A pre-built centroider: any object with a ``centroid(cutouts) -> (N, 2)``
        method. None (default) builds a `~bandaid.ballet.Ballet` from
        ``weights``.
    weights : str or None, optional
        Path to Ballet weights used when ``cnn`` is None; None downloads the
        defaults from HuggingFace.
    user_specific_metadata : dict or None, optional
        Per-frame user metadata recorded with each output. None (default) is an
        empty dict.
    build_l4 : bool, optional
        Whether to also produce the full-frame L4 luminance channel.
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
    forced_targets : astropy.coordinates.SkyCoord or None, optional
        Extra sky positions to photometer that are absent from the Gaia
        catalog, forwarded to `prepare_batch`. Any frame is accepted (e.g.
        FK5) and transformed to ICRS; a scalar `~astropy.coordinates.SkyCoord`
        is accepted and treated as one target. None (default) adds nothing.

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
    BatchPrepError
        If `prepare_batch` cannot build the once-per-batch preparation from
        the first frame (e.g. its header does not resolve to exactly one
        registered instrument profile). Re-raised here with the first frame's
        path folded into the message, chaining the original as ``__cause__``,
        so the failure is actionable instead of a bare traceback.
    """
    frames = expand_frame_paths(files)
    if not frames:
        msg = "no FITS frames found in the given files/directories"
        raise ValueError(msg)

    config = config or PhotometryConfig()
    if cnn is None:
        cnn = Ballet(model_file=weights)

    # Open the first frame once and hand the load to both stages --
    # prepare_batch derives the prep from it and process_batch photometers it
    # -- so the whole run opens each frame exactly once (issue #44).
    first_frame = _load_frame(frames[0])
    try:
        prep = prepare_batch(
            frames[0],
            cnn=cnn,
            config=config,
            build_l4=build_l4,
            forced_targets=forced_targets,
            frame=first_frame,
        )
    except BatchPrepError as exc:
        # prepare_batch's own message has no idea which file it was given --
        # this is the caller that knows the first frame's path, so fold it in
        # instead of letting a bare BatchPrepError surface uncaught.
        msg = f"could not prepare the batch from the first frame ({frames[0]}): {exc}"
        raise BatchPrepError(msg) from exc
    results = process_batch(
        frames,
        prep,
        user_specific_metadata=user_specific_metadata or {},
        output_dir=output_dir,
        output_suffix=output_suffix,
        write_frame=write_frame,
        fail_fast=fail_fast,
        write_qa_manifest=write_qa_manifest,
        first_frame=first_frame,
    )
    return frames, results
