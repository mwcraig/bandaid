"""
Photometry pipeline for Smart Telescope images.

Provides the per-image processing steps used by the bandaid pipeline: source
detection and FWHM estimation, WCS alignment against a Gaia reference, CNN
centroiding, aperture photometry with annulus background subtraction and error
estimation, bright-neighbor contamination flagging, and conversion of the
resulting tables into a ``StarList``.
"""

import contextlib
import io
import logging
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, search_around_sky
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.table import Table
from astropy.time import Time
from dateutil import parser
from eloy import centroid, detection, photometry, psf, utils
from eloy.centroid import ballet_centroid
from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture
from scipy.ndimage import shift as ndshift
from st_pipeline.schema_definition import StarList
from twirl import compute_wcs

from .config import (
    ApertureConfig,
    DriftConfig,
    InstrumentProfile,
    PhotometryConfig,
)
from .exceptions import (
    FrameMetadataError,
    NoUsableStarsError,
    TooFewStarsError,
    WCSSolveError,
)
from .image2sl_qt import bayer_balance_image
from .instruments import load_instrument

logger = logging.getLogger(__name__)

__all__ = [
    "ImageData",
    "align",
    "annulus_sigma_clip_stats",
    "build_photometry_table",
    "calculate_l4_quantities",
    "calibration_sequence",
    "centroid_drift_flag",
    "centroid_stars",
    "eloy_to_starlist",
    "good_star_mask",
    "measure_photometry",
    "metadata_from_header",
    "min_separation_fwhm",
    "neighbor_contamination_flag",
    "neighbor_contamination_flag_sky",
    "prepare_image",
    "process_one_image",
]

# Star counts fed to twirl's asterism matcher in `align` to compute the per-image
# WCS. The image (detected) and Gaia (reference) lists are sliced *independently*
# so the matcher can be handed more references than detections. The two ranked
# lists order by different quantities (measured peak vs catalog G mag), so under
# clouds/colour/saturation they diverge; a deeper Gaia pool then raises the
# bright-end overlap and is the documented robustness lever for starved frames.
# DETECTION_OPENING = 3 roughly doubles the detections, which reshuffles the
# brightest N_IMAGE_STARS_ALIGN handed to twirl; on SS Leo that cost 6 frames a
# match at the 15-star pool that all came back at 20. Rather than pay the deeper
# pool on every frame, `align` solves with N_GAIA_STARS_ALIGN first and only
# widens to N_GAIA_STARS_ALIGN_RETRY when that fails -- the ~99% of frames that
# solve immediately keep the cheap search (cost grows ~ C(N, 4), so the retry is
# ~3x slower and is reserved for the few starved frames that need it).
# N_GAIA_STARS_ALIGN_RETRY is also the per-batch minimum reference count
# (scripts.prepare_batch): a field that cannot fill the retry pool can never use
# the deeper search, so the batch fails up front.
N_IMAGE_STARS_ALIGN = 15
N_GAIA_STARS_ALIGN = 15
N_GAIA_STARS_ALIGN_RETRY = 20
# Pixel tolerance handed to twirl's WCS solve.
WCS_MATCH_TOLERANCE = 1

# Minimum number of detected stars required before an image can be processed.
MIN_DETECTED_STARS = 3

# At least two stars are needed before any neighbor pair can exist.
MIN_STARS_FOR_PAIRS = 2

# The user-tunable knobs below now live in `bandaid.config`; the module-level
# names are kept (as the defaults pulled from a default-constructed config) so the
# leaf-function signatures and any existing callers continue to read them. The
# config object is the single source of truth for these values.
_DEFAULT_APERTURES = ApertureConfig()
_DEFAULT_DRIFT = DriftConfig()
_DEFAULT_INSTRUMENT = InstrumentProfile()

# Source-detection threshold (in units of the background sigma) passed to
# `detection.stars_detection`.
THRESH = _DEFAULT_INSTRUMENT.thresh
# Size of the morphological-opening kernel passed to `detection.stars_detection`.
# A source must hold a solid opening x opening above-threshold core to survive, so
# this -- not THRESH -- is what gates faint-star detection. eloy's default of 5
# starved real fields (~10 of ~23 real stars), failing the plate solve; 3 recovers
# them.
DETECTION_OPENING = _DEFAULT_INSTRUMENT.detection_opening
# Half-width (px) of the square cutout used to build the effective PSF for the
# FWHM fit; 25 reproduces the long-standing 50x50 calibration window.
_FWHM_CUTOUT_HALF = _DEFAULT_INSTRUMENT.fwhm_cutout_half
# Cap on how many of the brightest unsaturated detections feed the FWHM fit. The
# fit needs only a few well-exposed stars; the thousands of faint detections that
# bayer-balanced detection now yields just slow the CNN re-centroiding and inflate
# the FWHM (faint sources are mis-centroided, smearing the stacked PSF).
_FWHM_N_STARS = _DEFAULT_INSTRUMENT.fwhm_n_stars

# Relative radii and annulus are multiplied by each image's FWHM to determine the
# actual aperture sizes. Only one radius is needed for STWG, but it must be in an
# iterable.
RELATIVE_RADII = np.array(_DEFAULT_APERTURES.radii)
ANNULUS = _DEFAULT_APERTURES.annulus

# Bright-neighbor rejection. A star is flagged if any brighter neighbor's PSF
# wings would contribute more than CONTAMINATION_TOLERANCE of the target flux
# inside the 1*FWHM aperture, modeled as a Moffat profile of index MOFFAT_BETA.
CONTAMINATION_TOLERANCE = _DEFAULT_INSTRUMENT.contamination_tolerance
MOFFAT_BETA = _DEFAULT_INSTRUMENT.moffat_beta

# Centroid-drift check. A star is flagged if its measured centroid wandered
# more than `min(DRIFT_TOLERANCE_FWHM * fwhm, DRIFT_CAP_PIX)` pixels from its
# aligned/expected position. The FWHM-relative term lets the allowance scale with
# seeing, while the absolute pixel cap keeps a pathologically large FWHM from
# licensing an enormous shift. These defaults are empirical starting points and are
# meant to be tuned against real frames (override via the kwargs on
# `centroid_drift_flag` / `build_photometry_table`).
DRIFT_TOLERANCE_FWHM = _DEFAULT_DRIFT.drift_tolerance_fwhm  # max drift, in FWHM
DRIFT_CAP_PIX = _DEFAULT_DRIFT.drift_cap_pix  # absolute pixel cap on allowed drift


def min_separation_fwhm(delta_mag, tolerance=CONTAMINATION_TOLERANCE, beta=MOFFAT_BETA):
    """
    Minimum target/neighbor separation (in FWHM) for clean 1*FWHM aperture photometry.

    Models the neighbor as a Moffat PSF and approximates its intensity as
    constant across the target aperture (good for d >~ 2*FWHM). Returns the
    separation at which the neighbor's spillover into the aperture equals
    `tolerance` times the target flux.

    Parameters
    ----------
    delta_mag : float or array-like
        How many magnitudes brighter the neighbor is than the target, i.e.
        ``mag_target - mag_neighbor``. This is *positive* when the neighbor is
        brighter than the target, so a dim star with a bright neighbor has a
        positive ``delta_mag`` and requires a large separation; a faint neighbor
        gives a negative ``delta_mag`` and requires essentially none. For
        example, ``delta_mag=10`` (neighbor 10 mag brighter) needs ~11 FWHM of
        separation, while ``delta_mag=-10`` needs zero.
    tolerance : float, optional
        Maximum tolerated fractional flux contamination.
    beta : float, optional
        Moffat wing index. Smaller beta -> wider wings -> larger separation.

    Returns
    -------
    ndarray
        Required separation in units of FWHM. Zero where the neighbor is
        not bright enough to require any separation.
    """
    a_factor = 4.0 * (2.0 ** (1.0 / beta) - 1.0)  # (FWHM / alpha)^2 for Moffat
    prefactor = a_factor * (beta - 1.0)
    flux_ratio = 10.0 ** (0.4 * np.asarray(delta_mag))
    rhs = (prefactor * flux_ratio / tolerance) ** (1.0 / beta)
    return np.sqrt(np.maximum((rhs - 1.0) / a_factor, 0.0))


def _contamination_flag(
    separations,
    mags,
    min_sep_scale,
    tolerance=CONTAMINATION_TOLERANCE,
    beta=MOFFAT_BETA,
):
    """
    Flag contaminated stars from a precomputed pairwise separation matrix.

    Shared core of `neighbor_contamination_flag` (pixel separations) and
    `neighbor_contamination_flag_sky` (angular separations). Contamination is
    checked symmetrically: each pair (target, neighbor) is evaluated as
    fractional spillover into the *target's* aperture, so the same physical pair
    can flag the fainter star at a larger separation than the brighter one.

    Parameters
    ----------
    separations : ndarray, shape (N, N)
        Pairwise separations between stars, in any unit. ``separations[i, j]`` is
        the separation between target ``i`` and neighbor ``j``.
    mags : array-like, shape (N,)
        Per-star magnitude (zero-point arbitrary; only differences matter).
        Non-finite values are treated as "no contamination" for that star's
        role in the pair.
    min_sep_scale : float
        PSF FWHM expressed in the *same unit as* ``separations`` (pixels for the
        pixel front end, arcsec for the sky front end). Multiplies the unitless
        `min_separation_fwhm` result to put the required separation in that unit.
    tolerance : float, optional
        See `min_separation_fwhm`.
    beta : float, optional
        See `min_separation_fwhm`.

    Returns
    -------
    ndarray of bool, shape (N,)
        True where any neighbor sits inside this star's contamination radius.
    """
    mags = np.asarray(mags, dtype=float)
    n = len(mags)
    if n < MIN_STARS_FOR_PAIRS:
        return np.zeros(n, dtype=bool)

    # Build the pairwise (target i, neighbor j) quantities by broadcasting the
    # (N,) magnitude vector into (N, N) matrices: axis 0 ([:, None]) is the
    # "target" i and axis 1 ([None, :]) is the "neighbor" j. delta_mag[i, j] is
    # how much brighter neighbor j is than target i; valid[i, j] marks pairs where
    # both magnitudes are finite (with the diagonal i==j removed so a star is
    # never its own neighbor).
    delta_mag = mags[:, None] - mags[None, :]

    finite = np.isfinite(mags)
    valid = finite[:, None] & finite[None, :]
    np.fill_diagonal(valid, val=False)

    min_sep = np.zeros_like(separations, dtype=float)
    if valid.any():
        min_sep[valid] = (
            min_separation_fwhm(
                delta_mag[valid],
                tolerance=tolerance,
                beta=beta,
            )
            * min_sep_scale
        )

    too_close = valid & (separations < min_sep)
    return too_close.any(axis=1)


def neighbor_contamination_flag(
    coords,
    mags,
    fwhm,
    tolerance=CONTAMINATION_TOLERANCE,
    beta=MOFFAT_BETA,
):
    """
    Flag stars whose 1*FWHM aperture is contaminated by a too-close neighbor.

    Pixel-space front end over `_contamination_flag`: pairwise separations are
    Euclidean pixel distances and the FWHM is in pixels. Equal-brightness pairs
    are flagged inside ~2.18 FWHM.

    Parameters
    ----------
    coords : array-like, shape (N, 2)
        Pixel coordinates of the stars.
    mags : array-like, shape (N,)
        Per-star magnitude (zero-point arbitrary; only differences matter).
        Non-finite values are treated as "no contamination" for that star's
        role in the pair.
    fwhm : float
        PSF FWHM in pixels.
    tolerance : float, optional
        See `min_separation_fwhm`.
    beta : float, optional
        See `min_separation_fwhm`.

    Returns
    -------
    ndarray of bool, shape (N,)
        True where any neighbor sits inside this star's contamination radius.
    """
    coords = np.asarray(coords)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return _contamination_flag(
        dist,
        mags,
        fwhm,
        tolerance=tolerance,
        beta=beta,
    )


def neighbor_contamination_flag_sky(
    radecs,
    mags,
    fwhm_arcsec,
    tolerance=CONTAMINATION_TOLERANCE,
    beta=MOFFAT_BETA,
    target_mask=None,
):
    """
    Flag contaminated stars directly from sky coordinates and an angular FWHM.

    Sky-space front end of the contamination model: pairwise separations are
    great-circle angular separations (arcsec) and the FWHM is in arcsec, so no
    WCS or pixel projection is needed. Same contamination model and result
    convention as `neighbor_contamination_flag`, but instead of the dense
    ``N x N`` separation matrix it uses a neighbor search capped at the largest
    separation any pair could require, so it stays fast on full Gaia fields
    (~10,000 sources).

    Parameters
    ----------
    radecs : array-like, shape (N, 2)
        Sky coordinates of the stars as ``(ra, dec)`` pairs in degrees.
    mags : array-like, shape (N,)
        Per-star magnitude (zero-point arbitrary; only differences matter).
        Non-finite values are treated as "no contamination" for that star's
        role in the pair.
    fwhm_arcsec : float
        PSF FWHM in arcsec.
    tolerance : float, optional
        See `min_separation_fwhm`.
    beta : float, optional
        See `min_separation_fwhm`.
    target_mask : array-like of bool, shape (N,), optional
        If given, only stars where ``target_mask`` is True are evaluated as
        contamination *targets* (i.e. can be flagged); every star still acts as
        a potential contaminating *neighbor*. This makes the check asymmetric: a
        deeper catalog of faint stars can flag a brighter target without the
        faint stars themselves ever being flagged. If ``None`` (default), every
        star is a target, matching the symmetric all-pairs behavior.

    Returns
    -------
    ndarray of bool, shape (N,)
        True where any neighbor sits inside this star's contamination radius.
        Entries outside ``target_mask`` are always False.
    """
    radecs = np.asarray(radecs, dtype=float)
    mags = np.asarray(mags, dtype=float)
    flagged = np.zeros(len(radecs), dtype=bool)
    if len(radecs) < MIN_STARS_FOR_PAIRS:
        return flagged

    finite = np.isfinite(mags)
    if not finite.any():
        return flagged

    # Targets are the stars eligible to be flagged: every finite-magnitude star
    # by default, or only the finite stars selected by `target_mask`. Neighbors
    # (contaminators) are always drawn from the full finite set.
    targets = finite if target_mask is None else finite & np.asarray(target_mask, bool)
    if not targets.any():
        return flagged

    # The largest separation any target can require is the faintest target with
    # the brightest neighbor; `min_separation_fwhm` is monotonic in delta_mag,
    # so capping the neighbor search there finds every pair the dense N x N
    # separation matrix would flag at a fraction of the time and memory.
    max_delta_mag = np.max(mags[targets]) - np.min(mags[finite])
    max_sep_arcsec = (
        min_separation_fwhm(max_delta_mag, tolerance=tolerance, beta=beta) * fwhm_arcsec
    )
    if max_sep_arcsec <= 0:
        return flagged

    coords = SkyCoord(radecs[:, 0], radecs[:, 1], unit="deg")
    # Seed the search only from the targets; search_around_sky indexes its first
    # catalog (the target subset), so map those back to full-array rows. Each
    # match is one (target, neighbor) pair, both in full-array coordinates.
    target_rows = np.nonzero(targets)[0]
    subset_target, pair_neighbor, sep2d, _ = search_around_sky(
        coords[target_rows],
        coords,
        max_sep_arcsec * u.arcsec,
    )
    pair_target = target_rows[subset_target]

    # Same pair convention as `_contamination_flag`: a star is never its own
    # neighbor (but distinct stars at zero separation are), and a pair only
    # counts when both magnitudes are finite (targets are finite by construction).
    keep = (pair_target != pair_neighbor) & finite[pair_neighbor]
    pair_target = pair_target[keep]
    pair_neighbor = pair_neighbor[keep]
    sep_arcsec = sep2d.arcsec[keep]

    min_sep = (
        min_separation_fwhm(
            mags[pair_target] - mags[pair_neighbor],
            tolerance=tolerance,
            beta=beta,
        )
        * fwhm_arcsec
    )
    flagged[pair_target[sep_arcsec < min_sep]] = True
    return flagged


def centroid_drift_flag(
    centroid_coords,
    aligned_coords,
    fwhm,
    tolerance=DRIFT_TOLERANCE_FWHM,
    cap=DRIFT_CAP_PIX,
):
    """
    Flag stars whose centroid drifted too far from its aligned position.

    Parameters
    ----------
    centroid_coords : array-like, shape (N, 2)
        Measured centroid pixel coordinates.
    aligned_coords : array-like, shape (N, 2)
        Aligned/expected pixel coordinates the centroids are compared against.
    fwhm : float
        PSF FWHM in pixels.
    tolerance : float, optional
        Maximum allowed drift in units of FWHM. Defaults to
        `DRIFT_TOLERANCE_FWHM`.
    cap : float, optional
        Absolute pixel cap on the allowed drift, applied as
        ``min(tolerance * fwhm, cap)``. Defaults to `DRIFT_CAP_PIX`.

    Returns
    -------
    ndarray of bool, shape (N,)
        True where the centroid drifted past the allowed threshold or is
        non-finite.

    Notes
    -----
    The drift is the pixel-space displacement between the measured centroid and
    the aligned/expected position; both inputs are already in pixel space, so no
    WCS round-trip is needed and the metric isolates centroid wander from WCS
    quality. A star is flagged when its drift exceeds
    ``min(tolerance * fwhm, cap)`` pixels. A large drift usually means the WCS is
    wrong, the star was too faint to centroid, or it was blocked by cloud or an
    obstruction. Non-finite centroids (e.g. failed faint-star centroids) are
    treated as drifted.
    """
    drift = np.linalg.norm(
        np.asarray(centroid_coords, dtype=float)
        - np.asarray(aligned_coords, dtype=float),
        axis=-1,
    )
    max_allowed = min(tolerance * fwhm, cap)
    # `nan > max_allowed` is False, so flag non-finite drift explicitly.
    return (drift > max_allowed) | ~np.isfinite(drift)


def _registered_epsf(data, coords_xy, max_adu, half=_FWHM_CUTOUT_HALF):
    """
    Median-stack sub-pixel-registered, peak-normalized cutouts into an effective PSF.

    Each source is cut out, shifted so its center (``coords_xy``) lands on the cutout
    center, peak-normalized, and median-combined. The sub-pixel registration is what
    keeps the stack from broadening when the input centers carry centroid error.

    Parameters
    ----------
    data : numpy.ndarray
        2D image.
    coords_xy : numpy.ndarray, shape (N, 2)
        Source centers as ``(x, y)`` in pixels (may be sub-pixel).
    max_adu : float
        Saturation ceiling; cutouts whose peak reaches it are dropped.
    half : int, optional
        Half-width of the (square) stacked cutout in pixels.

    Returns
    -------
    numpy.ndarray or None
        The ``(2*half, 2*half)`` effective PSF, or None if no usable cutout survived
        (every source off-edge, saturated, or non-finite).
    """
    # Cut a slightly larger box so the sub-pixel shift never pulls in edge fill.
    box = half + 3
    stack = []
    for cx, cy in np.asarray(coords_xy, dtype=float):
        ix, iy = round(cx), round(cy)
        if (
            iy - box < 0
            or ix - box < 0
            or iy + box > data.shape[0]
            or ix + box > data.shape[1]
        ):
            continue
        sub = data[iy - box : iy + box, ix - box : ix + box].astype(float)
        if not np.isfinite(sub).all():
            continue
        peak = sub.max()
        if peak >= max_adu or peak <= 0:
            continue
        # Shift the (off-center) source onto the cutout center before stacking.
        sub = ndshift(sub, shift=(iy - cy, ix - cx), order=3, mode="nearest")
        inner = sub[(box - half) : (box + half), (box - half) : (box + half)]
        stack.append(inner / np.nanmax(inner))
    if not stack:
        return None
    return np.nanmedian(stack, 0)


def _brightest_unsaturated(data, coords_xy, max_adu, n):
    """
    Keep the ``n`` highest-peak unsaturated detections for the FWHM fit.

    The peak is read at each detection's (rounded, in-bounds) centroid pixel.
    Sources whose peak reaches saturation (``>= max_adu``) or is non-positive are
    dropped -- a cheap single-pixel brightness pre-filter, *not* the full
    per-cutout test :func:`_registered_epsf` applies (which reads the peak from
    the cutout max and also excludes off-edge and non-finite cutouts; that test
    still runs downstream on the survivors). The surviving coords are returned
    highest-peak first, truncated to ``n`` (all of them when
    ``len(coords_xy) <= n``).

    Parameters
    ----------
    data : numpy.ndarray
        2D image.
    coords_xy : numpy.ndarray, shape (N, 2)
        Detected source centers as ``(x, y)`` in pixels.
    max_adu : float
        Saturation ceiling; detections at or above it are excluded.
    n : int
        Maximum number of detections to retain.

    Returns
    -------
    numpy.ndarray, shape (M, 2)
        The retained ``(x, y)`` coords, ``M <= n``, ordered brightest first.
    """
    # reshape(-1, 2) keeps empty (-> (0, 2)) and single ((2,) -> (1, 2)) inputs
    # from collapsing to a 1-D array that the column indexing below would reject.
    coords_xy = np.asarray(coords_xy, dtype=float).reshape(-1, 2)
    ix = np.clip(np.round(coords_xy[:, 0]).astype(int), 0, data.shape[1] - 1)
    iy = np.clip(np.round(coords_xy[:, 1]).astype(int), 0, data.shape[0] - 1)
    peaks = data[iy, ix]
    usable = np.flatnonzero((peaks > 0) & (peaks < max_adu))
    # Sort the usable detections by descending peak and keep the top n.
    brightest = usable[np.argsort(peaks[usable])[::-1][:n]]
    return coords_xy[brightest]


def _fwhm_from_coords(
    data,
    coords_xy,
    max_adu,
    *,
    cnn=None,
    fwhm_cutout_half=_FWHM_CUTOUT_HALF,
    n_stars=_FWHM_N_STARS,
):
    """
    Fit the image FWHM (px) from the effective PSF of the given sources.

    With ``cnn=None`` this reproduces the legacy behaviour: peak-normalized 50x50
    cutouts taken at ``coords_xy`` (integer-pixel) are median-stacked and fit with a
    Gaussian. The stack inherits any error in ``coords_xy`` as extra width, so a
    detection opening that yields jittery centroids inflates the FWHM -- and hence the
    photometry aperture, which is sized in FWHM.

    With a ``cnn`` (an ``eloy.centroid.Ballet``), the centers are first refined with
    `ballet_centroid` and the cutouts are sub-pixel-registered to them before
    stacking, so the measured FWHM tracks the true PSF regardless of the opening.

    Parameters
    ----------
    data : numpy.ndarray
        2D image.
    coords_xy : numpy.ndarray, shape (N, 2)
        Detected source centers as ``(x, y)`` in pixels.
    max_adu : float
        Saturation ceiling; saturated cutouts are excluded from the fit.
    cnn : eloy.centroid.Ballet or None, optional
        If given, re-centroid sources with the CNN and sub-pixel-register the cutouts
        before fitting. Default None (legacy integer-cutout stack).
    fwhm_cutout_half : int, optional
        Half-width (px) of the square cutout used to build the effective PSF.
        Defaults to ``_FWHM_CUTOUT_HALF`` (a 50x50 window).
    n_stars : int, optional
        Cap on how many of the brightest unsaturated detections feed the fit. The
        cut is applied *before* the CNN re-centroiding so the cost scales with the
        cap, not the (now thousands of) detections, and so faint mis-centroided
        sources cannot smear the stacked PSF. Defaults to ``_FWHM_N_STARS``.

    Returns
    -------
    float or None
        FWHM in pixels, or None if no source was usable for the fit.
    """
    coords_xy = _brightest_unsaturated(data, coords_xy, max_adu, n_stars)
    if cnn is not None:
        coords_xy = ballet_centroid(data, np.asarray(coords_xy, dtype=float), cnn)
        epsf = _registered_epsf(data, coords_xy, max_adu, half=fwhm_cutout_half)
    else:
        cutout_size = (2 * fwhm_cutout_half, 2 * fwhm_cutout_half)
        cutouts = utils.cutout(data, coords_xy, cutout_size)
        cutouts = np.array([c for c in cutouts if np.max(c) < max_adu])
        epsf = (
            None
            if len(cutouts) == 0
            else np.nanmedian(cutouts / np.nanmax(cutouts, (1, 2))[:, None, None], 0)
        )
    if epsf is None:
        return None
    params = psf.fit_gaussian(epsf)
    return psf.gaussian_sigma_to_fwhm * np.mean([params["sigma_x"], params["sigma_y"]])


def calibration_sequence(
    file,
    threshold=1,
    opening=DETECTION_OPENING,
    *,
    detect_on_bayer_balanced=False,
    cnn=None,
    fwhm_cutout_half=_FWHM_CUTOUT_HALF,
    fwhm_n_stars=_FWHM_N_STARS,
    profile=None,
) -> tuple:
    """
    Find sources and compute FWHM for an image.

    Parameters
    ----------
    file : str
        Path to the FITS file.
    threshold : float, optional
        Detection threshold for star finding, by default 1
    opening : int, optional
        Size of the morphological-opening kernel passed to
        `detection.stars_detection`; gates faint-star detection. By default
        ``DETECTION_OPENING``.
    detect_on_bayer_balanced : bool, optional
        When True, run source detection and the FWHM fit on a Bayer-balanced
        *copy* of the data. The returned ``calibrated_data`` is always the
        original unbalanced array, so downstream photometry still measures real
        counts. Default False preserves detection on the raw data.
    cnn : eloy.centroid.Ballet or None, optional
        If given, the FWHM is measured by re-centroiding detections with the CNN and
        sub-pixel-registering their cutouts before the PSF fit, so the FWHM (and the
        FWHM-sized photometry aperture) is independent of the detection ``opening``.
        Default None preserves the legacy integer-cutout FWHM. See
        `_fwhm_from_coords`.
    fwhm_cutout_half : int, optional
        Half-width (px) of the square cutout used to build the effective PSF for
        the FWHM fit. By default ``_FWHM_CUTOUT_HALF``.
    fwhm_n_stars : int, optional
        Cap on how many of the brightest unsaturated detections feed the FWHM
        fit; forwarded to `_fwhm_from_coords` as its ``n_stars``. By default
        ``_FWHM_N_STARS``.
    profile : InstrumentProfile or None, optional
        The instrument whose ``header_map`` resolves the frame metadata, passed
        through to `metadata_from_header`. Defaults to the bundled Seestar50
        profile.

    Returns
    -------
    tuple
        A tuple containing the calibrated data, metadata, region coordinates,
        FWHM, and regions.

    Raises
    ------
    TooFewStarsError
        If fewer than ``MIN_DETECTED_STARS`` are detected, or every detected
        source is saturated, so no usable PSF can be fit.
    FrameMetadataError
        If the header is missing a required keyword (propagated from
        `metadata_from_header`, with the source file attached).

    """
    data = fits.getdata(file)
    header = fits.getheader(file)

    try:
        metadata = metadata_from_header(header, profile=profile)
    except FrameMetadataError as exc:
        # metadata_from_header has only the header, not the path; label it here.
        exc.file = file
        raise
    max_adu = metadata["largest_usable_adu_value"]

    # Multiplying by 1 should force conversion from int to float data
    calibrated_data = 1.0 * data

    # Detection and the FWHM fit run on the Bayer-balanced data when requested
    # (the channel imbalance otherwise biases both), but photometry must see the
    # real counts, so balance a *copy* and keep calibrated_data untouched.
    if detect_on_bayer_balanced:
        detection_image = calibrated_data.copy()
        bayer_balance_image(detection_image)
    else:
        detection_image = calibrated_data

    regions = detection.stars_detection(
        detection_image, threshold=threshold, opening=opening
    )

    # in case we detect fewer than the minimum number of stars
    if len(regions) < MIN_DETECTED_STARS:
        msg = f"only {len(regions)} stars detected (need at least {MIN_DETECTED_STARS})"
        raise TooFewStarsError(msg, file=file)

    region_coords_xy = np.array([(r.centroid[1], r.centroid[0]) for r in regions])

    # Saturated sources are excluded inside the helper; if none survive there is
    # nothing to fit a PSF to.
    fwhm = _fwhm_from_coords(
        detection_image,
        region_coords_xy,
        max_adu,
        cnn=cnn,
        fwhm_cutout_half=fwhm_cutout_half,
        n_stars=fwhm_n_stars,
    )
    if fwhm is None:
        msg = "all detected sources are saturated"
        raise TooFewStarsError(msg, file=file)

    return calibrated_data, metadata, region_coords_xy, fwhm, regions


def _airmass_from_header(header):
    """
    Return the frame airmass, deriving it when the header lacks AIRMASS.

    Seestar ``.fit`` headers carry no ``AIRMASS`` keyword but do record the
    pointing (``RA``/``DEC``), site (``SITELAT``/``SITELONG``, optionally
    ``SITEELEV``) and time (``DATE-OBS``) -- enough to compute the field-center
    airmass via an AltAz transform. The header value is preferred when present.

    The relative optical airmass is computed with the Kasten & Young (1989)
    formula, which stays accurate at the high airmass (up to ~4) these frames
    reach -- where the plane-parallel ``sec(z)`` overestimates by several percent.

    Parameters
    ----------
    header : astropy.io.fits.Header or dict
        FITS header for the frame.

    Returns
    -------
    float
        The header ``AIRMASS`` value, or the derived Kasten-Young airmass.

    Raises
    ------
    FrameMetadataError
        If the header has neither a parseable ``AIRMASS`` nor the
        pointing/site/time keywords needed to derive one. Airmass is a standard
        input for extinction work, so an undiagnosable frame is skipped rather
        than carried with a NaN.
    """
    # Only the header reads and the date-string parse can fail *because of the
    # header*; keep exactly those inside the try so a well-understood missing or
    # malformed keyword maps to a skipped frame, while a bug in the astropy
    # object construction or the airmass formula below surfaces as itself.
    try:
        airmass = header.get("AIRMASS")
        if airmass is not None:
            return float(airmass)
        site_lat = float(header["SITELAT"])
        site_lon = float(header["SITELONG"])
        site_elev = float(header.get("SITEELEV", 0.0))
        ra = float(header["RA"])
        dec = float(header["DEC"])
        obs_datetime = parser.parse(header["DATE-OBS"])
    except (KeyError, ValueError, TypeError) as exc:
        msg = (
            "cannot determine AIRMASS: header has no parseable AIRMASS and is "
            "missing/unparseable RA/DEC/SITELAT/SITELONG/DATE-OBS"
        )
        raise FrameMetadataError(msg) from exc

    location = EarthLocation(
        lat=site_lat * u.deg, lon=site_lon * u.deg, height=site_elev * u.m
    )
    pointing = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    altaz = pointing.transform_to(AltAz(obstime=Time(obs_datetime), location=location))

    # Kasten & Young (1989) relative optical airmass:
    #   X = 1 / (sin(h) + 0.50572 * (h + 6.07995)**-1.6364),  h = apparent
    #   altitude in degrees. Accurate to high airmass where the plane-parallel
    #   sec(z) breaks down (sec(z) is ~3% high at airmass ~5.7).
    # Kasten & Young 1989, Applied Optics 28(22), 4735; doi:10.1364/AO.28.004735
    alt_deg = altaz.alt.to_value(u.deg)
    airmass = 1.0 / (
        np.sin(np.deg2rad(alt_deg)) + 0.50572 * (alt_deg + 6.07995) ** -1.6364
    )
    return float(airmass)


def _resolve_template_value(key, value, header, defaults):
    """
    Resolve one ``header_map`` directive against a FITS header.

    Parameters
    ----------
    key : str
        The metadata key being resolved (used in error messages).
    value : object
        The directive: ``"@KEY"`` header lookup, ``"!KEY func ..."`` function
        call, or a plain literal passed through untouched.
    header : astropy.io.fits.Header or dict
        FITS header to look up values in.
    defaults : dict
        Fallback values collected from the template's ``#key`` entries.

    Returns
    -------
    object
        The resolved metadata value.

    Raises
    ------
    FrameMetadataError
        If a ``!`` directive references a header keyword that is missing or has
        an unexpected shape.
    """
    if isinstance(value, str) and value.startswith("@"):
        return header.get(value[1:], defaults.get(key))
    if isinstance(value, str) and value.startswith("!"):
        parts = value[1:].split()
        header_key = parts[0]
        index = int(parts[2])
        # A missing keyword (KeyError) or an unexpected value shape
        # (AttributeError/IndexError) is a per-frame metadata problem.
        try:
            return header[header_key].split()[index]
        except (KeyError, AttributeError, IndexError) as exc:
            msg = f"could not read header keyword {header_key!r} for {key!r}"
            raise FrameMetadataError(msg) from exc
    return value


def metadata_from_header(header, *, profile=None):
    """
    Build a metadata dictionary from an instrument's header dialect and a header.

    Parameters
    ----------
    header : astropy.io.fits.Header or dict
        FITS header to look up values in.
    profile : InstrumentProfile or None, optional
        The instrument whose ``header_map`` resolves the header. Defaults to the
        bundled Seestar50 profile, preserving the historical behaviour for
        callers that do not pass one.

    Returns
    -------
    dict
        Metadata dictionary with header lookups resolved.

    Raises
    ------
    FrameMetadataError
        If a required header keyword is missing or cannot be parsed, or if the
        system gain (``egain``) is absent with no template default.
    """
    if profile is None:
        profile = load_instrument("Seestar50")
    template = profile.header_map

    # Collect fallback values from "#key" entries
    defaults = {
        key[1:]: value for key, value in template.items() if key.startswith("#")
    }

    # Comment/internal "_"/"#" keys are skipped; everything else is a directive.
    metadata = {
        key: _resolve_template_value(key, value, header, defaults)
        for key, value in template.items()
        if not key.startswith(("_", "#"))
    }

    try:
        metadata["width"] = header["NAXIS1"]
        metadata["height"] = header["NAXIS2"]
    except KeyError as exc:
        msg = f"missing required header keyword {exc.args[0]!r}"
        raise FrameMetadataError(msg) from exc

    # egain feeds the photometry noise model; a None value would silently poison
    # every SNR downstream, so fail the frame here with a clear message.
    if metadata.get("egain") is None:
        msg = "system gain (egain) is missing from the header and has no default"
        raise FrameMetadataError(msg)

    return metadata


def eloy_to_starlist(eloy_table, metadata):
    """
    Convert a single-image photometry table from eloy to a StarList.

    Parameters
    ----------
    eloy_table : astropy.table.Table
        Table containing photometry data from eloy for one image.
        Each row is one star. Must include columns matching StarItem fields:
        tot_count, count_err, bkgd_count, peak_count, x, y, ra, dec.
        Table meta must include fwhm.
    metadata : dict
        Dictionary of StarList metadata fields not available in the eloy table.
        Required keys: site_lat, site_lon, site_elev, observer, filter,
        block_filter, exposure, tel_manufac, tel_model, tel_firmware,
        adc_depth, largest_usable_adu_value, egain, width, height, refframe.

    Returns
    -------
    StarList

    Raises
    ------
    NoUsableStarsError
        If no rows survive filtering, so the frame would produce an empty
        StarList. The source file is not known here; the caller attaches it.
    """
    good = good_star_mask(eloy_table, metadata)
    if not np.any(good):
        msg = "no stars survived photometry filtering"
        raise NoUsableStarsError(msg)
    return StarList.from_table(eloy_table[good], metadata=metadata)


def good_star_mask(eloy_table, metadata):
    """
    Boolean mask of rows that survive photometry filtering.

    A star is "good" when it has a finite, positive net count and error, lies
    in-bounds, and is not flagged as contaminated. This is the same predicate
    `eloy_to_starlist` uses to decide which rows reach the output StarList, so
    QA tooling can count good stars without rebuilding the StarList.

    Parameters
    ----------
    eloy_table : astropy.table.Table
        Per-image photometry table; one row per star. Must include
        ``tot_count``, ``count_err``, ``x``, and ``y`` (and optionally
        ``contaminated``).
    metadata : dict
        Must include the frame ``width`` and ``height`` used for the in-bounds
        test.

    Returns
    -------
    numpy.ndarray
        Boolean array, ``True`` for rows that pass the filter.
    """
    # REPLACE THIS WITH FILTERING FROM IMAGE2SL_QT
    good = ~np.isnan(eloy_table["tot_count"])
    good &= eloy_table["tot_count"] > 0
    good &= np.isfinite(eloy_table["count_err"])
    good &= eloy_table["count_err"] > 0
    good &= (
        (eloy_table["x"] > 0)
        & (eloy_table["x"] < metadata["width"])
        & (eloy_table["y"] > 0)
        & (eloy_table["y"] < metadata["height"])
    )
    if "contaminated" in eloy_table.colnames:
        good &= ~eloy_table["contaminated"]
    return np.asarray(good)


@dataclass
class ImageData:
    """Per-image detection, alignment, and centroiding results."""

    calibrated_data: np.ndarray
    coords: np.ndarray
    fwhm: float
    centroid_coords: np.ndarray
    aligned_coords: np.ndarray
    wcs: object
    header: fits.Header
    input_photometry_coords: object = None
    metadata: dict = None


def align(coords, radecs=None, *, photometry_coords=None, wcs=None):
    """
    Compute per-image WCS and align reference coordinates into pixel space.

    Parameters
    ----------
    coords : numpy.ndarray
        Detected star **pixel** coordinates (x, y) in this image. Used for WCS
        alignment and, if photometry_coords is None, returned as the aligned
        coordinates as well.
    radecs : numpy.ndarray or None, optional
        Gaia reference sky coordinates (RA/Dec), paired against `coords` by
        twirl's asterism matcher to solve the WCS. Required only when `wcs` is
        None; ignored when a precomputed `wcs` is supplied. By default None.
    photometry_coords : `astropy.coordinates.SkyCoord` or None, optional
        If provided, these sky coordinates are projected through the WCS to
        produce the aligned pixel coordinates. By default None (aligned
        coordinates are just `coords`).
    wcs : astropy.wcs.WCS or None, optional
        If provided, this WCS is used instead of computing a new one from
        `coords` and `radecs`. By default None (WCS is computed from
        `coords` and `radecs`).

    Returns
    -------
    aligned_coords : numpy.ndarray
        Aligned star coordinates in pixel space.
    this_wcs : astropy.wcs.WCS
        World Coordinate System for the image.

    Raises
    ------
    WCSSolveError
        If `wcs` is None and twirl cannot solve a WCS from `coords` and
        `radecs` -- either it returns None (too few stars overlap to satisfy its
        min_match threshold) or it raises while matching. The raiser does not
        know the source file; callers attach it to the error before re-raising.
    """
    if wcs is None:
        # Feed the brightest N_IMAGE_STARS_ALIGN detections and the brightest
        # Gaia reference RA/Decs to twirl's asterism (quad) matcher. twirl matches
        # by geometric shape, so the two lists need NOT be in the same order or
        # even the same length -- there is no per-index correspondence. The slices
        # are independent so the matcher can be handed more references than
        # detections (see the constants above).
        #
        # Try the cheap shallow Gaia pool first and only widen to the deeper retry
        # pool if it fails: the larger pool's asterism search costs ~ C(N, 4), so
        # the ~99% of frames that solve immediately should not pay for it. A pool
        # that returns None or raises a too-few-stars error is a failure to retry;
        # the last failure (exception or None) decides the error if every pool
        # fails.
        # twirl's asterism matcher prints timing/diagnostic lines straight to
        # stdout (e.g. "Match took ... us"); redirect them to /dev/null so the
        # pipeline and notebooks stay quiet.
        this_wcs = None
        last_exc = None
        for n_gaia in (N_GAIA_STARS_ALIGN, N_GAIA_STARS_ALIGN_RETRY):
            last_exc = None
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    this_wcs = compute_wcs(
                        coords[0:N_IMAGE_STARS_ALIGN],
                        radecs[0:n_gaia],
                        tolerance=WCS_MATCH_TOLERANCE,
                    )
            except (IndexError, ValueError) as exc:
                # twirl's two too-few-stars exits: an IndexError when cross_match
                # finds zero pairs (empty float index array), or a ValueError out
                # of fit_wcs_from_points when too few matched points reach scipy's
                # least-squares fitter (the original SS Leo failure). Surface
                # either as a recoverable frame error; anything else is an
                # unexpected bug and is left to propagate. The original is
                # preserved for the log.
                last_exc = exc
                this_wcs = None
            if this_wcs is not None:
                break
        if this_wcs is None:
            if last_exc is not None:
                msg = "twirl raised while solving the WCS"
                raise WCSSolveError(msg) from last_exc
            msg = "twirl returned no WCS (too few matched stars)"
            raise WCSSolveError(msg)
    else:
        this_wcs = wcs

    if photometry_coords is not None:
        aligned_coords = this_wcs.world_to_pixel(photometry_coords)
        aligned_coords = np.array(aligned_coords).T
    else:
        aligned_coords = coords

    return aligned_coords, this_wcs


def centroid_stars(calibrated_data, aligned_coords, cnn):
    """
    Centroid stars at the given pixel coordinates.

    Parameters
    ----------
    calibrated_data : numpy.ndarray
        Calibrated image data.
    aligned_coords : numpy.ndarray
        Pixel coordinates at which to centroid.
    cnn : eloy.centroid.Ballet
        Centroiding CNN model.

    Returns
    -------
    centroid_coords : numpy.ndarray
        Centroided star coordinates in pixel space.
    """
    return centroid.ballet_centroid(calibrated_data, aligned_coords, cnn)


def annulus_sigma_clip_stats(data, coords, r_in, r_out, input_mask=None, sigma=3):
    """
    Compute the sigma-clipped median and standard deviation in an annulus.

    Parameters
    ----------
    data : numpy.ndarray
        2D image data.
    coords : numpy.ndarray
        Array of (x, y) coordinates.
    r_in : float
        Inner radius of the annulus.
    r_out : float
        Outer radius of the annulus.
    input_mask : numpy.ndarray or None, optional
        Optional mask to apply to the data (e.g., Bayer mask), by default None.
    sigma : float, optional
        Sigma for sigma-clipping, by default 3.

    Returns
    -------
    bkg_median : numpy.ndarray
        Sigma-clipped median background per pixel for each coordinate.
    bkg_std : numpy.ndarray
        Sigma-clipped standard deviation per pixel for each coordinate.
    """
    annulus = CircularAnnulus(coords, r_in, r_out)
    sigclip = SigmaClip(sigma=sigma)
    aperstats = ApertureStats(data, annulus, mask=input_mask, sigma_clip=sigclip)

    return aperstats.median, aperstats.std


def measure_photometry(
    calibrated_data,
    centroid_coords,
    aligned_coords,
    fwhm,
    egain,
    mask,
    *,
    radii=RELATIVE_RADII,
    annulus=ANNULUS,
):
    """
    Perform aperture photometry, background subtraction, and error calculation.

    Parameters
    ----------
    calibrated_data : numpy.ndarray
        Calibrated image data.
    centroid_coords : numpy.ndarray
        Centroided star coordinates.
    aligned_coords : numpy.ndarray
        Aligned star coordinates (used for peak measurement).
    fwhm : float
        FWHM of the PSF in pixels.
    egain : float
        System gain in e-/adu.
    mask : numpy.ndarray or None
        Bayer mask to apply to the image data.
    radii : array-like or float, optional
        Aperture radii in units of FWHM; multiplied by `fwhm` to get the actual
        aperture sizes. A scalar is treated as a single radius. Defaults to the
        module-level `RELATIVE_RADII`.
    annulus : tuple of float, optional
        Background annulus ``(inner, outer)`` radii in units of FWHM, with
        ``outer > inner``. Defaults to the module-level `ANNULUS`.

    Returns
    -------
    dict
        Keys: tot_count, count_err, bkgd_count, peak_count, snr,
        total_bkg, fluxes, aperture_radii, annulus_radii.

    Raises
    ------
    ValueError
        If `annulus` is not a 2-element sequence with the outer radius larger
        than the inner radius.

    Notes
    -----
    Per-star ``snr`` (and the noise terms feeding it) may be ``NaN`` for faint
    stars whose annulus background over-subtracts (negative ``net_count``) or for
    stars near the frame edge whose annulus statistics are undefined. These
    ``NaN``s are an accepted part of the contract: they are filtered out
    downstream by `eloy_to_starlist` (which keeps only rows with ``tot_count > 0``,
    finite ``count_err``, and in-bounds ``x``/``y``). The associated
    ``invalid``/``divide`` RuntimeWarnings are deliberately suppressed.
    """
    msg = (
        "annulus must be a 2-element (inner, outer) sequence with "
        f"outer > inner; got {annulus!r}."
    )
    # Unpacking turns both non-sequences (TypeError) and wrong-length sequences
    # (ValueError) into the same actionable ValueError promised in the docstring.
    try:
        inner, outer = annulus
    except (TypeError, ValueError):
        raise ValueError(msg) from None
    if outer <= inner:
        raise ValueError(msg)
    # Coerce to at least 1D float so a scalar radii (e.g. 1.0) is treated as a
    # single radius rather than a 0-d array (which is not iterable).
    apertures_radii = np.atleast_1d(np.asarray(radii, dtype=float)) * fwhm

    # The inner background radius is pushed out to at least the largest aperture
    # so the annulus never overlaps the photometry aperture. If that leaves the
    # outer radius at or inside the inner one, there is no usable annulus.
    r_in = np.max([np.max(apertures_radii), inner * fwhm])
    r_out = outer * fwhm
    if r_out <= r_in:
        radius_msg = (
            f"no usable background annulus: outer radius ({r_out}) is not larger "
            f"than the inner radius ({r_in}) after expanding it to the largest "
            "aperture. Use a larger annulus or smaller radii."
        )
        raise ValueError(radius_msg)
    annulus_radii = (r_in, r_out)

    flux = photometry.aperture_photometry(
        calibrated_data,
        centroid_coords,
        apertures_radii,
        mask=mask,
    )
    aperture_area = np.array(
        [
            a.area_overlap(calibrated_data, mask=mask)
            for a in [
                CircularAperture(
                    centroid_coords,
                    r=r,
                )
                for r in apertures_radii
            ]
        ]
    ).T

    bkg, bkg_std = annulus_sigma_clip_stats(
        calibrated_data,
        centroid_coords,
        *annulus_radii,
        input_mask=mask,
    )
    total_bkg = bkg[:, None] * aperture_area

    peaks = np.nanmax(
        utils.cutout(calibrated_data, aligned_coords, (25, 25)),
        axis=(1, 2),
    )

    net_count = flux - total_bkg
    # Background noise per pixel estimated from the annulus standard deviation
    tot_noise_bkgd = bkg_std[:, None] * np.sqrt(aperture_area)
    # NaN contract: a faint star whose annulus over-subtracts gives net_count < 0
    # (so sqrt yields NaN), and an edge-of-frame annulus can give a NaN
    # background/std that poisons tot_noise and snr. These NaNs are *expected*
    # intermediates -- eloy_to_starlist drops such rows downstream (tot_count > 0,
    # finite count_err, in-bounds x/y) -- so suppress the corresponding
    # invalid/divide RuntimeWarnings here rather than letting them spam the logs
    # and obscure real problems.
    with np.errstate(invalid="ignore", divide="ignore"):
        poiss_noise = np.sqrt(egain * net_count) / egain
        tot_noise = np.sqrt(poiss_noise**2 + tot_noise_bkgd**2)
        snr = net_count / tot_noise

    return {
        "tot_count": net_count[:, 0],
        "count_err": tot_noise[:, 0],
        "bkgd_count": bkg,
        "bkgd_std": bkg_std,
        "peak_count": peaks,
        "snr": snr[:, 0],
        "total_bkg": total_bkg,
        "fluxes": flux,
        "aperture_radii": float(apertures_radii[0]),
        "annulus_radii": annulus_radii,
        "aperture_area": aperture_area[:, 0],
    }


def prepare_image(
    file,
    radecs,
    cnn,
    *,
    config=None,
    detect_on_bayer_balanced=False,
    photometry_coords=None,
    user_specific_metadata=None,
    wcs=None,
):
    """
    Detect sources, align, and centroid for a single image.

    Parameters
    ----------
    file : str or Path
        Path to the FITS file.
    radecs : numpy.ndarray
        Gaia reference sky coordinates (RA/Dec) used for WCS alignment.
    cnn : eloy.centroid.Ballet
        Centroiding CNN model.
    config : PhotometryConfig or None, optional
        Photometry configuration. The ``instrument`` settings (detection
        threshold, opening, and FWHM cutout window) drive the detection call. If
        None (default), a default ``PhotometryConfig`` is used.
    detect_on_bayer_balanced : bool, optional
        Whether to detect sources on Bayer balanced data (default is False).
    photometry_coords : `astropy.coordinates.SkyCoord` or None, optional
        If provided, these are the coordinates used for centroiding instead of those
        detected in this image. This allows for centroiding on a different set of
        coordinates than those used for WCS alignment. By default None (centroiding is
        done on detected coords).
    user_specific_metadata : dict or None, optional
        User-specific metadata to include in the output. By default None.
    wcs : `astropy.wcs.WCS` or None, optional
        Precomputed WCS to reuse instead of solving one for this image; passed
        through to `align`. By default None.

    Returns
    -------
    ImageData
        Per-image detection/alignment results.

    Raises
    ------
    WCSSolveError
        If the per-image WCS cannot be solved. The source `file` is attached to
        the error before it propagates. (`calibration_sequence` may also raise
        `TooFewStarsError`, which propagates unchanged.)
    """
    # "calibrate" the data and get initial detections for WCS alignment and
    # FWHM estimation. calibration_sequence raises TooFewStarsError (a
    # FrameError) when the frame is unusable; let it propagate to the batch loop.
    # Pass the CNN so the FWHM that sizes the photometry aperture is measured by
    # re-centroiding detections, keeping it independent of the detection opening.
    config = config or PhotometryConfig()
    instrument = config.instrument
    calibrated_data, metadata, coords, fwhm, _ = calibration_sequence(
        file,
        threshold=instrument.thresh,
        opening=instrument.detection_opening,
        detect_on_bayer_balanced=detect_on_bayer_balanced,
        cnn=cnn,
        fwhm_cutout_half=instrument.fwhm_cutout_half,
        fwhm_n_stars=instrument.fwhm_n_stars,
        profile=instrument,
    )

    if user_specific_metadata is not None:
        metadata.update(user_specific_metadata)

    if detect_on_bayer_balanced:
        working_image = calibrated_data.copy()
        bayer_balance_image(working_image)
    else:
        working_image = calibrated_data

    try:
        aligned_coords, this_wcs = align(
            coords,
            radecs,
            photometry_coords=photometry_coords,
            wcs=wcs,
        )
    except WCSSolveError as exc:
        # align does not know the source file; attach it here so the batch
        # loop can report which frame failed.
        exc.file = file
        raise
    centroid_coords = centroid_stars(working_image, aligned_coords, cnn)

    header = fits.getheader(file)

    return ImageData(
        calibrated_data=calibrated_data,
        coords=coords,
        fwhm=fwhm,
        centroid_coords=centroid_coords,
        aligned_coords=aligned_coords,
        wcs=this_wcs,
        header=header,
        input_photometry_coords=photometry_coords,
        metadata=metadata,
    )


def build_photometry_table(
    img,
    mask,
    *,
    config=None,
    radii=None,
    annulus=None,
    drift_tolerance=None,
    drift_cap=None,
):
    """
    Run photometry with a given mask and build an output table.

    Parameters
    ----------
    img : ImageData
        Per-image detection/alignment results.
    mask : numpy.ndarray or None
        Bayer mask to apply to the image data.
    config : PhotometryConfig or None, optional
        Photometry configuration supplying the apertures/drift defaults. Any of
        the explicit keyword overrides below take precedence over it. If None
        (default), a default ``PhotometryConfig`` is used.
    radii : array-like or float or None, optional
        Aperture radii in units of FWHM, passed through to `measure_photometry`
        (a scalar is treated as a single radius). If None (default), taken from
        ``config.apertures.radii``.
    annulus : tuple of float or None, optional
        Background annulus inner and outer radii in units of FWHM, passed
        through to `measure_photometry`. If None (default), taken from
        ``config.apertures.annulus``.
    drift_tolerance : float or None, optional
        Maximum allowed centroid drift in units of FWHM, passed to
        `centroid_drift_flag`. If None (default), taken from
        ``config.drift.drift_tolerance_fwhm``.
    drift_cap : float or None, optional
        Absolute pixel cap on the allowed centroid drift, passed to
        `centroid_drift_flag`. If None (default), taken from
        ``config.drift.drift_cap_pix``.

    Returns
    -------
    Table
        Photometry table for this image and mask. Includes a boolean
        ``centroid_drift`` column flagging stars whose centroid wandered too far
        from its aligned position (see `centroid_drift_flag`).

    Raises
    ------
    FrameMetadataError
        If the image header has a missing or unparseable ``DATE-OBS``.
    """
    config = config or PhotometryConfig()
    if radii is None:
        radii = config.apertures.radii
    if annulus is None:
        annulus = config.apertures.annulus
    if drift_tolerance is None:
        drift_tolerance = config.drift.drift_tolerance_fwhm
    if drift_cap is None:
        drift_cap = config.drift.drift_cap_pix
    phot = measure_photometry(
        img.calibrated_data,
        img.centroid_coords,
        img.aligned_coords,
        img.fwhm,
        img.metadata["egain"],
        mask,
        radii=radii,
        annulus=annulus,
    )
    if img.input_photometry_coords is not None:
        # The caller supplied known sky coordinates; use them directly rather
        # than re-deriving RA/Dec from this image's WCS. input_photometry_coords
        # drives aligned_coords -> centroid_coords one-to-one, so rows line up.
        ra_deg = img.input_photometry_coords.ra.degree
        dec_deg = img.input_photometry_coords.dec.degree
    else:
        centroid_ra_dec = img.wcs.pixel_to_world(
            img.centroid_coords[..., 0],
            img.centroid_coords[..., 1],
        )
        ra_deg = centroid_ra_dec.ra.degree
        dec_deg = centroid_ra_dec.dec.degree

    data = Table()
    data["tot_count"] = phot["tot_count"]
    data["total_bkg"] = phot["total_bkg"]
    data["bkgd_count"] = phot["bkgd_count"]
    data["bkgd_std"] = phot["bkgd_std"]
    data["count_err"] = phot["count_err"]
    data["snr"] = phot["snr"]
    data["fluxes"] = phot["fluxes"]
    try:
        data["time"] = Time(parser.parse(img.header["DATE-OBS"])).jd
    except (KeyError, ValueError, TypeError) as exc:
        msg = "missing or unparseable DATE-OBS header keyword"
        raise FrameMetadataError(msg) from exc
    data["sky"] = np.mean(
        phot["total_bkg"] / (np.pi * (np.asarray(radii) * img.fwhm) ** 2),
    )
    data["airmass"] = _airmass_from_header(img.header)
    data["peak_count"] = phot["peak_count"]
    data["stars_in_exp"] = len(img.coords)
    data["ra"] = ra_deg
    data["dec"] = dec_deg
    data["x"] = img.centroid_coords[..., 0]
    data["y"] = img.centroid_coords[..., 1]
    data["centroid_drift"] = centroid_drift_flag(
        img.centroid_coords,
        img.aligned_coords,
        img.fwhm,
        tolerance=drift_tolerance,
        cap=drift_cap,
    )
    data["aperture_area"] = phot["aperture_area"]
    data.meta["fwhm"] = float(img.fwhm)
    data.meta["aperture_radii"] = phot["aperture_radii"]
    data.meta["annulus_radii"] = phot["annulus_radii"]

    return data


def process_one_image(
    file,
    user_specific_metadata,
    radecs,
    cnn,
    bayer_masks,
    *,
    config=None,
    bayer_balance_detection=True,
    input_photometry_coords=None,
):
    """
    Process a single image file and return one photometry table per input mask.

    Parameters
    ----------
    file : str or Path
        Path to the FITS file.
    user_specific_metadata : dict
        User-specific metadata to include in the output.
    radecs : numpy.ndarray
        Gaia reference sky coordinates (RA/Dec) used for WCS alignment.
    cnn : eloy.centroid.Ballet
        Centroiding CNN model.
    bayer_masks : dict of {str: numpy.ndarray or None}
        Dictionary mapping each filter name to the Bayer mask to apply to the
        image. The filter name is stamped into each returned table's metadata so
        the results can be grouped by filter. Each Bayer mask should have the same
        shape as the image data. To include the synthetic full-frame "L4"
        luminance channel, map "L4" to None; it must be ordered after the RGB
        channels (TR/TG/TB) it is built from.
    config : PhotometryConfig or None, optional
        Photometry configuration threaded through to `prepare_image` and
        `build_photometry_table`. If None (default), a default
        ``PhotometryConfig`` is used.
    bayer_balance_detection : bool, optional
        Whether to perform source detection on Bayer balanced data. This is usually
        desirable for data with a bayer pattern.
    input_photometry_coords : `astropy.coordinates.SkyCoord` or None, optional
        If provided, these sky coordinates are used for centroiding instead of those
        detected in this image. The sky coordinates passed in are recorded as the
        sky coordinates in the output.

    Returns
    -------
    dict of {str: Table}
        Dictionary mapping each filter name to the photometry table for that
        filter. If the frame cannot be processed, the `FrameError` raised by
        `prepare_image` (too few stars, unsolvable WCS, ...) propagates
        unchanged; `process_batch` catches it, logs it, and skips the frame.

    Notes
    -----
    When `bayer_masks` includes "L4", the "TR", "TG", and "TB" channels must be
    present and ordered before it; otherwise `calculate_l4_quantities` raises a
    `ValueError`.
    """
    # Calculate everything we need for all filters at once. prepare_image raises
    # a FrameError (TooFewStarsError / WCSSolveError) when the frame is unusable;
    # let it propagate to the batch loop.
    config = config or PhotometryConfig()
    img = prepare_image(
        file,
        radecs,
        cnn,
        config=config,
        detect_on_bayer_balanced=bayer_balance_detection,
        photometry_coords=input_photometry_coords,
        user_specific_metadata=user_specific_metadata,
    )

    by_filter_data = {}
    for filter_name, mask in bayer_masks.items():
        data = build_photometry_table(img, mask, config=config)
        data.meta["filter"] = filter_name
        data.meta["full_image_meta"] = img.metadata
        if filter_name == "L4":
            # L4 is the channel sum of TR/TG/TB, so those must already have been
            # processed. generate_bayer_masks orders L4 last to guarantee this;
            # calculate_l4_quantities validates that the RGB channels are present
            # and raises a clear ValueError if a caller passes a mask dict that
            # violates the ordering.
            calculate_l4_quantities(data, by_filter_data, img.metadata["egain"])
        by_filter_data[filter_name] = data

    return by_filter_data


def calculate_l4_quantities(final_data, by_filter_data, egain):
    """
    Calculate the "L4" photometry given RGB photometry on a Bayer array.

    Note that ``final_data`` is modified in place.

    Parameters
    ----------
    final_data : astropy.table.Table
        The final photometry table for the L4 filter, modified in place.
    by_filter_data : dict
        A dictionary containing the photometry data for each individual filter.
    egain : float
        The gain of the image.

    Raises
    ------
    ValueError
        If any of the "TR", "TG", or "TB" channels are missing from
        ``by_filter_data``; the L4 channel is built from all three.
    """
    # L4 is the channel sum of TR/TG/TB, so each must already be present. Fail
    # loudly with an actionable message rather than a bare KeyError raised deep
    # inside the combination below.
    missing = {"TR", "TG", "TB"} - by_filter_data.keys()
    if missing:
        msg = (
            f"calculate_l4_quantities requires {sorted(missing)} in by_filter_data "
            "before the L4 channel can be combined."
        )
        raise ValueError(msg)

    # L4 total count is sum of the individual filter total counts
    final_data["tot_count"] = (
        by_filter_data["TR"]["tot_count"]
        + by_filter_data["TG"]["tot_count"]
        + by_filter_data["TB"]["tot_count"]
    )

    # L4 aperture area is the sum of the individual filter aperture areas
    final_data["aperture_area"] = (
        by_filter_data["TR"]["aperture_area"]
        + by_filter_data["TG"]["aperture_area"]
        + by_filter_data["TB"]["aperture_area"]
    )

    # Background is the weighted average of the individual filter backgrounds
    final_data["bkgd_count"] = (
        by_filter_data["TR"]["bkgd_count"] * by_filter_data["TR"]["aperture_area"]
        + by_filter_data["TG"]["bkgd_count"] * by_filter_data["TG"]["aperture_area"]
        + by_filter_data["TB"]["bkgd_count"] * by_filter_data["TB"]["aperture_area"]
    ) / final_data["aperture_area"]

    # For peak count, create a numpy array of the individual filter peak counts
    # and take the max along the filter axis
    final_data["peak_count"] = np.max(
        [
            by_filter_data["TR"]["peak_count"],
            by_filter_data["TG"]["peak_count"],
            by_filter_data["TB"]["peak_count"],
        ],
        axis=0,
    )

    # For error, add the individual filter background errors in quadrature,
    # multiplied by the aperture areas, then add in quadrature to the Poisson
    # error from the total count.
    final_data["count_err"] = np.sqrt(
        (
            by_filter_data["TR"]["bkgd_std"] ** 2
            * by_filter_data["TR"]["aperture_area"]
            + by_filter_data["TG"]["bkgd_std"] ** 2
            * by_filter_data["TG"]["aperture_area"]
            + by_filter_data["TB"]["bkgd_std"] ** 2
            * by_filter_data["TB"]["aperture_area"]
        )
        + final_data["tot_count"] / egain  # Poisson error from total count
    )

    # Recompute SNR from the recombined L4 count and error. The table arrives
    # here from a full-frame photometry pass, so its snr column otherwise still
    # reflects the discarded full-frame measurement instead of the channel sum.
    final_data["snr"] = final_data["tot_count"] / final_data["count_err"]

    # fluxes/total_bkg/bkgd_std are not recombined across TR/TG/TB and have no
    # L4-consistent meaning, so drop them rather than leave the stale full-frame
    # values from the incoming table (issue #21). remove_columns is given only
    # the columns actually present so it is safe if the table is built without
    # them.
    stale_columns = [
        col for col in ("fluxes", "total_bkg", "bkgd_std") if col in final_data.colnames
    ]
    final_data.remove_columns(stale_columns)
