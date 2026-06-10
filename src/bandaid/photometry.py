"""
Photometry pipeline for Smart Telescope images.

Provides the per-image processing steps used by the bandaid pipeline: source
detection and FWHM estimation, WCS alignment against a Gaia reference, CNN
centroiding, aperture photometry with annulus background subtraction and error
estimation, bright-neighbor contamination flagging, and conversion of the
resulting tables into a ``StarList``.
"""

import json
from dataclasses import dataclass
from importlib.resources import files as package_files

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.table import Table
from astropy.time import Time
from dateutil import parser
from eloy import centroid, detection, photometry, psf, utils
from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture
from st_pipeline.schema_definition import StarList
from twirl import compute_wcs

from .image2sl_qt import bayer_balance_image

# Half-width (in pixels) of the cutout taken around each star for per-star
# processing.
CUTOUT = 500  # 120

# Number of (brightest) detected stars used to compute the per-image WCS in
# `align`. Only the first N_STARS_ALIGN detections/reference coords are paired up.
N_STARS_ALIGN = 15
# Source-detection threshold (in units of the background sigma) passed to
# `detection.stars_detection`.
THRESH = 0.5

# Minimum number of detected stars required before an image can be processed.
MIN_DETECTED_STARS = 3

# Relative radii and annulus are defined here. These radii are multiplied by
# each image's FWHM to determine the actual aperture sizes.

# Only need one radius for STWG, but it needs to be in an iterable
RELATIVE_RADII = np.array([1.0])  # np.linspace(0.1, 5, 30)
ANNULUS = (5, 8)

# Bright-neighbor rejection. A star is flagged if any brighter neighbor's PSF
# wings would contribute more than CONTAMINATION_TOLERANCE of the target flux
# inside the 1*FWHM aperture, modeled as a Moffat profile of index MOFFAT_BETA.
CONTAMINATION_TOLERANCE = 0.01
MOFFAT_BETA = 3.0
# At least two stars are needed before any neighbor pair can exist.
MIN_STARS_FOR_PAIRS = 2

# Centroid-drift check. A star is flagged if its measured centroid wandered
# more than `min(DRIFT_TOLERANCE_FWHM * fwhm, DRIFT_CAP_PIX)` pixels from its
# aligned/expected position. The FWHM-relative term lets the allowance scale with
# seeing, while the absolute pixel cap keeps a pathologically large FWHM from
# licensing an enormous shift. These defaults are empirical starting points and are
# meant to be tuned against real frames (override via the kwargs on
# `centroid_drift_flag` / `build_photometry_table`).
DRIFT_TOLERANCE_FWHM = 1.0  # max centroid drift, in units of FWHM
DRIFT_CAP_PIX = 4.0  # absolute pixel cap on allowed drift


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

    Returns
    -------
    ndarray of bool, shape (N,)
        True where any neighbor sits inside this star's contamination radius.
    """
    radecs = np.asarray(radecs, dtype=float)
    mags = np.asarray(mags, dtype=float)
    flagged = np.zeros(len(radecs), dtype=bool)
    if len(radecs) < MIN_STARS_FOR_PAIRS:
        return flagged

    finite = np.isfinite(mags)
    if not finite.any():
        return flagged

    # The largest separation any pair can require is the faintest target with
    # the brightest neighbor; `min_separation_fwhm` is monotonic in delta_mag,
    # so capping the neighbor search there finds every pair the dense N x N
    # separation matrix would flag at a fraction of the time and memory.
    max_delta_mag = np.max(mags[finite]) - np.min(mags[finite])
    max_sep_arcsec = (
        min_separation_fwhm(max_delta_mag, tolerance=tolerance, beta=beta) * fwhm_arcsec
    )
    if max_sep_arcsec <= 0:
        return flagged

    coords = SkyCoord(radecs[:, 0], radecs[:, 1], unit="deg")
    idx_target, idx_neighbor, sep2d, _ = search_around_sky(
        coords,
        coords,
        max_sep_arcsec * u.arcsec,
    )

    # Same pair convention as `_contamination_flag`: a star is never its own
    # neighbor (but distinct stars at zero separation are), and a pair only
    # counts when both magnitudes are finite.
    keep = (idx_target != idx_neighbor) & finite[idx_target] & finite[idx_neighbor]
    idx_target = idx_target[keep]
    idx_neighbor = idx_neighbor[keep]
    sep_arcsec = sep2d.arcsec[keep]

    min_sep = (
        min_separation_fwhm(
            mags[idx_target] - mags[idx_neighbor],
            tolerance=tolerance,
            beta=beta,
        )
        * fwhm_arcsec
    )
    flagged[idx_target[sep_arcsec < min_sep]] = True
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


def calibration_sequence(file, threshold=1) -> tuple:
    """
    Find sources and compute FWHM for an image.

    Parameters
    ----------
    file : str
        Path to the FITS file.
    threshold : float, optional
        Detection threshold for star finding, by default 1

    Returns
    -------
    tuple
        A tuple containing the calibrated data, metadata, region coordinates,
        FWHM, and regions.

    """
    data = fits.getdata(file)
    header = fits.getheader(file)

    metadata = metadata_from_header(header)
    max_adu = metadata["largest_usable_adu_value"]

    # Multiplying by 1 should force conversion from int to float data
    calibrated_data = 1.0 * data
    regions = detection.stars_detection(calibrated_data, threshold=threshold)

    # in case we detect fewer than the minimum number of stars
    if len(regions) < MIN_DETECTED_STARS:
        return None, [], None, None, None

    region_coords_xy = np.array([(r.centroid[1], r.centroid[0]) for r in regions])
    cutouts = utils.cutout(calibrated_data, region_coords_xy, (50, 50))

    # Drop any cutouts that are saturated -- NOTE THAT THIS LEAVES BEHIND
    # SATURATED REGIONS
    cutouts = np.array(list(filter(lambda data: np.max(data) < max_adu, cutouts)))

    # If every detected source was saturated there is nothing left to fit a PSF to
    if len(cutouts) == 0:
        return None, [], None, None, None

    # Drop any regions that are saturated for calculating the FWHM
    cutouts_normalized = cutouts / np.nanmax(cutouts, (1, 2))[:, None, None]

    # Average the cutouts...yolo I guess on whether these are good detections
    epsf = np.nanmedian(cutouts_normalized, 0)

    # Note fitting is only done to the normalized cutout
    psf_params = psf.fit_gaussian(epsf)
    fwhm = psf.gaussian_sigma_to_fwhm * np.mean(
        [psf_params["sigma_x"], psf_params["sigma_y"]],
    )

    # Saves a bit of memory, I guess, by forcing garbage collection
    del (
        cutouts_normalized,
        data,
        cutouts,
        epsf,
        header,
    )

    return calibrated_data, metadata, region_coords_xy, fwhm, regions


def metadata_from_header(header):
    """
    Build a metadata dictionary from a JSON template and a FITS header.

    Parameters
    ----------
    header : astropy.io.fits.Header or dict
        FITS header to look up values in.

    Returns
    -------
    dict
        Metadata dictionary with header lookups resolved.
    """
    json_path = package_files("bandaid").joinpath(
        "meta_json_files",
        "Seestar50",
        "basic.json",
    )
    with json_path.open() as f:
        template = json.load(f)

    # Collect fallback values from "#key" entries
    defaults = {}
    for key, value in template.items():
        if key.startswith("#"):
            defaults[key[1:]] = value

    metadata = {}
    for key, value in template.items():
        # Skip comment/internal keys
        if key.startswith(("_", "#")):
            continue

        if isinstance(value, str) and value.startswith("@"):
            header_key = value[1:]
            metadata[key] = header.get(header_key, defaults.get(key))
        elif isinstance(value, str) and value.startswith("!"):
            parts = value[1:].split()
            header_key = parts[0]
            index = int(parts[2])
            metadata[key] = header[header_key].split()[index]
        else:
            metadata[key] = value

    metadata["width"] = header["NAXIS1"]
    metadata["height"] = header["NAXIS2"]
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
    return StarList.from_table(eloy_table[good], metadata=metadata)


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
    """
    if wcs is None:
        # Feed the brightest N_STARS_ALIGN detections and Gaia reference
        # RA/Decs to twirl's asterism (quad) matcher. twirl matches by
        # geometric shape, so the two lists need NOT be in the same order or
        # even the same length -- there is no per-index correspondence. The
        # slicing just limits the matcher to the brightest, most reliable
        # stars. compute_wcs returns None if too few stars overlap to satisfy
        # its min_match threshold, so callers must supply enough stars (not
        # enough *matched* stars).
        this_wcs = compute_wcs(
            coords[0:N_STARS_ALIGN],
            radecs[0:N_STARS_ALIGN],
            tolerance=1,
        )
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
    relative_radii=RELATIVE_RADII,
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
    relative_radii : array-like or float, optional
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
    # Coerce to at least 1D float so a scalar relative_radii (e.g. 1.0) is
    # treated as a single radius rather than a 0-d array (which is not iterable).
    apertures_radii = np.atleast_1d(np.asarray(relative_radii, dtype=float)) * fwhm

    # The inner background radius is pushed out to at least the largest aperture
    # so the annulus never overlaps the photometry aperture. If that leaves the
    # outer radius at or inside the inner one, there is no usable annulus.
    r_in = np.max([np.max(apertures_radii), inner * fwhm])
    r_out = outer * fwhm
    if r_out <= r_in:
        radius_msg = (
            f"no usable background annulus: outer radius ({r_out}) is not larger "
            f"than the inner radius ({r_in}) after expanding it to the largest "
            "aperture. Use a larger annulus or smaller relative_radii."
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
    ImageData or None
        Per-image results, or None if too few stars were detected.
    """
    # "calibrate" the data and get initial detections for WCS alignment and
    # FWHM estimation
    calibrated_data, metadata, coords, fwhm, _ = calibration_sequence(
        file,
        threshold=THRESH,
    )

    # calibration_sequence returns the (None, [], None, None, None) sentinel when too
    # few usable stars were detected; honor the documented "return None" contract.
    if calibrated_data is None:
        return None

    if user_specific_metadata is not None:
        metadata.update(user_specific_metadata)

    if detect_on_bayer_balanced:
        working_image = calibrated_data.copy()
        bayer_balance_image(working_image)
    else:
        working_image = calibrated_data

    aligned_coords, this_wcs = align(
        coords,
        radecs,
        photometry_coords=photometry_coords,
        wcs=wcs,
    )
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
    relative_radii=RELATIVE_RADII,
    annulus=ANNULUS,
    drift_tolerance=DRIFT_TOLERANCE_FWHM,
    drift_cap=DRIFT_CAP_PIX,
):
    """
    Run photometry with a given mask and build an output table.

    Parameters
    ----------
    img : ImageData
        Per-image detection/alignment results.
    mask : numpy.ndarray or None
        Bayer mask to apply to the image data.
    relative_radii : array-like or float, optional
        Aperture radii in units of FWHM, passed through to `measure_photometry`
        (a scalar is treated as a single radius). Defaults to the module-level
        `RELATIVE_RADII`.
    annulus : tuple of float, optional
        Background annulus inner and outer radii in units of FWHM, passed
        through to `measure_photometry`. Defaults to the module-level `ANNULUS`.
    drift_tolerance : float, optional
        Maximum allowed centroid drift in units of FWHM, passed to
        `centroid_drift_flag`. Defaults to `DRIFT_TOLERANCE_FWHM`.
    drift_cap : float, optional
        Absolute pixel cap on the allowed centroid drift, passed to
        `centroid_drift_flag`. Defaults to `DRIFT_CAP_PIX`.

    Returns
    -------
    Table
        Photometry table for this image and mask. Includes a boolean
        ``centroid_drift`` column flagging stars whose centroid wandered too far
        from its aligned position (see `centroid_drift_flag`).
    """
    phot = measure_photometry(
        img.calibrated_data,
        img.centroid_coords,
        img.aligned_coords,
        img.fwhm,
        img.metadata["egain"],
        mask,
        relative_radii=relative_radii,
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
    data["time"] = Time(parser.parse(img.header["DATE-OBS"])).jd
    data["sky"] = np.mean(
        phot["total_bkg"] / (np.pi * (np.asarray(relative_radii) * img.fwhm) ** 2),
    )
    data["airmass"] = img.header.get("AIRMASS", np.nan)
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
    bayer_balance_detection : bool, optional
        Whether to perform source detection on Bayer balanced data. This is usually
        desirable for data with a bayer pattern.
    input_photometry_coords : `astropy.coordinates.SkyCoord` or None, optional
        If provided, these sky coordinates are used for centroiding instead of those
        detected in this image. The sky coordinates passed in are recorded as the
        sky coordinates in the output.

    Returns
    -------
    dict of {str: Table} or None
        Dictionary mapping each filter name to the photometry table for that
        filter, or None if the image could not be processed.

    Notes
    -----
    When `bayer_masks` includes "L4", the "TR", "TG", and "TB" channels must be
    present and ordered before it; otherwise `calculate_l4_quantities` raises a
    `ValueError`.
    """
    # Calculate everything we need for all filters at once.
    img = prepare_image(
        file,
        radecs,
        cnn,
        detect_on_bayer_balanced=bayer_balance_detection,
        photometry_coords=input_photometry_coords,
        user_specific_metadata=user_specific_metadata,
    )

    if img is None:
        return None

    by_filter_data = {}
    for filter_name, mask in bayer_masks.items():
        data = build_photometry_table(img, mask)
        data.meta["filter"] = filter_name
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
