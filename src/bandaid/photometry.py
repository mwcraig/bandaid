import json
from dataclasses import dataclass
from importlib.resources import files as package_files
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.table import Table
from astropy.time import Time
from dateutil import parser
from eloy import centroid, detection, photometry, psf, utils
from eloy.centroid import Ballet
from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture
from st_pipeline.schema_definition import StarList
from twirl import compute_wcs

from .image2sl_qt import bayer_balance_image

CUTOUT = 500 # 120

N_STARS_ALIGN = 15
THRESH = 0.5

# Relative radii and annulus are defined here. These radii are multiplied by
# each image's FWHM to determine the actual aperture sizes.

# Only need one radius for STWG, but it needs to be in an iterable
RELATIVE_RADII = np.array([1.0])  # np.linspace(0.1, 5, 30)
ANNULUS = (5, 8)

# Size of cutout for centroiding
CUTOUT_SHAPE = (21, 21)

# Bright-neighbor rejection. A star is flagged if any brighter neighbor's PSF
# wings would contribute more than CONTAMINATION_TOLERANCE of the target flux
# inside the 1*FWHM aperture, modeled as a Moffat profile of index MOFFAT_BETA.
CONTAMINATION_TOLERANCE = 0.01
MOFFAT_BETA = 3.0


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
        How many magnitudes brighter the neighbor is than the target.
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


def neighbor_contamination_flag(
    coords, mags, fwhm,
    tolerance=CONTAMINATION_TOLERANCE,
    beta=MOFFAT_BETA,
):
    """
    Flag stars whose 1*FWHM aperture is contaminated by a too-close neighbor.

    Contamination is checked symmetrically: each pair (target, neighbor) is
    evaluated as fractional spillover into the *target's* aperture, so the
    same physical pair can flag the fainter star at a larger separation than
    the brighter one. Equal-brightness pairs are flagged inside ~2.18 FWHM.

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
    tolerance, beta : see `min_separation_fwhm`.

    Returns
    -------
    ndarray of bool, shape (N,)
        True where any neighbor sits inside this star's contamination radius.
    """
    coords = np.asarray(coords)
    mags = np.asarray(mags, dtype=float)
    n = len(coords)
    if n < 2:
        return np.zeros(n, dtype=bool)

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    delta_mag = mags[:, None] - mags[None, :]

    finite = np.isfinite(mags)
    valid = finite[:, None] & finite[None, :]
    np.fill_diagonal(valid, False)

    min_sep_pix = np.zeros_like(dist)
    if valid.any():
        min_sep_pix[valid] = min_separation_fwhm(
            delta_mag[valid], tolerance=tolerance, beta=beta,
        ) * fwhm

    too_close = valid & (dist < min_sep_pix)
    return too_close.any(axis=1)


def calibration_sequence(file: str, threshold: float = 1) -> tuple:
    """
    Find sources and compute FWHM for an image.

    Parameters
    ----------
    file : str
        Path to the FITS file.
    threshold : float, optional
        Detection threshold for star finding, by default 1
    max_adu : int, optional
        Maximum ADU value to consider for detections (used to filter out saturated
        stars), by default 0 (no filtering)

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

    # in case we detect fewer than 3 stars
    if len(regions) < 3:
        return None, [], None, None, None

    region_coords_xy = np.array([(r.centroid[1], r.centroid[0]) for r in regions])
    cutouts = utils.cutout(calibrated_data, region_coords_xy, (50, 50))

    # Drop any cutouts that are saturated -- NOTE THAT THIS LEAVES BEHIND SATURATED REGIONS
    cutouts = np.array(list(filter(lambda data: np.max(data) < max_adu, cutouts)))

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
       "meta_json_files", "Seestar50", "basic.json",
    )
    with Path(json_path).open() as f:
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
class ReferenceData:
    """Reference image data used to process each science image."""

    sky_coords: SkyCoord
    radecs: np.ndarray
    cnn: Ballet

    @classmethod
    def from_pixel_coords(cls, coords, wcs, radecs, cnn):
        """Create from pixel coordinates, converting to sky coordinates."""
        sky_coords = wcs.pixel_to_world(coords[..., 0], coords[..., 1])
        return cls(sky_coords=sky_coords, radecs=radecs, cnn=cnn)


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
    metadata: dict = None


def align_and_centroid(calibrated_data, coords, ref, photometry_coords=None):
    """
    Compute per-image WCS, align reference coordinates, and centroid.

    Parameters
    ----------
    calibrated_data : numpy.ndarray
        Calibrated image data.
    coords : numpy.ndarray
        Detected star coordinates in this image. This is used for WCS alignment and, if
        photometry_coords is None, for centroiding as well.
    ref : ReferenceData
        Reference image data (coords, WCS, Gaia RA/Decs, CNN model).
    photometry_coords : `astropy.coordinates.SkyCoord` or None, optional
        If provided, these are the coordinates used for centroiding instead of `coords`.
        This allows for centroiding on a different set of coordinates than those used
        for WCS alignment. By default None (centroiding is done on `coords`).

    Returns
    -------
    centroid_coords, aligned_coords, this_wcs
        centroid_coords : numpy.ndarray
            Centroided star coordinates in pixel space.
        aligned_coords : numpy.ndarray
            Aligned star coordinates in pixel space.
        this_wcs : astropy.wcs.WCS
            World Coordinate System for the image.
    """
    this_wcs = compute_wcs(
        coords[0:N_STARS_ALIGN], ref.radecs[0:N_STARS_ALIGN], tolerance=1,
    )
    if photometry_coords is not None:
        aligned_coords = this_wcs.world_to_pixel(photometry_coords)
        aligned_coords = np.array(aligned_coords).T
    else:
        aligned_coords = this_wcs.world_to_pixel(ref.sky_coords)
        aligned_coords = np.array(aligned_coords).T
    centroid_coords = centroid.ballet_centroid(calibrated_data, aligned_coords, ref.cnn)
    print((((aligned_coords - centroid_coords)**2).sum(axis=1)**0.5).max())
    return centroid_coords, aligned_coords, this_wcs


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


def measure_photometry(  # noqa: PLR0913
        calibrated_data,
        centroid_coords,
        aligned_coords,
        fwhm,
        egain,
        mask,
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

    Returns
    -------
    dict
        Keys: tot_count, count_err, bkgd_count, peak_count, snr,
        total_bkg, fluxes, aperture_radii, annulus_radii.
    """
    apertures_radii = RELATIVE_RADII * fwhm
    flux = photometry.aperture_photometry(
        calibrated_data, centroid_coords, apertures_radii, mask=mask,
    )
    annulus_radii = (
        np.max([np.max(apertures_radii), ANNULUS[0] * fwhm]),
        ANNULUS[1] * fwhm,
    )
    aperture_area = np.array(
        [
            a.area_overlap(calibrated_data, mask=mask)
            for a in
            [
                CircularAperture(
                    centroid_coords,
                    r=r,
                )
                for r in apertures_radii
            ]
        ]
    ).T

    bkg, bkg_std = annulus_sigma_clip_stats(
        calibrated_data, centroid_coords, *annulus_radii, input_mask=mask,
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
        "snr": snr,
        "total_bkg": total_bkg,
        "fluxes": flux,
        "aperture_radii": float(apertures_radii[0]),
        "annulus_radii": annulus_radii,
        "aperture_area": aperture_area[:, 0],
    }


def prepare_image(file, ref, *, detect_on_bayer_balanced=False, photometry_coords=None, user_specific_metadata=None):
    """
    Detect sources, align, and centroid for a single image.

    Parameters
    ----------
    file : str or Path
        Path to the FITS file.
    ref : ReferenceData
        Reference image data (sky coords, Gaia RA/Decs, CNN model).
    detect_on_bayer_balanced : bool, optional
        Whether to detect sources on Bayer balanced data (default is False).
    photometry_coords : `astropy.coordinates.SkyCoord` or None, optional
        If provided, these are the coordinates used for centroiding instead of those
        detected in this image. This allows for centroiding on a different set of
        coordinates than those used for WCS alignment. By default None (centroiding is
        done on detected coords).
    user_specific_metadata : dict or None, optional
        User-specific metadata to include in the output. By default None.

    Returns
    -------
    ImageData or None
        Per-image results, or None if too few stars were detected.
    """
    # "calibrate" the data and get initial detections for WCS alignment and
    # FWHM estimation
    calibrated_data, metadata, coords, fwhm, _ = calibration_sequence(
        file, threshold=THRESH,
    )

    if user_specific_metadata is not None:
        metadata.update(user_specific_metadata)

    if len(coords) < N_STARS_ALIGN:
        return None

    if detect_on_bayer_balanced:
        working_image = calibrated_data.copy()
        bayer_balance_image(working_image)
    else:
        working_image = calibrated_data

    centroid_coords, aligned_coords, this_wcs = align_and_centroid(
        working_image, coords, ref, photometry_coords=photometry_coords,
    )

    header = fits.getheader(file)

    return ImageData(
        calibrated_data=calibrated_data,
        coords=coords,
        fwhm=fwhm,
        centroid_coords=centroid_coords,
        aligned_coords=aligned_coords,
        wcs=this_wcs,
        header=header,
        metadata=metadata,
    )


def build_photometry_table(img, mask):
    """
    Run photometry with a given mask and build an output table.

    Parameters
    ----------
    img : ImageData
        Per-image detection/alignment results.
    mask : numpy.ndarray or None
        Bayer mask to apply to the image data.

    Returns
    -------
    Table
        Photometry table for this image and mask.
    """
    # Maybe check here or somewhere else that the centroid hasn't moved too much?
    phot = measure_photometry(
        img.calibrated_data, img.centroid_coords, img.aligned_coords,
        img.fwhm, img.metadata["egain"], mask,
    )
    centroid_ra_dec = img.wcs.pixel_to_world(
        img.centroid_coords[..., 0],
        img.centroid_coords[..., 1],
    )

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
        phot["total_bkg"] / (np.pi * (RELATIVE_RADII * img.fwhm) ** 2),
    )
    data["airmass"] = img.header.get("AIRMASS", np.nan)
    data["peak_count"] = phot["peak_count"]
    data["stars_in_exp"] = len(img.coords)
    data["ra"] = centroid_ra_dec.ra.degree
    data["dec"] = centroid_ra_dec.dec.degree
    data["x"] = img.centroid_coords[..., 0]
    data["y"] = img.centroid_coords[..., 1]
    data["aperture_area"] = phot["aperture_area"]
    # with np.errstate(divide="ignore", invalid="ignore"):
    #     instr_mag = -2.5 * np.log10(phot["tot_count"])
    # data["contaminated"] = neighbor_contamination_flag(
    #     img.centroid_coords, instr_mag, img.fwhm,
    # )
    data.meta["fwhm"] = float(img.fwhm)
    data.meta["aperture_radii"] = phot["aperture_radii"]
    data.meta["annulus_radii"] = phot["annulus_radii"]

    return data
