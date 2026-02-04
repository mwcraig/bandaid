import json
from dataclasses import dataclass
from importlib.resources import files as package_files
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from dateutil import parser
from eloy import centroid, detection, photometry, psf, utils
from eloy.centroid import Ballet
from st_pipeline.schema_definition import StarList
from twirl import compute_wcs

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


def calibration_sequence(file: str, threshold: float = 1, max_adu=0) -> tuple:
    """
    Find sources and compute FWHM for an image.

    Parameters
    ----------
    file : str
        Path to the FITS file.
    threshold : float, optional
        Detection threshold for star finding, by default 1
    """
    data = fits.getdata(file)
    header = fits.getheader(file)

    # Multiplying by 1 should force conversion from int to float data
    calibrated_data = 1.0 * data
    regions = detection.stars_detection(calibrated_data, threshold=threshold)

    # in case we detect fewer than 3 stars
    if len(regions) < 3:
        return None, [], None, None

    region_coords = np.array([(r.centroid[1], r.centroid[0]) for r in regions])
    cutouts = utils.cutout(calibrated_data, region_coords, (50, 50))

    # Drop any cutouts that are saturated -- NOTE THAT THIS LEAVES BEHIND SATURATED REGIONS
    cutouts = np.array(list(filter(lambda data: np.max(data) < max_adu, cutouts)))

    # Drop any regions that are saturdated for calculating the FWHM
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

    return calibrated_data, region_coords, fwhm, regions


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
    # REPLCE THIS WITH FILTERING FROM IMAGE2SL_QT
    good = ~np.isnan(eloy_table["tot_count"])
    good &= eloy_table["tot_count"] > 0
    good &= (
        (eloy_table["x"] > 0)
        & (eloy_table["x"] < metadata["width"])
        & (eloy_table["y"] > 0)
        & (eloy_table["y"] < metadata["height"])
    )
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


def align_and_centroid(calibrated_data, coords, ref):
    """
    Compute per-image WCS, align reference coordinates, and centroid.

    Parameters
    ----------
    calibrated_data : numpy.ndarray
        Calibrated image data.
    coords : numpy.ndarray
        Detected star coordinates in this image.
    ref : ReferenceData
        Reference image data (coords, WCS, Gaia RA/Decs, CNN model).

    Returns
    -------
    centroid_coords, aligned_coords, this_wcs
    """
    this_wcs = compute_wcs(
        coords[0:N_STARS_ALIGN], ref.radecs[0:N_STARS_ALIGN], tolerance=1,
    )
    aligned_coords = this_wcs.world_to_pixel(ref.sky_coords)
    aligned_coords = np.array(aligned_coords).T
    centroid_coords = centroid.ballet_centroid(calibrated_data, aligned_coords, ref.cnn)
    return centroid_coords, aligned_coords, this_wcs


def measure_photometry(calibrated_data, centroid_coords, aligned_coords, fwhm, egain, mask):
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
    aperture_area = np.pi * apertures_radii**2

    bkg = photometry.annulus_sigma_clip_median(
        calibrated_data, centroid_coords, *annulus_radii,
    )
    total_bkg = bkg[:, None] * aperture_area[None, :]

    peaks = np.nanmax(
        utils.cutout(calibrated_data, aligned_coords, (25, 25)),
        axis=(1, 2),
    )

    net_count = flux - total_bkg
    noise_bkgd_per_pixel = bkg * egain
    tot_noise_bkgd = noise_bkgd_per_pixel[:, None] * aperture_area[None, :]
    poiss_noise = np.sqrt(egain * net_count)
    tot_noise = np.sqrt(poiss_noise**2 + tot_noise_bkgd**2) / egain
    snr = net_count / tot_noise

    return {
        "tot_count": net_count[:, 0],
        "count_err": tot_noise[:, 0],
        "bkgd_count": bkg,
        "peak_count": peaks,
        "snr": snr,
        "total_bkg": total_bkg,
        "fluxes": flux,
        "aperture_radii": float(apertures_radii[0]),
        "annulus_radii": annulus_radii,
    }


def prepare_image(file, ref, metadata):
    """
    Detect sources, align, and centroid for a single image.

    Parameters
    ----------
    file : str or Path
        Path to the FITS file.
    ref : ReferenceData
        Reference image data (sky coords, Gaia RA/Decs, CNN model).
    metadata : dict
        Metadata dictionary (must include 'largest_usable_adu_value').

    Returns
    -------
    ImageData or None
        Per-image results, or None if too few stars were detected.
    """
    calibrated_data, coords, fwhm, _ = calibration_sequence(
        file, threshold=THRESH, max_adu=metadata["largest_usable_adu_value"],
    )

    if len(coords) < N_STARS_ALIGN:
        return None

    centroid_coords, aligned_coords, this_wcs = align_and_centroid(
        calibrated_data, coords, ref,
    )

    header = fits.open(file)[0].header

    return ImageData(
        calibrated_data=calibrated_data,
        coords=coords,
        fwhm=fwhm,
        centroid_coords=centroid_coords,
        aligned_coords=aligned_coords,
        wcs=this_wcs,
        header=header,
    )


def build_photometry_table(img, metadata, mask):
    """
    Run photometry with a given mask and build an output table.

    Parameters
    ----------
    img : ImageData
        Per-image detection/alignment results.
    metadata : dict
        Metadata dictionary (must include 'egain').
    mask : numpy.ndarray or None
        Bayer mask to apply to the image data.

    Returns
    -------
    Table
        Photometry table for this image and mask.
    """
    phot = measure_photometry(
        img.calibrated_data, img.centroid_coords, img.aligned_coords,
        img.fwhm, metadata["egain"], mask,
    )
    centroid_ra_dec = img.wcs.pixel_to_world(
        img.centroid_coords[..., 0],
        img.centroid_coords[..., 1],
    )

    data = Table()
    data["tot_count"] = phot["tot_count"]
    data["total_bkg"] = phot["total_bkg"]
    data["bkgd_count"] = phot["bkgd_count"]
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
    data.meta["fwhm"] = float(img.fwhm)
    data.meta["aperture_radii"] = phot["aperture_radii"]
    data.meta["annulus_radii"] = phot["annulus_radii"]

    return data
