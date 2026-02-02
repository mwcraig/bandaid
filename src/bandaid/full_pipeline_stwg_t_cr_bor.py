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
from eloy import alignment, centroid, detection, photometry, psf, utils
from eloy.centroid import Ballet
from st_pipeline.schema_definition import StarList, StarListSet
from tqdm.auto import tqdm
from twirl import compute_wcs, gaia_radecs
from twirl.geometry import sparsify

from . import generate_bayer_masks

CUTOUT = 500 # 120

N_STARS_ALIGN = 15
THRESH = 0.5

# Relative radii and annulus are defined here. These radii are multiplied by
# each image's FWHM to determine the actual aperture sizes.

# Only need one radius for STWG, but it needs to be in an iterable
RELATIVE_RADII = [1.0]  # np.linspace(0.1, 5, 30)
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


files = sorted(Path("photometry_raw_data_t_cr_bor").glob("*.fit"))

# ## Reference Selection and Calibration
#
# Next, a reference image is selected for further processing.
# The middle image from the observation night is chosen as the reference.

images = np.array(files)

reference_image = images[len(images) // 2]

ref_data, ref_coords, ref_fwhm, _ = calibration_sequence(reference_image, threshold=THRESH)
ref_reference = alignment.twirl_reference(ref_coords[0:N_STARS_ALIGN])

# Create starlist metadata from input json and FITS header of the reference image
ref_header = fits.getheader(reference_image)

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


metadata     = metadata_from_header(ref_header)
# _ = logger.info(f"Reference FWHM: {ref_fwhm:.2f} pixels")

# size of the field-of-view in degrees -- only used to query Gaia
fov = metadata["fov_rad"]
# That is T CrB below
center = SkyCoord.from_name(metadata["object"])

# The WCS is computed using the twirl package.

# Get Gaia coordinates for this fov
all_radecs = gaia_radecs(
    center,
    1.5 * fov,
)

# we only keep stars 0.01 degree apart from each other
all_radecs = sparsify(all_radecs, 0.01)


# we only use the n brightest stars from Gaia -- WHY NOT N_STARS_ALIGN???
wcs, _ = compute_wcs(ref_coords[0:15], all_radecs[0:15], tolerance=1)

# Convert eloy table to starlist format
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
    good = ~np.isnan(eloy_table["tot_count"])
    return StarList.from_table(eloy_table[good], metadata=metadata)


@dataclass
class ReferenceData:
    """Reference image data used to process each science image."""

    coords: np.ndarray
    wcs: object
    radecs: np.ndarray
    cnn: Ballet


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
    aligned_coords = this_wcs.world_to_pixel(
        ref.wcs.pixel_to_world(ref.coords[..., 0], ref.coords[..., 1]),
    )
    aligned_coords = np.array(aligned_coords).T
    centroid_coords = centroid.ballet_centroid(calibrated_data, aligned_coords, ref.cnn)
    return centroid_coords, aligned_coords, this_wcs


def measure_photometry(calibrated_data, centroid_coords, aligned_coords, fwhm, egain):
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

    Returns
    -------
    dict
        Keys: tot_count, count_err, bkgd_count, peak_count, snr,
        total_bkg, fluxes, aperture_radii, annulus_radii.
    """
    apertures_radii = RELATIVE_RADII * fwhm
    flux = photometry.aperture_photometry(
        calibrated_data, centroid_coords, apertures_radii,
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


def process_image(file, ref, metadata):
    """
    Process a single image: detect, align, centroid, measure photometry.

    Parameters
    ----------
    file : str or Path
        Path to the FITS file.
    ref : ReferenceData
        Reference image data (coords, WCS, Gaia RA/Decs, CNN model).
    metadata : dict
        Metadata dictionary (must include 'egain').

    Returns
    -------
    Table or None
        Photometry table for this image, or None if the image was skipped.
    """
    calibrated_data, coords, fwhm, _ = calibration_sequence(
        file, threshold=THRESH,
    )

    if len(coords) < N_STARS_ALIGN:
        return None

    centroid_coords, aligned_coords, this_wcs = align_and_centroid(
        calibrated_data, coords, ref,
    )

    phot = measure_photometry(
        calibrated_data, centroid_coords, aligned_coords, fwhm, metadata["egain"],
    )

    centroid_ra_dec = this_wcs.pixel_to_world(
        centroid_coords[..., 0],
        centroid_coords[..., 1],
    )

    header = fits.open(file)[0].header

    data = Table()
    data["tot_count"] = phot["tot_count"]
    data["total_bkg"] = phot["total_bkg"]
    data["bkgd_count"] = phot["bkgd_count"]
    data["count_err"] = phot["count_err"]
    data["snr"] = phot["snr"]
    data["fluxes"] = phot["fluxes"]
    data["time"] = Time(parser.parse(header["DATE-OBS"])).jd
    data["sky"] = np.mean(
        phot["total_bkg"] / (np.pi * (RELATIVE_RADII * fwhm) ** 2),
    )
    data["airmass"] = header.get("AIRMASS", np.nan)
    data["peak_count"] = phot["peak_count"]
    data["stars_in_exp"] = len(coords)
    data["ra"] = centroid_ra_dec.ra.degree
    data["dec"] = centroid_ra_dec.dec.degree
    data["x"] = centroid_coords[..., 0]
    data["y"] = centroid_coords[..., 1]
    data.meta["fwhm"] = float(fwhm)
    data.meta["aperture_radii"] = phot["aperture_radii"]
    data.meta["annulus_radii"] = phot["annulus_radii"]

    return data


# ## Photometry

# NOTE -- this triggers a download from HuggingFace the first time it is run.
ref = ReferenceData(coords=ref_coords, wcs=wcs, radecs=all_radecs, cnn=Ballet())
bayer_masks = generate_bayer_masks(
    (metadata["height"], metadata["width"]),
    metadata,
)
bayer_masks["L4"] = np.ones(
    (metadata["height"], metadata["width"]),
    dtype=bool,
)

star_lists = []
for file in tqdm(images):

    for filter_name, mask in bayer_masks:
        metadata["filter"] = filter_name
        # Could save mask for later use if needed
        data = process_image(file, ref, metadata, mask=mask)
    if data is not None:
        star_lists.append(eloy_to_starlist(data, metadata))
