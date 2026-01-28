from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from dateutil import parser
from eloy import alignment, centroid, detection, photometry, psf, utils
from eloy.centroid import Ballet
from tqdm.auto import tqdm
from twirl import compute_wcs, gaia_radecs
from twirl.geometry import sparsify

SATURATED = 40000
CUTOUT = 500 # 120

N_STARS_ALIGN = 15
THRESH = 0.5

# Relative radii and annulus are defined here. These radii are multiplied by
# each image's FWHM to determine the actual aperture sizes.
RELATIVE_RADII = np.linspace(0.1, 5, 30)
ANNULUS = (5, 8)

# Max number of stars to use for photometry
N_STARS = 200
# Size of cutout for centroiding
CUTOUT_SHAPE = (21, 21)

EGAIN = 0.3116
YBAYROFF =  0

# Get the files
def observation_time(file):
    date_str = fits.getheader(file)["DATE-OBS"]
    return parser.parse(date_str)


def calibration_sequence(file: str, threshold: float = 1) -> tuple:
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

    else:  # noqa: RET505
        region_coords = np.array([(r.centroid[1], r.centroid[0]) for r in regions])
        cutouts = utils.cutout(calibrated_data, region_coords, (50, 50))

        # Drop any cutouts that are saturated -- NOTE THAT THIS LEAVES BEHIND SATURATED REGIONS
        cutouts = np.array(list(filter(lambda data: np.max(data) < SATURATED, cutouts)))

        # Drop any regions that are saturdated for calculating the FWHM
        cutouts_normalized = cutouts / np.nanmax(cutouts, (1, 2))[:, None, None]

        # Average the cutouts...yolo I guess on whether these are good detections
        epsf = np.nanmedian(cutouts_normalized, 0)

        # Note fitting is only done to the normalized cutout
        psf_params = psf.fit_gaussian(epsf)
        fwhm = psf.gaussian_sigma_to_fwhm * np.mean(
            [psf_params["sigma_x"], psf_params["sigma_y"]]
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

# _ = logger.info(f"Reference FWHM: {ref_fwhm:.2f} pixels")

# known pixel size in degrees ---- THIS IS ONLY USED TO FIND THE FOV
pixel_scale = 2.37 / 3600
# size of the field-of-view -- only used to query Gaia
fov = max(ref_data.shape) * pixel_scale
# RA/Dec coordinates of the image
ref_header = fits.getheader(reference_image)
centero = SkyCoord(ref_header["RA"], ref_header["DEC"], unit=("deg", "deg"))
# That is T CrB below
center = SkyCoord.from_name("T CrB")

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


# ## Photometry
# The photometry step follows the approach described in the [photometry tutorial](), with additional comments for clarity
# In the pipeline we also added logging information but keep it commented not to overcrowded this tutorial page. In practice, these logging info are very useful to check the pipeline progress and debug any issue.

data = defaultdict(list)

# NOTE -- this triggers a download from HuggingFace the first time it is run. We ought to be able to cache it somewhere
cnn = Ballet()

# logger.info("Starting full reduction")

# NEXT BREAK THIS INTO FUNCTIONS!!!
for i, file in enumerate(tqdm(images)):
    filename = Path(file).name
    # logger.info(f"Processing {filename} ({i + 1}/{len(images)})")

    # calibration and FWHM
    calibrated_data, coords, fwhm, regions = calibration_sequence(file, threshold=THRESH)
    # if calibrated_data is None:
    #     print("skipping, fewer than 3 star")
    #     continue
    # logger.info(f"{len(coords)} stars detected")
    #logger.info(f"FWHM: {fwhm:.2f} pixels")

    # skip images with too few stars
    if len(coords) < N_STARS_ALIGN:
        # logger.warning(f"{filename} discarded")
        continue
    # we only use the n brightest stars from Gaia
    this_wcs = compute_wcs(coords[0:N_STARS_ALIGN], all_radecs[0:N_STARS_ALIGN], tolerance=1)

    # KEEP THIS -- it uses the wcs we have calculated to get approximate pixel coordinates
    aligned_coords = this_wcs.world_to_pixel(wcs.pixel_to_world(ref_coords[:N_STARS, 0], ref_coords[:N_STARS, 1]))
    aligned_coords = np.array(aligned_coords).T
    dx, dy = np.median(ref_coords[0:N_STARS] - aligned_coords, 0)
    # logger.info(f"(X,Y) shift: ({dx:.2f}, {dy:.2f}) pixels")

    # centroiding
    centroid_coords = centroid.ballet_centroid(calibrated_data, aligned_coords, cnn)
    # aperture photometry -- PHOTOMETRY STARTS HERE -- need to look at eloy source to
    # see how it gets done so fast...HMMM, they just call photutils.aperture_photometry
    apertures_radii = RELATIVE_RADII * fwhm
    # This flux is the sum of the aperture counts within each radius
    flux = photometry.aperture_photometry(
        calibrated_data, centroid_coords, apertures_radii,
    )
    # annulus background correction -- IS THIS RIGHT? This leaves no gap
    # for the largest radius
    annulus_radii = np.max([np.max(apertures_radii), ANNULUS[0] * fwhm]), ANNULUS[1] * fwhm
    aperture_area = np.pi * apertures_radii**2

    # This is background per pixel
    bkg = photometry.annulus_sigma_clip_median(
        calibrated_data, centroid_coords, *annulus_radii,
    )
    # This bkg is TOTAL, not per pixel
    total_bkg = bkg[:, None] * aperture_area[None, :]

    # peaks
    peaks = np.nanmax(
        utils.cutout(calibrated_data, aligned_coords, (25, 25)),
        axis=(1, 2),
    )

    net_count = flux - total_bkg
    # noise_bkgd_per_pixel in units of e-/pixel
    noise_bkgd_per_pixel = bkg * EGAIN
    tot_noise_bkgd = noise_bkgd_per_pixel[:, None] * aperture_area[None, :]
    # Calculate errors using table columns and star flux error in column
    poiss_noise = np.sqrt(EGAIN * net_count)
    tot_noise = np.sqrt(poiss_noise**2 + tot_noise_bkgd**2) / EGAIN

    snr = net_count / tot_noise

    # getting data
    header = fits.open(file)[0].header
    data["net_count"].append(net_count)
    data["total_bkg"].append(total_bkg)
    data["bkg_per_pix"].append(bkg)
    data["snr"].append(snr)
    data["fluxes"].append(flux)
    data["fwhm"].append(fwhm)
    data["time"].append(Time(parser.parse(header["DATE-OBS"])).jd)
    data["dx"].append(dx)
    data["dy"].append(dy)
    data["sky"].append(np.mean(total_bkg / aperture_area[None, :]))
    data["airmass"].append(header.get("AIRMASS", np.nan))
    data["peak"].append(peaks)
    data["stars_in_exp"].append(len(coords))
    data["aperture_radii"].append(apertures_radii)
    data["annulus_radii"].append(annulus_radii)


for k, v in data.items():
    data[k] = np.array(v)
