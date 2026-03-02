from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from eloy import alignment
from eloy.centroid import Ballet
from st_pipeline.schema_definition import StarListSet
from tqdm.auto import tqdm
from twirl import compute_wcs, gaia_radecs
from twirl.geometry import sparsify

from . import (
    ReferenceData,
    build_photometry_table,
    calibration_sequence,
    eloy_to_starlist,
    generate_bayer_masks,
    metadata_from_header,
    prepare_image,
)
from .photometry import N_STARS_ALIGN, THRESH

# ## Reference Selection and Calibration
#
# Next, a reference image is selected for further processing.
# The middle image from the observation night is chosen as the reference.

files = sorted(Path("photometry_raw_data_t_cr_bor").glob("*.fit"))
images = np.array(files)

reference_image = images[len(images) // 2]

# Create starlist metadata from input json and FITS header of the reference image
ref_header = fits.getheader(reference_image)
metadata = metadata_from_header(ref_header)

# Get the sky coordinates for the stars on which to perform photometry from the
# reference image. This is done using the twirl package for astrometric calibration
# and eloy for source detection
_, ref_img_coords_xy, _, _ = calibration_sequence(
    reference_image, threshold=THRESH, max_adu=metadata["largest_usable_adu_value"],
)

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
wcs = compute_wcs(ref_img_coords_xy[0:15], all_radecs[0:15], tolerance=1)

# ## Photometry

# NOTE -- this triggers a download from HuggingFace the first time it is run.
ref = ReferenceData.from_pixel_coords(ref_img_coords_xy, wcs, all_radecs, Ballet())

bayer_masks = generate_bayer_masks(
    (metadata["height"], metadata["width"]),
    metadata,
)
bayer_masks.insert(0, ("L4", None))

output_dir = Path(files[0]).parent.parent / (Path(files[0]).parent.name + "_star")
output_dir.mkdir(exist_ok=True)

for file in tqdm(images):
    img = prepare_image(file, ref, metadata)
    if img is None:
        continue

    image_star_lists = []
    for filter_name, mask in bayer_masks:
        metadata["filter"] = filter_name
        data = build_photometry_table(img, metadata, mask)
        image_star_lists.append(eloy_to_starlist(data, metadata))

    star_list_set = StarListSet(star_lists=image_star_lists)
    output_path = output_dir / Path(file).with_suffix(".star").name
    output_path.write_text(star_list_set.model_dump_json(indent=2))
