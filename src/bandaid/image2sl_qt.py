# MIT License
#
# Copyright (c) 2025 AAVSO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Bayer (CFA) utilities for Smart Telescope images.

Builds per-color Bayer masks from the image metadata and balances the four CFA
channels of a raw frame so that downstream detection and centroiding see a
checkerboard-free image.
"""

import numpy as np


def generate_bayer_masks(shape, metadata, *, append_l4=False):
    """
    Generate mask for each color in a Bayer array.

    Parameters
    ----------
    shape : tuple
        The (ny, nx) shape of the image data array.
    metadata : dict
        The image metadata dictionary
    append_l4 : bool, optional
        If True, add an "L4" key mapped to None to the returned dict. The None
        mask signals a full-frame (unmasked) luminance channel. Default False.

    Returns
    -------
    dict
        A dictionary mapping each color filter name to its corresponding mask.
        For the optional "L4" luminance channel the value is None, signalling a
        full-frame (unmasked) channel.
    """
    pattern = metadata["bayerpat"]

    # Re-anchor the 2x2 pattern on the stored array. Per the standard FITS
    # convention (N.I.N.A./SGP writers, Siril/ASTAP readers) the CFA color of
    # image pixel (x, y) is
    #
    #     pattern[(y + YBAYROFF) % 2][(x + XBAYROFF) % 2]
    #
    # so an odd YBAYROFF swaps the pattern's top and bottom rows -- the same
    # transform as a bottom-up roworder, and the two cancel when both apply --
    # while an odd XBAYROFF swaps the columns within each row.
    if metadata["roworder"] == "bottom-up":
        pattern = pattern[2:4] + pattern[0:2]
    if metadata["ybayroff"] % 2 != 0:
        pattern = pattern[2:4] + pattern[0:2]
    if metadata.get("xbayroff", 0) % 2 != 0:
        pattern = pattern[1] + pattern[0] + pattern[3] + pattern[2]

    img_slice = {}
    img_slice[0] = (0, 0)
    img_slice[1] = (0, 1)
    img_slice[2] = (1, 0)
    img_slice[3] = (1, 1)

    bayer_info = {}  # maps filter -> img_mask, in insertion order

    for color in ["R", "B", "G"]:
        # In the mask, True means masked/ignore; False means yes/use/valid
        img_mask = np.ones(shape, dtype=bool)
        for channel in range(4):
            if pattern[channel] == color:
                slicer = img_slice[channel]
                img_mask[slicer[0] :: 2, slicer[1] :: 2] = False

        bayer_info["T" + color] = img_mask
    if append_l4:
        bayer_info["L4"] = None

    return bayer_info


def bayer_balance_image(image):
    """
    Adjust pixel values per the Bayer pattern.

    Perform a linear adjustment to
    each pixel value according to its position in the Bayer
    pattern. The adjustment factors are calculated so as to give the
    new file the same background grayness and the same background
    noise in each of the four Bayer pattern pixels.

    The image itself is changed by this operation.

    Parameters
    ----------
    image : np.ndarray
        The image to be modified
    """
    # Drop the casts to float -- this has already been done
    temp1 = image[0::2, 0::2]
    temp2 = image[0::2, 1::2]
    temp3 = image[1::2, 0::2]
    temp4 = image[1::2, 1::2]

    raw_avg = image.mean()
    raw_stdev = image.std()
    cutoff = raw_avg + 5 * raw_stdev

    ## sample the image into tempnx, to be used to generate statistics
    temp1x = temp1[(temp1 >= 0) & (temp1 < cutoff)]
    temp2x = temp2[(temp2 >= 0) & (temp2 < cutoff)]
    temp3x = temp3[(temp3 >= 0) & (temp3 < cutoff)]
    temp4x = temp4[(temp4 >= 0) & (temp4 < cutoff)]

    tempexes = [temp1x, temp2x, temp3x, temp4x]
    # adjust everything based on the overall mean
    target_stdev = np.mean([temp.std() for temp in tempexes])
    # The temp1x, etc arrays are not all the same size, so just
    # sum it all and divide by the number of items.
    summed_temps = sum(temp.sum() for temp in tempexes)
    summed_numbers = sum(len(temp) for temp in tempexes)
    target_mean = summed_temps / summed_numbers

    # Rescale each CFA sub-grid to the common stdev, then shift its rescaled mean
    # onto the common mean, so all four channels end up with the same background
    # grayness and noise. The four sub-grids are disjoint pixels, so the order and
    # the write-back are independent.
    slices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (row, col), tempx in zip(slices, tempexes, strict=True):
        factor = target_stdev / tempx.std()
        scaled = image[row::2, col::2] * factor
        image[row::2, col::2] = scaled - ((tempx * factor).mean() - target_mean)
