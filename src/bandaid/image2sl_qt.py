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

import numpy as np


def generate_bayer_masks(shape, metadata):
        """
        Generate mask for each color in a Bayer array.

        Parameters
        ----------
        shape : tuple
            The image data array
        metadata : dict
            The image metadata dictionary
        """
        pattern = metadata['bayerpat']

        # now re-jumble based on roworder and ybaryoff
        if metadata['roworder'] == 'bottom-up':
            pattern = pattern[2:3] + pattern[0:1]
        if metadata['ybayroff'] != 0:
            pattern = pattern[1] + pattern[0] + pattern[3] + pattern[2]

        img_slice = {}
        img_slice[0] = (0, 0)
        img_slice[1] = (0, 1)
        img_slice[2] = (1, 0)
        img_slice[3] = (1, 1)

        bayer_info = [] # list of tuples (filter, img_mask)

        for color in ['R', 'B', 'G']:
            # In the mask, True means masked/ignore; False means yes/use/valid
            img_mask = np.ones(shape, dtype=bool)
            for channel in range(4):
                if pattern[channel] == color:
                    slicer = img_slice[channel]
                    img_mask[slicer[0]::2, slicer[1]::2] = False

            bayer_info.append(('T' + color, img_mask))
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
    temp1 = image[0::2,0::2]
    temp2 = image[0::2,1::2]
    temp3 = image[1::2,0::2]
    temp4 = image[1::2,1::2]

    def print_img_stats(data):
        data = data
        print("        Min = ", min(data),
                ", Max = ", max(data),
                ", Avg = ", statistics.mean(data),
                ", Stdev = ", statistics.stdev(data))

    raw_avg = image.mean()
    raw_stdev = image.std()
    cutoff = raw_avg + 5*raw_stdev

    ## sample the image into tempnx, to be used to generate statistics
    temp1x = temp1[(0 <= temp1) &  (temp1 < cutoff)]
    temp2x = temp2[(0 <= temp2) &  (temp2 < cutoff)]
    temp3x = temp3[(0 <= temp3) &  (temp3 < cutoff)]
    temp4x = temp4[(0 <= temp4) &  (temp4 < cutoff)]

    print(" -------------- balancing ---------------")
    tempexes = [temp1x, temp2x, temp3x, temp4x]
    # adjust everything based on the overall mean
    target_stdev = np.mean([temp.std() for temp in tempexes])
    # The temp1x, etc arrays are not all the same size, so just
    # sum it all and divide by the number of items.
    summed_temps = sum(temp.sum() for temp in tempexes)
    summed_numbers = sum(len(temp) for temp in tempexes)
    target_mean = summed_temps / summed_numbers

    print("Overall mean is ", target_mean, ", overall stdev = ", target_stdev)

    m = temp1x.std()
    factor = target_stdev/m
    temp1 = temp1 * factor
    temp1x = temp1x * factor
    m = temp1x.mean()
    temp1 = temp1 - (m-target_mean)
    print("Bayer 1 factor = ", factor)

    m = temp2x.std()
    factor = target_stdev/m
    temp2 = temp2 * factor
    temp2x = temp2x * factor
    m = temp2x.mean()
    temp2 = temp2 - (m-target_mean)
    print("Bayer 2 factor = ", factor)

    m = temp3x.std()
    factor = target_stdev/m
    temp3 = temp3 * factor
    temp3x = temp3x * factor
    m = temp3x.mean()
    temp3 = temp3 - (m-target_mean)
    print("Bayer 3 factor = ", factor)

    m = temp4x.std()
    factor = target_stdev/m
    temp4 = temp4 * factor
    temp4x = temp4x * factor
    m = temp4x.mean()
    temp4 = temp4 - (m-target_mean)
    print("Bayer 4 factor = ", factor)

    image[0::2,0::2] = temp1
    image[0::2,1::2] = temp2
    image[1::2,0::2] = temp3
    image[1::2,1::2] = temp4


# def remove_background(image, metadata, do_color_balance=False):
#     """
#     Calculate and remove the background from an image

#     Parameters
#     ----------
#     image : np.ndarray
#         The image. The background will be estimated and then
#         subtracted from each pixel
#     do_color_balance : bool
#         If True, the four pixel color channels will be adjusted
#         with a linear transformation to achieve a flat gray
#         background that has the same noise level in each color
#         channel.
#     """
#     egain = self.metadata['egain']

#     if do_color_balance:
#         bayer_balance_image(image)
#     (self.bkgd_mean,
#         median,
#         self.std) = sigma_clipped_stats(image, sigma=3.0)
#     sigma_clip = SigmaClip(sigma=3.0)
#     bkg_estimator = MedianBackground()
#     full_background = Background2D(image,
#                                     (int(self.width/8),int(self.height/8)),
#                                     filter_size=(3,3),
#                                     exclude_percentile=80,
#                                     sigma_clip=sigma_clip,
#                                     bkg_estimator=bkg_estimator)
#     background = full_background.background
#     image -= background
#     (self.bkgd_mean,
#         _dummy,
#         bkgd_std) = sigma_clipped_stats(background, sigma=3.0)
#     print("background.median = ", full_background.background_median,
#             ", background.rms = ", full_background.background_rms_median)
#     self.noise_bkgd_per_pixel = full_background.background_rms_median * egain
#     self.background = background
