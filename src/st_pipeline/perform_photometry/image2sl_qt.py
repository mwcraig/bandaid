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

import argparse
import json
import math
import os
import platform
import shutil
import statistics
import sys
import tempfile
import warnings
from collections import namedtuple
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import numpy as np
import pytz
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
from astroquery.astrometry_net import AstrometryNet
from photutils import aperture, psf
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from pydantic import BaseModel, ConfigDict
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QFile, QIODevice, QSettings
from PySide6.QtGui import QGuiApplication
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QRadioButton,
    QVBoxLayout,
)
from timezonefinder import TimezoneFinder

from .. import __version__
from ..schema_definition import StarItem, StarList, StarListSet
from . import psf_fitting

warnings.filterwarnings('error', category=RuntimeWarning)

astrometry_api_key = None

################################################################
##        Algorithmic Stuff Comes First
################################################################

def de_bayer_file(filename, metadata, temp_dir):
    """Split an RGB image into four images, one for each Bayer channel
    The four channels are extracted using a string description of the
    Bayer sequence (e.g., 'BGGR'); each extracted sub-image is stored
    as a new FITS image file.
    Parameters
    ----------
    filename : str
        pathname to the original image to be de-Bayered
    metadata: dict
        The metadata for this image file
    temp_dir : str
        pathname to the temporary directory where the new images go

    Returns
    -------
    List of named tuples (will always be list of exactly 4 elements)
        The named tuples are created from namedtuple('filter','filename')
    filter : str
        The name of the Bayer filter color (e.g., 'TG1' or 'TB')
    filename : str
        The name of the file with that channel's image

    """
    with fits.open(filename) as hdul:
        temp1 = hdul[0].data[0::2,0::2]
        temp2 = hdul[0].data[0::2,1::2]
        temp3 = hdul[0].data[1::2,0::2]
        temp4 = hdul[0].data[1::2,1::2]

        array = [temp1, temp2, temp3, temp4] # roworder= top-down, ybayroff= 0
        if metadata['roworder'] == 'bottom-up':
            array = [array[i-1] for i in [3,4,1,2]]
        if metadata['ybayroff'] != 0: # left or right shift the same
            array = [array[i-1] for i in [2,1,4,3]]

        output_filenames = [] # each entry in this list is a tuple: (filter, filename)

        for index in range(4):
            color = metadata['bayerpat'][index]
            output_tgt = Path(temp_dir) / ("image"+str(index)+"_"+color+".fits")

            hdu = fits.PrimaryHDU()
            # push keywords in from the original file
            for keyword in hdul[0].header:
                if keyword not in ('COMMENT', 'HISTORY'):
                    value = hdul[0].header[keyword]
                    comment = hdul[0].header.comments[keyword]
                    hdu.header[keyword] = (value,comment)
                else: # Yes, this is a comment/history
                    comment = hdul[0].header[keyword]
                    for card in comment:
                        hdu.header[keyword] = card

            hdu.data = array[index]
            # fix up header to match the data
            hdu.header['FILTER'] = ('T'+color, 'Bayer color mask')
            hdu.header['BAYERPAT'] = ('NA', 'No longer a Bayered image')
            hdu.header['PIXSCALE'] = (metadata['pixscale'] * 2.0, 'arcsec/pix')
            hdu.header['NAXIS1'] = int(hdu.header['NAXIS1'] / 2)
            hdu.header['NAXIS2'] = int(hdu.header['NAXIS2'] / 2)

            # modify the wcs to reflect the new pixel scale
            if 'CD1_1' in hdu.header: hdu.header['CD1_1'] *= 2.0
            if 'CD1_2' in hdu.header: hdu.header['CD1_2'] *= 2.0
            if 'CD2_1' in hdu.header: hdu.header['CD2_1'] *= 2.0
            if 'CD2_2' in hdu.header: hdu.header['CD2_2'] *= 2.0
            if 'CRPIX_1' in hdu.header: hdu.header['CRPIX_1'] /= 2.0
            if 'CRPIX_2' in hdu.header: hdu.header['CRPIX_2'] /= 2.0
            if 'CDELT_1' in hdu.header: hdu.header['CDELT_1'] *= 2.0
            if 'CDELT_2' in hdu.header: hdu.header['CDELT_2'] *= 2.0
            hdu.header['CTYPE1'] = 'RA---TAN'
            hdu.header['CTYPE2'] = 'DEC--TAN'
            hdu.header['CTYPE2'] = hdu.header['CTYPE2'].replace('-SIP', '')
            # update_header will "fix" the header to match the data
            hdu.update_header()
            fits.writeto(output_tgt, array[index], header=hdu.header, overwrite=True)
            print("write debayered image: ", output_tgt) # so you can look at the image files
            ImageDescriptor = namedtuple('ImageDescriptor', ['filter','filename'])
            output_filenames.append(ImageDescriptor(color,output_tgt))

        # change the metadata to reflect the new state of the image
        metadata['pixscale'] = metadata['pixscale'] * 2.0
        metadata['bayerpat'] = 'NA'
        metadata['telescope_probe']= (metadata['telescope_probe'][0], 'mono')

        return output_filenames

# pattern triplet: (x_offset, y_offset, weight)
interp_pattern = [
    [ (0,0,9), (0,1,3),  (1,0,3),  (1,1,1)   ], # color 0
    [ (0,0,9), (-1,0,3), (0,1,3),  (-1,1,1)  ], # color 1
    [ (0,0,9), (1,0,3),  (0,-1,3), (1,-1,1)  ], # color 2
    [ (0,0,9), (-1,0,3), (0,-1,3), (-1,-1,1) ]  # color 3
]

def stack_images(channel_list, options, temp_dir):
    """ Create a stacked image from 4 individual Bayer sub-images

    Four Bayer sub-images are stacked into a single image. If
    `options` includes the interpolate_channels flag, the sub-images
    will be shifted as they are added, recognizing the offsets between
    the locations of the different Bayer colors. The resulting sum
    image will be stored as a new FITS file with floating point pixel
    values (to avoid overflow issues as the four pixel values are
    added). The resulting FITS file will have all the FITS
    keyword/value pairs found in the first sub-image, except that the
    FILTER keyword will be set to 'CV'.

    Parameters
    ----------
    channel_list : list of named tuples (taken from de_bayer_file)
        Each tuple is a (filtername, filename) pair specifying the
        input files.
    options : OptionBox
        Object of class OptionBox, describing options the user has
        selected
    temp_dir : str
        pathname to the temporary directory where the new images go

    Returns
    -------
    str
        pathname to the new FITS file holding the stacked image
    """
    output_tgt = Path(temp_dir) /  "image_S.fits"
    hdu = fits.PrimaryHDU()
    # push keywords in from the original file(s)
    filename = channel_list[0].filename
    with fits.open(filename) as hdul:
        (height,width) = np.shape(hdul[0].data)
        for keyword in hdul[0].header:
            if keyword not in ('COMMENT', 'HISTORY'):
                value = hdul[0].header[keyword]
                comment = hdul[0].header.comments[keyword]
                hdu.header[keyword] = (value,comment)
            else: # Yes, this is a comment/history
                comment = hdul[0].header[keyword]
                for card in comment:
                    hdu.header[keyword] = card
    hdu.data = np.zeros((height,width),dtype=np.float32)
    for (bayer_id,(_, channel)) in enumerate(channel_list):
        with fits.open(channel) as hdul:
            source_hdu = hdul[0].data
            if options.interpolate_channels:
                new_data = np.zeros(np.shape(hdul[0].data),dtype=np.float32)
                orig_data = source_hdu.astype(np.float32)
                source_hdu = new_data
                for y in range(height-1):
                    for x in range(width-1):
                        tgt = sum(p[2]*orig_data[y+p[1],x+p[0]]
                                   for p in interp_pattern[bayer_id])
                        new_data[y,x] = tgt

            hdu.data += source_hdu/16.0
    hdu.header['filter'] = 'CV'
    hdu.update_header()
    fits.writeto(output_tgt, hdu.data, header=hdu.header, overwrite=True)
    return output_tgt

def duplicate_file_with_new_image(hdul, new_data, new_filter, new_pathname):
    """ Copy a FITS file, replacing the pixel data with new pixel data

    Create a copy of a FITS file, replacing the original pixels with a
    new set of pixels (possibly with a different shape). FITS keywords
    will be copied over, except for those keywords affected by a shape
    change or a possible pixel numeric type change (e.g., int to float).

    Parameters
    ----------
    hdul : list of HDUs as provided by fits.open()
        The HDU list conveying the contents of the FITS file to be copied.
    new_data : numpy 2D array of pixel values
        The new pixel values (data type will be preserved into the new file)
    new_filter : str
        The name of the filter to be put into the new FITS header
    new_pathname : str
        The pathname of the file to be created

    Returns
    -------
    None

    """
    hdu = fits.PrimaryHDU()
    # push keywords in from the original file
    for keyword in hdul[0].header:
        if keyword not in ('COMMENT', 'HISTORY'):
            value = hdul[0].header[keyword]
            comment = hdul[0].header.comments[keyword]
            hdu.header[keyword] = (value,comment)
        else: # Yes, this is a comment/history
            comment = hdul[0].header[keyword]
            for card in comment:
                hdu.header[keyword] = card

    hdu.data = new_data
    hdu.header['filter'] = new_filter
    # update_header will "fix" the header to match the data
    hdu.update_header()
    fits.writeto(new_pathname, new_data, header=hdu.header, overwrite=True)

def bayer_balance_file(filename):
    """Duplicate an image file while adjusting pixel values per the Bayer pattern

    Duplicate an existing FITS file, performing a linear adjustment to
    each pixel value according to its position in the Bayer
    pattern. The adjustment factors are calculated so as to give the
    new file the same background grayness and the same background
    noise in each of the four Bayer pattern pixels. The new file will
    have float32 pixels to avoid pixel overflow problems that result
    from the adjustments. The new file will have the same filename as
    the original file, but with '_M' appended to the filename 'stem'.

    Parameters
    ----------
    filename : str
        Pathname to the file to be duplicated

    Returns
    -------
    str
        The full pathname of the new file.
    """
    with fits.open(filename) as hdul:
        temp1 = hdul[0].data[0::2,0::2].astype(np.float32)
        temp2 = hdul[0].data[0::2,1::2].astype(np.float32)
        temp3 = hdul[0].data[1::2,0::2].astype(np.float32)
        temp4 = hdul[0].data[1::2,1::2].astype(np.float32)

        def flatten_slice(slice, target):
            m = statistics.stdev(slice.flatten())
            factor = (target/m)
            slice = slice * factor
            print("  new slice mean = ", statistics.mean(slice.flatten()))

        def print_img_stats(data):
            data = data.astype(np.float32)
            print("        Min = ",
                  min(data),
                  ", Max = ",
                  max(data),
                  ", Avg = ",
                  statistics.mean(data),
                  ", Stdev = ",
                  statistics.stdev(data))

        raw_pixels = hdul[0].data.flatten().astype(np.float32)
        raw_avg = statistics.mean(raw_pixels)
        raw_stdev = statistics.stdev(raw_pixels)
        cutoff = raw_avg + 5*raw_stdev

        temp1x = temp1[(0 <= temp1) &  (temp1 < cutoff)].flatten()
        temp2x = temp2[(0 <= temp2) &  (temp2 < cutoff)].flatten()
        temp3x = temp3[(0 <= temp3) &  (temp3 < cutoff)].flatten()
        temp4x = temp4[(0 <= temp4) &  (temp4 < cutoff)].flatten()

        #print_img_stats(temp1x.flatten())
        #print_img_stats(temp2x.flatten())
        #print_img_stats(temp3x.flatten())
        #print_img_stats(temp4x.flatten())
        print(" -------------- balancing ---------------")

        # adjust everything based on the overall mean
        target_stdev = statistics.mean([statistics.stdev(x) for x in [temp1x,
                                                                      temp2x,
                                                                      temp3x,
                                                                      temp4x]])
        target_mean = statistics.mean(list(temp1x)+list(temp2x)+list(temp3x)+list(temp4x))


        #target_stdev = statistics.stdev(hdul[0].data.flatten().astype(np.float32))
        #target_mean = statistics.mean(hdul[0].data.flatten().astype(np.float32))
        print("Overall mean is ", target_mean, ", overall stdev = ", target_stdev)

        m = statistics.stdev(temp1x.flatten())
        factor = (target_stdev/m)
        temp1 = temp1 * factor
        temp1x = temp1x * factor
        m = statistics.mean(temp1x.flatten())
        temp1 = temp1 - (m-target_mean)
        print("Bayer 1 factor = ", factor)

        m = statistics.stdev(temp2x.flatten())
        factor = (target_stdev/m)
        temp2 = temp2 * factor
        temp2x = temp2x * factor
        m = statistics.mean(temp2x.flatten())
        temp2 = temp2 - (m-target_mean)
        print("Bayer 2 factor = ", factor)

        m = statistics.stdev(temp3x.flatten())
        factor = (target_stdev/m)
        temp3 = temp3 * factor
        temp3x = temp3x * factor
        m = statistics.mean(temp3x.flatten())
        temp3 = temp3 - (m-target_mean)
        print("Bayer 3 factor = ", factor)

        m = statistics.stdev(temp4x.flatten())
        factor = (target_stdev/m)
        temp4 = temp4 * factor
        temp4x = temp4x * factor
        m = statistics.mean(temp4x.flatten())
        temp4 = temp4 - (m-target_mean)
        print("Bayer 4 factor = ", factor)

        #print_img_stats(temp1x.flatten())
        #print_img_stats(temp2x.flatten())
        #print_img_stats(temp3x.flatten())
        #print_img_stats(temp4x.flatten())

        new_data = hdul[0].data.astype(np.float32)
        new_data[0::2,0::2] = temp1
        new_data[0::2,1::2] = temp2
        new_data[1::2,0::2] = temp3
        new_data[1::2,1::2] = temp4

        new_filename = filename.replace(".fit","_M.fit")
        duplicate_file_with_new_image(hdul, new_data, "M", new_filename)
        return new_filename

class OutputObject(BaseModel):
    """This class represents one logical output (result) starlist

    One starlist file can contain multiple logical starlists. An
    OutputObject instance represents one of those logical
    starlists. If the "One Starlist Per File" option is selected, each
    instance of OutputObject will receive its own file. Otherwise,
    multiple OutputObject instances will be combined into a single
    file.

    Attributes
    ----------
    logical_starlist: a single StarListSet
        A StarListSet that results from processing of an image
    filename: Path
        The filename path that the file will be stored into if each
        starlist is given its own file; otherwise, this is ignored
    """
    filename: Path
    orig_image_path: Path
    logical_starlist: StarListSet

    def write(self):
        """write this logical starlist to its own file

        Create a file holding this (single) logical starlist

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        with open(self.filename, 'w', encoding='utf-8') as fp:
            json.dump(self.logical_starlist.model_dump(),
                      fp, indent=2)

def probe_file_for_type(filename):
    """Figure out what kind of a smart telescope created an image

    Examine the FITS header keywords to determine what kind of smart
    telescope created the image and what format the image is in.
    The format strings have the following meanings:
        "bayered" - The image matches the sensor with a superimposed
    Bayer color filter array. Pixels alternate color.
        "mono" - The image is a true monochrome image. If filtered,
    the filter applies equally to all pixels
        "3Dstacked" - The FITS image has NAXIS=3, with the first axis
    holding 3 distinct images, one in R, G, and B. We can't determine
    order of the three colors here, so you have to rely on some other
    source to know the color order.
        "3Hstacked" - The FITS file contains 3 HDUs, one each for an
    R, G,  and B image. This is currently unimplemented.

    Parameters
    ----------
    filename : str
        The pathname of the image to be examined

    Returns
    -------
    Tuple: (str, str)
        First str is one of the following: "Unistellar", "Seestar",
        "Origin", "Dwarf2", "Dwarf3", "other"
        Second str is one of the following: "bayered", "mono",
        "3Dstacked", "3Hstacked", "unknown"

    Raises
    ------
    ValueError
        Raised if unable to determine which smart telescope type
    """
    with fits.open(filename) as hdul:
        hdu0h = hdul[0].header

        ################################
        ## Unistellar test
        ################################
        if 'ORIGIN' in hdu0h and 'Unistellar' in hdu0h['ORIGIN']:
            return ("Unistellar", "bayered")

        ################################
        ## Seestar test
        ################################
        if 'CREATOR' in hdu0h and 'Seestar' in hdu0h['CREATOR']:
            if hdu0h['NAXIS'] == 3:
                return ("Seestar50", "3Dstacked")
            return ("Seestar50", "bayered")

        ################################
        ## Celestron Origin test
        ################################
        if 'CREATOR' in hdu0h and 'Origin' in hdu0h['CREATOR']:
            return ("Origin", "bayered")

        ################################
        ## DWARF
        ################################
        if 'TELESCOP' in hdu0h and 'DWARFIII' in hdu0h['TELESCOP']:
            return ("Dwarf3", ["bayered", "3Dstacked"][hdu0h['NAXIS'] == 3])
        if 'TELESCOP' in hdu0h and 'DWARFII' in hdu0h['TELESCOP']:
            return ("Dwarf2", ["bayered", "3Dstacked"][hdu0h['NAXIS'] == 3])

        ################################
        ## Unrecognized
        ################################
        return ("other", "unknown")

################################################################
##                METADATA
## Metadata is managed with a dictionary. The index is the metadata
## variable name (see below). The value is either None or a legitamate
## value (either a string, int, or float).
##
## A subset of this metadata can be put into an AAVSO starlist's
## metadata section.
##
## Valid metadata indices that go into a starlist are:
##    OBS_TIME - a string, e.g., '2024-08-24T12:34:65.2' (UTC)
##    site_lat - a float, the observer's latitude in degrees
##    site_lon - a float, the observer's longitude in degrees
##    site_elev - a float, the observer's GPS elevation
##    OBSERVER - a string, the AAVSO observer code
##    FILTER - a 2-character string, one of TG,TB,TR
##    BLOCK_FILTER - a string, typically "UV+IR"
##    EXPOSURE - a float, total exposure time in secs
##    TEL_MANUFAC - a string, name of the telescope manufacturer
##    TEL_MODEL - a string, the telescope's model name
##    TEL_FIRMWARE - a string, the firmware ID
##    ADC_DEPTH - an integer, bit depth of the camera ADC
##    largest_usable_adu_value - an integer, the ADU level where saturation starts
##
## Additional metadata indices:
##    BAYERPAT - a 4-character string (e.g., 'BGGR')
##    PIXSCALE - a float, pixel scale *after* debayering, arcsec/pix
##    DEC - a float, nominal declination of image center (deg)
##    RA - a float, nominal RA of image center (deg)
##    FOV_RAD - a float, nominal field of view radius (deg)
##    telescope_probe - a str, value returned by probe_file_for_type()
################################################################

valid_meta_keys = ['schema_version',
                   'obs_time', # a string, e.g., '2024-08-24T12:34:65.2' (UTC)
                   'site_lat', # a float, the observer's latitude in degrees
                   'site_lon', # a float, the observer's longitude in degrees
                   'site_elev', # a float, the observer's GPS elevation in meters
                   'observer', # a string, the AAVSO observer code
                   'filter', # a 2-character string, one of TG,TB,TR
                   'block_filter', # a string, typically "UV+IR" or "IRCUT" or other
                   'exposure', # a float, total exposure time in secs
                   'tel_manufac', # a string, name of the telescope manufacturer (e.g. "ZWO")
                   'tel_model', #a string, the telescope's model name (e.g. "DwarfIII")
                   'tel_firmware', # a string, the firmware ID (e.g. "v1.2.3")
                   'adc_depth', # an integer, bit depth of the camera ADC (e.g. 12)
                   'largest_usable_adu_value', # an integer, the ADU level where saturation starts
                   'system_gain', # a float, the gain of the camera system, e-/ADU
                   'bayerpat', # a 4-character string (e.g., 'BGGR')
                   'pixscale', # a float, pixel scale *after* debayering, arcsec/pix
                   'epoch', # a string, (e.g., "J2000")
                   'refframe', # a string, (e.g., "ICRS")
                   'dec', # a float, nominal declination of image center (deg)
                   'ra', # a float, nominal RA of image center (deg)
                   'fov_rad', # a float, nominal field of view radius (deg) (half the diagonal)
                   'telescope_probe', # a tuple, with type and format of the image
                   'roworder', # a string, bayerpat modifier. "top-down" or "bottom-up"
                   'ybayroff' # an integer, bayerpat modifier. Column shift horizontally, 0 or 1
        ]

def get_json_value(data, keys):
    # data is a dictionary that was read from the JSON file
    # keys can be a string with '.' separators
    value = None
    for key in keys.split('.'):
        value = data[key]
    if value is None:
        print(f"WARNING: JSON key '{keys}' not found in metadata")    
    return value

class MetaValidator:
    """Class that tests metadata to see what's missing

    Tool for testing metadata to see what's missing and print
    intelligent error messages to the console

    Attributes
    ----------
    optional: list of str
        List of keywords that are not required in the input
    final: dict of {str, value}
        Metadata dictionary after all metadata has settled down; this
        is the metadata set that will be validated.
    json: dict of {str, value}
        The metadata that was pulled from the metadata.JSON file
    fits: dict of {str, value}
        The metadata that was pulled from the FITS file. (They keyword
        value will be the *metadata* form of the keyword, *not* the
        FITS form of the keyword. (A conversion is done for some smart
        telescope models.)
    """
    def __init__(self):
        """Create an instance of a MetaValidator
        """
        self.clear()
        self.optional = ['schema_version', 'filter', 'tel_firmware']
        self.final = {}         # These are here to make lint quiet
        self.json = {}
        self.fits = {}

    def clear(self):
        """clear state of the validator in preparation for another cycle
        """
        self.final = {}
        self.json = {}
        self.fits = {}

    def add_json_item(self, key, value, meta_dict):
        """Add a piece of metadata pulled from the JSON file

        Parameters
        ----------
        key: str
            The metadata keyword that was read
        value: any value
            The value that was read
        return: the value that belongs to this key, resolving any references,
            or None if the key is a comment
        """
        if key.startswith('_'): # json comment
            return None # skip
        if isinstance(value, str) and value.startswith('@'):
            # This is a reference to another key in the existing meta dir file
            self.json[key] = value # show we will get the value from the prior meta
            meta_dict[key]= get_json_value(meta_dict, value[1:]) # show that the fits had the value
            return meta_dict[key]
        if key.startswith('#'):
            # do not replace an existing key
            key= key[1:]
            if key in meta_dict:
                print(f"WARNING: {key} | {meta_dict[key]} not replaced with {value}")
                return None 
        if key in meta_dict:
            print(f"Replacing existing meta key '{key}' value '{meta_dict}' with new value '{value}'")
        meta_dict[key] = value
        return value

    def add_fits_item(self, key, value):
        """Add a piece of metadata pulled from the FITS file

        Parameters
        ----------
        key: str
            The metadata keyword that was read
        value: any value
            The value that was read
        """
        self.fits[key] = value

    def validate(self, final):
        """Check the metadata and generate error messages if wrong

        Parameters
        ----------
        final: dict {str, any value}
            The final metadata dictionary to be checked
        """
        self.final = final
        missing = []

        for key in valid_meta_keys:
            if key not in self.final and key not in self.optional:
                missing.append(key)

        self.dump_to_console(missing) # always print the results to the console
        if len(missing) > 0:
            return False # return False if there are missing required keys
        return True

    def console_dump_1_line(self, key, reqd, found_json, found_fits, final):
        """Print one metadata summary line on the console

        Parameters
        ----------
        key: str
            The metadata key
        reqd: str in ['Opt', 'Req']
            'Opt' if this metadata is optional, else 'Req'
        found_json: str
            The str(value) of the metadata pulled from the JSON file
        found_fits: str
            The str(value) of the metadata pulled from the FITS file
        final: str
            The str(value) of the final metadata used to generate the starlist
        """
        print(f'{key:<25} {reqd:4} {found_json:<15} {found_fits:<15} {final:<15}')

    def dump_to_console(self, missing):
        """Dump status of the validation to the console

        Parameters
        ----------
        missing: list of str
            List of the metadata keywords that don't have values, but
            are required keywords.
        """
        print("The following metadata key(s) are missing:")
        print(missing)

        print("\n Validation Table")
        self.console_dump_1_line('Key', 'Reqd', 'Found JSON', 'Found FITS', 'Final')
        for key in valid_meta_keys:
            json_value = '' if key not in self.json else str(self.json[key])
            fits_value = '' if key not in self.fits else str(self.fits[key])
            final_value = '' if key not in self.final else str(self.final[key])
            self.console_dump_1_line(key,
                                  'Opt' if key in self.optional else 'Req',
                                  json_value,
                                  fits_value,
                                  final_value)
        print('\n')

meta_validator = MetaValidator()

#
# The so-called JSON metadata file is a temporary band-aid for smart
# telescopes that are currently missing important FITS header
# keywords. (Early Origin scopes have this problem.) The JSON metdata
# file is merged with the metadata that comes from the FITS header,
# providing a way to deal with missing/incorrect FITS header info.
#

def read_meta_from_json(filename, meta_dict):
    """Pull metadata from a JSON metadata file

    Update a meta dictionary using the contents of the JSON metadata
    file to augment or replace entries in the meta
    dictionary. Metadata keywords are validated as they are
    encountered; unrecognized keywords generate a console message.

    Parameters
    ----------
    filename : str
        Pathname of the JSON metadata file
    meta_dict : dictionary
        Metadata dictionary to be modified/augmented

    Returns
    -------
    None

    """
    bytes = Path(filename).stat().st_size
    # This file should NEVER be more than 10K bytes long. In order to
    # limit cyber vulnerability, we immediately quit if we're given a
    # long file here.
    if bytes > 10000:
        print("ERROR: Refusing to read JSON metadata file that exceeds 10K bytes.")
        raise ValueError

    with open(filename, encoding='utf-8') as fp:
        try:
            data = json.load(fp)
        except json.JSONDecodeError:
            print("Parse error reading ", filename)
            raise

        for (keyword, value) in data.items():
            if val := meta_validator.add_json_item(keyword, value, meta_dict):
                meta_dict[keyword] = val


# Read metadata from a FITS header. The metadata that's found will be
# put into the dictionary that's passed as the argument "dict".
def read_meta_from_fits(filename, meta_dict):
    """Pull metadata from a FITS image

    Update a meta dictionary using the keywords found in a FITS image
    file. These values will augment or replace entries in the meta
    dictionary. The keywords pulled from the FITS file depend on the
    smart telescope type to account for different spellings used by
    the different vendors.

    Parameters
    ----------
    filename: str
        Pathname of the FITS file from the smart telescope
    meta_dict : dictionary
        Metadata dictionary to be modified/augmented

    Returns
    -------
    None
    """
    telescope_type= probe_file_for_type(filename)
    with fits.open(filename) as hdul:
        hdu0h = hdul[0].header
        meta_dict['telescope_probe'] = telescope_type

        # read in the whole header
        for key in hdu0h:
            meta_validator.add_fits_item(key, hdu0h[key])
            meta_dict[key] = hdu0h[key] # copy the value into the meta_dict

def wcs_text_2wcs(wcs_text):
    """Convert WCS FITS header text into an astropy WCS object

    Convert a set of FITS header keyword/value pairs describing a WCS
    into an astropy WCS object

    Parameters
    ----------
    wcs_text : str
        Long string holding relevant WCS FITS header/keyword values in
        the form of FITS records, concatenated together into a single
        string.

    Returns
    -------
    WCS object
        An astropy WCS object
    """
    card_list = []
    while len(wcs_text) >= 80:
        # Turn the long string into a list of fits.Card objects
        this_line = wcs_text[0:80]
        wcs_text = wcs_text[80:]
        card_list.append(fits.Card.fromstring(this_line))
        print(this_line)
    # Create a FITS header that contains only the WCS cards
    wcs_header = fits.Header(cards=card_list)
    return WCS(wcs_header)

def table_to_star_items(photometry_table):
    """
    Convert an astropy table with photometry to a list of star items.

    Parameters
    ----------
    photometry_table : astropy table
        The table to convert to a list of star items. The table must
        have as column each of the fields in the `StarItem` class.
    """
    star_items = []
    for row in photometry_table:
        missing_keys = set(StarItem.model_fields.keys()) - set(row.keys())
        if missing_keys:
            raise ValueError(f"Missing keys in table: {missing_keys}")
        star_items.append(
            StarItem(**{key: row[key] for key in StarItem.model_fields.keys()})
        )
    return star_items


def plate_solve_image(filename, metadata, temp_dir, wcs, sources, height, width)    :
    """Plate solve an image

    Plate solve an image using astrometry.net or other means. The image will be
    plate-solved and the WCS information will be stored in the metadata
    dictionary. The WCS information will be stored as a string in the
    metadata dictionary.

    Parameters
    ----------
    filename : str
        Pathname of the image to be plate-solved
    metadata : dict
        Dictionary of metadata values
    options : class OptionBox object
        An object holding all of the operator's conversion options
    temp_dir : str
        Pathname of a directory in which temporary files can be put
    wcs : WCS object
        WCS object to be used if it is already known
    sources, height, width : if you are going to use astrometry.net, you need these

    Returns
    -------
    wcs WCS Object
    """

    ################################
    ## WCS handling overall sequence
    ## 1. If wcs is passed in as parameter, use it. Do not
    ## plate-solve.
    ## 2. If the "solve-field" program is installed, use it to
    ## plate-solve. (Only tested on Linux so far.)
    ## 3. Otherwise, go out to astrometry.net and use the online
    ## plate-solver.
    ################################
    def build_local_command(temp_dir):
        print("WCS using temp_dir = ", temp_dir.name)
        temp_dirname = temp_dir.name
        plate_solve_dir = temp_dirname

        command = " " #"solve-field  will be added later"
        temp_dir_arg = " --dir " + str(plate_solve_dir) #.replace('\\', '/') # output dir
        command += temp_dir_arg
        print("plate_solve_dir = ", plate_solve_dir)
        print("temp_dir_arg = ", temp_dir_arg)
        #command += (" --config + config_file)
        if 'PIXSCALE' in metadata and metadata['PIXSCALE'] is not None:
            pixelscale = metadata['PIXSCALE']
        elif 'pixscale' in metadata and metadata['pixscale'] is not None:
            pixelscale = metadata['pixscale']
        else:
            raise ValueError("Pixelscale not defined.")
        pixel_low = pixelscale * 0.9
        pixel_hi = pixelscale * 1.1
        command += (" --scale-low " + str(pixel_low)
                    + " --scale-high " + str(pixel_hi)
                    + " --scale-units arcsecperpix ")

        if ('dec' in metadata and 'ra' in metadata and
              metadata['dec'] is not None and metadata['ra'] is not None):
            command += (" --ra " + str(metadata['ra'])
                        + " --dec " + str(metadata['dec']))
        else:
            raise ValueError("Dec/RA not defined")

        if 'fov_rad' in metadata and metadata['fov_rad'] is not None:
            command += (" --radius " + str(metadata['fov_rad'])) # in degrees
        else:
            raise AssertionError("Image FOV not defined")

        command += ' "' + str(filename) + '"'
        return command

    if not wcs:
        print("No WCS provided. Will plate-solve.")
        ################################
        ## Plate-solve the image
        ################################
        temp_dir = tempfile.TemporaryDirectory()
        local_system = platform.system()
        if local_system == 'Windows':
            p = Path(os.path.expandvars('%LOCALAPPDATA%'))
            q = p  / 'cygwin_ansvr' / 'bin' / 'solve-field'
            if q.exists():
                cmd = build_local_command(temp_dir)
                cmd2= (str(q) + cmd ).replace('\\', '/')
                full_cmd = f"%LOCALAPPDATA%\\cygwin_ansvr\\bin\\bash.exe --login -c '{cmd2}'"
                print("Executing: ", full_cmd)
                return_code= os.system(full_cmd)
                print("solve-field returned ", return_code)
                if return_code:
                    raise ValueError("Abnormal termination of solve-field")
            else:
                cmd = None
        else:
            if shutil.which('solve-field') is not None:
                cmd = 'solve-field ' + build_local_command(temp_dir)
                print("Executing: ", cmd)

                if os.system(cmd) != 0:
                    raise ValueError("Abnormal termination of solve-field")
            else:
                cmd = None

        if cmd is not None:
                wcs_basename = Path(filename).with_suffix(".new").name
                print("wcs_basename = ", wcs_basename)
                wcs_pathname = Path(temp_dir.name) / wcs_basename
                print("Looking for WCS info in ", str(wcs_pathname))
                with fits.open(str(wcs_pathname)) as hdul:
                    header = hdul[0].header
                    wcs = WCS(header)

        if wcs is None and sources is not None:
            print("Trying astrometry.net.")
            ast = AstrometryNet()
            if astrometry_api_key is None:
                global ui
                dlg = QMessageBox(ui.window)
                dlg.setWindowTitle("No astrometry.net API Key")
                dlg.setText("Must enter astrometry.net API Key via Menu Bar")
                dlg.exec()
                return None

            ast.api_key = astrometry_api_key
            # star_x and star_y were sorted by flux earlier... important here.
            wcs_header = ast.solve_from_source_list(sources['x'],
                                                    sources['y'],
                                                    width,
                                                    height,
                                                    solve_timeout= 120
                                                    #,
                                                    #verbose= True,
                                                    #center_dec=metadata['dec'],
                                                    #center_ra=metadata['ra'],
                                                    #scale_lower=0.9*metadata['pixscale'],
                                                    #scale_upper=1.1*metadata['pixscale']
                                                    )
            wcs = WCS(header=wcs_header)
    return wcs




# Process one (possibly de-Bayered) image
def process_single_image(filename, metadata, options, temp_dir,
                         starlist_json_path, passband_filter,
                         psf_builder, wcs=None):
    """Turn an image file into a starlist

    In a strictly one-to-one operation, turn an image into a starlist,
    and store the starlist as a JSON file in the AAVSO starlist
    format.

    Parameters
    ----------
    filename : str
        The pathname of the image to be turned into a starlist
    metadata : dict
        A dictionary of metadata values
    options : class OptionBox object
        An object holding all of the operator's conversion options
    temp_dir : str
        Pathname of a directory in which temporary files can be put
    starlist_json_path : str
        Pathname that will be used to store the resulting starlist
    filter : str
        Short string holding the AAVSO reporting name for the filter
        associated with this image
    psf_builder: psf_fitting.PSFBuilder reference
        Reference to the PSFBuilder() that is only used when
        psf_fitting is enabled.
    wcs : astropy WCS object
        WCS object that maps pixel coordinates to sky coordinates. If
        provided then astrometry.net fitting is skipped.
    Returns
    -------
    OutputObject
        An instance of an OutputObject containing the resulting
        StarListSet
    """
    # filter == 'M' is a special case == 'CV'
    if passband_filter == 'M':
        passband_filter = 'CV'

    # "G" needs to become "TG" if it hasn't already
    if len(passband_filter) == 1:
        passband_filter = 'T'+passband_filter

    width = None
    height = None

    # This is essentially creating a placeholder for the star items to
    # go into. The star items will be filled in later.

    # starlist = AAVSOStarlist(metadata, passband_filter)
    metadata['staritems'] = []
    metadata['filter'] = passband_filter
    metadata['gain'] = metadata['system_gain']
    #print("model_validate: ", metadata)


    starlist = StarList.model_validate(metadata)
    ccd_image = CCDData.read(filename, unit='adu')

    image_data = ccd_image.data.astype(np.float32)
    (height, width) = np.shape(image_data)
    print("height = ", height, ", width = ", width)

    # Estimate the background
    (mean, median, std) = sigma_clipped_stats(image_data, sigma=3.0)
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    full_background = Background2D(
        image_data,
        (int(width/8),int(height/8)),
        filter_size=(3,3),
        sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    background = full_background.background
    (bkgd_mean, _dummy, bkgd_std) = sigma_clipped_stats(background, sigma=3.0)
    print("background.median = ", full_background.background_median,
          ", background.rms = ", full_background.background_rms_median)
    noise_bkgd_per_pixel = full_background.background_rms_median * starlist.gain
    # Should this be 5 times the background RMS instead of the full image RMS?
    # How do we know the fwhm is roughly 3? is that the same for all smart telescopes?

    # What we *really* want is RMS of the original image after
    # masking all the stars. The RMS of "background" is way off,
    # because Background2D smooths the background, which destroys
    # the original background RMS.
    daofind = DAOStarFinder(fwhm=3.0, threshold=4.*std)
    clean_image = (image_data - background)
    (mean, median, std) = sigma_clipped_stats(clean_image, sigma=3.0)
    sources = daofind(clean_image)
    print("Initial quicklook found ", len(sources), " stars.")

    # Sort the table in-place by flux in reverse order
    sources.sort('flux', reverse=True)

    # Grab a subset of the brightest stars to estimate the FWHM
    subset_size = min(10, len(sources))
    subset = sources[:subset_size]
    fwhm = psf.fit_fwhm(
        clean_image,
        xypos=list(zip(subset['xcentroid'], subset['ycentroid'], strict=False)),
        fit_shape=15
    ).mean()

    print("Estimate FWHM from photutils = ", fwhm)
    metadata['fwhm'] = fwhm

    # Now that we know the *real* FWHM, re-find the stars
    daofind = DAOStarFinder(fwhm=fwhm, threshold=4.0*std,
                            sharplo=0.05, sharphi=3.0,
                            roundlo=-4.0, roundhi=4.0)
    sources = daofind(clean_image)
    print("Sources found before edge-culling: ", len(sources), " stars.")

    # eliminate stars too close to the edges
    EDGELIMIT = 15
    mask = np.array([row['xcentroid'] < EDGELIMIT
                     or row['xcentroid'] > width-EDGELIMIT
                     or row['ycentroid'] < EDGELIMIT
                     or row['ycentroid'] > height-EDGELIMIT
                     for row in sources])
    sources = sources[~mask]
    print("Official source extraction found ", len(sources), " stars.")
    sources.sort('flux', reverse=True)

    phot_radius = options.aperture_size_fwhm * fwhm
    annulus_inner = max(3*phot_radius, 4*fwhm)
    annulus_outer = math.sqrt(100*phot_radius**2 + annulus_inner**2)
    print(f"Aperture radius = {phot_radius:.2f} , with {math.pi * phot_radius * phot_radius:.2f} pixels total")

    # Perform the photometry
    positions = list(zip(sources['xcentroid'],
                         sources['ycentroid'], strict=False))
    apertures = aperture.CircularAperture(positions, r=phot_radius)
    tot_noise_bkgd = np.sqrt(apertures.area) * noise_bkgd_per_pixel

    if options.use_annulus:
        annuli = aperture.CircularAnnulus(positions, annulus_inner, annulus_outer)
        annulus_sigma_clip = SigmaClip(sigma=2.0)
        annulus_data = aperture.ApertureStats(clean_image,
                                              annuli,
                                              sigma_clip=annulus_sigma_clip,
                                              sum_method='center')

        central_sum = aperture.ApertureStats(clean_image,
                                             apertures,
                                             sum_method='exact',
                                             local_bkg=annulus_data.mean)
        centroids = central_sum.centroid
        sources['x'] = centroids[:, 0]
        sources['y'] = centroids[:, 1]
        sources['tot_flux'] = central_sum.sum - annulus_data.sum * apertures.area / annuli.area
    else: # not using an annulus
        result = aperture.aperture_photometry(clean_image, apertures)
        print(result)

        # Make some column names match the starlist schema
        sources['tot_flux'] = result['aperture_sum']
        # Use .value for these next two because they are astropy Quantity
        # objects with the unit "pixels" and we don't need the unit.
        sources['x'] = result['xcenter'].value
        sources['y'] = result['ycenter'].value

    # Clean up the sources table
    print("Sources cleanup starts with ", len(sources), " stars.")

    bad_rows = []
    min_adu = max(tot_noise_bkgd, 0.0)
    for row,content in enumerate(sources):
        if (content['x'] <= 3.0
            or content['y'] <= 3.0
            or content['x'] >= (width-3)
            or content['y'] >= (height-3)
            or content['peak'] <= min_adu
            or content['tot_flux'] <= min_adu):
            bad_rows.append(row)
    print("... removing ", len(bad_rows), " stars.")
    sources.remove_rows(bad_rows)
    print("... now have ", len(sources), " stars.")

    sources.rename_column('peak', 'peak_flux')
    # Sort so that order is well-defined and tests will pass
    sources.sort(keys='tot_flux', reverse=True)

    # Calculate errors using table columns and star flux error in column
    poiss_noise = np.sqrt(starlist.gain * sources['tot_flux'])
    tot_noise = np.sqrt(poiss_noise**2 + tot_noise_bkgd**2) / starlist.gain
    sources['flux_err'] = tot_noise

    # Set flux errors to zero for negative fluxes
    sources['flux_err'][sources['tot_flux'] < 0] = 0.0

    # Check if WCS is already present in the FITS header
    if wcs is None:
        wcs = ccd_image.wcs # This looks for the WCSAXES keyword

    wcs = plate_solve_image(filename, metadata, temp_dir, wcs, sources, height, width)

    # Calculate RA and Dec
    star_coords = wcs.pixel_to_world(sources['x'], sources['y'])
    sources['ra'] = star_coords.ra.deg
    sources['dec'] = star_coords.dec.deg

    # Populate the background flux column. The "+0.5" is to reproduce the
    # behavior of the original code.
    sources['bkgd_flux'] = [
        background[int(0.5+y), int(0.5+x)]
        for (x, y) in zip(sources['x'], sources['y'], strict=False)
    ]

    print(sources.colnames)
    print(StarItem.model_fields.keys())
    print("Creating starlist with ", len(sources), " stars.")
    starlist.staritems = table_to_star_items(sources)

    ################################
    ## Do PSF fitting, if requested
    ################################
    if options.use_psf_fitting:
        starlist.staritems.sort(key=lambda star:
                                star.tot_flux, reverse=True)
        psf_builder.add_image(clean_image,
                              metadata,
                              noise_bkgd_per_pixel,
                              starlist)

    print("starlist is ", type(starlist))
    print('creating OutputObject w/filename = ',
          metadata['orig_filename'])
    return OutputObject(
        filename=starlist_json_path,
        orig_image_path=metadata['orig_filename'],
        logical_starlist=StarListSet(star_lists=[starlist])
    )

def process_3d_file(filename, temp_dir):
    """Process an RGB image, converting it into one or more starlists

    Process a stacked, one-shot-color image using the user's selected options
    to turn it into some number of starlists. Depending on options,
    the image may be separated into color subchannels or treated
    as a monochrome image. Create a starlist for each resulting channel.

    Parameters
    ----------
    filename : str
        Pathname of the image to be processed
    temp_dir : str
        Pathname of a directory in which temporary files can be put

    Returns
    -------
    list of ImageDescriptors (tuples: (filename, filtername))
        These are the three files that this 3D file was converted into

    """
    output_filenames = [] # each entry in this list is a tuple: (filter, filename)
    with fits.open(filename) as hdul:
        image1 = hdul[0].data[0,0:,0:]
        image2 = hdul[0].data[1,0:,0:]
        image3 = hdul[0].data[2,0:,0:]

        image_list = [(image1, 'TR'),
                      (image2, 'TG'),
                      (image3, 'TB')]

        for (data, color) in image_list:
            output_tgt = Path(temp_dir) / ("image_"+color+".fits")

            hdu = fits.PrimaryHDU()
            # push keywords in from the original file
            for keyword in hdul[0].header:
                if keyword not in ('COMMENT', 'HISTORY'):
                    value = hdul[0].header[keyword]
                    comment = hdul[0].header.comments[keyword]
                    hdu.header[keyword] = (value,comment)
                else: # Yes, this is a comment/history
                    comment = hdul[0].header[keyword]
                    for card in comment:
                        hdu.header[keyword] = card

            hdu.data = data
            hdu.header['filter'] = (color, 'Bayer color mask')
            # update_header will "fix" the header to match the data
            hdu.update_header()
            fits.writeto(output_tgt, data, header=hdu.header, overwrite=True)
            ImageDescriptor = namedtuple('ImageDescriptor', ['filter','filename'])
            output_filenames.append(ImageDescriptor(color,output_tgt))
    return output_filenames

def process_rgb_file(filename, options, temp_dir, metadata,
                     starlist_tgtname, psf_builder, wcs=None):
    """Process an RGB image, converting it into one or more starlists

    Process a one-shot-color image using the user's selected options
    to turn it into some number of starlists. Depending on options,
    the image may be separated into Bayer color subchannels or treated
    as a monochrome image. Depending on options, pixels may or may not
    have their values adjusted. Depending on options, de-Bayered
    images may or may not be shifted into alignment and stacked into a
    luminance channel. Create a starlist for each resulting channel.

    Parameters
    ----------
    filename : str
        Pathname of the image to be processed
    options : class OptionBox object
        Reference to the OptionBox object that holds information on
        the user's processing options
    temp_dir : str
        Pathname of a directory in which temporary files can be put
    metadata : dict
        Dictionary of metadata for the image
    starlist_tgtname : str
        Pathname to be used (possibly as a template) for the resulting starlist(s).
    wcs : astropy WCS object, optional
        WCS object that maps pixel coordinates to sky coordinates. If
        provided then astrometry.net fitting is skipped.
    psf_builder: psf_fitting.PSFBuilder
        This option is always present (even when PSF fitting isn't
        being done). It's a reference to the PSF processing class.

    Returns
    -------
    list of OutputObjects
        These are the logical starlists that result.

    """
    if not meta_validator.validate(metadata):
        return []
    de_bayer = options.de_bayer
    fits_format = metadata['telescope_probe'][1] # fits_format
    # can't get here without a telescope_probe   if 'fits_format' in metadata else "bayered")
    output_objects = []
    single_color_files = []
    do_stacking = False
    # Copy the original metadata so that whatever changes we make here
    # don't haunt us later.
    adj_meta_dict = dict(metadata)
    if fits_format == "3Dstacked":
        # Need the user to choose between combining the R, G, and B
        # images into a single luminance channel vs. generating three
        # starlists, one for each of the three channels. Choice will
        # go straight into options.split_stacked_image.
        popup = Option3DPopup(options)
        ret_val = popup.exec()
        # if the popup returned 0, then the user selected the Cancel
        # button.
        if ret_val == 0:
            return []

        single_color_files = process_3d_file(filename, temp_dir)
        do_stacking = not options.split_stacked_image
        #adj_meta_dict['pixscale'] /= 2.0 # Correct for non-de-Bayered image
    elif de_bayer:
        single_color_files = de_bayer_file(filename, adj_meta_dict, temp_dir)
        do_stacking = options.stack_channels

    if do_stacking:
        print("Stacking images")
        stacked_image = stack_images(single_color_files, options, temp_dir)
        starlist_filename = starlist_tgtname.replace("$$","M")
        output_objects.append(process_single_image(stacked_image,
                                                   adj_meta_dict,
                                                   options,
                                                   temp_dir,
                                                   starlist_filename,
                                                   'M',
                                                   psf_builder,
                                                   wcs=wcs))
    elif len(single_color_files) > 1:
        print("Processing separate channels")
        tg_num = 1
        for (photfilter,file) in single_color_files:
            filter_file = photfilter
            # Hangle "TG" and "G" filters the same
            if photfilter in ['TG', 'G']:
                filter_file = "TG"+str(tg_num)
                tg_num += 1
            starlist_filename = starlist_tgtname.replace("$$",filter_file)
            output_objects.append(process_single_image(file,
                                                       adj_meta_dict,
                                                       options,
                                                       temp_dir,
                                                       starlist_filename,
                                                       photfilter,
                                                       psf_builder,
                                                       wcs=wcs))
    else:
        # Not de-Bayered; treat as single monochrome image
        print("Processing single monochrome image")
        starlist_filename = starlist_tgtname.replace("$$","M") # M==monochrome
        print(metadata)
        output_objects.append(process_single_image(filename,
                                                   adj_meta_dict,
                                                   options,
                                                   temp_dir,
                                                   starlist_filename,
                                                   'M',
                                                   psf_builder,
                                                   wcs=wcs))
    print(f"ProcessRGB: returning {len(output_objects)} output_objects.")
    return output_objects

################################################################
##        Display GUI Comes Next
################################################################

class FileChooser:
    """Select one or more files for processing

    FileChooser is an API that groups a set of Qt widgets into a group
    that provides the ability to select files to be used in
    processing, providing both visual feedback to the operator and an
    API to allow code to query the file(s) selected.

    Attributes
    ----------
    text_widget : Qt widget (either QLineEdit or QPlainTextEdit)
        Text entry widget that can hold the chosen file's
        filename. If `multiple_files_okay` is True, this is a
        multi-line field (QPlainTextEdit)
    popup_button : Qt button widget
        Button widget that is clicked to give the user a popup file
        chooser window

    """
    def __init__(self,
                 text_entry_widget,
                 chooser_button,
                 multiple_files_okay=False):
        """Create a file-chooser object

        Create a FileChooser object (used for dark, flat, metadata,
        bias, and image files).

        Parameters
        ----------
        text_entry_widget : Qt text entry widget (QLineEdit)
            Text entry widget that can hold the chosen file's
            filename. If `multiple_files_okay` is True, this is a
            multi-line field (QPlainTextEdit)
        chooser_button : Qt button widget
            The button to be activated in order to choose the file
        multiple_files_okay : bool, optional, default=only-one-file-allowed
            A flag to indicate whether this FileChooser is allowed to
            select multiple files (i.e., light image files) or just a
            single file (e.g., a master flat image)

        Returns
        -------
        FileChooser object
        """
        self.text_widget = text_entry_widget
        self.popup_button = chooser_button
        self.multiple_files_okay = multiple_files_okay
        chooser_button.clicked.connect(self.chooser_popup)

        if not multiple_files_okay:
            self.file_mode = QFileDialog.ExistingFile
        else: # Big entry for image filenames
            self.file_mode = QFileDialog.ExistingFiles

    def chooser_popup(self, _):
        """Create popup window to choose file(s)

        Initiate the popup window to select one (or more) files. This
        method blocks until the file selection has completed, so all
        other buttons and widgets in the application will be
        disabled. The selected filename will be put into the
        FileChooser's `text_entry_widget`.

        Returns
        -------
        None
        """
        dialog = QFileDialog(self.text_widget)
        dialog.setFileMode(self.file_mode)
        if dialog.exec():
            if self.multiple_files_okay:
                # Now append to the filelist
                entry_list = dialog.selectedFiles()
                for entry in entry_list:
                    self.text_widget.appendPlainText(entry + '\n')
            else:
                self.text_widget.setText(dialog.selectedFiles()[0])
        else:
            self.text_widget.setText("")

    def entered_filename(self):
        """Return the filename entered via this FileChooser

        Return the filename entered by this FileChooser if a filename
        was entered. If no filename was entered, returns None. This
        should only be used if `multiple_files_okay` was False.

        Parameters
        ----------
        None

        Returns
        -------
        None or str
            Return the filename entered by this FileChooser if a filename
            was entered. If no filename was entered, returns None
        """
        assert not self.multiple_files_okay, \
                "Call to EnteredFilename should be EnteredFilenameList"
        raw_text = self.text_widget.text()
        if raw_text is None or len(raw_text.strip()) == 0:
            return None
        return raw_text.strip()

    def entered_filename_list(self):
        """Return the filenames entered via this FileChooser

        Return the filenames entered by this FileChooser if a filename
        was entered. If no filename was entered, returns None. This
        should only be used if `multiple_files_okay` was True.

        Parameters
        ----------
        None

        Returns
        -------
        list of str
            Return the filenames entered by this FileChooser if a filename
            was entered (list might be empty).
        """
        assert self.multiple_files_okay, \
                "Call to EnteredFilenameList should be EnteredFilename"
        raw_text = self.text_widget.toPlainText()
        text_words = raw_text.split('\n')
        print("Files to process = ", text_words)
        return text_words

    def clear_filename(self):
        """clear the entered filename

        clear the filename entered for this FileChooser object

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.text_widget.setText("")

class OptionsUI:
    def __init__(self):
        """Create an object for the UI options, including file names

        Create an OptionsUI object (a singleton), and associate the
        related display button widgets with a set of available option
        queries.

        Parameters
        ----------
        None

        Returns
        -------
        An OptionsUI instance
        """
        # File choosers first...use the properties defined later
        # to get the file names out.
        self._bias_file = FileChooser(ui.window.bias_entry,
                                      ui.window.BiasButton)

        self._dark_file = FileChooser(ui.window.dark_entry,
                                      ui.window.DarkButton)
        self._flat_file = FileChooser(ui.window.flat_entry,
                                      ui.window.FlatButton)
        self._meta_file = FileChooser(ui.window.meta_entry,
                                      ui.window.MetaButton,
                                      multiple_files_okay=True)
        self._image_file = FileChooser(ui.window.image_filename_list,
                                       ui.window.AddImageButton,
                                       multiple_files_okay=True)

        # The remaining options
        self.pretend_monochrome = ui.window.MonochromeButton
        self.one_channel = ui.window.SingleChannelButton
        self.stacked_channels = ui.window.StackedButton
        self.interp_stack_channels = ui.window.StackInterpButton
        self.color_correx = ui.window.ColorBalanceButton
        self.psf_photometry = ui.window.PSFPhotButton
        self.aperture_photometry = ui.window.AperturePhotButton

        self.multiple_starlists = ui.window.OneSLPerFile
        self.add_wcs_to_image = ui.window.UpdateWCSButton
        self.aperture_size = ui.window.ApertureSize
        self.subtract_annulus = ui.window.AnnulusSubtractionCheckbox

        self.split_stacked_image = True

    @property
    def add_wcs(self):
        """Query whether WCS keywords need to be added to the image

        Return True if the input FITS file needs to have the WCS
        information added to it

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if WCS is to be added
        """
        return self.add_wcs_to_image.isChecked()

    @property
    def one_sl_per_file(self):
        """Query whether to save just one logical starlist per file

        Return True if the output starlist(s) should be split into
        multiple starlist files. The alternative is to pack multiple logical
        starlists into each starlist file.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if each starlist is to go into a separate file
        """
        return self.multiple_starlists.isChecked()

    @property
    def aperture_size_fwhm(self):
        """Query the aperture size factor

        The aperture size factor is multiplied by the image's average
        FWHM to establish the photometry aperture radius. This query
        returns that multiplicative factor.

        Parameters
        ----------
        None

        Returns
        -------
        float
            The factor that should be used. A value of 1.0 is returned
            if nothing is entered or if an entry is invalid.
        """
        entry_str = self.aperture_size.text()
        try:
            entry_float = float(entry_str)
        except ValueError:
            ErrorPopup("Invalid Aperture Size entry: " + entry_str)
            return 1.0
        if entry_float < 0.1 or entry_float > 10.0:
            ErrorPopup("Aperture size entry out of bounds (0.1 .. 10.0)")
            self.aperture_size.setText("1.0")
            return 1.0
        return entry_float

    @property
    def use_annulus(self):
        """Query whether an annulus aperture helps estimate background

        Return True if the sky background found in an annulus around
        the star centroid should be used during background
        subtraction. The alternative is to use only a slowly-varying
        background level across the entire image to estimate
        background.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if an annulus is to be used
        """
        return self.subtract_annulus.isChecked()

    @property
    def de_bayer(self):
        """Query whether input file(s) need to be de-Bayered

        Return True if the input images need to be split into separate
        color images (de-Bayer).

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if input images needed to be split
        """
        return not self.pretend_monochrome.isChecked()

    # return "psf" or "app_phot"
    @property
    def get_phot(self):
        """Query the kind of photometry to be done

        Return a string indicating whether aperture photometry or
        PSF-fitting photometry is to be done.

        Parameters
        ----------
        None

        Returns
        -------
        str
            'psf' if PSF-fitting is to be done; 'app_phot' if aperture
            photometry is to be done
        """
        return "psf" if self.psf_photometry.isChecked() else "app_phot"

    @property
    def stack_channels(self):
        """Query whether de-Bayered images are to be stacked

        Query whether de-Bayered images are to be stacked into a
        single sort-of-luminance channel image. Only makes sense to
        query this if de_bayer() returns True. If stacking was chosen,
        the method used for doing the stacking depends on the setting
        of the interpolate_channels() query.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if the four de-Bayered images are to be stacked
        """
        return (self.stacked_channels.isChecked()  or
                self.interp_stack_channels.isChecked())

    @property
    def interpolate_channels(self):
        """Query whether de-Bayered images get shifted into pixel alignment

        Query whether the four de-Bayered images are to be shifted
        into pixel alignment. Each color gets shifted one or two
        pixels left/right/up/down using flux-preserving bilinear
        interpolation in order to have the pixels in each color
        channel correspond to exactly the same Dec/RA sky location.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if the de-Bayered images need to be shifted slightly
            to bring them into sky coordinate alignment
        """
        return self.interp_stack_channels.isChecked()

    @property
    def get_color_balance(self):
        """Query whether the pixel values should be adjusted for color balance

        Query whether "color balancing" should be done. This is
        performed on the original RGB image with a linear
        transformation to adjust pixel values. Four different
        transformations are used; one for each color channel. The
        values for the transformation coefficients are chosen to give
        all four channels the same background average level and the
        same standard deviation around that common average.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if color balancing is to be done
        """
        return self.color_correx.isChecked()

    @property
    def use_psf_fitting(self):
        """Query whether PSF-fitting photometry is to be done

        Return a boolean indicating whether
        PSF-fitting photometry is to be done. This method is 100%
        redundant with get_phot() and should be retired.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if PSF-fitting is to be done
        """
        return self.psf_photometry.isChecked()

    @property
    def bias_file(self):
        """Return the pathname of the bias file

        Return the pathname of the bias file that was entered by the
        operator.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The pathname of the bias file
        """
        return self._bias_file.entered_filename()

    @property
    def dark_file(self):
        """Return the pathname of the dark file

        Return the pathname of the dark file that was entered by the
        operator.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The pathname of the dark file
        """
        return self._dark_file.entered_filename()

    @property
    def flat_file(self):
        """Return the pathname of the flat file

        Return the pathname of the flat file that was entered by the
        operator.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The pathname of the flat file
        """
        return self._flat_file.entered_filename()

    @property
    def meta_file(self):
        """Return the pathname of the metadata file

        Return the pathname of the metadata file that was entered by
        the operator.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The pathname of the metadata file
        """
        return self._meta_file.entered_filename_list()

    @property
    def image_file(self):
        """Return the pathname of the image files

        Return the pathname of the image files that were entered by
        the operator.

        Parameters
        ----------
        None

        Returns
        -------
        list of str
            The pathname of the image files
        """
        return self._image_file.entered_filename_list()


class BayerHandlingOptions(StrEnum):
    """Enumeration of Bayer handling options"""
    # See the python documentation for Enums work. Behind the scenes pythopn
    # creates a class whose attributes have the names listed below.
    PRETEND_MONOCHROME = "pretend_monochrome"
    STACKED_CHANNELS = "stacked_channels"
    INTERP_STACK_CHANNELS = "interp_stack_channels"
    SPLIT_STACKED_IMAGE = "split_stacked_image"


class PhotometryMethods(StrEnum):
    """Enumeration of photometry methods"""
    # See the python documentation for Enums work. Behind the scenes pythopn
    # creates a class whose attributes have the names listed below.
    APERTURE = "aperture"
    PSF = "psf"


class OptionsAPI(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_default=True, validate_assignment=True)
    bayer_handling: BayerHandlingOptions = BayerHandlingOptions.PRETEND_MONOCHROME
    color_correx: bool = False
    subtract_annulus: bool = False
    multiple_starlists: bool = False
    add_wcs_to_image: bool = False
    aperture_size: float = 1.0
    photometry_method: PhotometryMethods = PhotometryMethods.APERTURE
    astrometry_net_api_key: str = ""
    bias_file: str = ""
    dark_file: str = ""
    flat_file: str = ""
    meta_file: list[str] = [""]
    image_file: list[str] = [""]

    # These are accessed by the current code.
    @property
    def de_bayer(self):
        return self.bayer_handling != BayerHandlingOptions.PRETEND_MONOCHROME

    @property
    def interpolate_channels(self):
        return self.bayer_handling == BayerHandlingOptions.INTERP_STACK_CHANNELS

    @property
    def get_color_balance(self):
        return self.color_correx

    @property
    def stack_channels(self):
        return (
            self.bayer_handling == BayerHandlingOptions.STACKED_CHANNELS
            or self.bayer_handling == BayerHandlingOptions.INTERP_STACK_CHANNELS
        )

    @property
    def use_psf_fitting(self):
        return self.photometry_method == PhotometryMethods.PSF

    @property
    def add_wcs(self):
        return self.add_WCS_to_image

    @property
    def one_sl_per_file(self):
        return self.multiple_starlists

    @property
    def aperture_size_fwhm(self):
        return self.aperture_size

    @property
    def use_annulus(self):
        return self.subtract_annulus

class UI:
    """Singleton class used to connect Qt Designer to this app

    This class (and its singlton instance, "ui") are used to hold the
    widget hierarchy that is read in from the *.ui file created by Qt
    Designer for this app.

    Attributes
    ----------
    window : Qt window
        The root of the *.ui file.
    """
    def __init__(self):
        """Set up Qt widgets for this app

        Read in the .ui file created with the Qt designer app. Store
        all the resulting widgets as children of self.window. This
        requires the .ui file to be in the same directory as the
        Python script is located.

        Parameters
        ----------
        None

        Returns
        -------
        UI object
        """
        source_path = Path(__file__).resolve()
        ui_filename = source_path.parent / "image2sl.ui"
        ui_file = QFile(ui_filename)
        if not ui_file.open(QIODevice.ReadOnly):
            print(f"Cannot open {ui_filename}: {ui_file.errorString()}")
            sys.exit(-1)
        loader = QUiLoader()
        self.window = loader.load(ui_file, None)
        ui_file.close()
        if not self.window:
            print(loader.errorString())
            sys.exit(-1)
        self.window.ApertureSize.setText("1.0")
        self.window.actionQuit.triggered.connect(QApplication.quit)

class APIEntryDialog(QDialog):
    """The QDialog popup window used to enter the astrometry.net API key

    The QDialog popup window used to enter the astrometry.net API key.

    Attributes
    ----------
    prompt : QLabel widget
        Holds the prompt message telling the user what to do
    lineEdit : QLineEdit widget
        Text entry widget that holds the value of the API key. This
        will be initialized to any previously-entered key value.
    save_checkbox : QCheckBox widget
        Checkbox to indicate that the API key value should be saved
        for use in the future as the default API key value
    buttonBox : QDialogButtonBox widget
        Container widget that holds the buttons for the popup
    """
    def __init__(self, parent=None):
        """Create a popup dialog window

        Create a popup dialog window. The "Okay" standard button will
        have its default behavior overridden so that the entered API
        key value gets saved.

        Parameters
        ----------
        parent : Qt window
            The parent widget that "owns" this popup

        Returns
        -------
        Qt QDialog window
        """
        super().__init__(parent)

        self.setWindowTitle("Astrometry.net API Key Entry")
        layout = QVBoxLayout()
        self.prompt = QLabel('Enter your astrometry.net API key:')
        layout.addWidget(self.prompt)
        self.lineEdit = QLineEdit()
        if astrometry_api_key is not None:
            self.lineEdit.setText(astrometry_api_key)
        layout.addWidget(self.lineEdit)

        self.save_checkbox = QCheckBox('Save API Key')
        layout.addWidget(self.save_checkbox)

        q_btn = (QDialogButtonBox.StandardButton.Ok |
                QDialogButtonBox.StandardButton.Cancel)
        self.buttonBox = QDialogButtonBox(q_btn)
        self.buttonBox.accepted.connect(self.accept_key)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)

        self.setLayout(layout)

    def accept_key(self):
        """Intercept the dialog's "Okay" button

        Before executing the default "Okay" button behavior, save the
        API key value for use here in this program (as a global
        variable) and save in the user's STWG local storage folder.
        """
        global astrometry_api_key
        astrometry_api_key = self.lineEdit.text()
        if self.save_checkbox.isChecked():
            save_astrometry_key(astrometry_api_key)
        self.accept()           # execute default behavior (kills the popup)

def get_astrometry_key():
    """Lookup the saved value of the astrometry.net API key

    Look for the file containing the astrometry.net API key. If the
    file exists and contains some text, return that text as the
    key. It is *not* an error for the file to not exist -- in which
    case this function will silently return "None".

    Returns
    -------
    Either a string (the stored API key) or None
    """
    localdir = None
    local_system = platform.system()
    if local_system == 'Windows':
        localdir = Path.home() / "AppData" / "Local" / "STWG"
    elif local_system == 'Linux':
        localdir = Path.home() / ".stwg"
    elif local_system == 'Darwin':
        localdir = Path.home() / ".stwg"
    else:
        raise ValueError("OS Name not recognized")

    localdir.mkdir(parents=True, exist_ok=True)
    api_key_pathname = localdir / "astrometryAPIkey.txt"

    try:
        return api_key_pathname.read_text()
    except (PermissionError, FileNotFoundError):
        return None

def save_astrometry_key(key_value):
    """Store the value of the astrometry.net API key in a local file

    Store the value of the astrometry.net API key in a file in the
    user's app data area (windows) or home directory (Linus/MacOS).

    Parameters
    ----------
    key_value : string
        The value of the API key

    Returns
    -------
    None
    """
    localdir = None
    local_system = platform.system()
    if local_system == 'Windows':
        localdir = Path.home() / "AppData" / "Local" / "STWG"
    elif local_system == 'Linux':
        localdir = Path.home() / ".stwg"
    elif local_system == 'Darwin':
        localdir = Path.home() / ".stwg"
    else:
        raise ValueError("OS Name not recognized")

    localdir.mkdir(parents=True, exist_ok=True)
    api_key_pathname = localdir / "astrometryAPIkey.txt"

    try:
        api_key_pathname.write_text(key_value)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Error Trying to save API Key")
        raise

class MainWindow:
    def __init__(self, options, ui=None, wcs=None):
        """Set up main display window and key singleton objects

        Create the FileChooser objects for each file chooser button in
        the display. Create a temporary directory in which working
        files can be stored (and destroyed when the program
        exits). Create singleton objects for OptionBox and the
        ProgressBar that is displayed while generation of the
        starlists is performed.

        Parameters
        ----------
        options : an Options object
            Either a UI-based or API-based object that holds the
            options. Should be either an instance of OptionsUI or
            OptionsAPI.

        ui : UI object, optional
            The UI object that holds the Qt widgets that make up the
            display. If None, then the program is running in API mode
            and the starlists are generated without any user imput.

        wcs : astropy WCS object, optional
            WCS object that maps pixel coordinates to sky coordinates.
            If provided then astrometry.net fitting is skipped.

        Returns
        -------
        None
        """
        global astrometry_api_key

        temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir = temp_dir
        self.temp_dirname = self.temp_dir.name
        print("Working in temporary directory ", self.temp_dirname)
        print(f"===> st-pipeline version {__version__} <===")
        self.options = options
        self.ui = ui
        self._wcs = wcs
        # Try getting the API key from options, or return None if not there
        astrometry_api_key = getattr(self.options, "astrometry_net_api_key", None)
        if astrometry_api_key is None:
            astrometry_api_key = get_astrometry_key()

        self.have_ui = False
        if self.ui:
            self.have_ui = True
            self.progressbar = self.ui.window.progressBar
        else:
            self.generate_starlist()

    def get_key(self):
        dialog = APIEntryDialog(self.ui.window)
        dialog.exec()

    ################################
    ## generate_starlist button
    ## starts here
    ################################
    def do_generate_starlist(self):
        """Start the process of generating a starlist

        The current version of this Python script is single-threaded,
        making this method a little unnecessary, but it's still here
        to make the transition to a two-thread structure easier. (One
        thread would be the graphics thread, the other thread would do
        the image-to-starlist conversion.) If threading was being
        used, this is where the working thread would be started. This
        is where the progress bar is made active.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.progressbar.setRange(0,100) # indeterminate mode
        self.progressbar.setValue(20)
        self.progressbar.setTextVisible(True)
        self.progressbar.setFormat("...Running...")
        self.progressbar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressbar.show()
        self.generate_starlist()
        self.progressbar.hide()

    def generate_starlist(self):
        """Actually perform the conversion of image to starlist

        For each image selected by the user (in the image
        FileChooser), perform a conversion into from 1 to 5 starlists,
        depending on the selected options.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Returns False to indicate that the thread (if used) should self-terminate
        """
        meta_validator.clear()
        image_list = self.options.image_file
        dark_filename = self.options.dark_file
        flat_filename = self.options.flat_file
        bias_filename = self.options.bias_file
        metadata_list = self.options.meta_file

        psf_builder = psf_fitting.PSFBuilder(display_graphs=self.have_ui)
        all_output = [] #  this is a list of lists of OutputObjects

        for image_filename in image_list:
            QGuiApplication.processEvents()
            #Skip blank lines (if present)
            if image_filename is None:
                continue
            image_filename = image_filename.strip()
            if image_filename == '':
                continue

            working_filename = image_filename
            image_path = Path(image_filename)
            orig_dir = image_path.parent
            orig_file_base = image_path.stem
            starlist_tgtname = str(Path(orig_dir, orig_file_base+"_$$.star"))

            hdu_working = fits.open(image_filename)
            working_image = hdu_working[0].data.astype(float)
            if (dark_filename
                or flat_filename
                or bias_filename):
                calibrated_image = str(Path(self.temp_dirname, "light.fits"))
                with fits.open(image_filename) as hdu_working:
                    working_image = hdu_working[0].data.astype(float)
                    if bias_filename:
                        with fits.open(bias_filename) as hdul:
                            bias = hdul[0].data
                            working_image -= bias
                    if dark_filename:
                        with fits.open(dark_filename) as hdul:
                            dark = hdul[0].data
                            working_image -= dark
                    if flat_filename:
                        with fits.open(flat_filename) as hdul:
                            flat = hdul[0].data
                            flat = flat.astype(float) / np.median(flat)
                            working_image /= flat
                    hdu_working[0].update_header()
                    fits.writeto(calibrated_image, working_image,
                                 hdu_working[0].header, overwrite=True)
                working_filename = calibrated_image

            meta = {} # This is the metadata dictionary

            # Order matters here. The standalone metadata file is to override
            # whatever is found in the FITS header of the image file
            read_meta_from_fits(image_filename, meta)

            QGuiApplication.processEvents()
            # Now get the metadata from the standalone metadata file (sidecar)
            for metadata_filename in metadata_list:
                if metadata_filename is not None and metadata_filename != '':
                    meta_path = Path(metadata_filename)
                    if not (meta_path.is_file() and meta_path.exists() and
                            meta_path.stat().st_mode & 0o400):
                        print("Cannot read metadata from file ", metadata_filename)
                        raise ValueError("Cannot read metadata file")
                    print("Reading metadata from ", metadata_filename)
                    read_meta_from_json(metadata_filename, meta)        

            # save the filename in the metadata
            meta["filename"] = image_filename.split("/")[-1]

            # path to the meta_json_files directory
            mp = Path(get_pkg_data_filename("meta_json_files/Seestar50/basic.json")).parent.parent

            # read meta adjustment jsons
            #   apply basic.json
            mpp= Path(mp, meta["telescope_probe"][0], "basic.json")
            print("Reading basic.json from ", mpp)
            read_meta_from_json(mpp, meta)

            #   look for and apply type specific json
            mpp= Path(mp, meta["telescope_probe"][0], meta["telescope_probe"][1]+".json")
            if (mpp.is_file() and mpp.exists() and mpp.stat().st_mode & 0o400):
                read_meta_from_json(mpp, meta)

            # post processing of '!' keys in the metadata
            # utility to convert local time to UTC
            def Local2UTC(lat, long, local_time_str):
                # courtesy of GPT-4o
                # Parse the local time string into a datetime object
                local_time = datetime.strptime(local_time_str, '%Y-%m-%dT%H:%M:%S.%f')
                # Find the timezone
                tf = TimezoneFinder()
                timezone_str = tf.timezone_at(lng=long, lat=lat)
                if timezone_str is None:
                    raise ValueError("Could not find timezone for the given coordinates.")
                # Get the timezone object
                local_tz = pytz.timezone(timezone_str)
                # Localize the datetime to the found timezone
                local_dt = local_tz.localize(local_time)
                # Convert to UTC
                utc_dt = local_dt.astimezone(pytz.utc)
                return utc_dt.strftime('%Y-%m-%dT%H:%M:%S')

            # look for special processing keys
            for key, value in meta.items():
                # eg "ra": "!RA hr2deg"
                if isinstance(value, str) and value.startswith('!'):
                    tt= value[1:].split()
                    if tt[1] == "hr2deg": # convert decimal hours to degrees
                        if val := get_json_value(meta, tt[0]):
                            meta[key]= float(val) * 15.0
                    elif tt[1] == "Local2UTC": # convert local time to UTC
                        # eg  "obs_time": "!DATE-OBS Local2UTC"
                        meta[key]= Local2UTC(meta["site_lat"], meta["site_lon"], get_json_value(meta, tt[0]))
                    elif tt[1] == "refmtDate":
                        # "obs_time": "!StackedInfo.dateTime refmtDate %m-%d-%yB%H_%M_%S"
                        #   B is a blank space
                        d= datetime.strptime(get_json_value(meta, tt[0]), tt[2].replace('B', ' '))
                        meta[key]= d.strftime("%Y-%m-%dT%H:%M:%S")
                    elif tt[1] == "index":
                        # eg "tel_firmware" : "!CREATOR index 1"
                        meta[key]= get_json_value(meta, tt[0]).split()[int(tt[2])]

            print("Final metadata is ", meta)

            # Copy the file to the temporary directory (ie, don't touch input file)
            temp_image_filename = os.path.join(self.temp_dirname, os.path.basename(image_filename))
            shutil.copy(image_filename, temp_image_filename)
            print("modified input file is ", temp_image_filename)
            # is the file fits header missing necessary info?
            #   Should be complete enough so you can load into VPhot and be plate solved
            with fits.open(temp_image_filename, mode='update') as hdul:
                hdu0h = hdul[0].header
                if 'RA' not in hdu0h:  hdu0h['RA'] = meta['ra'] / 15.0 # FITS wants hours
                if 'DEC' not in hdu0h: hdu0h['DEC'] = meta['dec']
                if 'DATE-OBS' not in hdu0h: hdu0h['DATE-OBS'] = meta['obs_time']
                if 'EXPTIME' not in hdu0h: hdu0h['EXPTIME'] = meta['exposure']
                if 'BAYERPAT' not in hdu0h: hdu0h['BAYERPAT'] = meta['bayerpat']
                if 'ROWORDER' not in hdu0h: hdu0h['ROWORDER'] = meta['roworder']
                if 'YBAYROFF' not in hdu0h: hdu0h['YBAYROFF'] = meta['ybayroff']
                hdul[0].header = hdu0h
                hdul.flush()

            # if the WCS is not provided, then we should get it now
            if self._wcs is None:
                wcs = CCDData.read(temp_image_filename, unit='adu', format='fits').wcs
                if wcs is None:
                    wcs = plate_solve_image(temp_image_filename, meta, self.temp_dirname, wcs, None, None, None)
                    # Save the filename with its WCS
                    if wcs is not None:
                        wcs_header = wcs.to_header()
                        with fits.open(temp_image_filename, mode='update') as hdul:
                            hdul[0].header.extend(wcs_header, useblanks=False, update=True)
                            hdul.flush()
            else:
                wcs= self._wcs.copy()

            if meta_validator.validate(meta):
                meta['orig_filename'] = image_filename
                if self.options.get_color_balance:
                    working_filename = bayer_balance_file(working_filename)
                output_objs = process_rgb_file(working_filename,
                                               self.options,
                                               self.temp_dirname,
                                               meta,
                                               starlist_tgtname,
                                               psf_builder,
                                               wcs=wcs)
                all_output.append(output_objs)
        # Now that all images have been processed, let the psf_fitter
        # perform PSF photometry. If the option was not turned on,
        # this will return quietly without doing anything.
        psf_builder.build_psf()

        # Write the starlist files
        if self.options.one_sl_per_file:
            for obj in all_output:
                for output in obj:
                    output.Write()
        else:
            for output_objs in all_output:
                if len(output_objs) == 0:
                    continue
                path1 = output_objs[0].orig_image_path
                filename = path1.with_suffix('.star')
                sl_set = output_objs[0].logical_starlist
                for output in output_objs[1:]:
                    sl_set.star_lists.extend(output.logical_starlist.star_lists)
                print("Writing total of ",
                      len(sl_set.star_lists),
                      " logical starlists to ",
                      filename)

                filename.write_text(sl_set.model_dump_json(indent=2))
        return False

class ErrorPopup:
    """An error popup window

    Pop up a new window with an error message in it and wait for the
    user to acknowledge the error; then resume control with the next
    statement in the program flow.
    """
    def __init__(self, msg):
        """Create an error popup window

        Parameters
        ----------
        msg: str
            The error message to be displayed

        Returns
        -------
        None
        """
        dlg = QMessageBox(None)
        dlg.setWindowTitle("Error")
        dlg.setText(msg)
        dlg.exec()

class Option3DPopup(QDialog):
    """A popup window that appears if the FITS file contains 3 images

    If the FITS file has 3 images in it, then we need to ask the user
    what to do with the three images: stack them into a single
    luminance channel or process them into three distinct logical
    starlists.

    Attributes
    ----------
    options: OptionsUI reference
        A reference to the OptionsUI instance that holds all option
        info. The two buttons of this popup map into the
        split_stacked_image bool option.
    radio_stack: QRadioButton
        One of the two buttons. If checked, user is asking for
        stacking.
    radio_split: QRadioButton
        One of the two buttons. If checked, user is asking for three
        distinct logical starlists.
    """
    def __init__(self, options, parent=None):
        """Create a Option3DPopup window

        This window is a subclass of the QDialog popup.

        Parameters
        ----------
        options: OptionsUI reference
            A reference to the OptionsUI instance that holds all
            option info.
        parent: QWidget
            The parent to this window. The popup should be centered in
            the parent window, if one is specified.
        """
        super().__init__(parent)
        self.options = options

        self.setWindowTitle("image2sl: Stacked Image Options")
        q_btn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        button_box = QDialogButtonBox(q_btn)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        message = QLabel("Choose stacked image processing option")
        layout.addWidget(message)
        button_group = QButtonGroup()
        self.radio_stack = QRadioButton("Combine RGB into one monochrome image")
        self.radio_split = QRadioButton("Separate into R, G, B images")
        button_group.addButton(self.radio_stack)
        button_group.addButton(self.radio_split)
        self.radio_stack.toggled.connect(self.button_change)
        self.radio_split.toggled.connect(self.button_change)
        layout.addWidget(self.radio_stack)
        layout.addWidget(self.radio_split)
        layout.addWidget(button_box)
        self.setLayout(layout)
        self.show()

    def button_change(self):
        """Callback when either of the two radio buttons changes state """
        self.options.split_stacked_image = self.radio_split.isChecked()

def main():
    global ui

    ap = argparse.ArgumentParser(description="Convert an image into a starlist")
    ap.add_argument("--api", help="Run tool using input json instead of GUI")
    args = ap.parse_args()

    if args.api is not None:
        # This is the command-line version of the tool
        # It is not yet implemented
        p = Path(args.api)
        options = OptionsAPI.model_validate_json(p.read_text())
        ui = None
        not_a_window = MainWindow(options)
    else:
        app = QtWidgets.QApplication(sys.argv)
        ui = UI()
        ui.window.show()
        ui.window.setWindowTitle(f"image2sl version {__version__}")
        not_a_window = MainWindow(OptionsUI(), ui=ui)

        ui.window.progressBar.hide()
        ui.window.GenerateStarlistButton.clicked.connect(not_a_window.do_generate_starlist)
        ui.window.actionEnter_astrometry_net_API_key.triggered.connect(not_a_window.get_key)

        settings = QSettings("AAVSO_STWG", "image2sl")
        ui.window.bias_entry.setText(settings.value("bias", ""))
        ui.window.dark_entry.setText(settings.value("dark", ""))
        ui.window.flat_entry.setText(settings.value("flat", ""))
        ui.window.meta_entry.setPlainText(settings.value("metas", ""))
        ui.window.image_filename_list.setPlainText(settings.value("images", ""))

        appx= app.exec()
        
        settings.setValue("bias", ui.window.bias_entry.text())
        settings.setValue("dark", ui.window.dark_entry.text())
        settings.setValue("flat", ui.window.flat_entry.text())
        settings.setValue("metas", ui.window.meta_entry.toPlainText())
        settings.setValue("images", ui.window.image_filename_list.toPlainText())

        sys.exit(appx)


if __name__ == "__main__":
    main()
