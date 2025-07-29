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
import statistics
import sys
import tempfile
import warnings
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import numpy as np
from PIL import Image
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.utils.data import get_pkg_data_filename
from astropy.nddata import CCDData
from astropy.wcs import WCS
from photutils import aperture, psf
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from pydantic import BaseModel, ConfigDict
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QFile, QIODevice, QSettings
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QErrorMessage,
    QFileDialog,
    QLabel,
    QMessageBox,
    QRadioButton,
    QVBoxLayout,
)

from .. import __version__
from ..schema_definition import StarList, StarListSet
from . import field_solve, psf_fitting

warnings.filterwarnings('error', category=RuntimeWarning)
ui = None
LOW_SNR = 2.0

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
    try:
        with fits.open(filename, ignore_missing_simple=True) as hdul:
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
            if ('CREATOR' in hdu0h and 'Origin' in hdu0h['CREATOR']) or \
            ('SWCREATE' in hdu0h and 'Origin' in hdu0h['SWCREATE']):
                if hdu0h['NAXIS'] == 3: return ("Origin", "3Dstacked")
                return ("Origin", "bayered")
                # nb   no mono option for Origin

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
            if 'NAXIS' in hdu0h and hdu0h['NAXIS'] == 3:
                return ("other", "3Dstacked")
            if 'NAXIS' in hdu0h and hdu0h['NAXIS'] == 2:
                if 'BAYERPAT' in hdu0h:
                    return ("other", "bayered")
                else:
                    return ("other", "mono")
    except Exception as e:
        print(f"Error reading FITS file {filename}: {e}")
        raise ValueError(f"Unable to determine telescope type from {filename}") 
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
##    telescope_probe - a tuple value returned by probe_file_for_type()
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
                   'egain', # a float, the gain of the camera system, e-/ADU
                   'bayerpat', # a 4-character string (e.g., 'BGGR')
                   'pixscale', # a float, pixel scale *after* debayering, arcsec/pix
                   'refframe', # a string, (e.g., "ICRS")
                   'dec', # a float, nominal declination of image center (deg)
                   'ra', # a float, nominal RA of image center (deg)
                   'fov_rad', # a float, nominal field of view radius (deg) (half the diagonal)
                   'telescope_probe', # a tuple, with telescope_type and image_type
                   'roworder', # a string, bayerpat modifier. "top-down" or "bottom-up"
                   'ybayroff', # an integer, bayerpat modifier. Column shift horizontally, 0 or 1
                   'stack', # integer, number of images stacked in this fits image. default is 1
                   'system_id' # a string, the system identification of the smart telescope, if available
        ]

def get_json_value(data, keys):
    # data is a dictionary that was read from the JSON file
    # keys can be a string with '.' separators
    # this is pretty much a replacement for data[keys] that can handle simple and compound keys

    datav= data.copy()
    for key in keys.split('.'):
        try:
            datav = datav[key]
        except KeyError:
            print(f"WARNING: JSON key '{keys}' not found in metadata")
            return None
    return datav

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
        if key.startswith('#'):
            # do not replace an existing key
            key= key[1:]
            if key in meta_dict and meta_dict[key] is not None:
                print(f"WARNING: {key} | {meta_dict[key]} not replaced with {value}")
                return None
            else:
                print(f"Adding backup value to meta key '{key}' with value '{value}'")
                # fall through to process this new key:value
        if isinstance(value, str) and value.startswith('@'):
            # This is a reference to another key in the existing meta dir file
            self.json[key] = value # show we will get the value from the prior meta
            nv = get_json_value(meta_dict, value[1:]) # show that the fits had the value
            if nv is not None:
                meta_dict[key] = nv 
            return nv
        # look for special processing keys
        # eg "ra": "!RA hr2deg"
        if isinstance(value, str) and value.startswith('!'):
            try:
                tt = value[1:].split()
                if tt[1] == "hr2deg": # convert decimal hours to degrees
                    val = get_json_value(meta_dict, tt[0])
                    if val is not None:
                        nv= float(val) * 15.0 # convert hours to degrees
                elif tt[1] == "refmtDate":
                    # "obs_time": "!StackedInfo.dateTime refmtDate %m-%d-%yB%H_%M_%S"
                    #   B is a blank space
                    d = datetime.strptime(get_json_value(meta_dict, tt[0]), tt[2].replace('B', ' '))
                    nv = d.strftime("%Y-%m-%dT%H:%M:%S")
                elif tt[1] == "index":
                    # eg "tel_firmware" : "!CREATOR index 1"
                    nv = get_json_value(meta_dict, tt[0]).split()[int(tt[2])]
                else:
                    print(f"Unknown processing function for key {key} with value {value}")
                    nv= None
                meta_dict[key] = nv
                return nv
            except Exception as e:
                #raise RuntimeError(f"Error processing key {key} with value {value}: {tt}") from e
                print(f"Error processing key {key} with value {value}: {tt}") 
                return None
                # programmer's note: We don't raise an error here in the expectation 
                # that the basic.json file will be prepared to act as the catch mechanism.
                # If the validation fails, look for these messages in the console log
                # and examine how to improve the basic.json file.
                # Details of the basic.json features are described in
                #    perform_photometry\meta_json_files\meta_notes.txt
        # If we get here, we have a normal key:value pair
        if key in meta_dict and meta_dict[key] is not None:
            print(f"Replacing existing meta key '{key}' value '{meta_dict[key]}' with new value '{value}'")
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
        print("\nThe following metadata key(s) are missing:")
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
    len_bytes = Path(filename).stat().st_size
    # This file should NEVER be more than 10K bytes long. In order to
    # limit cyber vulnerability, we immediately quit if we're given a
    # long file here.
    if len_bytes > 10000:
        print("ERROR: Refusing to read JSON metadata file that exceeds 10K bytes.")
        raise ValueError

    with open(filename, encoding='utf-8') as fp:
        try:
            data = json.load(fp)
        except json.JSONDecodeError:
            print("Parse error reading ", filename)
            raise

        for (keyword, value) in data.items():
            val = meta_validator.add_json_item(keyword, value, meta_dict)
            if val is not None:
                meta_dict[keyword] = val
                print(f"Adding JSON metadata key '{keyword}' with value '{val}'")

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
    telescope_probe = probe_file_for_type(filename) # tuple of telescope_type and image_type
    with fits.open(filename) as hdul:
        hdu0h = hdul[0].header
        meta_dict['telescope_probe'] = telescope_probe 

        # read in the whole header
        for key in hdu0h:
            meta_validator.add_fits_item(key, hdu0h[key])
            meta_dict[key] = hdu0h[key] # copy the value into the meta_dict
            if comment:= hdu0h.comments[key]:
                # If the FITS header has a comment for this key, add it to the meta_dict
                meta_dict[key+"_comment"] = comment
    
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

# Three possibilities:
#    - a single raw RGB Bayered image
#    - a single stacked color channel image
#    - a single stacked luminance image (created from 3Dstacked image)

class StarlistGenerator:
    """Convert an image file into a StarlistSet

    This class does all the heavy lifting for converting an image file
    into a StarListSet. It has two externally-visible methods: a
    constructor and write_starlists(). All of the hard work is done
    within the constructor.

    Attributes
    ----------
    filename : Path
        The original image file
    height, width : int
        The dimensions of the original image
    metadata : dict
        The metadata for this image; the metadata will be (slightly)
        modified by this class.
    options : OptionsAPI or OptionsUI
        Run-time options
    wcs : WCS
        The WCS for the original image; the result of plate-solving
    working_image : np.ndarray
        The 2D array of pixel values
    interactive : bool
        True if this is being run with a GUI
    telescope_type : str
        The telescope type as found by probe_file_for_type()
    image_type : str
        One of the three image types, as found by probe_file_for_type()
    bkgd_mean : float
        A rough estimate of the image's background (sky) ADU level
    background : np.ndarray
        A 2D model of the sky background in the image. This blends
        together the red, green, and blue background into one
        slowly-varying gradient.
    std : float
        The standard deviation of the original image's background. An
        attempt is made to compensate for differing red, green, and
        blue background averages
    noise_bkgd_per_pixel : float
        The noise level of the sky background in e-/pixel
    source_table : astropy.Table
        The table listing all the stars in the image. This is always
        kept as the "best" set of stars (derived from luminance or
        fake monochrome channel).
    starlist_set : schema.StarListSet
        The StarListSet containing all the logical starlists extracted
        from this image.
    fwhm : float
        The best estimate of average FWHM for stars in this image
    field_solver : field_solve.FieldSolver
        The FieldSolver used to plate-solve this image
    """
    def __init__(self, full_path, meta, options, working_image,
                 wcs, interactive, ui, telescope_type, image_type):
        """Construct a GenerateStarlist instance

        Parameters
        ----------
        full_path : str
            The original image file
        meta : dict
            The metadata for this image; the metadata will be (slightly)
            modified by this class.
        options : OptionsAPI or OptionsUI
            Run-time options
        working_image : np.ndarray
            The 2D array of pixel values
        wcs : WCS
            The WCS for the original image; the result of plate-solving
        interactive : bool
            True if this is being run with a GUI
        ui : UI
            The instance of UI if this is being run interactively
        telescope_type : str
            The telescope type as found by probe_file_for_type()
        image_type : str
            One of the three image types, as found by probe_file_for_type()
        """
        print('Starting StarlistGenerator.__init__()')
        self.filename : Path  = full_path
        if working_image.ndim == 3:
            (_, self.height, self.width) = np.shape(working_image)
        else:
            (self.height, self.width) = np.shape(working_image)
        self.metadata : dict = meta
        self.options = options
        self.wcs : WCS = wcs
        self.working_image : np.ndarray = working_image
        self.interactive : bool = interactive
        self.telescope_type = telescope_type
        self.image_type = image_type
        self.bkgd_mean = None
        self.background = None
        self.std = None
        self.noise_bkgd_per_pixel = None
        self.source_table = None
        self.starlist_set = None
        self.fwhm = None
        self.field_solver = field_solve.FieldSolver(meta, ui, options)

        if image_type == 'bayered':
            self.starlist_set = self._process_bayer_file()
        elif image_type == 'mono':
            self.starlist_set = self._process_mono_file()
        elif image_type == '3Dstacked':
            self.starlist_set = self._process_3d_file()
        else:
            raise ValueError("UnknownImageType:"+image_type)

    def write_starlists(self):
        """Store the resulting StarListSet as a file
        """
        output_file = self.filename.with_suffix('.star')
        with open(output_file, 'w', encoding='utf-8') as fp:
            json.dump(self.starlist_set.model_dump(), fp, indent=2)
        print('All starlists written.')

    def _process_3d_file(self):
        """Process a stacked image with r, g, and b layers

        The three layers are stacked into a luminance layer. Star
        centroids are established from that layer. It becomes the L3
        starlist. The same set of centroids is used for
        photometry of each of the three separate layers; those three
        become TR, TG, and TB starlists.

        Returns
        -------
        StarListSet
            The StarListSet holding all four starlists.

        Bugs
        ----
            1. Assumes that all stacked images use the same format as the
        Seestar S50. When we find something that works differently,
        this will need to change.
            2. Does not perform PSF fitting photometry
        """
        assert self.working_image.ndim == 3
        image1 = self.working_image[0,0:,0:]
        image2 = self.working_image[1,0:,0:]
        image3 = self.working_image[2,0:,0:]

        # We know the following mapping is correct for Seestar. Need
        # examples from other vendors, though.
        image_list = [(image1, 'TR'),
                      (image2, 'TG'),
                      (image3, 'TB')]

        sum_image = image1 + image2 + image3
        sum_copy_image = sum_image.copy()
        self.source_table = self._find_sources(sum_copy_image)
        self.source_table = self._do_photometry(sum_image, self.source_table)
        self.wcs = self._setup_wcs(self.source_table, self.wcs)
        self.metadata['filter'] = 'L3'
        for key in ['height', 'width', 'fwhm']:
            self.metadata[key] = getattr(self, key)

        # This needs to be properly added to metadata, which is probably different
        # for each telescope.
        starlist = StarList.from_table(self.source_table, metadata=self.metadata)
        final_starlists = [starlist]

        for (image,color) in image_list:
            copy_source_table = self.source_table.copy()
            copy_source_table = self._do_photometry(image, copy_source_table)
            self.metadata['filter'] = color
            starlist = StarList.from_table(copy_source_table, metadata=self.metadata)
            final_starlists.append(starlist)

        return StarListSet(star_lists=final_starlists)

    def _process_mono_file(self):
        """Process a single (monochrome) image

        The single image will have stars identified, the image will be
        plate-solved, and photometry will be done on each star.

        Returns
        -------
        StarListSet
            A StarListSet containing a single StarList

        Bugs
        ----
        Does not perform PSF fitting properly
        """
        self.source_table = self._find_sources(self.working_image)
        self.source_table = self._do_photometry(self.working_image, self.source_table)
        self.wcs = self._setup_wcs(self.source_table, self.wcs)
        starlist = StarList.from_table(self.source_table, metadata=self.metadata)

        ################################
        ## Do PSF fitting, if requested
        ################################
        if self.options.use_psf_fitting:
            starlist.staritems.sort(key=lambda star:
                                    star.tot_flux, reverse=True)
            raise NotImplementedError("PSF fitting not implemented yet")
            # psf_builder.add_image(self.working_image,
            #                       self.metadata,
            #                       self.noise_bkgd_per_pixel,
            #                       starlist)

        return StarListSet(star_lists=[starlist])

    def _process_bayer_file(self):
        """Process a single raw (Bayered) image

        A fake monochrome image will be used to create a list of
        stars. That list will be used for photometry 4 times: once on
        the fake monochrome image and once each on the original image
        with only red, green, or blue pixels masked "true". The four
        resulting starlists will be packed into a StarListSet

        Returns
        -------
        StarListSet
            A StarListSet containing the four resulting StarLists

        Bugs
        ----
            Does not perform PSF fitting photometry
        """
        pattern = self.metadata['bayerpat']

        # now re-jumble based on roworder and ybaryoff
        print('extract_mono_and_rgb: initial pattern = ', pattern)
        if self.metadata['roworder'] == 'bottom-up':
            pattern = pattern[2:3] + pattern[0:1]
        if self.metadata['ybayroff'] != 0:
            pattern = pattern[1] + pattern[0] + pattern[3] + pattern[2]

        print('adjusted pattern = ', pattern)
        img_slice = {}
        img_slice[0] = (0, 0)
        img_slice[1] = (0, 1)
        img_slice[2] = (1, 0)
        img_slice[3] = (1, 1)

        bayer_info = [] # list of tuples (filter, img_mask)
        total_pixels = self.height * self.width
        for color in ['R', 'B', 'G']:
            # In the mask, True means masked/ignore; False means yes/use/valid
            img_mask = np.ones((self.height, self.width), dtype=bool)
            for channel in range(4):
                if pattern[channel] == color:
                    slicer = img_slice[channel]
                    img_mask[slicer[0]::2, slicer[1]::2] = False

            print('Color ', color, ' has ',
                  total_pixels - np.count_nonzero(img_mask),
                  'usable cells')
            bayer_info.append(('T'+color, img_mask))

        # Make a copy, since we're going to adjust pixels to get best
        # star detection & centroids
        full_image = np.copy(self.working_image)
        self.source_table = self._find_sources(full_image, do_color_balance=True)
        self.source_table = self._do_photometry(self.working_image, self.source_table)
        print('...top level has ', len(self.source_table), ' stars.')
        self.wcs = self._setup_wcs(self.source_table, self.wcs)
        self.metadata['filter'] = 'L4'
        for key in ['height', 'width', 'fwhm']:
            self.metadata[key] = getattr(self, key)

        self.metadata['stack'] = None
        starlist = StarList.from_table(self.source_table, metadata=self.metadata)
        final_starlists = [starlist]

        for (color, img_mask) in bayer_info:
            copy_source_table = self.source_table.copy()
            copy_source_table = self._do_photometry(self.working_image,
                                                    copy_source_table,
                                                    image_mask=img_mask)
            print('...top level has ', len(copy_source_table), ' stars.')
            self.metadata['filter'] = color
            starlist = StarList.from_table(copy_source_table, metadata=self.metadata)
            final_starlists.append(starlist)

        return StarListSet(star_lists=final_starlists)

    def _bayer_balance_image(self, image):
        """adjust pixel values per the Bayer pattern

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

        #print_img_stats(temp1x.flatten())
        #print_img_stats(temp2x.flatten())
        #print_img_stats(temp3x.flatten())
        #print_img_stats(temp4x.flatten())
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

        #print_img_stats(temp1x.flatten())
        #print_img_stats(temp2x.flatten())
        #print_img_stats(temp3x.flatten())
        #print_img_stats(temp4x.flatten())

        image[0::2,0::2] = temp1
        image[0::2,1::2] = temp2
        image[1::2,0::2] = temp3
        image[1::2,1::2] = temp4

    def _remove_background(self, image, do_color_balance=False):
        """Calculate and remove the background from an image

        Parameters
        ----------
        image : np.ndarray
            The image. The background will be estimated and then
            subtracted from each pixel
        do_color_balance : bool
            If True, the four pixel color channels will be adjusted
            with a linear transformation to achieve a flat gray
            background that has the same noise level in each color
            channel.
        """
        egain = self.metadata['egain']

        if do_color_balance:
            self._bayer_balance_image(image)
        (self.bkgd_mean,
         median,
         self.std) = sigma_clipped_stats(image, sigma=3.0)
        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground()
        full_background = Background2D(image,
                                       (int(self.width/8),int(self.height/8)),
                                       filter_size=(3,3),
                                       exclude_percentile=80,
                                       sigma_clip=sigma_clip,
                                       bkg_estimator=bkg_estimator)
        background = full_background.background
        image -= background
        (self.bkgd_mean,
         _dummy,
         bkgd_std) = sigma_clipped_stats(background, sigma=3.0)
        print("background.median = ", full_background.background_median,
              ", background.rms = ", full_background.background_rms_median)
        self.noise_bkgd_per_pixel = full_background.background_rms_median * egain
        self.background = background

    def _find_sources(self, working_image, do_color_balance=False):
        """Find the stars in an image

        Parameters
        ----------
        working_image : np.ndarray
            The image to be searched

        Returns
        -------
        astropy.Table
            A table of the stars found
        """
        # We find stars twice. First time (here) is sloppy and will
        # probably miss lots of stars because of combination of incorrect
        # FWHM and using default star shape thresholds. Doing this first
        # find anyway just to get better handle on FWHM and to extract
        # image statistics in the process.

        local_image = working_image.copy()
        self._remove_background(local_image, do_color_balance=do_color_balance)

        daofind = DAOStarFinder(fwhm=3.0, threshold=4.*self.std)
        sources = daofind(local_image)
        print("Initial quicklook found ", len(sources), " stars.")
        # Sort the table in-place by flux in reverse order
        sources.sort('flux', reverse=True)

        # Exclude rows where flux is saturated
        mask = (
            (sources['peak'] > self.metadata['largest_usable_adu_value'])
            | (sources['xcentroid'] < 3.0)
            | (sources['ycentroid'] < 3.0)
            | (sources['flux'] < 0.0)
        )
        sources = sources[~mask]
        print("after removal of saturated/poor stars, the count is ", len(sources), " stars.")

        # Grab a subset of the brightest stars to estimate the FWHM
        subset_size = min(10, len(sources))
        if subset_size == 0:
            print('No stars. Cannot estimate FWHM.')
            if ui is not None:
                msg = QErrorMessage()
                msg.showMessage(
                    f'No stars: no FWHM for file: {self.metadata["filename"]}')
                msg.exec()
            raise ValueError('NoStarsFound')
        subset = sources[:subset_size]
        fwhm = psf.fit_fwhm(local_image,
                            xypos=list(zip(subset['xcentroid'],
                                           subset['ycentroid'],
                                           strict=True)),
                            fit_shape=15).mean()
        print("Estimate FWHM from photutils = ", fwhm)
        self.metadata['fwhm'] = fwhm
        self.fwhm = fwhm

        # Now that we know the *real* FWHM, re-find the stars
        daofind = DAOStarFinder(fwhm=fwhm, threshold=4.0*self.std,
                                sharplo=0.05, sharphi=3.0,
                                roundlo=-4.0, roundhi=4.0)
        sources = daofind(local_image)
        print("Sources found before edge-culling: ", len(sources), " stars.")
        sources.rename_column('peak', 'peak_count')

        # eliminate stars too close to the edges
        edgelimit = 15
        mask = np.array([row['xcentroid'] < edgelimit
                         or row['xcentroid'] > self.width-edgelimit
                         or row['ycentroid'] < edgelimit
                         or row['ycentroid'] > self.height-edgelimit
                         for row in sources])
        sources = sources[~mask]
        sources.sort('flux', reverse=True)
        return sources

    def _do_photometry(self, working_image, sources, image_mask=None):
        """Perform photometry on a set of sources in an image

        Aperture photometry will be performed, using a sky annulus to
        determine the (residual) sky background for each star. The
        input "sources" table is both an input and output. Centroid
        locations will be taken from the table, while flux and
        background values will be written into the table.

        Parameters
        ----------
        working_image : np.ndarray
            The image
        sources: astropy.Table
            The table containing x- and y-coordinate centroids for
            each star to be measured. (Any existing flux or background
            values will be overwritten)
        image_mask : np.ndarray of bool
            Only pixels with a value of True will be used in the
            photometry. Must have the same shape as working_image.

        """
        egain = self.metadata['egain']
        phot_radius = self.options.aperture_size_fwhm * self.fwhm
        annulus_inner = max(3*phot_radius, 4*self.fwhm)
        annulus_outer = math.sqrt(100*phot_radius**2 + annulus_inner**2)
        print(f"Aperture radius = {phot_radius:.2f} , with {math.pi * phot_radius * phot_radius:.2f} pixels total")

        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground()
        full_background = Background2D(working_image,
                                       (int(self.width/8),int(self.height/8)),
                                       filter_size=(3,3),
                                       exclude_percentile=80,
                                       sigma_clip=sigma_clip,
                                       bkg_estimator=bkg_estimator)
        # noise_bkgd_per_pixel in units of e-/pixel
        noise_bkgd_per_pixel = full_background.background_rms_median * egain

        # Perform the photometry
        positions = list(zip(sources['xcentroid'],
                             sources['ycentroid'], strict=False))
        apertures = aperture.CircularAperture(positions, r=phot_radius)
        # Notice! tot_noise_bkgd is in units of electrons
        tot_noise_bkgd = np.sqrt(apertures.area) * noise_bkgd_per_pixel

        annuli = aperture.CircularAnnulus(positions, annulus_inner, annulus_outer)
        annulus_sigma_clip = SigmaClip(sigma=2.0)
        annulus_data = aperture.ApertureStats(working_image,
                                              annuli,
                                              sigma_clip=annulus_sigma_clip,
                                              mask=image_mask,
                                              sum_method='center')

        central_sum = aperture.ApertureStats(working_image,
                                             apertures,
                                             sum_method='exact',
                                             mask=image_mask,
                                             local_bkg=annulus_data.mean)
        sources['x'] = sources['xcentroid']
        sources['y'] = sources['ycentroid']
        sources['tot_count'] = central_sum.sum
        if 'bkgd_count' not in sources.columns:
            sources.add_column(annulus_data.mean, name='bkgd_count')
        else:
            sources['bkgd_count'] = annulus_data.mean
        sources['peak_count'] = central_sum.max + annulus_data.mean

        bad_rows = []
        min_adu = 1.0 # max(0.0, tot_noise_bkgd/starlist.gain)
        # Clean up the sources table
        print("Sources cleanup starts with ", len(sources), " stars.")
        print('   ... and min_adu of ', min_adu, ' and egain = ', egain)
        print('   ... and smallest peak_count of ', min(sources['peak_count']))
        print('   ... and smallest tot_count of ', min(sources['tot_count']))

        for row,content in enumerate(sources):
            if (content['x'] <= 3.0
                or content['y'] <= 3.0
                or content['x'] >= (self.width-3)
                or content['y'] >= (self.height-3)
                or content['tot_count'] <= min_adu
                or content['peak_count'] <= min_adu):
                bad_rows.append(row)
        print("... removing ", len(bad_rows), " stars.")
        sources.remove_rows(bad_rows)
        print("... now have ", len(sources), " stars.")

        # Turn this on to see the original image with "valid" stars
        # circled. You'll probably need to adjust the 600/1600 in imshow.
        if False:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
            fig,ax = plt.subplots(1)
            ax.imshow(self.working_image, cmap='Greys', vmin=600.0, vmax=1600.0)
            for row in self.sources:
                ax.add_patch(Circle((row['x'],row['y']), 8, fc=None, fill=False))
            plt.show()

        # Sort so that order is well-defined and tests will pass
        sources.sort(keys='tot_count', reverse=True)

        # Calculate errors using table columns and star flux error in column
        poiss_noise = np.sqrt(egain * sources['tot_count'])
        tot_noise = np.sqrt(poiss_noise**2 + tot_noise_bkgd**2) / egain
        sources['count_err'] = tot_noise

        # Set flux errors to zero for negative fluxes
        sources['count_err'][sources['tot_count'] < 0] = 0.0

        # Calculate SNR and drop stars with low SNR or with negative flux
        snr = sources['tot_count'] / sources['count_err']
        good_snr = (
            (snr > LOW_SNR)              # Only keep stars with decent SNR
            & ~np.isnan(snr)             # Drop any nan SNRs, likely from count_err=0
            & (sources['tot_count'] > 0)  # Drop any negative fluxes, which are unphysical
        )

        return sources[good_snr]

    def _setup_wcs(self, sources, wcs):
        """Plate-solve and set RA/Dec in the sources table

        Parmaters
        ---------
        sources : astropy.Table
            List of stars in the image
        wcs : WCS
            If not None, this WCS will prevent plate-solving from
            being performed.

        Returns
        -------
        WCS
            The WCS used to set Dec/RA for the stars.
        """
        wcs = self.field_solver.solve(sources, self.width, self.height, source_wcs=wcs)
        if wcs is None:
            print('field_solver failed to solve the field.')
            if self.interactive:
                msg = QErrorMessage()
                msg.showMessage(
                    f'Field_solver failed to solve the field for file: {self.filename}')
                msg.exec()
            raise ValueError('FieldSolver Failed')
        return wcs

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
                 multiple_files_okay=False,
                 last_directory=None):
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
        last_directory : str, optional, default=None
            The last directory used to select a file. If None, the
            directory will be the one in the text_entry_widget, if
            there is one.

        Returns
        -------
        FileChooser object
        """
        self.text_widget = text_entry_widget
        self.popup_button = chooser_button
        self.multiple_files_okay = multiple_files_okay
        self.last_directory = last_directory
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
        if self.multiple_files_okay:
            if self.last_directory is None:
                if pt := self.text_widget.toPlainText():
                    pt = pt.split('\n', 1)[0]
                    self.last_directory = str(Path(pt).parent)
        else:
            if self.last_directory is None:
                if pt := self.text_widget.text():
                    self.last_directory = str(Path(pt).parent)
        dialog.setDirectory(self.last_directory)
        if dialog.exec():
            if self.multiple_files_okay:
                # Now append to the filelist
                entry_list = dialog.selectedFiles()
                for entry in entry_list:
                    self.text_widget.appendPlainText(entry + '\n')
            else:
                self.text_widget.setText(dialog.selectedFiles()[0])
            self.last_directory = dialog.directory().path()
        else:
            self.text_widget.clear()

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
        text_words = [word.strip() for word in text_words if word.strip()]
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


class PhotometryMethods(StrEnum):
    """Enumeration of photometry methods"""
    # See the python documentation for Enums work. Behind the scenes pythopn
    # creates a class whose attributes have the names listed below.
    APERTURE = "aperture"
    PSF = "psf"


class OptionsAPI(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_default=True, validate_assignment=True)
    aperture_size: float = 1.0
    photometry_method: PhotometryMethods = PhotometryMethods.APERTURE
    astrometry_net_api_key: str = ""
    bias_file: str = ""
    dark_file: str = ""
    flat_file: str = ""
    meta_file: list[str] = [""]
    image_file: list[str] = [""]

    @property
    def use_psf_fitting(self):
        return self.photometry_method == PhotometryMethods.PSF

    @property
    def aperture_size_fwhm(self):
        return self.aperture_size

    @classmethod
    def from_ui(cls, ui):
        """Create an OptionsAPI object from a UI object

        Create an OptionsAPI object from a UI object. This is used to
        create the API-based options object from the UI-based options
        object.

        Parameters
        ----------
        ui : OptionsUI
            The UI object that holds the options

        Returns
        -------
        OptionsAPI
            The API-based options object
        """
        return cls(
            aperture_size=ui.aperture_size_fwhm,
            photometry_method=PhotometryMethods.PSF if ui.use_psf_fitting else PhotometryMethods.APERTURE,
            bias_file=ui.bias_file if ui.bias_file else "",
            dark_file=ui.dark_file if ui.dark_file else "",
            flat_file=ui.flat_file if ui.flat_file else "",
            meta_file=ui.meta_file,
            image_file=ui.image_file
        )


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
        self._bias_file = FileChooser(self.window.bias_entry,
                                      self.window.BiasButton)
        self._dark_file = FileChooser(self.window.dark_entry,
                                      self.window.DarkButton)
        self._flat_file = FileChooser(self.window.flat_entry,
                                      self.window.FlatButton)
        self._meta_file = FileChooser(self.window.meta_entry,
                                      self.window.MetaButton,
                                      multiple_files_okay=True)
        self._image_file = FileChooser(self.window.image_filename_list,
                                       self.window.AddImageButton,
                                       multiple_files_okay=True)
        self.psf_photometry = self.window.PSFPhotButton
        self.aperture_photometry = self.window.AperturePhotButton

        self.aperture_size = self.window.ApertureSize

    @property
    def use_psf_fitting(self):
        """Query whether PSF-fitting photometry is to be done

        Return a boolean indicating whether
        PSF-fitting photometry is to be done. This method is 100%
        redundant with get_phot() and should be retired.

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
class MainWindow:
    def __init__(self, options, ui=None):
        """Set up main display window and key singleton objects

        Create the FileChooser objects for each file chooser button in
        the display. Create a temporary directory in which working
        files can be stored (and destroyed when the program
        exits). Create singleton objects for OptionBox and the
        ProgressBar that is displayed while generation of the
        starlists is performed.

        Parameters
        ----------
        options : an OptionsAPI object or None
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
        temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir = temp_dir
        self.temp_dirname = self.temp_dir.name
        print("Working in temporary directory ", self.temp_dirname)
        print(f"===> st-pipeline version {__version__} <===")
        self.options = options
        self.ui = ui
        self._wcs = None
        self.have_ui = False
        if self.ui:
            self.have_ui = True
            self.progressbar = self.ui.window.progressBar
        else:
            self.generate_starlist()

    def get_key(self):
        dialog = field_solve.APIEntryDialog(self.ui.window)
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
        self.options = OptionsAPI.from_ui(self.ui)
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

        for image_filename in image_list:
            #QGuiApplication.processEvents()
            #Skip blank lines (if present)
            if image_filename is None:
                continue
            image_filename = image_filename.strip()
            if image_filename == '':
                continue

            telescope_type, image_type = probe_file_for_type(image_filename)

            ccd = CCDData.read(image_filename, unit="adu")
            if ccd.wcs is None:
                # 3D stacked files get read in without a WCS. Try a little
                # harder to get the WCS from the header.
                wcs = WCS(fits.getheader(image_filename), naxis=2)
                ccd.wcs = wcs if wcs.wcs.ctype[0] else None

            working_image = ccd.data.astype(float)
            self._wcs = ccd.wcs

            if (dark_filename
                or flat_filename
                or bias_filename):
                if telescope_type[1] == "3Dstacked":
                    print("Cannot calibrate a 3D stacked image")
                    continue
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
                        condition = flat == 0.0
                        flat[condition] = np.median(flat)
                        flat_median = np.median(flat)
                        flat = flat.astype(float) / flat_median
                        working_image /= flat
                        print('Flat correction applied.')

            meta = {} # This is the metadata dictionary

            # Order matters here. The standalone metadata file is to override
            # whatever is found in the FITS header of the image file
            print("Reading metadata from FITS header")
            read_meta_from_fits(image_filename, meta)

            #QGuiApplication.processEvents()
            personal_json= '' # no personal.json file by default
            # Now get the metadata from the standalone metadata file (sidecar)
            for metadata_filename in metadata_list:
                if metadata_filename is not None and metadata_filename != '':
                    meta_path = Path(metadata_filename)
                    if not (meta_path.is_file() and meta_path.exists() and
                            meta_path.stat().st_mode & 0o400):
                        print("Cannot read metadata from file ", metadata_filename)
                        raise ValueError("Cannot read metadata file")
                    # check the meta_file for a DoLast instruction, making it a personal.json file
                    with open(metadata_filename, 'r', encoding='utf-8') as meta_file:
                        if "DoLast" in meta_file.read():
                            personal_json= metadata_filename
                            continue # do not process this file yet
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

            #   look for and apply the personal.json
            if personal_json != '':
                print("Reading personal.json from ", personal_json)
                read_meta_from_json(personal_json, meta)

            print("Final metadata is: ")
            lback= False
            for key, value in meta.items():
                if "comment" in key:
                    print(f" ({key}: {value})")
                    lback= False
                else:
                    newline_or_not = '\n' if lback else ''
                    print(f"{newline_or_not}{key}: {value}", end="")
                    lback= True

            wcs = self._wcs.copy() if self._wcs is not None else None
            if meta_validator.validate(meta):
                # at this point the meta dictionary is complete, same as meta_validator.final
                meta['orig_filename'] = image_filename
                starlist_gen = StarlistGenerator(full_path = Path(image_filename),
                                                 meta = meta,
                                                 options = self.options,
                                                 working_image = working_image,
                                                 wcs = wcs,
                                                 interactive = self.have_ui,
                                                 ui = self.ui,
                                                 telescope_type = telescope_type,
                                                 image_type = image_type)
                starlist_gen.write_starlists()


        # Now that all images have been processed, let the psf_fitter
        # perform PSF photometry. If the option was not turned on,
        # this will return quietly without doing anything.
        psf_builder.build_psf()

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
        not_a_window = MainWindow(None, ui=ui)

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
