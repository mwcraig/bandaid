# MIT License
#
# Copyright (c) 2024 Mark J Munkacsy

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

from astropy.io import fits
from astropy.wcs import WCS
from photutils import background, detection, aperture, psf
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.table import Table
from astroquery.astrometry_net import AstrometryNet

import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from pydantic import BaseModel, ConfigDict

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

import threading
from pathlib import Path
import statistics
import numpy as np
import tempfile
import getopt
import datetime
import math
import sys
import json
import platform
from collections import namedtuple
import argparse
from typing import List
#import sep

from st_pipeline.perform_photometry import psf_fitting
from .. import __version__
from ..schema_definition import StarListSet

astrometry_api_key = None

################################################################
##        Algorithmic Stuff Comes First
################################################################

def DeBayerFile(filename, pattern, temp_dir):
    """Split an RGB image into four images, one for each Bayer channel
    The four channels are extracted using a string description of the
    Bayer sequence (e.g., 'BGGR'); each extracted sub-image is stored
    as a new FITS image file.
    Parameters
    ----------
    filename : str
        pathname to the original image to be de-Bayered
    pattern : str exactly 4 chars long
        The Bayer pattern (e.g., 'BGGR')
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
        temp2 = hdul[0].data[1::2,0::2]
        temp3 = hdul[0].data[0::2,1::2]
        temp4 = hdul[0].data[1::2,1::2]
        array = [temp1, temp2, temp3, temp4]
        output_filenames = [] # each entry in this list is a tuple: (filter, filename)

        for index in range(4):
            color = pattern[index]
            output_tgt = Path(temp_dir) / ("image"+str(index)+"_"+color+".fits")

            hdu = fits.PrimaryHDU()
            # push keywords in from the original file
            for keyword in hdul[0].header:
                if keyword != 'COMMENT' and keyword != 'HISTORY':
                    value = hdul[0].header[keyword]
                    comment = hdul[0].header.comments[keyword]
                    hdu.header[keyword] = (value,comment)
                else: # Yes, this is a comment/history
                    comment = hdul[0].header[keyword]
                    for card in comment:
                        hdu.header[keyword] = card

            hdu.data = array[index]
            hdu.header['filter'] = ('T'+color, 'Bayer color mask')
            # update_header will "fix" the header to match the data
            hdu.update_header()
            fits.writeto(output_tgt, array[index], header=hdu.header, overwrite=True)
            ImageDescriptor = namedtuple('ImageDescriptor', ['filter','filename'])
            output_filenames.append(ImageDescriptor(color,output_tgt))
        return output_filenames

# pattern triplet: (x_offset, y_offset, weight)
pattern = [
    [ (0,0,9), (0,1,3),  (1,0,3),  (1,1,1)   ], # color 0
    [ (0,0,9), (-1,0,3), (0,1,3),  (-1,1,1)  ], # color 1
    [ (0,0,9), (1,0,3),  (0,-1,3), (1,-1,1)  ], # color 2
    [ (0,0,9), (-1,0,3), (0,-1,3), (-1,-1,1) ]  # color 3
]

def StackImages(channel_list, options, temp_dir):
    """ Create a stacked image from 4 individual Bayer sub-images

    Four Bayer sub-images are stacked into a single image. If
    `options` includes the InterpolateChannels flag, the sub-images
    will be shifted as they are added, recognizing the offsets between
    the locations of the different Bayer colors. The resulting sum
    image will be stored as a new FITS file with floating point pixel
    values (to avoid overflow issues as the four pixel values are
    added). The resulting FITS file will have all the FITS
    keyword/value pairs found in the first sub-image, except that the
    FILTER keyword will be set to 'CV'.

    Parameters
    ----------
    channel_list : list of named tuples (taken from DeBayerFile)
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
            if keyword != 'COMMENT' and keyword != 'HISTORY':
                value = hdul[0].header[keyword]
                comment = hdul[0].header.comments[keyword]
                hdu.header[keyword] = (value,comment)
            else: # Yes, this is a comment/history
                comment = hdul[0].header[keyword]
                for card in comment:
                    hdu.header[keyword] = card
    hdu.data = np.zeros((height,width),dtype=np.float32)
    for (bayer_id,(filter,channel)) in enumerate(channel_list):
        with fits.open(channel) as hdul:
            source_hdu = hdul[0].data
            if options.InterpolateChannels:
                new_data = np.zeros(np.shape(hdul[0].data),dtype=np.float32)
                orig_data = source_hdu.astype(np.float32)
                source_hdu = new_data
                for y in range(height-1):
                    for x in range(width-1):
                        tgt = sum((p[2]*orig_data[y+p[1],x+p[0]] for p in pattern[bayer_id]))
                        new_data[y,x] = tgt

            hdu.data += source_hdu/16.0
    hdu.header['filter'] = 'CV'
    hdu.update_header()
    fits.writeto(output_tgt, hdu.data, header=hdu.header, overwrite=True)
    return output_tgt

def DuplicateFileWithNewImage(hdul, new_data, new_filter, new_pathname):
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
    output_tgt = new_pathname

    hdu = fits.PrimaryHDU()
    # push keywords in from the original file
    for keyword in hdul[0].header:
        if keyword != 'COMMENT' and keyword != 'HISTORY':
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

def BayerBalanceFile(filename, temp_dir):
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
    temp_dir : str
        Pathname of the directory where the new file will be placed

    Returns
    -------
    str
        The full pathname of the new file.
    """
    with fits.open(filename) as hdul:
        temp1 = hdul[0].data[0::2,0::2].astype(np.float32)
        temp2 = hdul[0].data[1::2,0::2].astype(np.float32)
        temp3 = hdul[0].data[0::2,1::2].astype(np.float32)
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

        temp1x = np.array([x for x in temp1.flatten() if x >= 0 and x < cutoff])
        temp2x = np.array([x for x in temp2.flatten() if x >= 0 and x < cutoff])
        temp3x = np.array([x for x in temp3.flatten() if x >= 0 and x < cutoff])
        temp4x = np.array([x for x in temp4.flatten() if x >= 0 and x < cutoff])

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
        new_data[1::2,0::2] = temp2
        new_data[0::2,1::2] = temp3
        new_data[1::2,1::2] = temp4

        new_filename = filename.replace(".fit","_M.fit")
        DuplicateFileWithNewImage(hdul, new_data, "M", new_filename)
        return new_filename

def ProbeFileForType(filename):
    """Figure out what kind of a smart telescope created an image

    Examine the FITS header keywords to determine what kind of smart
    telescope created the image.

    Parameters
    ----------
    filename : str
        The pathname of the image to be examined

    Returns
    -------
    str
        One of the following: "Unistellar", "Seestar", "Origin", "Dwarf"

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
            return "Unistellar"

        ################################
        ## Seestar test
        ################################
        if 'CREATOR' in hdu0h and 'Seestar' in hdu0h['CREATOR']:
            return "Seestar"

        ################################
        ## Celestron Origin test
        ################################
        if 'CREATOR' in hdu0h and 'Origin' in hdu0h['CREATOR']:
            return "Origin"

        ################################
        ## DWARF
        ################################
        if 'ORIGIN' in hdu0h and 'DWARFLAB' in hdu0h['ORIGIN']:
            return "Dwarf"

        ################################
        ## Unrecognized
        ################################
        print("Unable to figure out Smart Telescope Type for file ", filename);
        raise ValueError("Unable to establish telescope type")

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
################################################################

valid_meta_keys = ['schema_version',
                   'obs_time',
                   'site_lat',
                   'site_lon',
                   'site_elev',
                   'observer',
                   'filter',
                   'block_filter',
                   'exposure',
                   'tel_manufac',
                   'tel_model',
                   'tel_firmware',
                   'adc_depth',
                   'largest_usable_adu_value',
                   'system_gain',
                   'BAYERPAT',
                   'pixscale',
                   'dec',
                   'ra',
                   'fov_rad' ]

#
# The so-called JSON metadata file is a temporary band-aid for smart
# telescopes that are currently missing important FITS header
# keywords. (Early Origin scopes have this problem.) The JSON metdata
# file is merged with the metadata that comes from the FITS header,
# providing a way to deal with missing/incorrect FITS header info.
#

def ReadMetaFromJSON(filename, dict):
    """Pull metadata from a JSON metadata file

    Update a meta dictionary using the contents of the JSON metadata
    file to augment or replace entries in the meta
    dictionary. Metadata keywords are validated as they are
    encountered; unrecognized keywords generate a console message.

    Parameters
    ----------
    filename : str
        Pathname of the JSON metadata file
    dict : dictionary
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

    with open(filename, 'r') as fp:
        try:
            data = json.load(fp)
        except JSONDecodeError:
            print("Parse error reading ", filename)
            raise

        for (keyword,value) in data.items():
            if keyword not in valid_meta_keys:
                print("Bad keyword in ", filename, ": ", keyword)
            else:
                dict[keyword] = value


# Read metadata from a FITS header. The metadata that's found will be
# put into the dictionary that's passed as the argument "dict".
def ReadMetaFromFITS(filename, dict):
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
    dict : dictionary
        Metadata dictionary to be modified/augmented

    Returns
    -------
    None
    """
    telescope_type = ProbeFileForType(filename)
    with fits.open(filename) as hdul:
        hdu0h = hdul[0].header

        # Generate Metadata
        ################################
        ##          Seestar
        ################################
        if telescope_type == 'Seestar':
            for key,tgt in [('BAYERPAT','BAYERPAT'),
                            ('DATE-OBS','obs_time'),
                            ('filter','block_filter')]:
                if key in hdu0h:
                    dict[tgt] = hdu0h[key]
            ################
            ## FOV, Pixel scale
            ################
            dict['fov_rad'] = 0.7 # wild guess; needs validation

            for key,tgt in [('sitelat', 'site_lat'),
                            ('sitelong', 'site_lon'),
                            ('dec', 'dec'),
                            ('ra', 'ra'),
                            ('exposure', 'exposure')]:
                if key in hdu0h:
                    dict[tgt] = hdu0h[key]

            dict['tel_manufac'] = 'ZWO'
            dict['tel_model'] = 'Seestar'

        ################################
        ##       Origin
        ################################
        elif telescope_type == 'Origin':
            dict['BAYERPAT'] = 'RGGB' # Is this correct?
            dict['tel_manufac'] = 'Celestron'
            dict['tel_model'] = 'Origin'
            dict['adc_depth'] = 14
            dict['pixscale'] = 1.45*2 # after de-bayering

        ################################
        ##       Unistellar
        ################################
        elif telescope_type == 'Unistellar':
            for key,tgt in [('BAYERPAT','BAYERPAT'),
                            ('DATE-AVG','obs_time'),
                            ('EXPTIME','exposure'),
                            ('LATITUDE','site_lat'),
                            ('LONGITUD', 'site_lon'),
                            ('ALTITUDE', 'site_elev'),
                            ('FOVDEC','dec')]:
                if key in hdu0h:
                    dict[tgt] = hdu0h[key]
            dict['tel_manufac'] = 'Unistellar'
            dict['tel_model'] = 'eVscope'
            dict['pixscale'] = 1.5*2 # after de-bayering

        ################################
        ##        Dwarf
        ################################
        elif telescope_type == "Dwarf":
            for key,tgt in [('BAYERPAT','BAYERPAT'),
                            ('EXPTIME','exposure'),
                            ('DATE-OBS','obs_time'),
                            ('INSTRUME','tel_model'),
                            ('ra','ra'),
                            ('dec','dec')]:
                if key in hdu0h:
                    dict[tgt] = hdu0h[key]
            dict['tel_manufac'] = 'DwarfLab'
            dict['pixscale'] = 1.5*2 # after de-bayering

        else:
            print("Telescope type ", telescope_type, " not implemented yet.")


class AAVSOStarlist:
    """Implement an AAVSO Starlist

    This class "is" an AAVSO Starlist. Initial condition: all metadata
    is populated, but the starlist is otherwise empty.

    Attributes
    ----------
    metadata : dict
        Dictionary containing "extended" metadata
    gain : float
        System gain (e-/ADU) in the original (Bayered) image
    filter : str
        Two-letter filter name (e.g., 'TG')
    starlist : dict
        This dict is a literal image of the official AAVSO starlist,
        including both  metadata and the stars of the starlist
    """
    def __init__(self, metadata, filter):
        """Create a starlist and populate with metadata

        Create a starlist and pull metadata from a `metadata`
        dictionary that is put into the starlist object. (The stars
        themselves will be populated with a later invocation of one of
        the ReadFrom...() member functions.)

        Parameters
        ----------
        metadata : dict
            A dictionary of metadata. See the earlier block comment
            for a full description
        filter : str
            Name of the filter for this starlist

        Returns
        -------
        AAVSOStarlist
            A new object of the class
        """
        self.metadata = metadata

        self.starlist = {} # JSON-style dictionary
        self.starlist['gain'] = metadata['system_gain']
        self.starlist['filter'] = filter # e.g., "TG"

        # Sort out the metadata
        for x in ('obs_time',
                  'site_lat',
                  'site_lon',
                  'site_elev',
                  'observer',
                  'block_filter',
                  'exposure',
                  'tel_manufac',
                  'tel_firmware',
                  'tel_model',
                  'adc_depth',
                  'largest_usable_adu_value'):

            self.starlist[x] = metadata[x] if x in metadata else None

        self.starlist['epoch'] = "J2000"

        # Add a couple of missing items
        self.starlist['refframe'] = "ICRS"

        # Set proper type for tel_firmware
        if self.starlist['tel_firmware'] is None:
            self.starlist['tel_firmware'] = ""

        # Set proper type for adc_depth
        if self.starlist['adc_depth'] is None:
            self.starlist['adc_depth'] = 0

        # The "STARLIST' is a list of dictionaries
        self.starlist['staritems'] = [] # the starlist starts off empty

    def ReadFromSourceExtractor(self, filename):
        """Create starlist entries from SourceExtractor output table

        Ingest the output table from Source Extractor and create
        self.starlist entries for each star in the output table.

        Parameters
        ----------
        filename : str
            The pathname to the source extractor output table

        Returns
        -------
        None

        """
        # Decode() is a helper function that supports the process of reading a
        # SourceExtractor starlist and turning it into an AAVSO starlist.
        # column_labels: a list of source_extractor output table column label strings
        # words: a list of words in the current input line (a data line)
        # keyword: the keyword we want to fetch. "keyword" must match a column label
        # The return value is the (float) value of the number in the column
        # that corresponds to the keyword.
        def Decode(column_labels, words, keyword):
            try:
                i = column_labels.index(keyword)
                return float(words[i])
            except:
                print("Keyword ", keyword, " not in source extractor output.")
                raise

        column_labels = []      # indexed by column number
        with open(filename, 'r') as fp:
            for line in fp:
                # If the line starts with a '#', then its a column label line
                if line[0] == '#':
                    words = line[1:].strip().split(' ')
                    column_labels.append(words[1])
                else:
                    # Normal (star) line
                    words = line.split() # whitespace split
                    star = {}
                    star['flux_err'] = Decode(column_labels, words, 'FLUXERR_AUTO')
                    star['tot_flux'] = Decode(column_labels, words, 'FLUX_AUTO')
                    star['peak_flux'] = Decode(column_labels, words, 'FLUX_MAX')
                    star['x'] = Decode(column_labels, words, 'X_IMAGE')
                    star['y'] = Decode(column_labels, words, 'Y_IMAGE')
                    star['dec'] = Decode(column_labels, words, 'DELTA_J2000')
                    star['ra'] = Decode(column_labels, words, 'ALPHA_J2000')
                    star['bkgd_flux'] = Decode(column_labels, words, 'BACKGROUND')
                    self.starlist['staritems'].append(star)

    def ReadFromPhotUtils(self, sourcelist, background, metadata):
        """Create starlist entries from photutils output table

        Ingest the output table from photutils and create
        self.starlist entries for each star in the output

        Parameters
        ----------
        sourcelist : astropy table
            The astropy table created by one of the astropy source
            detection methods
        background : photutils 2D image
            A copy of the original image holding background levels in
            each pixel
        metadata : dict
            The metadata dictionary for this input file

        Returns
        -------
        None

        """
        for (xc,yc,peak,flux) in sourcelist.iterrows('xcentroid','ycentroid','peak','flux'):
            star = {}
            star['tot_flux'] = float(flux)
            star['peak_flux'] = float(peak)
            star['x'] = float(xc)
            star['y'] = float(yc)
            star['dec'] = None
            star['ra'] = None
            star['bkgd_flux'] = float(background[int(0.5+yc),int(0.5+xc)])
            star['flux_err'] = None
            self.starlist['staritems'].append(star)

    def WriteJSON(self, filename):
        """Create an AAVSO starlist set file from an AAVSOStarlist object

        Create an AAVSO starlist set file with the contents of the current
        starlist object.

        Parameters
        ----------
        filename : str
            Pathname of the JSON file to be created

        Returns
        -------
        None
        """
        # Version will be set to default value in schema definition,
        # no need to set it here.
        star_list_set = StarListSet(star_lists=[self.starlist])

        with open(filename, 'w') as fp:
            json.dump(star_list_set.model_dump(), fp, indent=2)

    def ApplyWCS(self, wcs):
        """Update star Dec/RA in the starlist using a provided WCS

        Update all star positions in the starlist, converting each
        star's (x,y) pixel-based centroid into a corresponding Dec/RA
        using the provided WCS. Stars that don't have a valid pixel
        location will not receive Dec/RA values.

        Parameters
        ----------
        wcs : astropy wcs object
            The WCS mapping pixel coordinates to sky coordinates

        Returns
        -------
        None
        """
        valid_stars = [star for star in self.starlist['staritems'] if star['x'] is not None and star['y'] is not None]
        x_array = [star['x'] for star in valid_stars]
        y_array = [star['y'] for star in valid_stars]
        (ra_array, dec_array) = wcs.pixel_to_world_values(x_array, y_array)
        for (star,ra,dec) in zip(valid_stars,ra_array,dec_array):
            star['dec'] = dec
            star['ra'] = ra

def WCStext2wcs(wcs_text):
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

# Process one (possibly de-Bayered) image
def ProcessSingleImage(filename, metadata, options, temp_dir,
                       starlist_json_path, filter, wcs=None):
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

    wcs : astropy WCS object
        WCS object that maps pixel coordinates to sky coordinates. If
        provided then astrometry.net fitting is skipped.
    Returns
    -------
    str
        The pathname of the starlist that was created
    """
    # filter == 'M' is a special case == 'CV'
    if filter == 'M':
        filter = 'CV'

    # "G" needs to become "TG" if it hasn't already
    if len(filter) == 1:
        filter = 'T'+filter

    width = None
    height = None
    starlist = AAVSOStarlist(metadata, filter)
    with fits.open(filename) as hdul:
        hdu0h = hdul[0].header
        image_data = hdul[0].data.astype(np.float32)
        (width,height) = np.shape(hdul[0].data)
        print("height = ", height, ", width = ", width)

        # Estimate the background
        (mean,median,std) = sigma_clipped_stats(image_data, sigma=3.0)
        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground()
        background = Background2D(image_data, (int(width/8),int(height/8)), filter_size=(3,3),
                                  sigma_clip=sigma_clip, bkg_estimator=bkg_estimator).background
        (bkgd_mean,_dummy,bkgd_std) = sigma_clipped_stats(background, sigma=3.0)
        noise_bkgd = bkgd_std/starlist.starlist['gain']
        daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
        clean_image = (image_data - background)
        (mean,median,std) = sigma_clipped_stats(clean_image, sigma=3.0)
        print("mean/median/std = ", (mean, median, std))
        sources = daofind(clean_image)
        #print(sources)
        starlist.ReadFromPhotUtils(sources,background,metadata)

        ## Now estimate an FWHM for these stars
        star_subset = starlist.starlist['staritems']
        star_subset.sort(key = lambda star : star['tot_flux'], reverse=True)
        subset_size = min(10, len(star_subset))
        fwhm = statistics.mean(psf.fit_fwhm(clean_image, xypos=[(s['x'],s['y']) for s in star_subset[0:subset_size]],fit_shape=15))
        print("Estimate FWHM from photutils = ", fwhm)

        phot_radius = 1.0 * fwhm
        star_x = [s['x'] for s in star_subset]
        star_y = [s['y'] for s in star_subset]
        radii = [phot_radius for s in star_subset]
        positions = zip(star_x, star_y)
        apertures = aperture.CircularAperture(positions, r=phot_radius)
        result = aperture.aperture_photometry(clean_image, apertures)
        print(result)

        xc = np.array(result['xcenter'])
        yc = np.array(result['ycenter'])
        fluxes = np.array(result['aperture_sum'])

        for s,flux,x,y in zip(starlist.starlist['staritems'],fluxes,xc,yc):
            s['tot_flux'] = float(flux)
            if flux >= 0.0:
                poiss_noise = starlist.starlist['gain']*(math.sqrt(float(flux)/starlist.starlist['gain']))
                tot_noise = math.sqrt(poiss_noise*poiss_noise+noise_bkgd*noise_bkgd)
                snr = float(flux)/tot_noise
                s['flux_err'] = tot_noise
            else:
                s['flux_err'] = 0.0
            s['x'] = float(x)
            s['y'] = float(y)

    ################################
    ## Plate Solve to get WCS transformation info if needed
    ## Things you need:
    ## Dec/RA (nominal) - metadata
    ## Plate scale (nominal) - metadata
    ## Field of View (nominal) - metadata
    ################################

    if not wcs:
        ast = AstrometryNet()
        if astrometry_api_key == None:
            global ui
            dlg = QMessageBox(ui.window)
            dlg.setWindowTitle("No astrometry.net API Key")
            dlg.setText("Must enter astrometry.net API Key via Menu Bar")
            dlg.exec()
            return

        ast.api_key = astrometry_api_key
        # star_x and star_y were sorted by flux earlier... important here.
        wcs_header = ast.solve_from_source_list(star_x,
                                                star_y,
                                                width,
                                                height,
                                                solve_timeout=120)
        wcs = WCS(header=wcs_header)

    if False:
        import astrometry
        wcs_text = astrometry.RunAstrometry(api_key = astrometry_api_key,
                                        x=star_x,
                                        y=star_y,
                                        dec_center_deg =metadata['dec'],
                                        ra_center_deg =metadata['ra'],
                                        radius_deg = metadata['fov_rad'],
                                        scale=(metadata['pixscale']*0.8,
                                               metadata['pixscale']*1.2),
                                        width=width,
                                        height=height)
        wcs = WCStext2wcs(wcs_text.decode('utf-8'))
        print("Image center is at ", wcs.pixel_to_world(width/2, height/2))
    starlist.ApplyWCS(wcs)

    ################################
    ## Do PSF fitting, if requested
    ################################
    if options.UsePSFFitting:
        starlist.starlist['staritems'].sort(key=lambda star:
                                           star['tot_flux'], reverse=True)
        psf_fitting.DoPSF(filename, starlist.starlist)

    starlist.WriteJSON(starlist_json_path)
    return starlist_json_path

def ProcessRGBFile(filename, options, temp_dir, metadata,
                   starlist_tgtname, wcs=None):
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

    Returns
    -------
    None

    """
    de_bayer = options.DeBayer
    if de_bayer:
        single_color_files = DeBayerFile(filename, metadata['BAYERPAT'], temp_dir)
        starlists = []

        if options.StackChannels:
            stacked_image = StackImages(single_color_files, options, temp_dir)
            starlist_filename = starlist_tgtname.replace("$$","M")
            starlist_file = ProcessSingleImage(stacked_image, dict(metadata),
                                               options,
                                               temp_dir,
                                               starlist_filename, 'M',
                                               wcs=wcs)
            print("Starlist stored in ", starlist_file)
        else:
            tg_num = 1
            for (filter,file) in single_color_files:
                filter_file = filter
                # Hangle "TG" and "G" filters the same
                if filter in ['TG', 'G']:
                    filter_file = "TG"+str(tg_num)
                    tg_num += 1
                starlist_filename = starlist_tgtname.replace("$$",filter_file)
                starlist_file = ProcessSingleImage(file, dict(metadata), options,
                                                   temp_dir, starlist_filename,filter,
                                                   wcs=wcs)
                starlists.append(starlist_file)

            print("Starlist(s) stored in ", starlists)
    else:
        # Not de-Bayered; treat as single monochrome image
        starlist_filename = starlist_tgtname.replace("$$","M") # M==monochrome
        print(metadata)
        adj_meta_dict = dict(metadata)
        adj_meta_dict['pixscale'] /= 2.0 # Correct for non-de-Bayered image
        starlist_file = ProcessSingleImage(filename, adj_meta_dict,
                                           options,
                                           temp_dir,
                                           starlist_filename, 'M', wcs=wcs)
        print("Starlist stored in ", starlist_file)


################################################################
##        Display GUI Comes Next
################################################################

from PySide6 import QtCore, QtWidgets, QtGui

from PySide6.QtWidgets import QFileDialog, QProgressBar, QDialog
from PySide6.QtWidgets import QVBoxLayout, QLabel, QCheckBox, QDialogButtonBox
from PySide6.QtWidgets import QLineEdit, QMessageBox
from PySide6.QtGui import QGuiApplication
from PySide6.QtCore import QFile, QIODevice
from PySide6.QtUiTools import QUiLoader

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

    def chooser_popup(self, button):
        """Create popup window to choose file(s)

        Initiate the popup window to select one (or more) files. This
        method blocks until the file selection has completed, so all
        other buttons and widgets in the application will be
        disabled. The selected filename will be put into the
        FileChooser's `text_entry_widget`.

        Parameters
        ----------
        button : QPushButton
            The button that triggered this popup window

        Returns
        -------
        None
        """
        dialog = QFileDialog(self.text_widget)
        dialog.setFileMode(self.file_mode)
        if dialog.exec():
            if self.multiple_files_okay:
                filename_list = dialog.selectedFiles()
                # Now append to the filelist
                entry_list = dialog.selectedFiles()
                for entry in entry_list:
                    self.text_widget.appendPlainText(entry + '\n')
            else:
                self.text_widget.setText(dialog.selectedFiles()[0])
        else:
            self.text_widget.setText("")

    def EnteredFilename(self):
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
        if self.multiple_files_okay:
            raise Exception("Call to EnteredFilename should be EnteredFilenameList")
        raw_text = self.text_widget.text()
        if raw_text is None or len(raw_text.strip()) == 0:
            return None
        else:
            return raw_text.strip()

    def EnteredFilenameList(self):
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
        if not self.multiple_files_okay:
            raise Exception("Call to EnteredFilenameList should be EnteredFilename")
        raw_text = self.text_widget.toPlainText()
        text_words = raw_text.split('\n')
        print("Files to process = ", text_words)
        return text_words

    def ClearFilename(self):
        """Clear the entered filename

        Clear the filename entered for this FileChooser object

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
                                      ui.window.MetaButton)
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

    @property
    def DeBayer(self):
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
    def GetPhot(self):
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
    def StackChannels(self):
        """Query whether de-Bayered images are to be stacked

        Query whether de-Bayered images are to be stacked into a
        single sort-of-luminance channel image. Only makes sense to
        query this if DeBayer() returns True. If stacking was chosen,
        the method used for doing the stacking depends on the setting
        of the InterpolateChannels() query.

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
    def InterpolateChannels(self):
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
    def GetColorBalance(self):
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
    def UsePSFFitting(self):
        """Query whether PSF-fitting photometry is to be done

        Return a boolean indicating whether
        PSF-fitting photometry is to be done. This method is 100%
        redundant with GetPhot() and should be retired.

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
        return self._bias_file.EnteredFilename()

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
        return self._dark_file.EnteredFilename()

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
        return self._flat_file.EnteredFilename()

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
        return self._meta_file.EnteredFilename()

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
        return self._image_file.EnteredFilenameList()


class OptionsAPI(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_default=True, validate_assignment=True)
    debayer: bool = False
    one_channel: bool = False
    stacked_channels: bool = False
    interp_stack_channels: bool = False
    color_correx: bool = False
    psf_photometry: bool = False
    astrometry_net_api_key: str = ""
    bias_file: str = ""
    dark_file: str = ""
    flat_file: str = ""
    meta_file: str = ""
    image_file: List[str] = [""]

    # These are accessed by the current code.
    @property
    def DeBayer(self):
        return self.debayer

    @property
    def InterpolateChannels(self):
        return self.interp_stack_channels

    @property
    def GetColorBalance(self):
        return self.color_correx

    @property
    def StackChannels(self):
        return (self.stacked_channels or self.interp_stack_channels)

    @property
    def UsePSFFitting(self):
        return self.psf_photometry


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

        QBtn = (QDialogButtonBox.StandardButton.Ok |
                QDialogButtonBox.StandardButton.Cancel)
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.AcceptKey)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)

        self.setLayout(layout)

    def AcceptKey(self):
        """Intercept the dialog's "Okay" button

        Before executing the default "Okay" button behavior, save the
        API key value for use here in this program (as a global
        variable) and save in the user's STWG local storage folder.
        """
        global astrometry_api_key
        astrometry_api_key = self.lineEdit.text()
        if self.save_checkbox.isChecked():
            SaveAstrometryKey(astrometry_api_key)
        self.accept()           # execute default behavior (kills the popup)

def GetAstrometryKey():
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
        raise ValueException("OS Name not recognized")

    localdir.mkdir(parents=True, exist_ok=True)
    APIKeypathname = localdir / "astrometryAPIkey.txt"

    try:
        return APIKeypathname.read_text()
    except:
        return None

def SaveAstrometryKey(key_value):
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
        raise ValueException("OS Name not recognized")

    localdir.mkdir(parents=True, exist_ok=True)
    APIKeypathname = localdir / "astrometryAPIkey.txt"

    try:
        APIKeypathname.write_text(key_value)
    except:
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

        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dirname = self.temp_dir.name
        print("Working in temporary directory ", self.temp_dirname)
        print(f"===> st-pipeline version {__version__} <===")
        self.options = options
        self.ui = ui
        self._wcs = wcs
        # Try getting the API key from options, or return None if not there
        astrometry_api_key = getattr(self.options, "astrometry_net_api_key", None)
        if astrometry_api_key is None:
            astrometry_api_key = GetAstrometryKey()

        if self.ui:
            self.progressbar = self.ui.window.progressBar
        else:
            self.GenerateStarlist()

    def GetKey(self):
        dialog = APIEntryDialog(self.ui.window)
        dialog.exec()

    ################################
    ## GenerateStarlist button
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
        self.GenerateStarlist()
        self.progressbar.hide()

    def GenerateStarlist(self):
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
        image_list = self.options.image_file
        dark_filename = self.options.dark_file
        flat_filename = self.options.flat_file
        bias_filename = self.options.bias_file
        metadata_filename = self.options.meta_file
        do_bayer_balance = self.options.GetColorBalance

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
            ReadMetaFromFITS(image_filename, meta)

            QGuiApplication.processEvents()
            # Now get the metadata from the standalone metadata file
            if metadata_filename is not None:
                meta_path = Path(metadata_filename)
                if not (meta_path.is_file() and meta_path.exists() and
                        meta_path.stat().st_mode & 0o400):
                    print("Cannot read metadata from file ", metadata_filename)
                    raise ValueError("Cannot read metadata file")
                print("Reading metadata from ", metadata_filename)
                ReadMetaFromJSON(metadata_filename, meta)

            print("Final metadata is ", meta)

            if self.options.GetColorBalance:
                working_filename = BayerBalanceFile(working_filename, self.temp_dirname)
            ProcessRGBFile(working_filename, self.options, self.temp_dirname, meta, starlist_tgtname, wcs=self._wcs)
        return False


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
        ui.window.actionEnter_astrometry_net_API_key.triggered.connect(not_a_window.GetKey)

        sys.exit(app.exec())


if __name__ == "__main__":
    main()
