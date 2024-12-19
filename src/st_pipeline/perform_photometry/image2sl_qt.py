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

import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

import threading
import os.path
import statistics
import numpy as np
import tempfile
import getopt
import datetime
import math
import sys
import json
import psf_fitting
#import sep

################################################################
##        Algorithmic Stuff Comes First
################################################################

use_PSF = False

# Returns a list of (filter,filename) tuples
#
# filename: pathname to the image to be de-Bayered
# pattern: Bayer pattern (e.g., 'BGGR')
# temp_dir: pathname to a temporary directory where stuff can be put
#
def DeBayerFile(filename, pattern, temp_dir):
    with fits.open(filename) as hdul:
        temp1 = hdul[0].data[0::2,0::2]
        temp2 = hdul[0].data[1::2,0::2]
        temp3 = hdul[0].data[0::2,1::2]
        temp4 = hdul[0].data[1::2,1::2]
        array = [temp1, temp2, temp3, temp4]
        output_filenames = [] # each entry in this list is a tuple: (filter, filename)

        for index in range(4):
            color = pattern[index]
            output_tgt = os.path.join(temp_dir, "image"+str(index)+"_"+color+".fits")
            
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
            hdu.header['FILTER'] = ('T'+color, 'Bayer color mask')
            # update_header will "fix" the header to match the data
            hdu.update_header()
            fits.writeto(output_tgt, array[index], header=hdu.header, overwrite=True)
            output_filenames.append((color,output_tgt))
        return output_filenames

# pattern triplet: (x_offset, y_offset, weight)
pattern = [
    [ (0,0,9), (0,1,3),  (1,0,3),  (1,1,1)   ], # color 0
    [ (0,0,9), (-1,0,3), (0,1,3),  (-1,1,1)  ], # color 1
    [ (0,0,9), (1,0,3),  (0,-1,3), (1,-1,1)  ], # color 2
    [ (0,0,9), (-1,0,3), (0,-1,3), (-1,-1,1) ]  # color 3
]

def StackImages(channel_list, options, temp_dir, metadata):
    output_tgt = os.path.join(temp_dir, "image_S.fits")
    hdu = fits.PrimaryHDU()
    # push keywords in from the original file(s)
    (_dummy,filename) = channel_list[0]
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
            if options.InterpolateChannels():
                new_data = np.zeros(np.shape(hdul[0].data),dtype=np.float32)
                source_hdu = new_data
                for y in range(height-1):
                    for x in range(width-1):
                        tgt = sum((p[2]*hdul[0].data[y+p[1],x+p[0]] for p in pattern[bayer_id]))
                        new_data[y,x] = tgt

            hdu.data += source_hdu
    hdu.header['FILTER'] = 'CV'
    hdu.update_header()
    fits.writeto(output_tgt, hdu.data, header=hdu.header, overwrite=True)
    return output_tgt
                
def DuplicateFileWithNewImage(hdul, new_data, new_filter, new_pathname):
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
    hdu.header['FILTER'] = new_filter
    # update_header will "fix" the header to match the data
    hdu.update_header()
    fits.writeto(new_pathname, new_data, header=hdu.header, overwrite=True)

def BayerBalanceFile(filename, temp_dir):
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

# Figure out what kind of a smart telescope created this image
def ProbeFileForType(filename):
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
        raise Exception("Unable to establish telescope type")

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
##    SITELAT - a float, the observer's latitude in degrees
##    SITELON - a float, the observer's longitude in degrees
##    SITEELEV - a float, the observer's GPS elevation
##    OBSERVER - a string, the AAVSO observer code
##    FILTER - a 2-character string, one of TG,TB,TR
##    BLOCK_FILTER - a string, typically "UV+IR"
##    EXPOSURE - a float, total exposure time in secs
##    TEL_MANUFAC - a string, name of the telescope manufacturer
##    TEL_MODEL - a string, the telescope's model name
##    TEL_FIRMWARE - a string, the firmware ID
##    ADC_DEPTH - an integer, bit depth of the camera ADC
##    DATAMAX - an integer, the ADU level where saturation starts
##
## Additional metadata indices:
##    BAYERPAT - a 4-character string (e.g., 'BGGR')
##    PIXSCALE - a float, pixel scale *after* debayering, arcsec/pix
##    DEC - a float, nominal declination of image center (deg)
##    RA - a float, nominal RA of image center (deg)
##    FOV_RAD - a float, nominal field of view radius (deg)
################################################################

valid_meta_keys = ['AAVSO_VER',
                   'OBS_TIME',
                   'SITELAT',
                   'SITELON',
                   'SITEELEV',
                   'OBSERVER',
                   'FILTER',
                   'BLOCK_FILTER',
                   'EXPOSURE',
                   'TEL_MANUFAC',
                   'TEL_MODEL',
                   'TEL_FIRMWARE',
                   'ADC_DEPTH',
                   'DATAMAX',
                   'SYSTEM_GAIN',
                   'BAYERPAT',
                   'PIXSCALE',
                   'DEC',
                   'RA',
                   'FOV_RAD' ]

#
# The so-called JSON metadata file is a temporary band-aid for smart
# telescopes that are currently missing important FITS header
# keywords. (Early Origin scopes have this problem.) The JSON metdata
# file is merged with the metadata that comes from the FITS header,
# providing a way to deal with missing/incorrect FITS header info.
#

# This function performs an *update* to the dictionary provided as an
# argument, using the content of the JSON metadata file to augment or
# replace entries in "dict".
def ReadMetaFromJSON(filename, dict):
    bytes = os.path.getsize(filename)
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
    telescope_type = ProbeFileForType(filename)
    with fits.open(filename) as hdul:
        hdu0h = hdul[0].header
        
        # Generate Metadata
        ################################
        ##          Seestar
        ################################
        if telescope_type == 'Seestar':
            for key,tgt in [('BAYERPAT','BAYERPAT'),
                            ('DATE-OBS','OBS_TIME'),
                            ('FILTER','BLOCK_FILTER')]:
                if key in hdu0h:
                    dict[tgt] = hdu0h[key]
            ################
            ## FOV, Pixel scale
            ################
            dict['FOV_RAD'] = 0.7 # wild guess; needs validation

            for key,tgt in [('SITELAT', 'SITELAT'),
                            ('SITELONG', 'SITELON'),
                            ('DEC', 'DEC'),
                            ('RA', 'RA'),
                            ('EXPOSURE', 'EXPOSURE')]:
                if key in hdu0h:
                    dict[tgt] = hdu0h[key]

            dict['TEL_MANUFAC'] = 'ZWO'
            dict['TEL_MODEL'] = 'Seestar'

        ################################
        ##       Origin
        ################################
        elif telescope_type == 'Origin':
            dict['BAYERPAT'] = 'RGGB' # Is this correct?
            dict['TEL_MANUFAC'] = 'Celestron'
            dict['TEL_MODEL'] = 'Origin'
            dict['ADC_DEPTH'] = 14
            dict['PIXSCALE'] = 1.45*2 # after de-bayering

        ################################
        ##       Unistellar
        ################################
        elif telescope_type == 'Unistellar':
            for key,tgt in [('BAYERPAT','BAYERPAT'),
                            ('DATE-AVG','OBS_TIME'),
                            ('EXPTIME','EXPOSURE'),
                            ('LATITUDE','SITELAT'),
                            ('LONGITUD', 'SITELON'),
                            ('ALTITUDE', 'SITEELEV'),
                            ('FOVDEC','DEC')]:
                if key in hdu0h:
                    dict[tgt] = hdu0h[key]
            dict['TEL_MANUFAC'] = 'Unistellar'
            dict['TEL_MODEL'] = 'eVscope'
            dict['PIXSCALE'] = 1.5*2 # after de-bayering

        ################################
        ##        Dwarf
        ################################
        elif telescope_type == "Dwarf":
            for key,tgt in [('BAYERPAT','BAYERPAT'),
                            ('EXPTIME','EXPOSURE'),
                            ('DATE-OBS','OBS_TIME'),
                            ('INSTRUME','TEL_MODEL'),
                            ('RA','RA'),
                            ('DEC','DEC')]:
                if key in hdu0h:
                    dict[tgt] = hdu0h[key]
            dict['TEL_MANUFAC'] = 'DwarfLab'
            dict['PIXSCALE'] = 1.5*2 # after de-bayering

        else:
            print("Telescope type ", telescope_type, " not implemented yet.")
            

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

# This class "is" an AAVSO Starlist. Initial condition: all metadata
# is populated, but the starlist is otherwise empty.
class AAVSOStarlist:
    def __init__(self, metadata, filter):
        self.metadata = metadata
        self.gain = metadata['SYSTEM_GAIN']
        
        self.starlist = {} # JSON-style dictionary
        self.filter = filter # e.g., "TG"

        # Sort out the metadata
        self.starlist['AAVSO_VER'] = "AA_001"
        for x in ('OBS_TIME',
                  'SITELAT',
                  'SITELON',
                  'SITEELEV',
                  'OBSERVER',
                  'BLOCK_FILTER',
                  'EXPOSURE',
                  'TEL_MANUFAC',
                  'TEL_FIRMWARE',
                  'TEL_MODEL',
                  'ADC_DEPTH',
                  'DATAMAX'):
            
            self.starlist[x] = metadata[x] if x in metadata else None
            
        self.starlist['EPOCH'] = "J2000"

        # The "STARLIST' is a list of dictionaries
        self.starlist['STARLIST'] = [] # the starlist starts off empty
        

    # Read stars from the SourceExtractor output table and create
    # entries in self.starlist['STARLIST']
    def ReadFromSourceExtractor(self, filename):
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
                    star['FLUX_ERR'] = Decode(column_labels, words, 'FLUXERR_AUTO')
                    star['TOT_FLUX'] = Decode(column_labels, words, 'FLUX_AUTO')
                    star['PEAK_FLUX'] = Decode(column_labels, words, 'FLUX_MAX')
                    star['X'] = Decode(column_labels, words, 'X_IMAGE')
                    star['Y'] = Decode(column_labels, words, 'Y_IMAGE')
                    star['DEC'] = Decode(column_labels, words, 'DELTA_J2000')
                    star['RA'] = Decode(column_labels, words, 'ALPHA_J2000')
                    star['BKGD_FLUX'] = Decode(column_labels, words, 'BACKGROUND')
                    self.starlist['STARLIST'].append(star)

    def ReadFromSEP(self, sep_array, background, metadata):
        gain = self.gain
        for sep_item in sep_array:
            star = {}
            star['TOT_FLUX'] = float(sep_item['flux'])
            if sep_item['flux'] > 0:
                star['FLUX_ERR'] = math.sqrt(gain*sep_item['flux'])/gain
            else:
                star['FLUX_ERR'] = None
            star['PEAK_FLUX'] = float(sep_item['peak'])
            star['X'] = float(sep_item['x'])
            star['Y'] = float(sep_item['y'])
            star['DEC'] = None
            star['RA'] = None
            star['BKGD_FLUX'] = float(background[int(0.5+sep_item['y']),int(0.5+sep_item['x'])])
            self.starlist['STARLIST'].append(star)
            ##for (key,value) in star.items():
            ##    print(key, value, type(value))
            
    def ReadFromPhotUtils(self, sourcelist, background, metadata):
        for (xc,yc,peak,flux) in sourcelist.iterrows('xcentroid','ycentroid','peak','flux'):
            star = {}
            star['TOT_FLUX'] = float(flux)
            star['PEAK_FLUX'] = float(peak)
            star['X'] = float(xc)
            star['Y'] = float(yc)
            star['DEC'] = None
            star['RA'] = None
            star['BKGD_FLUX'] = float(background[int(0.5+yc),int(0.5+xc)])
            star['FLUX_ERR'] = None
            self.starlist['STARLIST'].append(star)
            

    # Create a JSON AAVSO starlist, stored in the specified filename
    def WriteJSON(self, filename):
        with open(filename, 'w') as fp:
            json.dump(self.starlist, fp, indent=2)
        
    def ApplyWCS(self, wcs):
        valid_stars = [star for star in self.starlist['STARLIST'] if star['X'] is not None and star['Y'] is not None]
        x_array = [star['X'] for star in valid_stars]
        y_array = [star['Y'] for star in valid_stars]
        (ra_array, dec_array) = wcs.pixel_to_world_values(x_array, y_array)
        for (star,ra,dec) in zip(valid_stars,ra_array,dec_array):
            star['DEC'] = dec
            star['RA'] = ra
            print('(',star['X'],',',star['Y'],') -> ', ra, dec)

def WCStext2wcs(wcs_text):
    lines = []
    card_list = []
    while len(wcs_text) >= 80:
        this_line = wcs_text[0:80]
        wcs_text = wcs_text[80:]
        card_list.append(fits.Card.fromstring(this_line))
        print(this_line)
    wcs_header = fits.Header(cards=card_list)
    return WCS(wcs_header)
    
# Process one (possibly de-Bayered) image
def ProcessSingleImage(filename, metadata, options, temp_dir, starlist_json_path, filter):
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
        #image_data = hdul[0].data
        (width,height) = np.shape(hdul[0].data)
        print("height = ", height, ", width = ", width)
        # Estimate the background
        (mean,median,std) = sigma_clipped_stats(image_data, sigma=3.0)
        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground()
        background = Background2D(image_data, (int(width/8),int(height/8)), filter_size=(3,3),
                                  sigma_clip=sigma_clip, bkg_estimator=bkg_estimator).background
        (bkgd_mean,_dummy,bkgd_std) = sigma_clipped_stats(background, sigma=3.0)
        noise_bkgd = bkgd_std/starlist.gain
        #background = sep.Background(image_data)
        #background.subfrom(image_data)
        #sep_objects = sep.extract(image_data,background.globalrms*3.0,gain=starlist.gain)
        daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
        clean_image = (image_data - background)
        (mean,median,std) = sigma_clipped_stats(clean_image, sigma=3.0)
        print("mean/median/std = ", (mean, median, std))
        sources = daofind(clean_image)
        #norm = ImageNormalize(stretch=SqrtStretch())
        #plt.imshow(image_data - background, norm=norm, cmap='Greys_r', interpolation='nearest')
        #print("sep created list of ", len(sep_objects), " stars.")
        print(sources)
        starlist.ReadFromPhotUtils(sources,background,metadata)

        ## Now estimate an FWHM for these stars
        star_subset = starlist.starlist['STARLIST']
        star_subset.sort(key = lambda star : star['TOT_FLUX'], reverse=True)
        subset_size = min(10, len(star_subset))
        #(sigma_x, sigma_y, bkgd_rms, bayer_zero, bayer_scale) = psf_fitting.MeasurePSF(image_data, star_subset[0:subset_size])
        #fwhm = 2.35482*(sigma_x+sigma_y)/2.0
        #print("Estimated FWHM = ", fwhm)

        # Now estiamte an FWHM from photutils to compare
        fwhm = statistics.mean(psf.fit_fwhm(clean_image, xypos=[(s['X'],s['Y']) for s in star_subset[0:subset_size]],fit_shape=15))
        print("Estimate FWHM from photutils = ", fwhm)
                            
        phot_radius = 1.0 * fwhm
        star_x = [s['X'] for s in starlist.starlist['STARLIST']]
        star_y = [s['Y'] for s in starlist.starlist['STARLIST']]
        radii = [phot_radius for s in starlist.starlist['STARLIST']]
        positions = zip(star_x, star_y)
        apertures = aperture.CircularAperture(positions, r=phot_radius)
        result = aperture.aperture_photometry(clean_image, apertures)
        print(result)

        xc = np.array(result['xcenter'])
        yc = np.array(result['ycenter'])
        fluxes = np.array(result['aperture_sum'])
        
        for s,flux,x,y in zip(starlist.starlist['STARLIST'],fluxes,xc,yc):
            s['TOT_FLUX'] = float(flux)
            poiss_noise = starlist.gain*(math.sqrt(float(flux)/starlist.gain))
            tot_noise = math.sqrt(poiss_noise*poiss_noise+noise_bkgd*noise_bkgd)
            snr = float(flux)/tot_noise
            s['FLUX_ERR'] = tot_noise
            s['X'] = float(x)
            s['Y'] = float(y)
                                      

    ################################
    ## Plate Solve to get WCS transformation info
    ## Things you need:
    ## Dec/RA (nominal) - metadata
    ## Plate scale (nominal) - metadata
    ## Field of View (nominal) - metadata
    ################################

    import astrometry
    wcs_text = astrometry.RunAstrometry(api_key = "pygiszqhoszxlyss",
                                        x=star_x,
                                        y=star_y,
                                        dec_center_deg =metadata['DEC'],
                                        ra_center_deg =metadata['RA'],
                                        radius_deg = metadata['FOV_RAD'],
                                        scale=(metadata['PIXSCALE']*0.8,
                                               metadata['PIXSCALE']*1.2),
                                        width=width,
                                        height=height)
    wcs = WCStext2wcs(wcs_text.decode('utf-8'))
    print("Image center is at ", wcs.pixel_to_world(width/2, height/2))
    starlist.ApplyWCS(wcs)

    starlist.WriteJSON(starlist_json_path)

    ################################
    ## Do PSF fitting, if requested
    ################################
    if options.UsePSFFitting():
        starlist.starlist['STARLIST'].sort(key=lambda star:
                                           star['TOT_FLUX'], reverse=True)
        psf_fitting.DoPSF(filename, starlist.starlist)
    
    return starlist_json_path
    
# Process a one-shot-color image, turning it into separate images for
# each color channel. Create a starlist for each channel. Store the
# pathnames for those 4 starlists into "starlists".
def ProcessRGBFile(filename, options, temp_dir, metadata, starlist_tgtname):
    de_bayer = options.DeBayer()
    if de_bayer:
        single_color_files = DeBayerFile(filename, metadata['BAYERPAT'], temp_dir)
        starlists = []

        if options.StackChannels():
            stacked_image = StackImages(single_color_files, options, temp_dir, metadata)
            starlist_filename = starlist_tgtname.replace("$$","M")
            starlist_file = ProcessSingleImage(stacked_image, dict(metadata),
                                               temp_dir,
                                               starlist_filename, 'M')
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
                starlist_file = ProcessSingleImage(file, metadata, temp_dir, starlist_filename,filter)
                starlists.append(starlist_file)

            print("Starlist(s) stored in ", starlists)
    else:
        # Not de-Bayered; treat as single monochrome image
        starlist_filename = starlist_tgtname.replace("$$","M") # M==monochrome
        print(metadata)
        adj_meta_dict = dict(metadata)
        adj_meta_dict['PIXSCALE'] /= 2.0 # Correct for non-de-Bayered image
        starlist_file = ProcessSingleImage(filename, adj_meta_dict,
                                           options,
                                           temp_dir,
                                           starlist_filename, 'M')
        print("Starlist stored in ", starlist_file)
    

################################################################
##        Display GUI Comes Next
################################################################

from PySide6 import QtCore, QtWidgets, QtGui

from PySide6.QtWidgets import QFileDialog, QProgressBar
from PySide6.QtGui import QGuiApplication
from PySide6.QtCore import QFile, QIODevice
from PySide6.QtUiTools import QUiLoader

class FileChooser:
    def __init__(self,
                 text_entry_widget,
                 chooser_button,
                 multiple_files_okay=False):
        self.text_widget = text_entry_widget
        self.popup_button = chooser_button
        self.multiple_files_okay = multiple_files_okay
        chooser_button.clicked.connect(self.chooser_popup)

        if not multiple_files_okay:
            self.file_mode = QFileDialog.ExistingFile
        else: # Big entry for image filenames
            self.file_mode = QFileDialog.ExistingFiles
            
    def chooser_popup(self, button):
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
        if self.multiple_files_okay:
            raise Exception("Call to EnteredFilename should be EnteredFilenameList")
        raw_text = self.text_widget.text()
        if raw_text is None or len(raw_text.strip()) == 0:
            return None
        else:
            return raw_text.strip()

    def EnteredFilenameList(self):
        if not self.multiple_files_okay:
            raise Exception("Call to EnteredFilenameList should be EnteredFilename")
        raw_text = self.text_widget.toPlainText()
        text_words = raw_text.split('\n')
        print("Files to process = ", text_words)
        return text_words

    def ClearFilename(self):
        self.text_widget.setText("")

class OptionBox:
    def __init__(self):
        self.pretend_monochrome = ui.window.MonochromeButton
        self.one_channel = ui.window.SingleChannelButton
        self.stacked_channels = ui.window.StackedButton
        self.interp_stack_channels = ui.window.StackInterpButton
        self.color_correx = ui.window.ColorBalanceButton
        self.psf_photometry = ui.window.PSFPhotButton
        self.aperture_photometry = ui.window.AperturePhotButton

    # return boolean
    def DeBayer(self):
        return not self.pretend_monochrome.isChecked()

    # return "psf" or "app_phot"
    def GetPhot(self):
        return "psf" if self.psf_photometry.isChecked() else "app_phot"

    def StackChannels(self):
        return (self.stacked_channels.isChecked()  or
                self.interp_stack_channels.isChecked())

    def InterpolateChannels(self):
        return self.interp_stack_channels.isChecked()

    # return type: boolean
    def GetColorBalance(self):
        return self.color_correx.isChecked()

    def UsePSFFitting(self):
        return self.psf_photometry.isChecked()

class UI:
    def __init__(self):
        ui_filename = "image2sl.ui"
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

class MainWindow:
    def __init__(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dirname = self.temp_dir.name
        print("Working in temporary directory ", self.temp_dirname)
    
        super().__init__()

        global ui
        self.bias_file = FileChooser(ui.window.bias_entry,
                                     ui.window.BiasButton)

        self.dark_file = FileChooser(ui.window.dark_entry,
                                     ui.window.DarkButton)
        self.flat_file = FileChooser(ui.window.flat_entry,
                                     ui.window.FlatButton)
        self.meta_file = FileChooser(ui.window.meta_entry,
                                     ui.window.MetaButton)
        self.image_file = FileChooser(ui.window.image_filename_list,
                                      ui.window.AddImageButton,
                                      multiple_files_okay=True)

        self.options = OptionBox()
        self.progressbar = ui.window.progressBar

    ################################
    ## GenerateStarlist button
    ## starts here
    ################################
    def do_generate_starlist(self):
        self.progressbar.setRange(0,100) # indeterminate mode
        self.progressbar.setValue(20)
        self.progressbar.setTextVisible(True)
        self.progressbar.setFormat("...Running...")
        self.progressbar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressbar.show()
        self.GenerateStarlist()
        self.progressbar.hide()
    
    def GenerateStarlist(self):
        image_list = self.image_file.EnteredFilenameList()
        dark_filename = self.dark_file.EnteredFilename()
        flat_filename = self.flat_file.EnteredFilename()
        bias_filename = self.bias_file.EnteredFilename()
        metadata_filename = self.meta_file.EnteredFilename()
        do_bayer_balance = self.options.color_correx.isChecked()
        
        for image_filename in image_list:
            QGuiApplication.processEvents()
            #Skip blank lines (if present)
            if image_filename is None:
                continue
            image_filename = image_filename.strip()
            if image_filename == '':
                continue
            
            working_filename = image_filename
            (orig_dir, orig_file) = os.path.split(image_filename)
            (orig_file_base,not_used) = os.path.splitext(orig_file)
            starlist_tgtname = os.path.join(orig_dir, orig_file_base+"_$$.star")

            hdu_working = fits.open(image_filename)
            working_image = hdu_working[0].data.astype(float)
            if (dark_filename is not None
                or flat_filename is not None
                or bias_filename is not None):
                calibrated_image = os.path.join(self.temp_dirname, "light.fits")
                with fits.open(image_filename) as hdu_working:
                    working_image = hdu_working[0].data.astype(float)
                    if bias_filename is not None:
                        with fits.open(bias_filename) as hdul:
                            bias = hdul[0].data
                            working_image -= bias
                    if dark_filename is not None:
                        with fits.open(dark_filename) as hdul:
                            dark = hdul[0].data
                            working_image -= dark
                    if flat_filename is not None:
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
                if os.access(metadata_filename, os.R_OK) != True:
                    print("Cannot read metadata from file ", metadata_filename)
                    raise Exception("Cannot read metadata file")
                print("Reading metadata from ", metadata_filename)
                ReadMetaFromJSON(metadata_filename, meta)

            print("Final metadata is ", meta)
            
            if self.options.GetColorBalance():
                working_filename = BayerBalanceFile(working_filename, self.temp_dirname)
            ProcessRGBFile(working_filename, self.options, self.temp_dirname, meta, starlist_tgtname)
        return False

if __name__ == "__main__":
    global ui
    app = QtWidgets.QApplication(sys.argv)
    ui = UI()
    ui.window.show()
    not_a_window = MainWindow()

    ui.window.progressBar.hide()
    ui.window.GenerateStarlistButton.clicked.connect(not_a_window.do_generate_starlist)
    
    sys.exit(app.exec())


