# MIT License
#
# Copyright (c) 2026 AAVSO
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Add a docstring here for the init module.

This might include a very brief description of the package,
its purpose, and any important notes.
"""

import logging

from .catalog import cached_gaia_radecs
from .config import (
    ApertureConfig,
    DetectionConfig,
    InstrumentConfig,
    PhotometryConfig,
    QualityConfig,
)
from .exceptions import (
    BandaidError,
    BatchPrepError,
    FrameError,
    FrameMetadataError,
    NoUsableStarsError,
    TooFewStarsError,
    WCSSolveError,
)
from .image2sl_qt import bayer_balance_image, generate_bayer_masks
from .logging_setup import configure_logging
from .photometry import (
    ImageData,
    align,
    build_photometry_table,
    calibration_sequence,
    centroid_drift_flag,
    centroid_stars,
    eloy_to_starlist,
    measure_photometry,
    metadata_from_header,
    neighbor_contamination_flag,
    prepare_image,
)
from .scripts import prepare_batch

# Libraries should not configure logging; attach a NullHandler so the package
# can emit records without forcing handler configuration on the host
# application (and without "No handlers could be found" warnings). Callers route
# records explicitly via configure_logging().
logging.getLogger(__name__).addHandler(logging.NullHandler())
