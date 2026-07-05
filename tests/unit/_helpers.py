"""
Shared pure helpers for the split ``bandaid.photometry`` test modules.

These are plain importable functions (not fixtures) because they take arguments
and build objects; the split ``test_*`` files import from here rather than each
carrying its own copy. Fixtures that need pytest wiring live in ``conftest.py``.
"""

import numpy as np
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.table import Table
from astropy.wcs import WCS

from bandaid import measure_photometry
from bandaid.photometry import ANNULUS, RELATIVE_RADII, ImageData

# Fixed random seed for reproducible noise in generated test images.
SEED = 843032

# FWHM of the peak_count test scene; the bright neighbor in ``_bright_neighbor_scene``
# sits well outside the ~2*FWHM peak box at this width.
_PEAK_SCENE_FWHM = 3.0


def _single_source_photometry_inputs(make_test_image, fwhm=2.3, annulus=ANNULUS):
    """
    Build a noiseless single-source image and the ``measure_photometry`` inputs.

    Parameters
    ----------
    make_test_image : callable
        The ``make_test_image`` fixture factory.
    fwhm : float, optional
        FWHM (pixels) of the Gaussian source.
    annulus : tuple, optional
        Annulus (in FWHM) the image must be large enough to contain.

    Returns
    -------
    tuple
        ``(image, coords, fwhm, mask)`` ready to pass to ``measure_photometry``.
    """
    image_side = max(max(annulus) * fwhm * 2, 100)
    image_size = (image_side, image_side)
    source_x = image_size[1] / 2
    source_y = image_size[0] / 2
    source_properties = Table(
        {
            "amplitude": [100],
            "x_mean": [source_x],
            "y_mean": [source_y],
            "x_stddev": [fwhm * gaussian_fwhm_to_sigma],
            "y_stddev": [fwhm * gaussian_fwhm_to_sigma],
        },
    )
    image = make_test_image(
        image_size=image_size,
        source_properties=source_properties,
        include_noise=False,
        noise_mean=0,
        noise_stddev=0,
        seed=SEED,
    )
    coords = np.array([[source_x, source_y]])
    mask = np.zeros(image_size, dtype=bool)
    return image, coords, fwhm, mask


def _bright_neighbor_scene(make_test_image, fwhm=_PEAK_SCENE_FWHM, sky=10.0):
    """
    Build a scene with a faint target, a bright close neighbor, and a control.

    The target (peak ~110 with sky) has a 5000-count neighbor 10 px away --
    outside the 1-FWHM aperture and the ~2*FWHM peak box, but inside the old
    fixed 25x25 peak box. An identical isolated control star sits far from
    both. Both measured stars are centered on even (x, y) pixels, i.e. on R
    pixels of a top-down RGGB mosaic.

    Parameters
    ----------
    make_test_image : callable
        The ``make_test_image`` fixture factory.
    fwhm : float, optional
        FWHM (pixels) of the Gaussian sources.
    sky : float, optional
        Constant sky pedestal added to the image.

    Returns
    -------
    tuple
        ``(image, coords)`` where ``coords`` holds the target and the isolated
        control star (the bright neighbor is deliberately not measured).
    """
    shape = (200, 200)
    sigma = fwhm * gaussian_fwhm_to_sigma
    source_properties = Table(
        {
            "amplitude": [100.0, 5000.0, 100.0],
            "x_mean": [100.0, 110.0, 40.0],
            "y_mean": [100.0, 100.0, 40.0],
            "x_stddev": [sigma] * 3,
            "y_stddev": [sigma] * 3,
        },
    )
    image = make_test_image(shape, source_properties, include_noise=False) + sky
    coords = np.array([[100.0, 100.0], [40.0, 40.0]])
    return image, coords


def _peak_scene_photometry(image, centroid_coords, mask):
    """
    Run ``measure_photometry`` on the bright-neighbor scene.

    Parameters
    ----------
    image : numpy.ndarray
        Scene from `_bright_neighbor_scene`.
    centroid_coords : numpy.ndarray
        Measured centroid coordinates.
    mask : numpy.ndarray or None
        Bayer channel mask (True = excluded), or None for the full frame.

    Returns
    -------
    dict
        The ``measure_photometry`` output.
    """
    return measure_photometry(
        image,
        centroid_coords,
        _PEAK_SCENE_FWHM,
        1.0,
        mask,
        radii=(1.0,),
        annulus=(5.0, 8.0),
    )


def _make_tan_wcs(image_size=(500, 500), crval=(10.0, 20.0), pixscale=2.4):
    """
    Build a simple TAN WCS centered at ``crval`` for the given image size.

    ``pixscale`` (arcsec/pixel) sets the plate scale; the 2.4 default matches the
    Seestar50. Pass a different value to build a wrong-scale WCS for the plate-scale
    check tests.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [image_size[1] / 2, image_size[0] / 2]
    wcs.wcs.crval = list(crval)
    wcs.wcs.cdelt = [-pixscale / 3600, pixscale / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def _seestar_header(*, with_stackcnt=True):
    """Build a FITS header carrying the keys referenced by ``basic.json``."""
    header = fits.Header()
    header["DATE-OBS"] = "2024-01-01T00:00:00"
    header["SITELAT"] = 40.0
    header["SITELONG"] = -105.0
    header["SITEELEV"] = 1600.0
    header["obscode"] = "ABC"
    header["FILTER"] = "L"
    header["EXPTIME"] = 10.0
    header["CREATOR"] = "ZWO Seestar S50"
    header["INSTRUME"] = "Seestar S50"
    header["PROGRAM"] = "SeestarApp"
    header["BAYERPAT"] = "RGGB"
    header["DEC"] = 20.0
    header["RA"] = 10.0
    header["OBJECT"] = "WASP-12"
    header["TELESCOP"] = "Seestar"
    header["NAXIS1"] = 1080
    header["NAXIS2"] = 1920
    if with_stackcnt:
        header["STACKCNT"] = 7
    return header


def _fake_phot_factory(n_stars):
    """
    Return a stub for ``measure_photometry`` sized for ``n_stars`` sources.

    The stub bypasses the real aperture photometry (which needs realistic image
    data) so the tests can exercise only the RA/Dec column logic in
    ``build_photometry_table``. Shapes mirror the real return value: scalar-per-
    star arrays plus ``(n_stars, len(RELATIVE_RADII))`` arrays for the
    aperture-resolved quantities.
    """
    n_radii = len(RELATIVE_RADII)

    def _fake_measure_photometry(*_args: object, **_kwargs: object) -> dict:
        return {
            "tot_count": np.arange(n_stars, dtype=float),
            "count_err": np.ones(n_stars),
            "bkgd_count": np.ones(n_stars),
            "bkgd_std": np.ones(n_stars),
            "peak_count": np.ones(n_stars),
            "snr": np.ones(n_stars),
            "total_bkg": np.ones((n_stars, n_radii)),
            "fluxes": np.ones((n_stars, n_radii)),
            "aperture_radii": 1.0,
            "annulus_radii": (5.0, 8.0),
            "aperture_area": np.ones(n_stars),
        }

    return _fake_measure_photometry


def _make_image_data(
    wcs, centroid_coords, input_photometry_coords, aligned_coords=None
):
    """
    Build an ImageData with just enough fields for build_photometry_table.

    ``aligned_coords`` defaults to ``centroid_coords`` (zero drift) but can be
    supplied to exercise the centroid-drift flag.
    """
    if aligned_coords is None:
        aligned_coords = centroid_coords
    header = fits.Header()
    # Deliberately different from the resolved metadata's obs_time/airmass
    # below: the table must come from the resolved metadata, not a raw header
    # re-read (issue #59).
    header["DATE-OBS"] = "1999-12-31T00:00:00"
    header["AIRMASS"] = 9.9
    return ImageData(
        calibrated_data=np.zeros((50, 50)),
        coords=centroid_coords,
        fwhm=2.3,
        centroid_coords=centroid_coords,
        aligned_coords=aligned_coords,
        wcs=wcs,
        header=header,
        input_photometry_coords=input_photometry_coords,
        metadata={"egain": 1.0, "airmass": 1.2, "obs_time": "2020-01-01T00:00:00"},
    )


def align_coords(n):
    """Build an ``(n, 2)`` float coordinate array of ``0, 1, ... 2n-1``."""
    return np.arange(n * 2, dtype=float).reshape(n, 2)


def filter_table(tot, area, bkgd, bkgd_std, peak):
    """Build a per-channel photometry table for ``calculate_l4_quantities`` tests."""
    t = Table()
    t["tot_count"] = np.array(tot, dtype=float)
    t["aperture_area"] = np.array(area, dtype=float)
    t["bkgd_count"] = np.array(bkgd, dtype=float)
    t["bkgd_std"] = np.array(bkgd_std, dtype=float)
    t["peak_count"] = np.array(peak, dtype=float)
    return t


class _Region:
    """Minimal stand-in for an eloy detection region exposing ``.centroid``."""

    def __init__(self, y, x) -> None:
        self.centroid = (y, x)


def five_diagonal_regions(data, threshold=5, opening=5):  # noqa: ARG001
    """Detection stub returning five regions at ``(10i, 10i)`` for ``i`` in 1..5."""
    return [_Region(10 * i, 10 * i) for i in range(1, 6)]
