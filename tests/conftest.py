from unittest.mock import MagicMock

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.table import MaskedColumn, Table
from photutils.datasets import make_model_image, make_noise_image

from bandaid import catalog


@pytest.fixture
def gaia_table():
    """
    A table shaped like the real ``I/345/gaia2`` result, brightest-first.

    The faintest star has *masked* proper motions with NaN beneath the mask,
    mirroring the astroquery round-trip for DR2 sources without a PM solution:
    ``MaskedColumn.quantity`` yields a NaN-bearing plain ``Quantity`` for such
    rows -- not a ``numpy.ma.MaskedArray``, so ``np.ma.filled`` on it is a
    no-op (https://github.com/mwcraig/bandaid/issues/80).
    """
    t = Table()
    t["Gmag"] = np.array([8.8, 13.5, 14.1]) * u.mag
    t["RA_ICRS"] = np.array([239.8756, 239.9220, 239.8684]) * u.deg
    t["DE_ICRS"] = np.array([25.9202, 25.8932, 25.8698]) * u.deg
    t["pmRA"] = MaskedColumn(
        [-4.220, -0.957, np.nan], unit=u.mas / u.yr, mask=[False, False, True]
    )
    t["pmDE"] = MaskedColumn(
        [12.364, -1.993, np.nan], unit=u.mas / u.yr, mask=[False, False, True]
    )
    return t


@pytest.fixture
def fake_vizier(monkeypatch, gaia_table):
    """
    Patch in a Vizier stand-in returning ``gaia_table``; yield the class mock.

    The returned mock records construction args in ``fake_vizier.call_args`` and
    the query call in ``fake_vizier.return_value.query_region.call_args``.
    """
    instance = MagicMock(name="vizier_instance")
    instance.query_region.return_value = [gaia_table]
    vizier_cls = MagicMock(name="Vizier", return_value=instance)
    monkeypatch.setattr(catalog, "Vizier", vizier_cls)
    return vizier_cls


@pytest.fixture
def starlist_metadata():
    """
    A metadata dict covering every StarList field except fwhm (set on table meta).

    Returns
    -------
    dict
        Metadata suitable for ``eloy_to_starlist`` / ``StarList.from_table``.
    """
    return {
        "obs_time": "2024-01-01T00:00:00",
        "site_lat": 40.0,
        "site_lon": -105.0,
        "site_elev": 1600.0,
        "observer": "ABC",
        "filter": "TG",
        "block_filter": "L",
        "exposure": 10.0,
        "tel_manufac": "ZWO",
        "width": 100,
        "height": 100,
        "stack": 1,
        "tel_model": "S50",
        "tel_firmware": "1.0",
        "adc_depth": 12,
        "largest_usable_adu_value": 50000,
        "egain": 0.3,
        "refframe": "ICRS",
    }


@pytest.fixture
def eloy_table():
    """
    Factory fixture building eloy-style photometry tables for StarList tests.

    Returns
    -------
    callable
        ``_make(rows, *, contaminated=None)`` -> an ``astropy.table.Table`` of the
        given per-row StarItem dicts, with ``meta["fwhm"]`` set and an optional
        ``contaminated`` column.
    """

    def _make(rows, *, contaminated=None):
        table = Table(rows)
        if contaminated is not None:
            table["contaminated"] = contaminated
        table.meta["fwhm"] = 2.5
        return table

    return _make


@pytest.fixture
def make_test_image():
    """
    Factory fixture to create test images with Gaussian sources and optional noise.

    Returns
    -------
    callable
        A function that builds a test image; see its docstring for parameters.
    """

    def _make_test_image(
        image_size,
        source_properties,
        *,
        include_noise=True,
        noise_mean=0,
        noise_stddev=1,
        seed=None,
    ):
        """
        Create a test image with Gaussian sources and optional noise.

        Parameters
        ----------
        image_size : tuple
            Size of the test image (ny, nx).
        source_properties : astropy.table.Table
            Table containing properties of the Gaussian source (amplitude, x_mean,
            y_mean, x_stddev, y_stddev).
        include_noise : bool
            Whether to include Gaussian noise in the test image.
        noise_mean : float
            Mean of the Gaussian noise to be added to the image.
        noise_stddev : float
            Standard deviation of the Gaussian noise to be added to the image.
        seed : int, optional
            Random seed for reproducibility of the noise.

        Returns
        -------
        numpy.ndarray
            The generated test image.
        """
        # Create a Gaussian2D model for the source; photutils will scale
        # this appropriately based on source properties.
        model = Gaussian2D(
            x_stddev=1,
            y_stddev=1,
        )

        # Create an image of the Gaussian source
        source_image = make_model_image(
            image_size,
            model,
            source_properties,
            x_name="x_mean",
            y_name="y_mean",
        )

        if not include_noise:
            return source_image

        # Create a noise image
        noise_image = make_noise_image(
            image_size,
            mean=noise_mean,
            stddev=noise_stddev,
            seed=seed,
        )

        # Combine the source and noise to create the final test image
        return source_image + noise_image

    return _make_test_image
