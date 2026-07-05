from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.table import MaskedColumn, Table
from astropy.wcs import WCS
from photutils.datasets import make_model_image, make_noise_image

from bandaid import catalog, scripts
from bandaid.image2sl_qt import generate_bayer_masks


def _default_tan_wcs(image_size=(500, 500), crval=(10.0, 20.0), pixscale=2.4):
    """Build a TAN WCS for the ``align`` stub's default return value."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [image_size[1] / 2, image_size[0] / 2]
    wcs.wcs.crval = list(crval)
    wcs.wcs.cdelt = [-pixscale / 3600, pixscale / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


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


@pytest.fixture
def bayer_masks_rggb():
    """
    Factory building top-down RGGB Bayer masks for a given frame shape.

    Returns
    -------
    callable
        ``_make(shape, *, append_l4=False)`` -> the ``{channel: mask}`` mapping
        from ``generate_bayer_masks``.
    """

    def _make(shape, *, append_l4=False):
        return generate_bayer_masks(
            shape,
            {"bayerpat": "RGGB", "roworder": "top-down", "ybayroff": 0},
            append_l4=append_l4,
        )

    return _make


@pytest.fixture
def isolate_registry():
    """
    Factory context manager: snapshot/restore a module-level registry dict.

    Replaces the near-identical ``_isolate_registry`` autouse fixtures in the
    writer and instrument test modules. Each test file keeps a tiny autouse
    wrapper that calls this with its own ``(module, attr)`` target.

    Returns
    -------
    callable
        ``isolate(module, attr)`` -> a context manager restoring
        ``getattr(module, attr)`` (a dict) to its prior contents on exit.
    """

    @contextmanager
    def _isolate(module, attr):
        registry = getattr(module, attr)
        saved = dict(registry)
        try:
            yield
        finally:
            registry.clear()
            registry.update(saved)

    return _isolate


@pytest.fixture
def by_filter(eloy_table, starlist_metadata):
    """
    Factory for a ``{filter: Table}`` photometry result like ``process_one_image``.

    Each filter's table carries two good (finite, positive, in-bounds) rows plus
    the ``meta["fwhm"]`` and ``meta["full_image_meta"]`` that
    ``process_batch`` -> ``eloy_to_starlist`` requires. Both rows survive the
    converter's filtering, so each written StarList has two stars.

    Parameters
    ----------
    eloy_table : callable
        Fixture building an eloy-style photometry table from per-row dicts.
    starlist_metadata : dict
        Fixture providing the StarList metadata stored on each table.

    Returns
    -------
    callable
        ``_make(filters=("TR", "TG"))`` -> the per-filter table mapping.
    """
    rows = [
        {
            "x": 20.0,
            "y": 30.0,
            "ra": 10.0,
            "dec": 20.0,
            "tot_count": 100.0,
            "count_err": 5.0,
            "bkgd_count": 1.0,
            "peak_count": 200.0,
        },
        {
            "x": 70.0,
            "y": 60.0,
            "ra": 11.0,
            "dec": 21.0,
            "tot_count": 300.0,
            "count_err": 7.0,
            "bkgd_count": 1.0,
            "peak_count": 400.0,
        },
    ]

    def _make(filters=("TR", "TG")):
        result = {}
        for filter_name in filters:
            table = eloy_table(rows)
            table.meta["full_image_meta"] = starlist_metadata
            result[filter_name] = table
        return result

    return _make


@pytest.fixture
def patched_process_one_image(monkeypatch):
    """
    Factory patching ``scripts.process_one_image`` to return a fixed result.

    Replaces the repeated
    ``monkeypatch.setattr(scripts, "process_one_image", lambda *a, **k: ...)``
    preamble in the batch/disk tests.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        The built-in monkeypatch fixture used to install the patch.

    Returns
    -------
    callable
        ``_patch(result)`` -> installs the patch and returns ``result``.
    """

    def _patch(result):
        monkeypatch.setattr(scripts, "process_one_image", lambda *_a, **_k: result)
        return result

    return _patch


@pytest.fixture
def stub_prepare_image_externals(mocker):
    """
    Factory patching the four externals ``prepare_image`` reaches, via ``mocker``.

    Patches ``calibration_sequence`` (returns a configurable
    ``(calibrated, metadata, coords, fwhm, None)`` 5-tuple), ``align`` (returns
    ``(coords, wcs)``), ``centroid_stars`` (identity) and ``fits.getheader``
    (``{"creator": "spy"}``). Returns the four mocks so callers can assert on
    their ``.call_args``; override ``.return_value`` / ``.side_effect`` to tune a
    single external per test.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        The pytest-mock fixture used to patch the externals.

    Returns
    -------
    callable
        ``_stub(*, metadata=None, coords=None, calibrated=None, fwhm=2.0)`` -> a
        namespace with ``.calibration_sequence``, ``.align``, ``.centroid_stars``
        and ``.getheader`` mocks.
    """

    def _stub(*, metadata=None, coords=None, calibrated=None, fwhm=2.0):
        if metadata is None:
            metadata = {"creator": "spy", "pixscale": 2.4}
        if coords is None:
            coords = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        if calibrated is None:
            calibrated = np.zeros((10, 10))
        calibration_sequence = mocker.patch(
            "bandaid.photometry.calibration_sequence",
            return_value=(calibrated, metadata, coords, fwhm, None),
        )
        align = mocker.patch(
            "bandaid.photometry.align",
            side_effect=lambda coords, _radecs, **_kwargs: (coords, _default_tan_wcs()),
        )
        centroid_stars = mocker.patch(
            "bandaid.photometry.centroid_stars",
            side_effect=lambda _data, coords, _cnn: coords,
        )
        getheader = mocker.patch(
            "bandaid.photometry.fits.getheader",
            side_effect=lambda _file: {"creator": "spy"},
        )
        return SimpleNamespace(
            calibration_sequence=calibration_sequence,
            align=align,
            centroid_stars=centroid_stars,
            getheader=getheader,
        )

    return _stub
