"""Unit-only fixtures shared across the split test modules."""

from types import SimpleNamespace

import numpy as np
import pytest
from _helpers import _CONSISTENT_HEADER, _make_tan_wcs

from bandaid import scripts


@pytest.fixture
def _consistent_headers(monkeypatch):
    """
    Stub fits.getheader so every frame passes check_frame_consistency.

    process_batch reads each frame's header unconditionally; the process_batch
    tests use fake paths and exercise the processing/output paths, not the
    consistency check, so return a header that matches _dummy_prep for all of them.
    """
    monkeypatch.setattr(
        scripts.fits, "getheader", lambda _file: dict(_CONSISTENT_HEADER)
    )


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
            side_effect=lambda coords, _radecs, **_kwargs: (coords, _make_tan_wcs()),
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
