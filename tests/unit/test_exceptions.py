"""
Unit tests for the bandaid exception hierarchy and converted raise sites.

``align`` now raises ``WCSSolveError``; the ``calibration_sequence`` ->
``TooFewStarsError`` conversion is exercised by the fixtured tests in
``test_photometry.py``.
"""

import numpy as np
import pytest

from bandaid import photometry
from bandaid.exceptions import (
    BandaidError,
    BatchPrepError,
    FrameError,
    RemoteFetchError,
    TooFewStarsError,
    WCSSolveError,
)


class TestHierarchy:
    """The class relationships the batch driver relies on."""

    def test_frame_errors_are_recoverable_base(self):
        """The per-frame errors share the recoverable ``FrameError`` base."""
        assert issubclass(TooFewStarsError, FrameError)
        assert issubclass(WCSSolveError, FrameError)
        assert issubclass(FrameError, BandaidError)

    def test_batch_prep_error_is_not_a_frame_error(self):
        """``BatchPrepError`` is fatal, so it must not be a ``FrameError``."""
        assert issubclass(BatchPrepError, BandaidError)
        assert not issubclass(BatchPrepError, FrameError)

    def test_message_renders_reason_only_without_file(self):
        """With no file attached the message is just the reason."""
        err = WCSSolveError("twirl found no match")
        assert str(err) == "twirl found no match"
        assert err.file is None

    def test_remote_fetch_error_is_a_frame_error(self):
        """A failed download skips just that frame, so it must be a ``FrameError``."""
        assert issubclass(RemoteFetchError, FrameError)

    def test_remote_fetch_error_carries_file(self):
        """The remote name rides along via the standard ``file=`` kwarg."""
        err = RemoteFetchError("rclone copyto failed", file="frame_0001.fit")
        assert str(err) == "frame_0001.fit: rclone copyto failed"
        assert err.reason == "rclone copyto failed"
        assert err.file == "frame_0001.fit"

    def test_file_can_be_attached_after_construction(self):
        """The 'raise here, label at the caller' pattern updates the message."""
        err = WCSSolveError("twirl found no match")
        err.file = "frame.fits"
        assert str(err) == "frame.fits: twirl found no match"
        assert err.reason == "twirl found no match"


class TestAlignRaisesWCSSolveError:
    """``align`` turns a failed solve into a typed, recoverable error."""

    def test_none_result_raises(self, monkeypatch):
        """A None result from twirl (no match) becomes a ``WCSSolveError``."""
        monkeypatch.setattr(photometry, "compute_wcs", lambda *a, **k: None)
        coords = radecs = np.zeros((20, 2))
        with pytest.raises(WCSSolveError):
            photometry.align(coords, radecs)

    def test_underlying_exception_is_wrapped_and_chained(self, monkeypatch):
        """An error from twirl is wrapped, preserving the original as __cause__."""
        boom = ValueError("only one matching pair of points")

        def _raise(*_args: object, **_kwargs: object):
            raise boom

        monkeypatch.setattr(photometry, "compute_wcs", _raise)
        coords = radecs = np.zeros((20, 2))
        with pytest.raises(WCSSolveError) as excinfo:
            photometry.align(coords, radecs)
        # The original error is preserved for the log, not swallowed.
        assert excinfo.value.__cause__ is boom
