"""
Exception hierarchy for the bandaid pipeline.

Two failure classes are distinguished so the batch driver can react correctly:

* `FrameError` and its subclasses are **recoverable per-frame** failures -- the
  frame cannot be photometered, but the rest of the batch can continue.
  `process_batch` catches this base class, logs it, and skips the frame.
* `BatchPrepError` is **fatal**: the once-per-batch preparation could not be
  built (e.g. no usable Gaia catalog), so the whole run must stop.

Anything that is *not* one of these (a genuine bug) is deliberately left to
propagate, so it is not silently swallowed as a "bad frame".
"""

__all__ = [
    "BandaidError",
    "BatchPrepError",
    "FrameError",
    "TooFewStarsError",
    "WCSSolveError",
]


class BandaidError(Exception):
    """Base class for all bandaid-specific errors."""


class FrameError(BandaidError):
    """
    A recoverable, per-frame failure: this frame cannot be processed.

    `process_batch` catches this (and its subclasses), logs the reason and the
    chained cause, and skips the frame so the rest of the batch continues.

    Parameters
    ----------
    reason : str
        Human-readable explanation of why the frame was rejected.
    file : str or pathlib.Path or None, optional
        The offending frame. Raisers that do not know the path (e.g. `align`)
        leave it None; the caller that does attaches it before re-raising.
    """

    def __init__(self, reason, *, file=None) -> None:
        self.reason = reason
        self.file = file
        super().__init__(reason)

    def __str__(self) -> str:
        """
        Render the message from ``file`` and ``reason``.

        Recomputed on each call so it picks up a ``file`` attached after
        construction -- the common "raise here, label at the caller" pattern
        where the source file is set on the error by the caller that knows it.
        """
        return f"{self.file}: {self.reason}" if self.file is not None else self.reason


class TooFewStarsError(FrameError):
    """Too few usable stars were detected to process the frame."""


class WCSSolveError(FrameError):
    """A WCS could not be solved for the frame (twirl failed or found no match)."""


class BatchPrepError(BandaidError):
    """
    The once-per-batch preparation could not be built; the run must stop.

    Raised by `prepare_batch` -- not a `FrameError`, so it is not caught by the
    per-frame loop and instead aborts the whole batch.
    """
