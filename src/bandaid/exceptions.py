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
    "DegenerateBayerChannelError",
    "FrameError",
    "FrameMetadataError",
    "InstrumentDetectionError",
    "NoUsableStarsError",
    "StarListValidationError",
    "TooFewStarsError",
    "WCSPointingError",
    "WCSScaleError",
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


class WCSScaleError(WCSSolveError):
    """
    Every WCS solve for the frame came back at an implausible plate scale.

    twirl can return a self-consistent but geometrically wrong WCS (wrong plate
    scale). Such a solve is normally *recovered*, not dropped: the scale check
    rejects it and retries a deeper Gaia pool, which usually finds the
    correct-scale match. This error is raised only in the residual case where
    *every* pool solves out of tolerance, so the frame cannot be recovered and is
    skipped. It subclasses `WCSSolveError` so the batch loop still skips the
    frame, while staying distinguishable in logs/manifests from a genuine
    "no match".
    """


class WCSPointingError(WCSSolveError):
    """
    Every WCS solve for the frame landed far from the queried field center.

    The Gaia catalog is queried at the frame header's pointing, so a solved WCS
    whose frame center is more than one field radius (half-diagonal) from that
    location is a mispointed (false-asterism) solve -- the catalog would barely
    overlap the frame. Like a wrong-scale solve it is normally *recovered*: the
    pointing check rejects it and retries a deeper Gaia pool. This error is
    raised only when every pool solves mispointed. It subclasses `WCSSolveError`
    so the batch loop still skips the frame, while staying distinguishable in
    logs/manifests.
    """


class FrameMetadataError(FrameError):
    """A required FITS header keyword is missing or could not be parsed."""


class NoUsableStarsError(FrameError):
    """No stars survived photometry filtering, so the frame yields no output."""


class StarListValidationError(FrameError):
    """
    A photometry row passed filtering but failed AAVSO starlist validation.

    `good_star_mask` mirrors the schema constraints it knows about (finite
    positive counts, in-bounds nonnegative centers, nonnegative peak), but the
    schema is a git-pinned dependency and can grow constraints the mask does
    not check. This wrapper translates the pydantic ``ValidationError`` raised
    by ``StarList.from_table`` into a `FrameError`, so a schema rejection
    costs at worst the one frame instead of aborting the whole batch.
    """


class DegenerateBayerChannelError(FrameError):
    """A Bayer channel's pixel sample was empty or had zero variance."""


class BatchPrepError(BandaidError):
    """
    The once-per-batch preparation could not be built; the run must stop.

    Raised by `prepare_batch` -- not a `FrameError`, so it is not caught by the
    per-frame loop and instead aborts the whole batch.
    """


class InstrumentDetectionError(BatchPrepError):
    """
    A frame header did not resolve to exactly one registered instrument profile.

    Raised by `~bandaid.instruments.detect_instrument` when a
    `~bandaid.config.PhotometryConfig` carries no explicit ``instrument`` (the
    default): zero profiles matched the frame's header (no bundled/registered
    instrument claims it) or more than one did (an ambiguous registration).
    Fatal like its parent `BatchPrepError` -- an unresolvable instrument means
    the whole batch has no detection/PSF settings to run with, not just one bad
    frame. The message names the header values seen and the candidate profile
    names so the fix (register the right profile, or pass ``--instrument``
    explicitly) is obvious from the error alone.
    """
