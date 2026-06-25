"""
Tiered, validated configuration for the photometry pipeline.

The photometry pipeline historically carried its tuning knobs as module-level
constants in :mod:`bandaid.photometry`. Those constants fall into three risk
classes that this module makes explicit:

- **Science knobs** a user legitimately sets per run -- aperture radii, the
  background annulus, and the Gaia magnitude limits (`ApertureConfig`,
  `DetectionConfig`).
- **Quality cuts** that flag/drop suspect measurements -- centroid drift and
  bright-neighbour contamination (`QualityConfig`).
- **Instrument / per-telescope** settings that depend on the pixel scale and
  should change only when pointing a different telescope at the sky -- the
  detection threshold, the morphological-opening kernel, and the FWHM-fit window
  (`InstrumentConfig`).

The composing :class:`PhotometryConfig` bundles one of each. It is immutable
(frozen) so a batch cannot mutate its inputs mid-run, and its fields are
validated at construction so values that would silently break the pipeline are
rejected up front. The defaults reproduce the legacy constants exactly, so
existing callers that do not pass a config see no change in behaviour.

The solver-internal star counts and match tolerance (the twirl ``N_*_ALIGN``
values, ``WCS_MATCH_TOLERANCE``, ``MIN_DETECTED_STARS``, ``MIN_STARS_FOR_PAIRS``)
are deliberately *not* modelled here: mis-setting them stalls or breaks the WCS
solve, so they stay as locked module constants in :mod:`bandaid.photometry`.
"""

import numpy as np
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

# Default contaminant catalogue depth relative to the photometry target limit.
# A real star up to this many magnitudes fainter than the target limit can still
# spill into a brighter target's aperture, so contamination flagging runs against
# a list this much deeper than the measured-target list.
_CONTAMINANT_MAG_OFFSET = 3.0


class ApertureConfig(BaseModel, frozen=True):
    """
    Aperture photometry geometry, in units of the per-image FWHM.

    Attributes
    ----------
    relative_radii : tuple of float
        Aperture radii in units of FWHM; each is multiplied by the image FWHM to
        get an actual aperture size. All radii must be positive.
    annulus : tuple of float
        Background annulus ``(inner, outer)`` radii in units of FWHM, with
        ``outer > inner``.
    """

    relative_radii: tuple[float, ...] = (1.0,)
    annulus: tuple[float, float] = (5.0, 8.0)

    @field_validator("relative_radii", mode="before")
    @classmethod
    def _coerce_and_check_radii(cls, value):
        """
        Coerce a scalar or array of radii to a positive tuple of floats.

        Parameters
        ----------
        value : float or array-like
            The aperture radii, as a scalar or any array-like of radii.

        Returns
        -------
        tuple of float
            The validated radii.

        Raises
        ------
        ValueError
            If any radius is not strictly positive.
        """
        radii = tuple(float(r) for r in np.atleast_1d(np.asarray(value, dtype=float)))
        if any(r <= 0 for r in radii):
            msg = f"relative_radii must all be positive, got {radii!r}"
            raise ValueError(msg)
        return radii

    @field_validator("annulus")
    @classmethod
    def _check_annulus(cls, value):
        """
        Require the annulus inner radius to be strictly inside the outer.

        Parameters
        ----------
        value : tuple of float
            The ``(inner, outer)`` annulus radii.

        Returns
        -------
        tuple of float
            The validated annulus.

        Raises
        ------
        ValueError
            If the inner radius is not strictly less than the outer radius.
        """
        inner, outer = value
        if inner >= outer:
            msg = f"annulus inner radius must be < outer radius, got {value!r}"
            raise ValueError(msg)
        return value


class DetectionConfig(BaseModel, frozen=True):
    """
    Gaia magnitude limits for the photometry targets and contaminant catalogue.

    Attributes
    ----------
    gaia_mag_limit : float
        Magnitude limit for the photometry *targets* -- the stars actually
        measured and used to align each frame.
    contaminant_mag_limit : float
        Magnitude limit for the deeper *contaminant* catalogue used only for
        contamination flagging. If left unset it defaults to
        ``gaia_mag_limit + 3``; a value shallower than ``gaia_mag_limit`` is
        clamped up to it (the contaminant list is never shallower than the target
        list). Must be finite.
    """

    gaia_mag_limit: float = 15.0
    contaminant_mag_limit: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _resolve_contaminant_limit(cls, data):
        """
        Default, finiteness-check, and clamp the contaminant magnitude limit.

        Parameters
        ----------
        data : dict or typing.Any
            The raw construction input. Only acted on when it is a mapping; any
            other input is returned unchanged for pydantic to handle.

        Returns
        -------
        dict or typing.Any
            The input with ``contaminant_mag_limit`` resolved to a concrete,
            clamped value.

        Raises
        ------
        ValueError
            If an explicit ``contaminant_mag_limit`` is non-finite.
        """
        if not isinstance(data, dict):
            return data
        gaia = data.get("gaia_mag_limit", 15.0)
        contaminant = data.get("contaminant_mag_limit")
        if contaminant is None:
            contaminant = gaia + _CONTAMINANT_MAG_OFFSET
        elif not np.isfinite(contaminant):
            # max(nan, limit) silently returns nan, which downstream makes the
            # contaminant mask all-False and blows up as a length mismatch.
            msg = f"contaminant_mag_limit must be finite, got {contaminant!r}"
            raise ValueError(msg)
        return {**data, "contaminant_mag_limit": max(contaminant, gaia)}


class QualityConfig(BaseModel, frozen=True):
    """
    Quality cuts applied to per-star measurements.

    Attributes
    ----------
    drift_tolerance_fwhm : float
        Maximum allowed centroid drift in units of FWHM.
    drift_cap_pix : float
        Absolute pixel cap on the allowed centroid drift.
    contamination_tolerance : float
        Maximum fractional bright-neighbour spillover into the aperture before a
        star is flagged.
    moffat_beta : float
        Moffat wing index used to model neighbour spillover.
    """

    drift_tolerance_fwhm: float = Field(default=1.0, gt=0)
    drift_cap_pix: float = Field(default=4.0, gt=0)
    contamination_tolerance: float = Field(default=0.01, gt=0)
    moffat_beta: float = Field(default=3.0, gt=0)


class InstrumentConfig(BaseModel, frozen=True):
    """
    Per-telescope detection and FWHM settings (pixel-scale dependent).

    The defaults are the Seestar50 values. Change these only when pointing a
    different telescope at the sky; they depend on the plate scale and the PSF.

    Attributes
    ----------
    thresh : float
        Source-detection threshold in units of the background sigma.
    detection_opening : int
        Size of the morphological-opening kernel that gates faint detections.
    fwhm_cutout_half : int
        Half-width (px) of the square cutout used to build the PSF for the FWHM
        fit.
    """

    thresh: float = Field(default=0.5, gt=0)
    detection_opening: int = Field(default=3, ge=1)
    fwhm_cutout_half: int = Field(default=25, ge=1)


class PhotometryConfig(BaseModel, frozen=True):
    """
    The full, immutable photometry configuration carried once per batch.

    Bundles the four sub-configs so the pipeline threads a single object rather
    than a long tail of keyword arguments. The defaults reproduce the legacy
    module constants exactly.

    Attributes
    ----------
    apertures : ApertureConfig
        Aperture/annulus geometry.
    detection : DetectionConfig
        Gaia magnitude limits.
    quality : QualityConfig
        Centroid-drift and contamination cuts.
    instrument : InstrumentConfig
        Per-telescope detection/FWHM settings.
    """

    apertures: ApertureConfig = Field(default_factory=ApertureConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    instrument: InstrumentConfig = Field(default_factory=InstrumentConfig)
