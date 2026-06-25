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

from typing import Annotated

import numpy as np
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

# Default contaminant catalogue depth relative to the photometry target limit,
# used only when `contaminant_mag_limit` is left unset. A real star up to this
# many magnitudes fainter than the target limit can still spill into a brighter
# target's aperture, so contamination flagging runs against a list this much
# deeper than the measured-target list. The depth is user-settable via
# `DetectionConfig.contaminant_mag_offset`; this is just its default.
_DEFAULT_CONTAMINANT_MAG_OFFSET = 3.0


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

    relative_radii: tuple[Annotated[float, Field(gt=0)], ...] = (1.0,)
    annulus: tuple[Annotated[float, Field(gt=0)], Annotated[float, Field(gt=0)]] = (
        5.0,
        8.0,
    )

    @field_validator("relative_radii", mode="before")
    @classmethod
    def _coerce_radii(cls, value):
        """
        Coerce a scalar or array of radii to a tuple of floats.

        Positivity is enforced by the ``Field(gt=0)`` annotation on the elements;
        this only normalises the shape so a scalar or any array-like is accepted.

        Parameters
        ----------
        value : float or array-like
            The aperture radii, as a scalar or any array-like of radii.

        Returns
        -------
        tuple of float
            The radii as a flat tuple of floats.
        """
        return tuple(float(r) for r in np.atleast_1d(np.asarray(value, dtype=float)))

    @model_validator(mode="after")
    def _check_annulus_geometry(self):
        """
        Require ``max(relative_radii) < annulus_inner < annulus_outer``.

        The background annulus must sit strictly outside the largest photometry
        aperture (otherwise the background estimate is taken from inside the star)
        and have a positive width. These are cross-field constraints, so they live
        here rather than on a single field.

        Returns
        -------
        ApertureConfig
            The validated config.

        Raises
        ------
        ValueError
            If the inner radius is not strictly inside the outer radius, or does
            not strictly exceed the largest aperture radius.
        """
        inner, outer = self.annulus
        if inner >= outer:
            msg = f"annulus inner radius must be < outer radius, got {self.annulus!r}"
            raise ValueError(msg)
        largest_aperture = max(self.relative_radii)
        if inner <= largest_aperture:
            msg = (
                f"annulus inner radius ({inner}) must exceed the largest aperture "
                f"radius ({largest_aperture})"
            )
            raise ValueError(msg)
        return self


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
        ``gaia_mag_limit + contaminant_mag_offset``; a value shallower than
        ``gaia_mag_limit`` is clamped up to it (the contaminant list is never
        shallower than the target list). Must be finite.
    contaminant_mag_offset : float
        How many magnitudes deeper than ``gaia_mag_limit`` the defaulted
        contaminant catalogue runs. Only consulted when ``contaminant_mag_limit``
        is left unset. Must be positive.
    """

    # Finiteness is enforced by the `allow_inf_nan=False` annotation: a non-finite
    # limit makes max()/the downstream contaminant mask silently misbehave (a nan
    # limit yields an all-False mask and a length mismatch), so it is rejected at
    # construction rather than caught deep in a batch.
    gaia_mag_limit: Annotated[float, Field(allow_inf_nan=False)] = 15.0
    contaminant_mag_limit: Annotated[float, Field(allow_inf_nan=False)] | None = None
    contaminant_mag_offset: Annotated[float, Field(gt=0)] = (
        _DEFAULT_CONTAMINANT_MAG_OFFSET
    )

    @model_validator(mode="before")
    @classmethod
    def _resolve_contaminant_limit(cls, data):
        """
        Default and clamp the contaminant magnitude limit before field validation.

        Defaulting and clamping are inherently cross-field, so they run here on the
        raw input. Positivity (``contaminant_mag_offset``) and finiteness
        (``*_mag_limit``) are left to the field annotations, which run afterwards;
        this validator only coerces the numeric inputs it needs for the arithmetic
        so a stringly-typed value (e.g. from a TOML/env source) yields a clean
        ``ValidationError`` from the field rather than a raw ``TypeError`` here.

        Parameters
        ----------
        data : dict or typing.Any
            The raw construction input. Only acted on when it is a mapping; any
            other input is returned unchanged for pydantic to handle.

        Returns
        -------
        dict or typing.Any
            The input with ``contaminant_mag_limit`` resolved to a concrete,
            clamped value when it could be computed.
        """
        if not isinstance(data, dict):
            return data

        def _as_float(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        gaia = _as_float(data.get("gaia_mag_limit", 15.0))
        offset = _as_float(
            data.get("contaminant_mag_offset", _DEFAULT_CONTAMINANT_MAG_OFFSET)
        )
        raw_contaminant = data.get("contaminant_mag_limit")
        contaminant = None if raw_contaminant is None else _as_float(raw_contaminant)

        # If any input we need could not be coerced, leave the data untouched and
        # let pydantic's field validation raise the appropriate ValidationError.
        if (
            gaia is None
            or offset is None
            or (raw_contaminant is not None and contaminant is None)
        ):
            return data
        if contaminant is None:
            contaminant = gaia + offset
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

    drift_tolerance_fwhm: Annotated[float, Field(gt=0)] = 1.0
    drift_cap_pix: Annotated[float, Field(gt=0)] = 4.0
    contamination_tolerance: Annotated[float, Field(gt=0)] = 0.01
    moffat_beta: Annotated[float, Field(gt=0)] = 3.0


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

    thresh: Annotated[float, Field(gt=0)] = 0.5
    detection_opening: Annotated[int, Field(ge=1)] = 3
    fwhm_cutout_half: Annotated[int, Field(ge=1)] = 25


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
