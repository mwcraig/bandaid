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

from pydantic import (
    BaseModel,
    Field,
    computed_field,
)


class ApertureConfig(BaseModel, frozen=True):
    """
    Aperture photometry geometry, in units of the per-image FWHM.

    The background annulus is parametrised the way ``stellarphot`` does it -- as
    positive *increments* outward from the apertures rather than absolute radii.
    Because ``gap`` and ``annulus_width`` are strictly positive, the invariant the
    background estimate relies on (``max(radii) < inner_annulus < outer_annulus``)
    holds by construction, so no cross-field validator is needed. The derived
    ``inner_annulus``, ``outer_annulus``, and ``annulus`` properties expose the
    resulting background annulus.

    Attributes
    ----------
    radii : tuple of float
        Aperture radii in units of FWHM; each is multiplied by the image FWHM to
        get an actual aperture size. All radii must be positive.
    gap : float
        Gap, in units of FWHM, between the largest aperture and the inner edge of
        the background annulus. Must be positive.
    annulus_width : float
        Radial width of the background annulus in units of FWHM. Must be positive.
    """

    radii: tuple[Annotated[float, Field(gt=0)], ...] = (1.0,)
    gap: Annotated[float, Field(gt=0)] = 4.0
    annulus_width: Annotated[float, Field(gt=0)] = 3.0

    @computed_field
    @property
    def inner_annulus(self) -> float:
        """
        Inner background-annulus radius in units of FWHM.

        Returns
        -------
        float
            ``max(radii) + gap`` -- always larger than the largest aperture
            because ``gap`` is positive.
        """
        return max(self.radii) + self.gap

    @computed_field
    @property
    def outer_annulus(self) -> float:
        """
        Outer background-annulus radius in units of FWHM.

        Returns
        -------
        float
            ``inner_annulus + annulus_width`` -- always larger than the inner
            radius because ``annulus_width`` is positive.
        """
        return self.inner_annulus + self.annulus_width

    @property
    def annulus(self) -> tuple[float, float]:
        """
        Background annulus as an ``(inner, outer)`` pair, in units of FWHM.

        Returns
        -------
        tuple of float
            ``(inner_annulus, outer_annulus)``.
        """
        return (self.inner_annulus, self.outer_annulus)


class DetectionConfig(BaseModel, frozen=True):
    """
    Gaia magnitude limits for the photometry targets and contaminant catalogue.

    The derived ``contaminant_mag_limit`` property exposes the resulting
    contaminant-catalogue depth (``gaia_mag_limit + contaminant_mag_offset``);
    tune ``contaminant_mag_offset`` rather than setting it directly.

    Attributes
    ----------
    gaia_mag_limit : float
        Magnitude limit for the photometry *targets* -- the stars actually
        measured and used to align each frame. Must be finite.
    contaminant_mag_offset : float
        How many magnitudes deeper than ``gaia_mag_limit`` the *contaminant*
        catalogue runs. Must be positive (and finite), which guarantees the
        contaminant list is always deeper than the target list.
    """

    # Finiteness is enforced by the `allow_inf_nan=False` annotation: a non-finite
    # limit makes the downstream contaminant mask silently misbehave (a nan limit
    # yields an all-False mask and a length mismatch), so it is rejected at
    # construction rather than caught deep in a batch.
    gaia_mag_limit: Annotated[float, Field(allow_inf_nan=False)] = 15.0
    contaminant_mag_offset: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 3.0

    @computed_field
    @property
    def contaminant_mag_limit(self) -> float:
        """
        Magnitude limit for the deeper contaminant-flagging catalogue.

        Returns
        -------
        float
            ``gaia_mag_limit + contaminant_mag_offset`` -- always deeper than the
            target limit because the offset is positive.
        """
        return self.gaia_mag_limit + self.contaminant_mag_offset


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
