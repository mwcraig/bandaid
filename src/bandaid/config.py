"""
Tiered, validated configuration for the photometry pipeline.

The photometry pipeline historically carried its tuning knobs as module-level
constants in :mod:`bandaid.photometry`. Those constants fall into three risk
classes that this module makes explicit:

- **Science knobs** a user legitimately sets per run -- aperture radii, the
  background annulus, and the Gaia magnitude limits (`ApertureConfig`,
  `SourceSelectionConfig`).
- **Centroid-drift cuts** that flag/drop measurements whose centroid wandered
  (`DriftConfig`).
- **Instrument / per-telescope** settings that depend on the pixel scale, the
  PSF, and the instrument's sensitivity, and should change only when pointing a
  different telescope at the sky -- the detection threshold, the
  morphological-opening kernel, the FWHM-fit window, and the bright-neighbour
  contamination model (`InstrumentConfig`).

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

    Notes
    -----
    The background annulus is parametrised as positive *increments* outward from
    the apertures (``gap``, ``annulus_width``) rather than absolute radii. Because
    both are strictly positive, the invariant the background estimate relies on
    (``max(radii) < inner_annulus < outer_annulus``) holds by construction, so no
    cross-field validator is needed. The derived ``inner_annulus``,
    ``outer_annulus``, and ``annulus`` properties expose the resulting background
    annulus.
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


class SourceSelectionConfig(BaseModel, frozen=True):
    """
    Gaia magnitude limits selecting which catalogue stars are measured and flagged.

    These knobs do not perform source *detection* (that is the instrument's
    ``thresh``/``detection_opening``); they choose which Gaia stars become
    photometry *targets* and which deeper stars are treated as potential
    *contaminants*. The derived ``contaminant_mag_limit`` property exposes the
    resulting contaminant-catalogue depth (``gaia_mag_limit +
    contaminant_mag_offset``); tune ``contaminant_mag_offset`` rather than setting
    it directly.

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


class DriftConfig(BaseModel, frozen=True):
    """
    Centroid-drift cuts applied to per-star measurements.

    Attributes
    ----------
    drift_tolerance_fwhm : float
        Maximum allowed centroid drift in units of FWHM.
    drift_cap_pix : float
        Absolute pixel cap on the allowed centroid drift.
    """

    drift_tolerance_fwhm: Annotated[float, Field(gt=0)] = 1.0
    drift_cap_pix: Annotated[float, Field(gt=0)] = 4.0


class InstrumentConfig(BaseModel, frozen=True):
    """
    Per-telescope detection, FWHM, PSF, and sensitivity settings.

    The defaults are the Seestar50 values. Change these only when pointing a
    different telescope at the sky; they depend on the plate scale, the PSF, and
    the instrument's sensitivity to contamination.

    Attributes
    ----------
    thresh : float
        Source-detection threshold in units of the background sigma.
    detection_opening : int
        Size of the morphological-opening kernel that gates faint detections.
    fwhm_cutout_half : int
        Half-width (px) of the square cutout used to build the PSF for the FWHM
        fit.
    contamination_tolerance : float
        Maximum fractional bright-neighbour spillover into the aperture before a
        star is flagged. How much spillover is acceptable depends on the
        instrument's sensitivity, so it is an instrument setting rather than a
        per-run science knob.
    moffat_beta : float
        Moffat wing index used to model neighbour spillover -- a property of the
        instrument PSF.
    """

    thresh: Annotated[float, Field(gt=0)] = 0.5
    detection_opening: Annotated[int, Field(ge=1)] = 3
    fwhm_cutout_half: Annotated[int, Field(ge=1)] = 25
    contamination_tolerance: Annotated[float, Field(gt=0)] = 0.01
    moffat_beta: Annotated[float, Field(gt=0)] = 3.0


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
    source_selection : SourceSelectionConfig
        Gaia magnitude limits selecting the measured and flagged stars.
    drift : DriftConfig
        Centroid-drift cuts.
    instrument : InstrumentConfig
        Per-telescope detection, FWHM, PSF, and contamination settings.
    """

    apertures: ApertureConfig = Field(default_factory=ApertureConfig)
    source_selection: SourceSelectionConfig = Field(
        default_factory=SourceSelectionConfig
    )
    drift: DriftConfig = Field(default_factory=DriftConfig)
    instrument: InstrumentConfig = Field(default_factory=InstrumentConfig)
