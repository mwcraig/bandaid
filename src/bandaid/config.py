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
  morphological-opening kernel, the FWHM-fit window, the bright-neighbour
  contamination model, and the per-frame FITS-header dialect (`header_map`)
  that maps that telescope's headers onto the metadata the pipeline needs
  (`InstrumentProfile`).

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

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated

from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_serializer,
    field_validator,
)


def _default_seestar_header_map() -> dict:
    """
    Return the Seestar50 per-frame FITS-header dialect.

    Used as the ``header_map`` default for :class:`InstrumentProfile` so a
    bare profile behaves exactly like the historical hard-coded Seestar
    template. Imported lazily to avoid a circular import (``instruments``
    imports this module).

    Returns
    -------
    dict
        The Seestar50 ``header_map`` (the old ``basic.json`` content).
    """
    from .instruments import default_header_map  # noqa: PLC0415

    return default_header_map()


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


class InstrumentProfile(BaseModel, frozen=True):
    """
    A named telescope: detection/FWHM/PSF settings plus its FITS-header dialect.

    The defaults are the Seestar50 values. Change these only when pointing a
    different telescope at the sky; they depend on the plate scale, the PSF, and
    the instrument's sensitivity to contamination. A profile bundles the two
    halves of "what a telescope is": the detection tuning knobs *and* the
    ``header_map`` that resolves that telescope's per-frame FITS header into the
    metadata the pipeline needs. Named profiles live in
    :mod:`bandaid.instruments`; use :func:`~bandaid.instruments.load_instrument`
    to fetch a bundled one and :meth:`from_file`/:meth:`to_file` to share a
    user-tuned profile.

    Attributes
    ----------
    name : str
        The instrument's name, used as its registry key.
    thresh : float
        Source-detection threshold in units of the background sigma.
    detection_opening : int
        Size of the morphological-opening kernel that gates faint detections.
    fwhm_cutout_half : int
        Half-width (px) of the square cutout used to build the PSF for the FWHM
        fit.
    fwhm_n_stars : int
        Cap on how many of the brightest unsaturated detections feed the single
        FWHM fit. The fit needs only a handful of well-exposed stars; feeding
        every faint detection both slows the CNN re-centroiding and inflates the
        FWHM (the CNN mis-centroids faint sources, smearing the stacked PSF).
    contamination_tolerance : float
        Maximum fractional bright-neighbour spillover into the aperture before a
        star is flagged, relative to the target flux the aperture encloses (the
        *measured* flux). How much spillover is acceptable depends on the
        instrument's sensitivity, so it is an instrument setting rather than a
        per-run science knob.
    moffat_beta : float
        Moffat wing index used to model neighbour spillover -- a property of the
        instrument PSF.
    contamination_seeing_margin : float
        Seeing-pessimism factor for the once-per-batch bright-neighbour
        contamination flag. The flag is computed from the *first* frame's FWHM
        but applied to every frame of the batch, so it is evaluated at
        ``first-frame FWHM * contamination_seeing_margin``: pairs that would
        become contaminated as seeing softens during the night are dropped up
        front. Must be ``>= 1``; ``1.0`` evaluates the flag at exactly the
        first frame's seeing.
    wcs_scale_tolerance : float
        Maximum fractional deviation of a solved plate scale from the
        instrument's expected pixscale before the WCS is rejected as a
        wrong-scale solve (see :func:`~bandaid.photometry.align`). It is an
        instrument setting because it is a tolerance on *this* telescope's plate
        scale; the empirical basis for the ``0.05`` default is in issue #83.
    header_center_offset : tuple of float or None
        The fixed sky vector ``(Delta(RA*cos(dec)), Delta(dec))`` in degrees from
        this instrument's header pointing to the *true* field center. The Seestar
        reports a pointing that sits ~0.35 deg off the frame center (mid-left of
        the field), so centering the Gaia cone on the raw header clips the far
        side of the field and starves the plate-solve matcher (issue #83). When
        present -- and the class default carries the Seestar offset --
        :func:`~bandaid.scripts.resolve_field_center` walks the header pointing to
        the field center by this vector. Setting it to ``None`` restores the
        historical behaviour of centering on the raw header (routing through the
        ``from_name`` fallback), which is correct for an instrument whose header
        already points at the field center.
    cone_radius_margin : float
        Extra field radius in degrees added to ``fov_rad`` when the Gaia cone is
        centered on a resolved center (``header_center_offset`` estimate or an
        object-name lookup). ``0.0`` (the default) leaves the query radius
        unchanged, querying exactly the field. A live-DR2 A/B on SS Leo (issue
        #83, 635 frames over two nights) found that widening the cone is *net
        harmful*: the extra edge stars reshuffle the brightest-N asterisms fed to
        the plate-solver, breaking frames that solved on the unwidened cone
        (0.1 deg margin was net -46 frames vs 0.0). Keep it 0.0 unless a specific
        instrument is shown to need a buffer.
    header_map : collections.abc.Mapping
        The per-frame FITS-header dialect for this telescope: a mapping of
        metadata key to a directive resolved by
        :func:`~bandaid.photometry.metadata_from_header` (``@KEY`` header
        lookups, ``!`` function calls, ``#key`` fallbacks, and plain literals).
        Stored as a read-only mapping so a shared/cached profile cannot be
        mutated in place; serialises back to a plain ``dict``.
    """

    name: str = "Seestar50"
    thresh: Annotated[float, Field(gt=0)] = 0.5
    detection_opening: Annotated[int, Field(ge=1)] = 5
    fwhm_cutout_half: Annotated[int, Field(ge=1)] = 25
    fwhm_n_stars: Annotated[int, Field(ge=1)] = 25
    contamination_tolerance: Annotated[float, Field(gt=0)] = 0.01
    moffat_beta: Annotated[float, Field(gt=0)] = 3.0
    # ge=1: a sub-unity margin would *un*-flag pairs the measured first-frame
    # seeing already contaminates, silently shipping blended photometry.
    contamination_seeing_margin: Annotated[float, Field(ge=1.0)] = 1.25
    wcs_scale_tolerance: Annotated[float, Field(gt=0)] = 0.05
    # Seestar50 values (the class defaults are the Seestar): the header pointing
    # sits ~0.35 deg off the field center, so the default pipeline must walk to
    # the true center. The cone is NOT widened (margin 0.0): a live-DR2 A/B on SS
    # Leo showed widening reshuffles the plate-solver asterisms and loses frames
    # (issue #83). A different telescope overrides these (set header_center_offset
    # to None to fall back to object-name resolution).
    header_center_offset: tuple[float, float] | None = (-0.32, 0.15)
    cone_radius_margin: Annotated[float, Field(ge=0)] = 0.0
    header_map: Mapping = Field(
        default_factory=_default_seestar_header_map, validate_default=True
    )

    @field_validator("header_map", mode="after")
    @classmethod
    def _freeze_header_map(cls, value) -> Mapping:
        """
        Make ``header_map`` structurally read-only.

        ``frozen=True`` only blocks rebinding the attribute, not mutating the
        dict it points at. The bundled profiles are cached and shared, so an
        in-place edit would leak globally; wrapping the mapping in a
        :class:`~types.MappingProxyType` makes such mutation raise. The values
        are scalars, so a shallow freeze is sufficient.

        Parameters
        ----------
        value : Mapping
            The validated ``header_map`` contents.

        Returns
        -------
        Mapping
            A read-only view over a private copy of ``value``.
        """
        return MappingProxyType(dict(value))

    @field_serializer("header_map")
    def _serialize_header_map(self, value) -> dict:
        """
        Serialise ``header_map`` back to a plain ``dict`` for JSON output.

        Parameters
        ----------
        value : Mapping
            The read-only ``header_map`` view.

        Returns
        -------
        dict
            A plain dict so :meth:`to_file`/``model_dump_json`` round-trip.
        """
        return dict(value)

    @classmethod
    def from_file(cls, path) -> "InstrumentProfile":
        """
        Load a profile from a JSON file written by :meth:`to_file`.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a JSON file holding a serialised profile.

        Returns
        -------
        InstrumentProfile
            The validated profile.
        """
        return cls.model_validate_json(Path(path).read_text())

    def to_file(self, path) -> None:
        """
        Serialise this profile to a JSON file readable by :meth:`from_file`.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination path for the serialised profile.
        """
        Path(path).write_text(self.model_dump_json(indent=2))


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
    instrument : InstrumentProfile
        The named telescope: detection, FWHM, PSF, and contamination settings
        plus the per-frame FITS-header dialect.
    """

    apertures: ApertureConfig = Field(default_factory=ApertureConfig)
    source_selection: SourceSelectionConfig = Field(
        default_factory=SourceSelectionConfig
    )
    drift: DriftConfig = Field(default_factory=DriftConfig)
    instrument: InstrumentProfile = Field(default_factory=InstrumentProfile)
