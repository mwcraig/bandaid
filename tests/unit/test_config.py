"""
Unit tests for the tiered photometry configuration in :mod:`bandaid.config`.

The configuration object groups the user-settable photometry knobs (aperture
geometry, source-selection magnitude limits, centroid-drift cuts, and the
per-telescope detection/PSF settings) into one immutable, validated bundle. These
tests pin two things: the defaults reproduce the legacy module-level constants
exactly (so swapping the constants for the config changes no behaviour), and the
validators reject the values that would silently break the pipeline.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from bandaid.config import (
    ApertureConfig,
    DriftConfig,
    InstrumentProfile,
    PhotometryConfig,
    SourceSelectionConfig,
)

# Legacy module-level constants the config defaults must reproduce. Pinned as
# explicit literals (rather than read back off the config-derived photometry.*
# constants) so these tests actually catch an accidental default change.
EXPECTED_GAIA_MAG_LIMIT = 15
EXPECTED_CONTAMINANT_OFFSET = 3
EXPECTED_RADII = (1.0,)
EXPECTED_GAP = 4.0
EXPECTED_ANNULUS_WIDTH = 3.0
EXPECTED_ANNULUS = (5.0, 8.0)
EXPECTED_DRIFT_TOLERANCE_FWHM = 1.0
EXPECTED_DRIFT_CAP_PIX = 4.0
EXPECTED_CONTAMINATION_TOLERANCE = 0.01
EXPECTED_MOFFAT_BETA = 3.0
EXPECTED_THRESH = 0.5
EXPECTED_DETECTION_OPENING = 3
EXPECTED_FWHM_CUTOUT_HALF = 25


class TestDefaultsMatchLegacyConstants:
    """A default config reproduces the current module-level constants."""

    def test_apertures(self):
        """Aperture radii/gap/width default to the legacy literal values."""
        cfg = ApertureConfig()
        np.testing.assert_array_equal(cfg.radii, EXPECTED_RADII)
        assert cfg.gap == EXPECTED_GAP
        assert cfg.annulus_width == EXPECTED_ANNULUS_WIDTH
        # The derived annulus reproduces the legacy (inner, outer) pair exactly.
        assert tuple(cfg.annulus) == EXPECTED_ANNULUS

    def test_source_selection(self):
        """The Gaia limit defaults to 15 and the contaminant limit to limit + 3."""
        cfg = SourceSelectionConfig()
        assert cfg.gaia_mag_limit == EXPECTED_GAIA_MAG_LIMIT
        assert (
            cfg.contaminant_mag_limit
            == EXPECTED_GAIA_MAG_LIMIT + EXPECTED_CONTAMINANT_OFFSET
        )

    def test_drift(self):
        """Centroid-drift cuts default to the legacy literal values."""
        cfg = DriftConfig()
        assert cfg.drift_tolerance_fwhm == EXPECTED_DRIFT_TOLERANCE_FWHM
        assert cfg.drift_cap_pix == EXPECTED_DRIFT_CAP_PIX

    def test_instrument(self):
        """Detection/FWHM/PSF settings default to the legacy literal values."""
        cfg = InstrumentProfile()
        assert cfg.thresh == EXPECTED_THRESH
        assert cfg.detection_opening == EXPECTED_DETECTION_OPENING
        assert cfg.fwhm_cutout_half == EXPECTED_FWHM_CUTOUT_HALF
        assert cfg.contamination_tolerance == EXPECTED_CONTAMINATION_TOLERANCE
        assert cfg.moffat_beta == EXPECTED_MOFFAT_BETA

    def test_instrument_carries_seestar_header_map(self):
        """A bare profile defaults to the Seestar50 name and header dialect."""
        cfg = InstrumentProfile()
        assert cfg.name == "Seestar50"
        # The header_map is the per-frame FITS dialect (the old basic.json).
        assert cfg.header_map["obs_time"] == "@DATE-OBS"
        assert "egain" in cfg.header_map

    def test_photometry_config_composes_defaults(self):
        """PhotometryConfig nests one of each sub-config with default values."""
        cfg = PhotometryConfig()
        assert isinstance(cfg.apertures, ApertureConfig)
        assert isinstance(cfg.source_selection, SourceSelectionConfig)
        assert isinstance(cfg.drift, DriftConfig)
        assert isinstance(cfg.instrument, InstrumentProfile)


class TestImmutability:
    """The config is frozen so a batch cannot mutate its inputs mid-run."""

    def test_cannot_mutate(self):
        """Assigning to a field on a constructed config raises."""
        cfg = PhotometryConfig()
        with pytest.raises(ValidationError):
            cfg.instrument.detection_opening = 5

    def test_cannot_mutate_header_map(self):
        """
        In-place mutation of ``header_map`` raises.

        ``frozen=True`` blocks rebinding the attribute but not mutating the dict
        it points at; a user editing a shared/cached profile's ``header_map``
        would otherwise leak across every later use, so the mapping itself is
        structurally read-only.
        """
        cfg = InstrumentProfile()
        with pytest.raises(TypeError):
            cfg.header_map["obs_time"] = "@NOPE"


class TestValidators:
    """Validators reject the values that would break the pipeline."""

    @pytest.mark.parametrize("gap", [0, -1])
    def test_non_positive_gap_rejected(self, gap):
        """A zero or negative gap is rejected at construction."""
        with pytest.raises(ValidationError):
            ApertureConfig(gap=gap)

    @pytest.mark.parametrize("annulus_width", [0, -1])
    def test_non_positive_annulus_width_rejected(self, annulus_width):
        """A zero or negative annulus width is rejected at construction."""
        with pytest.raises(ValidationError):
            ApertureConfig(annulus_width=annulus_width)

    def test_annulus_geometry_is_structural(self):
        """The annulus sits strictly outside the largest aperture by construction."""
        # gap and annulus_width are positive *increments*, so for any valid config
        # max(radii) < inner_annulus < outer_annulus holds without a validator.
        cfg = ApertureConfig(radii=(2.0, 4.0), gap=1.5, annulus_width=2.0)
        assert cfg.inner_annulus == max(cfg.radii) + cfg.gap
        assert cfg.outer_annulus == cfg.inner_annulus + cfg.annulus_width
        assert max(cfg.radii) < cfg.inner_annulus < cfg.outer_annulus
        assert tuple(cfg.annulus) == (cfg.inner_annulus, cfg.outer_annulus)

    def test_negative_radius_rejected(self):
        """A negative aperture radius is rejected."""
        with pytest.raises(ValidationError):
            ApertureConfig(radii=[-1.0])

    def test_zero_radius_rejected(self):
        """A zero aperture radius is rejected."""
        with pytest.raises(ValidationError):
            ApertureConfig(radii=[0.0])

    def test_contaminant_default_tracks_gaia(self):
        """The derived contaminant limit follows a custom Gaia limit by +3."""
        gaia_limit = 14
        cfg = SourceSelectionConfig(gaia_mag_limit=gaia_limit)
        assert cfg.contaminant_mag_limit == gaia_limit + EXPECTED_CONTAMINANT_OFFSET

    def test_contaminant_offset_is_configurable(self):
        """A custom offset shifts the derived contaminant limit by that amount."""
        gaia_limit = 15
        offset = 2
        cfg = SourceSelectionConfig(
            gaia_mag_limit=gaia_limit, contaminant_mag_offset=offset
        )
        assert cfg.contaminant_mag_offset == offset
        assert cfg.contaminant_mag_limit == gaia_limit + offset

    def test_non_positive_contaminant_offset_rejected(self):
        """A zero or negative contaminant offset is rejected."""
        with pytest.raises(ValidationError):
            SourceSelectionConfig(contaminant_mag_offset=0)

    def test_non_finite_contaminant_offset_rejected(self):
        """A non-finite contaminant offset is rejected with a clear message."""
        with pytest.raises(ValidationError, match="contaminant_mag_offset"):
            SourceSelectionConfig(contaminant_mag_offset=float("inf"))

    def test_string_inputs_are_coerced(self):
        """String numeric inputs coerce cleanly instead of raising TypeError."""
        expected_gaia = 15.0
        expected_offset = 2.0
        cfg = SourceSelectionConfig(gaia_mag_limit="15", contaminant_mag_offset="2")
        assert cfg.gaia_mag_limit == expected_gaia
        assert cfg.contaminant_mag_offset == expected_offset
        assert cfg.contaminant_mag_limit == expected_gaia + expected_offset

    def test_non_finite_gaia_limit_rejected(self):
        """A non-finite Gaia magnitude limit is rejected."""
        with pytest.raises(ValidationError, match="gaia_mag_limit"):
            SourceSelectionConfig(gaia_mag_limit=float("inf"))

    def test_negative_drift_cap_rejected(self):
        """A negative pixel cap on centroid drift is rejected."""
        with pytest.raises(ValidationError):
            DriftConfig(drift_cap_pix=-1.0)


class TestOverrides:
    """Non-default values round-trip through construction."""

    def test_instrument_override(self):
        """A custom detection opening is preserved on the nested config."""
        opening = 5
        cfg = PhotometryConfig(instrument=InstrumentProfile(detection_opening=opening))
        assert cfg.instrument.detection_opening == opening

    def test_aperture_override(self):
        """Custom aperture geometry is preserved and drives the derived annulus."""
        radii, gap, annulus_width = (1.0,), 5.0, 4.0
        cfg = PhotometryConfig(
            apertures=ApertureConfig(radii=radii, gap=gap, annulus_width=annulus_width)
        )
        assert tuple(cfg.apertures.radii) == radii
        assert cfg.apertures.gap == gap
        assert cfg.apertures.annulus_width == annulus_width
        # inner = max(radii) + gap = 6.0; outer = inner + annulus_width = 10.0.
        assert tuple(cfg.apertures.annulus) == (
            max(radii) + gap,
            max(radii) + gap + annulus_width,
        )
