"""
Unit tests for the tiered photometry configuration in :mod:`bandaid.config`.

The configuration object groups the user-settable photometry knobs (aperture
geometry, detection magnitude limits, quality cuts, and the per-telescope
detection/FWHM settings) into one immutable, validated bundle. These tests pin
two things: the defaults reproduce the legacy module-level constants exactly (so
swapping the constants for the config changes no behaviour), and the validators
reject the values that would silently break the pipeline.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from bandaid import photometry
from bandaid.config import (
    ApertureConfig,
    DetectionConfig,
    InstrumentConfig,
    PhotometryConfig,
    QualityConfig,
)

DEFAULT_GAIA_MAG_LIMIT = 15
DEFAULT_CONTAMINANT_OFFSET = 3


class TestDefaultsMatchLegacyConstants:
    """A default config reproduces the current module-level constants."""

    def test_apertures(self):
        """Aperture radii and annulus default to the photometry.py constants."""
        cfg = ApertureConfig()
        np.testing.assert_array_equal(cfg.relative_radii, photometry.RELATIVE_RADII)
        assert tuple(cfg.annulus) == tuple(photometry.ANNULUS)

    def test_detection(self):
        """The Gaia limit defaults to 15 and the contaminant limit to limit + 3."""
        cfg = DetectionConfig()
        assert cfg.gaia_mag_limit == DEFAULT_GAIA_MAG_LIMIT
        assert (
            cfg.contaminant_mag_limit
            == DEFAULT_GAIA_MAG_LIMIT + DEFAULT_CONTAMINANT_OFFSET
        )

    def test_quality(self):
        """Drift and contamination cuts default to the photometry.py constants."""
        cfg = QualityConfig()
        assert cfg.drift_tolerance_fwhm == photometry.DRIFT_TOLERANCE_FWHM
        assert cfg.drift_cap_pix == photometry.DRIFT_CAP_PIX
        assert cfg.contamination_tolerance == photometry.CONTAMINATION_TOLERANCE
        assert cfg.moffat_beta == photometry.MOFFAT_BETA

    def test_instrument(self):
        """Detection/FWHM settings default to the photometry.py constants."""
        cfg = InstrumentConfig()
        assert cfg.thresh == photometry.THRESH
        assert cfg.detection_opening == photometry.DETECTION_OPENING
        assert cfg.fwhm_cutout_half == photometry._FWHM_CUTOUT_HALF  # noqa: SLF001

    def test_photometry_config_composes_defaults(self):
        """PhotometryConfig nests one of each sub-config with default values."""
        cfg = PhotometryConfig()
        assert isinstance(cfg.apertures, ApertureConfig)
        assert isinstance(cfg.detection, DetectionConfig)
        assert isinstance(cfg.quality, QualityConfig)
        assert isinstance(cfg.instrument, InstrumentConfig)
        assert cfg.instrument.detection_opening == photometry.DETECTION_OPENING


class TestImmutability:
    """The config is frozen so a batch cannot mutate its inputs mid-run."""

    def test_cannot_mutate(self):
        """Assigning to a field on a constructed config raises."""
        cfg = PhotometryConfig()
        with pytest.raises(ValidationError):
            cfg.instrument.detection_opening = 5


class TestValidators:
    """Validators reject the values that would break the pipeline."""

    def test_annulus_inner_must_be_less_than_outer(self):
        """An inner radius larger than the outer radius is rejected."""
        with pytest.raises(ValidationError, match="annulus"):
            ApertureConfig(annulus=(8, 5))

    def test_annulus_equal_radii_rejected(self):
        """Equal inner/outer radii leave no annulus and are rejected."""
        with pytest.raises(ValidationError, match="annulus"):
            ApertureConfig(annulus=(5, 5))

    def test_negative_radius_rejected(self):
        """A negative aperture radius is rejected."""
        with pytest.raises(ValidationError):
            ApertureConfig(relative_radii=[-1.0])

    def test_zero_radius_rejected(self):
        """A zero aperture radius is rejected."""
        with pytest.raises(ValidationError):
            ApertureConfig(relative_radii=[0.0])

    def test_non_finite_contaminant_limit_rejected(self):
        """A non-finite contaminant limit is rejected with a clear message."""
        with pytest.raises(ValidationError, match="contaminant_mag_limit"):
            DetectionConfig(contaminant_mag_limit=float("inf"))

    def test_contaminant_limit_clamped_up_to_gaia(self):
        """A contaminant limit shallower than the target limit is clamped up."""
        # The legacy code clamped a too-shallow contaminant list up to the target
        # limit rather than erroring; the validator preserves that.
        cfg = DetectionConfig(gaia_mag_limit=15, contaminant_mag_limit=12)
        assert cfg.contaminant_mag_limit == cfg.gaia_mag_limit

    def test_contaminant_default_tracks_gaia(self):
        """The default contaminant limit follows a custom Gaia limit by +3."""
        gaia_limit = 14
        cfg = DetectionConfig(gaia_mag_limit=gaia_limit)
        assert cfg.contaminant_mag_limit == gaia_limit + DEFAULT_CONTAMINANT_OFFSET

    def test_negative_drift_cap_rejected(self):
        """A negative pixel cap on centroid drift is rejected."""
        with pytest.raises(ValidationError):
            QualityConfig(drift_cap_pix=-1.0)


class TestOverrides:
    """Non-default values round-trip through construction."""

    def test_instrument_override(self):
        """A custom detection opening is preserved on the nested config."""
        opening = 5
        cfg = PhotometryConfig(instrument=InstrumentConfig(detection_opening=opening))
        assert cfg.instrument.detection_opening == opening

    def test_aperture_override(self):
        """A custom annulus is preserved on the nested config."""
        annulus = (6, 10)
        cfg = PhotometryConfig(apertures=ApertureConfig(annulus=annulus))
        assert tuple(cfg.apertures.annulus) == annulus
