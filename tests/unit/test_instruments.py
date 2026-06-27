"""
Unit tests for the named instrument-profile registry in :mod:`bandaid.instruments`.

An :class:`~bandaid.config.InstrumentProfile` bundles the two halves of "what a
telescope is": the detection/PSF tuning knobs and the per-frame FITS-header
dialect (``header_map``). The registry exposes the bundled profiles by name and
lets a user register or load their own from a file. These tests pin that the
bundled Seestar50 profile reproduces the class defaults, that the registry can be
extended, and that a profile round-trips through ``to_file``/``from_file``.
"""

import pytest

from bandaid.config import InstrumentProfile
from bandaid.instruments import (
    available_instruments,
    load_instrument,
    register_instrument,
)


class TestLoadInstrument:
    """``load_instrument`` returns the bundled profile for a known name."""

    def test_seestar_tuning_matches_class_defaults(self):
        """The bundled Seestar50 tuning equals a bare ``InstrumentProfile()``."""
        profile = load_instrument("Seestar50")
        default = InstrumentProfile()
        assert profile.name == "Seestar50"
        assert profile.thresh == default.thresh
        assert profile.detection_opening == default.detection_opening
        assert profile.fwhm_cutout_half == default.fwhm_cutout_half
        assert profile.contamination_tolerance == default.contamination_tolerance
        assert profile.moffat_beta == default.moffat_beta

    def test_seestar_header_map_carries_dialect(self):
        """The bundled profile carries the Seestar header dialect."""
        profile = load_instrument("Seestar50")
        assert profile.header_map["obs_time"] == "@DATE-OBS"
        assert profile.header_map["egain"] == pytest.approx(0.3116)

    def test_unknown_instrument_raises(self):
        """An unregistered, unbundled name raises rather than guessing."""
        with pytest.raises(ValueError, match="NoSuchScope"):
            load_instrument("NoSuchScope")


class TestAvailableInstruments:
    """``available_instruments`` lists the bundled profiles."""

    def test_lists_seestar(self):
        """Seestar50 is discoverable as a bundled profile."""
        assert "Seestar50" in available_instruments()


class TestRegister:
    """A user can register a custom profile and load it back by name."""

    def test_register_then_load(self):
        """A registered profile is returned by ``load_instrument`` and listed."""
        custom = InstrumentProfile(name="MyScope", thresh=1.5)
        register_instrument(custom)
        assert load_instrument("MyScope") is custom
        assert "MyScope" in available_instruments()


class TestFileRoundTrip:
    """A profile serialized to a file reloads equal."""

    def test_to_file_from_file_roundtrip(self, tmp_path):
        """``to_file`` then ``from_file`` reproduces the profile exactly."""
        profile = load_instrument("Seestar50")
        path = tmp_path / "s50.json"
        profile.to_file(path)
        assert InstrumentProfile.from_file(path) == profile
