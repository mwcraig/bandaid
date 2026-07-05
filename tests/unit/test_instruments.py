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

from bandaid import instruments
from bandaid.config import InstrumentProfile
from bandaid.instruments import (
    available_instruments,
    load_instrument,
    register_instrument,
)


@pytest.fixture(autouse=True)
def _isolate_registry(isolate_registry):
    """
    Restore the in-process profile registry after each test.

    ``register_instrument`` mutates a module-level dict, so without this a
    registered profile would leak into later tests (e.g. the exact-set check on
    ``available_instruments``). Delegates to the shared ``isolate_registry``
    factory with this module's private registry as the target.
    """
    with isolate_registry(instruments, "_REGISTERED"):
        yield


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
        # The framing constants (issue #83) live in profile.json but must match
        # the class defaults, so the bundled and default profiles agree.
        assert profile.header_center_offset == default.header_center_offset
        assert profile.cone_radius_margin == default.cone_radius_margin

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

    def test_lists_exactly_the_bundled_profiles(self):
        """
        The bundled set is exactly the profile directories shipped.

        Pins the *complete* discovered set (not just membership) so adding or
        dropping a bundled ``meta_json_files/<name>/profile.json`` is a
        deliberate, reviewed change to this list rather than a silent one.
        """
        assert set(available_instruments()) == {"Seestar50"}


class TestRegister:
    """A user can register a custom profile and load it back by name."""

    def test_register_then_load(self):
        """A registered profile is returned by ``load_instrument`` and listed."""
        custom_thresh = 1.5
        custom = InstrumentProfile(name="MyScope", thresh=custom_thresh)
        register_instrument(custom)
        loaded = load_instrument("MyScope")
        assert loaded is custom
        assert loaded.thresh == custom_thresh
        assert "MyScope" in available_instruments()


class TestFileRoundTrip:
    """A profile serialized to a file reloads equal."""

    def test_to_file_from_file_roundtrip(self, tmp_path):
        """``to_file`` then ``from_file`` reproduces the profile exactly."""
        profile = load_instrument("Seestar50")
        path = tmp_path / "s50.json"
        profile.to_file(path)
        assert InstrumentProfile.from_file(path) == profile
