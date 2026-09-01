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
from bandaid.config import HeaderMatchRule, InstrumentProfile
from bandaid.exceptions import InstrumentDetectionError
from bandaid.instruments import (
    available_instruments,
    detect_instrument,
    load_instrument,
    register_instrument,
)


class TestHeaderMatchRule:
    """Unit tests for ``HeaderMatchRule``'s header/pattern comparison."""

    def test_matches_case_insensitively(self):
        """A differently-cased header value still matches."""
        rule = HeaderMatchRule(keyword="INSTRUME", pattern="Seestar S50")
        assert rule.matches({"INSTRUME": "seestar s50"}) is True
        assert rule.matches({"INSTRUME": "SEESTAR S50"}) is True

    def test_no_match_returns_false(self):
        """A present but different header value does not match."""
        rule = HeaderMatchRule(keyword="INSTRUME", pattern="Seestar S50")
        assert rule.matches({"INSTRUME": "Some Other Scope"}) is False

    def test_absent_keyword_returns_false(self):
        """A header without the keyword at all does not match."""
        rule = HeaderMatchRule(keyword="INSTRUME", pattern="Seestar S50")
        assert rule.matches({}) is False

    def test_whitespace_is_stripped(self):
        """Leading/trailing whitespace on the header value is ignored."""
        rule = HeaderMatchRule(keyword="INSTRUME", pattern="Seestar S50")
        assert rule.matches({"INSTRUME": "  Seestar S50  "}) is True


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
        # DR2 A/B (issue #83) found widening the cone is net harmful, so the
        # default margin is 0.0 (no widening); guard against an accidental revert.
        assert default.cone_radius_margin == 0.0

    def test_seestar_bundle_carries_header_match_rule(self):
        """
        The bundled Seestar50 profile matches on INSTRUME, unlike the bare class.

        Device identity must be opt-in (see ``InstrumentProfile.header_match``
        docs), so only the *bundled* profile carries the rule; a bare
        ``InstrumentProfile()`` -- even with identical tuning -- carries none.
        """
        profile = load_instrument("Seestar50")
        assert profile.header_match == (
            HeaderMatchRule(keyword="INSTRUME", pattern="Seestar S50"),
        )
        assert InstrumentProfile().header_match == ()

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


class TestDetectInstrument:
    """``detect_instrument`` auto-selects a profile from a frame header."""

    def test_instrume_match_selects_seestar50(self):
        """A header carrying the Seestar50 INSTRUME value resolves to it."""
        profile = detect_instrument({"INSTRUME": "Seestar S50"})
        assert profile.name == "Seestar50"

    def test_real_telescop_serial_does_not_block_detection(self):
        """
        A per-device TELESCOP serial is irrelevant to the match.

        Real Seestar frames carry ``TELESCOP='S50_<serial>'`` (a per-device
        string) alongside the stable ``INSTRUME='Seestar S50'``; only INSTRUME
        is in the bundled rule, so an arbitrary TELESCOP serial must not
        prevent detection.
        """
        profile = detect_instrument(
            {"INSTRUME": "Seestar S50", "TELESCOP": "S50_0e597e9b"}
        )
        assert profile.name == "Seestar50"

    def test_unmatched_header_raises_naming_seen_values_and_available(self):
        """No matching profile raises, naming the header values and candidates."""
        with pytest.raises(InstrumentDetectionError, match="Seestar50") as excinfo:
            detect_instrument({"INSTRUME": "Some Other Scope"})
        assert "Some Other Scope" in str(excinfo.value)

    def test_missing_both_keywords_raises(self):
        """A header with neither INSTRUME nor TELESCOP raises rather than guessing."""
        with pytest.raises(InstrumentDetectionError):
            detect_instrument({})

    def test_ambiguous_match_raises_naming_candidates(self):
        """Two registered profiles matching the same header raise, naming both."""
        clone = InstrumentProfile(
            name="Clone",
            header_match=(HeaderMatchRule(keyword="INSTRUME", pattern="Seestar S50"),),
        )
        register_instrument(clone)

        with pytest.raises(InstrumentDetectionError, match="Seestar50") as excinfo:
            detect_instrument({"INSTRUME": "Seestar S50"})
        assert "Clone" in str(excinfo.value)

    def test_empty_header_match_profile_never_auto_selected(self):
        """A profile with no header_match rules is never returned by detection."""
        # header_match=() (the bare-class default) means this profile can never
        # be a detection candidate, even though the header value happens to
        # equal its name -- there is no rule to match against.
        register_instrument(InstrumentProfile(name="NoRules"))
        with pytest.raises(InstrumentDetectionError):
            detect_instrument({"INSTRUME": "NoRules"})


class TestFileRoundTrip:
    """A profile serialized to a file reloads equal."""

    def test_to_file_from_file_roundtrip(self, tmp_path):
        """``to_file`` then ``from_file`` reproduces the profile exactly."""
        profile = load_instrument("Seestar50")
        path = tmp_path / "s50.json"
        profile.to_file(path)
        assert InstrumentProfile.from_file(path) == profile
