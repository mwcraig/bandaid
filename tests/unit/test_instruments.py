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
from _helpers import SEESTAR_RULE
from pydantic import ValidationError

from bandaid import instruments
from bandaid.config import HeaderMatchRule, InstrumentProfile
from bandaid.exceptions import InstrumentDetectionError
from bandaid.instruments import (
    available_instruments,
    detect_instrument,
    load_instrument,
    register_instrument,
)

EXPECTED_CONE_MARGIN = 0.4
EXPECTED_WCS_POINTING_TOLERANCE = 0.30
EXPECTED_WCS_SCALE_TOLERANCE = 0.005
EXPECTED_SEESTAR_PIXSCALE = 2.376
EXPECTED_SOLVE_POOL_SCALE = 0.9


class TestHeaderMatchRule:
    """Unit tests for ``HeaderMatchRule``'s header/pattern comparison."""

    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            ({"INSTRUME": "Seestar S50"}, True),
            ({"INSTRUME": "seestar s50"}, True),
            ({"INSTRUME": "SEESTAR S50"}, True),
            ({"INSTRUME": "  Seestar S50  "}, True),
            ({"INSTRUME": "Some Other Scope"}, False),
            ({}, False),
        ],
        ids=[
            "exact",
            "lowercase",
            "uppercase",
            "whitespace",
            "different-value",
            "absent-keyword",
        ],
    )
    def test_matches(self, header, expected):
        """Matching is case-insensitive and strips whitespace; absence never matches."""
        rule = HeaderMatchRule(keyword="INSTRUME", pattern="Seestar S50")
        assert rule.matches(header) is expected


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
        assert profile.solve_pool_radius_scale == default.solve_pool_radius_scale
        assert profile.cone_radius_margin == default.cone_radius_margin

    def test_cone_radius_margin_default(self):
        """The cone margin defaults to 0.4 deg, so small pointing jitter is covered."""
        assert InstrumentProfile().cone_radius_margin == EXPECTED_CONE_MARGIN
        assert load_instrument("Seestar50").cone_radius_margin == EXPECTED_CONE_MARGIN

    def test_seestar_carries_the_tight_wcs_tolerances(self):
        """
        The Seestar50 profile sets the tight WCS tolerances; the class does not.

        The class defaults stay loose (a 5% scale window, and no fixed pointing
        angle so the field radius applies) because they apply to every
        instrument; only the Seestar50 profile has the measured pixscale that
        makes 0.5% and 0.30 deg safe.
        """
        default = InstrumentProfile()
        assert default.wcs_pointing_tolerance is None
        assert default.wcs_scale_tolerance != EXPECTED_WCS_SCALE_TOLERANCE

        seestar = load_instrument("Seestar50")
        assert seestar.wcs_pointing_tolerance == EXPECTED_WCS_POINTING_TOLERANCE
        assert seestar.wcs_scale_tolerance == EXPECTED_WCS_SCALE_TOLERANCE

    def test_solve_pool_radius_scale_default(self):
        """The solve-pool scale defaults to 0.9 and the bundle does not override it."""
        assert InstrumentProfile().solve_pool_radius_scale == EXPECTED_SOLVE_POOL_SCALE
        assert (
            load_instrument("Seestar50").solve_pool_radius_scale
            == EXPECTED_SOLVE_POOL_SCALE
        )

    def test_seestar_header_pointing_is_fk5_of_date(self):
        """
        The bundled Seestar50 declares its header pointing as FK5 of date.

        The Seestar writes RA/DEC in the equinox of the observation date; a bare
        ``InstrumentProfile()`` assumes an ICRS header so a custom telescope does
        not inherit that.
        """
        profile = load_instrument("Seestar50")
        assert (profile.header_frame, profile.header_equinox) == ("fk5", "date")

        default = InstrumentProfile()
        assert (default.header_frame, default.header_equinox) == ("icrs", "J2000")

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
        assert profile.header_map["pixscale"] == pytest.approx(
            EXPECTED_SEESTAR_PIXSCALE
        )

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

    def test_bundled_profiles_have_pairwise_disjoint_header_match_rules(self):
        """
        No two bundled profiles can ever match the same header.

        ``register_instrument``'s rule-conflict check only runs inside
        ``register_instrument`` itself; the bundled profiles under
        ``meta_json_files/`` are loaded directly and never pass through it.
        Today there is only one bundled profile, so this cannot
        yet fail -- but it pins the invariant that check exists to protect,
        so a future ``meta_json_files/<Name>/profile.json`` shipped with a
        ``header_match`` rule copy-pasted from another bundled profile (or
        genuinely overlapping one) fails CI instead of making
        ``detect_instrument`` ambiguous for every user of both telescopes.
        """
        seen = {}
        for name in sorted(instruments._bundled_names()):  # noqa: SLF001
            for rule in load_instrument(name).header_match:
                identity = instruments._rule_identity(rule)  # noqa: SLF001
                assert identity not in seen, (
                    f"{name!r}'s rule {rule.keyword}=={rule.pattern!r} "
                    f"conflicts with {seen.get(identity)!r}'s"
                )
                seen[identity] = name


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

    def test_registering_existing_bundled_name_raises(self):
        """Registering over a bundled name without ``replace=True`` raises."""
        with pytest.raises(ValueError, match="Seestar50"):
            register_instrument(InstrumentProfile(name="Seestar50"))

    def test_registering_duplicate_custom_name_raises(self):
        """Re-registering the same custom name without ``replace=True`` raises."""
        register_instrument(InstrumentProfile(name="MyScope"))
        with pytest.raises(ValueError, match="MyScope"):
            register_instrument(InstrumentProfile(name="MyScope"))

    def test_replace_true_overrides_bundled_name(self):
        """``replace=True`` deliberately overrides a bundled profile."""
        custom_thresh = 9.9
        register_instrument(
            InstrumentProfile(name="Seestar50", thresh=custom_thresh), replace=True
        )
        assert load_instrument("Seestar50").thresh == custom_thresh

    def test_replace_true_with_empty_header_match_inherits_the_previous_rules(self):
        """
        ``replace=True`` with a fresh profile keeps the replaced ``header_match``.

        The documented "override a bundled telescope in-process" path --
        ``InstrumentProfile(name='Seestar50', thresh=9.9)``, the natural
        "retune one knob" shape -- used to leave ``header_match`` empty (a
        bare ``InstrumentProfile()`` defaults it to ``()``), which silently
        stripped Seestar50's detection rule: `detect_instrument` then had no
        candidates, and the next flagless run on real Seestar frames raised
        `~bandaid.exceptions.InstrumentDetectionError` for the whole batch
        with no warning pointing at ``replace=True`` as the cause. Inheriting
        the replaced profile's ``header_match`` when
        the new one carries none keeps the "retune one knob" mental model
        working.
        """
        original = load_instrument("Seestar50")
        custom_thresh = 9.9
        register_instrument(
            InstrumentProfile(name="Seestar50", thresh=custom_thresh), replace=True
        )
        replaced = load_instrument("Seestar50")
        assert replaced.thresh == custom_thresh
        assert replaced.header_match == original.header_match

    def test_replace_true_explicit_empty_header_match_stays_empty(self):
        """An explicit ``header_match=()`` on a replacement stays empty."""
        register_instrument(
            InstrumentProfile(name="Seestar50", thresh=9.9, header_match=()),
            replace=True,
        )
        assert load_instrument("Seestar50").header_match == ()

    def test_replace_true_own_header_match_is_not_overridden(self):
        """A replacement that supplies its own ``header_match`` keeps it."""
        own_rule = (HeaderMatchRule(keyword="INSTRUME", pattern="Custom Seestar"),)
        register_instrument(
            InstrumentProfile(name="Seestar50", header_match=own_rule), replace=True
        )
        assert load_instrument("Seestar50").header_match == own_rule

    def test_replace_true_inherits_the_previous_header_frame(self):
        """
        ``replace=True`` with a fresh profile keeps the replaced header frame.

        A bare ``InstrumentProfile()`` defaults to an ICRS header, so the
        "retune one knob" shape would otherwise stop converting the Seestar's
        equinox-of-date pointing and move the Gaia cone off the field.
        """
        original = load_instrument("Seestar50")
        register_instrument(
            InstrumentProfile(name="Seestar50", thresh=9.9), replace=True
        )
        replaced = load_instrument("Seestar50")
        assert (replaced.header_frame, replaced.header_equinox) == (
            original.header_frame,
            original.header_equinox,
        )

    @pytest.mark.parametrize(
        ("frame_fields", "expected"),
        [
            ({"header_frame": "icrs"}, ("icrs", "J2000")),
            ({"header_frame": "fk5"}, ("fk5", "J2000")),
            ({"header_frame": "fk5", "header_equinox": "J2025.5"}, ("fk5", "J2025.5")),
        ],
        ids=["icrs", "fk5-default-equinox", "fk5-fixed-epoch"],
    )
    def test_replace_true_own_header_frame_is_not_overridden(
        self, frame_fields, expected
    ):
        """A replacement that sets either frame field keeps both as given."""
        register_instrument(
            InstrumentProfile(name="Seestar50", **frame_fields), replace=True
        )
        replaced = load_instrument("Seestar50")
        assert (replaced.header_frame, replaced.header_equinox) == expected

    def test_replace_true_overrides_custom_name(self):
        """``replace=True`` deliberately overrides a previously-registered profile."""
        register_instrument(InstrumentProfile(name="MyScope", thresh=1.5))
        register_instrument(InstrumentProfile(name="MyScope", thresh=2.5), replace=True)
        assert load_instrument("MyScope").thresh == 2.5  # noqa: PLR2004


class TestRegisterConflicts:
    """``register_instrument`` eagerly rejects a rule that collides with another."""

    def test_conflicting_rule_raises_naming_both_profiles(self):
        """
        A new profile's rule that duplicates an existing profile's rule raises.

        Rules are exact (keyword, casefolded value) matches, so two profiles
        sharing one would make ``detect_instrument`` ambiguous on any header
        that satisfies it -- reject the registration up front rather than
        letting that surface later as a detection-time error on a real frame.
        """
        clone = InstrumentProfile(
            name="Clone",
            header_match=(SEESTAR_RULE,),
        )
        with pytest.raises(ValueError, match="Seestar50") as excinfo:
            register_instrument(clone)
        assert "Clone" in str(excinfo.value)
        # The rejected profile must not have been registered.
        assert "Clone" not in available_instruments()

    def test_different_value_same_keyword_registers_fine(self):
        """A rule on the same keyword but a different value does not conflict."""
        other = InstrumentProfile(
            name="OtherScope",
            header_match=(
                HeaderMatchRule(keyword="INSTRUME", pattern="Some Other Scope"),
            ),
        )
        register_instrument(other)
        assert load_instrument("OtherScope") is other

    def test_no_header_match_registers_fine(self):
        """A profile with no ``header_match`` rules can never conflict."""
        bare = InstrumentProfile(name="Bare")
        register_instrument(bare)
        assert load_instrument("Bare") is bare

    def test_replace_true_self_conflict_allowed(self):
        """Re-registering a profile with its own unchanged rules is not a conflict."""
        seestar = load_instrument("Seestar50")
        updated = seestar.model_copy(update={"thresh": 9.9})
        register_instrument(updated, replace=True)
        assert load_instrument("Seestar50") is updated


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
            header_match=(SEESTAR_RULE,),
        )
        # register_instrument now eagerly rejects a colliding rule, so this
        # deliberately-ambiguous fixture is inserted directly into the
        # isolated registry, bypassing that check, to exercise the
        # detection-time ambiguity error -- which remains reachable in
        # practice for bundled profiles shipped with overlapping rules.
        instruments._REGISTERED["Clone"] = clone  # noqa: SLF001

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

    def test_no_match_error_lists_available_but_not_bare_profile_as_candidate(self):
        """
        A profile with no ``header_match`` is "available" but never a "candidate".

        The error message distinguishes the two: ``NoRules`` cannot be
        auto-detected (it carries no rule to match against) but is still a
        selectable ``--instrument``/``--profile`` name, so it must appear only
        in the "all available instruments" listing, not the "auto-detection
        candidates" one.
        """
        register_instrument(InstrumentProfile(name="NoRules"))

        with pytest.raises(InstrumentDetectionError) as excinfo:
            detect_instrument({"INSTRUME": "Some Other Scope"})

        message = str(excinfo.value)
        candidates_part, available_part = message.split("all available instruments")
        assert "NoRules" not in candidates_part
        assert "NoRules" in available_part


class TestFileRoundTrip:
    """A profile serialized to a file reloads equal."""

    def test_to_file_from_file_roundtrip(self, tmp_path):
        """``to_file`` then ``from_file`` reproduces the profile exactly."""
        profile = load_instrument("Seestar50")
        path = tmp_path / "s50.json"
        profile.to_file(path)
        assert InstrumentProfile.from_file(path) == profile

    def test_from_file_rejects_removed_header_center_offset(self, tmp_path):
        """A profile file still carrying ``header_center_offset`` fails to load."""
        path = tmp_path / "old.json"
        path.write_text('{"name": "Old", "header_center_offset": [-0.32, 0.15]}')
        with pytest.raises(ValidationError, match="header_equinox"):
            InstrumentProfile.from_file(path)

    def test_from_file_accepts_null_header_center_offset(self, tmp_path):
        """A profile file with ``"header_center_offset": null`` still loads."""
        path = tmp_path / "old.json"
        path.write_text('{"name": "Old", "header_center_offset": null}')
        assert InstrumentProfile.from_file(path).header_frame == "icrs"
