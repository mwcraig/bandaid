"""
Named instrument profiles: the registry of telescopes the pipeline can process.

An :class:`~bandaid.config.InstrumentProfile` bundles a telescope's detection
tuning with its per-frame FITS-header dialect (``header_map``). This module is
the registry over those profiles: it discovers the ones bundled with the package
(one ``meta_json_files/<name>/profile.json`` per telescope), lets a user register
their own in-process, and resolves a name to a profile. Adding a telescope is
dropping in a new ``profile.json`` (or calling :func:`register_instrument`), not
editing code.

The metadata layer that resolves a profile's ``header_map`` against a frame's
FITS header lives in :func:`~bandaid.photometry.metadata_from_header`; the
observer-identity layer (site/observer overrides applied last) is the separate
``user_specific_metadata`` dict threaded through the batch and is not modelled
here.
"""

import json
import logging
from functools import cache
from importlib.resources import files as package_files

from .config import InstrumentProfile
from .exceptions import InstrumentDetectionError

logger = logging.getLogger(__name__)

_META_DIR = "meta_json_files"
_PROFILE_FILENAME = "profile.json"

# User-registered profiles, keyed by name. Checked before the bundled profiles so
# a caller can override a bundled telescope in-process.
_REGISTERED: dict[str, InstrumentProfile] = {}


def _profiles_root():
    """
    Return a traversable for the bundled ``meta_json_files`` directory.

    Returns
    -------
    importlib.resources.abc.Traversable
        The package's ``meta_json_files`` directory.
    """
    return package_files("bandaid").joinpath(_META_DIR)


def _profile_path(name):
    """
    Return a traversable for a bundled instrument's ``profile.json``.

    Parameters
    ----------
    name : str
        The instrument name (the ``meta_json_files`` subdirectory).

    Returns
    -------
    importlib.resources.abc.Traversable
        Path to ``meta_json_files/<name>/profile.json``.
    """
    return _profiles_root().joinpath(name, _PROFILE_FILENAME)


@cache
def _bundled_names():
    """
    Return the names of the bundled profiles.

    Returns
    -------
    list of str
        Subdirectories of ``meta_json_files`` that hold a ``profile.json``.

    Notes
    -----
    Cached: the bundled directory does not change within a process, and this
    is on the per-frame batch-mixing-guard path (`~bandaid.scripts.
    check_frame_consistency`), so an uncached walk here would cost a
    directory listing on every frame.
    """
    return [
        entry.name
        for entry in _profiles_root().iterdir()
        if entry.is_dir() and entry.joinpath(_PROFILE_FILENAME).is_file()
    ]


@cache
def _load_bundled(name):
    """
    Load and cache a bundled profile by name.

    Parameters
    ----------
    name : str
        The instrument name.

    Returns
    -------
    InstrumentProfile
        The validated bundled profile.
    """
    return InstrumentProfile.model_validate_json(_profile_path(name).read_text())


def default_header_map():
    """
    Return the bundled Seestar50 ``header_map`` (the bare-class default).

    Reads the profile file directly (without constructing an
    :class:`~bandaid.config.InstrumentProfile`) so it can serve as the
    ``header_map`` default factory for that class without recursing. Seestar50
    is the bare-class default instrument; the ``header_map`` of any other
    bundled profile is reached via ``load_instrument(name).header_map``.

    Returns
    -------
    dict
        The Seestar50 ``header_map`` (its per-frame FITS-header dialect).
    """
    return json.loads(_profile_path("Seestar50").read_text())["header_map"]


def load_instrument(name):
    """
    Resolve an instrument name to its profile.

    Registered profiles take precedence over the bundled ones, so a caller can
    override a bundled telescope in-process via
    ``register_instrument(profile, replace=True)``.

    Parameters
    ----------
    name : str
        The instrument name.

    Returns
    -------
    InstrumentProfile
        The profile for ``name``.

    Raises
    ------
    ValueError
        If ``name`` is neither registered nor bundled.
    """
    if name in _REGISTERED:
        return _REGISTERED[name]
    if name in _bundled_names():
        return _load_bundled(name)
    available = ", ".join(available_instruments())
    msg = f"unknown instrument {name!r}; available: {available}"
    raise ValueError(msg)


def _rule_identity(rule):
    """
    Return a rule's normalised identity for conflict comparison.

    Parameters
    ----------
    rule : HeaderMatchRule
        The rule to normalise.

    Returns
    -------
    tuple of str
        ``(keyword.upper(), pattern.strip().casefold())``.

    Notes
    -----
    Two rules that agree on this pair are a conflict when they belong to
    differently-named profiles: they match exactly the same header values for
    an `astropy.io.fits.Header` input, whose ``.get`` is itself
    case-insensitive on the keyword. This normalisation is *not* the one
    :meth:`~bandaid.config.HeaderMatchRule.matches` applies for a plain
    `collections.abc.Mapping` input, though -- ``matches`` does no keyword-case
    folding of its own there, so a mapping whose keys are not uppercase FITS
    convention can disagree with this conflict check (issue #122 follow-up).
    """
    return (rule.keyword.upper(), rule.pattern.strip().casefold())


def register_instrument(profile, *, replace=False):
    """
    Register a profile so :func:`load_instrument` can resolve it by name.

    Parameters
    ----------
    profile : InstrumentProfile
        The profile to register; its ``name`` is the registry key.
    replace : bool, optional
        Whether to allow overriding a name that already resolves. Default
        False. When True, ``profile`` is also exempted from the rule-conflict
        check against its own prior registration (registering the *same*
        name with the *same* rules is not a conflict).

    Raises
    ------
    ValueError
        If ``profile.name`` is already registered or bundled and ``replace``
        is False, or if any of ``profile.header_match`` collides with a rule
        on a differently-named existing profile.

    Notes
    -----
    Two checks run before the registry is touched, both meant to catch a
    mistake at registration time rather than letting it surface later as a
    confusing detection-time failure on a real frame:

    - **Duplicate name.** Registering a name that already resolves (bundled or
      previously registered) raises unless ``replace=True`` is passed, so an
      accidental name collision does not silently shadow the wrong profile.
      ``replace=True`` keeps the deliberate "override a bundled telescope
      in-process" use case working. When ``replace=True`` and ``profile``
      carries no ``header_match`` of its own, the replaced profile's
      ``header_match`` is inherited (rather than silently emptied), so the
      "retune one knob" shape -- e.g.
      ``InstrumentProfile(name='Seestar50', thresh=9.9)`` -- keeps that name
      auto-detectable; pass an explicit ``header_match`` to change it too.
    - **Rule conflict.** A new profile whose ``header_match`` shares an exact
      ``(keyword, value)`` pair with a *differently-named* existing profile is
      rejected: `detect_instrument` cannot tell the two apart on a header that
      satisfies that rule, so the ambiguity is caught here instead of on some
      later frame.
    """
    existing_names = set(available_instruments())
    if profile.name in existing_names and not replace:
        msg = (
            f"instrument {profile.name!r} is already registered; pass "
            "replace=True to override it deliberately"
        )
        raise ValueError(msg)

    if replace and not profile.header_match and profile.name in existing_names:
        # The "retune one knob" override shape --
        # InstrumentProfile(name='Seestar50', thresh=9.9) -- otherwise leaves
        # header_match at the bare-class default (empty), silently stripping
        # the replaced profile's detection rule: detect_instrument would then
        # have no candidates for this name at all (issue #122).
        previous = load_instrument(profile.name)
        if previous.header_match:
            profile = profile.model_copy(update={"header_match": previous.header_match})

    new_rules = {_rule_identity(rule): rule for rule in profile.header_match}
    if new_rules:
        for other_name in sorted(existing_names):
            if replace and other_name == profile.name:
                continue
            other = load_instrument(other_name)
            for other_rule in other.header_match:
                identity = _rule_identity(other_rule)
                if identity in new_rules:
                    msg = (
                        f"instrument {profile.name!r}'s header_match rule "
                        f"{new_rules[identity].keyword}=="
                        f"{new_rules[identity].pattern!r} conflicts with "
                        f"already-registered instrument {other_name!r}'s rule "
                        f"{other_rule.keyword}=={other_rule.pattern!r} -- "
                        "detect_instrument could never tell them apart"
                    )
                    raise ValueError(msg)

    _REGISTERED[profile.name] = profile


def available_instruments():
    """
    Return the names of all resolvable instruments.

    Returns
    -------
    list of str
        Sorted union of the bundled and registered profile names.
    """
    return sorted(set(_bundled_names()) | set(_REGISTERED))


def detect_instrument(header):
    """
    Resolve a frame header to exactly one instrument profile.

    Every bundled/registered profile whose ``header_match`` is non-empty is a
    candidate; it matches the header if *any* of its rules match (OR). A
    profile with an empty ``header_match`` (the bare-class default) is never a
    candidate -- device identity must be opt-in. Used to resolve
    `~bandaid.config.PhotometryConfig.instrument` when it is left as ``None``
    (auto-detect) -- see `~bandaid.scripts.prepare_batch` and
    `~bandaid.photometry.prepare_image`.

    Parameters
    ----------
    header : astropy.io.fits.Header or collections.abc.Mapping
        The frame header (or header-like mapping) to match against.

    Returns
    -------
    InstrumentProfile
        The single matching profile.

    Raises
    ------
    InstrumentDetectionError
        If zero or more than one profile matches, naming the header values the
        candidate rules reference, the auto-detection candidates (the
        profiles a header could possibly resolve to), and separately the full
        set of available instrument names (including profiles that carry no
        ``header_match`` and so can never be auto-detected, only chosen
        explicitly).
    """
    profiles = [load_instrument(name) for name in available_instruments()]
    candidates = [profile for profile in profiles if profile.header_match]
    matched_profiles = [
        profile for profile in candidates if profile.matches_header(header)
    ]

    if len(matched_profiles) == 1:
        return matched_profiles[0]

    matched = sorted({profile.name for profile in matched_profiles})
    seen = {
        rule.keyword: header.get(rule.keyword)
        for profile in candidates
        for rule in profile.header_match
    }
    candidate_names = ", ".join(sorted({profile.name for profile in candidates}))
    available = ", ".join(available_instruments())
    detail = (
        f"auto-detection candidates: {candidate_names or 'none'}; all available "
        f"instruments (pass --instrument/--profile explicitly): {available}"
    )
    if not matched:
        msg = (
            f"no bundled/registered instrument profile's header_match matched "
            f"this frame's header (checked {seen}); {detail}"
        )
    else:
        msg = (
            f"ambiguous instrument: {', '.join(matched)} all matched this "
            f"frame's header (checked {seen}); {detail}"
        )
    raise InstrumentDetectionError(msg)


def resolve_profile(profile, header):
    """
    Return ``profile`` unchanged, or auto-detect and log it from ``header``.

    The single "``None`` means resolve from the header" step, shared by every
    place that accepts a possibly-unset profile: this used to be
    reimplemented inline, without logging, by
    `~bandaid.photometry.metadata_from_header`, while
    `resolve_config_instrument` (below) logged the detected name -- so a
    standalone `~bandaid.photometry.metadata_from_header` or
    `~bandaid.photometry.calibration_sequence` call left no trace of which
    instrument was picked (PR #122 follow-up). Both now funnel through this.

    Parameters
    ----------
    profile : InstrumentProfile or None
        The profile to return unchanged, or None to auto-detect from
        ``header``.
    header : astropy.io.fits.Header or collections.abc.Mapping
        The frame header to detect from; only consulted when ``profile`` is
        None.

    Returns
    -------
    resolved : InstrumentProfile
        ``profile`` unchanged, or the profile detected from ``header``. When
        ``profile`` needs resolving, `detect_instrument` may raise
        `~bandaid.exceptions.InstrumentDetectionError` (zero or more than one
        profile matched the header); that propagates unchanged.
    auto_detected : bool
        True if ``profile`` was resolved by detection (the incoming
        ``profile`` was None); False if it was already set explicitly.
    """
    if profile is not None:
        return profile, False
    detected = detect_instrument(header)
    logger.info(
        "auto-detected instrument profile %r from the frame header", detected.name
    )
    return detected, True


def resolve_config_instrument(config, header):
    """
    Return ``config`` (unchanged or with ``instrument`` auto-detected) and how.

    A thin `~bandaid.config.PhotometryConfig`-shaped wrapper around
    :func:`resolve_profile`, used by the two places a header is in hand early
    enough to resolve the instrument from it:
    `~bandaid.scripts.prepare_batch` (the batch path) and
    `~bandaid.photometry.prepare_image` (the direct/per-frame path). The
    second return value tells `~bandaid.scripts.prepare_batch` whether to mark
    its `~bandaid.scripts.BatchPrep.instrument_auto_detected`, which gates
    `~bandaid.scripts.check_frame_consistency`'s batch-mixing guard.

    Parameters
    ----------
    config : PhotometryConfig
        The config whose ``instrument`` may need resolving.
    header : astropy.io.fits.Header or collections.abc.Mapping
        The frame header to detect from; only consulted when
        ``config.instrument`` is None.

    Returns
    -------
    config : PhotometryConfig
        ``config`` unchanged if ``instrument`` was already set, otherwise a
        copy with the detected profile. When ``instrument`` needs resolving,
        `~bandaid.instruments.detect_instrument` may raise
        `~bandaid.exceptions.InstrumentDetectionError` (zero or more than one
        profile matched the header); that propagates unchanged.
    auto_detected : bool
        True if ``instrument`` was resolved by detection (the incoming
        ``config.instrument`` was None); False if it was already set
        explicitly.
    """
    detected, auto_detected = resolve_profile(config.instrument, header)
    if not auto_detected:
        return config, False
    return config.model_copy(update={"instrument": detected}), True
