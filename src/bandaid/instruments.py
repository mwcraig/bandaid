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

A profile can also name *star flaggers* (``star_flaggers``): instrument-specific
code, registered here by name with :func:`register_star_flagger`, that marks
stars whose measurement the camera itself has spoiled. :func:`apply_star_flags`
runs a profile's flaggers on one frame. Naming flaggers on the profile, rather
than keying them on the profile's ``name``, keeps a bare or user-built profile
free of instrument code unless it opts in.
"""

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib.resources import files as package_files

import numpy as np

from .config import InstrumentProfile
from .exceptions import InstrumentDetectionError

logger = logging.getLogger(__name__)

__all__ = [
    "StarFlagInputs",
    "StarFlagger",
    "apply_star_flags",
    "available_instruments",
    "default_header_map",
    "detect_instrument",
    "load_instrument",
    "register_instrument",
    "register_star_flagger",
    "resolve_config_instrument",
    "resolve_profile",
    "star_flaggers_for",
]

_META_DIR = "meta_json_files"
_PROFILE_FILENAME = "profile.json"

# User-registered profiles, keyed by name. Checked before the bundled profiles so
# a caller can override a bundled telescope in-process.
_REGISTERED: dict[str, InstrumentProfile] = {}

# Registered star flaggers, keyed by the name a profile's ``star_flaggers`` uses.
_STAR_FLAGGERS: dict[str, "StarFlagger"] = {}


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
    convention can disagree with this conflict check.
    """
    return (rule.keyword.upper(), rule.pattern.strip().casefold())


def _inherit_header_identity(profile, previous):
    """
    Carry a replaced profile's header identity over to its replacement.

    Parameters
    ----------
    profile : InstrumentProfile
        The replacement profile.
    previous : InstrumentProfile
        The profile being replaced.

    Returns
    -------
    InstrumentProfile
        ``profile``, with ``header_match``, ``star_flaggers`` and the
        ``header_frame``/``header_equinox`` pair taken from ``previous``
        wherever ``profile`` did not set them explicitly.
    """
    inherited = {}
    # The "retune one knob" override shape --
    # InstrumentProfile(name='Seestar50', thresh=9.9) -- otherwise leaves
    # header_match at the bare-class default (empty), silently stripping
    # the replaced profile's detection rule: detect_instrument would then
    # have no candidates for this name at all.
    if "header_match" not in profile.model_fields_set and previous.header_match:
        inherited["header_match"] = previous.header_match
    # Likewise the bare-class star_flaggers is empty, which would silently
    # switch off the replaced profile's instrument flags.
    if "star_flaggers" not in profile.model_fields_set and previous.star_flaggers:
        inherited["star_flaggers"] = previous.star_flaggers
    # Likewise the bare-class header frame is ICRS, which would silently stop
    # converting the replaced profile's header pointing. The two fields are
    # only meaningful together, so they are inherited as a pair.
    if profile.model_fields_set.isdisjoint({"header_frame", "header_equinox"}):
        inherited["header_frame"] = previous.header_frame
        inherited["header_equinox"] = previous.header_equinox
    return profile.model_copy(update=inherited) if inherited else profile


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
      omits ``header_match``, the replaced profile's ``header_match`` is
      inherited (rather than silently emptied), so the "retune one knob"
      shape -- e.g. ``InstrumentProfile(name='Seestar50', thresh=9.9)`` --
      keeps that name auto-detectable. An explicitly supplied
      ``header_match``, including ``()``, is honoured as given.
      ``star_flaggers`` is inherited the same way, so the replacement keeps
      running the replaced profile's instrument flags unless it sets
      ``star_flaggers`` itself (``()`` to run none).
      ``header_frame`` and ``header_equinox`` are inherited the same way, as a
      pair, when ``profile`` sets neither, so the header pointing keeps being
      converted as before.
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

    if replace and profile.name in existing_names:
        profile = _inherit_header_identity(profile, load_instrument(profile.name))

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
        explicitly). The error's ``matched`` attribute holds the names of the
        profiles that matched (empty when none did).
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
    raise InstrumentDetectionError(msg, matched=tuple(matched))


def resolve_profile(profile, header):
    """
    Return ``profile`` unchanged, or auto-detect and log it from ``header``.

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

    Notes
    -----
    The single "``None`` means resolve from the header" step, shared by
    every place that accepts a possibly-unset profile:
    `resolve_config_instrument` and `~bandaid.photometry.metadata_from_header`
    previously reimplemented it separately (one logged the detected name,
    one didn't); both now funnel through here.
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


# eq=False: the generated __eq__ would compare the numpy fields, whose truth
# value is ambiguous; keep identity semantics, like LoadedFrame.
@dataclass(frozen=True, eq=False)
class StarFlagInputs:
    """
    What a star flagger sees of one frame.

    Attributes
    ----------
    raw : numpy.ndarray
        The frame's pixel data exactly as read from the FITS file (the
        camera's integers), not the floating-point copy photometry runs on.
    centroids : numpy.ndarray
        ``(N, 2)`` pixel ``x, y`` of the measured stars; row ``i`` is row ``i``
        of every channel table.
    bayer_masks : collections.abc.Mapping
        Channel name to Bayer mask (True where a pixel is *not* in that
        channel, the `~bandaid.image2sl_qt.generate_bayer_masks` convention),
        or None for a channel measured over the whole frame.
    peak_count : collections.abc.Mapping
        Channel name to the ``(N,)`` ``peak_count`` column of that channel's
        table.
    bkgd_count : collections.abc.Mapping
        Channel name to the ``(N,)`` per-pixel sky (``bkgd_count``) of that
        channel's table.
    metadata : collections.abc.Mapping
        The frame's metadata (``tel_firmware``, ``system_id``, ...), as
        resolved by the profile's ``header_map``.
    """

    raw: np.ndarray
    centroids: np.ndarray
    bayer_masks: Mapping
    peak_count: Mapping
    bkgd_count: Mapping
    metadata: Mapping


@dataclass(frozen=True)
class StarFlagger:
    """
    Instrument-specific code that flags spoiled stars on a frame.

    Attributes
    ----------
    column : str
        Name of the boolean column the combined flag is written to on every
        filter's table, e.g. ``"core_replaced"``.
    flag : collections.abc.Callable
        ``flag(inputs, state) -> {channel: bool array}``: given the frame's
        `StarFlagInputs` and the flagger's per-batch state (None when the
        batch prepared none), return one ``(N,)`` boolean array per channel,
        True for a flagged star. A channel left out of the mapping flags
        nothing.
    prepare : collections.abc.Callable or None
        ``prepare(raw_frames, metadata) -> state``: build the per-batch state
        from a list of raw frame arrays and the metadata of the frame the
        batch is prepared from, with the user's metadata overrides applied:
        the same mapping a frame's ``StarFlagInputs.metadata`` carries (e.g.
        ``height``, ``width`` and ``largest_usable_adu_value``). Called once
        per batch. None (default) for a
        flagger that needs no state.
    n_state_frames : int
        How many raw frames, from the start of the batch, ``prepare`` wants.
        Default 0 (no state).
    """

    column: str
    flag: Callable[[StarFlagInputs, object], Mapping]
    prepare: Callable[[Sequence[np.ndarray], Mapping], object] | None = None
    n_state_frames: int = 0


def register_star_flagger(name, flagger, *, replace=False):
    """
    Register a star flagger so a profile can name it in ``star_flaggers``.

    Parameters
    ----------
    name : str
        The registry key a profile's ``star_flaggers`` refers to.
    flagger : StarFlagger
        The flagger to register.
    replace : bool, optional
        Whether to allow overriding a name that is already registered.
        Default False.

    Raises
    ------
    ValueError
        If ``name`` is already registered and ``replace`` is False.
    """
    if _is_star_flagger(name) and not replace:
        msg = (
            f"star flagger {name!r} is already registered; pass replace=True "
            "to override it deliberately"
        )
        raise ValueError(msg)
    _STAR_FLAGGERS[name] = flagger


def _is_star_flagger(name):
    """
    Return whether ``name`` is a registered star flagger.

    Parameters
    ----------
    name : str
        Star-flagger name.

    Returns
    -------
    bool
        True when `star_flaggers_for` can resolve ``name``.
    """
    return name in _STAR_FLAGGERS


def _star_flagger(name):
    """
    Look up a registered star flagger.

    Parameters
    ----------
    name : str
        A name for which `_is_star_flagger` is True.

    Returns
    -------
    StarFlagger
        The registered flagger.
    """
    return _STAR_FLAGGERS[name]


def star_flaggers_for(profile):
    """
    Resolve a profile's ``star_flaggers`` names to the registered flaggers.

    Parameters
    ----------
    profile : InstrumentProfile
        The profile whose flaggers to resolve.

    Returns
    -------
    tuple of (str, StarFlagger)
        ``(name, flagger)`` pairs in the profile's order; empty when the
        profile names none.

    Raises
    ------
    ValueError
        If the profile names a flagger that is not registered, or names two
        flaggers that write the same ``column``.

    Notes
    -----
    Two flaggers may not share a ``column``: each flag column has to say
    which flagger fired, and a second flagger would overwrite the first.
    """
    unknown = [name for name in profile.star_flaggers if not _is_star_flagger(name)]
    if unknown:
        available = ", ".join(sorted(_STAR_FLAGGERS))
        msg = (
            f"instrument {profile.name!r} names unknown star flagger(s) "
            f"{', '.join(map(repr, unknown))}; registered: {available}"
        )
        raise ValueError(msg)
    flaggers = tuple((name, _star_flagger(name)) for name in profile.star_flaggers)
    owners = {}
    for name, flagger in flaggers:
        if flagger.column in owners:
            msg = (
                f"instrument {profile.name!r} names star flaggers "
                f"{owners[flagger.column]!r} and {name!r}, which both write "
                f"column {flagger.column!r}"
            )
            raise ValueError(msg)
        owners[flagger.column] = name
    return flaggers


def apply_star_flags(by_filter_data, inputs, profile, *, state=None):
    """
    Run a profile's star flaggers on one frame and stamp their flags.

    Parameters
    ----------
    by_filter_data : dict of {str: astropy.table.Table}
        The frame's per-channel photometry tables, rows aligned with
        ``inputs.centroids``. Modified in place.
    inputs : StarFlagInputs
        What the flaggers see of the frame.
    profile : InstrumentProfile
        The frame's instrument; its ``star_flaggers`` names the flaggers.
    state : collections.abc.Mapping or None, optional
        Flagger name to the per-batch state its ``prepare`` built. A flagger
        missing from it, or every flagger when None (default), gets None.

    Raises
    ------
    ValueError
        If `star_flaggers_for` rejects the profile, or a flagger's ``column``
        is already a column of a table that does not list it in
        ``meta["reject_flags"]``.

    Notes
    -----
    Each flagger's per-channel flags are combined with OR -- a star flagged
    in any channel is flagged -- and the combined boolean column is written,
    under the flagger's ``column``, to *every* table, so the star is dropped
    from every filter's star list. The column name is appended to each
    table's ``meta["reject_flags"]`` (a tuple), which
    `~bandaid.photometry.good_star_mask` reads to reject flagged rows.

    A column already listed in a table's ``reject_flags`` -- left by an
    earlier run -- is overwritten. Any other existing column of that name is
    measured data, so the frame fails before any flagger runs rather than
    replace it with a flag.

    When the profile names no flagger the tables are left untouched: no
    column, no ``reject_flags`` key.
    """
    flaggers = star_flaggers_for(profile)
    for _, flagger in flaggers:
        for filter_name, table in by_filter_data.items():
            if (
                flagger.column in table.colnames
                and flagger.column not in table.meta.get("reject_flags", ())
            ):
                msg = (
                    f"star flagger column {flagger.column!r} is already a column "
                    f"of the {filter_name!r} table and is not a reject flag"
                )
                raise ValueError(msg)
    state = state or {}
    n_stars = len(inputs.centroids)
    for name, flagger in flaggers:
        flags = flagger.flag(inputs, state.get(name))
        combined = np.zeros(n_stars, dtype=bool)
        for channel_flags in flags.values():
            combined |= np.asarray(channel_flags, dtype=bool)
        for table in by_filter_data.values():
            table[flagger.column] = combined
            reject_flags = tuple(table.meta.get("reject_flags", ()))
            if flagger.column not in reject_flags:
                reject_flags = (*reject_flags, flagger.column)
            table.meta["reject_flags"] = reject_flags
