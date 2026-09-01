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


def _bundled_names():
    """
    Return the names of the bundled profiles.

    Returns
    -------
    list of str
        Subdirectories of ``meta_json_files`` that hold a ``profile.json``.
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
    override a bundled telescope in-process via :func:`register_instrument`.

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


def register_instrument(profile):
    """
    Register a profile so :func:`load_instrument` can resolve it by name.

    Re-registering a name (bundled or not) overrides the previous profile.

    Parameters
    ----------
    profile : InstrumentProfile
        The profile to register; its ``name`` is the registry key.
    """
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
        candidate rules reference and the available/ambiguous profile names.
    """
    profiles = [load_instrument(name) for name in available_instruments()]
    rules = [(profile, rule) for profile in profiles for rule in profile.header_match]
    matched = sorted({profile.name for profile, rule in rules if rule.matches(header)})

    if len(matched) == 1:
        return load_instrument(matched[0])

    seen = {rule.keyword: header.get(rule.keyword) for _, rule in rules}
    available = ", ".join(available_instruments())
    if not matched:
        msg = (
            f"no bundled/registered instrument profile's header_match matched "
            f"this frame's header (checked {seen}); available instruments: "
            f"{available}"
        )
    else:
        msg = (
            f"ambiguous instrument: {', '.join(matched)} all matched this "
            f"frame's header (checked {seen}); available instruments: {available}"
        )
    raise InstrumentDetectionError(msg)


def resolve_config_instrument(config, header):
    """
    Return ``config`` (unchanged or with ``instrument`` auto-detected) and how.

    The single "``None`` means resolve from the header" step shared by the two
    places a header is in hand early enough to do it:
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
    if config.instrument is not None:
        return config, False
    detected = detect_instrument(header)
    logger.info(
        "auto-detected instrument profile %r from the frame header", detected.name
    )
    return config.model_copy(update={"instrument": detected}), True
