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
from functools import cache
from importlib.resources import files as package_files

from .config import InstrumentProfile

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
