"""
Pluggable per-frame output writers: how a photometered frame is recorded to disk.

`process_batch` measures each frame into a ``{filter: astropy.table.Table}``
result and then hands it to a *frame writer* to record. The bundled writer
(:func:`write_starlist_set`) reproduces the historical behaviour -- one
``StarListSet`` JSON document per frame -- but the write step is no longer
hard-wired: this module is the registry over frame writers. A caller passes a
writer callable directly (``write_frame=`` on `process_batch` /
`photometer_frames`), or registers one under a name so the CLI can select it by
``--output-format``. Adding an output format is writing a small function and
(optionally) registering it, not editing the batch loop.

A frame writer is any callable ``write(frame_result, output_path) -> Path |
list of Path``:

* ``frame_result`` is the frame's ``{filter: astropy.table.Table}`` mapping. Each
  table carries the full set of measured columns (``tot_count``, ``count_err``,
  ``snr``, ``bkgd_count``, ``airmass``, ``centroid_drift``, ...) -- a superset
  of what a ``.star`` file records -- and ``meta["full_image_meta"]`` /
  ``meta["fwhm"]``.
  A writer that wants AAVSO-starlist semantics can call
  :func:`~bandaid.photometry.good_star_mask` /
  :func:`~bandaid.photometry.eloy_to_starlist` itself.
* ``output_path`` is the resolved per-frame path (stem + the batch's
  ``output_suffix``); a writer emitting one file per filter derives per-filter
  names from it.
* the return value is stored as the frame's entry in the `process_batch` result
  mapping -- typically the ``Path`` (or list of paths) actually written.
"""

import logging

from aavso_starlist_schema import StarListSet

from .exceptions import NoUsableStarsError
from .photometry import eloy_to_starlist

logger = logging.getLogger(__name__)

# User-registered writers, keyed by name. The bundled ``starlist`` writer is
# registered at import (see bottom of module); re-registering a name overrides it.
_WRITERS = {}


def write_starlist_set(frame_result, output_path):
    """
    Write one frame as a ``StarListSet`` JSON document (the default writer).

    Reproduces bandaid's historical ``.star`` output: each filter's photometry
    table is filtered and converted to a `StarList` via
    :func:`~bandaid.photometry.eloy_to_starlist`, the per-filter lists are bundled
    into a ``StarListSet``, and the set is written as indented JSON.

    Parameters
    ----------
    frame_result : dict
        The frame's ``{filter: astropy.table.Table}`` photometry result. Each
        table must carry ``meta["full_image_meta"]`` (the StarList metadata).
    output_path : pathlib.Path
        Path to write the JSON document to.

    Returns
    -------
    pathlib.Path
        ``output_path``, the file written.

    Raises
    ------
    NoUsableStarsError
        If no filter yields any usable stars, so there is nothing to write.
        The batch loop records the frame as a skip.

    Notes
    -----
    A filter whose table yields *no* usable stars is dropped from the set
    (with a warning naming the dropped filters) rather than failing the whole
    frame -- with an SNR floor configured, an all-faint channel (e.g. TB on a
    red-star field) is routine, and the surviving channels still carry good
    data. ``StarListSet`` puts no per-filter requirement on its ``star_lists``,
    so a partial set is schema-valid.
    """
    star_lists = []
    dropped = []
    for filter_name, table in frame_result.items():
        try:
            star_lists.append(
                eloy_to_starlist(
                    table,
                    table.meta["full_image_meta"],
                    min_snr=table.meta.get("min_snr"),
                )
            )
        except NoUsableStarsError:
            dropped.append(filter_name)
    if not star_lists:
        msg = "no stars survived photometry filtering in any filter"
        raise NoUsableStarsError(msg)
    if dropped:
        logger.warning(
            "%s: dropping filter(s) %s with no stars surviving filtering; "
            "writing the %d surviving filter(s)",
            output_path.name,
            ", ".join(dropped),
            len(star_lists),
        )
    star_list_set = StarListSet(star_lists=star_lists)
    output_path.write_text(star_list_set.model_dump_json(indent=2))
    return output_path


def register_writer(name, writer):
    """
    Register a frame writer so :func:`get_writer` can resolve it by name.

    Re-registering a name (including a bundled one) overrides the previous writer.

    Parameters
    ----------
    name : str
        The registry key, e.g. the value passed to the CLI ``--output-format``.
    writer : collections.abc.Callable
        A frame writer ``write(frame_result, output_path)`` (see the module
        docstring for the contract).
    """
    _WRITERS[name] = writer


def get_writer(name):
    """
    Resolve a writer name to its callable.

    Parameters
    ----------
    name : str
        The registered writer name.

    Returns
    -------
    collections.abc.Callable
        The frame writer registered under ``name``.

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    """
    try:
        return _WRITERS[name]
    except KeyError:
        available = ", ".join(available_writers())
        msg = f"unknown output format {name!r}; available: {available}"
        raise ValueError(msg) from None


def available_writers():
    """
    Return the names of all registered writers.

    Returns
    -------
    list of str
        Sorted registered writer names.
    """
    return sorted(_WRITERS)


register_writer("starlist", write_starlist_set)
