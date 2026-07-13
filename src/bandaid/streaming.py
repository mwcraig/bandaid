"""
Streaming transport for ``bandaid stream``.

Downloads frames one at a time from an rclone remote so a batch far larger
than the local disk can be photometered: the remote listing supplies the frame
*names* up front (they stay the keys for progress logs, results, and the QA
manifest), while each frame's bytes are fetched just in time and deleted once
its outcome is decided. rclone is shelled out to (list-form argv, never a
shell) so the user's already-configured remotes, credentials, and retry
behaviour are reused with zero new Python dependencies.
"""

import subprocess
from pathlib import Path

from .exceptions import RemoteFetchError
from .scripts import _is_fits

__all__ = [
    "fetch_remote_file",
    "list_remote_fits",
]


def _run_rclone(args):
    """
    Run one rclone command and return its completed process.

    The single seam between this module and the outside world: every rclone
    interaction goes through here, so tests fake exactly one function and the
    argv contract (list form, never a shell) is enforced in one place.

    Parameters
    ----------
    args : list of str
        The rclone subcommand and its arguments, e.g. ``["lsf", remote]``.
        Each element travels to rclone verbatim, so remotes with spaces need
        no quoting.

    Returns
    -------
    subprocess.CompletedProcess
        The finished process, with ``stdout``/``stderr`` captured as text.

    Raises
    ------
    subprocess.CalledProcessError
        If rclone exits non-zero (``check=True``); its ``stderr`` attribute
        carries rclone's explanation.
    """
    # List-form argv straight to the binary -- no shell, so remote names with
    # spaces or metacharacters are inert -- and "rclone" resolved via PATH
    # like any user-invoked tool.
    return subprocess.run(  # noqa: S603
        ["rclone", *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )


def list_remote_fits(remote):
    """
    List the FITS frame names on an rclone remote, lexically sorted.

    Only names with a recognised FITS suffix survive -- the remote may also
    hold thumbnails, logs, and the like -- and sorting makes the processing
    order (and therefore the QA manifest) reproducible run to run, matching
    the sorted order `bandaid.scripts.expand_frame_paths` gives local batches.

    Parameters
    ----------
    remote : str
        The rclone remote path, e.g. ``"gdrive:LS Psc from Rick"``. Passed to
        rclone as a single argument, so spaces need no quoting.

    Returns
    -------
    list of str
        The sorted FITS file names found directly on the remote.
    """
    listing = _run_rclone(["lsf", "--files-only", remote])
    return sorted(name for name in listing.stdout.splitlines() if _is_fits(name))


def fetch_remote_file(remote, name, dest_dir):
    """
    Download one frame from the remote into ``dest_dir``.

    A failed download must not leave a truncated FITS file behind -- a later
    run (or a cached-file short-circuit) would mistake it for a complete
    frame -- so any partial file is deleted before the failure is re-raised
    as a `RemoteFetchError`, which the batch loop treats as a recoverable
    per-frame skip.

    Parameters
    ----------
    remote : str
        The rclone remote path the frame lives on.
    name : str
        The frame's file name on the remote.
    dest_dir : str or pathlib.Path
        Local directory to download into.

    Returns
    -------
    pathlib.Path
        The local path of the downloaded frame, ``dest_dir / name``.

    Raises
    ------
    RemoteFetchError
        If rclone exits non-zero. The message carries rclone's stderr, the
        error's ``file`` is ``name``, and the ``CalledProcessError`` is
        chained as the cause.
    """
    local = Path(dest_dir) / name
    try:
        _run_rclone(["copyto", f"{remote.rstrip('/')}/{name}", str(local)])
    except subprocess.CalledProcessError as exc:
        local.unlink(missing_ok=True)
        msg = f"rclone copyto failed for {name!r}: {exc.stderr}"
        raise RemoteFetchError(msg, file=name) from exc
    return local
