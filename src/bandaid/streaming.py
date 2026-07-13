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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .exceptions import RemoteFetchError
from .scripts import _is_fits

__all__ = [
    "Prefetcher",
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


class Prefetcher:
    """
    Bounded look-ahead downloader for a streamed batch.

    Downloads (seconds each) overlap frame processing by running in a small
    thread pool, but only ``lookahead`` names ever have outstanding futures
    at once, so the incoming directory holds at most a handful of frames no
    matter how large the batch is. A name whose local file already exists is
    accounted as satisfied and served straight from disk -- in particular the
    first frame, which was already fetched to seed ``prepare_batch``.

    Parameters
    ----------
    remote : str
        The rclone remote path the frames live on.
    names : collections.abc.Iterable of str
        The frame names to download, in processing order.
    incoming_dir : str or pathlib.Path
        Local directory the frames are downloaded into.
    workers : int, optional
        Number of concurrent download threads. Default 2, conservative
        enough to stay under Drive rate limits.
    lookahead : int or None, optional
        Maximum number of names with outstanding downloads at once. Default
        None: ``2 * workers``, enough to keep every worker busy with the next
        download already queued behind it.
    fetch_one : collections.abc.Callable or None, optional
        ``fetch_one(remote, name, dest_dir) -> Path`` used for each download.
        Default None: `fetch_remote_file`, resolved at construction time so a
        test that patches ``streaming.fetch_remote_file`` is honoured.
    """

    def __init__(
        self,
        remote,
        names,
        incoming_dir,
        *,
        workers=2,
        lookahead=None,
        fetch_one=None,
    ) -> None:
        self._remote = remote
        self._incoming = Path(incoming_dir)
        self._fetch_one = fetch_one if fetch_one is not None else fetch_remote_file
        self._lookahead = lookahead if lookahead is not None else 2 * workers
        self._executor = ThreadPoolExecutor(max_workers=workers)
        self._names = iter(names)
        self._futures = {}
        self._top_up()

    def _top_up(self):
        """
        Submit downloads until ``lookahead`` names are outstanding.

        A name whose local file already exists is not re-downloaded: it is
        skipped here (served from disk by `fetch`) and does not count against
        the cap, which therefore bounds real in-flight downloads only.
        """
        while len(self._futures) < self._lookahead:
            name = next(self._names, None)
            if name is None:
                break
            if (self._incoming / name).exists():
                continue
            self._futures[name] = self._executor.submit(
                self._fetch_one, self._remote, name, self._incoming
            )

    def fetch(self, name):
        """
        Return the local path of ``name``, downloading it if necessary.

        Serves, in order of preference: the name's outstanding future
        (waiting for it if still in flight), the already-downloaded local
        file, or -- for a name never queued, e.g. one requested out of
        order -- a synchronous download. The future is consulted before the
        file so a download still in flight is never mistaken for a complete
        frame. Either way the queue is topped back up afterwards, keeping
        ``lookahead`` downloads in flight. A failed download raises from here
        (a `RemoteFetchError` from the transport), poisoning only this name;
        every other queued name is unaffected.

        Parameters
        ----------
        name : str
            The frame name to materialize.

        Returns
        -------
        pathlib.Path
            The frame's local path in the incoming directory.
        """
        try:
            future = self._futures.pop(name, None)
            if future is not None:
                return future.result()
            local = self._incoming / name
            if local.exists():
                return local
            return self._fetch_one(self._remote, name, self._incoming)
        finally:
            self._top_up()

    def close(self):
        """
        Cancel the outstanding downloads and shut the pool down.

        Downloads not yet started are cancelled outright; ones mid-flight are
        allowed to finish (an rclone copy cannot be interrupted from here
        anyway) and then discarded, so close returns promptly without
        hanging. Safe to call more than once.
        """
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._futures.clear()
