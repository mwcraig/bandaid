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

import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eloy.ballet.model import Ballet

from .config import PhotometryConfig
from .exceptions import RemoteFetchError
from .scripts import _is_fits, _quiet_hf_xet, prepare_batch, process_batch
from .writers import write_starlist_set

__all__ = [
    "Prefetcher",
    "fetch_remote_file",
    "list_remote_fits",
    "stream_frames",
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


def stream_frames(
    remote,
    *,
    config=None,
    cnn=None,
    weights=None,
    user_specific_metadata=None,
    append_l4=True,
    output_dir=".",
    output_suffix=".star",
    write_frame=write_starlist_set,
    fail_fast=False,
    write_qa_manifest=True,
    incoming_dir=None,
    keep_local=False,
    download_workers=2,
):
    """
    Photometer every FITS frame on an rclone remote, downloading just in time.

    The streaming counterpart of `bandaid.scripts.photometer_frames` and the
    convenience behind ``bandaid stream``: it lists the remote, builds the
    Ballet centroider, fetches the first frame to seed `prepare_batch`, and
    runs `process_batch` over the remote *names* with a `Prefetcher` supplying
    each frame's bytes just before they are needed and deleting them once the
    frame's outcome is decided. Local disk therefore holds only a handful of
    frames at a time, so a batch far larger than the disk can be processed;
    the remote is never modified.

    Parameters
    ----------
    remote : str
        The rclone remote path holding the frames, e.g.
        ``"gdrive:LS Psc from Rick"``.
    config : PhotometryConfig or None, optional
        Configuration carried through the batch. None (default) uses a
        default `PhotometryConfig` (Seestar50).
    cnn : object or None, optional
        A pre-built Ballet centroider. None (default) builds one from
        ``weights``.
    weights : str or None, optional
        Path to Ballet weights used when ``cnn`` is None; None downloads the
        defaults from HuggingFace.
    user_specific_metadata : dict or None, optional
        Per-frame user metadata recorded with each output. None (default) is
        an empty dict.
    append_l4 : bool, optional
        Whether to add a full-frame L4 luminance channel to the Bayer masks.
        Default True.
    output_dir : str or pathlib.Path or None, optional
        Directory to write the per-frame ``.star`` files (and QA manifest)
        into. Default ``"."``; None runs in in-memory mode (see
        `bandaid.scripts.process_batch`).
    output_suffix : str, optional
        Suffix for the per-frame output files. Default ``".star"``.
    write_frame : collections.abc.Callable, optional
        Per-frame writer used in write-to-disk mode (see
        `bandaid.scripts.process_batch` and `bandaid.writers`). Default
        `write_starlist_set` (the ``.star`` format).
    fail_fast : bool, optional
        Whether to re-raise unexpected per-frame errors instead of skipping
        the frame. Default False (the robust mode for unattended runs).
    write_qa_manifest : bool, optional
        Whether to write a per-frame QA manifest alongside the outputs.
        Default True.
    incoming_dir : str or pathlib.Path or None, optional
        Local staging directory for the downloads. A given directory is
        created if needed and left in place afterwards (only the per-frame
        files are cleaned up); None (default) uses a fresh temporary
        directory that is removed when the run ends.
    keep_local : bool, optional
        Whether to keep each frame's local copy instead of deleting it after
        the frame's outcome is decided. Default False -- deleting is the
        point of streaming. True also preserves an owned temporary
        ``incoming_dir``.
    download_workers : int, optional
        Number of concurrent download threads for the `Prefetcher`. Default
        2; the look-ahead is derived from it (``2 * workers``).

    Returns
    -------
    tuple of (list of str, dict)
        The remote frame names and the `process_batch` result mapping (each
        successfully-processed name to its output). A frame whose download
        fails mid-batch is skipped like any other per-frame failure and
        recorded in the QA manifest.

    Raises
    ------
    ValueError
        If the remote holds no FITS frames. A failure to list the remote
        (`subprocess.CalledProcessError`) or to fetch the *first* frame
        (`RemoteFetchError`) also propagates -- there is no batch without
        prep.
    """
    names = list_remote_fits(remote)
    if not names:
        msg = f"no FITS frames found on the remote {remote!r}"
        raise ValueError(msg)

    # The incoming directory is *owned* (and so removed at the end) only when
    # this run created it; a user-supplied directory is theirs to keep.
    owns_incoming = incoming_dir is None
    if owns_incoming:
        incoming = Path(tempfile.mkdtemp(prefix="bandaid-stream-"))
    else:
        incoming = Path(incoming_dir)
        incoming.mkdir(parents=True, exist_ok=True)

    config = config or PhotometryConfig()
    if cnn is None:
        _quiet_hf_xet()
        cnn = Ballet(model_file=weights)

    # The first frame seeds prepare_batch, so a failed download here is fatal
    # and deliberately propagates. The local copy is left in place: the
    # Prefetcher serves it from disk instead of downloading it twice.
    first_local = fetch_remote_file(remote, names[0], incoming)
    prep = prepare_batch(first_local, cnn=cnn, config=config, append_l4=append_l4)

    prefetcher = Prefetcher(remote, names, incoming, workers=download_workers)

    def _after_frame(name, _status):
        """Delete the frame's local copy once its outcome is decided."""
        if not keep_local:
            (incoming / name).unlink(missing_ok=True)

    try:
        results = process_batch(
            names,
            prep,
            user_specific_metadata=user_specific_metadata or {},
            output_dir=output_dir,
            output_suffix=output_suffix,
            write_frame=write_frame,
            fail_fast=fail_fast,
            write_qa_manifest=write_qa_manifest,
            fetch=prefetcher.fetch,
            after_frame=_after_frame,
        )
    finally:
        # Wind the downloads down and remove an owned staging directory even
        # when the batch aborts (fail-fast bug, systemic write failure), so a
        # crashed run does not leak temp dirs full of FITS frames.
        prefetcher.close()
        if owns_incoming and not keep_local:
            shutil.rmtree(incoming, ignore_errors=True)
    return names, results
