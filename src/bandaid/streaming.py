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

import atexit
import logging
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .exceptions import RemoteFetchError
from .scripts import _is_fits, _resolve_batch_inputs, prepare_batch, process_batch
from .writers import write_starlist_set

logger = logging.getLogger(__name__)

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

    Attributes
    ----------
    download_seconds : dict
        Wall-clock duration of each name's download, recorded as it finishes
        (failed downloads included -- the time was still spent). Names served
        from an already-present local file have no entry: there was no
        download to time.
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
        self.download_seconds = {}
        self._top_up()

    def _timed_fetch_one(self, name):
        """
        Download one name, recording its wall-clock duration.

        Recorded in a finally so a failed download's time is kept too -- it
        was spent either way, and the timing report should show it.

        Parameters
        ----------
        name : str
            The frame name to download.

        Returns
        -------
        pathlib.Path
            The downloaded frame's local path, from ``fetch_one``.
        """
        start = time.monotonic()
        try:
            return self._fetch_one(self._remote, name, self._incoming)
        finally:
            self.download_seconds[name] = time.monotonic() - start

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
            self._futures[name] = self._executor.submit(self._timed_fetch_one, name)

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
            return self._timed_fetch_one(name)
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


class _BatchTimer:
    """
    Per-frame download/stall/processing timing for a streamed batch.

    Wraps the `Prefetcher`'s ``fetch`` to measure the *stall* (how long
    processing sat blocked waiting for a download -- the number that says
    whether ``--download-workers`` needs raising) and clocks each frame from
    fetch-return to outcome as its *bandaid* time. The download time itself
    comes from the Prefetcher, which times the actual transfer in the worker;
    with prefetch overlap it is usually much larger than the stall.

    Parameters
    ----------
    prefetcher : Prefetcher
        The batch's prefetcher, wrapped by `timed_fetch` and consulted for
        the per-name download durations.
    """

    def __init__(self, prefetcher) -> None:
        self._prefetcher = prefetcher
        self._fetch_done = {}
        self._stall = {}
        self._frames = 0
        self._download_total = 0.0
        self._stall_total = 0.0
        self._bandaid_total = 0.0

    def timed_fetch(self, name):
        """
        Fetch ``name`` via the prefetcher, recording the stall it cost.

        Recorded in a finally so a failed fetch's stall is kept too --
        processing sat blocked either way, and the frame still reaches
        `record` via the batch's skip path.

        Parameters
        ----------
        name : str
            The frame name to materialize.

        Returns
        -------
        pathlib.Path
            The frame's local path, from ``Prefetcher.fetch``.
        """
        start = time.monotonic()
        try:
            return self._prefetcher.fetch(name)
        finally:
            done = time.monotonic()
            self._stall[name] = done - start
            self._fetch_done[name] = done

    def record(self, name):
        """
        Close out one frame's timing and log its line at INFO.

        Called from the batch's ``after_frame`` hook, so the frame's bandaid
        time spans everything from fetch-return to decided outcome (header
        check, photometry, and the write).

        Parameters
        ----------
        name : str
            The frame whose outcome was just decided.
        """
        now = time.monotonic()
        bandaid = now - self._fetch_done.pop(name, now)
        stall = self._stall.pop(name, 0.0)
        download = self._prefetcher.download_seconds.get(name)
        self._frames += 1
        self._stall_total += stall
        self._bandaid_total += bandaid
        if download is None:
            logger.info(
                "timing %s: download cached, stall %.1f s, bandaid %.1f s",
                name,
                stall,
                bandaid,
            )
        else:
            self._download_total += download
            logger.info(
                "timing %s: download %.1f s, stall %.1f s, bandaid %.1f s",
                name,
                download,
                stall,
                bandaid,
            )

    def log_summary(self):
        """Log the batch's timing totals and per-frame averages at INFO."""
        if not self._frames:
            return
        frames = self._frames
        logger.info(
            "timing summary over %d frames: download avg %.1f s (total %.1f s), "
            "stall avg %.1f s (total %.1f s), bandaid avg %.1f s (total %.1f s)",
            frames,
            self._download_total / frames,
            self._download_total,
            self._stall_total / frames,
            self._stall_total,
            self._bandaid_total / frames,
            self._bandaid_total,
        )


def _resolve_incoming_dir(incoming_dir, keep_local):
    """
    Resolve the staging directory and whether this run owns it.

    An *owned* directory (one this run created because the caller gave none)
    is removed when the run ends; a user-supplied directory is theirs to keep.
    An owned directory that ``keep_local`` will preserve is announced at INFO:
    the caller never named it, so without the log line the kept frames would
    sit in an unfindable temp dir.

    Parameters
    ----------
    incoming_dir : str or pathlib.Path or None
        The caller's staging directory, or None for a fresh temporary one.
    keep_local : bool
        Whether the run will keep the downloaded frames (and therefore an
        owned temporary directory) instead of deleting them.

    Returns
    -------
    tuple of (pathlib.Path, bool)
        The directory (created if needed) and whether this run owns it.
    """
    if incoming_dir is None:
        incoming = Path(tempfile.mkdtemp(prefix="bandaid-stream-"))
        if keep_local:
            logger.info("keeping downloaded frames in %s", incoming)
        return incoming, True
    incoming = Path(incoming_dir)
    incoming.mkdir(parents=True, exist_ok=True)
    return incoming, False


def _fetch_seed_frame(remote, name, incoming):
    """
    Materialize the batch's seed frame, reusing a cached copy when present.

    A copy surviving from a prior ``--keep`` run is served as-is instead of
    re-downloaded; a failed download propagates (as `RemoteFetchError`) --
    there is no batch without the seed.

    Parameters
    ----------
    remote : str
        The rclone remote path the frame lives on.
    name : str
        The seed frame's file name on the remote.
    incoming : pathlib.Path
        The local staging directory.

    Returns
    -------
    tuple of (pathlib.Path, float or None)
        The frame's local path and the download's wall-clock seconds, or
        None when a cached copy was served from disk (no download to time).
    """
    local = incoming / name
    if local.exists():
        return local, None
    start = time.monotonic()
    local = fetch_remote_file(remote, name, incoming)
    return local, time.monotonic() - start


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
        ``incoming_dir``, whose path is logged at INFO so the kept frames
        can be found.
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

    # Resolve the config and centroider before creating the staging directory:
    # a failed weights download must not leave an owned temp dir behind.
    config, cnn = _resolve_batch_inputs(config, cnn, weights)

    incoming, owns_incoming = _resolve_incoming_dir(incoming_dir, keep_local)
    cleanup_incoming = owns_incoming and not keep_local

    def _remove_incoming():
        """Remove the owned staging directory, tolerating its absence."""
        shutil.rmtree(incoming, ignore_errors=True)

    if cleanup_incoming:
        # Safety net for an interrupted shutdown: a download thread still in
        # flight when the in-line rmtree below runs can finish afterwards and
        # recreate the directory (rclone creates destination parents). The
        # interpreter joins those threads at exit *before* atexit handlers
        # run, so this handler is the one cleanup guaranteed to see the
        # directory's final state. Harmless when the in-line rmtree already
        # succeeded.
        atexit.register(_remove_incoming)

    prefetcher = None
    try:
        # The first frame seeds prepare_batch, so a failed download here is
        # fatal and deliberately propagates. The local copy is left in place:
        # the Prefetcher serves it from disk instead of downloading it twice.
        first_local, first_seconds = _fetch_seed_frame(remote, names[0], incoming)
        prep = prepare_batch(first_local, cnn=cnn, config=config, append_l4=append_l4)
        prefetcher = Prefetcher(remote, names, incoming, workers=download_workers)
        if first_seconds is not None:
            # The seed download happened outside the Prefetcher; hand it the
            # duration so the first frame's timing line shows the real
            # transfer instead of "cached".
            prefetcher.download_seconds[names[0]] = first_seconds
        timer = _BatchTimer(prefetcher)

        def _after_frame(name, _status):
            """Log the frame's timing, then delete its local copy."""
            timer.record(name)
            if not keep_local:
                (incoming / name).unlink(missing_ok=True)

        results = process_batch(
            names,
            prep,
            user_specific_metadata=user_specific_metadata or {},
            output_dir=output_dir,
            output_suffix=output_suffix,
            write_frame=write_frame,
            fail_fast=fail_fast,
            write_qa_manifest=write_qa_manifest,
            fetch=timer.timed_fetch,
            after_frame=_after_frame,
        )
        timer.log_summary()
    finally:
        # Wind the downloads down and remove an owned staging directory
        # whenever the run does not complete -- a fatal first-frame fetch, a
        # prep failure, a fail-fast bug, a systemic write failure -- so a
        # crashed run does not leak temp dirs full of FITS frames. The removal
        # is nested in its own finally because close() itself can be
        # interrupted: a terminal Ctrl-C signals the whole process group (and
        # `uv run` forwards it besides), so a second KeyboardInterrupt often
        # lands while close() is still waiting out the in-flight downloads.
        try:
            if prefetcher is not None:
                prefetcher.close()
            # close() returned normally, so no worker thread survives to
            # recreate the directory after the rmtree below: the exit-time
            # safety net has nothing left to do, and dropping it keeps
            # handlers from piling up in a long-lived process that streams
            # many batches. When close() raises instead (a second Ctrl-C),
            # this is skipped and the net stays registered.
            if cleanup_incoming:
                atexit.unregister(_remove_incoming)
        finally:
            if cleanup_incoming:
                _remove_incoming()
    return names, results
