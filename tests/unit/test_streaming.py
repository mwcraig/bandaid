"""Unit tests for the rclone streaming transport in ``bandaid.streaming``."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from bandaid import streaming
from bandaid.exceptions import RemoteFetchError

# A remote path with spaces, matching the real-world "gdrive:LS Psc from Rick"
# use case: it must always travel to rclone as a single argv element.
SPACEY_REMOTE = "gdrive:LS Psc from Rick"


def _completed(stdout=""):
    """Build a minimal ``CompletedProcess`` stand-in with the given stdout."""
    return SimpleNamespace(stdout=stdout, stderr="", returncode=0)


class TestListRemoteFits:
    """Unit tests for ``list_remote_fits``."""

    def test_lsf_argv_passes_spacey_remote_as_single_element(self, monkeypatch):
        """The remote (spaces and all) is one argv element of an exact lsf call."""
        calls = []

        def fake_run(args):
            calls.append(args)
            return _completed()

        monkeypatch.setattr(streaming, "_run_rclone", fake_run)

        streaming.list_remote_fits(SPACEY_REMOTE)

        assert calls == [["lsf", "--files-only", SPACEY_REMOTE]]

    def test_keeps_only_fits_names_sorted(self, monkeypatch):
        """Non-FITS names are dropped and the survivors come back sorted."""
        listing = "stack.fits\nnotes.txt\nb.fit\nthumb.jpg\na.fits.gz\n"
        monkeypatch.setattr(streaming, "_run_rclone", lambda _args: _completed(listing))

        names = streaming.list_remote_fits("gdrive:field")

        assert names == ["a.fits.gz", "b.fit", "stack.fits"]

    def test_empty_listing_returns_empty_list(self, monkeypatch):
        """An empty remote listing yields an empty name list, not an error."""
        monkeypatch.setattr(streaming, "_run_rclone", lambda _args: _completed(""))

        assert streaming.list_remote_fits("gdrive:field") == []


class TestFetchRemoteFile:
    """Unit tests for ``fetch_remote_file``."""

    def test_copyto_argv_joins_remote_and_name(self, monkeypatch, tmp_path):
        """The copyto argv gets ``remote/name`` as one element plus the dest path."""
        calls = []

        def fake_run(args):
            calls.append(args)
            return _completed()

        monkeypatch.setattr(streaming, "_run_rclone", fake_run)

        local = streaming.fetch_remote_file(SPACEY_REMOTE, "frame1.fits", tmp_path)

        assert calls == [
            ["copyto", f"{SPACEY_REMOTE}/frame1.fits", str(tmp_path / "frame1.fits")]
        ]
        assert local == tmp_path / "frame1.fits"

    def test_trailing_slash_on_remote_does_not_double_up(self, monkeypatch, tmp_path):
        """A trailing slash on the remote is stripped before the name is joined."""
        calls = []

        def fake_run(args):
            calls.append(args)
            return _completed()

        monkeypatch.setattr(streaming, "_run_rclone", fake_run)

        streaming.fetch_remote_file("gdrive:field/", "a.fits", tmp_path)

        assert calls[0][1] == "gdrive:field/a.fits"

    def test_success_returns_path_and_keeps_file(self, monkeypatch, tmp_path):
        """On success the downloaded file is left at the returned path."""

        def fake_run(args):
            Path(args[2]).write_text("frame data")
            return _completed()

        monkeypatch.setattr(streaming, "_run_rclone", fake_run)

        local = streaming.fetch_remote_file("gdrive:field", "a.fits", tmp_path)

        assert local == tmp_path / "a.fits"
        assert local.read_text() == "frame data"

    def test_failure_raises_remote_fetch_error_with_stderr(self, monkeypatch, tmp_path):
        """A failed copyto surfaces rclone's stderr and names the frame."""

        def fake_run(args):
            raise subprocess.CalledProcessError(
                1, ["rclone", *args], stderr="didn't find section in config file"
            )

        monkeypatch.setattr(streaming, "_run_rclone", fake_run)

        with pytest.raises(RemoteFetchError, match="didn't find section") as excinfo:
            streaming.fetch_remote_file("gdrive:field", "bad.fits", tmp_path)

        assert excinfo.value.file == "bad.fits"
        # Chained with `from exc` so the CalledProcessError stays debuggable.
        assert isinstance(excinfo.value.__cause__, subprocess.CalledProcessError)

    def test_failure_removes_partial_download(self, monkeypatch, tmp_path):
        """A partial file left behind by a failed copyto is deleted."""

        def fake_run(args):
            Path(args[2]).write_text("partial bytes")
            raise subprocess.CalledProcessError(1, ["rclone", *args], stderr="boom")

        monkeypatch.setattr(streaming, "_run_rclone", fake_run)

        with pytest.raises(RemoteFetchError, match="boom"):
            streaming.fetch_remote_file("gdrive:field", "bad.fits", tmp_path)

        assert not (tmp_path / "bad.fits").exists()


def _recording_fetch_one(recorded, fail=()):
    """
    Build a blocking-free ``fetch_one`` fake for `Prefetcher` tests.

    The fake records each downloaded name in ``recorded``, raises a
    `RemoteFetchError` for names in ``fail``, and otherwise writes a small
    file at the destination and returns its path -- the same contract as
    ``fetch_remote_file``.
    """

    def fetch_one(_remote, name, dest_dir):
        recorded.append(name)
        if name in fail:
            msg = "rclone copyto failed"
            raise RemoteFetchError(msg, file=name)
        local = Path(dest_dir) / name
        local.write_text("frame data")
        return local

    return fetch_one


class TestPrefetcher:
    """Unit tests for the bounded look-ahead ``Prefetcher``."""

    def test_downloads_run_in_listing_order(self, tmp_path):
        """Names download in listing order (a single worker serializes them)."""
        recorded = []
        names = [f"frame{i}.fits" for i in range(5)]
        prefetcher = streaming.Prefetcher(
            "gdrive:field",
            names,
            tmp_path,
            workers=1,
            fetch_one=_recording_fetch_one(recorded),
        )
        try:
            for name in names:
                assert prefetcher.fetch(name) == tmp_path / name
        finally:
            prefetcher.close()

        assert recorded == names

    def test_lookahead_bounds_outstanding_submissions(self, tmp_path):
        """Only the first ``2 * workers`` names are in flight after construction."""
        recorded = []
        names = [f"frame{i}.fits" for i in range(10)]
        prefetcher = streaming.Prefetcher(
            "gdrive:field",
            names,
            tmp_path,
            workers=2,
            fetch_one=_recording_fetch_one(recorded),
        )
        try:
            # The private futures map is the only race-free view of what has
            # been submitted; anything execution-based races the pool.
            assert set(prefetcher._futures) == set(names[:4])  # noqa: SLF001
        finally:
            prefetcher.close()

    def test_fetch_tops_the_queue_back_up(self, tmp_path):
        """Serving one name promotes the next listed name into the window."""
        recorded = []
        names = ["a.fits", "b.fits", "c.fits", "d.fits"]
        prefetcher = streaming.Prefetcher(
            "gdrive:field",
            names,
            tmp_path,
            workers=1,
            lookahead=2,
            fetch_one=_recording_fetch_one(recorded),
        )
        try:
            assert set(prefetcher._futures) == {"a.fits", "b.fits"}  # noqa: SLF001

            local = prefetcher.fetch("a.fits")

            assert local == tmp_path / "a.fits"
            assert "c.fits" in prefetcher._futures  # noqa: SLF001
            assert "d.fits" not in prefetcher._futures  # noqa: SLF001
        finally:
            prefetcher.close()

    def test_existing_local_file_served_without_download(self, tmp_path):
        """A file already in incoming (the prep frame) is never re-fetched."""
        recorded = []
        names = ["a.fits", "b.fits"]
        (tmp_path / "a.fits").write_text("already here")
        prefetcher = streaming.Prefetcher(
            "gdrive:field",
            names,
            tmp_path,
            workers=1,
            fetch_one=_recording_fetch_one(recorded),
        )
        try:
            local = prefetcher.fetch("a.fits")
        finally:
            prefetcher.close()

        assert local == tmp_path / "a.fits"
        assert local.read_text() == "already here"
        # close() waited for the pool, so recorded is complete: only b.fits.
        assert "a.fits" not in recorded

    def test_failed_download_poisons_only_its_own_fetch(self, tmp_path):
        """One RemoteFetchError raises from that fetch; later names still arrive."""
        recorded = []
        names = ["bad.fits", "good.fits"]
        prefetcher = streaming.Prefetcher(
            "gdrive:field",
            names,
            tmp_path,
            workers=1,
            fetch_one=_recording_fetch_one(recorded, fail={"bad.fits"}),
        )
        try:
            with pytest.raises(RemoteFetchError, match="rclone copyto failed"):
                prefetcher.fetch("bad.fits")

            assert prefetcher.fetch("good.fits") == tmp_path / "good.fits"
        finally:
            prefetcher.close()

    def test_unqueued_name_is_fetched_synchronously(self, tmp_path):
        """A name outside the look-ahead window is downloaded on demand."""
        recorded = []
        names = [f"frame{i}.fits" for i in range(8)]
        prefetcher = streaming.Prefetcher(
            "gdrive:field",
            names,
            tmp_path,
            workers=1,
            lookahead=2,
            fetch_one=_recording_fetch_one(recorded),
        )
        try:
            local = prefetcher.fetch(names[-1])
        finally:
            prefetcher.close()

        assert local == tmp_path / names[-1]
        assert names[-1] in recorded

    def test_close_is_prompt_and_idempotent(self, tmp_path):
        """close() returns without hanging and tolerates a second call."""
        recorded = []
        names = [f"frame{i}.fits" for i in range(6)]
        prefetcher = streaming.Prefetcher(
            "gdrive:field",
            names,
            tmp_path,
            workers=2,
            fetch_one=_recording_fetch_one(recorded),
        )

        prefetcher.close()
        prefetcher.close()


def _interrupted_close(_self):
    """Stand-in for ``Prefetcher.close`` hit by a second Ctrl-C mid-shutdown."""
    raise KeyboardInterrupt


def _install_stream_fakes(monkeypatch, names):
    """
    Patch ``stream_frames``'s collaborators with recording fakes.

    The fake ``fetch_remote_file`` writes a small file at the destination (so
    local copies really exist on disk), ``prepare_batch`` records its call and
    returns a sentinel, and ``process_batch`` drives the passed ``fetch``/
    ``after_frame`` hooks once per name -- mimicking the real per-frame loop
    so the delete-after-process contract is exercised. Returns a namespace
    recording everything the fakes saw.
    """
    rec = SimpleNamespace(
        fetched=[],
        prep_args=[],
        batch=None,
        exists_before_after_frame=[],
        exists_after_after_frame=[],
    )
    monkeypatch.setattr(streaming, "list_remote_fits", lambda _remote: list(names))
    monkeypatch.setattr(streaming, "Ballet", lambda **_kwargs: object())

    def fake_fetch(_remote, name, dest_dir):
        rec.fetched.append(name)
        local = Path(dest_dir) / name
        local.write_text("frame data")
        return local

    monkeypatch.setattr(streaming, "fetch_remote_file", fake_fetch)

    def fake_prepare_batch(first_local, **kwargs: object):
        rec.prep_args.append((Path(first_local), kwargs))
        return "prep-sentinel"

    monkeypatch.setattr(streaming, "prepare_batch", fake_prepare_batch)

    def fake_process_batch(files, prep, *, fetch, after_frame, **kwargs: object):
        rec.batch = {
            "files": list(files),
            "prep": prep,
            "fetch": fetch,
            "after_frame": after_frame,
            **kwargs,
        }
        results = {}
        for name in files:
            local = fetch(name)
            rec.exists_before_after_frame.append(local.exists())
            after_frame(name, "ok")
            rec.exists_after_after_frame.append(local.exists())
            results[name] = local
        return results

    monkeypatch.setattr(streaming, "process_batch", fake_process_batch)
    return rec


class TestStreamFrames:
    """Unit tests for the ``stream_frames`` orchestration."""

    def test_local_copies_deleted_after_each_frame_by_default(
        self, monkeypatch, tmp_path
    ):
        """Each frame's incoming copy exists while processed and is gone after."""
        names = ["a.fits", "b.fits"]
        rec = _install_stream_fakes(monkeypatch, names)
        incoming = tmp_path / "incoming"

        streaming.stream_frames("gdrive:field", incoming_dir=incoming)

        assert rec.exists_before_after_frame == [True, True]
        assert rec.exists_after_after_frame == [False, False]

    def test_keep_local_retains_the_downloads(self, monkeypatch, tmp_path):
        """With keep_local=True every frame's local copy survives the run."""
        names = ["a.fits", "b.fits"]
        rec = _install_stream_fakes(monkeypatch, names)
        incoming = tmp_path / "incoming"

        streaming.stream_frames("gdrive:field", incoming_dir=incoming, keep_local=True)

        assert rec.exists_after_after_frame == [True, True]
        assert sorted(p.name for p in incoming.iterdir()) == names

    def test_user_incoming_dir_is_created_and_preserved(self, monkeypatch, tmp_path):
        """A user-supplied incoming dir is made if needed and never removed."""
        _install_stream_fakes(monkeypatch, ["a.fits"])
        incoming = tmp_path / "deep" / "incoming"

        streaming.stream_frames("gdrive:field", incoming_dir=incoming)

        assert incoming.is_dir()

    def test_owned_temp_incoming_dir_removed_on_return(self, monkeypatch):
        """With no incoming_dir a bandaid-stream temp dir is made, then removed."""
        rec = _install_stream_fakes(monkeypatch, ["a.fits"])

        streaming.stream_frames("gdrive:field")

        temp_dir = rec.prep_args[0][0].parent
        assert temp_dir.name.startswith("bandaid-stream-")
        assert not temp_dir.exists()

    def test_owned_temp_dir_removed_when_process_batch_raises(self, monkeypatch):
        """A mid-batch crash still cleans up the owned temp directory."""
        rec = _install_stream_fakes(monkeypatch, ["a.fits"])

        def boom(*_args: object, **_kwargs: object):
            msg = "mid-batch crash"
            raise RuntimeError(msg)

        monkeypatch.setattr(streaming, "process_batch", boom)

        with pytest.raises(RuntimeError, match="mid-batch crash"):
            streaming.stream_frames("gdrive:field")

        temp_dir = rec.prep_args[0][0].parent
        assert not temp_dir.exists()

    def test_owned_temp_dir_removed_when_close_is_interrupted(self, monkeypatch):
        """
        A Ctrl-C landing inside prefetcher.close() still cleans the temp dir.

        In a terminal the whole process group gets SIGINT (and ``uv run``
        forwards it besides), so a second KeyboardInterrupt routinely arrives
        while the finally block is still winding the downloads down; the
        staging-dir removal must not depend on close() finishing.
        """
        rec = _install_stream_fakes(monkeypatch, ["a.fits"])
        monkeypatch.setattr(streaming.Prefetcher, "close", _interrupted_close)

        with pytest.raises(KeyboardInterrupt):
            streaming.stream_frames("gdrive:field")

        temp_dir = rec.prep_args[0][0].parent
        assert not temp_dir.exists()

    def test_atexit_net_cleans_a_dir_recreated_by_a_late_worker(self, monkeypatch):
        """
        The atexit handler removes an owned dir a worker recreated post-rmtree.

        When close() is interrupted, still-running downloads can finish *after*
        the in-line rmtree and recreate the staging dir (rclone creates
        destination parents). The interpreter joins those threads at exit,
        before atexit handlers run, so a registered handler is the one cleanup
        that is guaranteed to see the dir's final state.
        """
        rec = _install_stream_fakes(monkeypatch, ["a.fits"])
        registered = []
        monkeypatch.setattr(
            streaming.atexit, "register", lambda func: registered.append(func) or func
        )
        monkeypatch.setattr(streaming.Prefetcher, "close", _interrupted_close)

        with pytest.raises(KeyboardInterrupt):
            streaming.stream_frames("gdrive:field")

        temp_dir = rec.prep_args[0][0].parent
        assert not temp_dir.exists()
        # A late worker recreates the dir after the in-line rmtree ran ...
        temp_dir.mkdir()
        (temp_dir / "late.fits").write_text("frame data")
        # ... and the registered exit handler still removes it.
        assert len(registered) == 1
        registered[0]()
        assert not temp_dir.exists()

    def test_empty_listing_raises_before_any_prep(self, monkeypatch):
        """An empty remote is a ValueError and never fetches or preps."""
        rec = _install_stream_fakes(monkeypatch, [])

        with pytest.raises(ValueError, match="no FITS frames found on the remote"):
            streaming.stream_frames("gdrive:field")

        assert rec.fetched == []
        assert rec.prep_args == []

    def test_first_frame_fetch_failure_is_fatal(self, monkeypatch, tmp_path):
        """A RemoteFetchError on the prep frame propagates; no prep is built."""
        rec = _install_stream_fakes(monkeypatch, ["a.fits"])

        def failing_fetch(_remote, name, _dest_dir):
            msg = "rclone copyto failed"
            raise RemoteFetchError(msg, file=name)

        monkeypatch.setattr(streaming, "fetch_remote_file", failing_fetch)

        with pytest.raises(RemoteFetchError, match="rclone copyto failed"):
            streaming.stream_frames("gdrive:field", incoming_dir=tmp_path)

        assert rec.prep_args == []

    def test_owned_temp_dir_removed_on_fatal_first_frame_failure(
        self, monkeypatch, tmp_path
    ):
        """A fatal failure before the batch starts still removes the owned dir."""
        _install_stream_fakes(monkeypatch, ["a.fits"])
        made = []

        def fake_mkdtemp(prefix):
            made.append(tmp_path / f"{prefix}owned")
            made[-1].mkdir()
            return str(made[-1])

        monkeypatch.setattr(streaming.tempfile, "mkdtemp", fake_mkdtemp)

        def failing_fetch(_remote, name, _dest_dir):
            msg = "rclone copyto failed"
            raise RemoteFetchError(msg, file=name)

        monkeypatch.setattr(streaming, "fetch_remote_file", failing_fetch)

        with pytest.raises(RemoteFetchError, match="rclone copyto failed"):
            streaming.stream_frames("gdrive:field")

        assert made
        assert not made[0].exists()

    def test_returns_names_and_process_batch_results(self, monkeypatch, tmp_path):
        """The remote listing and the batch results pass straight through."""
        names = ["a.fits", "b.fits"]
        _install_stream_fakes(monkeypatch, names)

        got_names, results = streaming.stream_frames(
            "gdrive:field", incoming_dir=tmp_path
        )

        assert got_names == names
        assert results == {name: tmp_path / name for name in names}

    def test_batch_kwargs_and_hooks_are_forwarded(self, monkeypatch, tmp_path):
        """process_batch receives the streaming hooks and normalized kwargs."""
        rec = _install_stream_fakes(monkeypatch, ["a.fits"])

        streaming.stream_frames(
            "gdrive:field",
            incoming_dir=tmp_path,
            output_dir="out",
            output_suffix=".starlist",
            fail_fast=True,
            write_qa_manifest=False,
        )

        assert rec.batch["files"] == ["a.fits"]
        assert rec.batch["prep"] == "prep-sentinel"
        # user_specific_metadata=None is normalized to an empty dict...
        assert rec.batch["user_specific_metadata"] == {}
        # ...and the streaming hooks really reach process_batch.
        assert callable(rec.batch["fetch"])
        assert callable(rec.batch["after_frame"])
        assert rec.batch["output_dir"] == "out"
        assert rec.batch["output_suffix"] == ".starlist"
        assert rec.batch["fail_fast"] is True
        assert rec.batch["write_qa_manifest"] is False
