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
