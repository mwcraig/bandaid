"""Unit tests for frame-path discovery and the photometer-frames entry point."""

import os
from pathlib import Path

import pytest
from _helpers import _CONSISTENT_HEADER, _dummy_prep

from bandaid import scripts
from bandaid.config import (
    PhotometryConfig,
)


class TestExpandFramePaths:
    """Unit tests for the ``expand_frame_paths`` file-name convenience."""

    def test_directory_expands_to_sorted_fits_only(self, tmp_path):
        """A directory yields its FITS frames, sorted, with non-FITS dropped."""
        night = tmp_path / "night"
        night.mkdir()
        for name in ["c.fit", "a.fit", "b.fit"]:
            (night / name).write_bytes(b"")
        (night / "notes.txt").write_text("not a frame")

        result = scripts.expand_frame_paths([str(night)])

        assert result == sorted(str(p.resolve()) for p in night.glob("*.fit"))

    def test_glob_matches_are_filtered_to_fits(self, tmp_path):
        """A wildcard that catches non-FITS files keeps only the FITS ones."""
        night = tmp_path / "night"
        night.mkdir()
        (night / "a.fit").write_bytes(b"")
        (night / "b.txt").write_text("not a frame")

        result = scripts.expand_frame_paths([str(night / "*")])

        assert result == [str((night / "a.fit").resolve())]

    def test_missing_literal_raises_file_not_found(self, tmp_path):
        """A non-existent literal path fails fast instead of reaching prepare_batch."""
        with pytest.raises(FileNotFoundError):
            scripts.expand_frame_paths([str(tmp_path / "nope.fits")])

    def test_non_fits_literal_raises_value_error(self, tmp_path):
        """An existing literal that is not a FITS frame is rejected with a message."""
        other = tmp_path / "foo.txt"
        other.write_text("not a frame")
        with pytest.raises(ValueError, match="FITS"):
            scripts.expand_frame_paths([str(other)])

    def test_same_name_in_different_dirs_kept_distinct(self, tmp_path):
        """Identically named frames in different directories are NOT collapsed."""
        n1 = tmp_path / "n1"
        n2 = tmp_path / "n2"
        n1.mkdir()
        n2.mkdir()
        (n1 / "img.fit").write_bytes(b"")
        (n2 / "img.fit").write_bytes(b"")

        result = scripts.expand_frame_paths([str(n1), str(n2)])

        expected = sorted(
            [str((n1 / "img.fit").resolve()), str((n2 / "img.fit").resolve())]
        )
        assert result == expected

    def test_same_file_referenced_two_ways_is_deduplicated(self, tmp_path):
        """The same file reached via a directory and an explicit path appears once."""
        night = tmp_path / "night"
        night.mkdir()
        frame = night / "a.fit"
        frame.write_bytes(b"")

        result = scripts.expand_frame_paths([str(night), str(frame)])

        assert result == [str(frame.resolve())]

    def test_compressed_fits_extension_accepted(self, tmp_path):
        """A compressed ``.fits.gz`` frame is recognised as a FITS frame."""
        frame = tmp_path / "a.fits.gz"
        frame.write_bytes(b"")

        assert scripts.expand_frame_paths([str(frame)]) == [str(frame.resolve())]

    def test_directory_with_fits_named_subdir_is_skipped(self, tmp_path):
        """A sub-directory whose name ends in a FITS suffix is not a frame."""
        night = tmp_path / "night"
        night.mkdir()
        (night / "a.fit").write_bytes(b"")
        (night / "bundle.fits").mkdir()  # a directory, not a frame

        result = scripts.expand_frame_paths([str(night)])

        assert result == [str((night / "a.fit").resolve())]

    def test_glob_matching_fits_named_dir_is_skipped(self, tmp_path):
        """A glob that catches a FITS-named directory keeps only real files."""
        night = tmp_path / "night"
        night.mkdir()
        (night / "a.fit").write_bytes(b"")
        (night / "bundle.fit").mkdir()

        result = scripts.expand_frame_paths([str(night / "*.fit")])

        assert result == [str((night / "a.fit").resolve())]

    def test_literal_non_regular_file_raises_value_error(self, tmp_path):
        """A literal FITS-named path that is not a regular file is rejected."""
        fifo = tmp_path / "pipe.fits"
        os.mkfifo(fifo)  # exists, ends in .fits, but is not a frame
        with pytest.raises(ValueError, match="FITS"):
            scripts.expand_frame_paths([str(fifo)])


class TestPhotometerFrames:
    """Unit tests for the high-level ``photometer_frames`` convenience entry point."""

    def test_expands_builds_cnn_and_wires_both_steps(self, monkeypatch, tmp_path):
        """It expands the args, builds the CNN from weights, and threads both steps."""
        night = tmp_path / "night"
        night.mkdir()
        for name in ["b.fit", "a.fit"]:
            (night / name).write_bytes(b"")
        weights = tmp_path / "w.npz"
        weights.write_bytes(b"npz")

        calls = {}
        cnn_sentinel = object()
        prep_sentinel = object()

        def fake_ballet(model_file=None):
            calls["ballet"] = model_file
            return cnn_sentinel

        def fake_prepare(first_file, *, cnn, config=None, append_l4=False):
            calls["prepare"] = {
                "first_file": first_file,
                "cnn": cnn,
                "config": config,
                "append_l4": append_l4,
            }
            return prep_sentinel

        def fake_process(files, prep, **kwargs: object):
            files = list(files)
            calls["process"] = {"files": files, "prep": prep, "kwargs": kwargs}
            return {f: f + ".star" for f in files}

        monkeypatch.setattr(scripts, "Ballet", fake_ballet)
        monkeypatch.setattr(scripts, "prepare_batch", fake_prepare)
        monkeypatch.setattr(scripts, "process_batch", fake_process)

        config = PhotometryConfig()
        frames, results = scripts.photometer_frames(
            [str(night)],
            config=config,
            weights=str(weights),
            user_specific_metadata={"observer": "MWC"},
            append_l4=True,
            output_dir=str(tmp_path / "out"),
            output_suffix=".sl",
            fail_fast=True,
            write_qa_manifest=False,
        )

        expected = sorted(str(p.resolve()) for p in night.glob("*.fit"))
        assert frames == expected
        assert calls["ballet"] == str(weights)
        assert calls["prepare"]["first_file"] == expected[0]
        assert calls["prepare"]["cnn"] is cnn_sentinel
        assert calls["prepare"]["config"] is config
        assert calls["prepare"]["append_l4"] is True
        assert calls["process"]["prep"] is prep_sentinel
        assert calls["process"]["files"] == expected
        kwargs = calls["process"]["kwargs"]
        assert kwargs["user_specific_metadata"] == {"observer": "MWC"}
        assert kwargs["output_dir"] == str(tmp_path / "out")
        assert kwargs["output_suffix"] == ".sl"
        assert kwargs["fail_fast"] is True
        assert kwargs["write_qa_manifest"] is False
        assert results == {f: f + ".star" for f in expected}

    def test_no_frames_raises_value_error(self, tmp_path):
        """An argument set that expands to nothing is a clean ValueError."""
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="no FITS"):
            scripts.photometer_frames([str(empty)])

    def test_defaults_download_weights_and_append_l4(self, monkeypatch, tmp_path):
        """Omitting options downloads weights, appends L4, and uses robust defaults."""
        frame = tmp_path / "a.fit"
        frame.write_bytes(b"")

        calls = {}
        monkeypatch.setattr(
            scripts, "Ballet", lambda model_file=None: calls.update(ballet=model_file)
        )

        def fake_prepare(first_file, *, cnn, config=None, append_l4=False):
            calls["prepare"] = (first_file, cnn)
            calls["append_l4"] = append_l4
            calls["config"] = config
            return object()

        monkeypatch.setattr(scripts, "prepare_batch", fake_prepare)
        monkeypatch.setattr(
            scripts,
            "process_batch",
            lambda files, prep, **kwargs: calls.update(kwargs=kwargs) or {},
        )

        scripts.photometer_frames([str(frame)])

        assert calls["ballet"] is None
        assert calls["append_l4"] is True
        assert isinstance(calls["config"], PhotometryConfig)
        kwargs = calls["kwargs"]
        assert kwargs["user_specific_metadata"] == {}
        assert kwargs["fail_fast"] is False
        assert kwargs["write_qa_manifest"] is True
        assert kwargs["output_dir"] == "."
        assert kwargs["output_suffix"] == ".star"

    def test_identical_names_write_distinct_starlists(
        self, monkeypatch, tmp_path, by_filter
    ):
        """End-to-end: two same-named frames in different dirs give distinct outputs."""
        n1 = tmp_path / "n1"
        n2 = tmp_path / "n2"
        n1.mkdir()
        n2.mkdir()
        (n1 / "img.fit").write_bytes(b"")
        (n2 / "img.fit").write_bytes(b"")
        out = tmp_path / "out"
        inputs = [str(n1), str(n2)]

        monkeypatch.setattr(scripts, "Ballet", lambda model_file=None: object())
        monkeypatch.setattr(
            scripts,
            "prepare_batch",
            lambda first_file, *, cnn, config=None, append_l4=False: _dummy_prep(),
        )
        monkeypatch.setattr(scripts, "process_one_image", lambda *a, **k: by_filter())
        monkeypatch.setattr(
            scripts.fits, "getheader", lambda _file: dict(_CONSISTENT_HEADER)
        )

        frames, results = scripts.photometer_frames(inputs, output_dir=str(out))

        # Two source directories => mirrored tree, so each frame lands under its
        # own <dirname>/ subdir instead of overwriting a shared flat name.
        written = sorted(p.relative_to(out) for p in out.rglob("*.star"))
        assert len(frames) == len(inputs)
        assert written == [Path("n1/img.star"), Path("n2/img.star")]
        assert len(results) == len(frames)
        assert len({str(v) for v in results.values()}) == len(results)
