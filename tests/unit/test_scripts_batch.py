"""Unit tests for the per-frame batch loop and batch-to-disk orchestration."""

import csv
import dataclasses
import json
import logging
import weakref
from pathlib import Path

import numpy as np
import pytest
from _helpers import _CONSISTENT_HEADER, _dummy_prep
from aavso_starlist_schema import StarListSet
from astropy.table import Table

import bandaid
from bandaid import scripts
from bandaid.exceptions import (
    CentroidPlaneError,
    FrameError,
    FrameMetadataError,
    NoUsableStarsError,
    StarListValidationError,
    TooFewStarsError,
    WCSScaleError,
    WCSSolveError,
)
from bandaid.scripts import (
    _CENTROID_MODEL_COLUMNS,
    FrameOutcome,
    FrameResult,
    _centroid_model_record,
    process_frame,
    write_qa_manifest,
)
from bandaid.writers import write_starlist_set


def _read_manifest(tmp_path):
    """Return the QA manifest rows written under ``tmp_path`` as a list of dicts."""
    with (tmp_path / scripts.QA_MANIFEST_FILENAME).open(newline="") as f:
        return list(csv.DictReader(f))


def _starlist_names(tmp_path):
    """Sorted starlist filenames in ``tmp_path``, ignoring the QA manifest sibling."""
    return sorted(
        p.name for p in tmp_path.iterdir() if p.name != scripts.QA_MANIFEST_FILENAME
    )


def _raise_on(trigger, exc, otherwise):
    """
    Build a ``process_one_image`` stub that raises ``exc`` for one frame.

    The returned stub raises ``exc`` when called for the ``trigger`` filename and
    otherwise returns ``otherwise()`` (evaluated per call, for a fresh result).
    """

    def _stub(file, *_args: object, **_kwargs: object):
        if file == trigger:
            raise exc
        return otherwise()

    return _stub


def _stub_load_frame_calls(mocker):
    """
    Stub ``scripts._load_frame`` with a counting mock; return the mock.

    The stub's frame carries a header matching ``_dummy_prep`` so every frame
    passes ``check_frame_consistency``; the mock's ``.call_args_list`` records
    the file arguments in call order, for the open-each-frame-once assertions
    (#44).

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        The pytest-mock fixture used to install the stub.

    Returns
    -------
    unittest.mock.MagicMock
        The mock installed in place of ``scripts._load_frame``.
    """
    return mocker.patch(
        "bandaid.scripts._load_frame",
        return_value=scripts.LoadedFrame(np.zeros((2, 2)), dict(_CONSISTENT_HEADER)),
    )


@pytest.mark.usefixtures("_consistent_headers")
class TestProcessBatch:
    """Unit tests for ``process_batch``."""

    def test_one_result_per_frame_with_shared_prep(self, mocker, by_filter):
        """Each frame is processed once with the same shared prep objects."""
        prep = _dummy_prep()
        user_meta = {"observer": "abc"}
        process_one_image = mocker.patch(
            "bandaid.scripts.process_one_image",
            return_value=by_filter(),
        )

        files = ["a.fits", "b.fits"]
        results = scripts.process_batch(files, prep, user_specific_metadata=user_meta)

        assert list(results) == files
        assert process_one_image.call_count == len(files)
        for file, call in zip(files, process_one_image.call_args_list, strict=True):
            cfile, meta, radecs, cnn, masks = call.args
            assert cfile == file
            assert meta is user_meta
            assert radecs is prep.radecs
            assert cnn is prep.cnn
            assert masks is prep.bayer_masks
            assert call.kwargs["input_photometry_coords"] is prep.photometry_coords

    def test_passes_the_preps_gaia_g_and_cut(self, mocker, by_filter):
        """``process_one_image`` receives the prep's row-aligned G and magnitude cut."""
        prep = dataclasses.replace(
            _dummy_prep(), gaia_g=np.array([9.0, 10.0]), g_cut=9.5
        )
        process_one_image = mocker.patch(
            "bandaid.scripts.process_one_image",
            return_value=by_filter(),
        )

        scripts.process_batch(["a.fits"], prep, user_specific_metadata={})

        kwargs = process_one_image.call_args.kwargs
        assert kwargs["input_gaia_g"] is prep.gaia_g
        assert kwargs["g_cut"] == prep.g_cut

    @pytest.mark.parametrize("build_l4", [True, False])
    def test_passes_the_preps_build_l4_flag(self, mocker, build_l4, by_filter):
        """``process_one_image`` receives the flag the prep carries."""
        prep = dataclasses.replace(_dummy_prep(), build_l4=build_l4)
        process_one_image = mocker.patch(
            "bandaid.scripts.process_one_image",
            return_value=by_filter(),
        )

        scripts.process_batch(["a.fits"], prep, user_specific_metadata={})

        assert process_one_image.call_args.kwargs["build_l4"] is build_l4

    def test_emits_progress_log_per_frame(
        self, patched_process_one_image, caplog, by_filter
    ):
        """Each frame logs a ``processing i/N: name`` line at DEBUG."""
        patched_process_one_image(by_filter())

        # Identically-named frames from different directories (a supported
        # mirrored-tree batch): the line logs the full path, not just the
        # basename, so the two "a.fits" frames stay distinguishable.
        files = ["night1/a.fits", "night2/a.fits", "night2/b.fits"]
        with caplog.at_level(logging.DEBUG, logger="bandaid"):
            scripts.process_batch(files, _dummy_prep(), user_specific_metadata={})

        progress = [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("processing ")
        ]
        assert progress == [
            "processing 1/3: night1/a.fits",
            "processing 2/3: night2/a.fits",
            "processing 3/3: night2/b.fits",
        ]

    def test_info_level_summarises_each_frame_and_the_batch(
        self, mocker, caplog, by_filter
    ):
        """At INFO each frame gets an outcome line, with the batch bracketed."""
        mocker.patch(
            "bandaid.scripts.process_one_image",
            side_effect=[
                by_filter(),
                TooFewStarsError("too few", file="b.fits"),
            ],
        )

        with caplog.at_level(logging.INFO, logger="bandaid"):
            scripts.process_batch(
                ["a.fits", "b.fits"], _dummy_prep(), user_specific_metadata={}
            )

        messages = [record.getMessage() for record in caplog.records]
        assert messages[0].startswith("photometering 2 frames against ")
        assert messages[0].endswith("; output to memory")
        assert any(m.startswith("[1/2] a.fits: ok, WCS solved,") for m in messages)
        assert any(
            m.startswith("[2/2] b.fits: skipped: TooFewStarsError, WCS unknown")
            for m in messages
        )
        assert messages[-1] == "finished: 1 of 2 frames photometered"

    def test_unexpected_error_still_gets_an_outcome_line(self, mocker, caplog):
        """In robust mode an unexpected error is summarised like any other frame."""
        mocker.patch(
            "bandaid.scripts.process_one_image", side_effect=RuntimeError("boom")
        )

        with caplog.at_level(logging.INFO, logger="bandaid"):
            scripts.process_batch(
                ["a.fits"], _dummy_prep(), user_specific_metadata={}, fail_fast=False
            )

        messages = [record.getMessage() for record in caplog.records]
        assert any(
            m.startswith("[1/1] a.fits: error: RuntimeError, WCS unknown")
            for m in messages
        )

    def test_info_level_hides_stage_detail(
        self, patched_process_one_image, caplog, by_filter
    ):
        """Per-stage detail is DEBUG-only, so -v stays one line per frame."""
        patched_process_one_image(by_filter())

        with caplog.at_level(logging.INFO, logger="bandaid"):
            scripts.process_batch(["a.fits"], _dummy_prep(), user_specific_metadata={})

        messages = [record.getMessage() for record in caplog.records]
        assert not [m for m in messages if "detected" in m or "quality cuts" in m]

    def test_debug_level_reports_stage_counts_and_the_write(
        self, patched_process_one_image, tmp_path, caplog, by_filter
    ):
        """At DEBUG a frame also logs its star counts and the file it wrote."""
        patched_process_one_image(by_filter())

        with caplog.at_level(logging.DEBUG, logger="bandaid"):
            scripts.process_batch(
                ["a.fits"],
                _dummy_prep(),
                user_specific_metadata={},
                output_dir=tmp_path,
                write_frame=lambda _by_filter, path: path,
            )

        messages = [record.getMessage() for record in caplog.records]
        assert any(
            m.startswith("a.fits: ") and "pass the quality cuts" in m for m in messages
        )
        assert any(m.startswith("a.fits: wrote ") for m in messages)
        assert any(m.startswith("wrote the QA manifest to ") for m in messages)

    def test_failed_frames_are_skipped(self, mocker, by_filter):
        """A frame whose ``process_one_image`` raises a FrameError is omitted."""
        prep = _dummy_prep()
        mocker.patch(
            "bandaid.scripts.process_one_image",
            side_effect=_raise_on(
                "bad.fits",
                TooFewStarsError("too few stars", file="bad.fits"),
                by_filter,
            ),
        )

        results = scripts.process_batch(
            ["good.fits", "bad.fits"],
            prep,
            user_specific_metadata={},
        )

        assert list(results) == ["good.fits"]

    def test_unexpected_error_propagates_when_fail_fast(self, mocker):
        """A non-FrameError bug aborts the batch by default (fail_fast=True)."""

        def _boom(*_args: object, **_kwargs: object):
            msg = "a real bug"
            raise RuntimeError(msg)

        mocker.patch("bandaid.scripts.process_one_image", side_effect=_boom)

        with pytest.raises(RuntimeError, match="a real bug"):
            scripts.process_batch(
                ["a.fits", "b.fits"],
                _dummy_prep(),
                user_specific_metadata={},
            )

    def test_unexpected_error_skipped_when_not_fail_fast(self, mocker, by_filter):
        """With fail_fast=False, an unexpected bug is logged and skipped."""
        mocker.patch(
            "bandaid.scripts.process_one_image",
            side_effect=_raise_on(
                "bad.fits",
                RuntimeError("a real bug"),
                by_filter,
            ),
        )

        results = scripts.process_batch(
            ["good.fits", "bad.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            fail_fast=False,
        )

        assert list(results) == ["good.fits"]

    def test_starless_frame_is_skipped_in_memory_mode(self, mocker, by_filter):
        """A frame with no usable star in any filter is omitted, as on disk."""

        def _maybe(file, *_args: object, **_kwargs: object):
            result = by_filter()
            if file == "starless.fits":
                for table in result.values():
                    table["tot_count"] = [-1.0, 0.0]
            return result

        mocker.patch("bandaid.scripts.process_one_image", side_effect=_maybe)

        results = scripts.process_batch(
            ["starless.fits", "good.fits"], _dummy_prep(), user_specific_metadata={}
        )

        assert list(results) == ["good.fits"]

    def test_loads_each_frame_exactly_once(
        self, patched_process_one_image, mocker, by_filter
    ):
        """process_batch loads each frame exactly once via the shared loader (#44)."""
        patched_process_one_image(by_filter())
        load_frame = _stub_load_frame_calls(mocker)

        files = ["a.fits", "b.fits"]
        scripts.process_batch(files, _dummy_prep(), user_specific_metadata={})

        assert [call.args[0] for call in load_frame.call_args_list] == files

    def test_provided_first_frame_is_not_reloaded(
        self, patched_process_one_image, mocker, by_filter
    ):
        """
        process_batch reuses a caller-provided first-frame load (#44).

        Encodes "exactly once per run" including the first-frame double-open:
        ``photometer_frames`` already opened the first file for ``prepare_batch``
        and hands that load in as ``first_frame=``, so ``process_batch`` must not
        reopen it, while still loading every other frame normally.
        """
        patched_process_one_image(by_filter())
        load_frame = _stub_load_frame_calls(mocker)
        first = scripts.LoadedFrame(np.zeros((2, 2)), dict(_CONSISTENT_HEADER))

        scripts.process_batch(
            ["first.fits", "second.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            first_frame=first,
        )

        # Only the second frame reaches the loader; the first came in preloaded.
        assert [call.args[0] for call in load_frame.call_args_list] == ["second.fits"]


@pytest.mark.usefixtures("_consistent_headers")
class TestProcessBatchToDisk:
    """Unit tests for the ``output_dir`` (write starlists to disk) path."""

    def test_writes_one_file_per_frame(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """Each processed frame produces one ``<stem>.star`` file in output_dir."""
        patched_process_one_image(by_filter())

        scripts.process_batch(
            ["a.fits", "b.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        # Ignore the QA manifest sibling; this test is about the starlist files.
        assert _starlist_names(tmp_path) == ["a.star", "b.star"]

    def test_output_filename_is_stem_plus_default_suffix(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """The output name is the input *stem* + ``.star``; the input dir is dropped."""
        patched_process_one_image(by_filter())

        scripts.process_batch(
            ["sub/frame1.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        assert _starlist_names(tmp_path) == ["frame1.star"]

    def test_same_basename_different_dirs_mirror_source_tree(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """Same-named frames from different dirs are written under mirrored subdirs."""
        patched_process_one_image(by_filter())

        inputs = ["n1/img.fits", "n2/img.fits"]
        results = scripts.process_batch(
            inputs,
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        # A mix of source directories mirrors the tree: <dirname>/<stem>.star,
        # keeping clean basenames while staying distinct on disk.
        written = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*.star"))
        assert written == [Path("n1/img.star"), Path("n2/img.star")]
        # Both inputs are kept in the result, each mapped to its own output path.
        assert set(results) == set(inputs)
        assert len({str(v) for v in results.values()}) == len(results)

    def test_distinct_dirs_sharing_a_basename_get_unique_subdirs(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """Two different source dirs with the same name still mirror distinctly."""
        patched_process_one_image(by_filter())

        # Both parent directories are named "night" but live in different trees.
        (tmp_path / "a" / "night").mkdir(parents=True)
        (tmp_path / "b" / "night").mkdir(parents=True)
        inputs = [
            str(tmp_path / "a" / "night" / "img.fits"),
            str(tmp_path / "b" / "night" / "img.fits"),
        ]
        out = tmp_path / "out"

        results = scripts.process_batch(
            inputs,
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=out,
        )

        written = sorted(p.relative_to(out) for p in out.rglob("*.star"))
        # The colliding "night" subdir name is disambiguated with a numeric suffix.
        assert written == [Path("night/img.star"), Path("night_1/img.star")]
        assert len({str(v) for v in results.values()}) == len(results)

    def test_same_stem_one_dir_falls_back_to_numeric_suffix(
        self, monkeypatch, patched_process_one_image, tmp_path, by_filter
    ):
        """Two single-dir inputs differing only by extension stay distinct + flat."""
        patched_process_one_image(by_filter())
        # Both inputs live in the same directory (the cwd), so the layout stays
        # flat; their shared stem "img" is disambiguated with a numeric suffix
        # rather than a leading-underscore or directory prefix.
        # monkeypatch.chdir is the right tool here (not mocker): it restores the
        # original cwd after the test even on failure, which mocker has no
        # equivalent for.
        monkeypatch.chdir(tmp_path)

        inputs = ["img.fit", "img.fits"]
        scripts.process_batch(
            inputs,
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        assert _starlist_names(tmp_path) == ["img.star", "img_1.star"]

    def test_custom_output_suffix_is_honored(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """An explicit ``output_suffix`` replaces the default ``.star``."""
        patched_process_one_image(by_filter())

        scripts.process_batch(
            ["frame1.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
            output_suffix=".starlist",
        )

        assert _starlist_names(tmp_path) == ["frame1.starlist"]

    def test_written_file_round_trips_through_starlistset(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """The file is a valid StarListSet: one StarList per filter, stars intact."""
        filters = ("TR", "TG", "TB")
        patched_process_one_image(by_filter(filters))

        scripts.process_batch(
            ["frame1.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        text = (tmp_path / "frame1.star").read_text()
        star_list_set = StarListSet.model_validate_json(text)

        assert len(star_list_set.star_lists) == len(filters)
        for star_list in star_list_set.star_lists:
            kept_x = sorted(item.x for item in star_list.staritems)
            assert kept_x == [20.0, 70.0]

    def test_disk_mode_returns_path_mapping(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """Disk mode returns each input file mapped to its written output path."""
        patched_process_one_image(by_filter())

        results = scripts.process_batch(
            ["a.fits", "b.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        assert results == {
            "a.fits": tmp_path / "a.star",
            "b.fits": tmp_path / "b.star",
        }

    def test_failed_frames_write_no_file(self, mocker, tmp_path, by_filter):
        """A frame whose ``process_one_image`` raises a FrameError writes nothing."""
        mocker.patch(
            "bandaid.scripts.process_one_image",
            side_effect=_raise_on(
                "bad.fits",
                WCSSolveError("twirl found no match", file="bad.fits"),
                by_filter,
            ),
        )

        results = scripts.process_batch(
            ["good.fits", "bad.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        assert _starlist_names(tmp_path) == ["good.star"]
        assert results == {"good.fits": tmp_path / "good.star"}

    def test_writes_qa_manifest(self, mocker, tmp_path, by_filter):
        """A per-frame QA manifest records ok and skipped frames (#31)."""
        mocker.patch(
            "bandaid.scripts.process_one_image",
            side_effect=_raise_on(
                "bad.fits",
                WCSSolveError("twirl found no match", file="bad.fits"),
                by_filter,
            ),
        )

        scripts.process_batch(
            ["good.fits", "bad.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        assert (tmp_path / scripts.QA_MANIFEST_FILENAME).exists()
        rows = _read_manifest(tmp_path)

        expected_columns = {
            "file",
            "status",
            "n_detected",
            "sky_median",
            "fwhm",
            "wcs_solved",
            "n_good_stars",
            "n_centroid_drift",
            "n_drift_rejected",
        }
        assert expected_columns <= set(rows[0])
        by_file = {row["file"]: row for row in rows}
        assert set(by_file) == {"good.fits", "bad.fits"}

        good = by_file["good.fits"]
        assert good["status"] == "ok"
        assert good["wcs_solved"] == "True"
        # Both fixture rows are finite/positive/in-bounds, so both are "good".
        assert good["n_good_stars"] == "2"

        bad = by_file["bad.fits"]
        assert bad["status"].startswith("skipped")
        # A WCS solve failure is recorded as an explicit non-solve.
        assert bad["wcs_solved"] == "False"

    def test_qa_manifest_records_pointing_offset(self, mocker, tmp_path, by_filter):
        """
        ``pointing_offset_deg`` is blank only when no offset was computed.

        A drift rejection carries its offset on the ``FrameError``, so that
        frame's row records it; a frame rejected before the pointing comparison
        has no offset and stays blank.
        """
        assert "pointing_offset_deg" in scripts.QA_MANIFEST_COLUMNS

        def _consistency(file, _header, _prep):
            if file == "drifted.fits":
                exc = FrameError("frame pointing drifted", file=file)
                exc.pointing_offset = 1.234567
                raise exc
            if file == "nopointing.fits":
                msg = "no usable pointing"
                raise FrameMetadataError(msg, file=file)
            return 0.123456

        mocker.patch(
            "bandaid.scripts.check_frame_consistency", side_effect=_consistency
        )
        mocker.patch(
            "bandaid.scripts.process_one_image",
            side_effect=_raise_on(
                "nosolve.fits",
                WCSSolveError("twirl found no match", file="nosolve.fits"),
                by_filter,
            ),
        )

        scripts.process_batch(
            ["good.fits", "nosolve.fits", "drifted.fits", "nopointing.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        by_file = {row["file"]: row for row in _read_manifest(tmp_path)}
        assert float(by_file["good.fits"]["pointing_offset_deg"]) == pytest.approx(
            0.1235
        )
        assert float(by_file["nosolve.fits"]["pointing_offset_deg"]) == pytest.approx(
            0.1235
        )
        assert by_file["drifted.fits"]["status"].startswith("skipped")
        assert float(by_file["drifted.fits"]["pointing_offset_deg"]) == pytest.approx(
            1.2346
        )
        assert by_file["nopointing.fits"]["status"].startswith("skipped")
        assert by_file["nopointing.fits"]["pointing_offset_deg"] == ""

    def test_qa_manifest_records_edge_dropped(self, mocker, tmp_path, by_filter):
        """The manifest carries the per-frame edge-drop count from the table meta."""
        assert "n_edge_dropped" in scripts.QA_MANIFEST_COLUMNS

        def _process(file, *_args: object, **_kwargs: object):
            result = by_filter()
            if file == "edge.fits":
                for table in result.values():
                    table.meta["n_edge_dropped"] = 17
            return result

        mocker.patch("bandaid.scripts.process_one_image", side_effect=_process)

        scripts.process_batch(
            ["edge.fits", "unstamped.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        by_file = {row["file"]: row for row in _read_manifest(tmp_path)}
        assert by_file["edge.fits"]["n_edge_dropped"] == "17"
        assert by_file["unstamped.fits"]["n_edge_dropped"] == ""

    def test_qa_manifest_records_centroid_model_diagnostics(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """The plane summary stamped on the tables lands in the manifest row."""
        summary = {
            "g_cut": 12.1234,
            "n_cnn_class": 26,
            "plane_fallback": False,
            "plane_n_used": 28,
            "plane_n_clipped": 2,
            "plane_rms": 0.123456,
            "plane_dx_center": 0.301234,
            "plane_dy_center": -0.201234,
            "plane_dx_slope_x": 0.101234,
            "plane_dx_slope_y": -0.051234,
            "plane_dy_slope_x": 0.041234,
            "plane_dy_slope_y": 0.081234,
        }
        for column in summary:
            assert column in scripts.QA_MANIFEST_COLUMNS
        result = by_filter()
        for table in result.values():
            table.meta["centroid_model"] = summary
        patched_process_one_image(result)

        scripts.process_batch(
            ["a.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        row = _read_manifest(tmp_path)[0]
        assert row["n_cnn_class"] == "26"
        assert row["plane_fallback"] == "False"
        assert row["plane_n_used"] == "28"
        assert row["plane_n_clipped"] == "2"
        assert float(row["g_cut"]) == pytest.approx(12.1234)
        assert float(row["plane_rms"]) == pytest.approx(0.1235)
        assert float(row["plane_dx_center"]) == pytest.approx(0.3012)
        assert float(row["plane_dy_center"]) == pytest.approx(-0.2012)
        assert float(row["plane_dx_slope_x"]) == pytest.approx(0.1012)
        assert float(row["plane_dy_slope_y"]) == pytest.approx(0.0812)

    def test_centroid_model_columns_are_manifest_columns(self):
        """Every policy column is a manifest column, and the record has all of them."""
        for column in _CENTROID_MODEL_COLUMNS:
            assert column in scripts.QA_MANIFEST_COLUMNS
        assert set(_centroid_model_record(None)) == set(_CENTROID_MODEL_COLUMNS)

    def test_qa_manifest_centroid_model_blank_without_a_summary(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """A frame the policy did not run on records the plane columns blank."""
        patched_process_one_image(by_filter())

        scripts.process_batch(
            ["a.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        row = _read_manifest(tmp_path)[0]
        assert row["plane_n_used"] == ""
        assert row["plane_fallback"] == ""
        assert row["n_cnn_class"] == ""

    def test_qa_manifest_records_solve_quality(self, mocker, tmp_path, by_filter):
        """
        The manifest carries the solved scale, solve offset and SNR >= 20 count.

        ``wcs_pixscale`` and ``solve_offset_deg`` are read from the table meta
        that ``process_one_image`` stamps; ``n_snr20`` counts representative-channel
        rows that pass ``good_star_mask`` with ``snr >= 20``. A frame that never
        solved has no such numbers and records them blank, except that a frame
        rejected for its plate scale still records the scale it measured, so a
        fully rejected run can be used to calibrate ``pixscale``.
        """
        for column in ("wcs_pixscale", "solve_offset_deg", "n_snr20"):
            assert column in scripts.QA_MANIFEST_COLUMNS

        def _solved():
            result = by_filter()
            for table in result.values():
                table["snr"] = [25.0, 10.0]
                table.meta["wcs_pixscale"] = 2.37654
                table.meta["solve_offset_deg"] = 0.04321
            return result

        errors = {
            "nosolve.fits": WCSSolveError("twirl found no match"),
            "badscale.fits": WCSScaleError("out of tolerance", measured_scale=2.43217),
        }

        def _process(file, *_args: object, **_kwargs: object):
            if file in errors:
                raise errors[file]
            return _solved()

        mocker.patch("bandaid.scripts.process_one_image", side_effect=_process)

        scripts.process_batch(
            ["good.fits", "nosolve.fits", "badscale.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        by_file = {row["file"]: row for row in _read_manifest(tmp_path)}
        good = by_file["good.fits"]
        assert float(good["wcs_pixscale"]) == pytest.approx(2.3765)
        assert float(good["solve_offset_deg"]) == pytest.approx(0.0432)
        assert good["n_snr20"] == "1"
        nosolve = by_file["nosolve.fits"]
        assert nosolve["wcs_pixscale"] == ""
        assert nosolve["solve_offset_deg"] == ""
        assert nosolve["n_snr20"] == ""
        badscale = by_file["badscale.fits"]
        assert float(badscale["wcs_pixscale"]) == pytest.approx(2.4322)
        assert badscale["solve_offset_deg"] == ""

    def test_a_poorly_fitting_plane_skips_the_frame_and_the_batch_continues(
        self, mocker, tmp_path, by_filter
    ):
        """A `CentroidPlaneError` is a skipped status with its rms; later frames run."""
        error = CentroidPlaneError("plane rms 2.18 px", plane_rms=2.18, n_fit_stars=30)

        def _process(file, *_args: object, **_kwargs: object):
            if file == "trailed.fits":
                raise error
            return by_filter()

        mocker.patch("bandaid.scripts.process_one_image", side_effect=_process)

        results = scripts.process_batch(
            ["trailed.fits", "good.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        by_file = {row["file"]: row for row in _read_manifest(tmp_path)}
        assert by_file["trailed.fits"]["status"] == "skipped: CentroidPlaneError"
        assert float(by_file["trailed.fits"]["plane_rms"]) == pytest.approx(2.18)
        assert by_file["good.fits"]["status"] == "ok"
        assert list(results) == ["good.fits"]

    def test_qa_manifest_sky_median_is_median_of_bkgd_count(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """
        ``sky_median`` is the median of the per-star ``bkgd_count`` (#52).

        The broken ``sky`` column is gone, so the manifest derives its sky
        estimate from ``bkgd_count`` -- the correct per-star per-pixel annulus
        background -- and is finally an actual median, as the docs describe.
        """
        result = by_filter()
        # Distinct per-star backgrounds in the representative (first) table so
        # the median is unambiguous: median([2, 8]) = 5.
        result["TR"]["bkgd_count"] = [2.0, 8.0]
        patched_process_one_image(result)

        scripts.process_batch(
            ["a.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        rows = _read_manifest(tmp_path)

        assert float(rows[0]["sky_median"]) == pytest.approx(5.0)

    def test_qa_manifest_sky_median_ignores_nan_bkgd_count(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """
        A NaN ``bkgd_count`` (edge-of-frame annulus) stays out of ``sky_median``.

        Per the NaN contract in `bandaid.photometry.measure_photometry`, an
        edge-of-frame or fully-masked annulus yields NaN; one such star must not
        poison the frame's QA value.
        """
        result = by_filter()
        result["TR"]["bkgd_count"] = [np.nan, 7.0]
        patched_process_one_image(result)

        scripts.process_batch(
            ["a.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        rows = _read_manifest(tmp_path)

        assert float(rows[0]["sky_median"]) == pytest.approx(7.0)

    def test_qa_manifest_drift_rejected_counts_flagged_star_that_survives_filtering(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """
        A drift-flagged star that still passes ``good_star_mask`` is "rejected" (#60).

        ``n_drift_rejected`` is the marginal effect a future gate would have:
        stars that are both drift-flagged and would otherwise reach the output.
        Both fixture rows are already finite/positive/in-bounds ("good"), so
        flagging one as drifted makes it count in both totals.
        """
        result = by_filter()
        result["TR"]["centroid_drift"] = [True, False]
        patched_process_one_image(result)

        scripts.process_batch(
            ["a.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        rows = _read_manifest(tmp_path)

        assert rows[0]["n_centroid_drift"] == "1"
        assert rows[0]["n_drift_rejected"] == "1"

    def test_qa_manifest_drift_rejected_excludes_star_failing_flux_cut(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """
        A drift-flagged star already dropped by the flux cut is not "rejected" (#60).

        Most drifted stars are already excluded by the existing flux/error/bounds
        cuts, so ``n_drift_rejected`` must stay 0 for a star that is both
        drift-flagged and fails ``good_star_mask`` on its own -- only
        ``n_centroid_drift`` (the raw flag count) should see it.
        """
        result = by_filter()
        # good_star_mask requires tot_count > 0; fail it for the drifted row.
        result["TR"]["tot_count"] = [-1.0, 300.0]
        result["TR"]["centroid_drift"] = [True, False]
        patched_process_one_image(result)

        scripts.process_batch(
            ["a.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        rows = _read_manifest(tmp_path)

        assert rows[0]["n_centroid_drift"] == "1"
        assert rows[0]["n_drift_rejected"] == "0"

    def test_qa_manifest_n_good_stars_honors_default_min_snr_floor(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """
        ``n_good_stars`` applies the SNR >= 2 default floor (#101).

        ``by_filter`` tables carry no stamped ``min_snr`` meta, so
        `qa_record_ok` -> `good_star_mask` falls back to the 2.0 default; that
        fallback is exactly what this test pins.
        """
        result = by_filter()
        result["TR"]["snr"] = [5.0, 1.0]
        patched_process_one_image(result)

        scripts.process_batch(
            ["a.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        rows = _read_manifest(tmp_path)

        assert rows[0]["n_good_stars"] == "1"

    def test_qa_manifest_dropped_filters_names_starved_filter(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """
        A filter with no ``good_star_mask`` survivors is named in ``dropped_filters``.

        The writer drops such a filter from the ``.star`` output but still
        writes the frame from its surviving siblings (#109); the manifest must
        record which filter was dropped even though the row's own ``status``
        stays "ok".
        """
        result = by_filter(("TR", "TG"))
        # good_star_mask requires tot_count > 0; starve every row in TR.
        result["TR"]["tot_count"] = [-1.0, 0.0]
        patched_process_one_image(result)

        scripts.process_batch(
            ["a.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        rows = _read_manifest(tmp_path)

        assert rows[0]["status"] == "ok"
        assert rows[0]["dropped_filters"] == "TR"

    def test_qa_manifest_dropped_filters_is_empty_for_clean_frame(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """A frame where every filter keeps at least one star records no drops."""
        patched_process_one_image(by_filter())

        scripts.process_batch(
            ["a.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        rows = _read_manifest(tmp_path)

        assert rows[0]["dropped_filters"] == ""

    def test_qa_manifest_can_be_disabled(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """``write_manifest=False`` writes only starlists, no manifest."""
        patched_process_one_image(by_filter())

        scripts.process_batch(
            ["a.fits", "b.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
            write_manifest=False,
        )

        assert not (tmp_path / scripts.QA_MANIFEST_FILENAME).exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["a.star", "b.star"]

    def test_qa_manifest_name_is_honored(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """An explicit ``qa_manifest_name`` overrides the default filename."""
        patched_process_one_image(by_filter())

        scripts.process_batch(
            ["a.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
            qa_manifest_name="run_quality.csv",
        )

        assert (tmp_path / "run_quality.csv").exists()
        assert not (tmp_path / scripts.QA_MANIFEST_FILENAME).exists()

    def test_custom_write_frame_gets_rich_tables_and_path(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """A custom ``write_frame`` is called once per frame with the rich tables."""
        calls = []

        def spy_writer(frame_result, output_path):
            calls.append((frame_result, output_path))
            output_path.write_text("recorded")
            return output_path

        patched_process_one_image(by_filter())

        scripts.process_batch(
            ["a.fits", "b.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
            write_frame=spy_writer,
        )

        # One call per frame, each handed its resolved output path.
        assert [path for _result, path in calls] == [
            tmp_path / "a.star",
            tmp_path / "b.star",
        ]
        # The writer receives the rich astropy tables (not pre-built StarLists):
        # the full {filter: Table} mapping, each table keeping its columns + meta.
        frame_result, _path = calls[0]
        assert set(frame_result) == {"TR", "TG"}
        table = frame_result["TR"]
        assert isinstance(table, Table)
        assert "tot_count" in table.colnames
        assert "full_image_meta" in table.meta

    def test_write_frame_return_value_lands_in_results(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """Whatever ``write_frame`` returns is stored as ``results[file]``."""
        sentinel = tmp_path / "somewhere" / "custom.out"

        def writer(_frame_result, _output_path):
            return sentinel

        patched_process_one_image(by_filter())

        results = scripts.process_batch(
            ["a.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
            write_frame=writer,
        )

        assert results == {"a.fits": sentinel}

    def test_write_frame_not_called_in_memory_mode(
        self, patched_process_one_image, by_filter
    ):
        """In-memory mode ignores ``write_frame`` and returns the tables."""

        def boom(_frame_result, _output_path):
            msg = "write_frame must not run in in-memory mode"
            raise AssertionError(msg)

        patched_process_one_image(by_filter())

        results = scripts.process_batch(
            ["a.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=None,
            write_frame=boom,
        )

        assert set(results) == {"a.fits"}
        assert set(results["a.fits"]) == {"TR", "TG"}

    def test_default_write_frame_writes_starlist(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """With no ``write_frame`` given, the default still writes a StarListSet."""
        patched_process_one_image(by_filter())

        scripts.process_batch(
            ["frame1.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        # A valid StarListSet document, exactly as before the writer seam existed.
        StarListSet.model_validate_json((tmp_path / "frame1.star").read_text())

    def test_writer_frame_error_skips_frame_not_batch(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """A ``FrameError`` raised at write time is skipped, not batch-fatal."""
        patched_process_one_image(by_filter())

        def rejecting(frame_result, output_path):
            # The one FrameError the default writer can still raise after
            # process_frame reported the frame ok: a schema rejection.
            if output_path.stem == "rejected":
                msg = "schema rejected a row"
                raise StarListValidationError(msg)
            return write_starlist_set(frame_result, output_path)

        inputs = ["rejected.fits", "good.fits"]
        results = scripts.process_batch(
            inputs,
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
            write_frame=rejecting,
        )

        # The batch survives the rejected frame: the later frame is still
        # written, and only it appears in the results.
        assert results == {"good.fits": tmp_path / "good.star"}
        assert _starlist_names(tmp_path) == ["good.star"]

        rows = _read_manifest(tmp_path)

        # One row per input frame: the rejected frame's ok record from
        # process_frame is replaced by the skip, not duplicated alongside it.
        by_file = {row["file"]: row for row in rows}
        assert len(rows) == len(by_file) == len(inputs)
        assert by_file["rejected.fits"]["status"] == "skipped: StarListValidationError"
        assert by_file["good.fits"]["status"] == "ok"

    def test_writer_non_frame_error_still_propagates(
        self, patched_process_one_image, tmp_path, by_filter
    ):
        """A genuine write failure (not a FrameError) still aborts the batch."""

        def denied(_frame_result, _output_path):
            msg = "simulated unwritable output"
            raise PermissionError(msg)

        patched_process_one_image(by_filter())

        with pytest.raises(PermissionError, match="simulated unwritable output"):
            scripts.process_batch(
                ["a.fits"],
                _dummy_prep(),
                user_specific_metadata={},
                output_dir=tmp_path,
                write_frame=denied,
            )


@pytest.mark.usefixtures("_consistent_headers")
class TestProcessFrame:
    """Unit tests for ``process_frame``, the per-frame public API."""

    def test_qa_record_builders_are_root_exports(self):
        """The QA record builders import from the package root."""
        assert bandaid.qa_record_ok is scripts.qa_record_ok
        assert bandaid.qa_record_skipped is scripts.qa_record_skipped
        assert bandaid.qa_record_error is scripts.qa_record_error

    def test_ok_outcome_carries_tables_and_qa(
        self, patched_process_one_image, by_filter
    ):
        """A clean frame is ``ok`` with its tables, no error and a full QA row."""
        tables = patched_process_one_image(by_filter())

        result = process_frame("a.fits", _dummy_prep(), user_specific_metadata={})

        assert isinstance(result, FrameResult)
        assert result.outcome is FrameOutcome.OK
        assert result.ok
        assert result.by_filter is tables
        assert result.dropped_filters == ()
        assert result.error is None
        assert result.reason_type is None
        assert set(result.qa) == set(scripts.QA_MANIFEST_COLUMNS)
        assert result.qa["status"] == "ok"
        assert result.qa["n_good_stars"] == len(tables["TR"])
        assert result.schema_version == 1

    def test_qa_values_keep_full_precision(self, mocker, by_filter):
        """The QA row is unrounded; rounding belongs to the manifest writer."""
        # Six decimals, so a 4-decimal rounding would show.
        pointing_offset = 0.123456
        pixscale = 2.37654
        solve_offset = 0.04321
        plane_rms = 0.123456
        mocker.patch(
            "bandaid.scripts.check_frame_consistency", return_value=pointing_offset
        )
        tables = by_filter()
        for table in tables.values():
            table.meta["wcs_pixscale"] = pixscale
            table.meta["solve_offset_deg"] = solve_offset
            table.meta["centroid_model"] = {"plane_rms": plane_rms}
        mocker.patch("bandaid.scripts.process_one_image", return_value=tables)

        result = process_frame("a.fits", _dummy_prep(), user_specific_metadata={})

        assert result.pointing_offset == pointing_offset
        assert result.qa["pointing_offset_deg"] == pointing_offset
        assert result.qa["wcs_pixscale"] == pixscale
        assert result.qa["solve_offset_deg"] == solve_offset
        assert result.qa["plane_rms"] == plane_rms

    def test_file_is_the_path_unless_a_label_is_given(
        self, patched_process_one_image, by_filter
    ):
        """``file`` and the QA row name the frame by path, or by the label's string."""
        patched_process_one_image(by_filter())
        prep = _dummy_prep()

        by_path = process_frame(Path("night/a.fits"), prep, user_specific_metadata={})
        labelled = process_frame(
            "a.fits", prep, user_specific_metadata={}, label=Path("p/q.fits")
        )

        assert by_path.file == str(Path("night/a.fits"))
        assert by_path.qa["file"] == str(Path("night/a.fits"))
        assert labelled.file == str(Path("p/q.fits"))
        assert labelled.qa["file"] == str(Path("p/q.fits"))

    def test_label_names_the_frame_on_the_error_and_in_the_log(self, mocker, caplog):
        """A label, not the path, reaches the skipped error, the log and the steps."""
        error = TooFewStarsError("too few stars")
        process_one_image = mocker.patch(
            "bandaid.scripts.process_one_image", side_effect=error
        )
        consistency = mocker.patch(
            "bandaid.scripts.check_frame_consistency", return_value=0.0
        )

        with caplog.at_level(logging.WARNING, logger="bandaid"):
            result = process_frame(
                "/data/night/a.fits",
                _dummy_prep(),
                user_specific_metadata={},
                label="night/a.fits",
            )

        assert result.error.file == "night/a.fits"
        assert "skipping night/a.fits: too few stars" in caplog.text
        assert "/data/" not in caplog.text
        assert consistency.call_args.args[0] == "night/a.fits"
        assert process_one_image.call_args.args[0] == "night/a.fits"

    def test_label_names_the_frame_in_the_unexpected_error_log(self, mocker, caplog):
        """The ``error`` outcome's log line names the label too."""
        mocker.patch(
            "bandaid.scripts.process_one_image", side_effect=RuntimeError("a bug")
        )

        with caplog.at_level(logging.ERROR, logger="bandaid"):
            process_frame(
                "/data/night/a.fits",
                _dummy_prep(),
                user_specific_metadata={},
                label="night/a.fits",
                fail_fast=False,
            )

        assert "unexpected error on night/a.fits" in caplog.text
        assert "/data/" not in caplog.text

    @pytest.mark.parametrize(
        ("error", "fail_fast"),
        [(TooFewStarsError("too few stars"), True), (RuntimeError("a bug"), False)],
        ids=["skipped", "error"],
    )
    def test_failed_frame_does_not_keep_the_pixels_alive(
        self, mocker, error, fail_fast
    ):
        """A skip or error releases the pixels; the error keeps its traceback."""
        data = np.zeros((2, 2))
        pixels = weakref.ref(data)
        frame = scripts.LoadedFrame(data, dict(_CONSISTENT_HEADER))

        def failing(_file, *_args: object, frame, **_kwargs: object):
            # A local holding the pixels, as the real pipeline's frames do; an
            # uncleared traceback would keep it, and the array, alive.
            held = frame.data
            assert held.shape == (2, 2)
            raise error

        mocker.patch("bandaid.scripts.process_one_image", new=failing)

        result = process_frame(
            "a.fits",
            _dummy_prep(),
            user_specific_metadata={},
            frame=frame,
            fail_fast=fail_fast,
        )
        del data, frame

        assert result.error is error
        assert result.error.__traceback__ is not None
        assert pixels() is None

    def test_pointing_drift_skip_reports_the_offset(self, mocker):
        """A drift rejection's offset reaches both the result and its QA row."""
        offset = 1.5
        error = FrameError("frame pointing drifted")
        error.pointing_offset = offset
        mocker.patch("bandaid.scripts.check_frame_consistency", side_effect=error)

        result = process_frame("a.fits", _dummy_prep(), user_specific_metadata={})

        assert result.outcome is FrameOutcome.SKIPPED
        assert result.pointing_offset == offset
        assert result.qa["pointing_offset_deg"] == offset

    def test_qa_manifest_columns_are_the_schema(self):
        """The manifest columns are fixed; a change here bumps ``schema_version``."""
        assert scripts.QA_MANIFEST_COLUMNS == (
            "file",
            "status",
            "n_detected",
            "sky_median",
            "fwhm",
            "wcs_solved",
            "pointing_offset_deg",
            "wcs_pixscale",
            "solve_offset_deg",
            "n_good_stars",
            "n_snr20",
            "dropped_filters",
            "n_centroid_drift",
            "n_drift_rejected",
            "n_forced_measured",
            "n_edge_dropped",
            "g_cut",
            "n_cnn_class",
            "plane_fallback",
            "plane_n_used",
            "plane_n_clipped",
            "plane_rms",
            "plane_dx_center",
            "plane_dy_center",
            "plane_dx_slope_x",
            "plane_dx_slope_y",
            "plane_dy_slope_x",
            "plane_dy_slope_y",
        )
        assert FrameResult.schema_version == 1

    def test_dropped_filters_names_the_starved_filter(
        self, patched_process_one_image, by_filter
    ):
        """A filter with no survivor is dropped while the frame stays ``ok``."""
        tables = by_filter(("TR", "TG"))
        tables["TR"]["tot_count"] = [-1.0, 0.0]
        patched_process_one_image(tables)

        result = process_frame("a.fits", _dummy_prep(), user_specific_metadata={})

        assert result.ok
        assert result.dropped_filters == ("TR",)
        assert result.qa["dropped_filters"] == "TR"

    def test_frame_error_is_a_skip(self, mocker, caplog):
        """A ``FrameError`` gives a skipped result carrying the error, and a warning."""
        error = TooFewStarsError("too few stars")
        mocker.patch("bandaid.scripts.process_one_image", side_effect=error)

        with caplog.at_level(logging.WARNING, logger="bandaid"):
            result = process_frame("a.fits", _dummy_prep(), user_specific_metadata={})

        assert result.outcome is FrameOutcome.SKIPPED
        assert not result.ok
        assert result.error is error
        assert error.file == "a.fits"
        assert result.reason_type == "TooFewStarsError"
        assert result.by_filter == {}
        assert result.dropped_filters == ()
        assert result.qa["status"] == "skipped: TooFewStarsError"
        assert result.qa["wcs_solved"] is None
        assert "skipping a.fits: too few stars" in caplog.text

    def test_rejected_solve_fills_the_solve_columns(self, mocker):
        """A scale rejection records the non-solve and its measured scale."""
        measured_scale = 2.43217
        mocker.patch(
            "bandaid.scripts.process_one_image",
            side_effect=WCSScaleError(
                "out of tolerance", measured_scale=measured_scale
            ),
        )

        result = process_frame("a.fits", _dummy_prep(), user_specific_metadata={})

        assert result.qa["wcs_solved"] is False
        assert result.qa["wcs_pixscale"] == measured_scale

    def test_no_usable_star_in_any_filter_is_a_skip(
        self, patched_process_one_image, by_filter
    ):
        """A starless frame is final on return: skipped, no tables."""
        tables = by_filter()
        for table in tables.values():
            table["tot_count"] = [-1.0, 0.0]
        patched_process_one_image(tables)

        result = process_frame("a.fits", _dummy_prep(), user_specific_metadata={})

        assert result.outcome is FrameOutcome.SKIPPED
        assert isinstance(result.error, NoUsableStarsError)
        assert result.by_filter == {}
        assert result.qa["status"] == "skipped: NoUsableStarsError"

    def test_starless_frame_logs_its_stages_before_the_skip(
        self, patched_process_one_image, by_filter, caplog
    ):
        """A starless frame still logs its stage counts, ahead of the skip warning."""
        tables = by_filter()
        for table in tables.values():
            table["tot_count"] = [-1.0, 0.0]
        patched_process_one_image(tables)

        with caplog.at_level(logging.DEBUG, logger="bandaid"):
            process_frame("a.fits", _dummy_prep(), user_specific_metadata={})

        messages = [record.getMessage() for record in caplog.records]
        stages = [
            i
            for i, m in enumerate(messages)
            if m.startswith("a.fits: ") and "pass the quality cuts" in m
        ]
        skips = [i for i, m in enumerate(messages) if "skipping" in m]
        assert stages
        assert skips
        assert stages[0] < skips[0]

    def test_qa_values_are_plain_python_types(
        self, patched_process_one_image, by_filter
    ):
        """Numpy scalars stamped in table meta come out as plain Python types."""
        tables = by_filter()
        for table in tables.values():
            table.meta["fwhm"] = np.float64(2.5)
            table.meta["n_edge_dropped"] = np.int64(3)
            table.meta["centroid_model"] = {
                "n_cnn_class": np.int64(26),
                "plane_fallback": np.False_,
            }
        patched_process_one_image(tables)

        result = process_frame("a.fits", _dummy_prep(), user_specific_metadata={})

        assert type(result.qa["fwhm"]) is float
        assert type(result.qa["n_edge_dropped"]) is int
        assert type(result.qa["n_cnn_class"]) is int
        assert type(result.qa["plane_fallback"]) is bool
        assert json.dumps(result.qa)

    def test_unexpected_error_propagates_by_default(self, mocker):
        """A non-``FrameError`` is re-raised when ``fail_fast`` is on."""
        mocker.patch(
            "bandaid.scripts.process_one_image", side_effect=RuntimeError("a real bug")
        )

        with pytest.raises(RuntimeError, match="a real bug"):
            process_frame("a.fits", _dummy_prep(), user_specific_metadata={})

    def test_unexpected_error_is_an_error_outcome_when_not_fail_fast(
        self, mocker, caplog
    ):
        """With ``fail_fast`` off a bug becomes an ``error`` result, logged."""
        error = RuntimeError("a real bug")
        mocker.patch("bandaid.scripts.process_one_image", side_effect=error)

        with caplog.at_level(logging.ERROR, logger="bandaid"):
            result = process_frame(
                "a.fits", _dummy_prep(), user_specific_metadata={}, fail_fast=False
            )

        assert result.outcome is FrameOutcome.ERROR
        assert result.error is error
        assert result.reason_type == "RuntimeError"
        assert result.by_filter == {}
        assert result.qa["status"] == "error: RuntimeError"
        assert "unexpected error on a.fits" in caplog.text

    def test_elapsed_is_recorded(self, patched_process_one_image, by_filter):
        """The wall time of the frame is reported in seconds."""
        patched_process_one_image(by_filter())

        result = process_frame("a.fits", _dummy_prep(), user_specific_metadata={})

        assert isinstance(result.elapsed_s, float)
        assert result.elapsed_s >= 0

    def test_provided_frame_is_not_loaded(
        self, patched_process_one_image, by_filter, mocker
    ):
        """A caller-provided load is used as is; otherwise the file is opened once."""
        patched_process_one_image(by_filter())
        load_frame = _stub_load_frame_calls(mocker)
        frame = scripts.LoadedFrame(np.zeros((2, 2)), dict(_CONSISTENT_HEADER))

        process_frame("a.fits", _dummy_prep(), user_specific_metadata={}, frame=frame)
        assert load_frame.call_count == 0

        process_frame("b.fits", _dummy_prep(), user_specific_metadata={})
        assert [call.args[0] for call in load_frame.call_args_list] == ["b.fits"]

    def test_metadata_defaults_to_an_empty_dict(self, mocker, by_filter):
        """``user_specific_metadata=None`` reaches ``process_one_image`` as ``{}``."""
        process_one_image = mocker.patch(
            "bandaid.scripts.process_one_image", return_value=by_filter()
        )

        process_frame("a.fits", _dummy_prep())

        assert process_one_image.call_args.args[1] == {}

    def test_result_is_frozen(self, patched_process_one_image, by_filter):
        """The result cannot be rebound after the fact."""
        patched_process_one_image(by_filter())

        result = process_frame("a.fits", _dummy_prep(), user_specific_metadata={})

        with pytest.raises(dataclasses.FrozenInstanceError):
            result.outcome = FrameOutcome.ERROR

    def test_qa_is_json_native_for_every_outcome(self, mocker, by_filter):
        """Every QA value is a plain Python type, keyed exactly by the columns."""
        drift = FrameError("frame pointing drifted")
        drift.pointing_offset = np.float64(1.5)
        # The skipped cases carry numpy scalars on the error, as the real
        # raisers stamp them, so the record builder's coercion is exercised.
        mocker.patch(
            "bandaid.scripts.process_one_image",
            side_effect=[
                by_filter(),
                WCSScaleError("out of tolerance", measured_scale=np.float64(2.4)),
                CentroidPlaneError("plane too loose", plane_rms=np.float64(0.3)),
                RuntimeError("a real bug"),
            ],
        )
        prep = _dummy_prep()
        results = [
            process_frame(file, prep, user_specific_metadata={}, fail_fast=False)
            for file in ("ok.fits", "scale.fits", "plane.fits", "error.fits")
        ]
        mocker.patch("bandaid.scripts.check_frame_consistency", side_effect=drift)
        results.append(process_frame("drift.fits", prep, user_specific_metadata={}))

        assert [r.outcome for r in results] == [
            FrameOutcome.OK,
            FrameOutcome.SKIPPED,
            FrameOutcome.SKIPPED,
            FrameOutcome.ERROR,
            FrameOutcome.SKIPPED,
        ]
        for result in results:
            assert tuple(result.qa) == scripts.QA_MANIFEST_COLUMNS
            for value in result.qa.values():
                assert type(value) in {str, int, float, bool, type(None)}, result.qa
            json.dumps(result.qa)


class TestWriteQaManifest:
    """Unit tests for ``write_qa_manifest``'s rounding and blanks."""

    def test_rounds_the_solve_and_plane_columns_only(self, tmp_path):
        """Solve offsets, scale and plane values get 4 decimals; the rest is as is."""
        record = dict.fromkeys(scripts.QA_MANIFEST_COLUMNS)
        record.update(
            {
                "file": "a.fits",
                "status": "ok",
                "pointing_offset_deg": 0.123456,
                "wcs_pixscale": 2.37654,
                "solve_offset_deg": 0.04321,
                "sky_median": 5.123456789,
                "fwhm": 2.123456789,
                "plane_rms": 0.123456,
                "plane_fallback": False,
                "plane_n_used": 28,
            }
        )
        path = tmp_path / "qa.csv"

        write_qa_manifest(path, [record])

        with path.open(newline="") as f:
            row = next(csv.DictReader(f))
        assert row["pointing_offset_deg"] == "0.1235"
        assert row["wcs_pixscale"] == "2.3765"
        assert row["solve_offset_deg"] == "0.0432"
        assert row["plane_rms"] == "0.1235"
        assert row["sky_median"] == "5.123456789"
        assert row["fwhm"] == "2.123456789"
        assert row["plane_fallback"] == "False"
        assert row["plane_n_used"] == "28"
        assert row["n_good_stars"] == ""
