"""Unit tests for the per-frame batch loop and batch-to-disk orchestration."""

import csv
import logging
import os
from pathlib import Path

import numpy as np
import pytest
from _helpers import _dummy_prep
from aavso_starlist_schema import StarListSet
from astropy.table import Table

from bandaid import scripts
from bandaid.exceptions import (
    TooFewStarsError,
    WCSSolveError,
)
from bandaid.scripts import _quiet_hf_xet


@pytest.mark.usefixtures("_consistent_headers")
class TestProcessBatch:
    """Unit tests for ``process_batch``."""

    def test_one_result_per_frame_with_shared_prep(self, monkeypatch):
        """Each frame is processed once with the same shared prep objects."""
        prep = _dummy_prep()
        user_meta = {"observer": "abc"}
        calls = []

        def fake_process_one_image(
            file,
            meta,
            radecs,
            cnn,
            masks,
            *,
            input_photometry_coords,
            **_kwargs: object,
        ):
            calls.append((file, meta, radecs, cnn, masks, input_photometry_coords))
            return {"TR": Table({"tot_count": [1.0]})}

        monkeypatch.setattr(scripts, "process_one_image", fake_process_one_image)

        files = ["a.fits", "b.fits"]
        results = scripts.process_batch(files, prep, user_specific_metadata=user_meta)

        assert list(results) == files
        assert len(calls) == len(files)
        for file, call in zip(files, calls, strict=True):
            cfile, meta, radecs, cnn, masks, phot_coords = call
            assert cfile == file
            assert meta is user_meta
            assert radecs is prep.radecs
            assert cnn is prep.cnn
            assert masks is prep.bayer_masks
            assert phot_coords is prep.photometry_coords

    def test_emits_progress_log_per_frame(self, monkeypatch, caplog):
        """Each frame logs a ``processing i/N: name`` line at INFO for --verbose."""
        monkeypatch.setattr(
            scripts,
            "process_one_image",
            lambda *_a, **_k: {"TR": Table({"tot_count": [1.0]})},
        )

        # Identically-named frames from different directories (a supported
        # mirrored-tree batch): the line logs the full path, not just the
        # basename, so the two "a.fits" frames stay distinguishable.
        files = ["night1/a.fits", "night2/a.fits", "night2/b.fits"]
        with caplog.at_level(logging.INFO, logger="bandaid"):
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

    def test_failed_frames_are_skipped(self, monkeypatch):
        """A frame whose ``process_one_image`` raises a FrameError is omitted."""
        prep = _dummy_prep()

        def _maybe(file, *_args: object, **_kwargs: object):
            if file == "bad.fits":
                msg = "too few stars"
                raise TooFewStarsError(msg, file=file)
            return {"TR": Table({"tot_count": [1.0]})}

        monkeypatch.setattr(scripts, "process_one_image", _maybe)

        results = scripts.process_batch(
            ["good.fits", "bad.fits"],
            prep,
            user_specific_metadata={},
        )

        assert list(results) == ["good.fits"]

    def test_unexpected_error_propagates_when_fail_fast(self, monkeypatch):
        """A non-FrameError bug aborts the batch by default (fail_fast=True)."""

        def _boom(*_args: object, **_kwargs: object):
            msg = "a real bug"
            raise RuntimeError(msg)

        monkeypatch.setattr(scripts, "process_one_image", _boom)

        with pytest.raises(RuntimeError, match="a real bug"):
            scripts.process_batch(
                ["a.fits", "b.fits"],
                _dummy_prep(),
                user_specific_metadata={},
            )

    def test_unexpected_error_skipped_when_not_fail_fast(self, monkeypatch):
        """With fail_fast=False, an unexpected bug is logged and skipped."""

        def _maybe(file, *_args: object, **_kwargs: object):
            if file == "bad.fits":
                msg = "a real bug"
                raise RuntimeError(msg)
            return {"TR": Table({"tot_count": [1.0]})}

        monkeypatch.setattr(scripts, "process_one_image", _maybe)

        results = scripts.process_batch(
            ["good.fits", "bad.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            fail_fast=False,
        )

        assert list(results) == ["good.fits"]


class TestQuietHfXet:
    """Unit tests for the best-effort ``_quiet_hf_xet`` HF-warning silencer."""

    def test_sets_disable_xet_when_unset(self, monkeypatch):
        """With no user setting, xet is disabled to avoid its stderr warning."""
        monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
        _quiet_hf_xet()
        assert os.environ["HF_HUB_DISABLE_XET"] == "1"

    def test_preserves_user_value(self, monkeypatch):
        """A user who set the var (e.g. to keep xet) is never overridden."""
        monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
        _quiet_hf_xet()
        assert os.environ["HF_HUB_DISABLE_XET"] == "0"


@pytest.mark.usefixtures("_consistent_headers")
class TestProcessBatchToDisk:
    """Unit tests for the ``output_dir`` (write starlists to disk) path."""

    def test_writes_one_file_per_frame(self, monkeypatch, tmp_path, by_filter):
        """Each processed frame produces one ``<stem>.star`` file in output_dir."""
        monkeypatch.setattr(scripts, "process_one_image", lambda *a, **k: by_filter())

        scripts.process_batch(
            ["a.fits", "b.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        # Ignore the QA manifest sibling; this test is about the starlist files.
        written = sorted(
            p.name for p in tmp_path.iterdir() if p.name != scripts.QA_MANIFEST_FILENAME
        )
        assert written == ["a.star", "b.star"]

    def test_output_filename_is_stem_plus_default_suffix(
        self, monkeypatch, tmp_path, by_filter
    ):
        """The output name is the input *stem* + ``.star``; the input dir is dropped."""
        monkeypatch.setattr(scripts, "process_one_image", lambda *a, **k: by_filter())

        scripts.process_batch(
            ["sub/frame1.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        assert [
            p.name for p in tmp_path.iterdir() if p.name != scripts.QA_MANIFEST_FILENAME
        ] == ["frame1.star"]

    def test_same_basename_different_dirs_mirror_source_tree(
        self, monkeypatch, tmp_path, by_filter
    ):
        """Same-named frames from different dirs are written under mirrored subdirs."""
        monkeypatch.setattr(scripts, "process_one_image", lambda *a, **k: by_filter())

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
        self, monkeypatch, tmp_path, by_filter
    ):
        """Two different source dirs with the same name still mirror distinctly."""
        monkeypatch.setattr(scripts, "process_one_image", lambda *a, **k: by_filter())

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
        self, monkeypatch, tmp_path, by_filter
    ):
        """Two single-dir inputs differing only by extension stay distinct + flat."""
        monkeypatch.setattr(scripts, "process_one_image", lambda *a, **k: by_filter())
        # Both inputs live in the same directory (the cwd), so the layout stays
        # flat; their shared stem "img" is disambiguated with a numeric suffix
        # rather than a leading-underscore or directory prefix.
        monkeypatch.chdir(tmp_path)

        inputs = ["img.fit", "img.fits"]
        scripts.process_batch(
            inputs,
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        written = sorted(
            p.name for p in tmp_path.iterdir() if p.name != scripts.QA_MANIFEST_FILENAME
        )
        assert written == ["img.star", "img_1.star"]

    def test_custom_output_suffix_is_honored(self, monkeypatch, tmp_path, by_filter):
        """An explicit ``output_suffix`` replaces the default ``.star``."""
        monkeypatch.setattr(scripts, "process_one_image", lambda *a, **k: by_filter())

        scripts.process_batch(
            ["frame1.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
            output_suffix=".starlist",
        )

        assert [
            p.name for p in tmp_path.iterdir() if p.name != scripts.QA_MANIFEST_FILENAME
        ] == ["frame1.starlist"]

    def test_written_file_round_trips_through_starlistset(
        self, monkeypatch, tmp_path, by_filter
    ):
        """The file is a valid StarListSet: one StarList per filter, stars intact."""
        filters = ("TR", "TG", "TB")
        monkeypatch.setattr(
            scripts, "process_one_image", lambda *a, **k: by_filter(filters)
        )

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

    def test_disk_mode_returns_path_mapping(self, monkeypatch, tmp_path, by_filter):
        """Disk mode returns each input file mapped to its written output path."""
        monkeypatch.setattr(scripts, "process_one_image", lambda *a, **k: by_filter())

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

    def test_failed_frames_write_no_file(self, monkeypatch, tmp_path, by_filter):
        """A frame whose ``process_one_image`` raises a FrameError writes nothing."""

        def _maybe(file, *_args: object, **_kwargs: object):
            if file == "bad.fits":
                msg = "twirl found no match"
                raise WCSSolveError(msg, file=file)
            return by_filter()

        monkeypatch.setattr(scripts, "process_one_image", _maybe)

        results = scripts.process_batch(
            ["good.fits", "bad.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        assert [
            p.name for p in tmp_path.iterdir() if p.name != scripts.QA_MANIFEST_FILENAME
        ] == ["good.star"]
        assert results == {"good.fits": tmp_path / "good.star"}

    def test_writes_qa_manifest(self, monkeypatch, tmp_path, by_filter):
        """A per-frame QA manifest records ok and skipped frames (#31)."""

        def _maybe(file, *_args: object, **_kwargs: object):
            if file == "bad.fits":
                msg = "twirl found no match"
                raise WCSSolveError(msg, file=file)
            return by_filter()

        monkeypatch.setattr(scripts, "process_one_image", _maybe)

        scripts.process_batch(
            ["good.fits", "bad.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        manifest = tmp_path / "qa_manifest.csv"
        assert manifest.exists()

        with manifest.open(newline="") as f:
            rows = list(csv.DictReader(f))

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

    def test_qa_manifest_sky_median_is_median_of_bkgd_count(
        self, monkeypatch, tmp_path, by_filter
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
        monkeypatch.setattr(scripts, "process_one_image", lambda *a, **k: result)

        scripts.process_batch(
            ["a.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        with (tmp_path / scripts.QA_MANIFEST_FILENAME).open(newline="") as f:
            rows = list(csv.DictReader(f))

        assert float(rows[0]["sky_median"]) == pytest.approx(5.0)

    def test_qa_manifest_sky_median_ignores_nan_bkgd_count(
        self, monkeypatch, tmp_path, by_filter
    ):
        """
        A NaN ``bkgd_count`` (edge-of-frame annulus) stays out of ``sky_median``.

        Per the NaN contract in `bandaid.photometry.measure_photometry`, an
        edge-of-frame or fully-masked annulus yields NaN; one such star must not
        poison the frame's QA value.
        """
        result = by_filter()
        result["TR"]["bkgd_count"] = [np.nan, 7.0]
        monkeypatch.setattr(scripts, "process_one_image", lambda *a, **k: result)

        scripts.process_batch(
            ["a.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        with (tmp_path / scripts.QA_MANIFEST_FILENAME).open(newline="") as f:
            rows = list(csv.DictReader(f))

        assert float(rows[0]["sky_median"]) == pytest.approx(7.0)

    def test_qa_manifest_drift_rejected_counts_flagged_star_that_survives_filtering(
        self, monkeypatch, tmp_path, by_filter
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
        monkeypatch.setattr(scripts, "process_one_image", lambda *a, **k: result)

        scripts.process_batch(
            ["a.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        with (tmp_path / scripts.QA_MANIFEST_FILENAME).open(newline="") as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["n_centroid_drift"] == "1"
        assert rows[0]["n_drift_rejected"] == "1"

    def test_qa_manifest_drift_rejected_excludes_star_failing_flux_cut(
        self, monkeypatch, tmp_path, by_filter
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
        monkeypatch.setattr(scripts, "process_one_image", lambda *a, **k: result)

        scripts.process_batch(
            ["a.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        with (tmp_path / scripts.QA_MANIFEST_FILENAME).open(newline="") as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["n_centroid_drift"] == "1"
        assert rows[0]["n_drift_rejected"] == "0"

    def test_qa_manifest_can_be_disabled(self, monkeypatch, tmp_path, by_filter):
        """``write_qa_manifest=False`` writes only starlists, no manifest."""
        monkeypatch.setattr(scripts, "process_one_image", lambda *a, **k: by_filter())

        scripts.process_batch(
            ["a.fits", "b.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
            write_qa_manifest=False,
        )

        assert not (tmp_path / scripts.QA_MANIFEST_FILENAME).exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["a.star", "b.star"]

    def test_qa_manifest_name_is_honored(self, monkeypatch, tmp_path, by_filter):
        """An explicit ``qa_manifest_name`` overrides the default filename."""
        monkeypatch.setattr(scripts, "process_one_image", lambda *a, **k: by_filter())

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
        self, monkeypatch, tmp_path, by_filter
    ):
        """A custom ``write_frame`` is called once per frame with the rich tables."""
        calls = []

        def spy_writer(frame_result, output_path):
            calls.append((frame_result, output_path))
            output_path.write_text("recorded")
            return output_path

        monkeypatch.setattr(scripts, "process_one_image", lambda *a, **k: by_filter())

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
        self, monkeypatch, tmp_path, by_filter
    ):
        """Whatever ``write_frame`` returns is stored as ``results[file]``."""
        sentinel = tmp_path / "somewhere" / "custom.out"

        def writer(_frame_result, _output_path):
            return sentinel

        monkeypatch.setattr(scripts, "process_one_image", lambda *a, **k: by_filter())

        results = scripts.process_batch(
            ["a.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
            write_frame=writer,
        )

        assert results == {"a.fits": sentinel}

    def test_write_frame_not_called_in_memory_mode(self, monkeypatch, by_filter):
        """In-memory mode ignores ``write_frame`` and returns the tables."""

        def boom(_frame_result, _output_path):
            msg = "write_frame must not run in in-memory mode"
            raise AssertionError(msg)

        monkeypatch.setattr(scripts, "process_one_image", lambda *a, **k: by_filter())

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
        self, monkeypatch, tmp_path, by_filter
    ):
        """With no ``write_frame`` given, the default still writes a StarListSet."""
        monkeypatch.setattr(scripts, "process_one_image", lambda *a, **k: by_filter())

        scripts.process_batch(
            ["frame1.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        # A valid StarListSet document, exactly as before the writer seam existed.
        StarListSet.model_validate_json((tmp_path / "frame1.star").read_text())

    def test_writer_frame_error_skips_frame_not_batch(
        self, monkeypatch, tmp_path, by_filter
    ):
        """A no-usable-stars frame at write time is skipped, not batch-fatal (#78)."""

        def _maybe(file, *_args: object, **_kwargs: object):
            result = by_filter()
            if file == "starless.fits":
                # No row survives good_star_mask (it requires tot_count > 0),
                # so the default writer raises NoUsableStarsError at write time.
                for table in result.values():
                    table["tot_count"] = [-1.0, 0.0]
            return result

        monkeypatch.setattr(scripts, "process_one_image", _maybe)

        inputs = ["starless.fits", "good.fits"]
        results = scripts.process_batch(
            inputs,
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        # The batch survives the starless frame: the later frame is still
        # written, and only it appears in the results.
        assert results == {"good.fits": tmp_path / "good.star"}
        assert sorted(
            p.name for p in tmp_path.iterdir() if p.name != scripts.QA_MANIFEST_FILENAME
        ) == ["good.star"]

        with (tmp_path / scripts.QA_MANIFEST_FILENAME).open(newline="") as f:
            rows = list(csv.DictReader(f))

        # One row per input frame: the starless frame's provisional ok record
        # is replaced by the skip, not duplicated alongside it.
        by_file = {row["file"]: row for row in rows}
        assert len(rows) == len(by_file) == len(inputs)
        assert by_file["starless.fits"]["status"] == "skipped: NoUsableStarsError"
        assert by_file["good.fits"]["status"] == "ok"

    def test_writer_non_frame_error_still_propagates(
        self, monkeypatch, tmp_path, by_filter
    ):
        """A genuine write failure (not a FrameError) still aborts the batch."""

        def denied(_frame_result, _output_path):
            msg = "simulated unwritable output"
            raise PermissionError(msg)

        monkeypatch.setattr(scripts, "process_one_image", lambda *a, **k: by_filter())

        with pytest.raises(PermissionError, match="simulated unwritable output"):
            scripts.process_batch(
                ["a.fits"],
                _dummy_prep(),
                user_specific_metadata={},
                output_dir=tmp_path,
                write_frame=denied,
            )
