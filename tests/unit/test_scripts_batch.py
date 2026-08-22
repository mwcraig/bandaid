"""Unit tests for the per-frame batch loop and batch-to-disk orchestration."""

import csv
import logging
from pathlib import Path

import numpy as np
import pytest
from _helpers import _CONSISTENT_HEADER, _dummy_prep, _make_tan_wcs
from aavso_starlist_schema import StarListSet
from astropy.table import Table

from bandaid import scripts
from bandaid.exceptions import (
    TooFewStarsError,
    WCSSolveError,
)


def _wcs_for_dummy_prep():
    """Return a TAN WCS matching ``_dummy_prep``'s shape and (RA, Dec) center."""
    prep = _dummy_prep()
    return _make_tan_wcs(image_size=prep.shape, crval=prep.center, pixscale=2.4)


def _fake_process_one_image_with_wcs(wcs, calls, *, fails_on=()):
    """
    Build a ``process_one_image`` stub recording its calls and carrying ``wcs``.

    Every call that does not match a file in ``fails_on`` is appended to
    ``calls`` (a list the caller supplies) as ``(args, kwargs)``, and every
    returned table carries ``meta["wcs"] = wcs``.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The WCS to stamp into every returned table's ``meta["wcs"]``.
    calls : list
        List to append each successful call's ``(args, kwargs)`` to, in call
        order.
    fails_on : collections.abc.Container of str, optional
        Filenames for which the stub raises `WCSSolveError` instead of
        returning a result. By default empty (every frame succeeds).

    Returns
    -------
    collections.abc.Callable
        A ``process_one_image``-compatible stub.
    """

    def _fake(file, *args: object, **kwargs: object):
        if file in fails_on:
            msg = "could not solve"
            raise WCSSolveError(msg, file=file)
        calls.append((args, kwargs))
        table = Table({"tot_count": [1.0]})
        table.meta["wcs"] = wcs
        return {"TR": table}

    return _fake


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

    def test_one_result_per_frame_with_shared_prep(self, mocker):
        """Each frame is processed once with the same shared prep objects."""
        prep = _dummy_prep()
        user_meta = {"observer": "abc"}
        process_one_image = mocker.patch(
            "bandaid.scripts.process_one_image",
            return_value={"TR": Table({"tot_count": [1.0]})},
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

    def test_emits_progress_log_per_frame(self, patched_process_one_image, caplog):
        """Each frame logs a ``processing i/N: name`` line at INFO for --verbose."""
        patched_process_one_image({"TR": Table({"tot_count": [1.0]})})

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

    def test_failed_frames_are_skipped(self, mocker):
        """A frame whose ``process_one_image`` raises a FrameError is omitted."""
        prep = _dummy_prep()
        mocker.patch(
            "bandaid.scripts.process_one_image",
            side_effect=_raise_on(
                "bad.fits",
                TooFewStarsError("too few stars", file="bad.fits"),
                lambda: {"TR": Table({"tot_count": [1.0]})},
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

    def test_unexpected_error_skipped_when_not_fail_fast(self, mocker):
        """With fail_fast=False, an unexpected bug is logged and skipped."""
        mocker.patch(
            "bandaid.scripts.process_one_image",
            side_effect=_raise_on(
                "bad.fits",
                RuntimeError("a real bug"),
                lambda: {"TR": Table({"tot_count": [1.0]})},
            ),
        )

        results = scripts.process_batch(
            ["good.fits", "bad.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            fail_fast=False,
        )

        assert list(results) == ["good.fits"]

    def test_loads_each_frame_exactly_once(self, patched_process_one_image, mocker):
        """process_batch loads each frame exactly once via the shared loader (#44)."""
        patched_process_one_image({"TR": Table({"tot_count": [1.0]})})
        load_frame = _stub_load_frame_calls(mocker)

        files = ["a.fits", "b.fits"]
        scripts.process_batch(files, _dummy_prep(), user_specific_metadata={})

        assert [call.args[0] for call in load_frame.call_args_list] == files

    def test_provided_first_frame_is_not_reloaded(
        self, patched_process_one_image, mocker
    ):
        """
        process_batch reuses a caller-provided first-frame load (#44).

        Encodes "exactly once per run" including the first-frame double-open:
        ``photometer_frames`` already opened the first file for ``prepare_batch``
        and hands that load in as ``first_frame=``, so ``process_batch`` must not
        reopen it, while still loading every other frame normally.
        """
        patched_process_one_image({"TR": Table({"tot_count": [1.0]})})
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

    def test_clips_photometry_catalog_after_first_solved_frame(self, mocker):
        """
        Frames after the first successfully solved one get only the in-frame stars.

        `_dummy_prep`'s two photometry_coords sit one in-frame and one off-frame
        against a WCS matching its shape/center: the first frame (which supplies
        the WCS) still gets the full list, and every later frame gets the
        clipped, single-star list (issue #115).
        """
        prep = _dummy_prep()
        wcs = _wcs_for_dummy_prep()
        calls = []
        mocker.patch(
            "bandaid.scripts.process_one_image",
            side_effect=_fake_process_one_image_with_wcs(wcs, calls),
        )

        files = ["a.fits", "b.fits", "c.fits"]
        scripts.process_batch(files, prep, user_specific_metadata={})

        assert len(calls) == len(files)
        _, first_kwargs = calls[0]
        assert len(first_kwargs["input_photometry_coords"]) == len(
            prep.photometry_coords
        )
        for later_args, later_kwargs in calls[1:]:
            assert len(later_kwargs["input_photometry_coords"]) == 1
            # prep.radecs (the WCS-solve reference list) is untouched by the clip.
            assert later_args[1] is prep.radecs

    def test_clip_deferred_until_first_successful_frame(self, mocker):
        """
        "First frame" means the first frame that solves, not literally frame 1.

        When frame 1 raises `WCSSolveError` and is skipped, the clip is deferred
        to the first frame that actually processes cleanly.
        """
        prep = _dummy_prep()
        wcs = _wcs_for_dummy_prep()
        calls = []
        mocker.patch(
            "bandaid.scripts.process_one_image",
            side_effect=_fake_process_one_image_with_wcs(
                wcs, calls, fails_on={"bad.fits"}
            ),
        )

        files = ["bad.fits", "good1.fits", "good2.fits"]
        scripts.process_batch(files, prep, user_specific_metadata={})

        # bad.fits never reaches the stub's calls list (it raises before appending).
        assert len(calls) == len(files) - 1
        _, good1_kwargs = calls[0]
        assert len(good1_kwargs["input_photometry_coords"]) == len(
            prep.photometry_coords
        )
        _, good2_kwargs = calls[1]
        assert len(good2_kwargs["input_photometry_coords"]) == 1

    def test_no_wcs_in_meta_leaves_catalog_unclipped(self, mocker):
        """
        A result with no ``meta["wcs"]`` never triggers the clip.

        Keeps existing stubbed ``process_one_image`` results (which carry no
        WCS) valid: the catalog is passed through unchanged on every frame.
        """
        prep = _dummy_prep()
        calls = []

        def _fake_no_wcs(file, *args: object, **kwargs: object):  # noqa: ARG001
            calls.append(kwargs)
            return {"TR": Table({"tot_count": [1.0]})}

        mocker.patch(
            "bandaid.scripts.process_one_image",
            side_effect=_fake_no_wcs,
        )

        files = ["a.fits", "b.fits", "c.fits"]
        scripts.process_batch(files, prep, user_specific_metadata={})

        assert len(calls) == len(files)
        for kwargs in calls:
            assert len(kwargs["input_photometry_coords"]) == len(prep.photometry_coords)


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
        `_qa_record_ok` -> `good_star_mask` falls back to the 2.0 default; that
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
        """``write_qa_manifest=False`` writes only starlists, no manifest."""
        patched_process_one_image(by_filter())

        scripts.process_batch(
            ["a.fits", "b.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
            write_qa_manifest=False,
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
        self, mocker, tmp_path, by_filter
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

        mocker.patch("bandaid.scripts.process_one_image", side_effect=_maybe)

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
        assert _starlist_names(tmp_path) == ["good.star"]

        rows = _read_manifest(tmp_path)

        # One row per input frame: the starless frame's provisional ok record
        # is replaced by the skip, not duplicated alongside it.
        by_file = {row["file"]: row for row in rows}
        assert len(rows) == len(by_file) == len(inputs)
        assert by_file["starless.fits"]["status"] == "skipped: NoUsableStarsError"
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
