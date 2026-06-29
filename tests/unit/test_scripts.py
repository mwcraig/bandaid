"""
Unit tests for the batch photometry driver in :mod:`bandaid.scripts`.

Covers the once-per-batch preparation (``prepare_batch`` building a
``BatchPrep`` from the first frame) and the per-frame loop (``process_batch``),
with the heavy/network dependencies (``calibration_sequence``,
``cached_gaia_radecs``, ``process_one_image``) monkeypatched out.
"""

import csv
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
from st_pipeline.schema_definition import StarListSet

from bandaid import scripts
from bandaid.config import InstrumentProfile, PhotometryConfig, SourceSelectionConfig
from bandaid.exceptions import (
    BatchPrepError,
    FrameError,
    FrameMetadataError,
    TooFewStarsError,
    WCSSolveError,
)
from bandaid.photometry import neighbor_contamination_flag_sky


def _batch_metadata():
    """Return a metadata dict like the one ``calibration_sequence`` produces."""
    return {
        "ra": 10.0,
        "dec": 0.0,
        "fov_rad": 0.74,
        "pixscale": 2.4,
        "width": 1080,
        "height": 1920,
        "bayerpat": "GRBG",
        "roworder": "top-down",
        "ybayroff": 0,
        "egain": 0.3116,
    }


def _batch_radecs_mags():
    """
    Sky positions + mags with one tight equal-brightness pair to be dropped.

    The first two stars sit ~1 arcsec apart at equal magnitude, so both are
    contaminated; the remaining two are degrees away and survive.
    """
    radecs = np.array(
        [
            [10.0, 0.0],
            [10.0 + 1.0 / 3600.0, 0.0],
            [10.1, 0.0],
            [10.2, 0.0],
        ],
    )
    mags = np.array([12.0, 12.0, 10.0, 11.0])
    return radecs, mags


def _patch_prep(monkeypatch, *, metadata=None, radecs_mags=None, fwhm_pix=2.0):
    """Monkeypatch the heavy prep dependencies and return the spied call args."""
    metadata = metadata if metadata is not None else _batch_metadata()
    radecs, mags = radecs_mags if radecs_mags is not None else _batch_radecs_mags()

    # These tests exercise the mag-cut/contamination plumbing with deliberately
    # tiny synthetic catalogs, so relax the "enough Gaia stars to solve a WCS"
    # floor; the floor itself is covered by TestPrepareBatch's guard tests.
    monkeypatch.setattr(scripts, "N_GAIA_STARS_ALIGN_RETRY", 1)

    calls = {}

    def fake_calibration_sequence(file, *, cnn=None, profile=None, **_kwargs: object):
        calls["calibration_file"] = file
        calls["calibration_cnn"] = cnn
        calls["calibration_profile"] = profile
        return np.zeros((4, 4)), metadata, np.zeros((3, 2)), fwhm_pix, object()

    def fake_cached_gaia_radecs(center, fov):
        calls["center"] = center
        calls["fov"] = fov
        return radecs, mags

    monkeypatch.setattr(scripts, "calibration_sequence", fake_calibration_sequence)
    monkeypatch.setattr(scripts, "cached_gaia_radecs", fake_cached_gaia_radecs)
    return calls, metadata, radecs, mags, fwhm_pix


class TestPrepareBatch:
    """Unit tests for ``prepare_batch``."""

    def test_returns_batchprep_with_expected_fields(self, monkeypatch):
        """The bundle carries the Gaia list, the cnn, and the three CFA masks."""
        _, _, radecs, _, _ = _patch_prep(monkeypatch)
        cnn = object()

        prep = scripts.prepare_batch("frame1.fits", cnn=cnn)

        assert isinstance(prep, scripts.BatchPrep)
        np.testing.assert_array_equal(prep.radecs, radecs)
        assert prep.cnn is cnn
        assert set(prep.bayer_masks) == {"TR", "TB", "TG"}

    def test_first_frame_resolved_with_config_instrument_profile(self, monkeypatch):
        """
        The config's instrument is threaded into the first-frame calibration.

        Without this, ``prepare_batch`` would resolve the first frame's metadata
        with the bundled-Seestar50 fallback rather than ``config.instrument`` --
        wrong for any other telescope.
        """
        calls, *_ = _patch_prep(monkeypatch)
        instrument = InstrumentProfile(name="MyScope")
        scripts.prepare_batch(
            "frame1.fits",
            cnn=object(),
            config=PhotometryConfig(instrument=instrument),
        )
        assert calls["calibration_profile"] is instrument

    def test_append_l4_adds_luminance_channel(self, monkeypatch):
        """``append_l4`` adds the full-frame "L4" channel as a None mask."""
        _patch_prep(monkeypatch)
        prep = scripts.prepare_batch("frame1.fits", cnn=object(), append_l4=True)
        assert set(prep.bayer_masks) == {"TR", "TB", "TG", "L4"}
        assert prep.bayer_masks["L4"] is None

    def test_gaia_queried_at_metadata_center_and_doubled_fov_rad(self, monkeypatch):
        """Gaia is queried at the frame pointing over twice the field radius."""
        calls, metadata, _, _, _ = _patch_prep(monkeypatch)
        scripts.prepare_batch("frame1.fits", cnn=object())

        assert calls["center"] == (metadata["ra"], metadata["dec"])
        # fov_rad is a field *radius*; the query takes the full field (2 * radius).
        assert calls["fov"] == pytest.approx(2 * metadata["fov_rad"])

    def test_contaminated_stars_dropped_from_photometry_coords(self, monkeypatch):
        """The contaminated pair is removed from ``photometry_coords``."""
        _, metadata, radecs, mags, fwhm_pix = _patch_prep(monkeypatch)

        prep = scripts.prepare_batch("frame1.fits", cnn=object())

        fwhm_arcsec = fwhm_pix * metadata["pixscale"]
        flagged = neighbor_contamination_flag_sky(radecs, mags, fwhm_arcsec)
        expected = SkyCoord(radecs[~flagged], unit="deg")

        # The tight equal-mag pair is dropped; the two isolated stars remain.
        assert flagged.tolist() == [True, True, False, False]
        np.testing.assert_allclose(prep.photometry_coords.ra.deg, expected.ra.deg)
        np.testing.assert_allclose(prep.photometry_coords.dec.deg, expected.dec.deg)

    def test_default_gaia_mag_limit_drops_faint_stars(self, monkeypatch):
        """Stars fainter than the default limit of 15 are cut; 15.0 itself is kept."""
        radecs = np.array([[10.0, 0.0], [10.1, 0.0], [10.2, 0.0], [10.3, 0.0]])
        mags = np.array([12.0, 15.0, 15.1, 16.0])
        _patch_prep(monkeypatch, radecs_mags=(radecs, mags))

        prep = scripts.prepare_batch("frame1.fits", cnn=object())

        np.testing.assert_array_equal(prep.radecs, radecs[:2])
        # The kept stars are degrees apart, so none are contamination-flagged.
        np.testing.assert_allclose(prep.photometry_coords.ra.deg, radecs[:2, 0])

    def test_custom_gaia_mag_limit_is_honored(self, monkeypatch):
        """An explicit ``gaia_mag_limit`` cuts at that magnitude instead."""
        radecs = np.array([[10.0, 0.0], [10.1, 0.0], [10.2, 0.0], [10.3, 0.0]])
        mags = np.array([12.0, 15.0, 15.1, 16.0])
        _patch_prep(monkeypatch, radecs_mags=(radecs, mags))

        prep = scripts.prepare_batch(
            "frame1.fits",
            cnn=object(),
            config=PhotometryConfig(
                source_selection=SourceSelectionConfig(gaia_mag_limit=12.0)
            ),
        )

        np.testing.assert_array_equal(prep.radecs, radecs[:1])

    def test_faint_real_star_contaminates_brighter_target(self, monkeypatch):
        """
        A real star fainter than the photometry limit still flags a brighter target.

        The mag-16 star sits ~1 arcsec from the mag-14 star -- well inside the
        ~7 arcsec the contamination model requires for that pair at this FWHM. It
        is fainter than the photometry limit of 15, so it is *not* a photometry
        target, but it is within the default contaminant limit (gaia_mag_limit + 3
        = 18), so it still contaminates the mag-14 target. The mag-14 star is
        therefore flagged and dropped from ``photometry_coords``; only the
        isolated mag-10 star survives. ``radecs`` (the alignment catalog) keeps
        both targets regardless of contamination. Fixes
        https://github.com/mwcraig/bandaid/issues/24.
        """
        radecs = np.array([[10.0, 0.0], [10.0 + 1.0 / 3600.0, 0.0], [10.2, 0.0]])
        mags = np.array([14.0, 16.0, 10.0])
        _patch_prep(monkeypatch, radecs_mags=(radecs, mags))

        prep = scripts.prepare_batch("frame1.fits", cnn=object())

        # Targets (mag <= 15) are the mag-14 and mag-10 stars; both stay in radecs.
        np.testing.assert_array_equal(prep.radecs, radecs[[0, 2]])
        # The mag-14 target is now flagged by the faint mag-16 neighbor, leaving
        # only the far mag-10 star.
        np.testing.assert_allclose(prep.photometry_coords.ra.deg, radecs[[2], 0])

    def test_contaminant_mag_offset_bounds_the_flagging_catalog(self, monkeypatch):
        """
        ``contaminant_mag_offset`` caps which faint stars can flag a target.

        Same close pair as ``test_faint_real_star_contaminates_brighter_target``,
        but a small ``contaminant_mag_offset=0.5`` shrinks the contaminant limit to
        ``gaia_mag_limit + 0.5 = 15.5``, which excludes the mag-16 neighbor from the
        contaminant catalog entirely, so the mag-14 target is no longer flagged and
        survives into ``photometry_coords``.
        """
        radecs = np.array([[10.0, 0.0], [10.0 + 1.0 / 3600.0, 0.0], [10.2, 0.0]])
        mags = np.array([14.0, 16.0, 10.0])
        _patch_prep(monkeypatch, radecs_mags=(radecs, mags))

        prep = scripts.prepare_batch(
            "frame1.fits",
            cnn=object(),
            config=PhotometryConfig(
                source_selection=SourceSelectionConfig(contaminant_mag_offset=0.5),
            ),
        )

        np.testing.assert_array_equal(prep.radecs, radecs[[0, 2]])
        np.testing.assert_allclose(prep.photometry_coords.ra.deg, radecs[[0, 2], 0])

    def test_nan_magnitude_dropped_by_mag_limit(self, monkeypatch):
        """A star with no Gaia magnitude fails the cut and is dropped entirely."""
        radecs = np.array([[10.0, 0.0], [10.1, 0.0], [10.2, 0.0]])
        mags = np.array([12.0, np.nan, 10.0])
        _patch_prep(monkeypatch, radecs_mags=(radecs, mags))

        prep = scripts.prepare_batch("frame1.fits", cnn=object())

        np.testing.assert_array_equal(prep.radecs, radecs[[0, 2]])

    def test_raises_when_too_few_stars_detected(self, monkeypatch):
        """A first-frame TooFewStarsError becomes a fatal BatchPrepError."""

        def _too_few(file, **_kwargs: object):
            msg = "only 1 stars detected"
            raise TooFewStarsError(msg, file=file)

        monkeypatch.setattr(scripts, "calibration_sequence", _too_few)
        with pytest.raises(BatchPrepError, match="too few stars"):
            scripts.prepare_batch("frame1.fits", cnn=object())

    def test_empty_gaia_field_raises_batchpreperror(self, monkeypatch):
        """An empty Gaia cone is fatal -- no reference stars to solve any WCS."""
        _patch_prep(monkeypatch, radecs_mags=(np.empty((0, 2)), np.empty(0)))
        # Use the real floor, not _patch_prep's relaxed one, for the guard.
        monkeypatch.setattr(scripts, "N_GAIA_STARS_ALIGN_RETRY", 20)
        with pytest.raises(BatchPrepError, match="Gaia returned only 0"):
            scripts.prepare_batch("frame1.fits", cnn=object())

    def test_sparse_gaia_field_raises_batchpreperror(self, monkeypatch):
        """Fewer than N_GAIA_STARS_ALIGN_RETRY references is fatal for the batch."""
        radecs = np.column_stack([np.linspace(9.0, 11.0, 5), np.zeros(5)])
        _patch_prep(monkeypatch, radecs_mags=(radecs, np.full(5, 12.0)))
        monkeypatch.setattr(scripts, "N_GAIA_STARS_ALIGN_RETRY", 20)
        with pytest.raises(BatchPrepError, match="Gaia returned only 5"):
            scripts.prepare_batch("frame1.fits", cnn=object())

    def test_gaia_network_error_raises_batchpreperror(self, monkeypatch):
        """A Gaia query failure is surfaced as a fatal BatchPrepError."""
        monkeypatch.setattr(
            scripts,
            "calibration_sequence",
            lambda file, *, cnn=None, **_kwargs: (
                np.zeros((4, 4)),
                _batch_metadata(),
                None,
                2.0,
                object(),
            ),
        )

        def _boom(*_args: object, **_kwargs: object):
            msg = "no network"
            raise ConnectionError(msg)

        monkeypatch.setattr(scripts, "cached_gaia_radecs", _boom)
        with pytest.raises(BatchPrepError, match="could not query Gaia"):
            scripts.prepare_batch("frame1.fits", cnn=object())


class TestCheckFrameConsistency:
    """Unit tests for the per-frame pointing/shape guard."""

    @staticmethod
    def _prep(**overrides: object) -> scripts.BatchPrep:
        """A BatchPrep carrying consistency fields, overridable per test."""
        fields = {"center": (10.0, 0.0), "fov_rad": 0.74, "shape": (1920, 1080)}
        fields.update(overrides)
        return scripts.BatchPrep(
            radecs=np.zeros((1, 2)),
            photometry_coords=SkyCoord([0.0], [0.0], unit="deg"),
            cnn=object(),
            bayer_masks={},
            **fields,
        )

    def test_consistent_frame_passes(self):
        """A frame matching the prep's shape and pointing is accepted."""
        header = {"NAXIS1": 1080, "NAXIS2": 1920, "RA": 10.0, "DEC": 0.0}
        scripts.check_frame_consistency("ok.fits", header, self._prep())

    def test_shape_mismatch_raises_frameerror(self):
        """A different image shape is rejected."""
        header = {"NAXIS1": 1000, "NAXIS2": 1920, "RA": 10.0, "DEC": 0.0}
        with pytest.raises(FrameError, match="shape"):
            scripts.check_frame_consistency("bad.fits", header, self._prep())

    def test_offfield_pointing_raises_frameerror(self):
        """A frame pointing beyond the field radius is rejected."""
        header = {"NAXIS1": 1080, "NAXIS2": 1920, "RA": 12.0, "DEC": 0.0}
        with pytest.raises(FrameError, match="pointing"):
            scripts.check_frame_consistency("bad.fits", header, self._prep())

    def test_missing_keyword_raises_metadata_error(self):
        """A header missing a needed keyword is a metadata error."""
        header = {"NAXIS1": 1080, "RA": 10.0, "DEC": 0.0}  # no NAXIS2
        with pytest.raises(FrameMetadataError):
            scripts.check_frame_consistency("bad.fits", header, self._prep())

    def test_inconsistent_frame_is_skipped_by_batch(self, monkeypatch):
        """process_batch skips an off-field frame and keeps the good one."""
        prep = self._prep()

        def _header(file):
            ra = 10.0 if file == "good.fits" else 50.0
            return {"NAXIS1": 1080, "NAXIS2": 1920, "RA": ra, "DEC": 0.0}

        monkeypatch.setattr(scripts.fits, "getheader", _header)
        monkeypatch.setattr(
            scripts,
            "process_one_image",
            lambda *a, **k: {"TR": Table({"tot_count": [1.0]})},
        )
        results = scripts.process_batch(
            ["good.fits", "bad.fits"],
            prep,
            user_specific_metadata={},
        )
        assert list(results) == ["good.fits"]


def _dummy_prep():
    """Return a BatchPrep with recognizable sentinel fields for identity checks."""
    return scripts.BatchPrep(
        radecs=np.array([[10.0, 0.0], [10.1, 0.0]]),
        photometry_coords=SkyCoord([10.0, 10.1], [0.0, 0.0], unit="deg"),
        cnn=object(),
        bayer_masks={"TR": np.zeros((2, 2), dtype=bool)},
        center=(10.0, 0.0),
        fov_rad=0.74,
        shape=(1920, 1080),
    )


# Header matching _dummy_prep's center/shape, so check_frame_consistency passes.
_CONSISTENT_HEADER = {"NAXIS1": 1080, "NAXIS2": 1920, "RA": 10.0, "DEC": 0.0}


@pytest.fixture
def _consistent_headers(monkeypatch):
    """
    Stub fits.getheader so every frame passes check_frame_consistency.

    process_batch reads each frame's header unconditionally; the process_batch
    tests use fake paths and exercise the processing/output paths, not the
    consistency check, so return a header that matches _dummy_prep for all of them.
    """
    monkeypatch.setattr(
        scripts.fits, "getheader", lambda _file: dict(_CONSISTENT_HEADER)
    )


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


@pytest.fixture
def by_filter(eloy_table, starlist_metadata):
    """
    Factory for a ``{filter: Table}`` photometry result like ``process_one_image``.

    Each filter's table carries two good (finite, positive, in-bounds) rows plus
    the ``meta["fwhm"]`` and ``meta["full_image_meta"]`` that
    ``process_batch`` -> ``eloy_to_starlist`` requires. Both rows survive the
    converter's filtering, so each written StarList has two stars.

    Parameters
    ----------
    eloy_table : callable
        Fixture building an eloy-style photometry table from per-row dicts.
    starlist_metadata : dict
        Fixture providing the StarList metadata stored on each table.

    Returns
    -------
    callable
        ``_make(filters=("TR", "TG"))`` -> the per-filter table mapping.
    """
    rows = [
        {
            "x": 20.0,
            "y": 30.0,
            "ra": 10.0,
            "dec": 20.0,
            "tot_count": 100.0,
            "count_err": 5.0,
            "bkgd_count": 1.0,
            "peak_count": 200.0,
        },
        {
            "x": 70.0,
            "y": 60.0,
            "ra": 11.0,
            "dec": 21.0,
            "tot_count": 300.0,
            "count_err": 7.0,
            "bkgd_count": 1.0,
            "peak_count": 400.0,
        },
    ]

    def _make(filters=("TR", "TG")):
        result = {}
        for filter_name in filters:
            table = eloy_table(rows)
            table.meta["full_image_meta"] = starlist_metadata
            result[filter_name] = table
        return result

    return _make


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
