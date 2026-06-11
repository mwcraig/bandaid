"""
Unit tests for the batch photometry driver in :mod:`bandaid.scripts`.

Covers the once-per-batch preparation (``prepare_batch`` building a
``BatchPrep`` from the first frame) and the per-frame loop (``process_batch``),
with the heavy/network dependencies (``calibration_sequence``,
``cached_gaia_radecs``, ``process_one_image``) monkeypatched out.
"""

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
from st_pipeline.schema_definition import StarListSet

from bandaid import scripts
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

    calls = {}

    def fake_calibration_sequence(file):
        calls["calibration_file"] = file
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

        prep = scripts.prepare_batch("frame1.fits", cnn=object(), gaia_mag_limit=12.0)

        np.testing.assert_array_equal(prep.radecs, radecs[:1])

    def test_mag_limit_applied_before_contamination_flagging(self, monkeypatch):
        """
        A too-faint star is cut before it can contamination-flag a neighbor.

        The mag-16 star sits ~1 arcsec from the mag-14 star -- well inside the
        ~7 arcsec the contamination model requires for that pair at this FWHM --
        but it is fainter than the default limit of 15, so it is removed before
        flagging and the mag-14 star survives into ``photometry_coords``. If the
        flagging ran first, the mag-14 star would be dropped too.

        NOTE: this ordering is a known design flaw -- the mag-16 star really is
        on the sky and really does contaminate the mag-14 star, so we arguably
        should flag the mag-14 star. Tracked in
        https://github.com/mwcraig/bandaid/issues/24.
        """
        radecs = np.array([[10.0, 0.0], [10.0 + 1.0 / 3600.0, 0.0], [10.2, 0.0]])
        mags = np.array([14.0, 16.0, 10.0])
        _patch_prep(monkeypatch, radecs_mags=(radecs, mags))

        prep = scripts.prepare_batch("frame1.fits", cnn=object())

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
        """The all-None sentinel from ``calibration_sequence`` raises clearly."""
        # calibration_sequence returns the documented all-None/empty sentinel.
        monkeypatch.setattr(
            scripts,
            "calibration_sequence",
            lambda file: (None, [], None, None, None),
        )
        monkeypatch.setattr(
            scripts,
            "cached_gaia_radecs",
            lambda center, fov: (np.zeros((0, 2)), np.zeros(0)),
        )
        with pytest.raises(ValueError, match="too few stars"):
            scripts.prepare_batch("frame1.fits", cnn=object())


def _dummy_prep():
    """Return a BatchPrep with recognizable sentinel fields for identity checks."""
    return scripts.BatchPrep(
        radecs=np.array([[10.0, 0.0], [10.1, 0.0]]),
        photometry_coords=SkyCoord([10.0, 10.1], [0.0, 0.0], unit="deg"),
        cnn=object(),
        bayer_masks={"TR": np.zeros((2, 2), dtype=bool)},
    )


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
        """A frame whose ``process_one_image`` returns None is omitted."""
        prep = _dummy_prep()

        monkeypatch.setattr(
            scripts,
            "process_one_image",
            lambda file, *a, **k: (
                None if file == "bad.fits" else {"TR": Table({"tot_count": [1.0]})}
            ),
        )

        results = scripts.process_batch(
            ["good.fits", "bad.fits"],
            prep,
            user_specific_metadata={},
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

        written = sorted(p.name for p in tmp_path.iterdir())
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

        assert [p.name for p in tmp_path.iterdir()] == ["frame1.star"]

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

        assert [p.name for p in tmp_path.iterdir()] == ["frame1.starlist"]

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
        """A frame whose ``process_one_image`` returns None writes nothing."""
        monkeypatch.setattr(
            scripts,
            "process_one_image",
            lambda file, *a, **k: None if file == "bad.fits" else by_filter(),
        )

        results = scripts.process_batch(
            ["good.fits", "bad.fits"],
            _dummy_prep(),
            user_specific_metadata={},
            output_dir=tmp_path,
        )

        assert [p.name for p in tmp_path.iterdir()] == ["good.star"]
        assert results == {"good.fits": tmp_path / "good.star"}
