"""Unit tests for ``build_photometry_table`` and ``calculate_l4_quantities``."""

import warnings

import astropy.units as u
import numpy as np
import pytest
from _helpers import (
    _bright_neighbor_scene,
    _fake_phot_factory,
    _make_image_data,
    _make_tan_wcs,
    _peak_scene_photometry,
    _single_source_photometry_inputs,
)
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from bandaid.config import InstrumentProfile
from bandaid.image2sl_qt import generate_bayer_masks
from bandaid.photometry import (
    ANNULUS,
    RELATIVE_RADII,
    ImageData,
    build_photometry_table,
    calculate_l4_quantities,
    metadata_from_header,
)


class TestBuildPhotometryTable:
    def test_uses_photometry_coords_when_provided(self, monkeypatch):
        """RA/Dec come straight from photometry_coords, not the WCS round-trip."""
        n_stars = 3
        monkeypatch.setattr(
            "bandaid.photometry.measure_photometry",
            _fake_phot_factory(n_stars),
        )
        wcs = _make_tan_wcs()
        # Centroids near the image center; their WCS sky positions are close to
        # the WCS crval (10, 20) -- deliberately NOT the photometry_coords below.
        centroid_coords = np.array([[245.0, 250.0], [255.0, 260.0], [250.0, 240.0]])
        photometry_coords = SkyCoord(
            ra=[100.0, 150.0, 200.0] * u.deg,
            dec=[-30.0, -10.0, 5.0] * u.deg,
        )
        img = _make_image_data(wcs, centroid_coords, photometry_coords)

        table = build_photometry_table(img, mask=None)

        # ra/dec equal the supplied sky coordinates exactly...
        np.testing.assert_allclose(table["ra"], photometry_coords.ra.degree)
        np.testing.assert_allclose(table["dec"], photometry_coords.dec.degree)

        # ...and are NOT what the WCS round-trip would have produced.
        wcs_radec = wcs.pixel_to_world(centroid_coords[..., 0], centroid_coords[..., 1])
        assert not np.allclose(table["ra"], wcs_radec.ra.degree)
        assert not np.allclose(table["dec"], wcs_radec.dec.degree)

        # Rows line up with the per-star x/y (centroid) columns.
        assert len(table) == n_stars
        assert len(table["ra"]) == len(table["x"])

    def test_airmass_comes_from_resolved_metadata(self, monkeypatch):
        """The airmass column uses img.metadata, not a raw header re-read (#59)."""
        n_stars = 3
        monkeypatch.setattr(
            "bandaid.photometry.measure_photometry",
            _fake_phot_factory(n_stars),
        )
        wcs = _make_tan_wcs()
        centroid_coords = np.array([[245.0, 250.0], [255.0, 260.0], [250.0, 240.0]])
        img = _make_image_data(wcs, centroid_coords, input_photometry_coords=None)

        table = build_photometry_table(img, mask=None)

        # _make_image_data plants conflicting values: metadata airmass 1.2 vs a
        # raw-header AIRMASS of 9.9. Only the resolved metadata may win.
        np.testing.assert_allclose(table["airmass"], 1.2)

    def test_falls_back_to_wcs_when_no_photometry_coords(self, monkeypatch):
        """With no photometry_coords, RA/Dec are derived from the image WCS."""
        n_stars = 3
        monkeypatch.setattr(
            "bandaid.photometry.measure_photometry",
            _fake_phot_factory(n_stars),
        )
        wcs = _make_tan_wcs()
        centroid_coords = np.array([[245.0, 250.0], [255.0, 260.0], [250.0, 240.0]])
        img = _make_image_data(wcs, centroid_coords, input_photometry_coords=None)

        table = build_photometry_table(img, mask=None)

        wcs_radec = wcs.pixel_to_world(centroid_coords[..., 0], centroid_coords[..., 1])
        np.testing.assert_allclose(table["ra"], wcs_radec.ra.degree)
        np.testing.assert_allclose(table["dec"], wcs_radec.dec.degree)

    def test_overrides_reach_photometry_and_show_in_output(self, make_test_image):
        """Custom radii/annulus flow through to the real photometry output."""
        # Drive build_photometry_table end-to-end on a real single-source image
        # (no monkeypatching): the overrides only reach the output table meta if
        # build_photometry_table actually forwarded them to measure_photometry.
        radii = [1.0, 2.0]
        annulus = (6, 10)
        image, coords, fwhm, _ = _single_source_photometry_inputs(
            make_test_image,
            annulus=annulus,
        )
        img = _make_image_data(_make_tan_wcs(image.shape), coords, None)
        img.calibrated_data = image

        table = build_photometry_table(
            img,
            mask=None,
            radii=radii,
            annulus=annulus,
        )

        # One flux column per requested radius, and the meta echoes the overrides.
        assert table["fluxes"].shape == (len(coords), len(radii))
        assert table.meta["aperture_radii"] == pytest.approx(radii[0] * fwhm)
        max_aper = max(radii) * fwhm
        expected_annulus = (max(max_aper, annulus[0] * fwhm), annulus[1] * fwhm)
        assert table.meta["annulus_radii"] == pytest.approx(expected_annulus)

    def test_defaults_show_module_constants_in_output(self, make_test_image):
        """With no overrides, the output reflects the module-level constants."""
        image, coords, fwhm, _ = _single_source_photometry_inputs(make_test_image)
        img = _make_image_data(_make_tan_wcs(image.shape), coords, None)
        img.calibrated_data = image

        table = build_photometry_table(img, mask=None)

        assert table.meta["aperture_radii"] == pytest.approx(RELATIVE_RADII[0] * fwhm)
        max_aper = np.max(RELATIVE_RADII) * fwhm
        expected_annulus = (max(max_aper, ANNULUS[0] * fwhm), ANNULUS[1] * fwhm)
        assert table.meta["annulus_radii"] == pytest.approx(expected_annulus)

    def test_centroid_drift_column(self, monkeypatch):
        """Output has a bool ``centroid_drift`` column reflecting per-star drift."""
        n_stars = 3
        monkeypatch.setattr(
            "bandaid.photometry.measure_photometry",
            _fake_phot_factory(n_stars),
        )
        wcs = _make_tan_wcs()
        # fwhm in _make_image_data is 2.3, so max_allowed = min(1.0 * 2.3, 4.0)
        # = 2.3 px. Build aligned vs centroid pairs straddling that threshold.
        aligned_coords = np.array([[250.0, 250.0], [250.0, 250.0], [250.0, 250.0]])
        centroid_coords = np.array(
            [
                [250.0, 250.0],  # zero drift -> not flagged
                [250.0 + 1.0, 250.0],  # 1 px drift -> not flagged
                [250.0 + 5.0, 250.0],  # 5 px drift (> 2.3 and > cap) -> flagged
            ],
        )
        img = _make_image_data(
            wcs,
            centroid_coords,
            input_photometry_coords=None,
            aligned_coords=aligned_coords,
        )

        table = build_photometry_table(img, mask=None)

        assert "centroid_drift" in table.colnames
        assert table["centroid_drift"].dtype == bool
        assert len(table["centroid_drift"]) == n_stars
        np.testing.assert_array_equal(
            table["centroid_drift"],
            [False, False, True],
        )

    # --- Regression tests for issue #57: ``time`` is the mid-exposure JD. ---

    def _time_column(self, monkeypatch, metadata):
        """Build a one-star table overlaying the given metadata; return ``time``."""
        monkeypatch.setattr(
            "bandaid.photometry.measure_photometry",
            _fake_phot_factory(1),
        )
        img = _make_image_data(_make_tan_wcs(), np.array([[250.0, 250.0]]), None)
        # Overlay rather than replace, keeping _make_image_data's baseline
        # obs_time/airmass so the table can always be built.
        img.metadata = {**img.metadata, **metadata}
        table = build_photometry_table(img, mask=None)
        return float(table["time"][0])

    def test_time_is_mid_exposure_for_stacked_frames(self, monkeypatch):
        """``time`` is the obs_time start plus half the effective stack exposure."""
        # 60 subs of 10 s: mid-exposure is 300 s after the obs_time start
        # ("2020-01-01T00:00:00" in _make_image_data's metadata). Recording the
        # start would be wrong by 5 minutes for this stack (issue #57).
        exposure = 10.0
        stack = 60
        start_jd = Time("2020-01-01T00:00:00").jd

        time = self._time_column(
            monkeypatch,
            {"egain": 1.0, "exposure": exposure, "stack": stack},
        )

        expected = start_jd + exposure * stack / 2.0 / 86400.0
        assert time == pytest.approx(expected, abs=1e-9)
        # And it is distinguishable from the old start-time convention.
        assert time != pytest.approx(start_jd, abs=1.0 / 86400.0)

    def test_time_mid_exposure_defaults_to_single_sub(self, monkeypatch):
        """Without a ``stack`` count, ``time`` shifts by half one exposure."""
        exposure = 10.0
        start_jd = Time("2020-01-01T00:00:00").jd

        time = self._time_column(monkeypatch, {"egain": 1.0, "exposure": exposure})

        assert time == pytest.approx(start_jd + exposure / 2.0 / 86400.0, abs=1e-9)

    def test_time_falls_back_to_start_when_exposure_unknown(self, monkeypatch):
        """With no exposure metadata the obs_time start is recorded."""
        start_jd = Time("2020-01-01T00:00:00").jd

        time = self._time_column(monkeypatch, {"egain": 1.0})

        assert time == pytest.approx(start_jd, abs=1e-9)

    def test_time_comes_from_resolved_obs_time(self, monkeypatch):
        """``time`` uses the header_map-resolved obs_time, not raw DATE-OBS (#59)."""
        # A dialect whose observation time lives under DATE-LOC; the raw header
        # in _make_image_data still carries a (different) DATE-OBS, which must
        # not be consulted.
        custom_map = {"obs_time": "@DATE-LOC", "egain": 1.0, "airmass": 1.5}
        profile = InstrumentProfile(name="Renamed", header_map=custom_map)
        header = fits.Header()
        header["NAXIS1"] = 50
        header["NAXIS2"] = 50
        header["DATE-LOC"] = "2021-05-05T05:00:00"
        metadata = metadata_from_header(header, profile=profile)

        time = self._time_column(monkeypatch, metadata)

        assert time == pytest.approx(Time("2021-05-05T05:00:00").jd, abs=1e-9)

    # --- Regression tests for issue #52: the broken ``sky`` column is gone; ---
    # --- ``bkgd_count`` is the correct per-star per-pixel background.       ---

    @staticmethod
    def _uniform_frame_image(data, coords) -> ImageData:
        """Wrap a plain background frame + pixel coords in an ImageData."""
        img = _make_image_data(_make_tan_wcs(data.shape), coords, None)
        img.calibrated_data = data
        return img

    @pytest.mark.parametrize("channel", ["TR", "TG", "TB", "L4"])
    def test_bkgd_count_is_true_per_pixel_sky_on_every_channel(self, channel):
        """
        A uniform frame reports the true per-pixel sky on TR/TG/TB/L4 (#52).

        The removed ``sky`` column under-reported the per-pixel sky by the Bayer
        fill factor (~2.7x/5.0x/2.3x low on TR/TG/TB); ``bkgd_count`` -- the
        per-star sigma-clipped annulus background -- is the correct value on
        every channel, and there must be no ``sky`` column left to mislead.
        """
        true_sky = 10.0
        shape = (256, 256)
        masks = generate_bayer_masks(
            shape,
            {"bayerpat": "RGGB", "roworder": "top-down", "ybayroff": 0},
            append_l4=True,
        )
        coords = np.array([[100.0, 100.0], [150.0, 120.0], [90.0, 160.0]])
        img = self._uniform_frame_image(np.full(shape, true_sky), coords)

        table = build_photometry_table(img, masks[channel])

        assert "sky" not in table.colnames
        np.testing.assert_allclose(
            np.asarray(table["bkgd_count"]),
            true_sky,
            rtol=0.05,
        )

    def test_bkgd_count_is_per_star_not_a_frame_wide_scalar(self):
        """
        Stars on either side of a sky step report their own local background.

        The removed ``sky`` column collapsed the whole frame to one scalar; the
        surviving ``bkgd_count`` must track each star's local annulus (#52).
        """
        true_sky = 10.0
        shape = (256, 256)
        # Left half sky=10, right half sky=40; one star centered in each half.
        data = np.full(shape, true_sky)
        data[:, 128:] = 4 * true_sky
        coords = np.array([[60.0, 128.0], [200.0, 128.0]])  # (x, y)
        img = self._uniform_frame_image(data, coords)

        table = build_photometry_table(img, mask=None)

        assert "sky" not in table.colnames
        bkgd = np.asarray(table["bkgd_count"])
        assert bkgd[0] != bkgd[1]
        np.testing.assert_allclose(bkgd, [true_sky, 4 * true_sky], rtol=0.05)


class TestCalculateL4Quantities:
    """Unit tests for the RGB->L4 combination ``calculate_l4_quantities``."""

    def test_missing_channel_raises(self):
        """A missing TR/TG/TB channel raises an actionable ValueError."""
        by_filter = {"TR": Table(), "TG": Table()}
        with pytest.raises(ValueError, match=r"\['TB'\]"):
            calculate_l4_quantities(Table(), by_filter, 0.5)

    def test_combines_rgb_filters(self):
        """L4 columns are the documented combinations of the TR/TG/TB tables."""
        egain = 0.5

        def _filter_table(tot, area, bkgd, bkgd_std, peak):
            t = Table()
            t["tot_count"] = np.array(tot, dtype=float)
            t["aperture_area"] = np.array(area, dtype=float)
            t["bkgd_count"] = np.array(bkgd, dtype=float)
            t["bkgd_std"] = np.array(bkgd_std, dtype=float)
            t["peak_count"] = np.array(peak, dtype=float)
            return t

        by_filter = {
            "TR": _filter_table([100, 200], [10, 12], [5, 6], [2, 3], [50, 90]),
            "TG": _filter_table([110, 210], [11, 13], [4, 7], [1, 2], [70, 80]),
            "TB": _filter_table([120, 220], [9, 14], [6, 5], [3, 1], [60, 95]),
        }
        final_data = Table()

        calculate_l4_quantities(final_data, by_filter, egain)

        tr, tg, tb = by_filter["TR"], by_filter["TG"], by_filter["TB"]
        expected_tot = tr["tot_count"] + tg["tot_count"] + tb["tot_count"]
        expected_area = tr["aperture_area"] + tg["aperture_area"] + tb["aperture_area"]
        expected_bkgd = (
            tr["bkgd_count"] * tr["aperture_area"]
            + tg["bkgd_count"] * tg["aperture_area"]
            + tb["bkgd_count"] * tb["aperture_area"]
        ) / expected_area
        expected_peak = np.max(
            [tr["peak_count"], tg["peak_count"], tb["peak_count"]],
            axis=0,
        )
        expected_err = np.sqrt(
            (
                tr["bkgd_std"] ** 2 * tr["aperture_area"]
                + tg["bkgd_std"] ** 2 * tg["aperture_area"]
                + tb["bkgd_std"] ** 2 * tb["aperture_area"]
            )
            + expected_tot / egain
        )
        expected_snr = expected_tot / expected_err

        np.testing.assert_allclose(final_data["tot_count"], expected_tot)
        np.testing.assert_allclose(final_data["aperture_area"], expected_area)
        np.testing.assert_allclose(final_data["bkgd_count"], expected_bkgd)
        np.testing.assert_allclose(final_data["peak_count"], expected_peak)
        np.testing.assert_allclose(final_data["count_err"], expected_err)
        np.testing.assert_allclose(final_data["snr"], expected_snr)

    def test_peak_count_is_a_real_cross_channel_max(self, make_test_image):
        """
        The L4 peak max varies with genuinely different channel peaks (#54).

        Before issue #54 the channel mask was never applied to the peak
        cutout, so the TR/TG/TB ``peak_count`` inputs were bit-identical and
        the "max across channels" was a no-op. Measured end-to-end through
        ``measure_photometry`` per channel, the inputs must now differ for
        every star and the L4 value must be their elementwise maximum.
        """
        image, coords = _bright_neighbor_scene(make_test_image)
        masks = generate_bayer_masks(
            image.shape,
            {"bayerpat": "RGGB", "roworder": "top-down", "ybayroff": 0},
            append_l4=False,
        )

        by_filter = {}
        for name, mask in masks.items():
            phot = _peak_scene_photometry(image, coords, mask)
            t = Table()
            for col in (
                "tot_count",
                "aperture_area",
                "bkgd_count",
                "bkgd_std",
                "peak_count",
            ):
                t[col] = phot[col]
            by_filter[name] = t

        final_data = Table()
        calculate_l4_quantities(final_data, by_filter, egain=1.0)

        channel_peaks = np.array(
            [by_filter[c]["peak_count"] for c in ("TR", "TG", "TB")],
        )
        # The per-channel inputs genuinely differ for every star ...
        assert np.all(np.ptp(channel_peaks, axis=0) > 0)
        # ... so the cross-channel max is a real max, not a no-op.
        np.testing.assert_array_equal(
            np.asarray(final_data["peak_count"]),
            channel_peaks.max(axis=0),
        )

    def test_drops_stale_full_frame_columns(self):
        """
        fluxes/total_bkg/bkgd_std are not recombined, so they are dropped.

        ``final_data`` arrives from a full-frame photometry pass carrying these
        columns at their discarded full-frame values. ``calculate_l4_quantities``
        overwrites the columns it can recombine (tot_count, count_err, snr, ...)
        but leaves these three with no L4-consistent meaning, so they must be
        removed rather than left stale (issue #21).
        """
        egain = 0.5

        def _filter_table(tot, area, bkgd, bkgd_std, peak):
            t = Table()
            t["tot_count"] = np.array(tot, dtype=float)
            t["aperture_area"] = np.array(area, dtype=float)
            t["bkgd_count"] = np.array(bkgd, dtype=float)
            t["bkgd_std"] = np.array(bkgd_std, dtype=float)
            t["peak_count"] = np.array(peak, dtype=float)
            return t

        by_filter = {
            "TR": _filter_table([100, 200], [10, 12], [5, 6], [2, 3], [50, 90]),
            "TG": _filter_table([110, 210], [11, 13], [4, 7], [1, 2], [70, 80]),
            "TB": _filter_table([120, 220], [9, 14], [6, 5], [3, 1], [60, 95]),
        }

        # Seed the stale full-frame columns the L4 table really arrives with.
        final_data = Table()
        final_data["fluxes"] = np.array([1.0, 2.0])
        final_data["total_bkg"] = np.array([3.0, 4.0])
        final_data["bkgd_std"] = np.array([5.0, 6.0])

        calculate_l4_quantities(final_data, by_filter, egain)

        for stale in ("fluxes", "total_bkg", "bkgd_std"):
            assert stale not in final_data.colnames

    def test_zero_denominators_do_not_warn(self):
        """
        Zero aperture area / error make the L4 divisions NaN without warning.

        Dropped or blocked stars can sum to zero aperture area and zero error,
        so the ``bkgd_count`` (``.../aperture_area``) and ``snr``
        (``tot_count/count_err``) divisions legitimately hit ``0/0``. The NaN is
        an expected intermediate filtered downstream, so the function must not
        spray ``RuntimeWarning: invalid value encountered in divide``.
        """
        egain = 0.5

        def _filter_table(tot, area, bkgd, bkgd_std, peak):
            t = Table()
            t["tot_count"] = np.array(tot, dtype=float)
            t["aperture_area"] = np.array(area, dtype=float)
            t["bkgd_count"] = np.array(bkgd, dtype=float)
            t["bkgd_std"] = np.array(bkgd_std, dtype=float)
            t["peak_count"] = np.array(peak, dtype=float)
            return t

        # Every channel of the single star has zero aperture area and zero
        # error: aperture_area sums to 0 (bkgd_count = 0/0) and count_err is
        # sqrt(0) = 0 (snr = 0/0).
        by_filter = {
            "TR": _filter_table([0.0], [0.0], [5.0], [0.0], [0.0]),
            "TG": _filter_table([0.0], [0.0], [4.0], [0.0], [0.0]),
            "TB": _filter_table([0.0], [0.0], [6.0], [0.0], [0.0]),
        }
        final_data = Table()

        with warnings.catch_warnings():
            # Promote RuntimeWarning specifically to an error so the test fails
            # if the function emits it; other warning categories are left untouched.
            warnings.simplefilter("error", RuntimeWarning)
            calculate_l4_quantities(final_data, by_filter, egain)

        assert np.isnan(final_data["bkgd_count"][0])
        assert np.isnan(final_data["snr"][0])

    def test_recombined_l4_table_carries_no_sky_column(self, monkeypatch):
        """
        The recombined L4 table has no ``sky`` column at all (#52).

        The L4 table starts as a full-frame ``build_photometry_table`` pass, and
        ``calculate_l4_quantities`` never stripped ``sky`` (unlike the stale
        fluxes/total_bkg/bkgd_std), so the recombined table used to carry a
        stale full-frame ``sky`` value. With the column deleted at the source,
        it must not appear anywhere in the L4 output.
        """
        n_stars = 2
        monkeypatch.setattr(
            "bandaid.photometry.measure_photometry",
            _fake_phot_factory(n_stars),
        )
        coords = np.array([[245.0, 250.0], [255.0, 260.0]])
        # The real L4 input: a full-frame (mask=None) photometry table.
        final_data = build_photometry_table(
            _make_image_data(_make_tan_wcs(), coords, None),
            mask=None,
        )

        def _filter_table(tot, area, bkgd, bkgd_std, peak):
            t = Table()
            t["tot_count"] = np.array(tot, dtype=float)
            t["aperture_area"] = np.array(area, dtype=float)
            t["bkgd_count"] = np.array(bkgd, dtype=float)
            t["bkgd_std"] = np.array(bkgd_std, dtype=float)
            t["peak_count"] = np.array(peak, dtype=float)
            return t

        by_filter = {
            "TR": _filter_table([100, 200], [10, 12], [5, 6], [2, 3], [50, 90]),
            "TG": _filter_table([110, 210], [11, 13], [4, 7], [1, 2], [70, 80]),
            "TB": _filter_table([120, 220], [9, 14], [6, 5], [3, 1], [60, 95]),
        }

        calculate_l4_quantities(final_data, by_filter, egain=0.5)

        assert "sky" not in final_data.colnames
