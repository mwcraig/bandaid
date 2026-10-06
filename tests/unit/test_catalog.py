"""
Unit tests for the cached Gaia catalog query in :mod:`bandaid.catalog`.

These tests never touch the network: they patch ``bandaid.catalog.Vizier`` with a
``unittest.mock`` stand-in whose ``query_region`` returns a small in-memory table
shaped like the real VizieR ``I/345/gaia2`` result (columns ``Gmag, RA_ICRS,
DE_ICRS, pmRA, pmDE``) -- the shared ``gaia_table`` / ``fake_vizier`` fixtures
from ``tests/conftest.py``. They check the reshaping contract that the notebook
relies on (matching ``twirl.gaia_radecs``), the optional proper-motion
propagation via ``SkyCoord.apply_space_motion``, the empty-result path, and that
the query is issued against the right catalog with the requested row limit and
sort. Recorded calls are inspected via the mock's ``call_args`` rather than any
side-effect state.
"""

import logging
import math

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from bandaid.catalog import (
    GAIA_DR2_EPOCH,
    GAIA_DR2_VIZIER_CATALOG,
    cached_gaia_radecs,
    query_field_catalog,
)
from bandaid.exceptions import BatchPrepError, CatalogTruncationError

# Arbitrary row limit used to assert it is forwarded to Vizier unchanged.
ROW_LIMIT_PROBE = 1234
# Floor (deg) the proper-motion shift must exceed; nine years of ~10 mas/yr is
# ~2.5e-5 deg, comfortably above this.
PM_SHIFT_FLOOR_DEG = 1e-6


@pytest.fixture
def center():
    """A nominal field center near the fixture stars."""
    return SkyCoord(ra=239.9 * u.deg, dec=25.9 * u.deg)


def test_returns_radecs_and_mags_with_correct_shapes(fake_vizier, center, gaia_table):
    """magnitude=True returns (radecs, mags) with twirl's shapes and order."""
    radecs, mags = cached_gaia_radecs(center, 0.2, magnitude=True)

    fake_vizier.return_value.query_region.assert_called_once()
    assert radecs.shape == (len(gaia_table), 2)
    assert mags.shape == (len(gaia_table),)
    # Order is preserved (brightest-first, as returned by VizieR's "+Gmag").
    np.testing.assert_allclose(radecs[:, 0], gaia_table["RA_ICRS"].value)
    np.testing.assert_allclose(radecs[:, 1], gaia_table["DE_ICRS"].value)
    np.testing.assert_allclose(mags, gaia_table["Gmag"].value)


def test_magnitude_false_returns_only_radecs(fake_vizier, center, gaia_table):
    """magnitude=False returns just the (n, 2) radecs array."""
    result = cached_gaia_radecs(center, 0.2, magnitude=False)

    fake_vizier.return_value.query_region.assert_called_once()
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(gaia_table), 2)


def test_query_uses_catalog_row_limit_and_brightness_sort(fake_vizier, center):
    """The query targets I/345/gaia2 with the given row limit and a Gmag sort."""
    cached_gaia_radecs(center, 0.2, limit=ROW_LIMIT_PROBE)

    init_kwargs = fake_vizier.call_args.kwargs
    query_kwargs = fake_vizier.return_value.query_region.call_args.kwargs
    assert query_kwargs["catalog"] == GAIA_DR2_VIZIER_CATALOG
    assert init_kwargs["row_limit"] == ROW_LIMIT_PROBE
    # "+Gmag" requests an ascending (brightest-first) sort from VizieR.
    assert "+Gmag" in init_kwargs["columns"]


def test_radius_is_half_the_min_fov(fake_vizier, center):
    """The cone radius is min(fov)/2, matching twirl (notebook passes 2*fov)."""
    cached_gaia_radecs(center, 0.4)

    radius = fake_vizier.return_value.query_region.call_args.kwargs["radius"]
    assert u.Quantity(radius).to_value(u.deg) == pytest.approx(0.2)


def test_no_proper_motion_by_default(fake_vizier, center, gaia_table):
    """With obs_epoch=None, positions are returned at the DR2 epoch unchanged."""
    radecs, _ = cached_gaia_radecs(center, 0.2, magnitude=True)

    fake_vizier.return_value.query_region.assert_called_once()
    np.testing.assert_allclose(radecs[:, 0], gaia_table["RA_ICRS"].value)
    np.testing.assert_allclose(radecs[:, 1], gaia_table["DE_ICRS"].value)


def test_proper_motion_applied_matches_apply_space_motion(
    fake_vizier, center, gaia_table
):
    """obs_epoch propagates positions exactly like SkyCoord.apply_space_motion."""
    obs_epoch = Time("2024-06-01")
    radecs, _ = cached_gaia_radecs(center, 0.2, obs_epoch=obs_epoch)

    fake_vizier.return_value.query_region.assert_called_once()
    # The masked (no-PM) fixture star is propagated with zero proper motion.
    expected = SkyCoord(
        ra=gaia_table["RA_ICRS"],
        dec=gaia_table["DE_ICRS"],
        pm_ra_cosdec=gaia_table["pmRA"].filled(0).quantity,
        pm_dec=gaia_table["pmDE"].filled(0).quantity,
        obstime=Time(GAIA_DR2_EPOCH, format="jyear"),
    ).apply_space_motion(new_obstime=obs_epoch)

    np.testing.assert_allclose(radecs[:, 0], expected.ra.deg, rtol=0, atol=1e-9)
    np.testing.assert_allclose(radecs[:, 1], expected.dec.deg, rtol=0, atol=1e-9)
    # And the propagated positions actually moved off the catalog epoch.
    shift = np.abs(radecs[:, 1] - gaia_table["DE_ICRS"].value).max()
    assert shift > PM_SHIFT_FLOOR_DEG


def test_masked_pm_yields_finite_catalog_epoch_positions(
    fake_vizier, center, gaia_table
):
    """
    Masked-PM rows propagate to finite positions at their catalog coordinates.

    On a real VizieR result, ``MaskedColumn.quantity`` converts masked proper
    motions to NaN in a *plain* ``Quantity`` -- ``np.ma.filled`` on it is a
    no-op -- so DR2 sources without PM solutions used to come back with NaN
    positions and crash ``search_around_sky`` downstream in ``prepare_batch``.
    Fixes https://github.com/mwcraig/bandaid/issues/80.
    """
    # Guard the fixture's realism first: it must mimic the astroquery
    # round-trip that triggered the bug (plain NaN-bearing Quantity), and the
    # buggy fill must demonstrably be a no-op on it.
    pm_quantity = gaia_table["pmRA"].quantity
    assert not isinstance(pm_quantity, np.ma.MaskedArray)
    assert np.isnan(pm_quantity.value[2])
    assert np.isnan(np.ma.filled(pm_quantity, 0 * u.mas / u.yr).value[2])

    radecs, mags = cached_gaia_radecs(
        center, 0.2, obs_epoch=Time("2026-04-18T02:39:48")
    )

    fake_vizier.return_value.query_region.assert_called_once()
    assert np.isfinite(radecs).all()
    assert mags.shape == (len(gaia_table),)
    # Zero PM means the masked star stays at its J2015.5 catalog position.
    np.testing.assert_allclose(
        radecs[2],
        [gaia_table["RA_ICRS"][2], gaia_table["DE_ICRS"][2]],
        rtol=0,
        atol=1e-9,
    )


def test_literal_nonfinite_pm_treated_as_zero(fake_vizier, center, gaia_table):
    """
    An unmasked non-finite proper-motion value is neutralized to zero.

    Belt and suspenders for https://github.com/mwcraig/bandaid/issues/80: a
    literal NaN/inf PM (no mask at all) means "no proper motion", so the star
    is propagated with zero PM instead of yielding a NaN position.
    """
    gaia_table["pmRA"] = np.array([np.inf, -0.957, 2.839]) * (u.mas / u.yr)
    gaia_table["pmDE"] = np.array([np.nan, -1.993, -7.168]) * (u.mas / u.yr)

    radecs, _ = cached_gaia_radecs(center, 0.2, obs_epoch=Time("2026-04-18T02:39:48"))

    fake_vizier.return_value.query_region.assert_called_once()
    assert np.isfinite(radecs).all()
    np.testing.assert_allclose(
        radecs[0],
        [gaia_table["RA_ICRS"][0], gaia_table["DE_ICRS"][0]],
        rtol=0,
        atol=1e-9,
    )


def test_empty_result_returns_shaped_empties(fake_vizier, center):
    """An empty VizieR TableList yields (0, 2) / (0,) arrays, not an IndexError."""
    fake_vizier.return_value.query_region.return_value = []

    radecs, mags = cached_gaia_radecs(center, 0.2, magnitude=True)
    assert radecs.shape == (0, 2)
    assert mags.shape == (0,)
    assert cached_gaia_radecs(center, 0.2, magnitude=False).shape == (0, 2)


def test_mag_limit_adds_vizier_column_filter(fake_vizier, center):
    """A mag_limit is forwarded to Vizier as a Gmag column filter."""
    cached_gaia_radecs(center, 0.2, mag_limit=17.5)

    assert fake_vizier.call_args.kwargs["column_filters"] == {"Gmag": "<=17.5"}


def test_no_mag_limit_passes_no_column_filter(fake_vizier, center):
    """Without mag_limit the Vizier call has no column_filters kwarg at all."""
    cached_gaia_radecs(center, 0.2)

    assert "column_filters" not in fake_vizier.call_args.kwargs


# Field geometry and magnitude limits used by the field-catalog tests.
FOV_RAD = 0.74
CONE_MARGIN = 0.4
GAIA_LIMIT = 15.0
CONTAMINANT_LIMIT = 18.0
ROW_LIMIT = 10000
# Equals math.ceil(10000 * ((0.74 + 0.4) / 0.74) ** 2).
SCALED_ROW_LIMIT = 23733


class TestQueryFieldCatalog:
    """``query_field_catalog`` scales the row limit and checks for truncation."""

    @staticmethod
    def _patch(mocker, mags) -> object:
        mags = np.asarray(mags, dtype=float)
        radecs = np.zeros((len(mags), 2))
        return mocker.patch(
            "bandaid.catalog.cached_gaia_radecs", return_value=(radecs, mags)
        )

    @staticmethod
    def _query(center, **kwargs: object) -> tuple:
        defaults = {
            "gaia_mag_limit": GAIA_LIMIT,
            "contaminant_mag_limit": CONTAMINANT_LIMIT,
            "row_limit": ROW_LIMIT,
        }
        defaults.update(kwargs)
        return query_field_catalog(center, FOV_RAD, **defaults)

    def test_zero_margin_keeps_row_limit(self, mocker, center):
        """With no margin the row limit and cone are unchanged."""
        mock = self._patch(mocker, [10.0])
        self._query(center)

        args, kwargs = mock.call_args
        assert args == (center, 2 * FOV_RAD)
        assert kwargs["limit"] == ROW_LIMIT
        assert kwargs["mag_limit"] == CONTAMINANT_LIMIT

    def test_margin_scales_limit_with_cone_area(self, mocker, center):
        """The row limit grows with the cone area; the cone is widened."""
        mock = self._patch(mocker, [10.0])
        self._query(center, cone_margin=CONE_MARGIN)

        args, kwargs = mock.call_args
        assert args[1] == pytest.approx(2 * (FOV_RAD + CONE_MARGIN))
        assert kwargs["limit"] == SCALED_ROW_LIMIT
        assert kwargs["limit"] == math.ceil(ROW_LIMIT * (1.14 / 0.74) ** 2)

    def test_obs_epoch_forwarded(self, mocker, center):
        """obs_epoch is passed through to the cached query."""
        mock = self._patch(mocker, [10.0])
        epoch = Time("2026-01-01")
        self._query(center, obs_epoch=epoch)

        assert mock.call_args.kwargs["obs_epoch"] == epoch

    def test_returns_radecs_and_mags(self, mocker, center):
        """The query result is returned unchanged."""
        self._patch(mocker, [10.0, 11.0])
        radecs, mags = self._query(center)

        assert radecs.shape == (2, 2)
        np.testing.assert_allclose(mags, [10.0, 11.0])

    def test_nonpositive_fov_raises(self, center):
        """fov_rad must be positive."""
        with pytest.raises(ValueError, match="fov_rad"):
            query_field_catalog(
                center,
                0.0,
                gaia_mag_limit=GAIA_LIMIT,
                contaminant_mag_limit=CONTAMINANT_LIMIT,
            )

    def test_truncation_inside_target_range_raises(self, mocker, center):
        """Truncation brighter than the target limit loses targets and is fatal."""
        self._patch(mocker, np.linspace(9.0, GAIA_LIMIT, ROW_LIMIT))
        with pytest.raises(CatalogTruncationError, match="targets") as excinfo:
            self._query(center)

        assert isinstance(excinfo.value, BatchPrepError)

    def test_truncation_in_contaminant_range_warns(self, mocker, center, caplog):
        """Truncation between the two limits only warns."""
        self._patch(mocker, np.linspace(9.0, 16.5, ROW_LIMIT))
        with caplog.at_level(logging.WARNING, logger="bandaid.catalog"):
            self._query(center)

        assert any("incomplete" in r.getMessage() for r in caplog.records)

    def test_truncation_at_contaminant_limit_is_silent(self, mocker, center, caplog):
        """Reaching the contaminant limit exactly at the row limit is fine."""
        self._patch(mocker, np.linspace(9.0, CONTAMINANT_LIMIT, ROW_LIMIT))
        with caplog.at_level(logging.WARNING, logger="bandaid.catalog"):
            self._query(center)

        assert not caplog.records

    def test_below_row_limit_is_silent(self, mocker, center, caplog):
        """No truncation check when fewer rows than the limit came back."""
        self._patch(mocker, np.linspace(9.0, 12.0, ROW_LIMIT - 1))
        with caplog.at_level(logging.WARNING, logger="bandaid.catalog"):
            self._query(center)

        assert not caplog.records

    def test_empty_result_is_silent(self, mocker, center, caplog):
        """An empty catalog neither raises nor warns."""
        self._patch(mocker, [])
        with caplog.at_level(logging.WARNING, logger="bandaid.catalog"):
            _, mags = self._query(center)

        assert len(mags) == 0
        assert not caplog.records
