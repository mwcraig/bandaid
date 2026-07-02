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

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from bandaid.catalog import GAIA_DR2_EPOCH, GAIA_DR2_VIZIER_CATALOG, cached_gaia_radecs

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
    expected = SkyCoord(
        ra=gaia_table["RA_ICRS"],
        dec=gaia_table["DE_ICRS"],
        pm_ra_cosdec=gaia_table["pmRA"],
        pm_dec=gaia_table["pmDE"],
        obstime=Time(GAIA_DR2_EPOCH, format="jyear"),
    ).apply_space_motion(new_obstime=obs_epoch)

    np.testing.assert_allclose(radecs[:, 0], expected.ra.deg, rtol=0, atol=1e-9)
    np.testing.assert_allclose(radecs[:, 1], expected.dec.deg, rtol=0, atol=1e-9)
    # And the propagated positions actually moved off the catalog epoch.
    shift = np.abs(radecs[:, 1] - gaia_table["DE_ICRS"].value).max()
    assert shift > PM_SHIFT_FLOOR_DEG


def test_empty_result_returns_shaped_empties(fake_vizier, center):
    """An empty VizieR TableList yields (0, 2) / (0,) arrays, not an IndexError."""
    fake_vizier.return_value.query_region.return_value = []

    radecs, mags = cached_gaia_radecs(center, 0.2, magnitude=True)
    assert radecs.shape == (0, 2)
    assert mags.shape == (0,)
    assert cached_gaia_radecs(center, 0.2, magnitude=False).shape == (0, 2)
