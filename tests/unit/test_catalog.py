"""
Unit tests for the cached Gaia catalog query in :mod:`bandaid.catalog`.

These tests never touch the network: they replace ``bandaid.catalog.Vizier``
with a fake whose ``query_region`` returns a small in-memory table shaped like
the real VizieR ``I/345/gaia2`` result (columns ``Gmag, RA_ICRS, DE_ICRS,
pmRA, pmDE``). They check the reshaping contract that the notebook relies on
(matching ``twirl.gaia_radecs``), the optional proper-motion propagation via
``SkyCoord.apply_space_motion``, and that the query is issued against the right
catalog with the requested row limit and sort.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

from bandaid import catalog
from bandaid.catalog import GAIA_DR2_EPOCH, GAIA_DR2_VIZIER_CATALOG, cached_gaia_radecs

# Arbitrary row limit used to assert it is forwarded to Vizier unchanged.
ROW_LIMIT_PROBE = 1234
# Floor (deg) the proper-motion shift must exceed; nine years of ~10 mas/yr is
# ~2.5e-5 deg, comfortably above this.
PM_SHIFT_FLOOR_DEG = 1e-6


def _fixture_table():
    """Build a table shaped like the real ``I/345/gaia2`` result, brightest-first."""
    t = Table()
    t["Gmag"] = np.array([8.8, 13.5, 14.1]) * u.mag
    t["RA_ICRS"] = np.array([239.8756, 239.9220, 239.8684]) * u.deg
    t["DE_ICRS"] = np.array([25.9202, 25.8932, 25.8698]) * u.deg
    t["pmRA"] = np.array([-4.220, -0.957, 2.839]) * (u.mas / u.yr)
    t["pmDE"] = np.array([12.364, -1.993, -7.168]) * (u.mas / u.yr)
    return t


class FakeVizier:
    """Stand-in for ``astroquery.vizier.Vizier`` that records its construction/call."""

    last_init = None
    last_call = None

    def __init__(self, columns=None, row_limit=None) -> None:
        FakeVizier.last_init = {"columns": columns, "row_limit": row_limit}
        self.columns = columns
        self.row_limit = row_limit

    def query_region(self, center, radius=None, catalog=None):
        """Record the query parameters and return the fixture table."""
        FakeVizier.last_call = {
            "center": center,
            "radius": radius,
            "catalog": catalog,
        }
        # astroquery returns a TableList (list-like); a plain list is enough here.
        return [_fixture_table()]


@pytest.fixture
def fake_vizier(monkeypatch):
    """Replace the module-level Vizier with the recording fake."""
    FakeVizier.last_init = None
    FakeVizier.last_call = None
    monkeypatch.setattr(catalog, "Vizier", FakeVizier)
    return FakeVizier


@pytest.fixture
def center():
    """A nominal field center near the fixture stars."""
    return SkyCoord(ra=239.9 * u.deg, dec=25.9 * u.deg)


@pytest.mark.usefixtures("fake_vizier")
def test_returns_radecs_and_mags_with_correct_shapes(center):
    """magnitude=True returns (radecs, mags) with twirl's shapes and order."""
    radecs, mags = cached_gaia_radecs(center, 0.2, magnitude=True)
    table = _fixture_table()

    assert radecs.shape == (len(table), 2)
    assert mags.shape == (len(table),)
    # Order is preserved (brightest-first, as returned by VizieR's "+Gmag").
    np.testing.assert_allclose(radecs[:, 0], table["RA_ICRS"].value)
    np.testing.assert_allclose(radecs[:, 1], table["DE_ICRS"].value)
    np.testing.assert_allclose(mags, table["Gmag"].value)


@pytest.mark.usefixtures("fake_vizier")
def test_magnitude_false_returns_only_radecs(center):
    """magnitude=False returns just the (n, 2) radecs array."""
    result = cached_gaia_radecs(center, 0.2, magnitude=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(_fixture_table()), 2)


def test_query_uses_catalog_row_limit_and_brightness_sort(fake_vizier, center):
    """The query targets I/345/gaia2 with the given row limit and a Gmag sort."""
    cached_gaia_radecs(center, 0.2, limit=ROW_LIMIT_PROBE)
    assert fake_vizier.last_call["catalog"] == GAIA_DR2_VIZIER_CATALOG
    assert fake_vizier.last_init["row_limit"] == ROW_LIMIT_PROBE
    # "+Gmag" requests an ascending (brightest-first) sort from VizieR.
    assert "+Gmag" in fake_vizier.last_init["columns"]


def test_radius_is_half_the_min_fov(fake_vizier, center):
    """The cone radius is min(fov)/2, matching twirl (notebook passes 2*fov)."""
    cached_gaia_radecs(center, 0.4)
    radius = fake_vizier.last_call["radius"]
    assert u.Quantity(radius).to_value(u.deg) == pytest.approx(0.2)


@pytest.mark.usefixtures("fake_vizier")
def test_no_proper_motion_by_default(center):
    """With obs_epoch=None, positions are returned at the DR2 epoch unchanged."""
    radecs, _ = cached_gaia_radecs(center, 0.2, magnitude=True)
    table = _fixture_table()
    np.testing.assert_allclose(radecs[:, 0], table["RA_ICRS"].value)
    np.testing.assert_allclose(radecs[:, 1], table["DE_ICRS"].value)


@pytest.mark.usefixtures("fake_vizier")
def test_proper_motion_applied_matches_apply_space_motion(center):
    """obs_epoch propagates positions exactly like SkyCoord.apply_space_motion."""
    obs_epoch = Time("2024-06-01")
    radecs, _ = cached_gaia_radecs(center, 0.2, obs_epoch=obs_epoch)

    table = _fixture_table()
    expected = SkyCoord(
        ra=table["RA_ICRS"],
        dec=table["DE_ICRS"],
        pm_ra_cosdec=table["pmRA"],
        pm_dec=table["pmDE"],
        obstime=Time(GAIA_DR2_EPOCH, format="jyear"),
    ).apply_space_motion(new_obstime=obs_epoch)

    np.testing.assert_allclose(radecs[:, 0], expected.ra.deg, rtol=0, atol=1e-9)
    np.testing.assert_allclose(radecs[:, 1], expected.dec.deg, rtol=0, atol=1e-9)
    # And the propagated positions actually moved off the catalog epoch.
    assert np.abs(radecs[:, 1] - table["DE_ICRS"].value).max() > PM_SHIFT_FLOOR_DEG
