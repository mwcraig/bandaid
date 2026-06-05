"""
Cached Gaia catalog queries for the bandaid pipeline.

The pipeline needs a magnitude-limited Gaia catalog around each target to align
images and to drive forced photometry. ``twirl.gaia_radecs`` does this by hitting
the Gaia TAP archive on *every* call, which is slow and occasionally times out.

This module provides :func:`cached_gaia_radecs`, a drop-in replacement that queries
the same Gaia DR2 data through VizieR (catalog ``I/345/gaia2``) using
``astroquery.vizier``. astroquery caches VizieR query results automatically (on by
default, one-week timeout, persisted under :attr:`Vizier.cache_location`), so
repeated calls with identical parameters are served from disk with no network
access and no caching code of our own. Inspect the cache with
``Vizier.cache_location`` and clear it with ``Vizier.clear_cache()``.

The return value matches ``twirl.gaia_radecs``: an ``(n, 2)`` array of RA/Dec in
degrees, optionally paired with the Gaia G magnitudes. Unlike twirl's optional
manual proper-motion correction, propagation to an observation epoch is done with
astropy's :meth:`~astropy.coordinates.SkyCoord.apply_space_motion`.
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.vizier import Vizier

# Gaia DR2 in VizieR and the reference epoch (Julian year) of its positions.
GAIA_DR2_VIZIER_CATALOG = "I/345/gaia2"
GAIA_DR2_EPOCH = 2015.5

# VizieR column labels for the I/345/gaia2 table.
_RA_COL = "RA_ICRS"
_DEC_COL = "DE_ICRS"
_PMRA_COL = "pmRA"
_PMDEC_COL = "pmDE"
_MAG_COL = "Gmag"


def _fov_to_radius_deg(fov):
    """
    Convert a field-of-view to a cone-search radius in degrees.

    Mirrors ``twirl.gaia_radecs``: the radius is half of the smaller FOV
    dimension.

    Parameters
    ----------
    fov : float or astropy.units.Quantity
        Field of view. A scalar is interpreted as degrees; a length-2 value is
        treated as ``(ra_fov, dec_fov)``.

    Returns
    -------
    float
        Cone-search radius in degrees.
    """
    if not isinstance(fov, u.Quantity):
        fov = fov * u.deg
    fov = fov.to(u.deg).value
    if np.ndim(fov) == 1:
        ra_fov, dec_fov = fov
    else:
        ra_fov = dec_fov = fov
    return np.min([ra_fov, dec_fov]) / 2


def cached_gaia_radecs(center, fov, *, limit=10000, magnitude=True, obs_epoch=None):
    """
    Return Gaia DR2 RA/Dec (and magnitudes) in a field, cached via VizieR.

    Drop-in replacement for ``twirl.gaia_radecs`` as used by the bandaid
    pipeline. Results are cached automatically by astroquery's VizieR cache, so
    repeated calls with the same parameters do not re-query the server.

    Parameters
    ----------
    center : astropy.coordinates.SkyCoord or tuple
        Center of the field. A tuple is interpreted as ``(ra, dec)`` in degrees.
    fov : float or astropy.units.Quantity
        Field of view. A scalar is interpreted as degrees. The cone-search
        radius is ``min(fov) / 2`` (matching ``twirl.gaia_radecs``).
    limit : int, optional
        Maximum number of (brightest) sources to retrieve. Default 10000.
    magnitude : bool, optional
        If ``True`` (default), also return the Gaia G magnitudes.
    obs_epoch : astropy.time.Time or str, optional
        If given, propagate positions from the Gaia DR2 epoch (2015.5) to this
        epoch using proper motions via
        :meth:`~astropy.coordinates.SkyCoord.apply_space_motion`. If ``None``
        (default), positions are returned at the catalog epoch with no
        proper-motion correction (matching the notebook's current behavior).

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        An ``(n, 2)`` array of RA/Dec in degrees. If ``magnitude`` is ``True``,
        a ``(radecs, mags)`` tuple where ``mags`` is the length-``n`` array of
        Gaia G magnitudes.
    """
    if not isinstance(center, SkyCoord):
        ra, dec = center
        center = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    radius = _fov_to_radius_deg(fov)

    # "+Gmag" asks VizieR to sort ascending (brightest first); combined with
    # row_limit this returns the N brightest sources in the cone, matching
    # twirl's "SELECT top N ... ORDER BY phot_g_mean_mag".
    vizier = Vizier(
        columns=["+" + _MAG_COL, _RA_COL, _DEC_COL, _PMRA_COL, _PMDEC_COL],
        row_limit=limit,
    )
    result = vizier.query_region(
        center, radius=radius * u.deg, catalog=GAIA_DR2_VIZIER_CATALOG
    )
    table = result[0]

    if obs_epoch is None:
        ra = np.asarray(table[_RA_COL].value, dtype=float)
        dec = np.asarray(table[_DEC_COL].value, dtype=float)
    else:
        # Some DR2 sources lack proper motions; treat missing PM as zero so the
        # space-motion calculation does not fail.
        pmra = np.ma.filled(table[_PMRA_COL].quantity, 0 * u.mas / u.yr)
        pmdec = np.ma.filled(table[_PMDEC_COL].quantity, 0 * u.mas / u.yr)
        coords = SkyCoord(
            ra=table[_RA_COL].quantity,
            dec=table[_DEC_COL].quantity,
            pm_ra_cosdec=pmra,
            pm_dec=pmdec,
            obstime=Time(GAIA_DR2_EPOCH, format="jyear"),
        ).apply_space_motion(new_obstime=Time(obs_epoch))
        ra = coords.ra.deg
        dec = coords.dec.deg

    radecs = np.array([ra, dec]).T

    if magnitude:
        mags = np.asarray(table[_MAG_COL].value, dtype=float)
        return radecs, mags
    return radecs
