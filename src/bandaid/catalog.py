"""
Cached Gaia catalog queries for the bandaid pipeline.

The pipeline needs a magnitude-limited Gaia catalog around each target to align
images and to drive forced photometry. ``twirl.gaia_radecs`` does this by hitting
the Gaia TAP archive on *every* call, which is slow and occasionally times out.

This module provides :func:`cached_gaia_radecs`, a drop-in replacement that queries
Gaia DR3 data through VizieR (catalog ``I/355/gaiadr3``) using
``astroquery.vizier``. astroquery caches VizieR query results automatically (on by
default, one-week timeout, persisted under :attr:`Vizier.cache_location`), so
repeated calls with identical parameters are served from disk with no network
access and no caching code of our own. Inspect the cache with
``Vizier.cache_location`` and clear it with ``Vizier.clear_cache()``.

When a server times out or answers with an error, the query moves on to the
next server in :data:`VIZIER_SERVERS`, and the failed answer is removed from
astroquery's cache so a retry asks the server again.

The return value matches ``twirl.gaia_radecs``: an ``(n, 2)`` array of RA/Dec in
degrees, optionally paired with the Gaia G magnitudes. Unlike twirl's optional
manual proper-motion correction, propagation to an observation epoch is done with
astropy's :meth:`~astropy.coordinates.SkyCoord.apply_space_motion`.
"""

import contextlib
import logging
import math
import warnings
from functools import partial
from io import BytesIO

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import votable
from astropy.table import MaskedColumn
from astropy.time import Time
from astroquery.vizier import Vizier
from erfa import ErfaWarning

from bandaid.exceptions import CatalogTruncationError, GaiaQueryError

logger = logging.getLogger(__name__)

# Gaia DR3 in VizieR and the reference epoch (Julian year) of its positions.
GAIA_DR3_VIZIER_CATALOG = "I/355/gaiadr3"
GAIA_DR3_EPOCH = 2016.0

# VizieR servers tried in order: CDS, then the two mirrors that answer Gaia
# DR3 cones themselves and send CORS headers (the others hang or are gone).
VIZIER_SERVERS = (
    "vizier.cds.unistra.fr",
    "vizier.cfa.harvard.edu",
    "vizier.idia.ac.za",
)
# Seconds to wait to connect to a server, and for each read from it.
SERVER_TIMEOUT = 30

# VizieR column labels for the I/355/gaiadr3 table.
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


def _pm_or_zero(column):
    """
    Return a proper-motion column as a Quantity with missing values as zero.

    Filling must happen on the *column*: on a real VizieR result the PM columns
    are ``MaskedColumn`` and their ``.quantity`` converts masked entries to NaN
    in a plain `~astropy.units.Quantity` -- not a ``numpy.ma.MaskedArray`` --
    so ``np.ma.filled`` on it is a no-op and the NaNs propagate into the
    computed positions (https://github.com/mwcraig/bandaid/issues/80). Literal
    non-finite values are neutralized too: a non-finite proper motion means
    "no proper motion".

    Parameters
    ----------
    column : astropy.table.Column or astropy.table.MaskedColumn
        The ``pmRA``/``pmDE`` column from the VizieR table.

    Returns
    -------
    astropy.units.Quantity
        The proper motions, with masked or non-finite entries replaced by zero.
    """
    if isinstance(column, MaskedColumn):
        column = column.filled(0)
    values = np.asarray(column, dtype=float)
    return np.where(np.isfinite(values), values, 0.0) * column.unit


def _read_votable(server, response):
    """
    Return the first non-empty table in a server's VOTable response.

    Parameters
    ----------
    server : str
        The server that sent the response, for the error message.
    response : requests.Response
        The server's response.

    Returns
    -------
    astropy.table.Table or None
        The first table with rows, or ``None`` when the response holds no rows
        and no error, which is how a healthy VizieR reports an empty field.

    Raises
    ------
    GaiaQueryError
        If the body is not a VOTable, or holds no rows and an error ``INFO``
        element; the message carries the server's error text.

    Notes
    -----
    The body is read before the HTTP status so that an error document is
    reported with the server's own text whatever the status; a failure status
    without an error document raises ``requests.HTTPError``. VizieR answers an
    empty cone with no ``TABLE`` at all, so a missing table alone does not
    mean failure; the ``INFO`` element named ``Error`` does.
    """
    try:
        tree = votable.parse(BytesIO(response.content), verify="ignore")
    except Exception as exc:
        response.raise_for_status()
        msg = f"{server} answered with something other than a VOTable"
        raise GaiaQueryError(msg) from exc
    for table in tree.iter_tables():
        if len(table.array):
            return table.to_table(use_names_over_ids=True)
    errors = [info for info in tree.iter_info() if info.name == "Error"]
    if errors:
        text = (errors[0].content or errors[0].value or "").strip()
        msg = f"{server} answered with an error: {text}"
        raise GaiaQueryError(msg)
    response.raise_for_status()
    return None


# pydoclint's DOC501/DOC503 are silenced on the docstring because the bare
# raise only re-raises a server failure after evicting its cache entry.
def _query_vizier(server, center, radius, *, limit, mag_limit):
    """
    Run the Gaia cone search on one VizieR server.

    Parameters
    ----------
    server : str
        VizieR host name.
    center : astropy.coordinates.SkyCoord
        Cone center.
    radius : float
        Cone radius in degrees.
    limit : int
        Maximum number of (brightest) sources to retrieve.
    mag_limit : float or None
        Faintest G magnitude to return, or ``None`` for no cut.

    Returns
    -------
    astropy.table.Table or None
        The sources, brightest first, or ``None`` for an empty field.

    Notes
    -----
    astroquery caches every response it receives, including an error answer,
    so a failed response is removed from the cache before the error is
    re-raised; otherwise the same query would return the cached failure for
    a week. Removing it also clears a failure cached by an earlier run. A
    genuine empty field stays cached.
    """  # noqa: DOC501, DOC503
    # "+Gmag" asks VizieR to sort ascending (brightest first); combined with
    # row_limit this returns the N brightest sources in the cone, matching
    # twirl's "SELECT top N ... ORDER BY phot_g_mean_mag".
    vizier_kwargs = {}
    if mag_limit is not None:
        vizier_kwargs["column_filters"] = {_MAG_COL: f"<={mag_limit}"}
    vizier = Vizier(
        columns=["+" + _MAG_COL, _RA_COL, _DEC_COL, _PMRA_COL, _PMDEC_COL],
        row_limit=limit,
        timeout=SERVER_TIMEOUT,
        vizier_server=server,
        **vizier_kwargs,
    )
    response = vizier.query_region_async(
        center, radius=radius * u.deg, catalog=GAIA_DR3_VIZIER_CATALOG
    )
    try:
        return _read_votable(server, response)
    except Exception:
        # astroquery exposes cache-entry removal only on the last query, and
        # raises FileNotFoundError when the entry is already gone (it tests
        # ``Path.exists`` without calling it), which must not mask this error.
        with contextlib.suppress(FileNotFoundError):
            vizier._last_query.remove_cache_file(vizier.cache_location)  # noqa: SLF001
        raise


def cached_gaia_radecs(
    center, fov, *, limit=10000, magnitude=True, obs_epoch=None, mag_limit=None
):
    """
    Return Gaia DR3 RA/Dec (and magnitudes) in a field, cached via VizieR.

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
        If given, propagate positions from the Gaia DR3 epoch (2016.0) to this
        epoch using proper motions via
        :meth:`~astropy.coordinates.SkyCoord.apply_space_motion`. If ``None``
        (default), positions are returned at the catalog epoch with no
        proper-motion correction (matching the notebook's current behavior).
    mag_limit : float, optional
        If given, ask VizieR to return only sources with ``Gmag <= mag_limit``.
        If ``None`` (default) no filter is sent, so the query (and its cache
        entry) is identical to an unfiltered one.

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        An ``(n, 2)`` array of RA/Dec in degrees. If ``magnitude`` is ``True``,
        a ``(radecs, mags)`` tuple where ``mags`` is the length-``n`` array of
        Gaia G magnitudes.

    Raises
    ------
    GaiaQueryError
        If no server returned a catalog. Its message names the servers tried
        and ends with the last server's failure, which is also its
        ``__cause__``.

    Notes
    -----
    Servers are tried in order, each with a :data:`SERVER_TIMEOUT` second
    timeout, and every skipped server is logged at warning level. The timeout
    applies to connecting and to each wait for data, not to the whole answer,
    so a slow server can take longer than that and a query that falls through
    every server can take that many times as long. An error
    answer is never left in astroquery's cache, while a healthy answer for an
    empty field is cached and returned as shaped empties without trying
    another server, since every mirror would return the same empty field.
    Every exception counts as a server failure, not only ``requests`` ones: in
    the browser kernel a timeout or blocked request surfaces as
    ``pyodide.ffi.JsException``.
    """
    if not isinstance(center, SkyCoord):
        ra, dec = center
        center = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    radius = _fov_to_radius_deg(fov)

    servers = [(server, partial(_query_vizier, server)) for server in VIZIER_SERVERS]
    last_error = None
    for server, query in servers:
        try:
            table = query(center, radius, limit=limit, mag_limit=mag_limit)
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Gaia query to %s failed, skipping it: %s: %s",
                server,
                type(exc).__name__,
                exc,
            )
            last_error = exc
    else:
        tried = ", ".join(server for server, _ in servers)
        msg = (
            f"no Gaia server answered the query; tried {tried}; "
            f"last error: {type(last_error).__name__}: {last_error}"
        )
        raise GaiaQueryError(msg) from last_error

    if table is None:
        radecs = np.empty((0, 2))
        return (radecs, np.empty(0)) if magnitude else radecs

    if obs_epoch is None:
        ra = np.asarray(table[_RA_COL].value, dtype=float)
        dec = np.asarray(table[_DEC_COL].value, dtype=float)
    else:
        # Some Gaia sources lack proper motions; treat missing PM as zero so the
        # space-motion calculation does not fail.
        pmra = _pm_or_zero(table[_PMRA_COL])
        pmdec = _pm_or_zero(table[_PMDEC_COL])
        coords = SkyCoord(
            ra=table[_RA_COL].quantity,
            dec=table[_DEC_COL].quantity,
            pm_ra_cosdec=pmra,
            pm_dec=pmdec,
            obstime=Time(GAIA_DR3_EPOCH, format="jyear"),
        )
        # No parallax or radial velocity is supplied, so ERFA substitutes a
        # default distance and warns once per call. That substitution is
        # expected here; silence only this message, only around this call.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message='ERFA function "pmsafe" yielded', category=ErfaWarning
            )
            coords = coords.apply_space_motion(new_obstime=Time(obs_epoch))
        ra = coords.ra.deg
        dec = coords.dec.deg
        # Defensive final guard: if propagation still produced a non-finite
        # position, fall back to that row's catalog (J2016.0) position --
        # semantically the same as "no usable proper motion", and immune to
        # upstream surprises like the masked-to-NaN quantity conversion above.
        nonfinite = ~(np.isfinite(ra) & np.isfinite(dec))
        if nonfinite.any():
            ra = np.where(nonfinite, np.asarray(table[_RA_COL].value, dtype=float), ra)
            dec = np.where(
                nonfinite, np.asarray(table[_DEC_COL].value, dtype=float), dec
            )

    radecs = np.array([ra, dec]).T

    if magnitude:
        mags = np.asarray(table[_MAG_COL].value, dtype=float)
        return radecs, mags
    return radecs


def query_field_catalog(
    center,
    fov_rad,
    *,
    cone_margin=0.0,
    obs_epoch=None,
    gaia_mag_limit,
    contaminant_mag_limit,
    row_limit=10000,
):
    """
    Return Gaia (radecs, mags) for a field, with a row-limit truncation check.

    Parameters
    ----------
    center : astropy.coordinates.SkyCoord or tuple
        Center of the field. A tuple is interpreted as ``(ra, dec)`` in degrees.
    fov_rad : float
        Field radius in degrees. Must be positive.
    cone_margin : float, optional
        Extra radius in degrees added to ``fov_rad`` for the cone search.
    obs_epoch : astropy.time.Time or str, optional
        Epoch to propagate positions to; see `cached_gaia_radecs`.
    gaia_mag_limit : float
        Faintest G magnitude of photometry targets.
    contaminant_mag_limit : float
        Faintest G magnitude of contaminant stars; the catalog is filtered to
        ``Gmag <= contaminant_mag_limit``.
    row_limit : int, optional
        Row limit for a cone of radius ``fov_rad``; scaled up with the cone
        area when ``cone_margin`` is non-zero. Default 10000.

    Returns
    -------
    radecs : numpy.ndarray
        An ``(n, 2)`` array of RA/Dec in degrees, brightest first.
    mags : numpy.ndarray
        Length-``n`` array of Gaia G magnitudes.

    Raises
    ------
    ValueError
        If ``fov_rad`` is not positive.
    CatalogTruncationError
        If the query returned exactly the scaled row limit and its faintest
        source is no fainter than ``gaia_mag_limit``, so targets were lost.

    Notes
    -----
    The VizieR magnitude filter returns the same rows in the same order as an
    unfiltered query cut at the same G, but moves 3-4 times less data. The row
    limit is scaled with the cone area so the per-area depth stays constant
    when the cone is widened. Dense fields can legitimately truncate inside the
    contaminant range; that case only logs a warning, whereas truncation inside
    the target range is an error.
    """
    if not fov_rad > 0:
        msg = f"fov_rad must be positive, got {fov_rad}"
        raise ValueError(msg)

    radius = fov_rad + cone_margin
    scaled_limit = math.ceil(row_limit * (radius / fov_rad) ** 2)

    radecs, mags = cached_gaia_radecs(
        center,
        2 * radius,
        obs_epoch=obs_epoch,
        mag_limit=contaminant_mag_limit,
        limit=scaled_limit,
    )

    if len(mags) == scaled_limit:
        faintest = mags.max()
        if faintest <= gaia_mag_limit:
            msg = (
                f"Gaia query hit its row limit ({scaled_limit}) with the faintest "
                f"returned source at G={faintest:.2f}; photometry targets brighter "
                f"than gaia_mag_limit={gaia_mag_limit} were lost."
            )
            raise CatalogTruncationError(msg)
        if faintest < contaminant_mag_limit:
            logger.warning(
                "Gaia query hit its row limit (%d); the contaminant catalog is "
                "incomplete below G=%.2f (contaminant_mag_limit=%s).",
                scaled_limit,
                faintest,
                contaminant_mag_limit,
            )

    return radecs, mags
