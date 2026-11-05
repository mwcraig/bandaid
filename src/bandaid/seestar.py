"""
Seestar-specific star flags.

The Seestar S50 camera runs a hot-pixel correction that sometimes overwrites a
bright star's brightest core pixel with ``floor`` of the mean of its four
same-color axis neighbors at +-2 px. The star loses roughly 20 % of its flux
in that channel in that frame. On the camera's raw integers the replaced pixel
``c`` satisfies ``0 <= S4 - 4c <= 3``, where ``S4`` is the sum of the pixels at
``(+-2, 0)`` and ``(0, +-2)``; the remainder of the integer division is all the
slack there is. Some firmware/units also replace a fixed list of pixels in every
frame. There the replacement value differs from the neighbor mean by a fixed
per-pixel amount, additive or proportional to the sky in that pixel's color
depending on the unit, so `static_replacement_map` finds the list across a
batch by the constancy of ``S4 - 4c`` from frame to frame, not by its size.

+-2 px is the nearest same-color pixel along each axis for R, B and both G
sub-lattices of any 2x2 Bayer pattern, so none of these tests needs to know the
pattern; only the final per-channel attribution does.

`CORE_REPLACED` packages these tests as the star flagger the bundled Seestar50
profile names, ``seestar_core_replaced``.
"""

import logging

import numpy as np

from .instruments import StarFlagger

__all__ = [
    "CORE_RADIUS_PX",
    "CORE_REPLACED",
    "EDGE_PX",
    "MIN_PEAK_ABOVE_SKY_ADU",
    "STATIC_MAP_FRAMES",
    "STATIC_MAP_MIN_FRAMES",
    "STATIC_SCATTER_FLOOR_ADU",
    "STATIC_SCATTER_SKY_FRACTION",
    "core_replaced_flags",
    "replacement_map",
    "replacement_residual",
    "static_replacement_map",
]

logger = logging.getLogger(__name__)

# Distance to the nearest same-color pixel along each axis in a 2x2 Bayer mosaic.
_NEIGHBOR_STEP = 2

# Below ~2500 ADU above sky a replaced core is indistinguishable from noise, and
# faint stars would collect chance exact hits (~0.2 % per pixel). Gating a bit
# below that keeps every detectable event. Measured on the SS Leo study:
# https://github.com/mwcraig/bandaid/issues/182
MIN_PEAK_ABOVE_SKY_ADU = 2000

# The replaced pixel is the star's brightest; allow for the centroid sitting up
# to a pixel diagonal away from it.
CORE_RADIUS_PX = 1.5

# Pixels in the outer EDGE_PX rows/columns have at least one neighbor off the
# frame, so the exact test cannot be evaluated there: never flagged, never in
# the static map.
EDGE_PX = _NEIGHBOR_STEP

# A statically replaced pixel's S4 - 4c is a fixed per-pixel amount: its robust
# scatter from frame to frame is <= 10 ADU, while an ordinary pixel's scatters by
# 130-1300 ADU on real frames. A 20 ADU ceiling keeps every measured list and
# admits 0-1 chance pixels per batch. The scatter is of S4 - 4c, four times the
# pixel's offset from its neighbor mean, so 20 ADU here is 5 ADU per pixel.
# Survey of every frame of five units:
# https://github.com/mwcraig/bandaid/pull/183#issuecomment-6647298305
STATIC_SCATTER_FLOOR_ADU = 20.0

# On a bright sky the ceiling is this fraction of the sky of the pixel's own
# Bayer phase instead, where that is larger: a residual proportional to sky
# carries the sky's own frame-to-frame jitter. Like the floor, it bounds the
# scatter of S4 - 4c (four times the per-pixel offset from the neighbor mean).
STATIC_SCATTER_SKY_FRACTION = 0.01

# The diagonal neighbors at (+-2, +-2), which no firmware uses, give a control
# map of pure chance matches. More than this many pixels, or this fraction of
# the axis map, means the sky is too flat for the scatter test to tell a
# replaced pixel from an ordinary one, so the batch gets no static map. The
# control is 0 on every real batch at the floor above, and the fraction only
# matters above 1000 axis pixels:
# https://github.com/mwcraig/bandaid/pull/183#discussion_r4476325981
_CONTROL_MAX_PIXELS = 10
_CONTROL_MAX_FRACTION = 0.01

# A star clipped at the ADC ceiling has a residual of 0 in its core against
# both the four axis neighbors at (+-2, 0), (0, +-2) and the four diagonal
# neighbors at (+-2, +-2), so if it stays put across the frames its core reads
# as a constant residual. A pixel is left out of both maps when it or a
# neighbor at +-2 px along an axis or a diagonal reaches the saturation ceiling
# in at least this fraction of the frames.
# Masking on any one frame would also drop 3, 6 and 16 genuine list pixels on
# three real units, where a passing star breaks them in a few frames only; at
# half the frames none is lost on any of five real batches:
# https://github.com/mwcraig/bandaid/pull/183#discussion_r4484501546
_SATURATED_FRAME_FRACTION = 0.5

# Frames a batch should feed to `static_replacement_map`. A sweep over the
# first N frames of four real units showed 20 to be the smallest count with no
# chance match in the diagonal control on every unit; 30 or 40 frames changed
# the map by 0-2 pixels:
# https://github.com/mwcraig/bandaid/pull/183#discussion_r4484500188
STATIC_MAP_FRAMES = 20

# Below this many frames no static map is built at all. The control (diagonal)
# count was measured on the first N frames of four real units: at 10 frames it
# showed 2-5 chance matches on three units and 137 on the fourth (V816 Oph),
# which tripped the control and switched that batch's map off; at 15 it was 0-1
# on all four, and the maps matched the 20-frame maps to within a few pixels; at
# 20 and above it was 0. Same sweep as for STATIC_MAP_FRAMES above.
STATIC_MAP_MIN_FRAMES = 15

# 1.4826 * MAD estimates the standard deviation of normally distributed values.
_MAD_TO_SIGMA = 1.4826

# Interior rows of the frames' residuals held and reduced at a time, to bound
# the memory of the static test: a block of N frames costs N * _ROW_BLOCK *
# width * 4 bytes (2.75 MB for 20 portrait Seestar frames at 32 rows, 4.9 MB
# landscape), and about three blocks are live during the reduction. The block
# size has no measurable effect on the run time, which the medians dominate.
_ROW_BLOCK = 32

# floor(S4 / 4) leaves S4 - 4c in 0..3.
_EXACT_MAX_RESIDUAL = 3

# Fails the exact test; the static test never evaluates the border.
_BORDER_RESIDUAL = np.iinfo(np.int32).max

# Candidate core pixels lie within CORE_RADIUS_PX of the centroid, hence within
# CORE_RADIUS_PX + 0.5 px of the centroid's nearest pixel along each axis.
_CORE_HALF = int(np.ceil(CORE_RADIUS_PX + 0.5))

# Per-star window half-width: the candidate core plus room for its neighbors.
_WINDOW_HALF = _CORE_HALF + _NEIGHBOR_STEP


def _is_integer_frame(raw):
    return np.issubdtype(np.asarray(raw).dtype, np.integer)


_AXIS_OFFSETS = (
    (-_NEIGHBOR_STEP, 0),
    (_NEIGHBOR_STEP, 0),
    (0, -_NEIGHBOR_STEP),
    (0, _NEIGHBOR_STEP),
)
_DIAGONAL_OFFSETS = (
    (-_NEIGHBOR_STEP, -_NEIGHBOR_STEP),
    (-_NEIGHBOR_STEP, _NEIGHBOR_STEP),
    (_NEIGHBOR_STEP, -_NEIGHBOR_STEP),
    (_NEIGHBOR_STEP, _NEIGHBOR_STEP),
)


def _residual(raw, offsets):
    """
    Compute the sum of four neighbors minus four times each pixel.

    Parameters
    ----------
    raw : numpy.ndarray
        Raw 2-D frame of any integer dtype whose values fit in 29 bits (the
        arithmetic is int32 and holds four times the largest value).
    offsets : sequence of tuple of int
        Four ``(dy, dx)`` neighbor offsets, each component at most
        `_NEIGHBOR_STEP` in size.

    Returns
    -------
    numpy.ndarray
        int32 array of the shape of ``raw``, with `_BORDER_RESIDUAL` in the outer
        `EDGE_PX` border.
    """
    # int32 so that the sum of four bright pixels (up to 4 * 65535 on uint16)
    # and a negative difference both fit; the frame's own dtype cannot hold either.
    r = np.asarray(raw).astype(np.int32)
    s = _NEIGHBOR_STEP
    ny, nx = r.shape
    # The residual exists only where all four neighbors are on the frame, i.e.
    # s pixels in from every edge; the border gets a value no test accepts.
    out = np.full(r.shape, _BORDER_RESIDUAL, dtype=np.int32)
    # Start from -4c on the interior, then add each neighbor: the slice offset by
    # (dy, dx) lines neighbor (y + dy, x + dx) up with interior pixel (y, x), so
    # the four adds are whole-array operations with no per-pixel loop.
    acc = -4 * r[s:-s, s:-s]
    for dy, dx in offsets:
        acc += r[s + dy : ny - s + dy, s + dx : nx - s + dx]
    out[s:-s, s:-s] = acc
    return out


def replacement_residual(raw):
    """
    Compute ``S4 - 4c`` for every pixel of a raw frame.

    Parameters
    ----------
    raw : numpy.ndarray
        Raw 2-D frame of any integer dtype whose values fit in 29 bits; any
        16-bit camera does.

    Returns
    -------
    numpy.ndarray
        int32 array of the shape of ``raw``. Interior pixels hold the sum of
        their four axis neighbors at +-2 px minus four times their own value;
        the outer `EDGE_PX` border holds ``np.iinfo(np.int32).max``, which fails
        every test in this module.

    Notes
    -----
    The frame is cast to int32 before any arithmetic: on uint16 the sum of four
    bright pixels overflows and the difference wraps. int32 holds ``S4 - 4c``
    exactly for values up to 29 bits, since ``S4`` and ``4c`` are each at most
    four times the largest value; any 16-bit camera is well inside that.
    """
    return _residual(raw, _AXIS_OFFSETS)


def replacement_map(raw):
    """
    Map the pixels of a raw frame that are exact neighbor-mean replacements.

    Parameters
    ----------
    raw : numpy.ndarray
        Raw 2-D frame of any integer dtype whose values fit in 29 bits; any
        16-bit camera does.

    Returns
    -------
    numpy.ndarray
        Boolean array of the shape of ``raw``, True where
        ``0 <= S4 - 4c <= 3``. The outer `EDGE_PX` border is always False, and
        so is every pixel of a non-integer frame.

    Notes
    -----
    The relation only holds on the camera's own integers; a float frame (for
    example one that has been calibrated or synthesized) gets no hits rather
    than a guess.
    """
    if not _is_integer_frame(raw):
        logger.debug("Frame is not integer; no replacement map.")
        return np.zeros(np.shape(raw), dtype=bool)
    d = replacement_residual(raw)
    return (d >= 0) & (d <= _EXACT_MAX_RESIDUAL)


def _phase_skies(raw):
    """
    Return a frame's sky in each of its four Bayer phases.

    Parameters
    ----------
    raw : numpy.ndarray
        Raw 2-D frame.

    Returns
    -------
    numpy.ndarray
        float64 array of shape (2, 2): element ``[py, px]`` is the median of the
        interior pixels at frame rows ``py`` and columns ``px`` modulo 2.

    Notes
    -----
    Each phase is one color on any 2x2 Bayer pattern, so these are the
    color planes' skies without knowing the pattern.
    """
    s = _NEIGHBOR_STEP
    interior = np.asarray(raw)[s:-s, s:-s]
    # Interior row (column) 0 is frame row (column) s.
    return np.array(
        [
            [np.median(interior[(py - s) % 2 :: 2, (px - s) % 2 :: 2]) for px in (0, 1)]
            for py in (0, 1)
        ]
    )


def _saturated_mask(raw_frames, saturated_adu):
    """
    Map pixels near a pixel saturated in at least half the frames.

    Parameters
    ----------
    raw_frames : sequence of numpy.ndarray
        Raw 2-D integer frames, all of the same shape.
    saturated_adu : float
        Raw value at and above which a pixel counts as saturated.

    Returns
    -------
    numpy.ndarray
        Boolean array of the frames' shape, True where the pixel itself, one of
        its axis neighbors or one of its diagonal neighbors is at least
        ``saturated_adu`` in at least `_SATURATED_FRAME_FRACTION` of the
        frames. The outer `EDGE_PX` border, which neither map covers, is False.

    Notes
    -----
    A pixel is mapped when it, or a neighbor at +-2 px along an axis or a
    diagonal, is saturated in at least half the frames.

    The frames are counted one at a time into a single integer plane, so no
    stack of frames is built.
    """
    s = _NEIGHBOR_STEP
    shape = np.shape(raw_frames[0])
    ny, nx = shape
    count = np.zeros((ny - 2 * s, nx - 2 * s), dtype=np.int16)
    for frame in raw_frames:
        saturated = np.asarray(frame) >= saturated_adu
        reached = saturated[s:-s, s:-s].copy()
        for dy, dx in _AXIS_OFFSETS + _DIAGONAL_OFFSETS:
            reached |= saturated[s + dy : ny - s + dy, s + dx : nx - s + dx]
        count += reached
    out = np.zeros(shape, dtype=bool)
    out[s:-s, s:-s] = count >= _SATURATED_FRAME_FRACTION * len(raw_frames)
    return out


def _constant_residual_map(raw_frames, offsets, skies):
    """
    Map the interior pixels whose residual is constant across the frames.

    Parameters
    ----------
    raw_frames : sequence of numpy.ndarray
        Raw 2-D integer frames, all of the same shape.
    offsets : sequence of tuple of int
        Neighbor offsets passed to `_residual`.
    skies : numpy.ndarray
        Per-phase sky of each frame, shape (frames, 2, 2), from `_phase_skies`.

    Returns
    -------
    numpy.ndarray
        Boolean array of the frames' shape, False on the outer `EDGE_PX`
        border.

    Notes
    -----
    The rule is stated in the Notes of `static_replacement_map`.
    """
    s = _NEIGHBOR_STEP
    shape = np.shape(raw_frames[0])
    n_rows = shape[0] - 2 * s
    # The residual is defined on the interior only, s pixels in from each edge;
    # the blocks below walk its rows.
    n_cols = shape[1] - 2 * s

    # Per phase: the ceiling on the scatter is the floor or a fraction of the
    # batch's median sky in that phase, whichever is larger, so a fainter plane
    # gets a tighter ceiling; the scale brings each frame's sky in that phase to
    # the batch median. Both are indexed [py, px] by frame row and column
    # parity; the scale has a leading frame axis. A phase whose sky is 0 in
    # some frame (one color plane of a dark sky on some units) has no sky to
    # scale by, so its scale is 1 and only its raw scatter counts.
    sky = np.median(skies, axis=0)
    phase_threshold = np.maximum(
        STATIC_SCATTER_FLOOR_ADU, STATIC_SCATTER_SKY_FRACTION * sky
    )
    has_sky = (skies > 0).all(axis=0)
    phase_scale = np.where(has_sky, sky / np.where(has_sky, skies, 1), 1)
    phase_scale = phase_scale.astype(np.float32)
    # Interior column j is frame column s + j.
    col_parity = (s + np.arange(n_cols)) % 2

    def median_of_sorted(ordered):
        # Median along the leading (frame) axis of an array already sorted
        # along it; the same values np.median gives.
        n = len(ordered)
        if n % 2:
            return ordered[n // 2]
        return np.mean(ordered[n // 2 - 1 : n // 2 + 1], axis=0)

    def scatter(values):
        # Robust standard deviation along the frame axis: 1.4826 x the median
        # absolute deviation from each pixel's median over the frames. The
        # medians sort rather than call np.median: np.median partitions each
        # 20-element lane with introselect, while np.sort on a short axis uses
        # numpy's SIMD sort, measured ~5x faster here with identical results.
        # About three block-sized arrays are live at once: the block, the
        # sorted copy np.sort makes of it (then ``dev``, which reuses its
        # place) and the per-frame ``scale`` of the caller.
        dev = values - median_of_sorted(np.sort(values, axis=0))
        np.abs(dev, out=dev)
        # dev is finished with, so sort it in place rather than copy it.
        dev.sort(axis=0)
        return _MAD_TO_SIGMA * median_of_sorted(dev)

    out = np.zeros(shape, dtype=bool)
    # The residuals are computed and reduced _ROW_BLOCK interior rows at a time
    # to bound memory. Interior rows y0 to y0 + rows of a frame's residual are
    # those of the slab of frame rows y0 to y0 + rows + 2 * s, less the slab's
    # own border: the extra s rows on each side are the neighbors at +-s that
    # the residual needs.
    for y0 in range(0, n_rows, _ROW_BLOCK):
        rows = min(_ROW_BLOCK, n_rows - y0)
        # Block row i is frame row s + y0 + i. Indexing the per-phase arrays by
        # the parities gives the threshold of every pixel of the block, shaped
        # (rows, columns), and the scale, shaped (frames, rows, columns) to
        # broadcast over the block.
        row_parity = ((s + y0 + np.arange(rows)) % 2)[:, None]
        threshold = phase_threshold[row_parity, col_parity]
        scale = phase_scale[:, row_parity, col_parity]

        # Interior rows y0 .. y0 + rows of every frame's residual, stacked along
        # a leading frame axis: (frames, rows, interior columns). float32 holds
        # every residual of a uint16 frame exactly; the residuals are not
        # clipped to a smaller type, since a hot pixel clipped in every frame
        # would read as constant.
        block = np.empty((len(raw_frames), rows, n_cols), dtype=np.float32)
        for i, frame in enumerate(raw_frames):
            # _residual marks the slab's own border as unusable; slicing s off
            # each side leaves exactly the block's rows and the interior columns.
            slab = np.asarray(frame)[y0 : y0 + rows + 2 * s]
            block[i] = _residual(slab, offsets)[s:-s, s:-s]

        # The raw scatter is constant for a replacement that differs from the
        # neighbor mean by a fixed amount.
        rsd = scatter(block)
        # The scaled scatter is constant for one proportional to the sky. The
        # raw block is finished with, so scale it in place.
        block *= scale
        rsd = np.minimum(rsd, scatter(block))
        # Static when either scatter is within the ceiling.
        out[s + y0 : s + y0 + rows, s:-s] = rsd <= threshold
    return out


def static_replacement_map(raw_frames, *, saturated_adu):
    """
    Map the pixels whose neighbor-mean residual is constant across a batch.

    Parameters
    ----------
    raw_frames : sequence of numpy.ndarray
        Raw 2-D frames of one camera, all of the same shape. A batch should pass
        `STATIC_MAP_FRAMES` of them.
    saturated_adu : float
        Raw value at and above which a pixel counts as saturated: the
        instrument's largest usable ADU value.

    Returns
    -------
    numpy.ndarray
        Boolean array of the frames' shape, True where the robust scatter of
        ``S4 - 4c`` (four times the pixel's offset from its neighbor mean, so
        20 ADU is 5 ADU per pixel) across the frames is within
        `STATIC_SCATTER_FLOOR_ADU` or `STATIC_SCATTER_SKY_FRACTION` of the sky
        of the pixel's own Bayer phase, whichever is larger, and the pixel is
        not next to a saturated one in half the frames. All False when fewer
        than `STATIC_MAP_MIN_FRAMES` frames are given or any frame is not
        integer (logged at INFO), or when the diagonal control fails (logged at
        WARNING).

    Notes
    -----
    The module docstring explains why a constant ``S4 - 4c`` marks a replaced
    pixel. A pixel is kept when::

        min(rsd, rsd_norm) <= max(STATIC_SCATTER_FLOOR_ADU,
                                  STATIC_SCATTER_SKY_FRACTION * sky)

    where

    - ``rsd`` is 1.4826 times the median absolute deviation of ``S4 - 4c``
      over the frames;
    - ``rsd_norm`` is the same after scaling each frame's residual by
      ``sky`` over that frame's own median of the pixel's Bayer phase, which
      removes a drifting sky's effect on a residual proportional to sky; where
      that frame median is 0 in some frame the phase is not scaled;
    - ``sky`` is the batch median of the per-frame median of the pixel's own
      Bayer phase over the frame interior.

    ``S4 - 4c`` involves only the pixel's own phase, since its neighbors at
    +-2 px are the same color, so that phase's sky is the one it can be
    proportional to. A dead pixel's residual follows its neighbors' noise, so
    it never qualifies.

    A star clipped at the ADC ceiling and stationary across the frames has a
    residual of 0 in its core against both the four axis neighbors at
    (+-2, 0), (0, +-2) and the four diagonal neighbors at (+-2, +-2), so
    pixels where the pixel or a neighbor at +-2 px along an axis or a diagonal
    is at least ``saturated_adu`` in at least half the frames are left out of
    both maps. In a later, unsaturated frame such a pixel would otherwise give
    a false flag.

    The same statistic on the diagonal neighbors at (+-2, +-2), which no
    firmware uses, counts chance matches. When it maps more than
    ``max(10, 0.01 * n_axis)`` pixels (a sky too flat or too dark for the test)
    no static map is used and a WARNING names both counts; otherwise one INFO
    line reports them.
    """
    shape = np.shape(raw_frames[0])
    if len(raw_frames) < STATIC_MAP_MIN_FRAMES:
        # A short batch (one frame on the command line, say) is ordinary, so
        # this is not a warning.
        logger.info(
            "Only %d %s given for the static replacement map; at least %d "
            "are needed to rule out chance matches, so no static map is used.",
            len(raw_frames),
            "frame" if len(raw_frames) == 1 else "frames",
            STATIC_MAP_MIN_FRAMES,
        )
        return np.zeros(shape, dtype=bool)
    if not all(_is_integer_frame(f) for f in raw_frames):
        logger.info(
            "Static replacement map needs the camera's raw integer frames; "
            "these are %s, so no static map is used.",
            next(np.asarray(f).dtype for f in raw_frames if not _is_integer_frame(f)),
        )
        return np.zeros(shape, dtype=bool)

    skies = np.array([_phase_skies(f) for f in raw_frames])
    unsaturated = ~_saturated_mask(raw_frames, saturated_adu)
    static = _constant_residual_map(raw_frames, _AXIS_OFFSETS, skies) & unsaturated
    n_axis = int(static.sum())
    diagonal = _constant_residual_map(raw_frames, _DIAGONAL_OFFSETS, skies)
    n_diag = int((diagonal & unsaturated).sum())
    if n_diag > max(_CONTROL_MAX_PIXELS, _CONTROL_MAX_FRACTION * n_axis):
        logger.warning(
            "Static replacement map: the diagonal control matches %d pixels "
            "against %d for the axis neighbors, so chance matches survive and "
            "this batch cannot support a static list; no static map is used.",
            n_diag,
            n_axis,
        )
        return np.zeros(shape, dtype=bool)
    logger.info(
        "Static replacement map: %d pixels (diagonal control %d)", n_axis, n_diag
    )
    return static


def core_replaced_flags(
    raw,
    centroids,
    bayer_masks,
    peak_count,
    bkgd_count,
    *,
    saturated_adu,
    static_map=None,
):
    """
    Flag star-channels whose core pixel was replaced by the camera.

    Parameters
    ----------
    raw : numpy.ndarray
        Raw 2-D frame, indexed ``[y, x]``, of any integer dtype whose values fit
        in 29 bits; any 16-bit camera does.
    centroids : array-like
        ``(N, 2)`` star centroids in pixels, columns ``x`` then ``y``.
    bayer_masks : mapping of str to numpy.ndarray or None
        Channel name to a boolean array of the frame's shape that is True where
        a pixel is *not* in that channel, as from `generate_bayer_masks`.
        Channels whose mask is None are skipped.
    peak_count : mapping of str to numpy.ndarray
        Channel name to the length-N raw maximum pixel of each star in that
        channel.
    bkgd_count : mapping of str to numpy.ndarray
        Channel name to the length-N per-pixel sky of each star in that channel.
    saturated_adu : float
        Raw value at and above which a star's ``peak_count`` counts as
        saturated: the instrument's largest usable ADU value. A clipped core is
        a flat plateau, where ``S4 - 4c`` is 0 and every plateau pixel looks
        like an exact replacement.
    static_map : numpy.ndarray, optional
        Boolean map from `static_replacement_map`; its pixels count as replaced
        without the exact test.

    Returns
    -------
    dict of str to numpy.ndarray
        One length-N boolean array per channel whose mask is not None.

    Notes
    -----
    A star-channel is flagged when some pixel of that channel, centered within
    `CORE_RADIUS_PX` of the centroid and not in the outer `EDGE_PX` of the
    frame, satisfies ``0 <= S4 - 4c <= 3`` or is in ``static_map``; and the
    channel's ``peak_count - bkgd_count >= MIN_PEAK_ABOVE_SKY_ADU``; and its
    ``peak_count < saturated_adu``. A star with a non-finite centroid or
    ``peak_count`` is never flagged, and neither is any star of a non-integer
    frame (see the module docstring).

    Only a small window around each star is examined, gathered for all stars at
    once by fancy indexing. Window indices are clipped to the frame; a candidate
    pixel counts only if it lies outside the outer `EDGE_PX`, which is exactly
    the condition for all its neighbors to be on the frame, so a clipped (and
    therefore wrong) neighbor value never produces a counted hit.
    """
    centroids = np.asarray(centroids, dtype=float).reshape(-1, 2)
    n = len(centroids)
    channels = [ch for ch, mask in bayer_masks.items() if mask is not None]
    if not _is_integer_frame(raw):
        logger.debug("Frame is not integer; no core-replacement flags.")
        return {ch: np.zeros(n, dtype=bool) for ch in channels}

    ny, nx = raw.shape
    # Plane-fit failures leave NaN centroid rows; examine pixel (0, 0) for
    # them instead, which keeps the integer cast warning-free, and drop their
    # flags at the end.
    finite = np.isfinite(centroids).all(axis=1)
    cx, cy = np.where(finite[:, None], centroids, 0.0).T
    offsets = np.arange(-_WINDOW_HALF, _WINDOW_HALF + 1)
    rows = np.rint(cy).astype(int)[:, None] + offsets  # (N, W)
    cols = np.rint(cx).astype(int)[:, None] + offsets  # (N, W)
    rows_c = np.clip(rows, 0, ny - 1)
    cols_c = np.clip(cols, 0, nx - 1)
    win = raw[rows_c[:, :, None], cols_c[:, None, :]].astype(np.int32)  # (N, W, W)

    # Candidate core pixels are the inner (2 * _CORE_HALF + 1)^2 of each window;
    # their neighbors at +-_NEIGHBOR_STEP are the window shifted by that much.
    s = _NEIGHBOR_STEP
    inner = slice(s, -s)
    c = win[:, inner, inner]
    resid = (
        win[:, : -2 * s, inner]
        + win[:, 2 * s :, inner]
        + win[:, inner, : -2 * s]
        + win[:, inner, 2 * s :]
        - 4 * c
    )
    hit = (resid >= 0) & (resid <= _EXACT_MAX_RESIDUAL)  # (N, K, K)

    # Absolute coordinates of the candidates, kept separable: rows (N, K, 1) and
    # columns (N, 1, K) broadcast to (N, K, K) wherever they are combined.
    crow = rows[:, inner][:, :, None]
    ccol = cols[:, inner][:, None, :]
    crow_c = rows_c[:, inner][:, :, None]
    ccol_c = cols_c[:, inner][:, None, :]

    if static_map is not None:
        hit |= static_map[crow_c, ccol_c]

    dx = ccol - cx[:, None, None]
    dy = crow - cy[:, None, None]
    in_core = dx**2 + dy**2 <= CORE_RADIUS_PX**2
    interior = (
        (crow >= EDGE_PX)
        & (crow < ny - EDGE_PX)
        & (ccol >= EDGE_PX)
        & (ccol < nx - EDGE_PX)
    )
    candidate = hit & in_core & interior

    flags = {}
    for ch in channels:
        in_channel = ~bayer_masks[ch][crow_c, ccol_c]
        peak = np.asarray(peak_count[ch], dtype=float)
        sky = np.asarray(bkgd_count[ch], dtype=float)
        flags[ch] = (
            finite
            & (candidate & in_channel).any(axis=(1, 2))
            & (peak - sky >= MIN_PEAK_ABOVE_SKY_ADU)
            & (peak < saturated_adu)
        )
    return flags


def _flag_core_replaced(inputs, state):
    """
    Adapt `core_replaced_flags` to the star-flagger interface.

    Parameters
    ----------
    inputs : bandaid.instruments.StarFlagInputs
        The frame's raw data, centroids, Bayer masks, gate columns and
        metadata, whose ``"largest_usable_adu_value"`` is the saturation
        ceiling.
    state : numpy.ndarray or None
        The batch's static map from `static_replacement_map`, or None.

    Returns
    -------
    dict of str to numpy.ndarray
        One length-N boolean array per channel, as from `core_replaced_flags`.
    """
    return core_replaced_flags(
        inputs.raw,
        inputs.centroids,
        inputs.bayer_masks,
        inputs.peak_count,
        inputs.bkgd_count,
        saturated_adu=inputs.metadata["largest_usable_adu_value"],
        static_map=state,
    )


def _prepare_core_replaced(raw_frames, metadata):
    """
    Adapt `static_replacement_map` to the star-flagger interface.

    Parameters
    ----------
    raw_frames : sequence of numpy.ndarray
        The batch's first raw frames.
    metadata : mapping
        The batch's frame metadata, whose ``"largest_usable_adu_value"`` is the
        saturation ceiling.

    Returns
    -------
    numpy.ndarray
        The batch's static map.
    """
    return static_replacement_map(
        raw_frames, saturated_adu=metadata["largest_usable_adu_value"]
    )


# The Seestar core-replacement star flagger, built in as "seestar_core_replaced".
CORE_REPLACED = StarFlagger(
    column="core_replaced",
    flag=_flag_core_replaced,
    prepare=_prepare_core_replaced,
    n_state_frames=STATIC_MAP_FRAMES,
)
