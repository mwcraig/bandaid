"""
Unit tests for the Seestar core-replacement flags in :mod:`bandaid.seestar`.

The Seestar S50 hot-pixel correction sometimes overwrites a bright star's core
pixel with ``floor`` of the mean of its four same-colour axis neighbours at
+-2 px. On the raw integer frame that leaves ``0 <= S4 - 4c <= 3``. These tests
build small deterministic uint16 frames with Gaussian stars, plant such
replacements by hand, and pin which star-channels get flagged.
"""

import logging
import re
import tracemalloc

import numpy as np
import pytest

from bandaid import seestar
from bandaid.seestar import (
    MIN_PEAK_ABOVE_SKY_ADU,
    STATIC_MAP_FRAMES,
    STATIC_MAP_MIN_FRAMES,
    STATIC_SCATTER_FLOOR_ADU,
    STATIC_SCATTER_SKY_FRACTION,
    _CONTROL_MAX_FRACTION,
    _CONTROL_MAX_PIXELS,
    _phase_skies,
    core_replaced_flags,
    replacement_map,
    replacement_residual,
    static_replacement_map,
)

SHAPE = (64, 64)
SKY = 500
STAR_PEAK = 20000
CHANNELS = ("TR", "TB", "TG")
SENTINEL = np.iinfo(np.int32).max

# The saturation ceiling the tests pass, the Seestar S50 profile's
# largest_usable_adu_value.
SATURATED_ADU = 50000


def _frame(
    stars=((32.0, 30.0),),
    *,
    seed=0,
    noise=15,
    sky=SKY,
    dtype=np.uint16,
    shape=SHAPE,
):
    """
    Build a deterministic frame: textured sky plus Gaussian stars.

    ``stars`` holds ``(x, y)`` centres. The sky gets a fixed-seed integer
    texture rather than a flat value: a flat sky has ``S4 - 4c == 0`` at every
    pixel, which would make every sky pixel an exact "replacement".
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[: shape[0], : shape[1]]
    img = sky + rng.integers(-noise, noise + 1, size=shape).astype(float)
    for sx, sy in stars:
        img += STAR_PEAK * np.exp(-((x - sx) ** 2 + (y - sy) ** 2) / 2)
    return np.rint(img).astype(dtype)


def _neighbour_coords(frame, x, y):
    """Axis neighbours at +-2 px of ``(x, y)``, clipped to the frame."""
    ny, nx = frame.shape
    return [
        (min(max(y + dy, 0), ny - 1), min(max(x + dx, 0), nx - 1))
        for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2))
    ]


def _plant(frame, x, y):
    """
    Overwrite pixel ``(x, y)`` the way the camera does.

    The value is ``floor`` of the mean of the four axis neighbours at +-2 px.
    Near the frame edge the neighbours are clipped to the frame, so the planted
    value is one that a window gathered with clipped indices would read as an
    exact hit; the edge rule, not luck, must keep such pixels from counting.
    A clipped neighbour can land on the pixel itself, in which case the pixel's
    own value contributes and the remaining neighbours are averaged instead.
    """
    coords = _neighbour_coords(frame, x, y)
    others = [frame[c] for c in coords if c != (y, x)]
    frame[y, x] = sum(int(v) for v in others) // len(others)
    return frame


def _set_residual(frame, x, y, d):
    """Make ``S4 - 4c`` at ``(x, y)`` exactly ``d`` by adjusting the north neighbour."""
    _, s, w, e = (int(frame[c]) for c in _neighbour_coords(frame, x, y))
    frame[y - 2, x] = 4 * int(frame[y, x]) + d - (s + w + e)
    return frame


def _bayer_masks(pattern):
    """
    Mirror ``generate_bayer_masks`` for a top-down frame with no offsets.

    True means "not this channel", as in the pipeline.
    """
    masks = {}
    for color in "RBG":
        mask = np.ones(SHAPE, dtype=bool)
        for k, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            if pattern[k] == color:
                mask[dy::2, dx::2] = False
        masks["T" + color] = mask
    return masks


def _gates(n, peak=STAR_PEAK + SKY, sky=SKY):
    """Per-channel ``peak_count`` and ``bkgd_count`` arrays of length ``n``."""
    peak_count = {ch: np.full(n, peak, dtype=float) for ch in CHANNELS}
    bkgd_count = {ch: np.full(n, sky, dtype=float) for ch in CHANNELS}
    return peak_count, bkgd_count


def _flags(
    raw, centroids, pattern="RGGB", *, peak=STAR_PEAK + SKY, sky=SKY, static_map=None
):
    """Run `core_replaced_flags` with uniform gate values for every star."""
    centroids = np.asarray(centroids, dtype=float).reshape(-1, 2)
    peak_count, bkgd_count = _gates(len(centroids), peak=peak, sky=sky)
    return core_replaced_flags(
        raw,
        centroids,
        _bayer_masks(pattern),
        peak_count,
        bkgd_count,
        saturated_adu=SATURATED_ADU,
        static_map=static_map,
    )


def _static_map(frames, saturated_adu=SATURATED_ADU):
    """Run `static_replacement_map` with the tests' saturation ceiling."""
    return static_replacement_map(frames, saturated_adu=saturated_adu)


def _assert_flags(flags, expected):
    """``expected`` maps channel -> list of bools; every channel must be present."""
    assert set(flags) == set(expected)
    for ch, want in expected.items():
        assert flags[ch].dtype == bool
        np.testing.assert_array_equal(flags[ch], want, err_msg=ch)


# --------------------------------------------------------------------------
# replacement_residual
# --------------------------------------------------------------------------


def test_residual_exact_values_and_border_sentinel():
    """Exact residual where uint16 S4 would overflow; border is the sentinel."""
    # S4 = 4 * 60000 overflows uint16; the residual must not wrap.
    raw = np.zeros((5, 5), dtype=np.uint16)
    raw[0, 2] = raw[4, 2] = raw[2, 0] = raw[2, 4] = 60000
    raw[2, 2] = 59999
    expected = np.full((5, 5), SENTINEL, dtype=np.int32)
    expected[2, 2] = 4 * 60000 - 4 * 59999
    result = replacement_residual(raw)
    assert result.dtype == np.int32
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.int64, np.uint8])
def test_residual_matches_direct_sum(dtype):
    """The vectorised residual matches a direct per-pixel sum for any integer dtype."""
    rng = np.random.default_rng(1)
    raw = rng.integers(0, 200, size=(7, 9)).astype(dtype)
    wide = raw.astype(np.int64)
    expected = np.full(raw.shape, SENTINEL, dtype=np.int64)
    for y in range(2, raw.shape[0] - 2):
        for x in range(2, raw.shape[1] - 2):
            s4 = wide[y - 2, x] + wide[y + 2, x] + wide[y, x - 2] + wide[y, x + 2]
            expected[y, x] = s4 - 4 * wide[y, x]
    np.testing.assert_array_equal(replacement_residual(raw), expected)


# --------------------------------------------------------------------------
# replacement_map
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("d", "expected"),
    [(-1, False), (0, True), (1, True), (2, True), (3, True), (4, False)],
)
def test_replacement_map_exact_rule(d, expected):
    """Remainders 0..3 are exact replacements; -1 and 4 are not."""
    raw = np.full((5, 5), 1000, dtype=np.uint16)
    raw[2, 4] = 1000 + d
    want = np.zeros((5, 5), dtype=bool)
    want[2, 2] = expected
    result = replacement_map(raw)
    assert result.dtype == bool
    np.testing.assert_array_equal(result, want)


def test_replacement_map_border_never_flagged():
    """A flat frame hits everywhere in the interior and nowhere on the border."""
    # A flat frame is an exact hit everywhere in the interior, none on the border.
    raw = np.full((8, 8), 700, dtype=np.uint16)
    want = np.zeros((8, 8), dtype=bool)
    want[2:-2, 2:-2] = True
    np.testing.assert_array_equal(replacement_map(raw), want)


def test_replacement_map_float_frame_is_all_false():
    """A float frame has no camera integers to test, so nothing is flagged."""
    raw = np.full((8, 8), 700.0)
    np.testing.assert_array_equal(replacement_map(raw), np.zeros((8, 8), dtype=bool))


# --------------------------------------------------------------------------
# static_replacement_map
# --------------------------------------------------------------------------

# Planted pixels sit well apart so adjusting one never touches another's S4.
STATIC_PIXELS = {
    "large_offset": (10, 10),
    "small_negative": (20, 10),
    "exact": (30, 10),
    "jumps": (40, 10),
    "wide": (10, 30),
    "dead": (20, 30),
    "hot": (30, 30),
}

# The residuals S4 - 4c planted at each pixel, frame by frame. A replaced pixel
# differs from the neighbour mean by a fixed amount of any size, so it is the
# constancy of the residual that marks it.
STATIC_KEPT = {"large_offset", "small_negative", "exact"}


def _planted_residual(name, i, rng):
    if name == "large_offset":
        return 300
    if name == "small_negative":
        return -7
    if name == "exact":
        return int(rng.integers(0, 4))
    if name == "jumps":
        # Three levels, so that no one level holds a majority of the frames
        # (and so sets the median deviation to 0) for an odd frame count.
        return (0, 60, 120)[i % 3]
    if name == "wide":
        return (-30, -15, 0, 15, 30)[i % 5]
    raise ValueError(name)


# Sky noise per frame, wide enough that an ordinary pixel's residual scatters by
# far more than the static threshold (sd of S4 - 4c is about 4.5 * 46 ADU).
STATIC_NOISE = 80


def _static_frames(k=STATIC_MAP_MIN_FRAMES, dtype=np.uint16):
    rng = np.random.default_rng(1)
    frames = []
    for i in range(k):
        f = _frame(stars=(), seed=100 + i, noise=STATIC_NOISE)
        for name, (x, y) in STATIC_PIXELS.items():
            if name not in {"dead", "hot"}:
                _set_residual(f, x, y, d=_planted_residual(name, i, rng))
        x, y = STATIC_PIXELS["dead"]
        f[y, x] = 0
        x, y = STATIC_PIXELS["hot"]
        f[y, x] = 60000
        frames.append(f.astype(dtype))
    return frames


def _yx(names):
    return sorted([STATIC_PIXELS[n][1], STATIC_PIXELS[n][0]] for n in names)


@pytest.mark.parametrize("k", [STATIC_MAP_MIN_FRAMES, STATIC_MAP_FRAMES])
def test_static_map_keeps_pixels_with_a_constant_residual(k):
    """
    Pixels whose residual is constant across frames survive, whatever its size.

    A large constant offset and a small negative one are kept, as is an exact
    replacement; a residual that jumps or wanders by tens of ADU is dropped, and
    so are a dead and a hot pixel, whose residuals follow their neighbours'
    noise however far from zero they are.
    """
    result = _static_map(_static_frames(k=k))
    assert result.dtype == bool
    assert result.shape == SHAPE
    np.testing.assert_array_equal(np.argwhere(result).tolist(), _yx(STATIC_KEPT))


def test_static_map_border_never_on_the_map():
    """The outer border, whose sentinel residual is constant, is never mapped."""
    result = _static_map(_static_frames())
    interior = np.zeros(SHAPE, dtype=bool)
    interior[2:-2, 2:-2] = True
    assert not result[~interior].any()


def _residual_scatter(values):
    values = np.asarray(values, dtype=float)
    return 1.4826 * np.median(np.abs(values - np.median(values)))


def test_static_map_proportional_residual_on_a_drifting_sky():
    """
    A residual proportional to sky is kept even when the sky drifts.

    The sky ramps from 500 to 2500 ADU across the batch; the planted residual is
    4 % of it, so the raw residual scatters by more than the threshold but the
    sky-normalised one does not.
    """
    k = STATIC_MAP_FRAMES
    skies = np.linspace(500, 2500, k).round().astype(int)
    x, y = STATIC_PIXELS["large_offset"]
    planted = [round(0.04 * sky) for sky in skies]
    assert _residual_scatter(planted) > STATIC_SCATTER_FLOOR_ADU
    frames = []
    for i, sky in enumerate(skies):
        f = _frame(stars=(), seed=200 + i, noise=int(sky) // 10, sky=int(sky))
        _set_residual(f, x, y, d=planted[i])
        frames.append(f)
    result = _static_map(frames)
    np.testing.assert_array_equal(np.argwhere(result).tolist(), [[y, x]])


BRIGHT_SKY = 2500


@pytest.mark.parametrize(
    ("levels", "kept"),
    [
        (((BRIGHT_SKY, BRIGHT_SKY), (BRIGHT_SKY, BRIGHT_SKY)), True),
        (((SKY, SKY), (SKY, SKY)), False),
        (((BRIGHT_SKY, SKY), (SKY, SKY)), True),
        (((SKY, BRIGHT_SKY), (BRIGHT_SKY, SKY)), False),
    ],
    ids=["bright", "dark", "own-phase-bright", "other-phases-bright"],
)
def test_static_map_threshold_scales_with_bright_sky(levels, kept):
    """
    On a bright sky the threshold is 1 % of the pixel's own phase's sky.

    A residual alternating +-16 ADU has a robust scatter of about 24 ADU: inside
    the 25 ADU threshold of a 2500 ADU sky, outside the 20 ADU floor of a dark
    one. The planted pixel is in the (0, 0) phase, and only that phase's sky
    sets its threshold: bright neighbouring phases do not loosen it.
    """
    x, y = STATIC_PIXELS["large_offset"]
    planted = [16 if i % 2 else -16 for i in range(STATIC_MAP_FRAMES)]
    bright_threshold = STATIC_SCATTER_SKY_FRACTION * BRIGHT_SKY
    assert STATIC_SCATTER_FLOOR_ADU < _residual_scatter(planted) <= bright_threshold
    frames = []
    for i, d in enumerate(planted):
        f = _phase_frame(levels, seed=300 + i)
        _set_residual(f, x, y, d=d)
        frames.append(f)
    result = _static_map(frames)
    expected = [[y, x]] if kept else []
    np.testing.assert_array_equal(np.argwhere(result).tolist(), expected)


def test_static_map_logs_both_counts_at_info(caplog):
    """A built map logs one INFO line with its size and the diagonal control's."""
    with caplog.at_level(logging.DEBUG, logger="bandaid.seestar"):
        _static_map(_static_frames())
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    info = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(info) == 1
    message = info[0].getMessage()
    assert f"{len(STATIC_KEPT)} pixels" in message
    assert "diagonal control 0" in message


def test_static_map_flat_sky_fails_the_diagonal_control(caplog):
    """
    On a noise-free sky every residual is constant, so the map is switched off.

    The diagonal neighbours, which no firmware uses, match as often as the axis
    ones; the WARNING names both counts and the map is empty.
    """
    frames = [_frame(stars=(), noise=0) for _ in range(STATIC_MAP_FRAMES)]
    with caplog.at_level(logging.DEBUG, logger="bandaid.seestar"):
        result = _static_map(frames)
    np.testing.assert_array_equal(result, np.zeros(SHAPE, dtype=bool))
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    n_interior = (SHAPE[0] - 4) * (SHAPE[1] - 4)
    counts = re.findall(r"\d+", warnings[0].getMessage())
    assert counts == [str(n_interior)] * 2
    assert not [r for r in caplog.records if r.levelno == logging.INFO]


# Diagonal-only matches for the control: (x, y) pixels, well clear of the
# STATIC_PIXELS and of each other's neighbours.
DIAGONAL_PIXELS = [(x, y) for y in (46, 54) for x in range(6, 60, 6)]


def _set_diagonal_residual(frame, x, y, d):
    """Make the diagonal residual at ``(x, y)`` exactly ``d`` via its NW neighbour."""
    others = sum(int(frame[y + dy, x + dx]) for dy, dx in ((-2, 2), (2, -2), (2, 2)))
    frame[y - 2, x - 2] = 4 * int(frame[y, x]) + d - others
    return frame


@pytest.mark.parametrize(
    ("n_diag", "switched_off"),
    [
        (_CONTROL_MAX_PIXELS, False),
        (_CONTROL_MAX_PIXELS + 1, True),
    ],
)
def test_static_map_diagonal_control_pixel_limit(caplog, n_diag, switched_off):
    """
    A short axis list is kept up to exactly 10 diagonal matches, not 11.

    With 3 axis pixels the fraction limit is far below 1, so the pixel limit
    decides.
    """
    frames = _static_frames(k=STATIC_MAP_FRAMES)
    for f in frames:
        for x, y in DIAGONAL_PIXELS[:n_diag]:
            _set_diagonal_residual(f, x, y, d=300)
    with caplog.at_level(logging.DEBUG, logger="bandaid.seestar"):
        result = _static_map(frames)
    if switched_off:
        assert not result.any()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        counts = re.findall(r"\d+", warnings[0].getMessage())
        assert counts == [str(n_diag), str(len(STATIC_KEPT))]
    else:
        np.testing.assert_array_equal(np.argwhere(result).tolist(), _yx(STATIC_KEPT))
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        info = [r for r in caplog.records if r.levelno == logging.INFO]
        assert f"diagonal control {n_diag})" in info[0].getMessage()


# A frame tall and wide enough for an axis list of more than 1000 pixels: a
# lattice 4 px apart in x and 6 px in y (so that no planted pixel's adjusted
# north neighbour is another's neighbour) over the upper rows, and room below
# it for diagonal matches.
LONG_LIST_SHAPE = (240, 200)
LONG_LIST_PIXELS = [(x, y) for y in range(4, 180, 6) for x in range(4, 197, 4)]
LONG_LIST_DIAGONAL = [(x, y) for y in (200, 210, 220) for x in range(8, 192, 8)]


@pytest.mark.parametrize("extra", [0, 1])
def test_static_map_diagonal_control_fraction_limit(caplog, extra):
    """
    On a long axis list the limit is 1 % of it, not 10 pixels.

    1470 axis pixels allow 14 diagonal matches; 15 switch the map off.
    """
    n_axis = len(LONG_LIST_PIXELS)
    limit = int(_CONTROL_MAX_FRACTION * n_axis)
    assert limit > _CONTROL_MAX_PIXELS
    n_diag = limit + extra
    frames = []
    for i in range(STATIC_MAP_FRAMES):
        f = _frame(stars=(), seed=600 + i, noise=STATIC_NOISE, shape=LONG_LIST_SHAPE)
        for x, y in LONG_LIST_PIXELS:
            _set_residual(f, x, y, d=300)
        for x, y in LONG_LIST_DIAGONAL[:n_diag]:
            _set_diagonal_residual(f, x, y, d=300)
        frames.append(f)
    with caplog.at_level(logging.DEBUG, logger="bandaid.seestar"):
        result = _static_map(frames)
    if extra:
        assert not result.any()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        counts = re.findall(r"\d+", warnings[0].getMessage())
        assert counts == [str(n_diag), str(n_axis)]
    else:
        expected = sorted([y, x] for x, y in LONG_LIST_PIXELS)
        np.testing.assert_array_equal(np.argwhere(result).tolist(), expected)


# Levels of the four 2x2 phases, indexed [row % 2][column % 2], all distinct.
PHASE_LEVELS = ((100, 400), (300, 50))


@pytest.mark.parametrize("shape", [SHAPE, (SHAPE[0] + 1, SHAPE[1] + 1)])
def test_phase_skies_are_the_four_bayer_phase_medians(shape):
    """
    The sky of each 2x2 phase is that phase's own median.

    Phases are counted in frame coordinates, on even and odd frame sizes.
    """
    frame = np.empty(shape, dtype=np.uint16)
    for py in (0, 1):
        for px in (0, 1):
            frame[py::2, px::2] = PHASE_LEVELS[py][px]
    skies = _phase_skies(frame)
    assert skies.shape == (2, 2)
    np.testing.assert_array_equal(skies, PHASE_LEVELS)


def _phase_frame(levels, *, seed, noise=STATIC_NOISE, shape=SHAPE):
    """
    Build a textured sky whose four 2x2 phases sit on their own levels.

    ``levels[py][px]`` is the sky of the pixels at frame rows ``py`` and columns
    ``px`` modulo 2, as on a Bayer sensor under a coloured sky.
    """
    f = _frame(stars=(), seed=seed, noise=noise, sky=0, dtype=np.int32, shape=shape)
    f += np.tile(np.asarray(levels, dtype=np.int32), (shape[0] // 2, shape[1] // 2))
    return f.astype(np.uint16)


def test_static_map_proportional_residual_under_a_changing_sky_colour():
    """
    A residual proportional to its own phase's sky is kept when the colour shifts.

    Over the batch the (0, 0) phase brightens from 500 to 3000 ADU while the
    (1, 1) phase fades from 2500 to 500, as when the sky colour changes in
    twilight. The planted residual at a (1, 1) pixel is 6 % of its own phase's
    sky. It scatters beyond the ceiling raw, and also when scaled by a single
    sky per frame, the brightest phase's; scaled by its own phase's sky, it is
    constant.
    """
    k = STATIC_MAP_FRAMES
    bright = np.linspace(500, 3000, k).round().astype(int)
    fading = np.linspace(2500, 500, k).round().astype(int)
    planted = np.array([round(0.06 * sky) for sky in fading])
    brightest = np.maximum(bright, fading)
    one_sky_ceiling = max(
        STATIC_SCATTER_FLOOR_ADU, STATIC_SCATTER_SKY_FRACTION * np.median(brightest)
    )
    one_sky = planted * np.median(brightest) / brightest
    assert _residual_scatter(planted) > one_sky_ceiling
    assert _residual_scatter(one_sky) > one_sky_ceiling

    x, y = 11, 11
    frames = []
    for i in range(k):
        f = _phase_frame(((bright[i], SKY), (SKY, fading[i])), seed=800 + i)
        _set_residual(f, x, y, d=int(planted[i]))
        frames.append(f)
    result = _static_map(frames)
    np.testing.assert_array_equal(np.argwhere(result).tolist(), [[y, x]])


def test_static_map_dark_phase_is_left_to_the_diagonal_control(caplog):
    """
    A Bayer phase whose sky is 0 is tested on its raw residual alone.

    On some units one colour plane reads 0 on a dark sky. There is no sky to
    scale by, and the plane is flat, so every residual in it is constant on
    both kernels; the diagonal control sees them and switches the map off.
    """
    frames = []
    for i in range(STATIC_MAP_FRAMES):
        f = _phase_frame(((SKY, SKY), (SKY, SKY)), seed=900 + i)
        f[1::2, 0::2] = 0
        frames.append(f)
    with caplog.at_level(logging.DEBUG, logger="bandaid.seestar"):
        result = _static_map(frames)
    assert not result.any()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    # Interior rows 3, 5, ..., 61 and columns 2, 4, ..., 60 of the dark phase.
    n_dark = 30 * 30
    counts = re.findall(r"\d+", warnings[0].getMessage())
    assert counts == [str(n_dark)] * 2


@pytest.mark.parametrize(("n_broken", "kept"), [(9, True), (11, False)])
def test_static_map_tolerates_a_minority_of_outlier_frames(n_broken, kept):
    """
    A constant residual survives being broken in a minority of the frames.

    A star crossing the pixel, say, throws its residual far off in a few frames;
    the median-based scatter ignores up to half of them, but not more.
    """
    rng = np.random.default_rng(7)
    x, y = STATIC_PIXELS["large_offset"]
    frames = []
    for i in range(STATIC_MAP_FRAMES):
        f = _frame(stars=(), seed=700 + i, noise=STATIC_NOISE)
        d = 300 + int(rng.integers(1000, 5000)) if i < n_broken else 300
        _set_residual(f, x, y, d=d)
        frames.append(f)
    result = _static_map(frames)
    expected = [[y, x]] if kept else []
    np.testing.assert_array_equal(np.argwhere(result).tolist(), expected)


# (x, y) centre of a stationary saturated star, clear of the STATIC_PIXELS.
PLATEAU_CENTRE = (48, 48)


@pytest.mark.parametrize("half", [3, 4])
def test_static_map_ignores_a_stationary_saturated_core(caplog, half):
    """
    A clipped core that stays put is neither mapped nor a diagonal match.

    Inside a plateau at the ADC ceiling every residual is 0 in every frame, on
    both kernels. A 7x7 plateau would add 9 pixels to the map; a 9x9 one adds
    25 diagonal matches and would switch the map off.
    """
    cx, cy = PLATEAU_CENTRE
    frames = _static_frames(k=STATIC_MAP_FRAMES)
    for f in frames:
        f[cy - half : cy + half + 1, cx - half : cx + half + 1] = 65535
    with caplog.at_level(logging.DEBUG, logger="bandaid.seestar"):
        result = _static_map(frames)
    np.testing.assert_array_equal(np.argwhere(result).tolist(), _yx(STATIC_KEPT))
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    info = [r for r in caplog.records if r.levelno == logging.INFO]
    assert "diagonal control 0)" in info[0].getMessage()


@pytest.mark.parametrize(
    ("n_saturated", "kept"),
    [(STATIC_MAP_FRAMES // 2 - 1, True), (STATIC_MAP_FRAMES // 2, False)],
)
def test_static_map_saturation_mask_needs_half_the_frames(n_saturated, kept):
    """
    A pixel is masked when its kernel reaches saturation in half the frames.

    A diagonal neighbour of a constant-residual pixel is set to
    `SATURATED_ADU`, which leaves its axis residual alone: a star passing
    by in a minority of the frames costs a real list pixel nothing, one that
    sits there in half of them masks it.
    """
    x, y = STATIC_PIXELS["large_offset"]
    frames = _static_frames(k=STATIC_MAP_FRAMES)
    for f in frames[:n_saturated]:
        f[y + 2, x + 2] = SATURATED_ADU
    result = _static_map(frames)
    names = STATIC_KEPT if kept else STATIC_KEPT - {"large_offset"}
    np.testing.assert_array_equal(np.argwhere(result).tolist(), _yx(names))


@pytest.mark.parametrize(
    ("k", "count"),
    [
        (STATIC_MAP_MIN_FRAMES - 1, f"{STATIC_MAP_MIN_FRAMES - 1} frames"),
        # 10 frames leave chance matches on real units.
        (10, "10 frames"),
        (1, "1 frame "),
    ],
)
def test_static_map_too_few_frames_logs_info_and_is_empty(caplog, k, count):
    """
    Below the minimum frame count the map is empty and one INFO line says so.

    A short batch (a single frame on the command line, say) is ordinary, so
    nothing is logged at WARNING; the INFO line names the count, pluralised.
    """
    frames = _static_frames(k=k)
    with caplog.at_level(logging.DEBUG, logger="bandaid.seestar"):
        result = _static_map(frames)
    np.testing.assert_array_equal(result, np.zeros(SHAPE, dtype=bool))
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    info = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(info) == 1
    message = info[0].getMessage()
    assert f"Only {count}" in message
    assert "no static map is used" in message


# Interior rows 2 * 32 + 7: two full row blocks and a partial one, and not a
# multiple of the block size.
TALL_SHAPE = (2 * 32 + 7 + 4, 64)

# (x, y) of the constant-residual pixels in TALL_SHAPE frames. Interior row
# y - 2 falls in block (y - 2) // 32: rows 2..33 are the first block, 34..65
# the second and 66..72 the partial third. Columns differ so that no pixel's
# planted neighbour is another's neighbour.
BLOCK_PIXELS = (
    (10, 10),  # first block
    (20, 33),  # last row of the first block
    (30, 34),  # first row of the second block
    (40, 65),  # last row of the second block
    (50, 66),  # first row of the partial block
    (12, 69),  # inside the partial block
    (24, 72),  # last interior row
)


def _tall_frames(k=STATIC_MAP_FRAMES):
    frames = []
    for i in range(k):
        f = _frame(stars=(), seed=400 + i, noise=STATIC_NOISE, shape=TALL_SHAPE)
        for x, y in BLOCK_PIXELS:
            _set_residual(f, x, y, d=300)
        frames.append(f)
    return frames


def test_static_map_across_row_blocks(monkeypatch):
    """
    Constant-residual pixels are found in every row block, at both block edges.

    The map is the same as when the whole frame is reduced as one block.
    """
    frames = _tall_frames()
    result = _static_map(frames)
    expected = sorted([y, x] for x, y in BLOCK_PIXELS)
    np.testing.assert_array_equal(np.argwhere(result).tolist(), expected)

    monkeypatch.setattr(seestar, "_ROW_BLOCK", 10 * TALL_SHAPE[0])
    np.testing.assert_array_equal(_static_map(frames), result)


def test_static_map_never_holds_every_residual_plane():
    """
    The peak memory of the static map stays well below a full residual stack.

    Holding every frame's residuals at once would take N float32 planes of the
    frame interior, about 9.2 MB here; reducing a block of rows at a time needs
    only a few copies of an N-frame block, whatever the frame height. The
    frames themselves are allocated before tracing starts.
    """
    if tracemalloc.is_tracing():
        pytest.skip("tracemalloc is already on, so its peak is not this call's")
    shape = (1200, 100)
    frames = []
    for i in range(STATIC_MAP_FRAMES):
        f = _frame(stars=(), seed=500 + i, noise=STATIC_NOISE, shape=shape)
        _set_residual(f, 10, 10, d=300)
        frames.append(f)
    full_stack = STATIC_MAP_FRAMES * (shape[0] - 4) * (shape[1] - 4) * 4

    tracemalloc.start()
    try:
        result = _static_map(frames)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    np.testing.assert_array_equal(np.argwhere(result).tolist(), [[10, 10]])
    assert peak < full_stack / 2


def test_static_map_float_frames_are_all_false():
    """Float frames give an empty static map."""
    frames = _static_frames(dtype=np.float64)
    np.testing.assert_array_equal(_static_map(frames), np.zeros(SHAPE, dtype=bool))


# --------------------------------------------------------------------------
# core_replaced_flags
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pattern", "x", "y", "channel"),
    [
        ("RGGB", 32, 30, "TR"),
        ("RGGB", 33, 30, "TG"),
        ("RGGB", 32, 31, "TG"),
        ("RGGB", 33, 31, "TB"),
        ("GRBG", 32, 30, "TG"),
        ("GRBG", 33, 30, "TR"),
        ("GRBG", 32, 31, "TB"),
        ("GRBG", 33, 31, "TG"),
    ],
)
def test_replaced_core_flags_only_its_channel(pattern, x, y, channel):
    """A replaced core pixel flags its own channel and no other, for either pattern."""
    raw = _plant(_frame(stars=((x, y),)), x, y)
    flags = _flags(raw, [(x, y)], pattern)
    _assert_flags(flags, {ch: [ch == channel] for ch in CHANNELS})


@pytest.mark.parametrize("centre", [(32.0, 30.0), (32.49, 30.49), (32.5, 29.5)])
@pytest.mark.parametrize("pattern", ["RGGB", "GRBG"])
def test_untouched_star_not_flagged(centre, pattern):
    """A star with no replacement is never flagged."""
    raw = _frame(stars=(centre,))
    _assert_flags(_flags(raw, [centre], pattern), {ch: [False] for ch in CHANNELS})


@pytest.mark.parametrize(
    ("dx", "dy", "expected"),
    [
        (1, 0, True),
        (0, -1, True),
        (1, 1, True),
        (-1, 1, True),
        (2, 0, False),
        (0, -2, False),
        (2, 1, False),
    ],
)
def test_replacement_offset_from_centroid(dx, dy, expected):
    """Only replacements within the core radius of the centroid count."""
    # The RGGB channel of the planted pixel is the only one that may flag.
    x0, y0 = 32, 30
    raw = _plant(_frame(stars=((x0, y0),)), x0 + dx, y0 + dy)
    channel = _channel_of("RGGB", x0 + dx, y0 + dy)
    flags = _flags(raw, [(x0, y0)])
    _assert_flags(flags, {ch: [expected and ch == channel] for ch in CHANNELS})


def _channel_of(pattern, x, y):
    return "T" + pattern[2 * (y % 2) + (x % 2)]


@pytest.mark.parametrize(
    ("centre", "pixel"),
    [
        # Centroid just short of a pixel boundary; replacement at its nearest pixel.
        ((32.49, 30.49), (32, 30)),
        # Nearest pixel is 33, but the replaced pixel 32 is still within the core.
        ((32.6, 30.0), (32, 30)),
        ((31.5, 30.5), (32, 31)),
    ],
)
def test_sub_pixel_centroid(centre, pixel):
    """Sub-pixel centroids near pixel boundaries still find the replaced pixel."""
    raw = _plant(_frame(stars=(centre,)), *pixel)
    channel = _channel_of("RGGB", *pixel)
    _assert_flags(_flags(raw, [centre]), {ch: [ch == channel] for ch in CHANNELS})


def test_several_stars_are_flagged_independently():
    """Each star's flag depends only on its own core."""
    stars = [(20.0, 20.0), (44.0, 44.0), (21.0, 44.0)]
    raw = _plant(_frame(stars=stars), 21, 44)
    raw = _plant(raw, 20, 20)
    flags = _flags(raw, stars)
    # (20, 20) is R and (21, 44) is G in RGGB.
    _assert_flags(
        flags,
        {"TR": [True, False, False], "TG": [False, False, True], "TB": [False] * 3},
    )


@pytest.mark.parametrize(
    ("peak", "sky", "expected"),
    [
        (STAR_PEAK + SKY, SKY, True),
        (SKY + MIN_PEAK_ABOVE_SKY_ADU, SKY, True),
        (SKY + MIN_PEAK_ABOVE_SKY_ADU - 1, SKY, False),
        (SATURATED_ADU - 1, SKY, True),
        (SATURATED_ADU, SKY, False),
        (60000, SKY, False),
    ],
)
def test_brightness_and_saturation_gates(peak, sky, expected):
    """Faint and saturated stars are not flagged even with a replaced core."""
    raw = _plant(_frame(), 32, 30)
    flags = _flags(raw, [(32.0, 30.0)], peak=peak, sky=sky)
    _assert_flags(flags, {ch: [expected and ch == "TR"] for ch in CHANNELS})


def test_gates_are_read_per_channel():
    """
    Each channel's flag is gated on that channel's own peak and sky.

    A replaced TR core flags TR when only the TR peak clears the brightness
    gate; with only the TG peak clearing it, nothing is flagged.
    """
    x, y = 32, 30
    raw = _plant(_frame(stars=((x, y),)), x, y)
    bright = SKY + MIN_PEAK_ABOVE_SKY_ADU
    faint = bright - 1
    bkgd_count = {ch: np.full(1, SKY, dtype=float) for ch in CHANNELS}

    def flags_for(peaks):
        peak_count = {ch: np.full(1, peaks[ch], dtype=float) for ch in CHANNELS}
        return core_replaced_flags(
            raw,
            [(x, y)],
            _bayer_masks("RGGB"),
            peak_count,
            bkgd_count,
            saturated_adu=SATURATED_ADU,
        )

    _assert_flags(
        flags_for({"TR": bright, "TG": faint, "TB": faint}),
        {"TR": [True], "TG": [False], "TB": [False]},
    )
    _assert_flags(
        flags_for({"TR": faint, "TG": bright, "TB": faint}),
        {ch: [False] for ch in CHANNELS},
    )


@pytest.mark.parametrize(
    ("dx", "dy", "expected"),
    [(0, 0, True), (1, 0, True), (1, 1, True), (2, 0, False), (2, 2, False)],
)
def test_static_map_pixel(dx, dy, expected):
    """A static-map pixel within the core flags its channel without a dynamic hit."""
    # No dynamic replacement planted: only the static map can flag.
    x0, y0 = 32, 30
    static_map = np.zeros(SHAPE, dtype=bool)
    static_map[y0 + dy, x0 + dx] = True
    channel = _channel_of("RGGB", x0 + dx, y0 + dy)
    flags = _flags(_frame(), [(x0, y0)], static_map=static_map)
    _assert_flags(flags, {ch: [expected and ch == channel] for ch in CHANNELS})


def test_static_map_respects_gates():
    """A static-map hit is still subject to the saturation gate."""
    static_map = np.zeros(SHAPE, dtype=bool)
    static_map[30, 32] = True
    flags = _flags(_frame(), [(32.0, 30.0)], peak=SATURATED_ADU, static_map=static_map)
    _assert_flags(flags, {ch: [False] for ch in CHANNELS})


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (0, 30, False),
        (1, 30, False),
        (2, 30, True),
        (61, 30, True),
        (62, 30, False),
        (63, 30, False),
        (30, 0, False),
        (30, 1, False),
        (30, 62, False),
        (30, 63, False),
    ],
)
def test_outer_edge_never_flagged(x, y, expected):
    """Outer-2-px replacements never count, even when clipping makes them exact."""
    raw = _plant(_frame(stars=((x, y),)), x, y)
    channel = _channel_of("RGGB", x, y)
    flags = _flags(raw, [(x, y)])
    _assert_flags(flags, {ch: [expected and ch == channel] for ch in CHANNELS})


def test_static_map_on_outer_edge_never_flagged():
    """Static-map pixels in the outer 2 px never count."""
    static_map = np.ones(SHAPE, dtype=bool)
    static_map[2:-2, 2:-2] = False
    flags = _flags(_frame(stars=((1, 30),)), [(1.0, 30.0)], static_map=static_map)
    _assert_flags(flags, {ch: [False] for ch in CHANNELS})


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_frame_gets_no_flags(dtype):
    """A float frame gets no flags in any channel, static map or not."""
    raw = _plant(_frame(), 32, 30).astype(dtype)
    static_map = np.ones(SHAPE, dtype=bool)
    flags = _flags(raw, [(32.0, 30.0)], static_map=static_map)
    _assert_flags(flags, {ch: [False] for ch in CHANNELS})


def test_channel_without_mask_is_omitted():
    """A channel whose mask is None is left out of the result."""
    raw = _plant(_frame(), 32, 30)
    masks = _bayer_masks("RGGB")
    masks["TB"] = None
    peak_count, bkgd_count = _gates(1)
    flags = core_replaced_flags(
        raw,
        np.array([[32.0, 30.0]]),
        masks,
        peak_count,
        bkgd_count,
        saturated_adu=SATURATED_ADU,
    )
    _assert_flags(flags, {"TR": [True], "TG": [False]})


@pytest.mark.parametrize("dtype", [np.uint16, np.float64])
def test_no_stars(dtype):
    """An empty star list gives empty flag arrays for every channel."""
    flags = _flags(_frame().astype(dtype), np.empty((0, 2)))
    _assert_flags(flags, {ch: [] for ch in CHANNELS})


@pytest.mark.parametrize("bad", [(np.nan, 30.0), (32.0, np.nan), (np.nan, np.nan)])
def test_non_finite_centroid_never_flagged(bad):
    """
    A row with a NaN centroid is not flagged; the other rows are unaffected.

    Plane-fit failures leave NaN centroid rows in the pipeline's tables.
    """
    raw = _plant(_frame(stars=((20.0, 20.0),)), 20, 20)
    flags = _flags(raw, [bad, (20.0, 20.0)])
    _assert_flags(
        flags, {"TR": [False, True], "TG": [False, False], "TB": [False, False]}
    )


def test_nan_peak_count_never_flagged():
    """A star whose peak_count is NaN is not flagged, even with a replaced core."""
    raw = _plant(_frame(stars=((20.0, 20.0), (44.0, 44.0))), 20, 20)
    raw = _plant(raw, 44, 44)
    peak_count, bkgd_count = _gates(2)
    peak_count = {ch: np.array([np.nan, STAR_PEAK + SKY]) for ch in CHANNELS}
    flags = core_replaced_flags(
        raw,
        np.array([[20.0, 20.0], [44.0, 44.0]]),
        _bayer_masks("RGGB"),
        peak_count,
        bkgd_count,
        saturated_adu=SATURATED_ADU,
    )
    _assert_flags(
        flags, {"TR": [False, True], "TG": [False, False], "TB": [False, False]}
    )
