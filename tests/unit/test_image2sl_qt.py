"""
Unit tests for :mod:`bandaid.image2sl_qt`.

Covers Bayer-mask generation (``generate_bayer_masks``) -- the optional
``append_l4`` luminance-channel entry and the ``roworder``/``xbayroff``/
``ybayroff`` re-anchoring of the CFA pattern, pinned both to the standard
FITS keyword convention and to Han Kleijn's Bayer conformance images -- and
the per-channel ``bayer_balance_image`` flattening of a raw Bayer frame.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from bandaid.exceptions import DegenerateBayerChannelError
from bandaid.image2sl_qt import bayer_balance_image, generate_bayer_masks

SHAPE = (4, 4)
METADATA = {"bayerpat": "RGGB", "roworder": "top-down", "ybayroff": 0}


def test_append_l4_true_by_default():
    """Without ``append_l4`` an "L4" -> None entry is added last (issue #61)."""
    default = generate_bayer_masks(SHAPE, METADATA)
    explicit_true = generate_bayer_masks(SHAPE, METADATA, append_l4=True)
    without_l4 = generate_bayer_masks(SHAPE, METADATA, append_l4=False)

    assert isinstance(default, dict)
    # Key order is preserved (R, B, G), with L4 last.
    assert list(default) == ["TR", "TB", "TG", "L4"]
    assert default["L4"] is None
    # Passing the default explicitly behaves the same as omitting it.
    assert list(explicit_true) == list(default)
    # append_l4=False still omits the entry entirely.
    assert list(without_l4) == ["TR", "TB", "TG"]
    assert "L4" not in without_l4


def _assert_valid_positions(mask, positions):
    """
    Assert ``mask`` marks exactly ``positions`` as valid (False).

    ``positions`` is an iterable of ``(row_parity, col_parity)`` pairs naming the
    2x2 CFA sub-lattice(s) that should be unmasked for this color; every other
    pixel must be masked (True).
    """
    expected_valid = np.zeros(mask.shape, dtype=bool)
    for row_parity, col_parity in positions:
        expected_valid[row_parity::2, col_parity::2] = True
    # In the returned mask, False means valid/use, True means masked/ignore.
    np.testing.assert_array_equal(~mask, expected_valid)


# Each case pins the expected valid (unmasked) CFA sub-lattice for the RGGB
# pattern under the standard FITS convention (as written by N.I.N.A./SGP and
# read by Siril/ASTAP): the CFA color of stored pixel (x, y) is given by
# ``pattern[(y + YBAYROFF + roworder_flip) % 2][(x + XBAYROFF) % 2]``,
# so YBAYROFF is a ROW offset (same top/bottom-row swap as a bottom-up
# roworder -- the two compose, cancelling when both apply) and XBAYROFF is a
# COLUMN offset (within-row swap). Effective patterns pinned below:
#   - top-down,  x=0, y=0 -> "RGGB"
#   - bottom-up, x=0, y=0 -> "GBRG" (row swap)
#   - top-down,  x=0, y=1 -> "GBRG" (row swap)
#   - bottom-up, x=0, y=1 -> "RGGB" (the two row swaps cancel)
#   - top-down,  x=1, y=0 -> "GRBG" (column swap)
#   - bottom-up, x=1, y=0 -> "BGGR" (row + column swap)
#   - top-down,  x=1, y=1 -> "BGGR" (row + column swap)
#   - bottom-up, x=1, y=1 -> "GRBG" (column swap only, rows cancel)
_RGGB_LATTICE = {"TR": [(0, 0)], "TG": [(0, 1), (1, 0)], "TB": [(1, 1)]}
_GBRG_LATTICE = {"TR": [(1, 0)], "TG": [(0, 0), (1, 1)], "TB": [(0, 1)]}
_GRBG_LATTICE = {"TR": [(0, 1)], "TG": [(0, 0), (1, 1)], "TB": [(1, 0)]}
_BGGR_LATTICE = {"TR": [(1, 1)], "TG": [(0, 1), (1, 0)], "TB": [(0, 0)]}


@pytest.mark.parametrize(
    ("roworder", "xbayroff", "ybayroff", "expected"),
    [
        ("top-down", 0, 0, _RGGB_LATTICE),
        ("bottom-up", 0, 0, _GBRG_LATTICE),
        ("top-down", 0, 1, _GBRG_LATTICE),
        ("bottom-up", 0, 1, _RGGB_LATTICE),
        ("top-down", 1, 0, _GRBG_LATTICE),
        ("bottom-up", 1, 0, _BGGR_LATTICE),
        ("top-down", 1, 1, _BGGR_LATTICE),
        ("bottom-up", 1, 1, _GRBG_LATTICE),
    ],
)
def test_generate_bayer_masks_roworder_offsets(roworder, xbayroff, ybayroff, expected):
    """roworder/xbayroff/ybayroff re-anchor the CFA pattern correctly."""
    metadata = {
        "bayerpat": "RGGB",
        "roworder": roworder,
        "xbayroff": xbayroff,
        "ybayroff": ybayroff,
    }
    masks = generate_bayer_masks(SHAPE, metadata, append_l4=False)

    assert set(masks) == set(expected)
    for color, positions in expected.items():
        _assert_valid_positions(masks[color], positions)


def _cfa_colors(shape, xbayroff=0, ybayroff=0, pattern="RGGB"):
    """
    Color letter of each image pixel per the standard CFA-offset convention.

    Parameters
    ----------
    shape : tuple
        The (ny, nx) shape of the image.
    xbayroff : int, optional
        The FITS ``XBAYROFF`` CFA column offset. Default 0.
    ybayroff : int, optional
        The FITS ``YBAYROFF`` CFA row offset. Default 0.
    pattern : str, optional
        The 4-character Bayer pattern. Default "RGGB".

    Returns
    -------
    numpy.ndarray
        Array of one-letter color codes, one per pixel.
    """
    pat = np.array(list(pattern)).reshape(2, 2)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return pat[(yy + ybayroff) % 2, (xx + xbayroff) % 2]


# Regression tests for issue #51: ground truth is a synthetic CFA color map
# built from the standard convention -- physics, not the function's own
# arithmetic. For RGGB + ybayroff=1 the buggy column swap traded the R and B
# masks exactly.
@pytest.mark.parametrize("ybayroff", [0, 1])
def test_masks_select_the_right_colors_for_ybayroff(ybayroff):
    """Each mask selects only pixels of its own color for a shifted CFA (row offset)."""
    colors = _cfa_colors(SHAPE, ybayroff=ybayroff)
    masks = generate_bayer_masks(
        SHAPE,
        {"bayerpat": "RGGB", "roworder": "top-down", "ybayroff": ybayroff},
        append_l4=False,
    )
    for name, mask in masks.items():
        selected = colors[~mask]  # in the mask, False means use/valid
        assert set(selected) == {name[1]}, (
            f"ybayroff={ybayroff}: {name} mask selects color(s) "
            f"{sorted(set(selected))}, expected {name[1]!r}"
        )


def test_xbayroff_is_honored():
    """Each mask selects only pixels of its own color for a column-offset CFA."""
    colors = _cfa_colors(SHAPE, xbayroff=1)
    masks = generate_bayer_masks(
        SHAPE,
        {"bayerpat": "RGGB", "roworder": "top-down", "ybayroff": 0, "xbayroff": 1},
        append_l4=False,
    )
    for name, mask in masks.items():
        selected = colors[~mask]
        assert set(selected) == {name[1]}, (
            f"xbayroff=1: {name} mask selects color(s) "
            f"{sorted(set(selected))}, expected {name[1]!r}"
        )


# Han Kleijn's Bayer conformance images (hnsky.org / ASTAP,
# https://free-astro.org/download/Bayer_test_pattern_v6.tar.gz) -- the de-facto
# ground truth the demosaicers (Siril, ASTAP, N.I.N.A.) validate against. One
# synthetic RGGB frame of the same chart per ROWORDER x XBAYROFF x YBAYROFF
# combination; committed gzipped (astropy reads .fits.gz transparently).
_BAYER_V6_DIR = Path(__file__).parent.parent / "data" / "bayer_test_pattern_v6"
_BAYER_V6_FILES = [
    "bayer_v6_top-down.fits.gz",
    "bayer_v6_bottom-up.fits.gz",
    "bayer_v6_top-down_x0_y1.fits.gz",
    "bayer_v6_top-down_x1_y0.fits.gz",
    "bayer_v6_top-down_x1_y1.fits.gz",
    "bayer_v6_bottom-up_x0_y1.fits.gz",
    "bayer_v6_bottom-up_x1_y0.fits.gz",
    "bayer_v6_bottom-up_x1_y1.fits.gz",
]
# Interior (top, bottom, left, right) pixel rectangles, in the chart's display
# orientation, of the solid red/green/blue blocks of the conformance chart. In
# a solid single-color block only the CFA pixels of that color are nonzero, so
# the lit pixels inside each rectangle are the physical ground truth for that
# color's mask. Margins of >= 2 px keep the rectangles inside the blocks (and
# clear of the one-pixel scene shift in the offset variants and of the text
# rendered lower in each block).
_BAYER_V6_SOLID_BLOCKS = {
    "TR": (50, 60, 4, 30),
    "TG": (46, 60, 38, 70),
    "TB": (46, 60, 78, 108),
}


@pytest.mark.parametrize("fixture_name", _BAYER_V6_FILES)
def test_generate_bayer_masks_against_han_kleijn_conformance_images(fixture_name):
    """Masks agree with the CFA alignment of Han Kleijn's Bayer test images."""
    with fits.open(_BAYER_V6_DIR / fixture_name) as hdul:
        header = hdul[0].header
        data = hdul[0].data

    # Resolve the metadata straight from the FITS keywords the chart's author
    # wrote; the plain top-down/bottom-up variants omit XBAYROFF/YBAYROFF.
    metadata = {
        "bayerpat": header["BAYERPAT"],
        "roworder": header["ROWORDER"].lower(),
        "xbayroff": header.get("XBAYROFF", 0),
        "ybayroff": header.get("YBAYROFF", 0),
    }
    masks = generate_bayer_masks(data.shape, metadata, append_l4=False)

    n_rows = data.shape[0]
    for color, (top, bottom, left, right) in _BAYER_V6_SOLID_BLOCKS.items():
        row_range = (top, bottom)
        if metadata["roworder"] == "bottom-up":
            # Bottom-up files store the chart vertically flipped; the masks
            # apply to the stored array, so flip the rectangle to match.
            row_range = (n_rows - bottom, n_rows - top)
        block = np.s_[row_range[0] : row_range[1], left:right]
        lit = data[block] > 0

        for name, mask in masks.items():
            valid = ~mask[block]  # in the mask, False means use/valid
            if name == color:
                # Every pixel of this block's color is lit and vice versa.
                np.testing.assert_array_equal(
                    valid,
                    lit,
                    err_msg=(
                        f"{fixture_name}: {name} mask disagrees with the lit "
                        f"pixels of the solid {color[1]} block"
                    ),
                )
            else:
                assert not (valid & lit).any(), (
                    f"{fixture_name}: {name} mask selects lit pixels inside "
                    f"the solid {color[1]} block"
                )


def _make_bayer_imbalanced_image(side=120, seed=20240611):
    """
    Build a raw Bayer frame whose four CFA sub-grids differ in mean and spread.

    Returns the image plus the per-channel ``(mean, stddev)`` used to generate
    it. ``side`` is even so each of the four ``[a::2, b::2]`` sub-grids has
    ``(side / 2) ** 2`` samples.
    """
    rng = np.random.default_rng(seed)
    # Deliberately different background level and noise in each Bayer channel.
    channel_stats = {
        (0, 0): (100.0, 5.0),
        (0, 1): (160.0, 12.0),
        (1, 0): (80.0, 8.0),
        (1, 1): (200.0, 3.0),
    }
    image = np.zeros((side, side), dtype=float)
    for (row_parity, col_parity), (mean, stddev) in channel_stats.items():
        sub = image[row_parity::2, col_parity::2]
        image[row_parity::2, col_parity::2] = rng.normal(mean, stddev, size=sub.shape)
    return image, channel_stats


def _channel_views(image):
    """Return the four CFA sub-grids of ``image`` in a fixed order."""
    return [
        image[0::2, 0::2],
        image[0::2, 1::2],
        image[1::2, 0::2],
        image[1::2, 1::2],
    ]


def test_bayer_balance_image_equalizes_channels():
    """The four CFA channels share a common mean and spread after balancing."""
    image, _ = _make_bayer_imbalanced_image()
    original = image.copy()

    # Pre-condition: the channels really are imbalanced to start with.
    pre_means = [c.mean() for c in _channel_views(original)]
    pre_stds = [c.std() for c in _channel_views(original)]
    min_initial_mean_spread = 50.0
    min_initial_std_spread = 5.0
    assert max(pre_means) - min(pre_means) > min_initial_mean_spread
    assert max(pre_stds) - min(pre_stds) > min_initial_std_spread

    bayer_balance_image(image)

    # The operation mutates the array in place.
    assert not np.allclose(image, original)

    post_means = [c.mean() for c in _channel_views(image)]
    post_stds = [c.std() for c in _channel_views(image)]

    # The documented goal: equal background grayness and equal background noise
    # across the four channels. Tolerances are loose to absorb the sampling
    # scatter from ~3600 pixels per channel.
    assert max(post_means) - min(post_means) == pytest.approx(0, abs=1.0)
    assert max(post_stds) - min(post_stds) == pytest.approx(0, abs=0.5)

    # The common mean/std land on the across-channel targets the function uses.
    assert np.mean(post_means) == pytest.approx(np.mean(pre_means), abs=2.0)
    assert np.mean(post_stds) == pytest.approx(np.mean(pre_stds), abs=1.0)


def test_bayer_balance_image_ignores_out_of_range_pixels():
    """Negative and far-above-background pixels are excluded from the statistics."""
    image, _ = _make_bayer_imbalanced_image()
    # Inject a handful of pixels that the ``0 <= v < mean + 5*std`` window must
    # reject so they do not skew the per-channel mean/std used for balancing.
    image[0, 0] = 1e6  # well above the cutoff (lives in the (0, 0) channel)
    image[0, 2] = -500.0  # negative, below the lower bound

    bayer_balance_image(image)

    # Balancing still runs and the bulk of each channel is brought into line;
    # the outliers do not blow up the result (they would if not excluded).
    post_means = [c.mean() for c in _channel_views(image)]
    # The injected 1e6 pixel lives in the (0, 0) channel; if it had leaked into
    # the statistics that channel's mean would be enormous.
    assert np.isfinite(post_means).all()
    sane_mean_ceiling = 1e4
    assert max(post_means) < sane_mean_ceiling


def test_bayer_balance_image_raises_on_uniform_image():
    """
    A fully uniform (constant) image raises rather than writing inf/NaN.

    Every channel is constant, so its stdev is 0 and the balancing-window
    cutoff (``mean + 5*std``) collapses onto the shared value; the strict
    ``< cutoff`` filter then excludes every pixel, so this hits the "empty
    sample" guard rather than the zero-variance one. Either way,
    ``bayer_balance_image`` must raise instead of dividing by zero and
    silently corrupting the image (issue #61).
    """
    fill_value = 100.0
    image = np.full((8, 8), fill_value)

    with pytest.raises(DegenerateBayerChannelError, match="no pixels"):
        bayer_balance_image(image)

    # The guard must fire before any write-back mutates the array.
    assert np.all(image == fill_value)


def test_bayer_balance_image_raises_on_constant_channel():
    """One constant CFA sub-grid (zero variance) raises, not just a uniform frame."""
    image, _ = _make_bayer_imbalanced_image()
    # Force the (0, 0) sub-grid to a single repeated value -- std() == 0 -- while
    # leaving the other three channels normally distributed.
    image[0::2, 0::2] = 42.0

    with pytest.raises(DegenerateBayerChannelError, match="zero"):
        bayer_balance_image(image)


def test_bayer_balance_image_raises_on_empty_channel_sample():
    """A CFA sub-grid with every pixel excluded by the ``>= 0`` filter raises."""
    image, _ = _make_bayer_imbalanced_image()
    # The (0, 0) sub-grid is entirely negative, so the ``0 <= v < cutoff``
    # balancing window excludes every one of its pixels (an empty sample),
    # independent of the zero-variance case above (issue #61).
    image[0::2, 0::2] = -5.0

    with pytest.raises(DegenerateBayerChannelError, match="no pixels"):
        bayer_balance_image(image)
