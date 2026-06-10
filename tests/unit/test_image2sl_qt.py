"""
Unit tests for :mod:`bandaid.image2sl_qt`.

Covers Bayer-mask generation (``generate_bayer_masks``) -- the optional
``append_l4`` luminance-channel entry and the ``roworder``/``ybayroff``
re-jumbling of the CFA pattern -- and the per-channel ``bayer_balance_image``
flattening of a raw Bayer frame.
"""

import numpy as np
import pytest

from bandaid.image2sl_qt import bayer_balance_image, generate_bayer_masks

SHAPE = (4, 4)
METADATA = {"bayerpat": "RGGB", "roworder": "top-down", "ybayroff": 0}


def test_append_l4_false_by_default():
    """Without ``append_l4`` there is no L4 entry, just the three colors."""
    default = generate_bayer_masks(SHAPE, METADATA)
    explicit_false = generate_bayer_masks(SHAPE, METADATA, append_l4=False)

    assert isinstance(default, dict)
    # Key order is preserved (R, B, G).
    assert list(default) == ["TR", "TB", "TG"]
    assert "L4" not in default
    # Passing the default explicitly behaves the same as omitting it.
    assert list(explicit_false) == list(default)


def test_append_l4_true_appends_entry():
    """With ``append_l4=True`` an "L4" -> None entry is added last."""
    default = generate_bayer_masks(SHAPE, METADATA)
    with_l4 = generate_bayer_masks(SHAPE, METADATA, append_l4=True)

    assert len(with_l4) == len(default) + 1
    assert "L4" in with_l4
    assert with_l4["L4"] is None
    # The L4 entry comes last so the RGB masks are available before it.
    assert list(with_l4)[-1] == "L4"


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
# pattern after the documented roworder/ybayroff re-jumbling, so the assertions
# do not just re-derive the function's own arithmetic:
#   - top-down,  ybayroff=0  -> pattern "RGGB"
#   - bottom-up, ybayroff=0  -> pattern "GBRG" (top/bottom rows swapped)
#   - top-down,  ybayroff!=0 -> pattern "GRBG" (columns swapped within rows)
#   - bottom-up, ybayroff!=0 -> pattern "BGGR" (both swaps)
@pytest.mark.parametrize(
    ("roworder", "ybayroff", "expected"),
    [
        (
            "top-down",
            0,
            {"TR": [(0, 0)], "TG": [(0, 1), (1, 0)], "TB": [(1, 1)]},
        ),
        (
            "bottom-up",
            0,
            {"TR": [(1, 0)], "TG": [(0, 0), (1, 1)], "TB": [(0, 1)]},
        ),
        (
            "top-down",
            1,
            {"TR": [(0, 1)], "TG": [(0, 0), (1, 1)], "TB": [(1, 0)]},
        ),
        (
            "bottom-up",
            1,
            {"TR": [(1, 1)], "TG": [(0, 1), (1, 0)], "TB": [(0, 0)]},
        ),
    ],
)
def test_generate_bayer_masks_roworder_ybayroff(roworder, ybayroff, expected):
    """roworder/ybayroff re-jumble the CFA pattern onto the right sub-lattice."""
    metadata = {"bayerpat": "RGGB", "roworder": roworder, "ybayroff": ybayroff}
    masks = generate_bayer_masks(SHAPE, metadata)

    assert set(masks) == set(expected)
    for color, positions in expected.items():
        _assert_valid_positions(masks[color], positions)


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
