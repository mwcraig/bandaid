"""
Unit tests for :mod:`bandaid.image2sl_qt`.

Covers Bayer-mask generation (``generate_bayer_masks``), in particular the
optional ``append_l4`` luminance-channel entry.
"""

from bandaid.image2sl_qt import generate_bayer_masks

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
