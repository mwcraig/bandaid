"""Unit-only fixtures shared across the split ``bandaid.scripts`` test modules."""

import pytest
from _helpers import _CONSISTENT_HEADER

from bandaid import scripts


@pytest.fixture
def _consistent_headers(monkeypatch):
    """
    Stub fits.getheader so every frame passes check_frame_consistency.

    process_batch reads each frame's header unconditionally; the process_batch
    tests use fake paths and exercise the processing/output paths, not the
    consistency check, so return a header that matches _dummy_prep for all of them.
    """
    monkeypatch.setattr(
        scripts.fits, "getheader", lambda _file: dict(_CONSISTENT_HEADER)
    )
