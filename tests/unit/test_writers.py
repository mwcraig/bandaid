"""
Unit tests for the pluggable frame-writer registry in :mod:`bandaid.writers`.

A frame writer turns one frame's ``{filter: Table}`` photometry result into an
on-disk record. The registry exposes the bundled ``starlist`` writer by name and
lets a user register their own, so the batch driver (and the CLI) can pick an
output format without editing the write loop. These tests pin the registry
contract (resolve by name, register/override, unknown-name error) and that the
default ``starlist`` writer still emits a valid ``StarListSet`` document.
"""

import pytest
from aavso_starlist_schema import StarListSet

from bandaid import writers
from bandaid.writers import (
    available_writers,
    get_writer,
    register_writer,
    write_starlist_set,
)


@pytest.fixture(autouse=True)
def _isolate_registry(isolate_registry):
    """
    Restore the in-process writer registry after each test.

    ``register_writer`` mutates a module-level dict, so without this a registered
    writer would leak into later tests (e.g. the membership check on
    ``available_writers``). Delegates to the shared ``isolate_registry`` factory.
    """
    with isolate_registry(writers, "_WRITERS"):
        yield


class TestGetWriter:
    """``get_writer`` resolves a registered writer by name."""

    def test_starlist_resolves_to_default(self):
        """The bundled ``starlist`` name resolves to ``write_starlist_set``."""
        assert get_writer("starlist") is write_starlist_set

    def test_unknown_writer_raises(self):
        """An unregistered name raises rather than guessing a format."""
        with pytest.raises(ValueError, match="no-such-format"):
            get_writer("no-such-format")


class TestAvailableWriters:
    """``available_writers`` lists the registered writers."""

    def test_lists_the_bundled_writer(self):
        """The ``starlist`` writer ships and is always listed."""
        assert "starlist" in available_writers()


class TestRegister:
    """A user can register a custom writer and resolve it back by name."""

    def test_register_then_get(self):
        """A registered writer is returned by ``get_writer`` and listed."""

        def my_writer(_frame_result, output_path):
            return output_path

        register_writer("mine", my_writer)
        assert get_writer("mine") is my_writer
        assert "mine" in available_writers()

    def test_reregister_overrides(self):
        """Re-registering a name replaces the previous writer."""

        def replacement(_frame_result, output_path):
            return output_path

        register_writer("starlist", replacement)
        assert get_writer("starlist") is replacement


class TestWriteStarlistSet:
    """The default writer emits a valid ``StarListSet`` and returns its path."""

    def test_writes_valid_starlistset_and_returns_path(self, tmp_path, by_filter):
        """One StarList per filter, stars intact, the written path returned."""
        output_path = tmp_path / "frame1.star"
        frame_result = by_filter()

        returned = write_starlist_set(frame_result, output_path)

        assert returned == output_path
        star_list_set = StarListSet.model_validate_json(output_path.read_text())
        assert len(star_list_set.star_lists) == len(frame_result)
        for star_list in star_list_set.star_lists:
            kept_x = sorted(item.x for item in star_list.staritems)
            assert kept_x == [20.0, 70.0]
