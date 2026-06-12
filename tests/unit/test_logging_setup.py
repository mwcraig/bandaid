"""
Unit tests for :func:`bandaid.configure_logging`.

The library configures only the ``bandaid`` logger (never the root logger), can
route records to a file, and is idempotent so repeated calls in a notebook do
not duplicate output.
"""

import logging

import pytest

from bandaid import configure_logging


@pytest.fixture(autouse=True)
def _reset_bandaid_logger():
    """Restore the ``bandaid`` logger to a clean state around each test."""
    logger = logging.getLogger("bandaid")
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    yield
    for handler in list(logger.handlers):
        if handler not in saved_handlers:
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(saved_level)


def _managed_handlers(logger):
    """Return the handlers ``configure_logging`` is responsible for."""
    return [h for h in logger.handlers if getattr(h, "_bandaid_managed", False)]


def test_writes_records_to_logfile(tmp_path):
    """A record emitted after configuration lands in the requested file."""
    logfile = tmp_path / "run.log"
    configure_logging(logging.DEBUG, logfile=str(logfile), stream=False)

    logging.getLogger("bandaid.test").warning("hello-from-test")

    assert logfile.exists()
    assert "hello-from-test" in logfile.read_text()


def test_only_touches_bandaid_logger(tmp_path):
    """Root logging is left untouched; only the bandaid logger is configured."""
    root = logging.getLogger()
    root_handlers_before = list(root.handlers)
    root_level_before = root.level

    configure_logging(logging.DEBUG, logfile=str(tmp_path / "run.log"))

    assert list(root.handlers) == root_handlers_before
    assert root.level == root_level_before
    assert logging.getLogger("bandaid").level == logging.DEBUG


def test_idempotent_no_duplicate_handlers(tmp_path):
    """Repeated calls replace, not stack, the managed handlers."""
    logger = logging.getLogger("bandaid")

    configure_logging(logging.INFO, logfile=str(tmp_path / "run.log"))
    count_after_first = len(_managed_handlers(logger))

    configure_logging(logging.INFO, logfile=str(tmp_path / "run.log"))
    count_after_second = len(_managed_handlers(logger))

    assert count_after_first == count_after_second
