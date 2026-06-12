"""
Opt-in logging configuration for the ``bandaid`` package.

The library itself only *emits* log records (each module logs through its own
``logging.getLogger(__name__)``) and never configures handlers, so it does not
impose logging policy on the host application. :func:`configure_logging` is a
convenience for notebooks and scripts that want bandaid's records routed
somewhere -- the console, a file, or both -- without touching the root logger.
"""

import logging

__all__ = ["configure_logging"]

#: Name of the package-level logger that :func:`configure_logging` configures.
_PACKAGE_LOGGER = "bandaid"

#: Name of the attribute stamped on handlers this module owns, so repeat calls
#: can replace, not stack, them.
_MANAGED_ATTR = "_bandaid_managed"

_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def _clear_managed_handlers(logger):
    """Remove (and close) only the handlers this module previously added."""
    for handler in list(logger.handlers):
        if getattr(handler, _MANAGED_ATTR, False):
            logger.removeHandler(handler)
            handler.close()


def configure_logging(level=logging.INFO, *, logfile=None, stream=True):
    """
    Route ``bandaid`` log records to the console and/or a file.

    Configures only the package-level ``bandaid`` logger -- never the root
    logger -- so it does not interfere with the host application's logging. The
    call is idempotent: handlers it added on a previous call are replaced rather
    than stacked, so calling it repeatedly (e.g. in a notebook) does not
    duplicate output.

    Parameters
    ----------
    level : int, optional
        Logging level for the ``bandaid`` logger. Default ``logging.INFO``.
    logfile : str or pathlib.Path or None, optional
        If given, add a `~logging.FileHandler` writing to this path. Default
        None (no file handler).
    stream : bool, optional
        If True (default), add a `~logging.StreamHandler` writing to stderr.

    Returns
    -------
    logging.Logger
        The configured ``bandaid`` logger.
    """
    logger = logging.getLogger(_PACKAGE_LOGGER)
    logger.setLevel(level)
    _clear_managed_handlers(logger)

    formatter = logging.Formatter(_FORMAT)
    handlers = []
    if stream:
        handlers.append(logging.StreamHandler())
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile))

    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _MANAGED_ATTR, True)
        logger.addHandler(handler)

    return logger
