"""
Logging configuration for Ruby.

Provides a simple stdlib-based logging setup suitable for lightweight
deployments (e.g. Raspberry Pi).  Call ``setup_logging()`` once at
process startup — every subsequent ``get_logger()`` call will return
a standard ``logging.Logger``.

Usage::

    from lib.logging_config import setup_logging, get_logger

    setup_logging(level="INFO")
    logger = get_logger("engine")

    logger.info("engine started, account_size=%d, interval=%s", 150_000, "5m")
"""

import logging
import os
import sys
from typing import Any

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(
    *,
    level: str | None = None,
    service: str = "futures",
) -> None:
    """Configure stdlib ``logging`` for the whole process.

    Parameters
    ----------
    level:
        Root log level.  Falls back to the ``LOG_LEVEL`` env var, then
        ``"INFO"``.
    service:
        Accepted for backward compatibility but unused in this
        lightweight implementation.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    numeric_level = getattr(logging, level, logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    root_logger = logging.getLogger()
    # Preserve handlers that are NOT StreamHandlers (e.g. _RingBufferHandler
    # from trainer_server) to avoid dropping or duplicating them.
    _keep = [
        h for h in root_logger.handlers if not isinstance(h, logging.StreamHandler)
    ]
    root_logger.handlers.clear()
    for h in _keep:
        root_logger.addHandler(h)
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    # Quiet down noisy third-party loggers
    for noisy in (
        "uvicorn.access",
        "httpx",
        "httpcore",
        "websockets",
        "urllib3",
        "sqlalchemy.engine",
        "hmmlearn",
        "matplotlib",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None, **initial_binds: Any) -> logging.Logger:
    """Return a stdlib logger.

    Parameters
    ----------
    name:
        Logger name.  When ``None``, returns the root logger.
    **initial_binds:
        Accepted for backward compatibility with the previous
        structlog-based interface but silently ignored.

    Examples
    --------
    >>> logger = get_logger("engine")
    >>> logger.info("refresh complete, bars=%d", 500)
    """
    return logging.getLogger(name)
