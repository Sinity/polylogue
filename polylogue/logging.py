"""Structured logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Any, TextIO

import structlog
from structlog.types import Processor


class _StderrProxy:
    """File-like proxy that always delegates to the current sys.stderr.

    structlog's PrintLoggerFactory captures the file object at creation
    time and caches the logger. If tests redirect sys.stderr, the cached
    logger's file handle becomes stale (closed). This proxy avoids that
    by always reading sys.stderr at write time.
    """

    def write(self, s: str) -> int:
        return sys.stderr.write(s)

    def flush(self) -> None:
        sys.stderr.flush()

    def isatty(self) -> bool:
        return sys.stderr.isatty()

    def fileno(self) -> int:
        return sys.stderr.fileno()


_stderr_proxy: TextIO = _StderrProxy()  # type: ignore[assignment]


# Configure structlog
def configure_logging(verbose: bool = False, json_logs: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO

    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=sys.stderr.isatty(),
            )
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=_stderr_proxy),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)
