"""Structured logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Any

try:
    import structlog
    from structlog.types import Processor
except ImportError:
    structlog = None  # type: ignore
    Processor = Any  # type: ignore
    import warnings

    warnings.warn("structlog not installed, falling back to standard logging", RuntimeWarning, stacklevel=2)


# Configure structlog
def configure_logging(verbose: bool = False, json_logs: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    if not structlog:
        logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        return

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
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    if structlog:
        return structlog.get_logger(name)
    return logging.getLogger(name)
