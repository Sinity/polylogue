"""Structured logging configuration.

structlog is imported lazily — the default CLI path (no --verbose, no --json-logs)
never pays the ~600ms import cost. Only configure_logging() triggers the import.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable, Iterator
from types import TracebackType
from typing import TYPE_CHECKING, Any, BinaryIO, Protocol, TextIO

from polylogue.config import load_polylogue_config

if TYPE_CHECKING:
    from structlog.types import Processor


class BoundLoggerLike(Protocol):
    """Logger methods used by first-party call sites."""

    def bind(self, **new_values: object) -> BoundLoggerLike: ...

    def debug(self, message: str, *args: object, **event_kw: object) -> object: ...

    def info(self, message: str, *args: object, **event_kw: object) -> object: ...

    def warning(self, message: str, *args: object, **event_kw: object) -> object: ...

    def error(self, message: str, *args: object, **event_kw: object) -> object: ...

    def exception(self, message: str, *args: object, **event_kw: object) -> object: ...


class _StderrProxy(TextIO):
    """File-like proxy that always delegates to the current sys.stderr.

    structlog's PrintLoggerFactory captures the file object at creation
    time and caches the logger. If tests redirect sys.stderr, the cached
    logger's file handle becomes stale (closed). This proxy avoids that
    by always reading sys.stderr at write time.
    """

    def write(self, s: str) -> int:
        return sys.stderr.write(s)

    def writelines(self, lines: Iterable[str]) -> None:
        sys.stderr.writelines(lines)

    def flush(self) -> None:
        sys.stderr.flush()

    def close(self) -> None:
        sys.stderr.close()

    @property
    def closed(self) -> bool:
        return sys.stderr.closed

    def isatty(self) -> bool:
        return sys.stderr.isatty()

    def fileno(self) -> int:
        return sys.stderr.fileno()

    def read(self, n: int = -1, /) -> str:
        return sys.stderr.read(n)

    def readable(self) -> bool:
        return sys.stderr.readable()

    def readline(self, limit: int = -1, /) -> str:
        return sys.stderr.readline(limit)

    def readlines(self, hint: int = -1, /) -> list[str]:
        return sys.stderr.readlines(hint)

    def seek(self, offset: int, whence: int = 0, /) -> int:
        return sys.stderr.seek(offset, whence)

    def seekable(self) -> bool:
        return sys.stderr.seekable()

    def tell(self) -> int:
        return sys.stderr.tell()

    def truncate(self, size: int | None = None, /) -> int:
        return sys.stderr.truncate(size)

    def writable(self) -> bool:
        return sys.stderr.writable()

    def __iter__(self) -> Iterator[str]:
        return iter(sys.stderr)

    def __next__(self) -> str:
        return next(sys.stderr)

    def __enter__(self) -> TextIO:
        return self

    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None:
        return None

    @property
    def buffer(self) -> BinaryIO:
        return sys.stderr.buffer

    @property
    def encoding(self) -> str:
        return sys.stderr.encoding

    @property
    def errors(self) -> str | None:
        return sys.stderr.errors

    @property
    def line_buffering(self) -> bool:
        return bool(sys.stderr.line_buffering)

    @property
    def newlines(self) -> object:
        return sys.stderr.newlines


_stderr_proxy = _StderrProxy()

_structlog_configured = False
_log_level = logging.INFO


class _StdlibBoundLogger:
    """Lightweight BoundLoggerLike backed by stdlib logging.

    Used when structlog hasn't been configured yet — the common case for
    plain CLI invocations. Avoids the ~600ms structlog import penalty.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def bind(self, **new_values: object) -> _StdlibBoundLogger:
        return self  # no-op: stdlib doesn't support structured context

    def debug(self, message: str, *args: object, **event_kw: object) -> None:
        self._logger.debug(message, *args, **_stdlib_log_kwargs(event_kw))

    def info(self, message: str, *args: object, **event_kw: object) -> None:
        self._logger.info(message, *args, **_stdlib_log_kwargs(event_kw))

    def warning(self, message: str, *args: object, **event_kw: object) -> None:
        self._logger.warning(message, *args, **_stdlib_log_kwargs(event_kw))

    def error(self, message: str, *args: object, **event_kw: object) -> None:
        self._logger.error(message, *args, **_stdlib_log_kwargs(event_kw))

    def exception(self, message: str, *args: object, **event_kw: object) -> None:
        self._logger.exception(message, *args, **_stdlib_log_kwargs(event_kw))


def _stdlib_log_kwargs(event_kw: dict[str, object]) -> dict[str, Any]:
    """Forward stdlib-supported logging kwargs from structlog-style calls."""
    return {key: value for key, value in event_kw.items() if key in {"exc_info", "stack_info", "stacklevel", "extra"}}


def get_logger(name: str | None = None) -> BoundLoggerLike:
    """Return a logger — stdlib-backed before structlog is configured.

    After configure_logging() is called, subsequent calls return the
    structlog logger. Before that, a lightweight stdlib logger is used
    to avoid the ~600ms structlog import cost.
    """
    if _structlog_configured:
        import structlog

        logger: BoundLoggerLike = structlog.get_logger(name)
        return logger

    stdlib_logger = logging.getLogger(name)
    stdlib_logger.setLevel(_log_level)
    if not stdlib_logger.handlers:
        stdlib_logger.addHandler(logging.StreamHandler(sys.stderr))
    return _StdlibBoundLogger(stdlib_logger)


def configure_logging(verbose: bool = False, json_logs: bool = False) -> None:
    """Configure structlog. Only called when structured logging is needed.

    The default CLI path (no --verbose, no --json-logs) never calls this,
    so structlog is never imported and startup stays fast.
    """
    import structlog

    global _structlog_configured, _log_level

    _log_level = logging.DEBUG if verbose else logging.INFO

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
        env_force = load_polylogue_config().get("force_plain")
        if isinstance(env_force, bool):
            force_plain = env_force
        elif isinstance(env_force, str):
            force_plain = env_force.lower() not in {"0", "false", "no", ""}
        else:
            force_plain = bool(env_force)
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=sys.stderr.isatty() and not force_plain,
            )
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(_log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=_stderr_proxy),
        cache_logger_on_first_use=True,
    )

    _structlog_configured = True
