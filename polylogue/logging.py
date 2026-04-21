"""Structured logging configuration."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable, Iterator
from types import TracebackType
from typing import BinaryIO, Protocol, TextIO

import structlog
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


def get_logger(name: str | None = None) -> BoundLoggerLike:
    logger: BoundLoggerLike = structlog.get_logger(name)
    return logger
