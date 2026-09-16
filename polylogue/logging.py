"""Structured logging configuration.

structlog is imported lazily — the default CLI path (no --verbose, no --json-logs)
never pays the ~600ms import cost. Only configure_logging() triggers the import.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass
from dataclasses import field as _dc_field
from datetime import UTC, datetime
from types import TracebackType
from typing import TYPE_CHECKING, Any, BinaryIO, ParamSpec, Protocol, TextIO, TypeVar

from polylogue.logging_fields import (
    OUTCOMES,
    QUARANTINED_FIELDS,
    TEXT_FIELD_MAX_CHARS,
    rejection_reason,
)

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
        from polylogue.config import load_polylogue_config

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


# ---------------------------------------------------------------------------
# Structured event layer
#
# Everything above this line is the legacy prose-logging shim retained for
# call sites that have not been converted yet. Note in particular that
# ``_StdlibBoundLogger.bind()`` is a no-op and ``_stdlib_log_kwargs`` discards
# every keyword except ``exc_info``/``stack_info``/``stacklevel``/``extra`` —
# on the default CLI path (no ``--verbose``, no ``--json-logs``) the structured
# fields those call sites pass are silently thrown away. That is the defect the
# layer below exists to retire; see ``docs/structured-logging.md``.
# ---------------------------------------------------------------------------

TRACE = 5
DEBUG = 10
INFO = 20
WARNING = 30
ERROR = 40

_LEVEL_NAMES: dict[int, str] = {TRACE: "trace", DEBUG: "debug", INFO: "info", WARNING: "warning", ERROR: "error"}
_LEVEL_VALUES: dict[str, int] = {name: value for value, name in _LEVEL_NAMES.items()}

#: Emit threshold. Compared before any work is done, so a suppressed event
#: costs one integer comparison plus the caller's own kwargs dict.
_threshold: int = INFO

Event = Mapping[str, object]
Sink = Callable[[Event], None]

_sinks: list[Sink] = []
_sinks_lock = threading.Lock()

# The correlation carrier. A ContextVar propagates automatically across
# ``await`` boundaries and across ``asyncio.to_thread`` (which copies the
# context), and across ``polylogue.daemon.write_coordinator._run_in_daemon_thread``
# (which copies it explicitly before crossing the writer-lease thread hop).
# Raw ``threading.Thread`` targets and ``Executor.submit`` callables do NOT
# inherit it; wrap those with :func:`propagate`.
_context: contextvars.ContextVar[Mapping[str, object]] = contextvars.ContextVar("polylogue_log_context")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def current_context() -> Mapping[str, object]:
    """Return the correlation fields bound at this point of execution."""
    return _context.get({})


@contextlib.contextmanager
def bind(**fields: object) -> Iterator[None]:
    """Bind correlation fields for the duration of the block.

    Fields are validated on bind, so an unregistered name fails loudly at the
    binding site rather than being dropped once per event downstream.
    """
    accepted, rejected = _validate(fields)
    for name, reason in rejected.items():
        _emit_raw(WARNING, "log.field_rejected", {"reason": reason, "field": name})
    merged = {**current_context(), **accepted}
    token = _context.set(merged)
    try:
        yield
    finally:
        _context.reset(token)


_P = ParamSpec("_P")
_R = TypeVar("_R")


def propagate(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Wrap ``function`` so it runs with the *current* correlation context.

    Needed for raw ``threading.Thread(target=...)`` and
    ``ThreadPoolExecutor.submit(...)``, neither of which copies contextvars.

    Generic in both directions so wrapping is type-transparent: a caller that
    awaits a typed future must not have to cast it back. An ``object``-erasing
    signature made every wrapped ``submit`` lose its result type, which is a
    silent invitation to skip the wrapper rather than fix the annotation.
    """
    context = contextvars.copy_context()

    def runner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return context.run(lambda: function(*args, **kwargs))

    return runner


def _validate(fields: Mapping[str, object]) -> tuple[dict[str, object], dict[str, str]]:
    """Split ``fields`` into the allowlisted subset and the rejected names."""
    accepted: dict[str, object] = {}
    rejected: dict[str, str] = {}
    for name, value in fields.items():
        reason = rejection_reason(name)
        if reason is not None:
            rejected[name] = reason
            continue
        if name in QUARANTINED_FIELDS:
            accepted[name] = _truncate(value)
        else:
            accepted[name] = _scalar(value)
    return accepted, rejected


def _truncate(value: object) -> str:
    text = str(value)
    if len(text) <= TEXT_FIELD_MAX_CHARS:
        return text
    return text[:TEXT_FIELD_MAX_CHARS] + "...<truncated>"


def _scalar(value: object) -> object:
    """Coerce to a JSON-safe scalar without serializing anything large.

    A registered field is expected to be a scalar already; anything else is
    reduced to its type name rather than stringified, so a rogue object can
    never smuggle content in through ``repr``.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return f"<{type(value).__name__}>"


def _emit_raw(level: int, event: str, fields: Mapping[str, object]) -> None:
    record: dict[str, object] = dict(fields)
    # Reserved keys are assigned last: a caller field can never rename the
    # event, restate its level, or forge its timestamp.
    record["ts"] = _now()
    record["level"] = _LEVEL_NAMES.get(level, "info")
    record["event"] = event
    with _sinks_lock:
        sinks = tuple(_sinks)
    for sink in sinks:
        # A broken sink must never break the caller it is observing.
        with contextlib.suppress(Exception):
            sink(record)


def emit(event: str, /, level: int = INFO, **fields: object) -> None:
    """Emit one structured event.

    ``event`` is a stable dotted token (``daemon.stage.ok``), never prose —
    the human sentence is a rendering concern, not the storage form.
    """
    if level < _threshold:
        return
    accepted, rejected = _validate(fields)
    for name, reason in rejected.items():
        _emit_raw(WARNING, "log.field_rejected", {"reason": reason, "field": name, "source_event": event})
    merged = dict(current_context())
    merged.update(accepted)
    _emit_raw(level, event, merged)


def is_enabled(level: int) -> bool:
    """True when ``level`` would be emitted. Use to guard expensive field work."""
    return level >= _threshold


@dataclass
class Span:
    """One unit of work, emitted as ``<name>.start`` and a terminal event.

    The terminal event is emitted from ``__exit__``, which runs *before* any
    enclosing ``except`` clause. An exception swallowed by a broad handler
    upstream is therefore still recorded here — the log cannot be made to lie
    by a caller who decides to carry on.

    A span that exits without a declared outcome is reported as ``unmeasured``
    at WARNING, not as success.
    """

    name: str
    trace_id: str
    span_id: str
    fields: MutableMapping[str, object] = _dc_field(default_factory=dict)
    _outcome: str | None = None
    _level: int = INFO
    _start: float = 0.0

    def set(self, **fields: object) -> None:
        """Attach fields to the terminal event."""
        accepted, rejected = _validate(fields)
        for name, reason in rejected.items():
            _emit_raw(WARNING, "log.field_rejected", {"reason": reason, "field": name, "source_event": self.name})
        self.fields.update(accepted)

    def _finish(self, outcome: str, level: int, **fields: object) -> None:
        if outcome not in OUTCOMES:
            outcome = "unmeasured"
        self._outcome = outcome
        self._level = level
        self.set(**fields)

    def ok(self, **fields: object) -> None:
        self._finish("ok", INFO, **fields)

    def empty(self, **fields: object) -> None:
        self._finish("empty", INFO, **fields)

    def skipped(self, **fields: object) -> None:
        self._finish("skipped", DEBUG, **fields)

    def degraded(self, reason: str, **fields: object) -> None:
        self._finish("degraded", WARNING, reason=reason, **fields)

    def refused(self, reason: str, **fields: object) -> None:
        """A deliberate, typed refusal. Never renders as success."""
        self._finish("refused", WARNING, reason=reason, **fields)

    def unmeasured(self, reason: str, **fields: object) -> None:
        """No answer was obtained — a timed-out probe, an unreached branch."""
        self._finish("unmeasured", WARNING, reason=reason, **fields)


@contextlib.contextmanager
def span(name: str, /, **fields: object) -> Iterator[Span]:
    """Open a correlated span. See :class:`Span` for the outcome contract."""
    parent = current_context()
    trace_id = str(parent.get("trace_id") or uuid.uuid4().hex[:16])
    span_id = uuid.uuid4().hex[:16]
    accepted, rejected = _validate(fields)
    for bad, reason in rejected.items():
        _emit_raw(WARNING, "log.field_rejected", {"reason": reason, "field": bad, "source_event": name})

    correlation: dict[str, object] = {"trace_id": trace_id, "span_id": span_id}
    parent_span = parent.get("span_id")
    if parent_span is not None:
        correlation["parent_span_id"] = parent_span

    bound = {**parent, **correlation, **accepted}
    token = _context.set(bound)
    active = Span(name=name, trace_id=trace_id, span_id=span_id, _start=time.perf_counter())
    try:
        emit(f"{name}.start", level=DEBUG)
        yield active
    except BaseException as exc:
        duration_ms = round((time.perf_counter() - active._start) * 1000, 3)
        emit(
            f"{name}.error",
            level=ERROR,
            outcome="error",
            duration_ms=duration_ms,
            error_type=type(exc).__name__,
            error_detail=str(exc),
            **active.fields,
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - active._start) * 1000, 3)
        outcome = active._outcome
        if outcome is None:
            emit(
                f"{name}.unmeasured",
                level=WARNING,
                outcome="unmeasured",
                reason="span_exited_without_outcome",
                duration_ms=duration_ms,
                **active.fields,
            )
        else:
            emit(
                f"{name}.{outcome}",
                level=active._level,
                outcome=outcome,
                duration_ms=duration_ms,
                **active.fields,
            )
    finally:
        _context.reset(token)


# -- rendering views --------------------------------------------------------


def render_json(record: Event, *, redact: bool = False) -> str:
    """The storage form: one JSON object per line, stable key order.

    ``redact`` strips the quarantined free-text fields, exactly as
    :func:`render_console` does. The storage form is the one an unattended
    rebuild keeps on disk and the one most likely to be handed to someone
    else, so ``POLYLOGUE_LOG_REDACT=1`` has to reach it too -- a redaction
    switch that only cleaned the operator's terminal would be a promise the
    retained artefact does not keep.
    """
    if redact:
        record = {key: value for key, value in record.items() if key not in QUARANTINED_FIELDS}
    return json.dumps(record, separators=(",", ":"), sort_keys=True, default=str)


_CONSOLE_LEAD = ("ts", "level", "event")


def _console_safe(value: object) -> str:
    """Render one console value with control characters made visible.

    Several registered fields carry text that a remote or untrusted party
    controls -- a browser-capture ``Origin`` header, a provider identifier, a
    quarantined ``error_detail``. Written raw to a terminal, a newline in such
    a value forges an additional log line and an ESC introduces a terminal
    control sequence. Escaping is lossless: nothing is dropped or shortened,
    so the record is still fully readable and the JSON storage form (which
    escapes these itself) is unaffected.
    """
    text = str(value)
    if not any(ch < " " or ch == "\x7f" or "\x80" <= ch <= "\x9f" for ch in text):
        return text
    return "".join(
        ch if not (ch < " " or ch == "\x7f" or "\x80" <= ch <= "\x9f") else f"\\x{ord(ch):02x}" for ch in text
    )


def render_console(record: Event, *, redact: bool = False) -> str:
    """The human view. A rendering of the record, never its storage form."""
    ts = _console_safe(record.get("ts", ""))[11:23]
    level = _console_safe(record.get("level", "info")).upper()[:5]
    event = _console_safe(record.get("event", ""))
    rest = {
        key: value
        for key, value in record.items()
        if key not in _CONSOLE_LEAD and not (redact and key in QUARANTINED_FIELDS)
    }
    trailer = " ".join(f"{key}={_console_safe(value)}" for key, value in sorted(rest.items()))
    return f"{ts} {level:<5} {event} {trailer}".rstrip()


# -- configuration surface --------------------------------------------------


def make_stream_sink(stream: object, *, fmt: str = "json", redact: bool = False) -> Sink:
    """Build a sink writing rendered records to ``stream``, one line each."""

    def sink(record: Event) -> None:
        line = render_json(record, redact=redact) if fmt == "json" else render_console(record, redact=redact)
        write = getattr(stream, "write", None)
        if write is None:
            return
        write(line + "\n")
        flush = getattr(stream, "flush", None)
        if flush is not None:
            flush()

    return sink


def add_sink(sink: Sink) -> Sink:
    """Register ``sink``. Returns it, so it can be passed to :func:`remove_sink`."""
    with _sinks_lock:
        _sinks.append(sink)
    return sink


def remove_sink(sink: Sink) -> None:
    with _sinks_lock:
        if sink in _sinks:
            _sinks.remove(sink)


def set_level(level: int | str) -> int:
    """Set the emit threshold. Returns the previous value."""
    global _threshold
    previous = _threshold
    _threshold = _LEVEL_VALUES[level] if isinstance(level, str) else level
    return previous


@contextlib.contextmanager
def capture() -> Iterator[list[dict[str, object]]]:
    """Collect emitted records into a list. For tests and for `devtools`."""
    records: list[dict[str, object]] = []
    sink = add_sink(lambda record: records.append(dict(record)))
    try:
        yield records
    finally:
        remove_sink(sink)


def reset_events() -> None:
    """Drop all sinks, correlation context and the stdlib bridge.

    Event configuration is process-global by nature. Tests (and any embedded
    host that re-enters a CLI command in one process) need a way back to a
    known state, or one command's sink renders into the next command's
    captured output.
    """
    global _default_sink, _stdlib_bridge, _threshold
    with _sinks_lock:
        _sinks.clear()
    _default_sink = None
    if _stdlib_bridge is not None:
        logging.getLogger().removeHandler(_stdlib_bridge)
        _stdlib_bridge = None
    _context.set({})
    _threshold = INFO


class _StdlibBridge(logging.Handler):
    """Route surviving ``logging.getLogger`` records into the event stream.

    Without this, a converted daemon would lose sight of the 19 unconverted
    modules and of third-party libraries entirely. Prose messages land in the
    quarantined ``error_detail`` field, which is exactly the right signal: an
    event carrying prose is one that has not been converted yet.
    """

    def emit(self, record: logging.LogRecord) -> None:
        level = {10: DEBUG, 20: INFO, 30: WARNING, 40: ERROR, 50: ERROR}.get(record.levelno, INFO)
        if level < _threshold:
            return
        try:
            detail = record.getMessage()
        except Exception:
            detail = "<unformattable log record>"
        fields: dict[str, object] = {"logger": record.name, "error_detail": detail}
        if record.exc_info and record.exc_info[0] is not None:
            fields["error_type"] = record.exc_info[0].__name__
        emit("stdlib.record", level=level, **fields)


_stdlib_bridge: _StdlibBridge | None = None
_default_sink: Sink | None = None


def set_run_context(**fields: object) -> None:
    """Set process-wide base correlation fields (e.g. one daemon ``run_id``).

    Unlike :func:`bind` this is not scoped: it establishes the floor that every
    later context builds on. Call it once, at process start. Passing no fields
    clears it, which is what makes repeated in-process entry points (tests,
    embedded invocations) reproducible.
    """
    accepted, rejected = _validate(fields)
    for name, reason in rejected.items():
        _emit_raw(WARNING, "log.field_rejected", {"reason": reason, "field": name})
    _context.set(accepted)


def configure_events(
    *,
    stream: object | None = None,
    fmt: str | None = None,
    level: int | str | None = None,
    redact: bool | None = None,
    bridge_stdlib: bool = True,
) -> None:
    """Install the default sink and threshold from arguments or environment.

    Environment: ``POLYLOGUE_LOG_FORMAT`` (json|console),
    ``POLYLOGUE_LOG_LEVEL``, ``POLYLOGUE_LOG_FILE``, ``POLYLOGUE_LOG_REDACT``.
    """
    global _stdlib_bridge

    resolved_fmt = fmt or os.environ.get("POLYLOGUE_LOG_FORMAT") or "console"
    resolved_level = level if level is not None else os.environ.get("POLYLOGUE_LOG_LEVEL", "info")
    resolved_redact = (
        redact if redact is not None else os.environ.get("POLYLOGUE_LOG_REDACT", "").lower() in {"1", "true", "yes"}
    )
    set_level(resolved_level)

    target = stream
    if target is None:
        log_file = os.environ.get("POLYLOGUE_LOG_FILE")
        target = open(log_file, "a", encoding="utf-8") if log_file else _stderr_proxy  # noqa: SIM115

    # Replace rather than stack: a second call (a test, a re-entered CLI
    # command in the same process) must not double every event.
    global _default_sink
    if _default_sink is not None:
        remove_sink(_default_sink)
    _default_sink = add_sink(make_stream_sink(target, fmt=resolved_fmt, redact=resolved_redact))

    if bridge_stdlib and _stdlib_bridge is None:
        _stdlib_bridge = _StdlibBridge()
        logging.getLogger().addHandler(_stdlib_bridge)
        logging.getLogger().setLevel(logging.DEBUG if _threshold <= DEBUG else logging.INFO)
