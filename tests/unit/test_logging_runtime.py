from __future__ import annotations

import io
import logging
import sys
from typing import Any, cast
from unittest.mock import patch

import pytest

from polylogue import logging as logging_mod
from polylogue.logging import BoundLoggerLike


class _FakeStderr:
    def __init__(self) -> None:
        self._stream = io.StringIO("alpha\nbeta\n")
        self._buffer = io.BytesIO()

    def write(self, s: str) -> int:
        return self._stream.write(s)

    def writelines(self, lines: list[str]) -> None:
        self._stream.writelines(lines)

    def flush(self) -> None:
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()

    @property
    def closed(self) -> bool:
        return self._stream.closed

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return 42

    def read(self, n: int = -1, /) -> str:
        return self._stream.read(n)

    def readable(self) -> bool:
        return self._stream.readable()

    def readline(self, limit: int = -1, /) -> str:
        return self._stream.readline(limit)

    def readlines(self, hint: int = -1, /) -> list[str]:
        return self._stream.readlines(hint)

    def seek(self, offset: int, whence: int = 0, /) -> int:
        return self._stream.seek(offset, whence)

    def seekable(self) -> bool:
        return self._stream.seekable()

    def tell(self) -> int:
        return self._stream.tell()

    def truncate(self, size: int | None = None, /) -> int:
        return self._stream.truncate(size)

    def writable(self) -> bool:
        return self._stream.writable()

    def __iter__(self) -> _FakeStderr:
        return self

    def __next__(self) -> str:
        line = self._stream.readline()
        if not line:
            raise StopIteration
        return line

    def __enter__(self) -> _FakeStderr:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    @property
    def buffer(self) -> io.BytesIO:
        return self._buffer

    @property
    def encoding(self) -> str:
        return "utf-8"

    @property
    def errors(self) -> str | None:
        return None

    @property
    def line_buffering(self) -> bool:
        return True

    @property
    def newlines(self) -> str | tuple[str, ...] | None:
        return "\n"


def test_stderr_proxy_delegates_to_current_sys_stderr() -> None:
    fake = _FakeStderr()

    with patch.object(sys, "stderr", cast(Any, fake)):
        assert logging_mod._stderr_proxy.write("prefix-") == len("prefix-")
        logging_mod._stderr_proxy.writelines(["line-1", "line-2"])
        logging_mod._stderr_proxy.flush()
        assert logging_mod._stderr_proxy.closed is False
        assert logging_mod._stderr_proxy.isatty() is True
        assert logging_mod._stderr_proxy.fileno() == 42

        logging_mod._stderr_proxy.seek(0)
        assert logging_mod._stderr_proxy.read(6) == "prefix"
        assert logging_mod._stderr_proxy.readable() is True
        logging_mod._stderr_proxy.seek(0)
        assert logging_mod._stderr_proxy.readline()
        logging_mod._stderr_proxy.seek(0)
        assert logging_mod._stderr_proxy.readlines()
        logging_mod._stderr_proxy.seek(0)
        assert logging_mod._stderr_proxy.seekable() is True
        assert logging_mod._stderr_proxy.tell() == 0
        assert logging_mod._stderr_proxy.truncate(5) == 5
        assert logging_mod._stderr_proxy.writable() is True

        fake.seek(0)
        assert list(logging_mod._stderr_proxy)
        fake.seek(0)
        assert next(logging_mod._stderr_proxy)

        with logging_mod._stderr_proxy as entered:
            assert entered is logging_mod._stderr_proxy

        assert logging_mod._stderr_proxy.buffer is fake.buffer
        assert logging_mod._stderr_proxy.encoding == fake.encoding
        assert logging_mod._stderr_proxy.errors == fake.errors
        assert logging_mod._stderr_proxy.line_buffering is True
        assert logging_mod._stderr_proxy.newlines == "\n"

        logging_mod._stderr_proxy.close()
        assert fake.closed is True


def test_configure_logging_supports_console_and_json_modes_and_get_logger() -> None:

    with (
        patch.dict("os.environ", {}, clear=False) as env,
        patch("structlog.configure") as configure,
        patch("structlog.dev.ConsoleRenderer", return_value="console-renderer") as console_renderer,
        patch("structlog.processors.JSONRenderer", return_value="json-renderer") as json_renderer,
        patch("sys.stderr.isatty", return_value=True),
    ):
        env.pop("POLYLOGUE_FORCE_PLAIN", None)
        logging_mod.configure_logging(verbose=True, json_logs=False)
        console_processors = configure.call_args.kwargs["processors"]
        assert console_processors[-1] == "console-renderer"
        console_renderer.assert_called_once()
        assert console_renderer.call_args.kwargs["colors"] is True

        logging_mod.configure_logging(verbose=False, json_logs=True)
        json_processors = configure.call_args.kwargs["processors"]
        assert json_processors[-1] == "json-renderer"
        json_renderer.assert_called_once_with()

    bound_logger = cast(BoundLoggerLike, object())
    with patch("structlog.get_logger", return_value=bound_logger) as get_logger:
        assert logging_mod.get_logger("polylogue.tests") is bound_logger
    get_logger.assert_called_once_with("polylogue.tests")


def test_configure_logging_accepts_typed_force_plain_config() -> None:

    with (
        patch("polylogue.config.load_polylogue_config", return_value={"force_plain": True}),
        patch("structlog.configure"),
        patch("structlog.dev.ConsoleRenderer", return_value="console-renderer") as console_renderer,
        patch("sys.stderr.isatty", return_value=True),
    ):
        logging_mod.configure_logging(verbose=False, json_logs=False)

    console_renderer.assert_called_once()
    assert console_renderer.call_args.kwargs["colors"] is False


def test_console_traceback_never_reads_frame_locals(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A logged exception renders without touching the failing frame's locals.

    Anti-vacuity: the default ``ConsoleRenderer`` formatter shows locals, so
    the ``len()`` below raises out of ``logger.error`` and the secret local
    reaches stderr.
    """
    import structlog

    class _ClosedIndex(frozenset[str]):
        def __len__(self) -> int:
            raise RuntimeError("closed")

        def __iter__(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("closed")

    previous = structlog.get_config()
    monkeypatch.setattr(logging_mod, "_structlog_configured", logging_mod._structlog_configured)
    monkeypatch.setattr(logging_mod, "_log_level", logging_mod._log_level)
    try:
        with patch("sys.stderr.isatty", return_value=False):
            logging_mod.configure_logging(verbose=False, json_logs=False)
        logger = structlog.get_logger("polylogue.tests.locals")

        def fail(closed: frozenset[str], secret: str) -> None:
            raise ValueError("boom")

        try:
            # Joined at run time so the rendered source lines never hold it.
            fail(_ClosedIndex(), "-".join(("private", "transcript", "text")))
        except ValueError:
            logger.error("failed", exc_info=True)
    finally:
        structlog.configure(**previous)
    captured = capsys.readouterr()
    rendered = captured.out + captured.err
    assert "ValueError" in rendered
    assert "private-transcript-text" not in rendered


def test_stdlib_bound_logger_forwards_exc_info_before_structlog_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callers pass exc_info/extra structlog-style before configure_logging() runs.

    Previously these kwargs were silently dropped, so a `logger.warning(...,
    exc_info=True)` call made before structlog configuration (e.g. daemon
    convergence probe failure paths) lost the traceback entirely.
    """
    captured: dict[str, object] = {}

    def _record(message: str, *args: object, **kwargs: object) -> None:
        captured["message"] = message
        captured["kwargs"] = kwargs

    stdlib_logger = logging.getLogger("polylogue.tests.exc-forwarding")
    stdlib_logger.warning = _record  # type: ignore[assignment]

    # An earlier test in the same worker may have run configure_logging();
    # this test is specifically about the pre-configuration stdlib path.
    monkeypatch.setattr(logging_mod, "_structlog_configured", False)

    with patch("polylogue.logging.logging.getLogger", return_value=stdlib_logger):
        bound = logging_mod.get_logger("polylogue.tests.exc-forwarding")

    bound.warning("probe failed", exc_info=True, extra={"stage": "fts"}, unsupported_kw="dropped")

    forwarded = captured["kwargs"]
    assert isinstance(forwarded, dict)
    assert forwarded["exc_info"] is True
    forwarded_extra = forwarded["extra"]
    assert isinstance(forwarded_extra, dict)
    assert forwarded_extra["stage"] == "fts"
    assert forwarded_extra["_polylogue_event_fields"] == {"stage": "fts"}


def test_pre_configuration_logger_checks_level_before_field_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validating before the level check makes the discarded debug call warn."""
    import json

    monkeypatch.setattr(logging_mod, "_structlog_configured", False)
    stdlib_logger = logging.getLogger("polylogue.tests.level-first")
    monkeypatch.setattr(stdlib_logger, "level", logging.INFO)
    with patch("polylogue.logging.logging.getLogger", return_value=stdlib_logger):
        bound = logging_mod.get_logger("polylogue.tests.level-first")
    stream = io.StringIO()
    sink = logging_mod.add_sink(logging_mod.make_stream_sink(stream, fmt="json"))
    try:
        bound.debug("Scanning source", unregistered_probe_field="x")
        assert stream.getvalue() == ""
        bound.warning("Probe failed", unregistered_probe_field="x")
    finally:
        logging_mod.remove_sink(sink)
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    rejections = [record for record in records if record["event"] == "log.field_rejected"]
    assert len(rejections) == 1
    assert rejections[0]["field"] == "unregistered_probe_field"
    assert rejections[0]["source_event"] == "Probe failed"
    assert rejections[0]["logger"] == "polylogue.tests.level-first"


def test_pre_configuration_logger_skips_fields_below_the_event_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validating an INFO call the event threshold discards makes it warn."""
    import json

    monkeypatch.setattr(logging_mod, "_structlog_configured", False)
    bound = logging_mod.get_logger("polylogue.tests.threshold-first")
    stream = io.StringIO()
    try:
        logging_mod.configure_events(stream=stream, fmt="json", level="warning")
        bound.info("Routine detail", unregistered_probe_field="x")
        logging_mod.flush_events(timeout_s=1)
    finally:
        logging_mod.reset_events()
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert not [record for record in records if record["event"] == "log.field_rejected"]


def test_bridge_keeps_an_explicit_error_type_without_exc_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reserving ``error_type`` in the bridge drops the caller's classification."""
    import json

    monkeypatch.setattr(logging_mod, "_structlog_configured", False)
    bound = logging_mod.get_logger("polylogue.tests.error-type")
    stream = io.StringIO()
    try:
        logging_mod.configure_events(stream=stream, fmt="json")
        bound.warning("Detection failed", error_type="ValueError")
        try:
            raise KeyError("probe")
        except KeyError:
            bound.warning("Traceback wins", error_type="ValueError", exc_info=True)
        logging_mod.flush_events(timeout_s=1)
    finally:
        logging_mod.reset_events()
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    bridged = {record["error_detail"]: record for record in records if record["event"] == "stdlib.record"}
    assert bridged["Detection failed"]["error_type"] == "ValueError"
    assert bridged["Traceback wins"]["error_type"] == "KeyError"
