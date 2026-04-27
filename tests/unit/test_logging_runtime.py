from __future__ import annotations

import io
import sys
from typing import Any, cast
from unittest.mock import patch

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
        # POLYLOGUE_FORCE_PLAIN=1 (set by CI and devshell) forces colors off
        # at the env-read site. The test exercises the colored-output branch,
        # so isolate it from ambient env.
        patch.dict("os.environ", {}, clear=False) as env,
        patch("polylogue.logging.structlog.configure") as configure,
        patch("polylogue.logging.structlog.dev.ConsoleRenderer", return_value="console-renderer") as console_renderer,
        patch("polylogue.logging.structlog.processors.JSONRenderer", return_value="json-renderer") as json_renderer,
        patch("sys.stderr.isatty", return_value=True),
    ):
        env.pop("POLYLOGUE_FORCE_PLAIN", None)
        logging_mod.configure_logging(verbose=True, json_logs=False)
        console_processors = configure.call_args.kwargs["processors"]
        assert console_processors[-1] == "console-renderer"
        console_renderer.assert_called_once_with(colors=True)

        logging_mod.configure_logging(verbose=False, json_logs=True)
        json_processors = configure.call_args.kwargs["processors"]
        assert json_processors[-1] == "json-renderer"
        json_renderer.assert_called_once_with()

    bound_logger = cast(BoundLoggerLike, object())
    with patch("polylogue.logging.structlog.get_logger", return_value=bound_logger) as get_logger:
        assert logging_mod.get_logger("polylogue.tests") is bound_logger
    get_logger.assert_called_once_with("polylogue.tests")
