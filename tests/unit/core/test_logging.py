from __future__ import annotations

import sys
from io import StringIO

from polylogue.logging import _StderrProxy, configure_logging, get_logger


def test_configure_logging_accepts_verbose_mode() -> None:
    configure_logging(verbose=True, json_logs=False)


def test_configure_logging_accepts_json_mode() -> None:
    configure_logging(verbose=False, json_logs=True)


def test_get_logger_returns_structlog_logger() -> None:
    assert get_logger("test.module") is not None


def test_stderr_proxy_write_delegates_to_current_stderr() -> None:
    proxy = _StderrProxy()
    original_stderr = sys.stderr
    try:
        sys.stderr = StringIO()
        proxy.write("test message")
        assert sys.stderr.getvalue() == "test message"
    finally:
        sys.stderr = original_stderr


def test_stderr_proxy_exposes_terminal_capabilities() -> None:
    proxy = _StderrProxy()
    assert isinstance(proxy.isatty(), bool)
    assert isinstance(proxy.fileno(), int)
