"""Regression: daemon HTTP suppresses BrokenPipeError from client disconnect (#1677).

When a webui client navigates away, refreshes, or closes the tab while
the daemon is still writing the response, ``self.wfile.write(...)``
raises ``BrokenPipeError``. The stdlib ``http.server`` lets the traceback
escape to stderr/journal; that noise was misread as "the webui crashed."

The fix wraps ``do_GET`` / ``do_POST`` / ``do_DELETE`` so the disconnect
classes (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)
are caught and logged at debug level instead of propagating.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from polylogue.daemon.http import DaemonAPIHandler

_DISCONNECT_ERRORS = [
    BrokenPipeError(32, "Broken pipe"),
    ConnectionResetError(104, "Connection reset by peer"),
    ConnectionAbortedError(103, "Software caused connection abort"),
]


def _make_handler() -> DaemonAPIHandler:
    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.path = "/healthz"
    handler.headers = MagicMock()
    return handler


@pytest.mark.parametrize("error", _DISCONNECT_ERRORS, ids=lambda e: type(e).__name__)
def test_do_get_swallows_client_disconnect(error: OSError) -> None:
    handler = _make_handler()
    with patch.object(DaemonAPIHandler, "_parse_path", side_effect=error):
        # Must not raise. If the fix regresses, the bare exception escapes
        # to the SocketServer error handler and the daemon journal fills
        # with tracebacks on every page navigation.
        handler.do_GET()


@pytest.mark.parametrize("error", _DISCONNECT_ERRORS, ids=lambda e: type(e).__name__)
def test_do_post_swallows_client_disconnect(error: OSError) -> None:
    handler = _make_handler()
    with patch.object(DaemonAPIHandler, "_do_post_impl", side_effect=error):
        handler.do_POST()


@pytest.mark.parametrize("error", _DISCONNECT_ERRORS, ids=lambda e: type(e).__name__)
def test_do_delete_swallows_client_disconnect(error: OSError) -> None:
    handler = _make_handler()
    with patch.object(DaemonAPIHandler, "_do_delete_impl", side_effect=error):
        handler.do_DELETE()


def test_do_get_does_not_swallow_unrelated_exceptions() -> None:
    """Non-disconnect errors still escape so the SocketServer logs them properly."""
    handler = _make_handler()
    with patch.object(DaemonAPIHandler, "_parse_path", side_effect=RuntimeError("real bug")):
        with pytest.raises(RuntimeError, match="real bug"):
            handler.do_GET()
