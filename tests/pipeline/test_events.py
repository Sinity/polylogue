"""Tests for pipeline event handlers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from polylogue.pipeline.events import (
    CompositeSyncHandler,
    ExecHandler,
    NotificationHandler,
    SyncEvent,
    WebhookHandler,
)


def _make_event(count: int = 5) -> SyncEvent:
    """Create a test SyncEvent."""
    return SyncEvent(new_conversations=count, run_result=MagicMock())


class TestNotificationHandler:
    def test_sends_notification(self):
        with patch("subprocess.run") as mock_run:
            handler = NotificationHandler()
            handler.on_sync(_make_event(3))
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "notify-send" in args
            assert "3" in str(args)

    def test_skips_when_no_new(self):
        with patch("subprocess.run") as mock_run:
            handler = NotificationHandler()
            handler.on_sync(_make_event(0))
            mock_run.assert_not_called()

    def test_handles_missing_notifysend(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            handler = NotificationHandler()
            handler.on_sync(_make_event(1))  # Should not raise


class TestWebhookHandler:
    def test_posts_to_url(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            handler = WebhookHandler("http://example.com/hook")
            handler.on_sync(_make_event(7))
            mock_urlopen.assert_called_once()
            req = mock_urlopen.call_args[0][0]
            assert req.get_full_url() == "http://example.com/hook"
            assert req.get_method() == "POST"

    def test_payload_format(self):
        import json

        with patch("urllib.request.urlopen"):
            with patch("urllib.request.Request") as mock_req:
                handler = WebhookHandler("http://example.com/hook")
                handler.on_sync(_make_event(7))
                call_kwargs = mock_req.call_args[1]
                payload = json.loads(call_kwargs["data"].decode())
                assert payload["event"] == "sync"
                assert payload["new_conversations"] == 7

    def test_skips_when_no_new(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            handler = WebhookHandler("http://example.com/hook")
            handler.on_sync(_make_event(0))
            mock_urlopen.assert_not_called()

    def test_logs_on_failure(self):
        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            handler = WebhookHandler("http://example.com/hook")
            handler.on_sync(_make_event(1))  # Should not raise


class TestExecHandler:
    def test_executes_command(self):
        with patch("subprocess.run") as mock_run:
            handler = ExecHandler("echo hello")
            handler.on_sync(_make_event(3))
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["env"]["POLYLOGUE_NEW_COUNT"] == "3"
            assert call_kwargs["shell"] is True

    def test_skips_when_no_new(self):
        with patch("subprocess.run") as mock_run:
            handler = ExecHandler("echo hello")
            handler.on_sync(_make_event(0))
            mock_run.assert_not_called()


class TestCompositeSyncHandler:
    def test_dispatches_to_all_handlers(self):
        h1 = MagicMock()
        h2 = MagicMock()
        composite = CompositeSyncHandler([h1, h2])
        event = _make_event(5)
        composite.on_sync(event)
        h1.on_sync.assert_called_once_with(event)
        h2.on_sync.assert_called_once_with(event)

    def test_continues_on_handler_failure(self):
        h1 = MagicMock()
        h1.on_sync.side_effect = RuntimeError("boom")
        h2 = MagicMock()
        composite = CompositeSyncHandler([h1, h2])
        event = _make_event(5)
        composite.on_sync(event)
        # h2 should still be called even though h1 raised
        h2.on_sync.assert_called_once_with(event)

    def test_empty_handlers_list(self):
        composite = CompositeSyncHandler([])
        composite.on_sync(_make_event(5))  # Should not raise
