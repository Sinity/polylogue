"""Tests for pipeline run observers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from polylogue.pipeline.observers import (
    CompositeObserver,
    ExecObserver,
    NotificationObserver,
    WebhookObserver,
)


def _make_result(count: int = 5) -> MagicMock:
    """Create a mock RunResult."""
    return MagicMock(counts={"conversations": count})


class TestNotificationObserver:
    def test_sends_notification(self):
        with patch("subprocess.run") as mock_run:
            handler = NotificationObserver()
            handler.on_completed(_make_result(3))
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "notify-send" in args
            assert "3" in str(args)

    def test_skips_when_no_new(self):
        with patch("subprocess.run") as mock_run:
            handler = NotificationObserver()
            handler.on_completed(_make_result(0))
            mock_run.assert_not_called()

    def test_handles_missing_notifysend(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            handler = NotificationObserver()
            handler.on_completed(_make_result(1))  # Should not raise


class TestWebhookObserver:
    """Test WebhookObserver with mocked DNS to avoid network access in tests."""

    _FAKE_ADDRINFO = [(2, 1, 6, "", ("93.184.216.34", 80))]

    def test_posts_to_url(self):
        with patch("polylogue.pipeline.observers.socket.getaddrinfo", return_value=self._FAKE_ADDRINFO):
            with patch("urllib.request.urlopen") as mock_urlopen:
                handler = WebhookObserver("http://example.com/hook")
                handler.on_completed(_make_result(7))
                mock_urlopen.assert_called_once()
                req = mock_urlopen.call_args[0][0]
                assert req.get_full_url() == "http://example.com/hook"
                assert req.get_method() == "POST"

    def test_payload_format(self):
        import json

        with patch("polylogue.pipeline.observers.socket.getaddrinfo", return_value=self._FAKE_ADDRINFO):
            with patch("urllib.request.urlopen"):
                with patch("urllib.request.Request") as mock_req:
                    handler = WebhookObserver("http://example.com/hook")
                    handler.on_completed(_make_result(7))
                    call_kwargs = mock_req.call_args[1]
                    payload = json.loads(call_kwargs["data"].decode())
                    assert payload["event"] == "sync"
                    assert payload["new_conversations"] == 7

    def test_skips_when_no_new(self):
        with patch("polylogue.pipeline.observers.socket.getaddrinfo", return_value=self._FAKE_ADDRINFO):
            with patch("urllib.request.urlopen") as mock_urlopen:
                handler = WebhookObserver("http://example.com/hook")
                handler.on_completed(_make_result(0))
                mock_urlopen.assert_not_called()

    def test_logs_on_failure(self):
        with patch("polylogue.pipeline.observers.socket.getaddrinfo", return_value=self._FAKE_ADDRINFO):
            with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
                handler = WebhookObserver("http://example.com/hook")
                handler.on_completed(_make_result(1))  # Should not raise


class TestExecObserver:
    def test_executes_command(self):
        with patch("subprocess.run") as mock_run:
            handler = ExecObserver("echo hello")
            handler.on_completed(_make_result(3))
            mock_run.assert_called_once()
            # Verify command is executed as argv list with shell=False (security)
            call_args = mock_run.call_args[0]
            assert call_args[0] == ["echo", "hello"]
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["env"]["POLYLOGUE_NEW_COUNT"] == "3"

    def test_skips_when_no_new(self):
        with patch("subprocess.run") as mock_run:
            handler = ExecObserver("echo hello")
            handler.on_completed(_make_result(0))
            mock_run.assert_not_called()


class TestCompositeObserver:
    def test_dispatches_to_all_observers(self):
        h1 = MagicMock()
        h2 = MagicMock()
        composite = CompositeObserver([h1, h2])
        result = _make_result(5)
        composite.on_completed(result)
        h1.on_completed.assert_called_once_with(result)
        h2.on_completed.assert_called_once_with(result)

    def test_continues_on_observer_failure(self):
        h1 = MagicMock()
        h1.on_completed.side_effect = RuntimeError("boom")
        h2 = MagicMock()
        composite = CompositeObserver([h1, h2])
        result = _make_result(5)
        composite.on_completed(result)
        # h2 should still be called even though h1 raised
        h2.on_completed.assert_called_once_with(result)

    def test_empty_observer_list(self):
        composite = CompositeObserver([])
        composite.on_completed(_make_result(5))  # Should not raise
