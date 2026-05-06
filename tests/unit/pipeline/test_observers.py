"""Tests for pipeline run observers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from polylogue.pipeline.observers import (
    CompositeObserver,
    ExecObserver,
    NotificationObserver,
    WebhookObserver,
)
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend


def _make_result(count: int = 5) -> MagicMock:
    """Create a mock RunResult."""
    return MagicMock(
        counts={
            "conversations": count,
            "new_conversations": count,
            "changed_conversations": 0,
        },
        drift={"new": {"conversations": count}, "changed": {"conversations": 0}},
    )


class TestNotificationObserver:
    def test_sends_notification(self: object) -> None:
        with patch("subprocess.run") as mock_run:
            handler = NotificationObserver()
            handler.on_completed(_make_result(3))
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "notify-send" in args
            assert "3" in str(args)

    def test_skips_when_no_new(self: object) -> None:
        with patch("subprocess.run") as mock_run:
            handler = NotificationObserver()
            handler.on_completed(_make_result(0))
            mock_run.assert_not_called()

    def test_handles_missing_notifysend(self: object) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            handler = NotificationObserver()
            handler.on_completed(_make_result(1))  # Should not raise


class TestWebhookObserver:
    """Test WebhookObserver with mocked DNS to avoid network access in tests."""

    _FAKE_ADDRINFO = [(2, 1, 6, "", ("93.184.216.34", 80))]

    def test_posts_to_url(self: object) -> None:
        with patch("polylogue.pipeline.observers.socket.getaddrinfo", return_value=TestWebhookObserver._FAKE_ADDRINFO):
            with patch("polylogue.pipeline.observers._post_webhook") as mock_post:
                handler = WebhookObserver("http://example.com/hook")
                handler.on_completed(_make_result(7))
                mock_post.assert_called_once()
                assert mock_post.call_args.args[0] == "http://example.com/hook"

    def test_payload_format(self: object) -> None:
        import json

        with patch("polylogue.pipeline.observers.socket.getaddrinfo", return_value=TestWebhookObserver._FAKE_ADDRINFO):
            with patch("polylogue.pipeline.observers._post_webhook") as mock_post:
                handler = WebhookObserver("http://example.com/hook")
                handler.on_completed(_make_result(7))
                payload = json.loads(mock_post.call_args.args[1].decode())
                assert payload["event"] == "sync"
                assert payload["conversation_activity_count"] == 7
                assert payload["new_conversations"] == 7
                assert payload["changed_conversations"] == 0

    def test_skips_when_no_new(self: object) -> None:
        with patch("polylogue.pipeline.observers.socket.getaddrinfo", return_value=TestWebhookObserver._FAKE_ADDRINFO):
            with patch("polylogue.pipeline.observers._post_webhook") as mock_post:
                handler = WebhookObserver("http://example.com/hook")
                handler.on_completed(_make_result(0))
                mock_post.assert_not_called()

    def test_logs_on_failure(self: object) -> None:
        with patch("polylogue.pipeline.observers.socket.getaddrinfo", return_value=TestWebhookObserver._FAKE_ADDRINFO):
            with patch("polylogue.pipeline.observers._post_webhook", side_effect=ConnectionError("refused")):
                handler = WebhookObserver("http://example.com/hook")
                handler.on_completed(_make_result(1))  # Should not raise


class TestExecObserver:
    def test_executes_command(self: object) -> None:
        with patch("subprocess.run") as mock_run:
            handler = ExecObserver("echo hello")
            handler.on_completed(_make_result(3))
            mock_run.assert_called_once()
            # Verify command is executed as argv list with shell=False (security)
            call_args = mock_run.call_args[0]
            assert call_args[0] == ["echo", "hello"]
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["env"]["POLYLOGUE_ACTIVITY_COUNT"] == "3"
            assert call_kwargs["env"]["POLYLOGUE_NEW_CONVERSATION_COUNT"] == "3"
            assert call_kwargs["env"]["POLYLOGUE_CHANGED_CONVERSATION_COUNT"] == "0"

    def test_skips_when_no_new(self: object) -> None:
        with patch("subprocess.run") as mock_run:
            handler = ExecObserver("echo hello")
            handler.on_completed(_make_result(0))
            mock_run.assert_not_called()


class TestCompositeObserver:
    def test_dispatches_to_all_observers(self: object) -> None:
        h1 = MagicMock()
        h2 = MagicMock()
        composite = CompositeObserver([h1, h2])
        result = _make_result(5)
        composite.on_completed(result)
        h1.on_completed.assert_called_once_with(result)
        h2.on_completed.assert_called_once_with(result)

    def test_continues_on_observer_failure(self: object) -> None:
        h1 = MagicMock()
        h1.on_completed.side_effect = RuntimeError("boom")
        h2 = MagicMock()
        composite = CompositeObserver([h1, h2])
        result = _make_result(5)
        composite.on_completed(result)
        # h2 should still be called even though h1 raised
        h2.on_completed.assert_called_once_with(result)

    def test_empty_observer_list(self: object) -> None:
        composite = CompositeObserver([])
        composite.on_completed(_make_result(5))  # Should not raise


class TestIndexProgressCallback:
    """Verify progress_callback firing during indexing is safe."""

    async def test_index_update_without_callback_succeeds(self: object, sqlite_backend: SQLiteBackend) -> None:
        """Index update works without a progress_callback."""
        from polylogue.config import Config
        from polylogue.pipeline.services.indexing import IndexService

        config = Config(
            archive_root=Path("/tmp"),
            render_root=Path("/tmp/render"),
            sources=[],
        )
        service = IndexService(config, backend=sqlite_backend)

        result = await service.update_index([])
        assert result is True

    async def test_index_rebuild_without_callback_succeeds(self: object, sqlite_backend: SQLiteBackend) -> None:
        """Index rebuild works without a progress_callback."""
        from polylogue.config import Config
        from polylogue.pipeline.services.indexing import IndexService

        config = Config(
            archive_root=Path("/tmp"),
            render_root=Path("/tmp/render"),
            sources=[],
        )
        service = IndexService(config, backend=sqlite_backend)

        result = await service.rebuild_index()
        assert result is True
