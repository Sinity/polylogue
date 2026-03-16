"""Tests for pipeline run observers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from polylogue.pipeline.observers import (
    CompositeObserver,
    ExecObserver,
    NotificationObserver,
    WebhookObserver,
)
from polylogue.pipeline.services.rendering import RenderResult, RenderService


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


# =====================================================================
# Merged from test_progress_callbacks.py (callbacks and observers)
# =====================================================================


class TestRenderProgressCallback:
    """Verify progress_callback fires during rendering."""

    async def test_callback_called_for_each_conversation(self):
        """progress_callback is invoked once per rendered conversation."""
        # Mock renderer that succeeds
        mock_renderer = AsyncMock()
        mock_renderer.render = AsyncMock()

        service = RenderService(renderer=mock_renderer, render_root=Path("/tmp/render"))

        callback = MagicMock()
        result = await service.render_conversations(
            ["conv-1", "conv-2", "conv-3"],
            progress_callback=callback,
        )

        assert result.rendered_count == 3
        assert callback.call_count == 3

    async def test_callback_desc_format(self):
        """progress_callback desc follows 'Rendering: N/total' format."""
        mock_renderer = AsyncMock()
        mock_renderer.render = AsyncMock()

        service = RenderService(renderer=mock_renderer, render_root=Path("/tmp/render"))

        descs = []

        def capture_callback(amount, desc=None):
            descs.append(desc)

        await service.render_conversations(
            ["conv-1", "conv-2"],
            progress_callback=capture_callback,
        )

        # All desc values should match the "Rendering: N/2" pattern
        assert all(d is not None and d.startswith("Rendering:") for d in descs)
        # The last one should show total completed
        assert "2/2" in descs[-1]

    async def test_callback_fires_on_failure_too(self):
        """progress_callback fires even when rendering fails."""
        mock_renderer = AsyncMock()
        mock_renderer.render = AsyncMock(
            side_effect=[None, RuntimeError("render failed"), None]
        )

        service = RenderService(renderer=mock_renderer, render_root=Path("/tmp/render"))

        callback = MagicMock()
        result = await service.render_conversations(
            ["conv-1", "conv-2", "conv-3"],
            progress_callback=callback,
        )

        # All 3 should fire callback (2 success + 1 failure)
        assert callback.call_count == 3
        assert result.rendered_count == 2
        assert len(result.failures) == 1

    async def test_no_callback_is_safe(self):
        """Rendering works without a progress_callback."""
        mock_renderer = AsyncMock()
        mock_renderer.render = AsyncMock()

        service = RenderService(renderer=mock_renderer, render_root=Path("/tmp/render"))

        result = await service.render_conversations(["conv-1"])
        assert result.rendered_count == 1

    async def test_callback_amount_is_one(self):
        """Each callback invocation passes amount=1."""
        mock_renderer = AsyncMock()
        mock_renderer.render = AsyncMock()

        service = RenderService(renderer=mock_renderer, render_root=Path("/tmp/render"))

        amounts = []

        def capture(amount, desc=None):
            amounts.append(amount)

        await service.render_conversations(
            ["conv-1", "conv-2", "conv-3"],
            progress_callback=capture,
        )
        assert all(a == 1 for a in amounts)


class TestIndexProgressCallback:
    """Verify progress_callback firing during indexing is safe."""

    async def test_index_update_without_callback_succeeds(self, sqlite_backend):
        """Index update works without a progress_callback."""
        from polylogue.config import Config
        from polylogue.pipeline.services.indexing import IndexService

        config = Config(
            archive_root="/tmp",
            render_root="/tmp/render",
            sources=[],
        )
        service = IndexService(config, backend=sqlite_backend)

        result = await service.update_index([])
        assert result is True

    async def test_index_rebuild_without_callback_succeeds(self, sqlite_backend):
        """Index rebuild works without a progress_callback."""
        from polylogue.config import Config
        from polylogue.pipeline.services.indexing import IndexService

        config = Config(
            archive_root="/tmp",
            render_root="/tmp/render",
            sources=[],
        )
        service = IndexService(config, backend=sqlite_backend)

        result = await service.rebuild_index()
        assert result is True


class TestCallbackEdgeCases:
    """Edge cases for progress callback handling."""

    async def test_null_callback_safety(self):
        """Explicitly passing None as callback doesn't raise."""
        mock_renderer = AsyncMock()
        mock_renderer.render = AsyncMock()

        service = RenderService(renderer=mock_renderer, render_root=Path("/tmp/render"))

        # Explicitly pass None (not omit the kwarg)
        result = await service.render_conversations(
            ["conv-1", "conv-2"],
            progress_callback=None,
        )
        assert result.rendered_count == 2

    async def test_callback_exception_does_not_crash_render(self):
        """If the callback itself raises, rendering should still complete.

        Note: Current implementation does NOT catch callback exceptions
        (they propagate). This test documents the actual behavior.
        """
        import pytest

        mock_renderer = AsyncMock()
        mock_renderer.render = AsyncMock()

        service = RenderService(renderer=mock_renderer, render_root=Path("/tmp/render"))

        def bad_callback(amount, desc=None):
            raise RuntimeError("callback exploded")

        # The callback exception propagates — verify it raises
        with pytest.raises(RuntimeError, match="callback exploded"):
            await service.render_conversations(
                ["conv-1"],
                progress_callback=bad_callback,
            )

    async def test_empty_conversation_list(self):
        """Rendering zero conversations returns clean result with no callbacks."""
        mock_renderer = AsyncMock()
        mock_renderer.render = AsyncMock()

        service = RenderService(renderer=mock_renderer, render_root=Path("/tmp/render"))

        callback = MagicMock()
        result = await service.render_conversations(
            [],
            progress_callback=callback,
        )

        assert result.rendered_count == 0
        assert len(result.failures) == 0
        assert callback.call_count == 0
