"""Concrete CLI query execution side-effect tests.

Law/contract coverage for query routing, no-results handling, output helpers,
summary output, mutations, and stats lives in `test_query_exec_laws.py`.
This file keeps only concrete seams that are better expressed as direct,
example-driven side-effect tests:
- execute_query stream target resolution/warnings
- stream_conversation I/O behavior
- browser/clipboard/result opening side effects
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.cli.types import AppEnv
from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Conversation, ConversationSummary, Message
from polylogue.services import build_runtime_services
from tests.infra.mutmut import preserved_mutmut_env

# =============================================================================
# Test Helpers for Building Test Data
# =============================================================================


def _make_msg(
    id: str = "msg-1",
    role: str = "user",
    text: str | None = "Hello",
    **kwargs,
) -> Message:
    """Create a test message."""
    return Message(
        id=id,
        role=role,
        text=text,
        timestamp=kwargs.get("timestamp"),
        attachments=kwargs.get("attachments", []),
        provider_meta=kwargs.get("provider_meta"),
    )


def _make_conv(
    id: str = "test-conv-123",
    provider: str = "claude",
    title: str = "Test Conversation",
    messages: list[Message] | None = None,
    **kwargs,
) -> Conversation:
    """Create a test conversation."""
    if messages is None:
        messages = [_make_msg("msg-1", "user", "Hello"), _make_msg("msg-2", "assistant", "Hi there")]

    return Conversation(
        id=id,
        provider=provider,
        title=title,
        messages=MessageCollection(messages=messages),
        created_at=kwargs.get("created_at"),
        updated_at=kwargs.get("updated_at"),
        metadata={"tags": kwargs.get("tags", []), "summary": kwargs.get("summary")},
    )


def _make_summary(
    id: str = "conv-1",
    provider: str = "claude",
    title: str = "Test",
    tags: list[str] | None = None,
) -> ConversationSummary:
    """Create a test conversation summary."""
    return ConversationSummary(
        id=id,
        provider=provider,
        title=title,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
        metadata={"tags": tags or [], "summary": "A test conversation"},
    )


def _make_env(*, repo: MagicMock | None = None, config: MagicMock | None = None) -> AppEnv:
    """Create a mock AppEnv."""
    ui = MagicMock()
    ui.plain = True
    ui.console = MagicMock()
    ui.confirm = MagicMock(return_value=True)
    return AppEnv(ui=ui, services=build_runtime_services(config=config, repository=repo))


def _make_params(**overrides) -> dict:
    """Build default params dict with overrides."""
    defaults = {
        "conv_id": None,
        "query": (),
        "contains": (),
        "exclude_text": (),
        "provider": None,
        "exclude_provider": None,
        "tag": None,
        "exclude_tag": None,
        "title": None,
        "has_type": (),
        "since": None,
        "until": None,
        "latest": False,
        "sort": None,
        "reverse": False,
        "limit": None,
        "sample": None,
        "count_only": False,
        "list_mode": False,
        "stream": False,
        "output_format": None,
        "output": None,
        "transform": None,
        "dialogue_only": False,
        "stats_only": False,
        "stats_by": None,
        "set_meta": None,
        "add_tag": None,
        "delete_matched": False,
        "force": False,
        "dry_run": False,
        "open_result": False,
        "fields": None,
    }
    defaults.update(overrides)
    return defaults


def _make_mock_repo(
    *,
    update_metadata: AsyncMock | None = None,
    add_tag: AsyncMock | None = None,
    delete_conversation: AsyncMock | None = None,
) -> MagicMock:
    """Create a mock ConversationRepository with common async methods."""
    repo = MagicMock()
    repo.update_metadata = update_metadata or AsyncMock()
    repo.add_tag = add_tag or AsyncMock()
    repo.delete_conversation = delete_conversation or AsyncMock()
    return repo


# =============================================================================
# Parametrization Data Tables (Module-level Constants)
# =============================================================================

STREAM_TARGET_TEST_CASES = (
    ({"latest": True}, True, "latest-conv-id"),
    ({"conv_id": "abc"}, True, "full-conv-id-12345"),
    ({}, False, None),
)


class TestExecuteQueryStream:
    """Test execute_query with --stream flag."""

    def _fn(self, env: AppEnv, params: dict):
        from polylogue.cli.query import execute_query

        return execute_query(env, params)

    @pytest.mark.parametrize(
        "param_overrides,resolves_id,expected_id",
        STREAM_TARGET_TEST_CASES,
    )
    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("polylogue.cli.query_output.stream_conversation", new_callable=AsyncMock)
    @patch("click.echo")
    def test_stream_target_resolution(
        self,
        mock_echo,
        mock_stream,
        MockFilter,
        mock_vector_provider,
        mock_load_config,
        param_overrides,
        resolves_id,
        expected_id,
    ) -> None:
        """Stream resolves targets correctly (latest, conv_id, or errors on missing)."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None
        mock_repo = MagicMock()

        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter

        if "latest" in param_overrides:
            summary = _make_summary(id="latest-conv-id")
            mock_filter.sort.return_value.limit.return_value.list_summaries = AsyncMock(return_value=[summary])
        elif "conv_id" in param_overrides:
            mock_repo.resolve_id = AsyncMock(return_value="full-conv-id-12345")

        env = _make_env(repo=mock_repo, config=mock_load_config.return_value)
        params = _make_params(stream=True, **param_overrides)

        if not resolves_id:
            with pytest.raises(SystemExit) as exc_info:
                self._fn(env, params)
            assert exc_info.value.code == 1
        else:
            self._fn(env, params)
            mock_stream.assert_called_once()
            call_args = mock_stream.call_args[0]
            assert call_args[2] == expected_id

    @pytest.mark.parametrize(
        "param_overrides",
        [
            {"latest": True, "transform": "strip-tools"},
            {"latest": True, "output": "/tmp/out.txt"},
        ],
    )
    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("polylogue.storage.search_providers.create_vector_provider")
    @patch("polylogue.lib.filters.ConversationFilter")
    @patch("polylogue.cli.query_output.stream_conversation", new_callable=AsyncMock)
    @patch("click.echo")
    def test_stream_warns_on_conflict(
        self,
        mock_echo,
        mock_stream,
        MockFilter,
        mock_vector_provider,
        mock_load_config,
        param_overrides,
    ) -> None:
        """Stream prints warnings for transform and output file conflicts."""
        mock_load_config.return_value = MagicMock()
        mock_vector_provider.return_value = None

        summary = _make_summary(id="conv-id")
        mock_filter = MagicMock()
        MockFilter.return_value = mock_filter
        mock_filter.sort.return_value.limit.return_value.list_summaries = AsyncMock(return_value=[summary])

        env = _make_env(config=mock_load_config.return_value)
        params = _make_params(stream=True, **param_overrides)

        self._fn(env, params)

        # Check that warning was printed
        warning_calls = [call for call in mock_echo.call_args_list if "Warning" in str(call)]
        assert len(warning_calls) > 0

# =============================================================================
# Tests for stream_conversation()
# =============================================================================


class TestStreamConversation:
    async def _fn(self, env: AppEnv, repo, conv_id: str, **kwargs):
        from polylogue.cli.query_output import stream_conversation

        return await stream_conversation(env, repo, conv_id, **kwargs)

    @staticmethod
    def _make_async_repo(**overrides):
        """Create a mock repo with async methods matching stream_conversation's usage."""
        repo = MagicMock()
        repo.get_conversation = AsyncMock(return_value=overrides.get("conv_record"))
        repo.get_conversation_stats = AsyncMock(return_value=overrides.get("stats", {}))

        # iter_messages returns an async iterator
        messages = overrides.get("messages", [])

        async def _async_iter(*args, **kwargs):
            for m in messages:
                yield m

        repo.iter_messages = _async_iter
        return repo

    async def test_conversation_not_found(self) -> None:
        """Conversation not found raises SystemExit."""
        env = _make_env()
        repo = self._make_async_repo(conv_record=None)

        with pytest.raises(SystemExit) as exc_info:
            await self._fn(env, repo, "nonexistent")

        assert exc_info.value.code == 1

    @pytest.mark.parametrize(
        "output_format,has_header,has_footer",
        [
            ("markdown", True, True),
            ("json-lines", True, True),
        ],
    )
    @patch("sys.stdout", new_callable=StringIO)
    async def test_stream_formats(self, mock_stdout, output_format, has_header, has_footer) -> None:
        """Test markdown and json-lines streaming formats with headers/footers."""
        env = _make_env()

        conv_record = MagicMock()
        conv_record.title = "Test Title"
        repo = self._make_async_repo(
            conv_record=conv_record,
            stats={"total_messages": 5, "dialogue_messages": 3},
        )

        await self._fn(env, repo, "conv-1", output_format=output_format)

        output = mock_stdout.getvalue()

        if output_format == "markdown":
            if has_header:
                assert "# Test Title" in output
            if has_footer:
                assert "---" in output
        elif output_format == "json-lines":
            lines = output.strip().split("\n")
            if has_header:
                header = json.loads(lines[0])
                assert header["type"] == "header"
            if has_footer:
                footer = json.loads(lines[-1])
                assert footer["type"] == "footer"

    @patch("sys.stdout", new_callable=StringIO)
    async def test_returns_message_count(self, mock_stdout) -> None:
        """Returns number of messages streamed."""
        env = _make_env()

        conv_record = MagicMock()
        conv_record.title = "Test"
        msg1 = _make_msg("msg-1", "user", "Hello")
        msg2 = _make_msg("msg-2", "assistant", "Hi")
        repo = self._make_async_repo(
            conv_record=conv_record,
            messages=[msg1, msg2],
        )

        count = await self._fn(env, repo, "conv-1")

        assert count == 2


# =============================================================================
# Tests for _open_in_browser()
# =============================================================================


class TestQueryOpenInBrowser:
    def _fn(self, env: AppEnv, content: str, output_format: str, conv=None):
        from polylogue.cli.query_output import _open_in_browser

        return _open_in_browser(env, content, output_format, conv)

    @pytest.mark.parametrize(
        "content,output_format,use_conv,is_html_content",
        [
            ("<html><body>Test</body></html>", "html", False, True),
            ("Plain text", "text", False, False),
        ],
    )
    @patch("webbrowser.open")
    def test_browser_content_handling(
        self, mock_browser, content, output_format, use_conv, is_html_content
    ) -> None:
        """Test HTML pass-through and content wrapping for non-HTML formats."""
        env = _make_env()
        conv = _make_conv() if use_conv else None

        self._fn(env, content, output_format, conv=conv)

        mock_browser.assert_called_once()
        # Verify file:// URL was used
        call_arg = mock_browser.call_args[0][0]
        assert call_arg.startswith("file://")

    @patch("webbrowser.open")
    def test_browser_with_conversation(self, mock_browser) -> None:
        """When conv provided, converts to HTML."""
        env = _make_env()
        conv = _make_conv()

        self._fn(env, "ignored", "text", conv=conv)

        mock_browser.assert_called_once()


# =============================================================================
# Tests for _copy_to_clipboard()
# =============================================================================


class TestCopyToClipboard:
    def _fn(self, env: AppEnv, content: str):
        from polylogue.cli.query_output import _copy_to_clipboard

        return _copy_to_clipboard(env, content)

    @pytest.mark.parametrize(
        "tool_success,should_fail",
        [
            (True, False),  # xclip succeeds
            (False, True),  # all tools fail
        ],
    )
    @patch("subprocess.run")
    @patch("click.echo")
    def test_clipboard_tools(self, mock_echo, mock_run, tool_success, should_fail) -> None:
        """Test clipboard tool success and failure scenarios."""
        env = _make_env()
        if tool_success:
            mock_run.return_value = MagicMock(returncode=0)
            self._fn(env, "test content")
            mock_run.assert_called()
            env.ui.console.print.assert_called()
        else:
            mock_run.side_effect = FileNotFoundError()
            self._fn(env, "test")
            assert mock_echo.called


# =============================================================================
# Tests for _open_result()
# =============================================================================


class TestOpenResult:
    def _fn(self, env: AppEnv, results: list, params: dict):
        from polylogue.cli.query_output import _open_result

        return _open_result(env, results, params)

    def test_no_results_exits(self) -> None:
        """No results raises SystemExit."""
        env = _make_env()
        params = _make_params()

        with pytest.raises(SystemExit) as exc_info:
            self._fn(env, [], params)

        assert exc_info.value.code == 2

    @patch("polylogue.cli.helpers.load_effective_config")
    @patch("click.echo")
    def test_no_render_root_errors(self, mock_echo, mock_config) -> None:
        """No render_root shows error."""
        env = _make_env()
        mock_config.return_value = None
        conv = _make_conv("c1")
        params = _make_params()

        with patch.dict("os.environ", preserved_mutmut_env(), clear=True):
            with pytest.raises(SystemExit) as exc_info:
                self._fn(env, [conv], params)

            assert exc_info.value.code == 1

    @patch("webbrowser.open")
    @patch("polylogue.cli.helpers.load_effective_config")
    def test_opens_html_file(self, mock_config, mock_browser) -> None:
        """Opens HTML file when found and opens in browser."""
        env = _make_env()
        conv = _make_conv("c1")
        params = _make_params()

        with tempfile.TemporaryDirectory() as tmpdir:
            render_root = Path(tmpdir) / "renders" / "c1234567"
            render_root.mkdir(parents=True)
            html_file = render_root / "conversation.html"
            html_file.write_text("<html>test</html>")

            mock_config.return_value = MagicMock()
            mock_config.return_value.render_root = str(Path(tmpdir) / "renders")

            self._fn(env, [conv], params)

            mock_browser.assert_called_once()


# =============================================================================
# Helpers for mocking click.echo
# =============================================================================


MockEchoCalls = []


@pytest.fixture(autouse=True)
def reset_echo_calls():
    """Reset echo call tracker."""
    global MockEchoCalls
    MockEchoCalls = []
    yield
    MockEchoCalls = []
