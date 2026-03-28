"""Focused entrypoint tests for CLI query execution.

Formatting, output, routing, and mutation contracts live in:
- test_query_fmt.py
- test_query_exec.py
- test_query_exec_laws.py
- test_query_plan.py

This file keeps only the direct entrypoint and error-handling seams that
are not worth expressing through the broader contract suites.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
<<<<<<< ours
import yaml

from polylogue.cli import query
from polylogue.lib import formatting
from polylogue.lib.models import Conversation, Message


@pytest.fixture
def sample_conversations():
    """Create sample conversations for testing."""
    now = datetime(2024, 6, 15, 10, 0)

    conv1 = Conversation(
        id="conv1-abc123",
        provider="chatgpt",
        messages=[
            Message(id="m1", role="user", text="Hello world", timestamp=now),
            Message(id="m2", role="assistant", text="Hi there!", timestamp=now),
        ],
        title="First Conversation",
        updated_at=now,
        metadata={"tags": ["work", "important"]},  # tags via metadata
    )
||||||| base
import yaml

from polylogue.cli.query_output import _format_list
from polylogue.lib import formatting
from polylogue.lib.models import Conversation, Message


@pytest.fixture
def sample_conversations():
    """Create sample conversations for testing."""
    now = datetime(2024, 6, 15, 10, 0)

    conv1 = Conversation(
        id="conv1-abc123",
        provider="chatgpt",
        messages=[
            Message(id="m1", role="user", text="Hello world", timestamp=now),
            Message(id="m2", role="assistant", text="Hi there!", timestamp=now),
        ],
        title="First Conversation",
        updated_at=now,
        metadata={"tags": ["work", "important"]},  # tags via metadata
    )
=======
>>>>>>> theirs

from polylogue.config import ConfigError


def _make_env() -> MagicMock:
    env = MagicMock()
    env.repository = MagicMock()
    return env


def test_execute_query_runs_async_core() -> None:
    from polylogue.cli.query import execute_query

    env = _make_env()
    params = {"query": ("hello",)}

    with (
        patch("polylogue.cli.query._async_execute_query", new_callable=AsyncMock) as mock_async_execute,
        patch("asyncio.run") as mock_asyncio_run,
    ):
        execute_query(env, params)

    mock_async_execute.assert_called_once_with(env, params)
    mock_asyncio_run.assert_called_once()
    dispatched = mock_asyncio_run.call_args.args[0]
    assert asyncio.iscoroutine(dispatched)
    dispatched.close()


<<<<<<< ours
    @pytest.mark.parametrize(
        "fields,should_include,should_exclude",
        [
            (None, ["id", "provider", "title", "messages", "tags"], []),
            ("id,provider", ["id", "provider"], ["title", "messages"]),
            ("id, title, tags", ["id", "title", "tags"], ["provider"]),
        ],
        ids=["full_dict", "selected_fields", "fields_with_spaces"],
    )
    def test_field_selection(self, sample_conversations, fields, should_include, should_exclude):
        """Tests _conv_to_dict field selection."""
        conv = sample_conversations[0]
        result = formatting._conv_to_dict(conv, fields)

        for field in should_include:
            assert field in result, f"Expected field '{field}' to be in result"
        for field in should_exclude:
            assert field not in result, f"Expected field '{field}' to NOT be in result"

        if fields is None:
            assert result["id"] == "conv1-abc123"
            assert result["provider"] == "chatgpt"
            assert result["title"] == "First Conversation"
            assert result["messages"] == 2
            assert result["tags"] == ["work", "important"]


class TestFormatHelpers:
    """Consolidated tests for format conversion helpers."""

    @pytest.mark.parametrize(
        "formatter,assertion",
        [
            (formatting._conv_to_markdown, lambda r: "# First Conversation" in r),
            (formatting._conv_to_html, lambda r: "<!DOCTYPE html>" in r and "<html" in r),
            (formatting._conv_to_obsidian, lambda r: r.startswith("---")),
            (formatting._conv_to_org, lambda r: "#+TITLE:" in r),
        ],
        ids=["markdown", "html", "obsidian", "org"],
    )
    def test_format_structure(self, sample_conversations, formatter, assertion):
        """Tests format conversion helpers produce expected structure."""
        conv = sample_conversations[0]
        result = formatter(conv)
        assert assertion(result), f"Failed for {formatter.__name__}"

    def test_markdown_includes_messages(self, sample_conversations):
        """Markdown includes all message content."""
        conv = sample_conversations[0]
        result = formatting._conv_to_markdown(conv)
        assert "Hello world" in result and "Hi there!" in result

    def test_markdown_includes_provider(self, sample_conversations):
        """Markdown includes provider info."""
        conv = sample_conversations[0]
        result = formatting._conv_to_markdown(conv)
        assert "chatgpt" in result.lower()

    def test_html_escapes_special_chars(self, sample_conversations):
        """HTML properly escapes special characters."""
        conv = Conversation(
            id="test",
            provider="test",
            messages=[Message(id="m1", role="user", text="<script>alert('xss')</script>")],
        )
        result = formatting._conv_to_html(conv)
        assert "<script>" not in result and "&lt;script&gt;" in result

    def test_html_includes_css(self, sample_conversations):
        """HTML includes CSS styles."""
        conv = sample_conversations[0]
        result = formatting._conv_to_html(conv)
        assert "<style>" in result and "message-user" in result

    def test_obsidian_includes_tags(self, sample_conversations):
        """Obsidian format includes tags in frontmatter."""
        conv = sample_conversations[0]
        result = formatting._conv_to_obsidian(conv)
        assert "tags:" in result

    def test_org_uses_headings(self, sample_conversations):
        """Org-mode uses * for headings."""
        conv = sample_conversations[0]
        result = formatting._conv_to_org(conv)
        assert "* USER" in result and "* ASSISTANT" in result


class TestQueryFormatList:
    """Tests for _format_list helper."""

    @pytest.mark.parametrize(
        "format_type,parser,expected_type,first_id",
        [
            ("json", json.loads, list, "conv1-abc123"),
            ("yaml", yaml.safe_load, list, "conv1-abc123"),
        ],
        ids=["json_format", "yaml_list_format"],
    )
    def test_format_list(self, sample_conversations, format_type, parser, expected_type, first_id):
        """Tests _format_list with various formats."""
        result = query._format_list(sample_conversations, format_type, None)
        parsed = parser(result)

        assert isinstance(parsed, expected_type)
        assert len(parsed) == 3
        assert parsed[0]["id"] == first_id

    def test_default_format(self, sample_conversations):
        """Default format returns text list."""
        result = query._format_list(sample_conversations, "markdown", None)

        assert "conv1-abc123" in result
        assert "conv2-def456" in result
        assert "First Conversation" in result


class TestQueryFormatConversation:
    """Tests for _format_conversation helper."""

    @pytest.mark.parametrize(
        "format_type,assertion",
        [
            ("json", lambda r: json.loads(r)["id"] == "conv1-abc123"),
            ("html", lambda r: "<!DOCTYPE html>" in r),
            ("obsidian", lambda r: r.startswith("---")),
            ("org", lambda r: "#+TITLE:" in r),
            ("yaml", lambda r: yaml.safe_load(r)["id"] == "conv1-abc123"),
            ("plaintext", lambda r: "Hello world" in r and "##" not in r),
        ],
        ids=[
            "json_format",
            "html_format",
            "obsidian_format",
            "org_format",
            "yaml_format",
            "plaintext_format",
        ],
    )
    def test_format_conversation(self, sample_conversations, format_type, assertion):
        """Tests _format_conversation with various formats."""
        conv = sample_conversations[0]
        result = formatting.format_conversation(conv, format_type, None)

        assert assertion(result), f"Failed assertion for format {format_type}"


class TestDryRunMode:
    """Tests for --dry-run functionality in modifiers and delete."""

    async def test_dry_run_modifiers_shows_preview(self, mock_env, sample_conversations, capsys):
        """Dry-run mode shows preview without modifying."""
        params = {
            "add_tag": ("test-tag",),
            "dry_run": True,
        }

        await query._apply_modifiers(mock_env, sample_conversations, params)

        captured = capsys.readouterr()
        assert "DRY-RUN" in captured.out
        assert "3" in captured.out  # Count of conversations
        assert "add tags" in captured.out.lower()

    async def test_dry_run_modifiers_shows_sample(self, mock_env, sample_conversations):
        """Dry-run shows sample of affected conversations."""
        params = {
            "rm_tag": ("old-tag",),
            "dry_run": True,
        }

        await query._apply_modifiers(mock_env, sample_conversations, params)

        calls = mock_env.ui.console.print.call_args_list
        output = " ".join(str(c) for c in calls)
        assert "conv1" in output or "chatgpt" in output

    async def test_dry_run_delete_shows_preview(self, mock_env, sample_conversations, capsys):
        """Dry-run delete shows preview without deleting."""
        params = {"dry_run": True}

        await query._delete_conversations(mock_env, sample_conversations, params)

        captured = capsys.readouterr()
        assert "DRY-RUN" in captured.out
        assert "delete" in captured.out.lower()
        assert "3" in captured.out


class TestBulkOperationConfirmation:
    """Tests for bulk operation confirmation (>10 items requires --force)."""

    async def test_modifiers_require_confirmation_for_bulk(self, mock_env, capsys):
        """Modifiers prompt for confirmation for >10 items."""
        # Create 15 mock conversations
        convs = [MagicMock(id=f"conv{i}", display_title=f"Conv {i}", provider="test") for i in range(15)]
        params = {"add_tag": ("bulk-tag",), "force": False}

        # Decline confirmation
        mock_env.ui.confirm.return_value = False
        await query._apply_modifiers(mock_env, convs, params)

        mock_env.ui.confirm.assert_called_once()
        captured = capsys.readouterr()
        assert "15" in captured.out

    async def test_modifiers_proceed_with_force(self, mock_env):
        """Modifiers proceed with --force for bulk operations."""
        # Create 15 mock conversations
        convs = [MagicMock(id=f"conv{i}", display_title=f"Conv {i}", provider="test") for i in range(15)]
        params = {"add_tag": ("bulk-tag",), "force": True}

        mock_backend = MagicMock()
        mock_backend.add_tag = AsyncMock()
        mock_env.repository = mock_backend
        await query._apply_modifiers(mock_env, convs, params)
        assert mock_backend.add_tag.call_count == 15

    async def test_delete_requires_confirmation_for_bulk(self, mock_env, capsys):
        """Delete prompts for confirmation for >10 items."""
        convs = [
            MagicMock(id=f"conv{i}", display_title=f"Conv {i}", provider="test", created_at=None, updated_at=None)
            for i in range(15)
        ]
        params = {"force": False}

        # Decline confirmation
        mock_env.ui.confirm.return_value = False
        await query._delete_conversations(mock_env, convs, params)

        mock_env.ui.confirm.assert_called_once()
        captured = capsys.readouterr()
        assert "DELETE" in captured.err

    async def test_delete_proceeds_with_force(self, mock_env):
        """Delete proceeds with --force for bulk operations."""
        convs = [MagicMock(id=f"conv{i}", display_title=f"Conv {i}", provider="test", created_at=None) for i in range(15)]
        params = {"force": True}

        mock_backend = MagicMock()
        mock_backend.delete_conversation = AsyncMock(return_value=True)
        mock_env.repository = mock_backend
        await query._delete_conversations(mock_env, convs, params)
        assert mock_backend.delete_conversation.call_count == 15

    async def test_small_operations_proceed_without_force(self, mock_env):
        """Operations with <=10 items proceed without --force."""
        convs = [MagicMock(id=f"conv{i}", display_title=f"Conv {i}", provider="test") for i in range(5)]
        params = {"add_tag": ("small-tag",), "force": False}

        mock_backend = MagicMock()
        mock_backend.add_tag = AsyncMock()
        mock_env.repository = mock_backend
        await query._apply_modifiers(mock_env, convs, params)
        assert mock_backend.add_tag.call_count == 5


class TestOperationReporting:
    """Tests for operation result reporting."""

    async def test_add_tag_reports_count(self, mock_env, sample_conversations, capsys):
        """Add tag reports count of affected conversations."""
        params = {"add_tag": ("new-tag",)}

        mock_backend = MagicMock()
        mock_backend.add_tag = AsyncMock()
        mock_env.repository = mock_backend
        await query._apply_modifiers(mock_env, sample_conversations, params)
        captured = capsys.readouterr()
        assert "Added tags" in captured.out
        assert "3" in captured.out

    async def test_delete_reports_count(self, mock_env, sample_conversations, capsys):
        """Delete reports count of deleted conversations."""
        params = {}

        mock_backend = MagicMock()
        mock_backend.delete_conversation = AsyncMock(return_value=True)
        mock_env.repository = mock_backend
        await query._delete_conversations(mock_env, sample_conversations, params)
        captured = capsys.readouterr()
        assert "Deleted" in captured.out
        assert "3" in captured.out


class TestQueryConvToJson:
    """Tests for _conv_to_json helper."""

    @pytest.mark.parametrize(
        "fields,has_messages,expected_id",
        [
            (None, True, "conv1-abc123"),
            ("id,title", False, "conv1-abc123"),
            ("id,messages", True, "conv1-abc123"),
        ],
        ids=[
            "full_json_includes_message_content",
            "field_selection_excludes_messages",
            "field_selection_includes_messages_when_selected",
        ],
    )
    def test_json_field_selection(self, sample_conversations, fields, has_messages, expected_id):
        """Tests _conv_to_json field selection."""
        conv = sample_conversations[0]
        result = formatting._conv_to_json(conv, fields)
        parsed = json.loads(result)

        assert parsed["id"] == expected_id
        if has_messages:
            assert isinstance(parsed["messages"], list)
            assert len(parsed["messages"]) == 2
            if fields is None:
                assert parsed["messages"][0]["id"] == "m1"
                assert parsed["messages"][0]["role"] == "user"
                assert parsed["messages"][0]["text"] == "Hello world"
        else:
            assert "messages" not in parsed
            assert "provider" not in parsed

    def test_empty_conversation_produces_valid_json(self):
        """Empty conversation produces valid JSON with empty messages array."""
        conv = Conversation(
            id="empty-conv",
            provider="test",
            messages=[],
            title="Empty",
        )
        result = formatting._conv_to_json(conv, None)
        parsed = json.loads(result)

        assert parsed["id"] == "empty-conv"
        assert parsed["messages"] == []

    @pytest.mark.parametrize(
        "message_text,is_none",
        [
            ("Hello", False),
            (None, True),
        ],
        ids=["message_with_text", "message_with_none_text"],
||||||| base
    @pytest.mark.parametrize(
        "fields,should_include,should_exclude",
        [
            (None, ["id", "provider", "title", "messages", "tags"], []),
            ("id,provider", ["id", "provider"], ["title", "messages"]),
            ("id, title, tags", ["id", "title", "tags"], ["provider"]),
        ],
        ids=["full_dict", "selected_fields", "fields_with_spaces"],
    )
    def test_field_selection(self, sample_conversations, fields, should_include, should_exclude):
        """Tests _conv_to_dict field selection."""
        conv = sample_conversations[0]
        result = formatting._conv_to_dict(conv, fields)

        for field in should_include:
            assert field in result, f"Expected field '{field}' to be in result"
        for field in should_exclude:
            assert field not in result, f"Expected field '{field}' to NOT be in result"

        if fields is None:
            assert result["id"] == "conv1-abc123"
            assert result["provider"] == "chatgpt"
            assert result["title"] == "First Conversation"
            assert result["messages"] == 2
            assert result["tags"] == ["work", "important"]


class TestFormatHelpers:
    """Consolidated tests for format conversion helpers."""

    @pytest.mark.parametrize(
        "formatter,assertion",
        [
            (formatting._conv_to_markdown, lambda r: "# First Conversation" in r),
            (formatting._conv_to_html, lambda r: "<!DOCTYPE html>" in r and "<html" in r),
            (formatting._conv_to_obsidian, lambda r: r.startswith("---")),
            (formatting._conv_to_org, lambda r: "#+TITLE:" in r),
        ],
        ids=["markdown", "html", "obsidian", "org"],
    )
    def test_format_structure(self, sample_conversations, formatter, assertion):
        """Tests format conversion helpers produce expected structure."""
        conv = sample_conversations[0]
        result = formatter(conv)
        assert assertion(result), f"Failed for {formatter.__name__}"

    def test_markdown_includes_messages(self, sample_conversations):
        """Markdown includes all message content."""
        conv = sample_conversations[0]
        result = formatting._conv_to_markdown(conv)
        assert "Hello world" in result and "Hi there!" in result

    def test_markdown_includes_provider(self, sample_conversations):
        """Markdown includes provider info."""
        conv = sample_conversations[0]
        result = formatting._conv_to_markdown(conv)
        assert "chatgpt" in result.lower()

    def test_html_escapes_special_chars(self, sample_conversations):
        """HTML properly escapes special characters."""
        conv = Conversation(
            id="test",
            provider="test",
            messages=[Message(id="m1", role="user", text="<script>alert('xss')</script>")],
        )
        result = formatting._conv_to_html(conv)
        assert "<script>" not in result and "&lt;script&gt;" in result

    def test_html_includes_css(self, sample_conversations):
        """HTML includes CSS styles."""
        conv = sample_conversations[0]
        result = formatting._conv_to_html(conv)
        assert "<style>" in result and "message-user" in result

    def test_obsidian_includes_tags(self, sample_conversations):
        """Obsidian format includes tags in frontmatter."""
        conv = sample_conversations[0]
        result = formatting._conv_to_obsidian(conv)
        assert "tags:" in result

    def test_org_uses_headings(self, sample_conversations):
        """Org-mode uses * for headings."""
        conv = sample_conversations[0]
        result = formatting._conv_to_org(conv)
        assert "* USER" in result and "* ASSISTANT" in result


class TestQueryFormatList:
    """Tests for _format_list helper."""

    @pytest.mark.parametrize(
        "format_type,parser,expected_type,first_id",
        [
            ("json", json.loads, list, "conv1-abc123"),
            ("yaml", yaml.safe_load, list, "conv1-abc123"),
        ],
        ids=["json_format", "yaml_list_format"],
    )
    def test_format_list(self, sample_conversations, format_type, parser, expected_type, first_id):
        """Tests _format_list with various formats."""
        result = _format_list(sample_conversations, format_type, None)
        parsed = parser(result)

        assert isinstance(parsed, expected_type)
        assert len(parsed) == 3
        assert parsed[0]["id"] == first_id

    def test_default_format(self, sample_conversations):
        """Default format returns text list."""
        result = _format_list(sample_conversations, "markdown", None)

        assert "conv1-abc123" in result
        assert "conv2-def456" in result
        assert "First Conversation" in result


class TestQueryFormatConversation:
    """Tests for _format_conversation helper."""

    @pytest.mark.parametrize(
        "format_type,assertion",
        [
            ("json", lambda r: json.loads(r)["id"] == "conv1-abc123"),
            ("html", lambda r: "<!DOCTYPE html>" in r),
            ("obsidian", lambda r: r.startswith("---")),
            ("org", lambda r: "#+TITLE:" in r),
            ("yaml", lambda r: yaml.safe_load(r)["id"] == "conv1-abc123"),
            ("plaintext", lambda r: "Hello world" in r and "##" not in r),
        ],
        ids=[
            "json_format",
            "html_format",
            "obsidian_format",
            "org_format",
            "yaml_format",
            "plaintext_format",
        ],
    )
    def test_format_conversation(self, sample_conversations, format_type, assertion):
        """Tests _format_conversation with various formats."""
        conv = sample_conversations[0]
        result = formatting.format_conversation(conv, format_type, None)

        assert assertion(result), f"Failed assertion for format {format_type}"


class TestQueryConvToJson:
    """Tests for _conv_to_json helper."""

    @pytest.mark.parametrize(
        "fields,has_messages,expected_id",
        [
            (None, True, "conv1-abc123"),
            ("id,title", False, "conv1-abc123"),
            ("id,messages", True, "conv1-abc123"),
        ],
        ids=[
            "full_json_includes_message_content",
            "field_selection_excludes_messages",
            "field_selection_includes_messages_when_selected",
        ],
    )
    def test_json_field_selection(self, sample_conversations, fields, has_messages, expected_id):
        """Tests _conv_to_json field selection."""
        conv = sample_conversations[0]
        result = formatting._conv_to_json(conv, fields)
        parsed = json.loads(result)

        assert parsed["id"] == expected_id
        if has_messages:
            assert isinstance(parsed["messages"], list)
            assert len(parsed["messages"]) == 2
            if fields is None:
                assert parsed["messages"][0]["id"] == "m1"
                assert parsed["messages"][0]["role"] == "user"
                assert parsed["messages"][0]["text"] == "Hello world"
        else:
            assert "messages" not in parsed
            assert "provider" not in parsed

    def test_empty_conversation_produces_valid_json(self):
        """Empty conversation produces valid JSON with empty messages array."""
        conv = Conversation(
            id="empty-conv",
            provider="test",
            messages=[],
            title="Empty",
        )
        result = formatting._conv_to_json(conv, None)
        parsed = json.loads(result)

        assert parsed["id"] == "empty-conv"
        assert parsed["messages"] == []

    @pytest.mark.parametrize(
        "message_text,is_none",
        [
            ("Hello", False),
            (None, True),
        ],
        ids=["message_with_text", "message_with_none_text"],
=======
@pytest.mark.parametrize("exc_type", [ValueError, ImportError])
def test_create_query_vector_provider_swallows_expected_setup_errors(exc_type: type[Exception]) -> None:
    from polylogue.cli.query import _create_query_vector_provider

    with patch(
        "polylogue.storage.search_providers.create_vector_provider",
        side_effect=exc_type("vector unavailable"),
    ):
        assert _create_query_vector_provider(MagicMock()) is None


def test_create_query_vector_provider_logs_unexpected_failure() -> None:
    from polylogue.cli.query import _create_query_vector_provider

    with (
        patch(
            "polylogue.storage.search_providers.create_vector_provider",
            side_effect=RuntimeError("boom"),
        ),
        patch("polylogue.cli.query.logger.warning") as mock_warning,
    ):
        assert _create_query_vector_provider(MagicMock()) is None

    mock_warning.assert_called_once()
    assert mock_warning.call_args.args[0] == "Vector search setup failed: %s"


def test_async_execute_query_fails_on_config_error() -> None:
    from polylogue.cli.query import _async_execute_query

    env = _make_env()

    with (
        patch("polylogue.cli.helpers.load_effective_config", side_effect=ConfigError("bad config")),
        patch("polylogue.cli.helpers.fail", side_effect=SystemExit("query: bad config")) as mock_fail,
        pytest.raises(SystemExit, match="query: bad config"),
    ):
        asyncio.run(_async_execute_query(env, {}))

    mock_fail.assert_called_once_with("query", "bad config")


def test_async_execute_query_reports_query_plan_error() -> None:
    from polylogue.cli.query import _async_execute_query
    from polylogue.cli.query_plan import QueryPlanError

    env = _make_env()

    with (
        patch("polylogue.cli.helpers.load_effective_config", return_value=MagicMock()),
        patch("polylogue.cli.query._create_query_vector_provider", return_value=None),
        patch(
            "polylogue.cli.query.build_query_execution_plan",
            side_effect=QueryPlanError("bad query plan"),
        ),
        patch("click.echo") as mock_echo,
        pytest.raises(SystemExit) as exc_info,
    ):
        asyncio.run(_async_execute_query(env, {}))

    assert exc_info.value.code == 1
    mock_echo.assert_called_once_with("Error: bad query plan", err=True)


def test_async_execute_query_passes_vector_provider_into_filter_build() -> None:
    from polylogue.cli.query import _async_execute_query
    from polylogue.cli.query_plan import QueryAction, QueryExecutionPlan, QueryMutationSpec, QueryOutputSpec
    from polylogue.lib.query_spec import ConversationQuerySpec

    env = _make_env()
    filter_chain = MagicMock()
    filter_chain.can_use_summaries.return_value = False
    filter_chain.list = AsyncMock(return_value=[])
    selection = MagicMock(spec=ConversationQuerySpec)
    selection.build_filter.return_value = filter_chain
    vector_provider = object()
    plan = QueryExecutionPlan(
        selection=selection,
        action=QueryAction.SHOW,
        output=QueryOutputSpec("markdown", ("stdout",), None, False, None, False),
        mutation=QueryMutationSpec((), (), False, False, False),
>>>>>>> theirs
    )

    with (
        patch("polylogue.cli.helpers.load_effective_config", return_value=MagicMock()),
        patch("polylogue.cli.query._create_query_vector_provider", return_value=vector_provider),
        patch("polylogue.cli.query.build_query_execution_plan", return_value=plan),
        patch("polylogue.cli.query_output._output_results") as mock_output_results,
    ):
        asyncio.run(_async_execute_query(env, {}))

    selection.build_filter.assert_called_once_with(env.repository, vector_provider=vector_provider)
    mock_output_results.assert_called_once_with(env, [], {})
