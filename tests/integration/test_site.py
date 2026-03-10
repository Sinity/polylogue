"""Tests for static site builder.

Tests the SiteBuilder class including:
1. SiteBuilder.build() succeeds with conversations in database (P0 regression test)
2. ConversationIndex doesn't reference conv.source (regression test for AttributeError)
3. The site CLI command doesn't leak Rich markup tags
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Conversation, ConversationSummary, Message

# =============================================================================
# Async mock helpers
# =============================================================================


async def _empty_messages(*args, **kwargs):
    """Async generator that yields nothing — used to mock iter_messages."""
    return
    yield  # noqa: unreachable — makes this an async generator


async def _iter_messages(payloads):
    """Yield lightweight message-like objects for conversation-page tests."""
    for index, (role, text) in enumerate(payloads, start=1):
        yield MagicMock(id=f"msg-{index}", role=role, text=text)


def _async_repo(summaries):
    """Build an AsyncMock ConversationRepository pre-configured with summaries."""
    repo = AsyncMock()
    repo.list_summaries.return_value = summaries
    repo.iter_messages = _empty_messages
    return repo


def _async_backend(counts):
    """Build an AsyncMock SQLiteBackend pre-configured with message counts."""
    backend = AsyncMock()
    backend.get_message_counts_batch.return_value = counts
    return backend


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_conversation():
    """Create a mock conversation for testing."""
    messages = [
        Message(
            id="msg-1",
            role="user",
            text="Hello, how are you?",
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        ),
        Message(
            id="msg-2",
            role="assistant",
            text="I'm doing well, thank you!",
            timestamp=datetime(2024, 1, 15, 10, 30, 30, tzinfo=timezone.utc),
        ),
        Message(
            id="msg-3",
            role="user",
            text="Can you help me with something?",
            timestamp=datetime(2024, 1, 15, 10, 31, 0, tzinfo=timezone.utc),
        ),
    ]
    return Conversation(
        id="test-conv-001",
        provider="claude",
        title="Test Conversation",
        messages=MessageCollection(messages=messages),
        created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 15, 10, 31, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def mock_conversation_no_title():
    """Create a conversation without title (tests fallback to ID)."""
    messages = [
        Message(
            id="msg-1",
            role="user",
            text="Testing conversations without titles",
            timestamp=datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
        ),
    ]
    return Conversation(
        id="no-title-conv-xyz",
        provider="chatgpt",
        title=None,
        messages=MessageCollection(messages=messages),
        created_at=datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def mock_conversation_gemini():
    """Create a conversation from Gemini provider."""
    messages = [
        Message(
            id="msg-1",
            role="user",
            text="This is from Gemini",
            timestamp=datetime(2024, 1, 20, 12, 0, 0, tzinfo=timezone.utc),
        ),
        Message(
            id="msg-2",
            role="assistant",
            text="Gemini response",
            timestamp=datetime(2024, 1, 20, 12, 0, 30, tzinfo=timezone.utc),
        ),
    ]
    return Conversation(
        id="gemini-conv-001",
        provider="gemini",
        title="Gemini Conversation",
        messages=MessageCollection(messages=messages),
        created_at=datetime(2024, 1, 20, 12, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 20, 12, 0, 30, tzinfo=timezone.utc),
    )


@pytest.fixture
def mock_summary():
    """Create a mock conversation summary for testing."""
    return ConversationSummary(
        id="test-conv-001",
        provider="claude",
        title="Test Conversation",
        created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 15, 10, 31, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def mock_summary_no_title():
    """Create a summary without title (tests fallback to ID)."""
    return ConversationSummary(
        id="no-title-conv-xyz",
        provider="chatgpt",
        title=None,
        created_at=datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def mock_summary_gemini():
    """Create a summary from Gemini provider."""
    return ConversationSummary(
        id="gemini-conv-001",
        provider="gemini",
        title="Gemini Conversation",
        created_at=datetime(2024, 1, 20, 12, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 20, 12, 0, 30, tzinfo=timezone.utc),
    )


# =============================================================================
# SiteBuilder.build() Tests - P0 Regression Test
# =============================================================================


class TestSiteBuilderBuild:
    """Tests for SiteBuilder.build() method."""

    def test_build_succeeds_with_conversations(self, tmp_path, mock_summary):
        """SiteBuilder.build() succeeds with conversations in database (P0 regression test)."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        # Mock the ConversationRepository - patch where they're imported (inside _build_index)
        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = AsyncMock()
            mock_backend_class.return_value = mock_backend
            mock_backend.get_message_counts_batch.return_value = {"test-conv-001": 3}

            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.iter_messages = _empty_messages
            mock_repo.list_summaries.return_value = [mock_summary]

            config = SiteConfig(
                title="Test Archive",
                description="Test Description",
                enable_search=True,
                include_dashboard=True,
            )
            builder = SiteBuilder(output_dir=output_dir, config=config)

            # This should not raise AttributeError about missing 'source'
            result = builder.build()

            # Verify results
            assert result["conversations"] == 1
            assert result["index_pages"] >= 1  # root + provider indexes
            assert (output_dir / "index.html").exists()

    def test_build_creates_output_directory(self, tmp_path, mock_summary):
        """SiteBuilder.build() creates output directory if it doesn't exist."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "nonexistent" / "site"
        assert not output_dir.exists()

        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = AsyncMock()
            mock_backend_class.return_value = mock_backend
            mock_backend.get_message_counts_batch.return_value = {"test-conv-001": 3}

            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.iter_messages = _empty_messages
            mock_repo.list_summaries.return_value = [mock_summary]

            config = SiteConfig(title="Test Archive")
            builder = SiteBuilder(output_dir=output_dir, config=config)
            builder.build()

            assert output_dir.exists()

    def test_build_empty_database(self, tmp_path):
        """SiteBuilder.build() handles empty database gracefully."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = AsyncMock()
            mock_backend_class.return_value = mock_backend
            mock_backend.get_message_counts_batch.return_value = {}

            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.iter_messages = _empty_messages
            mock_repo.list_summaries.return_value = []

            config = SiteConfig(title="Empty Archive")
            builder = SiteBuilder(output_dir=output_dir, config=config)
            result = builder.build()

            assert result["conversations"] == 0
            assert (output_dir / "index.html").exists()

    def test_build_index_requests_all_summaries_without_silent_cap(self, tmp_path, mock_summary):
        """Site index should not silently cap archive summaries."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = AsyncMock()
            mock_backend_class.return_value = mock_backend
            mock_backend.get_message_counts_batch.return_value = {"test-conv-001": 3}

            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.iter_messages = _empty_messages
            mock_repo.list_summaries.return_value = [mock_summary]

            builder = SiteBuilder(output_dir=output_dir, config=SiteConfig(title="Test"))
            asyncio.run(builder._build_index())

            mock_repo.list_summaries.assert_awaited_once_with(limit=None)

    def test_build_multiple_providers(
        self, tmp_path, mock_summary, mock_summary_gemini
    ):
        """SiteBuilder.build() groups conversations by provider."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = AsyncMock()
            mock_backend_class.return_value = mock_backend
            mock_backend.get_message_counts_batch.return_value = {
                "test-conv-001": 3,
                "gemini-conv-001": 2,
            }

            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.iter_messages = _empty_messages
            mock_repo.list_summaries.return_value = [mock_summary, mock_summary_gemini]

            config = SiteConfig(title="Multi-Provider Archive")
            builder = SiteBuilder(output_dir=output_dir, config=config)
            result = builder.build()

            assert result["conversations"] == 2
            # root index + claude provider + gemini provider + dashboard = 4
            assert result["index_pages"] == 4
            # Both provider directories should exist
            assert (output_dir / "claude").exists()
            assert (output_dir / "gemini").exists()

    def test_build_without_dashboard(self, tmp_path, mock_summary):
        """SiteBuilder.build() respects include_dashboard config."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = AsyncMock()
            mock_backend_class.return_value = mock_backend
            mock_backend.get_message_counts_batch.return_value = {"test-conv-001": 3}

            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.iter_messages = _empty_messages
            mock_repo.list_summaries.return_value = [mock_summary]

            config = SiteConfig(title="No Dashboard", include_dashboard=False)
            builder = SiteBuilder(output_dir=output_dir, config=config)
            result = builder.build()

            # root index + 1 provider index = 2 (no dashboard)
            assert result["index_pages"] == 2
            assert not (output_dir / "dashboard.html").exists()

    def test_build_with_dashboard(self, tmp_path, mock_summary):
        """SiteBuilder.build() includes dashboard when enabled."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = AsyncMock()
            mock_backend_class.return_value = mock_backend
            mock_backend.get_message_counts_batch.return_value = {"test-conv-001": 3}

            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.iter_messages = _empty_messages
            mock_repo.list_summaries.return_value = [mock_summary]

            config = SiteConfig(title="With Dashboard", include_dashboard=True)
            builder = SiteBuilder(output_dir=output_dir, config=config)
            result = builder.build()

            # root index + 1 provider index + dashboard = 3
            assert result["index_pages"] == 3
            assert (output_dir / "dashboard.html").exists()


# =============================================================================
# ConversationIndex Tests - Regression Test for 'source' Bug
# =============================================================================


class TestConversationIndex:
    """Tests for ConversationIndex dataclass."""

    def test_conversation_index_no_source_attribute_reference(
        self, tmp_path, mock_summary
    ):
        """ConversationIndex doesn't try to access conv.source attribute.

        Regression test: Previously the code tried to access conv.source which
        doesn't exist on Conversation objects, causing:
        'Conversation' object has no attribute 'source'
        """
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = AsyncMock()
            mock_backend_class.return_value = mock_backend
            mock_backend.get_message_counts_batch.return_value = {"test-conv-001": 3}

            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.iter_messages = _empty_messages
            mock_repo.list_summaries.return_value = [mock_summary]

            config = SiteConfig(title="Test")
            builder = SiteBuilder(output_dir=output_dir, config=config)

            # This should not raise AttributeError
            conversations = asyncio.run(builder._build_index())

            # Verify the index was built successfully
            assert len(conversations) == 1
            index = conversations[0]

            # Verify ConversationIndex structure
            assert index.id == "test-conv-001"
            assert index.title == "Test Conversation"
            assert index.provider == "claude"
            # source should be None (not accessed from conversation)
            assert index.source is None
            assert index.message_count == 3

    def test_conversation_index_title_fallback(self, tmp_path, mock_summary_no_title):
        """ConversationIndex falls back to ID when conversation has no title."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = AsyncMock()
            mock_backend_class.return_value = mock_backend
            mock_backend.get_message_counts_batch.return_value = {"no-title-conv-xyz": 1}

            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.iter_messages = _empty_messages
            mock_repo.list_summaries.return_value = [mock_summary_no_title]

            config = SiteConfig(title="Test")
            builder = SiteBuilder(output_dir=output_dir, config=config)
            conversations = asyncio.run(builder._build_index())

            assert len(conversations) == 1
            index = conversations[0]
            # Should use truncated ID when no title
            assert index.title.startswith("no-title")

    def test_conversation_index_message_count(self, tmp_path, mock_summary):
        """ConversationIndex correctly counts messages."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = AsyncMock()
            mock_backend_class.return_value = mock_backend
            mock_backend.get_message_counts_batch.return_value = {"test-conv-001": 3}

            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.iter_messages = _empty_messages
            mock_repo.list_summaries.return_value = [mock_summary]

            config = SiteConfig(title="Test")
            builder = SiteBuilder(output_dir=output_dir, config=config)
            conversations = asyncio.run(builder._build_index())

            index = conversations[0]
            assert index.message_count == 3

    def test_conversation_index_preview_extraction(self, tmp_path, mock_summary):
        """ConversationIndex extracts preview from summary metadata."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        # Create a summary with metadata containing a summary text
        summary_with_content = ConversationSummary(
            id="test-conv-001",
            provider="claude",
            title="Test Conversation",
            created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 15, 10, 31, 0, tzinfo=timezone.utc),
            metadata={"summary": "Hello, how are you?"},
        )

        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = AsyncMock()
            mock_backend_class.return_value = mock_backend
            mock_backend.get_message_counts_batch.return_value = {"test-conv-001": 3}

            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.iter_messages = _empty_messages
            mock_repo.list_summaries.return_value = [summary_with_content]

            config = SiteConfig(title="Test")
            builder = SiteBuilder(output_dir=output_dir, config=config)
            conversations = asyncio.run(builder._build_index())

            index = conversations[0]
            assert index.preview == "Hello, how are you?"

    def test_conversation_index_timestamp_formatting(self, tmp_path, mock_summary):
        """ConversationIndex formats timestamps correctly."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = AsyncMock()
            mock_backend_class.return_value = mock_backend
            mock_backend.get_message_counts_batch.return_value = {"test-conv-001": 3}

            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.iter_messages = _empty_messages
            mock_repo.list_summaries.return_value = [mock_summary]

            config = SiteConfig(title="Test")
            builder = SiteBuilder(output_dir=output_dir, config=config)
            conversations = asyncio.run(builder._build_index())

            index = conversations[0]
            assert index.created_at == "2024-01-15"
            # updated_at includes time
            assert index.updated_at.startswith("2024-01-15")


# =============================================================================
# CLI Command Tests - Rich Markup Regression Test
# =============================================================================


class TestSiteCommand:
    """Tests for the site CLI command."""

    def test_site_command_success(self, workspace_env):
        """site command completes successfully."""
        from polylogue.cli.commands.site import site_command

        runner = CliRunner()
        output_dir = workspace_env["archive_root"] / "site"

        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = AsyncMock()
            mock_backend_class.return_value = mock_backend
            mock_backend.get_message_counts_batch.return_value = {}

            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.iter_messages = _empty_messages
            mock_repo.list_summaries.return_value = []

            # Create a mock AppEnv object
            from polylogue.ui import UI

            mock_ui = MagicMock(spec=UI)
            mock_ui.plain = True

            from polylogue.cli.types import AppEnv
            mock_env = AppEnv(ui=mock_ui)

            result = runner.invoke(
                site_command,
                ["--output", str(output_dir)],
                obj=mock_env,
            )

            assert result.exit_code == 0

    def test_site_command_no_rich_markup_in_output(self, workspace_env):
        """site command output doesn't contain Rich markup tags.

        Regression test: Output should not contain Rich markup like [bold], [green], etc.
        """
        from polylogue.cli.commands.site import site_command

        runner = CliRunner()
        output_dir = workspace_env["archive_root"] / "site"

        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = AsyncMock()
            mock_backend_class.return_value = mock_backend
            mock_backend.get_message_counts_batch.return_value = {}

            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.iter_messages = _empty_messages
            mock_repo.list_summaries.return_value = []

            from polylogue.cli.types import AppEnv
            from polylogue.ui import UI

            mock_ui = MagicMock(spec=UI)
            mock_ui.plain = True
            mock_env = AppEnv(ui=mock_ui)

            result = runner.invoke(
                site_command,
                ["--output", str(output_dir)],
                obj=mock_env,
            )

            output = result.output

            # Check for common Rich markup patterns
            rich_patterns = [
                "[bold]",
                "[/bold]",
                "[green]",
                "[/green]",
                "[red]",
                "[/red]",
                "[yellow]",
                "[/yellow]",
                "[cyan]",
                "[/cyan]",
            ]

            for pattern in rich_patterns:
                assert pattern not in output, f"Found Rich markup: {pattern} in output"

    def test_site_command_with_custom_title(self, workspace_env):
        """site command accepts custom title."""
        from polylogue.cli.commands.site import site_command

        runner = CliRunner()
        output_dir = workspace_env["archive_root"] / "site"

        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = AsyncMock()
            mock_backend_class.return_value = mock_backend
            mock_backend.get_message_counts_batch.return_value = {}

            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.iter_messages = _empty_messages
            mock_repo.list_summaries.return_value = []

            from polylogue.cli.types import AppEnv
            from polylogue.ui import UI

            mock_ui = MagicMock(spec=UI)
            mock_ui.plain = True
            mock_env = AppEnv(ui=mock_ui)

            result = runner.invoke(
                site_command,
                ["--output", str(output_dir), "--title", "My Custom Archive"],
                obj=mock_env,
            )

            assert result.exit_code == 0

    def test_site_command_with_search_disabled(self, workspace_env):
        """site command respects --no-search flag."""
        from polylogue.cli.commands.site import site_command

        runner = CliRunner()
        output_dir = workspace_env["archive_root"] / "site"

        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = AsyncMock()
            mock_backend_class.return_value = mock_backend
            mock_backend.get_message_counts_batch.return_value = {}

            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.iter_messages = _empty_messages
            mock_repo.list_summaries.return_value = []

            from polylogue.cli.types import AppEnv
            from polylogue.ui import UI

            mock_ui = MagicMock(spec=UI)
            mock_ui.plain = True
            mock_env = AppEnv(ui=mock_ui)

            result = runner.invoke(
                site_command,
                ["--output", str(output_dir), "--no-search"],
                obj=mock_env,
            )

            assert result.exit_code == 0

    def test_site_command_error_handling(self, workspace_env):
        """site command handles build errors gracefully."""
        from polylogue.cli.commands.site import site_command

        runner = CliRunner()
        output_dir = workspace_env["archive_root"] / "site"

        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class:
            mock_backend_class.side_effect = RuntimeError("Database error")

            from polylogue.cli.types import AppEnv
            from polylogue.ui import UI

            mock_ui = MagicMock(spec=UI)
            mock_ui.plain = True
            mock_env = AppEnv(ui=mock_ui)

            result = runner.invoke(
                site_command,
                ["--output", str(output_dir)],
                obj=mock_env,
            )

            # Should fail with error message
            assert result.exit_code != 0

    def test_site_command_generates_html_files(self, workspace_env):
        """site command generates HTML files in output directory."""
        from polylogue.cli.commands.site import site_command

        runner = CliRunner()
        output_dir = workspace_env["archive_root"] / "site"

        # Create a test summary
        test_summary = ConversationSummary(
            id="test-conv-123",
            provider="test",
            title="Test Conv",
            created_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        )

        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = AsyncMock()
            mock_backend_class.return_value = mock_backend
            mock_backend.get_message_counts_batch.return_value = {"test-conv-123": 1}

            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.iter_messages = _empty_messages
            mock_repo.list_summaries.return_value = [test_summary]

            from polylogue.cli.types import AppEnv
            from polylogue.ui import UI

            mock_ui = MagicMock(spec=UI)
            mock_ui.plain = True
            mock_env = AppEnv(ui=mock_ui)

            result = runner.invoke(
                site_command,
                ["--output", str(output_dir)],
                obj=mock_env,
            )

            assert result.exit_code == 0
            # Verify index.html was created
            assert (output_dir / "index.html").exists()


# =============================================================================
# Integration Tests
# =============================================================================


class TestSiteBuilderIntegration:
    """Integration tests for the site builder."""

    def test_build_generates_valid_html(self, tmp_path):
        """SiteBuilder generates valid HTML with proper structure."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        summary = ConversationSummary(
            id="conv-123",
            provider="test",
            title="Test",
            created_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        )

        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = AsyncMock()
            mock_backend_class.return_value = mock_backend
            mock_backend.get_message_counts_batch.return_value = {"conv-123": 1}

            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.iter_messages = _empty_messages
            mock_repo.list_summaries.return_value = [summary]

            config = SiteConfig(title="Test Archive", include_dashboard=True)
            builder = SiteBuilder(output_dir=output_dir, config=config)
            builder.build()

            # Check index.html content
            index_html = (output_dir / "index.html").read_text()
            assert "<!DOCTYPE html>" in index_html
            assert "Test Archive" in index_html
            assert "Test" in index_html

            # Check dashboard.html content
            dashboard_html = (output_dir / "dashboard.html").read_text()
            assert "<!DOCTYPE html>" in dashboard_html
            assert "Dashboard" in dashboard_html or "Archive" in dashboard_html

    def test_build_search_index_generation(self, tmp_path):
        """SiteBuilder generates search index for lunr.js."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        summary = ConversationSummary(
            id="conv-123",
            provider="test",
            title="Test",
            created_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        )

        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = AsyncMock()
            mock_backend_class.return_value = mock_backend
            mock_backend.get_message_counts_batch.return_value = {"conv-123": 1}

            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.iter_messages = _empty_messages
            mock_repo.list_summaries.return_value = [summary]

            config = SiteConfig(
                title="Test",
                enable_search=True,
                search_provider="lunr",
            )
            builder = SiteBuilder(output_dir=output_dir, config=config)
            builder.build()

            # Check search index
            search_index = (output_dir / "search-index.json").read_text()
            search_data = json.loads(search_index)
            assert len(search_data) == 1
            assert search_data[0]["id"] == "conv-123"
            assert search_data[0]["title"] == "Test"
            assert search_data[0]["provider"] == "unknown"

    def test_build_conversation_page_keeps_tail_messages(self, tmp_path):
        """Conversation pages should include messages beyond the old 500-message cap."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"
        summary = ConversationSummary(
            id="conv-123",
            provider="claude",
            title="Long Conversation",
            created_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 15, 10, 5, 0, tzinfo=timezone.utc),
        )

        payloads = [("user" if i % 2 == 0 else "assistant", f"message {i}") for i in range(501)]

        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = AsyncMock()
            mock_backend_class.return_value = mock_backend
            mock_backend.get_message_counts_batch.return_value = {"conv-123": 501}

            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.iter_messages = lambda *args, **kwargs: _iter_messages(payloads)
            mock_repo.list_summaries.return_value = [summary]

            builder = SiteBuilder(output_dir=output_dir, config=SiteConfig(title="Test"))
            builder.build()

            conversation_html = next(output_dir.rglob("conversation.html")).read_text(encoding="utf-8")
            assert "message 0" in conversation_html
            assert "message 500" in conversation_html

    def test_build_conversation_page_preserves_long_message_bodies(self, tmp_path):
        """Conversation pages should not silently truncate long messages."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"
        summary = ConversationSummary(
            id="conv-456",
            provider="claude",
            title="Long Message Conversation",
            created_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 15, 10, 5, 0, tzinfo=timezone.utc),
        )
        long_text = ("abcdef " * 900) + "tail-marker"

        with patch(
            "polylogue.storage.backends.async_sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = AsyncMock()
            mock_backend_class.return_value = mock_backend
            mock_backend.get_message_counts_batch.return_value = {"conv-456": 1}

            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.iter_messages = lambda *args, **kwargs: _iter_messages([("assistant", long_text)])
            mock_repo.list_summaries.return_value = [summary]

            builder = SiteBuilder(output_dir=output_dir, config=SiteConfig(title="Test"))
            builder.build()

            conversation_html = next(output_dir.rglob("conversation.html")).read_text(encoding="utf-8")
            assert "tail-marker" in conversation_html
            assert "[... truncated ...]" not in conversation_html
