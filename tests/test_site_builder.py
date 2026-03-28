"""Tests for static site builder.

Tests the SiteBuilder class including:
1. SiteBuilder.build() succeeds with conversations in database (P0 regression test)
2. ConversationIndex doesn't reference conv.source (regression test for AttributeError)
3. The site CLI command doesn't leak Rich markup tags
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.lib.models import Conversation, Message
from polylogue.lib.messages import MessageCollection


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


# =============================================================================
# SiteBuilder.build() Tests - P0 Regression Test
# =============================================================================


class TestSiteBuilderBuild:
    """Tests for SiteBuilder.build() method."""

    def test_build_succeeds_with_conversations(self, tmp_path, mock_conversation):
        """SiteBuilder.build() succeeds with conversations in database (P0 regression test)."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        # Mock the ConversationRepository - patch where they're imported (inside _build_index)
        with patch(
            "polylogue.storage.backends.sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = MagicMock()
            mock_backend_class.return_value = mock_backend

            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.list.return_value = [mock_conversation]

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

    def test_build_creates_output_directory(self, tmp_path, mock_conversation):
        """SiteBuilder.build() creates output directory if it doesn't exist."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "nonexistent" / "site"
        assert not output_dir.exists()

        with patch(
            "polylogue.storage.backends.sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = MagicMock()
            mock_backend_class.return_value = mock_backend

            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.list.return_value = [mock_conversation]

            config = SiteConfig(title="Test Archive")
            builder = SiteBuilder(output_dir=output_dir, config=config)
            builder.build()

            assert output_dir.exists()

    def test_build_empty_database(self, tmp_path):
        """SiteBuilder.build() handles empty database gracefully."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        with patch(
            "polylogue.storage.backends.sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = MagicMock()
            mock_backend_class.return_value = mock_backend

            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.list.return_value = []

            config = SiteConfig(title="Empty Archive")
            builder = SiteBuilder(output_dir=output_dir, config=config)
            result = builder.build()

            assert result["conversations"] == 0
            assert (output_dir / "index.html").exists()

    def test_build_multiple_providers(
        self, tmp_path, mock_conversation, mock_conversation_gemini
    ):
        """SiteBuilder.build() groups conversations by provider."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        with patch(
            "polylogue.storage.backends.sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = MagicMock()
            mock_backend_class.return_value = mock_backend

            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.list.return_value = [mock_conversation, mock_conversation_gemini]

            config = SiteConfig(title="Multi-Provider Archive")
            builder = SiteBuilder(output_dir=output_dir, config=config)
            result = builder.build()

            assert result["conversations"] == 2
            # root index + claude provider + gemini provider + dashboard = 4
            assert result["index_pages"] == 4
            # Both provider directories should exist
            assert (output_dir / "claude").exists()
            assert (output_dir / "gemini").exists()

    def test_build_without_dashboard(self, tmp_path, mock_conversation):
        """SiteBuilder.build() respects include_dashboard config."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        with patch(
            "polylogue.storage.backends.sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = MagicMock()
            mock_backend_class.return_value = mock_backend

            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.list.return_value = [mock_conversation]

            config = SiteConfig(title="No Dashboard", include_dashboard=False)
            builder = SiteBuilder(output_dir=output_dir, config=config)
            result = builder.build()

            # root index + 1 provider index = 2 (no dashboard)
            assert result["index_pages"] == 2
            assert not (output_dir / "dashboard.html").exists()

    def test_build_with_dashboard(self, tmp_path, mock_conversation):
        """SiteBuilder.build() includes dashboard when enabled."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        with patch(
            "polylogue.storage.backends.sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = MagicMock()
            mock_backend_class.return_value = mock_backend

            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.list.return_value = [mock_conversation]

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
        self, tmp_path, mock_conversation
    ):
        """ConversationIndex doesn't try to access conv.source attribute.

        Regression test: Previously the code tried to access conv.source which
        doesn't exist on Conversation objects, causing:
        'Conversation' object has no attribute 'source'
        """
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        with patch(
            "polylogue.storage.backends.sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = MagicMock()
            mock_backend_class.return_value = mock_backend

            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.list.return_value = [mock_conversation]

            config = SiteConfig(title="Test")
            builder = SiteBuilder(output_dir=output_dir, config=config)

            # This should not raise AttributeError
            conversations = builder._build_index()

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

    def test_conversation_index_title_fallback(self, tmp_path, mock_conversation_no_title):
        """ConversationIndex falls back to ID when conversation has no title."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        with patch(
            "polylogue.storage.backends.sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = MagicMock()
            mock_backend_class.return_value = mock_backend

            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.list.return_value = [mock_conversation_no_title]

            config = SiteConfig(title="Test")
            builder = SiteBuilder(output_dir=output_dir, config=config)
            conversations = builder._build_index()

            assert len(conversations) == 1
            index = conversations[0]
            # Should use truncated ID when no title
            assert index.title.startswith("no-title")

    def test_conversation_index_message_count(self, tmp_path, mock_conversation):
        """ConversationIndex correctly counts messages."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        with patch(
            "polylogue.storage.backends.sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = MagicMock()
            mock_backend_class.return_value = mock_backend

            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.list.return_value = [mock_conversation]

            config = SiteConfig(title="Test")
            builder = SiteBuilder(output_dir=output_dir, config=config)
            conversations = builder._build_index()

            index = conversations[0]
            assert index.message_count == 3

    def test_conversation_index_preview_extraction(self, tmp_path, mock_conversation):
        """ConversationIndex extracts preview from first user message."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        with patch(
            "polylogue.storage.backends.sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = MagicMock()
            mock_backend_class.return_value = mock_backend

            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.list.return_value = [mock_conversation]

            config = SiteConfig(title="Test")
            builder = SiteBuilder(output_dir=output_dir, config=config)
            conversations = builder._build_index()

            index = conversations[0]
            assert index.preview == "Hello, how are you?"

    def test_conversation_index_timestamp_formatting(self, tmp_path, mock_conversation):
        """ConversationIndex formats timestamps correctly."""
        from polylogue.site.builder import SiteBuilder, SiteConfig

        output_dir = tmp_path / "site"

        with patch(
            "polylogue.storage.backends.sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = MagicMock()
            mock_backend_class.return_value = mock_backend

            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.list.return_value = [mock_conversation]

            config = SiteConfig(title="Test")
            builder = SiteBuilder(output_dir=output_dir, config=config)
            conversations = builder._build_index()

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
            "polylogue.storage.backends.sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = MagicMock()
            mock_backend_class.return_value = mock_backend

            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.list.return_value = []

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
            "polylogue.storage.backends.sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = MagicMock()
            mock_backend_class.return_value = mock_backend

            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.list.return_value = []

            from polylogue.ui import UI
            from polylogue.cli.types import AppEnv

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
            "polylogue.storage.backends.sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = MagicMock()
            mock_backend_class.return_value = mock_backend

            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.list.return_value = []

            from polylogue.ui import UI
            from polylogue.cli.types import AppEnv

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
            "polylogue.storage.backends.sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = MagicMock()
            mock_backend_class.return_value = mock_backend

            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.list.return_value = []

            from polylogue.ui import UI
            from polylogue.cli.types import AppEnv

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
            "polylogue.storage.backends.sqlite.SQLiteBackend"
        ) as mock_backend_class:
            mock_backend_class.side_effect = RuntimeError("Database error")

            from polylogue.ui import UI
            from polylogue.cli.types import AppEnv

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
        from polylogue.lib.models import Message
        from polylogue.lib.messages import MessageCollection

        runner = CliRunner()
        output_dir = workspace_env["archive_root"] / "site"

        # Create a real test conversation
        test_conv = Conversation(
            id="test-conv-123",
            provider="test",
            title="Test Conv",
            messages=MessageCollection(
                messages=[
                    Message(
                        id="m1",
                        role="user",
                        text="Hello",
                        timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
                    ),
                ]
            ),
            created_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        )

        with patch(
            "polylogue.storage.backends.sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = MagicMock()
            mock_backend_class.return_value = mock_backend

            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.list.return_value = [test_conv]

            from polylogue.ui import UI
            from polylogue.cli.types import AppEnv

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

        messages = [
            Message(
                id="m1",
                role="user",
                text="Test question",
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            ),
        ]
        conv = Conversation(
            id="conv-123",
            provider="test",
            title="Test",
            messages=MessageCollection(messages=messages),
        )

        with patch(
            "polylogue.storage.backends.sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = MagicMock()
            mock_backend_class.return_value = mock_backend

            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.list.return_value = [conv]

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

        messages = [
            Message(
                id="m1",
                role="user",
                text="Test question",
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            ),
        ]
        conv = Conversation(
            id="conv-123",
            provider="test",
            title="Test",
            messages=MessageCollection(messages=messages),
        )

        with patch(
            "polylogue.storage.backends.sqlite.SQLiteBackend"
        ) as mock_backend_class, patch(
            "polylogue.storage.repository.ConversationRepository"
        ) as mock_repo_class:
            mock_backend = MagicMock()
            mock_backend_class.return_value = mock_backend

            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.list.return_value = [conv]

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
            assert search_data[0]["provider"] == "test"
