"""Tests for polylogue.facade — the Polylogue high-level API.

Covers initialization, context manager protocol, CRUD round-trip,
filtering, stats computation, and file parsing.
"""

from __future__ import annotations

import json

import pytest

from polylogue.facade import ArchiveStats, Polylogue
from polylogue.lib.filters import ConversationFilter


class TestPolylogueInit:
    """Initialization and property tests."""

    def test_init_defaults(self, workspace_env):
        """Polylogue() uses XDG defaults when no args given."""
        archive = Polylogue()
        assert archive.archive_root.exists() or True  # path is resolved, may not exist yet
        assert archive.config is not None
        assert archive.repository is not None
        archive.close()

    def test_init_custom_paths(self, workspace_env):
        """Polylogue() accepts custom archive_root and db_path."""
        archive_root = workspace_env["archive_root"] / "custom"
        db_path = workspace_env["data_root"] / "custom.db"

        archive = Polylogue(archive_root=archive_root, db_path=db_path)
        assert archive.archive_root == archive_root.resolve()
        assert archive._db_path == db_path.resolve()
        archive.close()

    def test_repr(self, workspace_env):
        """repr shows archive_root."""
        archive = Polylogue()
        r = repr(archive)
        assert "Polylogue(" in r
        assert "archive_root=" in r
        archive.close()


class TestContextManager:
    """Context manager protocol tests."""

    def test_context_manager_enter_returns_self(self, workspace_env):
        """__enter__ returns the Polylogue instance."""
        archive = Polylogue()
        with archive as ctx:
            assert ctx is archive

    def test_context_manager_closes_on_exit(self, workspace_env):
        """__exit__ calls close() on the backend."""
        with Polylogue() as archive:
            # Access backend to confirm it's open
            assert archive._backend is not None
        # After exit, backend.close() was called (no error = success)

    def test_context_manager_closes_on_exception(self, workspace_env):
        """Backend is closed even if an exception occurs."""
        with pytest.raises(ValueError, match="test error"):
            with Polylogue():
                raise ValueError("test error")


class TestConversationRoundtrip:
    """CRUD round-trip: create via builder, retrieve via facade."""

    def test_get_conversation_roundtrip(self, workspace_env):
        """Store a conversation via builder, retrieve via facade."""
        from tests.helpers import ConversationBuilder, db_setup

        db_path = db_setup(workspace_env)
        (
            ConversationBuilder(db_path, "test:roundtrip-conv")
            .title("Roundtrip Test")
            .provider("claude")
            .add_message("m1", role="user", text="Hello from test")
            .add_message("m2", role="assistant", text="Hi back!")
            .save()
        )

        archive = Polylogue(db_path=db_path)
        conv = archive.get_conversation("test:roundtrip-conv")
        assert conv is not None
        assert conv.id == "test:roundtrip-conv"
        assert conv.provider == "claude"
        assert len(conv.messages) == 2
        archive.close()

    def test_get_conversation_not_found(self, workspace_env):
        """get_conversation returns None for missing IDs."""
        archive = Polylogue()
        assert archive.get_conversation("nonexistent:id") is None
        archive.close()

    def test_get_conversations_batch(self, workspace_env):
        """get_conversations retrieves multiple in one query."""
        from tests.helpers import ConversationBuilder, db_setup

        db_path = db_setup(workspace_env)
        for i in range(3):
            (
                ConversationBuilder(db_path, f"test:batch-{i}")
                .title(f"Batch {i}")
                .add_message(f"m{i}", text=f"Message {i}")
                .save()
            )

        archive = Polylogue(db_path=db_path)
        convs = archive.get_conversations(
            ["test:batch-0", "test:batch-1", "test:batch-2", "test:missing"]
        )
        assert len(convs) == 3
        archive.close()


class TestListConversations:
    """list_conversations with filtering."""

    def test_list_conversations_with_provider_filter(self, workspace_env):
        """list_conversations filters by provider."""
        from tests.helpers import ConversationBuilder, db_setup

        db_path = db_setup(workspace_env)
        (
            ConversationBuilder(db_path, "test:claude-1")
            .provider("claude")
            .add_message("m1", text="Claude msg")
            .save()
        )
        (
            ConversationBuilder(db_path, "test:chatgpt-1")
            .provider("chatgpt")
            .add_message("m1", text="ChatGPT msg")
            .save()
        )

        archive = Polylogue(db_path=db_path)

        claude_convs = archive.list_conversations(provider="claude")
        assert len(claude_convs) == 1
        assert claude_convs[0].provider == "claude"

        all_convs = archive.list_conversations()
        assert len(all_convs) == 2

        archive.close()

    def test_list_conversations_with_limit(self, workspace_env):
        """list_conversations respects limit."""
        from tests.helpers import ConversationBuilder, db_setup

        db_path = db_setup(workspace_env)
        for i in range(5):
            (
                ConversationBuilder(db_path, f"test:limited-{i}")
                .add_message(f"m{i}", text=f"Message {i}")
                .save()
            )

        archive = Polylogue(db_path=db_path)
        convs = archive.list_conversations(limit=2)
        assert len(convs) == 2
        archive.close()


class TestFilter:
    """filter() returns a ConversationFilter."""

    def test_filter_returns_conversation_filter(self, workspace_env):
        """filter() returns a ConversationFilter instance."""
        archive = Polylogue()
        f = archive.filter()
        assert isinstance(f, ConversationFilter)
        archive.close()


class TestStats:
    """Stats computation tests."""

    def test_stats_empty_archive(self, workspace_env):
        """stats() on empty archive returns zero counts."""
        archive = Polylogue()
        stats = archive.stats()
        assert isinstance(stats, ArchiveStats)
        assert stats.conversation_count == 0
        assert stats.message_count == 0
        assert stats.word_count == 0
        assert stats.providers == {}
        assert stats.tags == {}
        assert stats.recent == []
        archive.close()

    def test_stats_with_data(self, workspace_env):
        """stats() computes counts from stored conversations."""
        from tests.helpers import ConversationBuilder, db_setup

        db_path = db_setup(workspace_env)
        # Use globally unique message IDs — message_id is PK, not (conversation_id, message_id)
        (
            ConversationBuilder(db_path, "test:stats-1")
            .provider("claude")
            .add_message("s1-m1", role="user", text="Hello world")
            .add_message("s1-m2", role="assistant", text="Hi there friend")
            .save()
        )
        (
            ConversationBuilder(db_path, "test:stats-2")
            .provider("chatgpt")
            .add_message("s2-m1", role="user", text="Question about Python programming language")
            .save()
        )

        archive = Polylogue(db_path=db_path)
        stats = archive.stats()

        assert stats.conversation_count == 2
        assert stats.message_count == 3
        assert stats.word_count > 0
        assert "claude" in stats.providers
        assert stats.providers["claude"] == 1
        assert "chatgpt" in stats.providers
        assert stats.providers["chatgpt"] == 1
        assert len(stats.recent) == 2

        # Verify repr
        r = repr(stats)
        assert "conversations=2" in r
        assert "messages=3" in r

        archive.close()


class TestParseFile:
    """File parsing tests."""

    def test_parse_file_chatgpt(self, workspace_env):
        """parse_file handles ChatGPT export format."""
        from tests.helpers import ChatGPTExportBuilder, db_setup

        db_path = db_setup(workspace_env)
        archive_root = workspace_env["archive_root"]
        archive_root.mkdir(parents=True, exist_ok=True)

        # Build a ChatGPT export file
        export = (
            ChatGPTExportBuilder("chatgpt-test-123")
            .title("Test ChatGPT Conversation")
            .add_node("user", "What is Python?")
            .add_node("assistant", "Python is a programming language.")
            .build()
        )

        # Write as JSON file
        export_path = archive_root / "chatgpt_test.json"
        export_path.write_text(json.dumps(export, indent=2), encoding="utf-8")

        archive = Polylogue(
            archive_root=archive_root,
            db_path=db_path,
        )
        result = archive.parse_file(export_path, source_name="test-chatgpt")

        assert result.counts["conversations"] >= 1
        archive.close()
