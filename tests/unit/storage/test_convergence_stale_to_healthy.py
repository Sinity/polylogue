"""Convergence laws: stale → repair → healthy transitions.

Proves that archive debt detected by health/repair queries converges
to zero after repair execution. These are temporal invariants — they
test state transitions, not just snapshots.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.config import Config, get_config
from tests.infra.storage_records import ConversationBuilder, db_setup


@pytest.fixture()
def archive_with_orphans(workspace_env: dict[str, Path]) -> Config:
    """Create a DB with valid conversations and injected orphan debt."""
    db_path = db_setup(workspace_env)

    ConversationBuilder(db_path, "healthy-1").provider("chatgpt").title("Healthy").add_message(
        role="user", text="A valid message"
    ).add_message(role="assistant", text="A valid reply").save()

    ConversationBuilder(db_path, "healthy-2").provider("claude-code").title("Also healthy").add_message(
        role="user", text="Another message"
    ).save()

    from polylogue.storage.sqlite.connection import open_connection

    with open_connection(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            "INSERT INTO messages (message_id, conversation_id, role, text, content_hash, version, "
            "provider_name, word_count, has_tool_use, has_thinking) "
            "VALUES ('orphan-m1', 'deleted-conv', 'user', 'orphan text', 'oh1', 1, 'test', 2, 0, 0)"
        )
        conn.execute(
            "INSERT INTO messages (message_id, conversation_id, role, text, content_hash, version, "
            "provider_name, word_count, has_tool_use, has_thinking) "
            "VALUES ('orphan-m2', 'deleted-conv', 'assistant', 'orphan reply', 'oh2', 1, 'test', 2, 0, 0)"
        )
        conn.commit()
        conn.execute("PRAGMA foreign_keys = ON")

    return get_config()


@pytest.fixture()
def archive_with_empty_conversations(workspace_env: dict[str, Path]) -> Config:
    """Create a DB with valid conversations and injected empty conversation debt."""
    db_path = db_setup(workspace_env)

    ConversationBuilder(db_path, "has-msgs").provider("chatgpt").title("Has messages").add_message(
        role="user", text="Real message"
    ).save()

    from polylogue.storage.sqlite.connection import open_connection

    with open_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO conversations (conversation_id, provider_name, provider_conversation_id, content_hash, version) "
            "VALUES ('empty-1', 'test', 'empty-prov-1', 'eh1', 1)"
        )
        conn.execute(
            "INSERT INTO conversations (conversation_id, provider_name, provider_conversation_id, content_hash, version) "
            "VALUES ('empty-2', 'test', 'empty-prov-2', 'eh2', 1)"
        )
        conn.commit()

    return get_config()


class TestOrphanMessageConvergence:
    """debt(orphaned messages) > 0 → repair → debt = 0 → queries exclude orphans."""

    def test_orphan_detected_then_repaired_to_zero(self, archive_with_orphans: Config) -> None:
        from polylogue.storage.repair import (
            count_orphaned_messages_sync,
            repair_orphaned_messages,
        )
        from polylogue.storage.sqlite.connection import open_connection

        with open_connection(archive_with_orphans.db_path) as conn:
            before = count_orphaned_messages_sync(conn)
        assert before == 2, f"Expected 2 orphans, got {before}"

        result = repair_orphaned_messages(archive_with_orphans, dry_run=False)
        assert result.repaired_count == 2

        with open_connection(archive_with_orphans.db_path) as conn:
            after = count_orphaned_messages_sync(conn)
        assert after == 0, f"After repair, expected 0 orphans, got {after}"

    def test_healthy_messages_survive_orphan_repair(self, archive_with_orphans: Config) -> None:
        """Repair must not touch non-orphaned messages."""
        from polylogue.storage.repair import repair_orphaned_messages
        from polylogue.storage.sqlite.connection import open_connection

        with open_connection(archive_with_orphans.db_path) as conn:
            healthy_before = conn.execute(
                "SELECT COUNT(*) FROM messages m "
                "WHERE EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = m.conversation_id)"
            ).fetchone()[0]

        repair_orphaned_messages(archive_with_orphans, dry_run=False)

        with open_connection(archive_with_orphans.db_path) as conn:
            healthy_after = conn.execute(
                "SELECT COUNT(*) FROM messages m "
                "WHERE EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = m.conversation_id)"
            ).fetchone()[0]

        assert healthy_after == healthy_before, f"Repair damaged healthy messages: {healthy_before} → {healthy_after}"

    def test_repair_is_idempotent(self, archive_with_orphans: Config) -> None:
        """Second repair after convergence is a no-op."""
        from polylogue.storage.repair import repair_orphaned_messages

        repair_orphaned_messages(archive_with_orphans, dry_run=False)
        result2 = repair_orphaned_messages(archive_with_orphans, dry_run=False)
        assert result2.repaired_count == 0


class TestEmptyConversationConvergence:
    """debt(empty conversations) > 0 → repair → debt = 0."""

    def test_empty_detected_then_repaired_to_zero(self, archive_with_empty_conversations: Config) -> None:
        from polylogue.storage.repair import (
            count_empty_conversations_sync,
            repair_empty_conversations,
        )
        from polylogue.storage.sqlite.connection import open_connection

        with open_connection(archive_with_empty_conversations.db_path) as conn:
            before = count_empty_conversations_sync(conn)
        assert before == 2, f"Expected 2 empty conversations, got {before}"

        result = repair_empty_conversations(archive_with_empty_conversations, dry_run=False)
        assert result.repaired_count == 2

        with open_connection(archive_with_empty_conversations.db_path) as conn:
            after = count_empty_conversations_sync(conn)
        assert after == 0

    def test_non_empty_conversations_survive(self, archive_with_empty_conversations: Config) -> None:
        from polylogue.storage.repair import repair_empty_conversations
        from polylogue.storage.sqlite.connection import open_connection

        repair_empty_conversations(archive_with_empty_conversations, dry_run=False)

        with open_connection(archive_with_empty_conversations.db_path) as conn:
            remaining = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        assert remaining == 1, "Only the conversation with messages should survive"

    def test_repair_is_idempotent(self, archive_with_empty_conversations: Config) -> None:
        from polylogue.storage.repair import repair_empty_conversations

        repair_empty_conversations(archive_with_empty_conversations, dry_run=False)
        result2 = repair_empty_conversations(archive_with_empty_conversations, dry_run=False)
        assert result2.repaired_count == 0


class TestDryRunSafety:
    """Dry-run must never mutate archive state."""

    def test_dry_run_orphan_repair_preserves_state(self, archive_with_orphans: Config) -> None:
        from polylogue.storage.repair import (
            count_orphaned_messages_sync,
            repair_orphaned_messages,
        )
        from polylogue.storage.sqlite.connection import open_connection

        with open_connection(archive_with_orphans.db_path) as conn:
            before = count_orphaned_messages_sync(conn)

        repair_orphaned_messages(archive_with_orphans, dry_run=True)

        with open_connection(archive_with_orphans.db_path) as conn:
            after = count_orphaned_messages_sync(conn)

        assert after == before, "Dry-run mutated state"

    def test_dry_run_empty_repair_preserves_state(self, archive_with_empty_conversations: Config) -> None:
        from polylogue.storage.repair import (
            count_empty_conversations_sync,
            repair_empty_conversations,
        )
        from polylogue.storage.sqlite.connection import open_connection

        with open_connection(archive_with_empty_conversations.db_path) as conn:
            before = count_empty_conversations_sync(conn)

        repair_empty_conversations(archive_with_empty_conversations, dry_run=True)

        with open_connection(archive_with_empty_conversations.db_path) as conn:
            after = count_empty_conversations_sync(conn)

        assert after == before, "Dry-run mutated state"
