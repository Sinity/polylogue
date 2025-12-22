"""Comprehensive idempotency tests for Polylogue operations.

Tests that operations can be safely re-run after interruption without
data loss or duplication.
"""
import json
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest

from polylogue import util
from polylogue.commands import CommandEnv, SyncOptions, sync_command
from polylogue.conversation import process_conversation
from polylogue.db import replace_messages
from polylogue.document_store import persist_document
from polylogue.pipeline_runner import Pipeline
from polylogue.render import MarkdownDocument
from polylogue.services.conversation_registrar import ConversationRegistrar
from polylogue.ui import create_ui


def _read_state_entry(state_home: Path, provider: str, conversation_id: str) -> dict:
    conn = sqlite3.connect(state_home / "polylogue.db")
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT metadata_json FROM conversations WHERE provider = ? AND conversation_id = ?",
            (provider, conversation_id),
        ).fetchone()
        if not row or not row["metadata_json"]:
            return {}
        return json.loads(row["metadata_json"])
    finally:
        conn.close()


class TestDatabaseIdempotency:
    """Test database operations are atomic and idempotent."""

    def test_replace_messages_atomic_on_interruption(self, tmp_path):
        """Verify replace_messages uses SAVEPOINT to prevent data loss."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        # Create schema
        conn.execute("CREATE TABLE messages (provider TEXT, conversation_id TEXT, branch_id TEXT, message_id TEXT, parent_id TEXT, position INT, timestamp TEXT, role TEXT, content_hash TEXT, rendered_text TEXT, raw_json TEXT, token_count INT, word_count INT, attachment_count INT, attachment_names TEXT, metadata_json TEXT)")
        conn.execute("CREATE TABLE messages_fts (provider TEXT, conversation_id TEXT, branch_id TEXT, message_id TEXT, content TEXT)")

        # Insert initial messages
        initial_messages = [
            {"message_id": "msg1", "rendered_text": "Hello"},
            {"message_id": "msg2", "rendered_text": "World"},
        ]

        for msg in initial_messages:
            conn.execute(
                "INSERT INTO messages (provider, conversation_id, branch_id, message_id, parent_id, position, timestamp, role, content_hash, rendered_text, raw_json, token_count, word_count, attachment_count, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("test", "conv1", "main", msg["message_id"], None, 0, None, "user", None, msg["rendered_text"], None, 0, 0, 0, None)
            )
        conn.commit()

        # Verify initial state
        count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert count == 2

        # Create a trigger that will cause INSERT to fail
        # This simulates a crash or error during the INSERT phase
        conn.execute("""
            CREATE TRIGGER prevent_insert
            BEFORE INSERT ON messages
            BEGIN
                SELECT RAISE(ABORT, 'Simulated failure');
            END;
        """)
        conn.commit()

        new_messages = [
            {"message_id": "msg3", "rendered_text": "New1", "parent_id": None,
             "position": 0, "timestamp": None, "role": "user", "content_hash": None,
             "raw_json": None, "token_count": 0, "word_count": 0, "attachment_count": 0,
             "metadata": None},
        ]

        # This should fail due to trigger
        try:
            replace_messages(
                conn,
                provider="test",
                conversation_id="conv1",
                branch_id="main",
                messages=new_messages
            )
        except sqlite3.IntegrityError:
            pass  # Expected due to trigger

        # Verify: OLD messages still present (SAVEPOINT rolled back)
        count = conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id='conv1'").fetchone()[0]
        assert count == 2, "Old messages should still exist after rollback"

        messages = conn.execute("SELECT message_id FROM messages ORDER BY message_id").fetchall()
        assert [m[0] for m in messages] == ["msg1", "msg2"]

    def test_replace_messages_commits_on_success(self, tmp_path):
        """Verify replace_messages commits when successful."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        
        conn.execute("CREATE TABLE messages (provider TEXT, conversation_id TEXT, branch_id TEXT, message_id TEXT, parent_id TEXT, position INT, timestamp TEXT, role TEXT, content_hash TEXT, rendered_text TEXT, raw_json TEXT, token_count INT, word_count INT, attachment_count INT, attachment_names TEXT, metadata_json TEXT)")
        conn.execute("CREATE TABLE messages_fts (provider TEXT, conversation_id TEXT, branch_id TEXT, message_id TEXT, content TEXT)")
        
        initial_messages = [{"message_id": "msg1", "rendered_text": "Hello"}]
        for msg in initial_messages:
            conn.execute(
                "INSERT INTO messages (provider, conversation_id, branch_id, message_id, parent_id, position, timestamp, role, content_hash, rendered_text, raw_json, token_count, word_count, attachment_count, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("test", "conv1", "main", msg["message_id"], None, 0, None, "user", None, msg["rendered_text"], None, 0, 0, 0, None)
            )
        conn.commit()
        
        # Replace messages successfully
        new_messages = [
            {
                "message_id": "msg2",
                "parent_id": None,
                "position": 0,
                "timestamp": None,
                "role": "assistant",
                "content_hash": None,
                "rendered_text": "Replaced",
                "raw_json": None,
                "token_count": 1,
                "word_count": 1,
                "attachment_count": 0,
                "metadata": None,
            }
        ]
        
        replace_messages(
            conn,
            provider="test",
            conversation_id="conv1",
            branch_id="main",
            messages=new_messages
        )
        
        # Verify new messages present
        count = conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id='conv1'").fetchone()[0]
        assert count == 1
        
        messages = conn.execute("SELECT message_id, rendered_text FROM messages").fetchall()
        assert messages[0] == ("msg2", "Replaced")


class TestForceAndAllowDirty:
    """Test --force and --allow-dirty flag interaction."""

    def test_force_alone_rejects_dirty_files(self, tmp_path):
        """Verify --force without --allow-dirty raises error on dirty files."""
        # Create conversation directory and file
        conv_dir = tmp_path / "test-conv"
        conv_dir.mkdir()
        md_path = conv_dir / "conversation.md"
        md_path.write_text("---\npolylogue:\n  localHash: abc123\n---\nOriginal content")

        # Create state with different hash (simulating user edit)
        registrar = Mock()
        registrar.get_state.return_value = {
            "localHash": "different",  # Different hash = dirty
            "contentHash": "xyz",  # Different from actual content hash
            "lastUpdated": "2024-01-01T00:00:00Z",
        }

        doc = MarkdownDocument(
            body="New content from remote",
            metadata={"polylogue": {}},
            attachments=[],
            stats={},
        )

        # Attempt persist with force=True but allow_dirty=False
        with pytest.raises(ValueError, match="local edits"):
            persist_document(
                provider="test",
                conversation_id="conv1",
                title="Test",
                document=doc,
                output_dir=tmp_path,
                collapse_threshold=24,
                attachments=[],
                updated_at="2024-01-01T00:00:00Z",
                created_at="2024-01-01T00:00:00Z",
                html=False,
                html_theme="light",
                force=True,
                allow_dirty=False,  # Should raise error
                registrar=registrar,
                slug_hint="test-conv",  # Control the slug
            )

    def test_force_with_allow_dirty_overwrites(self, tmp_path):
        """Verify --force --allow-dirty overwrites dirty files."""
        # Create conversation directory and file
        conv_dir = tmp_path / "test-conv"
        conv_dir.mkdir()
        md_path = conv_dir / "conversation.md"
        md_path.write_text("---\npolylogue:\n  localHash: abc123\n---\nUser edited content")

        registrar = Mock()
        registrar.get_state.return_value = {
            "localHash": "different",
            "contentHash": "xyz",  # Different from actual content hash
            "lastUpdated": "2024-01-01T00:00:00Z",
        }

        doc = MarkdownDocument(
            body="New content from remote",
            metadata={"polylogue": {}},
            attachments=[],
            stats={},
        )

        # Should succeed with both flags
        result = persist_document(
            provider="test",
            conversation_id="conv1",
            title="Test",
            document=doc,
            output_dir=tmp_path,
            collapse_threshold=24,
            attachments=[],
            updated_at="2024-01-01T00:00:00Z",
            created_at="2024-01-01T00:00:00Z",
            html=False,
            html_theme="light",
            force=True,
            allow_dirty=True,  # Both flags = allow overwrite
            registrar=registrar,
            slug_hint="test-conv",  # Control the slug
        )

        assert not result.skipped
        assert "New content from remote" in md_path.read_text()


class TestSyncIdempotency:
    """Test sync operations are idempotent."""

    def test_sync_twice_produces_same_result(self, tmp_path, state_env):
        """Running sync twice on same data produces identical output."""
        chats = [
            {"id": "drive-1", "name": "Alpha Thread", "modifiedTime": "2024-01-02T00:00:00Z"},
            {"id": "drive-2", "name": "Beta Thread", "modifiedTime": "2024-01-03T00:00:00Z"},
        ]
        payloads = {
            "drive-1": {
                "chunkedPrompt": {
                    "chunks": [
                        {"role": "user", "text": "Question one"},
                        {"role": "model", "text": "Answer one"},
                    ]
                }
            },
            "drive-2": {
                "chunkedPrompt": {
                    "chunks": [
                        {"role": "user", "text": "Question two"},
                        {"role": "model", "text": "Answer two"},
                    ]
                }
            },
        }

        class StubDrive:
            def resolve_folder_id(self, folder_name, folder_id):  # noqa: ARG002
                return folder_id or "folder-123"

            def list_chats(self, folder_name, folder_id):  # noqa: ARG002
                return list(chats)

            def download_chat_bytes(self, file_id):
                return json.dumps(payloads[file_id]).encode("utf-8")

        env = CommandEnv(ui=create_ui(plain=True))
        env.drive = StubDrive()

        options = SyncOptions(
            folder_name="AI Studio",
            folder_id="folder-123",
            output_dir=tmp_path / "drive",
            collapse_threshold=12,
            download_attachments=False,
            dry_run=False,
            force=False,
            prune=False,
            since=None,
            until=None,
            name_filter=None,
            html=False,
            html_theme="light",
            diff=False,
            prefetched_chats=list(chats),
        )

        first_result = sync_command(options, env)
        assert first_result.count == len(chats)
        rendered = {
            util.sanitize_filename(chat["name"]): (options.output_dir / util.sanitize_filename(chat["name"]) / "conversation.md").read_text()
            for chat in chats
        }

        second_result = sync_command(options, env)
        assert second_result.count == 0
        assert second_result.items == []
        assert second_result.total_stats.get("skipped") == len(chats)

        for chat in chats:
            slug = util.sanitize_filename(chat["name"])
            md_path = options.output_dir / slug / "conversation.md"
            assert md_path.read_text() == rendered[slug]

        conn = sqlite3.connect(state_env / "polylogue.db")
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT conversation_id, COUNT(*) AS c FROM conversations WHERE provider = ? GROUP BY conversation_id",
                ("drive-sync",),
            ).fetchall()
        finally:
            conn.close()
        assert [row["conversation_id"] for row in rows] == ["drive-1", "drive-2"]
        assert all(row["c"] == 1 for row in rows)

    def test_interrupted_sync_resumable(self, tmp_path, state_env, monkeypatch):
        """Interrupted sync can resume without duplicating work."""
        chats = [
            {"id": "drive-1", "name": "Alpha Thread", "modifiedTime": "2024-01-02T00:00:00Z"},
            {"id": "drive-2", "name": "Beta Thread", "modifiedTime": "2024-01-03T00:00:00Z"},
        ]
        payloads = {
            "drive-1": {
                "chunkedPrompt": {
                    "chunks": [
                        {"role": "user", "text": "Question one"},
                        {"role": "model", "text": "Answer one"},
                    ]
                }
            },
            "drive-2": {
                "chunkedPrompt": {
                    "chunks": [
                        {"role": "user", "text": "Question two"},
                        {"role": "model", "text": "Answer two"},
                    ]
                }
            },
        }

        class StubDrive:
            def resolve_folder_id(self, folder_name, folder_id):  # noqa: ARG002
                return folder_id or "folder-123"

            def list_chats(self, folder_name, folder_id):  # noqa: ARG002
                return list(chats)

            def download_chat_bytes(self, file_id):
                payload = payloads[file_id]
                return json.dumps(payload).encode("utf-8")

        ui = create_ui(plain=True)
        env = CommandEnv(ui=ui)
        env.drive = StubDrive()

        options = SyncOptions(
            folder_name="AI Studio",
            folder_id="folder-123",
            output_dir=tmp_path / "drive",
            collapse_threshold=12,
            download_attachments=False,
            dry_run=False,
            force=False,
            prune=False,
            since=None,
            until=None,
            name_filter=None,
            html=False,
            html_theme="light",
            diff=False,
            prefetched_chats=list(chats),
        )

        original_run = Pipeline.run
        should_interrupt = True

        def _interrupting_run(self, context):
            nonlocal should_interrupt
            metadata = context.get("metadata")
            if should_interrupt and isinstance(metadata, dict) and metadata.get("id") == "drive-2":
                should_interrupt = False
                raise KeyboardInterrupt("simulated interrupt")
            return original_run(self, context)

        monkeypatch.setattr(Pipeline, "run", _interrupting_run)

        with pytest.raises(KeyboardInterrupt):
            sync_command(options, env)

        output_dir = options.output_dir
        slug_one = util.sanitize_filename(chats[0]["name"])
        slug_two = util.sanitize_filename(chats[1]["name"])
        conv_one = output_dir / slug_one / "conversation.md"
        conv_two = output_dir / slug_two / "conversation.md"

        assert conv_one.exists()
        assert not conv_two.exists()

        first_state = _read_state_entry(state_env, "drive-sync", "drive-1")
        assert first_state["slug"] == slug_one
        assert first_state["outputPath"] == str(conv_one)
        assert _read_state_entry(state_env, "drive-sync", "drive-2") == {}

        result = sync_command(options, env)

        assert conv_one.exists()
        assert conv_two.exists()
        assert any(item.slug == slug_two for item in result.items)

        second_state = _read_state_entry(state_env, "drive-sync", "drive-2")
        assert second_state["slug"] == slug_two
        assert second_state["outputPath"] == str(conv_two)

        conn = sqlite3.connect(state_env / "polylogue.db")
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT conversation_id FROM conversations WHERE provider = ? ORDER BY conversation_id",
                ("drive-sync",),
            ).fetchall()
        finally:
            conn.close()
        assert [row["conversation_id"] for row in rows] == ["drive-1", "drive-2"]

        md_paths = sorted(path.relative_to(output_dir) for path in output_dir.rglob("conversation.md"))
        assert md_paths == [
            Path(slug_one) / "conversation.md",
            Path(slug_two) / "conversation.md",
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
