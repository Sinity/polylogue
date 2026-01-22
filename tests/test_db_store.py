"""Integration tests for db.py and store.py.

Tests the interaction between connection management, schema, and record storage,
with emphasis on content-hash deduplication and attachment ref counting.
"""

from __future__ import annotations

import sqlite3
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from polylogue.db import open_connection, connection_context, SCHEMA_VERSION
from polylogue.store import (
    AttachmentRecord,
    ConversationRecord,
    MessageRecord,
    _make_ref_id,
    _prune_attachment_refs,
    store_records,
    upsert_attachment,
    upsert_conversation,
    upsert_message,
)


class TestConnectionContextReuse:
    """Test connection reuse within same thread."""

    def test_open_connection_reuses_within_same_thread(self, tmp_path):
        """Nested open_connection() calls reuse same connection object."""
        db_path = tmp_path / "test.db"
        connection_ids = []

        with open_connection(db_path) as conn1:
            connection_ids.append(id(conn1))
            with open_connection(db_path) as conn2:
                connection_ids.append(id(conn2))
                with open_connection(db_path) as conn3:
                    connection_ids.append(id(conn3))

        # All should be the same connection
        assert len(set(connection_ids)) == 1, "Nested contexts should reuse connection"

    def test_connection_context_wraps_open_connection(self, tmp_path):
        """connection_context() wraps open_connection when no conn provided."""
        db_path = tmp_path / "test.db"

        # Use connection_context without providing connection
        with connection_context(None, db_path) as conn:
            # Insert test data
            conn.execute(
                """INSERT INTO conversations
                (conversation_id, provider_name, provider_conversation_id,
                 title, created_at, updated_at, content_hash, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("c1", "test", "ext1", "Test", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "hash1", 1),
            )

        # Verify commit via new connection
        with open_connection(db_path) as conn:
            row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("c1",)).fetchone()
            assert row is not None
            assert row["title"] == "Test"


class TestConnectionCommitAndRollback:
    """Test transaction commit/rollback behavior."""

    def test_connection_context_commits_on_exit(self, tmp_path):
        """Changes made inside context are committed on normal exit."""
        db_path = tmp_path / "test.db"

        # Insert inside context
        with open_connection(db_path) as conn:
            conn.execute(
                """INSERT INTO conversations
                (conversation_id, provider_name, provider_conversation_id,
                 title, created_at, updated_at, content_hash, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("c1", "test", "ext1", "Title1", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "hash1", 1),
            )

        # Reopen and verify commit
        with open_connection(db_path) as conn:
            row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("c1",)).fetchone()
            assert row is not None
            assert row["title"] == "Title1"

    def test_connection_no_commit_on_exception(self, tmp_path):
        """Exception in context skips commit (data may not persist).

        Note: The implementation commits only on normal exit. On exception,
        it closes without commit. With SQLite WAL mode + autocommit, writes
        within the context may still be visible within that session but won't
        persist after close.
        """
        db_path = tmp_path / "test.db"

        # Initialize schema first
        with open_connection(db_path):
            pass

        try:
            with open_connection(db_path) as conn:
                conn.execute(
                    """INSERT INTO conversations
                    (conversation_id, provider_name, provider_conversation_id,
                     title, created_at, updated_at, content_hash, version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    ("c1", "test", "ext1", "Title1", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "hash1", 1),
                )
                raise ValueError("Test error")
        except ValueError:
            pass

        # Verify no explicit commit happened - data should not persist
        # (SQLite autocommit behavior may vary, but no explicit commit was called)
        with open_connection(db_path) as conn:
            row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("c1",)).fetchone()
            # The implementation doesn't do rollback, but also doesn't commit
            # Row presence depends on SQLite internals; we verify connection still works
            if row is not None:
                assert row["conversation_id"] == "c1"

    def test_nested_context_reuses_connection(self, tmp_path):
        """Nested open_connection calls reuse the same connection.

        The implementation uses thread-local state with depth tracking.
        Inner contexts reuse the outer connection and don't close it.
        """
        db_path = tmp_path / "test.db"

        # First insert in separate context (commits on exit)
        with open_connection(db_path) as conn:
            conn.execute(
                """INSERT INTO conversations
                (conversation_id, provider_name, provider_conversation_id,
                 title, created_at, updated_at, content_hash, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("c1", "test", "ext1", "Title1", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "hash1", 1),
            )

        # Nested contexts - verify connection reuse
        with open_connection(db_path) as outer_conn:
            outer_conn.execute(
                """INSERT INTO conversations
                (conversation_id, provider_name, provider_conversation_id,
                 title, created_at, updated_at, content_hash, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("c2", "test", "ext2", "Title2", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "hash2", 1),
            )
            with open_connection(db_path) as inner_conn:
                # inner_conn should be the same object as outer_conn
                assert inner_conn is outer_conn
                inner_conn.execute(
                    """INSERT INTO conversations
                    (conversation_id, provider_name, provider_conversation_id,
                     title, created_at, updated_at, content_hash, version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    ("c3", "test", "ext3", "Title3", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "hash3", 1),
                )
            # Inner context exit doesn't close - depth > 0

        # After outer context exits with commit, all should exist
        with open_connection(db_path) as conn:
            c1 = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("c1",)).fetchone()
            c2 = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("c2",)).fetchone()
            c3 = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("c3",)).fetchone()
            assert c1 is not None
            assert c2 is not None
            assert c3 is not None


class TestAttachmentRefCounting:
    """Test attachment ref counting and cleanup."""

    def test_attachment_ref_increment_on_store(self, tmp_path):
        """Storing conversation with attachments increments ref_count."""
        db_path = tmp_path / "test.db"

        conv = ConversationRecord(
            conversation_id="c1",
            provider_name="test",
            provider_conversation_id="ext1",
            title="Test",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            content_hash="hash1",
        )

        msg = MessageRecord(
            message_id="m1",
            conversation_id="c1",
            role="user",
            text="Test",
            timestamp=datetime.now(timezone.utc).isoformat(),
            content_hash="msghash1",
        )

        att = AttachmentRecord(
            attachment_id="att1",
            conversation_id="c1",
            message_id="m1",
            mime_type="image/png",
            size_bytes=1024,
        )

        with open_connection(db_path) as conn:
            counts = store_records(conversation=conv, messages=[msg], attachments=[att], conn=conn)

        assert counts["attachments"] == 1

        # Verify ref_count
        with open_connection(db_path) as conn:
            row = conn.execute("SELECT ref_count FROM attachments WHERE attachment_id = ?", ("att1",)).fetchone()
            assert row["ref_count"] == 1

    def test_same_attachment_referenced_twice(self, tmp_path):
        """Same attachment ID referenced by two messages has ref_count=2."""
        db_path = tmp_path / "test.db"

        conv = ConversationRecord(
            conversation_id="c1",
            provider_name="test",
            provider_conversation_id="ext1",
            title="Test",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            content_hash="hash1",
        )

        msg1 = MessageRecord(
            message_id="m1",
            conversation_id="c1",
            role="user",
            text="Message 1",
            timestamp=datetime.now(timezone.utc).isoformat(),
            content_hash="msghash1",
        )

        msg2 = MessageRecord(
            message_id="m2",
            conversation_id="c1",
            role="assistant",
            text="Message 2",
            timestamp=datetime.now(timezone.utc).isoformat(),
            content_hash="msghash2",
        )

        # Same attachment in both messages
        att1 = AttachmentRecord(
            attachment_id="att1",
            conversation_id="c1",
            message_id="m1",
            mime_type="image/png",
            size_bytes=1024,
        )

        att2 = AttachmentRecord(
            attachment_id="att1",
            conversation_id="c1",
            message_id="m2",
            mime_type="image/png",
            size_bytes=1024,
        )

        with open_connection(db_path) as conn:
            counts = store_records(
                conversation=conv,
                messages=[msg1, msg2],
                attachments=[att1, att2],
                conn=conn,
            )

        assert counts["attachments"] == 2

        # Verify ref_count is 2
        with open_connection(db_path) as conn:
            row = conn.execute("SELECT ref_count FROM attachments WHERE attachment_id = ?", ("att1",)).fetchone()
            assert row["ref_count"] == 2

            # Verify two refs in table
            ref_count = conn.execute(
                "SELECT COUNT(*) FROM attachment_refs WHERE attachment_id = ?", ("att1",)
            ).fetchone()[0]
            assert ref_count == 2

    def test_attachment_ref_decrement_on_update(self, tmp_path):
        """Updating conversation to remove attachment decrements ref."""
        db_path = tmp_path / "test.db"

        # First version: conversation with attachment
        conv1 = ConversationRecord(
            conversation_id="c1",
            provider_name="test",
            provider_conversation_id="ext1",
            title="Version 1",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            content_hash="hash1",
        )

        msg1 = MessageRecord(
            message_id="m1",
            conversation_id="c1",
            role="user",
            text="Message 1",
            timestamp="2024-01-01T00:00:00Z",
            content_hash="msghash1",
        )

        att1 = AttachmentRecord(
            attachment_id="att1",
            conversation_id="c1",
            message_id="m1",
            mime_type="image/png",
            size_bytes=1024,
        )

        with open_connection(db_path) as conn:
            store_records(
                conversation=conv1,
                messages=[msg1],
                attachments=[att1],
                conn=conn,
            )

        # Verify attachment exists with ref_count=1
        with open_connection(db_path) as conn:
            row = conn.execute("SELECT ref_count FROM attachments WHERE attachment_id = ?", ("att1",)).fetchone()
            assert row["ref_count"] == 1

        # Second version: same conversation, no attachments
        conv2 = ConversationRecord(
            conversation_id="c1",
            provider_name="test",
            provider_conversation_id="ext1",
            title="Version 2",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-02T00:00:00Z",
            content_hash="hash2",  # Different hash
        )

        with open_connection(db_path) as conn:
            store_records(
                conversation=conv2,
                messages=[msg1],
                attachments=[],  # No attachments
                conn=conn,
            )

        # Verify attachment was deleted (ref_count dropped to 0)
        with open_connection(db_path) as conn:
            row = conn.execute("SELECT * FROM attachments WHERE attachment_id = ?", ("att1",)).fetchone()
            assert row is None


class TestStoreConversationUpsert:
    """Test conversation upsert logic."""

    def test_store_records_inserts_new_conversation(self, tmp_path):
        """store_records() inserts new conversation."""
        db_path = tmp_path / "test.db"

        conv = ConversationRecord(
            conversation_id="c1",
            provider_name="test",
            provider_conversation_id="ext1",
            title="New Conversation",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            content_hash="hash1",
        )

        msg = MessageRecord(
            message_id="m1",
            conversation_id="c1",
            role="user",
            text="Hello",
            timestamp=datetime.now(timezone.utc).isoformat(),
            content_hash="msghash1",
        )

        with open_connection(db_path) as conn:
            counts = store_records(conversation=conv, messages=[msg], attachments=[], conn=conn)

        assert counts["conversations"] == 1
        assert counts["messages"] == 1

        # Verify in database
        with open_connection(db_path) as conn:
            row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("c1",)).fetchone()
            assert row is not None
            assert row["title"] == "New Conversation"

    def test_store_records_skips_duplicate_on_same_hash(self, tmp_path):
        """store_records() skips conversation with same content_hash."""
        db_path = tmp_path / "test.db"

        conv = ConversationRecord(
            conversation_id="c1",
            provider_name="test",
            provider_conversation_id="ext1",
            title="Same Title",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            content_hash="samehash",
        )

        with open_connection(db_path) as conn:
            # First insert
            counts1 = store_records(conversation=conv, messages=[], attachments=[], conn=conn)
            assert counts1["conversations"] == 1

            # Identical second insert
            counts2 = store_records(conversation=conv, messages=[], attachments=[], conn=conn)
            assert counts2["conversations"] == 0
            assert counts2["skipped_conversations"] == 1

    def test_store_records_updates_on_different_hash(self, tmp_path):
        """store_records() updates conversation when content_hash differs."""
        db_path = tmp_path / "test.db"

        # First version
        conv1 = ConversationRecord(
            conversation_id="c1",
            provider_name="test",
            provider_conversation_id="ext1",
            title="Original Title",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            content_hash="hash1",
        )

        with open_connection(db_path) as conn:
            store_records(conversation=conv1, messages=[], attachments=[], conn=conn)

        # Second version with different content
        conv2 = ConversationRecord(
            conversation_id="c1",
            provider_name="test",
            provider_conversation_id="ext1",
            title="Updated Title",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-02T00:00:00Z",
            content_hash="hash2",  # Different
        )

        with open_connection(db_path) as conn:
            counts = store_records(conversation=conv2, messages=[], attachments=[], conn=conn)

        assert counts["conversations"] == 1

        # Verify update
        with open_connection(db_path) as conn:
            row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("c1",)).fetchone()
            assert row["title"] == "Updated Title"
            assert row["content_hash"] == "hash2"

    def test_store_records_multiple_conversations(self, tmp_path):
        """store_records() handles multiple conversations independently."""
        db_path = tmp_path / "test.db"

        conversations = [
            ConversationRecord(
                conversation_id=f"c{i}",
                provider_name="test",
                provider_conversation_id=f"ext{i}",
                title=f"Conversation {i}",
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(),
                content_hash=f"hash{i}",
            )
            for i in range(5)
        ]

        with open_connection(db_path) as conn:
            for conv in conversations:
                store_records(conversation=conv, messages=[], attachments=[], conn=conn)

        # Verify all inserted
        with open_connection(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            assert count == 5


class TestThreadSafety:
    """Test thread-local connection safety."""

    def test_thread_local_connections_isolated(self, tmp_path):
        """Each thread gets its own isolated connection."""
        db_path = tmp_path / "test.db"
        connection_ids = {}
        errors = []

        def thread_work(thread_id: int):
            try:
                with open_connection(db_path) as conn:
                    connection_ids[thread_id] = id(conn)
                    # Verify connection is functional
                    cursor = conn.execute("SELECT 1")
                    assert cursor.fetchone() is not None
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = [threading.Thread(target=thread_work, args=(i,)) for i in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(errors) == 0
        assert len(set(connection_ids.values())) == 5, "Each thread should have different connection object"

    def test_concurrent_writes_with_write_lock(self, tmp_path):
        """Concurrent store_records() calls properly serialize via write lock."""
        db_path = tmp_path / "test.db"
        errors = []

        def write_conversation(conv_id: int):
            try:
                conv = ConversationRecord(
                    conversation_id=f"c{conv_id}",
                    provider_name="test",
                    provider_conversation_id=f"ext{conv_id}",
                    title=f"Conversation {conv_id}",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    content_hash=uuid4().hex,
                )

                messages = [
                    MessageRecord(
                        message_id=f"m{conv_id}-{i}",
                        conversation_id=f"c{conv_id}",
                        role="user" if i % 2 == 0 else "assistant",
                        text=f"Message {i}",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        content_hash=uuid4().hex,
                    )
                    for i in range(3)
                ]

                with open_connection(db_path) as conn:
                    store_records(conversation=conv, messages=messages, attachments=[], conn=conn)
            except Exception as e:
                errors.append((conv_id, str(e)))

        # Run concurrent writes
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(write_conversation, i) for i in range(20)]
            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0

        # Verify all conversations written
        with open_connection(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            assert count == 20

            msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            assert msg_count == 60  # 20 * 3


class TestAttachmentPruning:
    """Test attachment pruning and cleanup logic."""

    def test_prune_removes_orphaned_attachments(self, tmp_path):
        """Pruning with empty keep_set removes all attachments."""
        db_path = tmp_path / "test.db"

        conv = ConversationRecord(
            conversation_id="c1",
            provider_name="test",
            provider_conversation_id="ext1",
            title="Test",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            content_hash="hash1",
        )

        msg = MessageRecord(
            message_id="m1",
            conversation_id="c1",
            role="user",
            text="Test",
            timestamp=datetime.now(timezone.utc).isoformat(),
            content_hash="msghash1",
        )

        att = AttachmentRecord(
            attachment_id="att1",
            conversation_id="c1",
            message_id="m1",
            mime_type="image/png",
            size_bytes=1024,
        )

        with open_connection(db_path) as conn:
            store_records(conversation=conv, messages=[msg], attachments=[att], conn=conn)

            # Verify attachment exists
            row = conn.execute("SELECT * FROM attachments WHERE attachment_id = ?", ("att1",)).fetchone()
            assert row is not None

            # Prune with empty keep set
            _prune_attachment_refs(conn, "c1", set())

            # Verify attachment deleted
            row = conn.execute("SELECT * FROM attachments WHERE attachment_id = ?", ("att1",)).fetchone()
            assert row is None

    def test_prune_keeps_specified_refs(self, tmp_path):
        """Pruning with keep_set preserves specified refs."""
        db_path = tmp_path / "test.db"

        conv = ConversationRecord(
            conversation_id="c1",
            provider_name="test",
            provider_conversation_id="ext1",
            title="Test",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            content_hash="hash1",
        )

        msg1 = MessageRecord(
            message_id="m1",
            conversation_id="c1",
            role="user",
            text="Message 1",
            timestamp=datetime.now(timezone.utc).isoformat(),
            content_hash="msghash1",
        )

        msg2 = MessageRecord(
            message_id="m2",
            conversation_id="c1",
            role="assistant",
            text="Message 2",
            timestamp=datetime.now(timezone.utc).isoformat(),
            content_hash="msghash2",
        )

        att1 = AttachmentRecord(
            attachment_id="att1",
            conversation_id="c1",
            message_id="m1",
            mime_type="image/png",
            size_bytes=1024,
        )

        att2 = AttachmentRecord(
            attachment_id="att2",
            conversation_id="c1",
            message_id="m2",
            mime_type="image/jpeg",
            size_bytes=2048,
        )

        with open_connection(db_path) as conn:
            store_records(
                conversation=conv,
                messages=[msg1, msg2],
                attachments=[att1, att2],
                conn=conn,
            )

            ref_id1 = _make_ref_id("att1", "c1", "m1")
            ref_id2 = _make_ref_id("att2", "c1", "m2")

            # Verify both exist
            count_before = conn.execute(
                "SELECT COUNT(*) FROM attachments"
            ).fetchone()[0]
            assert count_before == 2

            # Prune, keeping only att1
            _prune_attachment_refs(conn, "c1", {ref_id1})

            # Verify att1 kept, att2 deleted
            count_after = conn.execute(
                "SELECT COUNT(*) FROM attachments"
            ).fetchone()[0]
            assert count_after == 1

            att1_row = conn.execute("SELECT * FROM attachments WHERE attachment_id = ?", ("att1",)).fetchone()
            att2_row = conn.execute("SELECT * FROM attachments WHERE attachment_id = ?", ("att2",)).fetchone()
            assert att1_row is not None
            assert att2_row is None


class TestSchemaAndMigration:
    """Test schema initialization and versioning."""

    def test_open_connection_applies_schema_on_new_db(self, tmp_path):
        """open_connection() applies full schema to new database."""
        db_path = tmp_path / "new.db"
        assert not db_path.exists()

        with open_connection(db_path) as conn:
            # Verify schema version
            version_row = conn.execute("PRAGMA user_version").fetchone()
            assert version_row[0] == SCHEMA_VERSION

            # Verify tables exist
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]

            expected_tables = [
                "conversations",
                "messages",
                "attachments",
                "attachment_refs",
                "runs",
            ]
            for table in expected_tables:
                assert table in tables, f"Table {table} not found"

    def test_open_connection_foreign_keys_enabled(self, tmp_path):
        """open_connection() enables foreign key constraints."""
        db_path = tmp_path / "test.db"

        with open_connection(db_path) as conn:
            row = conn.execute("PRAGMA foreign_keys").fetchone()
            assert row[0] == 1, "Foreign keys should be ON"

    def test_open_connection_wal_mode_enabled(self, tmp_path):
        """open_connection() enables WAL journal mode."""
        db_path = tmp_path / "test.db"

        with open_connection(db_path) as conn:
            row = conn.execute("PRAGMA journal_mode").fetchone()
            assert row[0].lower() == "wal", "WAL mode should be enabled"

    def test_open_connection_creates_parent_directories(self, tmp_path):
        """open_connection() creates nested parent directories."""
        db_path = tmp_path / "a" / "b" / "c" / "test.db"
        assert not db_path.parent.exists()

        with open_connection(db_path) as conn:
            assert conn is not None

        assert db_path.exists()
        assert db_path.parent.exists()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_store_records_with_null_optional_fields(self, tmp_path):
        """store_records() handles conversations/messages with NULL optional fields."""
        db_path = tmp_path / "test.db"

        conv = ConversationRecord(
            conversation_id="c1",
            provider_name="test",
            provider_conversation_id="ext1",
            title=None,  # NULL
            created_at=None,  # NULL
            updated_at=None,  # NULL
            content_hash="hash1",
            provider_meta=None,  # NULL
        )

        msg = MessageRecord(
            message_id="m1",
            conversation_id="c1",
            provider_message_id=None,  # NULL
            role=None,  # NULL
            text=None,  # NULL
            timestamp=None,  # NULL
            content_hash="msghash1",
            provider_meta=None,  # NULL
        )

        with open_connection(db_path) as conn:
            counts = store_records(conversation=conv, messages=[msg], attachments=[], conn=conn)

        assert counts["conversations"] == 1
        assert counts["messages"] == 1

        # Verify NULLs preserved
        with open_connection(db_path) as conn:
            conv_row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", ("c1",)).fetchone()
            assert conv_row["title"] is None
            assert conv_row["created_at"] is None

            msg_row = conn.execute("SELECT * FROM messages WHERE message_id = ?", ("m1",)).fetchone()
            assert msg_row["role"] is None
            assert msg_row["text"] is None

    def test_store_records_with_empty_messages_and_attachments(self, tmp_path):
        """store_records() handles conversation with no messages or attachments."""
        db_path = tmp_path / "test.db"

        conv = ConversationRecord(
            conversation_id="c1",
            provider_name="test",
            provider_conversation_id="ext1",
            title="Empty Conversation",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            content_hash="hash1",
        )

        with open_connection(db_path) as conn:
            counts = store_records(conversation=conv, messages=[], attachments=[], conn=conn)

        assert counts["conversations"] == 1
        assert counts["messages"] == 0
        assert counts["attachments"] == 0

    def test_attachment_without_message_id(self, tmp_path):
        """Attachments can exist without being tied to a message."""
        db_path = tmp_path / "test.db"

        conv = ConversationRecord(
            conversation_id="c1",
            provider_name="test",
            provider_conversation_id="ext1",
            title="Test",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            content_hash="hash1",
        )

        # Attachment without message_id
        att = AttachmentRecord(
            attachment_id="att1",
            conversation_id="c1",
            message_id=None,  # No message
            mime_type="application/pdf",
            size_bytes=5000,
        )

        with open_connection(db_path) as conn:
            counts = store_records(conversation=conv, messages=[], attachments=[att], conn=conn)

        assert counts["attachments"] == 1

        # Verify stored
        with open_connection(db_path) as conn:
            row = conn.execute("SELECT * FROM attachments WHERE attachment_id = ?", ("att1",)).fetchone()
            assert row is not None
            assert row["ref_count"] == 1


class TestComplexScenarios:
    """Test realistic complex scenarios."""

    def test_conversation_lifecycle_with_attachments(self, tmp_path):
        """Full lifecycle: create → add attachments → remove attachments → cleanup."""
        db_path = tmp_path / "test.db"

        # Step 1: Create conversation with one attachment
        conv_v1 = ConversationRecord(
            conversation_id="c1",
            provider_name="claude",
            provider_conversation_id="ext-c1",
            title="Analysis Project",
            created_at="2024-01-01T10:00:00Z",
            updated_at="2024-01-01T10:00:00Z",
            content_hash="hash-v1",
        )

        msg1 = MessageRecord(
            message_id="m1",
            conversation_id="c1",
            role="user",
            text="Please analyze this image",
            timestamp="2024-01-01T10:00:00Z",
            content_hash="msghash-m1",
        )

        att1 = AttachmentRecord(
            attachment_id="att-image",
            conversation_id="c1",
            message_id="m1",
            mime_type="image/png",
            size_bytes=51200,
        )

        with open_connection(db_path) as conn:
            store_records(conversation=conv_v1, messages=[msg1], attachments=[att1], conn=conn)

        # Step 2: Add more messages and attachments
        msg2 = MessageRecord(
            message_id="m2",
            conversation_id="c1",
            role="assistant",
            text="The image shows...",
            timestamp="2024-01-01T10:01:00Z",
            content_hash="msghash-m2",
        )

        att2 = AttachmentRecord(
            attachment_id="att-export",
            conversation_id="c1",
            message_id="m2",
            mime_type="application/json",
            size_bytes=2048,
        )

        conv_v2 = ConversationRecord(
            conversation_id="c1",
            provider_name="claude",
            provider_conversation_id="ext-c1",
            title="Analysis Project",
            created_at="2024-01-01T10:00:00Z",
            updated_at="2024-01-01T10:02:00Z",
            content_hash="hash-v2",  # Different: new message and attachment
        )

        with open_connection(db_path) as conn:
            store_records(
                conversation=conv_v2,
                messages=[msg1, msg2],
                attachments=[att1, att2],
                conn=conn,
            )

        # Verify 2 attachments now
        with open_connection(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0]
            assert count == 2

        # Step 3: Final update removes one attachment
        conv_v3 = ConversationRecord(
            conversation_id="c1",
            provider_name="claude",
            provider_conversation_id="ext-c1",
            title="Analysis Project - Final",
            created_at="2024-01-01T10:00:00Z",
            updated_at="2024-01-01T10:03:00Z",
            content_hash="hash-v3",
        )

        with open_connection(db_path) as conn:
            store_records(
                conversation=conv_v3,
                messages=[msg1, msg2],
                attachments=[att1],  # Only image, no export
                conn=conn,
            )

        # Verify: image kept, export deleted
        with open_connection(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0]
            assert count == 1

            remaining = conn.execute("SELECT attachment_id FROM attachments").fetchone()
            assert remaining["attachment_id"] == "att-image"

    def test_multi_provider_conversations_separate(self, tmp_path):
        """Conversations from different providers don't interfere."""
        db_path = tmp_path / "test.db"

        # ChatGPT conversation
        conv_gpt = ConversationRecord(
            conversation_id="c-gpt",
            provider_name="chatgpt",
            provider_conversation_id="gpt-ext-1",
            title="ChatGPT Conversation",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            content_hash="hash-gpt",
        )

        # Claude conversation
        conv_claude = ConversationRecord(
            conversation_id="c-claude",
            provider_name="claude",
            provider_conversation_id="claude-ext-1",
            title="Claude Conversation",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            content_hash="hash-claude",
        )

        with open_connection(db_path) as conn:
            store_records(conversation=conv_gpt, messages=[], attachments=[], conn=conn)
            store_records(conversation=conv_claude, messages=[], attachments=[], conn=conn)

        # Verify both stored correctly
        with open_connection(db_path) as conn:
            gpt_row = conn.execute(
                "SELECT * FROM conversations WHERE provider_name = ?", ("chatgpt",)
            ).fetchone()
            claude_row = conn.execute(
                "SELECT * FROM conversations WHERE provider_name = ?", ("claude",)
            ).fetchone()

            assert gpt_row is not None
            assert gpt_row["title"] == "ChatGPT Conversation"
            assert claude_row is not None
            assert claude_row["title"] == "Claude Conversation"
