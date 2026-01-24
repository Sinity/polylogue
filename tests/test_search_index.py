"""Tests for search.py and index.py modules.

Tests cover FTS5 index creation, incremental updates, search functionality,
ranking, special characters, and edge cases.
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from polylogue.storage.db import open_connection
from polylogue.storage.index import ensure_index, rebuild_index, update_index_for_conversations
from polylogue.storage.search import search_messages
from polylogue.storage.store import ConversationRecord, MessageRecord, store_records


@pytest.fixture
def test_db_with_schema(tmp_path):
    """Create a test database with schema applied."""
    db_path = tmp_path / "test.db"
    with open_connection(db_path) as conn:
        pass  # Schema applied automatically
    return db_path


@pytest.fixture
def test_conn(test_db_with_schema):
    """Provide a connection to test database."""
    with open_connection(test_db_with_schema) as conn:
        yield conn


@pytest.fixture
def archive_root(tmp_path):
    """Create an archive root directory with render subdirectory."""
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "render").mkdir()
    return archive


# ============================================================================
# Tests for rebuild_index()
# ============================================================================


def test_rebuild_index_creates_fts_table(test_conn):
    """rebuild_index() creates messages_fts table if it doesn't exist."""
    # Verify table doesn't exist yet
    result = test_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
    ).fetchone()
    assert result is None or result[0] != "messages_fts"

    # Call rebuild_index
    rebuild_index(test_conn)

    # Verify table exists
    result = test_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
    ).fetchone()
    assert result is not None
    assert result[0] == "messages_fts"


def test_rebuild_index_populates_fts_from_messages(test_conn):
    """rebuild_index() populates FTS table from existing messages."""
    # Insert conversation and messages
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Test Conversation",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        content_hash="hash1",
        version=1,
    )
    messages = [
        MessageRecord(
            message_id="msg1",
            conversation_id="conv1",
            role="user",
            text="hello world",
            timestamp="2024-01-01T00:00:00Z",
            content_hash="hash-msg1",
            version=1,
        ),
        MessageRecord(
            message_id="msg2",
            conversation_id="conv1",
            role="assistant",
            text="goodbye world",
            timestamp="2024-01-01T00:01:00Z",
            content_hash="hash-msg2",
            version=1,
        ),
    ]

    store_records(conversation=conv, messages=messages, attachments=[], conn=test_conn)

    # Rebuild index
    rebuild_index(test_conn)

    # Verify FTS table has entries
    count = test_conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
    assert count == 2

    # Verify content is indexed
    hello_hits = test_conn.execute(
        'SELECT message_id FROM messages_fts WHERE messages_fts MATCH ?', ("hello",)
    ).fetchall()
    assert len(hello_hits) == 1
    assert hello_hits[0]["message_id"] == "msg1"

    goodbye_hits = test_conn.execute(
        'SELECT message_id FROM messages_fts WHERE messages_fts MATCH ?', ("goodbye",)
    ).fetchall()
    assert len(goodbye_hits) == 1
    assert goodbye_hits[0]["message_id"] == "msg2"


def test_rebuild_index_clears_previous_index(test_conn):
    """rebuild_index() clears old FTS entries when rebuilding."""
    # First insert and index
    conv1 = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="First",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        content_hash="hash1",
        version=1,
    )
    msg1 = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        role="user",
        text="first message",
        timestamp="2024-01-01T00:00:00Z",
        content_hash="hash-msg1",
        version=1,
    )
    store_records(conversation=conv1, messages=[msg1], attachments=[], conn=test_conn)
    rebuild_index(test_conn)

    # Verify first message is indexed
    count_before = test_conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
    assert count_before == 1

    # Delete first conversation and add a new one
    test_conn.execute("DELETE FROM conversations WHERE conversation_id = ?", ("conv1",))

    conv2 = ConversationRecord(
        conversation_id="conv2",
        provider_name="test",
        provider_conversation_id="ext-conv2",
        title="Second",
        created_at="2024-01-02T00:00:00Z",
        updated_at="2024-01-02T00:00:00Z",
        content_hash="hash2",
        version=1,
    )
    msg2 = MessageRecord(
        message_id="msg2",
        conversation_id="conv2",
        role="user",
        text="second message",
        timestamp="2024-01-02T00:00:00Z",
        content_hash="hash-msg2",
        version=1,
    )
    store_records(conversation=conv2, messages=[msg2], attachments=[], conn=test_conn)

    # Rebuild index
    rebuild_index(test_conn)

    # Old message should not be searchable
    old_hits = test_conn.execute(
        'SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?', ("first",)
    ).fetchone()[0]
    assert old_hits == 0

    # New message should be searchable
    new_hits = test_conn.execute(
        'SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?', ("second",)
    ).fetchone()[0]
    assert new_hits == 1


def test_rebuild_index_skips_null_text(test_conn):
    """rebuild_index() skips messages with NULL text."""
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        content_hash="hash1",
        version=1,
    )
    messages = [
        MessageRecord(
            message_id="msg1",
            conversation_id="conv1",
            role="user",
            text="hello",
            timestamp="2024-01-01T00:00:00Z",
            content_hash="hash-msg1",
            version=1,
        ),
        MessageRecord(
            message_id="msg2",
            conversation_id="conv1",
            role="assistant",
            text=None,  # NULL text
            timestamp="2024-01-01T00:01:00Z",
            content_hash="hash-msg2",
            version=1,
        ),
    ]

    store_records(conversation=conv, messages=messages, attachments=[], conn=test_conn)
    rebuild_index(test_conn)

    # Only one message should be indexed
    count = test_conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
    assert count == 1


def test_rebuild_index_preserves_conversation_metadata(test_conn):
    """rebuild_index() includes provider_name and conversation_id in FTS."""
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="claude",
        provider_conversation_id="ext-conv1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        content_hash="hash1",
        version=1,
    )
    msg = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        role="user",
        text="hello",
        timestamp="2024-01-01T00:00:00Z",
        content_hash="hash-msg1",
        version=1,
    )

    store_records(conversation=conv, messages=[msg], attachments=[], conn=test_conn)
    rebuild_index(test_conn)

    # Check FTS columns
    row = test_conn.execute(
        "SELECT message_id, conversation_id, provider_name FROM messages_fts LIMIT 1"
    ).fetchone()
    assert row["message_id"] == "msg1"
    assert row["conversation_id"] == "conv1"
    assert row["provider_name"] == "claude"


# ============================================================================
# Tests for update_index_for_conversations()
# ============================================================================


def test_update_index_incremental_single_conversation(test_conn):
    """update_index_for_conversations() updates only specified conversations."""
    # Create and index two conversations
    conv1 = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Conv 1",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        content_hash="hash1",
        version=1,
    )
    conv2 = ConversationRecord(
        conversation_id="conv2",
        provider_name="test",
        provider_conversation_id="ext-conv2",
        title="Conv 2",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        content_hash="hash2",
        version=1,
    )
    msg1 = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        role="user",
        text="first",
        timestamp="2024-01-01T00:00:00Z",
        content_hash="hash-msg1",
        version=1,
    )
    msg2 = MessageRecord(
        message_id="msg2",
        conversation_id="conv2",
        role="user",
        text="second",
        timestamp="2024-01-01T00:00:00Z",
        content_hash="hash-msg2",
        version=1,
    )

    store_records(conversation=conv1, messages=[msg1], attachments=[], conn=test_conn)
    store_records(conversation=conv2, messages=[msg2], attachments=[], conn=test_conn)

    # Initial rebuild
    rebuild_index(test_conn)

    # Add new message to conv1
    msg3 = MessageRecord(
        message_id="msg3",
        conversation_id="conv1",
        role="assistant",
        text="updated",
        timestamp="2024-01-01T00:01:00Z",
        content_hash="hash-msg3",
        version=1,
    )
    test_conn.execute(
        """
        INSERT INTO messages
        (message_id, conversation_id, role, text, timestamp, content_hash, version)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("msg3", "conv1", "assistant", "updated", "2024-01-01T00:01:00Z", "hash-msg3", 1),
    )

    # Update only conv1
    update_index_for_conversations(["conv1"], test_conn)

    # Conv1 should have 2 messages indexed now
    conv1_count = test_conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE conversation_id = ?", ("conv1",)
    ).fetchone()[0]
    assert conv1_count == 2

    # Conv2 should still have only 1 message (unchanged)
    conv2_count = test_conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE conversation_id = ?", ("conv2",)
    ).fetchone()[0]
    assert conv2_count == 1


def test_update_index_incremental_multiple_conversations(test_conn):
    """update_index_for_conversations() can update multiple conversations at once."""
    # Create three conversations
    conversations = []
    messages = []
    for i in range(3):
        conv = ConversationRecord(
            conversation_id=f"conv{i}",
            provider_name="test",
            provider_conversation_id=f"ext-conv{i}",
            title=f"Conv {i}",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            content_hash=f"hash{i}",
            version=1,
        )
        msg = MessageRecord(
            message_id=f"msg{i}",
            conversation_id=f"conv{i}",
            role="user",
            text=f"text {i}",
            timestamp="2024-01-01T00:00:00Z",
            content_hash=f"hash-msg{i}",
            version=1,
        )
        store_records(conversation=conv, messages=[msg], attachments=[], conn=test_conn)

    rebuild_index(test_conn)

    # Add new messages to conv0 and conv2
    for cid in [0, 2]:
        test_conn.execute(
            """
            INSERT INTO messages
            (message_id, conversation_id, role, text, timestamp, content_hash, version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"msg{cid}-new",
                f"conv{cid}",
                "assistant",
                f"new text {cid}",
                "2024-01-01T00:01:00Z",
                f"hash-msg{cid}-new",
                1,
            ),
        )

    # Update conv0 and conv2
    update_index_for_conversations(["conv0", "conv2"], test_conn)

    # Check counts
    for cid in range(3):
        count = test_conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE conversation_id = ?", (f"conv{cid}",)
        ).fetchone()[0]
        expected = 2 if cid in [0, 2] else 1
        assert count == expected, f"Conv {cid} should have {expected} messages"


def test_update_index_empty_list_does_nothing(test_conn):
    """update_index_for_conversations([]) does nothing."""
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        content_hash="hash1",
        version=1,
    )
    msg = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        role="user",
        text="hello",
        timestamp="2024-01-01T00:00:00Z",
        content_hash="hash-msg1",
        version=1,
    )

    store_records(conversation=conv, messages=[msg], attachments=[], conn=test_conn)
    rebuild_index(test_conn)

    count_before = test_conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]

    # Call with empty list
    update_index_for_conversations([], test_conn)

    count_after = test_conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
    assert count_before == count_after


def test_update_index_handles_large_batch(test_conn):
    """update_index_for_conversations() handles large batches (>200 conversations)."""
    # Create 250 conversations with messages
    num_convs = 250
    for i in range(num_convs):
        conv = ConversationRecord(
            conversation_id=f"conv{i}",
            provider_name="test",
            provider_conversation_id=f"ext-conv{i}",
            title=f"Conv {i}",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            content_hash=f"hash{i}",
            version=1,
        )
        msg = MessageRecord(
            message_id=f"msg{i}",
            conversation_id=f"conv{i}",
            role="user",
            text=f"text {i}",
            timestamp="2024-01-01T00:00:00Z",
            content_hash=f"hash-msg{i}",
            version=1,
        )
        store_records(conversation=conv, messages=[msg], attachments=[], conn=test_conn)

    # Update all at once (tests chunking with size=200)
    conv_ids = [f"conv{i}" for i in range(num_convs)]
    update_index_for_conversations(conv_ids, test_conn)

    # Verify all indexed
    count = test_conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
    assert count == num_convs


# ============================================================================
# Tests for search_messages()
# ============================================================================


def test_search_finds_matching_text(workspace_env, storage_repository):
    """search_messages() finds conversations with matching text."""
    from polylogue.ingestion import IngestBundle, ingest_bundle

    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Test Conv",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        content_hash="hash1",
        version=1,
    )
    msg = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        role="user",
        text="python programming language",
        timestamp="2024-01-01T00:00:00Z",
        content_hash="hash-msg1",
        version=1,
    )

    ingest_bundle(IngestBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)
    rebuild_index()

    results = search_messages("python", archive_root=workspace_env["archive_root"], limit=10)

    assert len(results.hits) == 1
    assert results.hits[0].conversation_id == "conv1"
    assert results.hits[0].message_id == "msg1"


def test_search_returns_multiple_matches(workspace_env, storage_repository):
    """search_messages() returns multiple matching conversations."""
    from polylogue.ingestion import IngestBundle, ingest_bundle

    for i in range(3):
        conv = ConversationRecord(
            conversation_id=f"conv{i}",
            provider_name="test",
            provider_conversation_id=f"ext-conv{i}",
            title=f"Conv {i}",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            content_hash=f"hash{i}",
            version=1,
        )
        msg = MessageRecord(
            message_id=f"msg{i}",
            conversation_id=f"conv{i}",
            role="user",
            text="testing framework",
            timestamp="2024-01-01T00:00:00Z",
            content_hash=f"hash-msg{i}",
            version=1,
        )
        ingest_bundle(IngestBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)

    rebuild_index()

    results = search_messages("testing", archive_root=workspace_env["archive_root"], limit=10)

    assert len(results.hits) == 3
    hit_conv_ids = {hit.conversation_id for hit in results.hits}
    assert hit_conv_ids == {"conv0", "conv1", "conv2"}


def test_search_no_results_returns_empty(workspace_env, storage_repository):
    """search_messages() returns empty list when no matches found."""
    from polylogue.ingestion import IngestBundle, ingest_bundle

    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        content_hash="hash1",
        version=1,
    )
    msg = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        role="user",
        text="hello world",
        timestamp="2024-01-01T00:00:00Z",
        content_hash="hash-msg1",
        version=1,
    )

    ingest_bundle(IngestBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)
    rebuild_index()

    results = search_messages("nonexistent", archive_root=workspace_env["archive_root"], limit=10)

    assert len(results.hits) == 0


def test_search_respects_limit(workspace_env, storage_repository):
    """search_messages() respects limit parameter."""
    from polylogue.ingestion import IngestBundle, ingest_bundle

    for i in range(10):
        conv = ConversationRecord(
            conversation_id=f"conv{i}",
            provider_name="test",
            provider_conversation_id=f"ext-conv{i}",
            title=f"Conv {i}",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            content_hash=f"hash{i}",
            version=1,
        )
        msg = MessageRecord(
            message_id=f"msg{i}",
            conversation_id=f"conv{i}",
            role="user",
            text="search limit",
            timestamp="2024-01-01T00:00:00Z",
            content_hash=f"hash-msg{i}",
            version=1,
        )
        ingest_bundle(IngestBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)

    rebuild_index()

    results = search_messages("search", archive_root=workspace_env["archive_root"], limit=3)
    assert len(results.hits) == 3


def test_search_includes_snippet(workspace_env, storage_repository):
    """search_messages() includes text snippet in results."""
    from polylogue.ingestion import IngestBundle, ingest_bundle

    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        content_hash="hash1",
        version=1,
    )
    msg = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        role="user",
        text="The quick brown fox jumps over the lazy dog",
        timestamp="2024-01-01T00:00:00Z",
        content_hash="hash-msg1",
        version=1,
    )

    ingest_bundle(IngestBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)
    rebuild_index()

    results = search_messages("quick", archive_root=workspace_env["archive_root"], limit=10)

    assert len(results.hits) == 1
    # Snippet should contain the query term or nearby context
    assert results.hits[0].snippet is not None
    assert isinstance(results.hits[0].snippet, str)


def test_search_includes_conversation_metadata(workspace_env, storage_repository):
    """search_messages() includes conversation metadata in results."""
    from polylogue.ingestion import IngestBundle, ingest_bundle

    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="claude",
        provider_conversation_id="ext-conv1",
        title="My Conversation",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        content_hash="hash1",
        version=1,
        provider_meta={"source": "my-source"},
    )
    msg = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        role="user",
        text="search query",
        timestamp="2024-01-01T10:30:00Z",
        content_hash="hash-msg1",
        version=1,
    )

    ingest_bundle(IngestBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)
    rebuild_index()

    results = search_messages("search", archive_root=workspace_env["archive_root"], limit=10)

    assert len(results.hits) == 1
    hit = results.hits[0]
    assert hit.conversation_id == "conv1"
    assert hit.provider_name == "claude"
    assert hit.title == "My Conversation"
    assert hit.message_id == "msg1"
    assert hit.timestamp == "2024-01-01T10:30:00Z"
    assert hit.source_name == "my-source"


# ============================================================================
# Tests for search ranking and special characters
# ============================================================================


def test_search_with_special_characters(workspace_env, storage_repository):
    """search_messages() handles special characters in text."""
    from polylogue.ingestion import IngestBundle, ingest_bundle

    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        content_hash="hash1",
        version=1,
    )
    msg = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        role="user",
        text="C++ programming with @mentions and #hashtags",
        timestamp="2024-01-01T00:00:00Z",
        content_hash="hash-msg1",
        version=1,
    )

    ingest_bundle(IngestBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)
    rebuild_index()

    # Search for word containing special chars
    results = search_messages("programming", archive_root=workspace_env["archive_root"], limit=10)
    assert len(results.hits) == 1


def test_search_with_quotes_in_text(workspace_env, storage_repository):
    """search_messages() handles quoted text correctly."""
    from polylogue.ingestion import IngestBundle, ingest_bundle

    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        content_hash="hash1",
        version=1,
    )
    msg = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        role="user",
        text='She said "hello world" to me',
        timestamp="2024-01-01T00:00:00Z",
        content_hash="hash-msg1",
        version=1,
    )

    ingest_bundle(IngestBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)
    rebuild_index()

    results = search_messages("hello", archive_root=workspace_env["archive_root"], limit=10)
    assert len(results.hits) == 1


def test_search_with_unicode_text(workspace_env, storage_repository):
    """search_messages() handles unicode characters."""
    from polylogue.ingestion import IngestBundle, ingest_bundle

    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Unicode Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        content_hash="hash1",
        version=1,
    )
    msg = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        role="user",
        text="Hello 世界 مرحبا мир café",
        timestamp="2024-01-01T00:00:00Z",
        content_hash="hash-msg1",
        version=1,
    )

    ingest_bundle(IngestBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)
    rebuild_index()

    results = search_messages("café", archive_root=workspace_env["archive_root"], limit=10)
    assert len(results.hits) == 1


def test_search_with_hyphenated_words(workspace_env, storage_repository):
    """search_messages() handles words in hyphenated phrases."""
    from polylogue.ingestion import IngestBundle, ingest_bundle

    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        content_hash="hash1",
        version=1,
    )
    msg = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        role="user",
        text="The state-of-the-art algorithm",
        timestamp="2024-01-01T00:00:00Z",
        content_hash="hash-msg1",
        version=1,
    )

    ingest_bundle(IngestBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)
    rebuild_index()

    # Search for individual words within hyphenated phrase (FTS5 tokenizes on hyphens)
    results = search_messages("state", archive_root=workspace_env["archive_root"], limit=10)
    # Should find the message containing "state" (from state-of-the-art)
    assert len(results.hits) == 1


# ============================================================================
# Tests for edge cases and error handling
# ============================================================================


def test_ensure_index_idempotent(test_conn):
    """ensure_index() can be called multiple times safely."""
    ensure_index(test_conn)
    ensure_index(test_conn)
    ensure_index(test_conn)

    # Verify table exists
    result = test_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
    ).fetchone()
    assert result is not None


def test_rebuild_index_with_empty_database(test_conn):
    """rebuild_index() handles empty database gracefully."""
    rebuild_index(test_conn)

    count = test_conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
    assert count == 0


def test_search_returns_searchresult_object(workspace_env, storage_repository):
    """search_messages() returns SearchResult with hits list."""
    from polylogue.ingestion import IngestBundle, ingest_bundle

    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        content_hash="hash1",
        version=1,
    )
    msg = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        role="user",
        text="search result",
        timestamp="2024-01-01T00:00:00Z",
        content_hash="hash-msg1",
        version=1,
    )

    ingest_bundle(IngestBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)
    rebuild_index()

    results = search_messages("search", archive_root=workspace_env["archive_root"], limit=10)

    # Verify SearchResult structure
    assert hasattr(results, "hits")
    assert isinstance(results.hits, list)
    if results.hits:
        hit = results.hits[0]
        assert hasattr(hit, "conversation_id")
        assert hasattr(hit, "message_id")
        assert hasattr(hit, "provider_name")
        assert hasattr(hit, "snippet")
        assert hasattr(hit, "title")
        assert hasattr(hit, "timestamp")
        assert hasattr(hit, "conversation_path")


def test_rebuild_index_with_multiple_messages_per_conversation(test_conn):
    """rebuild_index() correctly indexes all messages in a conversation."""
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Multi-message Conv",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        content_hash="hash1",
        version=1,
    )
    messages = [
        MessageRecord(
            message_id=f"msg{i}",
            conversation_id="conv1",
            role="user" if i % 2 == 0 else "assistant",
            text=f"message number {i}",
            timestamp=f"2024-01-01T00:{i:02d}:00Z",
            content_hash=f"hash-msg{i}",
            version=1,
        )
        for i in range(10)
    ]

    store_records(conversation=conv, messages=messages, attachments=[], conn=test_conn)
    rebuild_index(test_conn)

    count = test_conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE conversation_id = ?", ("conv1",)
    ).fetchone()[0]
    assert count == 10


def test_update_index_deletes_old_entries_from_conversation(test_conn):
    """update_index_for_conversations() removes old index entries for updated conversations."""
    conv = ConversationRecord(
        conversation_id="conv1",
        provider_name="test",
        provider_conversation_id="ext-conv1",
        title="Test",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        content_hash="hash1",
        version=1,
    )
    msg = MessageRecord(
        message_id="msg1",
        conversation_id="conv1",
        role="user",
        text="original message",
        timestamp="2024-01-01T00:00:00Z",
        content_hash="hash-msg1",
        version=1,
    )

    store_records(conversation=conv, messages=[msg], attachments=[], conn=test_conn)
    rebuild_index(test_conn)

    # Delete original message
    test_conn.execute("DELETE FROM messages WHERE message_id = ?", ("msg1",))

    # Add new message
    test_conn.execute(
        """
        INSERT INTO messages
        (message_id, conversation_id, role, text, timestamp, content_hash, version)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("msg2", "conv1", "user", "new message", "2024-01-01T00:01:00Z", "hash-msg2", 1),
    )

    # Update index
    update_index_for_conversations(["conv1"], test_conn)

    # Old message should not be in index
    old_hits = test_conn.execute(
        'SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?', ("original",)
    ).fetchone()[0]
    assert old_hits == 0

    # New message should be indexed
    new_hits = test_conn.execute(
        'SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?', ("new",)
    ).fetchone()[0]
    assert new_hits == 1


__all__ = [
    "test_rebuild_index_creates_fts_table",
    "test_rebuild_index_populates_fts_from_messages",
    "test_rebuild_index_clears_previous_index",
    "test_rebuild_index_skips_null_text",
    "test_update_index_incremental_single_conversation",
    "test_update_index_incremental_multiple_conversations",
    "test_search_finds_matching_text",
    "test_search_returns_multiple_matches",
    "test_search_no_results_returns_empty",
    "test_search_respects_limit",
    "test_search_with_special_characters",
    "test_search_with_quotes_in_text",
    "test_search_with_unicode_text",
]
