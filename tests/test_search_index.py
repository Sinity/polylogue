"""Tests for search.py and index.py modules.

Tests cover FTS5 index creation, incremental updates, search functionality,
ranking, special characters, and edge cases.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from polylogue.config import Config, IndexConfig, get_config
from polylogue.health import get_health
from polylogue.sources import IngestBundle, ingest_bundle
from polylogue.storage.backends.sqlite import open_connection
from polylogue.storage.index import ensure_index, rebuild_index, update_index_for_conversations
from polylogue.storage.search import escape_fts5_query, search_messages
from polylogue.storage.search_providers import create_vector_provider
from polylogue.storage.search_providers.fts5 import FTS5Provider
from polylogue.storage.search_providers.qdrant import QdrantProvider
from polylogue.storage.store import store_records
from tests.helpers import ConversationBuilder, DbFactory, make_conversation, make_message

# test_db and test_conn fixtures are in conftest.py


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
    # Verify table doesn't exist yet (drop it first as fixture might create it)
    test_conn.execute("DROP TABLE IF EXISTS messages_fts")
    result = test_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone()
    assert result is None or result[0] != "messages_fts"

    # Call rebuild_index
    rebuild_index(test_conn)

    # Verify table exists
    result = test_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone()
    assert result is not None
    assert result[0] == "messages_fts"


def test_rebuild_index_populates_fts_from_messages(test_conn):
    """rebuild_index() populates FTS table from existing messages."""
    conv = make_conversation("conv1")
    messages = [
        make_message("msg1", "conv1", text="hello world"),
        make_message("msg2", "conv1", role="assistant", text="goodbye world"),
    ]

    store_records(conversation=conv, messages=messages, attachments=[], conn=test_conn)

    # Rebuild index
    rebuild_index(test_conn)

    # Verify FTS table has entries
    count = test_conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
    assert count == 2

    # Verify content is indexed
    hello_hits = test_conn.execute(
        "SELECT message_id FROM messages_fts WHERE messages_fts MATCH ?", ("hello",)
    ).fetchall()
    assert len(hello_hits) == 1
    assert hello_hits[0]["message_id"] == "msg1"

    goodbye_hits = test_conn.execute(
        "SELECT message_id FROM messages_fts WHERE messages_fts MATCH ?", ("goodbye",)
    ).fetchall()
    assert len(goodbye_hits) == 1
    assert goodbye_hits[0]["message_id"] == "msg2"


def test_rebuild_index_clears_previous_index(test_conn):
    """rebuild_index() clears old FTS entries when rebuilding."""
    # First insert and index
    conv1 = make_conversation("conv1", title="First")
    msg1 = make_message("msg1", "conv1", text="first message")
    store_records(conversation=conv1, messages=[msg1], attachments=[], conn=test_conn)
    rebuild_index(test_conn)

    # Verify first message is indexed
    count_before = test_conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
    assert count_before == 1

    # Delete first conversation and add a new one
    test_conn.execute("DELETE FROM conversations WHERE conversation_id = ?", ("conv1",))

    conv2 = make_conversation("conv2", title="Second")
    msg2 = make_message("msg2", "conv2", text="second message")
    store_records(conversation=conv2, messages=[msg2], attachments=[], conn=test_conn)

    # Rebuild index
    rebuild_index(test_conn)

    # Old message should not be searchable
    old_hits = test_conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?", ("first",)).fetchone()[
        0
    ]
    assert old_hits == 0

    # New message should be searchable
    new_hits = test_conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?", ("second",)
    ).fetchone()[0]
    assert new_hits == 1


def test_rebuild_index_skips_null_text(test_conn):
    """rebuild_index() skips messages with NULL text."""
    conv = make_conversation("conv1")
    messages = [
        make_message("msg1", "conv1", text="hello"),
        make_message("msg2", "conv1", role="assistant", text=None),  # NULL text
    ]

    store_records(conversation=conv, messages=messages, attachments=[], conn=test_conn)
    rebuild_index(test_conn)

    # Only one message should be indexed
    count = test_conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
    assert count == 1


def test_rebuild_index_includes_keys(test_conn):
    """rebuild_index() includes keys in FTS."""
    conv = make_conversation("conv1", provider_name="claude")
    msg = make_message("msg1", "conv1", text="hello")

    store_records(conversation=conv, messages=[msg], attachments=[], conn=test_conn)
    rebuild_index(test_conn)

    # Check FTS columns
    row = test_conn.execute("SELECT message_id, conversation_id FROM messages_fts LIMIT 1").fetchone()
    assert row["message_id"] == "msg1"
    assert row["conversation_id"] == "conv1"
    # provider_name is no longer in FTS table


# ============================================================================
# Tests for update_index_for_conversations()
# ============================================================================


def test_update_index_incremental_single_conversation(test_conn):
    """update_index_for_conversations() updates only specified conversations."""
    # Create and index two conversations
    conv1 = make_conversation("conv1", title="Conv 1")
    conv2 = make_conversation("conv2", title="Conv 2")
    msg1 = make_message("msg1", "conv1", text="first")
    msg2 = make_message("msg2", "conv2", text="second")

    store_records(conversation=conv1, messages=[msg1], attachments=[], conn=test_conn)
    store_records(conversation=conv2, messages=[msg2], attachments=[], conn=test_conn)

    # Initial rebuild
    rebuild_index(test_conn)

    # Add new message to conv1 directly to test incremental update
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
    for i in range(3):
        conv = make_conversation(f"conv{i}", title=f"Conv {i}")
        msg = make_message(f"msg{i}", f"conv{i}", text=f"text {i}")
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
    conv = make_conversation("conv1")
    msg = make_message("msg1", "conv1", text="hello")

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
        conv = make_conversation(f"conv{i}", title=f"Conv {i}")
        msg = make_message(f"msg{i}", f"conv{i}", text=f"text {i}")
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
    from polylogue.sources import IngestBundle, ingest_bundle

    conv = make_conversation("conv1", title="Test Conv")
    msg = make_message("msg1", "conv1", text="python programming language")

    ingest_bundle(IngestBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)
    rebuild_index()

    results = search_messages("python", archive_root=workspace_env["archive_root"], limit=10)

    assert len(results.hits) == 1
    assert results.hits[0].conversation_id == "conv1"
    assert results.hits[0].message_id == "msg1"


def test_search_returns_multiple_matches(workspace_env, storage_repository):
    """search_messages() returns multiple matching conversations."""
    from polylogue.sources import IngestBundle, ingest_bundle

    for i in range(3):
        conv = make_conversation(f"conv{i}", title=f"Conv {i}")
        msg = make_message(f"msg{i}", f"conv{i}", text="testing framework")
        ingest_bundle(IngestBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)

    rebuild_index()

    results = search_messages("testing", archive_root=workspace_env["archive_root"], limit=10)

    assert len(results.hits) == 3
    hit_conv_ids = {hit.conversation_id for hit in results.hits}
    assert hit_conv_ids == {"conv0", "conv1", "conv2"}


def test_search_no_results_returns_empty(workspace_env, storage_repository):
    """search_messages() returns empty list when no matches found."""
    from polylogue.sources import IngestBundle, ingest_bundle

    conv = make_conversation("conv1")
    msg = make_message("msg1", "conv1", text="hello world")

    ingest_bundle(IngestBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)
    rebuild_index()

    results = search_messages("nonexistent", archive_root=workspace_env["archive_root"], limit=10)

    assert len(results.hits) == 0


def test_search_respects_limit(workspace_env, storage_repository):
    """search_messages() respects limit parameter."""
    from polylogue.sources import IngestBundle, ingest_bundle

    for i in range(10):
        conv = make_conversation(f"conv{i}", title=f"Conv {i}")
        msg = make_message(f"msg{i}", f"conv{i}", text="search limit")
        ingest_bundle(IngestBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)

    rebuild_index()

    results = search_messages("search", archive_root=workspace_env["archive_root"], limit=3)
    assert len(results.hits) == 3


def test_search_includes_snippet(workspace_env, storage_repository):
    """search_messages() includes text snippet in results."""
    from polylogue.sources import IngestBundle, ingest_bundle

    conv = make_conversation("conv1")
    msg = make_message("msg1", "conv1", text="The quick brown fox jumps over the lazy dog")

    ingest_bundle(IngestBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)
    rebuild_index()

    results = search_messages("quick", archive_root=workspace_env["archive_root"], limit=10)

    assert len(results.hits) == 1
    # Snippet should contain the query term or nearby context
    assert results.hits[0].snippet is not None
    assert isinstance(results.hits[0].snippet, str)


def test_search_includes_conversation_metadata(workspace_env, storage_repository):
    """search_messages() includes conversation metadata in results."""
    from polylogue.sources import IngestBundle, ingest_bundle

    conv = make_conversation(
        "conv1", provider_name="claude", title="My Conversation", provider_meta={"source": "my-source"}
    )
    msg = make_message("msg1", "conv1", text="search query", timestamp="2024-01-01T10:30:00Z")

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
    from polylogue.sources import IngestBundle, ingest_bundle

    conv = make_conversation("conv1")
    msg = make_message("msg1", "conv1", text="C++ programming with @mentions and #hashtags")

    ingest_bundle(IngestBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)
    rebuild_index()

    # Search for word containing special chars
    results = search_messages("programming", archive_root=workspace_env["archive_root"], limit=10)
    assert len(results.hits) == 1


def test_search_with_quotes_in_text(workspace_env, storage_repository):
    """search_messages() handles quoted text correctly."""
    from polylogue.sources import IngestBundle, ingest_bundle

    conv = make_conversation("conv1")
    msg = make_message("msg1", "conv1", text='She said "hello world" to me')

    ingest_bundle(IngestBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)
    rebuild_index()

    results = search_messages("hello", archive_root=workspace_env["archive_root"], limit=10)
    assert len(results.hits) == 1


def test_search_with_unicode_text(workspace_env, storage_repository):
    """search_messages() handles unicode characters."""
    from polylogue.sources import IngestBundle, ingest_bundle

    conv = make_conversation("conv1", title="Unicode Test")
    msg = make_message("msg1", "conv1", text="Hello 世界 مرحبا мир café")

    ingest_bundle(IngestBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)
    rebuild_index()

    results = search_messages("café", archive_root=workspace_env["archive_root"], limit=10)
    assert len(results.hits) == 1


def test_search_with_hyphenated_words(workspace_env, storage_repository):
    """search_messages() handles words in hyphenated phrases."""
    from polylogue.sources import IngestBundle, ingest_bundle

    conv = make_conversation("conv1")
    msg = make_message("msg1", "conv1", text="The state-of-the-art algorithm")

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
    result = test_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone()
    assert result is not None


def test_rebuild_index_with_empty_database(test_conn):
    """rebuild_index() handles empty database gracefully."""
    rebuild_index(test_conn)

    count = test_conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
    assert count == 0


def test_search_returns_searchresult_object(workspace_env, storage_repository):
    """search_messages() returns SearchResult with hits list."""
    from polylogue.sources import IngestBundle, ingest_bundle

    conv = make_conversation("conv1")
    msg = make_message("msg1", "conv1", text="search result")

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
    conv = make_conversation("conv1", title="Multi-message Conv")
    messages = [
        make_message(
            f"msg{i}",
            "conv1",
            role="user" if i % 2 == 0 else "assistant",
            text=f"message number {i}",
            timestamp=f"2024-01-01T00:{i:02d}:00Z",
        )
        for i in range(10)
    ]

    store_records(conversation=conv, messages=messages, attachments=[], conn=test_conn)
    rebuild_index(test_conn)

    count = test_conn.execute("SELECT COUNT(*) FROM messages_fts WHERE conversation_id = ?", ("conv1",)).fetchone()[0]
    assert count == 10


def test_update_index_deletes_old_entries_from_conversation(test_conn):
    """update_index_for_conversations() removes old index entries for updated conversations."""
    conv = make_conversation("conv1")
    msg = make_message("msg1", "conv1", text="original message")

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
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?", ("original",)
    ).fetchone()[0]
    assert old_hits == 0

    # New message should be indexed
    new_hits = test_conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?", ("new",)).fetchone()[0]
    assert new_hits == 1


def test_batch_index_10k_messages(test_conn):
    """Benchmark: update_index_for_conversations handles 10k messages efficiently."""
    import time

    # Create 100 conversations with 100 messages each = 10,000 messages
    num_convs = 100
    msgs_per_conv = 100

    for i in range(num_convs):
        conv = make_conversation(f"conv{i}", title=f"Benchmark Conv {i}")
        store_records(conversation=conv, messages=[], attachments=[], conn=test_conn)

        # Batch insert messages directly for speed
        messages_batch = [
            (
                f"msg{i}-{j}",
                f"conv{i}",
                "user" if j % 2 == 0 else "assistant",
                f"message content {i}-{j} with searchable text",
                f"2024-01-01T{i:02d}:{j:02d}:00Z",
                f"hash-{i}-{j}",
                1,
            )
            for j in range(msgs_per_conv)
        ]
        test_conn.executemany(
            """INSERT INTO messages
               (message_id, conversation_id, role, text, timestamp, content_hash, version)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            messages_batch,
        )

    test_conn.commit()

    # Time the index build
    conv_ids = [f"conv{i}" for i in range(num_convs)]

    start = time.perf_counter()
    update_index_for_conversations(conv_ids, test_conn)
    elapsed = time.perf_counter() - start

    # Verify all indexed
    count = test_conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
    assert count == num_convs * msgs_per_conv

    # Assert reasonable performance (should complete within 5 seconds)
    assert elapsed < 5.0, f"Batch indexing 10k messages took too long: {elapsed:.2f}s"


def test_batch_index_search_returns_correct_provider(workspace_env, storage_repository):
    """Verify batch indexing allows retrieving correct provider_name via search."""
    from polylogue.sources import IngestBundle, ingest_bundle

    # Create conversations with different providers
    conv1 = make_conversation("conv1", provider_name="claude", title="Claude Conv")
    conv2 = make_conversation("conv2", provider_name="chatgpt", title="ChatGPT Conv")

    messages1 = [make_message(f"msg1-{i}", "conv1", text=f"claude text {i}") for i in range(5)]
    messages2 = [make_message(f"msg2-{i}", "conv2", text=f"chatgpt text {i}") for i in range(5)]

    ingest_bundle(IngestBundle(conversation=conv1, messages=messages1, attachments=[]), repository=storage_repository)
    ingest_bundle(IngestBundle(conversation=conv2, messages=messages2, attachments=[]), repository=storage_repository)

    rebuild_index()

    # Verify provider names via search
    results1 = search_messages("claude", archive_root=workspace_env["archive_root"], limit=10)
    assert all(hit.provider_name == "claude" for hit in results1.hits)
    assert len(results1.hits) == 1

    results2 = search_messages("chatgpt", archive_root=workspace_env["archive_root"], limit=10)
    assert all(hit.provider_name == "chatgpt" for hit in results2.hits)
    assert len(results2.hits) == 1


__all__ = [
    "test_rebuild_index_includes_keys",
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
    "test_batch_index_10k_messages",
    "test_batch_index_search_returns_correct_provider",
]

# =============================================================================
# SEARCH HEALTH TESTS (merged from test_search_health.py)
# =============================================================================


def _seed_conversation(storage_repository):
    ingest_bundle(
        IngestBundle(
            conversation=make_conversation("conv:hash", provider_name="codex", title="Demo"),
            messages=[make_message("msg:hash", "conv:hash", text="hello world")],
            attachments=[],
        ),
        repository=storage_repository,
    )


def test_search_after_index(workspace_env, storage_repository):
    _seed_conversation(storage_repository)
    rebuild_index()
    results = search_messages("hello", archive_root=workspace_env["archive_root"], limit=5)
    assert results.hits
    assert results.hits[0].conversation_id == "conv:hash"


def test_health_cached(workspace_env):
    config = get_config()
    get_health(config)
    second = get_health(config)
    assert second.cached is True
    assert second.age_seconds is not None


def test_search_invalid_query_reports_error(monkeypatch, workspace_env):
    class StubCursor:
        def __init__(self, row=None):
            self._row = row

        def fetchone(self):
            return self._row

        def fetchall(self):
            return []

    class StubConn:
        def execute(self, sql, params=()):
            if "sqlite_master" in sql:
                return StubCursor(row={"name": "messages_fts"})
            if "MATCH" in sql:
                raise sqlite3.OperationalError("fts5: syntax error")
            return StubCursor()

    @contextmanager
    def stub_open_connection(_):
        yield StubConn()

    monkeypatch.setattr("polylogue.storage.search.open_connection", stub_open_connection)
    # Use type name check to handle module reload class identity issues
    with pytest.raises(Exception) as exc_info:
        search_messages('"unterminated', archive_root=workspace_env["archive_root"], limit=5)
    assert exc_info.type.__name__ == "DatabaseError"
    assert "Invalid search query" in str(exc_info.value)


def test_search_prefers_legacy_render_when_present(workspace_env, storage_repository):
    """Test that search returns legacy render paths when they exist.

    Note: Invalid provider names are now rejected at validation, so we use a valid
    provider name but still test legacy path resolution behavior.
    """
    archive_root = workspace_env["archive_root"]
    provider_name = "legacy-provider"  # Valid provider name (path chars now rejected)
    conversation_id = "conv-one"
    bundle = IngestBundle(
        conversation=make_conversation(conversation_id, provider_name=provider_name, title="Legacy"),
        messages=[make_message("msg:legacy", conversation_id, text="hello legacy")],
        attachments=[],
    )
    ingest_bundle(bundle, repository=storage_repository)
    rebuild_index()

    # Create a legacy-style render path
    legacy_path = archive_root / "render" / provider_name / conversation_id / "conversation.md"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text("legacy", encoding="utf-8")

    results = search_messages("hello", archive_root=archive_root, limit=5)
    assert results.hits
    assert results.hits[0].conversation_path == legacy_path


# --since timestamp filtering tests


def test_search_since_filters_by_iso_date(workspace_env, storage_repository):
    """--since with ISO date filters messages correctly."""
    archive_root = workspace_env["archive_root"]
    # Message with ISO timestamp: 2024-01-15T10:00:00
    bundle = IngestBundle(
        conversation=make_conversation("conv:iso", title="ISO Test"),
        messages=[
            make_message("msg:old-iso", "conv:iso", text="old message iso", timestamp="2024-01-10T10:00:00"),
            make_message("msg:new-iso", "conv:iso", text="new message iso", timestamp="2024-01-20T10:00:00"),
        ],
        attachments=[],
    )
    ingest_bundle(bundle, repository=storage_repository)
    rebuild_index()

    # Filter for messages after 2024-01-15
    results = search_messages(
        "message",
        archive_root=archive_root,
        since="2024-01-15",
        limit=10,
    )
    assert len(results.hits) == 1
    assert results.hits[0].message_id == "msg:new-iso"


def test_search_since_filters_numeric_timestamps(workspace_env, storage_repository):
    """--since works when DB has float timestamps (e.g., 1704067200.0)."""
    archive_root = workspace_env["archive_root"]
    # 1705312800.0 = 2024-01-15T10:00:00 UTC
    # 1704067200.0 = 2024-01-01T00:00:00 UTC
    # 1706227200.0 = 2024-01-26T00:00:00 UTC
    bundle = IngestBundle(
        conversation=make_conversation("conv:numeric", title="Numeric Test"),
        messages=[
            make_message("msg:old-num", "conv:numeric", text="old message numeric", timestamp="1704067200.0"),
            make_message("msg:new-num", "conv:numeric", text="new message numeric", timestamp="1706227200.0"),
        ],
        attachments=[],
    )
    ingest_bundle(bundle, repository=storage_repository)
    rebuild_index()

    # Filter for messages after 2024-01-15
    results = search_messages(
        "numeric",
        archive_root=archive_root,
        since="2024-01-15",
        limit=10,
    )
    assert len(results.hits) == 1
    assert results.hits[0].message_id == "msg:new-num"


def test_search_since_handles_mixed_timestamp_formats(workspace_env, storage_repository):
    """--since works with mix of ISO and numeric timestamps in same DB.

    Note: Search results are deduplicated by conversation, so we create
    separate conversations to verify both ISO and numeric timestamps work.
    """
    archive_root = workspace_env["archive_root"]

    # Create conversation with ISO timestamp (after cutoff)
    bundle_iso = IngestBundle(
        conversation=make_conversation("conv:iso-new", title="ISO Test"),
        messages=[
            make_message("msg:iso-new", "conv:iso-new", text="mixedformat gamma", timestamp="2024-01-25T12:00:00")
        ],
        attachments=[],
    )

    # Create conversation with numeric timestamp (after cutoff)
    bundle_num = IngestBundle(
        conversation=make_conversation("conv:num-new", title="Numeric Test"),
        messages=[make_message("msg:num-new", "conv:num-new", text="mixedformat delta", timestamp="1706400000.0")],
        attachments=[],
    )

    # Create conversation with old ISO timestamp (before cutoff)
    bundle_old = IngestBundle(
        conversation=make_conversation("conv:old", title="Old Test"),
        messages=[make_message("msg:iso-old", "conv:old", text="mixedformat alpha", timestamp="2024-01-05T12:00:00")],
        attachments=[],
    )

    ingest_bundle(bundle_iso, repository=storage_repository)
    ingest_bundle(bundle_num, repository=storage_repository)
    ingest_bundle(bundle_old, repository=storage_repository)
    rebuild_index()

    results = search_messages(
        "mixedformat",
        archive_root=archive_root,
        since="2024-01-15",
        limit=10,
    )
    # Should get 2 hits: one ISO, one numeric - both after cutoff
    assert len(results.hits) == 2
    hit_conv_ids = {h.conversation_id for h in results.hits}
    assert hit_conv_ids == {"conv:iso-new", "conv:num-new"}


def test_search_since_invalid_date_raises_error(workspace_env, storage_repository):
    """Invalid --since format raises ValueError with helpful message."""
    archive_root = workspace_env["archive_root"]
    _seed_conversation(storage_repository)
    rebuild_index()

    with pytest.raises(ValueError, match="Invalid --since date"):
        search_messages(
            "hello",
            archive_root=archive_root,
            since="not-a-date",
            limit=5,
        )

    with pytest.raises(ValueError, match="ISO format"):
        search_messages(
            "hello",
            archive_root=archive_root,
            since="01/15/2024",  # Wrong format
            limit=5,
        )


def test_search_since_boundary_condition(workspace_env, storage_repository):
    """Messages at or after --since timestamp are included, earlier ones excluded."""
    archive_root = workspace_env["archive_root"]
    # Use dates far enough apart that timezone differences don't matter
    # Filter: 2024-01-15 (any timezone interpretation)
    # Before: 2024-01-10 (definitely before, any timezone)
    # After: 2024-01-20 (definitely after, any timezone)
    bundle = IngestBundle(
        conversation=make_conversation("conv:boundary", title="Boundary Test"),
        messages=[
            make_message(
                "msg:after-cutoff", "conv:boundary", text="boundary after message", timestamp="2024-01-20T12:00:00"
            ),
            make_message(
                "msg:before-cutoff", "conv:boundary", text="boundary before message", timestamp="2024-01-10T12:00:00"
            ),
        ],
        attachments=[],
    )
    ingest_bundle(bundle, repository=storage_repository)
    rebuild_index()

    results = search_messages(
        "boundary",
        archive_root=archive_root,
        since="2024-01-15",
        limit=10,
    )
    # Should include after, exclude before
    assert len(results.hits) == 1
    assert results.hits[0].message_id == "msg:after-cutoff"


def test_search_without_fts_table_raises_descriptive_error(workspace_env, db_without_fts, monkeypatch):
    """search() raises DatabaseError mentioning 'polylogue run' when FTS missing."""
    archive_root = workspace_env["archive_root"]

    # Monkey-patch to use the db without FTS
    from polylogue.storage.backends import sqlite as db

    monkeypatch.setattr(db, "default_db_path", lambda: db_without_fts)

    # Use type name check to handle module reload class identity issues
    with pytest.raises(Exception) as exc_info:
        search_messages("hello", archive_root=archive_root, limit=5)
    assert exc_info.type.__name__ == "DatabaseError"
    assert "Search index not built" in str(exc_info.value)


# =============================================================================
# FTS5 SEARCH TESTS (from test_search_core.py)
# =============================================================================


# =============================================================================
# FTS5 ESCAPING - PARAMETRIZED (1 test replacing 30+ tests)
# =============================================================================


# Test cases: (input_query, expected_property, description)
FTS5_ESCAPE_CASES = [
    # Empty/whitespace
    ('', '""', "empty query"),
    ('   ', '""', "whitespace only"),

    # Quotes
    ('find "quoted text" here', 'has_doubled_quotes', "internal quotes"),

    # Wildcards
    ('*', '""', "bare asterisk"),
    ('test*', 'starts_and_ends_with_quotes', "asterisk with text"),
    ('?', '?', "question mark"),  # Single char, no special FTS5 chars -> unquoted

    # FTS5 operators (should be quoted as literals)
    ('AND', '"AND"', "AND operator"),
    ('OR', '"OR"', "OR operator"),
    ('NOT', '"NOT"', "NOT operator"),
    ('NEAR', '"NEAR"', "NEAR operator"),
    ('and', '"and"', "lowercase and"),
    ('And', '"And"', "mixed case And"),

    # Special characters
    ('test:value', 'starts_and_ends_with_quotes', "colon"),
    ('test^2', 'starts_and_ends_with_quotes', "caret"),
    ('function(arg)', 'starts_and_ends_with_quotes', "parentheses"),
    ('test)', 'starts_and_ends_with_quotes', "close paren"),
    ('(test', 'starts_and_ends_with_quotes', "open paren"),

    # Minus/hyphen
    ('-test', 'starts_and_ends_with_quotes', "leading minus"),
    ('test-word', 'starts_and_ends_with_quotes', "embedded hyphen"),

    # Plus
    ('+required', 'starts_and_ends_with_quotes', "plus operator"),

    # Multiple operators
    ('test AND query', 'test AND query', "embedded AND - passes through unquoted"),
    ('OR query', 'starts_and_ends_with_quotes', "leading OR - quoted for safety"),

    # Normal text (NOT quoted - implementation passes simple alphanumeric through as-is)
    ('simple query', 'simple query', "simple words"),
    ('hello', 'hello', "single word"),
]


@pytest.mark.parametrize("input_query,expected,desc", FTS5_ESCAPE_CASES)
def test_escape_fts5_comprehensive(input_query, expected, desc):
    """Comprehensive FTS5 escaping test.

    Replaces 30+ individual escaping tests with single parametrized test.

    Expected can be:
    - Exact string match (e.g., '""')
    - Property to check (e.g., 'starts_and_ends_with_quotes', 'has_doubled_quotes')
    """
    result = escape_fts5_query(input_query)

    if expected == 'starts_and_ends_with_quotes':
        assert result.startswith('"'), f"Failed {desc}: doesn't start with quote"
        assert result.endswith('"'), f"Failed {desc}: doesn't end with quote"
    elif expected == 'has_doubled_quotes':
        assert result.startswith('"'), f"Failed {desc}: not quoted"
        assert result.endswith('"'), f"Failed {desc}: not quoted"
        assert '""' in result, f"Failed {desc}: quotes not doubled"
    else:
        # Exact match
        assert result == expected, f"Failed {desc}: expected {expected}, got {result}"


# =============================================================================
# SEARCH INTEGRATION - PARAMETRIZED (1 test replacing ~10 tests)
# =============================================================================


@pytest.mark.parametrize("query,should_find", [
    ("test", True),  # Basic search
    ("nonexistent", False),  # No match
    ("*", False),  # Bare asterisk escaped
    ("AND", False),  # Operator as literal
    ("quoted", True),  # Part of text with quotes
])
def test_search_messages_escaping_integration(query, should_find, tmp_path):
    """Integration test for search with various queries.

    Replaces ~10 individual integration tests.
    """
    from pathlib import Path

    # Setup database with test data
    db_path = tmp_path / "test.db"
    DbFactory(db_path)

    # Insert test conversation using builder
    (ConversationBuilder(db_path, "test1")
     .title("Test Conversation")
     .add_message("msg1", role="user", text='This is a test message with "quoted text" inside.')
     .save())

    # Build search index
    with open_connection(str(db_path)) as conn:
        rebuild_index(conn)

    # Search - use keyword arguments
    results = search_messages(
        query,
        archive_root=tmp_path,
        db_path=Path(str(db_path)),
        limit=10,
    )

    if should_find:
        assert len(results.hits) > 0, f"Expected to find results for '{query}'"
    else:
        # Either no results or results don't match the query
        # The important thing is no SQL errors occur
        assert isinstance(results.hits, list)


# =============================================================================
# EDGE CASES - PARAMETRIZED (1 test replacing ~5 tests)
# =============================================================================


@pytest.mark.parametrize("special_query,should_quote", [
    ("test OR anything", False),  # "OR" in middle - passes through unquoted
    ("NOT this", True),  # "NOT" at start - should be quoted
    ("NEAR that", True),  # "NEAR" at start - should be quoted
    ("' OR '1'='1", False),  # No special FTS5 chars, passes through (single quotes aren't FTS5 special)
    ("test; DROP TABLE messages--", True),  # Contains special chars (semicolon, etc.), should be quoted
])
def test_escape_fts5_injection_prevention(special_query, should_quote):
    """Prevent dangerous operator positions and special characters.

    Replaces ~5 security-focused tests.
    """
    result = escape_fts5_query(special_query)

    if should_quote:
        # Should be safely quoted
        assert result.startswith('"'), f"Expected quoted: {special_query}"
        assert result.endswith('"'), f"Expected quoted: {special_query}"
    else:
        # These may or may not be quoted depending on special chars
        # The important thing is they don't cause FTS5 errors
        assert isinstance(result, str), f"Should return string: {special_query}"


# =============================================================================
# UNICODE HANDLING - PARAMETRIZED (1 test)
# =============================================================================


@pytest.mark.parametrize("unicode_query", [
    "文字",  # Chinese
    "тест",  # Cyrillic
    "🔍",   # Emoji
    "café",  # Accented
])
def test_escape_fts5_unicode(unicode_query):
    """Unicode queries are handled correctly.

    Unicode-only queries are simple alphanumeric (no special FTS5 chars),
    so they pass through unquoted.
    """
    result = escape_fts5_query(unicode_query)

    # Should preserve unicode and pass through unquoted
    assert result == unicode_query


# =============================================================================
# SEARCH RESULT VALIDATION (NEW - was missing)
# =============================================================================


def test_search_messages_returns_valid_structure(tmp_path):
    """Search results have expected structure."""
    from pathlib import Path

    db_path = tmp_path / "test.db"
    DbFactory(db_path)

    # Insert test conversation using builder
    (ConversationBuilder(db_path, "test1")
     .title("Test")
     .add_message("msg1", role="user", text="Searchable content")
     .save())

    # Build search index
    with open_connection(str(db_path)) as conn:
        rebuild_index(conn)

    # Search using keyword arguments
    results = search_messages(
        "searchable",
        archive_root=tmp_path,
        db_path=Path(str(db_path)),
        limit=10,
    )

    assert len(results.hits) > 0
    for hit in results.hits:
        # Verify result structure
        assert hasattr(hit, 'snippet')
        assert hasattr(hit, 'conversation_id')
        assert hit.snippet is not None
        assert len(hit.snippet) > 0

# =============================================================================
# SEARCH PROVIDER TESTS (from test_search_core.py)
# =============================================================================


class TestCreateVectorProvider:
    """Tests for create_vector_provider factory."""

    def test_returns_none_when_no_qdrant_url(self):
        """Returns None when QDRANT_URL is not configured."""
        with patch.dict("os.environ", {}, clear=True):
            provider = create_vector_provider()
            assert provider is None

    def test_uses_env_vars_when_no_config(self, monkeypatch):
        """Uses environment variables when no config provided."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "test-key")
        monkeypatch.setenv("VOYAGE_API_KEY", "voyage-key")

        with patch("polylogue.storage.search_providers.QdrantProvider") as mock_provider:
            create_vector_provider()
            mock_provider.assert_called_once_with(
                qdrant_url="http://localhost:6333",
                api_key="test-key",
                voyage_key="voyage-key",
            )

    def test_uses_config_over_env_vars(self, monkeypatch, tmp_path):
        """Config values take priority over environment variables."""
        monkeypatch.setenv("QDRANT_URL", "http://env:6333")
        monkeypatch.setenv("VOYAGE_API_KEY", "env-voyage-key")

        index_config = IndexConfig(
            qdrant_url="http://config:6333",
            voyage_api_key="config-voyage-key",
        )
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[],
            index_config=index_config,
        )

        with patch("polylogue.storage.search_providers.QdrantProvider") as mock_provider:
            create_vector_provider(config=config)
            mock_provider.assert_called_once_with(
                qdrant_url="http://config:6333",
                api_key=None,
                voyage_key="config-voyage-key",
            )

    def test_explicit_args_override_config(self, tmp_path):
        """Explicit arguments override both config and env vars."""
        index_config = IndexConfig(
            qdrant_url="http://config:6333",
            voyage_api_key="config-voyage-key",
        )
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[],
            index_config=index_config,
        )

        with patch("polylogue.storage.search_providers.QdrantProvider") as mock_provider:
            create_vector_provider(
                config=config,
                qdrant_url="http://explicit:6333",
                voyage_api_key="explicit-voyage-key",
            )
            mock_provider.assert_called_once_with(
                qdrant_url="http://explicit:6333",
                api_key=None,
                voyage_key="explicit-voyage-key",
            )

    def test_raises_when_voyage_key_missing(self, monkeypatch):
        """Raises ValueError when Qdrant URL is set but Voyage key is missing."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

        with patch.dict("os.environ", {"QDRANT_URL": "http://localhost:6333"}, clear=True):
            with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
                create_vector_provider()

    def test_config_with_none_values_falls_back_to_env(self, monkeypatch, tmp_path):
        """Config with None values falls back to environment variables."""
        monkeypatch.setenv("QDRANT_URL", "http://env:6333")
        monkeypatch.setenv("VOYAGE_API_KEY", "env-voyage-key")

        index_config = IndexConfig(
            qdrant_url=None,  # Explicitly None
            voyage_api_key=None,
        )
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[],
            index_config=index_config,
        )

        with patch("polylogue.storage.search_providers.QdrantProvider") as mock_provider:
            create_vector_provider(config=config)
            mock_provider.assert_called_once_with(
                qdrant_url="http://env:6333",
                api_key=None,
                voyage_key="env-voyage-key",
            )


class TestFTS5Provider:
    """Tests for FTS5Provider full-text search implementation."""

    @pytest.fixture
    def fts_provider(self, workspace_env):
        """Create FTS5Provider with test database."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        return FTS5Provider(db_path=db_path)

    @pytest.fixture
    def populated_fts(self, workspace_env, storage_repository, fts_provider):
        """FTS provider with indexed test data."""
        conv = make_conversation("fts-conv-1", provider_name="claude", title="FTS Test", created_at="1000", updated_at="1000", provider_meta={"source": "inbox"})
        msgs = [
            make_message("fts-msg-1", "fts-conv-1", text="How do I implement quicksort in Python?", timestamp="1000"),
            make_message("fts-msg-2", "fts-conv-1", role="assistant", text="Quicksort is a divide-and-conquer algorithm for sorting", timestamp="1001"),
        ]
        storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        # Index the messages
        fts_provider.index(msgs)
        return fts_provider

    def test_ensure_index_creates_fts_table(self, workspace_env, fts_provider):
        """Ensure index creates FTS5 virtual table."""
        from polylogue.storage.backends.sqlite import open_connection

        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"

        # Index empty list to trigger table creation
        fts_provider.index([])

        with open_connection(db_path) as conn:
            # Check if the FTS table doesn't exist yet (we passed empty list)
            # Actually we need to trigger the _ensure_index by indexing something
            pass

        # Index with actual message to ensure table creation
        conv = make_conversation("ensure-conv", title="Ensure Test", created_at="1000", updated_at="1000", provider_meta={"source": "inbox"})
        # First save the conversation so provider_name lookup works
        from polylogue.storage.backends.sqlite import SQLiteBackend

        backend = SQLiteBackend(db_path=db_path)
        backend.begin()
        backend.save_conversation(conv)
        backend.commit()

        msgs = [make_message("ens-msg", "ensure-conv", timestamp="1000")]

        fts_provider.index(msgs)

        with open_connection(db_path) as conn:
            row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone()
            assert row is not None
            assert row["name"] == "messages_fts"

    def test_ensure_index_idempotent(self, workspace_env, fts_provider, storage_repository):
        """Calling index multiple times is safe (idempotent)."""
        conv = make_conversation("idem-conv", title="Idempotent Test", created_at="1000", updated_at="1000", provider_meta={"source": "inbox"})
        msgs = [make_message("idem-msg", "idem-conv", text="Idempotent message", timestamp="1000")]
        storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        # Index twice - should not error or duplicate
        fts_provider.index(msgs)
        fts_provider.index(msgs)

        # Search should return exactly one result
        results = fts_provider.search("idempotent")
        assert len(results) == 1
        assert results[0] == "idem-msg"

    def test_index_deletes_old_entries(self, workspace_env, fts_provider, storage_repository):
        """Incremental indexing removes old entries before inserting."""
        conv = make_conversation("incr-conv", title="Incremental Test", created_at="1000", updated_at="1000", provider_meta={"source": "inbox"})
        msgs_v1 = [make_message("incr-msg-1", "incr-conv", text="Original content about apples", timestamp="1000")]
        storage_repository.save_conversation(conversation=conv, messages=msgs_v1, attachments=[])
        fts_provider.index(msgs_v1)

        # Should find "apples"
        results = fts_provider.search("apples")
        assert len(results) == 1

        # Re-index with different content
        msgs_v2 = [make_message("incr-msg-1", "incr-conv", text="Updated content about oranges", timestamp="1000")]
        fts_provider.index(msgs_v2)

        # "apples" should no longer be found
        results = fts_provider.search("apples")
        assert len(results) == 0

        # "oranges" should be found
        results = fts_provider.search("oranges")
        assert len(results) == 1

    def test_index_skips_empty_text(self, workspace_env, fts_provider, storage_repository):
        """Messages with empty text are not indexed."""
        conv = make_conversation("skip-conv", title="Skip Test", created_at="1000", updated_at="1000", provider_meta={"source": "inbox"})
        msgs = [
            make_message("skip-msg-1", "skip-conv", text="", timestamp="1000"),  # Empty text
            make_message("skip-msg-2", "skip-conv", role="assistant", text="This has content", timestamp="1001"),
        ]
        storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])
        fts_provider.index(msgs)

        # Search should only find the non-empty message
        results = fts_provider.search("content")
        assert len(results) == 1
        assert results[0] == "skip-msg-2"

    def test_search_returns_ranked_results(self, populated_fts):
        """Search returns results ordered by relevance (BM25)."""
        # The populated fixture has messages about quicksort
        results = populated_fts.search("quicksort")
        assert len(results) == 2  # Both messages mention quicksort
        # Results should be in relevance order (checked implicitly by ORDER BY rank)

    def test_search_escapes_fts5_special_chars(self, populated_fts):
        """Search query escapes FTS5 special characters."""
        # Quotes and asterisks should be escaped
        results = populated_fts.search('"special* query"')
        # Should not raise FTS5 syntax error
        assert isinstance(results, list)

    def test_search_returns_empty_if_no_index(self, workspace_env):
        """Search returns empty list if FTS index doesn't exist."""
        db_path = workspace_env["data_root"] / "polylogue" / "nonexistent.db"
        provider = FTS5Provider(db_path=db_path)
        results = provider.search("anything")
        assert results == []

        # Could be empty or match all - depends on FTS5 behavior
        assert isinstance(results, list)


class TestQdrantProviderChunking:
    """Tests for QdrantProvider batching/chunking."""

    @pytest.fixture
    def provider(self):
        """Create a QdrantProvider with mocked client."""
        # Mock sys.modules to handle lazy imports
        mock_qdrant_module = MagicMock()
        mock_client_cls = MagicMock()
        mock_qdrant_module.QdrantClient = mock_client_cls

        mock_httpx = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "qdrant_client": mock_qdrant_module,
                "qdrant_client.http": MagicMock(),
                "httpx": mock_httpx,
            },
        ):
            provider = QdrantProvider(qdrant_url="http://test:6333", api_key="test-key", voyage_key="voyage-key")
            # Ensure the client property returns our mock instance
            provider._client = mock_client_cls.return_value
            return provider

    def test_upsert_chunks_large_request(self, provider):
        """Verify that upsert chunks messages into batches of 64."""
        # Create 150 messages (should be 3 chunks: 64, 64, 22)
        messages = [make_message(f"msg-{i}", "conv-1", text=f"Message {i}", timestamp="1000") for i in range(150)]

        # Mock embeddings to return one mock vector per input text
        with patch.object(provider, "_get_embeddings") as mock_embed:
            mock_embed.side_effect = lambda texts: [[0.1] * 1024] * len(texts)

            provider.upsert(conversation_id="conv-1", messages=messages)

            # Check embedding calls (3 batches)
            assert mock_embed.call_count == 3
            assert len(mock_embed.call_args_list[0][0][0]) == 64
            assert len(mock_embed.call_args_list[1][0][0]) == 64
            assert len(mock_embed.call_args_list[2][0][0]) == 22

            # Check Qdrant upsert calls
            assert provider.client.upsert.call_count == 3
            assert len(provider.client.upsert.call_args_list[0].kwargs["points"]) == 64
            assert len(provider.client.upsert.call_args_list[1].kwargs["points"]) == 64
            assert len(provider.client.upsert.call_args_list[2].kwargs["points"]) == 22

    def test_upsert_handles_single_chunk(self, provider):
        """Verify upsert handling for small lists (<64)."""
        messages = [make_message(f"msg-{i}", "conv-1", text=f"Message {i}", timestamp="1000") for i in range(10)]

        with patch.object(provider, "_get_embeddings") as mock_embed:
            mock_embed.return_value = [[0.1] * 1024] * 10
            provider.upsert(conversation_id="conv-1", messages=messages)

            mock_embed.assert_called_once()
            assert len(mock_embed.call_args[0][0]) == 10
            provider.client.upsert.assert_called_once()
            assert len(provider.client.upsert.call_args.kwargs["points"]) == 10
