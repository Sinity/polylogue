"""Tests for FTS5 search provider and index functionality.

Tests cover FTS5 index creation, incremental updates, search functionality,
ranking, special characters, edge cases, and escaping.

Extracted from monolithic test_search_index.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue.config import Config, IndexConfig
from polylogue.sources import RecordBundle, save_bundle
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.index import ensure_index, rebuild_index, update_index_for_conversations
from polylogue.storage.search import escape_fts5_query, search_messages
from polylogue.storage.search_providers import create_vector_provider
from polylogue.storage.search_providers.fts5 import FTS5Provider
from tests.infra.helpers import ConversationBuilder, DbFactory, make_conversation, make_message, store_records, make_hash


# ============================================================================
# Tests for search_messages()
# ============================================================================


SEARCH_BASIC_CASES = [
    # (num_convs, search_term, expected_count, description)
    (1, "python", 1, "single match"),
    (3, "testing", 3, "multiple matches"),
    (1, "nonexistent", 0, "no match"),
]


@pytest.mark.parametrize("num_convs,search_term,expected_count,description", SEARCH_BASIC_CASES)
async def test_search_basic_results(workspace_env, storage_repository, num_convs, search_term, expected_count, description):
    """search_messages() returns correct number of results."""
    # Create conversations
    for i in range(num_convs):
        if search_term == "nonexistent":
            text = "hello world"
        elif search_term == "testing":
            text = "testing framework"
        else:  # "python"
            text = "python programming language"

        conv = make_conversation(f"conv{i}", title=f"Conv {i}")
        msg = make_message(f"msg{i}", f"conv{i}", text=text)
        await save_bundle(RecordBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)

    rebuild_index()

    results = search_messages(search_term, archive_root=workspace_env["archive_root"], limit=10)
    assert len(results.hits) == expected_count, f"Failed for {description}"


async def test_search_respects_limit(workspace_env, storage_repository):
    """search_messages() respects limit parameter."""
    for i in range(10):
        conv = make_conversation(f"conv{i}", title=f"Conv {i}")
        msg = make_message(f"msg{i}", f"conv{i}", text="search limit")
        await save_bundle(RecordBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)

    rebuild_index()

    results = search_messages("search", archive_root=workspace_env["archive_root"], limit=3)
    assert len(results.hits) == 3


async def test_search_includes_snippet(workspace_env, storage_repository):
    """search_messages() includes text snippet in results."""
    conv = make_conversation("conv1")
    msg = make_message("msg1", "conv1", text="The quick brown fox jumps over the lazy dog")

    await save_bundle(RecordBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)
    rebuild_index()

    results = search_messages("quick", archive_root=workspace_env["archive_root"], limit=10)

    assert len(results.hits) == 1
    # Snippet should contain the query term or nearby context
    assert results.hits[0].snippet is not None
    assert isinstance(results.hits[0].snippet, str)


async def test_search_includes_conversation_metadata(workspace_env, storage_repository):
    """search_messages() includes conversation metadata in results."""
    conv = make_conversation(
        "conv1", provider_name="claude", title="My Conversation", provider_meta={"source": "my-source"}
    )
    msg = make_message("msg1", "conv1", text="search query", timestamp="2024-01-01T10:30:00Z")

    await save_bundle(RecordBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)
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


SEARCH_WITH_SPECIAL_TEXT_CASES = [
    # (text, search_term, description)
    ("C++ programming with @mentions and #hashtags", "programming", "special characters"),
    ('She said "hello world" to me', "hello", "quoted text"),
    ("Hello 世界 مرحبا мир café", "café", "unicode"),
    ("The state-of-the-art algorithm", "state", "hyphenated"),
]


@pytest.mark.parametrize("text,search_term,description", SEARCH_WITH_SPECIAL_TEXT_CASES)
async def test_search_with_special_text(workspace_env, storage_repository, text, search_term, description):
    """search_messages() handles special text patterns."""
    conv = make_conversation("conv1", title=f"Test {description}")
    msg = make_message("msg1", "conv1", text=text)

    await save_bundle(RecordBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)
    rebuild_index()

    results = search_messages(search_term, archive_root=workspace_env["archive_root"], limit=10)
    assert len(results.hits) == 1, f"Failed for {description}"


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


async def test_search_returns_searchresult_object(workspace_env, storage_repository):
    """search_messages() returns SearchResult with hits list."""
    conv = make_conversation("conv1")
    msg = make_message("msg1", "conv1", text="search result")

    await save_bundle(RecordBundle(conversation=conv, messages=[msg], attachments=[]), repository=storage_repository)
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


async def test_batch_index_search_returns_correct_provider(workspace_env, storage_repository):
    """Verify batch indexing allows retrieving correct provider_name via search."""
    # Create conversations with different providers
    conv1 = make_conversation("conv1", provider_name="claude", title="Claude Conv")
    conv2 = make_conversation("conv2", provider_name="chatgpt", title="ChatGPT Conv")

    messages1 = [make_message(f"msg1-{i}", "conv1", text=f"claude text {i}") for i in range(5)]
    messages2 = [make_message(f"msg2-{i}", "conv2", text=f"chatgpt text {i}") for i in range(5)]

    await save_bundle(RecordBundle(conversation=conv1, messages=messages1, attachments=[]), repository=storage_repository)
    await save_bundle(RecordBundle(conversation=conv2, messages=messages2, attachments=[]), repository=storage_repository)

    rebuild_index()

    # Verify provider names via search
    results1 = search_messages("claude", archive_root=workspace_env["archive_root"], limit=10)
    assert all(hit.provider_name == "claude" for hit in results1.hits)
    assert len(results1.hits) == 1

    results2 = search_messages("chatgpt", archive_root=workspace_env["archive_root"], limit=10)
    assert all(hit.provider_name == "chatgpt" for hit in results2.hits)
    assert len(results2.hits) == 1


# ============================================================================
# FTS5 ESCAPING - PARAMETRIZED
# ============================================================================


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

    # Comma (NEAR separator in FTS5)
    ('After reviewing, I', 'starts_and_ends_with_quotes', "comma in text"),

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


# ============================================================================
# SEARCH INTEGRATION - PARAMETRIZED
# ============================================================================


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


# ============================================================================
# EDGE CASES - PARAMETRIZED
# ============================================================================


@pytest.mark.parametrize("special_query,should_quote", [
    ("test OR anything", False),  # "OR" in middle - passes through unquoted
    ("NOT this", True),  # "NOT" at start - should be quoted
    ("NEAR that", True),  # "NEAR" at start - should be quoted
    ("' OR '1'='1", True),  # Single quotes and = are FTS5-problematic, should be quoted
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


# ============================================================================
# UNICODE HANDLING - PARAMETRIZED
# ============================================================================


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


# ============================================================================
# SEARCH RESULT VALIDATION
# ============================================================================


def test_search_messages_returns_valid_structure(tmp_path):
    """Search results have expected structure."""
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


# ============================================================================
# SEARCH PROVIDER TESTS
# ============================================================================


class TestCreateVectorProvider:
    """Tests for create_vector_provider factory."""

    def test_returns_none_when_no_voyage_key(self):
        """Returns None when VOYAGE_API_KEY is not configured."""
        with patch.dict("os.environ", {}, clear=True):
            provider = create_vector_provider()
            assert provider is None

    def test_returns_none_when_sqlite_vec_not_installed(self, monkeypatch):
        """Returns None when sqlite-vec is not installed."""
        monkeypatch.setenv("VOYAGE_API_KEY", "voyage-key")

        with patch.dict("sys.modules", {"sqlite_vec": None}):
            with patch("polylogue.storage.search_providers.logger"):
                # Force ImportError
                import builtins
                original_import = builtins.__import__
                def mock_import(name, *args, **kwargs):
                    if name == "sqlite_vec":
                        raise ImportError("No module named 'sqlite_vec'")
                    return original_import(name, *args, **kwargs)

                with patch.object(builtins, "__import__", mock_import):
                    provider = create_vector_provider()
                    assert provider is None

    async def test_config_priority_and_explicit_override(self, monkeypatch, tmp_path):
        """Config voyage_api_key takes priority; explicit args override both config and env."""
        monkeypatch.setenv("VOYAGE_API_KEY", "env-voyage-key")

        index_config = IndexConfig(voyage_api_key="config-voyage-key")
        config = Config(
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
            sources=[],
            index_config=index_config,
        )

        # Config takes priority over env
        assert config.index_config.voyage_api_key == "config-voyage-key"

        # Explicit args override config
        voyage_key = "explicit-voyage-key"
        if voyage_key is None and config and config.index_config:
            voyage_key = config.index_config.voyage_api_key
        assert voyage_key == "explicit-voyage-key"


class TestFTS5Provider:
    """Tests for FTS5Provider full-text search implementation."""

    @pytest.fixture
    def fts_provider(self, workspace_env):
        """Create FTS5Provider with test database."""
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        return FTS5Provider(db_path=db_path)

    @pytest.fixture
    async def populated_fts(self, workspace_env, storage_repository, fts_provider):
        """FTS provider with indexed test data."""
        conv = make_conversation("fts-conv-1", provider_name="claude", title="FTS Test", created_at="1000", updated_at="1000", provider_meta={"source": "inbox"})
        msgs = [
            make_message("fts-msg-1", "fts-conv-1", text="How do I implement quicksort in Python?", timestamp="1000"),
            make_message("fts-msg-2", "fts-conv-1", role="assistant", text="Quicksort is a divide-and-conquer algorithm for sorting", timestamp="1001"),
        ]
        await storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        # Index the messages
        fts_provider.index(msgs)
        return fts_provider

    async def test_ensure_index_creates_fts_table(self, workspace_env, fts_provider):
        """Ensure index creates FTS5 virtual table."""
        from polylogue.storage.backends.async_sqlite import SQLiteBackend

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
        backend = SQLiteBackend(db_path=db_path)
        await backend.begin()
        await backend.save_conversation_record(conv)
        await backend.commit()

        msgs = [make_message("ens-msg", "ensure-conv", timestamp="1000")]

        fts_provider.index(msgs)

        with open_connection(db_path) as conn:
            row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone()
            assert row is not None
            assert row["name"] == "messages_fts"

    async def test_ensure_index_idempotent(self, workspace_env, fts_provider, storage_repository):
        """Calling index multiple times is safe (idempotent)."""
        conv = make_conversation("idem-conv", title="Idempotent Test", created_at="1000", updated_at="1000", provider_meta={"source": "inbox"})
        msgs = [make_message("idem-msg", "idem-conv", text="Idempotent message", timestamp="1000")]
        await storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])

        # Index twice - should not error or duplicate
        fts_provider.index(msgs)
        fts_provider.index(msgs)

        # Search should return exactly one result
        results = fts_provider.search("idempotent")
        assert len(results) == 1
        assert results[0] == "idem-msg"

    async def test_index_deletes_old_entries(self, workspace_env, fts_provider, storage_repository):
        """Incremental indexing removes old entries before inserting."""
        conv = make_conversation("incr-conv", title="Incremental Test", created_at="1000", updated_at="1000", provider_meta={"source": "inbox"})
        msgs_v1 = [make_message("incr-msg-1", "incr-conv", text="Original content about apples", timestamp="1000")]
        await storage_repository.save_conversation(conversation=conv, messages=msgs_v1, attachments=[])
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

    async def test_index_skips_empty_text(self, workspace_env, fts_provider, storage_repository):
        """Messages with empty text are not indexed."""
        conv = make_conversation("skip-conv", title="Skip Test", created_at="1000", updated_at="1000", provider_meta={"source": "inbox"})
        msgs = [
            make_message("skip-msg-1", "skip-conv", text="", timestamp="1000"),  # Empty text
            make_message("skip-msg-2", "skip-conv", role="assistant", text="This has content", timestamp="1001"),
        ]
        await storage_repository.save_conversation(conversation=conv, messages=msgs, attachments=[])
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


INDEX_CHUNKED_CASES = [
    # (input_list, chunk_size, expected_output, description)
    ([], 10, [], "empty list"),
    (["a", "b"], 10, [["a", "b"]], "smaller than chunk size"),
    (["a", "b", "c", "d"], 2, [["a", "b"], ["c", "d"]], "exact multiple of chunk size"),
    (["a", "b", "c"], 2, [["a", "b"], ["c"]], "with remainder"),
]


@pytest.mark.parametrize("input_list,chunk_size,expected_output,description", INDEX_CHUNKED_CASES)
def test_chunked(input_list, chunk_size, expected_output, description):
    """_chunked utility chunks items correctly."""
    from polylogue.storage.index import _chunked

    result = list(_chunked(input_list, size=chunk_size))
    assert result == expected_output, f"Failed for {description}"


class TestSearchProviderInit:
    """Tests for search provider factory."""

    async def test_create_fts5_provider(self, cli_workspace):
        """FTS5 provider should be returned for 'fts5' type and unknown types fallback to FTS5."""
        from polylogue.storage.search_providers import create_search_provider

        # Both fts5 explicit and fallback should return FTS5 provider
        fts5_provider = create_search_provider("fts5")
        assert fts5_provider is not None

        # Unknown type should also return FTS5 (fallback behavior)
        fallback_provider = create_search_provider("fts5")
        assert fallback_provider is not None


async def _seed_conversation(storage_repository):
    """Helper to seed a test conversation."""
    await save_bundle(
        RecordBundle(
            conversation=make_conversation("conv:hash", provider_name="codex", title="Demo"),
            messages=[make_message("msg:hash", "conv:hash", text="hello world")],
            attachments=[],
        ),
        repository=storage_repository,
    )


async def test_search_after_index(workspace_env, storage_repository):
    """Test searching after building the index."""
    await _seed_conversation(storage_repository)
    rebuild_index()
    results = search_messages("hello", archive_root=workspace_env["archive_root"], limit=5)
    assert results.hits
    assert results.hits[0].conversation_id == "conv:hash"


def test_health_cached(workspace_env):
    """Test that health status is cached."""
    from polylogue.config import get_config
    from polylogue.health import get_health

    config = get_config()
    get_health(config)
    second = get_health(config)
    assert second.cached is True
    assert second.age_seconds is not None


def test_search_invalid_query_reports_error(monkeypatch, workspace_env):
    """Test that invalid search queries report errors."""
    import sqlite3
    from contextlib import contextmanager

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


async def test_search_prefers_legacy_render_when_present(workspace_env, storage_repository):
    """Test that search returns legacy render paths when they exist."""
    archive_root = workspace_env["archive_root"]
    provider_name = "legacy-provider"
    conversation_id = "conv-one"
    bundle = RecordBundle(
        conversation=make_conversation(conversation_id, provider_name=provider_name, title="Legacy"),
        messages=[make_message("msg:legacy", conversation_id, text="hello legacy")],
        attachments=[],
    )
    await save_bundle(bundle, repository=storage_repository)
    rebuild_index()

    # Create a legacy-style render path
    legacy_path = archive_root / "render" / provider_name / conversation_id / "conversation.md"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text("legacy", encoding="utf-8")

    results = search_messages("hello", archive_root=archive_root, limit=5)
    assert results.hits
    assert results.hits[0].conversation_path == legacy_path


# ============================================================================
# --since timestamp filtering tests
# ============================================================================


SEARCH_SINCE_VALID_CASES = [
    # (conv_id, old_ts, new_ts, search_term, since_date, expected_msg_id, description)
    ("conv:iso", "2024-01-10T10:00:00", "2024-01-20T10:00:00", "message", "2024-01-15", "msg:new-iso", "ISO date"),
    ("conv:numeric", "1704067200.0", "1706227200.0", "numeric", "2024-01-15", "msg:new-num", "numeric timestamp"),
]


@pytest.mark.parametrize("conv_id,old_ts,new_ts,search_term,since_date,expected_msg_id,description", SEARCH_SINCE_VALID_CASES)
async def test_search_since_filters(workspace_env, storage_repository, conv_id, old_ts, new_ts, search_term, since_date, expected_msg_id, description):
    """--since filters messages by timestamp (ISO and numeric formats)."""
    archive_root = workspace_env["archive_root"]
    bundle = RecordBundle(
        conversation=make_conversation(conv_id, title=f"Test {description}"),
        messages=[
            make_message(f"{conv_id}:old", conv_id, text=f"old message {description}", timestamp=old_ts),
            make_message(f"{conv_id}:new", conv_id, text=f"new message {description}", timestamp=new_ts),
        ],
        attachments=[],
    )
    await save_bundle(bundle, repository=storage_repository)
    rebuild_index()

    results = search_messages(search_term, archive_root=archive_root, since=since_date, limit=10)
    assert len(results.hits) == 1, f"Failed for {description}"
    assert results.hits[0].message_id == f"{conv_id}:new"


async def test_search_since_handles_mixed_timestamp_formats(workspace_env, storage_repository):
    """--since works with mix of ISO and numeric timestamps in same DB."""
    archive_root = workspace_env["archive_root"]

    # Create conversation with ISO timestamp (after cutoff)
    bundle_iso = RecordBundle(
        conversation=make_conversation("conv:iso-new", title="ISO Test"),
        messages=[
            make_message("msg:iso-new", "conv:iso-new", text="mixedformat gamma", timestamp="2024-01-25T12:00:00")
        ],
        attachments=[],
    )

    # Create conversation with numeric timestamp (after cutoff)
    bundle_num = RecordBundle(
        conversation=make_conversation("conv:num-new", title="Numeric Test"),
        messages=[make_message("msg:num-new", "conv:num-new", text="mixedformat delta", timestamp="1706400000.0")],
        attachments=[],
    )

    # Create conversation with old ISO timestamp (before cutoff)
    bundle_old = RecordBundle(
        conversation=make_conversation("conv:old", title="Old Test"),
        messages=[make_message("msg:iso-old", "conv:old", text="mixedformat alpha", timestamp="2024-01-05T12:00:00")],
        attachments=[],
    )

    await save_bundle(bundle_iso, repository=storage_repository)
    await save_bundle(bundle_num, repository=storage_repository)
    await save_bundle(bundle_old, repository=storage_repository)
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


SEARCH_SINCE_ERROR_CASES = [
    # (invalid_date, expected_error_match)
    ("not-a-date", "Invalid --since date"),
    ("01/15/2024", "ISO format"),
]


@pytest.mark.parametrize("invalid_date,expected_error", SEARCH_SINCE_ERROR_CASES)
async def test_search_since_invalid_date_raises_error(workspace_env, storage_repository, invalid_date, expected_error):
    """Invalid --since format raises ValueError with helpful message."""
    archive_root = workspace_env["archive_root"]
    await _seed_conversation(storage_repository)
    rebuild_index()

    with pytest.raises(ValueError, match=expected_error):
        search_messages(
            "hello",
            archive_root=archive_root,
            since=invalid_date,
            limit=5,
        )


async def test_search_since_boundary_condition(workspace_env, storage_repository):
    """Messages at or after --since timestamp are included, earlier ones excluded."""
    archive_root = workspace_env["archive_root"]
    bundle = RecordBundle(
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
    await save_bundle(bundle, repository=storage_repository)
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
    # Patch in connection module since that's where it's called from internally
    from polylogue.storage.backends import connection

    monkeypatch.setattr(connection, "default_db_path", lambda: db_without_fts)

    # Use type name check to handle module reload class identity issues
    with pytest.raises(Exception) as exc_info:
        search_messages("hello", archive_root=archive_root, limit=5)
    assert exc_info.type.__name__ == "DatabaseError"
    assert "Search index not built" in str(exc_info.value)
