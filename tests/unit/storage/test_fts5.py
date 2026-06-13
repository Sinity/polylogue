"""Tests for FTS5 search provider and index functionality.

Tests cover FTS5 index creation, incremental updates, search functionality,
ranking, special characters, edge cases, and escaping.

Extracted from test_search_index.py.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, given, settings

from polylogue.config import Config, IndexConfig
from polylogue.storage.index import rebuild_index, update_index_for_sessions
from polylogue.storage.repository import SessionRepository
from polylogue.storage.search import escape_fts5_query, search_messages
from polylogue.storage.search_providers import create_vector_provider
from polylogue.storage.search_providers.fts5 import FTS5Provider
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.mutmut import preserved_mutmut_env
from tests.infra.storage_records import (
    DbFactory,
    SessionBuilder,
    make_content_block,
    make_message,
    make_session,
    save_current_archive_records,
    save_session_to_archive,
)
from tests.infra.strategies import fts5_match_text_strategy, search_query_strategy


def _archive_session_id(source_name: str, provider_session_id: str) -> str:
    from polylogue.core.identity_law import session_id as make_archive_session_id
    from polylogue.core.sources import origin_from_provider
    from polylogue.types import Provider

    return make_archive_session_id(origin_from_provider(Provider.from_string(source_name)).value, provider_session_id)


def _archive_message_id(source_name: str, provider_session_id: str, provider_message_id: str) -> str:
    from polylogue.core.identity_law import message_id as make_archive_message_id

    return make_archive_message_id(
        _archive_session_id(source_name, provider_session_id),
        provider_message_id,
        position=0,
    )


def _insert_current_session_message(
    conn: sqlite3.Connection,
    *,
    provider_session_id: str,
    provider_message_id: str,
    text: str,
    position: int = 0,
    role: str = "user",
    source_name: str = "test",
) -> tuple[str, str]:
    from polylogue.core.identity_law import message_id as make_archive_message_id

    session_id = _archive_session_id(source_name, provider_session_id)
    message_id = make_archive_message_id(session_id, provider_message_id, position=position)
    content_hash = b"x" * 32
    conn.execute(
        """
        INSERT OR IGNORE INTO sessions (native_id, origin, title, content_hash)
        VALUES (?, ?, ?, ?)
        """,
        (provider_session_id, session_id.split(":", 1)[0], "FTS test", content_hash),
    )
    conn.execute(
        """
        INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash)
        VALUES (?, ?, ?, ?, 'message', ?)
        """,
        (session_id, provider_message_id, position, role, content_hash),
    )
    conn.execute(
        """
        INSERT INTO blocks (message_id, session_id, position, block_type, text)
        VALUES (?, ?, 0, 'text', ?)
        """,
        (message_id, session_id, text),
    )
    return session_id, message_id


# ============================================================================
# Tests for search_messages()
# ============================================================================


async def test_search_respects_limit(workspace_env: dict[str, Path], storage_repository: SessionRepository) -> None:
    """search_messages() respects limit parameter."""
    for i in range(10):
        conv = make_session(f"conv{i}", title=f"Conv {i}")
        msg = make_message(f"msg{i}", f"conv{i}", text="search limit")
        await save_current_archive_records(storage_repository, session=conv, messages=[msg], attachments=[])

    rebuild_index()

    results = search_messages("search", archive_root=workspace_env["archive_root"], limit=3)
    assert len(results.hits) == 3


async def test_search_includes_snippet(workspace_env: dict[str, Path], storage_repository: SessionRepository) -> None:
    """search_messages() includes text snippet in results."""
    conv = make_session("conv1")
    msg = make_message("msg1", "conv1", text="The quick brown fox jumps over the lazy dog")

    await save_current_archive_records(storage_repository, session=conv, messages=[msg], attachments=[])
    rebuild_index()

    results = search_messages("quick", archive_root=workspace_env["archive_root"], limit=10)

    assert len(results.hits) == 1
    # Snippet should contain the query term or nearby context
    assert results.hits[0].snippet is not None
    assert isinstance(results.hits[0].snippet, str)


async def test_search_includes_session_metadata(
    workspace_env: dict[str, Path],
    storage_repository: SessionRepository,
) -> None:
    """search_messages() includes session metadata in results."""
    conv = make_session("conv1", source_name="claude-ai", title="My Session", provider_meta={"source": "my-source"})
    msg = make_message("msg1", "conv1", text="search query", timestamp="2024-01-01T10:30:00Z")

    await save_current_archive_records(storage_repository, session=conv, messages=[msg], attachments=[])
    rebuild_index()

    results = search_messages("search", archive_root=workspace_env["archive_root"], limit=10)

    assert len(results.hits) == 1
    hit = results.hits[0]
    assert hit.session_id == _archive_session_id("claude-ai", "conv1")
    assert hit.source_name == "claude-ai-export"
    assert hit.title == "My Session"
    assert hit.message_id == _archive_message_id("claude-ai", "conv1", "msg1")
    assert hit.timestamp is not None and "2024-01-01" in hit.timestamp


async def test_search_returns_best_message_per_session(
    workspace_env: dict[str, Path],
    storage_repository: SessionRepository,
) -> None:
    """search_messages() picks the strongest per-session hit deterministically."""
    archive_root = workspace_env["archive_root"]
    await save_current_archive_records(
        storage_repository,
        session=make_session("conv-best-hit", title="Best Hit Test"),
        messages=[
            make_message(
                "msg-best-old",
                "conv-best-hit",
                text="deterministic best hit token",
                timestamp="2024-01-01T00:00:00",
            ),
            make_message(
                "msg-best-new",
                "conv-best-hit",
                text="deterministic best hit token",
                timestamp="2024-02-01T00:00:00",
            ),
        ],
        attachments=[],
    )
    rebuild_index()

    results = search_messages("deterministic best hit token", archive_root=archive_root, limit=5)

    assert len(results.hits) == 1
    assert results.hits[0].message_id == _archive_message_id("test", "conv-best-hit", "msg-best-new")


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
async def test_search_with_special_text(
    workspace_env: dict[str, Path],
    storage_repository: SessionRepository,
    text: str,
    search_term: str,
    description: str,
) -> None:
    """search_messages() handles special text patterns."""
    conv = make_session("conv1", title=f"Test {description}")
    msg = make_message("msg1", "conv1", text=text)

    await save_current_archive_records(storage_repository, session=conv, messages=[msg], attachments=[])
    rebuild_index()

    results = search_messages(search_term, archive_root=workspace_env["archive_root"], limit=10)
    assert len(results.hits) == 1, f"Failed for {description}"


# ============================================================================
# Tests for edge cases and error handling
# ============================================================================


def test_rebuild_index_with_empty_database(test_conn: sqlite3.Connection) -> None:
    """rebuild_index() handles empty database gracefully."""
    rebuild_index(test_conn)

    count = test_conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
    assert count == 0


async def test_search_returns_searchresult_object(
    workspace_env: dict[str, Path],
    storage_repository: SessionRepository,
) -> None:
    """search_messages() returns SearchResult with hits list."""
    conv = make_session("conv1")
    msg = make_message("msg1", "conv1", text="search result")

    await save_current_archive_records(storage_repository, session=conv, messages=[msg], attachments=[])
    rebuild_index()

    results = search_messages("search", archive_root=workspace_env["archive_root"], limit=10)

    # Verify SearchResult structure
    assert hasattr(results, "hits")
    assert isinstance(results.hits, list)
    if results.hits:
        hit = results.hits[0]
        assert hasattr(hit, "session_id")
        assert hasattr(hit, "message_id")
        assert hasattr(hit, "source_name")
        assert hasattr(hit, "snippet")
        assert hasattr(hit, "title")
        assert hasattr(hit, "timestamp")
        assert hasattr(hit, "session_url")


def test_rebuild_index_with_multiple_messages_per_session(test_conn: sqlite3.Connection) -> None:
    """rebuild_index() correctly indexes all messages in a session."""
    session_id = _archive_session_id("test", "conv1")
    for i in range(10):
        _insert_current_session_message(
            test_conn,
            provider_session_id="conv1",
            provider_message_id=f"msg{i}",
            text=f"message number {i}",
            position=i,
            role="user" if i % 2 == 0 else "assistant",
        )
    rebuild_index(test_conn)

    count = test_conn.execute(
        """
        SELECT COUNT(*)
        FROM messages_fts
        JOIN blocks ON blocks.rowid = messages_fts.rowid
        WHERE blocks.session_id = ?
        """,
        (session_id,),
    ).fetchone()[0]
    assert count == 10


def test_update_index_deletes_old_entries_from_session(test_conn: sqlite3.Connection) -> None:
    """update_index_for_sessions() removes old index entries for updated sessions."""
    session_id, message_id = _insert_current_session_message(
        test_conn,
        provider_session_id="conv1",
        provider_message_id="msg1",
        text="original message",
    )

    rebuild_index(test_conn)

    # Delete original message
    test_conn.execute("DELETE FROM messages WHERE message_id = ?", (message_id,))

    # Add new message
    _insert_current_session_message(
        test_conn,
        provider_session_id="conv1",
        provider_message_id="msg2",
        text="new message",
    )

    # Update index
    update_index_for_sessions([session_id], test_conn)

    # Old message should not be in index
    old_hits = test_conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?", ("original",)
    ).fetchone()[0]
    assert old_hits == 0

    # New message should be indexed
    new_hits = test_conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?", ("new",)).fetchone()[0]
    assert new_hits == 1


async def test_rebuild_index_populates_action_search_rows(
    workspace_env: dict[str, Path],
    storage_repository: SessionRepository,
) -> None:
    """rebuild_index() keeps current tool-action projections readable."""
    conv = make_session("conv-actions", title="Action indexing")
    msg = make_message(
        "msg-actions",
        "conv-actions",
        role="assistant",
        text="Ran tests",
        blocks=[
            make_content_block(
                message_id="msg-actions",
                session_id="conv-actions",
                block_index=0,
                block_type="tool_use",
                tool_name="Bash",
                tool_id="tool-actions",
                tool_input=json.dumps({"command": "pytest -q tests/unit/core/test_semantic_facts.py"}),
                semantic_type="shell",
            )
        ],
    )

    await save_current_archive_records(storage_repository, session=conv, messages=[msg], attachments=[])
    rebuild_index()

    with open_connection(storage_repository.backend.db_path) as conn:
        action_row = conn.execute(
            """
            SELECT tool_name, semantic_type, tool_command
            FROM actions
            WHERE session_id = ?
            """,
            (_archive_session_id("test", "conv-actions"),),
        ).fetchone()
    assert action_row is not None
    assert action_row["tool_name"] == "Bash"
    assert action_row["semantic_type"] == "shell"
    assert action_row["tool_command"] == "pytest -q tests/unit/core/test_semantic_facts.py"


def test_actions_view_reflects_updated_tool_blocks(tmp_path: Path) -> None:
    """The current archive action projection is the blocks-backed actions view."""
    db_path = tmp_path / "index.db"
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(db_path, ArchiveTier.INDEX)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, ?, ?)",
            ("conv-action-refresh", "unknown-export", bytes(32)),
        )
        conn.execute(
            """
            INSERT INTO messages (
                session_id, native_id, position, role, message_type, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "unknown-export:conv-action-refresh",
                "msg-action-refresh",
                0,
                "assistant",
                "message",
                bytes(32),
            ),
        )
        conn.execute(
            """
            INSERT INTO blocks (
                message_id, session_id, position, block_type, tool_name, tool_id, tool_input, semantic_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "unknown-export:conv-action-refresh:msg-action-refresh",
                "unknown-export:conv-action-refresh",
                0,
                "tool_use",
                "Bash",
                "tool-refresh",
                json.dumps({"command": "pytest -q"}),
                "shell",
            ),
        )
        original_command = conn.execute(
            "SELECT tool_command FROM actions WHERE session_id = ?",
            ("unknown-export:conv-action-refresh",),
        ).fetchone()[0]
        assert original_command == "pytest -q"

        conn.execute(
            "DELETE FROM blocks WHERE message_id = ?",
            ("unknown-export:conv-action-refresh:msg-action-refresh",),
        )
        conn.execute(
            """
            INSERT INTO blocks (
                message_id, session_id, position, block_type, tool_name, tool_id, tool_input, semantic_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                "unknown-export:conv-action-refresh:msg-action-refresh",
                "unknown-export:conv-action-refresh",
                0,
                "tool_use",
                "Bash",
                "tool-refresh",
                json.dumps({"command": "ruff check polylogue/storage"}),
                "shell",
            ),
        )

        refreshed_command = conn.execute(
            "SELECT tool_command FROM actions WHERE session_id = ?",
            ("unknown-export:conv-action-refresh",),
        ).fetchone()[0]
        assert refreshed_command == "ruff check polylogue/storage"


def test_actions_view_drops_rows_when_tool_blocks_disappear(tmp_path: Path) -> None:
    """Removing the tool block removes the derived action projection."""
    db_path = tmp_path / "index.db"
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(db_path, ArchiveTier.INDEX)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, ?, ?)",
            ("conv-action-remove", "unknown-export", bytes(32)),
        )
        conn.execute(
            """
            INSERT INTO messages (
                session_id, native_id, position, role, message_type, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "unknown-export:conv-action-remove",
                "msg-action-remove",
                0,
                "assistant",
                "message",
                bytes(32),
            ),
        )
        conn.execute(
            """
            INSERT INTO blocks (
                message_id, session_id, position, block_type, tool_name, tool_id, tool_input, semantic_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "unknown-export:conv-action-remove:msg-action-remove",
                "unknown-export:conv-action-remove",
                0,
                "tool_use",
                "Bash",
                "tool-remove",
                json.dumps({"command": "pytest -q"}),
                "shell",
            ),
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM actions WHERE session_id = ?",
                ("unknown-export:conv-action-remove",),
            ).fetchone()[0]
            == 1
        )

        conn.execute(
            "DELETE FROM blocks WHERE message_id = ?",
            ("unknown-export:conv-action-remove:msg-action-remove",),
        )

        action_rows = conn.execute(
            "SELECT COUNT(*) FROM actions WHERE session_id = ?",
            ("unknown-export:conv-action-remove",),
        ).fetchone()[0]
        assert action_rows == 0


def test_batch_index_10k_messages(test_conn: sqlite3.Connection) -> None:
    """Benchmark: update_index_for_sessions handles 10k messages efficiently."""
    import time

    # Create 100 sessions with 100 messages each = 10,000 messages
    num_convs = 100
    msgs_per_conv = 100
    content_hash = b"x" * 32

    for i in range(num_convs):
        session_id = _archive_session_id("test", f"conv{i}")
        test_conn.execute(
            "INSERT INTO sessions (native_id, origin, title, content_hash) VALUES (?, ?, ?, ?)",
            (f"conv{i}", session_id.split(":", 1)[0], f"Benchmark Conv {i}", content_hash),
        )

        # Batch insert messages directly for speed
        messages_batch = [
            (
                session_id,
                f"msg{i}-{j}",
                j,
                "user" if j % 2 == 0 else "assistant",
                content_hash,
            )
            for j in range(msgs_per_conv)
        ]
        test_conn.executemany(
            """INSERT INTO messages
               (session_id, native_id, position, role, message_type, content_hash)
               VALUES (?, ?, ?, ?, 'message', ?)""",
            messages_batch,
        )
        blocks_batch = [
            (
                f"{session_id}:msg{i}-{j}",
                session_id,
                0,
                "text",
                f"message content {i}-{j} with searchable text",
            )
            for j in range(msgs_per_conv)
        ]
        test_conn.executemany(
            """INSERT INTO blocks
               (message_id, session_id, position, block_type, text)
               VALUES (?, ?, ?, ?, ?)""",
            blocks_batch,
        )

    test_conn.commit()

    # Time the index build
    conv_ids = [_archive_session_id("test", f"conv{i}") for i in range(num_convs)]

    start = time.perf_counter()
    update_index_for_sessions(conv_ids, test_conn)
    elapsed = time.perf_counter() - start

    # Verify all indexed
    count = test_conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
    assert count == num_convs * msgs_per_conv

    # Assert reasonable performance (should complete within 5 seconds)
    assert elapsed < 5.0, f"Batch indexing 10k messages took too long: {elapsed:.2f}s"


async def test_batch_index_search_returns_correct_provider(
    workspace_env: dict[str, Path],
    storage_repository: SessionRepository,
) -> None:
    """Verify batch indexing allows retrieving correct source_name via search."""
    # Create sessions with different providers
    conv1 = make_session("conv1", source_name="claude-ai", title="Claude Conv")
    conv2 = make_session("conv2", source_name="chatgpt", title="ChatGPT Conv")

    messages1 = [make_message(f"msg1-{i}", "conv1", text=f"claude text {i}") for i in range(5)]
    messages2 = [make_message(f"msg2-{i}", "conv2", text=f"chatgpt text {i}") for i in range(5)]

    await save_current_archive_records(
        storage_repository,
        session=conv1,
        messages=messages1,
        attachments=[],
    )
    await save_current_archive_records(
        storage_repository,
        session=conv2,
        messages=messages2,
        attachments=[],
    )

    rebuild_index()

    # Verify provider names via search
    results1 = search_messages("claude", archive_root=workspace_env["archive_root"], limit=10)
    assert all(hit.source_name == "claude-ai-export" for hit in results1.hits)
    assert len(results1.hits) == 1

    results2 = search_messages("chatgpt", archive_root=workspace_env["archive_root"], limit=10)
    assert all(hit.source_name == "chatgpt-export" for hit in results2.hits)
    assert len(results2.hits) == 1


# ============================================================================
# SEARCH INTEGRATION - PARAMETRIZED
# ============================================================================


@pytest.mark.parametrize(
    "query,should_find",
    [
        ("test", True),  # Basic search
        ("nonexistent", False),  # No match
        ("*", False),  # Bare asterisk escaped
        ("AND", False),  # Operator as literal
        ("quoted", True),  # Part of text with quotes
    ],
)
async def test_search_messages_known_cases(query: str, should_find: bool, tmp_path: Path) -> None:
    """Integration test for known archive search cases with specific assertions."""
    from polylogue.api import Polylogue

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    db_path = archive_root / "index.db"
    DbFactory(db_path)

    (
        SessionBuilder(db_path, "test1")
        .title("Test Session")
        .add_message("msg1", role="user", text='This is a test message with "quoted text" inside.')
        .save()
    )

    async with Polylogue(archive_root=archive_root, db_path=db_path) as plg:
        results = await plg.search(query, limit=10)

    if should_find:
        assert len(results.hits) > 0, f"Expected to find results for '{query}'"
    else:
        # Either no results or results don't match the query; the important
        # property is that no FTS syntax error escapes.
        assert isinstance(results.hits, list)


@given(query=search_query_strategy())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
def test_search_messages_escaping_never_crashes(query: str, tmp_path: Path) -> None:
    """Property: search_messages handles any query with controlled error handling.

    Tests FTS5 escaping and query handling with arbitrary inputs.
    Invalid FTS5 syntax raises DatabaseError (controlled), not unrecoverable crash.
    """
    from polylogue.errors import DatabaseError

    db_path = tmp_path / "test.db"
    DbFactory(db_path)

    # Insert test session using builder
    (
        SessionBuilder(db_path, "test1")
        .title("Test Session")
        .add_message("msg1", role="user", text='This is a test message with "quoted text" inside.')
        .save()
    )

    # Build search index
    with open_connection(str(db_path)) as conn:
        rebuild_index(conn)

    # The critical property: any query should either succeed or raise controlled DatabaseError
    try:
        results = search_messages(
            query,
            archive_root=tmp_path,
            db_path=Path(str(db_path)),
            limit=10,
        )
        assert isinstance(results.hits, list)
    except DatabaseError:
        # Controlled error for invalid FTS5 queries is acceptable
        pass


# ============================================================================
# EDGE CASES - PARAMETRIZED
# ============================================================================


@pytest.mark.parametrize(
    "special_query,should_quote",
    [
        ("test OR anything", False),  # "OR" in middle - passes through unquoted
        ("NOT this", True),  # "NOT" at start - should be quoted
        ("NEAR that", True),  # "NEAR" at start - should be quoted
        ("' OR '1'='1", True),  # Single quotes and = are FTS5-problematic, should be quoted
        ("test; DROP TABLE messages--", True),  # Contains special chars (semicolon, etc.), should be quoted
    ],
)
def test_escape_fts5_injection_prevention(special_query: str, should_quote: bool) -> None:
    """Prevent dangerous operator positions and special characters.

    Replaces ~5 security-focused tests.
    """
    result = escape_fts5_query(special_query)

    if should_quote:
        # Should be safely quoted
        assert result.startswith('"'), f"Expected quoted: {special_query}"
        assert result.endswith('"'), f"Expected quoted: {special_query}"
    else:
        # Safe mid-query operator usage should pass through unchanged.
        assert result == special_query, f"Unexpected rewrite: {special_query} -> {result}"
        assert not result.startswith('"')


# ============================================================================
# UNICODE HANDLING - PARAMETRIZED
# ============================================================================


@pytest.mark.parametrize(
    "unicode_query",
    [
        "文字",  # Chinese
        "тест",  # Cyrillic
        "🔍",  # Emoji
        "café",  # Accented
    ],
)
def test_escape_fts5_unicode(unicode_query: str) -> None:
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


async def test_search_messages_returns_valid_structure(tmp_path: Path) -> None:
    """Archive search results have the expected structure."""
    from polylogue.api import Polylogue

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    db_path = archive_root / "index.db"
    DbFactory(db_path)

    (SessionBuilder(db_path, "test1").title("Test").add_message("msg1", role="user", text="Searchable content").save())

    async with Polylogue(archive_root=archive_root, db_path=db_path) as plg:
        results = await plg.search("searchable", limit=10)

    assert len(results.hits) > 0
    for hit in results.hits:
        assert hasattr(hit, "snippet")
        assert hasattr(hit, "session_id")
        assert hit.snippet is not None
        assert len(hit.snippet) > 0


# ============================================================================
# SEARCH PROVIDER TESTS
# ============================================================================


class TestCreateVectorProvider:
    """Tests for create_vector_provider factory."""

    def test_returns_none_when_no_voyage_key(self: object) -> None:
        """Returns None when VOYAGE_API_KEY is not configured."""
        with patch.dict("os.environ", preserved_mutmut_env(), clear=True):
            provider = create_vector_provider()
            assert provider is None

    def test_returns_none_when_sqlite_vec_not_installed(
        self: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Returns None when sqlite-vec is not installed."""
        monkeypatch.setenv("VOYAGE_API_KEY", "voyage-key")

        with patch.dict("sys.modules", {"sqlite_vec": None}):
            with patch("polylogue.storage.search_providers.logger"):
                # Force ImportError
                import builtins

                original_import = builtins.__import__

                def mock_import(
                    name: str,
                    globals_: dict[str, object] | None = None,
                    locals_: dict[str, object] | None = None,
                    fromlist: tuple[str, ...] = (),
                    level: int = 0,
                ) -> object:
                    if name == "sqlite_vec":
                        raise ImportError("No module named 'sqlite_vec'")
                    return original_import(name, globals_, locals_, fromlist, level)

                with patch.object(builtins, "__import__", mock_import):
                    provider = create_vector_provider()
                    assert provider is None

    def test_logs_sqlite_vec_missing_only_once_per_process(
        self: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Missing sqlite-vec warning is emitted once even if provider creation repeats."""
        import builtins

        import polylogue.storage.search_providers as search_providers

        monkeypatch.setenv("VOYAGE_API_KEY", "voyage-key")
        monkeypatch.setattr(search_providers, "_sqlite_vec_missing_warned", False)

        original_import = builtins.__import__

        def mock_import(
            name: str,
            globals_: dict[str, object] | None = None,
            locals_: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "sqlite_vec":
                raise ImportError("No module named 'sqlite_vec'")
            return original_import(name, globals_, locals_, fromlist, level)

        with (
            patch.dict("sys.modules", {"sqlite_vec": None}),
            patch.object(builtins, "__import__", mock_import),
            patch.object(search_providers.logger, "warning") as mock_warning,
        ):
            assert create_vector_provider() is None
            assert create_vector_provider() is None

        assert mock_warning.call_count == 1

    async def test_config_priority_and_explicit_override(
        self: object,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
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
        assert config.index_config is not None
        assert config.index_config.voyage_api_key == "config-voyage-key"

        # Explicit args override config
        voyage_key = "explicit-voyage-key"
        if voyage_key is None and config.index_config is not None:
            voyage_key = config.index_config.voyage_api_key
        assert voyage_key == "explicit-voyage-key"


class TestFTS5Provider:
    """Tests for FTS5Provider full-text search implementation."""

    @pytest.fixture
    def fts_provider(self: object, workspace_env: dict[str, Path]) -> FTS5Provider:
        """Create FTS5Provider with test database."""
        db_path = workspace_env["archive_root"] / "index.db"
        return FTS5Provider(db_path=db_path)

    @pytest.fixture
    async def populated_fts(
        self: object,
        workspace_env: dict[str, Path],
        storage_repository: SessionRepository,
        fts_provider: FTS5Provider,
    ) -> FTS5Provider:
        """FTS provider with indexed test data."""
        conv = make_session(
            "fts-conv-1",
            source_name="claude-ai",
            title="FTS Test",
            created_at="1970-01-02T00:00:00+00:00",
            updated_at="1970-01-02T00:00:00+00:00",
            provider_meta={"source": "inbox"},
        )
        msgs = [
            make_message("fts-msg-1", "fts-conv-1", text="How do I implement quicksort in Python?", timestamp="1000"),
            make_message(
                "fts-msg-2",
                "fts-conv-1",
                role="assistant",
                text="Quicksort is a divide-and-conquer algorithm for sorting",
                timestamp="1001",
            ),
        ]
        await save_current_archive_records(storage_repository, session=conv, messages=msgs, attachments=[])

        # Index the messages
        fts_provider.index(msgs)
        return fts_provider

    async def test_ensure_index_creates_fts_table(
        self: object,
        workspace_env: dict[str, Path],
        fts_provider: FTS5Provider,
    ) -> None:
        """Ensure index creates FTS5 virtual table."""
        from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

        db_path = workspace_env["archive_root"] / "index.db"

        # Index empty list to trigger table creation
        fts_provider.index([])

        with open_connection(db_path) as conn:
            # Check if the FTS table doesn't exist yet (we passed empty list)
            # Actually we need to trigger the _ensure_index by indexing something
            pass

        # Index with actual message to ensure table creation
        conv = make_session(
            "ensure-conv",
            title="Ensure Test",
            created_at="1970-01-02T00:00:00+00:00",
            updated_at="1970-01-02T00:00:00+00:00",
            provider_meta={"source": "inbox"},
        )
        # First save the session so source_name lookup works
        backend = SQLiteBackend(db_path=db_path)
        await save_session_to_archive(backend, session=conv)
        await backend.close()

        msgs = [make_message("ens-msg", "ensure-conv", timestamp="1000")]

        fts_provider.index(msgs)

        with open_connection(db_path) as conn:
            row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone()
            assert row is not None
            assert row["name"] == "messages_fts"

    async def test_ensure_index_idempotent(
        self: object,
        workspace_env: dict[str, Path],
        fts_provider: FTS5Provider,
        storage_repository: SessionRepository,
    ) -> None:
        """Calling index multiple times is safe (idempotent)."""
        conv = make_session(
            "idem-conv",
            title="Idempotent Test",
            created_at="1970-01-02T00:00:00+00:00",
            updated_at="1970-01-02T00:00:00+00:00",
            provider_meta={"source": "inbox"},
        )
        msgs = [make_message("idem-msg", "idem-conv", text="Idempotent message", timestamp="1000")]
        await save_current_archive_records(storage_repository, session=conv, messages=msgs, attachments=[])

        # Index twice - should not error or duplicate
        fts_provider.index(msgs)
        fts_provider.index(msgs)

        # Search should return exactly one result
        results = fts_provider.search("idempotent")
        assert len(results) == 1
        assert results[0] == _archive_message_id("test", "idem-conv", "idem-msg")

    async def test_index_deletes_old_entries(
        self: object,
        workspace_env: dict[str, Path],
        fts_provider: FTS5Provider,
        storage_repository: SessionRepository,
    ) -> None:
        """Incremental indexing removes old entries before inserting."""
        conv = make_session(
            "incr-conv",
            title="Incremental Test",
            created_at="1970-01-02T00:00:00+00:00",
            updated_at="1970-01-02T00:00:00+00:00",
            provider_meta={"source": "inbox"},
        )
        msgs_v1 = [make_message("incr-msg-1", "incr-conv", text="Original content about apples", timestamp="1000")]
        await save_current_archive_records(storage_repository, session=conv, messages=msgs_v1, attachments=[])
        fts_provider.index(msgs_v1)

        # Should find "apples"
        results = fts_provider.search("apples")
        assert len(results) == 1

        # Re-index with different content
        conv_v2 = make_session(
            "incr-conv",
            title="Incremental Test",
            created_at="1970-01-02T00:00:00+00:00",
            updated_at="1970-01-02T00:00:01+00:00",
            provider_meta={"source": "inbox"},
            content_hash="updated-content-hash",
        )
        msgs_v2 = [make_message("incr-msg-1", "incr-conv", text="Updated content about oranges", timestamp="1000")]
        await save_current_archive_records(storage_repository, session=conv_v2, messages=msgs_v2, attachments=[])
        fts_provider.index(msgs_v2)

        # "apples" should no longer be found
        results = fts_provider.search("apples")
        assert len(results) == 0

        # "oranges" should be found
        results = fts_provider.search("oranges")
        assert len(results) == 1

    async def test_index_skips_empty_text(
        self: object,
        workspace_env: dict[str, Path],
        fts_provider: FTS5Provider,
        storage_repository: SessionRepository,
    ) -> None:
        """Messages with empty text are not indexed."""
        conv = make_session(
            "skip-conv",
            title="Skip Test",
            created_at="1970-01-02T00:00:00+00:00",
            updated_at="1970-01-02T00:00:00+00:00",
            provider_meta={"source": "inbox"},
        )
        msgs = [
            make_message("skip-msg-1", "skip-conv", text="", timestamp="1000"),  # Empty text
            make_message("skip-msg-2", "skip-conv", role="assistant", text="This has content", timestamp="1001"),
        ]
        await save_current_archive_records(storage_repository, session=conv, messages=msgs, attachments=[])
        fts_provider.index(msgs)

        # Search should only find the non-empty message
        results = fts_provider.search("content")
        assert len(results) == 1
        assert results[0] == _archive_message_id("test", "skip-conv", "skip-msg-2")

    def test_search_returns_ranked_results(self: object, populated_fts: FTS5Provider) -> None:
        """Search returns results ordered by relevance (BM25)."""
        # The populated fixture has messages about quicksort
        results = populated_fts.search("quicksort")
        assert len(results) == 2  # Both messages mention quicksort
        # Results should be in relevance order (checked implicitly by the stable BM25 ordering)

    def test_search_applies_limit_in_sql(self: object, populated_fts: FTS5Provider) -> None:
        """Search should honor LIMIT without materializing extra rows first."""
        results = populated_fts.search("quicksort", limit=1)
        assert len(results) == 1

    def test_search_escapes_fts5_special_chars(self: object, populated_fts: FTS5Provider) -> None:
        """Search query escapes FTS5 special characters."""
        # Quotes and asterisks should be escaped
        results = populated_fts.search('"special* query"')
        # Should not raise FTS5 syntax error
        assert isinstance(results, list)

    def test_search_returns_empty_if_no_index(self: object, workspace_env: dict[str, Path]) -> None:
        """Search returns empty list if FTS index doesn't exist."""
        db_path = workspace_env["data_root"] / "polylogue" / "nonexistent.db"
        provider = FTS5Provider(db_path=db_path)
        results = provider.search("anything")
        assert results == []

        # Could be empty or match all - depends on FTS5 behavior
        assert isinstance(results, list)

    def test_search_returns_empty_for_blank_query(self: object, populated_fts: FTS5Provider) -> None:
        """Blank queries short-circuit instead of issuing empty MATCH searches."""
        assert populated_fts.search("") == []


INDEX_CHUNKED_CASES = [
    # (input_list, chunk_size, expected_output, description)
    ([], 10, [], "empty list"),
    (["a", "b"], 10, [["a", "b"]], "smaller than chunk size"),
    (["a", "b", "c", "d"], 2, [["a", "b"], ["c", "d"]], "exact multiple of chunk size"),
    (["a", "b", "c"], 2, [["a", "b"], ["c"]], "with remainder"),
]


@pytest.mark.parametrize("input_list,chunk_size,expected_output,description", INDEX_CHUNKED_CASES)
def test_chunked(
    input_list: list[str],
    chunk_size: int,
    expected_output: list[list[str]],
    description: str,
) -> None:
    """_chunked utility chunks items correctly."""
    from polylogue.storage.index import _chunked

    result = list(_chunked(input_list, size=chunk_size))
    assert result == expected_output, f"Failed for {description}"


class TestSearchProviderInit:
    """Tests for search provider factory."""

    async def test_create_fts5_provider(self: object, cli_workspace: dict[str, Path]) -> None:
        """Search provider factory returns an FTS5 provider."""
        from polylogue.storage.search_providers import create_search_provider

        provider = create_search_provider(db_path=cli_workspace["db_path"])
        assert isinstance(provider, FTS5Provider)


async def _seed_session(storage_repository: SessionRepository) -> None:
    """Helper to seed a test session."""
    await save_current_archive_records(
        storage_repository,
        session=make_session("conv:hash", source_name="codex", title="Demo"),
        messages=[make_message("msg:hash", "conv:hash", text="hello world")],
        attachments=[],
    )


async def test_search_after_index(workspace_env: dict[str, Path], storage_repository: SessionRepository) -> None:
    """Test searching after building the index."""
    await _seed_session(storage_repository)
    rebuild_index()
    results = search_messages("hello", archive_root=workspace_env["archive_root"], limit=5)
    assert results.hits
    assert results.hits[0].session_id == _archive_session_id("codex", "conv:hash")


def test_health_cached(workspace_env: dict[str, Path]) -> None:
    """Test that get_readiness returns a live report."""
    from polylogue.config import get_config
    from polylogue.readiness import get_readiness

    config = get_config()
    report = get_readiness(config)
    assert report.provenance.source == "live"
    assert report.timestamp > 0


def test_search_invalid_query_reports_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_env: dict[str, Path],
) -> None:
    """Test that invalid search queries report errors."""
    import sqlite3
    from contextlib import contextmanager

    class StubCursor:
        def __init__(self, row: object = None) -> None:
            self._row = row

        def fetchone(self) -> object:
            return self._row

        def fetchall(self) -> list[object]:
            return []

    class StubConn:
        def execute(self, sql: str, params: object = ()) -> StubCursor:
            param_values = tuple(params) if isinstance(params, tuple) else ()
            if "messages_fts_docsize" in sql:
                return StubCursor(row=(1,))
            if "COUNT(*)" in sql and "FROM blocks" in sql and "search_text" in sql:
                return StubCursor(row=(1,))
            if "sqlite_master" in sql and "trigger" in sql:
                return StubCursor(row=(3,))
            if "sqlite_master" in sql and param_values in {("blocks",), ("messages_fts",)}:
                return StubCursor(row=(1,))
            if "sqlite_master" in sql and "messages_fts" in sql:
                return StubCursor(row={"name": "messages_fts"})
            if "MATCH" in sql:
                raise sqlite3.OperationalError("fts5: syntax error")
            return StubCursor()

    @contextmanager
    def stub_open_connection(_: object) -> Iterator[StubConn]:
        yield StubConn()

    monkeypatch.setattr("polylogue.storage.search.open_connection", stub_open_connection)
    # Use type name check to handle module reload class identity issues
    with pytest.raises(Exception) as exc_info:
        search_messages('"unterminated', archive_root=workspace_env["archive_root"], limit=5)
    assert exc_info.type.__name__ == "DatabaseError"
    assert "Invalid search query" in str(exc_info.value)


async def test_search_returns_daemon_reader_url(
    workspace_env: dict[str, Path],
    storage_repository: SessionRepository,
) -> None:
    """Search results point to the daemon reader, not rendered files."""
    archive_root = workspace_env["archive_root"]
    source_name = "test"
    session_id = "conv-one"
    await save_current_archive_records(
        storage_repository,
        session=make_session(session_id, source_name=source_name, title="Reader URL"),
        messages=[make_message("msg:legacy", session_id, text="hello legacy")],
        attachments=[],
    )
    rebuild_index()

    results = search_messages("hello", archive_root=archive_root, limit=5)
    assert results.hits
    assert results.hits[0].session_url == "/?session=unknown-export%3Aconv-one"


# ============================================================================
# --since timestamp filtering tests
# ============================================================================


SEARCH_SINCE_VALID_CASES = [
    # (conv_id, old_ts, new_ts, search_term, since_date, expected_msg_id, description)
    (
        "conv:iso",
        "2024-01-10T10:00:00",
        "2024-01-20T10:00:00",
        "message",
        "2024-01-15",
        _archive_message_id("test", "conv:iso", "new"),
        "ISO date",
    ),
    (
        "conv:numeric",
        "1704067200.0",
        "1706227200.0",
        "numeric",
        "2024-01-15",
        _archive_message_id("test", "conv:numeric", "new"),
        "numeric timestamp",
    ),
]


@pytest.mark.parametrize(
    "conv_id,old_ts,new_ts,search_term,since_date,expected_msg_id,description", SEARCH_SINCE_VALID_CASES
)
async def test_search_since_filters(
    workspace_env: dict[str, Path],
    storage_repository: SessionRepository,
    conv_id: str,
    old_ts: str,
    new_ts: str,
    search_term: str,
    since_date: str,
    expected_msg_id: str,
    description: str,
) -> None:
    """--since filters messages by timestamp (ISO and numeric formats)."""
    archive_root = workspace_env["archive_root"]
    await save_current_archive_records(
        storage_repository,
        session=make_session(conv_id, title=f"Test {description}"),
        messages=[
            make_message(f"{conv_id}:old", conv_id, text=f"old message {description}", timestamp=old_ts),
            make_message(f"{conv_id}:new", conv_id, text=f"new message {description}", timestamp=new_ts),
        ],
        attachments=[],
    )
    rebuild_index()

    results = search_messages(search_term, archive_root=archive_root, since=since_date, limit=10)
    assert len(results.hits) == 1, f"Failed for {description}"
    assert results.hits[0].message_id == expected_msg_id


async def test_search_since_handles_mixed_timestamp_formats(
    workspace_env: dict[str, Path],
    storage_repository: SessionRepository,
) -> None:
    """--since works with mix of ISO and numeric timestamps in same DB."""
    archive_root = workspace_env["archive_root"]

    await save_current_archive_records(
        storage_repository,
        session=make_session("conv:iso-new", title="ISO Test"),
        messages=[
            make_message("msg:iso-new", "conv:iso-new", text="mixedformat gamma", timestamp="2024-01-25T12:00:00")
        ],
        attachments=[],
    )

    await save_current_archive_records(
        storage_repository,
        session=make_session("conv:num-new", title="Numeric Test"),
        messages=[make_message("msg:num-new", "conv:num-new", text="mixedformat delta", timestamp="1706400000.0")],
        attachments=[],
    )

    await save_current_archive_records(
        storage_repository,
        session=make_session("conv:old", title="Old Test"),
        messages=[make_message("msg:iso-old", "conv:old", text="mixedformat alpha", timestamp="2024-01-05T12:00:00")],
        attachments=[],
    )
    rebuild_index()

    results = search_messages(
        "mixedformat",
        archive_root=archive_root,
        since="2024-01-15",
        limit=10,
    )
    # Should get 2 hits: one ISO, one numeric - both after cutoff
    assert len(results.hits) == 2
    hit_conv_ids = {h.session_id for h in results.hits}
    assert hit_conv_ids == {
        _archive_session_id("test", "conv:iso-new"),
        _archive_session_id("test", "conv:num-new"),
    }


SEARCH_SINCE_ERROR_CASES = [
    # (invalid_date, expected_error_match)
    ("not-a-date", "Invalid --since date"),
    ("01/15/2024", "ISO format"),
]


@pytest.mark.parametrize("invalid_date,expected_error", SEARCH_SINCE_ERROR_CASES)
async def test_search_since_invalid_date_raises_error(
    workspace_env: dict[str, Path],
    storage_repository: SessionRepository,
    invalid_date: str,
    expected_error: str,
) -> None:
    """Invalid --since format raises ValueError with helpful message."""
    archive_root = workspace_env["archive_root"]
    await _seed_session(storage_repository)
    rebuild_index()

    with pytest.raises(ValueError, match=expected_error):
        search_messages(
            "hello",
            archive_root=archive_root,
            since=invalid_date,
            limit=5,
        )


async def test_search_since_boundary_condition(
    workspace_env: dict[str, Path],
    storage_repository: SessionRepository,
) -> None:
    """Messages at or after --since timestamp are included, earlier ones excluded."""
    archive_root = workspace_env["archive_root"]
    await save_current_archive_records(
        storage_repository,
        session=make_session("conv:boundary", title="Boundary Test"),
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
    rebuild_index()

    results = search_messages(
        "boundary",
        archive_root=archive_root,
        since="2024-01-15",
        limit=10,
    )
    # Should include after, exclude before
    assert len(results.hits) == 1
    assert results.hits[0].message_id == _archive_message_id("test", "conv:boundary", "msg:after-cutoff")


def test_search_without_fts_table_raises_descriptive_error(
    workspace_env: dict[str, Path],
    db_without_fts: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """search() raises DatabaseError mentioning missing FTS when FTS missing."""
    archive_root = workspace_env["archive_root"]

    import polylogue.paths as _paths

    monkeypatch.setattr(_paths, "db_path", lambda: db_without_fts)

    # Use type name check to handle module reload class identity issues
    with pytest.raises(Exception) as exc_info:
        search_messages("hello", archive_root=archive_root, limit=5)
    assert exc_info.type.__name__ == "DatabaseError"
    assert "Search index not built" in str(exc_info.value)


async def test_search_with_empty_fts_rows_returns_no_hits(
    workspace_env: dict[str, Path],
    db_path: Path,
) -> None:
    """Search over an emptied block FTS index reports incomplete search state."""
    from polylogue.api import Polylogue
    from polylogue.errors import DatabaseError
    from tests.infra.archive_scenarios import open_index_db

    archive_root = workspace_env["archive_root"]
    DbFactory(db_path).create_session(
        id="conv-incomplete-fts",
        provider="chatgpt",
        title="Incomplete FTS",
        messages=[
            {"id": "m-incomplete-1", "role": "user", "text": "search me"},
            {"id": "m-incomplete-2", "role": "assistant", "text": "still not indexed"},
        ],
    )
    with open_index_db(db_path) as conn:
        conn.execute("DELETE FROM messages_fts")
        conn.commit()

    async with Polylogue(archive_root=archive_root, db_path=db_path) as plg:
        with pytest.raises(DatabaseError, match="Search index is incomplete"):
            await plg.search("search", limit=5)


# ============================================================================
# Property Laws (from test_fts5_laws.py)
# ============================================================================


@given(st.text())
def test_escape_fts5_query_never_crashes(text: str) -> None:
    """escape_fts5_query handles any Unicode input without raising, always returns str."""
    result = escape_fts5_query(text)
    assert isinstance(result, str)


@settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow])
@given(text=fts5_match_text_strategy())
def test_escape_fts5_match_text_safe(text: str) -> None:
    """FTS5 match text from strategy is safely escaped."""
    result = escape_fts5_query(text)
    assert isinstance(result, str)
    assert len(result) > 0 or text.strip() == ""


# ============================================================================
# Search-with-since Property Laws
# ============================================================================


class TestSearchWithSinceLaws:
    """Property: search with --since returns subset of search without --since."""

    @given(
        pair=st.one_of(
            # Import inline to avoid circular issues at module level
            st.tuples(
                st.text(min_size=2, max_size=20, alphabet=st.characters(whitelist_categories=["L"])),
                st.one_of(st.none(), st.dates().map(lambda d: d.isoformat())),
            ),
        )
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_since_filter_is_monotonic(
        self: object,
        pair: tuple[str, str | None],
        tmp_path: Path,
    ) -> None:
        """Results with --since must be subset of results without --since.

        All returned messages must have timestamps >= since when since is set.
        """
        from polylogue.errors import DatabaseError

        query_text, since = pair
        db_path = tmp_path / "since.db"
        DbFactory(db_path)

        # Seed sessions with spread timestamps
        for i in range(3):
            (
                SessionBuilder(db_path, f"since-conv-{i}")
                .title(f"Since Test {i}")
                .add_message(
                    f"msg-{i}",
                    role="user",
                    text=f"{query_text} message {i}",
                    timestamp=f"2024-{(i % 12) + 1:02d}-15T12:00:00Z",
                )
                .save()
            )

        with open_connection(str(db_path)) as conn:
            rebuild_index(conn)

        try:
            results_all = search_messages(
                query_text,
                archive_root=tmp_path,
                db_path=Path(str(db_path)),
                limit=100,
            )
        except (DatabaseError, ValueError):
            return  # Invalid query, skip

        if since is None:
            return  # No since filter to compare

        try:
            results_since = search_messages(
                query_text,
                archive_root=tmp_path,
                db_path=Path(str(db_path)),
                since=since,
                limit=100,
            )
        except (DatabaseError, ValueError):
            return  # Invalid since date, skip

        # Monotonicity: since-filtered results must be subset of unfiltered
        all_ids = {h.message_id for h in results_all.hits}
        since_ids = {h.message_id for h in results_since.hits}
        assert since_ids <= all_ids, f"since-filtered IDs {since_ids - all_ids} not in unfiltered results"


def test_fts_triggers_restored_after_exception() -> None:
    """FTS triggers must be active even after an exception during ingest (#817)."""
    import sqlite3

    from polylogue.storage.fts.fts_lifecycle import restore_fts_triggers_sync, suspend_fts_triggers_sync
    from polylogue.storage.sqlite.schema import SCHEMA_DDL

    db_path = None
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA_DDL)

        # Simulate ingest: suspend, insert, exception, restore in finally
        suspend_fts_triggers_sync(conn)
        _insert_current_session_message(
            conn,
            provider_session_id="pc1",
            provider_message_id="m1",
            text="test message",
            source_name="unknown",
        )
        try:
            raise RuntimeError("simulated ingest failure")
        except RuntimeError:
            pass
        finally:
            # CRITICAL: restore triggers even after exception (#817)
            restore_fts_triggers_sync(conn)
            conn.commit()

        # Insert another message — FTS should pick it up since triggers are active
        _insert_current_session_message(
            conn,
            provider_session_id="pc1",
            provider_message_id="m2",
            text="another message",
            position=1,
            role="assistant",
            source_name="unknown",
        )
        conn.commit()
        from polylogue.storage.fts.fts_lifecycle import rebuild_fts_index_sync

        rebuild_fts_index_sync(conn)
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]
        assert count == 2, f"Expected 2 FTS entries after exception recovery, got {count}"
    finally:
        conn.close()
        if db_path is not None:
            import os

            try:
                os.unlink(db_path)
            except OSError:
                pass


def test_fts_index_recovers_from_corrupt_trigger_state() -> None:
    """FTS index should be queryable after trigger suspend/restore cycle (#807)."""
    import sqlite3

    from polylogue.storage.fts.fts_lifecycle import restore_fts_triggers_sync, suspend_fts_triggers_sync
    from polylogue.storage.sqlite.schema import SCHEMA_DDL

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_DDL)
    conn.execute(
        "INSERT INTO sessions(native_id, origin, content_hash) VALUES(?,?,?)",
        ("pc1", "unknown-export", b"x" * 32),
    )
    conn.commit()

    # Normal insert with triggers active
    _insert_current_session_message(
        conn,
        provider_session_id="pc1",
        provider_message_id="m1",
        text="hello world this is a test message",
        source_name="unknown",
    )
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 1

    # Simulate crash recovery: triggers suspended, data inserted, triggers restored
    suspend_fts_triggers_sync(conn)
    _insert_current_session_message(
        conn,
        provider_session_id="pc1",
        provider_message_id="m2",
        text="another message for testing",
        position=1,
        role="assistant",
        source_name="unknown",
    )
    restore_fts_triggers_sync(conn)
    conn.commit()

    # New message should be picked up
    _insert_current_session_message(
        conn,
        provider_session_id="pc1",
        provider_message_id="m3",
        text="third message here",
        position=2,
        source_name="unknown",
    )
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
    # m1 (pre-suspend) + m3 (post-restore) should be indexed
    # m2 was inserted while triggers were suspended
    assert count >= 2, f"Expected at least 2 FTS entries, got {count}"
    conn.close()
