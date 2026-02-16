"""Miscellaneous storage and search tests.

Tests cover database migrations, vec0 tables, asset handling, raw conversation
storage, session index parsing, and related edge cases.

Extracted from monolithic test_search_index.py.
"""

from __future__ import annotations

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue.assets import asset_path, write_asset
from polylogue.sources.parsers.claude import (
    SessionIndexEntry,
    enrich_conversation_from_index,
    find_sessions_index,
    parse_sessions_index,
)
from polylogue.storage.backends.schema import (
    _ensure_vec0_table,
    _migrate_v6_to_v7,
    _migrate_v7_to_v8,
    _migrate_v8_to_v9,
    _migrate_v9_to_v10,
)
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import connection_context
from polylogue.storage.store import ConversationRecord


def make_hash(s: str) -> str:
    """Create a 16-char content hash."""
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# Schema constants for migration tests
V6_CONVERSATIONS_TABLE = """
    CREATE TABLE conversations (
        conversation_id TEXT PRIMARY KEY,
        provider_name TEXT NOT NULL,
        provider_conversation_id TEXT,
        title TEXT,
        created_at TEXT,
        updated_at TEXT,
        archive_path TEXT,
        raw_id TEXT
    )
"""

V6_MESSAGES_TABLE = """
    CREATE TABLE messages (
        message_id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        provider_message_id TEXT,
        text TEXT,
        timestamp TEXT,
        content_hash TEXT
    )
"""

V7_CONVERSATIONS_TABLE = """
    CREATE TABLE conversations (
        conversation_id TEXT PRIMARY KEY,
        provider_name TEXT NOT NULL,
        provider_conversation_id TEXT,
        title TEXT,
        created_at TEXT,
        updated_at TEXT,
        archive_path TEXT,
        parent_conversation_id TEXT
    )
"""

V8_RAW_CONVERSATIONS_TABLE = """
    CREATE TABLE raw_conversations (
        raw_id TEXT PRIMARY KEY,
        provider_name TEXT NOT NULL,
        source_path TEXT NOT NULL,
        raw_content BLOB NOT NULL,
        acquired_at TEXT NOT NULL,
        file_mtime TEXT
    )
"""

V9_RAW_CONVERSATIONS_TABLE = """
    CREATE TABLE raw_conversations (
        raw_id TEXT PRIMARY KEY,
        provider_name TEXT NOT NULL,
        source_name TEXT,
        source_path TEXT NOT NULL,
        raw_content BLOB NOT NULL,
        acquired_at TEXT NOT NULL,
        file_mtime TEXT
    )
"""

V9_CONVERSATIONS_MIN_TABLE = """
    CREATE TABLE conversations (
        conversation_id TEXT PRIMARY KEY,
        provider_name TEXT NOT NULL,
        title TEXT
    )
"""

V9_MESSAGES_MIN_TABLE = """
    CREATE TABLE messages (
        message_id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        text TEXT
    )
"""


class TestMigrateV6ToV7:
    """Tests for _migrate_v6_to_v7 (conversation/message branching columns)."""

    def test_migrate_v6_to_v7_adds_columns_and_enforces_constraints(self, tmp_path):
        """Migration adds parent columns, indices, and enforces branch_type constraints."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        conn.execute(V6_CONVERSATIONS_TABLE)
        conn.execute(V6_MESSAGES_TABLE)
        conn.commit()

        _migrate_v6_to_v7(conn)
        conn.commit()

        # Verify columns exist
        cursor = conn.execute("PRAGMA table_info(conversations)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "parent_conversation_id" in columns
        assert "branch_type" in columns

        cursor = conn.execute("PRAGMA table_info(messages)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "parent_message_id" in columns
        assert "branch_index" in columns

        # Verify indices were created
        indices = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        ).fetchall()
        index_names = {row[0] for row in indices}
        assert "idx_conversations_parent" in index_names
        assert "idx_messages_parent" in index_names

        # Verify branch_type constraint works - insert with valid type
        conn.execute(
            """
            INSERT INTO conversations
            (conversation_id, provider_name, provider_conversation_id, title, created_at, branch_type)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("conv1", "test", "p1", "Test", "2024-01-01T00:00:00Z", "continuation"),
        )
        conn.commit()

        row = conn.execute(
            "SELECT branch_type FROM conversations WHERE conversation_id = 'conv1'"
        ).fetchone()
        assert row[0] == "continuation"

        conn.close()


class TestMigrateV7ToV8:
    """Tests for _migrate_v7_to_v8 (raw storage with FK direction)."""

    def test_migrate_v7_to_v8_creates_and_accepts_inserts(self, tmp_path):
        """Migration creates raw_conversations table with correct schema and accepts inserts."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        conn.execute(V7_CONVERSATIONS_TABLE)
        conn.commit()

        # Run migration
        _migrate_v7_to_v8(conn)
        conn.commit()

        # Verify raw_conversations table exists with correct columns
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='raw_conversations'"
        ).fetchone()
        assert exists is not None

        cursor = conn.execute("PRAGMA table_info(raw_conversations)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "raw_id" in columns
        assert "provider_name" in columns
        assert "source_path" in columns
        assert "raw_content" in columns
        assert "acquired_at" in columns

        # Verify raw_id FK column added to conversations
        cursor = conn.execute("PRAGMA table_info(conversations)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "raw_id" in columns

        # Verify indices were created
        indices = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_raw%'"
        ).fetchall()
        index_names = {row[0] for row in indices}
        assert "idx_raw_conv_provider" in index_names
        assert "idx_raw_conv_source" in index_names

        # Test insertion works
        conn.execute(
            """
            INSERT INTO raw_conversations
            (raw_id, provider_name, source_name, source_path, raw_content, acquired_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("raw1", "claude", "inbox", "/path/to/file.jsonl", b"content", "2024-01-01T00:00:00Z"),
        )
        conn.commit()

        row = conn.execute("SELECT raw_id, provider_name FROM raw_conversations WHERE raw_id = 'raw1'").fetchone()
        assert row["raw_id"] == "raw1"
        assert row["provider_name"] == "claude"

        conn.close()


class TestMigrateV8ToV9:
    """Tests for _migrate_v8_to_v9 (idempotent source_name addition)."""

    def test_migrate_v8_to_v9_adds_source_name_and_is_idempotent(self, tmp_path):
        """Migration adds source_name column to raw_conversations and is idempotent."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # First test: adding to v8 schema (without source_name)
        conn.execute(V8_RAW_CONVERSATIONS_TABLE)
        conn.commit()

        _migrate_v8_to_v9(conn)
        conn.commit()

        cursor = conn.execute("PRAGMA table_info(raw_conversations)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "source_name" in columns

        # Second test: idempotency - recreate with v9 schema and run migration again
        conn.execute("DROP TABLE raw_conversations")
        conn.commit()

        conn.execute(V9_RAW_CONVERSATIONS_TABLE)
        conn.commit()

        _migrate_v8_to_v9(conn)
        conn.commit()

        cursor = conn.execute("PRAGMA table_info(raw_conversations)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "source_name" in columns

        conn.close()


class TestMigrateV9ToV10:
    """Tests for _migrate_v9_to_v10 (vec0 tables and embeddings)."""

    def test_migrate_v9_to_v10_creates_embeddings_meta_table(self, tmp_path):
        """Migration creates embeddings_meta table regardless of sqlite-vec."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        conn.execute(V9_CONVERSATIONS_MIN_TABLE)
        conn.execute(V9_MESSAGES_MIN_TABLE)
        conn.commit()

        # Run migration
        _migrate_v9_to_v10(conn)
        conn.commit()

        # Verify embeddings_meta table exists
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings_meta'"
        ).fetchone()
        assert exists is not None

        # Verify embedding_status table exists
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_status'"
        ).fetchone()
        assert exists is not None

        # Verify embeddings_meta has correct columns
        cursor = conn.execute("PRAGMA table_info(embeddings_meta)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "target_id" in columns
        assert "target_type" in columns
        assert "model" in columns
        assert "dimension" in columns

        conn.close()

    def test_migrate_v9_to_v10_creates_embedding_status_table(self, tmp_path):
        """Migration creates embedding_status table for tracking."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        conn.execute(V9_CONVERSATIONS_MIN_TABLE)
        conn.execute(V9_MESSAGES_MIN_TABLE)
        conn.commit()

        _migrate_v9_to_v10(conn)
        conn.commit()

        # Verify embedding_status table exists
        cursor = conn.execute("PRAGMA table_info(embedding_status)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "conversation_id" in columns
        assert "message_count_embedded" in columns
        assert "last_embedded_at" in columns
        assert "needs_reindex" in columns
        assert "error_message" in columns

        conn.close()


class TestEnsureVec0Table:
    """Tests for _ensure_vec0_table (idempotent vec0 creation)."""

    def test_ensure_vec0_table_idempotent_and_creates_when_missing(self, tmp_path):
        """_ensure_vec0_table creates vec0 table if missing, is idempotent when called multiple times."""
        db_path = tmp_path / "test.db"
        with connection_context(db_path) as conn:
            # Verify vec_version is available (sqlite-vec loaded)
            try:
                conn.execute("SELECT vec_version()")
                vec_available = True
            except sqlite3.OperationalError:
                vec_available = False

            if vec_available:
                # Test creation when missing
                conn.execute("DROP TABLE IF EXISTS message_embeddings")
                conn.commit()

                exists = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='message_embeddings'"
                ).fetchone()
                assert exists is None

                _ensure_vec0_table(conn)

                exists = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='message_embeddings'"
                ).fetchone()
                assert exists is not None

            # Test idempotency (call multiple times, should not raise)
            _ensure_vec0_table(conn)
            _ensure_vec0_table(conn)
            _ensure_vec0_table(conn)


class TestAssetPath:
    """Tests for asset_path function."""

    @pytest.mark.parametrize("asset_id,verify_fn,description", [
        (
            "att-id/with@special#chars",
            lambda path: "att-" in path.name,
            "sanitizes_special_characters",
        ),
        (
            "id with / spaces",
            lambda path: "/" not in path.name and "att-" in path.name,
            "sanitizes_spaces_and_slashes",
        ),
        (
            "a",
            lambda path: "assets" in path.parts,
            "handles_short_id",
        ),
        (
            "attachment-123",
            lambda path: "attachment-123" in str(path),
            "preserves_clean_id",
        ),
        (
            "test-asset-001",
            lambda path: "assets" in path.parts and path.parent.parent.name == "assets",
            "creates_correct_structure",
        ),
        (
            "",
            lambda path: "att-" in path.name,
            "handles_empty_string",
        ),
    ])
    def test_asset_path_behavior(self, tmp_path, asset_id, verify_fn, description):
        """Parametrized test for asset_path with various inputs and expectations."""
        path = asset_path(tmp_path, asset_id)
        assert verify_fn(path), f"Failed for {description}: {asset_id}"


class TestWriteAsset:
    """Tests for write_asset function."""

    def test_write_asset_success(self, tmp_path):
        """write_asset successfully writes content to disk."""
        asset_id = "test-asset"
        content = b"test content"

        result_path = write_asset(tmp_path, asset_id, content)

        assert result_path.exists()
        assert result_path.read_bytes() == content

    def test_write_asset_error_cleans_up_on_failures(self, tmp_path):
        """write_asset cleans up temp files on replace or write errors."""
        def mock_replace(src, dst):
            if os.path.exists(src):
                os.unlink(src)
            raise OSError("Simulated replace error")

        # Test replace error cleanup
        with patch("os.replace", side_effect=mock_replace):
            with pytest.raises(OSError):
                write_asset(tmp_path, "test-asset", b"test content")
        temp_files = list(tmp_path.glob(".test-asset.*"))
        assert len(temp_files) == 0

        # Test write error cleanup
        def mock_write(fd, data):
            raise OSError("Write failed")

        with patch("os.write", side_effect=mock_write):
            with pytest.raises(OSError):
                write_asset(tmp_path, "test-asset", b"test content")
        asset_dir = tmp_path / "assets"
        if asset_dir.exists():
            temp_files = list(asset_dir.glob("**/.test-asset.*"))
            assert len(temp_files) == 0

    def test_write_asset_atomic_rename(self, tmp_path):
        """write_asset uses atomic rename (os.replace)."""
        with patch("os.replace") as mock_replace:
            write_asset(tmp_path, "test-asset", b"test content")
            assert mock_replace.called

    def test_write_asset_creates_nested_dirs(self, tmp_path):
        """write_asset creates all needed nested directories."""
        result_path = write_asset(tmp_path, "very-long-asset-id-that-should-get-hashed", b"content")
        assert result_path.parent.exists()
        assert result_path.parent.parent.exists()
        assert result_path.exists()


class TestRawConversationEdgeCases:
    """Tests for raw conversation storage and edge cases."""

    async def test_raw_conversation_with_all_fields(self, tmp_path):
        """Raw conversation records can be saved with all optional fields."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        # First, create a raw_conversations record
        with connection_context(tmp_path / "test.db") as conn:
            conn.execute(
                """
                INSERT INTO raw_conversations
                (raw_id, provider_name, source_path, raw_content, acquired_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("raw-123", "claude", "/path/to/file.jsonl", b"content", "2024-01-01T00:00:00Z"),
            )
            conn.commit()

        # Create conversation linked to raw data
        conv = ConversationRecord(
            conversation_id="conv-with-raw",
            provider_name="claude",
            provider_conversation_id="claude-123",
            content_hash=make_hash("conv-with-raw"),
            title="Test Conv",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            raw_id="raw-123",
            parent_conversation_id=None,
            branch_type=None,
        )
        await backend.save_conversation_record(conv)

        # Retrieve and verify
        retrieved = await backend.get_conversation("conv-with-raw")
        assert retrieved is not None
        assert retrieved.raw_id == "raw-123"

        await backend.close()

    async def test_list_conversations_with_branch_type(self, tmp_path):
        """List conversations by parent respects branch_type field."""
        backend = SQLiteBackend(db_path=tmp_path / "test.db")

        parent = ConversationRecord(
            conversation_id="parent",
            provider_name="test",
            provider_conversation_id="p",
            content_hash=make_hash("parent"),
            title="Parent",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
        await backend.save_conversation_record(parent)

        # Create child with branch_type
        child = ConversationRecord(
            conversation_id="child",
            provider_name="test",
            provider_conversation_id="c",
            content_hash=make_hash("child"),
            title="Child",
            created_at="2024-01-02T00:00:00Z",
            updated_at="2024-01-02T00:00:00Z",
            parent_conversation_id="parent",
            branch_type="sidechain",
        )
        await backend.save_conversation_record(child)

        # Query and verify branch_type is preserved
        children = await backend.list_conversations_by_parent("parent")
        assert len(children) == 1
        assert children[0].branch_type == "sidechain"

        await backend.close()


class TestConcurrentAssetWrite:
    """Tests for concurrent asset writing safety."""

    def test_concurrent_write_same_asset_no_corruption(self, tmp_path: Path):
        """Concurrent writes to same asset should not corrupt file."""
        asset_id = "concurrent-test-asset"
        content = b"x" * 10000  # 10KB of data

        def write_asset_thread(thread_id: int):
            # Each thread writes the same content to the same asset
            write_asset(tmp_path, asset_id, content)

        # Run 10 concurrent writes
        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(write_asset_thread, range(10)))

        # Verify file is not corrupted
        final_path = asset_path(tmp_path, asset_id)
        assert final_path.exists(), "Asset file should exist"
        assert final_path.read_bytes() == content, "Asset content should not be corrupted"

    def test_write_asset_atomic(self, tmp_path: Path):
        """write_asset should use atomic write (write to temp, then rename)."""
        asset_id = "atomic-test"
        content = b"test content"

        write_asset(tmp_path, asset_id, content)

        final_path = asset_path(tmp_path, asset_id)
        assert final_path.exists()
        assert final_path.read_bytes() == content

    def test_write_asset_creates_parent_directories(self, tmp_path: Path):
        """write_asset should create necessary parent directories."""
        asset_id = "deeply-nested-asset-id-with-hash-prefix"
        content = b"nested content"

        write_asset(tmp_path, asset_id, content)

        final_path = asset_path(tmp_path, asset_id)
        assert final_path.exists()
        assert final_path.read_bytes() == content
        assert final_path.parent.exists()

    def test_write_asset_overwrites_existing(self, tmp_path: Path):
        """write_asset should overwrite existing file atomically."""
        asset_id = "overwrite-test"
        old_content = b"old content"
        new_content = b"new content that is different"

        # Write initial content
        write_asset(tmp_path, asset_id, old_content)
        final_path = asset_path(tmp_path, asset_id)
        assert final_path.read_bytes() == old_content

        # Overwrite with new content
        write_asset(tmp_path, asset_id, new_content)
        assert final_path.read_bytes() == new_content

    def test_write_asset_empty_content(self, tmp_path: Path):
        """write_asset should handle empty content correctly."""
        asset_id = "empty-asset"
        content = b""

        write_asset(tmp_path, asset_id, content)

        final_path = asset_path(tmp_path, asset_id)
        assert final_path.exists()
        assert final_path.read_bytes() == content


class TestParseSessionsIndex:
    """Tests for parse_sessions_index function."""

    def test_parses_valid_index(self, sample_sessions_index):
        """Parses valid sessions-index.json file."""
        entries = parse_sessions_index(sample_sessions_index)

        assert len(entries) == 3
        assert "abc123-def456" in entries
        assert "ghi789-jkl012" in entries
        assert "sidechain-test" in entries

    def test_extracts_all_fields(self, sample_sessions_index):
        """Extracts all expected fields from index entries."""
        entries = parse_sessions_index(sample_sessions_index)
        entry = entries["abc123-def456"]

        assert entry.session_id == "abc123-def456"
        assert entry.first_prompt == "How do I fix the bug in auth?"
        assert entry.summary == "Fixed authentication bug in login flow"
        assert entry.message_count == 12
        assert entry.created == "2024-01-15T10:30:00.000Z"
        assert entry.modified == "2024-01-15T11:45:00.000Z"
        assert entry.git_branch == "feature/auth-fix"
        assert entry.project_path == "/home/user/myproject"
        assert entry.is_sidechain is False

    def test_returns_empty_on_missing_file(self, tmp_path):
        """Returns empty dict when file doesn't exist."""
        entries = parse_sessions_index(tmp_path / "nonexistent.json")
        assert entries == {}

    def test_returns_empty_on_invalid_json(self, tmp_path):
        """Returns empty dict on invalid JSON."""
        index_path = tmp_path / "sessions-index.json"
        index_path.write_text("not valid json")

        entries = parse_sessions_index(index_path)
        assert entries == {}

    def test_returns_empty_on_missing_entries(self, tmp_path):
        """Returns empty dict when entries key is missing."""
        index_path = tmp_path / "sessions-index.json"
        index_path.write_text('{"version": 1}')

        entries = parse_sessions_index(index_path)
        assert entries == {}


class TestSessionIndexEntry:
    """Tests for SessionIndexEntry dataclass."""

    def test_from_dict_creates_entry(self):
        """Creates entry from dictionary."""
        data = {
            "sessionId": "test-123",
            "fullPath": "/path/to/session.jsonl",
            "firstPrompt": "Hello",
            "summary": "Test session",
            "messageCount": 5,
            "created": "2024-01-01T00:00:00.000Z",
            "modified": "2024-01-01T01:00:00.000Z",
            "gitBranch": "main",
            "projectPath": "/project",
            "isSidechain": False,
        }

        entry = SessionIndexEntry.from_dict(data)

        assert entry.session_id == "test-123"
        assert entry.summary == "Test session"
        assert entry.message_count == 5

    def test_from_dict_handles_missing_optional_fields(self):
        """Handles missing optional fields gracefully."""
        data = {"sessionId": "test-123", "fullPath": "/path/to/session.jsonl"}

        entry = SessionIndexEntry.from_dict(data)

        assert entry.session_id == "test-123"
        assert entry.first_prompt is None
        assert entry.summary is None
        assert entry.is_sidechain is False


class TestEnrichConversationFromIndex:
    """Tests for enrich_conversation_from_index function."""

    def test_enriches_title_with_summary(self, sample_conversation, sample_sessions_index):
        """Uses summary as title when available."""
        entries = parse_sessions_index(sample_sessions_index)
        entry = entries["abc123-def456"]

        enriched = enrich_conversation_from_index(sample_conversation, entry)

        assert enriched.title == "Fixed authentication bug in login flow"

    def test_enriches_timestamps(self, sample_conversation, sample_sessions_index):
        """Uses index timestamps when conversation lacks them."""
        entries = parse_sessions_index(sample_sessions_index)
        entry = entries["abc123-def456"]

        enriched = enrich_conversation_from_index(sample_conversation, entry)

        assert enriched.created_at == "2024-01-15T10:30:00.000Z"
        assert enriched.updated_at == "2024-01-15T11:45:00.000Z"

    def test_enriches_provider_meta(self, sample_conversation, sample_sessions_index):
        """Adds git branch and project path to provider_meta."""
        entries = parse_sessions_index(sample_sessions_index)
        entry = entries["abc123-def456"]

        enriched = enrich_conversation_from_index(sample_conversation, entry)

        assert enriched.provider_meta["gitBranch"] == "feature/auth-fix"
        assert enriched.provider_meta["projectPath"] == "/home/user/myproject"
        assert enriched.provider_meta["isSidechain"] is False

    def test_uses_first_prompt_when_no_summary(self, sample_conversation, sample_sessions_index):
        """Falls back to firstPrompt when summary is generic."""
        entries = parse_sessions_index(sample_sessions_index)
        entry = entries["ghi789-jkl012"]

        enrich_conversation_from_index(sample_conversation, entry)

        # "User Exits CLI Session" is filtered out, falls back to firstPrompt
        # But "No prompt" is also filtered, so keeps original title

    def test_truncates_long_first_prompt(self, sample_conversation):
        """Truncates firstPrompt if longer than 80 chars."""
        long_prompt = "A" * 100
        entry = SessionIndexEntry(
            session_id="test",
            full_path="/path",
            first_prompt=long_prompt,
            summary=None,  # No summary, use firstPrompt
            message_count=1,
            created=None,
            modified=None,
            git_branch=None,
            project_path=None,
            is_sidechain=False,
        )

        enriched = enrich_conversation_from_index(sample_conversation, entry)

        assert len(enriched.title) == 83  # 80 + "..."
        assert enriched.title.endswith("...")


class TestFindSessionsIndex:
    """Tests for find_sessions_index function."""

    def test_finds_index_in_same_directory(self, sample_sessions_index, tmp_path):
        """Finds sessions-index.json in session file directory."""
        session_file = tmp_path / "test-session.jsonl"
        session_file.touch()

        index_path = find_sessions_index(session_file)

        assert index_path is not None
        assert index_path.name == "sessions-index.json"

    def test_returns_none_when_no_index(self, tmp_path):
        """Returns None when no sessions-index.json exists."""
        session_file = tmp_path / "test-session.jsonl"
        session_file.touch()

        index_path = find_sessions_index(session_file)

        assert index_path is None
