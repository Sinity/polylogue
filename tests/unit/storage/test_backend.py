"""Focused infrastructure contracts for the SQLite backend."""

from __future__ import annotations

import asyncio
import importlib
import sqlite3
import threading
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite
import pytest

import polylogue.paths
from polylogue.archive.message.roles import Role
from polylogue.errors import DatabaseError
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import connection_context, open_connection, open_read_connection
from polylogue.storage.backends.connection_profile import (
    READ_CACHE_SIZE_KIB,
    READ_CONNECTION_PRAGMA_STATEMENTS,
    READ_CONNECTION_PROFILE,
    WRITE_CACHE_SIZE_KIB,
    WRITE_CONNECTION_PRAGMA_STATEMENTS,
    WRITE_CONNECTION_PROFILE,
)
from polylogue.storage.backends.schema import SCHEMA_VERSION, _ensure_schema
from polylogue.storage.backends.schema_bootstrap import (
    SchemaColumnExtensionDescriptor,
    SchemaIndexExtensionDescriptor,
    SchemaSnapshot,
    build_current_schema_extension_plan,
    decide_schema_bootstrap,
    schema_extension_snapshot_indexes,
    schema_extension_snapshot_tables,
)
from polylogue.storage.embeddings.models import EmbeddingStatsSnapshot
from polylogue.storage.query_models import ConversationRecordQuery
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.runtime import ConversationRecord
from tests.infra.storage_records import (
    make_attachment,
    make_content_block,
    make_conversation,
    make_message,
    make_raw_conversation,
)


def _replace_ensure_schema(
    backend: SQLiteBackend,
    handler: Callable[[aiosqlite.Connection], Awaitable[None]],
) -> None:
    object.__setattr__(backend, "_ensure_schema", handler)


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}


def _message_types(conn: sqlite3.Connection) -> dict[str, str]:
    return {row[0]: row[1] for row in conn.execute("SELECT message_id, message_type FROM messages").fetchall()}


def _create_v2_message_type_fixture(conn: sqlite3.Connection, *, include_message_type: bool = False) -> None:
    message_type_column = ", message_type TEXT NOT NULL DEFAULT 'message'" if include_message_type else ""
    message_type_insert_column = ", message_type" if include_message_type else ""
    explicit_value = ", 'summary'" if include_message_type else ""
    conn.executescript(
        f"""
        CREATE TABLE messages (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT,
            text TEXT,
            sort_key REAL,
            content_hash TEXT NOT NULL,
            version INTEGER NOT NULL,
            branch_index INTEGER NOT NULL DEFAULT 0,
            provider_name TEXT NOT NULL DEFAULT '',
            word_count INTEGER NOT NULL DEFAULT 0,
            has_tool_use INTEGER NOT NULL DEFAULT 0,
            has_thinking INTEGER NOT NULL DEFAULT 0,
            has_paste INTEGER NOT NULL DEFAULT 0
            {message_type_column}
        );
        CREATE TABLE content_blocks (
            block_id TEXT PRIMARY KEY,
            message_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            block_index INTEGER NOT NULL,
            type TEXT NOT NULL,
            text TEXT,
            tool_name TEXT,
            tool_id TEXT,
            tool_input TEXT,
            media_type TEXT,
            metadata TEXT,
            semantic_type TEXT,
            UNIQUE (message_id, block_index)
        );
        INSERT INTO messages (
            message_id, conversation_id, role, text, sort_key, content_hash,
            version, provider_name, has_tool_use, has_thinking{message_type_insert_column}
        ) VALUES
            ('normal', 'conv-1', 'user', 'hello', 1, 'h-normal', 1, 'codex', 0, 0{explicit_value}),
            ('thinking-block', 'conv-1', 'assistant', '', 2, 'h-thinking', 1, 'codex', 0, 0{explicit_value}),
            ('tool-result-block', 'conv-1', 'assistant', '', 3, 'h-tool-result', 1, 'codex', 1, 0{explicit_value}),
            ('tool-role', 'conv-1', 'tool', '', 4, 'h-tool-role', 1, 'codex', 1, 0{explicit_value}),
            ('tool-use-block', 'conv-1', 'assistant', '', 5, 'h-tool-use', 1, 'codex', 1, 0{explicit_value}),
            ('tool-flag', 'conv-1', 'assistant', '', 6, 'h-tool-flag', 1, 'codex', 1, 0{explicit_value});
        INSERT INTO content_blocks (
            block_id, message_id, conversation_id, block_index, type
        ) VALUES
            ('block-thinking', 'thinking-block', 'conv-1', 0, 'thinking'),
            ('block-tool-result', 'tool-result-block', 'conv-1', 0, 'tool_result'),
            ('block-tool-use', 'tool-use-block', 'conv-1', 0, 'tool_use');
        PRAGMA user_version = 2;
        """
    )
    conn.commit()


def _build_record(conversation_id: str = "conv-1") -> ConversationRecord:
    return make_conversation(
        conversation_id=conversation_id,
        provider_name="claude-ai",
        provider_conversation_id=conversation_id,
        title="Test",
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
        content_hash=f"hash-{conversation_id}",
        version=1,
    )


def test_ensure_schema_contract(tmp_path: Path) -> None:
    """Schema application must upgrade a fresh database and create the core tables."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
    _ensure_schema(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert {
        "artifact_observations",
        "conversations",
        "messages",
        "attachments",
        "attachment_refs",
        "runs",
        "publications",
    }.issubset(_table_names(conn))
    assert "message_meta" not in _table_names(conn)
    raw_columns = {row[1] for row in conn.execute("PRAGMA table_info(raw_conversations)").fetchall()}
    assert "blob_size" in raw_columns
    conn.close()


def test_ensure_schema_upgrades_v2_message_types_without_reimport(tmp_path: Path) -> None:
    """The supported v2->v3 path adds message_type without deleting archive rows."""
    db_path = tmp_path / "v2.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _create_v2_message_type_fixture(conn)

    before_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

    _ensure_schema(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == before_count
    assert "message_type" in {row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()}
    assert (
        conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_messages_conversation_message_type'"
        ).fetchone()
        is not None
    )
    assert _message_types(conn) == {
        "normal": "message",
        "thinking-block": "thinking",
        "tool-result-block": "tool_result",
        "tool-role": "tool_result",
        "tool-use-block": "tool_use",
        "tool-flag": "tool_use",
    }
    conn.close()


def test_ensure_schema_upgrades_v2_already_extended_without_overwriting(tmp_path: Path) -> None:
    """A v2 archive that already has message_type only needs the version marker."""
    db_path = tmp_path / "v2-already-extended.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _create_v2_message_type_fixture(conn, include_message_type=True)

    _ensure_schema(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert set(_message_types(conn).values()) == {"summary"}
    conn.close()


def test_ensure_schema_rejects_version_mismatch_without_mutating(tmp_path: Path) -> None:
    """Version mismatch fails clearly instead of partially patching old layouts."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE raw_conversations (
            raw_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            source_path TEXT NOT NULL,
            blob_size INTEGER NOT NULL,
            acquired_at TEXT NOT NULL
        );
        PRAGMA user_version = 2;
        """
    )
    conn.commit()

    with pytest.raises(DatabaseError) as exc_info:
        _ensure_schema(conn)

    assert "only supports explicitly declared in-place archive upgrades" in str(exc_info.value)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    assert conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'").fetchone() is None
    conn.close()


def test_schema_bootstrap_decision_returns_version_mismatch_action() -> None:
    """Version mismatch returns version_mismatch action instead of raising."""
    snapshot = SchemaSnapshot(current_version=999, table_columns={}, index_sql={})

    decision = decide_schema_bootstrap(snapshot)

    assert decision.action == "version_mismatch"
    assert decision.current_version == 999


def test_schema_extension_descriptors_detect_missing_columns_and_indexes() -> None:
    """Descriptor expansion should be table-gated and idempotent."""
    column_descriptor = SchemaColumnExtensionDescriptor(
        table_name="demo",
        column_name="extra",
        ddl="ALTER TABLE demo ADD COLUMN extra TEXT",
    )
    missing_column_snapshot = SchemaSnapshot(
        current_version=SCHEMA_VERSION,
        table_columns={"demo": frozenset({"id"})},
        index_sql={},
    )
    current_column_snapshot = SchemaSnapshot(
        current_version=SCHEMA_VERSION,
        table_columns={"demo": frozenset({"id", "extra"})},
        index_sql={},
    )
    absent_table_snapshot = SchemaSnapshot(current_version=SCHEMA_VERSION, table_columns={}, index_sql={})

    assert column_descriptor.statements(missing_column_snapshot) == ("ALTER TABLE demo ADD COLUMN extra TEXT",)
    assert column_descriptor.statements(current_column_snapshot) == ()
    assert column_descriptor.statements(absent_table_snapshot) == ()

    index_descriptor = SchemaIndexExtensionDescriptor(
        table_name="demo",
        index_name="idx_demo_id",
        ddl="CREATE INDEX IF NOT EXISTS idx_demo_id ON demo(id)",
        replace_on_drift=True,
    )
    missing_index_snapshot = SchemaSnapshot(
        current_version=SCHEMA_VERSION,
        table_columns={"demo": frozenset({"id"})},
        index_sql={"idx_demo_id": None},
    )
    current_index_snapshot = SchemaSnapshot(
        current_version=SCHEMA_VERSION,
        table_columns={"demo": frozenset({"id"})},
        index_sql={"idx_demo_id": "CREATE INDEX idx_demo_id ON demo(id)"},
    )
    drifted_index_snapshot = SchemaSnapshot(
        current_version=SCHEMA_VERSION,
        table_columns={"demo": frozenset({"id", "other_id"})},
        index_sql={"idx_demo_id": "CREATE INDEX idx_demo_id ON demo(other_id)"},
    )

    assert index_descriptor.statements(absent_table_snapshot) == ()
    assert index_descriptor.statements(missing_index_snapshot) == (
        "CREATE INDEX IF NOT EXISTS idx_demo_id ON demo(id)",
    )
    assert index_descriptor.statements(current_index_snapshot) == ()
    assert index_descriptor.statements(drifted_index_snapshot) == (
        "DROP INDEX IF EXISTS idx_demo_id",
        "CREATE INDEX IF NOT EXISTS idx_demo_id ON demo(id)",
    )


def test_schema_extension_catalog_declares_snapshot_scope() -> None:
    """Snapshot capture should follow the descriptor catalog instead of ad hoc lists."""
    table_names = schema_extension_snapshot_tables()
    index_names = schema_extension_snapshot_indexes()

    assert len(table_names) == len(set(table_names))
    assert len(index_names) == len(set(index_names))
    assert {"raw_conversations", "attachments", "session_profiles", "session_phases"}.issubset(table_names)
    assert {
        "idx_messages_conversation_message_type",
        "idx_raw_conv_source_mtime",
        "idx_raw_conv_effective_provider",
        "idx_content_blocks_tool_use_conversation",
        "idx_attachments_provider_meta_file_id",
        "idx_attachment_refs_provider_meta_drive_id",
    }.issubset(index_names)


def test_schema_extension_plan_expands_catalog_descriptors() -> None:
    """Current-version extension planning should expand descriptor facts into SQL."""
    snapshot = SchemaSnapshot(
        current_version=SCHEMA_VERSION,
        table_columns={
            "messages": frozenset({"message_id", "conversation_id"}),
            "raw_conversations": frozenset({"raw_id", "source_path", "file_mtime"}),
            "content_blocks": frozenset({"conversation_id", "type"}),
            "attachments": frozenset({"provider_meta"}),
            "session_profiles": frozenset({"conversation_id", "search_text"}),
        },
        index_sql=dict.fromkeys(schema_extension_snapshot_indexes()),
    )

    plan = build_current_schema_extension_plan(snapshot)

    assert "ALTER TABLE messages ADD COLUMN message_type TEXT NOT NULL DEFAULT 'message'" in plan.statements
    assert any("idx_messages_conversation_message_type" in statement for statement in plan.statements)
    assert "ALTER TABLE raw_conversations ADD COLUMN blob_size INTEGER NOT NULL DEFAULT 0" in plan.statements
    assert any("idx_content_blocks_tool_use_conversation" in statement for statement in plan.statements)
    assert any("idx_attachments_provider_meta_id" in statement for statement in plan.statements)
    assert any("idx_raw_conv_effective_provider" in statement for statement in plan.statements)
    assert any(
        "ALTER TABLE session_profiles ADD COLUMN evidence_search_text" in statement for statement in plan.statements
    )
    assert any("UPDATE session_profiles" in statement for statement in plan.statements)
    # Partial indexes scoping each backfill UPDATE to unbackfilled rows.
    assert any("idx_session_profiles_evidence_search_text_unbackfilled" in statement for statement in plan.statements)
    assert any("idx_session_profiles_inference_search_text_unbackfilled" in statement for statement in plan.statements)
    assert any("idx_session_profiles_enrichment_search_text_unbackfilled" in statement for statement in plan.statements)
    # Indexes precede the matching UPDATE so the optimizer can use them on first run.
    statements = list(plan.statements)
    for column in ("evidence_search_text", "inference_search_text", "enrichment_search_text"):
        index_position = next(i for i, s in enumerate(statements) if f"idx_session_profiles_{column}_unbackfilled" in s)
        update_position = next(i for i, s in enumerate(statements) if "UPDATE session_profiles" in s and column in s)
        assert index_position < update_position, f"{column}: index must be created before its backfill UPDATE"
    assert len(plan.scripts) == 5


def test_ensure_schema_rejects_old_raw_table_version_without_mutating(tmp_path: Path) -> None:
    """Old-version raw tables are blocked instead of being partially patched."""
    db_path = tmp_path / "legacy-v1.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE raw_conversations (
            raw_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            payload_provider TEXT,
            source_name TEXT,
            source_path TEXT NOT NULL,
            source_index INTEGER,
            blob_size INTEGER NOT NULL DEFAULT 0,
            acquired_at TEXT NOT NULL,
            file_mtime TEXT,
            parsed_at TEXT,
            parse_error TEXT,
            validated_at TEXT,
            validation_status TEXT CHECK (validation_status IN ('passed', 'failed', 'skipped') OR validation_status IS NULL),
            validation_error TEXT,
            validation_drift_count INTEGER DEFAULT 0,
            validation_provider TEXT,
            validation_mode TEXT
        );
        PRAGMA user_version = 2;
        """
    )
    conn.commit()

    with pytest.raises(DatabaseError) as exc_info:
        _ensure_schema(conn)

    assert "only supports explicitly declared in-place archive upgrades" in str(exc_info.value)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    assert conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'").fetchone() is None
    conn.close()


def test_ensure_schema_rejects_legacy_inline_raw_layout(tmp_path: Path) -> None:
    """Legacy inline-raw databases should fail with a clear reset/back-up message."""
    db_path = tmp_path / "legacy-inline-raw.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE raw_conversations (
            raw_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            payload_provider TEXT,
            source_name TEXT,
            source_path TEXT NOT NULL,
            source_index INTEGER,
            raw_content BLOB NOT NULL,
            acquired_at TEXT NOT NULL,
            file_mtime TEXT,
            parsed_at TEXT,
            parse_error TEXT,
            validated_at TEXT,
            validation_status TEXT,
            validation_error TEXT,
            validation_drift_count INTEGER DEFAULT 0,
            validation_provider TEXT,
            validation_mode TEXT
        );
        PRAGMA user_version = 1;
        """
    )
    conn.commit()

    with pytest.raises(Exception) as exc_info:
        _ensure_schema(conn)

    assert exc_info.type.__name__ == "DatabaseError"
    assert "legacy inline raw-content layout" in str(exc_info.value)
    conn.close()


def test_open_connection_contract(tmp_path: Path) -> None:
    """open_connection() must create directories, apply schema, and set sqlite pragmas."""
    db_path = tmp_path / "nested" / "path" / "test.db"

    with open_connection(db_path) as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
        assert conn.execute("PRAGMA cache_size").fetchone()[0] == -WRITE_CACHE_SIZE_KIB
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert {
            "artifact_observations",
            "conversations",
            "messages",
            "attachments",
            "attachment_refs",
            "publications",
        }.issubset(_table_names(conn))

    assert db_path.exists()
    assert db_path.parent.exists()


def test_connection_profiles_define_sqlite_tuning_once() -> None:
    """Read and write PRAGMA statements should be derived from the shared profiles."""
    import polylogue.pipeline.services.ingest_batch as ingest_batch_module
    import polylogue.storage.backends.async_sqlite as async_sqlite_module
    import polylogue.storage.backends.connection as connection_module

    assert WRITE_CONNECTION_PROFILE.pragma_statements == WRITE_CONNECTION_PRAGMA_STATEMENTS
    assert READ_CONNECTION_PROFILE.pragma_statements == READ_CONNECTION_PRAGMA_STATEMENTS
    assert connection_module.WRITE_CONNECTION_PRAGMA_STATEMENTS == WRITE_CONNECTION_PRAGMA_STATEMENTS
    assert vars(async_sqlite_module)["WRITE_CONNECTION_PRAGMA_STATEMENTS"] == WRITE_CONNECTION_PRAGMA_STATEMENTS
    assert vars(ingest_batch_module)["WRITE_CONNECTION_PRAGMA_STATEMENTS"] == WRITE_CONNECTION_PRAGMA_STATEMENTS
    assert connection_module.READ_CONNECTION_PRAGMA_STATEMENTS == READ_CONNECTION_PRAGMA_STATEMENTS
    assert vars(async_sqlite_module)["READ_CONNECTION_PRAGMA_STATEMENTS"] == READ_CONNECTION_PRAGMA_STATEMENTS
    assert WRITE_CONNECTION_PROFILE.busy_timeout_ms == 30000
    assert WRITE_CONNECTION_PROFILE.cache_size_kib == WRITE_CACHE_SIZE_KIB
    assert READ_CONNECTION_PROFILE.busy_timeout_ms == 1000
    assert READ_CONNECTION_PROFILE.cache_size_kib == READ_CACHE_SIZE_KIB


def test_open_read_connection_contract(tmp_path: Path) -> None:
    """open_read_connection() must avoid writer-style setup when the DB exists."""
    db_path = tmp_path / "readonly.db"
    with open_connection(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS sentinel(value INTEGER)")
        conn.execute("INSERT INTO sentinel(value) VALUES (1)")
        conn.commit()

    with open_read_connection(db_path) as conn:
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 0
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 1000
        assert conn.execute("PRAGMA cache_size").fetchone()[0] == -READ_CACHE_SIZE_KIB
        assert conn.execute("SELECT value FROM sentinel").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO sentinel(value) VALUES (2)")
            conn.commit()


def test_open_read_connection_rejects_old_schema_without_mutating(tmp_path: Path) -> None:
    """Read-only opens should report incompatible schemas before later SQL fails."""
    db_path = tmp_path / "old-readonly.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE raw_conversations (
            raw_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            source_path TEXT NOT NULL,
            blob_size INTEGER NOT NULL,
            acquired_at TEXT NOT NULL
        );
        PRAGMA user_version = 2;
        """
    )
    conn.commit()
    conn.close()

    with pytest.raises(DatabaseError) as exc_info:
        with open_read_connection(db_path):
            pass

    assert "only supports explicitly declared in-place archive upgrades" in str(exc_info.value)
    with sqlite3.connect(db_path) as verify_conn:
        assert verify_conn.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            verify_conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'").fetchone() is None
        )


@pytest.mark.slow
def test_open_connection_thread_isolation(tmp_path: Path) -> None:
    """Each thread must receive an isolated, usable connection object."""
    db_path = tmp_path / "threaded.db"
    barrier = threading.Barrier(3)
    connection_ids: list[int] = []
    errors: list[tuple[int, str]] = []

    with open_connection(db_path) as conn:
        conn.execute("SELECT 1").fetchone()

    def thread_func(thread_id: int) -> None:
        try:
            with open_connection(db_path) as conn:
                connection_ids.append(id(conn))
                barrier.wait()
                assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
                conn.execute("SELECT 1").fetchone()
        except Exception as exc:  # pragma: no cover - failure path assertion target
            errors.append((thread_id, str(exc)))

    threads = [threading.Thread(target=thread_func, args=(idx,)) for idx in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(set(connection_ids)) == 3


def test_connection_context_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """connection_context() must support explicit, default, and already-open connections."""
    explicit_path = tmp_path / "explicit.db"
    with connection_context(explicit_path) as explicit_conn:
        assert isinstance(explicit_conn, sqlite3.Connection)
        assert {
            "artifact_observations",
            "conversations",
            "messages",
            "attachments",
            "attachment_refs",
            "publications",
        }.issubset(_table_names(explicit_conn))

    data_home = tmp_path / "data"
    data_home.mkdir()
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    import polylogue.paths
    import polylogue.storage.backends.connection as connection_module

    importlib.reload(polylogue.paths)
    importlib.reload(connection_module)

    with connection_module.connection_context(None) as default_conn:
        db_file = Path(default_conn.execute("PRAGMA database_list").fetchone()[2])
        assert db_file.exists()
        assert str(data_home) in str(db_file)

    with connection_module.connection_context(explicit_path) as first_conn:
        with connection_module.connection_context(first_conn) as reused_conn:
            assert reused_conn is first_conn
        assert first_conn.execute("SELECT 1").fetchone()[0] == 1


def test_default_db_path_respects_xdg_data_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """polylogue.paths.db_path() must honor XDG_DATA_HOME."""
    xdg_data = tmp_path / "data"
    xdg_data.mkdir()
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data))

    import polylogue.storage.backends.connection as connection_module

    importlib.reload(connection_module)
    assert str(xdg_data) in str(polylogue.paths.db_path())


async def test_sqlite_backend_init_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SQLiteBackend init must preserve paths, create parents, and allocate the write lock."""
    custom_path = tmp_path / "custom" / "db.sqlite"
    custom_backend = SQLiteBackend(db_path=custom_path)
    assert custom_backend._db_path == custom_path
    assert custom_path.parent.exists()
    assert isinstance(custom_backend._write_lock, asyncio.Lock)
    await custom_backend.close()

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    import polylogue.paths

    importlib.reload(polylogue.paths)
    default_backend = SQLiteBackend(db_path=None)
    assert "polylogue" in str(default_backend._db_path)
    assert str(default_backend._db_path).endswith("polylogue.db")
    await default_backend.close()


async def test_async_backend_rejects_legacy_inline_raw_layout(tmp_path: Path) -> None:
    """Async backend writes should reject legacy inline-raw databases before acquisition begins."""
    db_path = tmp_path / "legacy-inline-raw.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE raw_conversations (
            raw_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            payload_provider TEXT,
            source_name TEXT,
            source_path TEXT NOT NULL,
            source_index INTEGER,
            raw_content BLOB NOT NULL,
            acquired_at TEXT NOT NULL,
            file_mtime TEXT,
            parsed_at TEXT,
            parse_error TEXT,
            validated_at TEXT,
            validation_status TEXT,
            validation_error TEXT,
            validation_drift_count INTEGER DEFAULT 0,
            validation_provider TEXT,
            validation_mode TEXT
        );
        PRAGMA user_version = 1;
        """
    )
    conn.commit()
    conn.close()

    backend = SQLiteBackend(db_path=db_path)
    with pytest.raises(Exception) as exc_info:
        await backend.get_raw_conversation_count()

    assert exc_info.type.__name__ == "DatabaseError"
    assert "legacy inline raw-content layout" in str(exc_info.value)
    await backend.close()


async def test_async_backend_schema_and_lock_contracts(tmp_path: Path) -> None:
    """Async schema guards and write locks must serialize correctly without blocking readers."""
    backend = SQLiteBackend(db_path=tmp_path / "async.db")

    results = await asyncio.gather(*[backend.get_conversation(f"conv:{idx}") for idx in range(10)])
    assert all(result is None for result in results)


async def test_async_backend_upgrades_v2_message_types_without_reimport(tmp_path: Path) -> None:
    """Async schema initialization should use the same v2->v3 upgrade path."""
    db_path = tmp_path / "v2-async.db"
    conn = sqlite3.connect(db_path)
    _create_v2_message_type_fixture(conn)
    conn.close()

    backend = SQLiteBackend(db_path=db_path)
    async with backend.connection():
        pass

    with sqlite3.connect(db_path) as verify_conn:
        assert verify_conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert verify_conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 6
        assert _message_types(verify_conn)["tool-result-block"] == "tool_result"
        assert _message_types(verify_conn)["thinking-block"] == "thinking"
    await backend.close()


async def test_async_backend_rejects_old_raw_table_version(tmp_path: Path) -> None:
    """Async backend init should not silently patch old-version raw tables."""
    db_path = tmp_path / "legacy-v1-async.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE raw_conversations (
            raw_id TEXT PRIMARY KEY,
            provider_name TEXT NOT NULL,
            payload_provider TEXT,
            source_name TEXT,
            source_path TEXT NOT NULL,
            source_index INTEGER,
            blob_size INTEGER NOT NULL DEFAULT 0,
            acquired_at TEXT NOT NULL,
            file_mtime TEXT,
            parsed_at TEXT,
            parse_error TEXT,
            validated_at TEXT,
            validation_status TEXT CHECK (validation_status IN ('passed', 'failed', 'skipped') OR validation_status IS NULL),
            validation_error TEXT,
            validation_drift_count INTEGER DEFAULT 0,
            validation_provider TEXT,
            validation_mode TEXT
        );
        PRAGMA user_version = 2;
        """
    )
    conn.commit()
    conn.close()

    backend = SQLiteBackend(db_path=db_path)
    with pytest.raises(DatabaseError) as exc_info:
        await backend.save_raw_conversation(
            make_raw_conversation(
                raw_id="raw-legacy",
                provider_name="chatgpt",
                source_path="/tmp/legacy.json",
                blob_size=123,
                acquired_at=datetime.now(timezone.utc).isoformat(),
            )
        )

    assert "only supports explicitly declared in-place archive upgrades" in str(exc_info.value)

    with sqlite3.connect(db_path) as verify_conn:
        assert verify_conn.execute("PRAGMA user_version").fetchone()[0] == 2
        assert verify_conn.execute("SELECT COUNT(*) FROM raw_conversations").fetchone()[0] == 0
    await backend.close()


async def test_async_backend_applies_current_session_profile_extensions(tmp_path: Path) -> None:
    """Async schema ensure should apply the same current-version profile extensions as the sync backend."""
    db_path = tmp_path / "legacy-profile-current.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        f"""
        CREATE TABLE session_profiles (
            conversation_id TEXT PRIMARY KEY,
            materializer_version INTEGER NOT NULL DEFAULT 5,
            materialized_at TEXT NOT NULL,
            source_updated_at TEXT,
            source_sort_key REAL,
            provider_name TEXT NOT NULL,
            title TEXT,
            first_message_at TEXT,
            repo_paths_json TEXT,
            repo_names_json TEXT,
            tags_json TEXT,
            auto_tags_json TEXT,
            message_count INTEGER NOT NULL DEFAULT 0,
            work_event_count INTEGER NOT NULL DEFAULT 0,
            word_count INTEGER NOT NULL DEFAULT 0,
            tool_use_count INTEGER NOT NULL DEFAULT 0,
            thinking_count INTEGER NOT NULL DEFAULT 0,
            total_cost_usd REAL NOT NULL DEFAULT 0,
            total_duration_ms INTEGER NOT NULL DEFAULT 0,
            wall_duration_ms INTEGER NOT NULL DEFAULT 0,
            search_text TEXT NOT NULL,
            inference_search_text TEXT NOT NULL DEFAULT ''
        );

        INSERT INTO session_profiles (
            conversation_id,
            materializer_version,
            materialized_at,
            source_updated_at,
            source_sort_key,
            provider_name,
            title,
            first_message_at,
            repo_paths_json,
            repo_names_json,
            tags_json,
            auto_tags_json,
            message_count,
            work_event_count,
            word_count,
            tool_use_count,
            thinking_count,
            total_cost_usd,
            total_duration_ms,
            wall_duration_ms,
            search_text,
            inference_search_text
        ) VALUES (
            'conv-1',
            5,
            '2026-04-01T00:00:00Z',
            '2026-04-01T00:00:00Z',
            1.0,
            'chatgpt',
            'Legacy profile',
            '2026-04-01T00:00:00Z',
            '[]',
            '[]',
            '[]',
            '[]',
            1,
            0,
            2,
            0,
            0,
            0.0,
            0,
            0,
            'search baseline',
            ''
        );

        PRAGMA user_version = {SCHEMA_VERSION};
        """
    )
    conn.commit()
    conn.close()

    backend = SQLiteBackend(db_path=db_path)
    async with backend.connection():
        pass

    with sqlite3.connect(db_path) as verify_conn:
        columns = {row[1] for row in verify_conn.execute("PRAGMA table_info(session_profiles)").fetchall()}
        row = verify_conn.execute(
            """
            SELECT evidence_search_text, inference_search_text, enrichment_search_text, enrichment_family
            FROM session_profiles
            WHERE conversation_id = ?
            """,
            ("conv-1",),
        ).fetchone()

    assert {"evidence_search_text", "enrichment_payload_json", "enrichment_search_text", "enrichment_family"}.issubset(
        columns
    )
    assert row is not None
    assert row[0] == "search baseline"
    assert row[1] == "search baseline"
    assert row[2] == "search baseline"
    assert row[3] == "scored_session_enrichment"
    await backend.close()


async def test_async_read_pool_uses_read_connection_settings(tmp_path: Path) -> None:
    """read_pool() should reuse read-oriented connections instead of writer-style ones."""
    backend = SQLiteBackend(db_path=tmp_path / "pooled.db")

    async with backend.connection() as conn:
        await conn.execute("CREATE TABLE IF NOT EXISTS sentinel(value INTEGER)")
        await conn.execute("INSERT INTO sentinel(value) VALUES (1)")
        await conn.commit()

    async with backend.read_pool(size=1):
        async with backend.read_connection() as conn:
            cursor = await conn.execute("PRAGMA busy_timeout")
            busy_timeout_row = await cursor.fetchone()
            assert busy_timeout_row is not None
            assert busy_timeout_row[0] == 1000
            cursor = await conn.execute("PRAGMA cache_size")
            cache_size_row = await cursor.fetchone()
            assert cache_size_row is not None
            assert cache_size_row[0] == -READ_CACHE_SIZE_KIB
            cursor = await conn.execute("SELECT value FROM sentinel")
            sentinel_row = await cursor.fetchone()
            assert sentinel_row is not None
            assert sentinel_row[0] == 1
            with pytest.raises(Exception) as exc_info:
                await conn.execute("INSERT INTO sentinel(value) VALUES (2)")
                await conn.commit()
            assert "readonly" in str(exc_info.value).lower() or "read-only" in str(exc_info.value).lower()

    await backend.close()


async def test_async_read_connection_closes_cleanly_after_pool_teardown(tmp_path: Path) -> None:
    """Checked-out read connections should not explode if the pool tears down first."""
    backend = SQLiteBackend(db_path=tmp_path / "pool-teardown.db")

    async with backend.connection() as conn:
        await conn.execute("CREATE TABLE IF NOT EXISTS sentinel(value INTEGER)")
        await conn.commit()

    pool_cm = backend.read_pool(size=1)
    await pool_cm.__aenter__()
    conn_cm = backend.read_connection()
    conn = await conn_cm.__aenter__()
    cursor = await conn.execute("SELECT 1")
    probe_row = await cursor.fetchone()
    assert probe_row is not None
    assert probe_row[0] == 1

    await pool_cm.__aexit__(None, None, None)
    await conn_cm.__aexit__(None, None, None)
    await backend.close()
    assert backend._schema_ensured is True

    init_count = 0
    original_ensure_schema = backend._ensure_schema

    async def counting_ensure_schema(conn: aiosqlite.Connection) -> None:
        nonlocal init_count
        init_count += 1
        await original_ensure_schema(conn)

    backend._schema_ensured = False
    _replace_ensure_schema(backend, counting_ensure_schema)
    await asyncio.gather(*[backend.queries.list_conversations(ConversationRecordQuery()) for _ in range(20)])
    assert init_count == 0

    slow_backend = SQLiteBackend(db_path=tmp_path / "slow.db")
    events: list[str] = []
    original_slow = slow_backend._ensure_schema

    async def slow_ensure_schema(conn: aiosqlite.Connection) -> None:
        events.append("start")
        await asyncio.sleep(0.05)
        await original_slow(conn)
        events.append("end")

    slow_backend._schema_ensured = False
    _replace_ensure_schema(slow_backend, slow_ensure_schema)
    await asyncio.gather(slow_backend.get_conversation("a"), slow_backend.get_conversation("b"))
    assert events.count("start") == 1
    assert events.count("end") == 1

    async with backend.transaction():
        assert backend._write_lock.locked()
    assert not backend._write_lock.locked()

    execution_order: list[int] = []

    async def write_operation(task_id: int) -> None:
        async with backend.transaction():
            execution_order.append(task_id)
            await asyncio.sleep(0.01)

    await asyncio.gather(*[write_operation(idx) for idx in range(5)])
    assert set(execution_order) == {0, 1, 2, 3, 4}

    await backend.queries.list_conversations(ConversationRecordQuery())
    assert not backend._write_lock.locked()

    read_completed = False

    async def slow_write() -> None:
        async with backend.transaction():
            await asyncio.sleep(0.1)

    async def quick_read() -> None:
        nonlocal read_completed
        await backend.queries.list_conversations(ConversationRecordQuery())
        read_completed = True

    write_task = asyncio.create_task(slow_write())
    await asyncio.sleep(0.01)
    await asyncio.gather(write_task, asyncio.create_task(quick_read()))
    assert read_completed is True

    second_backend = SQLiteBackend(db_path=backend.db_path)
    second_repo = ConversationRepository(backend=second_backend)

    cross_backend_read_completed = False

    async def cross_backend_read() -> None:
        nonlocal cross_backend_read_completed
        await asyncio.wait_for(second_backend.queries.list_conversations(ConversationRecordQuery()), timeout=2)
        await asyncio.wait_for(second_repo.get_archive_stats(), timeout=2)
        cross_backend_read_completed = True

    write_task = asyncio.create_task(slow_write())
    await asyncio.sleep(0.01)
    await asyncio.gather(write_task, asyncio.create_task(cross_backend_read()))
    assert cross_backend_read_completed is True

    await backend.close()
    await second_backend.close()
    await backend.close()
    await slow_backend.close()


async def test_async_backend_connection_error_surfaces() -> None:
    """Invalid targets must surface an error instead of silently succeeding."""
    with pytest.raises((OSError, PermissionError, Exception)):
        backend = SQLiteBackend(db_path=Path("/nonexistent/deeply/nested/path/db.db"))
        await backend.get_conversation("test")


async def test_async_read_connection_stays_responsive_during_sync_writer_lock(tmp_path: Path) -> None:
    """Fresh read backends must not take the writer-style path when a DB already exists."""
    db_path = tmp_path / "locked.db"
    with open_connection(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS sentinel(value INTEGER)")
        conn.execute("INSERT INTO sentinel(value) VALUES (1)")
        conn.commit()

    writer_started = threading.Event()
    writer_release = threading.Event()

    def hold_writer_lock() -> None:
        with sqlite3.connect(db_path, timeout=30) as conn:
            conn.execute("BEGIN IMMEDIATE")
            writer_started.set()
            writer_release.wait(timeout=5)
            conn.rollback()

    writer = threading.Thread(target=hold_writer_lock)
    writer.start()
    writer_started.wait(timeout=2)

    backend = SQLiteBackend(db_path=db_path)
    backend._schema_ensured = False
    started = time.perf_counter()
    try:
        records = await asyncio.wait_for(backend.queries.list_conversations(ConversationRecordQuery()), timeout=2)
    finally:
        writer_release.set()
        writer.join(timeout=2)
        await backend.close()

    elapsed = time.perf_counter() - started
    assert records == []
    assert elapsed < 1.5


async def test_paged_id_iteration_contract(tmp_path: Path) -> None:
    """Bounded ID iterators must preserve canonical descending sort order."""
    backend = SQLiteBackend(db_path=tmp_path / "paged.db")
    base = datetime(2026, 3, 10, tzinfo=timezone.utc)

    for idx in range(5):
        await backend.save_raw_conversation(
            make_raw_conversation(
                raw_id=f"raw-{idx}",
                provider_name="chatgpt",
                source_name="inbox-a",
                source_path=f"/tmp/raw-{idx}.json",
                blob_size=len(b'{"id":"x"}'),
                acquired_at=(base + timedelta(minutes=idx)).isoformat(),
            )
        )
        await backend.save_conversation_record(
            make_conversation(
                f"conv-{idx}",
                provider_name="chatgpt",
                title=f"Conversation {idx}",
                sort_key=float(idx),
            )
        )

    assert [raw_id async for raw_id in backend.iter_raw_ids(page_size=2)] == [
        "raw-4",
        "raw-3",
        "raw-2",
        "raw-1",
        "raw-0",
    ]
    assert [cid async for cid in backend.iter_conversation_ids(page_size=2)] == [
        "conv-4",
        "conv-3",
        "conv-2",
        "conv-1",
        "conv-0",
    ]
    assert await backend.count_conversation_ids() == 5
    await backend.close()


async def test_backend_transaction_contracts(tmp_path: Path) -> None:
    """Explicit and context-managed transactions must commit on success and roll back on failure."""
    backend = SQLiteBackend(db_path=tmp_path / "transaction.db")
    conv_rollback = make_conversation("conv-rollback", title="Rollback")
    await backend.begin()
    await backend.save_conversation_record(conv_rollback)
    await backend.rollback()
    assert await backend.get_conversation("conv-rollback") is None

    conv_commit = make_conversation("conv-commit", title="Commit")
    async with backend.transaction():
        await backend.save_conversation_record(conv_commit)
    committed = await backend.get_conversation("conv-commit")
    assert committed is not None
    assert committed.conversation_id == "conv-commit"

    with pytest.raises(ValueError):
        async with backend.transaction():
            await backend.save_conversation_record(make_conversation("conv-error", title="Error"))
            raise ValueError("Test error")
    assert await backend.get_conversation("conv-error") is None
    await backend.close()


async def test_save_conversation_replaces_runtime_rows_on_content_change(tmp_path: Path) -> None:
    """Content changes must replace stale runtime rows instead of accumulating them."""
    backend = SQLiteBackend(db_path=tmp_path / "replace.db")
    repo = ConversationRepository(backend=backend)

    conv_v1 = make_conversation(
        "conv-replace",
        provider_name="codex",
        title="Replace",
        content_hash="hash-v1",
    )
    msg1_v1 = make_message(
        "msg-1",
        "conv-replace",
        text="first",
        content_hash="msg-hash-v1-1",
        content_blocks=[
            make_content_block(
                message_id="msg-1",
                conversation_id="conv-replace",
                block_index=0,
                block_type="text",
                text="alpha",
            ),
            make_content_block(
                message_id="msg-1",
                conversation_id="conv-replace",
                block_index=1,
                block_type="text",
                text="beta",
            ),
        ],
    )
    msg2_v1 = make_message(
        "msg-2",
        "conv-replace",
        role="assistant",
        text="second",
        content_hash="msg-hash-v1-2",
    )
    att1 = make_attachment("att-1", "conv-replace", "msg-1", mime_type="image/png")
    att2 = make_attachment("att-2", "conv-replace", "msg-2", mime_type="image/jpeg")

    await repo.save_conversation(conv_v1, [msg1_v1, msg2_v1], [att1, att2])

    conv_v2 = make_conversation(
        "conv-replace",
        provider_name="codex",
        title="Replace",
        content_hash="hash-v2",
    )
    msg1_v2 = make_message(
        "msg-1",
        "conv-replace",
        text="first updated",
        content_hash="msg-hash-v2-1",
        content_blocks=[
            make_content_block(
                message_id="msg-1",
                conversation_id="conv-replace",
                block_index=0,
                block_type="text",
                text="alpha updated",
            )
        ],
    )
    att1_v2 = make_attachment("att-1", "conv-replace", "msg-1", mime_type="image/png")

    await repo.save_conversation(conv_v2, [msg1_v2], [att1_v2])

    messages = await backend.get_messages("conv-replace")
    assert [message.message_id for message in messages] == ["msg-1"]
    assert messages[0].text == "first updated"
    assert [(block.block_index, block.text) for block in messages[0].content_blocks] == [(0, "alpha updated")]

    attachments = sorted(await backend.get_attachments("conv-replace"), key=lambda record: record.attachment_id)
    assert [(attachment.attachment_id, attachment.message_id) for attachment in attachments] == [("att-1", "msg-1")]

    with open_connection(backend.db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                ("conv-replace",),
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM content_blocks WHERE conversation_id = ?",
                ("conv-replace",),
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM attachment_refs WHERE conversation_id = ?",
                ("conv-replace",),
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT message_id FROM attachment_refs WHERE conversation_id = ? AND attachment_id = ?",
                ("conv-replace", "att-1"),
            ).fetchone()[0]
            == "msg-1"
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM attachments WHERE attachment_id = ?",
                ("att-2",),
            ).fetchone()[0]
            == 0
        )
        stats_row = conn.execute(
            "SELECT message_count, word_count FROM conversation_stats WHERE conversation_id = ?",
            ("conv-replace",),
        ).fetchone()
        assert stats_row is not None
        assert stats_row["message_count"] == 1
        assert stats_row["word_count"] == 2

    await backend.close()


async def test_backend_delete_contracts(tmp_path: Path) -> None:
    """Deleting a conversation must remove rows, attachments, and FTS entries exactly once."""
    from polylogue.storage.index import ensure_index, update_index_for_conversations
    from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync

    backend = SQLiteBackend(db_path=tmp_path / "delete.db")
    repo = ConversationRepository(backend=backend)
    conv = make_conversation("conv-delete", title="Delete")
    msg1 = make_message("msg-1", "conv-delete", text="Hello")
    msg2 = make_message("msg-2", "conv-delete", role="assistant", text="Hi there")
    att = make_attachment("att-1", "conv-delete", "msg-1", mime_type="image/png", size_bytes=1024)

    async with backend.transaction():
        await backend.save_conversation_record(conv)
        await backend.save_messages([msg1, msg2])
        await backend.save_attachments([att])

    with open_connection(backend.db_path) as conn:
        ensure_index(conn)
        update_index_for_conversations(["conv-delete"], conn)
        rebuild_session_insights_sync(conn)
        conn.commit()
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM messages_fts WHERE conversation_id = ?",
                ("conv-delete",),
            ).fetchone()[0]
            > 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM session_profiles WHERE conversation_id = ?",
                ("conv-delete",),
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM session_work_events WHERE conversation_id = ?",
                ("conv-delete",),
            ).fetchone()[0]
            > 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM session_phases WHERE conversation_id = ?",
                ("conv-delete",),
            ).fetchone()[0]
            > 0
        )
        assert conn.execute("SELECT COUNT(*) FROM day_session_summaries").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM session_tag_rollups").fetchone()[0] >= 1

    assert await repo.delete_conversation("conv-delete") is True
    assert await backend.get_conversation("conv-delete") is None
    assert len(await backend.get_messages("conv-delete")) == 0
    assert len(await backend.get_attachments("conv-delete")) == 0
    assert await repo.delete_conversation("conv-delete") is False

    with open_connection(backend.db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM messages_fts WHERE conversation_id = ?",
                ("conv-delete",),
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM session_profiles WHERE conversation_id = ?",
                ("conv-delete",),
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM session_work_events WHERE conversation_id = ?",
                ("conv-delete",),
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM session_phases WHERE conversation_id = ?",
                ("conv-delete",),
            ).fetchone()[0]
            == 0
        )
        assert conn.execute("SELECT COUNT(*) FROM day_session_summaries").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM session_tag_rollups").fetchone()[0] == 0
    await backend.close()


async def test_backend_stream_messages_accepts_generic_role_filter(tmp_path: Path) -> None:
    backend = SQLiteBackend(db_path=tmp_path / "role-filter.db")
    conv = make_conversation("conv-role-filter", title="Role Filter")
    messages = [
        make_message("msg-user", "conv-role-filter", role="user", text="User text"),
        make_message("msg-assistant", "conv-role-filter", role="assistant", text="Assistant text"),
        make_message("msg-tool", "conv-role-filter", role="tool", text="Tool text"),
        make_message("msg-system", "conv-role-filter", role="system", text="System text"),
    ]

    async with backend.transaction():
        await backend.save_conversation_record(conv)
        await backend.save_messages(messages)

    user_messages = [message async for message in backend.iter_messages("conv-role-filter", message_roles=(Role.USER,))]
    dialogue_messages = [message async for message in backend.iter_messages("conv-role-filter", dialogue_only=True)]
    stats = await backend.get_conversation_stats("conv-role-filter")

    assert [message.message_id for message in user_messages] == ["msg-user"]
    assert [message.message_id for message in dialogue_messages] == ["msg-user", "msg-assistant"]
    assert stats["role_user_messages"] == 1
    assert stats["role_assistant_messages"] == 1
    assert stats["role_tool_messages"] == 1
    assert stats["role_system_messages"] == 1

    await backend.close()


async def test_backend_lifecycle_reopen_contract(tmp_path: Path) -> None:
    """Closing a backend must be idempotent and future operations may lazily reopen it."""
    backend = SQLiteBackend(db_path=tmp_path / "lifecycle.db")
    await backend.save_conversation_record(_build_record("conv-life"))
    await backend.close()
    await backend.close()

    async with backend._get_connection() as conn:
        assert conn is not None

    assert await backend.get_conversation("conv-life") is not None
    await backend.close()


@pytest.mark.asyncio
async def test_get_archive_stats_skips_retrieval_band_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = SQLiteBackend(db_path=tmp_path / "archive-stats.db")
    repo = ConversationRepository(backend=backend)
    observed: list[bool] = []

    async def fake_read_embedding_stats(
        conn: aiosqlite.Connection, *, include_retrieval_bands: bool = True
    ) -> EmbeddingStatsSnapshot:
        observed.append(include_retrieval_bands)
        return EmbeddingStatsSnapshot(
            embedded_conversations=0,
            embedded_messages=0,
            pending_conversations=0,
        )

    monkeypatch.setattr(
        "polylogue.storage.repository.vectors.repository_vectors.read_embedding_stats_async", fake_read_embedding_stats
    )

    stats = await repo.get_archive_stats()

    assert stats.total_conversations == 0
    assert observed == [False]
    await backend.close()
