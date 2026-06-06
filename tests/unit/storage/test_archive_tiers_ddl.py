from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    ARCHIVE_TIER_SPECS,
    initialize_archive_database,
    initialize_archive_tier,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_HASH = b"x" * 32


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _apply_tier(conn: sqlite3.Connection, tier: ArchiveTier) -> None:
    try:
        initialize_archive_tier(conn, tier)
    except RuntimeError as exc:
        if tier is ArchiveTier.EMBEDDINGS and "sqlite-vec" in str(exc):
            pytest.skip(str(exc))
        raise


@pytest.mark.parametrize("tier", list(ArchiveTier))
def test_archive_tiers_tier_ddl_builds_fresh_database(tmp_path: Path, tier: ArchiveTier) -> None:
    conn = _connect(tmp_path / f"{tier.value}.db")

    _apply_tier(conn, tier)

    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_archive_tiers_generated_ids_are_unique_not_primary_keys() -> None:
    """SQLite 3.51 rejects generated columns in PKs; archive pins the real fallback."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    with pytest.raises(sqlite3.OperationalError, match="generated columns cannot be part of the PRIMARY KEY"):
        conn.execute(
            """
            CREATE TABLE invalid_generated_pk (
                native_id TEXT NOT NULL,
                generated_id TEXT GENERATED ALWAYS AS ('origin:' || native_id) STORED,
                PRIMARY KEY(generated_id)
            ) STRICT
            """
        )

    _apply_tier(conn, ArchiveTier.INDEX)
    index_rows = conn.execute("PRAGMA index_list('sessions')").fetchall()
    unique_column_sets = {
        tuple(info["name"] for info in conn.execute(f"PRAGMA index_info({row['name']!r})").fetchall())
        for row in index_rows
        if row["unique"] == 1
    }
    assert ("session_id",) in unique_column_sets


def test_archive_tiers_index_generates_ids_and_actions_view(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    _apply_tier(conn, ArchiveTier.INDEX)

    conn.execute(
        """
        INSERT INTO sessions (
            native_id, origin, title, content_hash, created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("native-session", "codex-session", "Schema work", _HASH, 1_767_225_600_000, 1_767_225_601_000),
    )
    session = conn.execute("SELECT session_id, sort_key_ms FROM sessions").fetchone()
    assert session["session_id"] == "codex-session:native-session"
    assert session["sort_key_ms"] == 1_767_225_601_000

    conn.execute(
        """
        INSERT INTO messages (
            session_id, native_id, position, role, message_type, content_hash, occurred_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (session["session_id"], "native-message", 0, "assistant", "message", _HASH, 1_767_225_601_000),
    )
    conn.execute(
        """
        INSERT INTO messages (
            session_id, native_id, position, variant_index, role, message_type, content_hash
        ) VALUES (?, NULL, ?, ?, ?, ?, ?)
        """,
        (session["session_id"], 1, 0, "assistant", "message", _HASH),
    )
    conn.execute(
        """
        INSERT INTO messages (
            session_id, native_id, position, variant_index, role, message_type, content_hash
        ) VALUES (?, NULL, ?, ?, ?, ?, ?)
        """,
        (session["session_id"], 1, 1, "assistant", "message", _HASH),
    )
    messages = conn.execute("SELECT message_id FROM messages ORDER BY position, variant_index").fetchall()
    assert [row["message_id"] for row in messages] == [
        "codex-session:native-session:native-message",
        "codex-session:native-session:1.0",
        "codex-session:native-session:1.1",
    ]

    conn.execute(
        """
        INSERT INTO blocks (
            message_id, session_id, position, block_type, text
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (messages[0]["message_id"], session["session_id"], 0, "text", "needle prose"),
    )
    conn.execute(
        """
        INSERT INTO blocks (
            message_id, session_id, position, block_type, tool_name, tool_id, tool_input
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            messages[0]["message_id"],
            session["session_id"],
            1,
            "tool_use",
            "shell",
            "tool-1",
            '{"command": "pytest -q", "path": "tests"}',
        ),
    )
    conn.execute(
        """
        INSERT INTO blocks (
            message_id, session_id, position, block_type, text, tool_id
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (messages[0]["message_id"], session["session_id"], 2, "tool_result", "passed", "tool-1"),
    )

    blocks = conn.execute("SELECT block_id, tool_command, tool_path FROM blocks ORDER BY position").fetchall()
    assert [row["block_id"] for row in blocks] == [
        "codex-session:native-session:native-message:0",
        "codex-session:native-session:native-message:1",
        "codex-session:native-session:native-message:2",
    ]
    assert blocks[1]["tool_command"] == "pytest -q"
    assert blocks[1]["tool_path"] == "tests"

    action = conn.execute("SELECT tool_command, output_text FROM actions").fetchone()
    assert dict(action) == {"tool_command": "pytest -q", "output_text": "passed"}

    fts_row = conn.execute("SELECT block_id FROM blocks_fts WHERE blocks_fts MATCH 'needle'").fetchone()
    assert fts_row["block_id"] == "codex-session:native-session:native-message:0"


def test_archive_tiers_user_ops_and_embeddings_do_not_reference_index_tables() -> None:
    non_index = (
        ARCHIVE_DDL_BY_TIER[ArchiveTier.EMBEDDINGS],
        ARCHIVE_DDL_BY_TIER[ArchiveTier.USER],
        ARCHIVE_DDL_BY_TIER[ArchiveTier.OPS],
    )
    for ddl in non_index:
        assert "REFERENCES sessions" not in ddl
        assert "REFERENCES messages" not in ddl
        assert "REFERENCES blocks" not in ddl


def test_archive_tiers_messages_store_prose_only_in_blocks(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    _apply_tier(conn, ArchiveTier.INDEX)

    message_columns = {row["name"] for row in conn.execute("PRAGMA table_info('messages')").fetchall()}
    block_columns = {row["name"] for row in conn.execute("PRAGMA table_info('blocks')").fetchall()}

    assert "text" not in message_columns
    assert "text" in block_columns


def test_archive_tiers_blocks_fts_uses_external_content_over_blocks(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    _apply_tier(conn, ArchiveTier.INDEX)

    fts_sql = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'blocks_fts'").fetchone()[
        "sql"
    ]

    assert "content='blocks'" in fts_sql
    assert "content_rowid='rowid'" in fts_sql


def test_archive_tiers_sessions_raw_id_is_cross_tier_pointer_not_foreign_key(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    _apply_tier(conn, ArchiveTier.INDEX)

    foreign_keys = conn.execute("PRAGMA foreign_key_list('sessions')").fetchall()

    assert not any(row["from"] == "raw_id" and row["table"] == "raw_sessions" for row in foreign_keys)


def test_archive_tiers_database_split_keeps_source_index_embeddings_user_and_ops_separate() -> None:
    tier_tables: dict[ArchiveTier, set[str]] = {}
    for tier in ArchiveTier:
        conn = sqlite3.connect(":memory:")
        _apply_tier(conn, tier)
        tier_tables[tier] = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
    source = tier_tables[ArchiveTier.SOURCE]
    index = tier_tables[ArchiveTier.INDEX]
    embeddings = tier_tables[ArchiveTier.EMBEDDINGS]
    user = tier_tables[ArchiveTier.USER]
    ops = tier_tables[ArchiveTier.OPS]

    for expected in ("raw_sessions", "blob_refs", "raw_artifacts"):
        assert expected in source
        assert expected not in index
    for expected in ("sessions", "messages", "blocks"):
        assert expected in index
        assert expected not in source
    for expected in ("message_embeddings", "embeddings_meta", "embedding_status"):
        assert expected in embeddings
        assert expected not in index
    for expected in ("marks", "suppressions"):
        assert expected in user
        assert expected not in index
    for expected in ("ingest_cursor", "otlp_spans"):
        assert expected in ops
        assert expected not in index
    assert "sessions" not in user
    assert "sessions" not in ops


def test_archive_tiers_session_profiles_record_cost_price_basis(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    _apply_tier(conn, ArchiveTier.INDEX)

    columns = {row["name"] for row in conn.execute("PRAGMA table_info('session_profiles')").fetchall()}

    assert {"cost_credits", "cost_usd", "cost_provenance", "priced_with", "priced_at_ms"} <= columns


def test_archive_tier_specs_capture_file_and_backup_policy() -> None:
    assert {tier: spec.filename for tier, spec in ARCHIVE_TIER_SPECS.items()} == {
        ArchiveTier.SOURCE: "source.db",
        ArchiveTier.INDEX: "index.db",
        ArchiveTier.EMBEDDINGS: "embeddings.db",
        ArchiveTier.USER: "user.db",
        ArchiveTier.OPS: "ops.db",
    }
    assert {tier.value for tier, spec in ARCHIVE_TIER_SPECS.items() if spec.backup_required} == {
        "source",
        "embeddings",
        "user",
    }


@pytest.mark.parametrize("tier", list(ArchiveTier))
def test_archive_tiers_bootstrap_sets_user_version(tmp_path: Path, tier: ArchiveTier) -> None:
    db_path = tmp_path / ARCHIVE_TIER_SPECS[tier].filename
    conn = _connect(db_path)

    _apply_tier(conn, tier)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == ARCHIVE_TIER_SPECS[tier].version


def test_archive_tiers_database_bootstrap_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "user.db"

    initialize_archive_database(path, ArchiveTier.USER)

    conn = _connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == ARCHIVE_TIER_SPECS[ArchiveTier.USER].version
