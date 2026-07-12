from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import cast

import pytest

from devtools.index_fast_forward import (
    FAST_FORWARD_TO_VERSION,
    activate_generation,
    create_and_fast_forward_generation,
    fast_forward_clone,
    rollback_generation,
    validate_clone,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_OLD_MESSAGES_FTS = """
CREATE VIRTUAL TABLE messages_fts USING fts5(
    block_id UNINDEXED, message_id UNINDEXED, session_id UNINDEXED,
    block_type UNINDEXED, text, content='', contentless_delete=1,
    tokenize='unicode61'
)
"""

_OLD_WORK_FTS = """
CREATE VIRTUAL TABLE session_work_events_fts USING fts5(
    event_id UNINDEXED, session_id UNINDEXED, work_event_type UNINDEXED,
    text, tokenize='unicode61'
)
"""

_OLD_THREADS_FTS = """
CREATE VIRTUAL TABLE threads_fts USING fts5(
    thread_id UNINDEXED, root_id UNINDEXED, text, tokenize='unicode61'
)
"""


def _drop_fts(conn: sqlite3.Connection, table: str, triggers: tuple[str, ...]) -> None:
    for trigger in triggers:
        conn.execute(f'DROP TRIGGER IF EXISTS "{trigger}"')
    conn.execute(f'DROP TABLE IF EXISTS "{table}"')


def _downgrade_fts_to_v32(conn: sqlite3.Connection) -> None:
    _drop_fts(conn, "messages_fts", ("messages_fts_ai", "messages_fts_ad", "messages_fts_au"))
    conn.execute(_OLD_MESSAGES_FTS)
    conn.executescript(
        """
        CREATE TRIGGER messages_fts_ai AFTER INSERT ON blocks WHEN new.search_text != '' BEGIN
          INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text)
          VALUES (new.rowid, new.block_id, new.message_id, new.session_id, new.block_type, new.search_text);
        END;
        CREATE TRIGGER messages_fts_ad AFTER DELETE ON blocks WHEN old.search_text != '' BEGIN
          DELETE FROM messages_fts WHERE rowid = old.rowid;
        END;
        CREATE TRIGGER messages_fts_au AFTER UPDATE ON blocks BEGIN
          DELETE FROM messages_fts WHERE rowid = old.rowid;
          INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text)
          SELECT new.rowid, new.block_id, new.message_id, new.session_id, new.block_type, new.search_text
          WHERE new.search_text != '';
        END;
        """
    )
    conn.execute(
        """
        INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text)
        SELECT rowid, block_id, message_id, session_id, block_type, search_text
        FROM blocks WHERE search_text != ''
        """
    )

    _drop_fts(
        conn,
        "session_work_events_fts",
        ("session_work_events_fts_ai", "session_work_events_fts_ad", "session_work_events_fts_au"),
    )
    conn.execute(_OLD_WORK_FTS)
    conn.executescript(
        """
        CREATE TRIGGER session_work_events_fts_ai AFTER INSERT ON session_work_events BEGIN
          INSERT INTO session_work_events_fts(event_id, session_id, work_event_type, text)
          VALUES (new.event_id, new.session_id, new.work_event_type, new.search_text);
        END;
        CREATE TRIGGER session_work_events_fts_ad AFTER DELETE ON session_work_events BEGIN
          DELETE FROM session_work_events_fts WHERE event_id = old.event_id;
        END;
        CREATE TRIGGER session_work_events_fts_au AFTER UPDATE ON session_work_events BEGIN
          DELETE FROM session_work_events_fts WHERE event_id = old.event_id;
          INSERT INTO session_work_events_fts(event_id, session_id, work_event_type, text)
          VALUES (new.event_id, new.session_id, new.work_event_type, new.search_text);
        END;
        INSERT INTO session_work_events_fts(event_id, session_id, work_event_type, text)
        SELECT event_id, session_id, work_event_type, search_text FROM session_work_events;
        """
    )

    _drop_fts(conn, "threads_fts", ("threads_fts_ai", "threads_fts_ad", "threads_fts_au"))
    conn.execute(_OLD_THREADS_FTS)
    conn.executescript(
        """
        CREATE TRIGGER threads_fts_ai AFTER INSERT ON threads BEGIN
          INSERT INTO threads_fts(thread_id, root_id, text)
          VALUES (new.thread_id, new.thread_id, new.search_text);
        END;
        CREATE TRIGGER threads_fts_ad AFTER DELETE ON threads BEGIN
          DELETE FROM threads_fts WHERE thread_id = old.thread_id;
        END;
        CREATE TRIGGER threads_fts_au AFTER UPDATE ON threads BEGIN
          DELETE FROM threads_fts WHERE thread_id = old.thread_id;
          INSERT INTO threads_fts(thread_id, root_id, text)
          VALUES (new.thread_id, new.thread_id, new.search_text);
        END;
        INSERT INTO threads_fts(thread_id, root_id, text)
        SELECT thread_id, thread_id, search_text FROM threads;
        """
    )


def _downgrade_insight_check_to_v32(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE insight_materialization_v32 (
            insight_type TEXT NOT NULL CHECK(insight_type IN (
                'session_profile', 'work_events', 'phases', 'latency', 'thread',
                'runs', 'observed_events', 'context_snapshots'
            )),
            session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            materializer_version INTEGER NOT NULL,
            materialized_at_ms INTEGER NOT NULL,
            source_updated_at_ms INTEGER,
            source_sort_key_ms INTEGER,
            input_high_water_mark_ms INTEGER,
            input_high_water_mark_source TEXT,
            input_row_count INTEGER NOT NULL DEFAULT 0 CHECK(input_row_count >= 0),
            PRIMARY KEY(insight_type, session_id)
        ) STRICT;
        INSERT INTO insight_materialization_v32
        SELECT * FROM insight_materialization;
        DROP TABLE insight_materialization;
        ALTER TABLE insight_materialization_v32 RENAME TO insight_materialization;
        """
    )


def _seed_v32_fixture(path: Path) -> dict[str, int]:
    initialize_archive_database(path, ArchiveTier.INDEX)
    digest = b"x" * 32
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            "INSERT INTO sessions(native_id, origin, title, content_hash) VALUES (?, ?, ?, ?)",
            ("v32-fixture", "chatgpt-export", "fixture", digest),
        )
        session_id = str(conn.execute("SELECT session_id FROM sessions").fetchone()[0])
        conn.execute(
            """
            INSERT INTO messages(session_id, native_id, position, role, material_origin, content_hash)
            VALUES (?, 'message-1', 0, 'user', 'human_authored', ?)
            """,
            (session_id, digest),
        )
        message_id = str(conn.execute("SELECT message_id FROM messages").fetchone()[0])
        conn.execute(
            """
            INSERT INTO blocks(message_id, session_id, position, block_type, text, content_hash)
            VALUES (?, ?, 0, 'text', 'Zażółć gęślą jaźń — ŁATWO', ?)
            """,
            (message_id, session_id, digest),
        )
        conn.execute(
            """
            INSERT INTO threads(thread_id, search_text)
            VALUES (?, 'Zażółć gęślą jaźń — ŁATWO')
            """,
            (session_id,),
        )
        conn.execute(
            """
            INSERT INTO session_work_events(session_id, position, work_event_type, summary, search_text)
            VALUES (?, 0, 'implementation', 'fixture', 'Zażółć gęślą jaźń — ŁATWO')
            """,
            (session_id,),
        )
        conn.execute(
            """
            INSERT INTO insight_materialization(
                insight_type, session_id, materializer_version, materialized_at_ms
            ) VALUES ('session_profile', ?, 1, 1)
            """,
            (session_id,),
        )
        conn.commit()
        _downgrade_fts_to_v32(conn)
        _downgrade_insight_check_to_v32(conn)
        conn.execute("DROP INDEX IF EXISTS idx_web_constructs_message")
        conn.execute("DROP VIEW IF EXISTS delegations")
        conn.execute("CREATE VIEW delegations AS SELECT 'legacy' AS mapping_state")
        conn.execute("PRAGMA user_version = 32")
        conn.commit()
        counts = {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("sessions", "messages", "blocks", "session_work_events", "threads")
        }
    return counts


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fast_forward_v32_fixture_applies_exact_deltas_without_raw_reparse(tmp_path: Path) -> None:
    clone = tmp_path / "index.db"
    expected_counts = _seed_v32_fixture(clone)
    receipt_path = tmp_path / "receipt.json"

    receipt = fast_forward_clone(
        clone,
        receipt_path,
        max_io_full_avg10=1000,
        max_memory_full_avg10=1000,
        batch_rows=1,
    )

    assert receipt["status"] == "clone_ready"
    assert receipt["raw_reparse"] is False
    validation = validate_clone(
        clone,
        expected_counts=expected_counts,
        expected_version=FAST_FORWARD_TO_VERSION,
    )
    assert validation["ready"] is True
    assert validation["count_drift"] == {}
    assert validation["ddl_drift"] == {}
    assert validation["foreign_key_violations"] == 0
    assert validation["quick_check"] == ["ok"]
    fts = cast(dict[str, dict[str, object]], validation["fts"])
    fold_smoke = cast(dict[str, dict[str, object]], validation["fold_smoke"])
    assert all(bool(surface["ready"]) for surface in fts.values())
    assert all(bool(smoke["matched"]) for smoke in fold_smoke.values())
    with sqlite3.connect(clone) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 35
        assert conn.execute("SELECT 1 FROM insight_materialization WHERE insight_type='session_profile'").fetchone()
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_web_constructs_message'"
        ).fetchone()
        delegation_columns = [row[1] for row in conn.execute("PRAGMA table_info(delegations)")]
        assert "instruction_tool_use_block_id" in delegation_columns
        for fts_table in ("messages_fts", "session_work_events_fts", "threads_fts"):
            assert conn.execute(f"SELECT 1 FROM {fts_table} WHERE {fts_table} MATCH 'latwo'").fetchone()


def test_fast_forward_failure_keeps_user_version_and_source_counts(tmp_path: Path) -> None:
    source = tmp_path / "source-v32.db"
    expected_counts = _seed_v32_fixture(source)
    source_hash = _sha256(source)
    clone = tmp_path / "failed-clone.db"
    shutil.copy2(source, clone)

    with pytest.raises(RuntimeError, match="injected failure"):
        fast_forward_clone(
            clone,
            tmp_path / "failed-receipt.json",
            max_io_full_avg10=1000,
            max_memory_full_avg10=1000,
            batch_rows=1,
            fail_after_stage="v34-index-and-delegations",
        )

    assert _sha256(source) == source_hash
    with sqlite3.connect(clone) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 32
        assert {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in expected_counts
        } == expected_counts
    failed_receipt = json.loads((tmp_path / "failed-receipt.json").read_text())
    assert failed_receipt["status"] == "failed"


def test_generation_activation_and_rollback_only_swap_symlink(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    active_dir = archive_root / ".index-generations" / "gen-v32"
    active_dir.mkdir(parents=True)
    active_db = active_dir / "index.db"
    _seed_v32_fixture(active_db)
    active_hash = _sha256(active_db)
    active_link = archive_root / "index.db"
    active_link.symlink_to(active_db)
    receipt_path = archive_root / "recovery" / "v35.json"

    receipt = create_and_fast_forward_generation(
        active_link,
        receipt_path,
        max_io_full_avg10=1000,
        max_memory_full_avg10=1000,
        batch_rows=1,
    )

    clone = Path(str(receipt["clone_path"]))
    assert active_link.resolve() == active_db.resolve()
    assert _sha256(active_db) == active_hash
    activate_generation(receipt_path)
    assert active_link.resolve() == clone.resolve()
    assert _sha256(active_db) == active_hash
    rollback_generation(receipt_path)
    assert active_link.resolve() == active_db.resolve()
    assert _sha256(active_db) == active_hash


def test_receipt_hash_tamper_blocks_activation(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    active_dir = archive_root / ".index-generations" / "gen-v32"
    active_dir.mkdir(parents=True)
    active_db = active_dir / "index.db"
    _seed_v32_fixture(active_db)
    active_link = archive_root / "index.db"
    active_link.symlink_to(active_db)
    receipt_path = archive_root / "recovery" / "v35.json"
    create_and_fast_forward_generation(
        active_link,
        receipt_path,
        max_io_full_avg10=1000,
        max_memory_full_avg10=1000,
        batch_rows=1,
    )
    payload = json.loads(receipt_path.read_text())
    payload["clone_generation_id"] = "tampered"
    receipt_path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="receipt hash mismatch"):
        activate_generation(receipt_path)
    assert active_link.resolve() == active_db.resolve()


def test_activation_reuses_receipt_proof_and_rejects_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    active_dir = archive_root / ".index-generations" / "gen-v32"
    active_dir.mkdir(parents=True)
    active_db = active_dir / "index.db"
    _seed_v32_fixture(active_db)
    active_link = archive_root / "index.db"
    active_link.symlink_to(active_db)
    receipt_path = archive_root / "recovery" / "v35.json"
    receipt = create_and_fast_forward_generation(
        active_link,
        receipt_path,
        max_io_full_avg10=1000,
        max_memory_full_avg10=1000,
        batch_rows=1,
    )
    clone = Path(str(receipt["clone_path"]))
    from devtools import index_fast_forward as fast_forward_module

    assert receipt["clone_identity_after"] == asdict(fast_forward_module.file_identity(clone))
    assert cast(list[int], cast(dict[str, object], receipt["validation"])["final_checkpoint"])[0] == 0
    monkeypatch.setattr(
        fast_forward_module,
        "validate_clone",
        lambda *args, **kwargs: pytest.fail("activation must reuse the receipt proof"),
    )
    activate_generation(receipt_path)
    assert active_link.resolve() == clone.resolve()
    rollback_generation(receipt_path)

    os.utime(clone, ns=(clone.stat().st_atime_ns, clone.stat().st_mtime_ns + 1))
    payload = json.loads(receipt_path.read_text())
    payload["status"] = "clone_ready"
    payload["receipt_payload_sha256"] = fast_forward_module._receipt_hash(payload)
    receipt_path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="clone file identity changed"):
        activate_generation(receipt_path)
    assert active_link.resolve() == active_db.resolve()


def test_failed_restart_contract_automatically_restores_v32_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    active_dir = archive_root / ".index-generations" / "gen-v32"
    active_dir.mkdir(parents=True)
    active_db = active_dir / "index.db"
    _seed_v32_fixture(active_db)
    active_link = archive_root / "index.db"
    active_link.symlink_to(active_db)
    receipt_path = archive_root / "recovery" / "v35.json"
    receipt = create_and_fast_forward_generation(
        active_link,
        receipt_path,
        max_io_full_avg10=1000,
        max_memory_full_avg10=1000,
        batch_rows=1,
    )
    clone = Path(str(receipt["clone_path"]))
    from devtools import index_fast_forward as fast_forward_module

    service_checks = iter((False, True, True))
    monkeypatch.setattr(
        fast_forward_module,
        "_service_active",
        lambda service: next(service_checks, True),
    )
    monkeypatch.setattr(fast_forward_module, "_service_remains_stable", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr=""),
    )

    with pytest.raises(RuntimeError, match="v32 symlink was restored"):
        activate_generation(receipt_path, service="fake.service", restart=True)
    assert clone.exists()
    assert active_link.resolve() == active_db.resolve()
    rolled_back = json.loads(receipt_path.read_text())
    assert rolled_back["status"] == "rolled_back_after_failed_activation"
