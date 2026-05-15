from __future__ import annotations

import sqlite3
from pathlib import Path

from devtools.daemon_workload_probe import probe
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.schema import _ensure_schema


def test_daemon_workload_probe_reports_attempts_and_plan_shape(tmp_path: Path) -> None:
    db = tmp_path / "archive.sqlite"
    source = tmp_path / "session.jsonl"
    source.write_text('{"a": 1}\n')
    conn = sqlite3.connect(db)
    _ensure_schema(conn)
    conn.execute(
        """
        INSERT INTO raw_conversations (
            raw_id, provider_name, source_path, blob_size, acquired_at
        ) VALUES ('raw-1', 'codex', ?, 0, '2026-01-01T00:00:00Z')
        """,
        (str(source),),
    )
    conn.execute(
        """
        INSERT INTO conversations (
            conversation_id, provider_name, provider_conversation_id,
            source_name, content_hash, version, raw_id
        ) VALUES ('conv-1', 'codex', 'provider-1', 'codex', 'hash-1', 1, 'raw-1')
        """
    )
    conn.commit()
    conn.close()

    cursor = CursorStore(db)
    attempt_id = cursor.begin_ingest_attempt(paths=[source], input_bytes=10, queued_file_count=1)
    cursor.update_ingest_attempt(
        attempt_id,
        status="completed",
        phase="done",
        succeeded_file_count=1,
        source_payload_read_bytes=10,
        cursor_fingerprint_read_bytes=0,
    )

    payload = probe(db)

    assert payload["ok"] is True
    assert payload["attempt_counts"]["total"] == 1
    assert payload["recent_attempts"][0]["read_amplification"] == 1.0
    source_plan = payload["query_plans"]["source_path_lookup"]
    assert source_plan["hazards"] == []
    assert any("idx_raw_conv_source_path_raw_id" in item for item in source_plan["plan"])


def test_daemon_workload_probe_reports_missing_database(tmp_path: Path) -> None:
    payload = probe(tmp_path / "missing.sqlite")

    assert payload["ok"] is False
    assert payload["error"] == "database does not exist"
