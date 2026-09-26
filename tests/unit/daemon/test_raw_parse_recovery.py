"""Interrupted intake recovery through the canonical raw-observation derivation.

An interrupted cursor records an interrupted attempt and rewinds a source that
outran retained raw bytes. Recovery then comes from the common derivation's
authoritative required/inspect relation. It does not depend on a raw stage or
on disposable convergence debt.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.operations.raw_observation_derivation import converge_raw_observations
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root


def _write_unparsed_raw(archive_root: Path, *, source_path: Path, native_id: str) -> str:
    payload = [
        {
            "id": native_id,
            "title": native_id,
            "create_time": 1,
            "current_node": "message-1",
            "mapping": {
                "message-1": {
                    "id": "message-1",
                    "parent": None,
                    "children": [],
                    "message": {
                        "id": "message-1",
                        "author": {"role": "user"},
                        "create_time": 1,
                        "content": {"content_type": "text", "parts": [f"retained {native_id}"]},
                    },
                }
            },
        }
    ]
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        return archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=json.dumps(payload).encode(),
            source_path=str(source_path),
            acquired_at_ms=1,
        )


def _sessions_for_raw(archive_root: Path, raw_id: str) -> list[tuple[str, str]]:
    with sqlite3.connect(archive_root / "index.db") as conn:
        return [
            (str(native_id), str(found_raw_id))
            for native_id, found_raw_id in conn.execute(
                "SELECT native_id, raw_id FROM sessions WHERE raw_id = ? ORDER BY native_id", (raw_id,)
            )
        ]


def test_interrupted_attempt_is_recorded_without_raw_derivation_debt(tmp_path: Path) -> None:
    """Attempt telemetry remains observable without becoming recovery authority."""
    bootstrap_archive_root(tmp_path)
    source_path = tmp_path / "sources" / "interrupted.json"
    source_path.parent.mkdir()
    source_path.write_text("placeholder")
    store = CursorStore(tmp_path / "live.sqlite", ops_db_path=tmp_path / "ops.db")
    store.begin_ingest_attempt(paths=[source_path], input_bytes=1, queued_file_count=1)

    CursorStore(tmp_path / "live.sqlite", ops_db_path=tmp_path / "ops.db")

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        assert conn.execute(
            "SELECT status, phase FROM ingest_attempts ORDER BY started_at_ms DESC LIMIT 1"
        ).fetchone() == ("interrupted", "interrupted")
        assert conn.execute("SELECT COUNT(*) FROM convergence_debt WHERE stage = 'raw_parse_recovery'").fetchone() == (
            0,
        )


def test_restart_rewinds_cursor_then_common_raw_derivation_recovers_retained_bytes(tmp_path: Path) -> None:
    """A fresh pass reconstructs pending raw work after cursor and ops-state loss.

    Anti-vacuity: changing the raw adapter to trust cursor/debt telemetry, or
    omitting its required/inspect pass after restart, leaves the retained raw
    without its logical session.
    """
    bootstrap_archive_root(tmp_path)
    source_path = tmp_path / "sources" / "cursor-ahead.json"
    source_path.parent.mkdir()
    source_path.write_text("placeholder")
    raw_id = _write_unparsed_raw(tmp_path, source_path=source_path, native_id="cursor-ahead")

    store = CursorStore(tmp_path / "live.sqlite", ops_db_path=tmp_path / "ops.db")
    store.set(
        source_path,
        source_path.stat().st_size,
        byte_offset=source_path.stat().st_size,
        last_complete_newline=source_path.stat().st_size,
        parser_fingerprint="test-parser",
        content_fingerprint="claimed-complete",
        tail_hash="claimed-complete",
    )
    store.begin_ingest_attempt(paths=[source_path], input_bytes=source_path.stat().st_size, queued_file_count=1)

    restarted_store = CursorStore(tmp_path / "live.sqlite", ops_db_path=tmp_path / "ops.db")
    cursor = restarted_store.get_record(source_path)
    assert cursor is not None
    assert (cursor.byte_offset, cursor.last_complete_newline) == (0, 0)
    assert cursor.content_fingerprint is None
    assert cursor.tail_hash is None
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM convergence_debt WHERE stage = 'raw_parse_recovery'").fetchone() == (
            0,
        )

    recovered = converge_raw_observations(
        tmp_path,
        source_roots=(source_path.parent,),
        limit=1,
    )
    assert recovered.done == recovered.work.published == 1
    assert recovered.failed == recovered.pending == 0
    assert _sessions_for_raw(tmp_path, raw_id) == [("cursor-ahead", raw_id)]


def test_common_raw_derivation_restart_recovers_output_loss_without_ops_hints(tmp_path: Path) -> None:
    """Output loss and a later admission are both rediscovered after restart."""
    bootstrap_archive_root(tmp_path)
    source_root = tmp_path / "sources"
    source_root.mkdir()
    first = _write_unparsed_raw(tmp_path, source_path=source_root / "first.json", native_id="first")
    initial = converge_raw_observations(tmp_path, source_roots=(source_root,), limit=1)
    assert initial.done == 1 and initial.failed == initial.pending == 0

    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("DELETE FROM sessions WHERE raw_id = ?", (first,))
        conn.commit()
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute("DELETE FROM convergence_debt")
        conn.commit()
    second = _write_unparsed_raw(tmp_path, source_path=source_root / "second.json", native_id="second")

    restarted = converge_raw_observations(tmp_path, source_roots=(source_root,), limit=2)
    assert restarted.done == restarted.work.published == 2
    assert restarted.failed == restarted.pending == 0
    assert _sessions_for_raw(tmp_path, first) == [("first", first)]
    assert _sessions_for_raw(tmp_path, second) == [("second", second)]
