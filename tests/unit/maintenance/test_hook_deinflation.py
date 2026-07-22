"""Retroactive hook-session-inflation repair (polylogue-31r1)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.core.enums import Origin
from polylogue.maintenance.hook_deinflation import repair_hook_session_inflation
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveHookEvent, write_source_raw_session


def _seed_hook_raw(archive: ArchiveStore, *, event_id: str, session_id: str) -> str:
    """Reproduce the OLD buggy state: a hook persisted AS a raw_sessions row."""
    conn = archive._ensure_source_conn()
    source_path = f"/home/u/.local/share/polylogue/hooks/pending/{event_id}.json"
    payload = f'{{"event_id":"{event_id}","event_type":"PostToolUse"}}'.encode()
    return write_source_raw_session(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path=source_path,
        source_index=0,
        payload=payload,
        acquired_at_ms=1,
        hook_event=ArchiveHookEvent(
            hook_event_id=f"hook:{event_id}",
            origin=Origin.CODEX_SESSION,
            source_path=source_path,
            event_type="PostToolUse",
            payload={"event_id": event_id},
            observed_at_ms=1,
            native_id=f"{session_id}:PostToolUse:{event_id}",
            session_native_id=session_id,
        ),
    )


def _seed_real_raw(archive: ArchiveStore, *, native: str) -> str:
    conn = archive._ensure_source_conn()
    return write_source_raw_session(
        conn,
        origin=Origin.CODEX_SESSION,
        source_path=f"/home/u/.codex/sessions/{native}.jsonl",
        source_index=0,
        payload=f'{{"real":"{native}"}}'.encode(),
        acquired_at_ms=1,
        native_id=native,
    )


def _insert_index_session(index_db: Path, *, native: str, raw_id: str, message_count: int) -> None:
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash, raw_id, message_count) VALUES (?,?,?,?,?)",
            (native, "codex-session", b"\x00" * 32, raw_id, message_count),
        )


def test_repair_removes_hook_sessions_keeps_evidence_and_real_sessions(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        hook_raw_a = _seed_hook_raw(archive, event_id="evt-a", session_id="sess-1")
        hook_raw_b = _seed_hook_raw(archive, event_id="evt-b", session_id="sess-1")
        real_raw = _seed_real_raw(archive, native="conv-real")
        archive.commit()

    index_db = tmp_path / "index.db"
    # Two empty shells from hook raws (the bug), one real materialized session,
    # and one genuinely-empty NON-hook session that must survive.
    _insert_index_session(index_db, native="hook-a", raw_id=hook_raw_a, message_count=0)
    _insert_index_session(index_db, native="hook-b", raw_id=hook_raw_b, message_count=0)
    _insert_index_session(index_db, native="conv-real", raw_id=real_raw, message_count=5)
    _insert_index_session(index_db, native="empty-real", raw_id="not-a-hook-raw", message_count=0)

    dry = repair_hook_session_inflation(tmp_path, dry_run=True)
    assert dry.applied is False
    assert dry.hook_raw_sessions == 2
    assert dry.hook_index_sessions == 2
    assert dry.raw_hook_events_before == 2

    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 3  # nothing deleted yet

    report = repair_hook_session_inflation(tmp_path, dry_run=False)
    assert report.applied is True
    assert report.hook_raw_sessions == 2
    assert report.hook_index_sessions == 2
    # Hook EVIDENCE and bytes are retained; only the session rows are gone.
    assert report.raw_hook_events_after == 2
    assert report.hook_blob_refs_retained == 2

    with sqlite3.connect(tmp_path / "source.db") as conn:
        # Only the real raw session remains; hook raw_sessions deleted.
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 1
        assert conn.execute("SELECT source_path FROM raw_sessions").fetchone()[0].endswith("conv-real.jsonl")
        assert conn.execute("SELECT COUNT(*) FROM raw_hook_events").fetchone()[0] == 2

    with sqlite3.connect(index_db) as conn:
        rows = {r[0] for r in conn.execute("SELECT native_id FROM sessions")}
    assert rows == {"conv-real", "empty-real"}  # hook shells gone, real + empty-real kept
