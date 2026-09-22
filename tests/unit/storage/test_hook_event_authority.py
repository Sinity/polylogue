from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.core.enums import Origin, Provider
from polylogue.storage.hook_event_authority import census_hook_event_authority
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveHookEvent


def _write_event(tmp_path: Path, *, event_id: str = "event-1") -> Path:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    with ArchiveStore(archive_root) as archive:
        archive.write_hook_event(
            provider=Provider.CLAUDE_CODE,
            payload=b'{"event_id":"event-1"}',
            source_path="hooks/carriers/claude-code/2026-09-06/4242.ndjson",
            acquired_at_ms=100,
            hook_event=ArchiveHookEvent(
                hook_event_id=f"hook:{event_id}",
                origin=Origin.CLAUDE_CODE_SESSION,
                source_path="hooks/carriers/claude-code/2026-09-06/4242.ndjson",
                event_type="SessionStart",
                payload={"event_id": event_id, "event_type": "SessionStart"},
                observed_at_ms=100,
                native_id="session:SessionStart:event-1",
                session_native_id="session",
            ),
            carrier_source_id="primary",
            carrier_relative_path="pending/event-1.json",
        )
    return archive_root / "source.db"


def test_authority_census_is_clean_for_production_writer(tmp_path: Path) -> None:
    census = census_hook_event_authority(sqlite3.connect(_write_event(tmp_path)))
    assert census.source_sealable
    assert census.to_dict()["dispositions"] == {"clean": 1}
    assert census.carrier_role_counts == {"primary-writable": 1}


def test_authority_census_blocks_missing_inline_payload_and_blob(tmp_path: Path) -> None:
    source_db = _write_event(tmp_path)
    with sqlite3.connect(source_db) as conn:
        conn.execute("UPDATE raw_hook_events SET payload_json = 'not-json', blob_hash = NULL")
        conn.commit()
        census = census_hook_event_authority(conn)
    assert not census.source_sealable
    assert "hook:event-1:malformed-inline-payload" in census.issues
    assert "hook:event-1:missing-blob-hash" in census.issues


def test_authority_census_blocks_writer_reader_payload_disagreement(tmp_path: Path) -> None:
    source_db = _write_event(tmp_path)
    with sqlite3.connect(source_db) as conn:
        conn.execute("UPDATE raw_hook_events SET payload_json = ?", (json.dumps({"changed": True}),))
        conn.commit()
        census = census_hook_event_authority(conn)
    assert census.blocked_count == 1
    assert "hook:event-1:carrier-payload-disagreement" in census.issues


def test_authority_census_reports_missing_carrier_schema(tmp_path: Path) -> None:
    source_db = _write_event(tmp_path)
    with sqlite3.connect(source_db) as conn:
        conn.execute("DROP TABLE hook_event_carriers")
        conn.commit()
        census = census_hook_event_authority(conn)
    assert not census.source_sealable
    assert census.dispositions == {"schema-unavailable": 1}


def test_census_blocks_conflicting_payload_ref(tmp_path: Path) -> None:
    """A contradictory second ``hook_payload`` ref must not read as clean.

    ``blob_refs`` is keyed on ``(blob_hash, ref_id, ref_type)``, so an event can
    carry two hook-payload refs naming different bytes -- the damaged or
    partially reconciled state this census exists to detect. Counting only the
    refs that already matched the logical hash returned 1 and sealed it.

    Anti-vacuity: restoring the ``AND blob_hash = ?`` filter makes the filtered
    count 1 again and this test goes green-on-blocked → red. The opposite
    direction (a check that blocks every event) is pinned by
    ``test_authority_census_is_clean_for_production_writer`` above.
    """
    source_db = _write_event(tmp_path)
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            SELECT randomblob(32), ref_id, ref_type, source_path, size_bytes, acquired_at_ms
            FROM blob_refs WHERE ref_type = 'hook_payload'
            """
        )
        conn.commit()
        census = census_hook_event_authority(conn)
    assert not census.source_sealable
    assert "hook:event-1:blob-ref-disagreement" in census.issues


def test_census_answers_from_one_pinned_generation(tmp_path: Path) -> None:
    """A commit racing the census must not turn a blocked row into a clean one.

    The row scan and the per-row blob/carrier lookups are separate statements.
    In autocommit each takes its own snapshot, so a writer that removed the
    contradiction between them made the census report ``source_sealable`` for a
    generation that was never clean.

    Anti-vacuity: deleting the ``BEGIN``/``rollback`` scope lets the mid-census
    delete become visible and the assertion below flips to sealable. The
    opposite direction -- a census that always blocks -- is pinned by
    ``test_authority_census_is_clean_for_production_writer``.
    """
    source_db = _write_event(tmp_path)
    conn = sqlite3.connect(source_db)
    racer = sqlite3.connect(source_db)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        with conn:
            conn.execute(
                """
                INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
                SELECT randomblob(32), ref_id, ref_type, source_path, size_bytes, acquired_at_ms
                FROM blob_refs WHERE ref_type = 'hook_payload'
                """
            )
        scans = 0

        def _race(statement: str) -> None:
            nonlocal scans
            if "FROM raw_hook_events" not in statement:
                return
            scans += 1
            if scans != 1:
                return
            with racer:
                racer.execute(
                    "DELETE FROM blob_refs WHERE ref_type = 'hook_payload' AND blob_hash NOT IN "
                    "(SELECT blob_hash FROM raw_hook_events WHERE blob_hash IS NOT NULL)"
                )

        conn.set_trace_callback(_race)
        census = census_hook_event_authority(conn)
        conn.set_trace_callback(None)
    finally:
        conn.close()
        racer.close()
    assert scans >= 1, "the census never scanned raw_hook_events"
    assert not census.source_sealable
    assert "hook:event-1:blob-ref-disagreement" in census.issues
