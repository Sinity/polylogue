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
            source_path="hooks/pending/event-1.json",
            acquired_at_ms=100,
            hook_event=ArchiveHookEvent(
                hook_event_id=f"hook:{event_id}",
                origin=Origin.CLAUDE_CODE_SESSION,
                source_path="hooks/pending/event-1.json",
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
