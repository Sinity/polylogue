from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_audit import audit_user_overlay_storage
from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, mark_assertion_status, upsert_assertion


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    initialize_archive_tier(conn, ArchiveTier.USER)
    return conn


def _surface(payload: Mapping[str, object], name: str) -> Mapping[str, object]:
    surfaces = payload["surfaces"]
    assert isinstance(surfaces, list)
    for surface in surfaces:
        assert isinstance(surface, dict)
        if surface["name"] == name:
            return surface
    raise AssertionError(f"missing audit surface {name}")


def test_user_overlay_audit_reports_assertion_and_table_backed_surfaces(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        upsert_assertion(
            conn,
            assertion_id="mark-active",
            target_ref="session:s1",
            kind=AssertionKind.MARK,
            status="active",
            now_ms=1,
        )
        upsert_assertion(
            conn,
            assertion_id="mark-deleted",
            target_ref="session:s2",
            kind=AssertionKind.MARK,
            status="active",
            now_ms=2,
        )
        mark_assertion_status(conn, "mark-deleted", "deleted", now_ms=3)
        upsert_assertion(
            conn,
            assertion_id="candidate",
            target_ref="session:s1",
            kind=AssertionKind.TRANSFORM_CANDIDATE,
            status="candidate",
            now_ms=4,
        )
        conn.execute(
            """
            INSERT INTO session_tags (session_id, tag, tag_source, method, confidence, evidence_json)
            VALUES ('s1', 'important', 'user', NULL, NULL, NULL)
            """
        )
        conn.execute(
            """
            INSERT INTO session_metadata (session_id, key, value_json, created_at_ms, updated_at_ms)
            VALUES ('s1', 'owner', '{"name":"sinity"}', 5, 5)
            """
        )
        conn.commit()

        payload = audit_user_overlay_storage(conn).to_dict()
    finally:
        conn.close()

    marks = _surface(payload, "marks")
    assert marks["storage"] == "assertions"
    assert marks["assertion_kind"] == "mark"
    assert marks["total_count"] == 2
    assert marks["active_count"] == 1
    assert marks["deleted_count"] == 1

    candidates = _surface(payload, "transform_candidates")
    assert candidates["storage"] == "assertions"
    assert candidates["candidate_count"] == 1

    tags = _surface(payload, "session_tags")
    assert tags["storage"] == "table"
    assert tags["assertion_kind"] is None
    assert tags["total_count"] == 1
    assert "metadata table" in str(tags["rationale"])

    metadata = _surface(payload, "session_metadata")
    assert metadata["storage"] == "table"
    assert metadata["total_count"] == 1

    assert payload["legacy_tables_present"] == []
