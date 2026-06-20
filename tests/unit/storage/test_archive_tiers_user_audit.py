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


def test_user_overlay_audit_reports_assertion_backed_surfaces(tmp_path: Path) -> None:
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
        upsert_assertion(
            conn,
            assertion_id="decision",
            target_ref="session:s1",
            kind=AssertionKind.DECISION,
            status="active",
            now_ms=5,
        )
        upsert_assertion(
            conn,
            assertion_id="tag-assertion",
            target_ref="session:s1",
            kind=AssertionKind.TAG,
            status="active",
            now_ms=6,
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

    decisions = _surface(payload, "decisions")
    assert decisions["storage"] == "assertions"
    assert decisions["assertion_kind"] == "decision"
    assert decisions["active_count"] == 1

    tag_assertions = _surface(payload, "tag_assertions")
    assert tag_assertions["storage"] == "assertions"
    assert tag_assertions["assertion_kind"] == "tag"
    assert tag_assertions["active_count"] == 1


def test_user_overlay_audit_covers_every_assertion_kind(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        payload = audit_user_overlay_storage(conn).to_dict()
    finally:
        conn.close()

    surfaces = payload["surfaces"]
    assert isinstance(surfaces, list)
    assertion_kinds = {
        surface["assertion_kind"]
        for surface in surfaces
        if isinstance(surface, dict) and surface["storage"] == "assertions"
    }
    assert assertion_kinds == {kind.value for kind in AssertionKind}
