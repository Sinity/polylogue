"""Worker 12 regression sources; exercise public API routes on fresh archives."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.core.enums import AssertionKind
from polylogue.readiness import VerifyStatus
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER, user_write
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.archive_templates import bootstrap_archive_root


async def test_health_check_reports_schema_skew_without_raising(tmp_path: Path) -> None:
    """55.05: schema validation on the diagnostic open raises instead of reporting ERROR."""
    bootstrap_archive_root(tmp_path)
    expected = ARCHIVE_VERSION_BY_TIER[ArchiveTier.AUDIT]
    mismatched = expected + 1
    archive = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    try:
        with sqlite3.connect(tmp_path / "audit.db") as conn:
            conn.execute(f"PRAGMA user_version = {mismatched}")
        report = await archive.health_check()
    finally:
        await archive.close()

    check = next(item for item in report.checks if item.name == "archive_audit")
    assert check.status == VerifyStatus.ERROR
    assert f"v{mismatched}/{expected}" in check.summary
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == mismatched


@pytest.mark.parametrize("limit", [0, 1])
async def test_review_count_does_not_hydrate_off_page_judgments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    """55.10: restoring the unbounded second read hydrates all three off-page judgments."""
    bootstrap_archive_root(tmp_path)
    target = "session:w12-review-target"
    with sqlite3.connect(tmp_path / "user.db") as conn:
        for index in range(3):
            user_write.upsert_assertion(
                conn,
                assertion_id=f"w12-candidate-{index}",
                target_ref=target,
                kind=AssertionKind.LESSON,
                body_text=f"Review candidate {index}",
                author_ref="user:w12-fixture",
                author_kind="user",
                evidence_refs=(),
                status="candidate",
                # The review audit includes expired claims; its count must too.
                staleness={"expires_at_ms": 1} if index == 0 else None,
                now_ms=1_700_000_000_000 + index,
            )
        for suffix, selected_target, kind, status in (
            ("other-target", "session:elsewhere", AssertionKind.LESSON, "candidate"),
            ("other-kind", target, AssertionKind.CORRECTION, "candidate"),
            ("other-status", target, AssertionKind.LESSON, "active"),
        ):
            user_write.upsert_assertion(
                conn,
                assertion_id=f"w12-{suffix}",
                target_ref=selected_target,
                kind=kind,
                body_text="Excluded by the requested review selection",
                author_ref="user:w12-fixture",
                author_kind="user",
                evidence_refs=(),
                status=status,
                now_ms=1_700_000_000_010,
            )

    hydrated: list[str] = []
    original = user_write._latest_candidate_judgment

    def record_judgment_read(conn: sqlite3.Connection, candidate_assertion_id: str):
        hydrated.append(candidate_assertion_id)
        return original(conn, candidate_assertion_id)

    monkeypatch.setattr(user_write, "_latest_candidate_judgment", record_judgment_read)
    archive = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    try:
        payload = await archive.list_assertion_candidate_reviews(
            target_ref=target,
            kinds=(AssertionKind.LESSON,),
            statuses=("candidate",),
            limit=limit,
        )
    finally:
        await archive.close()

    assert payload.total == 3
    assert len(payload.items) == limit
    assert len(hydrated) == limit
