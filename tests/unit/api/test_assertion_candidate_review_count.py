"""A candidate-review page counts its selection without hydrating off-page reviews."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.core.enums import AssertionKind
from polylogue.storage.sqlite.archive_tiers import user_write
from tests.infra.archive_templates import bootstrap_archive_root


@pytest.mark.parametrize("limit", [0, 1])
async def test_review_count_does_not_hydrate_off_page_judgments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    """Fails if ``total`` comes from a second unbounded review read, which hydrates every judgment."""
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
