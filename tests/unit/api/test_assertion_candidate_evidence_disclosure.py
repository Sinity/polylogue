"""Evidence-disclosure contracts for candidate judgment review.

The tests traverse the production facade and resolver rather than fabricating
review rows. Anti-vacuity: dropping evidence resolution, removing the five-ref
bound, or allowing one resolver failure to abort the list makes these tests
fail.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.core.enums import AssertionKind, BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion


def _seed_candidate(root: Path) -> tuple[str, str]:
    with ArchiveStore(root) as archive:
        session_id = archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="evidence-review",
                title="Evidence review source",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="The evidence-bearing source message",
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TEXT,
                                text="The evidence-bearing source message",
                            )
                        ],
                    )
                ],
            )
        )
    evidence_refs = (
        f"session:{session_id}",
        "session:missing-evidence",
        "workspace:evidence-review",
        "query:evidence-review",
        "assertion:missing-evidence",
        "repo:polylogue",
        "run:missing-evidence",
    )
    with sqlite3.connect(root / "user.db") as conn:
        upsert_assertion(
            conn,
            assertion_id="candidate-evidence-review",
            target_ref=f"session:{session_id}",
            kind=AssertionKind.LESSON,
            body_text=("Evidence disclosure must remain bounded and explicit. " * 20),
            author_ref="agent:standing-queries",
            author_kind="agent",
            evidence_refs=evidence_refs,
            status="candidate",
            now_ms=1_700_000_000_000,
        )
    return session_id, "assertion:candidate-evidence-review"


async def test_review_discloses_bounded_typed_evidence_previews(tmp_path: Path) -> None:
    session_id, candidate_ref = _seed_candidate(tmp_path)
    archive = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    try:
        payload = await archive.list_assertion_candidate_reviews(
            target_ref=f"session:{session_id}",
            statuses=("candidate",),
            limit=20,
        )
    finally:
        await archive.close()

    assert payload.total == 1
    item = payload.items[0]
    assert item.candidate_ref == candidate_ref
    assert item.review_status == "pending"
    assert item.source_ref == "agent:standing-queries"
    assert item.source_kind == "agent"
    assert len(item.claim_summary) == 480
    assert item.claim_summary.endswith("...")
    assert item.age_ms > 0
    assert item.evidence_total_count == 7
    assert item.evidence_omitted_count == 2
    assert len(item.evidence_previews) == 5
    assert [preview.state for preview in item.evidence_previews] == [
        "resolved",
        "missing",
        "unsupported",
        "pending",
        "missing",
    ]
    assert item.evidence_resolution == "partial"
    resolved = item.evidence_previews[0]
    assert resolved.ref == f"session:{session_id}"
    assert resolved.title == "Evidence review source"
    assert resolved.excerpt == "1 messages"
    assert resolved.open_commands
    assert "polylogue find id:" in resolved.open_commands[0]


async def test_one_evidence_resolution_failure_degrades_only_that_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, _ = _seed_candidate(tmp_path)
    archive = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    original_resolve = archive.resolve_ref

    async def flaky_resolve(ref: str):
        if ref == "query:evidence-review":
            raise RuntimeError("resolver unavailable")
        return await original_resolve(ref)

    monkeypatch.setattr(archive, "resolve_ref", flaky_resolve)
    try:
        payload = await archive.list_assertion_candidate_reviews(
            target_ref=f"session:{session_id}",
            statuses=("candidate",),
            limit=20,
        )
    finally:
        await archive.close()

    item = payload.items[0]
    assert item.evidence_previews[3].state == "error"
    assert item.evidence_previews[3].reason == "resolver unavailable"
    assert item.evidence_previews[0].state == "resolved"
    assert payload.total == 1
