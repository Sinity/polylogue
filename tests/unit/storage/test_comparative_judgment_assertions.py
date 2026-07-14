"""Storage integration tests for comparative-judgment assertion rows (rxdo.9.11)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.core.enums import AssertionKind, AssertionStatus, ComparativeVerdict
from polylogue.insights.judgment.comparative import build_comparative_judgment
from polylogue.insights.judgment.types import JudgeIdentity
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import (
    list_assertion_claims,
    list_comparative_judgments,
    upsert_comparative_judgment_assertion,
)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    initialize_archive_tier(conn, ArchiveTier.USER)
    return conn


def test_operator_judgment_is_stored_active_and_injectable_gate_reflects_user_authorship(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        judgment = build_comparative_judgment(
            items=["finding:a", "finding:b"],
            dimension="correctness",
            verdict=ComparativeVerdict.PREFER_LEFT,
            judge=JudgeIdentity(actor_ref="user:local", execution_context_id="operator"),
            blinded=True,
            rubric_id="rubric-1",
            rubric_version=1,
            decided_at_ms=1000,
        )
        envelope = upsert_comparative_judgment_assertion(conn, judgment, author_kind="user")
        conn.commit()

        assert envelope.kind == AssertionKind.COMPARATIVE_JUDGMENT
        assert envelope.status == AssertionStatus.ACTIVE
        assert envelope.assertion_id == judgment.judgment_id
        assert set(envelope.evidence_refs) == {"finding:a", "finding:b"}
    finally:
        conn.close()


def test_agent_judgment_stays_candidate_pending_promotion(tmp_path: Path) -> None:
    """AC (rxdo.9.15 spine): agent judgments stay candidates; promotion still gated."""

    conn = _connect(tmp_path / "user.db")
    try:
        judgment = build_comparative_judgment(
            items=["finding:a", "finding:b"],
            dimension="correctness",
            verdict=ComparativeVerdict.PREFER_RIGHT,
            judge=JudgeIdentity(actor_ref="agent:sonnet", execution_context_id="ctx-a"),
            blinded=True,
            rubric_id="rubric-1",
            rubric_version=1,
            decided_at_ms=2000,
        )
        envelope = upsert_comparative_judgment_assertion(conn, judgment, author_kind="agent")
        conn.commit()

        assert envelope.status == AssertionStatus.CANDIDATE
        assert envelope.context_policy.get("inject") is False
        assert envelope.context_policy.get("promotion_required") is True

        claims = list_assertion_claims(
            conn, kinds=(AssertionKind.COMPARATIVE_JUDGMENT,), statuses=(AssertionStatus.CANDIDATE,)
        )
        assert any(claim.assertion_id == judgment.judgment_id for claim in claims)
    finally:
        conn.close()


def test_round_trips_through_list_comparative_judgments(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        judgment = build_comparative_judgment(
            items=["finding:a", "finding:b", "finding:c"],
            dimension="usefulness",
            verdict=["finding:c", "finding:a", "finding:b"],
            judge=JudgeIdentity(actor_ref="user:local", execution_context_id="operator"),
            blinded=False,
            rubric_id="rubric-2",
            rubric_version=3,
            decided_at_ms=3000,
            evidence_refs=["session:s1"],
            rationale="c wins because it fixed the root cause",
            rationale_visible=True,
        )
        upsert_comparative_judgment_assertion(conn, judgment, author_kind="user")
        conn.commit()

        restored = list_comparative_judgments(conn)
        assert len(restored) == 1
        got = restored[0]
        assert got.judgment_id == judgment.judgment_id
        assert got.items == judgment.items
        assert got.verdict == judgment.verdict
        assert got.is_ordering
        assert got.dimension == "usefulness"
        assert got.judge == judgment.judge
        assert got.rationale == "c wins because it fixed the root cause"
        assert got.evidence_refs == ("session:s1",)
    finally:
        conn.close()


def test_identical_verdict_written_twice_is_idempotent_not_duplicative(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "user.db")
    try:
        judgment = build_comparative_judgment(
            items=["finding:a", "finding:b"],
            dimension="correctness",
            verdict=ComparativeVerdict.TIE,
            judge=JudgeIdentity(actor_ref="user:local", execution_context_id="operator"),
            blinded=True,
            rubric_id="rubric-1",
            rubric_version=1,
            decided_at_ms=4000,
        )
        first = upsert_comparative_judgment_assertion(conn, judgment, author_kind="user")
        conn.commit()
        second = upsert_comparative_judgment_assertion(conn, judgment, author_kind="user")
        conn.commit()
        assert first.assertion_id == second.assertion_id
        assert len(list_comparative_judgments(conn)) == 1
    finally:
        conn.close()
