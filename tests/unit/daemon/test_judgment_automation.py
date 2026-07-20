"""Judgment automation actor: policy engine + daemon sweep (polylogue-6qjc).

Covers the pure policy engine (`evaluate_candidate`, `parse_judgment_automation_policy`)
directly, then exercises `run_judgment_automation_sweep_once` against a real
`user.db` fixture to prove the sweep actually calls the same
`judge_assertion_candidates` storage chokepoint the MCP `judge` dispatcher
uses (not a private reimplementation), and that the escalated residue lands
as a queryable `handoff` assertion rather than being silently dropped. The
periodic-loop wrapper's dual capability gate
(`judgment_automation_enabled` AND `mcp_judge_enabled`) is covered
separately against the real config resolver plus a stubbed write
coordinator, since that gate is the load-bearing safety property the bead
asks for (automation must not exercise judge authority the operator hasn't
independently granted).
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.core.enums import AssertionKind, AssertionStatus, AssertionVisibility
from polylogue.daemon.judgment_automation import (
    JUDGMENT_AUTOMATION_ACTOR_REF,
    JudgmentAutomationPolicyRule,
    evaluate_candidate,
    parse_judgment_automation_policy,
    periodic_judgment_automation_sweep,
    run_judgment_automation_sweep_once,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import (
    ArchiveAssertionEnvelope,
    list_assertion_claims,
    read_assertion_envelope,
    upsert_assertion,
)

# ---------------------------------------------------------------------------
# Policy engine (pure functions)
# ---------------------------------------------------------------------------


def _candidate(
    *,
    assertion_id: str = "cand-1",
    kind: AssertionKind = AssertionKind.PATHOLOGY,
    confidence: float | None,
) -> ArchiveAssertionEnvelope:
    return ArchiveAssertionEnvelope(
        assertion_id=assertion_id,
        scope_ref=None,
        target_ref="session:s1",
        key=None,
        kind=kind,
        value=None,
        body_text="candidate body",
        author_ref="actor:pathology-detector",
        author_kind="detector",
        evidence_refs=[],
        status=AssertionStatus.CANDIDATE,
        visibility=AssertionVisibility.PRIVATE,
        confidence=confidence,
        staleness=None,
        context_policy={"inject": False},
        supersedes=[],
        created_at_ms=1,
        updated_at_ms=1,
    )


def test_parse_judgment_automation_policy_decodes_and_drops_unknown() -> None:
    raw = {
        "pathology": {"auto_accept_min_confidence": 0.9, "auto_reject_max_confidence": 0.1},
        "not-a-real-kind": {"auto_accept_min_confidence": 0.9},
        "finding": {"auto_accept_min_confidence": 2.0},  # out of [0,1] -> dropped
        "note": "not a mapping",
    }

    policy = parse_judgment_automation_policy(raw)

    assert policy == {
        AssertionKind.PATHOLOGY: JudgmentAutomationPolicyRule(
            auto_accept_min_confidence=0.9, auto_reject_max_confidence=0.1
        )
    }


def test_evaluate_candidate_accepts_above_threshold() -> None:
    policy = {AssertionKind.PATHOLOGY: JudgmentAutomationPolicyRule(auto_accept_min_confidence=0.9)}
    decision = evaluate_candidate(_candidate(confidence=0.95), policy)
    assert decision.decision == "accept"
    assert decision.candidate_ref == "assertion:cand-1"


def test_evaluate_candidate_rejects_below_threshold() -> None:
    policy = {AssertionKind.PATHOLOGY: JudgmentAutomationPolicyRule(auto_reject_max_confidence=0.1)}
    decision = evaluate_candidate(_candidate(confidence=0.05), policy)
    assert decision.decision == "reject"


def test_evaluate_candidate_escalates_without_configured_policy() -> None:
    decision = evaluate_candidate(_candidate(confidence=0.99), {})
    assert decision.decision == "escalate"
    assert "no judgment-automation policy configured" in decision.reason


def test_evaluate_candidate_escalates_without_confidence_signal() -> None:
    policy = {AssertionKind.PATHOLOGY: JudgmentAutomationPolicyRule(auto_accept_min_confidence=0.9)}
    decision = evaluate_candidate(_candidate(confidence=None), policy)
    assert decision.decision == "escalate"
    assert "no confidence signal" in decision.reason


def test_evaluate_candidate_escalates_inside_undecided_band() -> None:
    policy = {
        AssertionKind.PATHOLOGY: JudgmentAutomationPolicyRule(
            auto_accept_min_confidence=0.9, auto_reject_max_confidence=0.1
        )
    }
    decision = evaluate_candidate(_candidate(confidence=0.5), policy)
    assert decision.decision == "escalate"
    assert "undecided band" in decision.reason


# ---------------------------------------------------------------------------
# Sweep against a real user.db fixture
# ---------------------------------------------------------------------------


def _init_user_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        initialize_archive_tier(conn, ArchiveTier.USER)
        conn.commit()
    finally:
        conn.close()


def _insert_candidate(
    root: Path,
    *,
    assertion_id: str,
    kind: AssertionKind,
    confidence: float | None,
    target_ref: str = "session:s1",
) -> None:
    conn = sqlite3.connect(root / "user.db")
    conn.row_factory = sqlite3.Row
    try:
        upsert_assertion(
            conn,
            assertion_id=assertion_id,
            target_ref=target_ref,
            kind=kind,
            body_text=f"{kind.value} candidate",
            author_ref="actor:test-detector",
            author_kind="detector",
            confidence=confidence,
        )
        conn.commit()
    finally:
        conn.close()


def test_sweep_accepts_and_escalates_via_the_real_judge_chokepoint(tmp_path: Path) -> None:
    _init_user_db(tmp_path / "user.db")
    _insert_candidate(tmp_path, assertion_id="cand-accept", kind=AssertionKind.PATHOLOGY, confidence=0.95)
    _insert_candidate(tmp_path, assertion_id="cand-escalate", kind=AssertionKind.FINDING, confidence=0.5)

    policy = {AssertionKind.PATHOLOGY: JudgmentAutomationPolicyRule(auto_accept_min_confidence=0.9)}

    result = run_judgment_automation_sweep_once(tmp_path, batch_limit=200, policy=policy)

    assert result.considered == 2
    assert result.accepted == 1
    assert result.escalated == 1
    assert result.failed == 0

    conn = sqlite3.connect(tmp_path / "user.db")
    conn.row_factory = sqlite3.Row
    try:
        accepted = read_assertion_envelope(conn, "cand-accept")
        assert accepted is not None
        assert accepted.status == AssertionStatus.ACCEPTED

        # The mutation that must make this fail: if the sweep judged through
        # a private reimplementation instead of `judge_assertion_candidates`,
        # no JUDGMENT-kind row referencing the automation actor would exist.
        judgment_rows = conn.execute(
            "SELECT author_ref FROM assertions WHERE kind = ? AND target_ref = ?",
            (AssertionKind.JUDGMENT.value, "assertion:cand-accept"),
        ).fetchall()
        assert any(row["author_ref"] == JUDGMENT_AUTOMATION_ACTOR_REF for row in judgment_rows)

        # Escalated candidate must be untouched (still awaiting a human)...
        escalated = read_assertion_envelope(conn, "cand-escalate")
        assert escalated is not None
        assert escalated.status == AssertionStatus.CANDIDATE

        # ...but discoverable via an explicit handoff pointing back at it.
        handoffs = list_assertion_claims(
            conn,
            kinds=(AssertionKind.HANDOFF,),
            statuses=None,
        )
        assert len(handoffs) == 1
        assert handoffs[0].target_ref == "assertion:cand-escalate"
        # Escalated for missing-policy (finding has none), not the
        # undecided-confidence-band reason.
        assert "undecided" not in (handoffs[0].body_text or "")
    finally:
        conn.close()


def test_sweep_is_idempotent_across_repeated_runs(tmp_path: Path) -> None:
    _init_user_db(tmp_path / "user.db")
    _insert_candidate(tmp_path, assertion_id="cand-escalate", kind=AssertionKind.FINDING, confidence=None)
    policy = {AssertionKind.FINDING: JudgmentAutomationPolicyRule(auto_accept_min_confidence=0.9)}

    run_judgment_automation_sweep_once(tmp_path, batch_limit=200, policy=policy)
    run_judgment_automation_sweep_once(tmp_path, batch_limit=200, policy=policy)

    conn = sqlite3.connect(tmp_path / "user.db")
    conn.row_factory = sqlite3.Row
    try:
        handoffs = conn.execute(
            "SELECT COUNT(*) FROM assertions WHERE kind = ?",
            (AssertionKind.HANDOFF.value,),
        ).fetchone()[0]
        assert handoffs == 1
    finally:
        conn.close()


def test_sweep_is_a_bounded_no_op_without_any_configured_policy(tmp_path: Path) -> None:
    _init_user_db(tmp_path / "user.db")
    _insert_candidate(tmp_path, assertion_id="cand-1", kind=AssertionKind.PATHOLOGY, confidence=0.95)

    result = run_judgment_automation_sweep_once(tmp_path, batch_limit=200, policy={})

    assert result.considered == 0
    conn = sqlite3.connect(tmp_path / "user.db")
    try:
        # The candidate must be untouched: an empty policy must escalate
        # everything implicitly by doing nothing, not silently judge it.
        status = conn.execute("SELECT status FROM assertions WHERE assertion_id = ?", ("cand-1",)).fetchone()[0]
        assert status == AssertionStatus.CANDIDATE.value
    finally:
        conn.close()


def test_sweep_is_a_no_op_without_a_user_db(tmp_path: Path) -> None:
    policy = {AssertionKind.PATHOLOGY: JudgmentAutomationPolicyRule(auto_accept_min_confidence=0.9)}
    result = run_judgment_automation_sweep_once(tmp_path, batch_limit=200, policy=policy)
    assert result.considered == 0


# ---------------------------------------------------------------------------
# Periodic loop: dual capability gate
# ---------------------------------------------------------------------------


class _StopLoopError(Exception):
    """Raised from a patched ``asyncio.sleep`` to end the infinite loop after one tick."""


async def _run_one_tick() -> None:
    # First `asyncio.sleep` call is the tick-pacing wait: let it return
    # normally so the loop body (the capability gate + sweep call) actually
    # runs. The second call -- reached only via `continue` or falling off
    # the end of the loop body -- raises to stop the otherwise-infinite loop
    # after exactly one full iteration.
    with pytest.raises(_StopLoopError):
        with patch("asyncio.sleep", AsyncMock(side_effect=[None, _StopLoopError()])):
            await periodic_judgment_automation_sweep()


@pytest.mark.parametrize(
    ("automation_enabled", "judge_enabled"),
    [(False, True), (True, False), (False, False)],
)
def test_periodic_sweep_never_judges_without_both_capability_flags(
    automation_enabled: bool, judge_enabled: bool
) -> None:
    """The sweep must not exercise judge authority on a single flag alone.

    The mutation that would make this fail: dropping the
    ``cfg.mcp_judge_enabled`` half of the gate in
    ``periodic_judgment_automation_sweep`` (leaving only
    ``judgment_automation_enabled``) would let this test's
    ``(True, False)`` case call the write coordinator.
    """
    cfg = SimpleNamespace(
        judgment_automation_enabled=automation_enabled,
        mcp_judge_enabled=judge_enabled,
        judgment_automation_interval_s=60,
        judgment_automation_batch_limit=200,
        judgment_automation_policy={},
    )
    write_coordinator = AsyncMock()
    with (
        patch("polylogue.daemon.judgment_automation.load_polylogue_config", return_value=cfg),
        patch("polylogue.daemon.write_coordinator.daemon_write_coordinator", return_value=write_coordinator),
    ):
        asyncio.run(_run_one_tick())

    write_coordinator.run_sync.assert_not_called()


def test_periodic_sweep_judges_once_both_capability_flags_are_set(tmp_path: Path) -> None:
    _init_user_db(tmp_path / "user.db")
    _insert_candidate(tmp_path, assertion_id="cand-accept", kind=AssertionKind.PATHOLOGY, confidence=0.95)

    cfg = SimpleNamespace(
        judgment_automation_enabled=True,
        mcp_judge_enabled=True,
        judgment_automation_interval_s=60,
        judgment_automation_batch_limit=200,
        judgment_automation_policy={"pathology": {"auto_accept_min_confidence": 0.9}},
    )
    write_coordinator = AsyncMock()

    async def _fake_run_sync(actor, function, *args, **kwargs):  # type: ignore[no-untyped-def]
        return function(*args, **kwargs)

    write_coordinator.run_sync.side_effect = _fake_run_sync

    with (
        patch("polylogue.daemon.judgment_automation.load_polylogue_config", return_value=cfg),
        patch("polylogue.daemon.write_coordinator.daemon_write_coordinator", return_value=write_coordinator),
        patch("polylogue.paths.archive_root", return_value=tmp_path),
    ):
        asyncio.run(_run_one_tick())

    write_coordinator.run_sync.assert_called_once()
    conn = sqlite3.connect(tmp_path / "user.db")
    try:
        status = conn.execute("SELECT status FROM assertions WHERE assertion_id = ?", ("cand-accept",)).fetchone()[0]
        assert status == AssertionStatus.ACCEPTED.value
    finally:
        conn.close()
