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
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.core.enums import AssertionKind, AssertionStatus, AssertionVisibility
from polylogue.daemon.events import emit_daemon_event
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
    ArchiveAssertionBulkJudgmentItemEnvelope,
    ArchiveAssertionEnvelope,
    judge_assertion_candidates,
    list_assertion_candidates,
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


def _init_ops_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        initialize_archive_tier(conn, ArchiveTier.OPS)
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


def test_sweep_receipt_and_bounded_retry_drain(tmp_path: Path) -> None:
    _init_user_db(tmp_path / "user.db")
    _init_ops_db(tmp_path / "ops.db")
    for index in range(3):
        _insert_candidate(tmp_path, assertion_id=f"cand-{index}", kind=AssertionKind.PATHOLOGY, confidence=0.95)
    policy = {AssertionKind.PATHOLOGY: JudgmentAutomationPolicyRule(auto_accept_min_confidence=0.9)}

    first = run_judgment_automation_sweep_once(tmp_path, batch_limit=2, policy=policy, now_ms=1_000)
    second = run_judgment_automation_sweep_once(tmp_path, batch_limit=2, policy=policy, now_ms=2_000)

    assert (first.accepted, first.considered) == (2, 2)
    assert (second.accepted, second.considered) == (1, 1)
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row = conn.execute(
            """
            SELECT ts_ms, payload_json
            FROM daemon_events
            WHERE kind = 'judgment-automation'
            ORDER BY ts_ms DESC, rowid DESC
            LIMIT 1
            """
        ).fetchone()
    assert row is not None
    receipt = json.loads(row[1])
    assert receipt["status"] == "completed"
    assert row[0] == 2_000
    assert receipt["accepted"] == 1


def test_empty_policy_receipt_parks_pending_queue_for_safe_retry(tmp_path: Path) -> None:
    _init_user_db(tmp_path / "user.db")
    _init_ops_db(tmp_path / "ops.db")
    _insert_candidate(tmp_path, assertion_id="cand-parked", kind=AssertionKind.PATHOLOGY, confidence=0.95)

    result = run_judgment_automation_sweep_once(tmp_path, batch_limit=200, policy={}, now_ms=3_000)

    assert result.considered == 0
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row = conn.execute(
            """
            SELECT ts_ms, payload_json
            FROM daemon_events
            WHERE kind = 'judgment-automation'
            ORDER BY ts_ms DESC, rowid DESC
            LIMIT 1
            """
        ).fetchone()
    assert row is not None
    receipt = json.loads(row[1])
    assert receipt["status"] == "parked"
    assert receipt["reason"] == "policy_empty"
    assert receipt["retryable"] is True


def test_failed_sweep_rolls_back_user_judgment_and_records_retryable_receipt(tmp_path: Path) -> None:
    _init_user_db(tmp_path / "user.db")
    _init_ops_db(tmp_path / "ops.db")
    _insert_candidate(tmp_path, assertion_id="cand-failed", kind=AssertionKind.PATHOLOGY, confidence=0.95)
    policy = {AssertionKind.PATHOLOGY: JudgmentAutomationPolicyRule(auto_accept_min_confidence=0.9)}

    with patch(
        "polylogue.storage.sqlite.archive_tiers.user_write.judge_assertion_candidates",
        side_effect=RuntimeError("forced scheduler failure"),
    ):
        with pytest.raises(RuntimeError, match="forced scheduler failure"):
            run_judgment_automation_sweep_once(tmp_path, batch_limit=200, policy=policy, now_ms=4_000)

    with sqlite3.connect(tmp_path / "user.db") as conn:
        assert conn.execute("SELECT status FROM assertions WHERE assertion_id = 'cand-failed'").fetchone() == (
            AssertionStatus.CANDIDATE.value,
        )
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row = conn.execute(
            """
            SELECT ts_ms, payload_json
            FROM daemon_events
            WHERE kind = 'judgment-automation'
            ORDER BY ts_ms DESC, rowid DESC
            LIMIT 1
            """
        ).fetchone()
    assert row is not None
    receipt = json.loads(row[1])
    assert receipt["status"] == "failed"
    assert receipt["retryable"] is True


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


async def _run_ticks(count: int) -> None:
    with pytest.raises(_StopLoopError):
        with patch("asyncio.sleep", AsyncMock(side_effect=[None] * count + [_StopLoopError()])):
            await periodic_judgment_automation_sweep()


@pytest.mark.parametrize(
    ("automation_enabled", "judge_enabled"),
    [(False, True), (True, False), (False, False)],
)
def test_periodic_sweep_never_judges_without_both_capability_flags(
    automation_enabled: bool, judge_enabled: bool, tmp_path: Path
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
        patch("polylogue.paths.archive_root", return_value=tmp_path / "nonexistent-test-archive"),
    ):
        asyncio.run(_run_one_tick())

    write_coordinator.run_sync.assert_not_called()


def test_periodic_sweep_judges_once_both_capability_flags_are_set(tmp_path: Path) -> None:
    _init_user_db(tmp_path / "user.db")
    _init_ops_db(tmp_path / "ops.db")
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


def test_periodic_reloads_config_after_sleep_and_serializes_parked_receipts(tmp_path: Path) -> None:
    _init_user_db(tmp_path / "user.db")
    _init_ops_db(tmp_path / "ops.db")
    disabled_cfg = SimpleNamespace(
        judgment_automation_enabled=False,
        mcp_judge_enabled=True,
        judgment_automation_interval_s=60,
        judgment_automation_batch_limit=200,
        judgment_automation_policy={},
    )
    enabled_cfg = SimpleNamespace(
        judgment_automation_enabled=True,
        mcp_judge_enabled=True,
        judgment_automation_interval_s=60,
        judgment_automation_batch_limit=200,
        judgment_automation_policy={"pathology": {"auto_accept_min_confidence": 0.9}},
    )
    write_coordinator = AsyncMock()

    async def _fake_run_sync(actor, function, *args, **kwargs):  # type: ignore[no-untyped-def]
        return function(*args, **kwargs)

    with (
        patch(
            "polylogue.daemon.judgment_automation.load_polylogue_config",
            # Tick one remains disabled. The flag flips during tick two's
            # sleep, so the post-sleep reload must run the enabled route.
            # The extra values cover the sweep's policy reload and the next
            # tick's pre-sleep snapshot before the stop sentinel is reached.
            side_effect=[disabled_cfg, disabled_cfg, disabled_cfg, enabled_cfg, enabled_cfg, enabled_cfg],
        ),
        patch("polylogue.daemon.write_coordinator.daemon_write_coordinator", return_value=write_coordinator),
        patch("polylogue.paths.archive_root", return_value=tmp_path),
    ):
        write_coordinator.run_sync.side_effect = _fake_run_sync
        asyncio.run(_run_ticks(2))

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        rows = conn.execute(
            """
            SELECT operation_id, payload_json
            FROM daemon_events
            WHERE kind = 'judgment-automation'
            ORDER BY id ASC
            """
        ).fetchall()
    assert len(rows) == 2
    receipts = [json.loads(row[1]) for row in rows]
    assert [receipt["status"] for receipt in receipts] == ["parked", "completed"]
    assert receipts[0]["reason"] == "capability_gate_disabled"
    assert receipts[1]["reason"] == "queue_empty"
    assert rows[0][0] is None
    assert str(rows[1][0]).startswith("judgment-automation:")
    assert [call.args[0] for call in write_coordinator.run_sync.await_args_list] == [
        "maintenance.judgment_automation.receipt",
        "maintenance.judgment_automation",
    ]


def test_periodic_coalesces_identical_disabled_receipt_through_default_interval(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A normal default-interval disabled tick does not duplicate telemetry."""

    _init_ops_db(tmp_path / "ops.db")
    default_interval_s = 60 * 60
    cfg = SimpleNamespace(
        judgment_automation_enabled=False,
        mcp_judge_enabled=True,
        judgment_automation_interval_s=default_interval_s,
        judgment_automation_batch_limit=200,
        judgment_automation_policy={},
    )
    write_coordinator = AsyncMock()
    base_ms = 1_800_000_000_000

    async def _fake_run_sync(actor, function, *args, **kwargs):  # type: ignore[no-untyped-def]
        return function(*args, **kwargs)

    with (
        patch("polylogue.daemon.judgment_automation.load_polylogue_config", return_value=cfg),
        patch("polylogue.daemon.write_coordinator.daemon_write_coordinator", return_value=write_coordinator),
        patch("polylogue.paths.archive_root", return_value=tmp_path),
        patch(
            "polylogue.daemon.events.current_epoch_ms",
            side_effect=[base_ms, base_ms + default_interval_s * 1000],
        ),
    ):
        write_coordinator.run_sync.side_effect = _fake_run_sync
        asyncio.run(_run_ticks(2))

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        rows = conn.execute(
            "SELECT ts_ms, payload_json FROM daemon_events WHERE kind = 'judgment-automation' ORDER BY id"
        ).fetchall()

    assert write_coordinator.run_sync.await_count == 2
    assert len(rows) == 1
    assert rows[0][0] == base_ms
    assert json.loads(rows[0][1])["reason"] == "capability_gate_disabled"
    assert "parked receipt was not persisted" not in caplog.text


def test_periodic_preserves_sweep_failure_reason_when_receipt_write_fails(tmp_path: Path) -> None:
    _init_user_db(tmp_path / "user.db")
    _init_ops_db(tmp_path / "ops.db")
    _insert_candidate(
        tmp_path,
        assertion_id="cand-sweep-failure-receipt",
        kind=AssertionKind.PATHOLOGY,
        confidence=0.95,
    )
    cfg = SimpleNamespace(
        judgment_automation_enabled=True,
        mcp_judge_enabled=True,
        judgment_automation_interval_s=60,
        judgment_automation_batch_limit=200,
        judgment_automation_policy={"pathology": {"auto_accept_min_confidence": 0.9}},
    )
    write_coordinator = AsyncMock()
    receipt_writes = 0

    async def _fake_run_sync(actor, function, *args, **kwargs):  # type: ignore[no-untyped-def]
        return function(*args, **kwargs)

    def _fail_detailed_receipt_once(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal receipt_writes
        receipt_writes += 1
        if receipt_writes == 1:
            raise RuntimeError("forced detailed receipt write failure")
        return emit_daemon_event(*args, **kwargs)

    with (
        patch("polylogue.daemon.judgment_automation.load_polylogue_config", return_value=cfg),
        patch("polylogue.daemon.write_coordinator.daemon_write_coordinator", return_value=write_coordinator),
        patch("polylogue.paths.archive_root", return_value=tmp_path),
        patch(
            "polylogue.storage.sqlite.archive_tiers.user_write.judge_assertion_candidates",
            side_effect=RuntimeError("forced sweep failure"),
        ),
        patch(
            "polylogue.daemon.events.emit_daemon_event",
            side_effect=_fail_detailed_receipt_once,
        ),
    ):
        write_coordinator.run_sync.side_effect = _fake_run_sync
        asyncio.run(_run_one_tick())

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row = conn.execute(
            "SELECT payload_json FROM daemon_events WHERE kind = 'judgment-automation' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row is not None
    receipt = json.loads(row[0])
    assert receipt["status"] == "failed"
    assert receipt["reason"] == "sweep_failed:RuntimeError"
    assert receipt["retryable"] is True
    assert receipt_writes == 2
    assert write_coordinator.run_sync.await_count == 2


def test_periodic_inner_failure_keeps_detailed_receipt_as_authoritative(tmp_path: Path) -> None:
    _init_user_db(tmp_path / "user.db")
    _init_ops_db(tmp_path / "ops.db")
    _insert_candidate(tmp_path, assertion_id="cand-periodic-failed", kind=AssertionKind.PATHOLOGY, confidence=0.95)
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

    with (
        patch("polylogue.daemon.judgment_automation.load_polylogue_config", return_value=cfg),
        patch("polylogue.daemon.write_coordinator.daemon_write_coordinator", return_value=write_coordinator),
        patch("polylogue.paths.archive_root", return_value=tmp_path),
        patch(
            "polylogue.storage.sqlite.archive_tiers.user_write.judge_assertion_candidates",
            side_effect=RuntimeError("forced detailed scheduler failure"),
        ),
    ):
        write_coordinator.run_sync.side_effect = _fake_run_sync
        asyncio.run(_run_one_tick())

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        rows = conn.execute(
            """
            SELECT operation_id, payload_json
            FROM daemon_events
            WHERE kind = 'judgment-automation'
            ORDER BY id DESC
            """
        ).fetchall()
    assert len(rows) == 1
    receipt = json.loads(rows[0][1])
    assert receipt["status"] == "failed"
    assert receipt["reason"] == "sweep_failed:RuntimeError"
    assert rows[0][0].startswith("judgment-automation:")


def test_periodic_receipt_write_failure_uses_one_serialized_fallback(tmp_path: Path) -> None:
    """A failed detailed receipt cannot turn a successful sweep into success.

    Anti-vacuity: patching out the receipt-context check or restoring the
    direct generic write would either leave no receipt after the injected
    first write failure or bypass the coordinator on the fallback call.
    """
    _init_user_db(tmp_path / "user.db")
    _init_ops_db(tmp_path / "ops.db")
    _insert_candidate(tmp_path, assertion_id="cand-receipt-accept", kind=AssertionKind.PATHOLOGY, confidence=0.95)
    _insert_candidate(tmp_path, assertion_id="cand-receipt-reject", kind=AssertionKind.PATHOLOGY, confidence=0.05)
    _insert_candidate(tmp_path, assertion_id="cand-receipt-escalate", kind=AssertionKind.FINDING, confidence=0.5)
    _insert_candidate(tmp_path, assertion_id="cand-receipt-fail", kind=AssertionKind.NOTE, confidence=0.5)
    _insert_candidate(tmp_path, assertion_id="cand-receipt-idempotent", kind=AssertionKind.PATHOLOGY, confidence=0.95)
    idempotent_policy = {AssertionKind.PATHOLOGY: JudgmentAutomationPolicyRule(auto_accept_min_confidence=0.9)}
    with sqlite3.connect(tmp_path / "user.db") as conn:
        candidate = read_assertion_envelope(conn, "cand-receipt-idempotent")
        assert candidate is not None
        decision = evaluate_candidate(candidate, idempotent_policy)
        judge_result = judge_assertion_candidates(
            conn,
            (
                ArchiveAssertionBulkJudgmentItemEnvelope(
                    candidate_ref=decision.candidate_ref,
                    decision=decision.decision,
                    reason=decision.reason,
                    actor_ref=JUDGMENT_AUTOMATION_ACTOR_REF,
                ),
            ),
            now_ms=1,
        )
        assert judge_result.applied_count == 1
        conn.commit()
    cfg = SimpleNamespace(
        judgment_automation_enabled=True,
        mcp_judge_enabled=True,
        judgment_automation_interval_s=60,
        judgment_automation_batch_limit=200,
        judgment_automation_policy={
            "pathology": {"auto_accept_min_confidence": 0.9, "auto_reject_max_confidence": 0.1},
            "finding": {"auto_accept_min_confidence": 0.9},
            "note": {"auto_accept_min_confidence": 0.0},
        },
    )
    write_coordinator = AsyncMock()

    async def _fake_run_sync(actor, function, *args, **kwargs):  # type: ignore[no-untyped-def]
        return function(*args, **kwargs)

    write_coordinator.run_sync.side_effect = _fake_run_sync
    receipt_writes = 0
    real_judge_assertion_candidates = judge_assertion_candidates
    real_list_assertion_candidates = list_assertion_candidates

    def _mixed_judge(conn, items, *, now_ms=None):  # type: ignore[no-untyped-def]
        mutated_items = tuple(
            replace(item, decision="invalid") if item.candidate_ref == "assertion:cand-receipt-fail" else item
            for item in items
        )
        return real_judge_assertion_candidates(conn, mutated_items, now_ms=now_ms)

    def _mixed_list(conn, *, limit=None, **kwargs):  # type: ignore[no-untyped-def]
        candidates = real_list_assertion_candidates(conn, limit=limit, **kwargs)
        idempotent_candidate = read_assertion_envelope(conn, "cand-receipt-idempotent")
        assert idempotent_candidate is not None
        return [*candidates, idempotent_candidate]

    def _fail_first_receipt_write(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal receipt_writes
        receipt_writes += 1
        if receipt_writes == 1:
            raise RuntimeError("forced receipt write failure")
        return emit_daemon_event(*args, **kwargs)

    with (
        patch("polylogue.daemon.judgment_automation.load_polylogue_config", return_value=cfg),
        patch("polylogue.daemon.write_coordinator.daemon_write_coordinator", return_value=write_coordinator),
        patch("polylogue.paths.archive_root", return_value=tmp_path),
        patch(
            "polylogue.daemon.events.emit_daemon_event",
            side_effect=_fail_first_receipt_write,
        ),
        patch(
            "polylogue.storage.sqlite.archive_tiers.user_write.judge_assertion_candidates",
            side_effect=_mixed_judge,
        ),
        patch(
            "polylogue.storage.sqlite.archive_tiers.user_write.list_assertion_candidates",
            side_effect=_mixed_list,
        ),
    ):
        asyncio.run(_run_one_tick())

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        rows = conn.execute(
            """
            SELECT operation_id, payload_json
            FROM daemon_events
            WHERE kind = 'judgment-automation'
            ORDER BY id ASC
            """
        ).fetchall()
    assert len(rows) == 1
    receipt = json.loads(rows[0][1])
    assert receipt["status"] == "failed"
    assert receipt["reason"] == "receipt_persistence_degraded"
    assert receipt["receipt_persistence_degraded"] is True
    assert receipt["retryable"] is True
    assert receipt["considered"] == 5
    assert receipt["accepted"] == 1
    assert receipt["rejected"] == 1
    assert receipt["escalated"] == 1
    assert receipt["idempotent"] == 1
    assert receipt["failed"] == 1
    with sqlite3.connect(tmp_path / "user.db") as conn:
        statuses = dict(
            conn.execute(
                "SELECT assertion_id, status FROM assertions WHERE assertion_id LIKE 'cand-receipt-%'"
            ).fetchall()
        )
        assert statuses == {
            "cand-receipt-accept": AssertionStatus.ACCEPTED.value,
            "cand-receipt-reject": AssertionStatus.REJECTED.value,
            "cand-receipt-escalate": AssertionStatus.CANDIDATE.value,
            "cand-receipt-fail": AssertionStatus.CANDIDATE.value,
            "cand-receipt-idempotent": AssertionStatus.ACCEPTED.value,
        }
    assert rows[0][0].startswith("judgment-automation:")
    assert receipt_writes == 2
    assert write_coordinator.run_sync.await_count == 2


def test_periodic_coordinator_failure_gets_one_serialized_fallback_receipt(tmp_path: Path) -> None:
    _init_ops_db(tmp_path / "ops.db")
    cfg = SimpleNamespace(
        judgment_automation_enabled=True,
        mcp_judge_enabled=True,
        judgment_automation_interval_s=60,
        judgment_automation_batch_limit=200,
        judgment_automation_policy={},
    )
    write_coordinator = AsyncMock()

    calls = 0

    async def _fake_run_sync(actor, function, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("coordinator unavailable")
        return function(*args, **kwargs)

    write_coordinator.run_sync.side_effect = _fake_run_sync

    with (
        patch("polylogue.daemon.judgment_automation.load_polylogue_config", return_value=cfg),
        patch("polylogue.daemon.write_coordinator.daemon_write_coordinator", return_value=write_coordinator),
        patch("polylogue.paths.archive_root", return_value=tmp_path),
    ):
        asyncio.run(_run_one_tick())

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row = conn.execute(
            """
            SELECT operation_id, payload_json
            FROM daemon_events
            WHERE kind = 'judgment-automation'
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    assert row is not None
    receipt = json.loads(row[1])
    assert receipt["status"] == "failed"
    assert receipt["reason"] == "sweep_exception"
    assert row[0].startswith("judgment-automation:")
    assert calls == 2


def test_periodic_false_coordinator_outcome_uses_retryable_fallback(tmp_path: Path) -> None:
    _init_ops_db(tmp_path / "ops.db")
    cfg = SimpleNamespace(
        judgment_automation_enabled=True,
        mcp_judge_enabled=True,
        judgment_automation_interval_s=60,
        judgment_automation_batch_limit=200,
        judgment_automation_policy={},
    )
    write_coordinator = AsyncMock()
    calls = 0

    async def _fake_run_sync(actor, function, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            return False
        return function(*args, **kwargs)

    write_coordinator.run_sync.side_effect = _fake_run_sync

    with (
        patch("polylogue.daemon.judgment_automation.load_polylogue_config", return_value=cfg),
        patch("polylogue.daemon.write_coordinator.daemon_write_coordinator", return_value=write_coordinator),
        patch("polylogue.paths.archive_root", return_value=tmp_path),
    ):
        asyncio.run(_run_one_tick())

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row = conn.execute(
            "SELECT payload_json FROM daemon_events WHERE kind = 'judgment-automation' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row is not None
    receipt = json.loads(row[0])
    assert receipt["status"] == "failed"
    assert receipt["reason"] == "sweep_exception"
    assert receipt["retryable"] is True
    assert calls == 2


def test_periodic_false_fallback_is_logged_as_unpersisted(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _init_ops_db(tmp_path / "ops.db")
    cfg = SimpleNamespace(
        judgment_automation_enabled=True,
        mcp_judge_enabled=True,
        judgment_automation_interval_s=60,
        judgment_automation_batch_limit=200,
        judgment_automation_policy={},
    )
    write_coordinator = AsyncMock()
    write_coordinator.run_sync.side_effect = [RuntimeError("coordinator unavailable"), False]

    with (
        patch("polylogue.daemon.judgment_automation.load_polylogue_config", return_value=cfg),
        patch("polylogue.daemon.write_coordinator.daemon_write_coordinator", return_value=write_coordinator),
        patch("polylogue.paths.archive_root", return_value=tmp_path),
    ):
        asyncio.run(_run_one_tick())

    assert write_coordinator.run_sync.await_count == 2
    assert "failure receipt fallback was not persisted" in caplog.text
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM daemon_events").fetchone() == (0,)


def test_periodic_disabled_receipt_failure_is_logged_and_retried(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _init_ops_db(tmp_path / "ops.db")
    cfg = SimpleNamespace(
        judgment_automation_enabled=False,
        mcp_judge_enabled=True,
        judgment_automation_interval_s=60,
        judgment_automation_batch_limit=200,
        judgment_automation_policy={},
    )
    write_coordinator = AsyncMock()

    attempts = 0

    async def _run_sync(actor, function, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("ops tier unavailable")
        return function(*args, **kwargs)

    write_coordinator.run_sync.side_effect = _run_sync

    with (
        patch("polylogue.daemon.judgment_automation.load_polylogue_config", return_value=cfg),
        patch("polylogue.daemon.write_coordinator.daemon_write_coordinator", return_value=write_coordinator),
        patch("polylogue.paths.archive_root", return_value=tmp_path),
    ):
        asyncio.run(_run_ticks(2))

    assert write_coordinator.run_sync.await_count == 2
    assert "parked receipt write failed; retrying next tick" in caplog.text
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row = conn.execute(
            "SELECT payload_json FROM daemon_events WHERE kind = 'judgment-automation' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row is not None
    assert json.loads(row[0])["reason"] == "capability_gate_disabled"
