"""Judgment automation actor: policy engine + trigger surface (polylogue-6qjc).

Most agent-authored assertion candidates will never be seen by a human --
the MCP `judge` dispatcher (polylogue-800m) is bulk-shaped but every
candidate defaults to waiting for an explicit human decision, which the
operator has said does not scale. This module adds the missing automated
actor: a periodic daemon sweep that

1. lists `CANDIDATE`-status assertions eligible for judgment
   (:func:`polylogue.storage.sqlite.archive_tiers.user_write.list_assertion_candidates`),
2. runs a small per-kind confidence policy engine over each one
   (:func:`evaluate_candidate`) that only ever answers accept/reject when
   the policy is unambiguous, defaulting to escalate otherwise, and
3. calls the *same* `judge_assertion_candidates` storage chokepoint the MCP
   `judge` dispatcher uses to apply accept/reject decisions, then writes an
   explicit `handoff`-kind assertion pointing at every escalated candidate
   so the residue is a queryable review queue instead of silent limbo.

Off by default (`judgment_automation_enabled`). Because this sweep exercises
the same judge write authority as the MCP `judge` dispatcher, it additionally
requires `mcp_judge_enabled` (polylogue-800m's independent capability
boundary) -- turning on the sweep's own opt-in must not, by itself, grant
judge authority the operator hasn't separately confirmed.
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from polylogue.config import JUDGMENT_AUTOMATION_BATCH_LIMIT_DEFAULT, load_polylogue_config
from polylogue.core.enums import AssertionKind, AssertionStatus
from polylogue.logging import get_logger
from polylogue.sources.live.sqlite_locking import is_transient_sqlite_lock

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.user_write import ArchiveAssertionEnvelope

logger = get_logger(__name__)

#: Actor identity recorded on every automated judgment/handoff so the
#: audit trail (``assertions.author_ref``) can distinguish automated
#: decisions from a human operator's.
JUDGMENT_AUTOMATION_ACTOR_REF = "actor:judgment-automation"

#: Non-``"user"`` author kind for escalation handoff rows -- routes through
#: the normal ``upsert_assertion`` promotion gate like any other automated
#: writer (polylogue-37t.15); ``handoff`` is not in
#: ``ASSERTION_CANDIDATE_JUDGMENT_KINDS`` so this can never create a
#: candidate the sweep would try to judge again.
JUDGMENT_AUTOMATION_AUTHOR_KIND = "automation"

#: Floor on the configured sweep interval so a misconfigured
#: ``judgment_automation_interval_s`` cannot turn this into a busy loop.
JUDGMENT_AUTOMATION_SWEEP_INTERVAL_FLOOR_SECONDS = 60

JUDGMENT_AUTOMATION_STAGE = "judgment-automation"
JUDGMENT_AUTOMATION_RECEIPT_GRACE_MIN_SECONDS = 5 * 60
JUDGMENT_AUTOMATION_RECEIPT_GRACE_MAX_SECONDS = 60 * 60
JudgmentAutomationReceiptStatus = Literal["completed", "parked", "failed"]

JudgmentAutomationDecisionKind = Literal["accept", "reject", "escalate"]


class JudgmentAutomationReceiptOutcome(StrEnum):
    """Outcome of the scheduler receipt operation."""

    PERSISTED = "persisted"
    COALESCED = "coalesced"
    FAILED = "failed"


def _judgment_automation_receipt_coalescing_horizon_ms(interval_s: int) -> int:
    """Return the bounded horizon for coalescing an identical scheduler receipt."""

    return judgment_automation_receipt_freshness_window_ms(interval_s)


def judgment_automation_receipt_freshness_window_ms(interval_s: int, *, parked: bool = False) -> int:
    """Return the bounded health window for one scheduler receipt.

    A parked receipt is coalesced at the next cadence boundary, so it remains
    truthful through that boundary and the following bounded grace period.
    """

    effective_interval_s = max(interval_s, JUDGMENT_AUTOMATION_SWEEP_INTERVAL_FLOOR_SECONDS)
    grace_s = min(
        max(effective_interval_s // 10, JUDGMENT_AUTOMATION_RECEIPT_GRACE_MIN_SECONDS),
        JUDGMENT_AUTOMATION_RECEIPT_GRACE_MAX_SECONDS,
    )
    cadence_count = 2 if parked else 1
    return (cadence_count * effective_interval_s + grace_s) * 1000


@dataclass(frozen=True, slots=True)
class JudgmentAutomationPolicyRule:
    """Per-kind confidence gate: below/above these bounds is auto-judgeable."""

    auto_accept_min_confidence: float | None = None
    auto_reject_max_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class JudgmentAutomationDecision:
    """One policy-engine verdict for one candidate."""

    candidate_ref: str
    decision: JudgmentAutomationDecisionKind
    reason: str


@dataclass(frozen=True, slots=True)
class JudgmentAutomationSweepResult:
    """Bounded, secret-safe summary of one sweep for daemon status/logging."""

    considered: int = 0
    accepted: int = 0
    rejected: int = 0
    escalated: int = 0
    idempotent: int = 0
    failed: int = 0


class _JudgmentAutomationReceiptPersistenceError(RuntimeError):
    """The scheduler could not persist the receipt for an attempted tick."""

    def __init__(
        self,
        message: str,
        *,
        result: JudgmentAutomationSweepResult | None = None,
        status: JudgmentAutomationReceiptStatus | None = None,
        reason: str | None = None,
        user_tier_committed: bool = False,
    ) -> None:
        super().__init__(message)
        self.result = result
        self.status = status
        self.reason = reason
        self.user_tier_committed = user_tier_committed


@dataclass(slots=True)
class _JudgmentAutomationReceiptContext:
    operation_id: str
    recorded: bool = False


def _receipt_state(payload: object) -> tuple[object, ...] | None:
    if not isinstance(payload, dict):
        return None
    return tuple(
        payload.get(key)
        for key in ("status", "reason", "retryable", "retry_route", "batch_limit", "receipt_persistence_degraded")
    )


def _require_positive_batch_limit(value: object) -> int:
    """Validate the scheduler's bounded-work authority at runtime."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("judgment automation batch_limit must be a positive integer")
    return value


def _judgment_automation_receipt_payload(
    *,
    status: JudgmentAutomationReceiptStatus,
    reason: str,
    batch_limit: int,
    result: JudgmentAutomationSweepResult | None,
    retryable: bool,
    receipt_persistence_degraded: bool,
    receipt_persistence_recovered: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": status,
        "reason": reason,
        "retryable": retryable,
        "retry_route": "next enabled judgment-automation tick",
        "batch_limit": batch_limit,
        "receipt_persistence_degraded": receipt_persistence_degraded,
        "receipt_persistence_recovered": receipt_persistence_recovered,
    }
    if result is not None:
        payload.update(
            considered=result.considered,
            accepted=result.accepted,
            rejected=result.rejected,
            escalated=result.escalated,
            idempotent=result.idempotent,
            failed=result.failed,
        )
    return payload


def _record_judgment_automation_receipt(
    root: Path,
    *,
    status: JudgmentAutomationReceiptStatus,
    reason: str,
    now_ms: int | None,
    batch_limit: int,
    result: JudgmentAutomationSweepResult | None = None,
    retryable: bool,
    operation_id: str | None = None,
    receipt_context: _JudgmentAutomationReceiptContext | None = None,
    suppress_identical_for_ms: int | None = None,
    receipt_persistence_degraded: bool = False,
    receipt_persistence_recovered: bool = False,
    user_tier_committed: bool = False,
) -> JudgmentAutomationReceiptOutcome:
    """Persist one scheduler outcome in the existing daemon event ledger."""

    _require_positive_batch_limit(batch_limit)
    if not (root / "ops.db").exists():
        return JudgmentAutomationReceiptOutcome.FAILED
    from polylogue.daemon.events import current_epoch_ms, emit_daemon_event, get_latest_daemon_event

    payload = _judgment_automation_receipt_payload(
        status=status,
        reason=reason,
        batch_limit=batch_limit,
        result=result,
        retryable=retryable,
        receipt_persistence_degraded=receipt_persistence_degraded,
        receipt_persistence_recovered=receipt_persistence_recovered,
    )
    observed_at_ms = current_epoch_ms() if now_ms is None else now_ms
    try:
        if suppress_identical_for_ms is not None:
            latest = get_latest_daemon_event(JUDGMENT_AUTOMATION_STAGE, archive_root_path=root)
            latest_ts_ms = latest.get("ts_ms") if latest is not None else None
            latest_state = _receipt_state(latest.get("payload")) if latest is not None else None
            if (
                isinstance(latest_ts_ms, int)
                and _receipt_state(payload) == latest_state
                and 0 <= observed_at_ms - latest_ts_ms <= suppress_identical_for_ms
            ):
                return JudgmentAutomationReceiptOutcome.COALESCED
        emit_daemon_event(
            JUDGMENT_AUTOMATION_STAGE,
            operation_id=operation_id,
            payload=payload,
            archive_root_path=root,
            observed_at_ms=observed_at_ms,
        )
        if receipt_context is not None:
            receipt_context.recorded = True
        return JudgmentAutomationReceiptOutcome.PERSISTED
    except Exception:
        logger.warning("judgment_automation: scheduler receipt write failed", exc_info=True)
        raise _JudgmentAutomationReceiptPersistenceError(
            "judgment automation scheduler receipt could not be persisted",
            result=result,
            status=status,
            reason=reason,
            user_tier_committed=user_tier_committed,
        ) from None


def _sweep_result_from_receipt_payload(payload: Mapping[str, object]) -> JudgmentAutomationSweepResult | None:
    """Rehydrate committed sweep counters stored in an outbox marker."""

    names = ("considered", "accepted", "rejected", "escalated", "idempotent", "failed")
    values = [payload.get(name) for name in names]
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return None
    typed_values = cast(tuple[int, int, int, int, int, int], tuple(values))
    return JudgmentAutomationSweepResult(
        considered=typed_values[0],
        accepted=typed_values[1],
        rejected=typed_values[2],
        escalated=typed_values[3],
        idempotent=typed_values[4],
        failed=typed_values[5],
    )


def _valid_judgment_automation_receipt_payload(payload: object) -> bool:
    """Check that a durable receipt contains enough evidence to acknowledge a marker."""

    if not isinstance(payload, dict):
        return False
    if payload.get("status") not in {"completed", "parked", "failed"}:
        return False
    if not isinstance(payload.get("reason"), str) or not payload["reason"]:
        return False
    if not isinstance(payload.get("retryable"), bool):
        return False
    if not isinstance(payload.get("retry_route"), str) or not payload["retry_route"]:
        return False
    try:
        _require_positive_batch_limit(payload.get("batch_limit"))
    except ValueError:
        return False
    for key in ("receipt_persistence_degraded", "receipt_persistence_recovered"):
        if not isinstance(payload.get(key), bool):
            return False
    counter_keys = {"considered", "accepted", "rejected", "escalated", "idempotent", "failed"}
    return not (counter_keys & payload.keys()) or _sweep_result_from_receipt_payload(payload) is not None


def recover_pending_judgment_automation_receipts(root: Path, *, now_ms: int | None = None) -> int:
    """Drain committed user-tier receipt markers using their original IDs.

    Recovery is deliberately idempotent: an existing ops event for the marker
    operation is acknowledged without another write; otherwise exactly that
    operation ID is used for the recovered event. A failed ops write leaves the
    marker active for the next process or tick.
    """

    user_db = root / "user.db"
    if not user_db.exists():
        return 0
    from polylogue.daemon.events import get_latest_daemon_event
    from polylogue.storage.sqlite.archive_tiers.user_write import (
        ack_judgment_automation_receipt_outbox,
        list_judgment_automation_receipt_outbox,
    )
    from polylogue.storage.sqlite.connection_profile import open_connection

    conn = open_connection(user_db)
    conn.row_factory = sqlite3.Row
    acknowledged = 0
    try:
        for marker in list_judgment_automation_receipt_outbox(conn):
            value = marker.value if isinstance(marker.value, dict) else {}
            operation_id = value.get("operation_id")
            receipt = value.get("receipt")
            if not isinstance(operation_id, str) or not operation_id or not isinstance(receipt, dict):
                logger.error("judgment_automation: malformed receipt outbox marker %s", marker.assertion_id)
                continue
            latest = get_latest_daemon_event(
                JUDGMENT_AUTOMATION_STAGE,
                operation_id=operation_id,
                archive_root_path=root,
            )
            if (
                latest is not None
                and latest.get("operation_id") == operation_id
                and _valid_judgment_automation_receipt_payload(latest.get("payload"))
            ):
                acknowledged += int(ack_judgment_automation_receipt_outbox(conn, marker, now_ms=now_ms))
                continue
            if latest is not None:
                logger.error("judgment_automation: refusing malformed receipt event for marker %s", marker.assertion_id)
            raw_status_value = receipt.get("status")
            raw_reason = receipt.get("reason")
            if not _valid_judgment_automation_receipt_payload(receipt):
                logger.error("judgment_automation: malformed receipt payload in marker %s", marker.assertion_id)
                continue
            if raw_status_value not in {"completed", "parked", "failed"} or not isinstance(raw_reason, str):
                continue
            raw_status = cast(JudgmentAutomationReceiptStatus, raw_status_value)
            raw_batch_limit = receipt.get("batch_limit", JUDGMENT_AUTOMATION_BATCH_LIMIT_DEFAULT)
            try:
                batch_limit = _require_positive_batch_limit(raw_batch_limit)
            except ValueError:
                batch_limit = JUDGMENT_AUTOMATION_BATCH_LIMIT_DEFAULT
            result = _sweep_result_from_receipt_payload(receipt)
            try:
                outcome = _record_judgment_automation_receipt(
                    root,
                    status=raw_status,
                    reason=raw_reason,
                    now_ms=now_ms,
                    batch_limit=batch_limit,
                    result=result,
                    retryable=bool(receipt.get("retryable", True)),
                    operation_id=operation_id,
                    receipt_persistence_recovered=True,
                )
            except Exception:
                logger.warning("judgment_automation: receipt outbox recovery write failed", exc_info=True)
                continue
            if outcome is JudgmentAutomationReceiptOutcome.PERSISTED:
                acknowledged += int(ack_judgment_automation_receipt_outbox(conn, marker, now_ms=now_ms))
        if acknowledged:
            conn.commit()
    finally:
        conn.close()
    return acknowledged


def _judgment_automation_receipt_outbox_pending(root: Path) -> bool:
    """Read whether recovery work exists without taking the daemon writer slot."""

    user_db = root / "user.db"
    if not user_db.exists():
        return False
    from polylogue.storage.sqlite.archive_tiers.user_write import (
        JUDGMENT_AUTOMATION_RECEIPT_OUTBOX_SCOPE,
    )
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    try:
        conn = open_readonly_connection(user_db)
    except (OSError, sqlite3.Error) as exc:
        logger.warning("judgment_automation: outbox probe could not open user.db: %s", exc)
        return False
    try:
        return (
            conn.execute(
                """
                SELECT 1
                FROM assertions
                WHERE scope_ref = ? AND kind = ? AND status = ?
                LIMIT 1
                """,
                (
                    JUDGMENT_AUTOMATION_RECEIPT_OUTBOX_SCOPE,
                    AssertionKind.RUN_STATE.value,
                    AssertionStatus.ACTIVE.value,
                ),
            ).fetchone()
            is not None
        )
    except sqlite3.Error as exc:
        logger.warning("judgment_automation: outbox probe query failed: %s", exc)
        return False
    finally:
        conn.close()


def _coerce_confidence(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= parsed <= 1.0:
        return None
    return parsed


def parse_judgment_automation_policy(
    raw: Mapping[str, object],
) -> dict[AssertionKind, JudgmentAutomationPolicyRule]:
    """Decode the raw ``[judgment_automation.policies]`` TOML table.

    Fails closed per-entry, not for the whole sweep: an unrecognized kind
    name or a malformed/out-of-range threshold is logged and dropped rather
    than raised, so one operator typo degrades that one kind to
    always-escalate instead of crashing every future sweep.
    """
    policy: dict[AssertionKind, JudgmentAutomationPolicyRule] = {}
    for raw_kind, raw_rule in raw.items():
        try:
            kind = AssertionKind.from_string(str(raw_kind))
        except ValueError:
            logger.warning("judgment_automation: unknown policy kind %r ignored", raw_kind)
            continue
        if not isinstance(raw_rule, Mapping):
            continue
        accept = _coerce_confidence(raw_rule.get("auto_accept_min_confidence"))
        reject = _coerce_confidence(raw_rule.get("auto_reject_max_confidence"))
        if accept is None and reject is None:
            continue
        policy[kind] = JudgmentAutomationPolicyRule(
            auto_accept_min_confidence=accept,
            auto_reject_max_confidence=reject,
        )
    return policy


def evaluate_candidate(
    candidate: ArchiveAssertionEnvelope,
    policy: Mapping[AssertionKind, JudgmentAutomationPolicyRule],
) -> JudgmentAutomationDecision:
    """Apply the policy engine to one candidate assertion.

    Escalates -- never guesses -- when: the candidate's kind has no
    configured policy, the candidate carries no ``confidence`` signal, or the
    confidence falls strictly between the accept/reject thresholds (or only
    one threshold is configured and the value doesn't clear it). This is the
    explicit residue path the bead asks for: silence is never mistaken for a
    decision.
    """
    candidate_ref = f"assertion:{candidate.assertion_id}"
    rule = policy.get(candidate.kind)
    if rule is None:
        return JudgmentAutomationDecision(
            candidate_ref=candidate_ref,
            decision="escalate",
            reason=f"no judgment-automation policy configured for kind={candidate.kind.value}",
        )
    confidence = candidate.confidence
    if confidence is None:
        return JudgmentAutomationDecision(
            candidate_ref=candidate_ref,
            decision="escalate",
            reason="candidate carries no confidence signal",
        )
    if rule.auto_accept_min_confidence is not None and confidence >= rule.auto_accept_min_confidence:
        return JudgmentAutomationDecision(
            candidate_ref=candidate_ref,
            decision="accept",
            reason=(f"confidence {confidence:.3f} >= auto_accept_min_confidence {rule.auto_accept_min_confidence:.3f}"),
        )
    if rule.auto_reject_max_confidence is not None and confidence <= rule.auto_reject_max_confidence:
        return JudgmentAutomationDecision(
            candidate_ref=candidate_ref,
            decision="reject",
            reason=(f"confidence {confidence:.3f} <= auto_reject_max_confidence {rule.auto_reject_max_confidence:.3f}"),
        )
    return JudgmentAutomationDecision(
        candidate_ref=candidate_ref,
        decision="escalate",
        reason=f"confidence {confidence:.3f} inside the undecided band for kind={candidate.kind.value}",
    )


def _handoff_assertion_id(candidate_assertion_id: str) -> str:
    """Deterministic id for the escalation handoff mirroring one candidate.

    One handoff row per candidate (re-sweeps upsert the same row with a
    fresh reason/timestamp instead of accumulating duplicates).
    """
    digest = hashlib.sha256()
    digest.update(candidate_assertion_id.encode("utf-8", errors="surrogatepass"))
    digest.update(b"\0")
    return f"assertion-{AssertionKind.HANDOFF.value}:{digest.hexdigest()}"


def _write_escalation_handoff(
    conn: sqlite3.Connection,
    candidate: ArchiveAssertionEnvelope,
    decision: JudgmentAutomationDecision,
    *,
    now_ms: int | None,
) -> None:
    from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion

    upsert_assertion(
        conn,
        assertion_id=_handoff_assertion_id(candidate.assertion_id),
        target_ref=decision.candidate_ref,
        kind=AssertionKind.HANDOFF,
        key="judgment-automation/escalation",
        value={
            "candidate_ref": decision.candidate_ref,
            "candidate_kind": candidate.kind.value,
            "reason": decision.reason,
        },
        body_text=decision.reason,
        author_ref=JUDGMENT_AUTOMATION_ACTOR_REF,
        author_kind=JUDGMENT_AUTOMATION_AUTHOR_KIND,
        evidence_refs=(decision.candidate_ref,),
        now_ms=now_ms,
    )


def run_judgment_automation_sweep_once(
    root: Path,
    *,
    batch_limit: int,
    policy: Mapping[AssertionKind, JudgmentAutomationPolicyRule] | None = None,
    now_ms: int | None = None,
    operation_id: str | None = None,
    _receipt_context: _JudgmentAutomationReceiptContext | None = None,
) -> JudgmentAutomationSweepResult:
    """Run one bounded judgment-automation sweep against ``<root>/user.db``.

    Reads up to ``batch_limit`` candidates, judges the ones the policy
    engine can decide (via the same ``judge_assertion_candidates`` chokepoint
    the MCP ``judge`` dispatcher calls), and writes an escalation handoff for
    every remaining candidate. A missing ``user.db`` or an empty policy is a
    bounded no-op, not an error -- the daemon must tolerate an archive that
    hasn't been judged even once yet.
    """
    _require_positive_batch_limit(batch_limit)
    from polylogue.storage.sqlite.archive_tiers.user_write import (
        ArchiveAssertionBulkJudgmentItemEnvelope,
        ack_judgment_automation_receipt_outbox,
        judge_assertion_candidates,
        list_assertion_candidates,
        upsert_judgment_automation_receipt_outbox,
    )
    from polylogue.storage.sqlite.connection_profile import open_connection

    def record_receipt(
        *,
        status: JudgmentAutomationReceiptStatus,
        reason: str,
        now_ms: int | None,
        retryable: bool,
        result: JudgmentAutomationSweepResult | None = None,
        receipt_persistence_degraded: bool = False,
        user_tier_committed: bool = False,
    ) -> JudgmentAutomationReceiptOutcome:
        try:
            recorded = _record_judgment_automation_receipt(
                root,
                status=status,
                reason=reason,
                now_ms=now_ms,
                batch_limit=batch_limit,
                result=result,
                retryable=retryable,
                operation_id=operation_id,
                receipt_context=_receipt_context,
                receipt_persistence_degraded=receipt_persistence_degraded,
                user_tier_committed=user_tier_committed,
            )
        except _JudgmentAutomationReceiptPersistenceError as exc:
            if result is not None:
                raise _JudgmentAutomationReceiptPersistenceError(
                    str(exc),
                    result=result,
                    status=status,
                    reason=reason,
                    user_tier_committed=user_tier_committed,
                ) from exc
            raise _JudgmentAutomationReceiptPersistenceError(
                str(exc),
                status=status,
                reason=reason,
                user_tier_committed=user_tier_committed,
            ) from exc
        if _receipt_context is not None and recorded is JudgmentAutomationReceiptOutcome.FAILED:
            raise _JudgmentAutomationReceiptPersistenceError(
                "judgment automation scheduler receipt could not be persisted",
                result=result,
                status=status,
                reason=reason,
                user_tier_committed=user_tier_committed,
            )
        return recorded

    resolved_policy = (
        policy
        if policy is not None
        else parse_judgment_automation_policy(load_polylogue_config().judgment_automation_policy)
    )
    if not resolved_policy:
        record_receipt(
            status="parked",
            reason="policy_empty",
            now_ms=now_ms,
            retryable=True,
        )
        return JudgmentAutomationSweepResult()

    user_db = root / "user.db"
    if not user_db.exists():
        record_receipt(
            status="parked",
            reason="user_db_unavailable",
            now_ms=now_ms,
            retryable=True,
        )
        return JudgmentAutomationSweepResult()

    conn = open_connection(user_db)
    conn.row_factory = sqlite3.Row
    try:
        candidates = list_assertion_candidates(conn, limit=batch_limit)
        if not candidates:
            record_receipt(
                status="completed",
                reason="queue_empty",
                now_ms=now_ms,
                result=JudgmentAutomationSweepResult(),
                retryable=False,
                user_tier_committed=False,
            )
            return JudgmentAutomationSweepResult()

        decisions = {f"assertion:{c.assertion_id}": evaluate_candidate(c, resolved_policy) for c in candidates}
        candidates_by_ref = {f"assertion:{c.assertion_id}": c for c in candidates}

        judgeable_refs = [ref for ref, d in decisions.items() if d.decision in ("accept", "reject")]
        escalated_refs = [ref for ref, d in decisions.items() if d.decision == "escalate"]

        accepted = rejected = idempotent = failed = 0
        if judgeable_refs:
            items = tuple(
                ArchiveAssertionBulkJudgmentItemEnvelope(
                    candidate_ref=ref,
                    decision=decisions[ref].decision,
                    reason=decisions[ref].reason,
                    inject=False,
                    actor_ref=JUDGMENT_AUTOMATION_ACTOR_REF,
                )
                for ref in judgeable_refs
            )
            bulk_result = judge_assertion_candidates(conn, items, now_ms=now_ms)
            for item_result in bulk_result.items:
                if item_result.outcome == "applied":
                    if decisions[item_result.candidate_ref].decision == "accept":
                        accepted += 1
                    else:
                        rejected += 1
                elif item_result.outcome == "idempotent":
                    idempotent += 1
                else:
                    failed += 1
                    logger.warning(
                        "judgment_automation: judge failed candidate_ref=%s error=%s",
                        item_result.candidate_ref,
                        item_result.error,
                    )

        for ref in escalated_refs:
            _write_escalation_handoff(conn, candidates_by_ref[ref], decisions[ref], now_ms=now_ms)
        result = JudgmentAutomationSweepResult(
            considered=len(candidates),
            accepted=accepted,
            rejected=rejected,
            escalated=len(escalated_refs),
            idempotent=idempotent,
            failed=failed,
        )
        receipt_status: JudgmentAutomationReceiptStatus = "failed" if result.failed else "completed"
        receipt_reason = "candidate_judgment_failures" if result.failed else "sweep_completed"
        outbox_marker = None
        if operation_id is not None:
            outbox_marker = upsert_judgment_automation_receipt_outbox(
                conn,
                operation_id=operation_id,
                receipt_payload=_judgment_automation_receipt_payload(
                    status=receipt_status,
                    reason=receipt_reason,
                    batch_limit=batch_limit,
                    result=result,
                    retryable=bool(result.failed),
                    receipt_persistence_degraded=False,
                    receipt_persistence_recovered=False,
                ),
                now_ms=now_ms,
            )
        conn.commit()
        try:
            recorded = record_receipt(
                status=receipt_status,
                reason=receipt_reason,
                now_ms=now_ms,
                result=result,
                retryable=bool(result.failed),
                user_tier_committed=True,
            )
            if operation_id is not None and recorded is JudgmentAutomationReceiptOutcome.PERSISTED:
                assert outbox_marker is not None
                try:
                    acknowledged = ack_judgment_automation_receipt_outbox(conn, outbox_marker, now_ms=now_ms)
                    if not acknowledged:
                        raise RuntimeError("judgment automation receipt outbox marker was not acknowledged")
                    conn.commit()
                except Exception as exc:
                    conn.rollback()
                    raise _JudgmentAutomationReceiptPersistenceError(
                        "judgment automation receipt acknowledgement could not be persisted",
                        result=result,
                        status=receipt_status,
                        reason=receipt_reason,
                        user_tier_committed=True,
                    ) from exc
        except _JudgmentAutomationReceiptPersistenceError as exc:
            raise _JudgmentAutomationReceiptPersistenceError(
                str(exc),
                result=result,
                status=receipt_status,
                reason=receipt_reason,
                user_tier_committed=True,
            ) from exc
        return result
    except _JudgmentAutomationReceiptPersistenceError:
        conn.rollback()
        raise
    except Exception as exc:
        conn.rollback()
        try:
            record_receipt(
                status="failed",
                reason=f"sweep_failed:{type(exc).__name__}",
                now_ms=now_ms,
                retryable=True,
            )
        except _JudgmentAutomationReceiptPersistenceError as receipt_exc:
            raise receipt_exc from exc
        raise
    finally:
        conn.close()


async def periodic_judgment_automation_sweep(
    *,
    catch_up_complete: asyncio.Event | None = None,
) -> None:
    """Periodically run one bounded judgment-automation sweep.

    ``daemon/cli.py`` schedules this loop unconditionally alongside the
    other periodic maintenance loops (``reload_behavior="daemon-loop"`` for
    both gating config keys) -- the loop itself re-checks
    ``judgment_automation_enabled and mcp_judge_enabled`` on every tick and
    no-ops otherwise, so flipping either flag in ``polylogue.toml`` takes
    effect on the *next* tick without a daemon restart, the same as
    ``_periodic_db_optimize``'s self-gating pattern.
    """
    from polylogue.daemon.cli import _await_catch_up_gate
    from polylogue.daemon.write_coordinator import daemon_write_coordinator
    from polylogue.paths import archive_root, data_home

    await _await_catch_up_gate(catch_up_complete, loop_name="judgment automation sweep")
    coordinator = daemon_write_coordinator()
    last_valid_root: Path | None = None

    def receipt_root() -> Path:
        """Resolve the archive root without letting a reload error kill the loop."""

        nonlocal last_valid_root
        try:
            root = archive_root()
        except Exception:
            logger.warning("judgment_automation: archive root resolution failed", exc_info=True)
            return last_valid_root if last_valid_root is not None else data_home()
        last_valid_root = root
        return root

    async def recover_receipt_outbox(root: Path) -> None:
        """Drain committed user-tier receipt markers before config can gate recovery."""

        if not root.exists() or not _judgment_automation_receipt_outbox_pending(root):
            return
        try:
            await coordinator.run_sync(
                "maintenance.judgment_automation.recover",
                recover_pending_judgment_automation_receipts,
                root,
            )
        except Exception:
            logger.warning("judgment_automation: receipt outbox recovery failed", exc_info=True)

    # Recovery is a durability obligation, not an enabled-feature sweep. Run it
    # before the first config load so a malformed reload cannot strand a marker
    # left by a prior process in the user tier.
    await recover_receipt_outbox(receipt_root())

    async def persist_failure_fallback(
        exc: Exception,
        *,
        default_reason: str,
        operation_id: str,
        root: Path,
        batch_limit: int,
    ) -> None:
        result = exc.result if isinstance(exc, _JudgmentAutomationReceiptPersistenceError) else None
        status = (
            exc.status
            if isinstance(exc, _JudgmentAutomationReceiptPersistenceError) and exc.status is not None
            else "failed"
        )
        reason = (
            exc.reason
            if isinstance(exc, _JudgmentAutomationReceiptPersistenceError) and exc.reason is not None
            else default_reason
        )
        degraded = isinstance(exc, _JudgmentAutomationReceiptPersistenceError) and exc.user_tier_committed
        try:
            recorded = await coordinator.run_sync(
                "maintenance.judgment_automation.receipt",
                _record_judgment_automation_receipt,
                root,
                status=status,
                reason=("receipt_persistence_degraded" if degraded else reason),
                now_ms=None,
                batch_limit=batch_limit,
                result=result,
                retryable=bool(result.failed) if result is not None else True,
                operation_id=operation_id,
                receipt_persistence_degraded=degraded,
            )
        except Exception:
            logger.warning("judgment_automation: failure receipt fallback failed", exc_info=True)
            return
        if recorded not in {
            JudgmentAutomationReceiptOutcome.PERSISTED,
            JudgmentAutomationReceiptOutcome.COALESCED,
        }:
            logger.warning("judgment_automation: failure receipt fallback was not persisted")

    last_valid_interval_s = JUDGMENT_AUTOMATION_SWEEP_INTERVAL_FLOOR_SECONDS
    last_valid_batch_limit = JUDGMENT_AUTOMATION_BATCH_LIMIT_DEFAULT
    while True:
        root = receipt_root()
        try:
            pacing_cfg = load_polylogue_config()
            pacing_interval_s = pacing_cfg.judgment_automation_interval_s
            pacing_batch_limit = pacing_cfg.judgment_automation_batch_limit
            _require_positive_batch_limit(pacing_batch_limit)
        except Exception as exc:
            if root.exists():
                await persist_failure_fallback(
                    exc,
                    default_reason="configuration_reload_failed",
                    operation_id=f"judgment-automation:{uuid.uuid4().hex}",
                    root=root,
                    batch_limit=last_valid_batch_limit,
                )
            logger.warning("judgment_automation: pre-sleep configuration reload failed", exc_info=True)
            interval = last_valid_interval_s
        else:
            last_valid_interval_s = max(
                pacing_interval_s,
                JUDGMENT_AUTOMATION_SWEEP_INTERVAL_FLOOR_SECONDS,
            )
            last_valid_batch_limit = pacing_batch_limit
            interval = last_valid_interval_s
        await asyncio.sleep(interval)
        # Config is reloadable at the daemon-loop boundary. The pre-sleep
        # snapshot controls pacing only; gating and batch settings must come
        # from the post-sleep snapshot so a config flip during the wait takes
        # effect on this tick.
        root = receipt_root()
        if not root.exists():
            continue
        try:
            cfg = load_polylogue_config()
            cfg_interval_s = cfg.judgment_automation_interval_s
            cfg_batch_limit = cfg.judgment_automation_batch_limit
            cfg_automation_enabled = cfg.judgment_automation_enabled
            cfg_judge_enabled = cfg.mcp_judge_enabled
            _require_positive_batch_limit(cfg_batch_limit)
            resolved_policy = parse_judgment_automation_policy(cfg.judgment_automation_policy)
        except Exception as exc:
            await persist_failure_fallback(
                exc,
                default_reason="configuration_reload_failed",
                operation_id=f"judgment-automation:{uuid.uuid4().hex}",
                root=root,
                batch_limit=last_valid_batch_limit,
            )
            logger.warning("judgment_automation: post-sleep configuration reload failed", exc_info=True)
            continue
        last_valid_interval_s = max(cfg_interval_s, JUDGMENT_AUTOMATION_SWEEP_INTERVAL_FLOOR_SECONDS)
        last_valid_batch_limit = cfg_batch_limit
        await recover_receipt_outbox(root)
        if not (cfg_automation_enabled and cfg_judge_enabled):
            try:
                recorded = await coordinator.run_sync(
                    "maintenance.judgment_automation.receipt",
                    _record_judgment_automation_receipt,
                    root,
                    status="parked",
                    reason="capability_gate_disabled",
                    now_ms=None,
                    batch_limit=cfg_batch_limit,
                    retryable=True,
                    suppress_identical_for_ms=_judgment_automation_receipt_coalescing_horizon_ms(cfg_interval_s),
                )
            except Exception:
                logger.warning("judgment_automation: parked receipt write failed; retrying next tick", exc_info=True)
            else:
                if recorded not in {
                    JudgmentAutomationReceiptOutcome.PERSISTED,
                    JudgmentAutomationReceiptOutcome.COALESCED,
                }:
                    logger.warning("judgment_automation: parked receipt was not persisted; retrying next tick")
            continue
        receipt_context = _JudgmentAutomationReceiptContext(operation_id=f"judgment-automation:{uuid.uuid4().hex}")
        try:
            result = await coordinator.run_sync(
                "maintenance.judgment_automation",
                run_judgment_automation_sweep_once,
                root,
                batch_limit=cfg_batch_limit,
                policy=resolved_policy,
                operation_id=receipt_context.operation_id,
                _receipt_context=receipt_context,
            )
            if isinstance(result, bool):
                if not result:
                    raise RuntimeError("judgment automation coordinator returned false without running the sweep")
                raise RuntimeError("judgment automation coordinator returned an invalid sweep result")
            if not isinstance(result, JudgmentAutomationSweepResult):
                raise RuntimeError("judgment automation coordinator returned an invalid sweep result")
            if result.considered:
                logger.info(
                    "judgment_automation: considered=%d accepted=%d rejected=%d escalated=%d idempotent=%d failed=%d",
                    result.considered,
                    result.accepted,
                    result.rejected,
                    result.escalated,
                    result.idempotent,
                    result.failed,
                )
        except sqlite3.OperationalError as exc:
            reason = "transient_sqlite_lock" if is_transient_sqlite_lock(exc) else "operational_error"
            if not receipt_context.recorded:
                await persist_failure_fallback(
                    exc,
                    default_reason=reason,
                    operation_id=receipt_context.operation_id,
                    root=root,
                    batch_limit=last_valid_batch_limit,
                )
            if reason == "transient_sqlite_lock":
                logger.info("judgment_automation: archive busy; retrying on next tick: %s", exc)
            else:
                logger.warning("judgment_automation: archive operation failed; retrying on next tick: %s", exc)
        except Exception as exc:
            if not receipt_context.recorded:
                await persist_failure_fallback(
                    exc,
                    default_reason=(
                        "receipt_persistence_failed"
                        if isinstance(exc, _JudgmentAutomationReceiptPersistenceError)
                        else "sweep_exception"
                    ),
                    operation_id=receipt_context.operation_id,
                    root=root,
                    batch_limit=last_valid_batch_limit,
                )
            logger.warning("judgment_automation: sweep failed", exc_info=True)


__all__ = [
    "JUDGMENT_AUTOMATION_ACTOR_REF",
    "JUDGMENT_AUTOMATION_AUTHOR_KIND",
    "JUDGMENT_AUTOMATION_STAGE",
    "JUDGMENT_AUTOMATION_SWEEP_INTERVAL_FLOOR_SECONDS",
    "JudgmentAutomationDecision",
    "JudgmentAutomationDecisionKind",
    "JudgmentAutomationPolicyRule",
    "JudgmentAutomationReceiptOutcome",
    "JudgmentAutomationReceiptStatus",
    "JudgmentAutomationSweepResult",
    "evaluate_candidate",
    "parse_judgment_automation_policy",
    "periodic_judgment_automation_sweep",
    "recover_pending_judgment_automation_receipts",
    "run_judgment_automation_sweep_once",
    "judgment_automation_receipt_freshness_window_ms",
]
