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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from polylogue.config import load_polylogue_config
from polylogue.core.enums import AssertionKind
from polylogue.logging import get_logger
from polylogue.sources.live.sqlite_locking import is_transient_sqlite_lock

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.user_write import ArchiveAssertionEnvelope

logger = get_logger(__name__)

#: Actor identity recorded on every automated judgment/handoff so the
#: audit trail (``assertions.author_ref``) can distinguish automated
#: decisions from a human operator's.
JUDGMENT_AUTOMATION_ACTOR_REF = "automation:judgment-policy"

#: Non-``"user"`` author kind for escalation handoff rows -- routes through
#: the normal ``upsert_assertion`` promotion gate like any other automated
#: writer (polylogue-37t.15); ``handoff`` is not in
#: ``ASSERTION_CANDIDATE_JUDGMENT_KINDS`` so this can never create a
#: candidate the sweep would try to judge again.
JUDGMENT_AUTOMATION_AUTHOR_KIND = "automation"

#: Floor on the configured sweep interval so a misconfigured
#: ``judgment_automation_interval_s`` cannot turn this into a busy loop.
JUDGMENT_AUTOMATION_SWEEP_INTERVAL_FLOOR_SECONDS = 60

JudgmentAutomationDecisionKind = Literal["accept", "reject", "escalate"]


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
) -> JudgmentAutomationSweepResult:
    """Run one bounded judgment-automation sweep against ``<root>/user.db``.

    Reads up to ``batch_limit`` candidates, judges the ones the policy
    engine can decide (via the same ``judge_assertion_candidates`` chokepoint
    the MCP ``judge`` dispatcher calls), and writes an escalation handoff for
    every remaining candidate. A missing ``user.db`` or an empty policy is a
    bounded no-op, not an error -- the daemon must tolerate an archive that
    hasn't been judged even once yet.
    """
    from polylogue.storage.sqlite.archive_tiers.user_write import (
        ArchiveAssertionBulkJudgmentItemEnvelope,
        judge_assertion_candidates,
        list_assertion_candidates,
    )
    from polylogue.storage.sqlite.connection_profile import open_connection

    resolved_policy = (
        policy
        if policy is not None
        else parse_judgment_automation_policy(load_polylogue_config().judgment_automation_policy)
    )
    if not resolved_policy:
        return JudgmentAutomationSweepResult()

    user_db = root / "user.db"
    if not user_db.exists():
        return JudgmentAutomationSweepResult()

    conn = open_connection(user_db)
    conn.row_factory = sqlite3.Row
    try:
        candidates = list_assertion_candidates(conn, limit=batch_limit)
        if not candidates:
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
        conn.commit()

        return JudgmentAutomationSweepResult(
            considered=len(candidates),
            accepted=accepted,
            rejected=rejected,
            escalated=len(escalated_refs),
            idempotent=idempotent,
            failed=failed,
        )
    finally:
        conn.close()


async def periodic_judgment_automation_sweep(
    *,
    catch_up_complete: asyncio.Event | None = None,
) -> None:
    """Periodically run one bounded judgment-automation sweep.

    The caller (``daemon/cli.py``) is expected to gate scheduling this loop
    at all on ``judgment_automation_enabled and mcp_judge_enabled`` -- this
    function re-checks both on every tick too, so a live config edit that
    turns either flag back off (`polylogue.toml` is re-read from disk each
    tick, unlike the daemon-startup-bound flags) stops the sweep from judging
    on the *next* tick without a daemon restart.
    """
    from polylogue.daemon.write_coordinator import daemon_write_coordinator
    from polylogue.paths import archive_root

    if catch_up_complete is not None:
        await catch_up_complete.wait()
    while True:
        cfg = load_polylogue_config()
        interval = max(cfg.judgment_automation_interval_s, JUDGMENT_AUTOMATION_SWEEP_INTERVAL_FLOOR_SECONDS)
        await asyncio.sleep(interval)
        if not (cfg.judgment_automation_enabled and cfg.mcp_judge_enabled):
            continue
        root = archive_root()
        if not root.exists():
            continue
        try:
            result = await daemon_write_coordinator().run_sync(
                "maintenance.judgment_automation",
                run_judgment_automation_sweep_once,
                root,
                batch_limit=cfg.judgment_automation_batch_limit,
            )
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
            if is_transient_sqlite_lock(exc):
                logger.info("judgment_automation: archive busy; retrying on next tick: %s", exc)
                continue
            logger.warning("judgment_automation: sweep failed", exc_info=True)
        except Exception:
            logger.warning("judgment_automation: sweep failed", exc_info=True)


__all__ = [
    "JUDGMENT_AUTOMATION_ACTOR_REF",
    "JUDGMENT_AUTOMATION_AUTHOR_KIND",
    "JUDGMENT_AUTOMATION_SWEEP_INTERVAL_FLOOR_SECONDS",
    "JudgmentAutomationDecision",
    "JudgmentAutomationDecisionKind",
    "JudgmentAutomationPolicyRule",
    "JudgmentAutomationSweepResult",
    "evaluate_candidate",
    "parse_judgment_automation_policy",
    "periodic_judgment_automation_sweep",
    "run_judgment_automation_sweep_once",
]
