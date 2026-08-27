"""The single admission boundary for context delivery.

Sources provide candidates; this module owns ordering, budgets, trust, and the
receipt of every decision.  In particular, source scores are never compared:
each source is ranked independently and receives a deterministic share.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol

from polylogue.core.refs import ExecutionContextRef

TrustClass = Literal["operator", "system", "quoted"]
Decision = Literal["included", "degraded", "dropped"]


@dataclass(frozen=True, slots=True)
class ContextItem:
    """A source candidate. ``ordinal_score`` is meaningful only per source."""

    ref: str
    content: str
    token_cost: int
    ordinal_score: int = 0
    priority_class: str = "default"
    trust_class: TrustClass = "quoted"
    material_class: str = "evidence"
    source: str = "unknown"
    expires_at_ms: int | None = None
    target_session: str | None = None
    policy_refs: tuple[str, ...] = ()
    kind: str | None = None
    author_kind: str | None = None
    author_ref: str | None = None
    status: str | None = None
    revoked: bool = False
    degrade: Callable[[ContextItem], ContextItem | None] | None = field(default=None, compare=False, repr=False)
    authority_reason: str = ""


class ContextSource(Protocol):
    """Provider contract; sources cannot allocate budgets or grant trust."""

    name: str

    def candidates(self, *, moment: str, target_session: str | None) -> Sequence[ContextItem]: ...


@dataclass(frozen=True, slots=True)
class ContextLedgerRow:
    decision: Decision
    source: str
    item_ref: str
    token_cost: int
    source_local_rank: int
    budget_before: int
    budget_after: int
    disclosure_verdict: str
    authority_verdict: str
    authority_reason: str
    policy_refs: tuple[str, ...]
    target_session: str | None
    execution_context_ref: str

    def as_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "source": self.source,
            "item_ref": self.item_ref,
            "token_cost": self.token_cost,
            "source_local_rank": self.source_local_rank,
            "budget_before": self.budget_before,
            "budget_after": self.budget_after,
            "disclosure_verdict": self.disclosure_verdict,
            "authority_verdict": self.authority_verdict,
            "authority_reason": self.authority_reason,
            "policy_refs": list(self.policy_refs),
            "target_session": self.target_session,
            "execution_context_ref": self.execution_context_ref,
        }


@dataclass(frozen=True, slots=True)
class ContextLedgerRecord:
    """One persisted scheduler decision read back from ``ops.db``."""

    ledger_id: str
    build_ref: str
    observed_at_ms: int
    row: ContextLedgerRow

    def as_dict(self) -> dict[str, object]:
        payload = self.row.as_dict()
        payload.update(
            {
                "ledger_id": self.ledger_id,
                "build_ref": self.build_ref,
                "observed_at_ms": self.observed_at_ms,
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class ContextAssembly:
    quoted_evidence: tuple[ContextItem, ...]
    executable_policy: tuple[ContextItem, ...]
    ledger: tuple[ContextLedgerRow, ...]
    token_cost: int
    budget: int
    execution_context_ref: ExecutionContextRef
    build_ref: str

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "build_ref": self.build_ref,
                "budget": self.budget,
                "token_cost": self.token_cost,
                "execution_context_ref": self.execution_context_ref.context_id,
                "quoted_evidence": [_item_dict(item) for item in self.quoted_evidence],
                "executable_policy": [_item_dict(item) for item in self.executable_policy],
                "ledger": [row.as_dict() for row in self.ledger],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )


def _item_dict(item: ContextItem) -> dict[str, object]:
    return {
        "ref": item.ref,
        "content": item.content,
        "token_cost": item.token_cost,
        "ordinal_score": item.ordinal_score,
        "priority_class": item.priority_class,
        "trust_class": item.trust_class,
        "material_class": item.material_class,
        "source": item.source,
        "expires_at_ms": item.expires_at_ms,
        "target_session": item.target_session,
        "policy_refs": list(item.policy_refs),
        "kind": item.kind,
        "author_kind": item.author_kind,
        "author_ref": item.author_ref,
        "status": item.status,
        "revoked": item.revoked,
    }


def _policy_is_authorized(item: ContextItem, *, target_session: str | None, now_ms: int) -> tuple[bool, str]:
    """Only an explicitly adopted, scoped policy may enter instructions."""
    if item.material_class != "policy" or item.kind != "policy" or item.trust_class != "operator":
        return False, "only operator-adopted policy may instruct"
    if item.author_kind != "user" or not item.author_ref or not item.author_ref.startswith("user:"):
        return False, "policy author is not an operator"
    if item.status != "active" or item.revoked:
        return False, "policy is not active"
    if not item.policy_refs or not all(ref.startswith("policy:") for ref in item.policy_refs):
        return False, "missing or malformed policy reference"
    if item.target_session is not None and item.target_session != target_session:
        return False, "policy target scope mismatch"
    if item.expires_at_ms is not None and item.expires_at_ms <= now_ms:
        return False, "policy expired"
    if not item.authority_reason.startswith("adopted:"):
        return False, "policy is not explicitly adopted"
    return True, "adopted policy scope and expiry valid"


def schedule_context(
    sources: Sequence[ContextSource],
    *,
    moment: str,
    target_session: str | None,
    execution_context: ExecutionContextRef,
    token_budget: int,
    now_ms: int | None = None,
    source_quota: int | None = None,
) -> ContextAssembly:
    """Collect, gate, and allocate every source through one deterministic route."""
    if token_budget < 0:
        raise ValueError("token_budget must be non-negative")
    now = now_ms if now_ms is not None else int(datetime.now(UTC).timestamp() * 1000)
    rows: list[ContextLedgerRow] = []
    included_evidence: list[ContextItem] = []
    included_policy: list[ContextItem] = []
    remaining = token_budget
    # Source order is an input to the policy, never an accidental dict order.
    for source in sorted(sources, key=lambda value: value.name):
        candidates = list(source.candidates(moment=moment, target_session=target_session))
        ranked = sorted(enumerate(candidates), key=lambda pair: (-pair[1].ordinal_score, pair[0], pair[1].ref))
        local_rank = {id(item): rank for rank, (_, item) in enumerate(ranked, start=1)}
        used_by_source = 0
        for _, item in ranked:
            rank = local_rank[id(item)]
            before = remaining
            if item.source != source.name or not item.ref or item.token_cost < 0:
                rows.append(
                    _row(
                        "dropped",
                        item,
                        rank,
                        before,
                        before,
                        "invalid",
                        "rejected",
                        "source identity or cost invalid",
                        execution_context,
                        target_session,
                    )
                )
                continue
            if item.expires_at_ms is not None and item.expires_at_ms <= now:
                rows.append(
                    _row(
                        "dropped",
                        item,
                        rank,
                        before,
                        before,
                        "expired",
                        "rejected",
                        "candidate expired",
                        execution_context,
                        target_session,
                    )
                )
                continue
            if item.revoked:
                rows.append(
                    _row(
                        "dropped",
                        item,
                        rank,
                        before,
                        before,
                        "revoked",
                        "rejected",
                        "candidate revoked",
                        execution_context,
                        target_session,
                    )
                )
                continue
            if item.trust_class not in {"operator", "system", "quoted"}:
                rows.append(
                    _row(
                        "dropped",
                        item,
                        rank,
                        before,
                        before,
                        "malformed",
                        "rejected",
                        "unknown trust class",
                        execution_context,
                        target_session,
                    )
                )
                continue
            authority_ok, authority_reason = _policy_is_authorized(item, target_session=target_session, now_ms=now)
            if item.material_class == "policy" and not authority_ok:
                rows.append(
                    _row(
                        "dropped",
                        item,
                        rank,
                        before,
                        before,
                        "policy",
                        "rejected",
                        authority_reason,
                        execution_context,
                        target_session,
                    )
                )
                continue
            if source_quota is not None and used_by_source >= source_quota:
                rows.append(
                    _row(
                        "dropped",
                        item,
                        rank,
                        before,
                        before,
                        "quota",
                        "accepted",
                        "source quota exhausted",
                        execution_context,
                        target_session,
                    )
                )
                continue
            chosen: ContextItem | None = item
            if item.token_cost > remaining and item.degrade is not None:
                chosen = item.degrade(item)
                if chosen is not None and 0 <= chosen.token_cost <= remaining:
                    decision: Decision = "degraded"
                else:
                    chosen = None
                    decision = "dropped"
            elif item.token_cost <= remaining:
                decision = "included"
            else:
                chosen = None
                decision = "dropped"
            after = remaining - (chosen.token_cost if chosen is not None else 0)
            if chosen is not None:
                remaining = after
                used_by_source += chosen.token_cost
                (included_policy if chosen.material_class == "policy" else included_evidence).append(chosen)
            rows.append(
                _row(
                    decision,
                    item,
                    rank,
                    before,
                    after,
                    "accepted" if chosen else "budget",
                    "accepted" if authority_ok else "quoted",
                    authority_reason or "quoted evidence",
                    execution_context,
                    target_session,
                )
            )
    payload = json.dumps(
        {
            "budget": token_budget,
            "execution_context": execution_context.context_id,
            "items": [_item_dict(item) for item in (*included_evidence, *included_policy)],
            "ledger": [row.as_dict() for row in rows],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    build_ref = "sha256:" + hashlib.sha256(payload.encode()).hexdigest()
    return ContextAssembly(
        tuple(included_evidence),
        tuple(included_policy),
        tuple(rows),
        token_budget - remaining,
        token_budget,
        execution_context,
        build_ref,
    )


def _row(
    decision: Decision,
    item: ContextItem,
    rank: int,
    before: int,
    after: int,
    disclosure: str,
    authority: str,
    reason: str,
    execution: ExecutionContextRef,
    target: str | None,
) -> ContextLedgerRow:
    return ContextLedgerRow(
        decision,
        item.source,
        item.ref,
        max(0, item.token_cost),
        rank,
        before,
        after,
        disclosure,
        authority,
        reason,
        item.policy_refs,
        target,
        execution.context_id,
    )


def record_context_ledger(conn: sqlite3.Connection, assembly: ContextAssembly, *, observed_at_ms: int) -> None:
    """Persist the append-only admission receipt in disposable ``ops.db``."""
    for index, row in enumerate(assembly.ledger):
        ledger_id = hashlib.sha256(f"{assembly.build_ref}:{index}:{row.item_ref}".encode()).hexdigest()
        conn.execute(
            "INSERT OR IGNORE INTO context_injection_ledger VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                ledger_id,
                assembly.build_ref,
                observed_at_ms,
                row.decision,
                row.source,
                row.item_ref,
                row.token_cost,
                row.source_local_rank,
                row.budget_before,
                row.budget_after,
                row.disclosure_verdict,
                row.authority_verdict,
                row.authority_reason,
                json.dumps(row.policy_refs, separators=(",", ":")),
                row.target_session,
                row.execution_context_ref,
            ),
        )
    conn.commit()


def read_context_ledger(
    conn: sqlite3.Connection,
    *,
    target_session: str | None = None,
    execution_context_ref: str | None = None,
    limit: int = 100,
) -> tuple[ContextLedgerRecord, ...]:
    """Read bounded scheduler receipts without creating or mutating the table."""

    if limit < 1:
        raise ValueError("context ledger limit must be positive")
    where: list[str] = []
    params: list[object] = []
    if target_session is not None:
        where.append("target_session = ?")
        params.append(target_session)
    if execution_context_ref is not None:
        where.append("execution_context_ref = ?")
        params.append(execution_context_ref)
    clause = " WHERE " + " AND ".join(where) if where else ""
    rows = conn.execute(
        f"""
        SELECT ledger_id, build_ref, observed_at_ms, decision, source, item_ref,
               token_cost, source_local_rank, budget_before, budget_after,
               disclosure_verdict, authority_verdict, authority_reason,
               policy_refs_json, target_session, execution_context_ref
        FROM context_injection_ledger{clause}
        ORDER BY observed_at_ms DESC, ledger_id DESC
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    records: list[ContextLedgerRecord] = []
    for row in rows:
        policy_refs = json.loads(str(row[13]))
        if not isinstance(policy_refs, list) or not all(isinstance(item, str) for item in policy_refs):
            raise ValueError("stored context ledger policy refs are not a string list")
        records.append(
            ContextLedgerRecord(
                ledger_id=str(row[0]),
                build_ref=str(row[1]),
                observed_at_ms=int(row[2]),
                row=ContextLedgerRow(
                    decision=row[3],
                    source=str(row[4]),
                    item_ref=str(row[5]),
                    token_cost=int(row[6]),
                    source_local_rank=int(row[7]),
                    budget_before=int(row[8]),
                    budget_after=int(row[9]),
                    disclosure_verdict=str(row[10]),
                    authority_verdict=str(row[11]),
                    authority_reason=str(row[12]),
                    policy_refs=tuple(policy_refs),
                    target_session=None if row[14] is None else str(row[14]),
                    execution_context_ref=str(row[15]),
                ),
            )
        )
    return tuple(records)


__all__ = [
    "ContextAssembly",
    "ContextItem",
    "ContextLedgerRecord",
    "ContextLedgerRow",
    "ContextSource",
    "read_context_ledger",
    "TrustClass",
    "record_context_ledger",
    "schedule_context",
]
