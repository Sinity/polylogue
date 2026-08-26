"""Plan/apply/receipt lifecycle for promoted query evidence.

The planner deliberately reports identities and counts, never canonical query
payloads.  The user-tier tombstone ledger is written before rows are removed;
query writers consult it, so a reset or later evaluator cannot recreate an
excised definition or relation.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Literal

from polylogue.core.enums import AssertionKind
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion

QueryTargetKind = Literal["query", "result-set"]


@dataclass(frozen=True, slots=True)
class QueryExcisionPlan:
    target_ref: str
    target_kind: QueryTargetKind
    query_hashes: tuple[str, ...]
    result_set_ids: tuple[str, ...]
    names: tuple[str, ...]
    edge_refs: tuple[str, ...]
    member_refs: tuple[str, ...]
    retained_run_ids: tuple[str, ...]
    evaluation_receipt_ids: tuple[str, ...]
    finding_report_refs: tuple[str, ...]
    vector_export_refs: tuple[str, ...]
    backed_replica_refs: tuple[str, ...]
    held_refs: tuple[str, ...] = ()
    unsupported_refs: tuple[str, ...] = ()
    operational_refs: tuple[str, ...] = ()

    @property
    def all_refs(self) -> tuple[str, ...]:
        return (
            self.query_hashes
            + self.result_set_ids
            + self.names
            + self.edge_refs
            + self.member_refs
            + self.retained_run_ids
            + self.evaluation_receipt_ids
            + self.finding_report_refs
            + self.vector_export_refs
            + self.backed_replica_refs
            + self.held_refs
            + self.unsupported_refs
            + self.operational_refs
        )


@dataclass(frozen=True, slots=True)
class QueryExcisionReceipt:
    target_ref: str
    status: Literal["applied", "held", "unsupported"]
    removed_refs: tuple[str, ...]
    tombstoned_refs: tuple[str, ...]
    held_refs: tuple[str, ...]
    unsupported_refs: tuple[str, ...]
    receipt_id: str


def _target(target_ref: str) -> tuple[QueryTargetKind, str]:
    if target_ref.startswith("query:"):
        return "query", target_ref.removeprefix("query:")
    if target_ref.startswith("result-set:"):
        return "result-set", target_ref.removeprefix("result-set:")
    raise ValueError("query excision target must be query:<hash> or result-set:<id>")


def _json_refs(value: object) -> tuple[str, ...]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(str(item) for item in parsed if isinstance(item, str))


def plan_query_excision(conn: sqlite3.Connection, target_ref: str) -> QueryExcisionPlan:
    """Enumerate the authorized graph without returning secret query text."""
    target_kind, target_id = _target(target_ref)
    query_hashes: tuple[str, ...]
    result_set_ids: tuple[str, ...]
    if target_kind == "query":
        query_hashes = (target_id,)
        result_set_ids = tuple(
            str(row[0])
            for row in conn.execute("SELECT result_set_id FROM result_sets WHERE query_hash = ?", (target_id,))
        )
    else:
        result_set_ids = (target_id,)
        row = conn.execute("SELECT query_hash FROM result_sets WHERE result_set_id = ?", (target_id,)).fetchone()
        query_hashes = (str(row[0]),) if row is not None else ()
    qmarks = ",".join("?" for _ in query_hashes) or "NULL"
    rmarks = ",".join("?" for _ in result_set_ids) or "NULL"
    names = tuple(
        str(row[0])
        for row in conn.execute(f"SELECT name FROM query_names WHERE query_hash IN ({qmarks})", query_hashes)
    )
    edges = tuple(
        f"query:{row[0]}->{row[1]}:{row[2]}"
        for row in conn.execute(
            f"SELECT src_query_hash, dst_query_hash, edge_kind FROM query_edges WHERE src_query_hash IN ({qmarks}) OR dst_query_hash IN ({qmarks})",
            query_hashes + query_hashes,
        )
    )
    members = tuple(
        str(row[0])
        for row in conn.execute(
            f"SELECT member_ref FROM result_set_members WHERE result_set_id IN ({rmarks})", result_set_ids
        )
    )
    retained = tuple(
        str(row[0])
        for row in conn.execute(
            f"SELECT run_id FROM retained_query_runs WHERE query_hash IN ({qmarks}) OR result_set_id IN ({rmarks})",
            query_hashes + result_set_ids,
        )
    )
    receipts = tuple(
        str(row[0])
        for row in conn.execute(
            f"SELECT receipt_id FROM query_evaluation_receipts WHERE query_hash IN ({qmarks}) OR result_set_id IN ({rmarks})",
            query_hashes + result_set_ids,
        )
    )
    refs = tuple("query:" + item for item in query_hashes) + tuple("result-set:" + item for item in result_set_ids)
    finding_refs: list[str] = []
    for row in conn.execute("SELECT assertion_id, target_ref, evidence_refs_json FROM assertions"):
        if str(row[1]) in refs or refs and set(_json_refs(row[2])) & set(refs):
            finding_refs.append(str(row[0]))
    return QueryExcisionPlan(
        target_ref=target_ref,
        target_kind=target_kind,
        query_hashes=query_hashes,
        result_set_ids=result_set_ids,
        names=names,
        edge_refs=edges,
        member_refs=members,
        retained_run_ids=retained,
        evaluation_receipt_ids=receipts,
        finding_report_refs=tuple(finding_refs),
        vector_export_refs=(),
        backed_replica_refs=(),
    )


def apply_query_excision(
    conn: sqlite3.Connection,
    plan: QueryExcisionPlan,
    *,
    reason: str,
    actor: str,
    now_ms: int,
) -> QueryExcisionReceipt:
    """Apply one previously resolved plan and emit a secret-free receipt."""
    if plan.held_refs or plan.unsupported_refs:
        return QueryExcisionReceipt(plan.target_ref, "held", (), (), plan.held_refs, plan.unsupported_refs, "")
    reason_digest = hashlib.sha256(reason.encode("utf-8", errors="surrogatepass")).hexdigest()
    query_hashes_to_remove = plan.query_hashes if plan.target_kind == "query" else ()
    ledger_ids: list[str] = []
    for query_hash in query_hashes_to_remove:
        ledger_id = f"query-excision:{hashlib.sha256(('query:' + query_hash).encode()).hexdigest()}"
        conn.execute(
            "INSERT OR IGNORE INTO query_excision_ledger (ledger_id, query_hash, excision_link, reason_digest, actor_ref, prior_revision, excised_at_ms) VALUES (?, ?, ?, ?, ?, 0, ?)",
            (ledger_id, query_hash, plan.target_ref, reason_digest, actor, now_ms),
        )
        ledger_ids.append(ledger_id)
    for result_set_id in plan.result_set_ids:
        ledger_id = f"query-excision:{hashlib.sha256(('result-set:' + result_set_id).encode()).hexdigest()}"
        conn.execute(
            "INSERT OR IGNORE INTO query_excision_ledger (ledger_id, result_set_id, excision_link, reason_digest, actor_ref, prior_revision, excised_at_ms) VALUES (?, ?, ?, ?, ?, 0, ?)",
            (ledger_id, result_set_id, plan.target_ref, reason_digest, actor, now_ms),
        )
        ledger_ids.append(ledger_id)
    if plan.finding_report_refs:
        marks = ",".join("?" for _ in plan.finding_report_refs)
        conn.execute(
            f"UPDATE assertions SET status = 'deleted', value_json = NULL, body_text = NULL, updated_at_ms = ? WHERE assertion_id IN ({marks})",
            (now_ms, *plan.finding_report_refs),
        )
    if plan.result_set_ids:
        marks = ",".join("?" for _ in plan.result_set_ids)
        conn.execute(f"DELETE FROM watched_query_baselines WHERE result_set_id IN ({marks})", plan.result_set_ids)
        conn.execute(f"DELETE FROM result_set_holdout_policies WHERE result_set_id IN ({marks})", plan.result_set_ids)
        conn.execute(f"DELETE FROM retained_query_runs WHERE result_set_id IN ({marks})", plan.result_set_ids)
        conn.execute(f"DELETE FROM query_evaluation_receipts WHERE result_set_id IN ({marks})", plan.result_set_ids)
        conn.execute(f"DELETE FROM result_set_members WHERE result_set_id IN ({marks})", plan.result_set_ids)
    if query_hashes_to_remove:
        marks = ",".join("?" for _ in query_hashes_to_remove)
        conn.execute(f"DELETE FROM watched_query_baselines WHERE query_hash IN ({marks})", query_hashes_to_remove)
        conn.execute(f"DELETE FROM retained_query_runs WHERE query_hash IN ({marks})", query_hashes_to_remove)
        conn.execute(f"DELETE FROM query_evaluation_receipts WHERE query_hash IN ({marks})", query_hashes_to_remove)
        conn.execute(
            f"UPDATE query_names SET supersedes_query_hash = NULL WHERE supersedes_query_hash IN ({marks})",
            query_hashes_to_remove,
        )
        conn.execute(f"DELETE FROM query_names WHERE query_hash IN ({marks})", query_hashes_to_remove)
        conn.execute(
            f"DELETE FROM query_edges WHERE src_query_hash IN ({marks}) OR dst_query_hash IN ({marks})",
            query_hashes_to_remove + query_hashes_to_remove,
        )
    if plan.result_set_ids:
        marks = ",".join("?" for _ in plan.result_set_ids)
        conn.execute(f"DELETE FROM result_sets WHERE result_set_id IN ({marks})", plan.result_set_ids)
    if query_hashes_to_remove:
        marks = ",".join("?" for _ in query_hashes_to_remove)
        conn.execute(f"DELETE FROM queries WHERE query_hash IN ({marks})", query_hashes_to_remove)
    receipt_id = (
        f"assertion-{AssertionKind.EXCISION_RECORD}:query:{hashlib.sha256(plan.target_ref.encode()).hexdigest()}"
    )
    upsert_assertion(
        conn,
        assertion_id=receipt_id,
        target_ref=plan.target_ref,
        kind=AssertionKind.EXCISION_RECORD,
        value={"status": "applied", "removed_count": len(plan.all_refs), "tombstone_count": len(ledger_ids)},
        author_ref=actor,
        author_kind="user",
        now_ms=now_ms,
        require_promotion=False,
    )
    return QueryExcisionReceipt(plan.target_ref, "applied", plan.all_refs, tuple(ledger_ids), (), (), receipt_id)


__all__ = ["QueryExcisionPlan", "QueryExcisionReceipt", "apply_query_excision", "plan_query_excision"]
