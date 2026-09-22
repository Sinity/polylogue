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
from dataclasses import dataclass, replace
from typing import Literal

from polylogue.core.enums import AssertionKind
from polylogue.core.sqlite_introspection import table_exists
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


def _blocking_holdout_receipts(conn: sqlite3.Connection, result_set_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Name the holdout access receipts that forbid deleting their own policy.

    ``holdout_access_receipts.result_set_id`` references
    ``result_set_holdout_policies`` ``ON DELETE RESTRICT``, and every holdout
    access writes one. Deleting the parent policy therefore raised
    ``sqlite3.IntegrityError`` *after* the ledger rows and the assertion scrub
    had already run in the caller's transaction. Whether a contamination
    receipt may be destroyed alongside the relation it records is a durable
    privacy decision, not something the applier may take silently: the plan
    reports the receipts and the apply refuses.
    """
    if not result_set_ids or not table_exists(conn, "holdout_access_receipts"):
        return ()
    marks = ",".join("?" for _ in result_set_ids)
    return tuple(
        f"holdout-access-receipt:{row[0]}"
        for row in conn.execute(
            f"SELECT receipt_id FROM holdout_access_receipts WHERE result_set_id IN ({marks}) ORDER BY receipt_id",
            result_set_ids,
        )
    )


def plan_query_excision(conn: sqlite3.Connection, target_ref: str) -> QueryExcisionPlan:
    """Enumerate the authorized graph without returning secret query text.

    A ``result-set:`` target is a *relation-only* excision: it deliberately
    preserves the owning query definition. The plan therefore enumerates only
    what such an excision removes. Scanning the owning query's refs as well
    put every assertion targeting or citing that surviving query into
    ``finding_report_refs``, so excising one snapshot marked unrelated notes,
    findings and annotations deleted and cleared their content -- irreplaceable
    user-tier state. Its query names, edges, retained runs and evaluation
    receipts were likewise reported as removed while the applier left them in
    place.
    """
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
    # The query-scoped relations a relation-only plan does not touch.
    query_scope = query_hashes if target_kind == "query" else ()
    qmarks = ",".join("?" for _ in query_scope) or "NULL"
    rmarks = ",".join("?" for _ in result_set_ids) or "NULL"
    names = tuple(
        str(row[0]) for row in conn.execute(f"SELECT name FROM query_names WHERE query_hash IN ({qmarks})", query_scope)
    )
    edges = tuple(
        f"query:{row[0]}->{row[1]}:{row[2]}"
        for row in conn.execute(
            f"SELECT src_query_hash, dst_query_hash, edge_kind FROM query_edges WHERE src_query_hash IN ({qmarks}) OR dst_query_hash IN ({qmarks})",
            query_scope + query_scope,
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
            query_scope + result_set_ids,
        )
    )
    receipts = tuple(
        str(row[0])
        for row in conn.execute(
            f"SELECT receipt_id FROM query_evaluation_receipts WHERE query_hash IN ({qmarks}) OR result_set_id IN ({rmarks})",
            query_scope + result_set_ids,
        )
    )
    refs = tuple("query:" + item for item in query_scope) + tuple("result-set:" + item for item in result_set_ids)
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
        held_refs=_blocking_holdout_receipts(conn, result_set_ids),
    )


def apply_query_excision(
    conn: sqlite3.Connection,
    plan: QueryExcisionPlan,
    *,
    reason: str,
    actor: str,
    now_ms: int,
) -> QueryExcisionReceipt:
    """Apply one plan against the graph as it stands in this transaction.

    The plan is a preview, and durable state can change between resolving it
    and applying it: the standing-query writer can add a finding citing the
    target, or a new result set can attach to it. Trusting the old plan
    returned ``applied`` while leaving that newly added assertion's content
    behind -- and a new result set made the later query delete fail on its
    foreign key after earlier mutations had already run. Single-writer
    ownership does not prevent other queued daemon writes between the two
    calls, so the graph is re-resolved here and a drift is refused rather than
    silently narrowed or widened.
    """
    if plan.held_refs or plan.unsupported_refs:
        return QueryExcisionReceipt(plan.target_ref, "held", (), (), plan.held_refs, plan.unsupported_refs, "")
    plan = _resolved_plan_or_hold(conn, plan)
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
        # Every content-bearing column, not only the two obvious ones. An
        # annotation can carry user text in `key`, a path or session in
        # `scope_ref`, and arbitrary operator JSON in `staleness_json` /
        # `context_policy_json` / `supersedes_json` / `evidence_refs_json`;
        # clearing `value_json`/`body_text` alone left all of those readable
        # while the receipt reported a successful excision. `author_ref` and
        # `author_kind` are deliberately retained: they are the accountability
        # record for the excised note, not its content.
        conn.execute(
            f"""
            UPDATE assertions
               SET status = 'deleted',
                   key = NULL,
                   value_json = NULL,
                   body_text = NULL,
                   scope_ref = NULL,
                   staleness_json = NULL,
                   confidence = NULL,
                   evidence_refs_json = '[]',
                   supersedes_json = '[]',
                   context_policy_json = '{{"inject":false}}',
                   visibility = 'private',
                   updated_at_ms = ?
             WHERE assertion_id IN ({marks})
            """,
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


def _resolved_plan_or_hold(conn: sqlite3.Connection, plan: QueryExcisionPlan) -> QueryExcisionPlan:
    """Re-resolve ``plan``'s target here, or hold the exact refs that drifted."""
    current = plan_query_excision(conn, plan.target_ref)
    if current.held_refs or current.unsupported_refs:
        return current
    drifted = tuple(sorted(set(current.all_refs) ^ set(plan.all_refs)))
    if drifted:
        return replace(current, held_refs=drifted)
    return current


__all__ = ["QueryExcisionPlan", "QueryExcisionReceipt", "apply_query_excision", "plan_query_excision"]
