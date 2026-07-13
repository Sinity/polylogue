"""Planner-owned execution seam for durable canonical query definitions.

Canonical query identity is deliberately not an executable serialization.  A
planner implementation receives its typed canonical definition here and owns
the language-version dispatch, temporal binding, and source access needed to
evaluate it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal, Protocol

from polylogue.archive.query.expression import RefOperand, RelationGrain, ResolvedRefOperand
from polylogue.core.refs import ObjectRef
from polylogue.storage.sqlite.query_objects import (
    EvaluationReceipt,
    QueryObject,
    get_query,
    get_result_set,
    get_result_set_members,
    get_retained_query_run,
    put_query_edge,
)

EvaluationPurpose = Literal["reference", "standing-watch", "finding-drift"]


class RetainedRelationUnavailableError(ValueError):
    """A reference requires a durable relation that was never retained."""


@dataclass(frozen=True, slots=True)
class QueryEvaluationRequest:
    query: QueryObject
    purpose: EvaluationPurpose
    changed_session_ids: tuple[str, ...] = ()
    excluded_scope_refs: tuple[str, ...] = ()
    excluded_origin_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    grain: RelationGrain
    member_refs: tuple[str, ...]
    corpus_epoch: str
    exactness: Literal["exact", "capped", "sampled", "estimate"]
    receipt: EvaluationReceipt
    cache_only: bool = False


class CanonicalPlanEvaluator(Protocol):
    """Evaluate canonical definitions without reverse-compiling identity JSON."""

    def evaluate(self, request: QueryEvaluationRequest) -> QueryEvaluation: ...

    def resolve_cohort(self, operand: RefOperand) -> QueryEvaluation: ...


class DurableRefResolver:
    """Resolve `from` operands through durable manifests and the planner seam."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        evaluator: CanonicalPlanEvaluator,
        *,
        owner_query_hash: str | None = None,
        created_at_ms: int = 0,
    ) -> None:
        self._conn = conn
        self._evaluator = evaluator
        self._owner_query_hash = owner_query_hash
        self._created_at_ms = created_at_ms

    def resolve_ref_operand(self, operand: RefOperand) -> ResolvedRefOperand:
        reference = operand.reference
        if reference.kind == "query":
            query = get_query(self._conn, reference.object_id)
            if query is None:
                raise KeyError(reference.format())
            evaluation = self._evaluator.evaluate(QueryEvaluationRequest(query=query, purpose="reference"))
            self._record_operand_edge(query.query_hash)
            return ResolvedRefOperand(
                operand=operand,
                grain=evaluation.grain,
                lineage=(query_ref_for(query),),
                member_refs=evaluation.member_refs,
            )
        if reference.kind == "query-run":
            retained = get_retained_query_run(self._conn, reference.object_id)
            if retained is None:
                raise RetainedRelationUnavailableError(
                    f"query run {reference.format()} has no retained relation; use query:<hash> to re-evaluate"
                )
            return self._retained_result(operand, retained.result_set_id, extra_lineage=(reference,))
        if reference.kind == "result-set":
            return self._retained_result(operand, reference.object_id)
        if reference.kind == "cohort":
            evaluation = self._evaluator.resolve_cohort(operand)
            return ResolvedRefOperand(
                operand=operand,
                grain=evaluation.grain,
                lineage=(reference,),
                member_refs=evaluation.member_refs,
            )
        raise ValueError(f"unsupported ref operand kind: {reference.kind}")

    def _retained_result(
        self,
        operand: RefOperand,
        result_set_id: str,
        *,
        extra_lineage: tuple[ObjectRef, ...] = (),
    ) -> ResolvedRefOperand:
        manifest = get_result_set(self._conn, result_set_id)
        if manifest is None:
            raise RetainedRelationUnavailableError(f"retained result-set:{result_set_id} is unavailable")
        members = get_result_set_members(self._conn, result_set_id)
        if manifest.member_count != len(members):
            raise RetainedRelationUnavailableError(
                f"result-set:{result_set_id} has no retained exact relation; use query:<hash> to re-evaluate"
            )
        query = get_query(self._conn, manifest.query_hash)
        lineage = (*extra_lineage, ObjectRef(kind="result-set", object_id=result_set_id))
        if query is not None:
            lineage = (*lineage, query_ref_for(query))
        return ResolvedRefOperand(
            operand=operand,
            grain=manifest.grain,  # type: ignore[arg-type]
            lineage=lineage,
            member_refs=members,
        )

    def _record_operand_edge(self, target_query_hash: str) -> None:
        if self._owner_query_hash is None:
            return
        put_query_edge(
            self._conn,
            src_query_hash=self._owner_query_hash,
            dst_query_hash=target_query_hash,
            edge_kind="operand-of",
            created_at_ms=self._created_at_ms,
        )


def query_ref_for(query: QueryObject) -> ObjectRef:
    return ObjectRef(kind="query", object_id=query.query_hash)


__all__ = [
    "CanonicalPlanEvaluator",
    "DurableRefResolver",
    "EvaluationPurpose",
    "QueryEvaluation",
    "QueryEvaluationRequest",
    "RetainedRelationUnavailableError",
]
