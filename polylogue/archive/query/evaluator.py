"""Planner-owned execution seam for durable canonical query definitions.

Canonical query identity is deliberately not an executable serialization.  A
planner implementation receives its typed canonical definition here and owns
the language-version dispatch, temporal binding, and source access needed to
evaluate it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from polylogue.archive.query.expression import RefOperand, RelationGrain, ResolvedRefOperand
from polylogue.core.query_identity import (
    LEGACY_QUERY_DEFINITION_PROTOCOL_VERSION,
    require_supported_definition_protocol_version,
)
from polylogue.core.refs import ObjectRef
from polylogue.storage.sqlite.holdout_cohorts import (
    HoldoutAccessError,
    require_non_holdout_access,
)
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

    def __post_init__(self) -> None:
        require_supported_definition_protocol_version(self.query.definition_protocol_version)


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


@runtime_checkable
class ScopedCanonicalPlanEvaluator(Protocol):
    """An evaluator that also publishes a bound on what a definition can match.

    Deliberately separate from :class:`CanonicalPlanEvaluator`: publishing a
    bound is an extra capability, not a new obligation on every planner. A
    caller narrows only against an evaluator that implements this, and keeps
    its exhaustive baseline against one that does not -- so an evaluator that
    has never proved a bound is never read as having proved an empty one.
    """

    def session_origin_scope(self, query: QueryObject) -> frozenset[str] | None:
        """Origins bounding this definition's membership, or ``None`` for unbounded."""
        ...


def session_origin(session_id: str) -> str | None:
    """Return the origin token embedded in a session identity, if it has one.

    ``sessions.session_id`` is the generated column ``origin || ':' || native_id``
    and no public origin token contains a colon, so the prefix before the first
    colon *is* the origin. It is part of the computed identity, which is why a
    session cannot migrate between origins without becoming a different session.

    An identifier with no colon is not a session identity this rule can read;
    the caller must treat it as unbounded rather than guess.
    """
    origin, separator, _native_id = session_id.partition(":")
    return origin if separator and origin else None


def declared_origin_scope(query: QueryObject) -> frozenset[str] | None:
    """Derive the origins a canonical definition's members must carry.

    The result is an over-approximation by construction: it is ``None``
    unless every satisfying session provably carries one of the returned
    origins. Only shapes where that implication holds contribute a bound --

    * an ``origin`` equality leaf bounds membership to its listed values;
    * ``and`` intersects whichever children are bounded, because a conjunct
      can only remove members;
    * ``or`` is bounded only when *every* branch is, because one unbounded
      branch can admit any origin;
    * ``not`` and every other predicate kind are unbounded, because a
      negation or a correlated/text predicate can match any origin.

    Nothing here inspects the archive: the bound is a property of the
    definition, so it cannot go stale against an index generation and it is
    recomputed from the current definition on every use.
    """
    from polylogue.archive.query.predicate import (
        QueryBoolPredicate,
        QueryFieldPredicate,
        QueryPredicate,
    )

    if query.definition_protocol_version == LEGACY_QUERY_DEFINITION_PROTOCOL_VERSION:
        return None
    if query.grain != "session":
        return None
    ast = query.canonical_plan.get("ast")
    if not isinstance(ast, dict):
        return None
    try:
        from polylogue.archive.query.predicate import predicate_from_payload

        predicate = predicate_from_payload(ast)
    except Exception:  # an unreadable definition is unbounded, not empty
        return None

    def bound(node: QueryPredicate) -> frozenset[str] | None:
        if isinstance(node, QueryFieldPredicate):
            if node.field != "origin" or node.op != "=" or not node.values:
                return None
            return frozenset(node.values)
        if isinstance(node, QueryBoolPredicate):
            child_bounds = [bound(child) for child in node.children]
            if node.op == "and":
                known = [item for item in child_bounds if item is not None]
                if not known:
                    return None
                return frozenset.intersection(*known)
            if node.op == "or":
                if not child_bounds or any(item is None for item in child_bounds):
                    return None
                return frozenset().union(*[item for item in child_bounds if item is not None])
            return None
        return None

    return bound(predicate)


class DurableRefResolver:
    """Resolve `from` operands through durable manifests and the planner seam."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        evaluator: CanonicalPlanEvaluator,
        *,
        owner_query_hash: str | None = None,
        created_at_ms: int = 0,
        declared_confirmation: bool = False,
    ) -> None:
        self._conn = conn
        self._evaluator = evaluator
        self._owner_query_hash = owner_query_hash
        self._created_at_ms = created_at_ms
        self._declared_confirmation = declared_confirmation

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
        try:
            require_non_holdout_access(
                self._conn,
                result_set_id,
                declared_confirmation=self._declared_confirmation,
            )
        except HoldoutAccessError as exc:
            raise RetainedRelationUnavailableError(str(exc)) from exc
        if extra_lineage:
            run = get_retained_query_run(self._conn, extra_lineage[0].object_id)
            if run is None or manifest.query_hash != run.query_hash:
                raise RetainedRelationUnavailableError(
                    f"query run {extra_lineage[0].format()} does not retain a relation for its query"
                )
        members = get_result_set_members(self._conn, result_set_id)
        if manifest.member_count != len(members):
            raise RetainedRelationUnavailableError(
                f"result-set:{result_set_id} has no retained exact relation; use query:<hash> to re-evaluate"
            )
        if manifest.exactness != "exact":
            raise RetainedRelationUnavailableError(
                f"result-set:{result_set_id} is {manifest.exactness}, not an exact set operand; use query:<hash> to re-evaluate"
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
    "ScopedCanonicalPlanEvaluator",
    "declared_origin_scope",
    "session_origin",
]
