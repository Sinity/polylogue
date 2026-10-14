"""Execute canonical saved-query plans and receipts on one admitted archive read.

Only session-grain current-protocol definitions execute. Earlier saved-view
shapes and other grains retain their explicit refusals.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import closing
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from polylogue.archive.query.evaluator import (
    CanonicalPlanEvaluator,
    QueryEvaluation,
    QueryEvaluationRequest,
    declared_origin_scope,
)
from polylogue.archive.query.expression import RefOperand
from polylogue.archive.query.filter_kwargs import plan_filter_kwargs
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.archive.query.predicate import predicate_from_payload
from polylogue.core.query_identity import LEGACY_QUERY_DEFINITION_PROTOCOL_VERSION
from polylogue.logging import get_logger
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.query_objects import EvaluationReceipt, QueryObject

logger = get_logger(__name__)

_SUPPORTED_GRAINS: frozenset[str] = frozenset({"session"})


class LegacyQueryDefinitionNotExecutableError(ValueError):
    """A protocol-v0 (legacy saved-view) query has no executable planner form."""


class UnsupportedEvaluationGrainError(ValueError):
    """The production evaluator does not yet execute this relation grain."""


def _polylogue_runtime_build_ref() -> str:
    try:
        return f"polylogue:{_package_version('polylogue')}"
    except PackageNotFoundError:
        return "polylogue:unknown"


def _tier_generation(conn: sqlite3.Connection, *, schema: str, label: str) -> str:
    aliases = {str(row[1]) for row in conn.execute("PRAGMA database_list")}
    if schema not in aliases:
        return f"{label}:absent"
    version = int(conn.execute(f"PRAGMA {schema}.user_version").fetchone()[0])
    return f"{label}:v{version}"


def _index_epoch(conn: sqlite3.Connection) -> str:
    version = int(conn.execute("PRAGMA main.user_version").fetchone()[0])
    row = conn.execute("SELECT MAX(updated_at_ms) FROM main.sessions").fetchone()
    watermark = int(row[0]) if row[0] is not None else 0
    return f"index:v{version}:{watermark}"


class ArchiveCanonicalPlanEvaluator(CanonicalPlanEvaluator):
    """Evaluate definitions and receipt facts on the caller's admitted reader."""

    def __init__(self, archive: ArchiveStore) -> None:
        self._archive = archive

    def _connection(self) -> sqlite3.Connection:
        self._archive.check_operation_read()
        connection = self._archive.index_connection
        if connection is None or not connection.in_transaction:
            raise RuntimeError("canonical evaluation requires an active pinned read")
        return connection

    def evaluate(self, request: QueryEvaluationRequest) -> QueryEvaluation:
        connection = self._connection()
        query = request.query
        if query.definition_protocol_version == LEGACY_QUERY_DEFINITION_PROTOCOL_VERSION:
            raise LegacyQueryDefinitionNotExecutableError(
                f"query:{query.query_hash} uses the legacy protocol-v0 definition; "
                "legacy saved-view identities predate the executable predicate grammar "
                "and cannot be re-evaluated by the planner"
            )
        if query.grain not in _SUPPORTED_GRAINS:
            raise UnsupportedEvaluationGrainError(
                f"query:{query.query_hash} has grain {query.grain!r}; the production "
                f"evaluator currently executes only {sorted(_SUPPORTED_GRAINS)!r}"
            )
        ast = query.canonical_plan.get("ast")
        if not isinstance(ast, dict):
            raise ValueError(f"query:{query.query_hash} canonical plan has no executable 'ast'")

        from polylogue.archive.query.expression import _bind_predicate_context  # planner-internal seam

        predicate = predicate_from_payload(ast)
        bound = _bind_predicate_context(predicate, unit="session")
        plan = SessionQueryPlan(boolean_predicate=bound)
        excluded_prefixes = tuple(request.excluded_origin_prefixes)
        try:
            with closing(self._archive.iter_session_identities(**plan_filter_kwargs(plan))) as identities:
                members: list[str] = []
                for identity in identities:
                    self._connection()
                    if not any(identity.origin.startswith(prefix) for prefix in excluded_prefixes):
                        members.append(f"session:{identity.session_id}")
                member_refs = tuple(members)
        except Exception:
            logger.warning("production-evaluator: evaluation failed for query:%s", query.query_hash, exc_info=True)
            raise

        excluded_refs = set(request.excluded_scope_refs)
        if excluded_refs:
            member_refs = tuple(ref for ref in member_refs if ref not in excluded_refs)

        self._connection()
        index_generation = _index_epoch(connection)
        receipt = EvaluationReceipt(
            receipt_id=f"receipt-{uuid.uuid4().hex}",
            source_generation=_tier_generation(connection, schema="source_tier", label="source"),
            user_generation=_tier_generation(connection, schema="user_tier", label="user"),
            index_generation=index_generation,
            runtime_build_ref=_polylogue_runtime_build_ref(),
        )
        evaluation = QueryEvaluation(
            grain="session",
            member_refs=member_refs,
            corpus_epoch=index_generation,
            exactness="exact",
            receipt=receipt,
        )
        self._connection()
        return evaluation

    def session_origin_scope(self, query: QueryObject) -> frozenset[str] | None:
        """Bound this definition's members by origin, from the definition alone.

        The same planner that executes the definition owns the claim about
        what it can match, so the bound cannot drift away from the predicate
        grammar this module already inverts. Shapes it has not proved a bound
        for return ``None``, and every caller then keeps its global baseline.
        """
        if query.definition_protocol_version == LEGACY_QUERY_DEFINITION_PROTOCOL_VERSION:
            return None
        if query.grain not in _SUPPORTED_GRAINS:
            return None
        return declared_origin_scope(query)

    def resolve_cohort(self, operand: RefOperand) -> QueryEvaluation:
        raise NotImplementedError(
            f"cohort substrate is not implemented yet ({operand.reference.format()}); see polylogue-rxdo.6"
        )


__all__ = [
    "ArchiveCanonicalPlanEvaluator",
    "LegacyQueryDefinitionNotExecutableError",
    "UnsupportedEvaluationGrainError",
]
