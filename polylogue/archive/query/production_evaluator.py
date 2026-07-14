"""Real (non-test-double) planner implementation for canonical query plans.

Before this module existed, :class:`~polylogue.archive.query.evaluator.
CanonicalPlanEvaluator` had exactly one kind of implementation anywhere in the
repository: hand-rolled fakes inside test files. ``ArchiveCanonicalPlanEvaluator``
is the first production evaluator -- it turns a durable :class:`QueryObject`'s
canonical, protocol-versioned AST back into an executable
:class:`~polylogue.archive.query.plan.SessionQueryPlan` and runs it through the
same :class:`~polylogue.archive.filter.filters.SessionFilter` every other
surface (CLI/MCP/API) uses, then records a bounded ``ops.db`` ``query_runs``
telemetry row for the execution (surface ``daemon-internal``).

Only the ``session`` grain and the current (v1) definition protocol are
evaluated. Legacy protocol v0 identities (opaque saved-view JSON, not this
predicate grammar) and non-session grains fail closed with a named error
rather than guessing -- this mirrors the "never reverse-compile lossy
identity JSON" doctrine: this module only inverts the *same* typed
``to_payload()`` shape ``polylogue.archive.query.predicate`` itself owns.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import closing
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import Literal

from polylogue.archive.filter.filters import SessionFilter
from polylogue.archive.query.evaluator import (
    CanonicalPlanEvaluator,
    QueryEvaluation,
    QueryEvaluationRequest,
)
from polylogue.archive.query.expression import RefOperand
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.archive.query.predicate import predicate_from_payload
from polylogue.core.query_identity import LEGACY_QUERY_DEFINITION_PROTOCOL_VERSION
from polylogue.logging import get_logger
from polylogue.storage.sqlite.query_objects import EvaluationReceipt

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


def _pragma_user_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row is not None else 0


def _tier_generation(db_path: Path, *, label: str) -> str:
    if not db_path.exists():
        return f"{label}:absent"
    try:
        with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)) as conn:
            return f"{label}:v{_pragma_user_version(conn)}"
    except sqlite3.Error:
        logger.warning("production-evaluator: could not read %s generation", label, exc_info=True)
        return f"{label}:unknown"


def _index_epoch(index_db: Path) -> str:
    """Index generation identifier: schema version plus a session watermark."""
    if not index_db.exists():
        return "index:absent"
    try:
        with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True, timeout=5.0)) as conn:
            version = _pragma_user_version(conn)
            row = conn.execute("SELECT MAX(updated_at_ms) FROM sessions").fetchone()
            watermark = int(row[0]) if row is not None and row[0] is not None else 0
            return f"index:v{version}:{watermark}"
    except sqlite3.Error:
        logger.warning("production-evaluator: could not read index epoch", exc_info=True)
        return "index:unknown"


class ArchiveCanonicalPlanEvaluator(CanonicalPlanEvaluator):
    """Evaluate durable canonical query definitions against the live archive.

    ``db_path`` is the ``index.db`` path (the same convention used throughout
    ``daemon/convergence_stages.py``); ``source.db``/``user.db``/``ops.db``
    are resolved as siblings.
    """

    def __init__(self, db_path: Path, *, surface: str = "daemon-internal") -> None:
        self._db_path = db_path
        self._archive_root = db_path.parent
        self._surface = surface

    def evaluate(self, request: QueryEvaluationRequest) -> QueryEvaluation:
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
        session_filter = SessionFilter.from_query_plan(plan, archive_root=self._archive_root)

        from polylogue.api.sync.bridge import run_coroutine_sync

        started_at_ms = int(time.time() * 1000)
        try:
            summaries = run_coroutine_sync(session_filter.list_summaries())
        except Exception:
            logger.warning("production-evaluator: evaluation failed for query:%s", query.query_hash, exc_info=True)
            raise
        duration_ms = int(time.time() * 1000) - started_at_ms

        excluded_prefixes = tuple(request.excluded_origin_prefixes)
        member_refs = tuple(
            f"session:{summary.id}"
            for summary in summaries
            if not any(str(summary.origin).startswith(prefix) for prefix in excluded_prefixes)
        )
        excluded_refs = set(request.excluded_scope_refs)
        if excluded_refs:
            member_refs = tuple(ref for ref in member_refs if ref not in excluded_refs)

        index_generation = _index_epoch(self._db_path)
        receipt = EvaluationReceipt(
            receipt_id=f"receipt-{uuid.uuid4().hex}",
            source_generation=_tier_generation(self._archive_root / "source.db", label="source"),
            user_generation=_tier_generation(self._archive_root / "user.db", label="user"),
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
        self._record_query_run(
            query_hash=query.query_hash,
            purpose=request.purpose,
            started_at_ms=started_at_ms,
            duration_ms=duration_ms,
            evaluation=evaluation,
        )
        return evaluation

    def resolve_cohort(self, operand: RefOperand) -> QueryEvaluation:
        raise NotImplementedError(
            f"cohort substrate is not implemented yet ({operand.reference.format()}); see polylogue-rxdo.6"
        )

    def _record_query_run(
        self,
        *,
        query_hash: str,
        purpose: str,
        started_at_ms: int,
        duration_ms: int,
        evaluation: QueryEvaluation,
    ) -> None:
        """Best-effort ops-tier telemetry. Never lets a recording failure fail a read."""
        ops_db = self._archive_root / "ops.db"
        if not ops_db.exists():
            return
        try:
            from polylogue.storage.sqlite.archive_tiers.ops_write import record_query_run

            with closing(sqlite3.connect(ops_db, timeout=5.0)) as conn:
                record_query_run(
                    conn,
                    run_id=f"qr_{uuid.uuid4().hex}",
                    query_hash=query_hash,
                    actor=None,
                    surface=self._surface,
                    verb=purpose,
                    request=None,
                    lowered_spec=None,
                    archive_epoch=evaluation.corpus_epoch,
                    started_at_ms=started_at_ms,
                    duration_ms=duration_ms,
                    status="ok",
                    degraded=None,
                    unit=evaluation.grain,
                    member_count=len(evaluation.member_refs),
                    exactness=evaluation.exactness,
                    result_fingerprint=None,
                    sample_refs=evaluation.member_refs[:20],
                )
        except Exception:
            logger.warning("production-evaluator: query_run recording failed", exc_info=True)


Surface = Literal["cli", "mcp", "daemon-web", "api", "daemon-internal"]


__all__ = [
    "ArchiveCanonicalPlanEvaluator",
    "LegacyQueryDefinitionNotExecutableError",
    "Surface",
    "UnsupportedEvaluationGrainError",
]
