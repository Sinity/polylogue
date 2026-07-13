"""Session-scoped standing-query convergence stage.

This module keeps the fourth convergence stage out of the historical
``convergence_stages`` hot file.  The canonical-plan evaluator remains an
injected planner contract: durable identity JSON is provenance, not source
syntax to reverse-compile.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from polylogue.archive.query.evaluator import CanonicalPlanEvaluator, QueryEvaluation, QueryEvaluationRequest
from polylogue.core.enums import AssertionKind, AssertionStatus
from polylogue.core.hashing import hash_payload
from polylogue.core.query_identity import query_ref, result_set_ref
from polylogue.daemon.convergence import ConvergenceStage, StageExecuteReturn
from polylogue.logging import get_logger
from polylogue.storage.sqlite.archive_tiers.user_write import (
    FindingAssertion,
    list_assertion_claims,
    upsert_findings_as_assertions,
)
from polylogue.storage.sqlite.connection_profile import open_daemon_connection
from polylogue.storage.sqlite.query_objects import (
    get_query,
    get_result_set,
    get_watched_query_baseline,
    list_watched_queries,
    membership_merkle_root,
    put_evaluation_receipt,
    put_result_set,
    put_watched_query_baseline,
)

logger = get_logger(__name__)


def make_standing_query_stage(
    db_path: Path,
    *,
    evaluator: CanonicalPlanEvaluator | None = None,
) -> ConvergenceStage:
    """Re-evaluate watched definitions after affected session convergence.

    The canonical-plan evaluator is deliberately injected. Identity JSON is
    not an executable plan, so the stage stays inert until the owning planner
    is supplied by broad runtime wiring. Its durable baseline, receipt, and
    candidate-finding semantics are nevertheless the production path.
    """

    def check(_path: Path) -> bool:
        return False

    def execute(_path: Path) -> StageExecuteReturn:
        return True

    def check_sessions(session_ids: Sequence[str]) -> set[str]:
        if evaluator is None or not session_ids:
            return set()
        user_db = _standing_user_db_path(db_path)
        if not user_db.exists():
            return set()
        try:
            with sqlite3.connect(f"file:{user_db}?mode=ro", uri=True, timeout=5.0) as conn:
                if list_watched_queries(conn) or _has_promoted_expected_findings(conn):
                    return set(session_ids)
                return set()
        except Exception:
            logger.warning("standing-queries: watch lookup failed", exc_info=True)
            return set(session_ids)

    def execute_sessions(session_ids: Sequence[str]) -> StageExecuteReturn:
        if evaluator is None:
            return True
        user_db = _standing_user_db_path(db_path)
        if not user_db.exists():
            return True
        now_ms = int(time.time() * 1000)
        ids = tuple(dict.fromkeys(str(session_id) for session_id in session_ids if session_id))
        try:
            conn = open_daemon_connection(user_db, timeout=30.0)
            try:
                for query in list_watched_queries(conn):
                    query_reference = query_ref(query.query_hash).format()
                    evaluation = evaluator.evaluate(
                        QueryEvaluationRequest(
                            query=query,
                            purpose="standing-watch",
                            changed_session_ids=ids,
                            excluded_scope_refs=(query_reference,),
                            excluded_origin_prefixes=("notice.",),
                        )
                    )
                    if evaluation.cache_only:
                        # An index-only relation after reset is not evidence of
                        # membership drift. Compare durable user-tier baselines only.
                        continue
                    _materialize_watch_evaluation(conn, query.query_hash, evaluation, now_ms=now_ms)
                _materialize_promoted_finding_drifts(conn, evaluator, now_ms=now_ms)
                conn.commit()
            finally:
                conn.close()
            return True
        except Exception:
            logger.warning("standing-queries: evaluation deferred", exc_info=True)
            return False

    return ConvergenceStage(
        name="standing-queries",
        description="Re-evaluate watched query definitions and emit candidate deltas",
        check=check,
        execute=execute,
        check_sessions=check_sessions,
        execute_sessions=execute_sessions,
        cpu_bound=False,
        false_means_pending=True,
    )


def _standing_user_db_path(db_path: Path) -> Path:
    return db_path.with_name("user.db")


def _watch_result_set_id(query_hash: str, member_refs: tuple[str, ...]) -> str:
    return f"watch-{hash_payload((query_hash, membership_merkle_root(member_refs)))}"


def _materialize_watch_evaluation(
    conn: sqlite3.Connection,
    query_hash: str,
    evaluation: QueryEvaluation,
    *,
    now_ms: int,
) -> None:
    baseline = get_watched_query_baseline(conn, query_hash)
    root = membership_merkle_root(evaluation.member_refs)
    if baseline is not None and baseline.membership_merkle_root == root:
        put_evaluation_receipt(
            conn,
            query_hash=query_hash,
            receipt=evaluation.receipt,
            result_set_id=baseline.result_set_id,
            created_at_ms=now_ms,
        )
        put_watched_query_baseline(
            conn,
            query_hash=query_hash,
            result_set_id=baseline.result_set_id,
            updated_at_ms=now_ms,
        )
        return
    result_set_id = _watch_result_set_id(query_hash, evaluation.member_refs)
    current = get_result_set(conn, result_set_id)
    if current is None:
        current = put_result_set(
            conn,
            result_set_id=result_set_id,
            query_hash=query_hash,
            grain=evaluation.grain,
            corpus_epoch=evaluation.corpus_epoch,
            member_refs=evaluation.member_refs,
            exactness=evaluation.exactness,
            persistence_class="watch",
            created_at_ms=now_ms,
        )
    put_evaluation_receipt(
        conn,
        query_hash=query_hash,
        receipt=evaluation.receipt,
        result_set_id=current.result_set_id,
        created_at_ms=now_ms,
    )
    put_watched_query_baseline(
        conn,
        query_hash=query_hash,
        result_set_id=current.result_set_id,
        updated_at_ms=now_ms,
    )
    if baseline is None:
        return
    query_reference = query_ref(query_hash).format()
    current_reference = result_set_ref(current.result_set_id).format()
    baseline_reference = result_set_ref(baseline.result_set_id).format()
    upsert_findings_as_assertions(
        conn,
        [
            FindingAssertion(
                claim_key="standing-query-membership-delta",
                target_ref=query_reference,
                body_text="Watched query membership changed after archive convergence.",
                finding_kind="query-delta",
                statistic={"op": "count", "value": current.member_count, "unit": "members"},
                n=current.member_count,
                query_ref=query_reference,
                result_set_ref=current_reference,
                baseline_ref=baseline_reference,
                current_ref=current_reference,
                detector_ref="agent:standing-query-detector.v1",
                scope_ref=query_reference,
            )
        ],
        now_ms=now_ms,
    )


def _materialize_promoted_finding_drifts(
    conn: sqlite3.Connection,
    evaluator: CanonicalPlanEvaluator,
    *,
    now_ms: int,
) -> None:
    """Emit a new candidate when an accepted expected-count finding diverges."""
    for finding in list_assertion_claims(
        conn,
        kinds=(AssertionKind.FINDING,),
        statuses=(AssertionStatus.ACCEPTED,),
    ):
        value = finding.value if isinstance(finding.value, dict) else {}
        expected = value.get("expected")
        query_reference = value.get("query_ref")
        if not isinstance(expected, dict) or not isinstance(query_reference, str):
            continue
        query_hash = query_reference.removeprefix("query:")
        query = get_query(conn, query_hash)
        if query is None:
            continue
        evaluation = evaluator.evaluate(
            QueryEvaluationRequest(
                query=query,
                purpose="finding-drift",
                excluded_scope_refs=(f"assertion:{finding.assertion_id}",),
                excluded_origin_prefixes=("notice.",),
            )
        )
        if evaluation.cache_only or _matches_expected_count(expected, len(evaluation.member_refs)):
            continue
        current_id = f"finding-{hash_payload((query_hash, membership_merkle_root(evaluation.member_refs)))}"
        current = get_result_set(conn, current_id)
        if current is None:
            current = put_result_set(
                conn,
                result_set_id=current_id,
                query_hash=query_hash,
                grain=evaluation.grain,
                corpus_epoch=evaluation.corpus_epoch,
                member_refs=evaluation.member_refs,
                exactness=evaluation.exactness,
                persistence_class="finding",
                created_at_ms=now_ms,
            )
        put_evaluation_receipt(
            conn,
            query_hash=query_hash,
            receipt=evaluation.receipt,
            result_set_id=current.result_set_id,
            created_at_ms=now_ms,
        )
        current_reference = result_set_ref(current.result_set_id).format()
        upsert_findings_as_assertions(
            conn,
            [
                FindingAssertion(
                    claim_key="promoted-finding-expected-count-drift",
                    target_ref=f"assertion:{finding.assertion_id}",
                    body_text="Promoted finding no longer matches its expected member count.",
                    finding_kind="query-drift",
                    statistic={"op": "count", "value": current.member_count, "unit": "members"},
                    n=current.member_count,
                    query_ref=query_reference,
                    result_set_ref=current_reference,
                    current_ref=current_reference,
                    expected=expected,
                    detector_ref="agent:standing-query-detector.v1",
                    scope_ref=f"assertion:{finding.assertion_id}",
                )
            ],
            now_ms=now_ms,
        )


def _has_promoted_expected_findings(conn: sqlite3.Connection) -> bool:
    """Return whether convergence must run expected findings without a watch."""
    for finding in list_assertion_claims(
        conn,
        kinds=(AssertionKind.FINDING,),
        statuses=(AssertionStatus.ACCEPTED,),
    ):
        value = finding.value if isinstance(finding.value, dict) else {}
        if isinstance(value.get("expected"), dict):
            return True
    return False


def _matches_expected_count(expected: Mapping[str, object], actual: int) -> bool:
    if expected.get("measure") != "member_count":
        return True
    value = expected.get("value")
    if isinstance(value, bool) or not isinstance(value, int):
        return True
    op = expected.get("op")
    if not isinstance(op, str):
        return True
    comparisons = {
        "=": actual == value,
        "!=": actual != value,
        ">": actual > value,
        ">=": actual >= value,
        "<": actual < value,
        "<=": actual <= value,
    }
    return comparisons.get(op, True)
