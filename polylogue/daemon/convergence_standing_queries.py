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
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from polylogue.archive.query.evaluator import (
    CanonicalPlanEvaluator,
    QueryEvaluation,
    QueryEvaluationRequest,
    ScopedCanonicalPlanEvaluator,
    session_origin,
)
from polylogue.core.enums import AssertionKind, AssertionStatus
from polylogue.core.hashing import hash_payload
from polylogue.core.query_identity import query_ref, result_set_ref
from polylogue.core.sqlite_locking import is_transient_sqlite_lock
from polylogue.daemon.convergence import ConvergenceStage, StageExecuteReturn
from polylogue.logging import INFO, WARNING, emit
from polylogue.storage.sqlite.archive_tiers.user_write import (
    FindingAssertion,
    list_assertion_claims,
    upsert_findings_as_assertions,
)
from polylogue.storage.sqlite.connection_profile import open_daemon_connection, open_readonly_connection
from polylogue.storage.sqlite.query_objects import (
    EvaluationReceipt,
    QueryObject,
    get_query,
    get_result_set,
    get_watched_query_baseline,
    list_watched_queries,
    membership_merkle_root,
    put_evaluation_receipt,
    put_result_set,
    put_watched_query_baseline,
)


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
            with closing(open_readonly_connection(user_db)) as conn:
                watched = list_watched_queries(conn)
                if not watched and not _has_promoted_expected_findings(conn):
                    return set()
                scope = _narrowed_origin_scope(conn, evaluator, watched)
                if scope is None:
                    return set(session_ids)
                return {
                    session_id
                    for session_id in session_ids
                    if (origin := session_origin(str(session_id))) is None or origin in scope
                }
        except Exception as exc:
            emit(
                "daemon.stage.check_failed",
                level=WARNING,
                outcome="degraded",
                stage="standing-queries",
                reason="watch_lookup_failed_assuming_work",
                path=user_db,
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
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
        except sqlite3.OperationalError as exc:
            transient = is_transient_sqlite_lock(exc)
            emit(
                "daemon.stage.execute_failed",
                level=INFO if transient else WARNING,
                outcome="skipped" if transient else "error",
                stage="standing-queries",
                reason="archive_busy" if transient else "evaluation_failed",
                sessions=len(ids),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            if transient:
                return False
            raise
        except Exception as exc:
            emit(
                "daemon.stage.execute_failed",
                level=WARNING,
                outcome="error",
                stage="standing-queries",
                reason="evaluation_deferred",
                sessions=len(ids),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            raise

    return ConvergenceStage(
        name="standing-queries",
        description="Re-evaluate watched query definitions and emit candidate deltas",
        check=check,
        execute=execute,
        check_sessions=check_sessions,
        execute_sessions=execute_sessions,
        false_means_pending=True,
    )


def _standing_user_db_path(db_path: Path) -> Path:
    return db_path.with_name("user.db")


def _narrowed_origin_scope(
    conn: sqlite3.Connection,
    evaluator: CanonicalPlanEvaluator,
    watched: Sequence[QueryObject],
) -> frozenset[str] | None:
    """Origins that can still affect a watch, or ``None`` for the global baseline.

    Narrowing is an in-memory, per-tick decision derived from the planner and
    the definitions themselves; nothing is persisted, so no stored fingerprint
    can go stale and suppress a real firing. It is only ever the union of
    *proved* per-definition bounds, and it is abandoned entirely -- keeping the
    global corpus_epoch baseline -- whenever any of the following holds.

    * The planner publishes no bounds, or cannot bound one watched definition.
      An unbounded predicate can match any origin.
    * A watched definition has no durable baseline yet. The first evaluation
      establishes a baseline silently, so skipping the tick that would have
      established it would move that silent first observation later and hide
      the delta a subsequent tick should have reported.
    * An accepted expected-count finding exists. Those drift against a stored
      expectation rather than against the previous membership, so a definition
      that is already drifting must be allowed to report it on the next tick
      whatever changed.
    """
    if not watched or not isinstance(evaluator, ScopedCanonicalPlanEvaluator):
        return None
    if _has_promoted_expected_findings(conn):
        return None
    union: set[str] = set()
    for query in watched:
        bound = evaluator.session_origin_scope(query)
        if bound is None:
            return None
        if get_watched_query_baseline(conn, query.query_hash) is None:
            return None
        union |= set(bound)
    return frozenset(union)


def _watch_result_set_id(query_hash: str, evaluation: QueryEvaluation) -> str:
    """Name an immutable watch snapshot by every persisted manifest field."""
    snapshot = (
        query_hash,
        evaluation.grain,
        evaluation.corpus_epoch,
        membership_merkle_root(evaluation.member_refs),
        hash_payload(list(evaluation.member_refs)),
        evaluation.exactness,
        "watch",
    )
    return f"watch-{hash_payload(snapshot)}"


#: The detector that owns every candidate this stage writes.
_DETECTOR_REF = "agent:standing-query-detector.v1"


def _degraded_receipt(evaluation: QueryEvaluation, *, purpose: str) -> EvaluationReceipt:
    """Stamp why an evaluation could not answer its question."""
    return replace(
        evaluation.receipt,
        degradation={
            **(evaluation.receipt.degradation or {}),
            "reason": "non-exact-evaluation",
            "exactness": evaluation.exactness,
            "purpose": purpose,
        },
    )


def _materialize_watch_evaluation(
    conn: sqlite3.Connection,
    query_hash: str,
    evaluation: QueryEvaluation,
    *,
    now_ms: int,
) -> None:
    if evaluation.exactness != "exact":
        _materialize_unmeasured_watch_evaluation(conn, query_hash, evaluation, now_ms=now_ms)
        return
    baseline = get_watched_query_baseline(conn, query_hash)
    root = membership_merkle_root(evaluation.member_refs)
    rank_hash = hash_payload(list(evaluation.member_refs))
    same_snapshot = baseline is not None and (
        baseline.grain == evaluation.grain
        and baseline.corpus_epoch == evaluation.corpus_epoch
        and baseline.membership_merkle_root == root
        and baseline.ordered_rank_hash == rank_hash
        and baseline.exactness == evaluation.exactness
        and baseline.persistence_class == "watch"
    )
    if same_snapshot:
        assert baseline is not None
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
    result_set_id = _watch_result_set_id(query_hash, evaluation)
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
    if baseline is None or baseline.membership_merkle_root == root:
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
                detector_ref=_DETECTOR_REF,
                scope_ref=query_reference,
            )
        ],
        now_ms=now_ms,
    )


def _materialize_unmeasured_watch_evaluation(
    conn: sqlite3.Connection,
    query_hash: str,
    evaluation: QueryEvaluation,
    *,
    now_ms: int,
) -> None:
    """Record a non-exact watch evaluation without claiming a membership delta.

    ``member_refs`` from a capped, sampled or estimated evaluation is not the
    watched relation, only a bounded view of it. Its merkle root therefore
    answers a different question than the baseline's, and a shifting cap window
    alone would emit an unconditional "membership changed" assertion
    (polylogue-uwm6y). So this path never advances the baseline, never writes a
    ``watch`` result set, and never emits ``query-delta``.

    Emitting *nothing* would be the mirror defect -- an unmeasured negative
    reported as a measured one -- so the degraded receipt is always written,
    and when a measured baseline existed (there was a real drift question this
    tick was supposed to answer) exactly one degraded candidate names the
    condition. With no baseline there is no question yet: the receipt is the
    whole record, and the next exact evaluation establishes the baseline it
    always would have.
    """
    put_evaluation_receipt(
        conn,
        query_hash=query_hash,
        receipt=_degraded_receipt(evaluation, purpose="standing-watch"),
        result_set_id=None,
        created_at_ms=now_ms,
    )
    baseline = get_watched_query_baseline(conn, query_hash)
    if baseline is None:
        return
    query_reference = query_ref(query_hash).format()
    baseline_reference = result_set_ref(baseline.result_set_id).format()
    upsert_findings_as_assertions(
        conn,
        [
            FindingAssertion(
                claim_key="standing-query-membership-unmeasured",
                target_ref=query_reference,
                body_text=(
                    "Watched query membership could not be compared after archive convergence: the "
                    f"evaluation was {evaluation.exactness}, not an exact enumeration. The stored "
                    "baseline is unchanged and no membership change is claimed."
                ),
                finding_kind="query-delta-unmeasured",
                statistic={
                    "op": "unmeasured",
                    "value": None,
                    "unit": "members",
                    "reason": f"evaluation exactness is {evaluation.exactness}",
                },
                n=0,
                query_ref=query_reference,
                result_set_ref=baseline_reference,
                baseline_ref=baseline_reference,
                detector_ref=_DETECTOR_REF,
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
        if evaluation.cache_only:
            continue
        if evaluation.exactness != "exact":
            _materialize_unmeasured_finding_drift(
                conn,
                assertion_id=finding.assertion_id,
                query_hash=query_hash,
                query_reference=query_reference,
                declared_result_set_ref=value.get("result_set_ref"),
                expected=expected,
                evaluation=evaluation,
                now_ms=now_ms,
            )
            continue
        if _matches_expected_count(expected, len(evaluation.member_refs)):
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
                    detector_ref=_DETECTOR_REF,
                    scope_ref=f"assertion:{finding.assertion_id}",
                )
            ],
            now_ms=now_ms,
        )


def _materialize_unmeasured_finding_drift(
    conn: sqlite3.Connection,
    *,
    assertion_id: str,
    query_hash: str,
    query_reference: str,
    declared_result_set_ref: object,
    expected: Mapping[str, object],
    evaluation: QueryEvaluation,
    now_ms: int,
) -> None:
    """Record that an expected-count check could not be performed.

    ``_matches_expected_count`` reads ``len(member_refs)`` as the relation's
    member count. On a capped, sampled or estimated evaluation that count is a
    property of the evaluation's bound, not of the relation, so comparing it
    against a stored expectation would promote a truncation into a definitive
    ``query-drift`` claim about accepted evidence (polylogue-uwm6y).
    """
    put_evaluation_receipt(
        conn,
        query_hash=query_hash,
        receipt=_degraded_receipt(evaluation, purpose="finding-drift"),
        result_set_id=None,
        created_at_ms=now_ms,
    )
    if not isinstance(declared_result_set_ref, str) or not declared_result_set_ref.strip():
        # finding.v1 always declares one; without it there is no relation to
        # name as the subject of the degraded claim, and inventing a ref would
        # be the fabrication this path exists to prevent.
        return
    upsert_findings_as_assertions(
        conn,
        [
            FindingAssertion(
                claim_key="promoted-finding-expected-count-unmeasured",
                target_ref=f"assertion:{assertion_id}",
                body_text=(
                    "Promoted finding could not be re-checked against its expected member count: the "
                    f"evaluation was {evaluation.exactness}, not an exact enumeration. No drift is claimed."
                ),
                finding_kind="query-drift-unmeasured",
                statistic={
                    "op": "unmeasured",
                    "value": None,
                    "unit": "members",
                    "reason": f"evaluation exactness is {evaluation.exactness}",
                },
                n=0,
                query_ref=query_reference,
                result_set_ref=declared_result_set_ref.strip(),
                expected=dict(expected),  # type: ignore[arg-type]
                detector_ref=_DETECTOR_REF,
                scope_ref=f"assertion:{assertion_id}",
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
