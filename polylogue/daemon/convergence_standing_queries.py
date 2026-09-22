"""Session-scoped standing-query convergence stage.

This module keeps the fourth convergence stage out of the historical
``convergence_stages`` hot file.  The canonical-plan evaluator remains an
injected planner contract: durable identity JSON is provenance, not source
syntax to reverse-compile.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from polylogue.archive.query.evaluator import (
    CanonicalPlanEvaluator,
    QueryEvaluation,
    QueryEvaluationRequest,
    ScopedCanonicalPlanEvaluator,
    session_origin,
)
from polylogue.archive.query.metadata import DATE_QUERY_FIELD_REGISTRY
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
    watched_query_baseline_updated_at_ms,
)

#: The declared clock boundary a watched definition is re-evaluated across.
#: One UTC day: relative bounds in this grammar are written in hours, days,
#: weeks and months (``_parse_relative_date``), so a day is the coarsest period
#: that cannot skip a whole generation of them, and it bounds a clock-only
#: re-evaluation to once per watch per day rather than once per convergence
#: pass.
CLOCK_BOUNDARY_MS = 24 * 60 * 60 * 1000

#: Predicate fields whose declared bound is an instant. Taken from the grammar's
#: own date registry plus the two relative-date spec filters it does not carry
#: (``since``/``until``, declared in ``archive/query/metadata.py``) and the
#: terminal-unit ``time`` field. A relative bound on any other field is not
#: expressible in this grammar, so probing every string value in the AST would
#: add only false positives from search terms and titles.
_CLOCK_BOUND_FIELDS = frozenset(DATE_QUERY_FIELD_REGISTRY) | {"since", "until", "time"}


def clock_boundary_start_ms(now_ms: int, *, boundary_ms: int = CLOCK_BOUNDARY_MS) -> int:
    """Return the start of the clock boundary ``now_ms`` falls in."""
    if boundary_ms <= 0:
        raise ValueError("clock boundary must be positive")
    return now_ms - (now_ms % boundary_ms)


def _resolve_at(value: str, base: datetime) -> datetime | None:
    """Resolve one declared bound against an explicit clock, never the wall clock."""
    import dateparser  # deferred: the timezone tables cost ~0.26s to import

    resolved: datetime | None = dateparser.parse(
        value,
        settings={
            "PREFER_DATES_FROM": "past",
            "RETURN_AS_TIMEZONE_AWARE": True,
            "TIMEZONE": "UTC",
            "RELATIVE_BASE": base,
        },
    )
    if resolved is not None and resolved.tzinfo is None:
        resolved = resolved.replace(tzinfo=timezone.utc)
    return resolved


def _clock_bound_values(node: object) -> Iterator[str]:
    """Yield every declared instant bound in one canonical predicate AST."""
    if isinstance(node, Mapping):
        field = node.get("field")
        if isinstance(field, str) and field in _CLOCK_BOUND_FIELDS:
            values = node.get("values")
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                for value in values:
                    if isinstance(value, str):
                        yield value
        for child in node.values():
            yield from _clock_bound_values(child)
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        for child in node:
            yield from _clock_bound_values(child)


def query_is_clock_relative(query: QueryObject) -> bool:
    """Whether this definition's own membership moves when only the clock moves.

    Measured, not pattern-matched: each declared instant bound is resolved
    against two explicit bases one boundary apart, and the definition is
    clock-relative exactly when some bound lands on a different instant. An
    absolute literal (``2026-01-01``) resolves identically against both bases
    and is therefore never scheduled by the clock; ``7 days ago`` -- what
    ``_parse_relative_date`` stores for ``7d`` -- is.
    """
    ast = query.canonical_plan.get("ast")
    if not isinstance(ast, Mapping):
        return False
    base = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
    later = base + timedelta(milliseconds=CLOCK_BOUNDARY_MS)
    for value in _clock_bound_values(ast):
        first = _resolve_at(value, base)
        second = _resolve_at(value, later)
        if first is None or second is None:
            continue
        if first != second:
            return True
    return False


def _clock_due_watches(conn: sqlite3.Connection, *, now_ms: int) -> tuple[QueryObject, ...]:
    """Return the enabled watches whose declared clock boundary has passed.

    AC1's other half. A watch on ``time >= 7 days ago`` changes membership with
    no ingestion event at all, so the session-scoped trigger below can never
    fire for it; without this the watch is enabled and silently never runs.
    Only watches the user actually enabled are considered, and a watch whose
    baseline was already advanced inside the current boundary is not due -- one
    boundary, one evaluation.
    """
    boundary_start = clock_boundary_start_ms(now_ms)
    due: list[QueryObject] = []
    for query in list_watched_queries(conn):
        if not query_is_clock_relative(query):
            continue
        updated_at_ms = watched_query_baseline_updated_at_ms(conn, query.query_hash)
        if updated_at_ms is None or updated_at_ms < boundary_start:
            due.append(query)
    return tuple(due)


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
        """Whole-archive trigger: a declared clock boundary, not an ingest event."""
        if evaluator is None:
            return False
        user_db = _standing_user_db_path(db_path)
        if not user_db.exists():
            return False
        with closing(open_readonly_connection(user_db)) as conn:
            return bool(_clock_due_watches(conn, now_ms=int(time.time() * 1000)))

    def execute(_path: Path) -> StageExecuteReturn:
        """Re-evaluate exactly the watches whose clock boundary has passed."""
        if evaluator is None:
            return True
        user_db = _standing_user_db_path(db_path)
        if not user_db.exists():
            return True
        now_ms = int(time.time() * 1000)
        conn = open_daemon_connection(user_db, timeout=30.0)
        try:
            due = _clock_due_watches(conn, now_ms=now_ms)
            if not due:
                return True
            for query in due:
                evaluation = evaluator.evaluate(
                    QueryEvaluationRequest(
                        query=query,
                        purpose="standing-watch",
                        changed_session_ids=(),
                        excluded_scope_refs=(query_ref(query.query_hash).format(),),
                        excluded_origin_prefixes=("notice.",),
                    )
                )
                if evaluation.cache_only:
                    continue
                _materialize_watch_evaluation(conn, query.query_hash, evaluation, now_ms=now_ms)
            conn.commit()
        finally:
            conn.close()
        emit(
            "daemon.stage.clock_boundary",
            level=INFO,
            outcome="ok",
            stage="standing-queries",
            reason="clock_boundary_passed",
            watches=len(due),
        )
        return True

    def check_many(paths: Sequence[Path]) -> set[Path]:
        # One clock read for the whole batch: the boundary is a property of the
        # archive, not of any one source file.
        if not paths:
            return set()
        return set(paths) if check(paths[0]) else set()

    def execute_many(paths: Sequence[Path]) -> StageExecuteReturn:
        if not paths:
            return True
        return execute(paths[0])

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
        check_many=check_many,
        execute_many=execute_many,
        check_sessions=check_sessions,
        execute_sessions=execute_sessions,
        false_means_pending=True,
        # The clock-boundary trigger is a function of the archive's watches and
        # the wall clock, not of the batch's subjects (polylogue-rxdo.5 AC1).
        whole_archive=True,
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
    """Advance one watch's baseline and report a measured membership delta.

    A stored baseline whose own ``exactness`` is not ``exact`` is *unmeasured*,
    not a prior observation. ``user.db`` files written before the guard in
    :func:`_materialize_unmeasured_watch_evaluation` persisted every evaluation
    as the baseline, so such a row can already exist; its merkle root is over a
    capped/sampled/estimated view and therefore answers a different question
    than an exact root. Comparing the two emits an unconditional "membership
    changed" claim -- the same false positive the guard prevents for new
    baselines (polylogue-uwm6y). The first exact evaluation therefore replaces
    a non-exact baseline silently and claims no delta against it.
    """
    if evaluation.exactness != "exact":
        _materialize_unmeasured_watch_evaluation(conn, query_hash, evaluation, now_ms=now_ms)
        return
    baseline = get_watched_query_baseline(conn, query_hash)
    comparable_baseline = baseline if baseline is not None and baseline.exactness == "exact" else None
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
    if comparable_baseline is None or comparable_baseline.membership_merkle_root == root:
        return
    query_reference = query_ref(query_hash).format()
    current_reference = result_set_ref(current.result_set_id).format()
    baseline_reference = result_set_ref(comparable_baseline.result_set_id).format()
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

    A stored baseline whose own ``exactness`` is not ``exact`` -- only
    reachable from a ``user.db`` written before this guard existed -- is the
    same "no question yet" case: there was never a measured membership to
    drift from, so it gets the receipt and nothing else.
    """
    put_evaluation_receipt(
        conn,
        query_hash=query_hash,
        receipt=_degraded_receipt(evaluation, purpose="standing-watch"),
        result_set_id=None,
        created_at_ms=now_ms,
    )
    baseline = get_watched_query_baseline(conn, query_hash)
    if baseline is None or baseline.exactness != "exact":
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
