from __future__ import annotations

import sqlite3

import pytest

from polylogue.archive.query.evaluator import (
    DurableRefResolver,
    QueryEvaluation,
    QueryEvaluationRequest,
    RetainedRelationUnavailableError,
)
from polylogue.archive.query.expression import RefOperand, resolve_ref_operand
from polylogue.core.refs import ObjectRef
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.query_objects import (
    EvaluationReceipt,
    QueryObject,
    put_query,
    put_result_set,
    put_retained_query_run,
)


class _Evaluator:
    def __init__(self) -> None:
        self.requests: list[QueryEvaluationRequest] = []

    def evaluate(self, request: QueryEvaluationRequest) -> QueryEvaluation:
        self.requests.append(request)
        return QueryEvaluation(
            grain="session",
            member_refs=("session:re-evaluated",),
            corpus_epoch="index:g1",
            exactness="exact",
            receipt=EvaluationReceipt("receipt", "source:g1", "user:g1", "index:g1", "build:test"),
        )

    def resolve_cohort(self, operand: RefOperand) -> QueryEvaluation:
        assert operand.reference.format() == "cohort:team"
        return QueryEvaluation(
            grain="session",
            member_refs=("session:cohort",),
            corpus_epoch="index:g1",
            exactness="exact",
            receipt=EvaluationReceipt("cohort-receipt", "source:g1", "user:g1", "index:g1", "build:test"),
        )


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.USER)
    return conn


def test_resolver_uses_planner_evaluation_and_retained_relations() -> None:
    conn = _conn()
    parent = put_query(
        conn,
        {"field": "title", "value": "parent"},
        grain="session",
        lane="dialogue",
        rank_policy="mixed",
        created_at_ms=1,
    )
    child = put_query(
        conn,
        {"field": "title", "value": "child"},
        grain="session",
        lane="dialogue",
        rank_policy="mixed",
        created_at_ms=1,
    )
    result = put_result_set(
        conn,
        result_set_id="retained",
        query_hash=child.query_hash,
        grain="session",
        corpus_epoch="index:g1",
        member_refs=("session:stored",),
        exactness="exact",
        persistence_class="pinned",
        created_at_ms=2,
    )
    put_retained_query_run(
        conn, run_id="qr_saved", query_hash=child.query_hash, result_set_id=result.result_set_id, retained_at_ms=3
    )
    evaluator = _Evaluator()
    resolver = DurableRefResolver(conn, evaluator, owner_query_hash=parent.query_hash, created_at_ms=4)

    dynamic = resolve_ref_operand(RefOperand(ObjectRef(kind="query", object_id=child.query_hash)), resolver)
    retained = resolve_ref_operand(RefOperand(ObjectRef(kind="query-run", object_id="qr_saved")), resolver)
    cohort = resolve_ref_operand(RefOperand(ObjectRef(kind="cohort", object_id="team")), resolver)

    assert dynamic.member_refs == ("session:re-evaluated",)
    assert evaluator.requests[0].query == child
    assert retained.member_refs == ("session:stored",)
    assert cohort.member_refs == ("session:cohort",)
    assert conn.execute("SELECT edge_kind FROM query_edges").fetchone()[0] == "operand-of"


def test_query_run_without_retained_relation_fails_closed() -> None:
    with pytest.raises(RetainedRelationUnavailableError, match="no retained relation"):
        resolve_ref_operand(
            RefOperand(ObjectRef(kind="query-run", object_id="qr_missing")),
            DurableRefResolver(_conn(), _Evaluator()),
        )


def test_retained_query_run_rejects_a_result_set_for_another_query() -> None:
    conn = _conn()
    first = put_query(
        conn,
        {"field": "title", "value": "first"},
        grain="session",
        lane="dialogue",
        rank_policy="mixed",
        created_at_ms=1,
    )
    second = put_query(
        conn,
        {"field": "title", "value": "second"},
        grain="session",
        lane="dialogue",
        rank_policy="mixed",
        created_at_ms=1,
    )
    result = put_result_set(
        conn,
        result_set_id="second-result",
        query_hash=second.query_hash,
        grain="session",
        corpus_epoch="index:g1",
        member_refs=("session:second",),
        exactness="exact",
        persistence_class="pinned",
        created_at_ms=2,
    )

    with pytest.raises(ValueError, match="same query"):
        put_retained_query_run(
            conn,
            run_id="qr_mismatch",
            query_hash=first.query_hash,
            result_set_id=result.result_set_id,
            retained_at_ms=3,
        )


def test_retained_query_run_id_cannot_rebind_to_a_different_execution() -> None:
    conn = _conn()
    first = put_query(
        conn,
        {"field": "title", "value": "first"},
        grain="session",
        lane="dialogue",
        rank_policy="mixed",
        created_at_ms=1,
    )
    second = put_query(
        conn,
        {"field": "title", "value": "second"},
        grain="session",
        lane="dialogue",
        rank_policy="mixed",
        created_at_ms=1,
    )
    first_result = put_result_set(
        conn,
        result_set_id="first-retained",
        query_hash=first.query_hash,
        grain="session",
        corpus_epoch="index:g1",
        member_refs=("session:first",),
        exactness="exact",
        persistence_class="pinned",
        created_at_ms=2,
    )
    second_result = put_result_set(
        conn,
        result_set_id="second-retained",
        query_hash=second.query_hash,
        grain="session",
        corpus_epoch="index:g1",
        member_refs=("session:second",),
        exactness="exact",
        persistence_class="pinned",
        created_at_ms=2,
    )
    put_retained_query_run(
        conn,
        run_id="qr_immutable",
        query_hash=first.query_hash,
        result_set_id=first_result.result_set_id,
        retained_at_ms=3,
    )

    with pytest.raises(ValueError, match="conflicts"):
        put_retained_query_run(
            conn,
            run_id="qr_immutable",
            query_hash=second.query_hash,
            result_set_id=second_result.result_set_id,
            retained_at_ms=4,
        )


def test_evaluation_request_rejects_an_unsupported_definition_protocol() -> None:
    unsupported = QueryObject(
        query_hash="a" * 64,
        canonical_plan={},
        grain="session",
        lane="dialogue",
        rank_policy="mixed",
        definition_protocol_version="polylogue.query-definition.v999",
    )

    with pytest.raises(ValueError, match="unsupported"):
        QueryEvaluationRequest(query=unsupported, purpose="reference")


def test_retained_sampled_result_set_fails_closed_as_a_set_operand() -> None:
    conn = _conn()
    query = put_query(
        conn,
        {"field": "title", "value": "sampled"},
        grain="session",
        lane="dialogue",
        rank_policy="mixed",
        created_at_ms=1,
    )
    result = put_result_set(
        conn,
        result_set_id="sampled-retained",
        query_hash=query.query_hash,
        grain="session",
        corpus_epoch="index:g1",
        member_refs=("session:sample",),
        exactness="sampled",
        persistence_class="pinned",
        created_at_ms=2,
    )

    with pytest.raises(RetainedRelationUnavailableError, match="not an exact set operand"):
        resolve_ref_operand(
            RefOperand(ObjectRef(kind="result-set", object_id=result.result_set_id)),
            DurableRefResolver(conn, _Evaluator()),
        )
