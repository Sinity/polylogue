from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.query.evaluator import QueryEvaluation, QueryEvaluationRequest
from polylogue.core.enums import AssertionKind, AssertionStatus
from polylogue.daemon.convergence import DaemonConverger
from polylogue.daemon.convergence_stages import make_standing_query_stage
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import (
    FindingAssertion,
    mark_assertion_status,
    upsert_findings_as_assertions,
)
from polylogue.storage.sqlite.query_objects import EvaluationReceipt, put_query, put_query_name


class _Evaluator:
    def __init__(self, *, members: tuple[str, ...], cache_only: bool = False) -> None:
        self.members = members
        self.cache_only = cache_only
        self.requests: list[QueryEvaluationRequest] = []

    def evaluate(self, request: QueryEvaluationRequest) -> QueryEvaluation:
        self.requests.append(request)
        return QueryEvaluation(
            grain="session",
            member_refs=self.members,
            corpus_epoch="index:g1",
            exactness="exact",
            cache_only=self.cache_only,
            receipt=EvaluationReceipt(
                receipt_id=f"receipt-{len(self.requests)}-{request.purpose}",
                source_generation="source:g1",
                user_generation="user:g1",
                index_generation="index:g1",
                runtime_build_ref="build:test",
            ),
        )

    def resolve_cohort(self, _operand: object) -> QueryEvaluation:
        raise AssertionError("standing queries do not resolve cohorts directly")


def _seed_watch(tmp_path: Path, *, watch: bool = True) -> tuple[Path, str]:
    index_db = tmp_path / "index.db"
    user_db = tmp_path / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    with sqlite3.connect(user_db) as conn:
        query = put_query(
            conn,
            {"field": "origin", "value": "codex-session"},
            grain="session",
            lane="dialogue",
            rank_policy="mixed",
            created_at_ms=1,
        )
        if watch:
            put_query_name(conn, name="codex", query_hash=query.query_hash, watch=True, updated_at_ms=2)
        conn.commit()
    return index_db, query.query_hash


def test_watched_session_delta_is_candidate_once_and_has_self_firewall(tmp_path: Path) -> None:
    index_db, query_hash = _seed_watch(tmp_path)
    evaluator = _Evaluator(members=("session:one",))
    stage = make_standing_query_stage(index_db, evaluator=evaluator)

    assert stage.check_sessions is not None and stage.execute_sessions is not None
    assert stage.check_sessions(("session:changed",)) == {"session:changed"}
    assert stage.execute_sessions(("session:changed",)) is True  # baseline only
    evaluator.members = ("session:one", "session:two")
    assert stage.execute_sessions(("session:changed",)) is True
    assert stage.execute_sessions(("session:changed",)) is True

    with sqlite3.connect(tmp_path / "user.db") as conn:
        rows = conn.execute("SELECT status, target_ref, value_json FROM assertions WHERE kind = 'finding'").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "candidate"
        assert rows[0][1] == f"query:{query_hash}"
        assert "query-delta" in str(rows[0][2])
        assert conn.execute("SELECT COUNT(*) FROM query_evaluation_receipts").fetchone()[0] == 3
    request = evaluator.requests[0]
    assert request.excluded_scope_refs == (f"query:{query_hash}",)
    assert request.excluded_origin_prefixes == ("notice.",)


def test_cache_only_evaluation_after_index_reset_never_compares_baseline(tmp_path: Path) -> None:
    index_db, _ = _seed_watch(tmp_path)
    evaluator = _Evaluator(members=("session:one",))
    stage = make_standing_query_stage(index_db, evaluator=evaluator)
    assert stage.execute_sessions is not None
    assert stage.execute_sessions(("session:changed",)) is True
    evaluator.members = ("session:one", "session:two")
    evaluator.cache_only = True
    assert stage.execute_sessions(("session:changed",)) is True

    with sqlite3.connect(tmp_path / "user.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM assertions WHERE kind = 'finding'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM result_sets WHERE persistence_class = 'watch'").fetchone()[0] == 1


def test_watched_aliases_materialize_one_query_once_per_convergence(tmp_path: Path) -> None:
    index_db, query_hash = _seed_watch(tmp_path)
    with sqlite3.connect(tmp_path / "user.db") as conn:
        put_query_name(conn, name="codex-alias", query_hash=query_hash, watch=True, updated_at_ms=3)
        conn.commit()

    evaluator = _Evaluator(members=("session:one",))
    stage = make_standing_query_stage(index_db, evaluator=evaluator)
    assert stage.execute_sessions is not None
    assert stage.execute_sessions(("session:changed",)) is True

    assert len(evaluator.requests) == 1
    with sqlite3.connect(tmp_path / "user.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM result_sets WHERE persistence_class = 'watch'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM assertions WHERE kind = 'finding'").fetchone()[0] == 0


def test_watch_baseline_tracks_return_to_a_prior_membership(tmp_path: Path) -> None:
    index_db, _ = _seed_watch(tmp_path)
    evaluator = _Evaluator(members=("session:a",))
    stage = make_standing_query_stage(index_db, evaluator=evaluator)
    assert stage.execute_sessions is not None
    assert stage.execute_sessions(("session:changed",)) is True  # baseline A
    evaluator.members = ("session:b",)
    assert stage.execute_sessions(("session:changed",)) is True  # A -> B
    evaluator.members = ("session:a",)
    assert stage.execute_sessions(("session:changed",)) is True  # B -> A

    with sqlite3.connect(tmp_path / "user.db") as conn:
        rows = conn.execute(
            "SELECT value_json FROM assertions WHERE kind = 'finding' ORDER BY created_at_ms, assertion_id"
        ).fetchall()
        assert len(rows) == 2
        assert "baseline_ref" in str(rows[1][0])
        assert "result-set:watch-" in str(rows[1][0])


def test_promoted_expected_count_divergence_targets_original_finding_without_watch(tmp_path: Path) -> None:
    index_db, query_hash = _seed_watch(tmp_path, watch=False)
    with sqlite3.connect(tmp_path / "user.db") as conn:
        original = upsert_findings_as_assertions(
            conn,
            [
                FindingAssertion(
                    claim_key="expected-count",
                    target_ref=f"query:{query_hash}",
                    body_text="Expected one member.",
                    finding_kind="measure",
                    statistic={"op": "count", "value": 1, "unit": "members"},
                    n=1,
                    query_ref=f"query:{query_hash}",
                    result_set_ref="result-set:original",
                    detector_ref="agent:test-detector",
                    expected={"measure": "member_count", "op": "=", "value": 1},
                )
            ],
            now_ms=1,
        )[0]
        mark_assertion_status(conn, original.assertion_id, AssertionStatus.ACCEPTED, now_ms=2)
        conn.commit()
    stage = make_standing_query_stage(index_db, evaluator=_Evaluator(members=("session:one", "session:two")))
    converger = DaemonConverger(stages=(stage,), max_workers=1)
    states, _ = converger.converge_sessions(("session:changed",))
    assert states["session:changed"].stages["standing-queries"].value == "done"

    with sqlite3.connect(tmp_path / "user.db") as conn:
        rows = conn.execute(
            "SELECT target_ref, value_json, status FROM assertions WHERE kind = ? ORDER BY created_at_ms, assertion_id",
            (AssertionKind.FINDING.value,),
        ).fetchall()
        assert len(rows) == 2
        assert rows[1][0] == f"assertion:{original.assertion_id}"
        assert '"finding_kind":"query-drift"' in str(rows[1][1])
        assert rows[1][2] == "candidate"
