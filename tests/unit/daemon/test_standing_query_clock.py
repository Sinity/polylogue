"""A watch on a relative time bound fires on the clock, not only on ingest.

polylogue-rxdo.5 AC1. ``make_standing_query_stage``'s whole-archive
``check``/``execute`` pair returned ``False``/``True`` unconditionally, so the
only trigger was ``check_sessions`` -- an accepted-input change. A definition
like ``date >= 7d`` changes membership when nothing is ingested at all, so it
was enabled and silently never re-evaluated.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.query.evaluator import QueryEvaluation, QueryEvaluationRequest
from polylogue.archive.query.expression import RefOperand
from polylogue.archive.query.watch_definition import compile_watch_definition
from polylogue.daemon.convergence_standing_queries import (
    CLOCK_BOUNDARY_MS,
    clock_boundary_start_ms,
    make_standing_query_stage,
    query_is_clock_relative,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.query_objects import (
    EvaluationReceipt,
    QueryObject,
    put_query,
    put_query_name,
    watched_query_baseline_updated_at_ms,
)
from tests.infra.frozen_clock import DEFAULT_FROZEN_EPOCH, FrozenClock

_RELATIVE = "sessions where origin:codex-session AND date >= 7d"
_ABSOLUTE = "sessions where origin:codex-session AND date >= 2026-01-01"


def _query(expression: str) -> QueryObject:
    return QueryObject(
        query_hash="probe",
        canonical_plan={"ast": compile_watch_definition(expression)},
        grain="session",
        lane="dialogue",
        rank_policy="mixed",
        definition_protocol_version="1",
    )


class _Evaluator:
    """The stage stays inert without an evaluator; this is the injected one."""

    def __init__(self) -> None:
        self.evaluated: list[str] = []

    def evaluate(self, request: QueryEvaluationRequest) -> QueryEvaluation:
        self.evaluated.append(str(request.query.query_hash))
        return QueryEvaluation(
            grain="session",
            member_refs=("session:codex-session:one",),
            corpus_epoch="index:v1:0",
            exactness="exact",
            receipt=EvaluationReceipt(
                receipt_id=f"receipt-{len(self.evaluated)}",
                source_generation="source:v1",
                user_generation="user:v1",
                index_generation="index:v1:0",
                runtime_build_ref="polylogue:test",
            ),
        )

    def session_origin_scope(self, query: QueryObject) -> frozenset[str] | None:
        del query
        return None

    def resolve_cohort(self, operand: RefOperand) -> QueryEvaluation:
        raise NotImplementedError(f"cohort substrate is not exercised here: {operand!r}")


def _seed(tmp_path: Path, expression: str, *, watch: bool = True) -> tuple[Path, str]:
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    user_db = archive_root / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    with sqlite3.connect(user_db) as conn:
        query = put_query(
            conn,
            compile_watch_definition(expression),
            grain="session",
            lane="dialogue",
            rank_policy="mixed",
            created_at_ms=1,
        )
        put_query_name(conn, name="clock-watch", query_hash=query.query_hash, watch=watch, updated_at_ms=2)
        conn.commit()
    return archive_root / "index.db", query.query_hash


def test_clock_boundary_start_is_a_declared_period() -> None:
    assert clock_boundary_start_ms(CLOCK_BOUNDARY_MS * 3 + 17) == CLOCK_BOUNDARY_MS * 3
    assert clock_boundary_start_ms(0) == 0
    with pytest.raises(ValueError):
        clock_boundary_start_ms(1, boundary_ms=0)


def test_relative_bound_is_clock_relative() -> None:
    """Measured against two explicit bases, never pattern-matched.

    ANTI-VACUITY: make ``query_is_clock_relative`` return ``True``
    unconditionally and the absolute case goes red; return ``False``
    unconditionally and the relative case does. One direction alone would admit
    a blanket answer.
    """
    assert query_is_clock_relative(_query(_RELATIVE)) is True
    assert query_is_clock_relative(_query(_ABSOLUTE)) is False
    assert query_is_clock_relative(_query("sessions where origin:codex-session")) is False


def test_clock_boundary_triggers_re_evaluation(tmp_path: Path, frozen_clock: FrozenClock) -> None:
    """A passed boundary makes the whole-archive check due, and once only.

    ANTI-VACUITY: restore ``check`` to ``return False`` (its state before
    polylogue-rxdo.5) and the first assertion goes red -- the watch is enabled,
    its bound has moved a whole day, and nothing schedules it.
    """
    index_db, query_hash = _seed(tmp_path, _RELATIVE)
    evaluator = _Evaluator()
    stage = make_standing_query_stage(index_db, evaluator=evaluator)

    assert stage.check(index_db) is True
    assert stage.execute(index_db) is True
    assert evaluator.evaluated == [query_hash]

    # One boundary, one evaluation: a second pass inside the same boundary is
    # not due, so a watch cannot be re-run on every convergence tick.
    assert stage.check(index_db) is False
    assert stage.execute(index_db) is True
    assert evaluator.evaluated == [query_hash]

    with sqlite3.connect(tmp_path / "archive" / "user.db") as conn:
        updated_at_ms = watched_query_baseline_updated_at_ms(conn, query_hash)
    assert updated_at_ms is not None
    assert updated_at_ms >= clock_boundary_start_ms(int(DEFAULT_FROZEN_EPOCH * 1000))

    # The next boundary makes it due again.
    frozen_clock.advance(CLOCK_BOUNDARY_MS / 1000)
    assert stage.check(index_db) is True


def test_absolute_watch_is_never_clock_due(tmp_path: Path, frozen_clock: FrozenClock) -> None:
    """A definition whose bound does not move is not periodic work.

    This is the "do not periodically evaluate every possible analysis" half of
    the acceptance criterion. Without it the trigger would degrade into a
    scanner that re-runs every watch on every boundary forever.
    """
    index_db, _query_hash = _seed(tmp_path, _ABSOLUTE)
    evaluator = _Evaluator()
    stage = make_standing_query_stage(index_db, evaluator=evaluator)

    assert stage.check(index_db) is False
    frozen_clock.advance(CLOCK_BOUNDARY_MS / 1000 * 4)
    assert stage.check(index_db) is False
    assert stage.execute(index_db) is True
    assert evaluator.evaluated == []


def test_disabled_watch_incurs_no_clock_evaluation(tmp_path: Path) -> None:
    """AC1's second half: only watches the user actually enabled run."""
    index_db, _query_hash = _seed(tmp_path, _RELATIVE, watch=False)
    evaluator = _Evaluator()
    stage = make_standing_query_stage(index_db, evaluator=evaluator)

    assert stage.check(index_db) is False
    assert stage.execute(index_db) is True
    assert evaluator.evaluated == []


def test_clock_trigger_stays_inert_without_an_evaluator(tmp_path: Path) -> None:
    """No injected planner means no evaluation, exactly as before."""
    index_db, _query_hash = _seed(tmp_path, _RELATIVE)
    stage = make_standing_query_stage(index_db)

    assert stage.check(index_db) is False
    assert stage.execute(index_db) is True
