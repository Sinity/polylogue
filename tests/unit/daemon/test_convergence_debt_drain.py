"""The generic convergence-debt drain is work-bound, not cadence-bound.

A cold build records deferred debt per admitted file for every stage it did
not run inline. One page of 100 rows per minute left a full archive's backlog
draining for days, and archive-wide stages re-ran once per page. These tests
drive the real ``_drain_convergence_debt_backlog`` over a real ops ledger with
stand-in stages, and count how often each stage actually runs.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest

from polylogue.core.compute import BoundedComputeAdapter
from polylogue.daemon import cli as daemon_cli
from polylogue.daemon.convergence import ConvergenceStage
from polylogue.sources.live.cursor import CursorStore
from tests.infra.archive_templates import bootstrap_archive_root


def _rows(root: Path) -> list[tuple[str, str]]:
    with sqlite3.connect(root / "ops.db") as conn:
        return [
            (str(stage), str(target))
            for stage, target in conn.execute("SELECT stage, target_id FROM convergence_debt ORDER BY stage, target_id")
        ]


def _seed(root: Path, stage: str, count: int) -> None:
    cursor = CursorStore(root / "index.db")
    for index in range(count):
        cursor.record_convergence_debt(
            stage=stage,
            subject_type="source_path",
            subject_id=str(root / "sources" / f"{stage}-{index:04d}.jsonl"),
            error="stage state: not_run",
            deferred=True,
        )
    with sqlite3.connect(root / "ops.db") as conn:
        conn.execute("UPDATE convergence_debt SET next_retry_at = '1970-01-01T00:00:00+00:00'")
        conn.commit()


class _Stage:
    def __init__(self, name: str, *, subject_independent: bool, converges: bool = True) -> None:
        self.name = name
        self.subject_independent = subject_independent
        self.converges = converges
        self.executions: list[tuple[Path, ...]] = []

    def build(self) -> ConvergenceStage:
        def check_many(paths: Sequence[Path]) -> set[Path]:
            return set(paths)

        def execute_many(paths: Sequence[Path]) -> bool:
            self.executions.append(tuple(paths))
            return self.converges

        return ConvergenceStage(
            name=self.name,
            description="stand-in",
            check=lambda _path: True,
            execute=lambda path: execute_many((path,)),
            check_many=check_many,
            execute_many=execute_many,
            whole_archive=self.subject_independent,
            subject_independent=self.subject_independent,
            false_means_pending=True,
            writer_admission="bridged",
        )


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    bootstrap_archive_root(tmp_path)
    return tmp_path


def _install(monkeypatch: pytest.MonkeyPatch, *stages: _Stage) -> None:
    built = tuple(stage.build() for stage in stages)
    monkeypatch.setattr(
        "polylogue.daemon.convergence_stages.make_default_convergence_stages", lambda _db, **_kwargs: built
    )


def test_subject_independent_stage_runs_once_for_its_whole_backlog(
    archive: Path,
    monkeypatch: pytest.MonkeyPatch,
    bounded_compute_adapter: BoundedComputeAdapter,
) -> None:
    """Anti-vacuity: without the subject-independent collapse the stage runs
    once per page (three times for 250 rows)."""
    stage = _Stage("archive_wide", subject_independent=True)
    _install(monkeypatch, stage)
    _seed(archive, "archive_wide", 250)

    daemon_cli._drain_convergence_debt_backlog(
        archive / "index.db", budget_s=60.0, compute_adapter=bounded_compute_adapter
    )

    assert len(stage.executions) == 1
    assert len(stage.executions[0]) == 1
    assert _rows(archive) == []


def test_unconverged_subject_independent_stage_keeps_its_rows(
    archive: Path, monkeypatch: pytest.MonkeyPatch, bounded_compute_adapter: BoundedComputeAdapter
) -> None:
    """A stage that is still pending settles nothing; its rows back off."""
    stage = _Stage("archive_wide", subject_independent=True, converges=False)
    _install(monkeypatch, stage)
    _seed(archive, "archive_wide", 30)

    daemon_cli._drain_convergence_debt_backlog(
        archive / "index.db", budget_s=60.0, compute_adapter=bounded_compute_adapter
    )

    assert len(stage.executions) == 1
    assert len(_rows(archive)) == 30


def test_one_tick_drains_more_than_one_page_of_subject_debt(
    archive: Path, monkeypatch: pytest.MonkeyPatch, bounded_compute_adapter: BoundedComputeAdapter
) -> None:
    """Anti-vacuity: a single page per tick leaves 150 of 250 rows behind."""
    stage = _Stage("per_subject", subject_independent=False)
    _install(monkeypatch, stage)
    _seed(archive, "per_subject", 250)

    daemon_cli._drain_convergence_debt_backlog(
        archive / "index.db", budget_s=60.0, compute_adapter=bounded_compute_adapter
    )

    assert _rows(archive) == []
    assert sum(len(paths) for paths in stage.executions) == 250


def test_rows_owned_by_another_drain_do_not_fill_the_page(
    archive: Path, monkeypatch: pytest.MonkeyPatch, bounded_compute_adapter: BoundedComputeAdapter
) -> None:
    """Anti-vacuity: filtering owned stages after the LIMIT lets 100 newer
    ``raw_retention`` rows fill the page, so the generic row is never retried."""
    stage = _Stage("per_subject", subject_independent=False)
    _install(monkeypatch, stage)
    _seed(archive, "per_subject", 1)
    _seed(archive, "raw_retention", 150)

    assert daemon_cli._drain_convergence_debt_once(archive / "index.db", compute_adapter=bounded_compute_adapter) == 1

    remaining = _rows(archive)
    assert all(stage_name == "raw_retention" for stage_name, _target in remaining)
    assert len(remaining) == 150


def test_debt_waits_while_a_cold_build_is_unsettled(archive: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: without the cold-build guard the archive-wide stage runs
    against the empty active generation and clears the candidate's rows."""
    import asyncio

    from polylogue.daemon import intake_adapters

    stage = _Stage("archive_wide", subject_independent=True)
    _install(monkeypatch, stage)
    _seed(archive, "archive_wide", 3)
    monkeypatch.setattr(intake_adapters, "active_cold_build_generation", lambda _root=None: object())

    asyncio.run(daemon_cli._retry_convergence_debt_once(archive / "index.db"))

    assert stage.executions == []
    assert len(_rows(archive)) == 3


def test_promotion_release_makes_deferred_rows_due_and_keeps_failure_backoff(archive: Path) -> None:
    """Anti-vacuity: releasing every row would drop a real failure's backoff;
    releasing none leaves the deferred row waiting out its backoff."""
    cursor = CursorStore(archive / "index.db")
    for stage, deferred in (("deferred_stage", True), ("failed_stage", False)):
        cursor.record_convergence_debt(
            stage=stage,
            subject_type="source_path",
            subject_id=str(archive / "sources" / f"{stage}.jsonl"),
            error="waiting",
            deferred=deferred,
        )
    with sqlite3.connect(archive / "ops.db") as conn:
        conn.execute("UPDATE convergence_debt SET next_retry_at = '2999-01-01T00:00:00+00:00'")
        conn.commit()

    assert cursor.release_deferred_convergence_debt() == 1

    with sqlite3.connect(archive / "ops.db") as conn:
        retry = dict(conn.execute("SELECT stage, next_retry_at FROM convergence_debt"))
    assert retry == {"deferred_stage": None, "failed_stage": "2999-01-01T00:00:00+00:00"}


def test_stage_clear_keeps_rows_recorded_in_the_run_start_millisecond(archive: Path) -> None:
    """Anti-vacuity: an inclusive cutoff deletes the row written in the same
    millisecond the converging run started, which that run may not have seen."""
    cursor = CursorStore(archive / "index.db")
    for name in ("before", "same"):
        cursor.record_convergence_debt(
            stage="archive_stage",
            subject_type="source_path",
            subject_id=str(archive / "sources" / f"{name}.jsonl"),
            error="waiting",
        )
    with sqlite3.connect(archive / "ops.db") as conn:
        conn.execute("UPDATE convergence_debt SET updated_at_ms = 1000 WHERE target_id LIKE '%before.jsonl'")
        conn.execute("UPDATE convergence_debt SET updated_at_ms = 2000 WHERE target_id LIKE '%same.jsonl'")
        conn.commit()

    assert cursor.clear_stage_convergence_debt(stage="archive_stage", recorded_before_ms=2000) == 1

    with sqlite3.connect(archive / "ops.db") as conn:
        remaining = [row[0] for row in conn.execute("SELECT target_id FROM convergence_debt")]
    assert [Path(target).name for target in remaining] == ["same.jsonl"]


def test_a_release_that_could_not_write_raises(archive: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: returning zero on a skipped write reports a release that
    never happened as success, and the rows keep their backoff."""
    import polylogue.sources.live.cursor as cursor_module

    cursor = CursorStore(archive / "index.db")
    monkeypatch.setattr(cursor_module, "best_effort_cursor_write", lambda _label, _write: False)
    with pytest.raises(RuntimeError, match="not released"):
        cursor.release_deferred_convergence_debt()


def test_a_row_re_recorded_after_the_run_started_survives_the_ledger(archive: Path) -> None:
    """Anti-vacuity: clearing the settled stage's rows per subject deletes the
    row live ingest re-recorded after the archive-wide run started."""
    cursor = CursorStore(archive / "index.db")
    subject = str(archive / "sources" / "late.jsonl")
    cursor.record_convergence_debt(stage="archive_wide", subject_type="source_path", subject_id=subject, error="x")
    with sqlite3.connect(archive / "ops.db") as conn:
        conn.execute("UPDATE convergence_debt SET updated_at_ms = 5000")
        conn.commit()
    [debt] = cursor.list_convergence_debt(limit=10)

    class _Converged:
        converged = True

    retried = daemon_cli._record_convergence_debt_retries(
        cursor,
        [debt],
        {("archive_wide", "source_path", subject): _Converged()},
        {"archive_wide": 4000},
    )
    assert retried == 1
    assert len(_rows(archive)) == 1


@pytest.mark.parametrize("stage", ["lineage", "convergence"])
def test_skipped_retry_preserves_unmeasured_debt(archive: Path, stage: str) -> None:
    """Aggregate convergence cannot settle a stage which never evaluated its subject."""
    from types import SimpleNamespace

    cursor = CursorStore(archive / "index.db")
    subject = "synthetic-session"
    cursor.record_convergence_debt(stage=stage, subject_type="session_id", subject_id=subject, error="owed")
    [debt] = cursor.list_convergence_debt(limit=10)
    state = SimpleNamespace(converged=True, stages={"lineage": "skipped"}, last_error=None)
    daemon_cli._record_convergence_debt_retries(cursor, [debt], {(stage, "session_id", subject): state})
    assert _rows(archive) == [("lineage", subject)]


def test_generic_retry_preserves_each_unevaluated_stage(archive: Path) -> None:
    from types import SimpleNamespace

    cursor = CursorStore(archive / "index.db")
    subject = "synthetic-session"
    cursor.record_convergence_debt(stage="convergence", subject_type="session_id", subject_id=subject, error="owed")
    [debt] = cursor.list_convergence_debt(limit=10)
    state = SimpleNamespace(
        converged=False, stages={"lineage": "skipped", "titles": "failed", "summary": "done"}, last_error="pending"
    )
    daemon_cli._record_convergence_debt_retries(cursor, [debt], {("convergence", "session_id", subject): state})
    assert _rows(archive) == [("lineage", subject), ("titles", subject)]


@pytest.mark.parametrize("due", [False, True])
@pytest.mark.parametrize("converges", [False, True])
def test_frontier_fallback_preserves_existing_debt_retry_schedule(
    archive: Path,
    monkeypatch: pytest.MonkeyPatch,
    bounded_compute_adapter: BoundedComputeAdapter,
    due: bool,
    converges: bool,
) -> None:
    """An existing frontier debt owns the census, even after its due retry."""
    stage = _Stage("raw_frontier_inspection", subject_independent=True, converges=converges)
    _install(monkeypatch, stage)
    monkeypatch.setattr(
        "polylogue.operations.raw_frontier_inspection.make_raw_frontier_inspection_stage",
        lambda _db, **_kwargs: stage.build(),
    )
    _seed(archive, stage.name, 1)
    if not due:
        with sqlite3.connect(archive / "ops.db") as conn:
            conn.execute("UPDATE convergence_debt SET next_retry_at = '2999-01-01T00:00:00+00:00'")

    daemon_cli._drain_convergence_debt_and_frontier(archive / "index.db", compute_adapter=bounded_compute_adapter)
    assert len(stage.executions) == int(due)
    assert len(_rows(archive)) == int(not (due and converges))


def test_blocked_frontier_fallback_retains_registered_telemetry_outcome(
    archive: Path, monkeypatch: pytest.MonkeyPatch, bounded_compute_adapter: BoundedComputeAdapter
) -> None:
    """The real event validator must retain the incomplete census outcome."""
    from polylogue import logging as plog

    stage = _Stage("raw_frontier_inspection", subject_independent=True, converges=False)
    _install(monkeypatch, stage)
    monkeypatch.setattr(
        "polylogue.operations.raw_frontier_inspection.make_raw_frontier_inspection_stage",
        lambda _db, **_kwargs: stage.build(),
    )
    with plog.capture() as records:
        daemon_cli._drain_convergence_debt_and_frontier(archive / "index.db", compute_adapter=bounded_compute_adapter)
    terminal = [r for r in records if r["event"] == "daemon.raw_frontier_inspection.pass.completed"]
    assert len(terminal) == 1
    assert terminal[0]["outcome"] == "degraded"
    assert terminal[0]["reason"] == "frontier_inspection_blocked"
    assert not [r for r in records if r["event"] == "log.field_rejected"]
