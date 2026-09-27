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
    archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: without the subject-independent collapse the stage runs
    once per page (three times for 250 rows)."""
    stage = _Stage("archive_wide", subject_independent=True)
    _install(monkeypatch, stage)
    _seed(archive, "archive_wide", 250)

    daemon_cli._drain_convergence_debt_backlog(archive / "index.db", budget_s=60.0)

    assert len(stage.executions) == 1
    assert len(stage.executions[0]) == 1
    assert _rows(archive) == []


def test_unconverged_subject_independent_stage_keeps_its_rows(archive: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stage that is still pending settles nothing; its rows back off."""
    stage = _Stage("archive_wide", subject_independent=True, converges=False)
    _install(monkeypatch, stage)
    _seed(archive, "archive_wide", 30)

    daemon_cli._drain_convergence_debt_backlog(archive / "index.db", budget_s=60.0)

    assert len(stage.executions) == 1
    assert len(_rows(archive)) == 30


def test_one_tick_drains_more_than_one_page_of_subject_debt(archive: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: a single page per tick leaves 150 of 250 rows behind."""
    stage = _Stage("per_subject", subject_independent=False)
    _install(monkeypatch, stage)
    _seed(archive, "per_subject", 250)

    daemon_cli._drain_convergence_debt_backlog(archive / "index.db", budget_s=60.0)

    assert _rows(archive) == []
    assert sum(len(paths) for paths in stage.executions) == 250


def test_rows_owned_by_another_drain_do_not_fill_the_page(archive: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: filtering owned stages after the LIMIT lets 100 newer
    ``raw_retention`` rows fill the page, so the generic row is never retried."""
    stage = _Stage("per_subject", subject_independent=False)
    _install(monkeypatch, stage)
    _seed(archive, "per_subject", 1)
    _seed(archive, "raw_retention", 150)

    assert daemon_cli._drain_convergence_debt_once(archive / "index.db") == 1

    remaining = _rows(archive)
    assert all(stage_name == "raw_retention" for stage_name, _target in remaining)
    assert len(remaining) == 150


def test_debt_waits_while_a_cold_build_is_unsettled(archive: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: without the cold-build guard the archive-wide stage runs
    against the empty active generation and clears the candidate's rows."""
    import asyncio

    from polylogue.sources.live import cold_build

    stage = _Stage("archive_wide", subject_independent=True)
    _install(monkeypatch, stage)
    _seed(archive, "archive_wide", 3)
    monkeypatch.setattr(cold_build, "active_cold_build_generation", lambda _root=None: object())

    asyncio.run(daemon_cli._retry_convergence_debt_once(archive / "index.db"))

    assert stage.executions == []
    assert len(_rows(archive)) == 3
