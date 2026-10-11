"""Exact retry ownership through the actual global product and Ops repository."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.product_retry import ProductObservation, ProductOutcome
from polylogue.operations.claude_workflow_convergence import make_claude_workflow_stage
from polylogue.storage.sqlite.archive_tiers.product_retry import (
    admit_product_retry,
    observe_product_retry,
    product_attempt_current,
    settle_product_retry,
)
from tests.infra.archive_templates import bootstrap_archive_root


def test_clock_observations_failures_duplicates_and_late_identity(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    with sqlite3.connect(tmp_path / "ops.db") as conn:

        def observe(identity: str, clock: int) -> None:
            observe_product_retry(
                conn,
                product="claude_workflow",
                target="current",
                observation=ProductObservation(identity, True),
                now_ms=clock,
            )

        def admit(identity: str, clock: int):
            return admit_product_retry(
                conn, product="claude_workflow", target="current", demand_identity=identity, owner="owner", now_ms=clock
            )

        observe("one", 0)
        first = admit("one", 0)
        assert first is not None
        assert settle_product_retry(conn, first, ProductOutcome.FAILED, now_ms=0, error="same")
        assert not settle_product_retry(conn, first, ProductOutcome.FAILED, now_ms=0, error="same")
        observe("two", 1)
        assert admit("two", 59999) is None
        second = admit("two", 60000)
        assert second is not None
        assert not product_attempt_current(conn, first)
        assert settle_product_retry(conn, second, ProductOutcome.FAILED, now_ms=60000, error="same")
        assert conn.execute("SELECT attempts,failure_streak,observation_count FROM convergence_debt").fetchone() == (
            2,
            2,
            2,
        )
        assert admit("two", 179999) is None
        third = admit("two", 180000)
        assert third is not None
        assert not product_attempt_current(conn, replace(third, ops_incarnation="replaced"))
        assert settle_product_retry(conn, third, ProductOutcome.CANCELLED, now_ms=180000)
        assert conn.execute("SELECT failure_streak,last_outcome FROM convergence_debt").fetchone() == (2, "cancelled")
        restarted = admit_product_retry(
            conn, product="claude_workflow", target="current", demand_identity="two", owner="restarted", now_ms=180001
        )
        assert restarted is not None
        observe("three", 180002)
        assert not settle_product_retry(conn, restarted, ProductOutcome.CURRENT, now_ms=180002)


def test_global_product_empty_discovery_and_actual_failure_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import polylogue.analysis.claude_workflow_materializer as materializer
    from polylogue.storage.archive_readiness import claude_workflow_materialization_status

    bootstrap_archive_root(tmp_path)
    clock = [0]
    stage = make_claude_workflow_stage(tmp_path / "index.db", now_ms=lambda: clock[0])
    assert stage.subject_independent and stage.owns_retry
    assert stage.check(tmp_path / "unrelated")
    assert stage.execute(tmp_path / "unrelated")
    assert not stage.check(tmp_path / "unrelated")
    status = claude_workflow_materialization_status(tmp_path / "ops.db")
    assert status is not None and status["status"] == "clean"
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute("DELETE FROM daemon_stage_events")
    starts = []

    def fail(_root: Path, **_kwargs: object):
        starts.append(clock[0])
        raise RuntimeError("same failure")

    monkeypatch.setattr(materializer, "materialize_claude_workflow_archive", fail)
    assert not stage.execute(tmp_path)
    clock[0] = 1
    assert stage.check(tmp_path)
    assert not stage.execute(tmp_path)
    assert starts == [0]
    clock[0] = 60000
    assert not stage.execute(tmp_path)
    assert starts == [0, 60000]
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        assert conn.execute(
            "SELECT target_type,target_id,attempts,failure_streak FROM convergence_debt"
        ).fetchall() == [("archive", "current", 2, 2)]
    assert claude_workflow_materialization_status(tmp_path / "ops.db")["status"] == "failed"


def test_actual_product_late_result_and_cancel_cannot_certify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import polylogue.analysis.claude_workflow_materializer as materializer

    bootstrap_archive_root(tmp_path)
    stage = make_claude_workflow_stage(tmp_path / "index.db", now_ms=lambda: 0, owner="old")
    original = materializer.materialize_claude_workflow_archive

    def supersede(root: Path, **kwargs: object):
        with sqlite3.connect(root / "ops.db") as conn:
            conn.execute("UPDATE product_retry_incarnation SET incarnation='new'")
        return original(root, **kwargs)

    monkeypatch.setattr(materializer, "materialize_claude_workflow_archive", supersede)
    result = stage.execute(tmp_path)
    assert not result and result.stale_preparation
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM daemon_stage_events").fetchone() == (0,)

    def cancelled(_root: Path, **_kwargs: object):
        raise DaemonOperationCancelled("cancelled")

    monkeypatch.setattr(materializer, "materialize_claude_workflow_archive", cancelled)
    with pytest.raises(DaemonOperationCancelled):
        make_claude_workflow_stage(tmp_path / "index.db", now_ms=lambda: 0, owner="restart").execute(tmp_path)
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        assert conn.execute(
            "SELECT admitted_attempt_id,last_outcome,failure_streak FROM convergence_debt"
        ).fetchone() == (None, "cancelled", 0)


def test_global_owner_has_no_subject_debt_translator() -> None:
    from polylogue.daemon.convergence import FileState, StageState
    from polylogue.sources.live.convergence_debt import convergence_debt_from_state
    from polylogue.sources.live.convergence_outcome import settled_convergence_stages

    state = FileState(Path("neutral"), owned_retry_stages=frozenset({"claude_workflow"}))
    state.stages["claude_workflow"] = StageState.PENDING
    assert convergence_debt_from_state(state.path, state) == []
    state.stages["claude_workflow"] = StageState.DONE
    assert settled_convergence_stages(state) == ()


def test_same_ops_bytes_on_a_new_inode_fence_the_old_attempt(tmp_path: Path) -> None:
    import shutil
    from contextlib import closing

    bootstrap_archive_root(tmp_path)
    ops = tmp_path / "ops.db"
    with closing(sqlite3.connect(ops)) as conn:
        observe_product_retry(
            conn, product="claude_workflow", target="current", observation=ProductObservation("one", True), now_ms=0
        )
        attempt = admit_product_retry(
            conn, product="claude_workflow", target="current", demand_identity="one", owner="owner", now_ms=0
        )
        assert attempt is not None
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    replacement = tmp_path / "replacement.db"
    shutil.copyfile(ops, replacement)
    replacement.replace(ops)
    with closing(sqlite3.connect(ops)) as conn:
        # The singleton and exact debt row survived the copy. Only physical
        # Ops identity distinguishes this replacement from the original owner.
        assert not product_attempt_current(conn, attempt)
        assert not settle_product_retry(conn, attempt, ProductOutcome.CURRENT, now_ms=1)
