"""Restart law for the typed session-profile convergence owner."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from polylogue.daemon.convergence import (
    DaemonConverger,
    SessionProfileConvergenceOwner,
    make_session_profile_derivation,
    make_session_profile_frame,
)
from polylogue.daemon.derivation import Budget
from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from tests.infra.convergence_harness import (
    converge_session_profiles,
    raw_authority_facts,
    seed_partial_convergence_archive,
    session_materialization_facts,
)


def _run_typed_owner_in_fresh_process(
    index_db: Path,
    archive_root: Path,
    session_ids: tuple[str, ...] | None,
) -> None:
    """Run the real owner after interpreter restart with a controlled clock."""
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    script = (
        "from pathlib import Path\n"
        "from tests.infra.convergence_harness import converge_session_profiles\n"
        f"converge_session_profiles(Path({str(index_db)!r}), Path({str(archive_root)!r}), {session_ids!r}, now=lambda: 0.0)\n"
        "print('CONVERGED=1')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (
        f"fresh-process typed owner failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "CONVERGED=1" in completed.stdout


@pytest.mark.contract
@pytest.mark.timeout(90)
def test_typed_session_owner_survives_restart_and_leaves_unrelated_sessions_absent(tmp_path: Path) -> None:
    """Restart re-enumerates durable output rather than a legacy debt row.

    Anti-vacuity: replace the typed owner with a generic-stage shortcut or make
    the owner retain an in-memory pending set and this no longer exercises the typed owner from a
    fresh interpreter against its output relation. The obsolete ``derived``
    debt is not retried here: production CLI filtering owns its disposal.
    """
    recovered = seed_partial_convergence_archive(tmp_path / "recovered", target_hot=False)
    raw_before = raw_authority_facts(recovered.source_db)

    _run_typed_owner_in_fresh_process(recovered.index_db, recovered.root, (recovered.target_session_id,))
    first = session_materialization_facts(recovered.index_db, session_id=recovered.target_session_id)
    assert first.profile is not None
    assert session_materialization_facts(recovered.index_db, session_id=recovered.unrelated_session_id).profile is None
    assert raw_authority_facts(recovered.source_db) == raw_before

    _run_typed_owner_in_fresh_process(recovered.index_db, recovered.root, (recovered.target_session_id,))
    assert session_materialization_facts(recovered.index_db, session_id=recovered.target_session_id) == first
    assert raw_authority_facts(recovered.source_db) == raw_before


@pytest.mark.contract
@pytest.mark.timeout(90)
def test_fresh_no_hint_owner_sweeps_all_session_profiles_before_legacy_debt_can_clear(tmp_path: Path) -> None:
    """A fresh no-hint owner run reconstructs and completes the whole archive.

    Anti-vacuity: replace ``scope=None`` with the changed-session callback,
    stop after the first page, or clear legacy debt without a terminal owner
    report and the unrelated partition remains absent despite durable source
    evidence requiring it.
    """
    recovered = seed_partial_convergence_archive(tmp_path / "recovered", target_hot=False)
    raw_before = raw_authority_facts(recovered.source_db)

    _run_typed_owner_in_fresh_process(recovered.index_db, recovered.root, None)
    target = session_materialization_facts(recovered.index_db, session_id=recovered.target_session_id)
    unrelated = session_materialization_facts(recovered.index_db, session_id=recovered.unrelated_session_id)
    assert target.profile is not None
    assert unrelated.profile is not None
    assert raw_authority_facts(recovered.source_db) == raw_before

    _run_typed_owner_in_fresh_process(recovered.index_db, recovered.root, None)
    assert session_materialization_facts(recovered.index_db, session_id=recovered.target_session_id) == target
    assert session_materialization_facts(recovered.index_db, session_id=recovered.unrelated_session_id) == unrelated
    assert raw_authority_facts(recovered.source_db) == raw_before


@pytest.mark.contract
@pytest.mark.timeout(90)
def test_fresh_no_hint_owner_retires_an_excess_profile_before_reporting_complete(tmp_path: Path) -> None:
    """A restart sweep reaches the output-only excess phase, not just sources.

    Anti-vacuity: stop the no-hint owner after required source keys and the
    profile for the removed session remains in the output relation forever.
    """
    recovered = seed_partial_convergence_archive(tmp_path / "recovered", target_hot=False)
    converge_session_profiles(
        recovered.index_db,
        recovered.root,
        (recovered.target_session_id, recovered.unrelated_session_id),
        now=lambda: 0.0,
    )
    raw_before = raw_authority_facts(recovered.source_db)
    with sqlite3.connect(recovered.index_db) as conn:
        assert conn.execute(
            "SELECT 1 FROM session_profiles WHERE session_id = ?",
            (recovered.unrelated_session_id,),
        ).fetchone()
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (recovered.unrelated_session_id,))
        conn.commit()

    _run_typed_owner_in_fresh_process(recovered.index_db, recovered.root, None)

    with sqlite3.connect(recovered.index_db) as conn:
        assert conn.execute(
            "SELECT 1 FROM session_profiles WHERE session_id = ?",
            (recovered.target_session_id,),
        ).fetchone()
        assert (
            conn.execute(
                "SELECT 1 FROM session_profiles WHERE session_id = ?",
                (recovered.unrelated_session_id,),
            ).fetchone()
            is None
        )
    assert raw_authority_facts(recovered.source_db) == raw_before


@pytest.mark.asyncio
async def test_real_factory_defers_hot_target_without_losing_the_no_hint_cursor(tmp_path: Path) -> None:
    """The production quiet predicate and owner preserve archive-sweep fairness.

    Anti-vacuity: bypass ``make_session_profile_derivation``'s real
    source-path hot check, retain a scoped frame's cursor, or resume the
    archive sweep from its beginning and this cannot show a hot earlier
    session deferred, then repaired by its callback, while the cold sibling
    still receives the next no-hint page.
    """
    recovered = seed_partial_convergence_archive(tmp_path / "recovered", target_hot=True)
    # The hot predicate receives an injected clock.  It is deliberately based
    # on the fixture file metadata rather than the ambient test process clock.
    observed_now = recovered.target_source.stat().st_mtime + 1.0
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator()
    adapter = make_session_profile_derivation(
        recovered.index_db,
        archive_root=recovered.root,
        now=lambda: observed_now,
    )
    converger = DaemonConverger((), derivations=[adapter])
    owner = SessionProfileConvergenceOwner(
        converger,
        compute_adapter=compute,
        write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
    )
    archive_frame = make_session_profile_frame(recovered.index_db, archive_root=recovered.root, scope=None)
    try:
        # Discovery is the relevant pass bound: quiet deferral consumes no
        # compute capacity, while an inspection limit stops before the quiet
        # predicate is evaluated. One discovered page therefore captures the
        # real no-hint continuation after the hot key was actually deferred.
        first = await owner.converge(archive_frame, budget=Budget(page=1, discovery=1, compute=1))
        assert first.pending == 1
        assert first.done == 0
        assert session_materialization_facts(recovered.index_db, session_id=recovered.target_session_id).profile is None
        assert (
            session_materialization_facts(recovered.index_db, session_id=recovered.unrelated_session_id).profile is None
        )
        archive_cursor = first.cursor
        assert not archive_cursor.position("session_profile").swept

        recovered.make_target_quiet()
        targeted = await owner.converge(
            make_session_profile_frame(
                recovered.index_db,
                archive_root=recovered.root,
                scope=(recovered.target_session_id,),
            ),
            budget=Budget(page=1, compute=1),
        )
        assert targeted.done == 1
        assert targeted.pending == 0
        assert (
            session_materialization_facts(recovered.index_db, session_id=recovered.target_session_id).profile
            is not None
        )
        assert converger._derivation_cursor == archive_cursor

        resumed = await owner.converge(archive_frame, budget=Budget(page=1, discovery=1, compute=1))
        assert resumed.done == 1
        assert resumed.pending == 0
        assert (
            session_materialization_facts(recovered.index_db, session_id=recovered.unrelated_session_id).profile
            is not None
        )
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)
