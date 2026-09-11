"""Restart law for the typed session-profile convergence owner."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.infra.convergence_harness import (
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
