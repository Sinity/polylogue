"""Contract tests for the reusable durability fault fixture."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.infra.durability_faults import (
    DurabilityFaultPoint,
    DurabilityFaultRegistry,
    InjectedCrash,
)


def test_registry_reaches_direct_and_pathlib_filesystem_boundaries(
    tmp_path: Path, durability_faults: DurabilityFaultRegistry
) -> None:
    """The fixture observes the calls users actually make, including Path.unlink."""
    target = tmp_path / "payload"
    target.write_bytes(b"payload")

    durability_faults.arm(DurabilityFaultPoint.UNLINK, action="kill")
    with pytest.raises(InjectedCrash):
        with durability_faults.installed():
            target.unlink()

    assert durability_faults.count(DurabilityFaultPoint.UNLINK) == 1
    assert target.exists()


def test_registry_wraps_sqlite_commit_without_replacing_connection_semantics(
    tmp_path: Path, durability_faults: DurabilityFaultRegistry
) -> None:
    """A commit fault is injected at the connection method, not SQL text."""
    database = tmp_path / "tier.db"
    durability_faults.arm(DurabilityFaultPoint.COMMIT)
    with pytest.raises(RuntimeError, match="injected commit fault"):
        with durability_faults.installed():
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE item (value TEXT)")
            connection.commit()

    assert durability_faults.count(DurabilityFaultPoint.COMMIT) == 1


def test_run_requires_reaching_the_selected_point_and_runs_recovery(durability_faults: DurabilityFaultRegistry) -> None:
    recovered: list[str] = []
    checked: list[str] = []

    result = durability_faults.run(
        DurabilityFaultPoint.FSYNC,
        lambda: _fsync_once(),
        recover=lambda: recovered.append("recovered"),
        assert_invariants=lambda: checked.append("sound"),
    )

    assert result.interrupted
    assert recovered == ["recovered"]
    assert checked == ["sound"]
    assert result.events == (result.events[0],)


def test_run_rejects_a_vacuous_operation(durability_faults: DurabilityFaultRegistry) -> None:
    with pytest.raises(AssertionError, match="was not reached"):
        durability_faults.run(
            DurabilityFaultPoint.REPLACE,
            lambda: None,
            recover=lambda: None,
            assert_invariants=lambda: None,
        )


def _fsync_once() -> None:
    import os

    descriptor = os.open("/dev/null", os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
