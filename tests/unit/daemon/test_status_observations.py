"""Status components distinguish a measured zero from an unreadable source.

The defect these guard against is a status payload that reports ``0`` after
a lock, permission, or corruption error and then derives readiness from it.
Every test here names the mutation that reddens it: translating an
unmeasured state back into ``0``, ``[]``, or an OK verdict.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from polylogue import paths as polylogue_paths
from polylogue.daemon.observation import (
    Observation,
    ObservationBoard,
    ObservationState,
    UnmeasuredObservationError,
    observe_bounded,
    worst_state,
)


def test_raw_failure_read_error_does_not_become_zero_or_ok(tmp_path: Path) -> None:
    """Anti-vacuity: a failed source read must retain an unavailable count."""
    from polylogue.daemon import status as status_module

    result = status_module._archive_raw_failure_info(tmp_path / "missing-source.db")

    assert result["raw_failure_lifecycle_available"] is False
    assert result["raw_failure_lifecycle_state"] == "unavailable"
    assert result["parse_failures"] is None
    assert result["validation_failures"] is None
    assert result["samples"] == []


def test_minimal_snapshot_does_not_invent_raw_failure_warning_count() -> None:
    """The request-safe path has no rich source evidence to count."""
    from polylogue.daemon.status_snapshot import _minimal_status_payload

    payload = _minimal_status_payload()

    assert payload["raw_detection_warnings"] is None
    assert payload["raw_failure_lifecycle_available"] is False


def test_unreadable_archive_tier_count_is_explicitly_unavailable(tmp_path: Path) -> None:
    """A corrupt tier is present, but its table count is not a measured zero."""
    from polylogue.daemon.status import _archive_tier_status

    path = tmp_path / "index.db"
    path.write_bytes(b"not sqlite")
    result = _archive_tier_status("index", path)

    assert result.exists is True
    assert result.table_count is None
    assert result.table_count_state == "unavailable"


def test_component_snapshot_metadata_keeps_collection_state_out_of_business_readiness() -> None:
    """The surface projection carries timeout evidence without inventing a value."""
    from polylogue.daemon.status import _status_component_metadata
    from polylogue.operations.status_protocol import ComponentSnapshot

    metadata = _status_component_metadata(
        {
            "slow": ComponentSnapshot(
                name="slow",
                scope="archive",
                state="timed_out",
                value=0,
                captured_at="now",
                age_s=0.0,
                deadline_s=0.1,
                error="collector exceeded deadline_s=0.1",
            )
        }
    )

    assert metadata == [
        {
            "component": "slow",
            "scope": "archive",
            "state": "timed_out",
            "captured_at": "now",
            "age_s": 0.0,
            "deadline_s": 0.1,
            "fingerprint": None,
            "error": "collector exceeded deadline_s=0.1",
            "last_good_at": None,
        }
    ]


def test_a_genuine_zero_is_measured() -> None:
    observation = observe_bounded("sessions", lambda: 0, budget_s=1.0)

    assert observation.state is ObservationState.MEASURED
    assert observation.value == 0
    assert observation.require() == 0


def test_a_sqlite_error_is_unreadable_not_zero() -> None:
    """Mutation: return 0 on ``sqlite3.Error`` and this reddens."""

    def collect() -> int:
        raise sqlite3.OperationalError("database is locked")

    observation = observe_bounded("sessions", collect, budget_s=1.0)

    assert observation.state is ObservationState.UNREADABLE
    assert observation.value is None
    assert "database is locked" in (observation.reason or "")
    with pytest.raises(UnmeasuredObservationError, match="sessions"):
        observation.require()


def test_a_missing_file_is_unreadable_not_empty() -> None:
    def collect() -> list[str]:
        raise FileNotFoundError("index.db")

    observation = observe_bounded("sources", collect, budget_s=1.0)

    assert observation.state is ObservationState.UNREADABLE
    assert observation.value is None


def test_an_arbitrary_raise_is_failed() -> None:
    def collect() -> int:
        raise ValueError("bad row")

    observation = observe_bounded("costs", collect, budget_s=1.0)

    assert observation.state is ObservationState.FAILED
    assert "ValueError" in (observation.reason or "")


@pytest.mark.uses_real_clock("measures a real thread's wall-clock budget")
def test_a_stalled_collector_times_out_within_its_budget() -> None:
    """Mutation: drop the budget and a stalled collector delays every answer."""
    release = threading.Event()

    def collect() -> int:
        release.wait(timeout=10.0)
        return 5

    started = time.monotonic()
    try:
        observation = observe_bounded("embeddings", collect, budget_s=0.2)
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert observation.state is ObservationState.TIMED_OUT
    assert observation.value is None
    assert elapsed < 2.0


@pytest.mark.uses_real_clock("measures a real thread's wall-clock budget")
def test_a_stalled_collector_does_not_delay_its_siblings() -> None:
    release = threading.Event()

    def stalled() -> int:
        release.wait(timeout=10.0)
        return 5

    board = ObservationBoard()
    started = time.monotonic()
    try:
        board.publish(observe_bounded("embeddings", stalled, budget_s=0.2))
        board.publish(observe_bounded("sessions", lambda: 12, budget_s=1.0))
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert board.get_or_unavailable("embeddings").state is ObservationState.TIMED_OUT
    assert board.get_or_unavailable("sessions").require() == 12
    assert elapsed < 3.0


def test_a_component_that_never_reported_is_unavailable() -> None:
    """Mutation: default to ``0`` here and an absent domain reads as empty."""
    board = ObservationBoard()

    observation = board.get_or_unavailable("blobs")

    assert observation.state is ObservationState.UNAVAILABLE
    assert observation.value is None


def test_a_frame_change_demotes_a_reading_to_stale() -> None:
    """Last-good evidence is advisory; it cannot certify the current frame."""
    measured = Observation.measured("sessions", 42, frame="generation-7")

    staled = measured.staled(reason="active generation is generation-8")

    assert staled.state is ObservationState.STALE
    assert staled.value is None
    assert staled.frame == "generation-7"


def test_unmeasured_cannot_construct_a_measured_observation() -> None:
    with pytest.raises(ValueError, match="cannot construct a MEASURED"):
        Observation.unmeasured("sessions", ObservationState.MEASURED, reason="impossible")


def test_a_serialized_unmeasured_component_carries_no_value() -> None:
    """Every surface renders one payload; none of them can invent a number."""
    payload = Observation.unmeasured("sessions", ObservationState.UNREADABLE, reason="database is locked").as_dict()

    assert payload["value"] is None
    assert payload["state"] == "unreadable"
    assert payload["reason"] == "database is locked"


def test_severity_cannot_be_ok_while_a_component_is_unavailable() -> None:
    """Mutation: rank UNAVAILABLE below MEASURED and a broken tier reads OK."""
    states = [
        ObservationState.MEASURED,
        ObservationState.MEASURED,
        ObservationState.UNAVAILABLE,
    ]

    assert worst_state(states) is ObservationState.UNAVAILABLE
    assert worst_state([]) is ObservationState.MEASURED
    assert worst_state([ObservationState.UNAVAILABLE, ObservationState.FAILED]) is ObservationState.FAILED
    assert worst_state([ObservationState.SKIPPED]) is ObservationState.SKIPPED


def test_reading_the_board_performs_no_collection() -> None:
    """Compact status costs a dictionary copy, never an archive walk."""
    collections = 0

    def collect() -> int:
        nonlocal collections
        collections += 1
        return 3

    board = ObservationBoard()
    board.publish(observe_bounded("sessions", collect, budget_s=1.0))

    for _ in range(50):
        board.snapshot()
        board.get_or_unavailable("sessions")

    assert collections == 1


def test_a_detail_operation_is_named_rather_than_executed() -> None:
    """Expensive exact denominators are referenced, not folded into compact."""
    observation = observe_bounded(
        "fts_backlog",
        lambda: 128,
        budget_s=1.0,
        detail_operation="daemon.fts.exact_backlog",
    )

    assert observation.detail_operation == "daemon.fts.exact_backlog"
    assert observation.as_dict()["detail_operation"] == "daemon.fts.exact_backlog"


# -- the daemon status payload names halted units ---------------------------


def test_daemon_status_names_every_halted_unit_and_is_not_ok(tmp_path: Path) -> None:
    """The third leg of the halt property, on the production status route.

    Mutation: drop ``and not halted_units`` from the payload's ``ok`` and a
    daemon with a dead source reports healthy again.
    """
    from polylogue.daemon.service_halt import HaltReason, HaltRegistry, UnitKind, unit_id
    from polylogue.daemon.status import daemon_status_payload, format_daemon_status_lines

    archive_root = Path(polylogue_paths.archive_root())
    halts = HaltRegistry(archive_root)
    halts.halt(
        unit_id(UnitKind.SOURCE, "claude-code"),
        reason=HaltReason.TERMINAL_REFUSAL,
        message="refusing further ingest until restart",
        frame="daemon:1",
    )

    payload = daemon_status_payload(sources=(), include_archive_debt=False)

    assert payload["ok"] is False
    halted = payload["halted_units"]
    assert isinstance(halted, list)
    records = [record for record in halted if isinstance(record, dict)]
    assert [record["unit"] for record in records] == ["source:claude-code"]
    assert records[0]["reason"] == "terminal_refusal"
    assert records[0]["frame"] == "daemon:1"

    lines = format_daemon_status_lines(payload)
    assert any("source:claude-code" in line for line in lines)
    assert any("HALTED" in line for line in lines)


def test_daemon_status_with_no_halt_reports_an_empty_list(tmp_path: Path) -> None:
    from polylogue.daemon.status import daemon_status_payload

    payload = daemon_status_payload(sources=(), include_archive_debt=False)

    assert payload["halted_units"] == []


def test_service_states_are_absent_outside_a_composed_daemon() -> None:
    """Mutation: default to ``{}`` and a one-shot CLI looks like a live daemon."""
    from polylogue.daemon.status import supervised_service_states

    assert supervised_service_states() is None
