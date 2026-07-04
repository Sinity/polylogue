"""Tests for routing replay failures to the daemon raw-failure surface.

Pins the acceptance criteria from issue #1198:

* ``route_failure_sample`` writes a per-record JSONL line under
  ``<archive_root>/.maintenance-state/failures.jsonl`` with the
  originating ``operation_id``, target, kind, locator, message, and
  routed timestamp;
* sensitive payloads (absolute paths embedded in messages and
  locators) are redacted at write time;
* read/count helpers cap at :data:`MAINTENANCE_FAILURE_SAMPLE_LIMIT`
  and never raise on a missing file;
* the replay executor calls ``route_failure_sample`` for every
  ``FailureSample`` it appends to its in-memory bounded envelope, so
  the on-disk surface matches the returned
  :class:`BackfillOperation`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from polylogue.config import Config
from polylogue.maintenance.failure_routing import (
    MAINTENANCE_FAILURE_SAMPLE_LIMIT,
    MaintenanceFailureRecord,
    clear_maintenance_failures,
    count_maintenance_failures,
    read_maintenance_failures,
    resolve_maintenance_failures,
    route_failure_sample,
)
from polylogue.maintenance.planner import FailureSample
from polylogue.maintenance.replay import execute_replay


def _make_config(tmp_path: Path) -> Config:
    archive_root = tmp_path / "archive"
    render_root = tmp_path / "render"
    archive_root.mkdir(parents=True, exist_ok=True)
    render_root.mkdir(parents=True, exist_ok=True)
    return Config(
        archive_root=archive_root,
        render_root=render_root,
        sources=[],
        db_path=tmp_path / "archive.db",
    )


# ---------------------------------------------------------------------------
# route_failure_sample contract
# ---------------------------------------------------------------------------


def test_route_failure_sample_persists_record(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    sample = FailureSample(
        kind="RuntimeError",
        locator="target:session_insights",
        message="boom",
    )
    record = route_failure_sample(
        sample,
        operation_id="op-abc",
        archive_root=Path(config.archive_root),
        target="session_insights",
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert record.operation_id == "op-abc"
    assert record.target == "session_insights"
    assert record.kind == "RuntimeError"
    assert record.locator == "target:session_insights"
    assert record.message == "boom"
    assert record.routed_at == "2026-01-01T00:00:00+00:00"

    persisted = read_maintenance_failures(Path(config.archive_root))
    assert len(persisted) == 1
    assert persisted[0] == record


def test_route_failure_sample_infers_target_from_locator(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    sample = FailureSample(
        kind="ValueError",
        locator="target:session_insights:session:abc123",
        message="oops",
    )
    record = route_failure_sample(
        sample,
        operation_id="op-xyz",
        archive_root=Path(config.archive_root),
    )
    assert record.target == "session_insights"


def test_resolve_maintenance_failures_removes_matching_target_and_kind(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    root = Path(config.archive_root)
    route_failure_sample(
        FailureSample(
            kind="UnsupportedReplayTargetError",
            locator="target:message_type_backfill",
            message="not wired",
        ),
        operation_id="op-old",
        archive_root=root,
        target="message_type_backfill",
    )
    route_failure_sample(
        FailureSample(
            kind="RepairReportedFailure",
            locator="target:message_type_backfill",
            message="provider call failed",
        ),
        operation_id="op-real",
        archive_root=root,
        target="message_type_backfill",
    )
    route_failure_sample(
        FailureSample(
            kind="UnsupportedReplayTargetError",
            locator="target:orphaned_messages",
            message="not wired",
        ),
        operation_id="op-other",
        archive_root=root,
        target="orphaned_messages",
    )

    removed = resolve_maintenance_failures(
        root,
        target="message_type_backfill",
        kinds=("UnsupportedReplayTargetError",),
    )

    assert removed == 1
    remaining = read_maintenance_failures(root)
    assert [(r.target, r.kind) for r in remaining] == [
        ("message_type_backfill", "RepairReportedFailure"),
        ("orphaned_messages", "UnsupportedReplayTargetError"),
    ]


def test_route_failure_sample_redacts_absolute_paths_in_message(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    sample = FailureSample(
        kind="OSError",
        locator="target:orphaned_messages",
        message="Failed to read /home/operator/data/secret.json: permission denied",
    )
    record = route_failure_sample(
        sample,
        operation_id="op-1",
        archive_root=Path(config.archive_root),
    )
    assert "/home/operator/data/secret.json" not in record.message
    assert "[redacted]" in record.message
    assert "permission denied" in record.message


def test_route_failure_sample_redacts_absolute_paths_in_locator(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    sample = FailureSample(
        kind="ValueError",
        locator="target:session_insights:session:c1:/home/operator/data.json",
        message="bad",
    )
    record = route_failure_sample(
        sample,
        operation_id="op-2",
        archive_root=Path(config.archive_root),
    )
    assert "/home/operator/data.json" not in record.locator
    assert "[redacted]" in record.locator


def test_route_failure_sample_truncates_long_messages(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    long_message = "x" * 2000
    sample = FailureSample(kind="K", locator="target:t", message=long_message)
    record = route_failure_sample(
        sample,
        operation_id="op-3",
        archive_root=Path(config.archive_root),
    )
    assert len(record.message) <= 500
    assert record.message.endswith("...")


def test_route_failure_sample_appends_multiple(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    for i in range(3):
        route_failure_sample(
            FailureSample(kind=f"K{i}", locator=f"target:t{i}", message=f"m{i}"),
            operation_id="op-multi",
            archive_root=Path(config.archive_root),
        )
    records = read_maintenance_failures(Path(config.archive_root))
    assert len(records) == 3
    assert [r.kind for r in records] == ["K0", "K1", "K2"]


def test_route_failure_sample_never_writes_raw_paths_to_disk(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    sample = FailureSample(
        kind="OSError",
        locator="target:session_insights:session:c1:/home/op/secret.json",
        message="Failed at /home/op/secret.json",
    )
    route_failure_sample(
        sample,
        operation_id="op-secret",
        archive_root=Path(config.archive_root),
    )
    raw = (Path(config.archive_root) / ".maintenance-state" / "failures.jsonl").read_text()
    assert "/home/op/secret.json" not in raw


# ---------------------------------------------------------------------------
# Reader helpers
# ---------------------------------------------------------------------------


def test_read_maintenance_failures_missing_file_returns_empty(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    assert read_maintenance_failures(Path(config.archive_root)) == []
    assert count_maintenance_failures(Path(config.archive_root)) == 0


def test_read_maintenance_failures_caps_at_limit(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    for i in range(MAINTENANCE_FAILURE_SAMPLE_LIMIT + 10):
        route_failure_sample(
            FailureSample(kind="K", locator=f"target:t:{i}", message=str(i)),
            operation_id="op-cap",
            archive_root=Path(config.archive_root),
        )
    records = read_maintenance_failures(Path(config.archive_root))
    assert len(records) == MAINTENANCE_FAILURE_SAMPLE_LIMIT
    # Most recent entries are returned (tail of the file)
    assert records[-1].message == str(MAINTENANCE_FAILURE_SAMPLE_LIMIT + 9)


def test_read_maintenance_failures_tolerates_unparseable_lines(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    state_dir = Path(config.archive_root) / ".maintenance-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "failures.jsonl"
    path.write_text(
        '{"operation_id":"a","target":"t","kind":"K","locator":"target:t","message":"m","routed_at":"x"}\nnot json\n\n'
    )
    records = read_maintenance_failures(Path(config.archive_root))
    assert len(records) == 1
    assert records[0].operation_id == "a"


def test_count_maintenance_failures_skips_blank_lines(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    state_dir = Path(config.archive_root) / ".maintenance-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "failures.jsonl"
    path.write_text('{"operation_id":"a"}\n\n{"operation_id":"b"}\n\n')
    assert count_maintenance_failures(Path(config.archive_root)) == 2


def test_clear_maintenance_failures_removes_file(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    route_failure_sample(
        FailureSample(kind="K", locator="target:t", message="m"),
        operation_id="op-clr",
        archive_root=Path(config.archive_root),
    )
    assert count_maintenance_failures(Path(config.archive_root)) == 1
    clear_maintenance_failures(Path(config.archive_root))
    assert count_maintenance_failures(Path(config.archive_root)) == 0


def test_maintenance_failure_record_roundtrip(tmp_path: Path) -> None:
    record = MaintenanceFailureRecord(
        operation_id="op",
        target="t",
        kind="K",
        locator="target:t",
        message="m",
        routed_at="2026-01-01T00:00:00+00:00",
    )
    payload = record.to_dict()
    parsed = MaintenanceFailureRecord.from_dict(payload)
    assert parsed == record


# ---------------------------------------------------------------------------
# Replay executor wiring
# ---------------------------------------------------------------------------


def test_execute_replay_routes_unsupported_target_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unresolvable target appends a FailureSample that is also routed to disk."""
    config = _make_config(tmp_path)

    # Force a target to be resolved but missing from the dispatch table.
    from polylogue.maintenance import replay as replay_mod
    from polylogue.maintenance.models import MaintenanceCategory
    from polylogue.maintenance.targets import MaintenanceTargetMode, MaintenanceTargetSpec

    class _FakeCatalog:
        def resolve(self, names: tuple[str, ...]) -> tuple[MaintenanceTargetSpec, ...]:
            return tuple(
                MaintenanceTargetSpec(
                    name=name,
                    mode=MaintenanceTargetMode.REPAIR,
                    category=MaintenanceCategory.DERIVED_REPAIR,
                    destructive=False,
                    description="",
                )
                for name in names
            )

    monkeypatch.setattr(replay_mod, "build_maintenance_target_catalog", lambda: _FakeCatalog())

    operation = execute_replay(
        config,
        targets=["__missing__"],
        operation_id="op-route",
        persist_state=False,
    )

    assert operation.status.value == "failed"
    assert len(operation.failure_samples.samples) == 1
    persisted = read_maintenance_failures(Path(config.archive_root))
    assert len(persisted) == 1
    assert persisted[0].operation_id == "op-route"
    assert persisted[0].target == "__missing__"
    assert persisted[0].kind == "UnsupportedReplayTargetError"


def test_execute_replay_routes_repair_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A repair fn that raises records a routed failure with the exception class."""
    config = _make_config(tmp_path)

    from polylogue.config import Config as ConfigT
    from polylogue.maintenance import replay as replay_mod
    from polylogue.storage.repair import RepairResult as RepairResultT

    def _bad_repair(_config: ConfigT, _dry_run: bool) -> RepairResultT:
        raise RuntimeError("session insight rebuild failed: /tmp/path/session_42")

    monkeypatch.setitem(replay_mod._REPLAY_DISPATCH, "session_insights", _bad_repair)

    operation = execute_replay(
        config,
        targets=["session_insights"],
        operation_id="op-raise",
        persist_state=False,
    )

    assert operation.status.value == "failed"
    persisted = read_maintenance_failures(Path(config.archive_root))
    assert len(persisted) == 1
    assert persisted[0].operation_id == "op-raise"
    assert persisted[0].target == "session_insights"
    assert persisted[0].kind == "RuntimeError"
    # Path is redacted at routing time.
    assert "/tmp/path/session_42" not in persisted[0].message


def test_execute_replay_routes_repair_reported_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A repair fn that returns ``success=False`` is routed with the canonical kind."""
    config = _make_config(tmp_path)

    from polylogue.maintenance import replay as replay_mod
    from polylogue.maintenance.models import MaintenanceCategory
    from polylogue.storage.repair import RepairResult

    def _failing_repair(_config: Config, _dry_run: bool) -> RepairResult:
        return RepairResult(
            name="session_insights",
            category=MaintenanceCategory.DERIVED_REPAIR,
            destructive=False,
            repaired_count=0,
            success=False,
            detail="insight repair could not converge",
        )

    monkeypatch.setitem(replay_mod._REPLAY_DISPATCH, "session_insights", _failing_repair)

    operation = execute_replay(
        config,
        targets=["session_insights"],
        operation_id="op-soft",
        persist_state=False,
    )

    assert operation.status.value == "failed"
    persisted = read_maintenance_failures(Path(config.archive_root))
    assert len(persisted) == 1
    assert persisted[0].kind == "RepairReportedFailure"
    assert "could not converge" in persisted[0].message
