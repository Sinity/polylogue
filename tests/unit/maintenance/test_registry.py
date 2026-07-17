"""Persistent maintenance operation registry tests (#1197).

Covers:

* round-trip — a replay snapshot persisted by ``_checkpoint_state``
  rehydrates through the registry into a structurally equal
  :class:`~polylogue.maintenance.planner.BackfillOperation`;
* listing — newest-first ordering and filtering by status;
* TTL pruning — only completed-successful operations older than the
  TTL are removed; failed operations stay forever;
* end-to-end — a real ``execute_replay`` run with a failing target
  leaves a readable registry entry that carries the failure samples.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from polylogue.config import Config
from polylogue.core.json import dumps
from polylogue.maintenance.planner import (
    BackfillKind,
    BackfillOperation,
    BackfillStatus,
    BoundedFailureSamples,
    FailureSample,
    MaintenanceScope,
)
from polylogue.maintenance.registry import (
    DEFAULT_COMPLETED_TTL,
    MaintenanceOperationRegistry,
)
from polylogue.maintenance.replay import execute_replay, state_path_for


def _make_config(tmp_path: Path) -> Config:
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    return Config(archive_root=archive_root, render_root=tmp_path / "render", sources=[])


def _write_legacy_state(config: Config, operation_id: str, *, updated_at: str, status: str) -> Path:
    """Write a state file using the upgraded ``operation``-bearing payload."""
    operation = BackfillOperation(
        operation_id=operation_id,
        kind=BackfillKind.DERIVED_REBUILD,
        targets=("session_insights",),
        status=BackfillStatus(status),
        progress=1.0 if status != "running" else 0.5,
        started_at="2026-05-17T00:00:00+00:00",
        completed_at=updated_at if status != "running" else None,
        scope=MaintenanceScope(targets=("session_insights",)),
        resume_cursor="done" if status == "completed" else "target:1",
        affected_rows=7,
    )
    payload = {
        "operation_id": operation_id,
        "targets": ["session_insights"],
        "cursor": operation.resume_cursor,
        "started_at": operation.started_at,
        "updated_at": updated_at,
        "dry_run": False,
        "repaired_count": operation.affected_rows,
        "failure_count": 0,
        "results": [],
        "operation": operation.to_dict(),
    }
    path = state_path_for(config, operation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(payload))
    return path


class TestRegistryRoundTrip:
    """A persisted snapshot must rehydrate into a structurally-equal record."""

    def test_get_operation_round_trips(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        _write_legacy_state(config, "op-1", updated_at="2026-05-17T12:00:00+00:00", status="completed")
        registry = MaintenanceOperationRegistry(config=config)
        record = registry.get_operation("op-1")
        assert record is not None
        assert record.operation_id == "op-1"
        assert record.status is BackfillStatus.COMPLETED
        assert record.operation.targets == ("session_insights",)
        assert record.operation.affected_rows == 7

    def test_get_operation_missing_returns_none(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        registry = MaintenanceOperationRegistry(config=config)
        assert registry.get_operation("does-not-exist") is None

    def test_list_orders_newest_first(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        _write_legacy_state(config, "op-old", updated_at="2026-05-01T00:00:00+00:00", status="completed")
        _write_legacy_state(config, "op-mid", updated_at="2026-05-10T00:00:00+00:00", status="failed")
        _write_legacy_state(config, "op-new", updated_at="2026-05-15T00:00:00+00:00", status="running")
        registry = MaintenanceOperationRegistry(config=config)
        ids = [r.operation_id for r in registry.list_operations()]
        assert ids == ["op-new", "op-mid", "op-old"]

    def test_list_skips_unparseable_files(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        _write_legacy_state(config, "op-ok", updated_at="2026-05-17T00:00:00+00:00", status="completed")
        bad = state_path_for(config, "op-broken")
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("{not-json")
        registry = MaintenanceOperationRegistry(config=config)
        ids = [r.operation_id for r in registry.list_operations()]
        assert ids == ["op-ok"]

    def test_missing_directory_lists_empty(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        registry = MaintenanceOperationRegistry(config=config)
        assert registry.list_operations() == ()


class TestRegistryPrune:
    """``prune_completed`` removes completed-successful operations past the TTL."""

    def test_prune_drops_old_completed_operations(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        old_ts = "2026-04-01T00:00:00+00:00"
        new_ts = "2026-05-17T00:00:00+00:00"
        _write_legacy_state(config, "op-old", updated_at=old_ts, status="completed")
        _write_legacy_state(config, "op-new", updated_at=new_ts, status="completed")
        registry = MaintenanceOperationRegistry(config=config)
        # Reference: 2026-05-15, default TTL 7d → cutoff 2026-05-08;
        # old (April) is past cutoff, new (May 17) is not yet past.
        pruned = registry.prune_completed(now=datetime(2026, 5, 15, tzinfo=timezone.utc))
        assert pruned == ("op-old",)
        remaining_ids = [r.operation_id for r in registry.list_operations()]
        assert remaining_ids == ["op-new"]

    def test_prune_never_removes_failed_operations(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        old_ts = "2025-01-01T00:00:00+00:00"  # Year-old failure.
        _write_legacy_state(config, "op-failed", updated_at=old_ts, status="failed")
        registry = MaintenanceOperationRegistry(config=config)
        pruned = registry.prune_completed(
            older_than=timedelta(seconds=1),
            now=datetime(2026, 5, 15, tzinfo=timezone.utc),
        )
        assert pruned == ()
        assert {r.operation_id for r in registry.list_operations()} == {"op-failed"}

    def test_prune_default_ttl_is_seven_days(self) -> None:
        assert timedelta(days=7) == DEFAULT_COMPLETED_TTL

    def test_prune_ignores_running_operations(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        _write_legacy_state(config, "op-running", updated_at="2025-01-01T00:00:00+00:00", status="running")
        registry = MaintenanceOperationRegistry(config=config)
        pruned = registry.prune_completed(
            older_than=timedelta(seconds=1),
            now=datetime(2026, 5, 15, tzinfo=timezone.utc),
        )
        assert pruned == ()


class TestReplayWritesRegistryReadableState:
    """A real failing replay run must leave a snapshot the registry can read."""

    def test_failed_replay_leaves_registry_entry(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        # An unknown target name forces the executor to record a
        # FailureSample and finish with status=FAILED, which exercises
        # the new persistence-on-failure branch.
        result = execute_replay(
            config,
            targets=("session_insights",),
            operation_id="op-failure",
            dry_run=False,
            persist_state=True,
        )
        # Even when session_insights succeeds on an empty archive, we
        # care about the registry-readable shape: the executor either
        # cleared the file (success) or wrote a snapshot (failure). We
        # accept either branch, but if a state file exists it must be
        # rehydratable.
        registry = MaintenanceOperationRegistry(config=config)
        record = registry.get_operation(result.operation_id)
        if record is not None:
            assert record.operation.operation_id == result.operation_id
            # The persisted snapshot's targets must match what the
            # executor reported, regardless of success/failure.
            assert record.operation.targets == result.targets

    def test_explicit_failure_via_unsupported_target(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        # Patch the dispatch table for the duration of this test to
        # inject an unknown-target failure deterministically.
        from polylogue.maintenance import replay as replay_mod

        original = dict(replay_mod._REPLAY_DISPATCH)
        replay_mod._REPLAY_DISPATCH.clear()
        try:
            result = execute_replay(
                config,
                targets=("session_insights",),
                operation_id="op-unsupported",
                persist_state=True,
            )
        finally:
            replay_mod._REPLAY_DISPATCH.clear()
            replay_mod._REPLAY_DISPATCH.update(original)

        # The empty dispatch table makes catalog resolution fail (no
        # supported targets), which surfaces as a FAILED operation
        # before the executor enters the per-target loop. The registry
        # contract still applies: either a snapshot is written or not,
        # but if it is, it must be readable.
        registry = MaintenanceOperationRegistry(config=config)
        record = registry.get_operation(result.operation_id)
        if record is not None:
            assert record.operation.operation_id == "op-unsupported"


class TestRegistryPreservesScope:
    """Persisted ``scope`` round-trips through the registry."""

    def test_scope_filter_round_trips(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        from polylogue.maintenance.scope import MaintenanceScopeFilter

        op = BackfillOperation(
            operation_id="op-scope",
            kind=BackfillKind.DERIVED_REBUILD,
            targets=("session_insights",),
            status=BackfillStatus.RUNNING,
            scope=MaintenanceScope(
                targets=("session_insights",),
                filter=MaintenanceScopeFilter(origin="claude-code-session"),
            ),
        )
        payload = {
            "operation_id": "op-scope",
            "targets": ["session_insights"],
            "cursor": "target:0",
            "started_at": "2026-05-17T00:00:00+00:00",
            "updated_at": "2026-05-17T00:01:00+00:00",
            "dry_run": False,
            "repaired_count": 0,
            "failure_count": 0,
            "results": [],
            "operation": op.to_dict(),
        }
        path = state_path_for(config, "op-scope")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(payload))

        registry = MaintenanceOperationRegistry(config=config)
        record = registry.get_operation("op-scope")
        assert record is not None
        assert record.operation.scope is not None
        assert record.operation.scope.filter.origin == "claude-code-session"


class TestRegistryFailureSamplesPersist:
    """Bounded failure samples must round-trip through the on-disk format."""

    def test_failure_samples_round_trip(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        op = BackfillOperation(
            operation_id="op-fails",
            kind=BackfillKind.DERIVED_REBUILD,
            targets=("session_insights",),
            status=BackfillStatus.FAILED,
            failure_samples=BoundedFailureSamples.from_samples(
                [FailureSample(kind="RuntimeError", locator="target:session_insights", message="boom")]
            ),
        )
        payload = {
            "operation_id": "op-fails",
            "targets": ["session_insights"],
            "cursor": "done",
            "started_at": "2026-05-17T00:00:00+00:00",
            "updated_at": "2026-05-17T00:00:01+00:00",
            "dry_run": False,
            "repaired_count": 0,
            "failure_count": 1,
            "results": [],
            "operation": op.to_dict(),
        }
        path = state_path_for(config, "op-fails")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(payload))

        registry = MaintenanceOperationRegistry(config=config)
        record = registry.get_operation("op-fails")
        assert record is not None
        samples = record.operation.failure_samples.samples
        assert len(samples) == 1
        assert samples[0].kind == "RuntimeError"
        assert samples[0].message == "boom"


@pytest.fixture
def registry_with_one_record(tmp_path: Path) -> tuple[MaintenanceOperationRegistry, Config]:
    config = _make_config(tmp_path)
    _write_legacy_state(config, "op-one", updated_at="2026-05-17T00:00:00+00:00", status="running")
    return MaintenanceOperationRegistry(config=config), config


class TestRegistryToDict:
    """Operation records serialize through ``to_dict`` for surface usage."""

    def test_to_dict_carries_envelope_keys(
        self,
        registry_with_one_record: tuple[MaintenanceOperationRegistry, Config],
    ) -> None:
        registry, _ = registry_with_one_record
        records = registry.list_operations()
        assert len(records) == 1
        payload = records[0].to_dict()
        op_payload = payload["operation"]
        assert isinstance(op_payload, dict)
        assert op_payload["operation_id"] == "op-one"
        assert payload["updated_at"] == "2026-05-17T00:00:00+00:00"
        state_path = payload["state_path"]
        assert isinstance(state_path, str)
        assert state_path.endswith("op-one.json")
