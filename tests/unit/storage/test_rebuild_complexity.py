"""Counter-based complexity, fairness, and resumability laws for raw rebuilds."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.sources import revision_backfill
from polylogue.storage import repair as repair_mod
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.growth_budgets import GrowthBudget, GrowthObservation, evaluate_growth_budgets
from tests.infra.sqlite_work_counter import sqlite_work_counter

_COMPONENT_DERIVED_WORK_BUDGET = GrowthBudget(metric="component_derived_vm_steps", max_step_multiplier=4.0)
# A terminal refresh is allowed to emit the bounded quartet's observed
# archive-wide statement envelope once per production-route pass. The law
# rejects paying that envelope once per selected component.
_COMPONENT_TERMINAL_REFRESH_STATEMENT_BUDGET = 9


def _config(root: Path) -> Config:
    return Config(archive_root=root, render_root=root, sources=[], db_path=root / "archive.db")


def _tool_call_payload(native_id: str) -> bytes:
    rows = [
        {"type": "session_meta", "payload": {"id": native_id, "timestamp": "2026-07-16T10:00:00Z"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": f"{native_id}-message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"run {native_id}"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "id": f"{native_id}-call",
                "call_id": f"{native_id}-call-id",
                "name": "exec_command",
                "arguments": json.dumps({"cmd": "printf hello"}),
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": f"{native_id}-call-id",
                "output": "hello",
            },
        },
    ]
    return b"".join(json.dumps(row, separators=(",", ":")).encode() + b"\n" for row in rows)


def _seed_raw_archive(root: Path, count: int, *, prefix: str = "session") -> list[str]:
    initialize_active_archive_root(root)
    raw_ids: list[str] = []
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for index in range(count):
            native_id = f"{prefix}-{index}"
            raw_ids.append(
                archive.write_raw_payload(
                    provider=Provider.CODEX,
                    payload=_tool_call_payload(native_id),
                    source_path=f"{native_id}.jsonl",
                    acquired_at_ms=index + 1,
                )
            )
    return raw_ids


def _materialize_all(root: Path, raw_count: int) -> None:
    result = repair_mod.repair_raw_materialization(_config(root), raw_artifact_limit=raw_count)
    assert result.success is True
    assert result.repaired_count == raw_count


def _quiesce_census(root: Path, *, limit: int) -> None:
    config = _config(root)
    for _ in range(100):
        result = repair_mod.repair_raw_materialization(config, dry_run=True, raw_artifact_limit=limit)
        assert result.census_receipt is not None
        if result.census_receipt.quiescent:
            return
    raise AssertionError("bounded census did not reach a fixed point")


def _run_component_measurement(
    tmp_path: Path,
    archive_size: int,
    monkeypatch: pytest.MonkeyPatch,
    *,
    component_count: int,
    install_mutation: Callable[[pytest.MonkeyPatch], None] | None = None,
) -> GrowthObservation:
    if component_count < 1 or component_count > archive_size:
        raise ValueError("component_count must be between one and archive_size")
    root = tmp_path / f"component-{archive_size}"
    _seed_raw_archive(root, archive_size, prefix="existing")
    _materialize_all(root, archive_size)

    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for index in range(component_count):
            native_id = f"target-{index}"
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=_tool_call_payload(native_id),
                source_path=f"{native_id}.jsonl",
                acquired_at_ms=archive_size + index + 1,
            )

    with monkeypatch.context() as mutation:
        if install_mutation is not None:
            install_mutation(mutation)
        with sqlite_work_counter(step_interval=1) as counter:
            result = repair_mod.repair_raw_materialization(_config(root), raw_artifact_limit=component_count)

    assert result.repaired_count == component_count
    return GrowthObservation(
        tier=str(archive_size),
        size=archive_size,
        metrics={
            "component_derived_vm_steps": float(counter.metric("derived_vm_steps")),
            "archive_wide_derived_statements": float(counter.metric("archive_wide_derived_statements")),
            "component_rows_scanned": result.metrics["raw_materialization_scanned_raw_count"],
            "component_rows_written": result.metrics["raw_materialization_replayed_logical_source_count"],
            "component_bytes": result.metrics["raw_materialization_selected_total_blob_bytes"],
            "component_passes": result.metrics["raw_materialization_executed_count"],
            "selected_component_count": float(component_count),
        },
    )


def _assert_component_shape(observations: list[GrowthObservation]) -> None:
    report = evaluate_growth_budgets(observations, [_COMPONENT_DERIVED_WORK_BUDGET])
    measured = "\n".join(f"  {observation.tier}: {dict(observation.metrics)}" for observation in observations)
    assert report.ok, (
        "component derived work exceeded the scale bound "
        f"{_COMPONENT_DERIVED_WORK_BUDGET.max_step_multiplier}x; "
        f"violations={report.violations}; measured counters:\n{measured}"
    )
    archive_wide = [observation.metric("archive_wide_derived_statements") for observation in observations]
    assert all(value <= _COMPONENT_TERMINAL_REFRESH_STATEMENT_BUDGET for value in archive_wide), (
        "incremental component route exceeded the one-terminal-refresh envelope; "
        f"declared statement budget={_COMPONENT_TERMINAL_REFRESH_STATEMENT_BUDGET}; measured counters:\n{measured}"
    )
    assert any(observation.metric("component_derived_vm_steps") > 0 for observation in observations), (
        f"production route reported no derived work; measured counters:\n{measured}"
    )


def _restore_deleted_archive_wide_refresh(mutation: pytest.MonkeyPatch) -> None:
    """Restore the deleted qsagp quartet through the real backfill seam."""
    from polylogue.storage.fts import fts_lifecycle
    from polylogue.storage.sqlite import action_pairs, delegation_facts

    original_backfill = revision_backfill.backfill_historical_revision_evidence

    def replay_with_deleted_regression(*args: Any, **kwargs: Any) -> Any:
        result = original_backfill(*args, **kwargs)
        archive_root = Path(args[0])
        with sqlite3.connect(archive_root / "index.db") as conn:
            conn.execute("PRAGMA busy_timeout = 600000")
            fts_lifecycle.rebuild_fts_index_sync(conn)
            fts_lifecycle.rebuild_command_trigram_index_sync(conn)
            action_pairs.rebuild_all_action_pairs_sync(conn)
            delegation_facts.rebuild_all_delegation_facts_sync(conn)
            conn.commit()
        return result

    mutation.setattr(revision_backfill, "backfill_historical_revision_evidence", replay_with_deleted_regression)


def test_one_component_derived_work_is_archive_scale_stable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observations = [
        _run_component_measurement(
            tmp_path,
            archive_size,
            monkeypatch,
            component_count=component_count,
        )
        for archive_size, component_count in ((2, 1), (8, 2), (32, 4))
    ]

    _assert_component_shape(observations)


def test_component_law_rejects_qsagp_archive_wide_per_item_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The law must turn red when the deleted archive-wide quartet is restored."""
    observations = [
        _run_component_measurement(
            tmp_path,
            archive_size,
            monkeypatch,
            component_count=component_count,
            install_mutation=_restore_deleted_archive_wide_refresh,
        )
        for archive_size, component_count in ((2, 1), (8, 2), (32, 4))
    ]

    with pytest.raises(AssertionError):
        _assert_component_shape(observations)
    assert all(observation.metric("archive_wide_derived_statements") > 0 for observation in observations), (
        "red mutation did not reach observed archive-wide SQL; "
        f"measured counters={[dict(observation.metrics) for observation in observations]}"
    )


def test_bounded_replay_work_is_batch_bounded_independent_of_backlog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_size = 2
    observed: list[tuple[int, int, int, int]] = []
    for archive_size in (4, 16, 64):
        root = tmp_path / f"batch-{archive_size}"
        _seed_raw_archive(root, archive_size, prefix="batch")
        _quiesce_census(root, limit=batch_size)
        selected_work = 0
        original_backfill = revision_backfill.backfill_historical_revision_evidence

        def counted_backfill(*args: Any, _original: Any = original_backfill, **kwargs: Any) -> Any:
            nonlocal selected_work
            result = _original(*args, **kwargs)
            selected_work += result.scanned
            return result

        with monkeypatch.context() as mutation:
            mutation.setattr(revision_backfill, "backfill_historical_revision_evidence", counted_backfill)
            result = repair_mod.repair_raw_materialization(_config(root), raw_artifact_limit=batch_size)

        assert result.metrics["raw_materialization_executed_count"] == float(batch_size)
        assert result.metrics["raw_materialization_scanned_raw_count"] <= float(batch_size)
        observed.append(
            (
                archive_size,
                selected_work,
                result.repaired_count,
                int(result.metrics["raw_materialization_executed_count"]),
            )
        )

    assert observed == [(4, 2, 2, 2), (16, 2, 2, 2), (64, 2, 2, 2)], (
        f"bounded replay exceeded batch bound={batch_size}; observed={observed}"
    )


def test_mixed_hot_cold_large_small_components_all_receive_a_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "fair-mixed"
    raw_ids = _seed_raw_archive(root, 4, prefix="mixed")
    large_raw_id = raw_ids[2]
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET blob_size = ? WHERE raw_id = ?",
            (repair_mod.RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES // 2, large_raw_id),
        )
        conn.commit()
    _quiesce_census(root, limit=1)

    original_backfill = revision_backfill.backfill_historical_revision_evidence
    attempted = Counter[str]()

    def fail_hot_component(*args: Any, **kwargs: Any) -> Any:
        selected_raw_ids = kwargs.get("selected_raw_ids")
        assert isinstance(selected_raw_ids, list)
        attempted[selected_raw_ids[0]] += 1
        if selected_raw_ids == [raw_ids[0]]:
            raise RuntimeError("injected hot component retry")
        return original_backfill(*args, **kwargs)

    with monkeypatch.context() as mutation:
        mutation.setattr(revision_backfill, "backfill_historical_revision_evidence", fail_hot_component)
        selected: list[str] = []
        for _ in range(3):
            result = repair_mod.repair_raw_materialization(_config(root), raw_artifact_limit=1)
            assert len(result.plan_outcomes) == 1
            selected.append(result.plan_outcomes[0].input_raw_ids[0])

    assert selected[0] == raw_ids[0]
    assert set(selected[1:]) == {raw_ids[1], raw_ids[2]}
    assert attempted[raw_ids[0]] == 1
    assert attempted[raw_ids[1]] == 1
    assert attempted[raw_ids[2]] == 1


def test_progress_counter_is_monotonic_and_resumable_across_bounded_passes(tmp_path: Path) -> None:
    root = tmp_path / "resumable"
    raw_count = 7
    _seed_raw_archive(root, raw_count, prefix="resume")
    _quiesce_census(root, limit=2)

    remaining: list[int] = []
    repaired: list[int] = []
    selected: list[str] = []
    config = _config(root)
    for _ in range(4):
        result = repair_mod.repair_raw_materialization(config, raw_artifact_limit=2)
        remaining.append(int(result.metrics["raw_materialization_remaining_candidate_count"]))
        repaired.append(result.repaired_count)
        selected.extend(
            outcome.input_raw_ids[0] for outcome in result.plan_outcomes if outcome.status.value == "executed"
        )
        if remaining[-1] == 0:
            break

    assert remaining == [5, 3, 1, 0]
    assert repaired == [2, 2, 2, 1]
    assert len(selected) == raw_count
    assert len(set(selected)) == raw_count
    assert repair_mod.repair_raw_materialization(config, raw_artifact_limit=2).success is True
