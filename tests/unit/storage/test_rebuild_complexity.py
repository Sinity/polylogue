"""Counter-based complexity, fairness, and resumability laws for raw rebuilds."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from polylogue.core.enums import Provider
from polylogue.daemon.derivation import Budget, DerivationRegistry, DerivationReport, PassCursor, converge
from polylogue.operations.raw_observation_derivation import raw_observation_frame
from polylogue.sources import revision_backfill
from polylogue.storage.derived.raw import RawObservationDerivation
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.growth_budgets import GrowthBudget, GrowthObservation, evaluate_growth_budgets
from tests.infra.sqlite_work_counter import sqlite_work_counter

_COMPONENT_DERIVED_WORK_BUDGET = GrowthBudget(metric="component_derived_vm_steps", max_step_multiplier=4.0)
# An incremental component replay must not emit archive-wide derived writes.
_COMPONENT_TERMINAL_REFRESH_STATEMENT_BUDGET = 9


def _run(
    root: Path, *, limit: int, raw_ids: tuple[str, ...] = (), cursor: PassCursor | None = None
) -> DerivationReport:
    return converge(
        DerivationRegistry((RawObservationDerivation(root),)),
        raw_observation_frame(root, raw_ids=raw_ids),
        budget=Budget(page=limit, discovery=limit, inspection=2 * limit, compute=limit, publication=limit),
        cursor=cursor,
    )


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
    result = _run(root, limit=raw_count)
    assert result.failed == result.pending == 0
    assert result.done == raw_count


def _run_component_measurement(
    tmp_path: Path,
    archive_size: int,
    *,
    component_count: int,
) -> GrowthObservation:
    if component_count < 1 or component_count > archive_size:
        raise ValueError("component_count must be between one and archive_size")
    root = tmp_path / f"component-{archive_size}"
    _seed_raw_archive(root, archive_size, prefix="existing")
    _materialize_all(root, archive_size)

    selected: list[str] = []
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        for index in range(component_count):
            native_id = f"target-{index}"
            selected.append(
                archive.write_raw_payload(
                    provider=Provider.CODEX,
                    payload=_tool_call_payload(native_id),
                    source_path=f"{native_id}.jsonl",
                    acquired_at_ms=archive_size + index + 1,
                )
            )

    with sqlite_work_counter(step_interval=1) as counter:
        result = _run(root, limit=component_count, raw_ids=tuple(selected))

    assert result.done == component_count
    return GrowthObservation(
        tier=str(archive_size),
        size=archive_size,
        metrics={
            "component_derived_vm_steps": float(counter.metric("derived_vm_steps")),
            "archive_wide_derived_statements": float(counter.metric("archive_wide_derived_statements")),
            "component_rows_scanned": float(result.work.discovered),
            "component_rows_written": float(result.work.published),
            "component_bytes": float(sum(len(_tool_call_payload(f"target-{i}")) for i in range(component_count))),
            "component_passes": float(result.done),
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


def test_one_component_derived_work_is_archive_scale_stable(tmp_path: Path) -> None:
    observations = [
        _run_component_measurement(
            tmp_path,
            archive_size,
            component_count=component_count,
        )
        for archive_size, component_count in ((2, 1), (8, 2), (32, 4))
    ]

    _assert_component_shape(observations)


def test_bounded_replay_work_is_batch_bounded_independent_of_backlog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_size = 2
    observed: list[tuple[int, int, int, int]] = []
    for archive_size in (4, 16, 64):
        root = tmp_path / f"batch-{archive_size}"
        _seed_raw_archive(root, archive_size, prefix="batch")
        selected_work = 0
        original_backfill = revision_backfill.backfill_historical_revision_evidence

        def counted_backfill(*args: Any, _original: Any = original_backfill, **kwargs: Any) -> Any:
            nonlocal selected_work
            result = _original(*args, **kwargs)
            selected_work += result.scanned
            return result

        with monkeypatch.context() as mutation:
            mutation.setattr(revision_backfill, "backfill_historical_revision_evidence", counted_backfill)
            result = _run(root, limit=batch_size)

        assert result.done == batch_size
        assert result.work.discovered <= batch_size
        observed.append(
            (
                archive_size,
                selected_work,
                result.done,
                result.work.published,
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
            (512 * 1024 * 1024, large_raw_id),
        )
        conn.commit()

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
        cursor = None
        for _ in range(4):
            result = _run(root, limit=1, cursor=cursor)
            cursor = result.cursor
            assert result.work.discovered == 1

    assert set(attempted) == set(raw_ids)
    assert all(count == 1 for count in attempted.values())


def test_progress_counter_is_monotonic_and_resumable_across_bounded_passes(tmp_path: Path) -> None:
    root = tmp_path / "resumable"
    raw_count = 7
    _seed_raw_archive(root, raw_count, prefix="resume")

    remaining: list[int] = []
    repaired: list[int] = []
    selected: list[str] = []
    cursor = None
    for _ in range(4):
        result = _run(root, limit=2, cursor=cursor)
        cursor = result.cursor
        with sqlite3.connect(root / "index.db") as conn:
            remaining.append(raw_count - conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
        repaired.append(result.done)
        selected.extend(outcome.key.key for outcome in result.outcomes if outcome.outcome.value == "done")
        if remaining[-1] == 0:
            break

    assert remaining == [5, 3, 1, 0]
    assert repaired == [2, 2, 2, 1]
    assert len(selected) == raw_count
    assert len(set(selected)) == raw_count
    assert _run(root, limit=2).work.published == 0
