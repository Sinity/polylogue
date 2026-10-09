"""Resident rebuild lifetime follows accepted custody and explicit caller policy."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from time import monotonic as monotonic_clock
from time import time as wall_clock

import pytest

from polylogue.daemon import operation_runtime
from polylogue.operations import daemon_insights
from polylogue.operations.insight_acceptance import AcceptedInsightPart
from tests.infra.daemon_operations import running_daemon_operations


@pytest.mark.parametrize("explicit_deadline", [False, True])
def test_resident_rebuild_past_five_minutes_keeps_only_implicit_work_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_deadline: bool
) -> None:
    """Restoring the implicit 300s deadline leaves the positive target unpublished."""
    offset = [0.0]
    original_time = wall_clock
    original_monotonic = monotonic_clock
    original_begin = daemon_insights.InsightExecution.begin
    original_part = daemon_insights.InsightExecution.part
    failures: list[str] = []

    def observe_part(execution: daemon_insights.InsightExecution, ordinal: int) -> AcceptedInsightPart:
        try:
            return original_part(execution, ordinal)
        except BaseException as exc:
            failures.append(f"part {type(exc).__name__}: {exc}")
            raise

    async def advance_before_begin(execution: daemon_insights.InsightExecution, part: AcceptedInsightPart) -> object:
        offset[0] = 600.0
        try:
            return await original_begin(execution, part)
        except BaseException as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
            raise

    monkeypatch.setattr(operation_runtime, "time", lambda: original_time() + offset[0])
    monkeypatch.setattr(operation_runtime, "monotonic", lambda: original_monotonic() + offset[0])
    monkeypatch.setattr(daemon_insights, "time", lambda: original_time() + offset[0])
    monkeypatch.setattr(time, "time", lambda: original_time() + offset[0])
    monkeypatch.setattr(daemon_insights.InsightExecution, "begin", advance_before_begin)
    monkeypatch.setattr(daemon_insights.InsightExecution, "part", observe_part)

    def seed(root: Path) -> None:
        with sqlite3.connect(root / "index.db") as conn:
            conn.execute(
                "INSERT INTO sessions (native_id, origin, content_hash) VALUES ('lifetime', 'codex-session', zeroblob(32))"
            )

    root = tmp_path / "archive"
    with running_daemon_operations(root, seed_archive=seed, session_derivation=True) as stack:
        result = stack.client.operation(
            "maintenance.insights.rebuild",
            {"session_ids": ["codex-session:lifetime"]},
            archive_root=str(root),
            deadline_ms=300_000 if explicit_deadline else None,
        )
        if result is not None:
            result = stack.client.follow_operation("maintenance.insights.rebuild", result, archive_root=str(root))
    assert result is not None
    with sqlite3.connect(root / "audit.db") as conn:
        deadline, stop = conn.execute(
            "SELECT accepted_deadline_unix_ms, stop_reason FROM machine_requests WHERE operation_name='maintenance.insights.rebuild'"
        ).fetchone()
    with sqlite3.connect(root / "index.db") as conn:
        published = conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0]
    if explicit_deadline:
        assert deadline is not None
        assert stop == "deadline", (result, failures)
        assert published == 0
    else:
        assert result["outcome"] == "completed", result
        assert result["result"]["profiles"] == 1
        assert deadline is None and stop is None
        assert published == 1
