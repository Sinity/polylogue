"""Semantic witness for the command-shape family decision."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from polylogue.analysis.command_shapes import (
    CommandShapeUsageQuery,
    build_command_shape_usage,
    normalize_command_shapes,
)


def _patch_scratch_creator(monkeypatch: pytest.MonkeyPatch, factory: type[Any]) -> None:
    """Create the fold's scratch on a measured subclass the law instruments.

    The fold opens scratch through ``scratch_connection_context``; patching its
    creator keeps the canonical custody owner and measured-creator stamps.
    """
    import os
    import sqlite3
    import threading

    from polylogue.storage import io_phase_metrics
    from polylogue.storage.sqlite import connection_profile

    def measured(database: str | Path, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn: io_phase_metrics._MeasuredConnection = sqlite3.connect(str(database), *args, factory=factory, **kwargs)
        conn._metric_tier = io_phase_metrics.tier_for_path(database)
        conn._native_creator = (os.getpid(), threading.current_thread())
        return conn

    monkeypatch.setattr(connection_profile, "connect_measured", measured)


def test_command_shape_library_preserves_shell_semantics_before_aggregation() -> None:
    """A SQL-style grouping would lose pipeline and wrapper boundaries."""

    command = "env CI=1 bash -lc 'pytest tests/unit --maxfail=1 | rg failed'"
    assert normalize_command_shapes(command) == ("pytest", "rg failed")

    rows = [
        {
            "origin": "codex",
            "repository": "polylogue",
            "session_id": "s1",
            "tool_command": command,
            "occurred_at_ms": 1_000,
        },
        {
            "origin": "codex",
            "repository": "polylogue",
            "session_id": "s2",
            "tool_command": "pytest tests/unit --maxfail=1",
            "occurred_at_ms": 2_000,
        },
    ]
    result = build_command_shape_usage(rows, CommandShapeUsageQuery(), materialized_at="now")

    assert [(item.command_shape, item.execution_count, item.session_count) for item in result] == [
        ("pytest", 2, 2),
        ("rg failed", 1, 1),
    ]
    assert all(item.provenance.materializer_version == 1 for item in result)


def test_streamed_fold_preserves_stage_multiplicity_and_pages_complete_totals() -> None:
    consumed = 0

    def rows() -> Iterator[dict[str, object]]:
        nonlocal consumed
        for index in range(1001):
            consumed += 1
            yield {
                "origin": "codex",
                "repository": None,
                "session_id": f"s{index % 3}",
                "tool_command": "foo bar | foo bar; other status",
                "occurred_at_ms": 0 if index == 0 else -1000,
            }

    result = build_command_shape_usage(rows(), CommandShapeUsageQuery(limit=1), materialized_at="now")
    assert consumed == 1001
    assert [(item.command_shape, item.execution_count, item.session_count) for item in result] == [("foo bar", 2002, 3)]
    assert result[0].repository is None
    assert result[0].last_used_at == "1970-01-01T00:00:00+00:00"
    following = build_command_shape_usage(rows(), CommandShapeUsageQuery(limit=None, offset=1), materialized_at="now")
    assert [(item.command_shape, item.execution_count, item.session_count) for item in following] == [
        ("other status", 1001, 3)
    ]


def test_fold_cancellation_preserves_identity_and_removes_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    import polylogue.analysis.command_shapes as module

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    failure = RuntimeError("injected cancellation")
    calls = 0

    def checkpoint() -> None:
        nonlocal calls
        calls += 1
        if calls == 6:
            raise failure

    rows = ({"origin": "codex", "session_id": "s", "tool_command": "foo"} for _ in range(100))
    with pytest.raises(RuntimeError) as caught:
        module.build_command_shape_usage(rows, CommandShapeUsageQuery(), materialized_at="now", checkpoint=checkpoint)
    assert caught.value is failure
    assert list(tmp_path.iterdir()) == []


def test_scratch_sql_cancellation_settles_connection_and_preserves_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3
    import tempfile

    import polylogue.analysis.command_shapes as module

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    aggregate_started = False
    closed = False
    failure = RuntimeError("injected SQL cancellation")

    class Cursor(sqlite3.Cursor):
        def execute(self, sql: str, parameters: Any = ()) -> Cursor:
            nonlocal aggregate_started
            if sql.startswith("SELECT origin"):
                aggregate_started = True
            return super().execute(sql, parameters)

    from polylogue.storage.io_phase_metrics import _MeasuredConnection

    class Connection(_MeasuredConnection):
        def cursor(self, factory: Any = None) -> Any:
            return super().cursor(Cursor if factory is None else factory)

        def close(self) -> None:
            nonlocal closed
            super().close()
            closed = True

    _patch_scratch_creator(monkeypatch, Connection)

    def checkpoint() -> None:
        if aggregate_started:
            raise failure

    rows = ({"origin": "codex", "session_id": f"s{i}", "tool_command": "foo"} for i in range(2000))
    with pytest.raises(RuntimeError) as caught:
        module.build_command_shape_usage(rows, CommandShapeUsageQuery(), materialized_at="now", checkpoint=checkpoint)
    assert caught.value is failure
    assert closed
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("same_failure", [False, True])
def test_scratch_cleanup_preserves_primary_and_distinct_faults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_failure: bool
) -> None:
    import tempfile

    import polylogue.analysis.command_shapes as module

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    primary = RuntimeError("injected fold failure")
    cleanup = primary if same_failure else RuntimeError("injected cleanup failure")
    closed = False

    from polylogue.storage.io_phase_metrics import _MeasuredConnection

    class Connection(_MeasuredConnection):
        def close(self) -> None:
            nonlocal closed
            super().close()
            closed = True
            raise cleanup

    _patch_scratch_creator(monkeypatch, Connection)

    def rows() -> Iterator[dict[str, object]]:
        yield {"origin": "codex", "session_id": "s", "tool_command": "foo"}
        raise primary

    from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError

    # The scratch custody owner reports a failed close as typed unsettled
    # custody that carries the close fault and chains the fold's own fault;
    # neither is dropped, and the scratch stays retained until SQL settles.
    with pytest.raises(NativeConnectionSettlementError) as caught:
        module.build_command_shape_usage(rows(), CommandShapeUsageQuery(), materialized_at="now")
    assert caught.value.failure is cleanup
    assert caught.value.__cause__ is primary
    assert closed
    assert [path.name.startswith("polylogue-command-shapes-") for path in tmp_path.iterdir()] == [True]


@pytest.mark.parametrize(("offset", "limit"), [(0, 10**100), (10**100, 1), (-1, None), (0, -1), (-2, 1), (1, -1)])
def test_scratch_page_preserves_declared_python_slice_operands(offset: int, limit: int | None) -> None:
    rows = ({"origin": "codex", "session_id": "s", "tool_command": shape} for shape in ("a", "b", "c"))
    result = build_command_shape_usage(rows, CommandShapeUsageQuery(offset=offset, limit=limit), materialized_at="now")
    stop = offset + limit if limit is not None else None
    assert [item.command_shape for item in result] == ["a", "b", "c"][offset:stop]
