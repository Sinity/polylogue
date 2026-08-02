"""End-to-end ``todo_snapshot`` admission + session-linked read model (polylogue-t0p).

Production dependencies exercised: configured source walk + OriginSpec path
classification (``_admit_non_session_origin_artifacts``), raw source/blob
persistence, and the ``claude_todo_projection`` read model built on top.

Anti-vacuity mutation: removing the ``todo_snapshot`` ``OriginArtifactRule``
(or its ``.json`` suffix) makes ``parse_sources_archive`` admit zero raw
rows for the todos fixture below, so ``load_claude_todo_plan_states`` would
return an empty tuple instead of two sessions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.insights.claude_todo_projection import load_claude_todo_plan_states
from polylogue.insights.measurement.registered_metrics import (
    DEFAULT_METRIC_REGISTRY,
    PLAN_COMPLETION_RATE_METRIC,
)
from polylogue.pipeline.services.archive_ingest import parse_sources_archive

_SESSION_A = "138e259e-435f-4259-8c68-dbd5aa9f9837"
_SESSION_B = "0092150d-6b81-43f9-85e2-ceeb2d1c8773"
_AGENT_B = "00afa7b1-f546-425e-95dc-d1a7093b09d2"


def _write_todos_fixture(root: Path) -> Path:
    todos = root / "todos"
    todos.mkdir(parents=True)
    (todos / f"{_SESSION_A}.json").write_text(
        json.dumps(
            [
                {"content": "bootstrap the repo", "status": "completed", "priority": "high", "id": "a1"},
                {"content": "write docs", "status": "in_progress", "priority": "medium", "id": "a2"},
            ]
        ),
        encoding="utf-8",
    )
    (todos / f"{_SESSION_B}-agent-{_AGENT_B}.json").write_text(
        json.dumps(
            [
                {"content": "subagent task", "status": "pending", "priority": "low", "id": "b1"},
            ]
        ),
        encoding="utf-8",
    )
    return todos


@pytest.mark.asyncio
async def test_todos_admitted_and_materialized_into_session_linked_plan_states(
    workspace_env: dict[str, Path],
) -> None:
    archive_root = workspace_env["archive_root"]
    todos_root = _write_todos_fixture(workspace_env["data_root"] / ".claude")

    result = await parse_sources_archive(
        archive_root,
        [Source(name="claude-code-todos", path=todos_root)],
    )
    assert result.parse_failures == 0

    plan_states = load_claude_todo_plan_states(archive_root)
    by_session = {state.session_id: state for state in plan_states}

    assert set(by_session) == {_SESSION_A, _SESSION_B}

    plan_a = by_session[_SESSION_A]
    assert plan_a.agent_id is None
    assert len(plan_a.snapshots) == 1
    assert plan_a.latest.snapshot.item_count == 2
    assert plan_a.plan_completion_rate == pytest.approx(0.5)
    assert plan_a.status_transitions() == {"a1": ("completed",), "a2": ("in_progress",)}

    plan_b = by_session[_SESSION_B]
    assert plan_b.agent_id == _AGENT_B
    assert plan_b.plan_completion_rate == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_second_observed_snapshot_extends_status_transition_history(
    workspace_env: dict[str, Path],
) -> None:
    """A later snapshot for the SAME session is a distinct raw revision, not an overwrite in place."""
    archive_root = workspace_env["archive_root"]
    todos_root = _write_todos_fixture(workspace_env["data_root"] / ".claude")

    await parse_sources_archive(archive_root, [Source(name="claude-code-todos", path=todos_root)])

    # Claude Code overwrites the same path on the next TodoWrite call; the
    # watcher observes this as a new file revision with an advanced status.
    (todos_root / f"{_SESSION_A}.json").write_text(
        json.dumps(
            [
                {"content": "bootstrap the repo", "status": "completed", "priority": "high", "id": "a1"},
                {"content": "write docs", "status": "completed", "priority": "medium", "id": "a2"},
            ]
        ),
        encoding="utf-8",
    )
    await parse_sources_archive(archive_root, [Source(name="claude-code-todos", path=todos_root)])

    plan_states = load_claude_todo_plan_states(archive_root)
    plan_a = next(state for state in plan_states if state.session_id == _SESSION_A)

    assert len(plan_a.snapshots) == 2
    assert plan_a.status_transitions()["a2"] == ("in_progress", "completed")
    assert plan_a.plan_completion_rate == pytest.approx(1.0)


def test_plan_completion_rate_metric_is_registered() -> None:
    """Plan-vs-outcome measure registration (polylogue-t0p AC), rxdo.9.1 identity layer.

    Mirrors the already-shipped ``session_cost_usd`` slice: proves the metric
    identity/registry machinery resolves this construct through the same
    process-wide registry, reachable via MCP ``get(ref="metric:plan_completion_rate")``.
    """
    resolved = DEFAULT_METRIC_REGISTRY.resolve("plan_completion_rate")
    assert resolved is not None
    assert resolved.ref == PLAN_COMPLETION_RATE_METRIC.ref
    assert resolved.measurement_authority == ("structural",)
    assert resolved.null_policy == "exclude"


def test_todos_source_name_resolves_to_claude_code_provider() -> None:
    """Admission gating (archive_ingest._admit_non_session_origin_artifacts) keys off this."""
    assert Provider.from_string("claude-code-todos") is Provider.CLAUDE_CODE
