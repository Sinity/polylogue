"""CLI tests for context-preamble composition via ``read --view context``.

The standalone ``context compose`` command was absorbed into the read-view
surface (#1842): ``find <seed> then read --view context``. The compose logic
lives in ``polylogue.cli.commands.context.run_context_compose``; the MCP
``compose_context_preamble`` tool exposes the same capability programmatically.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from polylogue.cli.commands.context import run_context_compose


def _session() -> SimpleNamespace:
    return SimpleNamespace(
        git_repository_url="https://github.com/Sinity/polylogue",
        git_branch="master",
    )


def test_run_context_compose_emits_preamble() -> None:
    env = MagicMock()
    env.polylogue.get_session = AsyncMock(return_value=_session())
    env.polylogue.get_session_topology = AsyncMock(
        return_value=SimpleNamespace(logical_session_id="root-1", parent_session_id="parent-1")
    )
    env.polylogue.find_resume_candidates = AsyncMock(
        return_value=[SimpleNamespace(session_id="rel-1", title="Related", terminal_state="open")]
    )

    preamble = json.loads(run_context_compose(env, session_id="target", related_limit=3))

    assert preamble["preamble_version"] == "1.0"
    assert preamble["session_lineage"]["logical_session_root"] == "root-1"
    assert preamble["session_lineage"]["parent_session_id"] == "parent-1"
    assert preamble["recent_related_sessions"][0]["session_id"] == "rel-1"
    assert preamble["project_state"]["repo"] == "https://github.com/Sinity/polylogue"
    assert preamble["project_state"]["branch"] == "master"


def test_run_context_compose_missing_session_exits() -> None:
    env = MagicMock()
    env.polylogue.get_session = AsyncMock(return_value=None)

    with pytest.raises(SystemExit):
        run_context_compose(env, session_id="nope")

    env.ui.error.assert_called_once()
