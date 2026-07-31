"""CLI tests for context-preamble composition via ``read --view context``.

The standalone ``context compose`` command was absorbed into the read-view
surface (#1842): ``find <seed> then read --view context``. The compose logic
lives in ``polylogue.context.preamble.compose_context_preamble``; the MCP
``compose_context_preamble`` tool exposes the same capability programmatically.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.context.preamble import compose_context_preamble
from polylogue.core.enums import AssertionKind


def _session() -> SimpleNamespace:
    return SimpleNamespace(
        git_repository_url="https://github.com/Sinity/polylogue",
        git_branch="master",
    )


def _mock_subprocess_failure(**kwargs: object) -> MagicMock:
    """Simulate a ``cwd`` with no local git checkout (``git`` exits non-zero)."""
    return MagicMock(returncode=1, stdout="", stderr="")


def test_compose_context_preamble_emits_preamble() -> None:
    env = MagicMock()
    env.polylogue.get_session = AsyncMock(return_value=_session())
    env.polylogue.get_session_topology = AsyncMock(
        return_value=SimpleNamespace(logical_session_id="root-1", parent_session_id="parent-1")
    )
    env.polylogue.find_resume_candidates = AsyncMock(
        return_value=[SimpleNamespace(session_id="rel-1", title="Related", terminal_state="open")]
    )

    with patch("subprocess.run", side_effect=_mock_subprocess_failure):
        preamble = json.loads(compose_context_preamble(env, session_id="target", related_limit=3))

    assert preamble["preamble_version"] == "1.0"
    assert preamble["session_lineage"]["logical_session_root"] == "root-1"
    assert preamble["session_lineage"]["parent_session_id"] == "parent-1"
    assert preamble["recent_related_sessions"][0]["session_id"] == "rel-1"
    # No local git checkout at the composing cwd: falls back to the session's
    # own recorded repo/branch metadata.
    assert preamble["project_state"]["repo"] == "https://github.com/Sinity/polylogue"
    assert preamble["project_state"]["branch"] == "master"


def test_compose_context_preamble_git_enrichment_overrides_branch() -> None:
    """When the composing cwd is inside a live git checkout, its current
    branch + recent commits enrich the preamble project state (the same
    enrichment the MCP ``context(intent="resume")`` route applies) —
    superseding the session's possibly-stale recorded branch."""
    env = MagicMock()
    env.polylogue.get_session = AsyncMock(return_value=_session())
    env.polylogue.get_session_topology = AsyncMock(return_value=None)
    env.polylogue.find_resume_candidates = AsyncMock(return_value=[])

    def _git_side_effect(args: list[str], **kwargs: object) -> MagicMock:
        if "rev-parse" in args:
            return MagicMock(returncode=0, stdout="feature/live-branch\n", stderr="")
        if "log" in args:
            return MagicMock(returncode=0, stdout="abc1234 feat: something\n", stderr="")
        return MagicMock(returncode=1, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_git_side_effect):
        preamble = json.loads(compose_context_preamble(env, session_id="target", related_limit=3))

    assert preamble["project_state"]["repo"] == "https://github.com/Sinity/polylogue"
    assert preamble["project_state"]["branch"] == "feature/live-branch"
    assert preamble["project_state"]["recent_commits"] == ["abc1234 feat: something"]


def test_compose_context_preamble_includes_injectable_assertion_claims() -> None:
    env = MagicMock()
    env.polylogue.get_session = AsyncMock(return_value=_session())
    env.polylogue.get_session_topology = AsyncMock(return_value=None)
    env.polylogue.find_resume_candidates = AsyncMock(return_value=[])
    env.polylogue.list_assertion_claim_payloads = AsyncMock(
        return_value=[
            SimpleNamespace(
                kind=AssertionKind.DECISION,
                body_text="Keep context claims behind explicit injection.",
                target_ref="session:target",
                scope_ref="repo:polylogue",
                evidence_refs=("target::m1",),
            )
        ]
    )

    preamble = json.loads(compose_context_preamble(env, session_id="target", related_limit=3))

    env.polylogue.list_assertion_claim_payloads.assert_awaited_once_with(
        target_ref="session:target",
        statuses=("active",),
        context_inject=True,
        limit=20,
    )
    assert preamble["guidance"]["assertions"] == [
        {
            "kind": "decision",
            "trust_class": "quoted",
            "quoted_evidence": {
                "format": "quoted-assertion-evidence",
                "text": "Keep context claims behind explicit injection.",
            },
            "target_ref": "session:target",
            "scope_ref": "repo:polylogue",
            "evidence_refs": ["target::m1"],
        }
    ]


def test_compose_context_preamble_missing_session_exits() -> None:
    env = MagicMock()
    env.polylogue.get_session = AsyncMock(return_value=None)

    with pytest.raises(SystemExit):
        compose_context_preamble(env, session_id="nope")

    env.ui.error.assert_called_once()
