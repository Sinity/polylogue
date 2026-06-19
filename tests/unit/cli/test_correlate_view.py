"""CLI tests for git/GitHub correlation via ``read --view correlation``.

The standalone ``correlate`` command was absorbed into the read-view surface
(#1842): ``find <seed> then read --view correlation``. The correlation logic
lives in ``polylogue.insights.correlation_view.run_correlation_view`` and
``polylogue.insights.session_commit``; the MCP ``correlate_session(s)`` tools
expose the same capability programmatically.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.core.refs import ObjectRef
from polylogue.insights.correlation_view import run_correlation_view
from polylogue.insights.session_commit import GitHubRef, SessionCommitEdge, SessionCorrelationResult


def _session() -> SimpleNamespace:
    return SimpleNamespace(
        created_at=datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 4, 22, 14, 0, tzinfo=timezone.utc),
        git_repository_url="https://github.com/Sinity/polylogue",
        working_directories=(),
        messages=[],
    )


def _result() -> SessionCorrelationResult:
    return SessionCorrelationResult(
        session_id="target",
        window_start="2026-04-22T10:00:00+00:00",
        window_end="2026-04-22T16:00:00+00:00",
        repo="https://github.com/Sinity/polylogue",
        commits=[
            SessionCommitEdge(
                session_id="target",
                commit_sha="abc123456789",
                detection_method="file_overlap",
                confidence=0.8,
                file_overlap_count=2,
            )
        ],
        issue_refs=[GitHubRef(owner="Sinity", repo="polylogue", number=1845, kind="issue", raw_match="#1845")],
        pr_refs=[GitHubRef(owner="Sinity", repo="polylogue", number=2149, kind="pr", raw_match="#2149")],
        file_paths=["polylogue/core/refs.py"],
    )


def test_run_correlation_view_json_emits_payload() -> None:
    env = MagicMock()
    env.polylogue.get_session = AsyncMock(return_value=_session())

    with patch("polylogue.insights.session_commit.build_correlation_result", return_value=_result()):
        run_correlation_view(env, session_id="target", output_format="json", github_api=False)

    # JSON output is printed through the console; capture the printed string.
    printed = "".join(str(call.args[0]) for call in env.ui.console.print.call_args_list if call.args)
    payload = json.loads(printed)
    assert payload["session_id"] == "target"
    assert payload["object_refs"] == [
        "commit:abc123456789",
        "github-issue:Sinity/polylogue#1845",
        "github-pr:Sinity/polylogue#2149",
        "file:polylogue/core/refs.py",
    ]
    assert all(ObjectRef.parse(ref).format() == ref for ref in payload["object_refs"])


def test_run_correlation_view_missing_session_exits() -> None:
    env = MagicMock()
    env.polylogue.get_session = AsyncMock(return_value=None)

    with pytest.raises(SystemExit):
        run_correlation_view(env, session_id="nope", github_api=False)

    env.ui.error.assert_called_once()


def test_run_correlation_view_plain_renders_window() -> None:
    env = MagicMock()
    env.polylogue.get_session = AsyncMock(return_value=_session())

    with patch("polylogue.insights.session_commit.build_correlation_result", return_value=_result()):
        run_correlation_view(env, session_id="target", github_api=False)

    rendered = " ".join(str(call.args[0]) for call in env.ui.console.print.call_args_list if call.args)
    assert "target" in rendered
    assert "Commits" in rendered or "No commits" in rendered
