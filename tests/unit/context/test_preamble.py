"""Unit tests for the shared context-preamble git enrichment.

``build_context_preamble_payload`` (used by both the CLI ``read --view
context`` route and the MCP ``context(intent="resume")`` route) owns reading
the composing cwd's current branch + recent commits via ``_git_project_state``
(polylogue-t46.7) -- this used to be duplicated MCP-surface-only code in
``mcp/server_cutover.py``. These tests exercise the enrichment against a real
throwaway git repo rather than mocking ``subprocess.run``, since spinning one
up is cheap and it proves the actual git command lines parse real output.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from polylogue.context.preamble import _git_project_state, build_context_preamble_payload


def _init_git_repo(path: Path, *, branch: str = "main") -> None:
    subprocess.run(["git", "init", "--initial-branch", branch, str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True, capture_output=True
    )
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True, capture_output=True)
    (path / "README.md").write_text("hello\n")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial commit"],
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_DATE": "2026-01-01T00:00:00",
            "GIT_COMMITTER_DATE": "2026-01-01T00:00:00",
        },
    )


class TestGitProjectStateRealRepo:
    """``_git_project_state`` against a real tmp git checkout."""

    def test_reads_branch_and_commits(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path, branch="feature/preamble-move")

        state = _git_project_state(str(tmp_path))

        assert state is not None
        assert state.branch == "feature/preamble-move"
        assert len(state.recent_commits) == 1
        assert "initial commit" in state.recent_commits[0]

    def test_non_git_directory_returns_none(self, tmp_path: Path) -> None:
        state = _git_project_state(str(tmp_path))

        assert state is None

    def test_missing_directory_never_raises(self) -> None:
        state = _git_project_state(str(Path("/nonexistent/definitely-not-a-repo-path")))

        assert state is None


class TestBuildContextPreambleGitEnrichment:
    """``build_context_preamble_payload`` merges git enrichment for cwd."""

    @pytest.mark.asyncio
    async def test_cwd_git_state_populates_project_state(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path, branch="feature/enrich")

        poly = MagicMock()
        poly.get_session = AsyncMock(return_value=None)
        poly.get_session_topology = AsyncMock(return_value=None)
        poly.find_resume_candidates = AsyncMock(return_value=[])
        poly.list_assertion_claim_payloads = AsyncMock(return_value=[])

        preamble = await build_context_preamble_payload(
            poly,
            session_id=None,
            cwd=str(tmp_path),
            require_session=False,
        )

        assert preamble is not None
        assert preamble.project_state is not None
        assert preamble.project_state.branch == "feature/enrich"
        assert len(preamble.project_state.recent_commits) == 1

    @pytest.mark.asyncio
    async def test_git_branch_supersedes_session_recorded_branch(self, tmp_path: Path) -> None:
        """A session's recorded git_branch reflects when the session ran; the
        composing cwd's live branch is the more current signal and wins,
        matching the prior MCP-only enrichment behavior this move preserves."""
        _init_git_repo(tmp_path, branch="feature/now")

        session = MagicMock(git_repository_url="https://example.invalid/repo", git_branch="main-stale")
        poly = MagicMock()
        poly.get_session = AsyncMock(return_value=session)
        poly.get_session_topology = AsyncMock(return_value=None)
        poly.find_resume_candidates = AsyncMock(return_value=[])
        poly.list_assertion_claim_payloads = AsyncMock(return_value=[])

        preamble = await build_context_preamble_payload(
            poly,
            session_id="seed",
            cwd=str(tmp_path),
        )

        assert preamble is not None
        assert preamble.project_state is not None
        assert preamble.project_state.repo == "https://example.invalid/repo"
        assert preamble.project_state.branch == "feature/now"

    @pytest.mark.asyncio
    async def test_no_cwd_git_state_falls_back_to_session_metadata(self) -> None:
        session = MagicMock(git_repository_url="https://example.invalid/repo", git_branch="recorded-branch")
        poly = MagicMock()
        poly.get_session = AsyncMock(return_value=session)
        poly.get_session_topology = AsyncMock(return_value=None)
        poly.find_resume_candidates = AsyncMock(return_value=[])
        poly.list_assertion_claim_payloads = AsyncMock(return_value=[])

        preamble = await build_context_preamble_payload(
            poly,
            session_id="seed",
            cwd=str(Path("/nonexistent/definitely-not-a-repo-path")),
        )

        assert preamble is not None
        assert preamble.project_state is not None
        assert preamble.project_state.repo == "https://example.invalid/repo"
        assert preamble.project_state.branch == "recorded-branch"
        assert preamble.project_state.recent_commits == []
