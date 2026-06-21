"""Unit tests for the compose_context_preamble MCP tool."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.core.enums import AssertionKind
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async, make_polylogue_mock


def _make_candidate(
    session_id: str = "test-session-1",
    title: str = "Test Session",
    date: str = "2026-05-01T10:00:00Z",
    terminal_state: str = "completed",
    summary: str | None = None,
    origin: str = "claude-code-session",
) -> MagicMock:
    """Build a mock resume candidate with specified fields."""
    c = MagicMock()
    c.logical_session_id = session_id
    c.session_id = session_id
    c.title = title
    c.date = date
    c.terminal_state = terminal_state
    c.summary = summary
    c.origin = origin
    return c


def _mock_subprocess_failure(**kwargs: object) -> MagicMock:
    """Return a MagicMock for subprocess.run that always fails."""
    return MagicMock(returncode=1, stdout="", stderr="")


class TestComposeContextPreambleRegistration:
    """Verify the compose_context_preamble tool is registered and callable."""

    def test_tool_is_registered_on_server(self, mcp_server: MCPServerUnderTest) -> None:
        """The tool is present in the server tool manager."""
        tools = mcp_server._tool_manager._tools
        assert "compose_context_preamble" in tools
        assert callable(tools["compose_context_preamble"].fn)


class TestComposeContextPreambleHappyPath:
    """Happy path: seeded session produces valid ContextPreamble JSON."""

    @pytest.mark.asyncio
    async def test_returns_valid_preamble_structure(self, mcp_server: MCPServerUnderTest) -> None:
        """A basic invocation returns a valid ContextPreamble JSON with all
        expected top-level keys."""
        with (
            patch("polylogue.mcp.server._get_polylogue") as mock_get_poly,
            patch("subprocess.run", return_value=_mock_subprocess_failure()),
        ):
            mock_poly = make_polylogue_mock()
            mock_poly.find_resume_candidates = AsyncMock(return_value=())
            mock_get_poly.return_value = mock_poly

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["compose_context_preamble"].fn,
                repo_path=".",
            )

        payload = json.loads(raw)
        assert "preamble_version" in payload
        assert payload["preamble_version"] == "1.0"
        assert "injected_at" in payload
        assert "source_tool_calls" in payload
        assert "recent_related_sessions" in payload
        assert isinstance(payload["recent_related_sessions"], list)
        # project_state is omitted when both branch and commits are empty.
        assert "project_state" not in payload

    @pytest.mark.asyncio
    async def test_recent_sessions_populated(self, mcp_server: MCPServerUnderTest) -> None:
        """Resume candidates returned by the facade appear in recent_related_sessions."""
        candidate = _make_candidate(
            session_id="test-session-1",
            title="Fix login bug",
            date="2026-05-01T10:00:00Z",
            terminal_state="completed",
            origin="claude-code-session",
        )

        with (
            patch("polylogue.mcp.server._get_polylogue") as mock_get_poly,
            patch("subprocess.run", return_value=_mock_subprocess_failure()),
        ):
            mock_poly = make_polylogue_mock()
            mock_poly.find_resume_candidates = AsyncMock(return_value=(candidate,))
            mock_get_poly.return_value = mock_poly

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["compose_context_preamble"].fn,
                repo_path=".",
            )

        payload = json.loads(raw)
        sessions = payload["recent_related_sessions"]
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "test-session-1"
        assert sessions[0]["title"] == "Fix login bug"
        assert sessions[0]["date"] == "2026-05-01T10:00:00Z"
        assert sessions[0]["terminal_state"] == "completed"
        assert sessions[0]["origin"] == "claude-code-session"

    @pytest.mark.asyncio
    async def test_candidate_without_logical_session_id_falls_back(self, mcp_server: MCPServerUnderTest) -> None:
        """When logical_session_id is absent, session_id is used."""
        c = MagicMock()
        c.logical_session_id = None
        c.session_id = "fallback-id"
        c.title = None
        c.date = None
        c.terminal_state = None
        c.summary = None
        c.origin = None

        with (
            patch("polylogue.mcp.server._get_polylogue") as mock_get_poly,
            patch("subprocess.run", return_value=_mock_subprocess_failure()),
        ):
            mock_poly = make_polylogue_mock()
            mock_poly.find_resume_candidates = AsyncMock(return_value=(c,))
            mock_get_poly.return_value = mock_poly

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["compose_context_preamble"].fn,
                repo_path=".",
            )

        payload = json.loads(raw)
        sessions = payload["recent_related_sessions"]
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "fallback-id"

    @pytest.mark.asyncio
    async def test_project_state_with_git_output(self, mcp_server: MCPServerUnderTest) -> None:
        """When git commands succeed, project_state is populated with branch
        and recent commits."""
        with (
            patch("polylogue.mcp.server._get_polylogue") as mock_get_poly,
            patch("subprocess.run") as mock_run,
        ):
            mock_poly = make_polylogue_mock()
            mock_poly.find_resume_candidates = AsyncMock(return_value=())
            mock_get_poly.return_value = mock_poly

            # Two git calls: first for branch, second for log.
            def _git_side_effect(args: list[str], **kwargs: object) -> MagicMock:
                if "rev-parse" in args:
                    return MagicMock(returncode=0, stdout="feature/test-branch\n", stderr="")
                if "log" in args:
                    return MagicMock(
                        returncode=0,
                        stdout="abc1234 feat: add new feature\ndef5678 fix: bug fix\n",
                        stderr="",
                    )
                return MagicMock(returncode=1, stdout="", stderr="")

            mock_run.side_effect = _git_side_effect

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["compose_context_preamble"].fn,
                repo_path=".",
            )

        payload = json.loads(raw)
        assert "project_state" in payload
        ps = payload["project_state"]
        assert ps["branch"] == "feature/test-branch"
        assert len(ps["recent_commits"]) == 2
        assert "abc1234 feat: add new feature" in ps["recent_commits"]

    @pytest.mark.asyncio
    async def test_repo_path_forwarded(self, mcp_server: MCPServerUnderTest) -> None:
        """The repo_path parameter is forwarded to find_resume_candidates."""
        with (
            patch("polylogue.mcp.server._get_polylogue") as mock_get_poly,
            patch("subprocess.run", return_value=_mock_subprocess_failure()),
        ):
            mock_poly = make_polylogue_mock()
            mock_poly.find_resume_candidates = AsyncMock(return_value=())
            mock_get_poly.return_value = mock_poly

            await invoke_surface_async(
                mcp_server._tool_manager._tools["compose_context_preamble"].fn,
                repo_path="/realm/project/sinex",
                cwd="/realm/project/sinex",
            )

        mock_poly.find_resume_candidates.assert_called_once_with(
            repo_path="/realm/project/sinex",
            cwd="/realm/project/sinex",
            recent_files=(),
            limit=5,
        )

    @pytest.mark.asyncio
    async def test_recent_files_forwarded(self, mcp_server: MCPServerUnderTest) -> None:
        """The recent_files tuple is forwarded as-is."""
        with (
            patch("polylogue.mcp.server._get_polylogue") as mock_get_poly,
            patch("subprocess.run", return_value=_mock_subprocess_failure()),
        ):
            mock_poly = make_polylogue_mock()
            mock_poly.find_resume_candidates = AsyncMock(return_value=())
            mock_get_poly.return_value = mock_poly

            await invoke_surface_async(
                mcp_server._tool_manager._tools["compose_context_preamble"].fn,
                repo_path=".",
                recent_files=("src/main.py", "tests/test_main.py"),
            )

        mock_poly.find_resume_candidates.assert_called_once_with(
            repo_path=".",
            cwd=None,
            recent_files=("src/main.py", "tests/test_main.py"),
            limit=5,
        )


class TestComposeContextPreambleEmptyArchive:
    """Empty archive: graceful handling."""

    @pytest.mark.asyncio
    async def test_empty_candidates_produces_valid_preamble(self, mcp_server: MCPServerUnderTest) -> None:
        """No resume candidates → empty recent_related_sessions list, valid JSON."""
        with (
            patch("polylogue.mcp.server._get_polylogue") as mock_get_poly,
            patch("subprocess.run", return_value=_mock_subprocess_failure()),
        ):
            mock_poly = make_polylogue_mock()
            mock_poly.find_resume_candidates = AsyncMock(return_value=())
            mock_get_poly.return_value = mock_poly

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["compose_context_preamble"].fn,
                repo_path="/nonexistent",
            )

        payload = json.loads(raw)
        assert payload["preamble_version"] == "1.0"
        assert payload["recent_related_sessions"] == []
        assert "injected_at" in payload

    @pytest.mark.asyncio
    async def test_find_resume_candidates_exception_is_graceful(self, mcp_server: MCPServerUnderTest) -> None:
        """When find_resume_candidates raises, the tool still returns a valid
        preamble with an empty sessions list."""
        with (
            patch("polylogue.mcp.server._get_polylogue") as mock_get_poly,
            patch("subprocess.run", return_value=_mock_subprocess_failure()),
        ):
            mock_poly = make_polylogue_mock()
            mock_poly.find_resume_candidates = AsyncMock(side_effect=RuntimeError("database is locked"))
            mock_get_poly.return_value = mock_poly

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["compose_context_preamble"].fn,
                repo_path=".",
            )

        payload = json.loads(raw)
        assert payload["recent_related_sessions"] == []
        assert "injected_at" in payload

    @pytest.mark.asyncio
    async def test_git_exception_is_graceful(self, mcp_server: MCPServerUnderTest) -> None:
        """When git commands fail (e.g. no repo), project_state is absent."""
        with (
            patch("polylogue.mcp.server._get_polylogue") as mock_get_poly,
            patch("subprocess.run", side_effect=FileNotFoundError("git not found")),
        ):
            mock_poly = make_polylogue_mock()
            mock_poly.find_resume_candidates = AsyncMock(return_value=())
            mock_get_poly.return_value = mock_poly

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["compose_context_preamble"].fn,
                repo_path=".",
            )

        payload = json.loads(raw)
        assert "project_state" not in payload
        assert "injected_at" in payload

    @pytest.mark.asyncio
    async def test_limit_is_clamped(self, mcp_server: MCPServerUnderTest) -> None:
        """The limit parameter is routed through hooks.clamp_limit."""
        with (
            patch("polylogue.mcp.server._get_polylogue") as mock_get_poly,
            patch("subprocess.run", return_value=_mock_subprocess_failure()),
        ):
            mock_poly = make_polylogue_mock()
            mock_poly.find_resume_candidates = AsyncMock(return_value=())
            mock_get_poly.return_value = mock_poly

            await invoke_surface_async(
                mcp_server._tool_manager._tools["compose_context_preamble"].fn,
                repo_path=".",
                limit=100,
            )

        # clamp_limit returns max(1, min(100, 1000)) = 100
        mock_poly.find_resume_candidates.assert_called_once_with(
            repo_path=".",
            cwd=None,
            recent_files=(),
            limit=100,
        )


class TestContextPreambleModel:
    """Test the ContextPreamble model with lineage, blackboard, and other
    optional fields that may be populated by future tool revisions."""

    def test_preamble_with_lineage(self) -> None:
        """ContextPreamble with session_lineage serializes lineage fields."""
        from polylogue.surfaces.payloads import (
            ContextPreamble,
            ContextPreambleLineage,
            ContextPreambleProjectState,
        )

        lineage = ContextPreambleLineage(
            logical_session_root="root-1",
            parent_session_id="parent-1",
            sibling_session_ids=["sib-1", "sib-2"],
            continuation_chain_depth=3,
        )
        preamble = ContextPreamble(
            preamble_version="1.0",
            injected_at="2026-05-01T00:00:00Z",
            session_lineage=lineage,
            project_state=ContextPreambleProjectState(branch="main", recent_commits=["abc1234 feat: test"]),
        )

        data = json.loads(preamble.model_dump_json(exclude_none=True))
        assert data["session_lineage"]["logical_session_root"] == "root-1"
        assert data["session_lineage"]["parent_session_id"] == "parent-1"
        assert data["session_lineage"]["continuation_chain_depth"] == 3
        assert len(data["session_lineage"]["sibling_session_ids"]) == 2
        assert data["project_state"]["branch"] == "main"

    def test_preamble_with_blackboard_notes(self) -> None:
        """ContextPreamble with blackboard_notes serializes note fields."""
        from polylogue.surfaces.payloads import (
            ContextPreamble,
            ContextPreambleBlackboardNote,
        )

        note = ContextPreambleBlackboardNote(
            key="finding-1",
            content="Need to fix auth module",
            repo="polylogue",
            created_at="2026-05-01T00:00:00Z",
        )
        preamble = ContextPreamble(
            preamble_version="1.0",
            injected_at="2026-05-01T00:00:00Z",
            blackboard_notes=[note],
        )

        data = json.loads(preamble.model_dump_json(exclude_none=True))
        assert len(data["blackboard_notes"]) == 1
        assert data["blackboard_notes"][0]["key"] == "finding-1"
        assert data["blackboard_notes"][0]["content"] == "Need to fix auth module"
        assert data["blackboard_notes"][0]["repo"] == "polylogue"

    def test_preamble_with_open_issues(self) -> None:
        """ContextPreamble with open_issues serializes issue fields."""
        from polylogue.surfaces.payloads import ContextPreamble, ContextPreambleIssue

        issue = ContextPreambleIssue(
            number=1721,
            title="Test coverage gaps",
            state="open",
            labels=["test", "coverage"],
            url="https://github.com/Sinity/polylogue/issues/1721",
        )
        preamble = ContextPreamble(
            preamble_version="1.0",
            injected_at="2026-05-01T00:00:00Z",
            open_issues=[issue],
        )

        data = json.loads(preamble.model_dump_json(exclude_none=True))
        assert len(data["open_issues"]) == 1
        assert data["open_issues"][0]["number"] == 1721
        assert data["open_issues"][0]["title"] == "Test coverage gaps"
        assert data["open_issues"][0]["state"] == "open"

    def test_preamble_with_guidance(self) -> None:
        """ContextPreamble with guidance serializes the guidance string."""
        from polylogue.surfaces.payloads import ContextPreamble

        preamble = ContextPreamble(
            preamble_version="1.0",
            injected_at="2026-05-01T00:00:00Z",
            guidance="Focus on #1721 test coverage gaps.",
        )

        data = json.loads(preamble.model_dump_json(exclude_none=True))
        assert data["guidance"] == "Focus on #1721 test coverage gaps."

    def test_preamble_with_structured_guidance_and_repo_state(self) -> None:
        """ContextPreamble carries structured assertion guidance across surfaces."""
        from polylogue.surfaces.payloads import (
            ContextPreamble,
            ContextPreambleAssertionGuidance,
            ContextPreambleGuidance,
            ContextPreambleProjectState,
        )

        preamble = ContextPreamble(
            preamble_version="1.0",
            project_state=ContextPreambleProjectState(repo="https://github.com/Sinity/polylogue", branch="master"),
            guidance=ContextPreambleGuidance(
                assertions=[
                    ContextPreambleAssertionGuidance(
                        kind=AssertionKind.DECISION,
                        text="Use the shared context-preamble builder.",
                        target_ref="session:target",
                        scope_ref="repo:polylogue",
                        evidence_refs=["target::m1"],
                    )
                ]
            ),
        )

        data = json.loads(preamble.model_dump_json(exclude_none=True))
        assert data["project_state"]["repo"] == "https://github.com/Sinity/polylogue"
        assert data["guidance"]["assertions"][0]["kind"] == "decision"

    def test_preamble_default_construction(self) -> None:
        """Default ContextPreamble has sensible empty values."""
        from polylogue.surfaces.payloads import ContextPreamble

        preamble = ContextPreamble()
        assert preamble.preamble_version == "1.0"
        assert preamble.injected_at is None
        assert preamble.session_lineage is None
        assert preamble.recent_related_sessions == []
        assert preamble.open_issues == []
        assert preamble.project_state is None
        assert preamble.blackboard_notes == []
        assert preamble.guidance is None
