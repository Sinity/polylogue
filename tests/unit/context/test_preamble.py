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
import sqlite3
import subprocess
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from polylogue.context.preamble import _git_project_state, build_context_preamble_payload
from polylogue.context.scheduler import read_context_ledger
from polylogue.core.refs import ExecutionContextRef
from polylogue.markers import parse_markers
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


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

    def test_non_git_directory_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # git discovers a repository by walking upward, and pytest's base
        # temporary directory may itself sit inside a checkout. Cap the walk so
        # the test describes the directory rather than where basetemp lives.
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.resolve().parent))

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

    @pytest.mark.asyncio
    async def test_precompact_uses_scheduler_and_records_real_boundary_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import polylogue.context.preamble as preamble_module
        from polylogue.context.scheduler import schedule_context

        captured: dict[str, object] = {}

        def capture_schedule(sources: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return schedule_context(sources, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(preamble_module, "schedule_context", capture_schedule)
        session = MagicMock(
            git_repository_url="https://example.invalid/repo",
            git_branch="main",
            origin="codex-session",
            model="gpt-test",
            permission_mode="default",
        )
        poly = MagicMock()
        poly.config.archive_root = tmp_path / "archive"
        poly.config.archive_root.mkdir()
        initialize_archive_database(poly.config.archive_root / "ops.db", ArchiveTier.OPS)
        poly.get_session = AsyncMock(return_value=session)
        poly.get_session_topology = AsyncMock(return_value=None)
        poly.find_resume_candidates = AsyncMock(return_value=[])
        poly.list_assertion_claim_payloads = AsyncMock(return_value=[])

        preamble = await build_context_preamble_payload(
            poly,
            session_id="seed",
            cwd=str(tmp_path),
            boundary="precompact",
        )

        assert preamble is not None
        with sqlite3.connect(tmp_path / "archive" / "ops.db") as conn:
            records = read_context_ledger(conn, target_session="seed")
        assert len(records) == 1
        assert records[0].row.source == "context-precompact"
        execution_context = cast(ExecutionContextRef, captured["execution_context"])
        assert execution_context.known_fields == (
            "boundary",
            "cwd",
            "model",
            "origin",
            "permission_mode",
            "related_limit",
            "session_id",
        )
        assert execution_context.unknown_fields == ("runtime",)


@pytest.mark.asyncio
async def test_declared_claim_survives_judgment_preamble_and_reboot_ref(
    tmp_path: Path,
) -> None:
    """Exercise the judged-memory loop through the real archive facade.

    The marker is the agent-facing authoring boundary; the candidate write
    route preserves its body and scope, the operator judgment promotes it,
    and a newly opened facade resolves the ref carried by the preamble.
    Mutating any stage to skip promotion, scope filtering, or public ref
    resolution makes one of the assertions below fail.
    """
    from polylogue.api import Polylogue
    from polylogue.core.enums import AssertionKind
    from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    initialize_archive_database(archive_root / "user.db", ArchiveTier.USER)
    initialize_archive_database(archive_root / "index.db", ArchiveTier.INDEX)
    initialize_archive_database(archive_root / "ops.db", ArchiveTier.OPS)
    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    session_ref = "session:codex:judged-memory-loop"
    repo_ref = "repo:polylogue"
    try:
        marker = parse_markers("::decision: Keep context as refs, not raw logs.\n")[0]
        assert marker.kind == "decision"

        with sqlite3.connect(archive_root / "user.db") as conn:
            upsert_assertion(
                conn,
                assertion_id="marker-judged-memory-loop",
                target_ref=session_ref,
                scope_ref=repo_ref,
                kind=AssertionKind.DECISION,
                body_text=marker.body,
                author_ref="agent:codex",
                author_kind="agent",
                evidence_refs=(session_ref,),
                status="candidate",
                visibility="private",
                context_policy={"inject": False},
                staleness={"expires_at_ms": 9_999_999_999_999},
                now_ms=1_700_000_000_000,
            )
        candidates = await archive.list_assertion_candidates(target_ref=session_ref, limit=1)
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.status is not None
        assert candidate.status.value == "candidate"
        assert candidate.scope_ref == repo_ref
        assert candidate.staleness is not None
        assert candidate.staleness["expires_at_ms"] > candidate.created_at_ms

        review = await archive.judge_assertion_candidate(
            candidate_ref=f"assertion:{candidate.assertion_id}",
            decision="accept",
            reason="The operator accepted this scoped context rule.",
            actor_ref="user:local",
            inject=True,
        )
        assert review.outcome == "applied"
        assert review.resulting_assertion is not None
        emitted_ref = f"assertion:{candidate.assertion_id}"
        assert review.resulting_assertion.status is not None
        assert review.resulting_assertion.status.value == "active"

        archive.get_session = AsyncMock(return_value=None)  # type: ignore[method-assign]
        archive.find_resume_candidates = AsyncMock(return_value=[])  # type: ignore[method-assign]
        preamble = await build_context_preamble_payload(
            archive,
            session_id=session_ref.removeprefix("session:"),
            repo_path=repo_ref,
            require_session=False,
        )
        assert preamble is not None
        assert preamble.injected_at is not None
        assert preamble.guidance is not None
        guidance = preamble.guidance
        assert not isinstance(guidance, str)
        assert guidance.assertions[0].quoted_evidence is not None
        assert guidance.assertions[0].evidence_refs == [session_ref, emitted_ref]
        assert guidance.assertions[0].scope_ref == repo_ref
        assert guidance.assertions[0].quoted_evidence.text == marker.body

    finally:
        await archive.close()

    rebooted = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    try:
        resolved = await rebooted.resolve_ref(emitted_ref)
        assert resolved.resolved is True
        assert resolved.payload_kind == "assertion-claim"
        assert resolved.payload is not None
        assert resolved.payload["assertion_id"] == emitted_ref.removeprefix("assertion:")
    finally:
        await rebooted.close()
