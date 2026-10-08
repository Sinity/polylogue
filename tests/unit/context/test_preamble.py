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
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from polylogue.context.preamble import _git_project_state, build_context_preamble_payload
from polylogue.core.refs import ExecutionContextRef
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

if TYPE_CHECKING:
    from polylogue.api import Polylogue


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

        state, failure = _git_project_state(str(tmp_path))

        assert failure is None
        assert state is not None
        assert state.branch == "feature/preamble-move"
        assert len(state.recent_commits) == 1
        assert "initial commit" in state.recent_commits[0]

    def test_non_git_directory_returns_none_without_a_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # git discovers a repository by walking upward, and pytest's base
        # temporary directory may itself sit inside a checkout. Cap the walk so
        # the test describes the directory rather than where basetemp lives.
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.resolve().parent))

        state, failure = _git_project_state(str(tmp_path))

        # A directory that is simply not a checkout is an answer, not a gap.
        assert state is None
        assert failure is None

    def test_missing_directory_never_raises_but_is_recorded(self) -> None:
        """A git read that *broke* must not look like a clean non-repo build.

        Anti-vacuity: restore the bare ``except Exception: pass`` and the
        failure string goes back to ``None`` here while
        ``test_non_git_directory_returns_none_without_a_failure`` keeps passing
        -- the two cases become indistinguishable, which is the defect.
        """
        state, failure = _git_project_state(str(Path("/nonexistent/definitely-not-a-repo-path")))

        assert state is None
        assert failure is not None
        assert failure.startswith("FileNotFoundError") or failure.startswith("NotADirectoryError")

    @pytest.mark.asyncio
    async def test_broken_git_read_reaches_component_failures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The recorded gap must reach the payload a caller actually reads."""
        poly = MagicMock()
        poly.get_session = AsyncMock(return_value=None)
        poly.compact_lineage = AsyncMock(return_value=None)
        poly.find_resume_candidates = AsyncMock(return_value=[])
        poly.list_assertion_claim_payloads = AsyncMock(return_value=[])
        poly.record_context_ledger = AsyncMock()

        preamble = await build_context_preamble_payload(
            poly,
            session_id=None,
            cwd=str(Path("/nonexistent/definitely-not-a-repo-path")),
            require_session=False,
        )

        assert preamble is not None
        assert "project_state" in preamble.component_failures
        assert preamble.component_failures["project_state"]


class TestBuildContextPreambleGitEnrichment:
    """``build_context_preamble_payload`` merges git enrichment for cwd."""

    @pytest.mark.asyncio
    async def test_cwd_git_state_populates_project_state(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path, branch="feature/enrich")

        poly = MagicMock()
        poly.get_session = AsyncMock(return_value=None)
        poly.compact_lineage = AsyncMock(return_value=None)
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
        poly.compact_lineage = AsyncMock(return_value=None)
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
        poly.compact_lineage = AsyncMock(return_value=None)
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
        poly.compact_lineage = AsyncMock(return_value=None)
        poly.find_resume_candidates = AsyncMock(return_value=[])
        poly.list_assertion_claim_payloads = AsyncMock(return_value=[])
        poly.record_context_ledger = AsyncMock()

        preamble = await build_context_preamble_payload(
            poly,
            session_id="seed",
            cwd=str(tmp_path),
            boundary="precompact",
        )

        assert preamble is not None
        poly.record_context_ledger.assert_awaited_once()
        assembly = poly.record_context_ledger.await_args.args[0]
        assert assembly.ledger[0].source == "context-precompact"
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


def _judging_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> tuple[Path, Polylogue]:
    """Start the resident writer and open a facade over its archive.

    Judging a candidate is a durable user-tier mutation, which only the
    resident daemon writes; the loop runs against the real operation stack.
    """
    from polylogue.api import Polylogue
    from polylogue.daemon.socket_path import daemon_socket_path
    from tests.infra.daemon_operations import running_daemon_operations

    archive_root = (tmp_path / "archive").resolve()
    monkeypatch.setattr("polylogue.daemon.api_auth.resolve_api_auth_token", lambda *_args, **_kwargs: None)
    from tests.infra.archive_templates import run_off_event_loop

    daemon = running_daemon_operations(archive_root, socket_path=daemon_socket_path(archive_root))
    # The stack bootstraps the archive under a synchronous write lease, which
    # may not block the async test's running loop.
    run_off_event_loop(daemon.__enter__)
    request.addfinalizer(lambda: daemon.__exit__(None, None, None))
    return archive_root, Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")


@pytest.mark.asyncio
async def test_lowered_marker_survives_judgment_and_reboot_ref(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """A marker reaches a judged claim only through production lowering.

    The candidate is written by ``candidates_for_block``/``lower_markers``, the
    route the accepted-marker consumer publishes through, so its identity,
    target, body, and evidence are derived rather than supplied by the test.
    Anti-vacuity: writing the candidate any other way, or changing the
    lowering's identity, target, or evidence derivation, fails the identity
    and provenance assertions; skipping promotion or public ref resolution
    fails the judgment and reboot assertions.
    """
    from polylogue.api import Polylogue
    from polylogue.markers import candidates_for_block, lower_markers
    from polylogue.markers.lowering import assertion_id_for_marker

    archive_root, archive = _judging_archive(tmp_path, monkeypatch, request)
    try:
        lowered = candidates_for_block(
            "judged-memory-loop-message",
            "judged-memory-loop-block",
            "::decision: Keep context as refs, not raw logs.\n",
        )
        assert len(lowered) == 1
        marker = lowered[0]
        with sqlite3.connect(archive_root / "user.db") as conn:
            assert lower_markers(conn, lowered, now_ms=1_700_000_000_000) == (assertion_id_for_marker(marker),)

        message_ref = "message:judged-memory-loop-message"
        candidates = await archive.list_assertion_candidates(target_ref=message_ref, limit=2)
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.assertion_id == assertion_id_for_marker(marker)
        assert candidate.status is not None
        assert candidate.status.value == "candidate"
        assert candidate.author_kind == "agent"
        assert candidate.body_text == marker.match.body
        assert candidate.evidence_refs == (message_ref, "block:judged-memory-loop-block")

        emitted_ref = f"assertion:{candidate.assertion_id}"
        review = await archive.judge_assertion_candidate(
            candidate_ref=emitted_ref,
            decision="accept",
            reason="The operator accepted this marker claim.",
            actor_ref="user:local",
            inject=True,
        )
        assert review.outcome == "applied"
        assert review.resulting_assertion is not None
        assert review.resulting_assertion.status is not None
        assert review.resulting_assertion.status.value == "active"
        assert review.resulting_assertion.body_text == marker.match.body
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


@pytest.mark.asyncio
async def test_judged_scoped_claim_reaches_the_session_preamble(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """An accepted, injectable session claim is carried by the preamble.

    The candidate is authored directly into the user tier with a session
    target and repository scope; marker lowering targets the message ref and
    carries no scope, so it is covered by the test above instead.
    Anti-vacuity: skipping promotion, dropping the ``inject`` policy, or
    losing the scope or evidence on the way to the preamble fails one of the
    guidance assertions.
    """
    from polylogue.core.enums import AssertionKind
    from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion

    archive_root, archive = _judging_archive(tmp_path, monkeypatch, request)
    session_ref = "session:codex:judged-memory-loop"
    repo_ref = "repo:polylogue"
    body = "Keep context as refs, not raw logs."
    try:
        with sqlite3.connect(archive_root / "user.db") as conn:
            upsert_assertion(
                conn,
                assertion_id="operator-judged-memory-loop",
                target_ref=session_ref,
                scope_ref=repo_ref,
                kind=AssertionKind.DECISION,
                body_text=body,
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
        assert candidate.scope_ref == repo_ref
        assert candidate.staleness is not None
        assert candidate.staleness["expires_at_ms"] > candidate.created_at_ms

        emitted_ref = f"assertion:{candidate.assertion_id}"
        review = await archive.judge_assertion_candidate(
            candidate_ref=emitted_ref,
            decision="accept",
            reason="The operator accepted this scoped context rule.",
            actor_ref="user:local",
            inject=True,
        )
        assert review.outcome == "applied"

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
        assert guidance.assertions[0].quoted_evidence.text == body
    finally:
        await archive.close()
