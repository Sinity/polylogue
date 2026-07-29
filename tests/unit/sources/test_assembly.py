"""Tests for provider assembly layer — sidecar discovery and session enrichment."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import MaterialOrigin, Provider, TitleSource
from polylogue.sources.assembly import SidecarData, get_assembly_spec
from polylogue.sources.assembly_claude_code import ClaudeCodeAssemblySpec
from polylogue.sources.assembly_codex import (
    CodexAssemblySpec,
    _parse_codex_history,
    _parse_codex_session_index,
    _parse_codex_state_titles,
)
from polylogue.sources.assembly_gemini import GeminiAssemblySpec
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.sources.parsers.claude.index import (
    SessionIndexEntry,
    _looks_like_git_branch,
    enrich_session_from_index,
)


def _parsed_message(
    provider_message_id: str,
    role: str,
    text: str,
    *,
    material_origin: MaterialOrigin | None = None,
) -> ParsedMessage:
    message = ParsedMessage(
        provider_message_id=provider_message_id,
        role=Role.normalize(role),
        text=text,
    )
    if material_origin is not None:
        message.material_origin = material_origin
    return message


def _authored_message(provider_message_id: str, text: str) -> ParsedMessage:
    return _parsed_message(
        provider_message_id,
        "user",
        text,
        material_origin=MaterialOrigin.HUMAN_AUTHORED,
    )


def _parsed_attachment(name: str | None = None) -> ParsedAttachment:
    return ParsedAttachment(
        provider_attachment_id="attachment-1",
        message_provider_id="m1",
        name=name,
    )


def _parsed_session(
    source_name: Provider,
    provider_session_id: str,
    title: str,
    messages: list[ParsedMessage],
    *,
    attachments: list[ParsedAttachment] | None = None,
) -> ParsedSession:
    return ParsedSession(
        source_name=source_name,
        provider_session_id=provider_session_id,
        title=title,
        created_at=None,
        updated_at=None,
        messages=messages,
        attachments=attachments or [],
    )


def _thread_sidecars(
    thread_names: dict[str, str] | None = None,
    history_titles: dict[str, str] | None = None,
    state_titles: dict[str, str] | None = None,
) -> SidecarData:
    return {
        "thread_names": {} if thread_names is None else thread_names,
        "history_titles": {} if history_titles is None else history_titles,
        "state_titles": {} if state_titles is None else state_titles,
    }


def _session_sidecars(
    session_index: dict[str, SessionIndexEntry] | None = None,
) -> SidecarData:
    return {"session_index": {} if session_index is None else session_index}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestGetAssemblySpec:
    def test_claude_code_returns_spec(self) -> None:
        spec = get_assembly_spec(Provider.CLAUDE_CODE)
        assert isinstance(spec, ClaudeCodeAssemblySpec)

    def test_codex_returns_spec(self) -> None:
        spec = get_assembly_spec(Provider.CODEX)
        assert isinstance(spec, CodexAssemblySpec)

    def test_gemini_returns_spec(self) -> None:
        spec = get_assembly_spec(Provider.GEMINI)
        assert isinstance(spec, GeminiAssemblySpec)

    @pytest.mark.parametrize("provider", [Provider.CHATGPT, Provider.CLAUDE_AI, Provider.UNKNOWN])
    def test_no_spec_for_other_providers(self, provider: Provider) -> None:
        assert get_assembly_spec(provider) is None


# ---------------------------------------------------------------------------
# Gemini Assembly
# ---------------------------------------------------------------------------


class TestGeminiAssemblySpec:
    def test_discover_sidecars_returns_empty_data(self, tmp_path: Path) -> None:
        session_file = tmp_path / "gemini.json"
        session_file.write_text("{}", encoding="utf-8")

        sidecar_data = GeminiAssemblySpec().discover_sidecars([session_file])

        assert sidecar_data == {}

    def test_meaningful_imported_title_is_preserved(self) -> None:
        conv = _parsed_session(
            Provider.GEMINI,
            "gemini-id-1234",
            "Gemini Session",
            [_parsed_message("m1", "user", "Summarize the roadmap")],
        )

        result = GeminiAssemblySpec().enrich_session(conv, {})

        assert result is conv

    def test_id_like_title_uses_first_user_message_display_label(self) -> None:
        conv = _parsed_session(
            Provider.GEMINI,
            "gemini-20250422-1234",
            "gemini-20250422-1234",
            [
                _parsed_message("m1", "assistant", "Opening context"),
                _parsed_message("m2", "user", "Summarize the retention plan for Q2."),
            ],
        )

        enriched = GeminiAssemblySpec().enrich_session(conv, {})

        assert enriched.title == "Summarize the retention plan for Q2."
        assert enriched.title_source == TitleSource.HEURISTIC

    def test_attachment_name_informs_display_label(self) -> None:
        conv = _parsed_session(
            Provider.GEMINI,
            "gemini-attachment-221",
            "gemini-attachment-221",
            [_parsed_message("m1", "user", "Please review the attached project plan.")],
            attachments=[_parsed_attachment("Project Plan")],
        )

        enriched = GeminiAssemblySpec().enrich_session(conv, {})

        assert enriched.title == "Project Plan: Please review the attached project plan."
        assert enriched.title_source == TitleSource.HEURISTIC

    def test_empty_title_uses_attachment_name_when_no_prompt_exists(self) -> None:
        conv = _parsed_session(
            Provider.GEMINI,
            "gemini-empty-title",
            "",
            [_parsed_message("m1", "assistant", "Ready")],
            attachments=[_parsed_attachment("Project Plan")],
        )

        enriched = GeminiAssemblySpec().enrich_session(conv, {})

        assert enriched.title == "Attachment: Project Plan"
        assert enriched.title_source == TitleSource.HEURISTIC

    def test_minimal_payload_without_label_evidence_is_unchanged(self) -> None:
        conv = _parsed_session(
            Provider.GEMINI,
            "gemini-minimal-221",
            "gemini-minimal-221",
            [_parsed_message("m1", "assistant", "Ready")],
        )

        result = GeminiAssemblySpec().enrich_session(conv, {})

        assert result is conv


# ---------------------------------------------------------------------------
# Claude Code Assembly
# ---------------------------------------------------------------------------


class TestClaudeCodeAssemblySpec:
    def test_discover_sidecars_parses_sessions_index(self, tmp_path: Path) -> None:
        """Discovers sessions-index.json and returns session_index dict."""
        index_data = {
            "entries": [
                {
                    "sessionId": "sess-1",
                    "fullPath": str(tmp_path / "sess-1.jsonl"),
                    "firstPrompt": "Hello",
                    "summary": "Greeting session",
                    "messageCount": 5,
                    "created": "2025-01-01T00:00:00Z",
                    "modified": "2025-01-02T00:00:00Z",
                    "gitBranch": "main",
                    "projectPath": "/project",
                    "isSidechain": False,
                },
                {
                    "sessionId": "sess-2",
                    "fullPath": str(tmp_path / "sess-2.jsonl"),
                    "summary": "Second session",
                    "messageCount": 3,
                },
            ]
        }
        (tmp_path / "sessions-index.json").write_text(json.dumps(index_data), encoding="utf-8")
        session_file = tmp_path / "sess-1.jsonl"
        session_file.touch()

        spec = ClaudeCodeAssemblySpec()
        sidecar_data = spec.discover_sidecars([session_file])

        assert "session_index" in sidecar_data
        idx = sidecar_data["session_index"]
        assert "sess-1" in idx
        assert "sess-2" in idx
        assert isinstance(idx["sess-1"], SessionIndexEntry)
        assert idx["sess-1"].summary == "Greeting session"

    def test_discover_sidecars_handles_missing_index(self, tmp_path: Path) -> None:
        """Returns empty session_index when no sessions-index.json exists."""
        session_file = tmp_path / "sess-1.jsonl"
        session_file.touch()

        spec = ClaudeCodeAssemblySpec()
        sidecar_data = spec.discover_sidecars([session_file])

        assert sidecar_data["session_index"] == {}

    def test_enrich_session_updates_title_from_summary(self) -> None:
        """Enriches session title from session index summary."""
        spec = ClaudeCodeAssemblySpec()
        conv = _parsed_session(
            Provider.CLAUDE_CODE,
            "sess-1",
            "sess-1",
            [_parsed_message("m1", "user", "hello")],
        )
        entry = SessionIndexEntry(
            session_id="sess-1",
            full_path="/tmp/sess-1.jsonl",
            first_prompt="Hello",
            summary="Build the parser",
            message_count=5,
            created="2025-01-01T00:00:00Z",
            modified="2025-01-02T00:00:00Z",
            git_branch="main",
            project_path="/project",
            is_sidechain=False,
        )
        sidecar_data = _session_sidecars({"sess-1": entry})

        enriched = spec.enrich_session(conv, sidecar_data)

        assert enriched.title == "Build the parser"
        assert enriched.title_source == "origin"

    def test_enrich_session_no_match_returns_original(self) -> None:
        """Returns original session when no session index match."""
        spec = ClaudeCodeAssemblySpec()
        conv = _parsed_session(Provider.CLAUDE_CODE, "sess-99", "original", [])
        sidecar_data = _session_sidecars()

        result = spec.enrich_session(conv, sidecar_data)

        assert result is conv
        assert result.title == "original"

    def test_enrich_session_from_index_preserves_semantic_fields(self, tmp_path: Path) -> None:
        session_file = tmp_path / "session-1.jsonl"
        session = ParsedSession(
            source_name=Provider.CLAUDE_CODE,
            provider_session_id="session-1",
            title="session-1",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            messages=[_parsed_message("m1", "user", "hello")],
            session_events=[
                ParsedSessionEvent(
                    event_type="compaction",
                    timestamp="2025-01-01T00:00:01Z",
                    payload={"summary": "compact"},
                )
            ],
            parent_session_provider_id="parent-session",
            branch_type=BranchType.SIDECHAIN,
        )
        entry = SessionIndexEntry(
            session_id="session-1",
            full_path=str(session_file),
            first_prompt="Summarize this repo",
            summary="Investigate parser contracts",
            message_count=12,
            created="2025-01-02T00:00:00Z",
            modified="2025-01-03T00:00:00Z",
            git_branch="main",
            project_path="/tmp/project",
            is_sidechain=True,
        )

        enriched = enrich_session_from_index(session, entry)

        assert enriched.session_events == session.session_events
        assert enriched.parent_session_provider_id == "parent-session"
        assert enriched.branch_type == BranchType.SIDECHAIN


# ---------------------------------------------------------------------------
# Codex Assembly
# ---------------------------------------------------------------------------


class TestCodexAssemblySpec:
    def test_discover_sidecars_parses_session_index_jsonl(self, tmp_path: Path) -> None:
        """Discovers session_index.jsonl and returns thread_names dict."""
        codex_dir = tmp_path / ".codex"
        sessions_dir = codex_dir / "sessions"
        sessions_dir.mkdir(parents=True)
        index_path = codex_dir / "session_index.jsonl"
        index_path.write_text(
            '{"id": "thread-1", "thread_name": "Build API client", "updated_at": "2025-01-01T00:00:00Z"}\n'
            '{"id": "thread-2", "thread_name": "Fix auth bug", "updated_at": "2025-01-02T00:00:00Z"}\n',
            encoding="utf-8",
        )
        session_file = sessions_dir / "thread-1" / "session.jsonl"
        session_file.parent.mkdir()
        session_file.touch()

        spec = CodexAssemblySpec()
        sidecar_data = spec.discover_sidecars([session_file])

        assert "thread_names" in sidecar_data
        names = sidecar_data["thread_names"]
        assert names["thread-1"] == "Build API client"
        assert names["thread-2"] == "Fix auth bug"

    def test_discover_sidecars_handles_missing_index(self, tmp_path: Path) -> None:
        """Returns empty thread_names when no session_index.jsonl exists."""
        codex_dir = tmp_path / ".codex"
        sessions_dir = codex_dir / "sessions"
        sessions_dir.mkdir(parents=True)
        session_file = sessions_dir / "thread-1" / "session.jsonl"
        session_file.parent.mkdir()
        session_file.touch()

        spec = CodexAssemblySpec()
        sidecar_data = spec.discover_sidecars([session_file])

        assert sidecar_data["thread_names"] == {}

    def test_enrich_session_uses_thread_name(self) -> None:
        """Enriches session title from thread name in sidecar data."""
        spec = CodexAssemblySpec()
        conv = _parsed_session(Provider.CODEX, "thread-1", "thread-1", [_parsed_message("m1", "user", "build client")])
        sidecar_data = _thread_sidecars({"thread-1": "Build API client"})

        enriched = spec.enrich_session(conv, sidecar_data)

        assert enriched.title == "Build API client"
        assert enriched.title_source == TitleSource.ORIGIN

    def test_enrich_session_falls_back_to_first_user_message(self) -> None:
        """Falls back to first user message when no thread name available."""
        spec = CodexAssemblySpec()
        conv = _parsed_session(
            Provider.CODEX,
            "thread-1",
            "thread-1",
            [
                _authored_message("m1", "Implement the payment gateway"),
                _parsed_message("m2", "assistant", "Sure, here is the code"),
            ],
        )
        sidecar_data = _thread_sidecars()

        enriched = spec.enrich_session(conv, sidecar_data)

        assert enriched.title == "Implement the payment gateway"
        assert enriched.title_source == "heuristic"

    def test_enrich_session_truncates_long_first_message(self) -> None:
        """Truncates first user message to 80 chars + ellipsis."""
        spec = CodexAssemblySpec()
        long_text = "A" * 100
        conv = _parsed_session(Provider.CODEX, "thread-1", "thread-1", [_authored_message("m1", long_text)])
        sidecar_data = _thread_sidecars()

        enriched = spec.enrich_session(conv, sidecar_data)

        assert enriched.title == "A" * 80 + "..."
        assert len(enriched.title) == 83

    def test_enrich_session_no_match_no_user_messages(self) -> None:
        """Returns original session when no enrichment possible."""
        spec = CodexAssemblySpec()
        conv = _parsed_session(Provider.CODEX, "thread-1", "thread-1", [_parsed_message("m1", "assistant", "response")])
        sidecar_data = _thread_sidecars()

        result = spec.enrich_session(conv, sidecar_data)

        assert result is conv

    def test_enrich_session_skips_empty_user_messages(self) -> None:
        """Skips empty user messages when looking for first-user-message fallback."""
        spec = CodexAssemblySpec()
        conv = _parsed_session(
            Provider.CODEX,
            "thread-1",
            "thread-1",
            [
                _authored_message("m1", ""),
                _authored_message("m2", "   "),
                _authored_message("m3", "Real message here"),
            ],
        )
        sidecar_data = _thread_sidecars()

        enriched = spec.enrich_session(conv, sidecar_data)

        assert enriched.title == "Real message here"

    def test_enrich_session_does_not_override_different_title(self) -> None:
        """Does not fall back to first-user-message when title differs from conv ID."""
        spec = CodexAssemblySpec()
        conv = _parsed_session(
            Provider.CODEX,
            "thread-1",
            "Already has a title",
            [_parsed_message("m1", "user", "some message")],
        )
        sidecar_data = _thread_sidecars()

        result = spec.enrich_session(conv, sidecar_data)

        assert result is conv
        assert result.title == "Already has a title"

    def test_enrich_session_downgrades_thread_name_that_echoes_first_prompt(self) -> None:
        """bd polylogue-6e7m: a sidecar title that just restates the opening
        prompt is not independent curation -- ORIGIN would overstate that."""
        spec = CodexAssemblySpec()
        conv = _parsed_session(
            Provider.CODEX,
            "thread-1",
            "thread-1",
            [_authored_message("m1", "Take over Claude's session 755b624d")],
        )
        sidecar_data = _thread_sidecars({"thread-1": "take over claude's session 755b624d"})

        enriched = spec.enrich_session(conv, sidecar_data)

        assert enriched.title == "take over claude's session 755b624d"
        assert enriched.title_source == TitleSource.HEURISTIC
        assert enriched.title_confidence == 0.5

    def test_enrich_session_keeps_thread_name_distinct_from_first_prompt(self) -> None:
        spec = CodexAssemblySpec()
        conv = _parsed_session(
            Provider.CODEX,
            "thread-1",
            "thread-1",
            [_authored_message("m1", "Take over Claude's session 755b624d")],
        )
        sidecar_data = _thread_sidecars({"thread-1": "Session handoff triage"})

        enriched = spec.enrich_session(conv, sidecar_data)

        assert enriched.title == "Session handoff triage"
        assert enriched.title_source == TitleSource.ORIGIN
        assert enriched.title_confidence == 1.0


class TestParseCodexSessionIndex:
    def test_parses_valid_jsonl(self, tmp_path: Path) -> None:
        codex_dir = tmp_path / ".codex"
        sessions_dir = codex_dir / "sessions"
        sessions_dir.mkdir(parents=True)
        (codex_dir / "session_index.jsonl").write_text(
            '{"id": "t1", "thread_name": "Alpha"}\n{"id": "t2", "thread_name": "Beta"}\n',
            encoding="utf-8",
        )

        result = _parse_codex_session_index(sessions_dir)

        assert result == {"t1": "Alpha", "t2": "Beta"}

    def test_latest_entry_wins(self, tmp_path: Path) -> None:
        codex_dir = tmp_path / ".codex"
        sessions_dir = codex_dir / "sessions"
        sessions_dir.mkdir(parents=True)
        (codex_dir / "session_index.jsonl").write_text(
            '{"id": "t1", "thread_name": "Old Name"}\n{"id": "t1", "thread_name": "New Name"}\n',
            encoding="utf-8",
        )

        result = _parse_codex_session_index(sessions_dir)

        assert result == {"t1": "New Name"}

    def test_handles_malformed_lines(self, tmp_path: Path) -> None:
        codex_dir = tmp_path / ".codex"
        sessions_dir = codex_dir / "sessions"
        sessions_dir.mkdir(parents=True)
        (codex_dir / "session_index.jsonl").write_text(
            '{"id": "t1", "thread_name": "Good"}\n'
            "not-json\n"
            '{"id": "t2"}\n'  # Missing thread_name
            '{"id": "t3", "thread_name": "Also Good"}\n',
            encoding="utf-8",
        )

        result = _parse_codex_session_index(sessions_dir)

        assert result == {"t1": "Good", "t3": "Also Good"}

    def test_supports_alternative_field_names(self, tmp_path: Path) -> None:
        """Supports thread_id/name as alternative to id/thread_name."""
        codex_dir = tmp_path / ".codex"
        sessions_dir = codex_dir / "sessions"
        sessions_dir.mkdir(parents=True)
        (codex_dir / "session_index.jsonl").write_text(
            '{"thread_id": "t1", "name": "Alt Names"}\n',
            encoding="utf-8",
        )

        result = _parse_codex_session_index(sessions_dir)

        assert result == {"t1": "Alt Names"}

    def test_handles_missing_file(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        result = _parse_codex_session_index(sessions_dir)

        assert result == {}


# ---------------------------------------------------------------------------
# Anti-title heuristics (git branch rejection)
# ---------------------------------------------------------------------------


class TestLooksLikeGitBranch:
    @pytest.mark.parametrize(
        "value",
        [
            "main",
            "master",
            "develop",
            "dev",
            "staging",
            "production",
            "HEAD",
            "feature/auth-fix",
            "fix/login-bug",
            "bugfix/cors-headers",
            "hotfix/prod-crash",
            "release/v2.0",
            "chore/deps-update",
            "refactor/cleanup",
            "test/add-coverage",
            "docs/readme",
            "ci/pipeline",
            "perf/query-optimization",
            "claude/phase_3",
        ],
    )
    def test_rejects_git_branch_names(self, value: str) -> None:
        assert _looks_like_git_branch(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "Fixed authentication bug in login flow",
            "Build the parser module",
            "Investigate memory leak",
            "User Exits CLI Session",
            "Set up CI/CD pipeline",
            "Update the feature flags documentation",
            "",
        ],
    )
    def test_accepts_real_titles(self, value: str) -> None:
        assert _looks_like_git_branch(value) is False

    def test_git_branch_summary_skipped_in_enrichment(self) -> None:
        """enrich_session_from_index skips summary that looks like a git branch."""
        conv = _parsed_session(
            Provider.CLAUDE_CODE,
            "sess-1",
            "sess-1",
            [_parsed_message("m1", "user", "hello")],
        )
        entry = SessionIndexEntry(
            session_id="sess-1",
            full_path="/tmp/sess-1.jsonl",
            first_prompt="Fix the bug",
            summary="feature/auth-fix",  # Git branch — should be rejected
            message_count=5,
            created=None,
            modified=None,
            git_branch="feature/auth-fix",
            project_path="/project",
            is_sidechain=False,
        )

        enriched = enrich_session_from_index(conv, entry)

        # Should fall back to first_prompt since summary looks like a git branch
        assert enriched.title == "Fix the bug"
        assert enriched.title_source == "heuristic"

    def test_git_branch_exact_match_skipped(self) -> None:
        """Exact branch names like 'main' are rejected as summaries."""
        conv = _parsed_session(
            Provider.CLAUDE_CODE,
            "sess-1",
            "sess-1",
            [_parsed_message("m1", "user", "hello")],
        )
        entry = SessionIndexEntry(
            session_id="sess-1",
            full_path="/tmp/sess-1.jsonl",
            first_prompt="Hello world",
            summary="main",
            message_count=1,
            created=None,
            modified=None,
            git_branch="main",
            project_path="/project",
            is_sidechain=False,
        )

        enriched = enrich_session_from_index(conv, entry)

        assert enriched.title == "Hello world"
        assert enriched.title_source == "heuristic"

    def test_git_branch_claude_prefix_skipped(self) -> None:
        """Claude Code's own auto-branch namespace (claude/phase_3) is rejected as a title.

        polylogue-cijx.3 triage (2026-07-29): the corpus observes real
        ``gitBranch`` values under a ``claude/`` prefix that the original
        ``_GIT_BRANCH_PREFIXES`` list did not cover.
        """
        conv = _parsed_session(
            Provider.CLAUDE_CODE,
            "sess-1",
            "sess-1",
            [_parsed_message("m1", "user", "hello")],
        )
        entry = SessionIndexEntry(
            session_id="sess-1",
            full_path="/tmp/sess-1.jsonl",
            first_prompt="Hello world",
            summary="claude/phase_3",
            message_count=1,
            created=None,
            modified=None,
            git_branch=None,
            project_path="/project",
            is_sidechain=False,
        )

        enriched = enrich_session_from_index(conv, entry)

        assert enriched.title == "Hello world"
        assert enriched.title_source == "heuristic"

    def test_typed_git_branch_preferred_over_shape_heuristic(self) -> None:
        """A summary that merely resembles a branch name, but isn't the known one, is kept.

        When typed ``gitBranch`` evidence exists, ``enrich_session_from_index``
        must compare the summary against that exact value rather than guessing
        from shape alone -- a shape-only guess both over-matches (this case)
        and under-matches (the previous ``claude/phase_3`` gap) real titles.
        """
        conv = _parsed_session(
            Provider.CLAUDE_CODE,
            "sess-1",
            "sess-1",
            [_parsed_message("m1", "user", "hello")],
        ).model_copy(update={"git_branch": "main"})
        entry = SessionIndexEntry(
            session_id="sess-1",
            full_path="/tmp/sess-1.jsonl",
            first_prompt="Hello world",
            summary="release/v2.0",  # shape-heuristic would reject this, but it isn't the real branch
            message_count=1,
            created=None,
            modified=None,
            git_branch=None,
            project_path="/project",
            is_sidechain=False,
        )

        enriched = enrich_session_from_index(conv, entry)

        assert enriched.title == "release/v2.0"
        assert enriched.title_source == "origin"
        assert enriched.git_branch == "main"


# ---------------------------------------------------------------------------
# Codex history.jsonl titles + material-origin title discipline (polylogue-ih67)
# ---------------------------------------------------------------------------


def _codex_root_with(
    tmp_path: Path,
    *,
    history_lines: list[str] | None = None,
    index_lines: list[str] | None = None,
) -> Path:
    codex_dir = tmp_path / ".codex"
    sessions_dir = codex_dir / "sessions"
    sessions_dir.mkdir(parents=True)
    if history_lines is not None:
        (codex_dir / "history.jsonl").write_text("\n".join(history_lines) + "\n", encoding="utf-8")
    if index_lines is not None:
        (codex_dir / "session_index.jsonl").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    session_file = sessions_dir / "thread-1" / "session.jsonl"
    session_file.parent.mkdir()
    session_file.touch()
    return session_file


class TestCodexHistoryTitles:
    def test_discover_sidecars_includes_history_titles(self, tmp_path: Path) -> None:
        """history.jsonl yields earliest authored text per session; junk skipped."""
        session_file = _codex_root_with(
            tmp_path,
            history_lines=[
                '{"session_id": "thread-1", "ts": 200, "text": "later prompt"}',
                '{"session_id": "thread-1", "ts": 100, "text": "Fix the ingest bug"}',
                "not json at all",
                '{"session_id": "", "ts": 1, "text": "no session"}',
                '{"session_id": "thread-2", "ts": 50, "text": "   "}',
                '{"session_id": "thread-2", "ts": 60, "text": "Second thread opening"}',
            ],
        )

        sidecar_data = CodexAssemblySpec().discover_sidecars([session_file])

        assert sidecar_data["history_titles"] == {
            "thread-1": "Fix the ingest bug",
            "thread-2": "Second thread opening",
        }

    def test_thread_name_beats_history(self, tmp_path: Path) -> None:
        spec = CodexAssemblySpec()
        conv = _parsed_session(Provider.CODEX, "thread-1", "thread-1", [])
        sidecar_data = _thread_sidecars(
            {"thread-1": "Named thread"},
            {"thread-1": "history prompt"},
        )

        enriched = spec.enrich_session(conv, sidecar_data)

        assert enriched.title == "Named thread"
        assert enriched.title_source == TitleSource.ORIGIN

    def test_history_beats_message_fallback(self) -> None:
        spec = CodexAssemblySpec()
        conv = _parsed_session(
            Provider.CODEX,
            "thread-1",
            "thread-1",
            [_authored_message("m1", "message body that would otherwise win")],
        )
        sidecar_data = _thread_sidecars(None, {"thread-1": "History opening prompt"})

        enriched = spec.enrich_session(conv, sidecar_data)

        assert enriched.title == "History opening prompt"
        assert enriched.title_source == TitleSource.ORIGIN

    def test_history_title_uses_first_line_bounded(self) -> None:
        spec = CodexAssemblySpec()
        conv = _parsed_session(Provider.CODEX, "thread-1", "thread-1", [])
        long_first_line = "B" * 100
        sidecar_data = _thread_sidecars(None, {"thread-1": f"\n\n{long_first_line}\nsecond line"})

        enriched = spec.enrich_session(conv, sidecar_data)

        assert enriched.title == "B" * 80 + "..."

    def test_history_downgrades_when_it_echoes_the_first_prompt(self) -> None:
        """history_titles is *by construction* the earliest authored prompt
        (see _parse_codex_history's docstring) -- it is essentially always
        an echo, and this test proves the runtime check catches it rather
        than trusting the sidecar label alone."""
        spec = CodexAssemblySpec()
        conv = _parsed_session(
            Provider.CODEX,
            "thread-1",
            "thread-1",
            [_authored_message("m1", "familiarize yourself with the repo and its full beads-set")],
        )
        sidecar_data = _thread_sidecars(None, {"thread-1": "familiarize yourself with the repo and its full beads-set"})

        enriched = spec.enrich_session(conv, sidecar_data)

        assert enriched.title_source == TitleSource.HEURISTIC
        assert enriched.title_confidence == 0.5

    def test_history_never_replaces_real_title(self) -> None:
        spec = CodexAssemblySpec()
        conv = _parsed_session(Provider.CODEX, "thread-1", "A real existing title", [])
        sidecar_data = _thread_sidecars(None, {"thread-1": "history prompt"})

        result = spec.enrich_session(conv, sidecar_data)

        assert result is conv

    def test_runtime_context_user_row_never_becomes_title(self) -> None:
        """An injected-context role=user row must not win over the authored request."""
        spec = CodexAssemblySpec()
        conv = _parsed_session(
            Provider.CODEX,
            "thread-1",
            "thread-1",
            [
                _parsed_message(
                    "m1",
                    "user",
                    "<AGENTS.md> repository instructions injected as context",
                    material_origin=MaterialOrigin.RUNTIME_CONTEXT,
                ),
                _authored_message("m2", "Please fix the payment bug"),
            ],
        )

        enriched = spec.enrich_session(conv, _thread_sidecars())

        assert enriched.title == "Please fix the payment bug"
        assert enriched.title_source == TitleSource.HEURISTIC

    def test_unknown_authorship_is_not_title_material(self) -> None:
        """role=user alone (material_origin UNKNOWN) is not proof a human typed it."""
        spec = CodexAssemblySpec()
        conv = _parsed_session(
            Provider.CODEX,
            "thread-1",
            "thread-1",
            [_parsed_message("m1", "user", "ambiguous channel content")],
        )

        result = spec.enrich_session(conv, _thread_sidecars())

        assert result is conv
        assert result.title == "thread-1"

    def test_history_cache_invalidates_on_file_change(self, tmp_path: Path) -> None:
        import os

        session_file = _codex_root_with(
            tmp_path,
            history_lines=['{"session_id": "thread-1", "ts": 1, "text": "first version"}'],
        )
        history_path = tmp_path / ".codex" / "history.jsonl"
        spec = CodexAssemblySpec()

        first = spec.discover_sidecars([session_file])["history_titles"]
        assert first == {"thread-1": "first version"}

        history_path.write_text('{"session_id": "thread-1", "ts": 1, "text": "rewritten"}\n', encoding="utf-8")
        stat = history_path.stat()
        os.utime(history_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

        second = spec.discover_sidecars([session_file])["history_titles"]
        assert second == {"thread-1": "rewritten"}

    def test_parse_codex_history_missing_file(self, tmp_path: Path) -> None:
        sessions_root = tmp_path / ".codex" / "sessions"
        sessions_root.mkdir(parents=True)
        assert _parse_codex_history(sessions_root) == {}


# ---------------------------------------------------------------------------
# Codex state_5.sqlite live thread titles (polylogue-ih67)
# ---------------------------------------------------------------------------


def _write_codex_state_db(codex_dir: Path, rows: list[tuple[str, str]]) -> Path:
    """Create a minimal ``state_5.sqlite`` with a ``threads(id, title)`` table."""
    state_path = codex_dir / "state_5.sqlite"
    conn = sqlite3.connect(state_path)
    try:
        conn.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '')")
        conn.executemany("INSERT INTO threads (id, title) VALUES (?, ?)", rows)
        conn.commit()
    finally:
        conn.close()
    return state_path


class TestCodexStateTitles:
    def test_discover_sidecars_includes_state_titles(self, tmp_path: Path) -> None:
        codex_dir = tmp_path / ".codex"
        sessions_dir = codex_dir / "sessions"
        sessions_dir.mkdir(parents=True)
        session_file = sessions_dir / "thread-1" / "session.jsonl"
        session_file.parent.mkdir()
        session_file.touch()
        _write_codex_state_db(
            codex_dir,
            [("thread-1", "Live-DB curated title"), ("thread-2", ""), ("thread-3", "  ")],
        )

        sidecar_data = CodexAssemblySpec().discover_sidecars([session_file])

        assert sidecar_data["state_titles"] == {"thread-1": "Live-DB curated title"}

    def test_discover_sidecars_handles_missing_state_db(self, tmp_path: Path) -> None:
        codex_dir = tmp_path / ".codex"
        sessions_dir = codex_dir / "sessions"
        sessions_dir.mkdir(parents=True)
        session_file = sessions_dir / "thread-1" / "session.jsonl"
        session_file.parent.mkdir()
        session_file.touch()

        sidecar_data = CodexAssemblySpec().discover_sidecars([session_file])

        assert sidecar_data["state_titles"] == {}

    def test_discover_sidecars_handles_malformed_state_db(self, tmp_path: Path) -> None:
        """A non-SQLite file at the expected path degrades to empty, never raises."""
        codex_dir = tmp_path / ".codex"
        sessions_dir = codex_dir / "sessions"
        sessions_dir.mkdir(parents=True)
        session_file = sessions_dir / "thread-1" / "session.jsonl"
        session_file.parent.mkdir()
        session_file.touch()
        (codex_dir / "state_5.sqlite").write_bytes(b"not a sqlite file at all")

        sidecar_data = CodexAssemblySpec().discover_sidecars([session_file])

        assert sidecar_data["state_titles"] == {}

    def test_discover_sidecars_handles_schema_mismatch(self, tmp_path: Path) -> None:
        """A state_5.sqlite without the expected threads/title shape degrades to empty."""
        codex_dir = tmp_path / ".codex"
        sessions_dir = codex_dir / "sessions"
        sessions_dir.mkdir(parents=True)
        session_file = sessions_dir / "thread-1" / "session.jsonl"
        session_file.parent.mkdir()
        session_file.touch()
        state_path = codex_dir / "state_5.sqlite"
        conn = sqlite3.connect(state_path)
        try:
            conn.execute("CREATE TABLE unrelated_table (foo TEXT)")
            conn.commit()
        finally:
            conn.close()

        sidecar_data = CodexAssemblySpec().discover_sidecars([session_file])

        assert sidecar_data["state_titles"] == {}

    def test_discover_sidecars_handles_locked_state_db(self, tmp_path: Path) -> None:
        """A concurrently write-locked state_5.sqlite degrades to empty, never raises."""
        codex_dir = tmp_path / ".codex"
        sessions_dir = codex_dir / "sessions"
        sessions_dir.mkdir(parents=True)
        session_file = sessions_dir / "thread-1" / "session.jsonl"
        session_file.parent.mkdir()
        session_file.touch()
        state_path = _write_codex_state_db(codex_dir, [("thread-1", "Curated title")])

        # Hold an exclusive write lock (default rollback-journal mode, not
        # WAL) to simulate Codex mid-write while we attempt a read.
        locker = sqlite3.connect(state_path, timeout=0)
        locker.execute("BEGIN EXCLUSIVE")
        try:
            sidecar_data = CodexAssemblySpec().discover_sidecars([session_file])
            assert sidecar_data["state_titles"] == {}
        finally:
            locker.rollback()
            locker.close()

    def test_state_title_fills_gap_when_no_thread_name_or_history(self) -> None:
        spec = CodexAssemblySpec()
        conv = _parsed_session(Provider.CODEX, "thread-1", "thread-1", [])
        sidecar_data = _thread_sidecars(state_titles={"thread-1": "Curated live title"})

        enriched = spec.enrich_session(conv, sidecar_data)

        assert enriched.title == "Curated live title"
        assert enriched.title_source == TitleSource.ORIGIN

    def test_thread_name_beats_state_title(self) -> None:
        spec = CodexAssemblySpec()
        conv = _parsed_session(Provider.CODEX, "thread-1", "thread-1", [])
        sidecar_data = _thread_sidecars(
            thread_names={"thread-1": "Named thread"},
            state_titles={"thread-1": "Curated live title"},
        )

        enriched = spec.enrich_session(conv, sidecar_data)

        assert enriched.title == "Named thread"

    def test_history_title_beats_state_title(self) -> None:
        spec = CodexAssemblySpec()
        conv = _parsed_session(Provider.CODEX, "thread-1", "thread-1", [])
        sidecar_data = _thread_sidecars(
            history_titles={"thread-1": "History opening prompt"},
            state_titles={"thread-1": "Curated live title"},
        )

        enriched = spec.enrich_session(conv, sidecar_data)

        assert enriched.title == "History opening prompt"

    def test_state_title_beats_first_message_fallback(self) -> None:
        spec = CodexAssemblySpec()
        conv = _parsed_session(
            Provider.CODEX,
            "thread-1",
            "thread-1",
            [_authored_message("m1", "message body that would otherwise win")],
        )
        sidecar_data = _thread_sidecars(state_titles={"thread-1": "Curated live title"})

        enriched = spec.enrich_session(conv, sidecar_data)

        assert enriched.title == "Curated live title"
        assert enriched.title_source == TitleSource.ORIGIN

    def test_state_title_uses_first_line_bounded(self) -> None:
        spec = CodexAssemblySpec()
        conv = _parsed_session(Provider.CODEX, "thread-1", "thread-1", [])
        long_first_line = "C" * 100
        sidecar_data = _thread_sidecars(state_titles={"thread-1": f"\n\n{long_first_line}\nsecond line"})

        enriched = spec.enrich_session(conv, sidecar_data)

        assert enriched.title == "C" * 80 + "..."

    def test_state_title_downgrades_when_it_echoes_the_first_prompt(self) -> None:
        """bd polylogue-6e7m's live scan found 166 of 2,771 state_5.sqlite
        threads.title values shared by >1 thread (worst case 78x) -- all
        confirmed echoes of the opening prompt, not curated titles."""
        spec = CodexAssemblySpec()
        conv = _parsed_session(
            Provider.CODEX,
            "thread-1",
            "thread-1",
            [_authored_message("m1", "find, using whatever means, either directly ~/.codex or polylogue")],
        )
        sidecar_data = _thread_sidecars(
            state_titles={"thread-1": "find, using whatever means, either directly ~/.codex or polylogue"}
        )

        enriched = spec.enrich_session(conv, sidecar_data)

        assert enriched.title_source == TitleSource.HEURISTIC
        assert enriched.title_confidence == 0.5

    def test_state_title_never_replaces_real_title(self) -> None:
        spec = CodexAssemblySpec()
        conv = _parsed_session(Provider.CODEX, "thread-1", "A real existing title", [])
        sidecar_data = _thread_sidecars(state_titles={"thread-1": "Curated live title"})

        result = spec.enrich_session(conv, sidecar_data)

        assert result is conv

    def test_parse_codex_state_titles_missing_file(self, tmp_path: Path) -> None:
        sessions_root = tmp_path / ".codex" / "sessions"
        sessions_root.mkdir(parents=True)
        assert _parse_codex_state_titles(sessions_root) == {}

    def test_state_titles_cache_invalidates_on_file_change(self, tmp_path: Path) -> None:
        import os

        codex_dir = tmp_path / ".codex"
        sessions_dir = codex_dir / "sessions"
        sessions_dir.mkdir(parents=True)
        session_file = sessions_dir / "thread-1" / "session.jsonl"
        session_file.parent.mkdir()
        session_file.touch()
        state_path = _write_codex_state_db(codex_dir, [("thread-1", "first version")])
        spec = CodexAssemblySpec()

        first = spec.discover_sidecars([session_file])["state_titles"]
        assert first == {"thread-1": "first version"}

        conn = sqlite3.connect(state_path)
        try:
            conn.execute("UPDATE threads SET title = ? WHERE id = ?", ("rewritten", "thread-1"))
            conn.commit()
        finally:
            conn.close()
        stat = state_path.stat()
        os.utime(state_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

        second = spec.discover_sidecars([session_file])["state_titles"]
        assert second == {"thread-1": "rewritten"}
