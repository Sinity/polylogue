"""Tests for provider assembly layer — sidecar discovery and conversation enrichment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.sources.assembly import SidecarData, get_assembly_spec
from polylogue.sources.assembly_claude_code import ClaudeCodeAssemblySpec
from polylogue.sources.assembly_codex import CodexAssemblySpec, _parse_codex_session_index
from polylogue.sources.assembly_gemini import GeminiAssemblySpec
from polylogue.sources.parsers.base import ParsedAttachment, ParsedConversation, ParsedMessage
from polylogue.sources.parsers.claude.index import (
    SessionIndexEntry,
    _looks_like_git_branch,
)
from polylogue.types import Provider


def _parsed_message(provider_message_id: str, role: str, text: str) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=provider_message_id,
        role=Role.normalize(role),
        text=text,
    )


def _parsed_attachment(name: str | None = None) -> ParsedAttachment:
    return ParsedAttachment(
        provider_attachment_id="attachment-1",
        message_provider_id="m1",
        name=name,
        provider_meta={"name": name} if name else None,
    )


def _parsed_conversation(
    provider_name: Provider,
    provider_conversation_id: str,
    title: str,
    messages: list[ParsedMessage],
    *,
    attachments: list[ParsedAttachment] | None = None,
    provider_meta: dict[str, object] | None = None,
) -> ParsedConversation:
    return ParsedConversation(
        provider_name=provider_name,
        provider_conversation_id=provider_conversation_id,
        title=title,
        created_at=None,
        updated_at=None,
        messages=messages,
        attachments=attachments or [],
        provider_meta=provider_meta,
    )


def _provider_meta(conversation: ParsedConversation) -> dict[str, object]:
    assert conversation.provider_meta is not None
    return conversation.provider_meta


def _thread_sidecars(thread_names: dict[str, str] | None = None) -> SidecarData:
    return {"thread_names": {} if thread_names is None else thread_names}


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
        conv = _parsed_conversation(
            Provider.GEMINI,
            "gemini-id-1234",
            "Gemini Session",
            [_parsed_message("m1", "user", "Summarize the roadmap")],
            provider_meta={"title_source": "imported:displayName"},
        )

        result = GeminiAssemblySpec().enrich_conversation(conv, {})

        assert result is conv
        assert _provider_meta(result)["title_source"] == "imported:displayName"

    def test_id_like_title_uses_first_user_message_display_label(self) -> None:
        conv = _parsed_conversation(
            Provider.GEMINI,
            "gemini-20250422-1234",
            "gemini-20250422-1234",
            [
                _parsed_message("m1", "assistant", "Opening context"),
                _parsed_message("m2", "user", "Summarize the retention plan for Q2."),
            ],
            provider_meta={"title_source": "fallback:id"},
        )

        enriched = GeminiAssemblySpec().enrich_conversation(conv, {})

        assert enriched.title == "gemini-20250422-1234"
        metadata = _provider_meta(enriched)
        assert metadata["title_source"] == "fallback:id"
        assert metadata["display_label"] == "Summarize the retention plan for Q2."
        assert metadata["display_label_source"] == "first-user-message"

    def test_attachment_name_informs_display_label(self) -> None:
        conv = _parsed_conversation(
            Provider.GEMINI,
            "gemini-attachment-221",
            "gemini-attachment-221",
            [_parsed_message("m1", "user", "Please review the attached project plan.")],
            attachments=[_parsed_attachment("Project Plan")],
            provider_meta={"title_source": "fallback:id"},
        )

        enriched = GeminiAssemblySpec().enrich_conversation(conv, {})

        metadata = _provider_meta(enriched)
        assert metadata["display_label"] == "Project Plan: Please review the attached project plan."
        assert metadata["display_label_source"] == "attachment-name:first-user-message"

    def test_empty_title_uses_attachment_name_when_no_prompt_exists(self) -> None:
        conv = _parsed_conversation(
            Provider.GEMINI,
            "gemini-empty-title",
            "",
            [_parsed_message("m1", "assistant", "Ready")],
            attachments=[_parsed_attachment("Project Plan")],
            provider_meta={"title_source": "fallback:id"},
        )

        enriched = GeminiAssemblySpec().enrich_conversation(conv, {})

        metadata = _provider_meta(enriched)
        assert metadata["display_label"] == "Attachment: Project Plan"
        assert metadata["display_label_source"] == "attachment-name"

    def test_minimal_payload_without_label_evidence_is_unchanged(self) -> None:
        conv = _parsed_conversation(
            Provider.GEMINI,
            "gemini-minimal-221",
            "gemini-minimal-221",
            [_parsed_message("m1", "assistant", "Ready")],
            provider_meta={"title_source": "fallback:id"},
        )

        result = GeminiAssemblySpec().enrich_conversation(conv, {})

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

    def test_enrich_conversation_updates_title_from_summary(self) -> None:
        """Enriches conversation title from session index summary."""
        spec = ClaudeCodeAssemblySpec()
        conv = _parsed_conversation(
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

        enriched = spec.enrich_conversation(conv, sidecar_data)

        assert enriched.title == "Build the parser"
        assert _provider_meta(enriched)["title_source"] == "session-index:summary"

    def test_enrich_conversation_no_match_returns_original(self) -> None:
        """Returns original conversation when no session index match."""
        spec = ClaudeCodeAssemblySpec()
        conv = _parsed_conversation(Provider.CLAUDE_CODE, "sess-99", "original", [])
        sidecar_data = _session_sidecars()

        result = spec.enrich_conversation(conv, sidecar_data)

        assert result is conv
        assert result.title == "original"


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

    def test_enrich_conversation_uses_thread_name(self) -> None:
        """Enriches conversation title from thread name in sidecar data."""
        spec = CodexAssemblySpec()
        conv = _parsed_conversation(
            Provider.CODEX, "thread-1", "thread-1", [_parsed_message("m1", "user", "build client")]
        )
        sidecar_data = _thread_sidecars({"thread-1": "Build API client"})

        enriched = spec.enrich_conversation(conv, sidecar_data)

        assert enriched.title == "Build API client"
        metadata = _provider_meta(enriched)
        assert metadata["title_source"] == "session-index:thread-name"
        assert metadata["thread_name"] == "Build API client"

    def test_enrich_conversation_falls_back_to_first_user_message(self) -> None:
        """Falls back to first user message when no thread name available."""
        spec = CodexAssemblySpec()
        conv = _parsed_conversation(
            Provider.CODEX,
            "thread-1",
            "thread-1",
            [
                _parsed_message("m1", "user", "Implement the payment gateway"),
                _parsed_message("m2", "assistant", "Sure, here is the code"),
            ],
        )
        sidecar_data = _thread_sidecars()

        enriched = spec.enrich_conversation(conv, sidecar_data)

        assert enriched.title == "Implement the payment gateway"
        assert _provider_meta(enriched)["title_source"] == "first-user-message"

    def test_enrich_conversation_truncates_long_first_message(self) -> None:
        """Truncates first user message to 80 chars + ellipsis."""
        spec = CodexAssemblySpec()
        long_text = "A" * 100
        conv = _parsed_conversation(Provider.CODEX, "thread-1", "thread-1", [_parsed_message("m1", "user", long_text)])
        sidecar_data = _thread_sidecars()

        enriched = spec.enrich_conversation(conv, sidecar_data)

        assert enriched.title == "A" * 80 + "..."
        assert len(enriched.title) == 83

    def test_enrich_conversation_no_match_no_user_messages(self) -> None:
        """Returns original conversation when no enrichment possible."""
        spec = CodexAssemblySpec()
        conv = _parsed_conversation(
            Provider.CODEX, "thread-1", "thread-1", [_parsed_message("m1", "assistant", "response")]
        )
        sidecar_data = _thread_sidecars()

        result = spec.enrich_conversation(conv, sidecar_data)

        assert result is conv

    def test_enrich_conversation_skips_empty_user_messages(self) -> None:
        """Skips empty user messages when looking for first-user-message fallback."""
        spec = CodexAssemblySpec()
        conv = _parsed_conversation(
            Provider.CODEX,
            "thread-1",
            "thread-1",
            [
                _parsed_message("m1", "user", ""),
                _parsed_message("m2", "user", "   "),
                _parsed_message("m3", "user", "Real message here"),
            ],
        )
        sidecar_data = _thread_sidecars()

        enriched = spec.enrich_conversation(conv, sidecar_data)

        assert enriched.title == "Real message here"

    def test_enrich_conversation_does_not_override_different_title(self) -> None:
        """Does not fall back to first-user-message when title differs from conv ID."""
        spec = CodexAssemblySpec()
        conv = _parsed_conversation(
            Provider.CODEX,
            "thread-1",
            "Already has a title",
            [_parsed_message("m1", "user", "some message")],
        )
        sidecar_data = _thread_sidecars()

        result = spec.enrich_conversation(conv, sidecar_data)

        assert result is conv
        assert result.title == "Already has a title"


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
        """enrich_conversation_from_index skips summary that looks like a git branch."""
        from polylogue.sources.parsers.claude.index import enrich_conversation_from_index

        conv = _parsed_conversation(
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

        enriched = enrich_conversation_from_index(conv, entry)

        # Should fall back to first_prompt since summary looks like a git branch
        assert enriched.title == "Fix the bug"
        assert _provider_meta(enriched)["title_source"] == "session-index:first-prompt"

    def test_git_branch_exact_match_skipped(self) -> None:
        """Exact branch names like 'main' are rejected as summaries."""
        from polylogue.sources.parsers.claude.index import enrich_conversation_from_index

        conv = _parsed_conversation(
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

        enriched = enrich_conversation_from_index(conv, entry)

        assert enriched.title == "Hello world"
        assert _provider_meta(enriched)["title_source"] == "session-index:first-prompt"
