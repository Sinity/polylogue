"""Tests for hook event taxonomy, classification, and security boundaries.

Exercises:
  - ArtifactKind.HOOK_EVENT is recognized by classify_artifact
  - Hook event stream detection via looks_like_hook_event / looks_like_hook_event_stream
  - Current Claude Code and Codex hook catalogs classify correctly
  - Paste ground truth (UserPromptSubmit) event shape
  - Tool annotation events (PreToolUse + PostToolUse)
  - Error subtype events (PostToolUseFailure)
  - Permission decision events (PermissionRequest + PermissionDenied)
  - Security/privacy: local path handling, session_id validation
  - Edge cases: empty payloads, malformed records, missing fields
"""

from __future__ import annotations

import pytest

from polylogue.archive.artifact_taxonomy import ArtifactKind, classify_artifact
from polylogue.archive.artifact_taxonomy.support import (
    looks_like_hook_event,
    looks_like_hook_event_stream,
)
from polylogue.core.json import JSONDocument, JSONValue
from polylogue.hooks import CLAUDE_CODE_EVENTS, CODEX_EVENTS


def _make_hook_record(
    event_type: str = "PreToolUse",
    session_id: str = "test-session-001",
    provider: str = "claude-code",
    payload: JSONDocument | None = None,
) -> JSONDocument:
    """Build a valid hook event record as a JSONDocument."""
    return {
        "event_type": event_type,
        "session_id": session_id,
        "timestamp": "2026-05-07T12:00:00Z",
        "provider": provider,
        "payload": payload or {"tool_name": "Read", "tool_input": {"file_path": "/tmp/test.txt"}},
    }


# ---------------------------------------------------------------------------
# Detection: looks_like_hook_event
# ---------------------------------------------------------------------------


def test_looks_like_hook_event_valid() -> None:
    """Valid hook event record is detected."""
    record = _make_hook_record()
    assert looks_like_hook_event(record) is True


def test_looks_like_hook_event_missing_event_type() -> None:
    """Record without event_type is not a hook event."""
    record: JSONDocument = {
        "session_id": "test-001",
        "timestamp": "2026-05-07T12:00:00Z",
        "provider": "claude-code",
    }
    assert looks_like_hook_event(record) is False


def test_looks_like_hook_event_missing_session_id() -> None:
    """Record without session_id is not a hook event."""
    record: JSONDocument = {
        "event_type": "PreToolUse",
        "timestamp": "2026-05-07T12:00:00Z",
        "provider": "claude-code",
    }
    assert looks_like_hook_event(record) is False


def test_looks_like_hook_event_missing_timestamp() -> None:
    """Record without timestamp is not a hook event."""
    record: JSONDocument = {
        "event_type": "PreToolUse",
        "session_id": "test-001",
        "provider": "claude-code",
    }
    assert looks_like_hook_event(record) is False


def test_looks_like_hook_event_invalid_provider() -> None:
    """Record with unknown provider is not a hook event."""
    record: JSONDocument = {
        "event_type": "PreToolUse",
        "session_id": "test-001",
        "timestamp": "2026-05-07T12:00:00Z",
        "provider": "unknown-provider",
    }
    assert looks_like_hook_event(record) is False


def test_looks_like_hook_event_non_dict() -> None:
    """Non-dict values are not hook events."""
    assert looks_like_hook_event("not a dict") is False
    assert looks_like_hook_event(42) is False
    assert looks_like_hook_event(None) is False


def test_looks_like_hook_event_empty_dict() -> None:
    """Empty dict is not a hook event."""
    assert looks_like_hook_event({}) is False


# ---------------------------------------------------------------------------
# Detection: looks_like_hook_event_stream
# ---------------------------------------------------------------------------


def test_looks_like_hook_event_stream_valid() -> None:
    """A list of valid hook records is detected as a hook event stream."""
    records: list[JSONDocument] = [
        _make_hook_record(event_type="PreToolUse"),
        _make_hook_record(event_type="PostToolUse"),
    ]
    assert looks_like_hook_event_stream(records) is True


def test_looks_like_hook_event_stream_single() -> None:
    """A single valid hook record is detected."""
    records: list[JSONDocument] = [_make_hook_record()]
    assert looks_like_hook_event_stream(records) is True


def test_looks_like_hook_event_stream_empty() -> None:
    """Empty list is not a hook event stream."""
    assert looks_like_hook_event_stream([]) is False


def test_looks_like_hook_event_stream_mixed() -> None:
    """Mixed valid/invalid records are not a hook event stream."""
    records: list[JSONDocument] = [
        _make_hook_record(),
        {"not": "a hook event"},
    ]
    assert looks_like_hook_event_stream(records) is False


# ---------------------------------------------------------------------------
# Classification: classify_artifact for hook events
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event_type",
    CLAUDE_CODE_EVENTS,
)
def test_classify_artifact_cc_stream(event_type: str) -> None:
    """Every current Claude Code hook event classifies as HOOK_EVENT."""
    records: JSONValue = [_make_hook_record(event_type=event_type, provider="claude-code")]
    artifact = classify_artifact(records, provider="claude-code")
    assert artifact.kind is ArtifactKind.HOOK_EVENT, f"{event_type} should be HOOK_EVENT, got {artifact.kind}"
    assert artifact.parse_as_session is False


@pytest.mark.parametrize(
    "event_type",
    CODEX_EVENTS,
)
def test_classify_artifact_codex_stream(event_type: str) -> None:
    """Every current Codex hook event classifies as HOOK_EVENT."""
    records: JSONValue = [_make_hook_record(event_type=event_type, provider="codex")]
    artifact = classify_artifact(records, provider="codex")
    assert artifact.kind is ArtifactKind.HOOK_EVENT, f"{event_type} should be HOOK_EVENT, got {artifact.kind}"
    assert artifact.parse_as_session is False


def test_classify_artifact_single_hook_dict() -> None:
    """A single dict hook record classifies as HOOK_EVENT."""
    record: JSONValue = _make_hook_record()
    artifact = classify_artifact(record, provider="claude-code")
    assert artifact.kind is ArtifactKind.HOOK_EVENT


# ---------------------------------------------------------------------------
# Paste ground truth (UserPromptSubmit)
# ---------------------------------------------------------------------------


def test_user_prompt_submit_before_expansion() -> None:
    """UserPromptSubmit captures paste markers before expansion."""
    payload: JSONDocument = {
        "prompt": "Look at [Pasted text #1] and [Pasted text #2]",
        "session_id": "test-session-001",
    }
    record: JSONValue = _make_hook_record(
        event_type="UserPromptSubmit",
        provider="claude-code",
        payload=payload,
    )

    artifact = classify_artifact(record, provider="claude-code")
    assert artifact.kind is ArtifactKind.HOOK_EVENT

    # Verify paste markers are preserved in the payload
    assert isinstance(record, dict)
    inner = record.get("payload")
    assert isinstance(inner, dict)
    prompt = inner.get("prompt")
    assert isinstance(prompt, str)
    assert "[Pasted text #1]" in prompt
    assert "[Pasted text #2]" in prompt


def test_user_prompt_submit_codex() -> None:
    """Codex UserPromptSubmit classifies correctly."""
    payload: JSONDocument = {
        "prompt": "Fix the bug in auth.py",
        "session_id": "codex-session-001",
    }
    record: JSONValue = _make_hook_record(
        event_type="UserPromptSubmit",
        provider="codex",
        payload=payload,
    )

    artifact = classify_artifact(record, provider="codex")
    assert artifact.kind is ArtifactKind.HOOK_EVENT


# ---------------------------------------------------------------------------
# Tool annotations (PreToolUse + PostToolUse)
# ---------------------------------------------------------------------------


def test_pre_tool_use_annotations() -> None:
    """PreToolUse captures tool_name and tool_input."""
    payload: JSONDocument = {
        "tool_name": "Bash",
        "tool_input": {"command": "pytest -q", "description": "Run tests"},
        "tool_call_id": "toolu_abc123",
    }
    record: JSONValue = _make_hook_record(
        event_type="PreToolUse",
        provider="claude-code",
        payload=payload,
    )

    artifact = classify_artifact(record, provider="claude-code")
    assert artifact.kind is ArtifactKind.HOOK_EVENT

    assert isinstance(record, dict)
    inner = record.get("payload")
    assert isinstance(inner, dict)
    assert inner.get("tool_name") == "Bash"
    assert isinstance(inner.get("tool_input"), dict)
    assert inner.get("tool_call_id") == "toolu_abc123"


def test_post_tool_use_output() -> None:
    """PostToolUse captures tool output."""
    payload: JSONDocument = {
        "tool_name": "Read",
        "tool_output": "file contents here...",
        "tool_call_id": "toolu_xyz789",
    }
    record: JSONValue = _make_hook_record(
        event_type="PostToolUse",
        provider="claude-code",
        payload=payload,
    )

    artifact = classify_artifact(record, provider="claude-code")
    assert artifact.kind is ArtifactKind.HOOK_EVENT

    assert isinstance(record, dict)
    inner = record.get("payload")
    assert isinstance(inner, dict)
    assert inner.get("tool_name") == "Read"
    assert inner.get("tool_output") == "file contents here..."


# ---------------------------------------------------------------------------
# Error subtypes (PostToolUseFailure)
# ---------------------------------------------------------------------------


def test_post_tool_use_failure() -> None:
    """PostToolUseFailure captures error details."""
    payload: JSONDocument = {
        "tool_name": "Bash",
        "error": "command not found: nosuch",
        "is_interrupt": False,
    }
    record: JSONValue = _make_hook_record(
        event_type="PostToolUseFailure",
        provider="claude-code",
        payload=payload,
    )

    artifact = classify_artifact(record, provider="claude-code")
    assert artifact.kind is ArtifactKind.HOOK_EVENT

    assert isinstance(record, dict)
    inner = record.get("payload")
    assert isinstance(inner, dict)
    assert inner.get("tool_name") == "Bash"
    assert "command not found" in str(inner.get("error"))


def test_post_tool_use_failure_interrupt() -> None:
    """PostToolUseFailure with is_interrupt flag."""
    payload: JSONDocument = {
        "tool_name": "Bash",
        "error": "user interrupt",
        "is_interrupt": True,
    }
    record: JSONValue = _make_hook_record(
        event_type="PostToolUseFailure",
        provider="claude-code",
        payload=payload,
    )

    artifact = classify_artifact(record, provider="claude-code")
    assert artifact.kind is ArtifactKind.HOOK_EVENT

    assert isinstance(record, dict)
    inner = record.get("payload")
    assert isinstance(inner, dict)
    assert inner.get("is_interrupt") is True


# ---------------------------------------------------------------------------
# Permission decisions (PermissionRequest + PermissionDenied)
# ---------------------------------------------------------------------------


def test_permission_request() -> None:
    """PermissionRequest captures proposed action."""
    payload: JSONDocument = {
        "tool_name": "Bash",
        "proposed_command": "rm -rf /tmp/test",
    }
    record: JSONValue = _make_hook_record(
        event_type="PermissionRequest",
        provider="claude-code",
        payload=payload,
    )

    artifact = classify_artifact(record, provider="claude-code")
    assert artifact.kind is ArtifactKind.HOOK_EVENT

    assert isinstance(record, dict)
    inner = record.get("payload")
    assert isinstance(inner, dict)
    assert inner.get("tool_name") == "Bash"
    assert inner.get("proposed_command") == "rm -rf /tmp/test"


def test_permission_denied() -> None:
    """PermissionDenied captures blocked action."""
    payload: JSONDocument = {
        "tool_name": "Bash",
    }
    record: JSONValue = _make_hook_record(
        event_type="PermissionDenied",
        provider="claude-code",
        payload=payload,
    )

    artifact = classify_artifact(record, provider="claude-code")
    assert artifact.kind is ArtifactKind.HOOK_EVENT

    assert isinstance(record, dict)
    inner = record.get("payload")
    assert isinstance(inner, dict)
    assert inner.get("tool_name") == "Bash"


# ---------------------------------------------------------------------------
# Session lifecycle events
# ---------------------------------------------------------------------------


def test_session_start_metadata() -> None:
    """SessionStart captures cwd, model, permission_mode."""
    payload: JSONDocument = {
        "session_id": "test-session-001",
        "cwd": "/home/user/project",
        "model": "claude-sonnet-4-20250514",
        "permission_mode": "default",
    }
    record: JSONValue = _make_hook_record(
        event_type="SessionStart",
        provider="claude-code",
        payload=payload,
    )

    artifact = classify_artifact(record, provider="claude-code")
    assert artifact.kind is ArtifactKind.HOOK_EVENT


def test_stop_event() -> None:
    """Stop captures session end reason."""
    payload: JSONDocument = {
        "session_id": "test-session-001",
        "reason": "user_exit",
    }
    record: JSONValue = _make_hook_record(
        event_type="Stop",
        provider="claude-code",
        payload=payload,
    )

    artifact = classify_artifact(record, provider="claude-code")
    assert artifact.kind is ArtifactKind.HOOK_EVENT


# ---------------------------------------------------------------------------
# Security / privacy boundaries
# ---------------------------------------------------------------------------


def test_local_path_handling() -> None:
    """Hook events with local file paths do not leak outside the archive."""
    payload: JSONDocument = {
        "file_path": "/home/user/secret/credentials.env",
        "diff_stats": {"lines_added": 0, "lines_removed": 1},
    }
    record: JSONValue = _make_hook_record(
        event_type="FileChanged",
        provider="claude-code",
        payload=payload,
    )

    artifact = classify_artifact(record, provider="claude-code")
    assert artifact.kind is ArtifactKind.HOOK_EVENT

    # Classification should never extract or expose the path contents
    assert "credentials.env" not in artifact.reason
    assert "secret" not in artifact.reason


def test_session_id_not_leaked_in_reason() -> None:
    """Classification reasons must not embed session IDs."""
    session_id = "very-sensitive-session-id-12345"
    record: JSONValue = _make_hook_record(session_id=session_id)
    artifact = classify_artifact(record, provider="claude-code")
    assert session_id not in artifact.reason


def test_payload_with_secrets_not_misclassified() -> None:
    """Hook payloads containing sensitive patterns still classify correctly."""
    payload: JSONDocument = {
        "tool_name": "Bash",
        "tool_input": {
            "command": "curl -H 'Authorization: Bearer sk-abc123' https://api.example.com",
        },
    }
    record: JSONValue = _make_hook_record(
        event_type="PreToolUse",
        provider="claude-code",
        payload=payload,
    )

    artifact = classify_artifact(record, provider="claude-code")
    assert artifact.kind is ArtifactKind.HOOK_EVENT

    # Classification reason should not embed the payload content
    assert "Bearer" not in artifact.reason
    assert "sk-abc123" not in artifact.reason


# ---------------------------------------------------------------------------
# Edge cases: codex session events
# ---------------------------------------------------------------------------


def test_codex_session_start() -> None:
    """Codex SessionStart with source field."""
    payload: JSONDocument = {
        "session_id": "codex-session-001",
        "cwd": "/home/user/project",
        "source": "manual",
    }
    record: JSONValue = _make_hook_record(
        event_type="SessionStart",
        provider="codex",
        payload=payload,
    )

    artifact = classify_artifact(record, provider="codex")
    assert artifact.kind is ArtifactKind.HOOK_EVENT


def test_codex_permission_request() -> None:
    """Codex PermissionRequest with proposed action."""
    payload: JSONDocument = {
        "proposed_action": "Execute shell command: npm install",
    }
    record: JSONValue = _make_hook_record(
        event_type="PermissionRequest",
        provider="codex",
        payload=payload,
    )

    artifact = classify_artifact(record, provider="codex")
    assert artifact.kind is ArtifactKind.HOOK_EVENT


# ---------------------------------------------------------------------------
# Hook stream precedence: hook events detected before record streams
# ---------------------------------------------------------------------------


def test_hook_event_stream_not_misclassified_as_record_stream() -> None:
    """Hook event streams must NOT be classified as record streams."""
    records: JSONValue = [
        _make_hook_record(event_type="PreToolUse"),
        _make_hook_record(event_type="PostToolUse"),
        _make_hook_record(event_type="PostToolUseFailure"),
    ]
    artifact = classify_artifact(records, provider="claude-code")
    assert artifact.kind is ArtifactKind.HOOK_EVENT


def test_hook_event_precedence_over_record_stream() -> None:
    """Hook event stream detection takes precedence over record stream."""
    records: JSONValue = [
        _make_hook_record(event_type="PreToolUse"),
        _make_hook_record(event_type="PostToolUse"),
    ]
    artifact = classify_artifact(records, provider="claude-code")
    # Must be HOOK_EVENT, not SESSION_RECORD_STREAM
    assert artifact.kind is ArtifactKind.HOOK_EVENT
    # Verify it's not a record stream
    assert artifact.kind.value != ArtifactKind.SESSION_RECORD_STREAM.value
