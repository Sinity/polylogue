"""Regression tests for surfaced parse-time record loss (#1745).

Each test asserts that a loss path is *surfaced* — raised as a typed error,
counted accurately, or escalated to a durable status — rather than silently
dropping records while the parse reports success.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import cast

import pytest

from polylogue.sources.decoder_json import (
    JsonValue,
    PartialJsonStreamError,
    iter_json_stream,
)
from polylogue.sources.parsers.antigravity import (
    AntigravityBinaryUnavailableError,
    AntigravityExportError,
    AntigravityLanguageServerClient,
    AntigravityPartialExportError,
    AntigravitySessionSummary,
    iter_language_server_exports,
)
from polylogue.sources.parsers.codex import parse as parse_codex

# ---------------------------------------------------------------------------
# AC #? — dense 2025-era Codex rollout envelopes must not silently parse to
# zero messages (polylogue-i415).
# ---------------------------------------------------------------------------


def test_codex_dense_reasoning_and_tool_call_rollout_yields_messages() -> None:
    """A structurally real 2025-10 rollout shape must not parse to 0 messages.

    Forensic audit of the live archive (polylogue-i415) found 11 codex
    rollouts up to 3.3MB that materialized with ``message_count=0`` despite
    containing real ``response_item``/``message`` records, alongside a heavy
    stream of ``reasoning``/``function_call``/``custom_tool_call`` records
    (verified sample: 55 message, 440 function_call, 473 reasoning records,
    0 archived messages). Re-parsing the exact archived raw bytes with the
    current parser produces the expected message counts (e.g. 1023 messages
    for the 3.3MB sample) -- the loss was a *stale materialization* from an
    older parser version, not a defect in the current one. This fixture
    (record shapes copied verbatim from a real rollout, prose replaced with
    placeholders) locks that in so it cannot silently regress.
    """
    records: list[object] = [
        {
            "timestamp": "2025-10-19T05:04:30.400Z",
            "type": "session_meta",
            "payload": {
                "id": "357c7da6-8703-4ba3-8f70-f5f253571c12",
                "timestamp": "2025-10-19T05:04:30.397Z",
                "cwd": "/repo",
                "originator": "codex_cli_rs",
                "cli_version": "0.45.0",
                "instructions": "Repository guidelines placeholder.",
            },
        },
        {
            "timestamp": "2025-10-19T05:04:30.447Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Do the audit."}],
            },
        },
        {
            "timestamp": "2025-10-19T05:04:30.447Z",
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "Do the audit."},
        },
        {
            "timestamp": "2025-10-19T05:04:30.447Z",
            "type": "turn_context",
            "payload": {
                "cwd": "/repo",
                "approval_policy": "never",
                "sandbox_policy": {"mode": "danger-full-access"},
                "model": "gpt-5-codex",
                "effort": "high",
                "summary": "auto",
            },
        },
        {
            "timestamp": "2025-10-19T05:04:30.447Z",
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Formulating a plan."}],
                "content": None,
                "encrypted_content": "opaque-encrypted-blob",
            },
        },
        {
            "timestamp": "2025-10-19T05:04:30.447Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "update_plan",
                "call_id": "call_1",
                "arguments": '{"plan":[{"status":"in_progress","step":"audit"}]}',
            },
        },
        {
            "timestamp": "2025-10-19T05:04:30.447Z",
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Plan updated",
            },
        },
        {
            "timestamp": "2025-10-19T05:04:30.464Z",
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call",
                "status": "completed",
                "call_id": "call_2",
                "name": "apply_patch",
                "input": "*** Begin Patch\n*** Update File: x.txt\n@@\n-old\n+new\n*** End Patch",
            },
        },
        {
            "timestamp": "2025-10-19T05:04:30.464Z",
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call_output",
                "call_id": "call_2",
                "output": '{"output":"Success.","metadata":{"exit_code":0,"duration_seconds":0.0}}',
            },
        },
        {
            "timestamp": "2025-10-19T05:04:31.000Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Audit complete."}],
            },
        },
    ]

    session = parse_codex(records, "357c7da6-8703-4ba3-8f70-f5f253571c12")

    assert session.messages, (
        f"Expected non-empty messages from a dense response_item stream with real "
        f"message/reasoning/function_call/custom_tool_call records, got 0 from "
        f"{len(records)} records -- this is the exact silent-zero-message shape "
        f"from polylogue-i415."
    )
    texts = [message.text for message in session.messages]
    assert "Do the audit." in texts
    assert "Audit complete." in texts
    roles = [message.role.value for message in session.messages]
    assert roles[0] == "user"
    assert roles[-1] == "assistant"


# ---------------------------------------------------------------------------
# AC #1 — mid-stream JSON corruption raises a typed partial-decode error
# ---------------------------------------------------------------------------


def test_mid_stream_corruption_raises_partial_decode_error() -> None:
    """A top-level array valid for the first N items then truncated must raise.

    The previous behaviour returned the records accumulated before the
    corruption with ``found_any=True`` and logged nothing, silently truncating
    the session set.
    """
    # Valid first two objects, then the array is cut off mid-token.
    truncated = b'[{"id": 1}, {"id": 2}, {"id": 3'
    handle = io.BytesIO(truncated)

    with pytest.raises(PartialJsonStreamError) as excinfo:
        list(iter_json_stream(handle, "sessions.json"))

    err = excinfo.value
    assert err.recovered >= 2
    assert "sessions.json" in str(err)


def test_clean_array_does_not_raise() -> None:
    """A well-formed array must still decode all records without raising."""
    handle = io.BytesIO(b'[{"id": 1}, {"id": 2}, {"id": 3}]')
    records = list(iter_json_stream(handle, "sessions.json"))
    ids = [cast(dict[str, JsonValue], r)["id"] for r in records]
    assert ids == [1, 2, 3]


def test_wrong_prefix_with_zero_items_falls_through_not_raises() -> None:
    """A single top-level object (no array) must not raise PartialJsonStreamError.

    Strategy 1 ("item") finds zero items and a JSONError there is a normal
    "try the next strategy" signal — it must be swallowed, not surfaced.
    """
    handle = io.BytesIO(b'{"sessions": [{"id": 1}]}')
    records = list(iter_json_stream(handle, "single.json"))
    # The object is yielded as a single record (dict payload, no unpack match).
    assert records


# ---------------------------------------------------------------------------
# AC #4 — Antigravity distinguishes missing-binary from mid-export failure
# ---------------------------------------------------------------------------


class _FakeClient:
    """Minimal stand-in for AntigravityLanguageServerClient."""

    def __init__(self, summaries: list[AntigravitySessionSummary], fail_at: int | None) -> None:
        self._summaries = summaries
        self._fail_at = fail_at
        self.closed = False

    def start(self) -> None:  # pragma: no cover - trivial
        pass

    def close(self) -> None:
        self.closed = True

    def search_sessions(self) -> list[AntigravitySessionSummary]:
        return self._summaries

    def export_markdown(self, cascade_id: str) -> str:
        index = [s.cascade_id for s in self._summaries].index(cascade_id)
        if self._fail_at is not None and index >= self._fail_at:
            raise AntigravityExportError(f"export failed for {cascade_id}")
        return f"### User Input\nhello {cascade_id}\n"


def _as_client(fake: _FakeClient) -> AntigravityLanguageServerClient:
    return cast(AntigravityLanguageServerClient, fake)


def _touch_conversation_pb(root: Path, *cascade_ids: str) -> None:
    """Create empty ``conversations/<cascade_id>.pb`` files.

    Cascade discovery is ground-truthed off this directory listing
    (polylogue-eo81) rather than the language server's own search/list RPCs.
    """
    conversations = root / "conversations"
    conversations.mkdir(parents=True, exist_ok=True)
    for cascade_id in cascade_ids:
        (conversations / f"{cascade_id}.pb").write_bytes(b"")


def test_antigravity_export_error_taxonomy_is_distinguishable(tmp_path: Path) -> None:
    """Missing-binary and mid-export are distinct typed subtypes of the export error."""
    _touch_conversation_pb(tmp_path, "c1")
    summary = AntigravitySessionSummary(cascade_id="c1")
    client = _FakeClient([summary], fail_at=None)
    # Sanity: with a working client all sessions are obtained.
    convos = list(iter_language_server_exports(tmp_path, client=_as_client(client)))
    assert len(convos) == 1

    # Both are AntigravityExportError subtypes so the broad fallback still
    # catches them, but callers can distinguish benign vs. lossy.
    assert issubclass(AntigravityBinaryUnavailableError, AntigravityExportError)
    assert issubclass(AntigravityPartialExportError, AntigravityExportError)


def test_antigravity_mid_export_failure_reports_obtained_vs_expected(tmp_path: Path) -> None:
    """A mid-iteration failure raises AntigravityPartialExportError with counts.

    Previously the generator yielded the sessions seen before the failure
    then aborted into the fallback, indistinguishable from "not installed".
    """
    summaries = [AntigravitySessionSummary(cascade_id=f"c{i}") for i in range(5)]
    _touch_conversation_pb(tmp_path, *(s.cascade_id for s in summaries))
    client = _FakeClient(summaries, fail_at=3)

    obtained: list[object] = []
    with pytest.raises(AntigravityPartialExportError) as excinfo:
        for convo in iter_language_server_exports(tmp_path, client=_as_client(client)):
            obtained.append(convo)

    err = excinfo.value
    assert err.expected == 5
    assert err.obtained == 3
    assert len(obtained) == 3
    # The lost remainder is visible in the message.
    assert "3 of 5" in str(err)
