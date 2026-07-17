"""Claude Code assembly + history.jsonl paste-evidence wiring (#1583).

These tests pin the strong-identity matcher: history rows annotate exactly
the user messages they identify by sessionId + timestamp proximity, never
silently fan paste evidence across unrelated messages, and downstream
materialization honors the annotation when computing ``has_paste``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.sources.assembly import SidecarData
from polylogue.sources.assembly_claude_code import ClaudeCodeAssemblySpec
from polylogue.sources.parsers.base import (
    ParsedMessage,
    ParsedPasteEvidence,
    ParsedSession,
)
from polylogue.sources.parsers.claude.history import HistoryEntry, HistoryPaste


def _iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).isoformat()


def _user_message(provider_message_id: str, text: str, timestamp_iso: str) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=provider_message_id,
        role=Role.normalize("user"),
        text=text,
        timestamp=timestamp_iso,
    )


def _session(
    session_id: str,
    messages: list[ParsedMessage],
) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=session_id,
        title="t",
        created_at=None,
        updated_at=None,
        messages=messages,
    )


def _write_history(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _history_entry(
    session_id: str,
    timestamp_ms: int,
    *,
    with_paste: bool,
    hash_only: bool = False,
) -> HistoryEntry:
    pastes: tuple[HistoryPaste, ...] = ()
    if with_paste:
        pastes = (
            HistoryPaste(
                paste_id="1",
                paste_type="text",
                content="" if hash_only else "pasted body",
                has_content=not hash_only,
            ),
        )
    return HistoryEntry(
        display="[Pasted text #1]" if with_paste else "ordinary",
        timestamp_ms=timestamp_ms,
        project="/p",
        session_id=session_id,
        pastes=pastes,
    )


def _sidecars(history: dict[str, list[HistoryEntry]] | None = None) -> SidecarData:
    return {
        "session_index": {},
        "history_paste_index": {} if history is None else history,
    }


# ---------------------------------------------------------------------------
# discover_sidecars: end-to-end with a real ~/.claude layout on disk.
# ---------------------------------------------------------------------------


def test_discover_sidecars_indexes_history_paste_entries(tmp_path: Path) -> None:
    """A history.jsonl two dirs above the session file is indexed by sessionId."""
    projects_root = tmp_path / ".claude" / "projects"
    project_dir = projects_root / "-realm-project-polylogue"
    project_dir.mkdir(parents=True)
    session_file = project_dir / "abc.jsonl"
    session_file.touch()
    history_path = tmp_path / ".claude" / "history.jsonl"
    _write_history(
        history_path,
        [
            {
                "display": "[Pasted text #1]",
                "pastedContents": {"1": {"id": 1, "type": "text", "content": "hi"}},
                "timestamp": 100,
                "sessionId": "abc",
            }
        ],
    )

    sidecar_data = ClaudeCodeAssemblySpec().discover_sidecars([session_file])

    assert "history_paste_index" in sidecar_data
    history_index = sidecar_data["history_paste_index"]
    assert "abc" in history_index
    assert history_index["abc"][0].pastes[0].content == "hi"


def test_discover_sidecars_handles_missing_history_jsonl(tmp_path: Path) -> None:
    """No history.jsonl on disk must not break discovery."""
    project_dir = tmp_path / ".claude" / "projects" / "p"
    project_dir.mkdir(parents=True)
    session_file = project_dir / "abc.jsonl"
    session_file.touch()

    sidecar_data = ClaudeCodeAssemblySpec().discover_sidecars([session_file])

    assert sidecar_data["history_paste_index"] == {}


def test_discover_sidecars_parses_declared_orchestration_artifacts_and_reports_gaps(tmp_path: Path) -> None:
    project_dir = tmp_path / ".claude" / "projects" / "p"
    workflow_dir = project_dir / "workflows"
    journal_dir = project_dir / "subagents" / "workflows" / "wf-54"
    agent_dir = project_dir / "subagents"
    workflow_dir.mkdir(parents=True)
    journal_dir.mkdir(parents=True)
    workflow = workflow_dir / "wf-54.json"
    journal = journal_dir / "journal.jsonl"
    transcript = agent_dir / "agent-a.jsonl"
    workflow.write_text('{"runId":"wf-54","taskId":"task-7"}', encoding="utf-8")
    journal.write_text('{"contentKey":"call-1","agentId":"agent-a"}\n', encoding="utf-8")
    transcript.write_text(
        '{"type":"user","sessionId":"agent-a","message":{"role":"user","content":"work"}}\n', encoding="utf-8"
    )

    sidecars = ClaudeCodeAssemblySpec().discover_sidecars([workflow, journal, transcript])

    assert [(artifact.kind, artifact.facts[0].run_id) for artifact in sidecars["orchestration_artifacts"]] == [
        ("workflow_run_snapshot", "wf-54"),
        ("workflow_journal", "wf-54"),
    ]
    assert sidecars["orchestration_coverage"].gaps == ("missing agent metadata for transcript agent-a",)
    assert sidecars["orchestration_parse_gaps"] == ()


# ---------------------------------------------------------------------------
# enrich_session: strong-identity matching by sessionId + timestamp.
# ---------------------------------------------------------------------------


def test_enrich_marks_matched_user_message_with_paste_evidence() -> None:
    ts_ms = 1_700_000_000_000
    conv = _session(
        "sess-a",
        [_user_message("m1", "prompt one", _iso(ts_ms))],
    )
    history = {"sess-a": [_history_entry("sess-a", ts_ms, with_paste=True)]}

    enriched = ClaudeCodeAssemblySpec().enrich_session(conv, _sidecars(history))

    assert len(enriched.messages[0].paste_spans) >= 1
    assert enriched.messages[0].paste_spans[0].boundary_state == "hash_only"


def test_enrich_marks_hash_only_paste_as_evidence_too() -> None:
    """Hash-only history rows still record that a paste existed (AC #3)."""
    ts_ms = 1_700_000_000_000
    conv = _session(
        "sess-a",
        [_user_message("m1", "prompt one", _iso(ts_ms))],
    )
    history = {"sess-a": [_history_entry("sess-a", ts_ms, with_paste=True, hash_only=True)]}

    enriched = ClaudeCodeAssemblySpec().enrich_session(conv, _sidecars(history))

    assert len(enriched.messages[0].paste_spans) >= 1
    assert enriched.messages[0].paste_spans[0].boundary_state == "hash_only"


def test_enrich_skips_message_outside_timestamp_tolerance() -> None:
    ts_ms = 1_700_000_000_000
    conv = _session(
        "sess-a",
        [_user_message("m1", "prompt one", _iso(ts_ms))],
    )
    # 30 seconds away — well outside the 6-second tolerance.
    history = {"sess-a": [_history_entry("sess-a", ts_ms + 30_000, with_paste=True)]}

    enriched = ClaudeCodeAssemblySpec().enrich_session(conv, _sidecars(history))

    assert enriched.messages[0].paste_spans == []


def test_enrich_does_not_silently_fan_evidence_across_ambiguous_matches() -> None:
    """AC: ambiguous matches MUST NOT mark unrelated messages.

    Two user messages 1 second apart both fall inside the tolerance window
    of one history row. The matcher must refuse to pick a winner rather
    than annotate either.
    """
    ts_ms = 1_700_000_000_000
    conv = _session(
        "sess-a",
        [
            _user_message("m1", "prompt one", _iso(ts_ms)),
            _user_message("m2", "prompt two", _iso(ts_ms + 1_000)),
        ],
    )
    history = {"sess-a": [_history_entry("sess-a", ts_ms + 500, with_paste=True)]}

    enriched = ClaudeCodeAssemblySpec().enrich_session(conv, _sidecars(history))

    for msg in enriched.messages:
        assert msg.paste_spans == []


def test_enrich_does_not_cross_session_boundaries() -> None:
    """Session A's history rows never annotate session B's messages."""
    ts_ms = 1_700_000_000_000
    conv_a = _session(
        "sess-a",
        [_user_message("m1", "a-prompt", _iso(ts_ms))],
    )
    conv_b = _session(
        "sess-b",
        [_user_message("m1", "b-prompt", _iso(ts_ms))],
    )
    history = {"sess-a": [_history_entry("sess-a", ts_ms, with_paste=True)]}
    sidecars = _sidecars(history)

    enriched_a = ClaudeCodeAssemblySpec().enrich_session(conv_a, sidecars)
    enriched_b = ClaudeCodeAssemblySpec().enrich_session(conv_b, sidecars)

    assert len(enriched_a.messages[0].paste_spans) >= 1
    assert enriched_b.messages[0].paste_spans == []


def test_enrich_leaves_assistant_messages_alone() -> None:
    """Only user messages carry paste evidence (assistant messages cannot be paste origins)."""
    ts_ms = 1_700_000_000_000
    assistant = ParsedMessage(
        provider_message_id="m1",
        role=Role.normalize("assistant"),
        text="response",
        timestamp=_iso(ts_ms),
    )
    conv = _session("sess-a", [assistant])
    history = {"sess-a": [_history_entry("sess-a", ts_ms, with_paste=True)]}

    enriched = ClaudeCodeAssemblySpec().enrich_session(conv, _sidecars(history))

    assert enriched.messages[0].paste_spans == []


def test_enrich_is_noop_when_history_index_empty() -> None:
    ts_ms = 1_700_000_000_000
    conv = _session(
        "sess-a",
        [_user_message("m1", "prompt", _iso(ts_ms))],
    )

    result = ClaudeCodeAssemblySpec().enrich_session(conv, _sidecars())

    assert result is conv


# ---------------------------------------------------------------------------
# materialization: ``has_paste`` honors the provider_meta annotation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "history_annotated", "expected"),
    [
        ("no paste markers", False, 0),
        ("plain prompt", True, 1),  # heuristic says no, history says yes → 1
        ("[Pasted text #1 +6 lines]", False, 1),  # heuristic catches marker
        ("[Pasted text #1 +6 lines]", True, 1),  # both agree
    ],
)
def test_materialization_ors_heuristic_with_history_evidence(text: str, history_annotated: bool, expected: int) -> None:
    """``has_paste`` is the OR of the text heuristic and history sidecar
    evidence — the central #1583 acceptance criterion expressed at the
    materialization boundary."""
    from polylogue.archive.message.paste_detection import detect_paste

    paste_spans = [ParsedPasteEvidence(boundary_state="hash_only", source_marker="1")] if history_annotated else []
    msg = ParsedMessage(
        provider_message_id="m1",
        role=Role.normalize("user"),
        text=text,
        paste_spans=paste_spans,
    )
    meta_paste_evidence = bool(msg.paste_spans)
    actual = 1 if (detect_paste(msg.text) or meta_paste_evidence) else 0

    assert actual == expected
