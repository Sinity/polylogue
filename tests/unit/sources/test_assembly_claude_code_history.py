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


# ---------------------------------------------------------------------------
# polylogue-ximhz: the same paste evidence must be reachable from retained
# bytes, without the original ``history.jsonl`` path.
#
# Anti-vacuity: make ``build_session_paste_index_bytes`` return ``{}`` (or
# revert ``parse_history_jsonl_bytes`` to requiring a live path) and
# ``test_retained_history_bytes_rebuild_the_same_paste_index`` goes red; the
# full production-route proof that a replay recovers the span with the tree
# deleted is
# ``tests/unit/pipeline/test_ingest_worker_assembly.py::test_claude_index_and_history_resolve_with_the_original_tree_gone``.
# ---------------------------------------------------------------------------


def test_retained_history_bytes_rebuild_the_same_paste_index(tmp_path: Path) -> None:
    """Reading the acquired blob is equivalent to reading the original file."""
    from polylogue.sources.parsers.claude.history import (
        build_session_paste_index,
        build_session_paste_index_bytes,
    )

    history = tmp_path / "history.jsonl"
    rows = [
        {
            "display": "first prompt",
            "timestamp": 1784541600000,
            "sessionId": "session-a",
            "pastedContents": {"1": {"type": "text", "content": "pasted body"}},
        },
        {"display": "no paste here", "timestamp": 1784541601000, "sessionId": "session-a"},
    ]
    payload = ("\n".join(json.dumps(row) for row in rows) + "\n").encode("utf-8")
    history.write_bytes(payload)

    from_path = build_session_paste_index(history)
    history.unlink()
    from_bytes = build_session_paste_index_bytes(payload, origin=str(history))

    assert list(from_bytes) == ["session-a"]
    assert from_bytes == from_path


def test_retained_claude_coordinates_follow_the_session_install(tmp_path: Path) -> None:
    """Scope is identity: each install anchors on its own index and history."""
    from polylogue.sources.retained_assembly import claude_code_sidecar_coordinates

    first = tmp_path / "install-a" / ".claude" / "projects" / "-p" / "s.jsonl"
    second = tmp_path / "install-b" / ".claude" / "projects" / "-p" / "s.jsonl"

    first_coordinates = claude_code_sidecar_coordinates(str(first))
    second_coordinates = claude_code_sidecar_coordinates(str(second))
    assert first_coordinates is not None and second_coordinates is not None
    first_index, first_history = first_coordinates
    second_index, second_history = second_coordinates

    assert first_index == str(first.parent / "sessions-index.json")
    assert first_history == str(tmp_path / "install-a" / ".claude" / "history.jsonl")
    assert first_index != second_index
    assert first_history != second_history


@pytest.mark.asyncio
async def test_retained_raw_replay_resolves_the_curated_title(tmp_path: Path) -> None:
    """The offline replay chokepoint reads the same retained evidence.

    ``sources/revision_backfill.py`` is a separate parse chokepoint from the
    daemon's ingest worker. Before polylogue-ximhz it supplied only retained
    Codex state titles, so a rebuild of a Claude Code session fell back to the
    first user message even though the archive still held the index bytes.

    Anti-vacuity: drop the ``source_conn``/``blob_root`` arguments from
    ``_enrich_retained_parse_results``'s call to
    ``_replay_safe_enrich_sessions`` and this returns to the heuristic title.
    """
    import sqlite3

    import polylogue.sources.live.watcher as live_watcher
    from polylogue import Polylogue
    from polylogue.core.enums import TitleSource
    from polylogue.sources.dispatch import parse_stream_payload
    from polylogue.sources.live import WatchSource
    from polylogue.sources.live.batch import LiveBatchProcessor
    from polylogue.sources.live.cursor import CursorStore
    from polylogue.sources.origin_specs import artifact_suffixes_for_provider
    from polylogue.sources.revision_backfill import _replay_safe_enrich_sessions

    archive_root = tmp_path / "archive"
    project = tmp_path / "live" / ".claude" / "projects" / "-realm-project-x"
    project.mkdir(parents=True)
    session_id = "aaaaaaaa-1111-2222-3333-444444444444"
    transcript = project / f"{session_id}.jsonl"
    records = [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": session_id,
            "timestamp": "2026-07-20T10:00:00.000Z",
            "message": {"role": "user", "content": "first prompt"},
        }
    ]
    transcript.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    index_path = project / "sessions-index.json"
    index_path.write_text(
        json.dumps(
            {"entries": [{"sessionId": session_id, "fullPath": str(transcript), "summary": "Curated index title"}]}
        ),
        encoding="utf-8",
    )

    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    processor = LiveBatchProcessor(
        archive,
        (
            WatchSource(
                name="claude-code",
                root=project.parent,
                suffixes=artifact_suffixes_for_provider(Provider.CLAUDE_CODE, defaults=(".jsonl",)),
            ),
        ),
        cursor=CursorStore(archive_root / "cursor.db"),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    try:
        await processor.ingest_files([index_path], emit_event=False)
    finally:
        await archive.close()

    index_path.unlink()
    transcript.unlink()

    sessions = parse_stream_payload(Provider.CLAUDE_CODE, iter(records), session_id, source_path=str(transcript))
    assert sessions[0].title_source is TitleSource.HEURISTIC

    conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    try:
        enriched = _replay_safe_enrich_sessions(
            provider=Provider.CLAUDE_CODE,
            sessions=sessions,
            index_conn=None,
            source_conn=conn,
            blob_root=archive_root / "blob",
            source_path=str(transcript),
        )
    finally:
        conn.close()

    assert enriched[0].title == "Curated index title"
    assert enriched[0].title_source is TitleSource.ORIGIN
