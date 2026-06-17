"""Batch ``[Pasted text #N]`` paste-marker detection and persistence.

Claude Code elides pasted content in the persisted JSONL, leaving a
``[Pasted text #N]`` marker in the user prompt. The live UserPromptSubmit hook
captures the same paste with a real content hash; batch re-ingest recovers the
marker's exact location and records it as a ``projected`` paste span. These
tests pin both the parser producer and the writer persistence.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.sources.dispatch import parse_payload
from polylogue.sources.parsers.base import ParsedMessage, ParsedPasteEvidence, ParsedSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def test_parse_code_detects_paste_marker_in_user_prompt() -> None:
    records = [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": "paste-sess",
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {"role": "user", "content": "Review [Pasted text #1 +12 lines] and fix it"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "timestamp": "2026-01-01T00:00:05Z",
            # An assistant message that quotes the marker must NOT be flagged.
            "message": {"role": "assistant", "content": "I see the [Pasted text #1] reference."},
        },
    ]

    parsed = parse_payload("claude-code", records, fallback_id="paste-sess")
    assert len(parsed) == 1
    session = parsed[0]

    user_messages = [m for m in session.messages if m.role == Role.USER]
    assert len(user_messages) == 1
    spans = user_messages[0].paste_spans
    assert len(spans) == 1
    assert spans[0].position == 1
    assert spans[0].boundary_state == "projected"
    assert spans[0].source_marker == "[Pasted text #1 +12 lines]"

    assistant_messages = [m for m in session.messages if m.role == Role.ASSISTANT]
    assert assistant_messages
    assert all(not m.paste_spans for m in assistant_messages)


def test_multiple_markers_in_one_prompt() -> None:
    records = [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": "paste-multi",
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {"role": "user", "content": "Compare [Pasted text #1] with [Pasted text #2 +3 lines]"},
        },
    ]

    session = parse_payload("claude-code", records, fallback_id="paste-multi")[0]
    spans = session.messages[0].paste_spans
    assert [s.position for s in spans] == [1, 2]
    assert all(s.boundary_state == "projected" for s in spans)


def test_paste_spans_roundtrip_through_writer(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    session = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="paste-rt",
        title="Paste round-trip",
        created_at="2026-01-01T00:00:00+00:00",
        messages=[
            ParsedMessage(
                provider_message_id="u1",
                role=Role.USER,
                text="See [Pasted text #1 +7 lines] for the trace",
                position=0,
                paste_spans=[
                    ParsedPasteEvidence(
                        position=1,
                        start_offset=4,
                        end_offset=30,
                        boundary_state="projected",
                        source_marker="[Pasted text #1 +7 lines]",
                    )
                ],
            ),
            ParsedMessage(provider_message_id="a1", role=Role.ASSISTANT, text="On it.", position=1),
        ],
    )

    session_id = write_parsed_session_to_archive(conn, session)

    has_paste, paste_boundary = conn.execute(
        "SELECT has_paste, paste_boundary FROM messages WHERE native_id = ?",
        ("u1",),
    ).fetchone()
    assert has_paste == 1
    assert paste_boundary == "projected"

    span_count = conn.execute(
        "SELECT COUNT(*) FROM paste_spans WHERE session_id = ?",
        (session_id,),
    ).fetchone()[0]
    assert span_count == 1

    paste_count = conn.execute(
        "SELECT paste_count FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()[0]
    assert paste_count == 1

    assistant_has_paste = conn.execute(
        "SELECT has_paste, paste_boundary FROM messages WHERE native_id = ?",
        ("a1",),
    ).fetchone()
    assert assistant_has_paste == (0, None)
