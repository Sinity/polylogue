"""Preserve exact event anchors and invocation-owned file-edit evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.core.enums import BlockType, Provider, Role
from polylogue.core.message_owner import MessageOwnerAmbiguityError
from polylogue.pipeline.ids import session_content_hash, session_revision_projection
from polylogue.sources.parsers.base import (
    ParsedContentBlock,
    ParsedFileEdit,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)
from polylogue.sources.parsers.codex import parse as parse_codex
from polylogue.sources.prepared_message_sink import SqliteMessageStore
from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.identity import archive_message_id
from tests.infra.live_ingest import ingest_session


@pytest.mark.parametrize("parent_proved", [False, True])
async def test_reused_tool_ids_keep_each_file_edit_on_its_call(tmp_path: Path, parent_proved: bool) -> None:
    backend = SQLiteBackend(db_path=tmp_path / "edits.db")
    repo = SessionRepository(backend=backend)
    messages: list[ParsedMessage] = []
    for suffix in ("a", "b"):
        messages.extend(
            [
                ParsedMessage(
                    provider_message_id=f"use-{suffix}",
                    role=Role.ASSISTANT,
                    blocks=[
                        ParsedContentBlock(
                            type=BlockType.TOOL_USE,
                            tool_name="Edit",
                            tool_id="same",
                            tool_input={"file_path": f"{suffix}.txt"},
                        )
                    ],
                ),
                ParsedMessage(
                    provider_message_id=f"result-{suffix}",
                    parent_message_provider_id=f"use-{suffix}" if parent_proved else None,
                    role=Role.TOOL,
                    blocks=[
                        ParsedContentBlock(
                            type=BlockType.TOOL_RESULT,
                            tool_id="same",
                            is_error=False,
                            file_edit=ParsedFileEdit(file_path=f"{suffix}.txt", old_string="before", new_string=suffix),
                        )
                    ],
                ),
            ]
        )
    if parent_proved:
        # Parent evidence still selects the first call when the second call
        # precedes that result in the export.
        messages = [messages[0], messages[2], messages[1], messages[3]]
    try:
        session_id = await ingest_session(
            ParsedSession(source_name=Provider.CLAUDE_CODE, provider_session_id="edits", messages=messages),
            backend=backend,
        )
        edits = await repo.get_file_edits(session_id)
        async with backend.connection() as conn:
            rows = await conn.execute_fetchall("SELECT block_id,message_id FROM blocks WHERE block_type='tool_use'")
        uses = {row["message_id"]: row["block_id"] for row in rows}
        assert len(edits) == 2
        assert {(edit.tool_use_block_id, edit.message_id, edit.file_path, edit.new_string) for edit in edits} == {
            (
                uses[archive_message_id(session_id, f"use-{suffix}")],
                archive_message_id(session_id, f"result-{suffix}"),
                f"{suffix}.txt",
                suffix,
            )
            for suffix in ("a", "b")
        }
    finally:
        await repo.close()


async def test_unproved_reused_tool_id_cannot_publish_a_guessed_edit(tmp_path: Path) -> None:
    backend = SQLiteBackend(db_path=tmp_path / "ambiguous.db")
    repo = SessionRepository(backend=backend)
    try:
        session = ParsedSession(
            source_name=Provider.CLAUDE_CODE,
            provider_session_id="ambiguous",
            messages=[
                ParsedMessage(
                    provider_message_id="use-a",
                    role=Role.ASSISTANT,
                    blocks=[
                        ParsedContentBlock(type=BlockType.TOOL_USE, tool_id="same", tool_name="Edit", tool_input={})
                    ],
                ),
                ParsedMessage(
                    provider_message_id="use-b",
                    role=Role.ASSISTANT,
                    blocks=[
                        ParsedContentBlock(type=BlockType.TOOL_USE, tool_id="same", tool_name="Edit", tool_input={})
                    ],
                ),
                ParsedMessage(
                    provider_message_id="result",
                    role=Role.TOOL,
                    blocks=[
                        ParsedContentBlock(
                            type=BlockType.TOOL_RESULT,
                            tool_id="same",
                            is_error=False,
                            file_edit=ParsedFileEdit(file_path="neutral.txt", new_string="after"),
                        )
                    ],
                ),
            ],
        )
        with pytest.raises(MessageOwnerAmbiguityError):
            await ingest_session(session, backend=backend)
        assert await repo.get_file_edits("claude-code:ambiguous") == []
    finally:
        await repo.close()


@pytest.mark.parametrize("native", ["part:n:other:b:0", " café ", " cafe\u0301 ", "native\ud800", "native\ufffd"])
async def test_event_native_anchor_roundtrips_exactly(tmp_path: Path, native: str) -> None:
    backend = SQLiteBackend(db_path=tmp_path / "events.db")
    repo = SessionRepository(backend=backend)
    try:
        session_id = await ingest_session(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="events",
                messages=[ParsedMessage(provider_message_id=native, role=Role.USER, text="neutral")],
                session_events=[ParsedSessionEvent(event_type="neutral", source_message_provider_id=native)],
            ),
            backend=backend,
        )
        events = await repo.queries.get_session_events(session_id)
        assert len(events) == 1
        assert events[0].source_message_provider_id == native
        assert events[0].source_message_id == archive_message_id(session_id, native)
    finally:
        await repo.close()


@pytest.mark.parametrize("field", ["boundary_start_position", "boundary_end_position", "boundary_message_position"])
async def test_changed_event_boundary_changes_hash_and_public_read(tmp_path: Path, field: str) -> None:
    backend = SQLiteBackend(db_path=tmp_path / "boundary.db")
    repo = SessionRepository(backend=backend)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="boundary",
        messages=[
            ParsedMessage(provider_message_id=f"m{i}", role=Role.USER, text=f"neutral {i}", position=i)
            for i in range(3)
        ],
        session_events=[ParsedSessionEvent(event_type="compaction", payload={"summary": "neutral"})],
    )
    changed_event = session.session_events[0].model_copy(update={field: 0})
    changed = session.model_copy(update={"session_events": [changed_event]})
    assert session_content_hash(session) != session_content_hash(changed)
    assert session_revision_projection(session).event_contents != session_revision_projection(changed).event_contents
    store = SqliteMessageStore(tmp_path / "prepared.db")
    try:
        messages = store.new_sink()
        messages.extend(changed.messages)
        events = store.new_event_sink()
        events.extend(changed.session_events)
        spilled = changed.model_copy(update={"messages": messages, "session_events": events})
        assert session_content_hash(spilled) == session_content_hash(changed)
        disk_projection = session_revision_projection(spilled)
        try:
            assert set(disk_projection.event_contents) == set(session_revision_projection(changed).event_contents)
        finally:
            disk_projection.close()
    finally:
        store.close()
    try:
        session_id = await ingest_session(session, backend=backend)
        await ingest_session(changed, backend=backend)
        read = (await repo.queries.get_session_events(session_id))[0]
        if field == "boundary_message_position":
            assert read.boundary_message_id == archive_message_id(session_id, "m0")
        else:
            assert getattr(read, field) == 0
    finally:
        await repo.close()


async def test_codex_boundary_only_reexport_keeps_changed_compaction_geometry(tmp_path: Path) -> None:
    header = {"type": "session_meta", "payload": {"id": "neutral-boundary", "timestamp": "2026-01-01T00:00:00Z"}}
    records = [
        {
            "type": "response_item",
            "timestamp": "2026-01-01T00:00:01Z",
            "payload": {
                "type": "message",
                "id": f"m{i}",
                "role": "user",
                "content": [{"type": "input_text", "text": f"neutral {i}"}],
            },
        }
        for i in range(3)
    ]
    compaction = {"type": "compacted", "timestamp": "2026-01-01T00:00:02Z", "payload": {"message": ""}}
    first = parse_codex([header, records[0], records[1], compaction, header, records[2]], "neutral")
    changed = parse_codex([header, records[0], header, compaction, records[1], records[2]], "neutral")
    assert first.messages == changed.messages
    assert first.session_events[0].payload == changed.session_events[0].payload
    assert first.session_events[0].boundary_end_position == 1
    assert changed.session_events[0].boundary_end_position == 0
    assert session_content_hash(first) != session_content_hash(changed)
    assert session_revision_projection(first).event_contents != session_revision_projection(changed).event_contents
    backend = SQLiteBackend(db_path=tmp_path / "reexport.db")
    repo = SessionRepository(backend=backend)
    try:
        session_id = await ingest_session(first, backend=backend)
        await ingest_session(changed, backend=backend)
        read = await repo.queries.get_session_events(session_id)
        assert len(read) == 1
        assert read[0].boundary_end_position == 0
    finally:
        await repo.close()


@pytest.mark.parametrize("anchor", [None, ""])
async def test_absent_event_native_anchor_keeps_event_unowned(tmp_path: Path, anchor: str | None) -> None:
    backend = SQLiteBackend(db_path=tmp_path / "unowned.db")
    repo = SessionRepository(backend=backend)
    try:
        session_id = await ingest_session(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="unowned",
                messages=[ParsedMessage(provider_message_id="", role=Role.USER, text="neutral")],
                session_events=[ParsedSessionEvent(event_type="neutral", source_message_provider_id=anchor)],
            ),
            backend=backend,
        )
        read = await repo.queries.get_session_events(session_id)
        assert len(read) == 1
        assert read[0].source_message_id is None
        assert read[0].source_message_provider_id is None
    finally:
        await repo.close()
