"""Contracts for the archive declaration-driven row and domain path."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.core.enums import BlockType, Origin, Provider, Role
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.hydrators import message_from_record
from polylogue.storage.runtime import SessionRecord
from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import BLOCKS_SPEC, MESSAGES_SPEC, SESSIONS_SPEC
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.queries import message_query_reads, sessions_reads
from tests.infra.identity import archive_message_id
from tests.infra.live_ingest import ingest_session


def test_archive_specs_have_unique_storage_and_record_projections() -> None:
    for spec in (MESSAGES_SPEC, BLOCKS_SPEC):
        storage_names = [column.name for column in spec.all_columns]
        record_names = [column.record_name for column in spec.record_columns]
        assert len(storage_names) == len(set(storage_names)), spec.table_name
        assert len(record_names) == len(set(record_names)), spec.table_name
        assert all(column.extract is not None for column in spec.writable_columns if column.extract_placeholder == "?")
        assert sum(column.name == "stop_reason" for column in spec.all_columns) <= 1


@pytest.mark.asyncio
async def test_real_archive_write_read_hydrates_spec_fields(tmp_path: Path) -> None:
    backend = SQLiteBackend(db_path=tmp_path / "index.db")
    parsed_session = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="spec-hydration",
        title="Spec hydration",
        messages=[
            ParsedMessage(
                provider_message_id="native-message",
                role=Role.ASSISTANT,
                text="finished",
                timestamp="2026-03-01T10:05:00+00:00",
                position=0,
                stop_reason="end_turn",
                is_active_path=True,
                is_active_leaf=True,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="finished")],
            )
        ],
    )
    try:
        session_id = await ingest_session(parsed_session, backend)
        async with backend.connection() as conn:
            records = await message_query_reads.get_messages(conn, session_id)
            blocks = await backend.get_blocks([records[0].message_id])
        record = records[0].model_copy(update={"blocks": blocks[str(records[0].message_id)]})
        hydrated = message_from_record(record, [], origin=records[0].source_name)
    finally:
        await backend.close()

    assert str(records[0].message_id) == archive_message_id(
        "claude-code-session:spec-hydration", "native-message", position=0
    )
    assert str(blocks[str(records[0].message_id)][0].block_id) == f"{records[0].message_id}:0"
    assert hydrated.stop_reason == "end_turn"
    assert hydrated.is_active_path is True
    assert hydrated.is_active_leaf is True
    assert hydrated.text == "finished"
    assert hydrated.timestamp is not None
    assert hydrated.timestamp.isoformat() == "2026-03-01T10:05:00+00:00"


def test_sessions_spec_declares_the_record_projection_the_mapper_consumes() -> None:
    """SESSIONS_SPEC must name every field ``_row_to_session`` builds a record from.

    Red when any ``record_name`` is dropped from SESSIONS_SPEC: an empty
    projection lowers to ``SELECT  FROM sessions``, and a partial one leaves a
    SessionRecord field with no column to read.
    """
    projection = SESSIONS_SPEC.record_select_column_names("sessions")
    assert projection.strip(), "sessions record projection is empty"

    declared = {column.record_name for column in SESSIONS_SPEC.record_columns}
    # The mapper builds these two fields from the raw JSON projections rather
    # than a same-named column.
    from_json_projection = {"run_settings": "run_settings_json", "pending_drafts": "pending_drafts_json"}
    missing = {
        field
        for field in SessionRecord.model_fields
        if field not in declared and from_json_projection.get(field) not in declared
    }
    assert not missing, f"SessionRecord fields with no sessions projection: {sorted(missing)}"


@pytest.mark.asyncio
async def test_session_reads_hydrate_records_through_the_declared_projection(tmp_path: Path) -> None:
    """``get_session`` and ``list_sessions`` return populated records.

    Red when the sessions projection is empty (both statements become invalid
    SQL) and when any single projected column loses its ``record_name`` (the
    corresponding record field falls back to its default).
    """
    backend = SQLiteBackend(db_path=tmp_path / "index.db")
    parsed_session = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="session-projection",
        title="Session projection",
        git_branch="feature/projection",
        messages=[
            ParsedMessage(
                provider_message_id="native-message",
                role=Role.USER,
                text="hello",
                timestamp="2026-03-01T10:05:00+00:00",
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hello")],
            )
        ],
    )
    try:
        session_id = await ingest_session(parsed_session, backend)
        async with backend.connection() as conn:
            record = await sessions_reads.get_session(conn, session_id)
            listed = await sessions_reads.list_sessions(conn, limit=10)
    finally:
        await backend.close()

    assert record is not None
    assert str(record.session_id) == session_id
    assert record.origin is Origin.CLAUDE_CODE_SESSION
    assert record.native_id == "session-projection"
    assert record.title == "Session projection"
    assert record.git_branch == "feature/projection"
    assert record.content_hash
    assert record.version == 1
    # The record-only projection is live: no working dirs still yields an array.
    assert record.working_directories_json == "[]"
    assert record.sort_key is not None
    assert record.updated_at is not None

    assert [str(item.session_id) for item in listed] == [session_id]
    assert listed[0].title == "Session projection"
