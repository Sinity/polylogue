"""Contracts for the archive declaration-driven row and domain path."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.core.enums import BlockType, Provider, Role
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.hydrators import message_from_record
from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import BLOCKS_SPEC, MESSAGES_SPEC
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.queries import message_query_reads
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
