from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
import pytest

from polylogue.api import Polylogue
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.sources.parsers.base import ParsedAttachment, ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.runtime import AttachmentRecord, BlockRecord
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.queries import attachments
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.prepared_replay import write_fixture_raw_session


@pytest.mark.asyncio
async def test_stored_block_fields_survive_ordinary_readers_and_bounded_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parsed = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="reader-blocks",
        messages=[
            ParsedMessage(
                provider_message_id=f"m{index}",
                role=Role.USER,
                material_origin=MaterialOrigin.HUMAN_AUTHORED if index % 2 else MaterialOrigin.RUNTIME_CONTEXT,
                blocks=[
                    ParsedContentBlock(type=BlockType.CODE, text=f"print({index})", metadata={"language": "python"}),
                    ParsedContentBlock(type=BlockType.IMAGE, text="neutral image", media_type="image/png"),
                ],
            )
            for index in range(105)
        ],
        attachments=[
            ParsedAttachment(
                provider_attachment_id=f"a{index}",
                message_provider_id=f"m{index}",
                name=f"neutral-{index}.txt",
                mime_type="text/plain",
                inline_bytes=b"neutral attachment",
                caption=f"caption {index}",
                direction="user_input",
                upload_origin="paste",
            )
            for index in range(105)
        ],
    )

    def seed() -> str:
        initialize_active_archive_root(tmp_path)
        with ArchiveStore(tmp_path) as archive:
            return write_fixture_raw_session(
                archive,
                parsed,
                payload=b"neutral reader blocks",
                source_path="reader-blocks.json",
                acquired_at_ms=1_767_000_000_000,
            ).session_id

    session_id = run_off_event_loop(seed)

    def check(messages: Sequence[Message]) -> None:
        for message in messages:
            assert len(message.attachments) == 1
            attachment = message.attachments[0]
            assert attachment.mime_type == "text/plain"
            assert attachment.name is not None
            assert attachment.name.startswith("neutral-")
            assert attachment.caption is not None
            assert attachment.availability is not None
            assert attachment.availability.state.value == "available"
            blocks = message.blocks
            assert [(block["type"], block["language"], block["media_type"]) for block in blocks] == [
                ("code", "python", None),
                ("image", None, "image/png"),
            ]

    async with Polylogue(archive_root=tmp_path) as api:
        single = await api.repository.get(session_id)
        assert single is not None
        check(list(single.messages))
        batch = await api.repository.get_many([session_id])
        check(list(batch[0].messages))
        context = await api.get_effective_context(session_id)
        assert context is not None
        context_blocks = context[0]["blocks"]
        assert isinstance(context_blocks, list)
        assert context_blocks[0]["language"] == "python"
        assert context_blocks[1]["media_type"] == "image/png"

        page, total, completeness = await api.repository.get_messages_paginated(session_id, limit=2)
        check(page)
        assert total == 105 and completeness.complete
        context_attachments = context[0]["attachments"]
        assert isinstance(context_attachments, list)
        assert isinstance(context_attachments[0], dict)
        assert context_attachments[0]["mime_type"] == "text/plain"

        batches: list[int] = []
        attachment_batches: list[int] = []
        original_attachments = attachments.get_message_attachments

        async def count_attachments(conn: aiosqlite.Connection, ids: list[str]) -> dict[str, list[AttachmentRecord]]:
            assert conn.in_transaction
            attachment_batches.append(len(ids))
            return await original_attachments(conn, ids)

        monkeypatch.setattr(attachments, "get_message_attachments", count_attachments)
        original_blocks = attachments.get_blocks

        async def count_blocks(conn: aiosqlite.Connection, ids: list[str]) -> dict[str, list[BlockRecord]]:
            batches.append(len(ids))
            return await original_blocks(conn, ids)

        monkeypatch.setattr(attachments, "get_blocks", count_blocks)
        streamed = [message async for message in api.iter_messages(session_id, limit=101)]
        check(streamed)
        assert len(streamed) == 101
        assert [message.id for message in streamed] == [message.id for message in single.messages][:101]
        assert batches == [100, 1]
        assert attachment_batches == [100, 1]

        queries = api.repository.queries
        original_connection = queries._connection_factory
        active = 0

        @asynccontextmanager
        async def held_connection() -> AsyncIterator[aiosqlite.Connection]:
            nonlocal active
            async with original_connection() as conn:
                active += 1
                try:
                    yield conn
                finally:
                    active -= 1

        monkeypatch.setattr(queries, "_connection_factory", held_connection)
        batches.clear()
        attachment_batches.clear()
        stream = api.iter_messages(session_id)
        first = await anext(stream)
        check([first])
        assert active == 1
        assert batches == [100]
        await stream.aclose()
        assert active == 0
        assert batches == [100]
        assert attachment_batches == [100]

        batches.clear()
        custom = [
            message async for message in api.repository._backend.iter_messages(session_id, chunk_size=7, limit=15)
        ]
        assert len(custom) == 15
        assert [len(message.blocks) for message in custom] == [2] * 15
        assert batches == [7, 7, 1]

        async def forbidden_full_session(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("a filtered iterator hydrated the entire session")

        monkeypatch.setattr(api, "get_session", forbidden_full_session)
        filtered = [
            message
            async for message in api.iter_messages(
                session_id, material_origin=(MaterialOrigin.HUMAN_AUTHORED,), limit=3
            )
        ]
        assert [message.id for message in filtered] == [
            message.id for message in single.messages if message.material_origin == MaterialOrigin.HUMAN_AUTHORED
        ][:3]
        check(filtered)
