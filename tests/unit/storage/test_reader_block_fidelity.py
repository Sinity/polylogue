from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.queries import attachments


@pytest.mark.asyncio
async def test_stored_block_fields_survive_ordinary_readers_and_bounded_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_active_archive_root(tmp_path)
    parsed = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="reader-blocks",
        messages=[
            ParsedMessage(
                provider_message_id=f"m{index}",
                role=Role.USER,
                blocks=[
                    ParsedContentBlock(type=BlockType.CODE, text=f"print({index})", metadata={"language": "python"}),
                    ParsedContentBlock(type=BlockType.IMAGE, text="neutral image", media_type="image/png"),
                ],
            )
            for index in range(105)
        ],
    )
    with ArchiveStore(tmp_path) as archive:
        session_id = archive.write_raw_and_parsed_result(
            parsed,
            payload=b"neutral reader blocks",
            source_path="reader-blocks.json",
            acquired_at_ms=1_767_000_000_000,
        ).session_id

    def check(messages: list[object]) -> None:
        for message in messages:
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
        assert context[0]["blocks"][0]["language"] == "python"
        assert context[0]["blocks"][1]["media_type"] == "image/png"

        batches: list[int] = []
        original_blocks = attachments.get_blocks

        async def count_blocks(conn, ids):
            batches.append(len(ids))
            return await original_blocks(conn, ids)

        monkeypatch.setattr(attachments, "get_blocks", count_blocks)
        streamed = [message async for message in api.iter_messages(session_id, limit=101)]
        check(streamed)
        assert len(streamed) == 101
        assert [message.id for message in streamed] == [message.id for message in single.messages][:101]
        assert batches == [100, 1]

        queries = api.repository.queries
        original_connection = queries._connection_factory
        active = 0

        @asynccontextmanager
        async def held_connection():
            nonlocal active
            async with original_connection() as conn:
                active += 1
                try:
                    yield conn
                finally:
                    active -= 1

        monkeypatch.setattr(queries, "_connection_factory", held_connection)
        batches.clear()
        stream = api.iter_messages(session_id)
        first = await anext(stream)
        check([first])
        assert active == 1
        assert batches == [100]
        await stream.aclose()
        assert active == 0
        assert batches == [100]

        batches.clear()
        custom = [
            message async for message in api.repository._backend.iter_messages(session_id, chunk_size=7, limit=15)
        ]
        assert len(custom) == 15
        assert [len(message.blocks) for message in custom] == [2] * 15
        assert batches == [7, 7, 1]
