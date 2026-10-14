"""Projection windows classify against the complete transcript dependency scope."""

import asyncio
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import pytest

from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.archive.semantic.projection_context import ContentProjectionContext
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.live_ingest import write_index_session
from tests.unit.api.test_facade_contracts import _archive

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize(
    "role_filter,type_filter,expected_total", [((), None, 501), ((Role.USER,), None, 500), ((), "tool_result", 0)]
)
async def test_projected_window_indexes_later_tool_use_before_counting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role_filter: tuple[Role, ...],
    type_filter: Literal["tool_result"] | None,
    expected_total: int,
) -> None:
    """A result before its use crosses raw pages without becoming ordinary output."""
    from polylogue.archive.semantic.content_projection import ContentProjectionSpec

    archive = _archive(tmp_path)
    messages = [
        ParsedMessage(
            provider_message_id="early-result",
            role=Role.USER,
            message_type=MessageType.TOOL_RESULT,
            text="private file result",
            blocks=[
                ParsedContentBlock(
                    type=BlockType.TOOL_RESULT,
                    tool_id="later-read",
                    text="private file result",
                    outcome_unknown_reason="not_reported",
                )
            ],
        ),
        *[
            ParsedMessage(provider_message_id=f"prose-{position}", role=Role.USER, text=f"prose {position}")
            for position in range(500)
        ],
        ParsedMessage(
            provider_message_id="later-use",
            role=Role.ASSISTANT,
            message_type=MessageType.TOOL_USE,
            text=None,
            blocks=[
                ParsedContentBlock(
                    type=BlockType.TOOL_USE,
                    tool_name="Read",
                    tool_id="later-read",
                    tool_input={"file_path": "synthetic.txt"},
                )
            ],
        ),
    ]

    def seed() -> str:
        with ArchiveStore(tmp_path) as archive_db:
            return write_index_session(
                archive_db,
                ParsedSession(
                    source_name=Provider.CLAUDE_CODE,
                    provider_session_id="result-before-use",
                    messages=messages,
                ),
            )

    session_id = await asyncio.to_thread(seed)

    paths: list[Path] = []
    original_initialize = ContentProjectionContext.__init__

    def initialize(
        context: ContentProjectionContext, connection: sqlite3.Connection, checkpoint: Callable[[], None]
    ) -> None:
        original_initialize(context, connection, checkpoint)
        paths.append(Path(connection.execute("PRAGMA database_list").fetchone()[2]))

    monkeypatch.setattr(ContentProjectionContext, "__init__", initialize)
    try:
        rows, total, _ = await archive.get_messages_paginated(
            session_id,
            limit=1,
            message_role=role_filter,
            message_type=type_filter,
            content_projection=ContentProjectionSpec(include_file_reads=False),
        )
        physical = await archive.get_session_page(
            session_id, limit=1, content_projection=ContentProjectionSpec(include_file_reads=False)
        )
        assert physical is not None
        assert physical.offset == 0 and physical.limit == 1
        assert physical.total_message_count == 502
        assert list(physical.session.messages) == []
    finally:
        await archive.close()
    assert len(paths) == 2
    assert all(not path.parent.exists() for path in paths)
    assert total == expected_total
    if expected_total:
        assert len(rows) == 1
        assert rows[0].text == "prose 0"
    else:
        assert rows == []


@pytest.mark.parametrize("fault", ["cancelled", "restart", "late_error"])
async def test_physical_projection_late_fault_releases_private_dependency_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    import sqlite3

    from polylogue.archive.query.execution_control import QueryCancelledError
    from polylogue.archive.semantic.content_projection import ContentProjectionSpec

    archive = _archive(tmp_path)

    def seed() -> str:
        with ArchiveStore(tmp_path) as writer:
            return write_index_session(
                writer,
                ParsedSession(
                    source_name=Provider.CLAUDE_CODE,
                    provider_session_id="physical-projection-fault",
                    messages=[
                        ParsedMessage(provider_message_id=f"message-{index}", role=Role.USER, text="neutral prose")
                        for index in range(501)
                    ],
                ),
            )

    session_id = await asyncio.to_thread(seed)
    paths: list[Path] = []
    original_initialize = ContentProjectionContext.__init__

    def initialize(
        context: ContentProjectionContext, connection: sqlite3.Connection, checkpoint: Callable[[], None]
    ) -> None:
        original_initialize(context, connection, checkpoint)
        paths.append(Path(connection.execute("PRAGMA database_list").fetchone()[2]))

    monkeypatch.setattr(ContentProjectionContext, "__init__", initialize)
    original_read = ArchiveStore.read_session_page

    def read(store: ArchiveStore, session_id: str, *, limit: int, offset: int, **kwargs: Any) -> Any:
        if limit == 500 and offset == 500:
            if fault == "cancelled":
                raise QueryCancelledError("synthetic late dependency cancellation")
            if fault == "late_error":
                raise ValueError("synthetic late dependency error")
            connection = store.index_connection
            assert connection is not None
            connection.rollback()
            connection.execute("BEGIN")
        return original_read(store, session_id, limit=limit, offset=offset, **kwargs)

    monkeypatch.setattr(ArchiveStore, "read_session_page", read)
    try:
        error = (
            QueryCancelledError if fault == "cancelled" else sqlite3.DatabaseError if fault == "restart" else ValueError
        )
        with pytest.raises(error):
            await archive.get_session_page(
                session_id, limit=1, content_projection=ContentProjectionSpec(include_file_reads=False)
            )
    finally:
        await archive.close()
    assert len(paths) == 1
    assert not paths[0].parent.exists()


async def test_projected_dependency_scan_cancellation_releases_private_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.archive.semantic.content_projection import ContentProjectionSpec

    archive = _archive(tmp_path)

    def seed() -> str:
        with ArchiveStore(tmp_path) as writer:
            return write_index_session(
                writer,
                ParsedSession(
                    source_name=Provider.CLAUDE_CODE,
                    provider_session_id="projection-cancel",
                    messages=[
                        ParsedMessage(provider_message_id=f"message-{index}", role=Role.USER, text="neutral prose")
                        for index in range(501)
                    ],
                ),
            )

    session_id = await asyncio.to_thread(seed)
    paths: list[Path] = []
    original_register = ContentProjectionContext.register_page

    def register(context: ContentProjectionContext, messages: Sequence[Message], *, offset: int) -> None:
        original_register(context, messages, offset=offset)
        if offset == 0:
            paths.append(Path(context.connection.execute("PRAGMA database_list").fetchone()[2]))
            task = asyncio.current_task()
            assert task is not None
            task.cancel()

    monkeypatch.setattr(ContentProjectionContext, "register_page", register)
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.create_task(
                archive.get_messages_paginated(
                    session_id, limit=1, content_projection=ContentProjectionSpec(include_file_reads=False)
                )
            )
    finally:
        await archive.close()
    assert len(paths) == 1
    assert not paths[0].parent.exists()


async def test_projected_window_uses_terminal_tool_semantics_across_pages(tmp_path: Path) -> None:
    from polylogue.archive.semantic.content_projection import ContentProjectionSpec

    archive = _archive(tmp_path)

    def seed() -> str:
        with ArchiveStore(tmp_path) as writer:
            return write_index_session(
                writer,
                ParsedSession(
                    source_name=Provider.CLAUDE_CODE,
                    provider_session_id="terminal-projection-kind",
                    messages=[
                        ParsedMessage(
                            provider_message_id="first-use",
                            role=Role.ASSISTANT,
                            blocks=[
                                ParsedContentBlock(
                                    type=BlockType.TOOL_USE,
                                    tool_name="Read",
                                    tool_id="reused",
                                    tool_input={"file_path": "synthetic.txt"},
                                )
                            ],
                        ),
                        ParsedMessage(
                            provider_message_id="early-result",
                            role=Role.USER,
                            message_type=MessageType.TOOL_RESULT,
                            blocks=[
                                ParsedContentBlock(
                                    type=BlockType.TOOL_RESULT,
                                    tool_id="reused",
                                    text="neutral output",
                                    outcome_unknown_reason="not_reported",
                                )
                            ],
                        ),
                        *[
                            ParsedMessage(provider_message_id=f"prose-{index}", role=Role.USER, text="neutral prose")
                            for index in range(500)
                        ],
                        ParsedMessage(
                            provider_message_id="terminal-use",
                            role=Role.ASSISTANT,
                            blocks=[
                                ParsedContentBlock(
                                    type=BlockType.TOOL_USE,
                                    tool_name="Bash",
                                    tool_id="reused",
                                    tool_input={"command": "printf neutral"},
                                )
                            ],
                        ),
                    ],
                ),
            )

    session_id = await asyncio.to_thread(seed)
    try:
        rows, total, _ = await archive.get_messages_paginated(
            session_id,
            limit=1,
            message_type="tool_result",
            content_projection=ContentProjectionSpec(include_file_reads=False),
        )
        assert total == 1
        assert len(rows) == 1
        assert rows[0].text == "neutral output"
        physical = await archive.get_session_page(
            session_id, limit=1, offset=1, content_projection=ContentProjectionSpec(include_file_reads=False)
        )
        assert physical is not None
        assert physical.total_message_count == 503
        assert len(physical.session.messages) == 1
        assert list(physical.session.messages)[0].text == "neutral output"
    finally:
        await archive.close()
