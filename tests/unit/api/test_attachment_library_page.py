"""Contract coverage for the bounded attachment-library facade seam."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from polylogue import Polylogue
from polylogue.storage.io_phase_metrics import connect_measured
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.index_writer import write_fixture_index_session

if TYPE_CHECKING:
    from polylogue.services import RuntimeServices
    from polylogue.storage.runtime import AttachmentRecord


class _Repository:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def get_attachment_library_page(self, **kwargs: object) -> list[tuple[object, str, str | None]]:
        self.calls.append(kwargs)
        return []

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"attachment facade must use the declared page read, not {name}")


@pytest.mark.asyncio
async def test_attachment_library_page_delegates_one_bounded_read() -> None:
    """The product facade carries page bounds and filters to the repository."""

    repository = _Repository()
    poly = Polylogue.__new__(Polylogue)
    poly._services = cast("RuntimeServices", SimpleNamespace(get_repository=lambda: repository))

    result = await poly._get_attachment_library_page(
        limit=21,
        offset=40,
        mime_filter="image/",
        session_filter="session-7",
        state_filter="available",
    )

    assert result == []
    assert repository.calls == [
        {
            "limit": 21,
            "offset": 40,
            "mime_filter": "image/",
            "session_filter": "session-7",
            "state_filter": "available",
        }
    ]


@pytest.mark.asyncio
async def test_attachment_library_pages_keep_session_and_transcript_order(
    workspace_env: dict[str, Path],
) -> None:
    """Anti-vacuity: ordering by session and message IDs puts the older
    lexically-first session, and within it the lexically-first later message,
    ahead of the newest session's first attachment."""
    import sqlite3

    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.sources.parsers.base import ParsedAttachment, ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    db_path = workspace_env["archive_root"] / "index.db"

    def seed() -> None:
        # The fixture writer prepares on its original measured physical creator.
        with closing(connect_measured(db_path)) as conn, conn:
            conn.row_factory = sqlite3.Row
            initialize_archive_tier(conn, ArchiveTier.INDEX)
            for native_id, timestamp in [("a-old", "2026-01-01T00:00:00Z"), ("z-new", "2026-01-02T00:00:00Z")]:
                write_fixture_index_session(
                    conn,
                    ParsedSession(
                        source_name=Provider.CODEX,
                        provider_session_id=native_id,
                        created_at=timestamp,
                        updated_at=timestamp,
                        messages=[
                            ParsedMessage(
                                provider_message_id="z-first",
                                role=Role.USER,
                                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="first message")],
                            ),
                            ParsedMessage(
                                provider_message_id="a-second",
                                role=Role.USER,
                                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="second message")],
                            ),
                        ],
                        attachments=[
                            ParsedAttachment(
                                provider_attachment_id="first", message_provider_id="z-first", name="first.txt"
                            ),
                            ParsedAttachment(
                                provider_attachment_id="second", message_provider_id="a-second", name="second.txt"
                            ),
                        ],
                    ),
                )

    run_off_event_loop(seed)
    poly = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    try:
        pages = [await poly._get_attachment_library_page(limit=1, offset=offset) for offset in range(4)]
        records = [cast("AttachmentRecord", page[0][0]) for page in pages]
        assert [(str(row.session_id), row.display_name) for row in records] == [
            (f"codex-session:{session}", name) for session in ("z-new", "a-old") for name in ("first.txt", "second.txt")
        ]
    finally:
        await poly.close()
