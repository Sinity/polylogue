"""The attachment library selects composed membership before its page cut."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import aiosqlite
import pytest

from polylogue.api import Polylogue
from polylogue.archive.query.transaction import archive_read_context
from polylogue.core.errors import DatabaseError
from polylogue.daemon.http import DaemonAPIHandler
from polylogue.operations.http_read_models import read_attachment_library_page
from polylogue.storage.runtime import AttachmentRecord
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.storage_records import seed_attachment_library_lineage_archive


@pytest.mark.asyncio
async def test_library_session_filter_pages_inherited_refs_and_preserves_physical_owner(tmp_path: Path) -> None:
    ids = run_off_event_loop(lambda: seed_attachment_library_lineage_archive(tmp_path))
    expected = [("own.txt", ids["child"]), ("prefix.txt", ids["parent"])]
    async with Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db") as poly:
        for offset, wanted in enumerate((expected[:1], expected[1:], [])):
            rows = await poly._get_attachment_library_page(limit=1, offset=offset, session_filter=ids["child"])
            assert [
                (cast(AttachmentRecord, row[0]).display_name, str(cast(AttachmentRecord, row[0]).session_id))
                for row in rows
            ] == wanted
            handler = cast(Any, DaemonAPIHandler.__new__(DaemonAPIHandler))
            payload = await handler._do_attachment_library(
                poly, limit=1, offset=offset, mime_filter="", state_filter="", session_filter=ids["child"]
            )
            assert [(item["name"], item["session_id"]) for item in payload["items"]] == wanted
            with archive_read_context(
                tmp_path, operation="http.archive.read", arguments={}, projection="http-read"
            ) as archive:
                sync_rows = read_attachment_library_page(
                    archive, limit=1, offset=offset, mime_filter="", state_filter="", session_filter=ids["child"]
                )
            assert [
                (cast(AttachmentRecord, row[0]).display_name, str(cast(AttachmentRecord, row[0]).session_id))
                for row in sync_rows
            ] == wanted
        all_rows = await poly._get_attachment_library_page(limit=10, offset=0)
        assert {cast(AttachmentRecord, row[0]).display_name for row in all_rows} == {
            "own.txt",
            "prefix.txt",
            "post-cut.txt",
            "foreign.txt",
        }
        # Both physical owners are readable despite having no materialized profiles.
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            assert archive._conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["missing", "content-skew"])
async def test_library_refuses_incomplete_lineage_instead_of_complete_child_tail(tmp_path: Path, fault: str) -> None:
    def seed_and_break_lineage() -> dict[str, str]:
        # The archive writer's synchronous lease must not block this event loop.
        seeded = seed_attachment_library_lineage_archive(tmp_path)
        with (
            write_lease("test.attachment-library-lineage-fault", archive_root=tmp_path),
            ArchiveStore.open_existing(tmp_path, read_only=False) as archive,
        ):
            if fault == "missing":
                archive._conn.execute(
                    "UPDATE session_links SET branch_point_message_id='missing' WHERE src_session_id=?",
                    (seeded["child"],),
                )
            else:
                archive._conn.execute(
                    "UPDATE session_links SET branch_point_content_address=? WHERE src_session_id=?",
                    (b"\x00" * 32, seeded["child"]),
                )
            archive._conn.commit()
        return seeded

    ids = run_off_event_loop(seed_and_break_lineage)
    async with Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db") as poly:
        with pytest.raises(DatabaseError):
            await poly._get_attachment_library_page(limit=1, offset=0, session_filter=ids["child"])
    with archive_read_context(tmp_path, operation="http.archive.read", arguments={}, projection="http-read") as archive:
        with pytest.raises(DatabaseError):
            read_attachment_library_page(
                archive, limit=1, offset=0, mime_filter="", state_filter="", session_filter=ids["child"]
            )


@pytest.mark.asyncio
async def test_library_consumes_lazy_pages_and_stops_at_empty_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.queries.attachment_records import get_attachment_library_page

    run_off_event_loop(lambda: seed_attachment_library_lineage_archive(tmp_path))
    original_fetchmany = aiosqlite.Cursor.fetchmany
    page_sizes: list[int] = []

    async def lazy_fetchmany(cursor: aiosqlite.Cursor, size: int | None = None) -> Iterable[aiosqlite.Row]:
        rows = tuple(await original_fetchmany(cursor, size))
        page_sizes.append(len(rows))
        if len(page_sizes) > 2:
            pytest.fail("empty lazy page must end attachment collection")
        return iter(rows)

    monkeypatch.setattr(aiosqlite.Cursor, "fetchmany", lazy_fetchmany)
    async with aiosqlite.connect(tmp_path / "index.db") as connection:
        connection.row_factory = aiosqlite.Row
        rows = await get_attachment_library_page(connection, limit=10, offset=0)
    assert page_sizes == [4, 0]
    assert {row[0].display_name for row in rows} == {"own.txt", "prefix.txt", "post-cut.txt", "foreign.txt"}
