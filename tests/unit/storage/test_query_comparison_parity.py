"""Exact comparison semantics through the production sync and async readers."""

import asyncio
from pathlib import Path

import aiosqlite
import pytest

from polylogue.core.enums import Provider
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.async_sqlite import configure_read_connection
from polylogue.storage.sqlite.queries.sessions_reads import count_sessions, list_sessions
from tests.infra.live_ingest import write_index_session


@pytest.mark.parametrize(
    ("prefix", "title", "selected"),
    [
        ("/repo/Project", None, [3, 2, 1, 0]),
        (" /repo//Project/ ", None, [3, 2, 1, 0]),
        ("/repo/P%_", None, [7, 6]),
        ("/", None, [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
        (None, "łódź", [3, 2, 1, 0]),
        (None, "%_", [7, 6]),
        (None, "STRASSE", [9]),
    ],
)
def test_production_comparisons_preserve_population_order_and_pages(
    tmp_path: Path, prefix: str | None, title: str | None, selected: list[int]
) -> None:
    paths = [
        "/repo/Project",
        "/repo/Project/child",
        "/repo//Project//child",
        "\\repo\\Project\\child",
        "/repo/project",
        "/repo/ProjectSibling",
        "/repo/P%_",
        "/repo/P%_/child",
        "/repo/Pxx/child",
        "/repo/else",
        "/repo/project/child",
    ]
    titles = [
        "ŁÓDŹ",
        "łódź",
        "Łódź notes",
        "notes ŁÓDŹ",
        "LODZ",
        "other",
        "100%_ done",
        "%_",
        "Straße",
        "STRASSE",
        "other",
    ]
    root = tmp_path / "archive"
    ids: list[str] = []
    with ArchiveStore(root) as store:
        for index, (path, original_title) in enumerate(zip(paths, titles, strict=True)):
            ids.append(
                write_index_session(
                    store,
                    ParsedSession(
                        source_name=Provider.CODEX,
                        provider_session_id=f"comparison-{index}",
                        title=original_title,
                        working_directories=[path],
                        updated_at=f"2026-01-02T10:00:{index:02d}Z",
                        messages=[],
                    ),
                )
            )
    expected = [ids[index] for index in selected]
    with ArchiveStore.open_existing(root) as store:
        assert store.count_sessions(cwd_prefix=prefix, title=title) == len(expected)
        assert [row.session_id for row in store.iter_summaries(cwd_prefix=prefix, title=title)] == expected
        assert [
            row.session_id for row in store.iter_summaries(cwd_prefix=prefix, title=title, limit=2, offset=1)
        ] == expected[1:3]
        for index, session_id in enumerate(ids):
            row = store._conn.execute("SELECT title FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            assert row[0] == titles[index]
            path_row = store._conn.execute(
                "SELECT path FROM session_working_dirs WHERE session_id=?", (session_id,)
            ).fetchone()
            assert path_row[0] == paths[index]

    async def check_async_reader() -> None:
        async with aiosqlite.connect(f"file:{root / 'index.db'}?mode=ro", uri=True) as connection:
            await configure_read_connection(connection, archive_root=root)
            assert await count_sessions(connection, cwd_prefix=prefix, title_contains=title) == len(expected)
            rows = await list_sessions(connection, cwd_prefix=prefix, title_contains=title)
            assert [row.session_id for row in rows] == expected
            page = await list_sessions(connection, cwd_prefix=prefix, title_contains=title, limit=2, offset=1)
            assert [row.session_id for row in page] == expected[1:3]

    asyncio.run(check_async_reader())
