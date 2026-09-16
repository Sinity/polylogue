"""The dispatcher's own cold-build shape (polylogue-6xcqj).

The cold-build write shape used to be reachable only from
``sources/revision_backfill.py``: the ordinary live ingest pass took the live
write profile and the full compare/replace writer no matter how empty the
index generation was. These tests pin the shape onto the route the daemon's
intake dispatcher actually uses -- ``LiveBatchProcessor.ingest_files`` -- and
pin the boundary that hands the generation back to live readers.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.watch_config import WatchSource
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _codex_session(native_id: str, text: str) -> bytes:
    return (
        f'{{"type":"session_meta","payload":{{"id":"{native_id}",'
        '"timestamp":"2026-06-02T00:00:00Z"}}\n'
        '{"type":"response_item","payload":{"type":"message","id":"message-0",'
        f'"role":"user","content":[{{"type":"input_text","text":"{text}"}}]}}}}\n'
    ).encode()


def _processor(archive_root: Path, root: Path) -> LiveBatchProcessor:
    index_db = archive_root / "index.db"
    return LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=archive_root, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )


def _index_pragma(archive_root: Path, pragma: str) -> object:
    conn = sqlite3.connect(archive_root / "index.db")
    try:
        return conn.execute(f"PRAGMA {pragma}").fetchone()[0]
    finally:
        conn.close()


def test_an_empty_generation_gives_the_live_pass_the_cold_build_shape(tmp_path: Path) -> None:
    """The store engages the shape from generation state, not a caller flag.

    Anti-vacuity: reverting ``_open_archive_for_live_write`` to the plain
    ``open_existing`` writer, or dropping the emptiness probe so the shape is
    never engaged, makes the ``engaged`` assertion red.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    (root / "one.jsonl").write_bytes(_codex_session("cold-one", "zero"))

    engaged: list[bool] = []
    processor = _processor(tmp_path, root)
    original = ArchiveStore.open_active_cold_build.__func__  # type: ignore[attr-defined]

    def recording(cls: type[ArchiveStore], archive_root: Path) -> ArchiveStore:
        store = cast(ArchiveStore, original(cls, archive_root))
        engaged.append(store.active_cold_build_engaged)
        return store

    ArchiveStore.open_active_cold_build = classmethod(recording)  # type: ignore[assignment]
    try:
        metrics = asyncio.run(processor.ingest_files([root / "one.jsonl"], emit_event=False))
    finally:
        ArchiveStore.open_active_cold_build = classmethod(original)  # type: ignore[assignment]

    assert metrics.succeeded_file_count == 1
    assert engaged and engaged[0] is True


def test_a_populated_generation_falls_back_to_the_live_shape(tmp_path: Path) -> None:
    """The shape is licensed by emptiness, so the second pass loses it.

    This is the transition back to the live shape: nothing latches, and no
    caller has to remember to turn it off.

    Anti-vacuity: making ``_engage_active_cold_build`` skip the
    ``SELECT 1 FROM sessions`` probe (or engage unconditionally) makes the
    second ``engaged`` assertion red.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    (root / "one.jsonl").write_bytes(_codex_session("cold-first", "zero"))
    (root / "two.jsonl").write_bytes(_codex_session("cold-second", "one"))

    engaged: list[bool] = []
    processor = _processor(tmp_path, root)
    original = ArchiveStore.open_active_cold_build.__func__  # type: ignore[attr-defined]

    def recording(cls: type[ArchiveStore], archive_root: Path) -> ArchiveStore:
        store = cast(ArchiveStore, original(cls, archive_root))
        engaged.append(store.active_cold_build_engaged)
        return store

    ArchiveStore.open_active_cold_build = classmethod(recording)  # type: ignore[assignment]
    try:
        assert asyncio.run(processor.ingest_files([root / "one.jsonl"], emit_event=False)).succeeded_file_count == 1
        assert asyncio.run(processor.ingest_files([root / "two.jsonl"], emit_event=False)).succeeded_file_count == 1
    finally:
        ArchiveStore.open_active_cold_build = classmethod(original)  # type: ignore[assignment]

    assert engaged == [True, False]


def test_the_cold_build_boundary_returns_live_durability_to_the_generation(tmp_path: Path) -> None:
    """A cold-built generation is readable and WAL/NORMAL when the pass ends.

    The cold shape runs with ``synchronous=OFF`` and ``foreign_keys=OFF``; the
    boundary verifies the skipped constraint and restores durability before
    anything else can read the generation.

    Anti-vacuity: deleting the ``finish_active_cold_build()`` call from the
    ingest pass leaves ``synchronous`` at 0 and makes this red.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    (root / "one.jsonl").write_bytes(_codex_session("cold-boundary", "zero"))
    processor = _processor(tmp_path, root)
    assert asyncio.run(processor.ingest_files([root / "one.jsonl"], emit_event=False)).succeeded_file_count == 1

    assert _index_pragma(tmp_path, "journal_mode") == "wal"
    conn = sqlite3.connect(tmp_path / "index.db")
    try:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()

    # The generation is an ordinary readable archive afterwards: had the pass
    # dropped reader indexes the way an owned inactive generation may, this
    # read-only open would refuse with a schema manifest mismatch.
    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        assert archive._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


def test_a_second_write_of_one_session_under_fresh_mode_is_refused(tmp_path: Path) -> None:
    """bp12n.9 AC2 on a generation the dispatcher route itself opened.

    Fresh mode skips the compare/replace preparation after proving the session
    id is absent, so a repeat is an assertion failure rather than a silent
    overwrite. The generation here is the ACTIVE one, opened through
    ``open_active_cold_build`` -- the shape this route now selects -- not an
    owned inactive generation opened by ``revision_backfill``.

    Anti-vacuity: removing the absence assertion in
    ``write_parsed_session_to_archive``'s fresh branch makes this red; so does
    letting ``open_active_cold_build`` engage on a non-empty generation, which
    would make the first write itself refuse.
    """
    import pytest

    from polylogue.core.enums import Provider
    from polylogue.pipeline.models import ParsedSession
    from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive

    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="fresh-twice",
        title="fresh twice",
        messages=[],
    )
    with ArchiveStore.open_active_cold_build(tmp_path) as archive:
        assert archive.active_cold_build_engaged is True
        batch: set[str] = set()
        write_parsed_session_to_archive(archive._conn, session, fresh_build=True, fresh_build_batch=batch)
        with pytest.raises(AssertionError, match="fresh_build requires an absent session_id"):
            write_parsed_session_to_archive(archive._conn, session, fresh_build=True, fresh_build_batch=batch)
