"""The live write route's cold-build shape (polylogue-6xcqj).

The cold-build write shape used to be reachable only from
``sources/revision_backfill.py``: the ordinary live ingest pass took the live
write profile and the full compare/replace writer no matter how empty the
index generation was. These tests pin the shape onto the live write open
(``_open_archive_for_live_write``), which the append route calls with
``cold_build=True`` on every pass, and pin the boundary that hands the
generation back to live readers.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from polylogue.sources.live import WatchSource
from polylogue.sources.live.archive_open import _open_archive_for_live_write
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.source_layout import export_drop_layout
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.index_writer import write_fixture_index_session
from tests.infra.raw_owner_routes import run_ingest_files


def _codex_session(native_id: str, text: str) -> bytes:
    return (
        f'{{"type":"session_meta","payload":{{"id":"{native_id}",'
        '"timestamp":"2026-06-02T00:00:00Z"}}\n'
        '{"type":"response_item","payload":{"type":"message","id":"message-0",'
        f'"role":"user","content":[{{"type":"input_text","text":"{text}"}}]}}}}\n'
    ).encode()


def _processor(archive_root: Path, root: Path) -> LiveBatchProcessor:
    index_db = archive_root / "index.db"
    # The daemon ingests into a bootstrapped root; source-only acquisition
    # refuses a missing Source tier.
    bootstrap_archive_root(archive_root)
    return LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=archive_root, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="codex", root=root, layout=export_drop_layout((".jsonl",))),),
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
    """The live write open engages the shape from generation state, not a caller flag.

    The full-ingest pass now writes Source only and publishes through the
    retained owner (its empty-archive cold build is the owned inactive
    generation, pinned by ``test_retained_cold_readiness``); the live append
    route is the one that asks ``_open_archive_for_live_write`` for this shape
    on every pass.

    Anti-vacuity: reverting ``_open_archive_for_live_write`` to the plain
    ``open_existing`` writer, or dropping the emptiness probe so the shape is
    never engaged, makes the ``engaged`` assertion red.
    """
    bootstrap_archive_root(tmp_path)
    with _open_archive_for_live_write(tmp_path, cold_build=True) as archive:
        assert archive.active_cold_build_engaged is True


def test_a_populated_generation_falls_back_to_the_live_shape(tmp_path: Path) -> None:
    """The shape is licensed by emptiness, so the next pass loses it.

    This is the transition back to the live shape: nothing latches, and no
    caller has to remember to turn it off.

    Anti-vacuity: making ``_engage_active_cold_build`` skip the
    ``SELECT 1 FROM sessions`` probe (or engage unconditionally) makes the
    second ``engaged`` assertion red.
    """
    from polylogue.core.enums import Provider
    from polylogue.sources.parsers.base import ParsedSession

    bootstrap_archive_root(tmp_path)
    engaged: list[bool] = []
    with _open_archive_for_live_write(tmp_path, cold_build=True) as archive:
        engaged.append(archive.active_cold_build_engaged)
        write_fixture_index_session(
            archive._conn,
            ParsedSession(source_name=Provider.CODEX, provider_session_id="cold-first", title="first", messages=[]),
        )
        archive._conn.commit()
    with _open_archive_for_live_write(tmp_path, cold_build=True) as archive:
        engaged.append(archive.active_cold_build_engaged)

    assert engaged == [True, False]


def test_the_cold_build_boundary_leaves_a_readable_wal_generation(tmp_path: Path) -> None:
    """A cold-built generation is an ordinary readable WAL archive afterwards.

    The cold shape relaxes only per-connection durability, so ``journal_mode``
    -- the one file-level setting -- stays WAL throughout and readers are
    never locked out or unmoored.

    Anti-vacuity: giving the cold profile ``journal_mode="MEMORY"`` (as the
    bulk-build profile has) makes the WAL assertion red; dropping reader
    indexes the way an owned inactive generation may makes the read-only open
    below raise a schema manifest mismatch.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    (root / "one.jsonl").write_bytes(_codex_session("cold-boundary", "zero"))
    processor = _processor(tmp_path, root)
    assert run_ingest_files(processor, [root / "one.jsonl"], emit_event=False).succeeded_file_count == 1

    assert _index_pragma(tmp_path, "journal_mode") == "wal"
    conn = sqlite3.connect(tmp_path / "index.db")
    try:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()

    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        assert archive._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


def test_the_cold_build_boundary_survives_an_open_transaction(tmp_path: Path) -> None:
    """The boundary must not touch pragmas SQLite forbids mid-transaction.

    Reproduces the defect directly: ``PRAGMA synchronous`` raises
    ``Safety level may not be changed inside a transaction``. The boundary
    does not own the connection's transaction state, so a boundary that
    restored durability pragmas raised OperationalError and failed the whole
    ingest pass -- turning a would-be-successful record into a failure on the
    one pass where the cold shape is engaged.

    Anti-vacuity: restoring WRITE_CONNECTION_PROFILE's pragmas in
    ``finish_active_cold_build`` makes this raise instead of returning.
    """
    with ArchiveStore.open_active_cold_build(tmp_path) as archive:
        assert archive.active_cold_build_engaged is True
        archive._conn.execute("BEGIN")
        # session_id is a generated column (origin || ':' || native_id), so it
        # is never inserted directly -- see the identity model in AGENTS.md.
        archive._conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, ?, ?)",
            ("open-txn", "codex-session", b"\x00" * 32),
        )
        assert archive._conn.in_transaction is True
        archive.finish_active_cold_build()
        assert archive.active_cold_build_engaged is False


def test_the_cold_shape_keeps_foreign_key_enforcement(tmp_path: Path) -> None:
    """The cold shape relaxes durability only, never enforcement.

    Enforcement parity is what lets the shape be selected automatically on the
    live route: it cannot change what a pass writes, defers or refuses.

    Anti-vacuity: setting ``foreign_keys=False`` on
    COLD_BUILD_ACTIVE_WRITE_CONNECTION_PROFILE makes this red.
    """
    with ArchiveStore.open_active_cold_build(tmp_path) as archive:
        assert archive.active_cold_build_engaged is True
        assert archive._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert archive._conn.execute("PRAGMA synchronous").fetchone()[0] == 0


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
    from polylogue.sources.parsers.base import ParsedSession

    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="fresh-twice",
        title="fresh twice",
        messages=[],
    )
    with ArchiveStore.open_active_cold_build(tmp_path) as archive:
        assert archive.active_cold_build_engaged is True
        batch: set[str] = set()
        write_fixture_index_session(archive._conn, session, fresh_build=True, fresh_build_batch=batch)
        with pytest.raises(AssertionError, match="fresh_build requires an absent session_id"):
            write_fixture_index_session(archive._conn, session, fresh_build=True, fresh_build_batch=batch)
