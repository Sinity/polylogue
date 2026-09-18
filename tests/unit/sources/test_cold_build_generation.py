"""The daemon's cold build as an owned inactive generation (polylogue-b7dkb).

The active-generation cold-build shape (``test_live_cold_build_route.py``)
cannot take index deferral, ``journal_mode=MEMORY`` or ``locking_mode=
EXCLUSIVE``, because live readers hold the file it writes. These tests pin the
shape that can: the same dispatcher-fed ``LiveBatchProcessor.ingest_files``
pass, writing its index rows into a generation created by
``IndexGenerationStore`` and invisible to readers until ``promote()``.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cold_build import (
    ColdBuildGeneration,
    active_index_generation_is_empty,
    clear_cold_build_generation,
    register_cold_build_generation,
)
from polylogue.sources.live.cursor import CursorStore
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


def _active_session_count(archive_root: Path) -> int:
    """Count sessions the way a reader resolving the active pointer sees them."""
    from polylogue.storage.archive_identity import resolve_active_index_path

    conn = sqlite3.connect(f"file:{resolve_active_index_path(archive_root)}?mode=ro", uri=True)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
    finally:
        conn.close()


def _candidate_index_names(generation: ColdBuildGeneration) -> set[str]:
    conn = sqlite3.connect(f"file:{generation.generation.index_path}?mode=ro", uri=True)
    try:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    finally:
        conn.close()


@pytest.fixture
def cold_build(tmp_path: Path) -> Iterator[ColdBuildGeneration]:
    assert active_index_generation_is_empty(tmp_path) is True
    generation = ColdBuildGeneration.begin(tmp_path, reason="test")
    register_cold_build_generation(generation)
    try:
        yield generation
    finally:
        clear_cold_build_generation()


def _ingest(archive_root: Path, root: Path, name: str, native_id: str) -> None:
    (root / name).write_bytes(_codex_session(native_id, native_id))
    metrics = asyncio.run(_processor(archive_root, root).ingest_files([root / name], emit_event=False))
    assert metrics.succeeded_file_count == 1, metrics


def test_the_live_pass_writes_into_the_owned_generation_not_the_active_one(
    tmp_path: Path, cold_build: ColdBuildGeneration
) -> None:
    """Readers keep the previous active generation for the whole build.

    Anti-vacuity: making ``_open_archive_for_live_write`` ignore the
    registered generation (returning ``open_active_cold_build``) writes the
    rows into the active index and makes the during-build assertion red;
    deleting the ``promote()`` pointer swap makes the after-promotion one red.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "owned-one")

    # During the build the candidate holds the row and no reader can see it.
    assert cold_build.session_count() == 1
    assert _active_session_count(tmp_path) == 0
    with ArchiveStore.open_existing(tmp_path, read_only=True) as reader:
        assert reader._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0

    promoted = cold_build.promote()
    assert promoted.state == "active"
    assert _active_session_count(tmp_path) == 1
    with ArchiveStore.open_existing(tmp_path, read_only=True) as reader:
        assert reader._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


def test_the_build_spans_passes_and_keeps_the_deferred_indexes_dropped(
    tmp_path: Path, cold_build: ColdBuildGeneration
) -> None:
    """A cold build is many dispatcher pages against one generation.

    The second page re-opens a generation that now has rows; the deferral must
    survive that rather than being refused for non-emptiness or silently
    recreated by the runtime-index ensure.

    Anti-vacuity: restoring the ``SELECT 1 FROM sessions`` refusal in the
    deferral branch makes the second ingest raise; letting
    ``ensure_runtime_indexes_sync`` run on a deferring open makes the
    ``idx_messages_role`` assertion red.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "owned-first")
    _ingest(tmp_path, root, "two.jsonl", "owned-second")

    assert cold_build.session_count() == 2
    assert "idx_messages_role" not in _candidate_index_names(cold_build)

    cold_build.promote()
    assert _active_session_count(tmp_path) == 2


def test_the_readiness_pass_restores_the_reader_shape_before_promotion(
    tmp_path: Path, cold_build: ColdBuildGeneration
) -> None:
    """One CREATE INDEX pass and one FTS build, then the generation is ordinary.

    A read-only open projects ``sqlite_master`` including indexes, so a
    promoted generation still missing its deferred indexes refuses with a
    schema mismatch -- which is exactly why the deferral needs the owned
    generation in the first place.

    Anti-vacuity: deleting ``restore_deferred_secondary_indexes_sync`` from
    ``run_generation_readiness_pass`` makes the read-only open raise; deleting
    the FTS rebuild makes the search assertion red.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "owned-ready")

    cold_build.promote()

    assert "idx_messages_role" in _candidate_index_names(cold_build)
    with ArchiveStore.open_existing(tmp_path, read_only=True) as reader:
        hits = reader._conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'owned'").fetchone()
        assert hits[0] >= 1


def test_a_never_promoted_generation_is_discarded_and_leaves_readers_alone(
    tmp_path: Path, cold_build: ColdBuildGeneration
) -> None:
    """Crash semantics: the build simply never becomes visible.

    Anti-vacuity: making ``discard`` promote instead, or having the ingest
    pass write through to the active generation, makes the active count
    assertion red.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "owned-doomed")
    generation_root = cold_build.generation_root

    assert cold_build.discard() is True
    assert not generation_root.exists()
    assert _active_session_count(tmp_path) == 0


def test_acquisition_stays_on_the_real_durable_tiers(tmp_path: Path, cold_build: ColdBuildGeneration) -> None:
    """The candidate directory carries read-through symlinks, not a second archive.

    The cold build acquires and materializes in one pass, so the raw row and
    its blob must land in the archive's own ``source.db`` -- otherwise a
    discarded generation would take the durable evidence with it.

    Anti-vacuity: dropping ``durable_writer`` (so the store keeps the inactive
    candidate's refusing blob publisher) makes the ingest fail outright; a
    generation created before the durable tiers exist would grow its own
    ``source.db`` and make the symlink assertion red.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "owned-durable")

    assert (cold_build.generation_root / "source.db").is_symlink()
    conn = sqlite3.connect(f"file:{tmp_path / 'source.db'}?mode=ro", uri=True)
    try:
        assert int(conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0]) >= 1
    finally:
        conn.close()
