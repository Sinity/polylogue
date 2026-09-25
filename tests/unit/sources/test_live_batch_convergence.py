from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue.daemon.intake import AdmissionOutcome, IntakeItem
from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter
from polylogue.sources.live import WatchSource, hook_paste_enrichment
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


class _SessionFirstConverger:
    def __init__(self) -> None:
        self.session_calls: list[tuple[str, ...]] = []
        self.batch_calls: list[tuple[Path, ...]] = []

    def converge_sessions(self, session_ids: tuple[str, ...]) -> tuple[dict[str, object], dict[str, float]]:
        self.session_calls.append(session_ids)
        return (
            {session_id: SimpleNamespace(converged=True, stages={"fts": "done"}) for session_id in session_ids},
            {"fts": 0.5},
        )

    def converge_batch(self, paths: tuple[Path, ...]) -> tuple[dict[Path, object], dict[str, float]]:
        self.batch_calls.append(paths)
        return ({path: SimpleNamespace(converged=True) for path in paths}, {"batch": 1.0})


def test_live_batch_converges_known_paths_by_source_path(tmp_path: Path) -> None:
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    source = tmp_path / "session.jsonl"
    source.write_text('{"a": 1}\n')
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    with sqlite3.connect(source_db) as conn:
        raw_id = write_source_raw_session(
            conn,
            origin="codex-session",
            source_path=str(source),
            source_index=0,
            payload=b'{"a": 1}\n',
            acquired_at_ms=1_767_225_600_000,
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, title, content_hash,
                created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "provider-1",
                "codex-session",
                raw_id,
                "hot session",
                bytes([7]) * 32,
                1_767_225_600_000,
                1_767_225_600_000,
            ),
        )
        conn.commit()

    converger = _SessionFirstConverger()
    processor = LiveBatchProcessor(
        MagicMock(archive_root=tmp_path),
        (WatchSource(name="projects", root=tmp_path),),
        cursor=CursorStore(index_db),
        converger=converger,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    completed, elapsed, timings, debts = processor._converge_paths([source])

    assert completed == {source}
    assert elapsed >= 0.0
    assert timings["batch"] == 1.0
    assert timings["hook_paste_enrichment"] >= 0.0
    assert debts == []
    assert converger.session_calls == []
    assert converger.batch_calls == [(source,)]


def test_hook_paste_failure_records_canonical_session_debt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_db = tmp_path / "index.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    cursor = CursorStore(index_db)
    processor = LiveBatchProcessor(
        MagicMock(archive_root=tmp_path),
        (),
        cursor=cursor,
        converger=_SessionFirstConverger(),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    session_id = "codex-session:paste-retry"

    def fail_enrichment(*_args: object, **_kwargs: object) -> int:
        raise OSError("temporary hook evidence read failure")

    monkeypatch.setattr(hook_paste_enrichment, "enrich_paste_from_hooks", fail_enrichment)

    processor._converge_paths((tmp_path / "session.jsonl",), session_ids=(session_id,))

    debt = cursor.list_convergence_debt()
    assert len(debt) == 1
    assert debt[0].stage == "hook_paste_enrichment"
    assert debt[0].subject_type == "session_id"
    assert debt[0].subject_id == session_id
    assert debt[0].last_error == "temporary hook evidence read failure"


class _GateTrackingCoordinator:
    """A write coordinator that reports whether its gate is currently held."""

    def __init__(self) -> None:
        self.depth = 0

    @property
    def held(self) -> bool:
        return self.depth > 0

    async def run(self, actor: str, operation: object) -> object:
        self.depth += 1
        try:
            return await cast(Any, operation)()
        finally:
            self.depth -= 1

    async def run_sync(self, actor: str, function: object, /, *args: object, **kwargs: object) -> object:
        self.depth += 1
        try:
            return cast(Any, function)(*args, **kwargs)
        finally:
            self.depth -= 1


@contextlib.asynccontextmanager
async def _held(coordinator: Any) -> AsyncIterator[None]:
    coordinator.depth += 1
    try:
        yield
    finally:
        coordinator.depth -= 1


@pytest.mark.asyncio
async def test_page_admission_runs_lease_free_owners_outside_the_writer_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding convergence for an ingest batch runs outside the coordinated region.

    The embed stage inside the coordinated ingest defers rather than calling a
    provider under the gate, so the owner call that follows is where the work
    actually happens -- and it must observe the gate released, or an unrelated
    archive writer would still be queued behind one network round trip.

    Anti-vacuity: move the owner call inside ``_ingest_files`` (or wrap the
    page in a coordinator hold again) and ``observed`` records
    ``gate_held=True`` for the owner.
    """
    index_db = tmp_path / "index.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    source = tmp_path / "session.jsonl"
    source.write_text('{"a": 1}\n')

    coordinator = _GateTrackingCoordinator()
    observed: list[tuple[str, bool, tuple[Path, ...]]] = []
    session_observed: list[tuple[bool, tuple[str, ...]]] = []

    async def owner(index_db_path: Path, paths: Sequence[Path], /) -> bool:
        observed.append(("embedding_owner", coordinator.held, tuple(paths)))
        return True

    async def session_owner(session_ids: Sequence[str], /) -> None:
        session_observed.append((coordinator.held, tuple(session_ids)))

    watcher = live_watcher.LiveWatcher(
        MagicMock(archive_root=tmp_path),
        (WatchSource(name="projects", root=tmp_path),),
        cursor=CursorStore(index_db),
        write_coordinator=cast(Any, coordinator),
        embedding_owner=owner,
        session_profile_callback=session_owner,
    )

    async def fake_ingest_files(paths: list[Path], **_kwargs: object) -> object:
        # The batch takes its own short writer admissions internally; this
        # double stands in for that whole region.
        async with _held(coordinator):
            observed.append(("ingest", coordinator.held, tuple(paths)))
        return SimpleNamespace(
            changed_session_ids=("session-1", "session-1", "session-2"),
            succeeded_file_count=1,
            succeeded_paths=(source,),
            failed_file_count=0,
            failed_paths=[],
            deferred_paths=(),
            excluded_paths={},
            stale_cursor_write_count=0,
            source_payload_read_bytes=source.stat().st_size,
        )

    monkeypatch.setattr(watcher, "_ingest_files", fake_ingest_files)
    monkeypatch.setattr(watcher, "select_ingest_candidates", lambda paths: tuple(paths))
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=watcher._sources),
        watcher._sources[0],
    )
    item = IntakeItem(item_id=f"file:{source}", class_name="projects", payload=source, estimated_cost=1)

    outcomes = await adapter.admit_page((item,))
    assert [result.outcome for result in outcomes.values()] == [AdmissionOutcome.ADMITTED]

    assert observed == [
        ("ingest", True, (source,)),
        ("embedding_owner", False, (source,)),
    ]
    assert session_observed == [(False, ("session-1", "session-2"))]
