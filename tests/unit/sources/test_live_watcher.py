"""Tests for the live filesystem watcher: cursor, ingest skip logic,
debounce, bootstrap scan, and end-to-end via the watchfiles event loop."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
import zipfile
from builtins import BaseExceptionGroup
from collections.abc import Awaitable, Callable, Iterable, Iterator, Sequence
from contextlib import closing
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

import polylogue.sources.live.batch as live_batch
import polylogue.sources.live.watcher as live_watcher
from polylogue import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.core.raw_failure_evidence import RetainedRawDecodeRefusalError
from polylogue.daemon.intake import AdmissionOutcome, IntakeItem
from polylogue.daemon.status import _archive_live_ingest_attempt_summary_info
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
from polylogue.logging import WARNING
from polylogue.operations.intake_adapters import (
    DaemonIntakeContext,
    FileIntakeAdapter,
    _bounded_source_paths,
)
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.batch import (
    _FULL_PARSE_PROGRESS_MAX_BYTES,
    _FULL_PARSE_PROGRESS_MAX_FILES,
    CursorAuthorityBlockedError,
    LiveBatchProcessor,
    _full_parse_progress_groups,
    _FullIngestResult,
    last_complete_newline_from_tail,
)
from polylogue.sources.live.batch_support import (
    LiveRetainedRunner,
    encode_cursor_hash_authority,
    tail_hash_from_path,
)
from polylogue.sources.live.cursor import CursorPathAuthority, CursorRecord, CursorStore
from polylogue.sources.live.metrics import REFUSED_NO_SESSIONS, LiveBatchMetrics
from polylogue.sources.live.watcher import WriteCoordinator, default_sources
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.sources.revision_backfill import (
    RetainedRawRetryableFailure,
    RetainedReplayOutcome,
)
from polylogue.sources.source_layout import export_drop_layout
from polylogue.sources.sqlite_snapshot import sqlite_source_revision
from polylogue.storage.blob_store import BlobStore, PreparedBlob
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.archive_templates import bootstrap_archive_root, run_off_event_loop
from tests.infra.cursor_authority import fixture_cursor_authority
from tests.infra.frozen_clock import FrozenClock
from tests.infra.live_ingest import write_index_session
from tests.infra.prepared_membership import publish_prepared_membership_classification
from tests.infra.raw_owner_routes import (
    ingest_files_with_owners,
    replay_retained_raws_async,
    seed_membership_census,
    supplied_live_owners,
)

# Well above any small-file assumption: input size is a parameter of the one
# retained route, never a branch or refusal.
_LARGE_INPUT_BYTES = 8 * 1024 * 1024


class _FullIngestMock:
    def __init__(self) -> None:
        self.await_count = 0
        self.side_effect: BaseException | type[BaseException] | None = None

    def reset_mock(self) -> None:
        self.await_count = 0

    async def __call__(
        self,
        paths: list[Path],
        *,
        source_name: str,
        heartbeat: object = None,
        attempt_id: str | None = None,
        max_pass_seconds: float | None = None,
        pass_started: float | None = None,
    ) -> _FullIngestResult:
        del source_name, heartbeat, attempt_id, max_pass_seconds, pass_started
        self.await_count += 1
        if self.side_effect is not None:
            if isinstance(self.side_effect, BaseException):
                raise self.side_effect
            if isinstance(self.side_effect, type):
                raise self.side_effect()
        return _FullIngestResult(
            succeeded=list(paths),
            failed=[],
            source_payload_read_bytes=sum(path.stat().st_size for path in paths),
            raw_fingerprints={path: f"raw:{path.name}" for path in paths},
            ingested_session_count=1,
            ingested_message_count=7,
            changed_session_count=1,
            stage_timings_s={"full.provider_parse": 0.01, "full.index_parsed_write": 0.02},
        )


class _FailingSecondPathConverger:
    def __init__(self, failing_path: Path) -> None:
        self.failing_path = failing_path

    def converge_batch(self, paths: tuple[Path, ...]) -> tuple[dict[Path, object], dict[str, float]]:
        if self.failing_path in paths:
            return (
                {path: SimpleNamespace(converged=path != self.failing_path) for path in paths},
                {"fake": 0.001},
            )
        return ({path: SimpleNamespace(converged=True) for path in paths}, {"fake": 0.001})


def _sqlite_snapshot(path: Path) -> tuple[tuple[str, tuple[tuple[object, ...], ...]], ...]:
    with sqlite3.connect(path) as conn:
        tables = tuple(
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        )
        return tuple((table, tuple(conn.execute(f'SELECT * FROM "{table}"').fetchall())) for table in tables)


async def _seed_live_cursor_authority_case(
    root: Path,
    *,
    force_full_fallback: bool = False,
    exact_frontier: bool = False,
) -> tuple[LiveBatchProcessor, LiveWatcher, CursorStore, Path]:
    source_root = root / "sessions"
    # The declared Codex position: YYYY/MM/DD/rollout-*.jsonl.
    source_path = source_root / "2026" / "05" / "01" / "rollout-session-1.jsonl"
    source_path.parent.mkdir(parents=True)
    prefix = (
        json.dumps(_codex_session_meta("session-1")).encode()
        + b"\n"
        + json.dumps(
            _codex_message(message_id="m1", role="user", text="hello", timestamp="2026-05-01T00:00:00Z")
        ).encode()
        + b"\n"
    )
    tail = (
        json.dumps(
            _codex_message(message_id="m2", role="assistant", text="reply", timestamp="2026-05-01T00:00:01Z")
        ).encode()
        + b"\n"
    )
    source_path.write_bytes(prefix + tail)
    await asyncio.to_thread(initialize_active_archive_root, root)

    def acquire_prefix() -> str:
        # A writable open takes a synchronous lease, which may not block the loop.
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=prefix,
                source_path=str(source_path),
                # Acquisition records the canonical coordinate; the live
                # frontier gate refuses an archive holding a raw without one.
                canonical_source_path=str(source_path.resolve()),
                acquired_at_ms=1,
                native_id="session-1",
            )
            archive.bind_raw_revision(
                raw_id,
                RawRevisionEnvelope(
                    "codex-session:session-1",
                    RawRevisionKind.FULL,
                    "revision-0",
                    0,
                    authority=RawRevisionAuthority.BYTE_PROVEN,
                ),
            )
            return raw_id

    raw_id = await asyncio.to_thread(acquire_prefix)
    # The accepted head is published from the acquired prefix through the
    # canonical retained replay route.
    await replay_retained_raws_async(root, [raw_id])

    cursor = CursorStore(root / "ops.db")
    stat = source_path.stat()
    cursor_offset = len(prefix) if exact_frontier else len(prefix) + 1
    local_prefix_hash = sha256(source_path.read_bytes()[:cursor_offset]).hexdigest()
    cursor.set(
        source_path,
        cursor_offset,
        byte_offset=cursor_offset,
        last_complete_newline=len(prefix),
        parser_fingerprint="stale-parser" if force_full_fallback else live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="revision-0",
        tail_hash=encode_cursor_hash_authority(local_prefix_hash, local_prefix_hash, ctime_ns=stat.st_ctime_ns),
        source_name="codex",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(source_path),
    )
    polylogue = SimpleNamespace(archive_root=root, backend=SimpleNamespace(db_path=root / "index.db"))
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="codex", root=source_root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    watcher = LiveWatcher(
        cast(Any, polylogue),
        (WatchSource(name="codex", root=source_root),),
        cursor=cursor,
    )
    watcher._batch_processor = processor
    return processor, watcher, cursor, source_path


def _live_archive_snapshot(root: Path) -> tuple[object, ...]:
    return (
        _sqlite_snapshot(root / "source.db"),
        _sqlite_snapshot(root / "index.db"),
        _sqlite_snapshot(root / "ops.db"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "force_full_fallback",
    [False, True],
    ids=["append-route", "full-fallback-route"],
)
async def test_live_watcher_refuses_ahead_cursor_before_append_or_full_write(
    tmp_path: Path,
    force_full_fallback: bool,
) -> None:
    """An ahead cursor with a locally matching prefix cannot select either live route."""
    processor, watcher, cursor, source_path = await _seed_live_cursor_authority_case(
        tmp_path,
        force_full_fallback=force_full_fallback,
    )
    record = cursor.get_record(source_path)
    assert record is not None
    if force_full_fallback:
        assert processor._append_plan(source_path, cursor=record) is None
    else:
        assert processor._append_plan(source_path, cursor=record) is not None
    before = _live_archive_snapshot(tmp_path)

    with pytest.raises(CursorAuthorityBlockedError, match="source-selection gate blocked"):
        await watcher._ingest_files([source_path])

    assert _live_archive_snapshot(tmp_path) == before
    watcher.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("force_full_fallback", [False, True], ids=["append-route", "full-fallback-route"])
async def test_page_admission_refuses_an_ahead_cursor_before_touching_cursor_state(
    tmp_path: Path,
    force_full_fallback: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission stops before initialize or the needs-work selection can run.

    Anti-vacuity: move the authority check after ``cursor.initialize`` (or
    after the selection) in ``FileIntakeAdapter.admit_page`` and the call
    counters below go non-zero.
    """
    _processor, watcher, cursor, source_path = await _seed_live_cursor_authority_case(
        tmp_path,
        force_full_fallback=force_full_fallback,
    )
    before = _live_archive_snapshot(tmp_path)
    initialize_calls = 0
    select_calls = 0

    def track_initialize() -> None:
        nonlocal initialize_calls
        initialize_calls += 1

    def fail_select(*args: object, **kwargs: object) -> bool:
        del args, kwargs
        nonlocal select_calls
        select_calls += 1
        raise AssertionError("cursor authority must gate admission before cursor filtering")

    monkeypatch.setattr(cursor, "initialize", track_initialize)
    monkeypatch.setattr(watcher, "_needs_work_from_state", fail_select)
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=watcher._sources),
        watcher._sources[0],
    )
    item = IntakeItem(
        item_id=f"file:{source_path}",
        class_name=watcher._sources[0].name,
        payload=source_path,
        estimated_cost=source_path.stat().st_size,
    )

    outcomes = await adapter.admit_page((item,))

    assert [result.outcome for result in outcomes.values()] == [AdmissionOutcome.RETRYABLE]
    assert "source-selection gate blocked" in str(outcomes[item.item_id].reason)
    assert initialize_calls == 0
    assert select_calls == 0
    assert _live_archive_snapshot(tmp_path) == before
    assert adapter._after is None
    watcher.stop()


@pytest.mark.asyncio
async def test_live_watcher_allows_append_at_authoritative_frontier(tmp_path: Path) -> None:
    """An exact accepted frontier retains the real append success path."""
    processor, watcher, _cursor, source_path = await _seed_live_cursor_authority_case(
        tmp_path,
        exact_frontier=True,
    )

    metrics = await ingest_files_with_owners(processor, [source_path], emit_event=False)

    assert metrics.succeeded_file_count == 1
    assert metrics.append_file_count == 1
    assert metrics.full_file_count == 0
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (2,)
    watcher.stop()


@pytest.mark.asyncio
async def test_active_index_pointer_keeps_shadow_index_unmodified(tmp_path: Path) -> None:
    """Live writes and cursor reconciliation follow the active index pointer.

    The cursor sits at an authoritative frontier but carries a stale parser
    fingerprint, so the ordinary live route (no repair authorization exists
    in production) re-reads the file in full: the replacement is published
    into the active generation and the root shadow index stays untouched.
    """
    processor, watcher, _cursor, source_path = await _seed_live_cursor_authority_case(
        tmp_path, exact_frontier=True, force_full_fallback=True
    )
    shadow_index = tmp_path / "index.db"
    active_index = tmp_path / "generations" / "active" / "index.db"
    active_index.parent.mkdir(parents=True)
    # A WAL-mode tier's committed pages may still live in its -wal file; copy
    # through SQLite so the generation holds every committed row.
    with closing(sqlite3.connect(shadow_index)) as source, closing(sqlite3.connect(active_index)) as target:
        source.backup(target)
    (tmp_path / ".index-active-pointer").write_text(f"{active_index}\n", encoding="utf-8")
    # Compare committed rows, not file bytes: a WAL tier's writes land in its
    # -wal file first.
    shadow_before = _sqlite_snapshot(shadow_index)
    active_before = _sqlite_snapshot(active_index)

    metrics = await ingest_files_with_owners(processor, [source_path], emit_event=False)

    assert metrics.succeeded_file_count == metrics.full_file_count == 1
    assert _sqlite_snapshot(shadow_index) == shadow_before
    assert _sqlite_snapshot(active_index) != active_before
    with sqlite3.connect(shadow_index) as conn:
        conn.execute("DELETE FROM sessions")
        conn.commit()
    assert (
        watcher._reconcile_archived_cursor(
            source_path, stat=source_path.stat(), expected=watcher._cursor.get_record(source_path)
        )
        is True
    )
    watcher.stop()


@pytest.mark.asyncio
async def test_cursor_authority_seam_blocks_normal_live_route_before_writes(tmp_path: Path) -> None:
    """The exact selector exercises the production live authority seam."""
    _processor, watcher, _cursor, source_path = await _seed_live_cursor_authority_case(tmp_path)
    before = _live_archive_snapshot(tmp_path)

    with pytest.raises(CursorAuthorityBlockedError, match="source-selection gate blocked"):
        await watcher._ingest_files([source_path])

    assert _live_archive_snapshot(tmp_path) == before
    watcher.stop()


@pytest.mark.asyncio
async def test_cursor_authority_refuses_only_the_named_path(tmp_path: Path) -> None:
    """One path's violation refuses that path; its healthy siblings still ingest.

    Anti-vacuity: restoring the global refusal (raising whenever the block
    reason is non-None) makes this batch raise instead of ingesting the
    sibling, and the sibling's raw never lands in source.db.
    """
    _processor, watcher, _cursor, blocked_path = await _seed_live_cursor_authority_case(tmp_path)
    sibling = blocked_path.parent / "sibling.jsonl"
    sibling.write_bytes(
        json.dumps(_codex_session_meta("session-2")).encode()
        + b"\n"
        + json.dumps(
            _codex_message(message_id="m9", role="user", text="sibling", timestamp="2026-05-01T00:00:00Z")
        ).encode()
        + b"\n"
    )
    before = _live_archive_snapshot(tmp_path)

    async with supplied_live_owners(watcher._batch_processor):
        metrics = await watcher._ingest_files([blocked_path, sibling])

    assert metrics.skipped_file_count == 1
    assert metrics.succeeded_file_count == 1
    assert _live_archive_snapshot(tmp_path) != before
    with sqlite3.connect(tmp_path / "source.db") as conn:
        paths = {row[0] for row in conn.execute("SELECT source_path FROM raw_sessions")}
    assert str(sibling) in paths
    assert str(blocked_path) in paths  # the seeded prefix raw, unchanged
    watcher.stop()


@pytest.mark.parametrize("partial", [False, True])
def test_live_ingest_metrics_log_separates_read_bytes_from_candidate_size(
    monkeypatch: pytest.MonkeyPatch,
    partial: bool,
) -> None:
    logger = MagicMock()
    monkeypatch.setattr(live_watcher, "logger", logger)
    from polylogue.core.raw_failure_evidence import PartialAdmission

    partial_paths = (
        {
            "synthetic.jsonl": PartialAdmission(
                reason="truncated_tail", complete_record_count=2, complete_prefix_bytes=72, source_bytes=100
            )
        }
        if partial
        else {}
    )
    metrics = LiveBatchMetrics(
        queued_file_count=2,
        needed_file_count=2,
        skipped_file_count=0,
        succeeded_file_count=2,
        failed_file_count=0,
        source_group_count=1,
        input_bytes=400_000_000,
        source_payload_read_bytes=40_000,
        cursor_fingerprint_read_bytes=0,
        ingest_worker_count_max=1,
        append_file_count=2,
        full_file_count=0,
        archive_bytes_before=0,
        archive_bytes_after=0,
        archive_write_bytes_delta=0,
        parse_time_s=0.5,
        convergence_time_s=0.25,
        total_time_s=1.0,
        stage_timings_s={"full_parse": 0.45, "fts": 0.05, "derived": 0.2},
        partial_admission_paths=partial_paths,
    )

    live_watcher._log_ingest_metrics("live.watcher: changed-file batch", metrics)

    message, *args = logger.info.call_args.args
    assert "read=%.1f MB input=%.1f MB read_amp=%.6fx" in message
    assert "stages=%s" in message
    assert "excluded=%d" in message
    assert args[:6] == [
        "live.watcher: changed-file batch",
        0.04,
        400.0,
        0.0001,
        2,
        0,
    ]
    # Partial admissions qualify success without hiding the refused tail.
    assert args[6:12] == [2, int(partial), "truncated_tail x1" if partial else "none", 28 if partial else 0, 0, 0]
    assert args[14:] == ["full_parse:0.450,derived:0.200,fts:0.050", False]


def test_live_ingest_stage_timing_summary_is_bounded_and_sorted() -> None:
    assert live_watcher._stage_timing_summary({}) == "none"
    assert (
        live_watcher._stage_timing_summary(
            {
                "parse": 1.25,
                "fts": 0.01,
                "derived": 0.4,
                "usage": 0.2,
                "checkpoint": 0.8,
            },
            limit=3,
        )
        == "parse:1.250,checkpoint:0.800,derived:0.400,+2 more"
    )


# --- CursorStore ---------------------------------------------------------------


def test_cursor_default_is_zero(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    assert store.get(tmp_path / "missing.jsonl") == 0


def test_cursor_round_trip(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    p = tmp_path / "session.jsonl"
    store.set(p, 42, record_count=3, authority=fixture_cursor_authority(p))
    assert store.get(p) == 42
    record = store.get_record(p)
    assert isinstance(record, CursorRecord)
    assert record.byte_size == 42
    assert record.byte_offset == 42
    assert record.last_complete_newline == 42
    assert record.record_count == 3


def test_cursor_upsert_overwrites(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    p = tmp_path / "session.jsonl"
    store.set(p, 100, authority=fixture_cursor_authority(p))
    store.set(p, 250, record_count=99, authority=fixture_cursor_authority(p))
    assert store.get(p) == 250


def test_cursor_isolated_per_path(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    store.set(a, 10, authority=fixture_cursor_authority(a))
    store.set(b, 20, authority=fixture_cursor_authority(b))
    assert store.get(a) == 10
    assert store.get(b) == 20


def test_cursor_fetches_records_in_bulk(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    missing = tmp_path / "missing.jsonl"

    store.set(
        first,
        11,
        parser_fingerprint="parser",
        content_fingerprint="first-hash",
        authority=fixture_cursor_authority(first),
    )
    store.set(
        second,
        22,
        parser_fingerprint="parser",
        content_fingerprint="second-hash",
        excluded=True,
        authority=fixture_cursor_authority(second),
    )

    records = store.get_records([first, second, missing, first])

    assert set(records) == {first, second}
    assert records[first].content_fingerprint == "first-hash"
    assert records[second].excluded is True


def test_cursor_persists_across_instances(tmp_path: Path) -> None:
    db = tmp_path / "live.sqlite"
    store_a = CursorStore(db)
    p = tmp_path / "session.jsonl"
    store_a.set(p, 555, authority=fixture_cursor_authority(p))
    store_b = CursorStore(db)
    assert store_b.get(p) == 555


def test_cursor_creates_table_if_missing(tmp_path: Path) -> None:
    db = tmp_path / "live.sqlite"
    CursorStore(db)
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ingest_cursor'").fetchall()
    assert rows == [("ingest_cursor",)]


def test_cursor_writes_updated_at(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    p = tmp_path / "s.jsonl"
    store.set(p, 1, authority=fixture_cursor_authority(p))
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row = conn.execute("SELECT updated_at_ms FROM ingest_cursor WHERE source_path=?", (str(p),)).fetchone()
    assert row[0]
    assert isinstance(row[0], int)


def test_cursor_records_live_ingest_attempt_progress(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    source = tmp_path / "session.jsonl"
    source.write_text('{"a":1}\n')

    attempt_id = store.begin_ingest_attempt(
        paths=[source],
        input_bytes=source.stat().st_size,
        queued_file_count=1,
    )
    store.update_ingest_attempt(
        attempt_id,
        phase="full_parse",
        succeeded_file_count=0,
        failed_file_count=0,
        source_payload_read_bytes=0,
        cursor_fingerprint_read_bytes=0,
        parse_time_s=0.25,
        current_source="codex",
        current_path=source,
        rss_current_mb=123.0,
    )
    store.record_ingest_stage_event(
        attempt_id,
        phase="full_parse",
        status="running",
        queued_file_count=1,
        needed_file_count=1,
        skipped_file_count=0,
        succeeded_file_count=0,
        failed_file_count=0,
        input_bytes=source.stat().st_size,
        source_payload_read_bytes=0,
        cursor_fingerprint_read_bytes=0,
        parse_time_s=0.25,
        current_source="codex",
        current_path=source,
    )
    running_summary = _archive_live_ingest_attempt_summary_info(tmp_path / "ops.db")
    assert running_summary is not None and running_summary.available
    assert running_summary.running_count == 1
    running = running_summary.recent[0]
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        events = conn.execute(
            """
            SELECT attempt_id, stage, payload_json
            FROM daemon_stage_events
            ORDER BY observed_at_ms DESC, event_id DESC
            """
        ).fetchall()

    assert running.status == "running"
    assert running.phase == "full_parse"
    assert running.current_path == str(source)
    matching_events = [event for event in events if event[0:2] == (attempt_id, "full_parse")]
    assert matching_events
    assert any(str(source) in event[2] for event in matching_events)

    store.finish_ingest_attempt(attempt_id, status="completed", phase="completed")
    completed_summary = _archive_live_ingest_attempt_summary_info(tmp_path / "ops.db")
    assert completed_summary is not None and completed_summary.available
    assert completed_summary.running_count == 0
    completed = completed_summary.recent[0]
    assert completed.status == "completed"
    assert completed.completed_at is not None

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        ops_attempt = conn.execute(
            """
            SELECT status, phase, source_path, parsed_raw_count, materialized_count
            FROM ingest_attempts
            WHERE attempt_id = ?
            """,
            (attempt_id,),
        ).fetchone()
        ops_events = conn.execute(
            """
            SELECT attempt_id, stage, status, payload_json
            FROM daemon_stage_events
            WHERE attempt_id = ?
            """,
            (attempt_id,),
        ).fetchall()

    assert ops_attempt == ("completed", "completed", str(source), 0, 0)
    full_parse_events = [event for event in ops_events if event[1] == "full_parse"]
    assert full_parse_events
    assert any(
        event[0] == attempt_id and event[2] == "running" and str(source) in event[3] for event in full_parse_events
    )


def test_cursor_archives_partial_success_as_completed_with_error(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    source = tmp_path / "session.jsonl"
    source.write_text('{"a":1}\n')

    attempt_id = store.begin_ingest_attempt(
        paths=[source],
        input_bytes=source.stat().st_size,
        queued_file_count=1,
    )
    store.update_ingest_attempt(
        attempt_id,
        phase="completed",
        status="completed",
        succeeded_file_count=1,
        failed_file_count=1,
        materialized_count=1,
    )
    store.finish_ingest_attempt(
        attempt_id,
        status="completed_with_failures",
        phase="completed",
        error="/tmp/skipped-or-failed.jsonl",
    )

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row = conn.execute(
            """
            SELECT status, phase, parsed_raw_count, materialized_count, error_message
            FROM ingest_attempts
            WHERE attempt_id = ?
            """,
            (attempt_id,),
        ).fetchone()

    assert row == ("completed_with_failures", "completed", 1, 1, "/tmp/skipped-or-failed.jsonl")


def test_cursor_failed_items_are_not_recorded_as_parsed_or_materialized(tmp_path: Path) -> None:
    """A wholly failed batch keeps both successful counters at zero.

    The failure count remains in the stage event payload, but it must not be
    copied into either the parsed-raw or materialized-session columns.
    """
    store = CursorStore(tmp_path / "live.sqlite")
    source = tmp_path / "failed.jsonl"
    source.write_text('{"not": "accepted"}\n')
    attempt_id = store.begin_ingest_attempt(paths=[source], input_bytes=source.stat().st_size, queued_file_count=1)

    store.update_ingest_attempt(
        attempt_id,
        phase="full_parse_failed",
        status="completed_with_failures",
        succeeded_file_count=None,
        failed_file_count=1,
        materialized_count=None,
        error="parse failed",
    )

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row = conn.execute(
            "SELECT parsed_raw_count, materialized_count FROM ingest_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        event_payload = conn.execute(
            "SELECT payload_json FROM daemon_stage_events WHERE attempt_id = ? AND stage = ?",
            (attempt_id, "full_parse_failed"),
        ).fetchone()[0]

    assert row == (0, 0)
    assert '"failed_file_count":1' in event_payload


def test_cursor_syncs_positions_to_archive_ops_db(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    source = tmp_path / "session.jsonl"
    source.write_text('{"a":1}\n')

    store.set(
        source,
        42,
        byte_offset=41,
        last_complete_newline=40,
        parser_fingerprint="parser-v1",
        content_fingerprint="content-v1",
        tail_hash="tail-v1",
        st_dev=1,
        st_ino=2,
        mtime_ns=3,
        failure_count=2,
        next_retry_at="2026-05-24T00:01:00+00:00",
        excluded=True,
        authority=fixture_cursor_authority(source),
    )

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row = conn.execute(
            """
            SELECT source_path, stat_size, byte_offset, last_complete_newline,
                   parser_fingerprint, content_fingerprint, tail_hash, st_dev, st_ino, mtime_ns,
                   failure_count, next_retry_at, excluded
            FROM ingest_cursor
            WHERE source_path = ?
            """,
            (str(source),),
        ).fetchone()

    assert row == (
        str(source),
        42,
        41,
        40,
        "parser-v1",
        "content-v1",
        "tail-v1",
        1,
        2,
        3,
        2,
        "2026-05-24T00:01:00+00:00",
        1,
    )


def test_cursor_syncs_convergence_debt_to_archive_ops_db(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")

    store.record_convergence_debt(
        stage="session_profile",
        subject_type="session",
        subject_id="conv-1",
        error="boom",
    )

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row = conn.execute(
            """
            SELECT stage, target_type, target_id, attempts, last_error
            FROM convergence_debt
            WHERE stage = 'session_profile'
            """,
        ).fetchone()
    assert row == ("session_profile", "session", "conv-1", 1, "boom")

    store.clear_convergence_debt(subject_type="session", subject_id="conv-1", stage="session_profile")

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        count = conn.execute("SELECT COUNT(*) FROM convergence_debt").fetchone()[0]
    assert count == 0


def test_cursor_records_genuine_deferral_as_deferred_not_failed(tmp_path: Path) -> None:
    """A ``false_means_pending`` stage's deliberate backpressure deferral
    (polylogue-6krh) must land as ``status = 'deferred'``, not ``'failed'``
    -- otherwise daemon health/alerting treats routine backpressure as a
    stage that is broken.
    """
    store = CursorStore(tmp_path / "live.sqlite")

    store.record_convergence_debt(
        stage="derived",
        subject_type="session_id",
        subject_id="conv-1",
        error="derived deferred until source quiet",
        deferred=True,
    )
    store.record_convergence_debt(
        stage="session_profile",
        subject_type="session_id",
        subject_id="conv-2",
        error="boom",
    )

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        rows = dict(
            conn.execute("SELECT target_id, status FROM convergence_debt WHERE target_type = 'session_id'").fetchall()
        )
    assert rows == {"conv-1": "deferred", "conv-2": "failed"}


@pytest.mark.frozen_clock_modules("polylogue.sources.live.cursor", "polylogue.sources.live.convergence_debt_retry")
def test_cursor_records_messages_fts_surface_debt_as_immediately_due(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    store = CursorStore(tmp_path / "live.sqlite")

    store.record_convergence_debt(
        stage="derived",
        subject_type="session_id",
        subject_id="conv-1",
        error="profile stale",
    )
    frozen_clock.advance(1)
    store.record_convergence_debt(
        stage="fts",
        subject_type="fts_surface",
        subject_id="messages_fts",
        error="startup found stale messages_fts freshness ledger",
    )

    debt = store.list_convergence_debt(limit=2)
    assert [item.subject_id for item in debt] == ["messages_fts", "conv-1"]
    retry_at = datetime.fromisoformat(debt[0].next_retry_at or "")
    failed_at = datetime.fromisoformat(debt[0].last_failed_at)
    assert debt[0].stage == "fts"
    assert retry_at == failed_at

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        priority = conn.execute(
            """
            SELECT priority
            FROM convergence_debt
            WHERE stage = 'fts' AND target_type = 'fts_surface' AND target_id = 'messages_fts'
            """,
        ).fetchone()[0]
    assert priority == 100


def test_cursor_marks_running_attempts_abandoned_on_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "live.sqlite"
    store = CursorStore(db_path)
    source = tmp_path / "session.jsonl"
    source.write_text('{"a":1}\n')
    attempt_id = store.begin_ingest_attempt(
        paths=[source],
        input_bytes=source.stat().st_size,
        queued_file_count=1,
    )
    store.update_ingest_attempt(
        attempt_id,
        phase="full_parse",
        succeeded_file_count=0,
        failed_file_count=0,
    )

    CursorStore(db_path)
    summary = _archive_live_ingest_attempt_summary_info(db_path.parent / "ops.db")
    assert summary is not None and summary.available
    assert summary.running_count == 0
    attempt = summary.recent[0]

    assert attempt.attempt_id == attempt_id
    assert attempt.status == "interrupted"
    assert attempt.phase == "interrupted"
    assert attempt.completed_at is not None
    assert attempt.error == "daemon stopped before completing this ingest attempt"
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row = conn.execute("SELECT status, phase FROM ingest_attempts WHERE attempt_id = ?", (attempt_id,)).fetchone()
    assert row == ("interrupted", "interrupted")


def test_cursor_mark_failed_creates_record_for_new_path(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    p = tmp_path / "new.jsonl"
    p.write_text('{"a":1}\n')

    store.mark_failed(p, authority=fixture_cursor_authority(p))

    record = store.get_record(p)
    assert record is not None
    assert record.byte_size == p.stat().st_size
    assert record.failure_count == 1
    assert record.next_retry_at is not None


def test_cursor_mark_failed_quarantines_repeated_failures(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    p = tmp_path / "poison.jsonl"
    p.write_text('{"a":1}\n')

    for _ in range(5):
        store.mark_failed(p, authority=fixture_cursor_authority(p))

    record = store.get_record(p)
    assert record is not None
    assert record.failure_count == 5
    assert record.next_retry_at is None
    assert record.excluded is True
    assert store.list_failed_with_retry() == []


def test_cursor_quarantine_binds_to_failed_replacement_observation(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    path = tmp_path / "capture.json"
    path.write_text('{"accepted":true}', encoding="utf-8")
    accepted = path.stat()
    store.set(
        path,
        accepted.st_size,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="accepted",
        st_dev=accepted.st_dev,
        st_ino=accepted.st_ino,
        mtime_ns=accepted.st_mtime_ns,
        authority=fixture_cursor_authority(path),
    )

    replacement = tmp_path / "replacement.json"
    replacement.write_text('{"malformed":', encoding="utf-8")
    replacement.replace(path)
    failed = path.stat()
    for _ in range(5):
        store.mark_failed(path, failed_stat=failed, authority=fixture_cursor_authority(path))

    record = store.get_record(path)
    assert record is not None
    assert record.excluded is True
    assert (record.byte_size, record.st_dev, record.st_ino, record.mtime_ns) == (
        failed.st_size,
        failed.st_dev,
        failed.st_ino,
        failed.st_mtime_ns,
    )


def test_cursor_round_trips_freshness_metadata(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    p = tmp_path / "session.jsonl"
    store.set(
        p,
        42,
        byte_offset=40,
        last_complete_newline=37,
        last_record_ts="2026-05-01T12:00:00+00:00",
        parser_fingerprint="parser-v1",
        content_fingerprint="abc123",
        source_name="codex",
        authority=fixture_cursor_authority(p),
    )

    record = store.get_record(p)

    assert record is not None
    assert record.byte_size == 42
    assert record.byte_offset == 40
    assert record.last_complete_newline == 37
    assert record.last_record_ts == "2026-05-01T12:00:00+00:00"
    assert record.parser_fingerprint == "parser-v1"
    assert record.content_fingerprint == "abc123"
    assert record.source_name == "codex"


def test_cursor_does_not_import_legacy_live_cursor_rows(tmp_path: Path) -> None:
    db = tmp_path / "live.sqlite"
    legacy_path = tmp_path / "legacy.jsonl"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE live_cursor (
                source_path TEXT PRIMARY KEY,
                byte_size INTEGER NOT NULL,
                record_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO live_cursor (source_path, byte_size, record_count, updated_at) VALUES (?, ?, ?, ?)",
            (str(legacy_path), 12, 2, "2026-05-01T00:00:00+00:00"),
        )
        conn.commit()

    store = CursorStore(db)
    record = store.get_record(legacy_path)

    assert record is None
    with sqlite3.connect(db) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(live_cursor)")}
    assert columns == {"source_path", "byte_size", "record_count", "updated_at"}
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row_count = conn.execute("SELECT COUNT(*) FROM ingest_cursor").fetchone()[0]
    assert row_count == 0


@pytest.mark.asyncio
async def test_live_full_ingest_streams_large_paths_before_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "projects"
    project = root / "project"
    project.mkdir(parents=True)
    source_path = project / "large-session.jsonl"
    records = [
        _claude_code_message(
            session_id="large-live-session",
            uuid=f"msg-{index}",
            role="user",
            text=f"message {index}",
            timestamp="2026-05-01T00:00:00Z",
        )
        for index in range(33)
    ]
    _write_jsonl(source_path, records)
    with source_path.open("r+b") as handle:
        handle.seek(_LARGE_INPUT_BYTES + 128)
        handle.write(b"\n")

    db_path = tmp_path / "archive.sqlite"
    # The live route acquires into an existing archive (#3952).
    bootstrap_archive_root(tmp_path)
    polylogue = MagicMock()
    polylogue.archive_root = tmp_path
    polylogue.backend.db_path = db_path
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        polylogue,
        (WatchSource(name="projects", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    calls: list[str] = []
    from polylogue.sources.acquisition_boundary import capture_bound_path as original_capture

    def spy_capture(blob_store: BlobStore, path: Path | str, *args: Any, **kwargs: Any) -> Any:
        # The capture streams the bound file through the acquisition boundary.
        calls.append(f"path:{Path(path).name}")
        return original_capture(blob_store, path, *args, **kwargs)

    def fail_from_bytes(_store: object, _payload: bytes, **_kwargs: object) -> PreparedBlob:
        raise AssertionError("large live full ingest should stream from path")

    monkeypatch.setattr("polylogue.sources.live.batch.capture_bound_path", spy_capture)
    monkeypatch.setattr("polylogue.sources.live.batch.BlobStore.prepare_from_bytes", fail_from_bytes)
    monkeypatch.setattr("polylogue.sources.live.batch.BlobStore.write_from_bytes", fail_from_bytes)

    async with supplied_live_owners(processor):
        result = await processor._ingest_full_paths([source_path], source_name="projects")

    assert result.succeeded == [source_path]
    assert result.failed == []
    assert calls == ["path:large-session.jsonl"]


# --- LiveWatcher: needs_work + ingest_files (batched) --------------------------


def _make_watcher(
    tmp_path: Path,
    root: Path,
    *,
    event_emitter: MagicMock | None = None,
    write_coordinator: WriteCoordinator | None = None,
    sources: tuple[WatchSource, ...] | None = None,
) -> tuple[LiveWatcher, _FullIngestMock]:
    polylogue = MagicMock()
    polylogue.archive_root = tmp_path
    cursor = CursorStore(tmp_path / "cursor.sqlite")
    sources = sources or (WatchSource(name="test", root=root),)
    watcher = LiveWatcher(
        polylogue,
        sources,
        cursor=cursor,
        event_emitter=event_emitter,
        write_coordinator=write_coordinator,
    )
    full_ingest = _FullIngestMock()
    watcher._batch_processor._ingest_full_paths = full_ingest  # type: ignore[method-assign]
    return watcher, full_ingest


def test_watcher_default_cursor_uses_archive_database(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    db_path = tmp_path / "ops.db"
    polylogue = cast(
        Any,
        SimpleNamespace(
            archive_root=tmp_path,
            backend=SimpleNamespace(db_path=db_path),
        ),
    )

    watcher = LiveWatcher(polylogue, (WatchSource(name="test", root=root),))

    assert watcher._cursor._db_path == db_path


def test_page_selection_uses_bulk_cursor_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One page reads its cursor rows in one bulk call, never one read per file.

    Anti-vacuity: replace the bulk ``get_records`` in
    ``classify_ingest_candidates`` with a per-path ``get_record`` loop and the
    stubbed per-file reader below raises.
    """
    root = tmp_path / "src"
    root.mkdir()
    files = [root / f"session-{index}.jsonl" for index in range(3)]
    for path in files:
        path.write_text('{"role":"user","content":"a"}\n')
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    bulk_calls = 0
    original_get_records = watcher._cursor.get_records

    def counted_get_records(paths: Iterable[Path]) -> dict[Path, CursorRecord]:
        nonlocal bulk_calls
        bulk_calls += 1
        return original_get_records(paths)

    def fail_get_record(path: Path) -> CursorRecord | None:
        raise AssertionError(f"catch-up should use bulk cursor reads, not per-file reads: {path}")

    monkeypatch.setattr(watcher._cursor, "get_records", counted_get_records)
    monkeypatch.setattr(watcher._cursor, "get_record", fail_get_record)

    assert list(watcher.classify_ingest_candidates(files)[0]) == files
    assert bulk_calls == 1


def test_page_classification_names_scheduled_retries_as_pending(tmp_path: Path) -> None:
    """A not-yet-due retry is pending; a settled exclusion is neither.

    Anti-vacuity (polylogue-b8of0): without the pending split the scheduled
    retry reads exactly like a file its cursor already accounts for.
    """
    root = tmp_path / "src"
    root.mkdir()
    owed, settled = root / "owed.jsonl", root / "settled.jsonl"
    for path in (owed, settled):
        path.write_text('{"role":"user","content":"a"}\n')
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    # Retry state belongs to the parser that recorded it; a cursor from
    # another parser is reattempted at once rather than waiting.
    watcher._cursor.set(
        owed,
        0,
        failure_count=1,
        next_retry_at="2999-01-01T00:00:00+00:00",
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        authority=fixture_cursor_authority(owed),
    )
    observed = settled.stat()
    watcher._cursor.set(
        settled,
        observed.st_size,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        st_dev=observed.st_dev,
        st_ino=observed.st_ino,
        mtime_ns=observed.st_mtime_ns,
        excluded=True,
        authority=fixture_cursor_authority(settled),
    )

    assert watcher.classify_ingest_candidates([owed, settled]) == ((), (owed,))


async def _ingest_one(watcher: LiveWatcher, path: Path) -> None:
    """Helper: check and ingest a single file (mimics old _ingest_if_grown)."""
    if watcher._needs_work(path):
        # Source bodies (raw compaction) run only on the daemon writer.
        async with supplied_live_owners(watcher._batch_processor):
            await watcher._ingest_files([path])


def _run_watcher_ingest(watcher: LiveWatcher, paths: list[Path], **kwargs: Any) -> None:
    """Run one watcher pass with the daemon's owners, as ``polylogued`` supplies them."""

    async def run() -> None:
        async with supplied_live_owners(watcher._batch_processor):
            await watcher._ingest_files(paths, **kwargs)

    asyncio.run(run())


def test_skip_when_file_not_grown(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"role":"user","content":"a"}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)

    asyncio.run(_ingest_one(watcher, f))
    assert parse_sources.await_count == 1

    asyncio.run(_ingest_one(watcher, f))
    assert parse_sources.await_count == 1  # cursor matches size


def test_reingest_when_file_grows(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"role":"user","content":"a"}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)

    asyncio.run(_ingest_one(watcher, f))
    f.write_text('{"a":1}\n{"b":2}\n{"c":3}\n')
    asyncio.run(_ingest_one(watcher, f))
    f.write_text('{"a":1}\n{"b":2}\n{"c":3}\n{"d":4}\n')
    asyncio.run(_ingest_one(watcher, f))

    assert parse_sources.await_count == 3


def test_size_only_cursor_reingests_to_populate_fingerprint(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)
    watcher._cursor.set(f, f.stat().st_size, authority=fixture_cursor_authority(f))

    asyncio.run(_ingest_one(watcher, f))
    asyncio.run(_ingest_one(watcher, f))

    assert parse_sources.await_count == 1
    record = watcher._cursor.get_record(f)
    assert record is not None
    assert record.content_fingerprint
    assert record.parser_fingerprint


def test_unchanged_file_uses_stat_fast_path_without_fingerprint_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    stat = f.stat()
    watcher._cursor.set(
        f,
        stat.st_size,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="already-known",
        tail_hash=encode_cursor_hash_authority(
            sha256(f.read_bytes()).hexdigest(),
            sha256(f.read_bytes()).hexdigest(),
            ctime_ns=stat.st_ctime_ns,
        ),
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(f),
    )

    def fail_fingerprint(path: Path) -> tuple[str, int]:
        raise AssertionError(f"unchanged file should not be fingerprinted: {path}")

    def fail_tail_hash(path: Path, size: int) -> tuple[str, int]:
        raise AssertionError(f"stat-stable unchanged file should not read tail hash: {path} ({size})")

    monkeypatch.setattr(live_watcher, "fingerprint_file", fail_fingerprint)
    monkeypatch.setattr(live_watcher, "tail_hash_from_path", fail_tail_hash)

    assert watcher._needs_work(f) is False

    watcher._cursor.set(
        f,
        stat.st_size,
        byte_offset=stat.st_size,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="already-known",
        tail_hash=encode_cursor_hash_authority(
            sha256(f.read_bytes()).hexdigest(),
            sha256(f.read_bytes()).hexdigest(),
            ctime_ns=stat.st_ctime_ns,
        ),
        st_dev=stat.st_dev + 1,
        st_ino=stat.st_ino + 1,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(f),
    )
    assert watcher._needs_work(f) is False


def test_new_file_needs_work_without_prefingerprint_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, _parse_sources = _make_watcher(tmp_path, root)

    def fail_fingerprint(path: Path) -> tuple[str, int]:
        raise AssertionError(f"new file should not be fingerprinted before ingest: {path}")

    monkeypatch.setattr(live_watcher, "fingerprint_file", fail_fingerprint)

    assert watcher._needs_work(f) is True


def test_parser_version_change_needs_work_without_prefingerprint_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    stat = f.stat()
    watcher._cursor.set(
        f,
        stat.st_size,
        parser_fingerprint="older-parser",
        content_fingerprint="already-known",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(f),
    )

    def fail_fingerprint(path: Path) -> tuple[str, int]:
        raise AssertionError(f"parser changes should not prefingerprint before ingest: {path}")

    monkeypatch.setattr(live_watcher, "fingerprint_file", fail_fingerprint)

    assert watcher._needs_work(f) is True


def test_replaced_excluded_file_is_revived_without_retrying_unchanged_poison(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    path = root / "capture.json"
    path.write_text('{"broken":true}', encoding="utf-8")
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    stat = path.stat()
    watcher._cursor.set(
        path,
        stat.st_size,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="broken",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        failure_count=5,
        excluded=True,
        authority=fixture_cursor_authority(path),
    )

    assert watcher._needs_work(path) is False

    replacement = root / "replacement.json"
    replacement.write_text('{"valid":"new capture"}', encoding="utf-8")
    replacement.replace(path)

    # The replacement is reported as work. The quarantine itself is lifted by
    # the acquisition pass that re-decides the path, so that a path which was
    # never admitted never presents a stale byte offset as committed ingest
    # authority to the raw-frontier cursor map.
    assert watcher._needs_work(path) is True
    still_quarantined = watcher._cursor.get_record(path)
    assert still_quarantined is not None
    assert still_quarantined.excluded is True


def test_excluded_file_revives_on_parser_fingerprint_change_without_identity_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-ix5r: a parser fix must revive an excluded cursor even when
    the poisoned file's own bytes never change. Exclusion revival was
    previously bound only to file identity (size/dev/inode/mtime), so a file
    that fails to parse stays permanently dark even after the parser bug that
    poisoned it is fixed -- the file on disk never changes, only the code
    that reads it. A ``_PARSER_FINGERPRINT`` bump (this module's existing,
    deliberately-versioned marker for a parser-semantics change) must be enough to
    trigger a fresh attempt through the real ``LiveWatcher._needs_work`` path,
    not merely through ``CursorStore.revive_replaced_exclusion`` directly.
    """
    root = tmp_path / "src"
    root.mkdir()
    path = root / "capture.json"
    path.write_text('{"broken":true}', encoding="utf-8")
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    stat = path.stat()
    watcher._cursor.set(
        path,
        stat.st_size,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="broken",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        failure_count=5,
        excluded=True,
        authority=fixture_cursor_authority(path),
    )

    # Identity-only revival check: unchanged bytes, unchanged parser -> stays
    # excluded and dark.
    assert watcher._needs_work(path) is False
    still_excluded = watcher._cursor.get_record(path)
    assert still_excluded is not None
    assert still_excluded.excluded is True

    # Simulate a parser fix shipping (the responsible parser's fingerprint
    # changes) with the file itself completely untouched.
    monkeypatch.setattr(live_watcher, "_PARSER_FINGERPRINT", "live-batched-v3-test")

    # ix5r's requirement is that the parser fix triggers a fresh attempt --
    # that is what ``_needs_work`` reporting True means. Clearing the
    # quarantine is the acquisition pass's job, not this check's.
    assert watcher._needs_work(path) is True
    still_quarantined = watcher._cursor.get_record(path)
    assert still_quarantined is not None
    assert still_quarantined.excluded is True


def test_full_cursor_uses_batch_raw_fingerprint_without_db_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, _parse_sources = _make_watcher(tmp_path, root)

    def fail_latest_raw_fingerprint(path: Path) -> str | None:
        raise AssertionError(f"raw fingerprint should already be carried by the batch result: {path}")

    monkeypatch.setattr(watcher._batch_processor, "_latest_raw_fingerprint", fail_latest_raw_fingerprint)

    bytes_read = watcher._batch_processor._record_full_cursor(f, raw_fingerprint="raw-sha256")
    record = watcher._cursor.get_record(f)

    # The cursor stores both a bounded tail and a complete accepted-prefix
    # hash; this one-record fixture makes both reads span the whole file.
    assert bytes_read == 2 * f.stat().st_size
    assert record is not None
    assert record.content_fingerprint == "raw-sha256"


def test_full_cursor_reuses_verified_acquisition_digest_at_eof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A restored prefix hash rereads the full acquired payload a third time."""
    root = tmp_path / "src"
    root.mkdir()
    path = root / "session.jsonl"
    payload = b'{"a":"' + (b"x" * 200_000) + b'"}\n'
    path.write_bytes(payload)
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    stat = path.stat()
    calls: list[tuple[int, int]] = []
    original_hash_range = cast(Callable[..., tuple[str, int]], live_batch.__dict__["sha256_range_from_path"])

    def count_hash_range(*args: Any, **kwargs: Any) -> tuple[str, int]:
        calls.append((int(kwargs["start_offset"]), int(kwargs["end_offset"])))
        return original_hash_range(*args, **kwargs)

    monkeypatch.setattr(live_batch, "sha256_range_from_path", count_hash_range)

    bytes_read = watcher._batch_processor._record_full_cursor(
        path,
        raw_fingerprint="raw-sha256",
        captured_content_hash=sha256(payload).hexdigest(),
        captured_file_observation=(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns),
    )

    assert calls == [(0, len(payload)), (0, len(payload))]
    assert bytes_read == 2 * len(payload) + 64 * 1024
    record = watcher._cursor.get_record(path)
    assert record is not None
    assert record.content_fingerprint == "raw-sha256"


def test_full_cursor_reused_digest_still_rejects_a_mutated_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second full-capture proof binds the cursor to the acquired bytes."""
    root = tmp_path / "src"
    root.mkdir()
    path = root / "session.jsonl"
    payload = b'{"a":"' + (b"x" * 200_000) + b'"}\n'
    path.write_bytes(payload)
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    stat = path.stat()
    original_hash_range = cast(Callable[..., tuple[str, int]], live_batch.__dict__["sha256_range_from_path"])
    hashes = 0

    def mutate_after_first_hash(
        path_to_hash: Path,
        *,
        start_offset: int,
        end_offset: int,
        **kwargs: Any,
    ) -> tuple[str, int]:
        nonlocal hashes
        result = original_hash_range(
            path_to_hash,
            start_offset=start_offset,
            end_offset=end_offset,
            **kwargs,
        )
        hashes += 1
        if hashes == 1:
            path.write_bytes(payload.replace(b"x", b"y"))
        return result

    monkeypatch.setattr(live_batch, "sha256_range_from_path", mutate_after_first_hash)

    watcher._batch_processor._record_full_cursor(
        path,
        raw_fingerprint="raw-sha256",
        captured_content_hash=sha256(payload).hexdigest(),
        captured_file_observation=(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns),
    )

    assert hashes == 1
    assert watcher._batch_processor._last_cursor_write_stale is True
    record = watcher._cursor.get_record(path)
    assert record is None or record.byte_offset == 0


@pytest.mark.parametrize("sqlite_input", [False, True], ids=["jsonl", "sqlite"])
@pytest.mark.parametrize("cursor_state", ["settled", "excluded", "deferred", "failed"])
@pytest.mark.frozen_clock_modules("polylogue.sources.live.watcher")
def test_hermes_profile_retarget_reopens_same_inode_cursor(
    tmp_path: Path, sqlite_input: bool, cursor_state: str, frozen_clock: FrozenClock
) -> None:
    from polylogue.core.enums import Provider
    from polylogue.sources.acquisition_boundary import capture_bound_path
    from polylogue.sources.sqlite_snapshot import snapshot_sqlite_to_blob

    first = tmp_path / "profile-a"
    second = tmp_path / "profile-b"
    external = tmp_path / "external"
    for root in (first, second, external):
        root.mkdir()
    # Both profiles hold the one physical input at its declared Hermes
    # position: the database at the home root (a hard link, so the inode is
    # shared), the ATOF stream below a shared observability tree.
    if sqlite_input:
        relative = Path("state.db")
        actual = external / "state.db"
        with sqlite3.connect(actual) as connection:
            connection.execute("CREATE TABLE state (value TEXT)")
            connection.execute("INSERT INTO state VALUES ('same accepted input')")
        for root in (first, second):
            os.link(actual, root / relative)
    else:
        relative = Path("observability") / "nemo-relay" / "atof" / "events.jsonl"
        actual = external / relative
        actual.parent.mkdir(parents=True)
        # A genuine Hermes ATOF event stream: a session-shaped record from
        # another harness at a Hermes location is refused as foreign origin.
        actual.write_bytes(
            (Path(__file__).parents[2] / "fixtures" / "origin-capability" / "hermes-session.jsonl").read_bytes()
        )
        for root in (first, second):
            (root / "observability").symlink_to(external / "observability", target_is_directory=True)
    alias = tmp_path / "profile"
    alias.symlink_to(first, target_is_directory=True)
    path = alias / relative
    # The production Hermes source admits its SQLite ledgers as well as JSONL.
    hermes = next(source for source in default_sources(hermes_root=alias) if source.name == "hermes")
    watcher, _ = _make_watcher(tmp_path, alias, sources=(hermes,))
    store = BlobStore(tmp_path / "blobs")

    def acquire() -> tuple[str, str]:
        if sqlite_input:
            snapshot = snapshot_sqlite_to_blob(path, store)
            assert snapshot.captured_profile_key is not None
            return snapshot.blob_hash, snapshot.captured_profile_key
        capture = capture_bound_path(store, path, Provider.HERMES)
        assert capture.captured_profile_key is not None
        return capture.blob_hash, capture.captured_profile_key

    original_stat = path.stat()
    blob_hash, original_profile = acquire()
    if sqlite_input:
        accepted_tail = sqlite_source_revision(path)
    else:
        # A settled JSONL cursor carries the prefix/tail hash authority of
        # the exact observation it accepted, including its ctime.
        accepted_prefix = sha256(path.read_bytes()).hexdigest()
        accepted_tail = encode_cursor_hash_authority(
            accepted_prefix, accepted_prefix, ctime_ns=original_stat.st_ctime_ns
        )

    def stamp(profile: str) -> None:
        watcher._cursor.set(
            path,
            original_stat.st_size,
            source_name="hermes",
            authority=CursorPathAuthority(str(path.resolve()), profile),
            parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
            content_fingerprint=None if cursor_state == "deferred" else blob_hash,
            tail_hash=accepted_tail,
            st_dev=original_stat.st_dev,
            st_ino=original_stat.st_ino,
            mtime_ns=original_stat.st_mtime_ns,
            excluded=cursor_state == "excluded",
            failure_count=int(cursor_state == "failed"),
            next_retry_at=(frozen_clock.now(UTC) + timedelta(days=1)).isoformat(),
        )

    stamp(original_profile)
    assert not watcher._needs_work(path)
    alias.unlink()
    alias.symlink_to(second, target_is_directory=True)
    assert (path.stat().st_dev, path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns) == (
        original_stat.st_dev,
        original_stat.st_ino,
        original_stat.st_size,
        original_stat.st_mtime_ns,
    )
    assert watcher._needs_work(path)
    new_blob, new_profile = acquire()
    assert new_blob == blob_hash
    assert new_profile != original_profile
    stamp(new_profile)
    assert not watcher._needs_work(path)
    alias.unlink()
    alias.symlink_to(first, target_is_directory=True)
    assert watcher._needs_work(path)
    assert acquire() == (blob_hash, original_profile)


def test_hermes_sqlite_profile_retarget_between_probe_and_bound_gate_requires_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager

    from polylogue.sources.sqlite_snapshot import snapshot_sqlite_to_blob

    external = tmp_path / "external"
    external.mkdir()
    actual = external / "state.db"
    with sqlite3.connect(actual) as connection:
        connection.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO sessions VALUES ('same inode and logical bytes')")
    profiles = (tmp_path / "profile-a", tmp_path / "profile-b")
    for profile in profiles:
        profile.mkdir()
        # The declared home-root position, sharing one inode across profiles.
        os.link(actual, profile / "state.db")
    alias = tmp_path / "profile"
    alias.symlink_to(profiles[0], target_is_directory=True)
    declared = alias / "state.db"
    hermes = next(source for source in default_sources(hermes_root=alias) if source.name == "hermes")
    watcher, _ = _make_watcher(tmp_path, alias, sources=(hermes,))
    store = BlobStore(tmp_path / "blobs")
    accepted = snapshot_sqlite_to_blob(declared, store)
    before = declared.stat()
    watcher._cursor.set(
        declared,
        before.st_size,
        source_name="hermes",
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint=accepted.source_revision,
        tail_hash=accepted.source_fingerprint,
        authority=CursorPathAuthority(str(declared.resolve()), accepted.captured_profile_key),
        st_dev=before.st_dev,
        st_ino=before.st_ino,
        mtime_ns=before.st_mtime_ns,
    )
    from polylogue.sources.source_staging import SourceInputBinding, bind_source_input

    @contextmanager
    def retarget_then_bind(path: Path) -> Iterator[SourceInputBinding]:
        alias.unlink()
        alias.symlink_to(profiles[1], target_is_directory=True)
        with bind_source_input(path) as binding:
            yield binding

    monkeypatch.setattr("polylogue.sources.live.watcher.bind_source_input", retarget_then_bind)
    assert watcher._needs_work(declared)
    captured = snapshot_sqlite_to_blob(declared, store)
    assert captured.blob_hash == accepted.blob_hash
    assert captured.captured_profile_key != accepted.captured_profile_key


def test_hermes_wal_revision_triggers_resnapshot_and_maps_sidecar_event(tmp_path: Path) -> None:
    root = tmp_path / "hermes"
    root.mkdir()
    state_db = root / "state.db"
    writer = sqlite3.connect(state_db)
    try:
        writer.execute("CREATE TABLE turns(id INTEGER PRIMARY KEY, text TEXT)")
        writer.commit()
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        watcher, _full_ingest = _make_watcher(
            tmp_path,
            root,
            sources=(WatchSource(name="hermes", root=root, layout=export_drop_layout((".db",))),),
        )
        initial_revision = sqlite_source_revision(state_db)
        from polylogue.sources.sqlite_snapshot import snapshot_sqlite_to_blob

        accepted = snapshot_sqlite_to_blob(state_db, BlobStore(tmp_path / "blobs"))
        stat = state_db.stat()
        watcher._cursor.set(
            state_db,
            stat.st_size,
            parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
            content_fingerprint="snapshot-hash",
            tail_hash=initial_revision,
            authority=CursorPathAuthority(str(state_db.resolve()), accepted.captured_profile_key),
            st_dev=stat.st_dev,
            st_ino=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
        )
        assert watcher._needs_work(state_db) is False

        writer.execute("INSERT INTO turns(text) VALUES ('WAL-only turn')")
        writer.commit()
        wal_path = state_db.with_name("state.db-wal")

        assert wal_path.stat().st_size > 0
        assert watcher._watch_filter(object(), str(wal_path)) is True
        assert watcher._canonical_watch_path(wal_path) == state_db
        assert watcher._needs_work(state_db) is True
    finally:
        writer.close()


def test_hermes_file_alias_wal_commit_reopens_actual_acquired_cursor(tmp_path: Path) -> None:
    """Using state.db-wal instead of the opened bundle.data-wal hides this commit."""
    from polylogue.sources.sqlite_snapshot import snapshot_sqlite_to_blob

    root = tmp_path / "hermes"
    root.mkdir()
    actual = tmp_path / "bundle.data"
    declared = root / "state.db"
    declared.symlink_to(actual)
    writer = sqlite3.connect(actual)
    try:
        writer.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY)")
        writer.commit()
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        hermes = next(source for source in default_sources(hermes_root=root) if source.name == "hermes")
        watcher, _ = _make_watcher(tmp_path, root, sources=(hermes,))
        accepted = snapshot_sqlite_to_blob(declared, BlobStore(tmp_path / "blobs"))
        before = declared.stat()
        watcher._cursor.set(
            declared,
            before.st_size,
            source_name="hermes",
            parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
            content_fingerprint=accepted.source_revision,
            tail_hash=accepted.source_fingerprint,
            authority=CursorPathAuthority(str(declared.resolve()), accepted.captured_profile_key),
            st_dev=before.st_dev,
            st_ino=before.st_ino,
            mtime_ns=before.st_mtime_ns,
        )
        assert not watcher._needs_work(declared)
        writer.execute("INSERT INTO sessions VALUES ('committed-only-in-physical-wal')")
        writer.commit()
        after = declared.stat()
        assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        assert not declared.with_name("state.db-wal").exists()
        assert actual.with_name("bundle.data-wal").stat().st_size > 0
        assert watcher._needs_work(declared)
    finally:
        writer.close()


def test_watch_filter_accepts_directories_but_not_unmatched_files_under_broad_roots(tmp_path: Path) -> None:
    """The watch backend wakes only for source suffixes or real directories."""

    root = tmp_path / "codex-state"
    root.mkdir()
    unmatched = root / "history.log"
    unmatched.write_text("noise", encoding="utf-8")
    child_directory = root / "new-session"
    child_directory.mkdir()
    watcher, _full_ingest = _make_watcher(
        tmp_path,
        root,
        sources=(WatchSource(name="codex-state", root=root, layout=export_drop_layout((".jsonl",))),),
    )

    assert watcher._watch_filter(object(), str(unmatched)) is False
    assert watcher._watch_filter(object(), str(child_directory)) is True


def test_added_directory_scan_rejects_file_symlinks_escaping_source_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recursive add recovery applies the same resolved containment as live events."""

    root = tmp_path / "watched"
    added = root / "new-directory"
    added.mkdir(parents=True)
    internal = added / "inside.jsonl"
    internal.write_text("{}\n", encoding="utf-8")
    external = tmp_path / "outside.jsonl"
    external.write_text("secret\n", encoding="utf-8")
    escaping = added / "escaping.jsonl"
    escaping.symlink_to(external)
    watcher, _full_ingest = _make_watcher(
        tmp_path,
        root,
        sources=(WatchSource(name="codex", root=root, layout=export_drop_layout((".jsonl",))),),
    )
    assert watcher._canonical_watch_path(escaping) is None
    watcher._enqueue_added_directory(added)

    source = watcher._sources[0]
    assert _bounded_source_paths(source, watcher._sources, limit=8, after=None) == [internal]


def test_added_directory_scan_retains_a_deeper_root_under_outer_ignore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An outer ignore rule cannot hide a configured nested source root."""

    outer = tmp_path / "codex-state"
    ignored = outer / "runtime"
    inner = ignored / "sessions"
    inner.mkdir(parents=True)
    session = inner / "session.jsonl"
    session.write_text("{}\n", encoding="utf-8")
    watcher, _full_ingest = _make_watcher(
        tmp_path,
        outer,
        sources=(
            # The declared codex-state layout never reaches ``runtime/``.
            WatchSource(name="codex-state", root=outer),
            WatchSource(name="codex", root=inner, layout=export_drop_layout((".jsonl",))),
        ),
    )
    assert watcher._watch_filter(object(), str(ignored)) is True
    watcher._enqueue_added_directory(ignored)
    assert watcher.intake_revision(watcher._sources[1]) > 0

    inner_source = watcher._sources[1]
    assert _bounded_source_paths(inner_source, watcher._sources, limit=8, after=None) == [session]


def test_hermes_cursor_records_acquisition_revision_not_live_tail(tmp_path: Path) -> None:
    root = tmp_path / "hermes"
    root.mkdir()
    state_db = root / "state.db"
    with sqlite3.connect(state_db) as conn:
        conn.execute("CREATE TABLE turns(id INTEGER PRIMARY KEY)")
    watcher, _full_ingest = _make_watcher(
        tmp_path,
        root,
        sources=(WatchSource(name="hermes", root=root, layout=export_drop_layout((".db",))),),
    )

    bytes_read = watcher._batch_processor._record_full_cursor(
        state_db,
        raw_fingerprint="snapshot-hash",
        raw_byte_size=999_999,
        source_revision="acquisition-revision",
    )
    record = watcher._cursor.get_record(state_db)

    assert bytes_read == 0
    assert record is not None
    assert record.byte_size == state_db.stat().st_size
    assert record.content_fingerprint == "acquisition-revision"
    assert record.tail_hash == sqlite_source_revision(state_db)


def test_hermes_cursor_keeps_snapshot_time_fingerprint(tmp_path: Path) -> None:
    root = tmp_path / "hermes"
    root.mkdir()
    state_db = root / "state.db"
    with sqlite3.connect(state_db) as conn:
        conn.execute("CREATE TABLE turns(id INTEGER PRIMARY KEY)")
    watcher, _full_ingest = _make_watcher(
        tmp_path,
        root,
        sources=(WatchSource(name="hermes", root=root, layout=export_drop_layout((".db",))),),
    )

    watcher._batch_processor._record_full_cursor(
        state_db,
        raw_fingerprint="snapshot-hash",
        source_revision="acquisition-revision",
        source_fingerprint="snapshot-time-fingerprint",
    )

    record = watcher._cursor.get_record(state_db)

    assert record is not None
    assert record.tail_hash == "snapshot-time-fingerprint"


def _archive_codex_session(archive_root: Path, native_id: str) -> None:
    """Archive the Codex session an append delta binds to.

    The planner emits an append plan only for a delta whose session identity
    is already bound (c07c4f44b1); without one the full route re-reads.
    """
    bootstrap_archive_root(archive_root)

    def seed() -> None:
        with ArchiveStore(archive_root) as store:
            write_index_session(
                store,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id=native_id,
                    title=native_id,
                    messages=[ParsedMessage(provider_message_id=f"{native_id}-0", role=Role.USER, text="seed")],
                ),
            )

    run_off_event_loop(seed)


def test_append_plan_reads_only_completed_tail(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    original = b'{"type":"session_meta","payload":{"id":"completed-tail"}}\n'
    completed = (
        b'{"type":"response_item","payload":{"type":"message","role":"user",'
        b'"content":[{"type":"input_text","text":"b"}]}}\n'
    )
    appended = completed + b'{"type":'
    f.write_bytes(original + appended)
    _archive_codex_session(tmp_path, "completed-tail")
    watcher, _parse_sources = _make_watcher(tmp_path, root, sources=(WatchSource(name="codex", root=root),))
    stat = f.stat()
    watcher._cursor.set(
        f,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="base",
        tail_hash=encode_cursor_hash_authority(
            sha256(original).hexdigest(),
            sha256(original).hexdigest(),
            ctime_ns=stat.st_ctime_ns,
        ),
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(f),
    )

    plan = watcher._batch_processor._append_plan(f)

    assert plan is not None
    append_plan = cast(Any, plan)
    assert append_plan.start_offset == len(original)
    assert append_plan.payload == completed
    assert append_plan.bytes_read == len(appended)
    assert append_plan.last_complete_newline == len(original) + len(completed)


def test_large_incomplete_jsonl_append_defers_until_the_file_changes(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    original = b'{"a":1}\n'
    f.write_bytes(original)
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    stat = f.stat()
    watcher._cursor.set(
        f,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="base",
        tail_hash=encode_cursor_hash_authority(
            sha256(original).hexdigest(),
            sha256(original).hexdigest(),
            ctime_ns=stat.st_ctime_ns,
        ),
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(f),
    )
    f.write_bytes(original + (b"x" * (live_watcher._INCOMPLETE_APPEND_PROBE_BYTES + 1)))

    # The first bounded probe observes no newline and records the unfinished
    # tail.  Repeating the same periodic scan must then be a stat-only skip.
    assert watcher._needs_work(f) is False
    record = watcher._cursor.get_record(f)
    assert record is not None
    assert record.byte_size == f.stat().st_size
    assert record.byte_offset == len(original)
    assert watcher._needs_work(f) is False


def test_a_failed_tail_probe_records_the_deferral_instead_of_retrying_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-dhkuu Finding A: an unreadable tail is still a recorded deferral.

    The probe used to ``handle.read(remaining_bytes)`` in one call and catch
    only ``OSError``. A ``MemoryError`` from an unterminated multi-GB tail
    therefore escaped, and even the ``OSError`` branch returned without
    calling ``record_deferred_append_cursor`` -- so the cursor kept its old
    observation and the identical probe was re-attempted on every catch-up
    pass, forever.

    Anti-vacuity: restore ``except OSError: return True`` (a bare return that
    records nothing) and ``record.byte_size`` below stays at the original
    length, so the second ``_needs_work`` call re-probes.
    """

    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    original = b'{"a":1}\n'
    f.write_bytes(original)
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    stat = f.stat()
    watcher._cursor.set(
        f,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="base",
        tail_hash=encode_cursor_hash_authority(
            sha256(original).hexdigest(),
            sha256(original).hexdigest(),
            ctime_ns=stat.st_ctime_ns,
        ),
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(f),
    )
    f.write_bytes(original + (b"x" * 4096))

    probes = 0

    def _explode(*_args: object, **_kwargs: object) -> bool:
        nonlocal probes
        probes += 1
        raise MemoryError("tail does not fit in memory")

    monkeypatch.setattr(live_watcher, "_tail_begins_a_complete_record", _explode)

    assert watcher._needs_work(f) is False
    assert probes == 1
    record = watcher._cursor.get_record(f)
    assert record is not None
    # The deferral was recorded despite the failed probe: the observed size
    # advanced while the offset stayed put, which is the state the stat-match
    # fast path skips.
    assert record.byte_size == f.stat().st_size
    assert record.byte_offset == len(original)

    # The identical pass must now be a stat-only skip, not a second probe.
    assert watcher._needs_work(f) is False
    assert probes == 1


def _backdate_cursor(tmp_path: Path, path: Path, *, now: datetime, seconds_ago: float) -> None:
    """Directly rewrite ``ingest_cursor.updated_at_ms`` into the past.

    Simulates a deferred append cursor that has sat at the same observed
    stat for a long time, without sleeping real wall-clock seconds in the
    test. ``now`` must come from ``frozen_clock.now()`` (or an equivalent
    already-guarded source), never a direct host-clock read.
    """
    past_ms = int((now.timestamp() - seconds_ago) * 1000)
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute("UPDATE ingest_cursor SET updated_at_ms = ? WHERE source_path = ?", (past_ms, str(path)))
        conn.commit()


def test_incomplete_probe_does_not_escalate_for_a_recent_in_progress_writer(tmp_path: Path) -> None:
    """An ordinary in-progress writer routinely leaves a genuinely incomplete
    trailing record between two polls milliseconds apart. That must NOT be
    treated as "the writer is done" just because the stat happened to match
    on an immediate second check -- only genuine staleness (see the age-gated
    escalation test below) justifies the more aggressive full-tail probe."""
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    original = b'{"a":1}\n'
    f.write_bytes(original)
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    stat = f.stat()
    watcher._cursor.set(
        f,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="base",
        tail_hash=encode_cursor_hash_authority(
            sha256(original).hexdigest(),
            sha256(original).hexdigest(),
            ctime_ns=stat.st_ctime_ns,
        ),
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(f),
    )
    tail = (b"x" * (live_watcher._INCOMPLETE_APPEND_PROBE_BYTES + 1)) + b'{"b":2}\n'
    f.write_bytes(original + tail)

    assert watcher._needs_work(f) is False
    # Second scan, immediately after -- no time has passed, so this must
    # still be treated as "maybe still writing", not escalated.
    assert watcher._needs_work(f) is False
    record = watcher._cursor.get_record(f)
    assert record is not None
    assert record.failure_count == 0


@pytest.mark.frozen_clock_modules("polylogue.sources.live.watcher", "polylogue.sources.live.cursor")
def test_incomplete_probe_escalates_to_full_scan_once_stat_stops_changing(
    tmp_path: Path, frozen_clock: FrozenClock
) -> None:
    """polylogue-2qrx: the bounded 64MB no-newline probe can miss a genuine
    complete trailing record sitting just past its window. Once the writer
    stops (stat never changes again) for long enough to rule out "still
    writing", the OLD behavior parked the cursor forever with zero durable
    signal -- measured live as 211 files / 414MB of stalled append backlog,
    up to 329h stale on one 94.8MB lag. The escalated full-tail probe on a
    stale stat-unchanged observation must find that record and unblock
    catch-up instead."""
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    original = b'{"a":1}\n'
    f.write_bytes(original)
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    stat = f.stat()
    watcher._cursor.set(
        f,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="base",
        tail_hash=encode_cursor_hash_authority(
            sha256(original).hexdigest(),
            sha256(original).hexdigest(),
            ctime_ns=stat.st_ctime_ns,
        ),
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(f),
    )
    # A trailing tail whose first PROBE_BYTES contain no newline, but a
    # complete record follows just past the bounded probe window.
    tail = (b"x" * (live_watcher._INCOMPLETE_APPEND_PROBE_BYTES + 1)) + b'{"b":2}\n'
    f.write_bytes(original + tail)

    # First scan: the bounded probe finds no \n in its window and defers.
    assert watcher._needs_work(f) is False
    _backdate_cursor(tmp_path, f, now=frozen_clock.now(), seconds_ago=live_watcher._STUCK_DEFERRED_APPEND_AGE_S + 1)

    # Second scan against the SAME unchanged stat, now stale enough that the
    # writer is implausibly still active: the escalated full-tail probe must
    # find the trailing record and report work is needed, instead of
    # trusting the stat-match fast path to skip it forever.
    assert watcher._needs_work(f) is True
    record = watcher._cursor.get_record(f)
    assert record is not None
    assert record.failure_count == 0


@pytest.mark.frozen_clock_modules("polylogue.sources.live.watcher", "polylogue.sources.live.cursor")
def test_incomplete_probe_marks_failed_when_no_newline_exists_anywhere(
    tmp_path: Path, frozen_clock: FrozenClock
) -> None:
    """polylogue-2qrx: when even the escalated full-tail probe finds no
    complete trailing record (a genuinely truncated/unterminated tail that
    will never resolve on its own), the cursor must get a durable failure
    record instead of parking silently forever."""
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    original = b'{"a":1}\n'
    f.write_bytes(original)
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    stat = f.stat()
    watcher._cursor.set(
        f,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="base",
        tail_hash=encode_cursor_hash_authority(
            sha256(original).hexdigest(),
            sha256(original).hexdigest(),
            ctime_ns=stat.st_ctime_ns,
        ),
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(f),
    )
    f.write_bytes(original + (b"x" * (live_watcher._INCOMPLETE_APPEND_PROBE_BYTES + 1)))

    assert watcher._needs_work(f) is False
    _backdate_cursor(tmp_path, f, now=frozen_clock.now(), seconds_ago=live_watcher._STUCK_DEFERRED_APPEND_AGE_S + 1)
    # Second scan, same unchanged stat but now stale: the escalated
    # full-tail probe also finds nothing (there really is no trailing
    # newline anywhere) -- this must become a durable, visible failure, not
    # another silent defer.
    assert watcher._needs_work(f) is False
    record = watcher._cursor.get_record(f)
    assert record is not None
    assert record.failure_count == 1


def test_last_complete_newline_from_tail_reads_only_final_chunk(tmp_path: Path) -> None:
    path = tmp_path / "large.jsonl"
    complete_prefix = b'{"a":"' + (b"x" * 200_000) + b'"}\n'
    path.write_bytes(complete_prefix + b'{"b":2}')

    offset, bytes_read = last_complete_newline_from_tail(path, path.stat().st_size)

    assert offset == len(complete_prefix)
    assert bytes_read < path.stat().st_size


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _claude_code_message(
    *,
    session_id: str,
    uuid: str,
    role: str,
    text: str,
    timestamp: str,
    parent_uuid: str | None = None,
) -> dict[str, object]:
    return {
        "type": role,
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "sessionId": session_id,
        "timestamp": timestamp,
        "message": {"role": role, "content": text if role == "user" else [{"type": "text", "text": text}]},
    }


def _codex_session_meta(session_id: str) -> dict[str, object]:
    return {"type": "session_meta", "payload": {"id": session_id, "timestamp": "2026-05-01T00:00:00Z"}}


def _codex_message(*, message_id: str, role: str, text: str, timestamp: str) -> dict[str, object]:
    block_type = "input_text" if role == "user" else "output_text"
    return {
        "type": "response_item",
        "payload": {
            "id": message_id,
            "role": role,
            "type": "message",
            "timestamp": timestamp,
            "content": [{"type": block_type, "text": text}],
        },
    }


@pytest.mark.asyncio
async def test_live_batch_processor_records_durable_attempt(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source_path = root / "session.jsonl"
    source_path.write_text(
        '{"type":"session_meta","payload":{"id":"s"}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"user",'
        '"content":[{"type":"input_text","text":"hello"}]}}\n',
        encoding="utf-8",
    )
    db_path = tmp_path / "live.sqlite"
    cursor = CursorStore(db_path)
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    metrics = await ingest_files_with_owners(processor, [source_path], emit_event=False)
    summary = _archive_live_ingest_attempt_summary_info(db_path.parent / "ops.db")
    assert summary is not None and summary.available
    attempts = summary.recent

    assert metrics.succeeded_file_count == 1
    assert len(attempts) == 1
    assert attempts[0].status == "completed"
    assert attempts[0].phase == "completed"
    assert attempts[0].needed_file_count == 1
    assert attempts[0].succeeded_file_count == 1
    assert attempts[0].source_payload_read_bytes == source_path.stat().st_size
    assert {
        "full.provider_parse",
        "full.source_raw_blob_ref_write",
        "full.index.session_upsert",
    }.issubset(metrics.stage_timings_s)


@pytest.mark.asyncio
async def test_live_batch_processor_records_cursor_after_each_converged_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    first_path = root / "first.jsonl"
    second_path = root / "second.jsonl"
    _write_jsonl(
        first_path,
        [
            _codex_session_meta("first-session"),
            _codex_message(
                message_id="first-message",
                role="user",
                text="first converged group",
                timestamp="2026-05-01T00:00:01Z",
            ),
        ],
    )
    _write_jsonl(
        second_path,
        [
            _codex_session_meta("second-session"),
            _codex_message(
                message_id="second-message",
                role="user",
                text="second unconverged group",
                timestamp="2026-05-01T00:00:01Z",
            ),
        ],
    )
    db_path = tmp_path / "live.sqlite"
    cursor = CursorStore(db_path)
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
        converger=_FailingSecondPathConverger(second_path),
    )
    monkeypatch.setattr(
        "polylogue.sources.live.batch._full_parse_progress_groups",
        lambda paths: ([path] for path in paths),
    )

    metrics = await ingest_files_with_owners(processor, [first_path, second_path], emit_event=False)

    first_cursor = cursor.get_record(first_path)
    second_cursor = cursor.get_record(second_path)
    assert metrics.succeeded_file_count == 2
    assert metrics.failed_file_count == 0
    assert first_cursor is not None
    assert first_cursor.parser_fingerprint == "test-parser"
    assert second_cursor is not None
    assert second_cursor.parser_fingerprint == "test-parser"
    debt = cursor.list_convergence_debt()
    assert len(debt) == 1
    assert debt[0].stage == "convergence"
    # Debt is keyed by the unit the converger failed on: on the retained
    # route that is the source observation, before any session exists.
    assert debt[0].subject_type == "source_path"
    assert debt[0].subject_id == str(second_path)


def test_full_parse_progress_groups_bounds_files_by_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [tmp_path / f"{index}.jsonl" for index in range(_FULL_PARSE_PROGRESS_MAX_FILES + 1)]
    monkeypatch.setattr("polylogue.sources.live.batch_support._path_size", lambda path: 1)

    groups = list(_full_parse_progress_groups(paths))

    assert groups == [paths[:_FULL_PARSE_PROGRESS_MAX_FILES], paths[_FULL_PARSE_PROGRESS_MAX_FILES:]]


def test_full_parse_progress_groups_bounds_files_by_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [tmp_path / f"{index}.jsonl" for index in range(5)]
    byte_size = (_FULL_PARSE_PROGRESS_MAX_BYTES // 3) + 1
    monkeypatch.setattr("polylogue.sources.live.batch_support._path_size", lambda path: byte_size)

    groups = list(_full_parse_progress_groups(paths))

    assert sum(byte_size for _ in groups[0]) <= _FULL_PARSE_PROGRESS_MAX_BYTES
    assert groups == [paths[:2], paths[2:4], paths[4:]]


def test_full_parse_progress_groups_admits_a_file_over_group_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [tmp_path / f"{index}.jsonl" for index in range(3)]
    sizes = {paths[0]: 1, paths[1]: _FULL_PARSE_PROGRESS_MAX_BYTES * 2, paths[2]: 1}
    monkeypatch.setattr("polylogue.sources.live.batch_support._path_size", sizes.__getitem__)

    assert list(_full_parse_progress_groups(paths)) == [[paths[0]], [paths[1]], [paths[2]]]


@pytest.mark.asyncio
async def test_live_full_ingest_offloads_sync_work_to_keep_loop_responsive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    source_path = root / "session.jsonl"
    source_path.write_text('{"type":"session_meta","payload":{"id":"responsive"}}\n', encoding="utf-8")
    cursor = CursorStore(tmp_path / "live.sqlite")
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    def slow_full_ingest(
        paths: list[Path],
        *,
        source_name: str,
        heartbeat: object = None,
        attempt_id: str | None = None,
        max_pass_seconds: float | None = None,
        pass_started: float | None = None,
        pre_writer_admissions: object = None,
    ) -> _FullIngestResult:
        del source_name, heartbeat, attempt_id, max_pass_seconds, pass_started, pre_writer_admissions
        time.sleep(0.2)
        return _FullIngestResult(
            succeeded=list(paths),
            failed=[],
            source_payload_read_bytes=sum(path.stat().st_size for path in paths),
            raw_fingerprints={path: f"raw:{path.name}" for path in paths},
        )

    monkeypatch.setattr(processor, "_ingest_full_paths_sync", slow_full_ingest)

    ingest_task = asyncio.create_task(ingest_files_with_owners(processor, [source_path], emit_event=False))
    await asyncio.sleep(0.02)

    assert not ingest_task.done()
    metrics = await ingest_task
    assert metrics.succeeded_file_count == 1


@pytest.mark.asyncio
async def test_live_full_ingest_admits_claude_originspec_fact_artifact(
    workspace_env: dict[str, Path],
) -> None:
    """A Workflow snapshot is raw authority, not a failed zero-session JSON file."""

    root = workspace_env["data_root"] / "claude-projects"
    run_path = root / "fixture" / "workflows" / "wf_live_fact.json"
    run_path.parent.mkdir(parents=True)
    run_path.write_text(
        json.dumps(
            {
                "runId": "wf_live_fact",
                "taskId": "live-fact-admission",
                "status": "running",
            }
        ),
        encoding="utf-8",
    )
    db_path = workspace_env["archive_root"] / "index.db"
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    run_off_event_loop(lambda: bootstrap_archive_root(workspace_env["archive_root"]))
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="claude-code", root=root, layout=export_drop_layout((".json", ".jsonl", ".ndjson"))),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    try:
        metrics = await ingest_files_with_owners(processor, [run_path], emit_event=False)

        # The snapshot is retained raw authority. It carries no session, so the
        # batch reports it as a settled no-session exclusion (xf8qp), never a
        # failure.
        assert metrics.failed_file_count == 0
        assert metrics.excluded_reasons == {REFUSED_NO_SESSIONS: 1}
        with sqlite3.connect(workspace_env["archive_root"] / "source.db") as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM raw_sessions WHERE source_path = ?",
                    (str(run_path),),
                ).fetchone()[0]
                == 1
            )

        from polylogue.analysis.claude_workflow_materializer import materialize_claude_workflow_archive

        summary = materialize_claude_workflow_archive(workspace_env["archive_root"])
        assert summary.current_artifact_count == 1
        assert summary.artifact_counts == {"workflow_run_snapshot": 1}
        assert summary.run_count == 1
        assert any("missing workflow journal" in gap for gap in summary.gaps)
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_live_full_ingest_preserves_complete_workflow_journal_revisions(
    workspace_env: dict[str, Path],
) -> None:
    """A growing Workflow journal retains full raw revisions and advances its current pointer."""

    root = workspace_env["data_root"] / "claude-projects"
    run_id = "wf_live_journal"
    run_path = root / "fixture" / "workflows" / f"{run_id}.json"
    journal_path = root / "fixture" / "subagents" / "workflows" / run_id / "journal.jsonl"
    run_path.parent.mkdir(parents=True)
    journal_path.parent.mkdir(parents=True)
    run_path.write_text(json.dumps({"runId": run_id, "status": "running"}), encoding="utf-8")
    _write_jsonl(
        journal_path,
        [
            {
                "runId": run_id,
                "event": "attempt_started",
                "contentKey": "call-00",
                "attemptId": "attempt-00",
            }
        ],
    )
    db_path = workspace_env["archive_root"] / "index.db"
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    run_off_event_loop(lambda: bootstrap_archive_root(workspace_env["archive_root"]))
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="claude-code", root=root, layout=export_drop_layout((".json", ".jsonl", ".ndjson"))),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    try:
        first = await ingest_files_with_owners(processor, [run_path, journal_path], emit_event=False)
        with journal_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "runId": run_id,
                        "event": "result",
                        "contentKey": "call-00",
                        "attemptId": "attempt-00",
                        "structuredResult": {"ok": True},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        second = await ingest_files_with_owners(processor, [journal_path], emit_event=False)

        # Workflow artifacts are retained raw evidence without sessions, so each
        # pass reports them as settled no-session exclusions (xf8qp), not
        # failures; the full revisions below are what they retain.
        assert first.failed_file_count == second.failed_file_count == 0
        assert first.excluded_reasons == {REFUSED_NO_SESSIONS: 2}
        assert second.excluded_reasons == {REFUSED_NO_SESSIONS: 1}
        with sqlite3.connect(workspace_env["archive_root"] / "source.db") as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM raw_sessions WHERE source_path = ?",
                    (str(journal_path),),
                ).fetchone()[0]
                == 2
            )

        from polylogue.analysis.claude_workflow_materializer import materialize_claude_workflow_archive

        summary = materialize_claude_workflow_archive(workspace_env["archive_root"])
        assert summary.current_artifact_count == 2
        assert summary.retained_raw_revision_count == 3
        assert summary.artifact_counts == {
            "workflow_journal": 1,
            "workflow_run_snapshot": 1,
        }
        assert summary.call_count == 1
        assert summary.journal_result_count == 1
        # Source selection reads the completed frontier inspection.
        await _inspect_accepted_frontier(workspace_env["archive_root"])
        processor.require_cursor_authority()
    finally:
        await archive.close()


def _atof_record(*, session_id: str, uuid: str, timestamp: str, name: str = "hermes.turn.start") -> dict[str, object]:
    return {
        "atof_version": "0.1",
        "kind": "mark",
        "uuid": uuid,
        "timestamp": timestamp,
        "name": name,
        "metadata": {"session_id": session_id},
    }


def _atof_event_uuids_by_session(archive_root: Path) -> dict[str, list[str]]:
    index_db = archive_root / "index.db"
    with sqlite3.connect(index_db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT s.native_id, se.payload_json
            FROM session_events se
            JOIN sessions s ON s.session_id = se.session_id
            WHERE se.event_type = 'hermes_context_span'
            ORDER BY s.native_id, se.position
            """
        ).fetchall()
    event_uuids_by_session: dict[str, list[str]] = {}
    for row in rows:
        payload = json.loads(row["payload_json"])
        event_uuids_by_session.setdefault(row["native_id"], []).append(payload["event_uuid"])
    return event_uuids_by_session


@pytest.mark.asyncio
async def test_live_append_atof_shared_file_multi_session_boundary_retains_all_events(
    workspace_env: dict[str, Path],
) -> None:
    """Regression test for polylogue-flxh: real ~/.hermes/observability/
    nemo-relay/atof/events.jsonl is ONE file shared across every Hermes
    session on the install (live evidence: 3+ distinct hermes session ids
    interleaved in one file), unlike Claude Code/Codex where one JSONL file
    is always exactly one session.

    Drives a growth batch whose bytes span two distinct hermes session ids
    -- the real shared-file shape -- and proves session A's new event
    (a-turn-2) is NOT lost while session B (b-turn-1) is created alongside
    it. Also proves idempotent replay: re-ingesting the identical growth
    batch a second time must not duplicate or otherwise change the stored
    state (the producer's own UUID+scope-phase dedup, unaffected by this
    fix, must still hold end-to-end).

    Root cause (found empirically, corrected the original polylogue-flxh
    diagnosis which assumed the bug was in the incremental-append path):
    the ATOF summary message's text used to embed live event counts, which
    changed on every reparse. The archive's membership classifier
    (session_revision_membership.classify_membership_revisions) requires
    message content to be an unchanging prefix across revisions to safely
    recognize append-only growth -- a count-bearing summary broke that
    invariant, so a genuinely-monotonic growth batch looked like a
    non-monotonic edit and the classifier conservatively kept the stale
    revision. Fixed by making the summary message text session-id-only
    (stable forever); ATOF is also now origin-scoped away from the
    incremental-append path entirely (Fable-adjudicated Direction 3),
    since a real ATOF file can span a session boundary in a growth delta
    in a way Claude Code/Codex/Beads JSONL files structurally cannot.
    """
    root = workspace_env["data_root"] / "hermes-observability"
    root.mkdir(parents=True)
    source_path = root / "events.jsonl"
    # The live writer binds ops writes to the archive root that owns them.
    db_path = workspace_env["archive_root"] / "index.db"
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="hermes", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    try:
        _write_jsonl(
            source_path,
            [
                _atof_record(session_id="atof-session-a", uuid="a-turn-1", timestamp="2026-07-18T00:00:00Z"),
            ],
        )
        await _inspect_accepted_frontier(workspace_env["archive_root"])
        await ingest_files_with_owners(processor, [source_path], emit_event=False)

        # Growth batch spans a session boundary -- the real shared-file shape.
        with source_path.open("a", encoding="utf-8") as handle:
            for record in (
                _atof_record(session_id="atof-session-a", uuid="a-turn-2", timestamp="2026-07-18T00:00:01Z"),
                _atof_record(session_id="atof-session-b", uuid="b-turn-1", timestamp="2026-07-18T00:00:02Z"),
            ):
                handle.write(json.dumps(record) + "\n")
        await _inspect_accepted_frontier(workspace_env["archive_root"])
        await ingest_files_with_owners(processor, [source_path], emit_event=False)

        # fs1.14: a resolvable profile root (the watched directory) now
        # artifact- AND profile-qualifies the observer session identity --
        # see hermes_identity.profile_key / hermes_spans.atof_session_provider_id.
        from polylogue.sources.parsers.hermes_identity import profile_key

        expected_key = profile_key(root)
        event_uuids_by_session = _atof_event_uuids_by_session(workspace_env["archive_root"])
        assert event_uuids_by_session.get(f"observer:atof:atof-session-a@profile-{expected_key}") == [
            "a-turn-1",
            "a-turn-2",
        ]
        assert event_uuids_by_session.get(f"observer:atof:atof-session-b@profile-{expected_key}") == ["b-turn-1"]

        # Idempotent replay: re-ingesting the SAME growth batch bytes again
        # (e.g. a poll cycle firing before the cursor advanced, or a daemon
        # restart replaying its tail) must not duplicate or lose anything.
        await _inspect_accepted_frontier(workspace_env["archive_root"])
        await ingest_files_with_owners(processor, [source_path], emit_event=False)
        replayed = _atof_event_uuids_by_session(workspace_env["archive_root"])
        assert replayed == event_uuids_by_session
        await _inspect_accepted_frontier(workspace_env["archive_root"])
        processor.require_cursor_authority()
    finally:
        await archive.close()


async def _inspect_accepted_frontier(archive_root: Path) -> None:
    """Run the daemon's accepted-frontier inspection convergence stage.

    Source selection refuses until the frontier is inspected, and the daemon
    runs that inspection between live passes, never inside one.
    """
    from polylogue.storage.frontier_inspection import inspect_prepared_raw_authority_frontier
    from tests.infra.live_ingest import prepared_live_convergence_owner

    async with prepared_live_convergence_owner(archive_root) as owner:
        await owner.run_convergence_sync(
            "test.live-watcher.frontier",
            inspect_prepared_raw_authority_frontier,
            archive_root,
            input_demand=owner._compute_adapter.amend_current_input_demand,
        )


def _browser_capture_payload(*, provider_session_id: str, assistant_turn_id: str, updated_at: str) -> dict[str, object]:
    return {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "capture_id": f"chatgpt-export:{provider_session_id}",
        "provenance": {
            "source_url": f"https://chatgpt.com/c/{provider_session_id}",
            "page_title": "title",
            "captured_at": updated_at,
            "adapter_name": "chatgpt-dom-v1",
            "capture_mode": "snapshot",
        },
        "session": {
            "provider": "chatgpt",
            "provider_session_id": provider_session_id,
            "title": "title",
            "updated_at": updated_at,
            "turns": [
                {"provider_turn_id": "u1", "role": "user", "text": "hello", "ordinal": 0},
                {
                    "provider_turn_id": assistant_turn_id,
                    "role": "assistant",
                    "text": "hi",
                    "ordinal": 1,
                    "attachments": [
                        {
                            "provider_attachment_id": "att-1",
                            "name": "f.md",
                            "mime_type": "text/markdown",
                            "url": "https://chatgpt.com/attachment/1",
                        }
                    ],
                },
            ],
        },
    }


@pytest.mark.asyncio
async def test_live_full_ingest_over_ambiguous_membership_preserves_durable_debt_without_retry(
    workspace_env: dict[str, Path],
) -> None:
    """Regression test for polylogue-emx2 (adoption idempotency), re-corrected.

    History of this exact scenario (two genuinely conflicting browser-capture
    snapshots of one session, where identity is not preserved and the content
    frontier does not strictly grow, so ``classify_membership_revisions``
    synchronously returns ``ambiguous``, never raising):

    - PR #3129 (de0b2df7a) shipped ``raw_membership_authority_complete()``, a
      boolean collapsing three distinct membership-decision states
      (``decision IS NULL`` / ``'ambiguous'`` / ``'deferred'``) into one,
      which counted this scenario as SUCCEEDED.
    - PR #3193 (f6cc1dd8e) narrowed that to only treat genuinely
      async-pending ``decision IS NULL`` as non-failure; a decided
      ``ambiguous``/``deferred`` outcome was made to surface as a fail-closed
      file failure, with a diagnostic log line. This test was rewritten then
      to assert exactly that (fail-closed).
    - PR #3282 (4120c40c2, "defer FTS repair off the live-ingest write path")
      deliberately superseded #3193's fail-closed-as-failure framing for this
      injection point: a decided ``ambiguous``/``deferred`` membership
      conflict is a durably-recorded outcome, not a transient source-file
      failure, and retrying the identical bytes on every daemon restart
      cannot supply the new evidence arbitration needs -- it only burns the
      live catch-up budget. The raw's membership decision now stays
      ``'ambiguous'`` in ``raw_session_memberships`` (still queryable,
      still not admitted to session content), but the cursor treats the
      *file observation* as complete (succeeded, not retried) so unchanged
      bytes are not reparsed forever. This is the case #3193 called
      "surfacing as failed, not deferred"; #3282 reversed that specific
      framing on purpose (see its PR body).

    This test now asserts the #3282 contract: the second, ambiguous
    observation counts as a succeeded file (cursor idempotent, no retry
    churn), the durable decision remains ``'ambiguous'`` for both raws, a
    diagnostic log line is still emitted (unlike the pre-#3129 silence), and
    the first accepted snapshot's content is untouched -- ambiguous status
    never gains deletion/overwrite authority over previously accepted
    content. The genuinely-NULL-must-defer case emx2 actually motivates, and
    the terminal (never-pending) empty-byte-revision-cohort case that *is*
    still fail-closed, are covered separately by
    ``test_raw_membership_decision_pending_distinguishes_null_from_ambiguous``
    and ``test_rewrite_plus_growth_before_planning_fails_closed_to_full_route``
    in ``tests/unit/sources/test_live_batch_support.py``.
    """
    root = workspace_env["data_root"] / "browser-capture"
    chatgpt_dir = root / "chatgpt"
    chatgpt_dir.mkdir(parents=True)
    source_path = chatgpt_dir / "conv-emx2.json"
    # The live writer binds ops writes to the archive root that owns them.
    db_path = workspace_env["archive_root"] / "index.db"
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="browser-capture", root=root, layout=export_drop_layout((".json",))),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    try:
        source_path.write_text(
            json.dumps(
                _browser_capture_payload(
                    provider_session_id="conv-emx2",
                    assistant_turn_id="a1",
                    updated_at="2026-07-18T00:00:00+00:00",
                )
            ),
            encoding="utf-8",
        )
        first = await ingest_files_with_owners(processor, [source_path], emit_event=False)
        assert first.succeeded_file_count == 1
        assert first.failed_file_count == 0

        # Second snapshot: same message count/shape but a genuinely
        # different assistant turn identity -- classify_membership_revisions
        # cannot pick a winner (identity not preserved, frontier doesn't
        # strictly grow) and correctly returns ambiguous, not accepted. Per
        # #3282 this is preserved as durable membership-decision debt, not
        # retried as a transient file failure: the observation itself
        # succeeded (durably acquired and parsed), it just never gained
        # session-content authority.
        source_path.write_text(
            json.dumps(
                _browser_capture_payload(
                    provider_session_id="conv-emx2",
                    assistant_turn_id="a2",
                    updated_at="2026-07-18T00:05:00+00:00",
                )
            ),
            encoding="utf-8",
        )
        await _inspect_accepted_frontier(workspace_env["archive_root"])
        second = await ingest_files_with_owners(processor, [source_path], emit_event=False)
        assert second.succeeded_file_count == 1, "ambiguous membership debt is not retried as a file failure (#3282)"
        assert second.failed_file_count == 0
        await _inspect_accepted_frontier(workspace_env["archive_root"])
        processor.require_cursor_authority()

        record = cursor.get_record(source_path)
        assert record is not None
        assert record.failure_count == 0

        with sqlite3.connect(workspace_env["archive_root"] / "source.db") as conn:
            decisions = conn.execute(
                "SELECT decision FROM raw_session_memberships WHERE logical_source_key = 'chatgpt-export:conv-emx2' "
                "ORDER BY raw_id"
            ).fetchall()
            assert decisions == [("ambiguous",), ("ambiguous",)]
            # Acquisition's placeholder converges to the origin the census parsed.
            assert conn.execute("SELECT DISTINCT origin FROM raw_sessions").fetchall() == [("chatgpt-export",)]

        # The first accepted snapshot's content remains queryable; the
        # ambiguous second observation has no deletion authority over it.
        with sqlite3.connect(workspace_env["archive_root"] / "index.db") as conn:
            assert conn.execute(
                """
                SELECT b.search_text
                FROM sessions AS s
                JOIN messages AS m USING (session_id)
                JOIN blocks AS b USING (message_id)
                WHERE s.native_id = 'conv-emx2'
                ORDER BY m.position, b.position
                """
            ).fetchall() == [("hello",), ("hi",)]
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_live_append_merges_tail_visible_through_public_archive_read(workspace_env: dict[str, Path]) -> None:
    root = workspace_env["data_root"] / "claude-projects"
    project = root / "project"
    project.mkdir(parents=True)
    source_path = project / "session.jsonl"
    # The live writer binds ops writes to the archive root that owns them.
    db_path = workspace_env["archive_root"] / "index.db"
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="claude-code", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    try:
        _write_jsonl(
            source_path,
            [
                _claude_code_message(
                    session_id="session-public-read",
                    uuid="msg-1",
                    role="user",
                    text="first live message",
                    timestamp="2026-05-01T00:00:00Z",
                ),
                _claude_code_message(
                    session_id="session-public-read",
                    uuid="msg-2",
                    parent_uuid="msg-1",
                    role="assistant",
                    text="second live reply",
                    timestamp="2026-05-01T00:00:01Z",
                ),
                _claude_code_message(
                    session_id="session-public-read",
                    uuid="msg-3",
                    parent_uuid="msg-2",
                    role="user",
                    text="third live followup",
                    timestamp="2026-05-01T00:00:02Z",
                ),
            ],
        )
        initial_metrics = await ingest_files_with_owners(processor, [source_path], emit_event=False)

        with source_path.open("a", encoding="utf-8") as handle:
            for record in (
                _claude_code_message(
                    session_id="session-public-read",
                    uuid="msg-4",
                    parent_uuid="msg-3",
                    role="assistant",
                    text="fourth appended reply",
                    timestamp="2026-05-01T00:00:03Z",
                ),
                _claude_code_message(
                    session_id="session-public-read",
                    uuid="msg-5",
                    parent_uuid="msg-4",
                    role="user",
                    text="fifth appended followup",
                    timestamp="2026-05-01T00:00:04Z",
                ),
            ):
                handle.write(json.dumps(record) + "\n")
        append_metrics = await ingest_files_with_owners(processor, [source_path], emit_event=False)

        session = await archive.get_session("claude-code-session:session-public-read")
        assert initial_metrics.full_file_count == 1
        assert append_metrics.append_file_count == 1
        assert append_metrics.full_file_count == 0
        assert append_metrics.source_payload_read_bytes < append_metrics.input_bytes
        assert session is not None
        assert [message.text for message in session.messages] == [
            "first live message",
            "second live reply",
            "third live followup",
            "fourth appended reply",
            "fifth appended followup",
        ]
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_live_ingest_metrics_carry_real_session_identity(workspace_env: dict[str, Path]) -> None:
    """polylogue-20d.13: full/append batches must name the real touched session.

    A regression that drops session-id threading (e.g. reverting
    ``_ArchiveFullWriteResult``/``_AppendResult`` identity fields back to an
    unscoped aggregate) makes this fail even though ``full_file_count`` /
    ``append_file_count`` still look correct -- the counts alone cannot
    prove the daemon can tell session A apart from session B.
    """
    root = workspace_env["data_root"] / "claude-projects"
    project = root / "project"
    project.mkdir(parents=True)
    source_path = project / "session.jsonl"
    # The live writer binds ops writes to the archive root that owns them.
    db_path = workspace_env["archive_root"] / "index.db"
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="claude-code", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    try:
        _write_jsonl(
            source_path,
            [
                _claude_code_message(
                    session_id="session-identity",
                    uuid="msg-1",
                    role="user",
                    text="first live message",
                    timestamp="2026-05-01T00:00:00Z",
                ),
            ],
        )
        full_metrics = await ingest_files_with_owners(processor, [source_path], emit_event=False)
        assert full_metrics.new_sessions == (("claude-code", "claude-code-session:session-identity"),)
        assert full_metrics.updated_sessions == ()

        with source_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    _claude_code_message(
                        session_id="session-identity",
                        uuid="msg-2",
                        parent_uuid="msg-1",
                        role="assistant",
                        text="second appended reply",
                        timestamp="2026-05-01T00:00:01Z",
                    )
                )
                + "\n"
            )
        append_metrics = await ingest_files_with_owners(processor, [source_path], emit_event=False)
        # The append route only ever grows an already-tracked file: this is
        # an EXISTING session growing, never a fresh one -- exactly the
        # session.updated semantics this bead introduces.
        assert append_metrics.new_sessions == ()
        assert append_metrics.updated_sessions == (("claude-code", "claude-code-session:session-identity"),)
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_live_full_ingest_expands_inbox_zip_members(
    workspace_env: dict[str, Path],
) -> None:
    """A ZIP dropped into a watched root ingests every session member (#1683).

    Regression: ZIP paths previously fell through to the byte-level detection
    branch, where ``orjson.loads`` over the ZIP container raised and the whole
    archive was silently marked excluded. The fix expands ZIP members through
    the maintenance acquisition path, so each member becomes a session.
    """
    root = workspace_env["data_root"] / "claude-projects"
    root.mkdir(parents=True)
    member_a = "\n".join(
        json.dumps(record)
        for record in (
            _claude_code_message(
                session_id="zip-session-a",
                uuid="a-1",
                role="user",
                text="first zip member message",
                timestamp="2026-05-01T00:00:00Z",
            ),
        )
    )
    member_b = "\n".join(
        json.dumps(record)
        for record in (
            _claude_code_message(
                session_id="zip-session-b",
                uuid="b-1",
                role="user",
                text="second zip member message",
                timestamp="2026-05-01T00:00:01Z",
            ),
        )
    )
    zip_path = root / "claude-export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("session-a.jsonl", member_a + "\n")
        zf.writestr("session-b.jsonl", member_b + "\n")

    # The live writer binds ops writes to the archive root that owns them.
    db_path = workspace_env["archive_root"] / "index.db"
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="claude-code", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    try:
        metrics = await ingest_files_with_owners(processor, [zip_path], emit_event=False)
        record = cursor.get_record(zip_path)

        session_a = await archive.get_session("claude-code-session:zip-session-a")
        session_b = await archive.get_session("claude-code-session:zip-session-b")

        assert metrics.succeeded_file_count == 1
        assert metrics.failed_file_count == 0
        assert metrics.source_payload_read_bytes > 0
        assert record is not None
        assert record.excluded is False
        assert session_a is not None
        assert session_b is not None
        assert [message.text for message in session_a.messages] == ["first zip member message"]
        assert [message.text for message in session_b.messages] == ["second zip member message"]
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_live_full_ingest_sniffs_zip_provider_for_non_session_siblings(
    workspace_env: dict[str, Path],
) -> None:
    """polylogue-hs3y: a GDPR export ZIP's non-conversation siblings inherit the
    ZIP's dominant provider instead of independently falling back to unknown.

    A ChatGPT GDPR export ZIP legitimately ships several sibling files
    alongside ``conversations.json`` -- ``message_feedback.json``,
    ``shared_conversations.json``, ``user.json``, etc. -- none of which are
    conversation-shaped on their own. When such a ZIP is dropped into a
    provider-agnostic watched root (source name not itself a provider
    token, so ``fallback_provider`` resolves to ``Provider.UNKNOWN``), each
    ZIP member used to be provider-detected independently with a fresh
    ``Provider.UNKNOWN`` seed: the real ``conversations.json`` detected
    cleanly as ChatGPT, but its non-conversation siblings landed in
    ``raw_sessions`` tagged ``origin='unknown-export'`` even though the ZIP
    as a whole is unambiguously a ChatGPT export (this is what the live
    archive's 25 ``unknown-export`` raw rows all turned out to be). Confirm
    every member -- including the non-session sibling -- now carries the
    ZIP's sniffed ``chatgpt-export`` origin.
    """
    root = workspace_env["data_root"] / "inbox"
    root.mkdir(parents=True)
    conversation = {
        "id": "chatgpt-zip-sniff-1",
        "conversation_id": "chatgpt-zip-sniff-1",
        "title": "GDPR export sniff fixture",
        "create_time": 1_704_067_200.0,
        "current_node": "a1",
        "mapping": {
            "u1": {
                "id": "u1",
                "parent": None,
                "children": ["a1"],
                "message": {
                    "id": "u1",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["hello from the export"]},
                    "create_time": 1_704_067_200.0,
                },
            },
            "a1": {
                "id": "a1",
                "parent": "u1",
                "children": [],
                "message": {
                    "id": "a1",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["hi there"]},
                    "create_time": 1_704_067_201.0,
                },
            },
        },
    }
    zip_path = root / "chatgpt-data-2026-04-23.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("conversations.json", json.dumps([conversation]))
        zf.writestr("message_feedback.json", json.dumps([{"conversation_id": "chatgpt-zip-sniff-1", "rating": 1}]))

    # The live writer binds ops writes to the archive root that owns them.
    db_path = workspace_env["archive_root"] / "index.db"
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="inbox", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    try:
        metrics = await ingest_files_with_owners(processor, [zip_path], emit_event=False)

        assert ("inbox", "chatgpt-export:chatgpt-zip-sniff-1") in metrics.new_sessions

        session = await archive.get_session("chatgpt-export:chatgpt-zip-sniff-1")
        assert session is not None

        with sqlite3.connect(workspace_env["archive_root"] / "source.db") as conn:
            rows = conn.execute(
                "SELECT source_path, origin FROM raw_sessions WHERE source_path LIKE ?",
                (f"{zip_path}:%",),
            ).fetchall()
        origins_by_member = {source_path.rsplit(":", 1)[-1]: origin for source_path, origin in rows}
        assert origins_by_member.get("conversations.json") == "chatgpt-export"
        assert origins_by_member.get("message_feedback.json") == "chatgpt-export", (
            f"non-conversation ZIP sibling should inherit the sniffed ZIP provider, got {origins_by_member!r}"
        )
        assert "unknown-export" not in origins_by_member.values()
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_live_full_ingest_detects_provider_when_source_name_is_not_provider(
    workspace_env: dict[str, Path],
) -> None:
    root = workspace_env["data_root"] / "projects"
    project = root / "project"
    project.mkdir(parents=True)
    source_path = project / "session.jsonl"
    # The live writer binds ops writes to the archive root that owns them.
    db_path = workspace_env["archive_root"] / "index.db"
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="projects", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    try:
        _write_jsonl(
            source_path,
            [
                _claude_code_message(
                    session_id="session-detected-provider",
                    uuid="msg-1",
                    role="user",
                    text="detected provider message",
                    timestamp="2026-05-01T00:00:00Z",
                ),
            ],
        )
        metrics = await ingest_files_with_owners(processor, [source_path], emit_event=False)

        session = await archive.get_session("claude-code-session:session-detected-provider")
        assert metrics.succeeded_file_count == 1
        assert metrics.failed_file_count == 0
        assert session is not None
        assert [message.text for message in session.messages] == ["detected provider message"]
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_live_full_ingest_excludes_non_session_sidecars_before_raw_storage(
    workspace_env: dict[str, Path],
) -> None:
    root = workspace_env["data_root"] / "projects"
    project = root / "project"
    project.mkdir(parents=True)
    source_path = project / "sessions-index.json"
    source_path.write_text(json.dumps({"sessions": [{"id": "metadata-only"}]}), encoding="utf-8")
    # The live writer binds ops writes to the archive root that owns them.
    db_path = workspace_env["archive_root"] / "index.db"
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="projects", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    try:
        metrics = await ingest_files_with_owners(processor, [source_path], emit_event=False)
        record = cursor.get_record(source_path)

        with sqlite3.connect(db_path) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            raw_count = (
                conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] if "raw_sessions" in tables else 0
            )
            session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] if "sessions" in tables else 0

        assert metrics.succeeded_file_count == 0
        assert metrics.failed_file_count == 0
        assert metrics.source_payload_read_bytes == 0
        assert raw_count == 0
        assert session_count == 0
        assert record is not None
        assert record.excluded is True
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_live_full_ingest_excludes_known_provider_invalid_jsonl_sidecars_before_raw_storage(
    workspace_env: dict[str, Path],
) -> None:
    root = workspace_env["data_root"] / "projects"
    project = root / "project" / "analysis"
    project.mkdir(parents=True)
    source_path = project / "architecture_discussions.jsonl"
    source_path.write_text("not json\nstill not json\n", encoding="utf-8")
    # The live writer binds ops writes to the archive root that owns them.
    db_path = workspace_env["archive_root"] / "index.db"
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="claude-code", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    try:
        metrics = await ingest_files_with_owners(processor, [source_path], emit_event=False)
        record = cursor.get_record(source_path)

        # Raw rows live in the Source tier; reading Index here could never fail.
        with sqlite3.connect(workspace_env["archive_root"] / "source.db") as conn:
            raw_count = conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0]

        assert metrics.succeeded_file_count == 0
        assert metrics.failed_file_count == 0
        assert metrics.source_payload_read_bytes == 0
        assert raw_count == 0
        assert record is not None
        assert record.excluded is True
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_codex_append_uses_existing_session_identity_when_tail_lacks_session_meta(
    workspace_env: dict[str, Path],
) -> None:
    root = workspace_env["data_root"] / "codex-sessions"
    project = root / "project"
    project.mkdir(parents=True)
    source_path = project / "codex-session.jsonl"
    # The live writer binds ops writes to the archive root that owns them.
    db_path = workspace_env["archive_root"] / "index.db"
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=db_path)
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    try:
        _write_jsonl(
            source_path,
            [
                _codex_session_meta("codex-real-session"),
                _codex_message(
                    message_id="msg-1",
                    role="user",
                    text="codex first",
                    timestamp="2026-05-01T00:00:00Z",
                ),
            ],
        )
        await ingest_files_with_owners(processor, [source_path], emit_event=False)

        with source_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    _codex_message(
                        message_id="msg-2",
                        role="assistant",
                        text="codex appended",
                        timestamp="2026-05-01T00:00:01Z",
                    )
                )
                + "\n"
            )
        append_metrics = await ingest_files_with_owners(processor, [source_path], emit_event=False)

        existing = await archive.get_session("codex-session:codex-real-session")
        fallback = await archive.get_session("codex-session:codex-session")
        assert append_metrics.append_file_count == 1
        assert append_metrics.full_file_count == 0
        # The append route acquires the delta, then publishes it through the
        # canonical raw owner (561dbe2ff0); those are its two timed stages.
        assert {"append.source_raw_write", "append.canonical_raw"}.issubset(append_metrics.stage_timings_s)
        assert existing is not None
        assert [message.text for message in existing.messages] == ["codex first", "codex appended"]
        assert fallback is None
    finally:
        await archive.close()


@pytest.mark.parametrize("cursor_state", ["settled", "excluded", "failed", "deferred"])
def test_v5_cursor_reprocesses_unchanged_bytes_through_live_batch(tmp_path: Path, cursor_state: str) -> None:
    root = tmp_path / "src"
    root.mkdir()
    path = root / "session.jsonl"
    path.write_text('{"a":1}\n')
    watcher, full_ingest = _make_watcher(tmp_path, root)
    stat = path.stat()
    watcher._cursor.set(
        path,
        stat.st_size,
        parser_fingerprint="live-batched-v5",
        content_fingerprint=None if cursor_state == "deferred" else "old-revision",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        failure_count=1 if cursor_state in {"excluded", "failed"} else 0,
        excluded=cursor_state == "excluded",
        next_retry_at="2999-01-01T00:00:00+00:00" if cursor_state in {"failed", "deferred"} else None,
        authority=fixture_cursor_authority(path),
    )

    # Use the dispatcher's bulk selection, then the actual batch cursor
    # publication path. Only provider work is substituted by this harness.
    selected, deferred = watcher.classify_ingest_candidates([path])
    assert selected == (path,)
    assert deferred == ()
    _run_watcher_ingest(watcher, list(selected))

    after = path.stat()
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
    )
    assert full_ingest.await_count == 1
    record = watcher._cursor.get_record(path)
    assert record is not None
    assert record.parser_fingerprint == live_watcher._PARSER_FINGERPRINT
    assert record.parser_fingerprint != "live-batched-v5"
    assert not record.excluded
    assert record.failure_count == 0
    assert watcher.classify_ingest_candidates([path]) == ((), ())


def test_parser_fingerprint_change_triggers_reingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)

    asyncio.run(_ingest_one(watcher, f))
    # Derive the changed value from the real constant. A hardcoded sentinel
    # silently becomes a no-op the moment the constant is bumped to it, and
    # this test then asserts nothing.
    changed = f"{live_watcher._PARSER_FINGERPRINT}-changed"
    monkeypatch.setattr(live_watcher, "_PARSER_FINGERPRINT", changed)
    asyncio.run(_ingest_one(watcher, f))

    assert parse_sources.await_count == 2
    record = watcher._cursor.get_record(f)
    assert record is not None
    assert record.parser_fingerprint == changed


def test_live_batch_processor_observes_dynamic_parser_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    monkeypatch.setattr(live_watcher, "_PARSER_FINGERPRINT", "live-batched-dynamic-test")

    assert watcher._batch_processor._current_parser_fingerprint() == "live-batched-dynamic-test"


def test_truncate_rewrite_triggers_reingest(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n{"b":2}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)

    asyncio.run(_ingest_one(watcher, f))
    f.write_text('{"c":3}\n')
    asyncio.run(_ingest_one(watcher, f))

    assert parse_sources.await_count == 2


def test_partial_trailing_line_keeps_cursor_at_last_newline(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    complete = b'{"a":1}\n'
    partial = b'{"b":'
    f.write_bytes(complete + partial)
    watcher, parse_sources = _make_watcher(tmp_path, root)

    asyncio.run(_ingest_one(watcher, f))
    asyncio.run(_ingest_one(watcher, f))

    record = watcher._cursor.get_record(f)
    assert parse_sources.await_count == 1
    assert record is not None
    assert record.byte_size == len(complete + partial)
    assert record.byte_offset == len(complete)
    assert record.last_complete_newline == len(complete)


def test_incomplete_append_event_defers_without_ingest_until_newline(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    complete = b'{"a":1}\n'
    partial = b'{"b":'
    f.write_bytes(complete)
    watcher, parse_sources = _make_watcher(tmp_path, root)
    stat = f.stat()
    watcher._cursor.set(
        f,
        len(complete),
        byte_offset=len(complete),
        last_complete_newline=len(complete),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="base",
        tail_hash=encode_cursor_hash_authority(
            sha256(complete).hexdigest(),
            sha256(complete).hexdigest(),
            ctime_ns=stat.st_ctime_ns,
        ),
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(f),
    )

    f.write_bytes(complete + partial)
    asyncio.run(_ingest_one(watcher, f))
    asyncio.run(_ingest_one(watcher, f))

    record = watcher._cursor.get_record(f)
    assert parse_sources.await_count == 0
    assert record is not None
    assert record.byte_size == len(complete + partial)
    assert record.byte_offset == len(complete)

    f.write_bytes(complete + b'{"b":2}\n')
    asyncio.run(_ingest_one(watcher, f))

    assert parse_sources.await_count == 1


def test_append_after_partial_line_reingests_completed_record(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n{"b":')
    watcher, parse_sources = _make_watcher(tmp_path, root)

    asyncio.run(_ingest_one(watcher, f))
    f.write_text('{"a":1}\n{"b":2}\n')
    asyncio.run(_ingest_one(watcher, f))

    assert parse_sources.await_count == 2
    record = watcher._cursor.get_record(f)
    assert record is not None
    assert record.byte_offset == f.stat().st_size


def test_year_old_file_resumed_triggers_reingest(tmp_path: Path) -> None:
    """A year-old session that gets new lines must be picked up — there
    is no 'live session' concept."""
    root = tmp_path / "src"
    root.mkdir()
    old = root / "old-session.jsonl"
    old.write_text('{"role":"user","content":"original"}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)
    asyncio.run(_ingest_one(watcher, old))
    parse_sources.reset_mock()

    old.write_text('{"role":"user","content":"original"}\n{"role":"user","content":"resumed"}\n')
    asyncio.run(_ingest_one(watcher, old))
    assert parse_sources.await_count == 1


def test_missing_file_is_silent(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    watcher, parse_sources = _make_watcher(tmp_path, root)
    assert not watcher._needs_work(root / "ghost.jsonl")
    assert parse_sources.await_count == 0


def test_parse_failure_is_recorded_and_backed_off(tmp_path: Path) -> None:
    """After a batch failure, cursor state records failure and backs off."""
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"role":"user","content":"a"}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)
    parse_sources.side_effect = RuntimeError("parser sad")

    # First attempt: fails, cursor is set
    asyncio.run(_ingest_one(watcher, f))
    assert parse_sources.await_count == 1
    record = watcher._cursor.get_record(f)
    assert record is not None
    assert record.failure_count == 1
    assert record.next_retry_at is not None

    # Second immediate attempt: file is skipped during backoff.
    parse_sources.reset_mock()
    asyncio.run(_ingest_one(watcher, f))
    assert parse_sources.await_count == 0


def test_ingest_files_emits_observable_batch_metrics(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"role":"user","content":"a"}\n')
    emit = MagicMock()
    watcher, _parse_sources = _make_watcher(tmp_path, root, event_emitter=emit)

    _run_watcher_ingest(watcher, [f], queued_file_count=3, skipped_file_count=2)

    emit.assert_called_once()
    kind = emit.call_args.args[0]
    payload = emit.call_args.args[1]
    assert kind == "ingestion_batch"
    assert payload["queued_file_count"] == 3
    assert payload["needed_file_count"] == 1
    assert payload["skipped_file_count"] == 2
    assert payload["succeeded_file_count"] == 1
    assert payload["failed_file_count"] == 0
    assert payload["source_group_count"] == 1
    assert payload["input_bytes"] == f.stat().st_size
    assert payload["source_payload_read_bytes"] == f.stat().st_size
    assert payload["cursor_fingerprint_read_bytes"] == 2 * f.stat().st_size
    assert payload["read_amplification"] == 1.0
    assert payload["files_per_second"] >= 0
    assert payload["source_mb_per_second"] >= 0
    assert payload["append_file_count"] == 0
    assert payload["full_file_count"] == 1
    assert payload["archive_write_bytes_delta"] >= 0
    assert payload["ingested_session_count"] == 1
    assert payload["ingested_message_count"] == 7
    assert payload["changed_session_count"] == 1
    assert payload["parse_time_s"] >= 0
    assert payload["total_time_s"] >= 0
    # The two parse/write stages this test pins are asserted by value; the
    # batch also times its post-commit raw compaction, whose duration is real
    # elapsed time and not a fixture value.
    stage_timings = payload["stage_timings_s"]
    assert isinstance(stage_timings, dict)
    assert stage_timings["full.index_parsed_write"] == 0.02
    assert stage_timings["full.provider_parse"] == 0.01
    assert set(stage_timings) <= {"full.index_parsed_write", "full.provider_parse", "raw_compaction"}
    assert payload["failed_paths"] == []
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        events = conn.execute(
            "SELECT stage FROM daemon_stage_events ORDER BY observed_at_ms DESC, event_id DESC LIMIT 10"
        ).fetchall()
    assert events[0] == ("completed",)
    assert ("planning",) in events


@pytest.mark.frozen_clock_modules("polylogue.sources.live.watcher", "polylogue.sources.live.cursor")
def test_parse_failure_retries_after_backoff(tmp_path: Path, frozen_clock: FrozenClock) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"role":"user","content":"a"}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)
    parse_sources.side_effect = RuntimeError("parser sad")

    asyncio.run(_ingest_one(watcher, f))
    parse_sources.reset_mock()
    parse_sources.side_effect = None
    past = (frozen_clock.now() - timedelta(seconds=1)).isoformat()
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute("UPDATE ingest_cursor SET next_retry_at = ? WHERE source_path = ?", (past, str(f)))
        conn.commit()

    asyncio.run(_ingest_one(watcher, f))

    assert parse_sources.await_count == 1
    record = watcher._cursor.get_record(f)
    assert record is not None
    assert record.failure_count == 0
    assert record.next_retry_at is None


# --- catch_up bootstrap --------------------------------------------------------


def test_page_admission_acquires_source_without_reading_unavailable_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real page-admission batch route remains source-only while derived-only."""

    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    with ArchiveStore.open_existing(tmp_path, read_only=False):
        pass
    root = tmp_path / "sessions"
    root.mkdir()
    path = root / "degraded-catch-up.jsonl"
    path.write_bytes(
        b'{"type":"session_meta","payload":{"id":"degraded-catch-up"}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"message-0","role":"user",'
        b'"content":[{"type":"input_text","text":"zero"}]}}\n'
    )
    pointer = tmp_path / ".index-active-pointer"
    pointer.write_bytes(b"\xff")
    # Retained preparation is the batch's off-writer route after acquisition.
    # Derived-only mode must not reach it; the guard records any call.
    prepared: list[tuple[str, ...]] = []

    async def recording_retained(
        raw_ids: Sequence[str],
        *,
        on_terminal_refusal: Callable[[tuple[str, ...], RetainedRawDecodeRefusalError], None] | None = None,
    ) -> RetainedReplayOutcome:
        prepared.append(tuple(raw_ids))
        return RetainedReplayOutcome()

    watcher = LiveWatcher(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="codex", root=root),),
        cursor=CursorStore(tmp_path / "cursor.sqlite", ops_db_path=tmp_path / "ops.db"),
        retained_runner=recording_retained,
    )
    set_degraded(
        DegradedReason(
            code="schema_version_mismatch",
            message="derived generation unavailable",
            derived_only=True,
        )
    )
    try:
        _run_watcher_ingest(watcher, [path], queued_file_count=1)
    finally:
        clear_degraded()

    assert prepared == []
    assert pointer.read_bytes() == b"\xff"
    with sqlite3.connect(tmp_path / "source.db") as conn:
        row = conn.execute(
            """SELECT parsed_at_ms, parse_error FROM raw_sessions
               WHERE source_path = ? ORDER BY acquired_at_ms DESC, raw_id DESC LIMIT 1""",
            (str(path),),
        ).fetchone()
    assert row == (None, None)


@pytest.mark.asyncio
async def test_retryable_retained_preparation_event_keeps_error_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retry event preserves the stale-seal comparison for diagnosis."""
    from polylogue.storage.sqlite.reference_seal import ReferenceSealStaleError

    source_path = tmp_path / "session.jsonl"
    source_path.write_bytes(b"{}\n")
    result = _FullIngestResult(
        succeeded=[source_path],
        failed=[],
        source_payload_read_bytes=source_path.stat().st_size,
        acquired_raw_ids=("raw-1",),
        raw_fingerprints={source_path: "raw-1"},
    )
    stale_error = ReferenceSealStaleError("the index.db file incarnation changed after preparation")
    failure = RetainedRawRetryableFailure(raw_id="raw-1", error=stale_error)

    async def source_writer(*_args: object, **_kwargs: object) -> _FullIngestResult:
        return result

    async def retained_runner(
        _raw_ids: Sequence[str],
        *,
        on_terminal_refusal: Callable[[tuple[str, ...], RetainedRawDecodeRefusalError], None] | None = None,
    ) -> RetainedReplayOutcome:
        del on_terminal_refusal
        return RetainedReplayOutcome(failures=(failure,))

    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path)),
        (),
        cursor=cast(CursorStore, SimpleNamespace(_db_path=tmp_path / "cursor.sqlite")),
        parser_fingerprint="test",
        retained_runner=cast(LiveRetainedRunner, retained_runner),
    )
    await asyncio.to_thread(initialize_active_archive_root, tmp_path)
    monkeypatch.setattr(processor, "_run_source_writer", source_writer)
    monkeypatch.setattr(live_batch, "_source_tier_acquisition_required", lambda: False)
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(live_batch, "emit", lambda event, **fields: events.append((event, fields)))

    await processor._ingest_full_paths_prepared(
        [source_path],
        source_name="codex",
        pre_writer_admissions={},
    )

    assert events == [
        (
            "live.ingest.retained_preparation_failed",
            {
                "level": WARNING,
                "outcome": "error",
                "reason": "retryable_preparation",
                "raw_id": "raw-1",
                "error_type": "ReferenceSealStaleError",
                "error_detail": "the index.db file incarnation changed after preparation",
            },
        )
    ]


def test_a_cursored_file_is_rediscovered_without_being_ingested_again(tmp_path: Path) -> None:
    """Anti-vacuity: drop the selection in ``admit_page`` and the ingest runs again."""
    root = tmp_path / "src"
    root.mkdir()
    proj = root / "my-project"
    proj.mkdir()
    f = proj / "s.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)
    asyncio.run(_ingest_one(watcher, f))
    parse_sources.reset_mock()

    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=watcher._sources),
        watcher._sources[0],
    )

    async def _admit() -> dict[str, Any]:
        # Page admission initializes the cursor on the daemon writer.
        coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
        watcher._write_coordinator = coordinator
        try:
            page = await adapter.discover(limit=8)
            assert [Path(cast(Any, item.payload)) for item in page] == [f]
            return dict(await adapter.admit_page(page))
        finally:
            watcher._write_coordinator = None
            assert await coordinator.shutdown(timeout=float("inf"))

    outcomes = asyncio.run(_admit())
    assert [result.outcome for result in outcomes.values()] == [AdmissionOutcome.DUPLICATE]
    assert parse_sources.await_count == 0


def test_page_selection_rebases_device_drift_after_one_prefix_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A remount must not make every later restart rehash stable history."""
    root = tmp_path / "src"
    root.mkdir()
    path = root / "session.jsonl"
    payload = b'{"a":1}\n'
    path.write_bytes(payload)
    watcher, parse_sources = _make_watcher(tmp_path, root)
    stat = path.stat()
    prefix_hash = sha256(payload).hexdigest()
    tail_hash, _ = tail_hash_from_path(path, len(payload))
    watcher._cursor.set(
        path,
        len(payload),
        byte_offset=len(payload),
        last_complete_newline=len(payload),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint=prefix_hash,
        tail_hash=encode_cursor_hash_authority(
            prefix_hash,
            tail_hash,
            ctime_ns=stat.st_ctime_ns,
        ),
        source_name="test",
        st_dev=stat.st_dev + 1,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(path),
    )

    calls = 0
    rebase_batches = 0
    real_hash = cast(Callable[..., tuple[str, int]], live_watcher.__dict__["sha256_range_from_path"])
    real_rebase = watcher._cursor.rebase_authoritative_observations

    def counted_hash(
        source: Path,
        *,
        start_offset: int,
        end_offset: int,
        chunk_size: int = 64 * 1024,
    ) -> tuple[str, int]:
        nonlocal calls
        calls += 1
        return real_hash(
            source,
            start_offset=start_offset,
            end_offset=end_offset,
            chunk_size=chunk_size,
        )

    monkeypatch.setattr(live_watcher, "sha256_range_from_path", counted_hash)

    def counted_rebase(rebases: Iterable[object]) -> int:
        nonlocal rebase_batches
        rebase_batches += 1
        return real_rebase(cast(Iterable[Any], rebases))

    monkeypatch.setattr(watcher._cursor, "rebase_authoritative_observations", counted_rebase)

    assert watcher.classify_ingest_candidates([path])[0] == ()
    rebased = watcher._cursor.get_record(path)
    assert calls == 1
    assert rebase_batches == 1
    assert parse_sources.await_count == 0
    assert rebased is not None
    assert rebased.st_dev == stat.st_dev

    assert watcher.classify_ingest_candidates([path])[0] == ()
    assert calls == 1
    assert parse_sources.await_count == 0


# --- debounce ------------------------------------------------------------------


# --- WatchSource ---------------------------------------------------------------


def test_watch_source_exists_true(tmp_path: Path) -> None:
    src = WatchSource(name="x", root=tmp_path)
    assert src.exists() is True


def test_watch_source_exists_false(tmp_path: Path) -> None:
    src = WatchSource(name="x", root=tmp_path / "nope")
    assert src.exists() is False


def test_watch_source_accepts_configured_suffixes(tmp_path: Path) -> None:
    src = WatchSource(name="x", root=tmp_path, layout=export_drop_layout((".json", ".jsonl")))
    assert src.accepts(tmp_path / "session.json") is True
    assert src.accepts(tmp_path / "session.jsonl") is True
    assert src.accepts(tmp_path / "README.md") is False


def test_claude_watch_source_accepts_declared_tool_result_extensions_and_extensionless_files(
    tmp_path: Path,
) -> None:
    """Watcher admission follows the declared Claude Code layout."""
    source = WatchSource(name="claude-code", root=tmp_path)
    session = tmp_path / "-home-user-repo" / "00000000-0000-4000-8000-000000000001"
    tool_results = session / "tool-results"

    assert source.accepts(tool_results / "toolu.json") is True
    assert source.accepts(tool_results / "toolu.txt") is True
    assert source.accepts(tool_results / "toolu.html") is True
    assert source.accepts(tool_results / "toolu") is True
    assert source.accepts(session / "notes.txt") is False


def test_source_accepts_prefers_most_specific_nested_root(tmp_path: Path) -> None:
    """A nested explicit root owns its files regardless of source order."""
    root = tmp_path / "codex"
    sessions = root / "sessions"
    sessions.mkdir(parents=True)
    path = sessions / "session.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    watcher = LiveWatcher(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (
            WatchSource(name="codex-state", root=root, layout=export_drop_layout((".sqlite",))),
            WatchSource(name="codex", root=sessions, layout=export_drop_layout((".jsonl",))),
        ),
        cursor=CursorStore(tmp_path / "cursor.db"),
    )
    assert watcher._source_accepts(path) is True
    assert watcher._source_name_for(path) == "codex"
    assert watcher._batch_processor._source_name_for(path) == "codex"
    directory_source = watcher._source_for_directory(sessions)
    assert directory_source is not None
    assert directory_source.name == "codex"
    discovered = _bounded_source_paths(
        watcher._sources[1], watcher._sources, limit=8, after=None
    ) + _bounded_source_paths(watcher._sources[0], watcher._sources, limit=8, after=None)
    assert discovered == [path]


def test_inbox_source_accepts_zip_and_archive_formats() -> None:
    """#1683: inbox must accept .zip (GDPR exports), .json, .jsonl, .ndjson."""
    from polylogue.sources.live.watcher import default_sources

    inbox = next(s for s in default_sources() if s.name == "inbox")
    for name in ("export.zip", "conversations.json", "session.jsonl", "events.ndjson"):
        assert inbox.accepts(inbox.root / name)
        assert inbox.accepts(inbox.root / "export" / name)
    assert not inbox.accepts(inbox.root / "readme.txt")


def test_claude_default_source_admits_only_its_declared_layout() -> None:
    """Claude live admission follows the declared layout the OriginSpec rules anchor to."""
    from polylogue.sources.live.watcher import default_sources

    claude = next(source for source in default_sources() if source.name == "claude-code")
    session = "138e259e-435f-4259-8c68-dbd5aa9f9837"
    assert claude.accepts(claude.root / "-home-user-repo" / f"{session}.jsonl")
    assert claude.accepts(claude.root / "-home-user-repo" / session / "workflows" / "wf.json")
    # Unanchored, the workflow rule used to admit this at any depth.
    assert not claude.accepts(claude.root / "workflows" / "wf.json")
    assert not claude.accepts(
        claude.root / ".claude" / "worktrees" / "agent-1" / "-home-user-repo" / f"{session}.jsonl"
    )


def test_claude_todos_default_source_watches_its_own_sibling_root() -> None:
    """polylogue-t0p: ~/.claude/todos/ is watched separately from ~/.claude/projects/."""
    from polylogue.sources.live.watcher import default_sources

    todos = next(source for source in default_sources() if source.name == "claude-code-todos")
    claude = next(source for source in default_sources() if source.name == "claude-code")

    assert todos.root != claude.root
    assert todos.root.name == "todos"
    assert todos.accepts(todos.root / "138e259e-435f-4259-8c68-dbd5aa9f9837.json")
    assert not todos.accepts(todos.root / "nested" / "138e259e-435f-4259-8c68-dbd5aa9f9837.json")


def test_claude_history_source_only_walks_direct_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import polylogue.paths as polylogue_paths
    import polylogue.sources.live.watcher as live_watcher
    from polylogue.sources.live.discovery import _bounded_source_paths

    claude_root = tmp_path / ".claude"
    (claude_root / "other").mkdir(parents=True)
    history = claude_root / "history.jsonl"
    history.write_text("{}\n")
    (claude_root / "other" / "history.jsonl").write_text("{}\n")
    monkeypatch.setattr(polylogue_paths, "claude_code_path", lambda: claude_root / "projects")
    source = next(source for source in live_watcher.default_sources() if source.name == "claude-code-history")

    assert not source.admits_directory(claude_root / "other")
    assert _bounded_source_paths(source, (source,), limit=8, after=None) == [history]


def test_browser_capture_spool_is_default_json_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import polylogue.paths as polylogue_paths
    from polylogue.sources.live.watcher import default_sources

    spool = tmp_path / "browser-capture"
    monkeypatch.setattr(
        polylogue_paths,
        "browser_capture_spool_root",
        lambda: spool,
    )

    browser_capture = next(s for s in default_sources() if s.name == "browser-capture")

    assert browser_capture.root == spool
    assert browser_capture.accepts(spool / "chatgpt" / "capture.json") is True
    assert browser_capture.accepts(spool / "chatgpt" / "capture.jsonl") is False
    assert browser_capture.accepts(spool / "browser-actions" / "a1" / "action.json") is False


# --- end-to-end via watchfiles -------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_files_max_pass_seconds_bounds_one_pass_and_preserves_progress(
    tmp_path: Path,
) -> None:
    """polylogue-11cg9: watcher.catch_up.chunk / watcher.live_ingest.full held
    the sole archive writer for however long a batch's full-ingest records
    took to parse and write, with only a file/byte size bound (4 files/16MiB
    for a catch-up chunk; nothing at all for a plain watch-flush batch) and
    no wall-clock bound -- de2a's own incident was an in-size-bounds 7 MB
    chunk that held the writer for 860s. A single session write cannot be
    split mid-transaction, so the checkpoint this budget can safely offer is
    *between* records/groups, never inside one: a zero budget still admits
    the first record to guarantee forward progress, so exactly
    one of three small same-progress-group codex sessions should be written
    this pass, the other two must be left as ordinary un-recorded backlog
    (not succeeded, not failed -- no cursor written, no retry backoff), and a
    follow-up call without a budget must finish them with nothing lost.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    paths = [root / f"session-{index}.jsonl" for index in range(3)]
    for index, path in enumerate(paths):
        _write_jsonl(
            path,
            [
                _codex_session_meta(f"budget-session-{index}"),
                _codex_message(
                    message_id=f"budget-message-{index}",
                    role="user",
                    text=f"time-budget checkpoint {index}",
                    timestamp="2026-08-02T00:00:00Z",
                ),
            ],
        )

    db_path = tmp_path / "live.sqlite"
    cursor = CursorStore(db_path)
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    # The zero-budget law admits the first item, then leaves its siblings
    # retryable. Keep asyncio and worker settlement on the real host clock.
    bounded = await ingest_files_with_owners(processor, paths, emit_event=False, max_pass_seconds=0.0)

    assert bounded.succeeded_file_count == 1
    assert bounded.failed_file_count == 0
    assert bounded.time_budget_exceeded is True
    recorded = [path for path in paths if cursor.get_record(path) is not None]
    assert len(recorded) == 1
    unrecorded = [path for path in paths if path not in recorded]
    assert len(unrecorded) == 2

    remainder = await ingest_files_with_owners(processor, unrecorded, emit_event=False)

    assert remainder.succeeded_file_count == 2
    assert remainder.failed_file_count == 0
    assert remainder.time_budget_exceeded is False
    for path in paths:
        assert cursor.get_record(path) is not None
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 3


@pytest.mark.asyncio
async def test_archive_write_budget_leaves_unwritten_page_tail_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.sources.live.metrics import REFUSED_UNATTEMPTED_TIME_BUDGET

    root = tmp_path / "sessions"
    root.mkdir()
    paths = [root / f"session-{index}.jsonl" for index in range(3)]
    for index, path in enumerate(paths):
        _write_jsonl(
            path,
            [
                _codex_session_meta(f"write-budget-{index}"),
                _codex_message(
                    message_id=f"message-{index}",
                    role="user",
                    text=f"write budget {index}",
                    timestamp="2026-08-02T00:00:00Z",
                ),
            ],
        )
    cursor = CursorStore(tmp_path / "live.sqlite")
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=None)),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    monkeypatch.setattr(
        live_batch,
        "_ingest_pass_exhausted",
        lambda *, max_pass_seconds, pass_started, checkpoint: checkpoint == "archive_write_record",
    )
    bounded = await ingest_files_with_owners(processor, paths, emit_event=False, max_pass_seconds=30.0)

    assert bounded.succeeded_file_count == 1
    assert bounded.time_budget_exceeded is True
    assert bounded.excluded_paths == {str(path): REFUSED_UNATTEMPTED_TIME_BUDGET for path in paths[1:]}
    assert [path for path in paths if cursor.get_record(path) is not None] == [paths[0]]


@pytest.mark.asyncio
async def test_acquisition_is_checkpointed_per_file_not_once_per_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """polylogue-ipyvj: the budget must be checked at every work item.

    Acquisition -- read, fingerprint, publish the blob -- runs under the same
    writer hold as the archive write and dominates it on real transcripts:
    a 15 MB catch-up chunk of three files spent 39.7 s there before the
    archive-write loop reached its first checkpoint, and the hold released
    44.7 s into a 30 s bound. Checking between acquired files bounds that
    overshoot by one file.

    Anti-vacuity: delete the ``full_acquisition_file`` checkpoint and all
    three files are read and blob-published before anything is refused, so
    the two deferred files come back as ``archive write skipped this raw``
    with every byte already read.
    """
    from polylogue.sources.live.metrics import REFUSED_UNATTEMPTED_TIME_BUDGET

    root = tmp_path / "sessions"
    root.mkdir()
    paths = [root / f"session-{index}.jsonl" for index in range(3)]
    for index, path in enumerate(paths):
        _write_jsonl(
            path,
            [
                _codex_session_meta(f"acquisition-session-{index}"),
                _codex_message(
                    message_id=f"acquisition-message-{index}",
                    role="user",
                    text=f"acquisition checkpoint {index}",
                    timestamp="2026-08-02T00:00:00Z",
                ),
            ],
        )

    cursor = CursorStore(tmp_path / "live.sqlite")
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    # A zero pass budget still admits the first item (the forward-progress
    # guarantee), then refuses the next item at the per-file checkpoint. This
    # avoids replacing the process-wide clock used by asyncio's executor
    # shutdown.
    bounded = await ingest_files_with_owners(processor, paths, emit_event=False, max_pass_seconds=0.0)

    assert bounded.succeeded_file_count == 1
    assert bounded.failed_file_count == 0
    assert bounded.time_budget_exceeded is True
    assert bounded.excluded_reasons == {REFUSED_UNATTEMPTED_TIME_BUDGET: 2}
    # Two files were refused before a byte of them was read.
    one_file_bytes = paths[0].stat().st_size
    assert bounded.source_payload_read_bytes < 2 * one_file_bytes
    assert [path for path in paths if cursor.get_record(path) is not None] == [paths[0]]


@pytest.mark.asyncio
@pytest.mark.parametrize("file_count", [1, 2])
async def test_a_file_whose_acquisition_outlasts_the_hold_still_lands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: FrozenClock, file_count: int
) -> None:
    """Admitted work finishes and publishes its cursor past diagnostic thresholds."""
    from polylogue.core.write_hold import enter_write_hold, exit_write_hold
    from polylogue.sources.live.batch import _file_observation

    root = tmp_path / "sessions"
    root.mkdir()
    paths = [root / f"slow-{index}.jsonl" for index in range(file_count)]
    for index, path in enumerate(paths):
        _write_jsonl(
            path,
            [
                _codex_session_meta(f"slow-session-{index}"),
                _codex_message(
                    message_id=f"slow-message-{index}",
                    role="user",
                    text=f"slow capture {index}",
                    timestamp="2026-08-02T00:00:00Z",
                ),
            ],
        )

    def slow_observation(stat: Any) -> tuple[int, int, int, int, int]:
        # Each file's acquisition under the writer outlasts the 30 s hold.
        frozen_clock.advance(31)
        return _file_observation(stat)

    monkeypatch.setattr("polylogue.sources.live.batch._file_observation", slow_observation)
    cursor = CursorStore(tmp_path / "live.sqlite")
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=None)),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    token = enter_write_hold("watcher.live_ingest.full", 30.0)
    try:
        result = await ingest_files_with_owners(processor, paths, emit_event=False)
    finally:
        exit_write_hold(token)

    assert result.failed_file_count == 0
    assert result.succeeded_file_count == file_count
    assert [path for path in paths if cursor.get_record(path) is not None] == paths
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone() == (file_count,)


@pytest.mark.asyncio
async def test_zero_hold_threshold_does_not_refuse_acquired_files(tmp_path: Path) -> None:
    """Admitted work finishes and publishes its cursor past diagnostic thresholds."""
    from polylogue.core.write_hold import enter_write_hold, exit_write_hold

    root = tmp_path / "sessions"
    root.mkdir()
    paths = [root / f"session-{index}.jsonl" for index in range(3)]
    for index, path in enumerate(paths):
        _write_jsonl(
            path,
            [
                _codex_session_meta(f"hold-session-{index}"),
                _codex_message(
                    message_id=f"hold-message-{index}",
                    role="user",
                    text=f"hold bound {index}",
                    timestamp="2026-08-02T00:00:00Z",
                ),
            ],
        )

    cursor = CursorStore(tmp_path / "live.sqlite")
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )

    token = enter_write_hold("watcher.catch_up.chunk", 0.0)
    try:
        result = await ingest_files_with_owners(processor, paths, emit_event=False)
    finally:
        exit_write_hold(token)

    assert result.succeeded_file_count == 3
    assert result.failed_file_count == 0
    assert all(cursor.get_record(path) is not None for path in paths)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 3


def test_lock_contention_is_retryable_and_corruption_is_not() -> None:
    """Anti-vacuity: treating every OperationalError as retryable would hide a
    malformed database behind a warning; treating none as retryable killed
    the daemon on 2026-09-05."""
    from polylogue.sources.live.watcher import _is_retryable_lock_error

    assert _is_retryable_lock_error(sqlite3.OperationalError("database is locked"))
    assert not _is_retryable_lock_error(sqlite3.OperationalError("database disk image is malformed"))


def test_decided_unresolved_membership_reconciles_the_cursor_instead_of_re_reading(tmp_path: Path) -> None:
    """A decided-ambiguous verdict stops catch-up re-reading the same bytes.

    polylogue-i03t8: such a raw is never parsed and never reaches the index,
    so ``_archived_cursor_row`` cannot see it and reconciliation reported
    INCOMPATIBLE -- ``_needs_work`` stayed True on every start and the daemon
    re-read the whole file to reach the same verdict. The retained bytes are
    the proof of what was consumed, so the cursor is restored from them and
    only a changed observation reopens full ingest.

    Anti-vacuity: without ``_decided_unresolved_cursor_row`` the first
    ``_needs_work`` here is True and the cursor stays absent.
    """
    from polylogue.archive.session_revision_membership import MembershipClassification
    from polylogue.pipeline.ids import session_revision_projection

    source_root = tmp_path / "src"
    source_root.mkdir()
    source_path = source_root / "decided-unresolved.jsonl"
    payload = b'{"native_id":"decided-unresolved"}\n'
    source_path.write_bytes(payload)

    initialize_active_archive_root(tmp_path)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="decided-unresolved",
        messages=[ParsedMessage(provider_message_id="m0", role=Role.USER, text="ambiguous content")],
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path=str(source_path),
            canonical_source_path=str(source_path),
            acquired_at_ms=1,
        )
    seed_membership_census(tmp_path, [(raw_id, [session])], parser_fingerprint="test-parser")
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        publish_prepared_membership_classification(
            archive,
            "codex-session:decided-unresolved",
            MembershipClassification((), (), (raw_id,)),
            {raw_id: session},
            {raw_id: session_revision_projection(session)},
            decided_at_ms=2,
        )

    watcher, _full_ingest = _make_watcher(
        tmp_path,
        source_root,
        sources=(WatchSource(name="codex", root=source_root),),
    )
    assert watcher._cursor.get_record(source_path) is None

    assert watcher._needs_work(source_path) is False
    record = watcher._cursor.get_record(source_path)
    assert record is not None
    assert record.byte_offset == len(payload)
    assert watcher._needs_work(source_path) is False

    source_path.write_bytes(payload + b'{"native_id":"decided-unresolved-2"}\n')
    assert watcher._needs_work(source_path) is True
    watcher.stop()


@pytest.mark.parametrize(
    ("materialized_at_ms", "decided_at_ms"),
    [(1, 2), (2, 1), (3, 3)],
    ids=["newer-decided", "newer-materialized", "same-acquisition"],
)
def test_cursor_reconciliation_restores_the_newest_archived_outcome(
    tmp_path: Path, materialized_at_ms: int, decided_at_ms: int
) -> None:
    """A missing cursor restores to the newest settled raw across both outcome classes.

    polylogue-ez5b9 (11.F065): reconciliation took any materialized raw and
    consulted a decided-unresolved raw only when none existed. With an older
    materialized A and a newer decided-ambiguous B that extends it, the
    cursor was restored to A's shorter prefix and every restart re-ingested
    B to reach the same verdict. The newest acquisition wins whichever class
    it is in; equal acquisition times fall back to ``raw_id``, as each class
    orders itself.

    Anti-vacuity: restore the materialized-first ``or`` and the
    newer-decided case restores ``len(short)`` and still needs work.
    """
    from polylogue.archive.revision_authority import decided_unresolved_membership_sql
    from polylogue.archive.session_revision_membership import MembershipClassification
    from polylogue.pipeline.ids import session_revision_projection

    source_root = tmp_path / "src"
    source_root.mkdir()
    source_path = source_root / "newest-outcome.jsonl"
    short = b'{"native_id":"newest-outcome"}\n'
    long = short + b'{"native_id":"newest-outcome","turn":2}\n'
    source_path.write_bytes(long)
    materialized_payload, decided_payload = (long, short) if materialized_at_ms > decided_at_ms else (short, long)

    def session(*message_ids: str) -> ParsedSession:
        return ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="newest-outcome",
            messages=[
                ParsedMessage(provider_message_id=message_id, role=Role.USER, text=message_id)
                for message_id in message_ids
            ],
        )

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        materialized = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=materialized_payload,
            source_path=str(source_path),
            canonical_source_path=str(source_path),
            acquired_at_ms=materialized_at_ms,
        )
        decided = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=decided_payload,
            source_path=str(source_path),
            canonical_source_path=str(source_path),
            acquired_at_ms=decided_at_ms,
        )
        parsed = {materialized: session("m0"), decided: session("m0", "m1")}
    seed_membership_census(
        tmp_path,
        [(raw_id, [parsed_session]) for raw_id, parsed_session in parsed.items()],
        parser_fingerprint="test-parser",
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        publish_prepared_membership_classification(
            archive,
            "codex-session:newest-outcome",
            MembershipClassification((materialized,), (), (decided,)),
            parsed,
            {raw_id: session_revision_projection(parsed_session) for raw_id, parsed_session in parsed.items()},
            decided_at_ms=4,
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert [
            row[0]
            for row in conn.execute(
                f"SELECT r.raw_id FROM raw_sessions AS r WHERE {decided_unresolved_membership_sql('r')}"
            )
        ] == [decided], "sanity: the second raw is a decided-unresolved outcome"
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert [row[0] for row in conn.execute("SELECT raw_id FROM sessions")] == [materialized], (
            "sanity: the first raw is the materialized outcome"
        )

    watcher, _full_ingest = _make_watcher(
        tmp_path,
        source_root,
        sources=(WatchSource(name="codex", root=source_root),),
    )
    newest_payload = max(
        (materialized_at_ms, materialized, materialized_payload),
        (decided_at_ms, decided, decided_payload),
    )[2]
    try:
        needs_work = watcher._needs_work(source_path)
        record = watcher._cursor.get_record(source_path)
        assert record is not None
        assert record.byte_offset == len(newest_payload)
        assert needs_work is (newest_payload != long)
    finally:
        watcher.stop()


def test_discovery_claims_nested_sessions_and_declared_suffixes_only(tmp_path: Path) -> None:
    """The one production walk: which files the dispatcher's discovery claims.

    A Claude Code projects root claims the session in its project directory
    and the subagent transcript below its session directory, and nothing
    else: not a stray file at the root, not undeclared files beside the
    session, not a runtime dependency tree.

    Anti-vacuity: accept every suffix and the ``.toml``/``.md`` files appear;
    walk outside the layout and the root orphan and ``site-packages`` file
    appear; stop descending and the subagent transcript disappears.
    """

    root = tmp_path / "src"
    subagents = root / "-my-project" / "some-uuid" / "subagents"
    subagents.mkdir(parents=True)
    session = root / "-my-project" / "session.jsonl"
    session.write_text('{"a":1}\n')
    orphan = root / "orphan.jsonl"
    orphan.write_text('{"a":1}\n')
    agent = subagents / "agent-abc123.jsonl"
    agent.write_text('{"a":1}\n')
    (root / "-my-project" / "config.toml").write_text("x=1")
    (root / "-my-project" / "README.md").write_text("# hi")
    dependency = root / "venv" / "lib" / "site-packages" / "generated.jsonl"
    dependency.parent.mkdir(parents=True)
    dependency.write_text('{"not":"a session"}\n')

    source = WatchSource(name="claude-code", root=root)
    assert set(_bounded_source_paths(source, (source,), limit=32, after=None)) == {session, agent}

    gemini_root = tmp_path / "gemini"
    gemini_root.mkdir()
    gemini_session = gemini_root / "session.json"
    gemini_session.write_text('{"sessionId":"s1","messages":[]}\n')
    (gemini_root / "notes.md").write_text("# no")
    gemini = WatchSource(name="gemini-cli", root=gemini_root, layout=export_drop_layout((".json", ".jsonl")))
    assert _bounded_source_paths(gemini, (gemini,), limit=32, after=None) == [gemini_session]


def test_a_watch_event_wakes_the_dispatcher_which_ingests_the_new_file(tmp_path: Path) -> None:
    """End to end on the one route: observe -> hint -> discover -> admit -> ingest.

    Anti-vacuity: stop bumping the intake revision (or stop setting the
    wakeup) in ``_note_intake_hint`` and the revision/wakeup assertions go
    red; break page admission and the ingest never runs.
    """

    root = tmp_path / "src"
    root.mkdir()
    # Page admission and the batch's Source bodies run on the daemon writer.
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    watcher, parse_sources = _make_watcher(tmp_path, root, write_coordinator=coordinator)
    wakeup = asyncio.Event()
    watcher._intake_wakeup = wakeup
    source = watcher._sources[0]
    before = watcher.intake_revision(source)

    created = root / "session.jsonl"
    created.write_text('{"role":"user","content":"a"}\n')
    watcher._note_intake_hint(created)

    assert watcher.intake_revision(source) > before
    assert wakeup.is_set()

    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=watcher._sources),
        source,
    )

    async def _admit() -> dict[str, Any]:
        try:
            page = await adapter.discover(limit=8)
            assert [Path(cast(Any, item.payload)) for item in page] == [created]
            return dict(await adapter.admit_page(page))
        finally:
            assert await coordinator.shutdown(timeout=float("inf"))

    outcomes = asyncio.run(_admit())
    assert [result.outcome for result in outcomes.values()] == [AdmissionOutcome.ADMITTED]
    assert parse_sources.await_count == 1


@pytest.mark.frozen_clock_modules("polylogue.sources.live.watcher", "polylogue.sources.live.cursor")
def test_stale_deferral_escalates_when_recorded_byte_size_lags_the_file(
    tmp_path: Path, frozen_clock: FrozenClock
) -> None:
    """polylogue-3r36h: the ``size != cursor.byte_size`` branch escalates too.

    Its sibling (``size == cursor.byte_size``) already ages a deferred
    observation out and records a durable failure. This branch returned False
    on a stat match with no escalation and no ``mark_failed``, so a cursor
    whose recorded ``byte_size`` lagged the file parked forever, invisible to
    ``list_retry_records``.

    Anti-vacuity: delete the age-gated escalation from that branch and the
    final ``failure_count == 1`` drops back to 0.
    """
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    original = b'{"a":1}\n'
    f.write_bytes(original)
    watcher, _parse_sources = _make_watcher(tmp_path, root)
    # Grow the file first, then record a cursor that carries the GROWN file's
    # stat but the ORIGINAL byte_size: the stat-match fast path fires while
    # ``size == cursor.byte_size`` is false, which is the branch under test.
    f.write_bytes(original + (b"x" * 4096))
    stat = f.stat()
    watcher._cursor.set(
        f,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="base",
        tail_hash=encode_cursor_hash_authority(
            sha256(original).hexdigest(),
            sha256(original).hexdigest(),
            ctime_ns=stat.st_ctime_ns,
        ),
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        authority=fixture_cursor_authority(f),
    )

    # Fresh observation: still plausibly an in-progress writer.
    assert watcher._needs_work(f) is False
    record = watcher._cursor.get_record(f)
    assert record is not None
    assert record.failure_count == 0

    _backdate_cursor(tmp_path, f, now=frozen_clock.now(), seconds_ago=live_watcher._STUCK_DEFERRED_APPEND_AGE_S + 1)

    # Same unchanged stat, now far too old to be an active writer: the
    # unbounded probe finds no complete trailing record anywhere, so this
    # becomes a durable, retryable failure instead of another silent park.
    assert watcher._needs_work(f) is False
    record = watcher._cursor.get_record(f)
    assert record is not None
    assert record.failure_count == 1


@pytest.mark.asyncio
async def test_a_budgeted_pass_with_a_no_session_file_stays_a_retryable_attempt(
    tmp_path: Path,
) -> None:
    """A pass that settles one no-session file and leaves the rest unattempted
    is not a whole-attempt UNSUPPORTED_SHAPE refusal.

    Anti-vacuity (Codex): the no-session disposition required only an empty
    retry list, and a time-budget omission is not in it, so ordinary backlog
    left by the budget was recorded as a non-retryable attempt.
    """
    from polylogue.core.enums import IngestOutcome
    from polylogue.sources.live.batch import LiveBatchProcessor

    root = tmp_path / "claude-projects"
    root.mkdir()
    paths = [root / f"silent-{index}.jsonl" for index in range(3)]
    for index, path in enumerate(paths):
        path.write_text(
            json.dumps(
                {
                    "type": "user",
                    "message": {"role": "user", "content": ""},
                    "uuid": f"u{index}",
                    "sessionId": f"silent-{index}",
                    "timestamp": "2026-01-01T00:00:00Z",
                }
            )
            + "\n",
            encoding="utf-8",
        )
    cursor = CursorStore(tmp_path / "live.sqlite")
    run_off_event_loop(lambda: bootstrap_archive_root(tmp_path))
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=None)),
        (WatchSource(name="claude-code", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    bounded = await ingest_files_with_owners(processor, paths, emit_event=False, max_pass_seconds=0.0)

    assert bounded.time_budget_exceeded is True
    assert set(bounded.settled_exclusion_paths.values()) == {REFUSED_NO_SESSIONS}
    with sqlite3.connect(cursor._ops_db_path) as ops:
        (outcome_code,) = ops.execute("SELECT outcome_code FROM ingest_attempts ORDER BY rowid DESC LIMIT 1").fetchone()
    assert outcome_code != IngestOutcome.UNSUPPORTED_SHAPE.value


def test_cold_build_cursor_corroboration_reads_the_candidate_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-slc55: a cold build corroborates cursors against its candidate.

    The writer publishes into the inactive candidate generation; the active
    index is empty until promotion. Anti-vacuity: resolving the active index
    here demotes every cursor the build just wrote and re-ingests the file.
    """
    from polylogue.sources.live import cold_build

    candidate = tmp_path / ".index-generations" / "gen-1" / "index.db"
    registered = SimpleNamespace(generation=SimpleNamespace(index_path=str(candidate)))
    monkeypatch.setattr(cold_build, "active_cold_build_generation", lambda _root=None: registered)
    assert live_watcher._published_index_path(tmp_path) == candidate

    monkeypatch.setattr(cold_build, "active_cold_build_generation", lambda _root=None: None)
    monkeypatch.setattr(live_watcher, "resolve_active_index_path", lambda root: root / "index.db")
    assert live_watcher._published_index_path(tmp_path) == tmp_path / "index.db"


@pytest.mark.asyncio
@pytest.mark.parametrize("moved_after_hash", [False, True], ids=["unchanged", "appended-before-admission"])
async def test_cursor_reconciliation_hashes_off_the_writer_and_rechecks_under_it(
    tmp_path: Path, moved_after_hash: bool
) -> None:
    """Intake selection hashes source bytes without the writer.

    Only the cursor restore is admitted, and under the writer it re-checks the
    file it hashed: a file appended while the restore awaits the writer (after
    every read the decision made) refuses the restore and is selected for
    ingest instead. Anti-vacuity: running selection through the
    writer (the old ``watcher.intake.select`` admission) holds the lease
    around the hash and fails the first assertion in ``observed_hash``; a
    restore without the re-check records a cursor for bytes it never proved.
    """
    from polylogue.archive.session_revision_membership import MembershipClassification
    from polylogue.core.write_lease import current_write_lease
    from polylogue.pipeline.ids import session_revision_projection
    from polylogue.storage.sqlite.write_lease import write_lease

    source_root = tmp_path / "src"
    source_root.mkdir()
    source_path = source_root / "off-writer.jsonl"
    payload = b'{"native_id":"off-writer"}\n'
    source_path.write_bytes(payload)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="off-writer",
        messages=[ParsedMessage(provider_message_id="m0", role=Role.USER, text="settled content")],
    )

    def settle_decided_raw() -> None:
        # Archive setup takes synchronous leases, so it runs off the loop.
        initialize_active_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path=str(source_path),
                canonical_source_path=str(source_path),
                acquired_at_ms=1,
            )
        seed_membership_census(tmp_path, [(raw_id, [session])], parser_fingerprint="test-parser")
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            publish_prepared_membership_classification(
                archive,
                "codex-session:off-writer",
                MembershipClassification((), (), (raw_id,)),
                {raw_id: session},
                {raw_id: session_revision_projection(session)},
                decided_at_ms=2,
            )

    await asyncio.to_thread(settle_decided_raw)

    admitted: list[str] = []

    class RecordingCoordinator:
        async def run(self, actor: str, operation: Callable[[], Awaitable[Any]], /) -> Any:
            raise AssertionError(f"selection must not take a whole-operation writer: {actor}")

        async def run_sync(self, actor: str, function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
            admitted.append(actor)
            if moved_after_hash:
                # The source's writer appends while the restore waits for
                # admission: after every read the decision made.
                with source_path.open("ab") as handle:
                    handle.write(b'{"native_id":"off-writer","turn":2}\n')

            def leased() -> Any:
                with write_lease(actor, archive_root=tmp_path):
                    return function(*args, **kwargs)

            return await asyncio.to_thread(leased)

    watcher, _full_ingest = _make_watcher(
        tmp_path,
        source_root,
        sources=(WatchSource(name="codex", root=source_root),),
        write_coordinator=RecordingCoordinator(),
    )
    from polylogue.sources.live.batch_support import sha256_range_from_path as real_hash

    hashed: list[Path] = []

    def observed_hash(path: Path, **kwargs: Any) -> Any:
        assert current_write_lease() is None, "the reconciliation hash ran under the writer"
        hashed.append(path)
        return real_hash(path, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(live_watcher, "sha256_range_from_path", observed_hash)
        selected, pending = await watcher.classify_ingest_candidates_off_writer([source_path])

    assert hashed == [source_path]
    assert pending == ()
    assert "watcher.intake.select" not in admitted
    assert admitted == ["watcher.intake.cursor_reconcile"]
    record = watcher._cursor.get_record(source_path)
    if moved_after_hash:
        assert selected == (source_path,)
        assert record is None
    else:
        assert selected == ()
        assert record is not None
        assert record.byte_offset == len(payload)
    watcher.stop()


@pytest.mark.asyncio
@pytest.mark.uses_real_clock("waits for a physical watcher selection thread to settle")
async def test_repeated_cancel_retains_watcher_selection_until_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second cancel cannot abandon the physical selection or its failure."""
    root = tmp_path / "src"
    root.mkdir()
    watcher, _ = _make_watcher(tmp_path, root, write_coordinator=cast(Any, object()))
    entered = threading.Event()
    release_cleanup = threading.Event()
    finished = threading.Event()

    def select_then_fail(_paths: Sequence[Path]) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
        entered.set()
        try:
            if not release_cleanup.wait(timeout=5):
                raise AssertionError("test did not release physical selection")
            raise RuntimeError("synthetic selection cleanup failure")
        finally:
            finished.set()

    monkeypatch.setattr(watcher, "classify_ingest_candidates", select_then_fail)
    caller = asyncio.get_running_loop().create_task(watcher.classify_ingest_candidates_off_writer(()))
    try:
        assert await asyncio.to_thread(entered.wait, 2), "physical selection never started"
        caller.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(caller), timeout=0.05)
        except TimeoutError:
            pass
        except BaseException:
            pass
        caller.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(caller), timeout=0.05)
        except TimeoutError:
            returned_before_cleanup = False
        except BaseException:
            returned_before_cleanup = True
        else:
            returned_before_cleanup = True
    finally:
        release_cleanup.set()
        assert await asyncio.to_thread(finished.wait, 2), "physical selection did not settle"

    try:
        await caller
    except BaseException as failure:
        observed = failure
    else:
        pytest.fail("cancelled selection unexpectedly returned successfully")

    assert not returned_before_cleanup, "repeated cancellation released selection ownership early"
    assert isinstance(observed, BaseExceptionGroup)
    flattened: list[BaseException] = []

    def collect(error: BaseException) -> None:
        if isinstance(error, BaseExceptionGroup):
            for nested in error.exceptions:
                collect(nested)
        else:
            flattened.append(error)

    collect(observed)
    assert any(isinstance(error, asyncio.CancelledError) for error in flattened)
    assert any(isinstance(error, RuntimeError) and "cleanup failure" in str(error) for error in flattened)
    watcher.stop()
