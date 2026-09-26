from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue.operations.intake_adapters import _bounded_source_paths
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.batch_support import _AppendPlan, encode_cursor_hash_authority
from polylogue.sources.live.cursor import CursorStore
from tests.infra.frozen_clock import FrozenClock


@dataclass(frozen=True, slots=True)
class _SelectionPlan:
    """What one page-admission selection decided, in the shape assertions read.

    The dispatcher's bounded walk plus ``LiveWatcher.select_ingest_candidates``
    is the one route that decides this now; ``_plan_catch_up`` and its scan were the
    deleted second one.
    """

    candidates: tuple[Path, ...]
    needed: tuple[Path, ...]

    @property
    def skipped_file_count(self) -> int:
        return len(self.candidates) - len(self.needed)

    @property
    def needed_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.needed)


def _select_plan(watcher: LiveWatcher, root: Path) -> _SelectionPlan:
    source = next(source for source in watcher._sources if source.root == root)
    candidates = tuple(_bounded_source_paths(source, watcher._sources, limit=64, after=None))
    return _SelectionPlan(candidates=candidates, needed=watcher.select_ingest_candidates(candidates))


def _write_archive_blob(archive_root: Path, blob_hash: bytes | str, payload: bytes) -> None:
    blob_hash_hex = blob_hash.hex() if isinstance(blob_hash, bytes) else blob_hash.lower()
    blob_path = archive_root / "blob" / blob_hash_hex[:2] / blob_hash_hex[2:]
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(payload)


def test_page_selection_carries_statted_candidates_without_payload_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    changed = root / "changed.jsonl"
    unchanged = root / "unchanged.jsonl"
    changed.write_text('{"role":"user","content":"new"}\n')
    unchanged.write_text('{"role":"user","content":"old"}\n')
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    cursor = CursorStore(tmp_path / "cursor.sqlite")
    watcher = LiveWatcher(cast(Any, polylogue), (WatchSource(name="test", root=root),), cursor=cursor)
    stat = unchanged.stat()
    unchanged_digest = hashlib.sha256(unchanged.read_bytes()).hexdigest()
    cursor.set(
        unchanged,
        stat.st_size,
        byte_offset=stat.st_size,
        last_complete_newline=stat.st_size,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="already-known",
        # Modern cursor authority (#2710): the hot stat-match skip only trusts
        # cursors carrying an encoded prefix/tail digest bound to this file's
        # ctime. A legacy cursor without one deliberately takes one full
        # route to (re-)establish that authority instead of hot-skipping.
        tail_hash=encode_cursor_hash_authority(
            unchanged_digest,
            unchanged_digest,
            ctime_ns=stat.st_ctime_ns,
        ),
        source_name="test",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )

    def fail_fingerprint_file(path: Path) -> tuple[str, int]:
        raise AssertionError(f"unchanged page selection should not fingerprint payloads: {path}")

    monkeypatch.setattr(live_watcher, "fingerprint_file", fail_fingerprint_file)
    plan = _select_plan(watcher, root)

    assert list(plan.candidates) == [changed, unchanged]
    assert plan.needed == (changed,)
    assert plan.skipped_file_count == 1
    assert plan.needed_bytes == changed.stat().st_size


def test_page_selection_repairs_missing_cursor_from_archive_source_row(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session_blob_ref
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "src"
    root.mkdir()
    archived = root / "archived.jsonl"
    archived.write_text('{"type":"session_meta","payload":{"id":"archived"}}\n', encoding="utf-8")
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    cursor = CursorStore(tmp_path / "ops.db")
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    with sqlite3.connect(source_db) as conn:
        # Reconciliation (#2710) now re-verifies the archived blob hash
        # against the live file's real bytes, so this must be the file's
        # actual digest rather than an arbitrary placeholder.
        blob_hash = hashlib.sha256(archived.read_bytes()).digest()
        raw_id = write_source_raw_session_blob_ref(
            conn,
            origin="codex-session",
            source_path=str(archived),
            source_index=0,
            blob_hash=blob_hash,
            blob_size=archived.stat().st_size,
            acquired_at_ms=1,
            native_id="archived",
        )
        # Archive reconciliation (#2676) only trusts a raw row that is both
        # parsed and materialized into a session, so mark it parsed here.
        conn.execute("UPDATE raw_sessions SET parsed_at_ms = ? WHERE raw_id = ?", (1, raw_id))
        conn.commit()
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, message_count, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("archived", "codex-session", raw_id, 1, b"c" * 32, 1, 1),
        )
        conn.commit()
    _write_archive_blob(tmp_path, blob_hash, archived.read_bytes())
    watcher = LiveWatcher(cast(Any, polylogue), (WatchSource(name="codex", root=root),), cursor=cursor)

    plan = _select_plan(watcher, root)
    record = cursor.get_record(archived)

    assert plan.needed == ()
    assert plan.skipped_file_count == 1
    assert record is not None
    assert record.byte_size == archived.stat().st_size
    assert record.content_fingerprint == blob_hash.hex()
    assert record.parser_fingerprint == live_watcher._PARSER_FINGERPRINT


def test_page_selection_does_not_repair_cursor_from_archive_row_with_missing_blob(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session_blob_ref
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "src"
    root.mkdir()
    archived = root / "archived.jsonl"
    archived.write_text('{"type":"session_meta","payload":{"id":"archived"}}\n', encoding="utf-8")
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    cursor = CursorStore(tmp_path / "ops.db")
    source_db = tmp_path / "source.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    with sqlite3.connect(source_db) as conn:
        write_source_raw_session_blob_ref(
            conn,
            origin="codex-session",
            source_path=str(archived),
            source_index=0,
            blob_hash=b"a" * 32,
            blob_size=archived.stat().st_size,
            acquired_at_ms=1,
            native_id="archived",
        )
    watcher = LiveWatcher(cast(Any, polylogue), (WatchSource(name="codex", root=root),), cursor=cursor)

    plan = _select_plan(watcher, root)

    assert plan.needed == (archived,)
    assert plan.skipped_file_count == 0
    assert cursor.get_record(archived) is None


def test_page_selection_reconciles_browser_capture_cursor_from_archive_origin(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session_blob_ref
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "browser-capture"
    root.mkdir()
    archived = root / "capture.json"
    archived.write_text('{"polylogue_capture_kind":"browser_llm_session"}', encoding="utf-8")
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    cursor = CursorStore(tmp_path / "ops.db")
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    with sqlite3.connect(source_db) as conn:
        # Reconciliation (#2710) now re-verifies the archived blob hash
        # against the live file's real bytes, so this must be the file's
        # actual digest rather than an arbitrary placeholder.
        blob_hash = hashlib.sha256(archived.read_bytes()).digest()
        raw_id = write_source_raw_session_blob_ref(
            conn,
            origin="chatgpt-export",
            source_path=str(archived),
            source_index=0,
            blob_hash=blob_hash,
            blob_size=archived.stat().st_size,
            acquired_at_ms=1,
            native_id="capture",
        )
        # Archive reconciliation (#2676) only trusts a raw row that is both
        # parsed and materialized into a session, so mark it parsed here.
        conn.execute("UPDATE raw_sessions SET parsed_at_ms = ? WHERE raw_id = ?", (1, raw_id))
        conn.commit()
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, message_count, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("capture", "chatgpt-export", raw_id, 1, b"d" * 32, 1, 1),
        )
        conn.commit()
    _write_archive_blob(tmp_path, blob_hash, archived.read_bytes())
    watcher = LiveWatcher(
        cast(Any, polylogue),
        (WatchSource(name="browser-capture", root=root, suffixes=(".json",)),),
        cursor=cursor,
    )

    plan = _select_plan(watcher, root)
    record = cursor.get_record(archived)

    assert plan.needed == ()
    assert plan.skipped_file_count == 1
    assert record is not None
    assert record.source_name == "chatgpt"
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        assert (
            conn.execute("SELECT origin FROM ingest_cursor WHERE source_path = ?", (str(archived),)).fetchone()[0]
            == "chatgpt-export"
        )


def test_codex_append_plan_recovers_identity_from_session_meta_when_source_row_missing(
    tmp_path: Path,
) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "src"
    root.mkdir()
    source = root / "rollout-2026-06-18T02-59-46-conv-hot.jsonl"
    prefix = b'{"timestamp":"2026-06-18T01:05:23.888Z","type":"session_meta","payload":{"id":"conv-hot"}}\n'
    old_content = prefix + b'{"type":"message","payload":{"role":"user","content":"old"}}\n'
    source.write_bytes(old_content)
    old_offset = source.stat().st_size
    with source.open("ab") as handle:
        handle.write(b'{"type":"message","payload":{"role":"assistant","content":"new"}}\n')
    stat = source.stat()
    old_content_digest = hashlib.sha256(old_content).hexdigest()

    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, message_count, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("conv-hot", "codex-session", "missing-source-raw", 1, b"b" * 32, 1, 1),
        )

    cursor = CursorStore(tmp_path / "ops.db")
    cursor.set(
        source,
        old_offset,
        byte_offset=old_offset,
        last_complete_newline=old_offset,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="already-known",
        # Modern cursor authority (#2710): append planning only trusts a
        # cursor carrying an encoded accepted-prefix digest. A legacy cursor
        # without one is correctly refused the append route.
        tail_hash=encode_cursor_hash_authority(
            old_content_digest,
            old_content_digest,
            ctime_ns=stat.st_ctime_ns,
        ),
        source_name="codex",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    plan = processor._append_plan(source)

    assert isinstance(plan, _AppendPlan)
    assert plan.start_offset == old_offset
    # #3539 (polylogue-u19l) retired splicing a synthetic session_meta header
    # into a Codex append payload before hashing/storing it -- the stored
    # blob must stay a literal byte-slice of the live file so live-source
    # byte-identity re-verification stays possible. Recovered identity now
    # flows as a sidecar hint (native_id_hint) instead, applied as the
    # parser's fallback_id at replay time.
    assert plan.native_id_hint == "conv-hot"
    assert plan.payload == source.read_bytes()[old_offset:]


@pytest.mark.frozen_clock_modules("polylogue.sources.live.cursor", "polylogue.sources.live.convergence_debt_retry")
def test_derived_convergence_debt_uses_exponential_retry(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    cursor = CursorStore(tmp_path / "cursor.sqlite")
    cursor.record_convergence_debt(
        stage="derived",
        subject_type="session_id",
        subject_id="conv-hot",
        error="derived stage returned False",
        deferred=True,
    )
    frozen_clock.advance(1)
    cursor.record_convergence_debt(
        stage="derived",
        subject_type="session_id",
        subject_id="conv-hot",
        error="derived stage returned False",
        deferred=True,
    )

    debt = cursor.list_convergence_debt(limit=1)[0]
    retry_at = datetime.fromisoformat(debt.next_retry_at or "")
    failed_at = datetime.fromisoformat(debt.last_failed_at)
    assert debt.status == "deferred"
    assert debt.failure_count == 1
    assert retry_at - failed_at == timedelta(seconds=60)


@pytest.mark.frozen_clock_modules("polylogue.sources.live.cursor", "polylogue.sources.live.convergence_debt_retry")
def test_derived_convergence_debt_advances_after_retry_is_due(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    cursor = CursorStore(tmp_path / "cursor.sqlite")
    cursor.record_convergence_debt(
        stage="derived",
        subject_type="session_id",
        subject_id="conv-hot",
        error="derived stage returned False",
        deferred=True,
    )
    frozen_clock.advance(61)
    cursor.record_convergence_debt(
        stage="derived",
        subject_type="session_id",
        subject_id="conv-hot",
        error="derived stage returned False",
        deferred=True,
    )

    debt = cursor.list_convergence_debt(limit=1)[0]
    retry_at = datetime.fromisoformat(debt.next_retry_at or "")
    failed_at = datetime.fromisoformat(debt.last_failed_at)
    assert debt.failure_count == 2
    assert retry_at - failed_at == timedelta(seconds=120)


def _seed_healthy_hot_skip_cursor(cursor: CursorStore, path: Path, *, source_name: str) -> None:
    """Write a cursor that the byte/fingerprint hot-skip path would trust on its own."""
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cursor.set(
        path,
        stat.st_size,
        byte_offset=stat.st_size,
        last_complete_newline=stat.st_size,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint=digest,
        tail_hash=encode_cursor_hash_authority(digest, digest, ctime_ns=stat.st_ctime_ns),
        source_name=source_name,
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )


def _seed_parsed_source_raw(
    conn: sqlite3.Connection, *, path: Path, native_id: str, raw_id: str, parsed_at_ms: int | None = 1
) -> None:
    conn.execute(
        """
        INSERT INTO raw_sessions (
            raw_id, origin, native_id, source_path, source_index,
            blob_hash, blob_size, acquired_at_ms, parsed_at_ms, revision_kind, revision_authority
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            raw_id,
            "codex-session",
            native_id,
            str(path),
            0,
            b"a" * 32,
            path.stat().st_size,
            1,
            parsed_at_ms,
            "full",
            "byte_proven",
        ),
    )


def test_page_selection_demotes_hot_skip_cursor_when_index_holds_no_sessions(tmp_path: Path) -> None:
    """polylogue-emx2: a cursor claiming acquisition must not skip when the
    index tier that would corroborate materialization is empty -- the
    signature left by a full index reset/rebuild (Finding 8: 14,879 cursors
    skipped 100% of files against an empty post-reset index)."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "src"
    root.mkdir()
    healthy = root / "healthy.jsonl"
    healthy.write_text('{"role":"user","content":"hello"}\n', encoding="utf-8")

    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    cursor = CursorStore(tmp_path / "cursor.sqlite")
    _seed_healthy_hot_skip_cursor(cursor, healthy, source_name="test")

    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    with sqlite3.connect(source_db) as conn:
        _seed_parsed_source_raw(conn, path=healthy, native_id="healthy", raw_id="raw-healthy")
        conn.commit()
    # index.db intentionally left with zero rows in `sessions` -- the
    # post-reset state this bead targets.

    watcher = LiveWatcher(cast(Any, polylogue), (WatchSource(name="test", root=root),), cursor=cursor)
    plan = _select_plan(watcher, root)

    assert plan.needed == (healthy,), "a cursor the index cannot corroborate must be demoted to needed, not hot-skipped"
    assert plan.skipped_file_count == 0


def test_page_selection_trusts_hot_skip_cursor_when_index_corroborates_it(tmp_path: Path) -> None:
    """The corroborated companion to the demotion test above: once the raw
    this cursor claims is actually materialized as a session in the index,
    the hot-skip path is trusted again and no re-ingest is scheduled."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "src"
    root.mkdir()
    healthy = root / "healthy.jsonl"
    healthy.write_text('{"role":"user","content":"hello"}\n', encoding="utf-8")

    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    cursor = CursorStore(tmp_path / "cursor.sqlite")
    _seed_healthy_hot_skip_cursor(cursor, healthy, source_name="test")

    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    with sqlite3.connect(source_db) as conn:
        _seed_parsed_source_raw(conn, path=healthy, native_id="healthy", raw_id="raw-healthy")
        conn.commit()
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, message_count, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("healthy", "codex-session", "raw-healthy", 1, b"c" * 32, 1, 1),
        )
        conn.commit()

    watcher = LiveWatcher(cast(Any, polylogue), (WatchSource(name="test", root=root),), cursor=cursor)
    plan = _select_plan(watcher, root)

    assert plan.needed == ()
    assert plan.skipped_file_count == 1


def test_page_selection_replays_missing_raw_when_index_has_another_session(tmp_path: Path) -> None:
    """A partial index rebuild must not let one session certify another file's cursor."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "src"
    root.mkdir()
    indexed = root / "a-indexed.jsonl"
    missing = root / "b-missing.jsonl"
    for path in (indexed, missing):
        path.write_text('{"role":"user","content":"hello"}\n', encoding="utf-8")

    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    cursor = CursorStore(tmp_path / "cursor.sqlite")
    for path in (indexed, missing):
        _seed_healthy_hot_skip_cursor(cursor, path, source_name="test")

    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    with sqlite3.connect(source_db) as conn:
        _seed_parsed_source_raw(conn, path=indexed, native_id="indexed", raw_id="raw-indexed")
        _seed_parsed_source_raw(conn, path=missing, native_id="missing", raw_id="raw-missing", parsed_at_ms=None)
        conn.commit()
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, message_count, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("indexed", "codex-session", "raw-indexed", 1, b"c" * 32, 1, 1),
        )
        conn.commit()

    watcher = LiveWatcher(cast(Any, polylogue), (WatchSource(name="test", root=root),), cursor=cursor)
    plan = _select_plan(watcher, root)

    assert plan.needed == (missing,)
    assert plan.skipped_file_count == 1


# ---------------------------------------------------------------------------
# Halted sources: excluded where work is SELECTED, and published as state
# ---------------------------------------------------------------------------


class _RecordingCoordinator:
    """Write coordinator double that records every lease it hands out."""

    def __init__(self) -> None:
        self.actors: list[str] = []

    async def run(self, actor: str, operation: Any) -> None:
        self.actors.append(actor)
        await operation()

    async def run_sync(self, actor: str, function: Any, /, *args: Any, **kwargs: Any) -> Any:
        self.actors.append(actor)
        return function(*args, **kwargs)


def _stub_metrics(succeeded: int, paths: list[Path]) -> Any:
    from polylogue.sources.live.metrics import LiveBatchMetrics

    offered = sum(path.stat().st_size for path in paths)
    return LiveBatchMetrics(
        queued_file_count=len(paths),
        needed_file_count=len(paths),
        skipped_file_count=0,
        succeeded_file_count=succeeded,
        failed_file_count=0,
        source_group_count=1,
        input_bytes=offered,
        ingested_bytes=offered if succeeded else 0,
        source_payload_read_bytes=offered if succeeded else 0,
        cursor_fingerprint_read_bytes=0,
        ingest_worker_count_max=1,
        append_file_count=0,
        full_file_count=len(paths),
        archive_bytes_before=0,
        archive_bytes_after=0,
        archive_write_bytes_delta=0,
        parse_time_s=0.0,
        convergence_time_s=0.0,
        total_time_s=0.01,
    )


def _two_source_watcher(
    tmp_path: Path,
    *,
    files_per_source: int = 3,
    write_coordinator: object | None = None,
    event_emitter: Any | None = None,
) -> tuple[Any, list[Path], list[Path]]:
    root = tmp_path / "sources"
    (root / "alpha").mkdir(parents=True)
    (root / "beta").mkdir(parents=True)
    alpha_files = []
    beta_files = []
    for index in range(files_per_source):
        alpha = root / "alpha" / f"a{index}.jsonl"
        beta = root / "beta" / f"b{index}.jsonl"
        alpha.write_text('{"type":"user"}\n', encoding="utf-8")
        beta.write_text('{"type":"user"}\n', encoding="utf-8")
        alpha_files.append(alpha)
        beta_files.append(beta)
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    watcher = LiveWatcher(
        cast(Any, polylogue),
        (
            WatchSource(name="alpha", root=root / "alpha"),
            WatchSource(name="beta", root=root / "beta"),
        ),
        cursor=CursorStore(tmp_path / "cursor.sqlite"),
        write_coordinator=cast(Any, write_coordinator),
        event_emitter=event_emitter,
    )
    return watcher, alpha_files, beta_files


def test_structural_database_error_halts_only_its_own_source() -> None:
    """The production handler records a per-source halt, not just a global flag.

    Anti-vacuity: drop the ``set_source_halt`` call in
    ``handle_structural_database_error`` and this goes red while the
    process-wide flag still passes -- the exact state that existed before,
    which no planner could act on.
    """
    from polylogue.core.degraded import clear_degraded, is_degraded
    from polylogue.core.errors import SchemaVersionMismatchError
    from polylogue.core.source_halts import clear_all_source_halts, source_halt
    from polylogue.sources.live.dedup import handle_structural_database_error, schema_warning_limiter

    clear_all_source_halts()
    clear_degraded()
    schema_warning_limiter.reset()
    try:
        handle_structural_database_error(
            "alpha",
            SchemaVersionMismatchError("index derived schema identity mismatch", current_version=1, expected_version=2),
        )
        halted = source_halt("alpha")
        healthy = source_halt("beta")
        process_wide = is_degraded()
    finally:
        clear_all_source_halts()
        clear_degraded()
        schema_warning_limiter.reset()

    assert halted is not None
    assert halted.code == "schema_version_mismatch"
    assert healthy is None
    assert process_wide is True


def _inbox_watcher(root: Path, archive_root: Path) -> LiveWatcher:
    polylogue = SimpleNamespace(archive_root=archive_root, backend=None)
    return LiveWatcher(
        cast(Any, polylogue),
        (WatchSource(name="inbox", root=root),),
        cursor=CursorStore(archive_root / "cursor.sqlite"),
    )


def _deny_scandir(monkeypatch: pytest.MonkeyPatch, blocked: Path) -> None:
    """Make exactly one directory unreadable to ``os.walk``.

    ``os.walk`` reaches the filesystem through ``os.scandir``, so denying that
    one call reproduces a real permission fault (and exercises ``os.walk``'s own
    ``onerror`` contract) without depending on the test process being
    unprivileged -- under root a ``chmod 000`` directory proves nothing.
    """
    real = os.scandir

    def scandir(directory: Any = ".", *args: Any, **kwargs: Any) -> Any:
        if Path(directory) == blocked:
            raise PermissionError(13, "Permission denied", str(blocked))
        return real(directory, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", scandir)
