"""Read-amplification regression tests for the live ingest path (#1003 follow-on).

Pins three scenarios against the steady-state contract:

- ``S1`` active-append: an active Claude Code session appended to once per
  batch must not re-read the entire file. The amplification ratio
  ``bytes_read_from_source / bytes_appended`` should be ≤ small constant.
- ``S2`` mtime-drift catch-up: a file whose mtime changed but whose content
  did NOT (e.g. backup tool touched it) should not trigger a full-file
  ``fingerprint_file`` call at the watcher stage.
- ``S3`` subagent append: a subagent file whose ``provider_session_id``
  is ``<parent_session>:agent-<id>`` and whose stem differs from the
  session id should still take the append path on subsequent batches,
  not fall back to full re-ingest.

These tests *fail loudly* if the amplification is present and become
regression pins once the fix lands. The harness uses a real
``LiveBatchProcessor`` against a real cursor DB; only ``_process_ingest_batch_sync``
(the heavyweight ingest) is mocked to keep the test fast — file reads
BEFORE that call are still real and counted.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.watcher import LiveWatcher, WatchSource
from tests.infra.io_counter import ReadCounter, read_counter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _claude_code_record(
    *,
    session_id: str,
    uuid: str,
    role: str = "user",
    text: str = "hello",
) -> dict[str, object]:
    return {
        "type": role,
        "uuid": uuid,
        "sessionId": session_id,
        "timestamp": "2026-05-10T00:00:00Z",
        "message": {"role": role, "content": text},
    }


def _claude_code_tool_result_record(*, uuid: str, text: str) -> dict[str, object]:
    """A record with NO sessionId field — common for tool_result lines."""
    return {
        "type": "tool_result",
        "uuid": uuid,
        "timestamp": "2026-05-10T00:00:00Z",
        "message": {"role": "tool", "content": text},
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, records: list[dict[str, object]]) -> int:
    """Append records to the JSONL, return bytes written."""
    payload = ("\n".join(json.dumps(record) for record in records) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(payload)
    return len(payload)


@pytest.fixture
def processor(tmp_path: Path) -> Iterator[tuple[LiveBatchProcessor, Path, Path]]:
    """Real ``LiveBatchProcessor`` with mocked heavyweight ingest.

    Returns ``(processor, source_root, source_file_path_factory)``.
    The factory is a directory the caller can write JSONL files into.
    """
    root = tmp_path / "projects"
    root.mkdir()
    db_path = tmp_path / "index.db"
    cursor = CursorStore(db_path)
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path), config=None)
    proc = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="claude-code", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    # Mock the heavyweight ingest to a no-op success. File reads BEFORE this
    # call (the ones we're measuring) still happen against the real FS.
    fake_summary = SimpleNamespace(
        failed_raw_ids=[],
        parse_failures=0,
        worker_count=1,
        worker_progress_total=0,
        worker_progress_in_flight=0,
        worker_progress_completed=0,
    )
    with patch("polylogue.sources.live.batch._process_ingest_batch_sync", return_value=fake_summary):
        yield proc, root, root


def _seed_initial_ingest(proc: LiveBatchProcessor, path: Path, *, session_id: str = "session-abc") -> None:
    """Run a first full ingest so the cursor + raw_sessions are populated.

    For Claude Code tests we also seed a ``sessions`` row so that
    ``_existing_provider_session_id`` returns the expected value.
    """
    import asyncio

    asyncio.run(proc.ingest_files([path], emit_event=False))
    # Seed the sessions row that ``_existing_provider_session_id``
    # queries — the mocked ingest pipeline doesn't write it.
    with proc._cursor._connect() as conn:
        from polylogue.storage.sqlite.schema import _ensure_schema

        _ensure_schema(conn)
        row = conn.execute(
            "SELECT raw_id FROM raw_sessions WHERE source_path = ? ORDER BY acquired_at DESC LIMIT 1",
            (str(path),),
        ).fetchone()
        assert row is not None, "first ingest must produce a raw_sessions row"
        raw_id = row[0]
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions
                (session_id, source_name, provider_session_id, title,
                 content_hash, version, raw_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"claude-code:{session_id}",
                "claude-code",
                session_id,
                "test session",
                "deadbeef" * 8,
                1,
                raw_id,
                "2026-05-10T00:00:00Z",
                "2026-05-10T00:00:00Z",
            ),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Scenario 1 — active Claude Code session appended to
# ---------------------------------------------------------------------------


class TestActiveAppendNoFullReread:
    """Pins H2 inverse: an active session should not re-read the file in
    full on every batch. After the first ingest, subsequent appends with
    well-formed ``sessionId`` lines must take the append path.
    """

    def test_append_with_sessionid_line_does_not_re_read_whole_file(
        self, processor: tuple[LiveBatchProcessor, Path, Path]
    ) -> None:
        import asyncio

        proc, root, _ = processor
        session_id = "session-abc"
        path = root / f"{session_id}.jsonl"
        _write_jsonl(
            path,
            [_claude_code_record(session_id=session_id, uuid=f"msg-{i}") for i in range(10)],
        )
        first_size = path.stat().st_size
        _seed_initial_ingest(proc, path, session_id=session_id)

        appended_bytes = _append_jsonl(
            path, [_claude_code_record(session_id=session_id, uuid=f"msg-new-{i}") for i in range(3)]
        )

        with read_counter() as counter:
            asyncio.run(proc.ingest_files([path], emit_event=False))

        assert counter.calls_by_site.get("fingerprint_file", 0) == 0, (
            f"append-only batch must not call fingerprint_file; got {counter.summary()}"
        )
        assert counter.calls_by_site.get("blob_store.write_from_path", 0) == 0, (
            f"append-only batch must not stream full file to blob; got {counter.summary()}"
        )
        # Source-payload reads should be small — the append payload plus
        # tail-newline scan at most. Allow a generous 8× safety margin.
        max_allowed = appended_bytes * 8 + 64 * 1024
        assert counter.total_bytes <= max_allowed, (
            f"source-read amplification: appended {appended_bytes} bytes, "
            f"read {counter.total_bytes} bytes\n{counter.summary()}\n"
            f"first_size={first_size}"
        )

    def test_append_with_only_tool_result_lines_does_not_full_reread(
        self, processor: tuple[LiveBatchProcessor, Path, Path]
    ) -> None:
        """Hypothesis H2 specifically: tool_result lines lack ``sessionId``.
        Today's ``_claude_code_tail_matches_existing_identity`` falls back
        to ``existing_id == path.stem``. For ``<session_id>.jsonl`` that
        matches, so this *should* still take the append path.

        Pins the contract so a future refactor that drops the stem-fallback
        doesn't silently amplify reads.
        """
        import asyncio

        proc, root, _ = processor
        session_id = "session-abc"
        path = root / f"{session_id}.jsonl"
        _write_jsonl(
            path,
            [_claude_code_record(session_id=session_id, uuid=f"msg-{i}") for i in range(5)],
        )
        _seed_initial_ingest(proc, path, session_id=session_id)

        appended_bytes = _append_jsonl(
            path,
            [_claude_code_tool_result_record(uuid=f"tr-{i}", text="...") for i in range(5)],
        )

        with read_counter() as counter:
            asyncio.run(proc.ingest_files([path], emit_event=False))

        assert counter.calls_by_site.get("fingerprint_file", 0) == 0, (
            f"tool-result-only append must not call fingerprint_file; got {counter.summary()}"
        )
        assert counter.calls_by_site.get("blob_store.write_from_path", 0) == 0, (
            f"tool-result-only append must not full-reread; got {counter.summary()}"
        )
        max_allowed = appended_bytes * 8 + 64 * 1024
        assert counter.total_bytes <= max_allowed, (
            f"tool-result amplification: appended {appended_bytes}, read {counter.total_bytes}\n{counter.summary()}"
        )


# ---------------------------------------------------------------------------
# Scenario 2 — mtime-drift catch-up (H1)
# ---------------------------------------------------------------------------


class TestMtimeDriftCatchUp:
    """Pins H1: a file whose mtime drifted but whose content is unchanged
    should NOT trigger ``fingerprint_file`` at the watcher's
    ``_needs_work_from_state`` stage.

    Today the slowpath at watcher.py:244 reads the whole file in this case.
    The fix is to trust the (dev, ino, size) triple when content_fingerprint
    is set — mtime alone is not a content-change signal.
    """

    def test_unchanged_content_with_drifted_mtime_does_not_trigger_full_read(self, tmp_path: Path) -> None:
        import asyncio

        root = tmp_path / "projects"
        root.mkdir()
        db_path = tmp_path / "index.db"
        cursor = CursorStore(db_path)
        polylogue = SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path), config=None)

        # Seed: one ingested file, cursor populated.
        path = root / "session-abc.jsonl"
        _write_jsonl(path, [_claude_code_record(session_id="abc", uuid=f"m-{i}") for i in range(5)])
        sources = (WatchSource(name="claude-code", root=root),)

        proc = LiveBatchProcessor(
            cast(Any, polylogue),
            sources,
            cursor=cursor,
            parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        )
        fake_summary = SimpleNamespace(
            failed_raw_ids=[],
            parse_failures=0,
            worker_count=1,
            worker_progress_total=0,
            worker_progress_in_flight=0,
            worker_progress_completed=0,
        )
        with patch("polylogue.sources.live.batch._process_ingest_batch_sync", return_value=fake_summary):
            asyncio.run(proc.ingest_files([path], emit_event=False))

        # Drift the mtime — nothing else.
        stat_before = path.stat()
        future = stat_before.st_mtime + 60
        os.utime(path, (future, future))
        assert path.stat().st_mtime_ns != stat_before.st_mtime_ns

        # Now exercise the watcher's `_needs_work_from_state` path. We
        # build a fresh watcher (which shares the same cursor DB) and
        # call its catch-up loop directly.
        watcher = LiveWatcher(cast(Any, polylogue), sources, cursor=cursor, debounce_s=0.0)

        with read_counter() as counter:
            asyncio.run(watcher._catch_up([root]))

        # The contract: an mtime-only change must not full-read the file.
        # Today this fails because `_needs_work_from_state` calls
        # `fingerprint_file(path)` when mtime drifted, even though size and
        # content_fingerprint are still authoritative.
        assert counter.calls_by_site.get("fingerprint_file", 0) == 0, (
            f"mtime-drift catch-up rehashed the file: {counter.summary()}"
        )


# ---------------------------------------------------------------------------
# Scenario 3 — subagent append (suspect: identity mismatch)
# ---------------------------------------------------------------------------


class TestSubagentAppendDoesNotFullReread:
    """Subagent files have ``provider_session_id = <parent>:agent-<id>``
    and a path stem that differs from both. Subsequent appends with
    ``sessionId = <parent>`` lines should take the append path via the
    ``existing_id.startswith(f"{session_id}:")`` branch.

    Pin the contract: subagent appends must not full-reread.
    """

    def test_subagent_append_with_parent_session_id_takes_append_path(
        self, processor: tuple[LiveBatchProcessor, Path, Path]
    ) -> None:
        import asyncio

        proc, root, _ = processor
        parent_session = "parent-session-xyz"
        subagent_id = f"{parent_session}:agent-007"
        # The on-disk stem is unrelated to the session id — that's
        # the realistic subagent case.
        path = root / "agent-007.jsonl"
        _write_jsonl(
            path,
            [_claude_code_record(session_id=parent_session, uuid=f"msg-{i}") for i in range(5)],
        )
        _seed_initial_ingest(proc, path, session_id=subagent_id)

        appended_bytes = _append_jsonl(
            path, [_claude_code_record(session_id=parent_session, uuid=f"msg-new-{i}") for i in range(3)]
        )

        with read_counter() as counter:
            asyncio.run(proc.ingest_files([path], emit_event=False))

        assert counter.calls_by_site.get("fingerprint_file", 0) == 0, (
            f"subagent append rehashed the file: {counter.summary()}"
        )
        assert counter.calls_by_site.get("blob_store.write_from_path", 0) == 0, (
            f"subagent append full-rewrote the blob: {counter.summary()}"
        )
        max_allowed = appended_bytes * 8 + 64 * 1024
        assert counter.total_bytes <= max_allowed, counter.summary()


class TestCatchUpReadsEachFileAtMostOnce:
    """AC from the prompt: catch-up after daemon restart over a 1000-session
    archive reads each file at most once (modulo failure-retry).

    This pins the steady-state property that the watcher's catch-up loop
    does NOT rehash files whose (dev, ino, size, content_fingerprint)
    state agrees with the cursor, even when mtime drifted between
    daemon restarts (filesystem touch ops, backup tools, etc.).
    """

    def test_catchup_does_not_rehash_unchanged_files_when_mtime_drifts(self, tmp_path: Path) -> None:
        import asyncio

        root = tmp_path / "projects"
        root.mkdir()
        db_path = tmp_path / "index.db"
        cursor = CursorStore(db_path)
        polylogue = SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path), config=None)
        sources = (WatchSource(name="claude-code", root=root),)

        # Seed 50 files with first-time full ingest (smaller than the AC's
        # 1000 to keep the test fast; the per-file behaviour is what we're
        # pinning, scale doesn't change the assertion).
        file_count = 50
        proc = LiveBatchProcessor(
            cast(Any, polylogue),
            sources,
            cursor=cursor,
            parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        )
        fake_summary = SimpleNamespace(
            failed_raw_ids=[],
            parse_failures=0,
            worker_count=1,
            worker_progress_total=0,
            worker_progress_in_flight=0,
            worker_progress_completed=0,
        )
        paths: list[Path] = []
        for i in range(file_count):
            session_id = f"session-{i:04d}"
            path = root / f"{session_id}.jsonl"
            _write_jsonl(
                path,
                [_claude_code_record(session_id=session_id, uuid=f"msg-{j}") for j in range(3)],
            )
            paths.append(path)
        with patch("polylogue.sources.live.batch._process_ingest_batch_sync", return_value=fake_summary):
            asyncio.run(proc.ingest_files(paths, emit_event=False))

        # Drift mtime on every file (simulating filesystem touch ops
        # accumulated between daemon shutdown and restart).
        for path in paths:
            stat_before = path.stat()
            future = stat_before.st_mtime + 3600
            os.utime(path, (future, future))

        # Now restart: a fresh watcher runs catch-up over all files.
        watcher = LiveWatcher(cast(Any, polylogue), sources, cursor=cursor, debounce_s=0.0)
        with read_counter() as counter:
            asyncio.run(watcher._catch_up([root]))

        # AC: catch-up reads each file at most once (modulo failure-retry).
        # Stronger here: catch-up reads NOTHING from any source file because
        # no file's content actually changed.
        full_file_reads = counter.calls_by_site.get("fingerprint_file", 0)
        assert full_file_reads == 0, (
            f"Catch-up rehashed {full_file_reads} files just because mtime drifted; "
            f"each file would be re-read at O(file_size) cost on every daemon restart. "
            f"This was the read-amplification storm in #1003.\n{counter.summary()}"
        )


def _print_counter_for_inspection(counter: ReadCounter) -> None:
    """Helper invoked from pytest `-s` mode to print measurements."""
    print(counter.summary())
