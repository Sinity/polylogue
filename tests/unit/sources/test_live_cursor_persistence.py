"""Cursor correctness at the raw-acquired/index-persistence boundary."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.core.enums import Provider
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveRawParsedWriteResult, ArchiveStore


def _claude_message(
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
        "message": {
            "role": role,
            "content": text if role == "user" else [{"type": "text", "text": text}],
        },
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _raw_parse_states(archive_root: Path, source_path: Path) -> list[tuple[int | None, str | None]]:
    with sqlite3.connect(archive_root / "source.db") as conn:
        return [
            (row[0], row[1])
            for row in conn.execute(
                "SELECT parsed_at_ms, parse_error FROM raw_sessions WHERE source_path = ? ORDER BY source_index",
                (str(source_path),),
            )
        ]


def _lock_first_index_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ArchiveStore._write_parsed_precedence_result
    attempts = 0

    def lock_once(
        self: ArchiveStore,
        session: ParsedSession,
        *,
        raw_id: str,
        source_index: int,
        stage_timings_s: dict[str, float] | None,
        stage_timing_prefix: str,
        manage_transaction: bool,
        preacquired_attachment_blobs: dict[int, tuple[bytes | None, int, str]] | None = None,
    ) -> ArchiveRawParsedWriteResult:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise sqlite3.OperationalError("database is locked")
        return original(
            self,
            session,
            raw_id=raw_id,
            source_index=source_index,
            stage_timings_s=stage_timings_s,
            stage_timing_prefix=stage_timing_prefix,
            manage_transaction=manage_transaction,
            preacquired_attachment_blobs=preacquired_attachment_blobs,
        )

    monkeypatch.setattr(ArchiveStore, "_write_parsed_precedence_result", lock_once)


def _watcher(archive: Polylogue, root: Path) -> LiveWatcher:
    return LiveWatcher(
        archive,
        (WatchSource(name="claude-code", root=root),),
        debounce_s=0,
        cursor=CursorStore(archive.archive_root / "index.db"),
    )


@pytest.mark.asyncio
async def test_full_ingest_lock_keeps_raw_pending_and_requeues_without_eof_reconciliation(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = workspace_env["data_root"] / "claude-projects"
    root.mkdir(parents=True)
    source_path = root / "session.jsonl"
    _write_jsonl(
        source_path,
        [
            _claude_message(
                session_id="full-lock",
                uuid="message-1",
                role="user",
                text="persist after retry",
                timestamp="2026-07-10T00:00:00Z",
            )
        ],
    )
    archive = Polylogue(
        archive_root=workspace_env["archive_root"],
        db_path=workspace_env["archive_root"] / "index.db",
    )
    watcher = _watcher(archive, root)
    _lock_first_index_persistence(monkeypatch)

    try:
        watcher._pending_paths.add(source_path)
        assert await watcher._flush_pending() is True

        assert watcher._cursor.get_record(source_path) is None
        assert watcher._needs_work(source_path) is True
        assert watcher._cursor.get_record(source_path) is None
        assert _raw_parse_states(workspace_env["archive_root"], source_path) == [(None, None)]
        assert source_path in watcher._pending_paths

        assert await watcher._flush_pending() is True

        cursor = watcher._cursor.get_record(source_path)
        session = await archive.get_session("claude-code:full-lock")
        assert cursor is not None
        assert cursor.byte_offset == source_path.stat().st_size
        assert session is not None
        assert [message.text for message in session.messages] == ["persist after retry"]
        assert all(
            parsed_at is not None and error is None
            for parsed_at, error in _raw_parse_states(workspace_env["archive_root"], source_path)
        )
    finally:
        watcher.cancel_pending()
        await archive.close()


@pytest.mark.asyncio
async def test_append_lock_preserves_prior_cursor_and_retries_tail_through_watcher(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = workspace_env["data_root"] / "claude-projects"
    root.mkdir(parents=True)
    source_path = root / "session.jsonl"
    first = _claude_message(
        session_id="append-lock",
        uuid="message-1",
        role="user",
        text="before lock",
        timestamp="2026-07-10T00:00:00Z",
    )
    second = _claude_message(
        session_id="append-lock",
        uuid="message-2",
        parent_uuid="message-1",
        role="assistant",
        text="after retry",
        timestamp="2026-07-10T00:00:01Z",
    )
    _write_jsonl(source_path, [first])
    archive = Polylogue(
        archive_root=workspace_env["archive_root"],
        db_path=workspace_env["archive_root"] / "index.db",
    )
    watcher = _watcher(archive, root)

    try:
        watcher._pending_paths.add(source_path)
        assert await watcher._flush_pending() is True
        prior_cursor = watcher._cursor.get_record(source_path)
        assert prior_cursor is not None

        with source_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(second) + "\n")
        _lock_first_index_persistence(monkeypatch)
        watcher._pending_paths.add(source_path)
        assert await watcher._flush_pending() is True

        locked_cursor = watcher._cursor.get_record(source_path)
        assert locked_cursor is not None
        assert locked_cursor.byte_offset == prior_cursor.byte_offset
        assert locked_cursor.failure_count == 0
        assert watcher._needs_work(source_path) is True
        assert (None, None) in _raw_parse_states(workspace_env["archive_root"], source_path)
        assert source_path in watcher._pending_paths

        assert await watcher._flush_pending() is True

        cursor = watcher._cursor.get_record(source_path)
        session = await archive.get_session("claude-code:append-lock")
        assert cursor is not None
        assert cursor.byte_offset == source_path.stat().st_size
        assert session is not None
        assert [message.text for message in session.messages] == ["before lock", "after retry"]
        assert all(
            parsed_at is not None and error is None
            for parsed_at, error in _raw_parse_states(workspace_env["archive_root"], source_path)
        )
    finally:
        watcher.cancel_pending()
        await archive.close()


def test_archived_cursor_reconciliation_rejects_parsed_raw_without_index_materialization(
    workspace_env: dict[str, Path],
) -> None:
    root = workspace_env["data_root"] / "claude-projects"
    root.mkdir(parents=True)
    source_path = root / "session.jsonl"
    _write_jsonl(
        source_path,
        [
            _claude_message(
                session_id="acquired-only",
                uuid="message-1",
                role="user",
                text="raw only",
                timestamp="2026-07-10T00:00:00Z",
            )
        ],
    )
    archive = Polylogue(
        archive_root=workspace_env["archive_root"],
        db_path=workspace_env["archive_root"] / "index.db",
    )
    watcher = _watcher(archive, root)
    with ArchiveStore.open_existing(workspace_env["archive_root"], read_only=False) as store:
        raw_id = store.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=source_path.read_bytes(),
            source_path=str(source_path),
            source_index=0,
            acquired_at_ms=1,
        )
        store.mark_raw_parse_succeeded(raw_id, provider=Provider.CLAUDE_CODE)

    try:
        assert watcher._needs_work(source_path) is True
        assert watcher._cursor.get_record(source_path) is None
    finally:
        watcher.cancel_pending()
