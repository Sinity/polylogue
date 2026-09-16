"""Cursor correctness at the raw-acquired/index-persistence boundary."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from polylogue import Polylogue
from polylogue.core.enums import Provider
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.archive_tiers import revision_governance as archive_revision_governance
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
    # polylogue-1r9c: _write_parsed_precedence_result is called internally by
    # revision_governance.py (a direct module-internal function reference),
    # not through ArchiveStore's `self.` dispatch -- patch it there.
    original = archive_revision_governance._write_parsed_precedence_result
    attempts = 0

    def lock_once(*args: Any, **kwargs: Any) -> ArchiveRawParsedWriteResult:
        # Deliberately signature-agnostic. Mirroring the production parameter
        # list here has drifted twice (``bulk_build`` in #3183, then
        # ``fresh_build``/``fresh_build_batch``/``prepared_required``/
        # ``prepared_write`` in #4924), and each time the stub raised TypeError
        # instead of the injected lock -- which the ingest routes then handled
        # as a genuine per-file failure, so these tests silently stopped
        # exercising the retryable-lock guard they exist to prove.
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise sqlite3.OperationalError("database is locked")
        return original(*args, **kwargs)

    monkeypatch.setattr(archive_revision_governance, "_write_parsed_precedence_result", lock_once)


async def _admit(watcher: LiveWatcher, path: Path) -> dict[str, object]:
    """Admit one path through the production intake route."""
    from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter

    adapter = FileIntakeAdapter(
        DaemonIntakeContext(
            archive_root=Path(watcher._polylogue.archive_root),
            watcher=watcher,
            sources=watcher._sources,
        ),
        watcher._sources[0],
    )
    return dict(await adapter.admit_page(await adapter.discover(limit=8)))


def _watcher(archive: Polylogue, root: Path) -> LiveWatcher:
    return LiveWatcher(
        archive,
        (WatchSource(name="claude-code", root=root),),
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
        with pytest.raises(sqlite3.OperationalError):
            await _admit(watcher, source_path)

        assert watcher._cursor.get_record(source_path) is None
        assert watcher._needs_work(source_path) is True
        assert watcher._cursor.get_record(source_path) is None
        assert _raw_parse_states(workspace_env["archive_root"], source_path) == [(None, None)]

        await _admit(watcher, source_path)

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
        watcher.stop()
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
        await _admit(watcher, source_path)
        prior_cursor = watcher._cursor.get_record(source_path)
        assert prior_cursor is not None

        with source_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(second) + "\n")
        _lock_first_index_persistence(monkeypatch)
        with pytest.raises(sqlite3.OperationalError):
            await _admit(watcher, source_path)

        locked_cursor = watcher._cursor.get_record(source_path)
        assert locked_cursor is not None
        assert locked_cursor.byte_offset == prior_cursor.byte_offset
        assert locked_cursor.failure_count == 0
        assert watcher._needs_work(source_path) is True
        assert (None, None) in _raw_parse_states(workspace_env["archive_root"], source_path)

        await _admit(watcher, source_path)

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
        watcher.stop()
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
        watcher.stop()
