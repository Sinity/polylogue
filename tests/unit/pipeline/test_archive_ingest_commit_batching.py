"""The one-shot compatibility API uses the canonical live intake owner."""

from __future__ import annotations

import asyncio
import fcntl
import multiprocessing
import os
import sqlite3
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.maintenance.offline_guard import ArchiveWriterOwnershipError
from polylogue.pipeline.services.archive_ingest import parse_sources_archive


def _session_file(root: Path) -> Path:
    source = root / "session.jsonl"
    source.write_text(
        '{"type":"user","uuid":"u1","sessionId":"s1","message":{"content":"hello"}}\n'
        '{"type":"assistant","uuid":"a1","sessionId":"s1","message":{"content":"hi"}}\n',
        encoding="utf-8",
    )
    return source


def _hold_daemon_pidfile(root: str, channel: multiprocessing.connection.Connection) -> None:
    path = Path(root) / "daemon.pid"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, str(os.getpid()).encode())
        os.fsync(fd)
        channel.send(True)
        channel.recv()
    finally:
        os.close(fd)
        channel.close()


def test_one_shot_ingest_uses_durable_raw_and_cursor_authority(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = _session_file(source_root)
    declaration = [Source(name="claude-code", path=source)]

    first = asyncio.run(parse_sources_archive(archive_root, declaration))
    second = asyncio.run(parse_sources_archive(archive_root, declaration))

    assert first.processed_ids == {"claude-code:s1"}
    assert second.processed_ids == set()
    with sqlite3.connect(archive_root / "source.db") as raw:
        assert raw.execute("SELECT COUNT(*) FROM raw_artifacts").fetchone() == (1,)
    with sqlite3.connect(archive_root / "index.db") as index:
        assert index.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)
        assert index.execute("SELECT COUNT(*) FROM messages").fetchone() == (2,)
    with sqlite3.connect(archive_root / "ops.db") as ops:
        assert ops.execute("SELECT COUNT(*) FROM ingest_cursor").fetchone() == (1,)


def test_one_shot_ingest_refuses_resident_daemon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.maintenance import offline_guard

    archive_root = tmp_path / "archive"
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = _session_file(source_root)
    monkeypatch.setattr(offline_guard, "resident_daemon_pid", lambda _root: 31415)

    with pytest.raises(ArchiveWriterOwnershipError, match="polylogued PID 31415"):
        asyncio.run(parse_sources_archive(archive_root, [Source(name="claude-code", path=source)]))
    assert not (archive_root / "source.db").exists()


def test_one_shot_ingest_refuses_daemon_held_pidfile(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = _session_file(source_root)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    holder = context.Process(target=_hold_daemon_pidfile, args=(str(archive_root), child))
    holder.start()
    child.close()
    try:
        assert parent.poll(10), "daemon lock holder did not start"
        assert parent.recv() is True
        with pytest.raises(ArchiveWriterOwnershipError, match="polylogued PID"):
            asyncio.run(parse_sources_archive(archive_root, [Source(name="claude-code", path=source)]))
    finally:
        parent.send(True)
        parent.close()
        holder.join(timeout=10)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=10)
    assert holder.exitcode == 0
    assert not (archive_root / "source.db").exists()
    assert not (archive_root / ".one-shot-ingest-owner").exists()


def test_one_shot_ingest_refuses_unclaimed_populated_archive(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = _session_file(source_root)
    with sqlite3.connect(archive_root / "source.db") as raw:
        raw.execute("CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY)")
        raw.execute("INSERT INTO raw_sessions VALUES ('real-session')")
    with pytest.raises(ArchiveWriterOwnershipError, match="already contains archive content"):
        asyncio.run(parse_sources_archive(archive_root, [Source(name="claude-code", path=source)]))
    assert not (archive_root / ".one-shot-ingest-owner").exists()


def test_one_shot_ingest_refuses_unclaimed_initialized_archive(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    with sqlite3.connect(archive_root / "source.db") as raw:
        raw.execute("CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY)")
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = _session_file(source_root)
    with pytest.raises(ArchiveWriterOwnershipError, match="existing archive"):
        asyncio.run(parse_sources_archive(archive_root, [Source(name="claude-code", path=source)]))
    assert not (archive_root / ".one-shot-ingest-owner").exists()
