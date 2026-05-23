from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import _MAX_APPEND_PLAN_PAYLOAD_BYTES, LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore


def test_full_ingest_heartbeats_small_file_groups_with_current_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    first = root / "first.jsonl"
    second = root / "second.jsonl"
    first.write_text('{"type":"session_meta","payload":{"id":"first"}}\n', encoding="utf-8")
    second.write_text('{"type":"session_meta","payload":{"id":"second"}}\n', encoding="utf-8")
    db_path = tmp_path / "archive.sqlite"
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))
    cursor = CursorStore(db_path)
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint="test-parser",
    )
    events: list[tuple[str, Path | None, int | None]] = []

    def heartbeat(
        phase: str,
        *,
        current_path: Path | None = None,
        source_payload_read_bytes: int | None = None,
        force: bool = False,
    ) -> None:
        del force
        events.append((phase, current_path, source_payload_read_bytes))

    def fake_write_from_bytes(_store: object, payload: bytes) -> tuple[str, int]:
        suffix = f"{len(events):064x}"[-64:]
        return suffix, len(payload)

    monkeypatch.setattr(
        "polylogue.sources.live.batch._jsonl_provider_and_conversation_artifact",
        lambda _path, fallback_provider: (fallback_provider, True),
    )
    monkeypatch.setattr("polylogue.sources.live.batch.BlobStore.write_from_bytes", fake_write_from_bytes)
    monkeypatch.setattr(
        "polylogue.sources.live.batch._process_ingest_batch_sync",
        lambda *args, **kwargs: SimpleNamespace(failed_raw_ids=set(), parse_failures=0, worker_count=1),
    )

    result = processor._ingest_full_paths_sync([first, second], source_name="codex", heartbeat=heartbeat)

    assert result.succeeded == [first, second]
    assert ("full_file_scan", first, 0) in events
    assert ("full_file_scan", second, first.stat().st_size) in events
    assert any(event == ("full_worker_wait", second, first.stat().st_size + second.stat().st_size) for event in events)


def test_fingerprint_file_streams_in_bounded_memory(tmp_path: Path) -> None:
    """``fingerprint_file`` must not load the whole file into memory.

    Regression: the previous implementation read the entire file via
    ``Path.read_bytes()``, producing an RSS peak proportional to file size.
    This test exercises the streaming path on a multi-megabyte synthetic
    file and asserts that the working set stays bounded by ``chunk_size``.
    """
    import hashlib
    import tracemalloc

    from polylogue.sources.live.batch_support import fingerprint_file

    payload = (b"x" * 4095 + b"\n") * 4096  # ~16 MiB, all lines newline-terminated
    target = tmp_path / "huge.jsonl"
    target.write_bytes(payload)
    expected_hash = hashlib.sha256(payload).hexdigest()
    expected_last_nl = len(payload)  # ends in newline

    tracemalloc.start()
    try:
        fp, last_nl = fingerprint_file(target, chunk_size=64 * 1024)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert fp == expected_hash
    assert last_nl == expected_last_nl
    # Peak Python-allocated memory must stay well under the file size. The
    # 1 MiB budget is generous (chunk_size is 64 KiB) but leaves room for
    # hasher state and chunk overhead without admitting a full-file read.
    assert peak < 1 * 1024 * 1024, f"fingerprint_file peak {peak} bytes is not bounded for a {len(payload)}-byte file"


def test_fingerprint_file_tracks_last_newline_across_chunk_boundary(tmp_path: Path) -> None:
    """The streaming fingerprint must locate the last newline even when it
    sits in an earlier chunk than the file tail."""
    from polylogue.sources.live.batch_support import fingerprint_file

    # 4 KiB of newline-terminated lines, followed by 4 KiB without any \n.
    head = (b"line\n") * 1000  # 5_000 bytes, ends with \n
    tail = b"y" * 5000  # no newline anywhere
    payload = head + tail
    target = tmp_path / "no-trailing-newline.jsonl"
    target.write_bytes(payload)

    _fp, last_nl = fingerprint_file(target, chunk_size=1024)
    assert last_nl == len(head), f"last_complete_newline should be at end-of-head ({len(head)}), got {last_nl}"


def test_fingerprint_file_empty_file(tmp_path: Path) -> None:
    import hashlib

    from polylogue.sources.live.batch_support import fingerprint_file

    target = tmp_path / "empty.jsonl"
    target.write_bytes(b"")

    fp, last_nl = fingerprint_file(target)
    assert fp == hashlib.sha256(b"").hexdigest()
    assert last_nl == 0


def test_append_plan_rejects_large_tail_for_streaming_full_ingest(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    path = root / "session.jsonl"
    original = b'{"a":1}\n'
    appended = b'{"b":"' + (b"x" * _MAX_APPEND_PLAN_PAYLOAD_BYTES) + b'"}\n'
    path.write_bytes(original + appended)
    db_path = tmp_path / "archive.sqlite"
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=db_path))),
        (WatchSource(name="claude-code", root=root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="test-parser",
    )
    stat = path.stat()
    processor._cursor.set(
        path,
        len(original),
        byte_offset=len(original),
        last_complete_newline=len(original),
        parser_fingerprint="test-parser",
        content_fingerprint="base",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )

    assert processor._append_plan(path) is None
