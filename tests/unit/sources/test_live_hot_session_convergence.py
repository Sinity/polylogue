from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue.sources.live.cursor import CursorStore
from tests.unit.sources.test_live_watcher import _ingest_one, _make_watcher


def test_cursor_rejects_stale_backward_write_for_same_parser(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    p = tmp_path / "session.jsonl"
    assert store.set(p, 250, byte_offset=250, parser_fingerprint="parser") is True

    assert store.set(p, 100, byte_offset=100, parser_fingerprint="parser") is False

    record = store.get_record(p)
    assert record is not None
    assert record.byte_size == 250
    assert record.byte_offset == 250


def test_cursor_allows_explicit_backward_write_for_truncation(tmp_path: Path) -> None:
    store = CursorStore(tmp_path / "live.sqlite")
    p = tmp_path / "session.jsonl"
    store.set(p, 250, byte_offset=250, parser_fingerprint="parser")

    assert store.set(p, 100, byte_offset=100, parser_fingerprint="parser", allow_backward=True) is True

    record = store.get_record(p)
    assert record is not None
    assert record.byte_size == 100
    assert record.byte_offset == 100


def test_same_size_prefix_rewrite_outside_tail_returns_to_full_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    original = '{"a":"alpha' + ("p" * (70 * 1024)) + '"}\n'
    rewritten = original.replace("alpha", "bravo", 1)
    assert len(original) == len(rewritten)
    assert original.encode()[-64 * 1024 :] == rewritten.encode()[-64 * 1024 :]
    f.write_text(original)
    watcher, parse_sources = _make_watcher(tmp_path, root)

    asyncio.run(_ingest_one(watcher, f))
    assert parse_sources.await_count == 1

    original_stat = f.stat()
    f.write_text(rewritten)
    rewritten_stat = f.stat()
    os.utime(f, ns=(rewritten_stat.st_atime_ns, original_stat.st_mtime_ns))
    restored_stat = f.stat()
    assert restored_stat.st_size == len(original)
    assert restored_stat.st_mtime_ns == original_stat.st_mtime_ns
    assert restored_stat.st_ctime_ns != original_stat.st_ctime_ns

    def fail_fingerprint(path: Path) -> tuple[str, int]:
        raise AssertionError(f"same-size rewrite should not full-fingerprint before ingest: {path}")

    monkeypatch.setattr(live_watcher, "fingerprint_file", fail_fingerprint)
    asyncio.run(_ingest_one(watcher, f))

    assert parse_sources.await_count == 2


def test_legacy_same_size_cursor_without_authority_requires_reauthorization(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"a":1}\n')
    watcher, parse_sources = _make_watcher(tmp_path, root)
    stat = f.stat()
    watcher._cursor.set(
        f,
        stat.st_size,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="legacy-full-hash",
    )

    f.write_text('{"b":2}\n')

    asyncio.run(_ingest_one(watcher, f))

    assert parse_sources.await_count == 1


def test_debounce_drains_live_events_serially_while_ingest_is_active(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    first = root / "first.jsonl"
    second = root / "second.jsonl"
    first.write_text('{"a":1}\n')
    second.write_text('{"b":2}\n')
    watcher, _parse_sources = _make_watcher(tmp_path, root, debounce_s=0.01)
    active = 0
    max_active = 0
    batches: list[list[Path]] = []
    entered_first_batch = asyncio.Event()
    release_first_batch = asyncio.Event()

    async def fake_ingest(
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
    ) -> None:
        del queued_file_count, skipped_file_count
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        batches.append(list(paths))
        if len(batches) == 1:
            entered_first_batch.set()
            await release_first_batch.wait()
        active -= 1

    async def _drive() -> None:
        watcher._ingest_files = fake_ingest  # type: ignore[assignment,method-assign]
        watcher._enqueue(first)
        await asyncio.wait_for(entered_first_batch.wait(), timeout=2.0)
        watcher._enqueue(second)
        await asyncio.sleep(0.05)
        release_first_batch.set()
        while watcher._pending_scheduled or watcher._pending_paths:
            await asyncio.sleep(0.02)

    asyncio.run(_drive())

    assert max_active == 1
    assert batches == [[first], [second]]
