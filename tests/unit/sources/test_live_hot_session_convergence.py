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
