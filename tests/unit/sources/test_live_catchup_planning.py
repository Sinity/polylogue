from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.cursor import CursorStore


def test_catch_up_plan_carries_statted_candidates_without_payload_reads(
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
    cursor.set(
        unchanged,
        stat.st_size,
        byte_offset=stat.st_size,
        last_complete_newline=stat.st_size,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="already-known",
        source_name="test",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )

    def fail_fingerprint_file(path: Path) -> tuple[str, int]:
        raise AssertionError(f"unchanged catch-up planning should not fingerprint payloads: {path}")

    monkeypatch.setattr(live_watcher, "fingerprint_file", fail_fingerprint_file)
    candidates = watcher._scan_catch_up_candidates([root])
    plan = watcher._plan_catch_up(candidates)

    assert [candidate.path for candidate in candidates] == [changed, unchanged]
    assert plan.needed == (changed,)
    assert plan.skipped_file_count == 1
    assert plan.needed_bytes == changed.stat().st_size


def test_catch_up_ingests_needed_files_in_bounded_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    files = [root / f"session-{index}.jsonl" for index in range(5)]
    for index, path in enumerate(files):
        path.write_text("x" * (index + 1))
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    watcher = LiveWatcher(cast(Any, polylogue), (WatchSource(name="test", root=root),))
    monkeypatch.setattr(live_watcher, "_CATCH_UP_MAX_BATCH_FILES", 2)
    monkeypatch.setattr(live_watcher, "_CATCH_UP_MAX_BATCH_BYTES", 100)

    calls: list[tuple[list[Path], int | None, int]] = []

    async def fake_ingest_files(
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
    ) -> None:
        calls.append((paths, queued_file_count, skipped_file_count))

    watcher._ingest_files = fake_ingest_files  # type: ignore[method-assign]

    asyncio.run(watcher._catch_up([root]))

    assert [paths for paths, _queued, _skipped in calls] == [files[:2], files[2:4], files[4:]]
    assert calls[0][1:] == (5, 0)
    assert calls[1][1:] == (2, 0)
    assert calls[2][1:] == (1, 0)
