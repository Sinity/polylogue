"""Raw-session search continuations resume a retained selected population.

The public token names a bounded metadata snapshot instead of carrying the
population or a digest of the whole live tree, so an unrelated live append
cannot invalidate an untouched historical scan (P08). Every test drives the
production ``raw_operation`` route unless it targets the store itself.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import zlib
from pathlib import Path
from typing import Any

import pytest

from polylogue.operations.raw_sessions import snapshot_store
from polylogue.operations.raw_sessions.sessions import (
    MAX_CURSOR_BYTES,
    OpaqueSessionCursor,
    SessionLogService,
    SessionSource,
    StaleContinuationError,
)
from polylogue.operations.session_contracts import RawMemorySearch, RawSearch, RawTimeline
from polylogue.operations.session_reads import raw_operation, session_operation_response
from polylogue.paths import state_home


def _write(path: Path, text: str, mtime_s: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, ns=(mtime_s * 1_000_000_000, mtime_s * 1_000_000_000))
    return path


def _sources(root: Path) -> tuple[SessionSource, ...]:
    return (SessionSource("codex", root),)


def _search(sources: tuple[SessionSource, ...], **fields: Any) -> Any:
    return raw_operation(RawSearch(origin="codex-session", query="needle", **fields), sources=sources)


def _snapshot_files() -> list[Path]:
    directory = state_home() / "raw-session-search"
    return sorted(directory.glob("*.snapshot")) if directory.is_dir() else []


def test_default_scale_population_keeps_the_token_small_and_roster_free(tmp_path: Path) -> None:
    """15,697 selected files resume through a token far below the cursor bound.

    Anti-vacuity: embedding the population (or one path per file) in the
    token would exceed MAX_CURSOR_BYTES and raise, and the second page would
    not find the match that lives only in the last-scanned file.
    """
    root = tmp_path / "codex"
    root.mkdir()
    for index in range(15_697):
        (root / f"s{index:05d}.jsonl").write_text("x\n")
    # Oldest file is scanned last and holds the only match.
    _write(root / "s00000.jsonl", "needle\n", 1)
    sources = _sources(root)

    first = _search(sources, scan_bytes=2)
    assert first.continuation is not None
    token = first.continuation
    assert len(token.encode()) < 1_024 < MAX_CURSOR_BYTES
    assert "s00001" not in json.dumps(first.model_dump())

    page = _search(sources, scan_bytes=8_388_608, continuation=token)
    assert [item.reference for item in page.items] == ["codex:s00000.jsonl"]
    assert page.continuation is None and page.outcome == "ok"


def test_reference_scope_cannot_be_resumed_without_its_filter(tmp_path: Path) -> None:
    root = tmp_path / "codex"
    _write(root / "a.jsonl", "x" * 64 + "needle\n", 2)
    _write(root / "b.jsonl", "needle\n", 1)
    sources = _sources(root)
    first = _search(sources, reference="codex:a.jsonl", scan_bytes=8)
    assert first.continuation is not None

    for other in (None, "codex:b.jsonl"):
        with pytest.raises(StaleContinuationError, match="original search scope"):
            _search(sources, reference=other, continuation=first.continuation)
    resumed = _search(sources, reference="codex:a.jsonl", continuation=first.continuation)
    assert [item.reference for item in resumed.items] == ["codex:a.jsonl"]

    envelope = asyncio.run(
        session_operation_response(
            None,
            RawSearch(origin="codex-session", query="needle", continuation=first.continuation),
            raw_sources=sources,
        )
    )
    assert envelope.model_dump()["code"] == "stale_continuation"


def test_pre_snapshot_v1_cursor_is_a_typed_stale_refusal(tmp_path: Path) -> None:
    root = tmp_path / "codex"
    _write(root / "a.jsonl", "needle\n", 1)
    sources = _sources(root)
    service = SessionLogService(sources=sources)
    key = hashlib.sha256(repr(service.sources).encode()).digest()
    legacy = OpaqueSessionCursor("polylogue-raw", key, "session-search").encode(
        {
            "principal": "polylogue-raw",
            "provider": "codex",
            "query_sha256": hashlib.sha256(b"needle").hexdigest(),
            "source_revision": "0" * 64,
        },
        {"file": 0, "offset": 0, "line": 1, "line_start": 0},
    )
    with pytest.raises(StaleContinuationError, match="predates retained search snapshots"):
        _search(sources, continuation=legacy)


def test_completed_live_append_does_not_disturb_a_historical_continuation(tmp_path: Path) -> None:
    """P08: the newest (live) file was fully scanned; appending to it later is irrelevant."""
    root = tmp_path / "codex"
    live = _write(root / "live.jsonl", "needle live\n", 2)
    historical = _write(root / "old.jsonl", "needle old\n", 1)
    sources = _sources(root)

    first = _search(sources, scan_bytes=live.stat().st_size)
    assert [item.reference for item in first.items] == ["codex:live.jsonl"]
    assert first.continuation is not None

    with live.open("a") as handle:
        handle.write("needle appended\n")
    second = _search(sources, continuation=first.continuation)
    assert [item.reference for item in second.items] == ["codex:old.jsonl"]
    assert second.items[0].text and "old" in second.items[0].text
    assert second.outcome == "ok" and second.coverage.complete and second.continuation is None
    assert historical.exists()


def test_unscanned_live_append_is_a_degraded_gap_not_a_restart(tmp_path: Path) -> None:
    root = tmp_path / "codex"
    first_file = _write(root / "a.jsonl", "needle a\n", 3)
    live = _write(root / "live.jsonl", "needle live\n", 2)
    _write(root / "z.jsonl", "needle z\n", 1)
    sources = _sources(root)

    first = _search(sources, scan_bytes=first_file.stat().st_size)
    assert [item.reference for item in first.items] == ["codex:a.jsonl"]
    with live.open("a") as handle:
        handle.write("more\n")

    second = _search(sources, continuation=first.continuation)
    assert [item.reference for item in second.items] == ["codex:z.jsonl"]
    assert second.outcome == "degraded" and not second.coverage.complete
    assert any("codex:live.jsonl" in gap and "changed after selection" in gap for gap in second.coverage.gaps)


def test_files_created_after_selection_are_excluded(tmp_path: Path) -> None:
    root = tmp_path / "codex"
    first_file = _write(root / "a.jsonl", "needle a\n", 3)
    _write(root / "b.jsonl", "needle b\n", 1)
    sources = _sources(root)
    first = _search(sources, scan_bytes=first_file.stat().st_size)
    _write(root / "new.jsonl", "needle new\n", 2)

    second = _search(sources, continuation=first.continuation)
    assert [item.reference for item in second.items] == ["codex:b.jsonl"]
    assert second.outcome == "ok"


def test_selected_file_that_vanishes_is_a_gap_and_the_skip_is_remembered(tmp_path: Path) -> None:
    root = tmp_path / "codex"
    first_file = _write(root / "a.jsonl", "needle a\n", 3)
    doomed = _write(root / "b.jsonl", "needle b\n", 2)
    last = _write(root / "c.jsonl", "needle c\n", 1)
    sources = _sources(root)
    first = _search(sources, scan_bytes=first_file.stat().st_size)
    doomed.unlink()

    second = _search(sources, scan_bytes=1, continuation=first.continuation)
    assert second.items == [] and second.outcome == "degraded"
    assert any("codex:b.jsonl" in gap and "disappeared" in gap for gap in second.coverage.gaps)
    third = _search(sources, scan_bytes=last.stat().st_size, continuation=second.continuation)
    assert [item.reference for item in third.items] == ["codex:c.jsonl"]
    # Anti-vacuity: dropping the carried skip count reports this final page complete.
    assert third.outcome == "degraded"
    assert any("1 selected files were skipped on earlier pages" in gap for gap in third.coverage.gaps)


def test_enumeration_to_open_race_is_a_gap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "codex"
    _write(root / "a.jsonl", "needle a\n", 2)
    _write(root / "b.jsonl", "needle b\n", 1)
    original = SessionLogService._files

    def enumerate_then_delete(source: SessionSource) -> Any:
        files = original(source)
        (root / "a.jsonl").unlink()
        return files

    monkeypatch.setattr(SessionLogService, "_files", staticmethod(enumerate_then_delete))
    page = _search(_sources(root))
    assert [item.reference for item in page.items] == ["codex:b.jsonl"]
    assert page.outcome == "degraded"
    assert any("codex:a.jsonl" in gap and "disappeared" in gap for gap in page.coverage.gaps)


def test_write_during_read_discards_the_block_and_reports_a_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Race injection between the pre-read and post-read descriptor checks."""
    root = tmp_path / "codex"
    racing = _write(root / "a.jsonl", "needle a\n", 2)
    _write(root / "b.jsonl", "needle b\n", 1)
    real_fstat = os.fstat
    calls = {"a": 0}
    racing_inode = racing.stat().st_ino

    def fstat(fd: int) -> os.stat_result:
        info = real_fstat(fd)
        if info.st_ino == racing_inode:
            calls["a"] += 1
            if calls["a"] == 2:
                with racing.open("a") as handle:
                    handle.write("late write\n")
                return real_fstat(fd)
        return info

    # The scanner reads descriptor identity through os.fstat before and after each block.
    monkeypatch.setattr(os, "fstat", fstat)
    page = _search(_sources(root))
    assert [item.reference for item in page.items] == ["codex:b.jsonl"]
    assert any("changed while it was being searched" in gap for gap in page.coverage.gaps)


def test_snapshot_expiry_is_a_degraded_outcome(tmp_path: Path, frozen_clock: Any) -> None:
    root = tmp_path / "codex"
    first_file = _write(root / "a.jsonl", "needle a\n", 2)
    _write(root / "b.jsonl", "needle b\n", 1)
    sources = _sources(root)
    first = _search(sources, scan_bytes=first_file.stat().st_size)
    frozen_clock.advance(snapshot_store.SNAPSHOT_TTL_MS / 1000 + 1)

    expired = _search(sources, continuation=first.continuation)
    assert expired.items == [] and expired.outcome == "degraded" and expired.continuation is None
    assert any("expired or was evicted" in gap for gap in expired.coverage.gaps)
    assert _snapshot_files() == []


def test_production_callers_share_one_global_lru_capacity(
    tmp_path: Path, frozen_clock: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every production caller reaches the store through ``raw_operation`` with one scope.

    Anti-vacuity: a per-principal cap of 4 (the former design) evicts the
    first of these six production-route continuations; a creation-time TTL
    or FIFO order evicts the handle that was just used instead of the idle one.
    """
    root = tmp_path / "codex"
    first_file = _write(root / "a.jsonl", "needle a\n", 2)
    _write(root / "b.jsonl", "needle b\n", 1)
    sources = _sources(root)
    tokens = []
    for _ in range(6):
        tokens.append(_search(sources, scan_bytes=first_file.stat().st_size).continuation)
        frozen_clock.advance(1)
    assert len(_snapshot_files()) == 6
    for token in tokens:
        assert [item.reference for item in _search(sources, continuation=token).items] == ["codex:b.jsonl"]

    monkeypatch.setattr(snapshot_store, "MAX_GLOBAL_SNAPSHOTS", 2)
    frozen_clock.advance(1)
    # Using the oldest handle makes it the most recently used survivor.
    assert _search(sources, continuation=tokens[0]).outcome == "ok"
    frozen_clock.advance(1)
    newest = _search(sources, scan_bytes=first_file.stat().st_size).continuation
    assert len(_snapshot_files()) == 2
    assert _search(sources, continuation=tokens[0]).outcome == "ok"
    assert _search(sources, continuation=newest).outcome == "ok"
    evicted = _search(sources, continuation=tokens[1])
    assert evicted.outcome == "degraded" and any("evicted" in gap for gap in evicted.coverage.gaps)


def test_ttl_slides_from_last_use_and_orphaned_temporaries_are_swept(tmp_path: Path, frozen_clock: Any) -> None:
    root = tmp_path / "codex"
    first_file = _write(root / "a.jsonl", "needle a\n", 2)
    _write(root / "b.jsonl", "needle b\n", 1)
    sources = _sources(root)
    token = _search(sources, scan_bytes=first_file.stat().st_size).continuation
    directory = state_home() / "raw-session-search"
    orphan = directory / ".crashed.json.abc.tmp"
    orphan.write_bytes(b"{}")
    os.utime(orphan, (frozen_clock.time(), frozen_clock.time()))

    ttl_s = snapshot_store.SNAPSHOT_TTL_MS / 1000
    for _ in range(3):
        frozen_clock.advance(ttl_s * 0.75)
        assert _search(sources, continuation=token).outcome == "ok"
    # The orphan is older than a TTL; the next write sweeps it.
    _search(sources, scan_bytes=first_file.stat().st_size)
    assert not orphan.exists()
    frozen_clock.advance(ttl_s + 1)
    assert _search(sources, continuation=token).outcome == "degraded"


def test_full_page_resumes_at_the_unreturned_match_not_an_earlier_scanned_file(tmp_path: Path) -> None:
    """A completed file that changes after the page is not a gap on the next page.

    Anti-vacuity: rewinding the continuation to just after the last accepted
    match (inside live.jsonl) makes the next page reopen the changed file and
    report it as a gap.
    """
    root = tmp_path / "codex"
    live = _write(root / "live.jsonl", "needle live\nrest\n", 2)
    _write(root / "old.jsonl", "needle\n", 1)
    sources = _sources(root)
    first = _search(sources, limit=1)
    assert [item.reference for item in first.items] == ["codex:live.jsonl"]
    assert first.continuation is not None
    with live.open("a") as handle:
        handle.write("appended\n")

    second = _search(sources, limit=1, continuation=first.continuation)
    assert [item.reference for item in second.items] == ["codex:old.jsonl"]
    assert second.outcome == "ok" and second.coverage.gaps == [] and second.continuation is None


def test_full_page_inside_one_file_resumes_without_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "codex"
    _write(root / "a.jsonl", "needle 1\nneedle 2\nneedle 3\n", 1)
    sources = _sources(root)
    seen: list[tuple[int | None, str | None]] = []
    page = _search(sources, limit=1)
    seen.extend((item.line, item.text) for item in page.items)
    while page.continuation is not None:
        page = _search(sources, limit=1, continuation=page.continuation)
        seen.extend((item.line, item.text) for item in page.items)
    assert [line for line, _ in seen] == [1, 2, 3]


def test_snapshot_survives_restart_and_retains_no_session_content(tmp_path: Path) -> None:
    root = tmp_path / "codex"
    first_file = _write(root / "a.jsonl", '{"secret":"needle alpha"}\n', 2)
    _write(root / "b.jsonl", '{"secret":"needle beta"}\n', 1)
    sources = _sources(root)
    first = _search(sources, scan_bytes=first_file.stat().st_size)

    # raw_operation builds a fresh service per call: nothing is process-resident.
    files = _snapshot_files()
    assert len(files) == 1
    stored = zlib.decompress(files[0].read_bytes()).decode()
    assert "secret" not in stored and "alpha" not in stored and "needle" not in stored
    assert files[0].stat().st_mode & 0o077 == 0
    resumed = _search(sources, continuation=first.continuation)
    assert [item.reference for item in resumed.items] == ["codex:b.jsonl"]


def test_a_search_that_completes_in_one_page_retains_nothing(tmp_path: Path) -> None:
    root = tmp_path / "codex"
    _write(root / "a.jsonl", "needle a\n", 1)
    page = _search(_sources(root))
    assert page.continuation is None and page.outcome == "ok"
    assert _snapshot_files() == []


def test_selected_empty_file_that_changes_before_its_turn_is_a_gap(tmp_path: Path) -> None:
    """Anti-vacuity: treating offset 0 == size 0 as 'already scanned' skips the open and reports complete."""
    root = tmp_path / "codex"
    first_file = _write(root / "a.jsonl", "needle a\n", 2)
    empty = _write(root / "empty.jsonl", "", 1)
    sources = _sources(root)
    first = _search(sources, scan_bytes=first_file.stat().st_size)
    assert first.continuation is not None
    empty.write_text("needle late\n")

    second = _search(sources, continuation=first.continuation)
    assert second.items == [] and second.outcome == "degraded"
    assert any("codex:empty.jsonl" in gap and "changed after selection" in gap for gap in second.coverage.gaps)


def test_memory_search_preserves_the_stale_continuation_code(tmp_path: Path) -> None:
    root = tmp_path / "codex"
    _write(root / "a.jsonl", "needle\n", 1)
    sources = _sources(root)
    key = hashlib.sha256(repr(SessionLogService(sources=sources).sources).encode()).digest()
    legacy = OpaqueSessionCursor("polylogue-raw", key, "session-search").encode(
        {"principal": "polylogue-raw", "provider": "codex", "query_sha256": "0" * 64, "source_revision": "0" * 64},
        {"file": 0, "offset": 0, "line": 1, "line_start": 0},
    )
    request = RawMemorySearch(query="needle", origins=["codex-session"], source_cursors={"codex-session": legacy})
    envelope = asyncio.run(session_operation_response(None, request, raw_sources=sources))
    assert envelope.model_dump()["code"] == "stale_continuation"


def test_snapshots_created_in_one_millisecond_never_evict_the_new_handle(
    tmp_path: Path, frozen_clock: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: pruning after the write lets a random-handle tie evict the snapshot just created."""
    monkeypatch.setattr(snapshot_store, "MAX_GLOBAL_SNAPSHOTS", 2)
    root = tmp_path / "codex"
    first_file = _write(root / "a.jsonl", "needle a\n", 2)
    _write(root / "b.jsonl", "needle b\n", 1)
    sources = _sources(root)
    for _ in range(12):
        token = _search(sources, scan_bytes=first_file.stat().st_size).continuation
        assert _search(sources, continuation=token).outcome == "ok"
        assert len(_snapshot_files()) <= 2


def test_timeline_fanout_preserves_the_stale_continuation_code(tmp_path: Path) -> None:
    root = tmp_path / "codex"
    for index in range(3):
        _write(root / f"s{index}.jsonl", f"needle {index}\n", index + 1)
    sources = _sources(root)
    first = raw_operation(RawTimeline(origins=["codex-session"], limit=1), sources=sources)
    assert first.continuation is not None
    with (root / "s0.jsonl").open("a") as handle:
        handle.write("changed\n")
    request = RawTimeline(origins=["codex-session"], limit=1, continuation=first.continuation)
    envelope = asyncio.run(session_operation_response(None, request, raw_sources=sources))
    assert envelope.model_dump()["code"] == "stale_continuation"


def test_short_resumed_budget_keeps_the_returned_match_high_water_mark(tmp_path: Path) -> None:
    """Anti-vacuity: clearing ``after`` at every block end re-emits match 1 via the replay tail."""
    root = tmp_path / "codex"
    _write(root / "a.jsonl", "needle 1\nneedle 2\nneedle 3\n", 1)
    sources = _sources(root)
    page = _search(sources, limit=1)
    lines = [item.line for item in page.items]
    for _ in range(100):
        if page.continuation is None:
            break
        page = _search(sources, limit=1, scan_bytes=4, continuation=page.continuation)
        lines.extend(item.line for item in page.items)
    assert lines == [1, 2, 3]


def test_resumes_racing_creations_at_capacity_never_fail_or_overfill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lookup, read and touch share the creation lock.

    Anti-vacuity: reading outside the lock lets a creator prune the handle
    between read and touch, and the lost rename then escapes as an error or
    returns a handle that is already gone.
    """
    import threading

    monkeypatch.setattr(snapshot_store, "MAX_GLOBAL_SNAPSHOTS", 2)
    store = snapshot_store.SnapshotStore(tmp_path / "store")
    binding = snapshot_store.SnapshotBinding("p", "codex", "0" * 64, None, tmp_path)
    resumed = store.create(binding, ()).handle
    errors: list[BaseException] = []
    loaded: list[str] = []
    barrier = threading.Barrier(12)

    def resume() -> None:
        barrier.wait()
        try:
            loaded.append(store.load(resumed, binding).handle)
        except snapshot_store.SnapshotUnavailableError:
            pass
        except BaseException as exc:  # pragma: no cover - the failure being guarded
            errors.append(exc)

    def create() -> None:
        barrier.wait()
        store.create(binding, ())

    threads = [threading.Thread(target=resume if index % 2 else create) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(list((tmp_path / "store").glob("*.snapshot"))) == 2
    if loaded:
        # A resume that succeeded touched the handle last or was followed by
        # creations that legitimately aged it out; it never returns a ghost.
        assert all(handle == resumed for handle in loaded)


def test_raced_reads_are_charged_to_the_scan_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: an uncharged raced block lets scan_bytes=1 read every selected file."""
    root = tmp_path / "codex"
    paths = [_write(root / f"s{index}.jsonl", f"needle {index}\n", index + 1) for index in range(4)]
    inodes = {path.stat().st_ino: path for path in paths}
    real_fstat = os.fstat
    seen: dict[int, int] = {}

    def fstat(fd: int) -> os.stat_result:
        info = real_fstat(fd)
        path = inodes.get(info.st_ino)
        if path is not None:
            seen[info.st_ino] = seen.get(info.st_ino, 0) + 1
            if seen[info.st_ino] == 2:
                with path.open("a") as handle:
                    handle.write("late\n")
                return real_fstat(fd)
        return info

    monkeypatch.setattr(os, "fstat", fstat)
    page = _search(_sources(root), scan_bytes=1)
    assert page.coverage.scanned_bytes == 1
    assert len(page.coverage.gaps) == 1 and page.continuation is not None


def test_long_reference_filter_fits_the_token_through_its_digest(tmp_path: Path) -> None:
    """Anti-vacuity: embedding a ~3.6 KB backslash path in the scope overflows MAX_CURSOR_BYTES after escaping."""
    root = tmp_path / "codex"
    directory = root
    for _ in range(14):
        directory = directory / ("\\" * 250)
    target = _write(directory / "s.jsonl", "x" * 64 + "needle\n", 1)
    reference = "codex:" + target.relative_to(root).as_posix()
    assert len(reference) > 3_500
    sources = _sources(root)
    first = _search(sources, reference=reference, scan_bytes=1)
    assert first.continuation is not None and len(first.continuation.encode()) < 1_024
    resumed = _search(sources, reference=reference, continuation=first.continuation)
    assert [item.reference for item in resumed.items] == [reference]
    with pytest.raises(StaleContinuationError):
        _search(sources, continuation=first.continuation)


def test_creation_bursts_hold_the_bound_and_keep_the_newest_handle(
    tmp_path: Path, frozen_clock: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: exempting young handles from pruning lets a same-millisecond burst exceed the bound."""
    monkeypatch.setattr(snapshot_store, "MAX_GLOBAL_SNAPSHOTS", 2)
    root = tmp_path / "codex"
    first_file = _write(root / "a.jsonl", "needle a\n", 2)
    _write(root / "b.jsonl", "needle b\n", 1)
    sources = _sources(root)
    for _ in range(5):
        token = _search(sources, scan_bytes=first_file.stat().st_size).continuation
        assert len(_snapshot_files()) <= 2
        assert _search(sources, continuation=token).outcome == "ok"


def test_threaded_creators_serialize_and_respect_the_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: without the creation lock, creators pruning from one stale listing leave more than the cap."""
    import threading

    monkeypatch.setattr(snapshot_store, "MAX_GLOBAL_SNAPSHOTS", 3)
    store = snapshot_store.SnapshotStore(tmp_path / "store")
    binding = snapshot_store.SnapshotBinding("p", "codex", "0" * 64, None, tmp_path)
    handles: list[str] = []
    barrier = threading.Barrier(8)

    def create() -> None:
        barrier.wait()
        handles.append(store.create(binding, ()).handle)

    threads = [threading.Thread(target=create) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(handles) == 8
    assert len(list((tmp_path / "store").glob("*.snapshot"))) == 3


def test_selected_path_replaced_by_a_fifo_is_a_gap_not_a_hang(tmp_path: Path) -> None:
    """Anti-vacuity: a blocking open waits forever for a FIFO writer."""
    root = tmp_path / "codex"
    first_file = _write(root / "a.jsonl", "needle a\n", 2)
    swapped = _write(root / "b.jsonl", "needle b\n", 1)
    sources = _sources(root)
    first = _search(sources, scan_bytes=first_file.stat().st_size)
    swapped.unlink()
    os.mkfifo(swapped)
    assert stat.S_ISFIFO(swapped.stat().st_mode)

    second = _search(sources, continuation=first.continuation)
    assert second.items == [] and second.outcome == "degraded"
    assert any("codex:b.jsonl" in gap for gap in second.coverage.gaps)


def test_deep_shared_prefix_roster_is_retained_compactly(tmp_path: Path) -> None:
    root = tmp_path / "codex"
    directory = root
    for _ in range(12):
        directory = directory / ("p" * 250)
    first_file = _write(directory / "s0000.jsonl", "needle\n", 3_000)
    for index in range(1, 2_000):
        _write(directory / f"s{index:04d}.jsonl", "x\n", 1)
    token = _search(_sources(root), scan_bytes=first_file.stat().st_size).continuation
    assert token is not None
    (snapshot,) = _snapshot_files()
    # ~3 KB of shared prefix on 2,000 rows is ~6 MB uncompressed.
    assert snapshot.stat().st_size < 200_000


def test_memory_fanout_reports_a_finished_providers_skips_on_its_terminal_page(tmp_path: Path) -> None:
    """Anti-vacuity: a ``None`` slot for the finished provider forgets its skipped file."""
    claude = tmp_path / "claude"
    locked = _write(claude / "a.jsonl", "needle locked\n", 2)
    _write(claude / "b.jsonl", "needle claude\n", 1)
    codex = tmp_path / "codex"
    _write(codex / "c.jsonl", "needle codex\n", 1)
    sources = (SessionSource("claude-code", claude), SessionSource("codex", codex))
    locked.chmod(0)
    try:
        first = raw_operation(RawMemorySearch(query="needle", limit=1), sources=sources)
        assert [item.reference for item in first.items] == ["claude-code:b.jsonl"]
        assert first.source_cursors is not None
        final = raw_operation(
            RawMemorySearch(query="needle", limit=1, source_cursors=first.source_cursors), sources=sources
        )
    finally:
        locked.chmod(0o600)
    assert [item.reference for item in final.items] == ["codex:c.jsonl"]
    assert final.outcome == "degraded"
    assert not any((final.source_cursors or {}).values())
    assert any("1 selected files were skipped on earlier pages" in gap for gap in final.coverage.gaps)


def test_timeline_continuation_reports_earlier_skips_on_its_terminal_page(tmp_path: Path) -> None:
    root = tmp_path / "codex"
    locked = _write(root / "s3.jsonl", "needle 3\n", 3)
    _write(root / "s2.jsonl", "needle 2\n", 2)
    _write(root / "s1.jsonl", "needle 1\n", 1)
    sources = _sources(root)
    locked.chmod(0)
    try:
        page = raw_operation(RawTimeline(origins=["codex-session"], query="needle", limit=1), sources=sources)
        pages = [page]
        while page.continuation is not None:
            page = raw_operation(
                RawTimeline(origins=["codex-session"], query="needle", limit=1, continuation=page.continuation),
                sources=sources,
            )
            pages.append(page)
    finally:
        locked.chmod(0o600)
    assert [item.reference for p in pages for item in p.items] == ["codex:s2.jsonl", "codex:s1.jsonl"]
    assert pages[-1].outcome == "degraded"
    assert any("1 selected files were skipped on earlier pages" in gap for gap in pages[-1].coverage.gaps)


def test_systemic_open_failure_pauses_the_scan_without_skipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: treating EMFILE as file-specific marks the file skipped and can end the continuation."""
    import errno

    root = tmp_path / "codex"
    _write(root / "a.jsonl", "needle a\n", 2)
    _write(root / "b.jsonl", "needle b\n", 1)
    sources = _sources(root)
    real_open = os.open
    failures = {"left": 1}

    def open_with_descriptor_exhaustion(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if str(path).endswith(".jsonl") and failures["left"]:
            failures["left"] -= 1
            raise OSError(errno.EMFILE, "Too many open files")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_with_descriptor_exhaustion)
    paused = _search(sources)
    assert paused.items == [] and paused.continuation is not None
    assert any("transient system error" in gap and "EMFILE" in gap for gap in paused.coverage.gaps)
    resumed = _search(sources, continuation=paused.continuation)
    assert [item.reference for item in resumed.items] == ["codex:a.jsonl", "codex:b.jsonl"]
    assert resumed.outcome == "ok" and resumed.coverage.gaps == []


def test_memory_resume_of_an_expired_snapshot_is_a_degraded_page(tmp_path: Path, frozen_clock: Any) -> None:
    """Anti-vacuity: the expired-snapshot result lacked the skip counters and memory raised KeyError."""
    root = tmp_path / "codex"
    first_file = _write(root / "a.jsonl", "needle a\n", 2)
    _write(root / "b.jsonl", "needle b\n", 1)
    sources = _sources(root)
    request = RawMemorySearch(query="needle", origins=["codex-session"], scan_bytes=first_file.stat().st_size)
    first = raw_operation(request, sources=sources)
    assert first.source_cursors and first.source_cursors.get("codex-session")
    frozen_clock.advance(snapshot_store.SNAPSHOT_TTL_MS / 1000 + 1)
    expired = raw_operation(request.model_copy(update={"source_cursors": first.source_cursors}), sources=sources)
    assert expired.outcome == "degraded" and expired.items == []
    assert any("expired or was evicted" in gap for gap in expired.coverage.gaps)


def test_a_touch_in_the_creation_millisecond_still_makes_the_handle_most_recent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: touching with the wall-clock stamp sorts the just-used handle before a newer synthetic stamp."""
    monkeypatch.setattr(snapshot_store, "MAX_GLOBAL_SNAPSHOTS", 2)
    monkeypatch.setattr(snapshot_store, "_now_ms", lambda: 1_000_000)
    store = snapshot_store.SnapshotStore(tmp_path / "store")
    binding = snapshot_store.SnapshotBinding("p", "codex", "0" * 64, None, tmp_path)
    first = store.create(binding, ()).handle
    second = store.create(binding, ()).handle
    store.load(first, binding)
    store.create(binding, ())
    store.load(first, binding)
    with pytest.raises(snapshot_store.SnapshotUnavailableError):
        store.load(second, binding)


def test_memory_fanout_reuses_one_completion_snapshot_across_pages(tmp_path: Path) -> None:
    """Anti-vacuity: minting a new completion token every page grows the store by one snapshot per page."""
    claude = tmp_path / "claude"
    locked = _write(claude / "a.jsonl", "needle locked\n", 2)
    _write(claude / "b.jsonl", "needle claude\n", 1)
    codex = tmp_path / "codex"
    for index in range(4):
        _write(codex / f"c{index}.jsonl", f"needle codex {index}\n", index + 1)
    sources = (SessionSource("claude-code", claude), SessionSource("codex", codex))
    locked.chmod(0)
    try:
        page = raw_operation(RawMemorySearch(query="needle", limit=1), sources=sources)
        counts = []
        references = [item.reference for item in page.items]
        while page.source_cursors and any(page.source_cursors.values()):
            page = raw_operation(
                RawMemorySearch(query="needle", limit=1, source_cursors=page.source_cursors), sources=sources
            )
            references.extend(item.reference for item in page.items)
            counts.append(len(_snapshot_files()))
    finally:
        locked.chmod(0o600)
    assert len(references) == 5 and len(set(references)) == 5
    assert max(counts) <= 2
    assert any("1 selected files were skipped on earlier pages" in gap for gap in page.coverage.gaps)


def test_memory_fanout_refuses_to_continue_when_it_cannot_retain_owed_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: recording the failed completion token as a finished source lets the terminal page forget the skip."""
    claude = tmp_path / "claude"
    locked = _write(claude / "a.jsonl", "needle locked\n", 2)
    _write(claude / "b.jsonl", "needle claude\n", 1)
    codex = tmp_path / "codex"
    _write(codex / "c.jsonl", "needle codex\n", 1)
    sources = (SessionSource("claude-code", claude), SessionSource("codex", codex))
    monkeypatch.setattr(SessionLogService, "completed_skips_token", lambda *_args, **_kwargs: None)
    locked.chmod(0)
    try:
        page = raw_operation(RawMemorySearch(query="needle", limit=1), sources=sources)
    finally:
        locked.chmod(0o600)
    assert page.outcome == "degraded"
    assert not any((page.source_cursors or {}).values())
    assert any("continuation unavailable" in gap and "restart the search" in gap for gap in page.coverage.gaps)


def test_transient_snapshot_read_failure_is_retryable_with_the_same_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: treating EIO like a missing snapshot returns a terminal 'expired' page and loses the continuation."""
    import errno

    root = tmp_path / "codex"
    first_file = _write(root / "a.jsonl", "needle a\n", 2)
    _write(root / "b.jsonl", "needle b\n", 1)
    sources = _sources(root)
    token = _search(sources, scan_bytes=first_file.stat().st_size).continuation
    real_read = Path.read_bytes

    def failing_read(self: Path) -> bytes:
        if self.suffix == ".snapshot":
            raise OSError(errno.EIO, "Input/output error")
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", failing_read)
    request = RawSearch(origin="codex-session", query="needle", continuation=token)
    envelope = asyncio.run(session_operation_response(None, request, raw_sources=sources))
    assert envelope.model_dump()["code"] == "retryable"
    monkeypatch.setattr(Path, "read_bytes", real_read)
    assert [item.reference for item in _search(sources, continuation=token).items] == ["codex:b.jsonl"]


def test_timeline_head_held_in_the_continuation_reports_its_enumerated_observation(tmp_path: Path) -> None:
    root = tmp_path / "codex"
    _write(root / "new.jsonl", "a\n", 2)
    held = _write(root / "old.jsonl", "b\n", 1)
    before = held.stat()
    sources = _sources(root)
    first = raw_operation(RawTimeline(origins=["codex-session"], limit=1), sources=sources)
    assert [item.reference for item in first.items] == ["codex:new.jsonl"]
    with held.open("a") as handle:
        handle.write("appended\n")
    second = raw_operation(
        RawTimeline(origins=["codex-session"], limit=1, continuation=first.continuation), sources=sources
    )
    assert [(item.reference, item.bytes, item.mtime_ns) for item in second.items] == [
        ("codex:old.jsonl", before.st_size, before.st_mtime_ns)
    ]


def test_memory_fanout_ends_when_an_unfinished_provider_loses_its_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: keeping the later provider's cursor lets a resume read the lost provider's ``None`` as finished."""
    import errno

    claude = tmp_path / "claude"
    _write(claude / "a.jsonl", "x" * 64 + "needle claude\n", 1)
    codex = tmp_path / "codex"
    _write(codex / "c.jsonl", "x" * 64 + "needle codex\n", 1)
    sources = (SessionSource("claude-code", claude), SessionSource("codex", codex))
    real_create = snapshot_store.SnapshotStore.create

    def create_fails_for_claude(
        self: snapshot_store.SnapshotStore, binding: snapshot_store.SnapshotBinding, files: Any
    ) -> snapshot_store.SearchSnapshot:
        if binding.provider == "claude-code":
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_create(self, binding, files)

    monkeypatch.setattr(snapshot_store.SnapshotStore, "create", create_fails_for_claude)
    page = raw_operation(RawMemorySearch(query="needle", limit=5, scan_bytes=8), sources=sources)
    assert page.outcome == "degraded" and not page.coverage.complete
    assert not any((page.source_cursors or {}).values())
    assert any("continuation unavailable" in gap for gap in page.coverage.gaps)


def test_transient_snapshot_directory_probe_failure_is_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: ``Path.is_dir()`` folds EIO into False and reports the snapshot expired."""
    import errno

    root = tmp_path / "codex"
    first_file = _write(root / "a.jsonl", "needle a\n", 2)
    _write(root / "b.jsonl", "needle b\n", 1)
    sources = _sources(root)
    token = _search(sources, scan_bytes=first_file.stat().st_size).continuation
    directory = state_home() / "raw-session-search"
    real_stat = os.stat

    def failing_stat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if Path(path) == directory:
            raise OSError(errno.EIO, "Input/output error")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", failing_stat)
    request = RawSearch(origin="codex-session", query="needle", continuation=token)
    envelope = asyncio.run(session_operation_response(None, request, raw_sources=sources))
    assert envelope.model_dump()["code"] == "retryable"
