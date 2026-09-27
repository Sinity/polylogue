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
from polylogue.operations.session_contracts import RawSearch
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
    return sorted(directory.glob("*.json")) if directory.is_dir() else []


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


def test_change_between_scan_and_emission_withholds_the_match_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "codex"
    target = _write(root / "a.jsonl", "needle a\n", 1)
    original = SessionLogService.search

    def search_then_append(self: SessionLogService, *args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(self, *args, **kwargs)
        with target.open("a") as handle:
            handle.write("appended\n")
        return result

    monkeypatch.setattr(SessionLogService, "search", search_then_append)
    page = _search(_sources(root))
    assert page.items == [] and page.outcome == "degraded"
    assert any("between scan and emission" in gap for gap in page.coverage.gaps)


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


def test_capacity_evicts_oldest_handles_per_principal_and_globally(
    tmp_path: Path, frozen_clock: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "codex"
    first_file = _write(root / "a.jsonl", "needle a\n", 2)
    _write(root / "b.jsonl", "needle b\n", 1)
    sources = _sources(root)
    tokens = []
    for _ in range(snapshot_store.MAX_PRINCIPAL_SNAPSHOTS + 1):
        tokens.append(_search(sources, scan_bytes=first_file.stat().st_size).continuation)
        frozen_clock.advance(1)
    assert len(_snapshot_files()) == snapshot_store.MAX_PRINCIPAL_SNAPSHOTS
    evicted = _search(sources, continuation=tokens[0])
    assert evicted.outcome == "degraded" and any("evicted" in gap for gap in evicted.coverage.gaps)
    assert [item.reference for item in _search(sources, continuation=tokens[-1]).items] == ["codex:b.jsonl"]

    monkeypatch.setattr(snapshot_store, "MAX_GLOBAL_SNAPSHOTS", 2)
    other = SessionLogService(sources=sources, scope="another-principal")
    key = b"k" * 32
    for _ in range(2):
        frozen_clock.advance(1)
        other.search("codex", "needle", 1, scan_bytes=first_file.stat().st_size, cursor_key=key)
    assert len(_snapshot_files()) == 2


def test_snapshot_survives_restart_and_retains_no_session_content(tmp_path: Path) -> None:
    root = tmp_path / "codex"
    first_file = _write(root / "a.jsonl", '{"secret":"needle alpha"}\n', 2)
    _write(root / "b.jsonl", '{"secret":"needle beta"}\n', 1)
    sources = _sources(root)
    first = _search(sources, scan_bytes=first_file.stat().st_size)

    # raw_operation builds a fresh service per call: nothing is process-resident.
    files = _snapshot_files()
    assert len(files) == 1
    stored = files[0].read_text()
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
