"""The first source walk stays visible before it yields an intake item."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.daemon.discovery_progress import (
    active_discovery_payload,
    advance_discovery,
    begin_discovery,
    end_discovery,
    reset_discovery_progress,
)
from polylogue.daemon.status_snapshot import (
    get_status_snapshot_payload,
    refresh_status_snapshot,
    reset_status_snapshot,
)
from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter
from polylogue.sources.live import WatchSource
from polylogue.sources.live import discovery as discovery_module


@pytest.mark.asyncio
async def test_status_shows_first_discovery_while_sibling_sort_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    accepted = source_root / "session.json"
    accepted.write_text("{}")
    source = WatchSource(name="synthetic", root=source_root, suffixes=(".json",))
    watcher = SimpleNamespace(intake_revision=lambda _source: 0)
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    entered = threading.Event()
    release = threading.Event()
    ordered_children = discovery_module._ordered_children

    def held_children(*args: object, **kwargs: object) -> object:
        entered.set()
        assert release.wait(3), "first directory listing was not released"
        return cast(Any, ordered_children)(*args, **kwargs)

    monkeypatch.setattr(discovery_module, "_ordered_children", held_children)
    monkeypatch.setattr("polylogue.daemon.status_snapshot._status_frame", lambda: None)
    refresh_status_snapshot(payload={"catchup": {"mode": "idle"}})
    task = asyncio.create_task(adapter.discover(limit=1))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        status = get_status_snapshot_payload()
        catchup = cast(dict[str, Any], status["catchup"])
        assert catchup["mode"] == "discovering"
        assert catchup["current_phase"] == "discovering"
        assert catchup["current_source"] == "synthetic"
        assert catchup["discovery_pending"] is True
        assert catchup["discovery_inspected_count"] == 0
        assert catchup["discovery_accepted_count"] == 0
        assert catchup["discovery_rejected_count"] == 0
        assert catchup["discovery_age_s"] >= 0
        assert catchup["discovery_last_advanced_age_s"] >= 0
        assert catchup["planned_file_count"] is None
        assert catchup["eta_s"] is None
        snapshot = cast(dict[str, Any], status["status_snapshot"])
        assert snapshot["age_s"] >= 0
    finally:
        release.set()
        try:
            result = await asyncio.wait_for(task, 3)
        finally:
            reset_status_snapshot()
            reset_discovery_progress()
    assert [item.payload for item in result] == [accepted]


def test_completed_discovery_restores_cached_ingest_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polylogue.daemon.status_snapshot._status_frame", lambda: None)
    refresh_status_snapshot(
        payload={
            "catchup": {
                "mode": "catching_up",
                "current_phase": "parsing",
                "current_source": "ingest-source",
                "current_path": "synthetic-path",
            }
        }
    )
    token = begin_discovery("walk-source")
    try:
        during = cast(dict[str, Any], get_status_snapshot_payload()["catchup"])
        assert during["current_phase"] == "discovering"
    finally:
        end_discovery(token)
    try:
        after = cast(dict[str, Any], get_status_snapshot_payload()["catchup"])
        assert after["mode"] == "catching_up"
        assert after["current_phase"] == "parsing"
        assert after["current_source"] == "ingest-source"
        assert after["current_path"] == "synthetic-path"
    finally:
        reset_status_snapshot()
        reset_discovery_progress()


def test_pending_source_contributes_while_another_source_runs() -> None:
    first = begin_discovery("first")
    try:
        advance_discovery(first, inspected=3, disposition="excluded")
        end_discovery(first, pending=True)
        second = begin_discovery("second")
        try:
            advance_discovery(second, inspected=2, disposition="accepted")
            progress = active_discovery_payload()
            assert progress is not None
            assert progress["current_source"] == "second"
            assert progress["discovery_active_walk_count"] == 1
            assert progress["discovery_pending_walk_count"] == 1
            assert progress["discovery_counter_scope"] == "all_active_and_pending_walks"
            assert progress["discovery_inspected_count"] == 5
            assert progress["discovery_accepted_count"] == 1
            assert progress["discovery_rejected_count"] == 1
        finally:
            end_discovery(second)
    finally:
        reset_discovery_progress()


@pytest.mark.asyncio
async def test_cancelled_discovery_keeps_worker_progress_until_walk_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    accepted = source_root / "session.json"
    accepted.write_text("{}")
    source = WatchSource(name="synthetic", root=source_root, suffixes=(".json",))
    watcher = SimpleNamespace(intake_revision=lambda _source: 0)
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=(source,)),  # type: ignore[arg-type]
        source,
    )
    entered = threading.Event()
    retry_entered = threading.Event()
    release = threading.Event()
    ordered_children = discovery_module._ordered_children
    observed_discover = adapter._observed_discover_sync
    calls = 0

    def held_children(*args: object, **kwargs: object) -> object:
        entered.set()
        assert release.wait(3)
        return cast(Any, ordered_children)(*args, **kwargs)

    def observed(limit: int, cancelled: threading.Event) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            retry_entered.set()
        return observed_discover(limit, cancelled)

    monkeypatch.setattr(discovery_module, "_ordered_children", held_children)
    monkeypatch.setattr(adapter, "_observed_discover_sync", observed)
    first = asyncio.create_task(adapter.discover(limit=1))
    retry: Any = None
    result: Any = ()
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        retry = asyncio.create_task(adapter.discover(limit=1))
        assert await asyncio.to_thread(retry_entered.wait, 2)
        progress = active_discovery_payload()
        assert progress is not None
        assert progress["current_phase"] == "discovering"
        assert progress["discovery_active_walk_count"] == 1
        assert progress["discovery_inspected_count"] == 0
    finally:
        release.set()
        try:
            if retry is not None:
                result = await asyncio.wait_for(retry, 3)
        finally:
            reset_discovery_progress()
    assert [item.payload for item in result] == [accepted]
