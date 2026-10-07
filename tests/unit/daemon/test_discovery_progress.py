"""The first source walk stays visible before it yields an intake item."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.daemon.discovery_progress import (
    abandon_discovery,
    active_discovery_payload,
    advance_discovery,
    begin_discovery,
    end_discovery,
    overlay_active_discovery,
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
from polylogue.sources.source_layout import export_drop_layout


def test_halted_owner_can_abandon_its_parked_discovery_progress() -> None:
    """A halted adapter no longer keeps stale pending progress in status.

    Anti-vacuity: remove abandon_discovery and the pending source remains
    projected despite the owner being permanently unschedulable.
    """
    reset_discovery_progress()

    class Owner:
        pass

    owner = Owner()
    token = begin_discovery("halted", owner=owner)
    end_discovery(token, pending=True)
    assert active_discovery_payload() is not None

    abandon_discovery(owner)

    assert active_discovery_payload() is None
    reset_discovery_progress()


@pytest.mark.asyncio
async def test_status_shows_first_discovery_while_sibling_sort_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    accepted = source_root / "session.json"
    accepted.write_text("{}")
    source = WatchSource(name="synthetic", root=source_root, layout=export_drop_layout((".json",)))
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
        assert catchup.get("planned_file_count") is None
        assert catchup.get("eta_s") is None
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


def test_aggregate_last_advance_age_uses_most_recent_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 0.0
    monkeypatch.setattr("polylogue.daemon.discovery_progress.time.monotonic", lambda: now)
    first = begin_discovery("stale-pending")
    end_discovery(first, pending=True)
    now = 10.0
    second = begin_discovery("active")
    now = 12.0
    advance_discovery(second, inspected=1)
    now = 15.0
    try:
        status = overlay_active_discovery({"catchup": {}})
        catchup = status["catchup"]
        assert isinstance(catchup, dict)
        assert catchup["discovery_inspected_count"] == 1
        assert catchup["discovery_counter_scope"] == "all_active_and_pending_walks"
        assert catchup["discovery_last_advanced_age_s"] == 3.0
    finally:
        end_discovery(second)
        reset_discovery_progress()


@pytest.mark.asyncio
async def test_cancelled_discovery_keeps_worker_progress_until_walk_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    accepted = source_root / "session.json"
    accepted.write_text("{}")
    source = WatchSource(name="synthetic", root=source_root, layout=export_drop_layout((".json",)))
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


def test_cold_build_preparation_is_visible_before_the_first_intake_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping the progress callback from the cold build observation or ``begin`` makes this red.

    The real baseline walk and revision hashing report through the daemon's
    preparation state; status observed mid-preparation carries the phase and
    counts, and the finished preparation stops describing the current phase.
    """
    import io
    import json

    from polylogue.daemon.discovery_progress import (
        advance_cold_build_preparation,
        begin_cold_build_preparation,
        end_cold_build_preparation,
    )
    from polylogue.logging import add_sink, make_stream_sink, remove_sink
    from polylogue.sources.live.cold_build import ColdBuildGeneration

    archive = tmp_path / "archive"
    archive.mkdir()
    source_root = tmp_path / "source"
    source_root.mkdir()
    # Intake excludes material it would not parse as a session before hashing it.
    payloads = [
        (
            f'{{"type":"session_meta","payload":{{"id":"progress-{index}","timestamp":"2026-06-02T00:00:00Z"}}}}\n'
            '{"type":"response_item","payload":{"type":"message","id":"message-0",'
            '"role":"user","content":[{"type":"input_text","text":"Synthetic"}]}}\n'
        ).encode()
        for index in range(2)
    ]
    for index, payload in enumerate(payloads):
        (source_root / f"session-{index}.jsonl").write_bytes(payload)
    source = WatchSource(name="codex", root=source_root, layout=export_drop_layout((".jsonl",)))
    observed: dict[str, dict[str, Any]] = {}

    def observing_progress(phase: str, **counts: int) -> None:
        advance_cold_build_preparation(phase, **counts)
        if phase not in observed:
            observed[phase] = cast(dict[str, Any], active_discovery_payload())

    monkeypatch.setattr("polylogue.daemon.status_snapshot._status_frame", lambda: None)
    refresh_status_snapshot(payload={"catchup": {"mode": "idle"}})
    stream = io.StringIO()
    sink = add_sink(make_stream_sink(stream, fmt="json"))
    begin_cold_build_preparation()
    try:
        ColdBuildGeneration.begin(
            archive,
            reason="test",
            observed=ColdBuildGeneration.observe_source_baseline((source,), progress=observing_progress),
            progress=observing_progress,
        )
        during = cast(dict[str, Any], get_status_snapshot_payload()["catchup"])
    finally:
        end_cold_build_preparation()
        remove_sink(sink)
    try:
        after = cast(dict[str, Any], get_status_snapshot_payload()["catchup"])
    finally:
        reset_status_snapshot()
        reset_discovery_progress()

    assert list(observed)[:2] == ["baseline_walk", "baseline_hash"]
    assert "capacity_projection" in observed and list(observed)[-1] == "generation_create"
    hashed = observed["capacity_projection"]
    assert hashed["mode"] == "cold_build_preparing"
    assert hashed["current_phase"] == "capacity_projection"
    assert hashed["preparation_inspected_count"] >= len(payloads)
    assert hashed["preparation_revision_count"] == len(payloads)
    assert hashed["preparation_hashed_bytes"] == sum(len(payload) for payload in payloads)
    assert hashed["planned_file_count"] is None and hashed["eta_s"] is None
    assert during["mode"] == "cold_build_preparing"
    assert during["current_phase"] == "generation_create"
    assert during["last_advanced_age_s"] >= 0
    assert after["mode"] == "idle"
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert not [record for record in records if record["event"] == "log.field_rejected"]
    phases = [record["phase"] for record in records if record["event"] == "daemon.cold_build.preparation"]
    assert phases[0] == "baseline_walk" and phases[-1] == "prepared"
    assert {"capacity_projection", "generation_create"} <= set(phases)
    final = [record for record in records if record["event"] == "daemon.cold_build.preparation"][-1]
    assert final["files"] == len(payloads)
    assert final["bytes"] == sum(len(payload) for payload in payloads)


def test_failed_cold_build_preparation_ends_with_error_and_clears_phase() -> None:
    from polylogue.daemon.discovery_progress import (
        advance_cold_build_preparation,
        begin_cold_build_preparation,
        end_cold_build_preparation,
    )

    begin_cold_build_preparation()
    try:
        advance_cold_build_preparation("baseline_walk", inspected=4)
        progress = active_discovery_payload()
        assert progress is not None
        assert progress["preparation_inspected_count"] == 4
    finally:
        end_cold_build_preparation(failed=True)
    try:
        assert active_discovery_payload() is None
        # Counts after the owner ended are not work the current build did.
        advance_cold_build_preparation("baseline_walk", inspected=1)
        assert active_discovery_payload() is None
    finally:
        reset_discovery_progress()


def test_status_lines_name_cold_build_preparation() -> None:
    from polylogue.daemon.catchup_status import format_catchup_status_lines

    lines = format_catchup_status_lines(
        {
            "mode": "cold_build_preparing",
            "current_phase": "baseline_hash",
            "preparation_inspected_count": 9,
            "preparation_revision_count": 4,
            "preparation_hashed_bytes": 512,
            "preparation_age_s": 3.0,
            "preparation_phase_age_s": 1.0,
            "last_advanced_age_s": 0.25,
        }
    )
    preparing = [line for line in lines if "cold build preparing" in line]
    assert preparing == [
        "  cold build preparing: phase=baseline_hash inspected=9 revisions=4 hashed=512 bytes "
        "age=3.0s phase_age=1.0s last_advance=0.25s planned=unknown"
    ]


def test_cancelled_cold_build_preparation_is_not_reported_as_an_error() -> None:
    """Routing cancellation through ``failed=True`` makes this red."""
    import io
    import json

    from polylogue.daemon.discovery_progress import (
        advance_cold_build_preparation,
        begin_cold_build_preparation,
        end_cold_build_preparation,
    )
    from polylogue.logging import add_sink, make_stream_sink, remove_sink

    stream = io.StringIO()
    sink = add_sink(make_stream_sink(stream, fmt="json"))
    begin_cold_build_preparation()
    try:
        advance_cold_build_preparation("baseline_hash", revisions=2, hashed_bytes=10)
    finally:
        end_cold_build_preparation(cancelled=True)
        remove_sink(sink)
        reset_discovery_progress()
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert not [record for record in records if record["event"] == "log.field_rejected"]
    final = [record for record in records if record["event"] == "daemon.cold_build.preparation"][-1]
    assert final["level"] == "info"
    assert final["outcome"] == "skipped"
    assert final["reason"] == "cancelled"
    assert final["phase"] == "baseline_hash"
    assert active_discovery_payload() is None


@pytest.mark.asyncio
async def test_cancelled_caller_keeps_preparation_until_admitted_writer_stops(tmp_path: Path) -> None:
    """Ending preparation in the caller's cancellation path makes this red.

    The write coordinator shields an admitted execution from caller
    cancellation; the hashing thread keeps running, so status must keep the
    phase until that execution completes.
    """
    from polylogue.daemon.discovery_progress import run_cold_build_preparation
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator

    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    started = threading.Event()
    release = threading.Event()

    def observing(*, progress: Any, cancelled: Any) -> str:
        progress("baseline_hash", revisions=1, hashed_bytes=3)
        return "observed"

    def preparing(*, observed: str, progress: Any) -> str:
        assert observed == "observed"
        started.set()
        assert release.wait(5), "writer was not released"
        progress("generation_create")
        return "generation"

    caller = asyncio.create_task(
        run_cold_build_preparation(coordinator, "daemon.cold_build.begin", observing, preparing)
    )
    try:
        assert await asyncio.to_thread(started.wait, 5)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        during = active_discovery_payload()
        assert during is not None
        assert during["current_phase"] == "baseline_hash"
        assert during["preparation_revision_count"] == 1
        release.set()
        assert await coordinator.shutdown(timeout=5)
        await asyncio.sleep(0)
        assert active_discovery_payload() is None
    finally:
        release.set()
        reset_discovery_progress()


@pytest.mark.asyncio
async def test_cold_build_source_observation_runs_before_and_outside_the_writer_call(tmp_path: Path) -> None:
    """Observing the sources inside the writer call makes this red.

    Pre-acquisition classification and hashing only read source files; the
    writer call receives their finished observation and binds it.
    """
    from polylogue.daemon.discovery_progress import run_cold_build_preparation
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator

    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    order: list[tuple[str, str]] = []

    def observing(*, progress: Any, cancelled: Any) -> str:
        order.append(("observe", threading.current_thread().name))
        progress("baseline_walk", inspected=1)
        assert not cancelled()
        return "observed"

    def preparing(*, observed: str, progress: Any) -> str:
        order.append(("begin", threading.current_thread().name))
        return f"bound:{observed}"

    try:
        result = await run_cold_build_preparation(coordinator, "daemon.cold_build.begin", observing, preparing)
    finally:
        assert await coordinator.shutdown(timeout=5)
        reset_discovery_progress()
    assert result == "bound:observed"
    assert [step for step, _thread in order] == ["observe", "begin"]
    assert order[0][1] == "cold-source-observation"
    assert order[1][1] != order[0][1]


@pytest.mark.asyncio
async def test_unadmitted_cancelled_preparation_ends_at_the_caller(tmp_path: Path) -> None:
    from polylogue.daemon.discovery_progress import run_cold_build_preparation
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator

    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    holder_started = asyncio.Event()
    holder_release = asyncio.Event()

    async def hold() -> None:
        holder_started.set()
        await holder_release.wait()

    holder = asyncio.create_task(coordinator.run("daemon.test.hold", hold))
    try:
        await holder_started.wait()
        caller = asyncio.create_task(
            run_cold_build_preparation(
                coordinator,
                "daemon.cold_build.begin",
                lambda *, progress, cancelled: None,
                lambda *, observed, progress: None,
            )
        )
        await asyncio.sleep(0.01)
        assert active_discovery_payload() is not None
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert active_discovery_payload() is None
    finally:
        holder_release.set()
        await holder
        reset_discovery_progress()


def test_discovery_paging_during_a_cold_build_keeps_the_build_eta_and_mode() -> None:
    """A paged walk inside a cold build adds counters; it never hides the ETA.

    Anti-vacuity: the previous overlay replaced ``mode``/``eta_s`` with
    ``discovery_pending``/``None`` for the whole paged walk.
    """
    walk = begin_discovery("synthetic")
    try:
        base: dict[str, object] = {"catchup": {"mode": "catching_up", "planned_raw_revision_count": 40, "eta_s": 12.5}}
        catchup = overlay_active_discovery(base)["catchup"]
        assert isinstance(catchup, dict)
        assert catchup["mode"] == "catching_up"
        assert catchup["eta_s"] == 12.5
        assert catchup["discovery_pending"] is True
    finally:
        end_discovery(walk)
        reset_discovery_progress()


def test_status_frame_names_a_root_without_an_active_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first cold build has a definite frame, so its snapshot can be fresh.

    Anti-vacuity: returning ``None`` for a missing active index made every
    status snapshot ``unavailable`` for the whole first build.
    """
    from polylogue.daemon import status_snapshot

    monkeypatch.setattr(status_snapshot, "archive_root", lambda: tmp_path)
    assert status_snapshot._status_frame() == status_snapshot.NO_ACTIVE_GENERATION_FRAME


def test_status_frame_distinguishes_a_broken_pointer_from_a_fresh_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present pointer naming a deleted generation is unavailable, not fresh.

    Anti-vacuity: collapsing every ``FileNotFoundError`` into the first-build
    sentinel makes a status snapshot report fresh while the archive's
    declared active generation is broken.
    """
    from polylogue.daemon import status_snapshot
    from polylogue.storage.archive_identity import ACTIVE_POINTER_FILENAME

    monkeypatch.setattr(status_snapshot, "archive_root", lambda: tmp_path)
    missing_generation = tmp_path / ".index-generations" / "gen-deleted" / "index.db"
    (tmp_path / ACTIVE_POINTER_FILENAME).write_text(str(missing_generation), encoding="utf-8")
    assert status_snapshot._status_frame() is None
