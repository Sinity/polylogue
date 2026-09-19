"""Measure the intake route ``polylogued run`` actually executes.

polylogue-oc9o0 / polylogue-v4dcc AC4. Rehearsal-4 (2026-09-03) and the
09-15 ``real_ingest_driver.py`` / ``hook_drain_driver.py`` numbers were
measured on the watcher chunk/catch-up/hook-drain route. That route is
deleted. Those receipts are historical for a deleted path, not current
production.

Production schedules through ``FairIntakeDispatcher`` +
``FileIntakeAdapter.admit_page``. Direct ``LiveBatchProcessor.ingest_files``
is the write entry the adapter calls, not the scheduler. This module
records both on the same synthetic page so a dispatcher that has become
an extra serial cost is visible.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.daemon.convergence import DaemonConverger
from polylogue.daemon.convergence_stages import make_default_convergence_stages
from polylogue.daemon.intake import FairIntakeDispatcher, IntakeClassSpec
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteEvent
from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.watcher import LiveWatcher, WatchSource
from polylogue.storage import raw_retention
from tests.infra.workload_declarations import convergence_corpus_specs

_MAX_DISPATCHER_PASSES = 32
_THROUGHPUT_BOUND = 1.5


@dataclass(frozen=True, slots=True)
class _DispatcherMeasurement:
    """One production-dispatcher receipt with its writer-hold split."""

    total_s: float
    payload_bytes: int
    end_to_end_mb_s: float
    succeeded_files: int
    failed_files: int
    files: int
    passes: int
    writer_hold_s: float | None = None
    outside_writer_hold_s: float | None = None
    raw_compaction_runs: int = 0
    raw_compaction_time_s: float | None = None


def _write_corpus(root: Path, *, prefix: str = "dispatcher-measure") -> Path:
    """Small deterministic Claude Code corpus: 10 files, 10 messages each."""
    spec = convergence_corpus_specs("xs-tiny-files")[0]
    project = root / "corpus" / "test-project"
    SyntheticCorpus.write_spec_artifacts(spec, project, prefix=prefix, index_width=4)
    return project.parent


def _jsonl_files(corpus_root: Path) -> list[Path]:
    return sorted(corpus_root.rglob("*.jsonl"))


def _payload_bytes(files: list[Path]) -> int:
    return sum(path.stat().st_size for path in files)


def _mb_s(payload_bytes: int, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        raise AssertionError("elapsed time was not measurable")
    return (payload_bytes / (1024 * 1024)) / elapsed_s


def _run_direct_ingest(corpus_root: Path, archive_root: Path) -> dict[str, float]:
    """The write entry without the dispatcher: historical comparison, not production scheduling."""
    files = _jsonl_files(corpus_root)
    db_path = archive_root / "index.db"
    converger = DaemonConverger(stages=make_default_convergence_stages(db_path))
    polylogue = SimpleNamespace(archive_root=archive_root, backend=SimpleNamespace(db_path=db_path))
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="claude-code", root=corpus_root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="dispatcher-measure-direct",
        converger=converger,
    )
    started = time.perf_counter()
    metrics = asyncio.run(processor.ingest_files(files, emit_event=False))
    elapsed = time.perf_counter() - started
    payload = _payload_bytes(files)
    return {
        "total_s": elapsed,
        "payload_bytes": float(payload),
        "end_to_end_mb_s": _mb_s(payload, elapsed),
        "succeeded_files": float(metrics.succeeded_file_count),
        "failed_files": float(metrics.failed_file_count),
        "files": float(len(files)),
        "passes": 1.0,
    }


def _run_dispatcher_ingest(
    corpus_root: Path,
    archive_root: Path,
    *,
    observe_writer_holds: bool = False,
) -> _DispatcherMeasurement:
    """The production scheduler: FairIntakeDispatcher + FileIntakeAdapter."""
    files = _jsonl_files(corpus_root)
    db_path = archive_root / "index.db"
    source = WatchSource(name="claude-code", root=corpus_root)
    converger = DaemonConverger(stages=make_default_convergence_stages(db_path))
    polylogue = SimpleNamespace(archive_root=archive_root, backend=SimpleNamespace(db_path=db_path))
    writer_events: list[DaemonWriteEvent] = []
    batch_payloads: list[dict[str, object]] = []

    def record_batch_event(name: str, payload: dict[str, object]) -> None:
        if name == "ingestion_batch":
            batch_payloads.append(payload)

    coordinator = (
        DaemonWriteCoordinator(observer=writer_events.append, archive_root=archive_root)
        if observe_writer_holds
        else None
    )
    watcher = LiveWatcher(
        cast(Any, polylogue),
        (source,),
        cursor=CursorStore(db_path),
        converger=converger,
        write_coordinator=coordinator,
        event_emitter=record_batch_event if observe_writer_holds else None,
    )
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=archive_root, watcher=watcher, sources=(source,)),
        source,
        class_name="configured_local",
    )
    dispatcher = FairIntakeDispatcher(
        (IntakeClassSpec(name="configured_local", adapter=adapter, page_size=32),),
        frame="test:dispatcher-measure",
    )

    async def drain() -> tuple[int, int]:
        admitted = 0
        passes = 0
        try:
            while passes < _MAX_DISPATCHER_PASSES:
                result = await dispatcher.run_once()
                passes += 1
                admitted += result.admitted
                if not result.progressed:
                    break
            else:
                raise AssertionError(f"dispatcher did not drain in {_MAX_DISPATCHER_PASSES} passes")
        finally:
            watcher.stop()
            if coordinator is not None:
                assert await coordinator.shutdown(timeout=1.0)
        return admitted, passes

    started = time.perf_counter()
    admitted, passes = asyncio.run(drain())
    elapsed = time.perf_counter() - started
    payload = _payload_bytes(files)
    released = [event for event in writer_events if event.phase == "released"]
    page_holds = [event.hold_seconds for event in released if event.actor == "watcher.live_ingest"]
    writer_hold_s = page_holds[0] if len(page_holds) == 1 else None
    batch_payload = batch_payloads[0] if len(batch_payloads) == 1 else {}
    stage_timings = cast(dict[str, object], batch_payload.get("stage_timings_s", {}))
    raw_compaction_runs = batch_payload.get("raw_compaction_runs")
    raw_compaction_time_s = stage_timings.get("raw_compaction")
    return _DispatcherMeasurement(
        total_s=elapsed,
        payload_bytes=payload,
        end_to_end_mb_s=_mb_s(payload, elapsed),
        succeeded_files=admitted,
        failed_files=0,
        files=len(files),
        passes=passes,
        writer_hold_s=writer_hold_s,
        outside_writer_hold_s=(elapsed - writer_hold_s) if writer_hold_s is not None else None,
        raw_compaction_runs=raw_compaction_runs if isinstance(raw_compaction_runs, int) else 0,
        raw_compaction_time_s=float(raw_compaction_time_s) if isinstance(raw_compaction_time_s, (int, float)) else None,
    )


def test_dispatcher_intake_is_within_direct_ingest_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dispatcher throughput stays within 1.5x of a direct ingest_files call.

    Anti-vacuity: schedule through ingest_files as the outer loop (the deleted
    chunk route) and this still records a number, but it is no longer the
    production scheduler. Dropping the dispatcher construction would make
    both probes identical and hide a serial cost the live daemon pays.

    Rehearsal-4 and 09-15 receipts measured that deleted chunk route.
    """
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "polylogue.toml"))

    dispatcher_corpus = _write_corpus(tmp_path / "dispatcher")
    direct_corpus = _write_corpus(tmp_path / "direct")
    dispatcher = _run_dispatcher_ingest(dispatcher_corpus, tmp_path / "dispatcher-archive")
    direct = _run_direct_ingest(direct_corpus, tmp_path / "direct-archive")

    assert dispatcher.files == direct["files"] > 0
    assert dispatcher.succeeded_files == dispatcher.files
    assert direct["succeeded_files"] == direct["files"]
    assert direct["failed_files"] == 0
    assert dispatcher.passes >= 1

    dispatcher_mb_s = dispatcher.end_to_end_mb_s
    direct_mb_s = direct["end_to_end_mb_s"]
    ratio = direct_mb_s / dispatcher_mb_s
    assert ratio <= _THROUGHPUT_BOUND, (
        f"dispatcher end_to_end_mb_s {dispatcher_mb_s:.4f} is {ratio:.2f}x slower than "
        f"direct ingest_files {direct_mb_s:.4f} (bound {_THROUGHPUT_BOUND}); "
        f"dispatcher_s={dispatcher.total_s:.4f} direct_s={direct['total_s']:.4f} "
        f"payload_bytes={dispatcher.payload_bytes} passes={dispatcher.passes}"
    )


def test_dispatcher_retention_authority_is_scoped_to_its_admitted_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live page never asks retention to inventory an unrelated archive.

    This reaches the production dispatcher, adapter, batch materialization,
    and raw-compaction callback. Anti-vacuity: removing the source-path scope
    from ``LiveBatchProcessor._compact_superseded_raw_snapshots`` leaves the
    batch successful, but makes this assertion red before the global query
    can return deletion authority.
    """
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "polylogue.toml"))
    corpus = _write_corpus(tmp_path / "dispatcher")
    expected_paths = frozenset(_jsonl_files(corpus))
    observed_scopes: list[frozenset[Path] | None] = []
    original = raw_retention.active_raw_retention_authority

    def recording_authority(*args: object, **kwargs: object) -> raw_retention.RawRetentionAuthority:
        paths = kwargs.get("authority_source_paths")
        observed_scopes.append(frozenset(cast(list[Path], paths)) if paths is not None else None)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(raw_retention, "active_raw_retention_authority", recording_authority)

    result = _run_dispatcher_ingest(corpus, tmp_path / "archive")

    assert result.succeeded_files == result.files
    assert observed_scopes == [expected_paths]


def test_dispatcher_page_compaction_cost_is_one_scoped_hold_at_archive_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real dispatcher pages keep compaction bounded as unrelated archive rows grow.

    This runs the ordinary dispatcher, adapter, live materialization, and
    compaction callback under the daemon coordinator.  It records actual
    released writer holds rather than timing a local helper.  The three archive
    populations distinguish an input-bounded retention call from a recurrence
    of archive-wide work.  Anti-vacuity: removing
    ``authority_source_paths=paths`` in the production compaction callback
    changes every observed scope to ``None`` while the page still completes.
    """
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "polylogue.toml"))

    observed_scopes: list[frozenset[Path] | None] = []
    original = raw_retention.active_raw_retention_authority

    def recording_authority(*args: object, **kwargs: object) -> raw_retention.RawRetentionAuthority:
        paths = kwargs.get("authority_source_paths")
        observed_scopes.append(frozenset(cast(list[Path], paths)) if paths is not None else None)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(raw_retention, "active_raw_retention_authority", recording_authority)
    measurements: list[_DispatcherMeasurement] = []
    expected_scopes: list[frozenset[Path]] = []
    for label, existing_batches in (("empty", 0), ("medium", 2), ("large", 4)):
        archive_root = tmp_path / f"{label}-archive"
        for batch in range(existing_batches):
            seed = _write_corpus(tmp_path / f"{label}-seed-{batch}", prefix=f"{label}-seed-{batch}")
            expected_scopes.append(frozenset(_jsonl_files(seed)))
            _run_dispatcher_ingest(seed, archive_root)
        corpus = _write_corpus(tmp_path / f"{label}-measurement", prefix=f"{label}-measurement")
        expected_scopes.append(frozenset(_jsonl_files(corpus)))
        measurements.append(_run_dispatcher_ingest(corpus, archive_root, observe_writer_holds=True))

    assert observed_scopes == expected_scopes
    assert all(measurement.succeeded_files == measurement.files > 0 for measurement in measurements)
    assert all(measurement.failed_files == 0 for measurement in measurements)
    assert all(measurement.writer_hold_s is not None and measurement.writer_hold_s > 0 for measurement in measurements)
    assert all(
        measurement.outside_writer_hold_s is not None and measurement.outside_writer_hold_s >= 0
        for measurement in measurements
    )
    assert all(measurement.raw_compaction_runs == 1 for measurement in measurements)
    assert all(
        measurement.raw_compaction_time_s is not None and measurement.raw_compaction_time_s > 0
        for measurement in measurements
    )


def test_rehearsal_chunk_route_numbers_are_labelled_deleted() -> None:
    """The module docstring is the receipt that those numbers are not production."""
    text = Path(__file__).read_text(encoding="utf-8")
    assert "Rehearsal-4" in text
    assert "deleted" in text
    assert "FairIntakeDispatcher" in text
