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
import sqlite3
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.daemon.intake import FairIntakeDispatcher, IntakeClassReport, IntakeClassSpec, IntakePass
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteEvent
from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.watcher import LiveWatcher, WatchSource
from polylogue.storage import frontier_existence, raw_retention
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.daemon_cold_start import write_fixture
from tests.infra.live_batch import prepared_live_batch_processor
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.workload_declarations import convergence_corpus_specs

_MAX_DISPATCHER_PASSES = 64
_THROUGHPUT_BOUND = 1.5
_MEASUREMENT_PAIRS = 3
_MEASURED_PAGE_FILES = 5


@dataclass(frozen=True, slots=True)
class _DispatcherMeasurement:
    """One production-dispatcher receipt with its actual writer-hold split."""

    total_s: float
    payload_bytes: int
    end_to_end_mb_s: float
    succeeded_files: int
    failed_files: int
    retried_files: int
    deferred_files: int
    files: int
    passes: int
    writer_hold_s: float | None = None
    outside_writer_hold_s: float | None = None
    writer_hold_count: int = 0
    raw_compaction_runs: int = 0
    raw_compaction_time_s: float | None = None


def _write_corpus(root: Path, *, prefix: str = "dispatcher-measure", keep_files: int | None = None) -> Path:
    """Small deterministic Claude Code corpus: 10 files, 10 messages each.

    ``keep_files`` trims the corpus to its first N files. Each file is an
    independent session, so a trimmed corpus is still a well-formed page.
    The throughput comparison runs eight ingests and each file costs both
    arms the same ~0.2 s, so the trim is what keeps that test inside a
    unit-test budget: 14 s idle here against 103 s untrimmed under load,
    with a 120 s default per-test timeout.
    """
    spec = convergence_corpus_specs("xs-tiny-files")[0]
    # The corpus root is a Claude Code projects directory; the writer puts
    # each session in a project directory below it.
    corpus = root / "corpus"
    SyntheticCorpus.write_spec_artifacts(spec, corpus, prefix=prefix, index_width=4)
    if keep_files is not None:
        for extra in _jsonl_files(corpus)[keep_files:]:
            extra.unlink()
    return corpus


def _jsonl_files(corpus_root: Path) -> list[Path]:
    return sorted(corpus_root.rglob("*.jsonl"))


def _payload_bytes(files: list[Path]) -> int:
    return sum(path.stat().st_size for path in files)


def _mb_s(payload_bytes: int, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        raise AssertionError("elapsed time was not measurable")
    return (payload_bytes / (1024 * 1024)) / elapsed_s


def _run_direct_ingest(corpus_root: Path, archive_root: Path) -> dict[str, float]:
    """The write entry without the dispatcher: historical comparison, not production scheduling.

    The call shape mirrors ``FileIntakeAdapter.admit_page`` exactly
    (``polylogue/operations/intake_adapters.py``): the adapter passes the
    page's paths with ``queued_file_count`` and ``whole_archive_convergence=
    False`` and leaves ``emit_event`` at its default. Leaving
    ``whole_archive_convergence`` at its own default here would put
    archive-wide convergence on one arm only, so the two arms would not be
    the same unit of work.

    Both arms parse through the processor's own preparation route.
    """
    files = _jsonl_files(corpus_root)
    # The fixture writer's lease names an existing archive root; it bootstraps the tiers.
    archive_root.mkdir(parents=True, exist_ok=True)

    async def run() -> tuple[Any, float]:
        async with prepared_live_batch_processor(
            archive_root,
            (WatchSource(name="claude-code", root=corpus_root),),
            parser_fingerprint="dispatcher-measure-direct",
        ) as processor:
            started = time.perf_counter()
            metrics = await processor.ingest_files(
                files,
                queued_file_count=len(files),
                emit_event=True,
                whole_archive_convergence=False,
            )
            return metrics, time.perf_counter() - started

    metrics, elapsed = asyncio.run(run())
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
    polylogue = SimpleNamespace(archive_root=archive_root, backend=SimpleNamespace(db_path=db_path))
    writer_events: list[DaemonWriteEvent] = []
    batch_payloads: list[dict[str, object]] = []

    def record_batch_event(name: str, payload: dict[str, object]) -> None:
        if name == "ingestion_batch":
            batch_payloads.append(payload)

    async def drain() -> tuple[int, int, int, int, int, float]:
        admitted = 0
        failed = 0
        retried = 0
        deferred = 0
        passes = 0
        # The writer's lease names an existing archive root; bootstrap fills it.
        archive_root.mkdir(parents=True, exist_ok=True)
        coordinator = DaemonWriteCoordinator(
            observer=writer_events.append if observe_writer_holds else None, archive_root=archive_root
        )
        await coordinator.run_sync("fixture.dispatcher.bootstrap", lambda: bootstrap_archive_root(archive_root))
        try:
            async with prepared_live_convergence_owner(archive_root, write_coordinator=coordinator) as owner:
                watcher = LiveWatcher(
                    cast(Any, polylogue),
                    (source,),
                    cursor=await coordinator.run_sync(
                        "fixture.dispatcher.cursor",
                        lambda: CursorStore(db_path, ops_db_path=archive_root / "ops.db"),
                    ),
                    write_coordinator=coordinator,
                    event_emitter=record_batch_event if observe_writer_holds else None,
                    append_runner=owner.ingest_append_plans,
                    retained_runner=owner.ingest_retained_raw_ids,
                    convergence_runner=owner.run_convergence_sync,
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
                # Time the scheduled passes only, the same unit the direct arm
                # times: archive bootstrap, owner composition and coordinator
                # shutdown are fixture setup on both arms.
                started = time.perf_counter()
                try:
                    while passes < _MAX_DISPATCHER_PASSES:
                        result = await dispatcher.run_once()
                        passes += 1
                        admitted += result.admitted
                        for report in result.classes:
                            retried += report.retried
                            deferred += report.deferred
                            failed += report.retried + report.isolated
                        if result.quiescent and not adapter.discovery_pending:
                            if admitted == len(files):
                                break
                            # A preparation deferral can leave a retry cursor due
                            # after this otherwise idle pass. Give it a bounded turn.
                            await asyncio.sleep(0.25)
                    else:
                        raise AssertionError(f"dispatcher did not drain in {_MAX_DISPATCHER_PASSES} passes")
                    elapsed = time.perf_counter() - started
                finally:
                    watcher.stop()
        finally:
            assert await coordinator.shutdown(timeout=float("inf"))
        return admitted, failed, retried, deferred, passes, elapsed

    admitted, failed, retried, deferred, passes, elapsed = asyncio.run(drain())
    assert admitted == len(files), f"dispatcher published {admitted} of {len(files)} expected files"
    payload = _payload_bytes(files)
    released = [event for event in writer_events if event.phase == "released"]
    # The dispatcher no longer takes one page-wide lease.  Its ordinary batch
    # processor instead self-admits each bounded publication (full write,
    # compaction, and ops receipts).  Sum those actual released holds; looking
    # for the deleted ``watcher.live_ingest`` wrapper would report no hold at
    # all and hide the work this production route still serializes.
    page_holds = [
        event.hold_seconds
        for event in released
        if event.actor.startswith("watcher.live_ingest.") and event.hold_seconds is not None
    ]
    writer_hold_s = sum(page_holds) if page_holds else None
    batch_payload = batch_payloads[0] if len(batch_payloads) == 1 else {}
    stage_timings = cast(dict[str, object], batch_payload.get("stage_timings_s", {}))
    raw_compaction_runs = batch_payload.get("raw_compaction_runs")
    raw_compaction_time_s = stage_timings.get("raw_compaction")
    return _DispatcherMeasurement(
        total_s=elapsed,
        payload_bytes=payload,
        end_to_end_mb_s=_mb_s(payload, elapsed),
        succeeded_files=admitted,
        failed_files=failed,
        retried_files=retried,
        deferred_files=deferred,
        files=len(files),
        passes=passes,
        writer_hold_s=writer_hold_s,
        outside_writer_hold_s=(elapsed - writer_hold_s) if writer_hold_s is not None else None,
        writer_hold_count=len(page_holds),
        raw_compaction_runs=raw_compaction_runs if isinstance(raw_compaction_runs, int) else 0,
        raw_compaction_time_s=float(raw_compaction_time_s) if isinstance(raw_compaction_time_s, (int, float)) else None,
    )


def test_frontier_pages_reconcile_changes_without_repeating_global_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publishing dispatcher pages after bootstrap consume changed keys only.

    Restoring the archive-wide seed check to each admission makes this red.
    """
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "polylogue.toml"))
    bootstrap_calls = 0
    changed_keys: list[str] = []
    selected_sizes: list[int] = []
    original_missing = frontier_existence._missing_reference
    original_changed = frontier_existence._first_missing_reference
    original_selected = raw_retention.raw_frontier_blocked_selected_paths

    def count_missing(conn: sqlite3.Connection) -> bool:
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        return original_missing(conn)

    def count_changed(conn: sqlite3.Connection, raw_ids: set[str]) -> str | None:
        changed_keys.extend(raw_ids)
        return original_changed(conn, raw_ids)

    def count_selected(root: Path, paths: list[Path]) -> raw_retention.RawFrontierBlockedPaths:
        selected_sizes.append(len(paths))
        return original_selected(root, paths)

    monkeypatch.setattr(frontier_existence, "_missing_reference", count_missing)
    monkeypatch.setattr(frontier_existence, "_first_missing_reference", count_changed)
    monkeypatch.setattr(raw_retention, "raw_frontier_blocked_selected_paths", count_selected)
    archive = tmp_path / "archive"
    initialize_active_archive_root(archive)
    measurements: list[_DispatcherMeasurement] = []
    first_page_bootstraps = 0
    for size in (0, 2, 5):
        corpus = _write_corpus(tmp_path / f"frontier-{size}", prefix=f"frontier-{size}", keep_files=size)
        measurements.append(_run_dispatcher_ingest(corpus, archive, observe_writer_holds=True))
        if size == 2:
            first_page_bootstraps = bootstrap_calls

    assert first_page_bootstraps >= 1
    assert bootstrap_calls == first_page_bootstraps
    assert changed_keys
    assert selected_sizes and max(selected_sizes) <= 5
    assert all(item.succeeded_files == item.files for item in measurements)
    assert measurements[0].files == 0 and measurements[0].passes >= 1
    assert all(item.writer_hold_s is not None and item.outside_writer_hold_s is not None for item in measurements[1:])
    with sqlite3.connect(archive / "source.db") as source, sqlite3.connect(archive / "ops.db") as ops:
        assert source.execute("SELECT count(*) FROM raw_sessions WHERE canonical_source_path IS NULL").fetchone() == (
            0,
        )
        assert ops.execute(
            "SELECT count(*) FROM ingest_cursor WHERE canonical_source_path IS NULL "
            "AND byte_offset IS NOT NULL AND excluded = 0"
        ).fetchone() == (0,)


def test_dispatcher_measurement_drains_rejected_prefix_and_counts_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected first page cannot complete the witness; a retry cannot read as zero failures."""
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "polylogue.toml"))
    corpus = tmp_path / "source"
    write_fixture(corpus, rejected=300)

    rejected: list[Path] = []
    monkeypatch.setattr(
        "polylogue.sources.live.discovery._log_unclaimed_intake_candidate",
        lambda path, **_kwargs: rejected.append(path),
    )
    original_run_once = FairIntakeDispatcher.run_once
    calls = 0
    first_pass_pending = False

    async def observed_pass(self: FairIntakeDispatcher, *, budget: int = 0) -> IntakePass:
        nonlocal calls, first_pass_pending
        if calls == 1:
            calls += 1
            return IntakePass(classes=(IntakeClassReport(name="configured_local", retried=1, deferred=1),))
        result = await original_run_once(self, budget=budget) if budget else await original_run_once(self)
        calls += 1
        if calls == 1:
            first_pass_pending = not result.progressed and any(
                bool(getattr(spec.adapter, "discovery_pending", False)) for spec in self.classes
            )
            assert len(rejected) == 256
        return result

    monkeypatch.setattr(FairIntakeDispatcher, "run_once", observed_pass)
    result = _run_dispatcher_ingest(corpus, tmp_path / "archive")

    assert first_pass_pending
    assert len(rejected) == 300
    assert result.files == result.succeeded_files == 3
    assert result.passes >= 4
    assert result.failed_files == result.retried_files >= 1
    assert result.deferred_files >= 1


@pytest.mark.timeout(600)
def test_dispatcher_intake_is_within_direct_ingest_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dispatcher throughput stays within 1.5x of a direct ingest_files call.

    The comparison is between two *warm* routes over ``_MEASUREMENT_PAIRS``
    alternating-order pairs, and the bound is applied to the median ratio.
    Three measured arm-design facts force that shape:

    * The first ingest in a process pays the six-tier bootstrap, lazy
      imports and SQLite page-cache fill. Unwarmed, the arm that happened
      to run first carried 0.12-4.5 s of that first-touch cost on a ~1 s
      batch -- the process's cost, not the scheduler's, and it read as a
      4.26x dispatcher regression when the dispatcher arm ran first.
    * Whichever route is measured first leaves the other warmer, so a fixed
      arm order biases the ratio by construction; the order alternates.
    * A single pair is not resolvable against this bound: one arm of one
      pair has been seen 4x slow on a busy host while every other reading
      in the same run sat at ~1.0. The median of three absorbs one such
      reading; it cannot absorb a real regression, which moves every pair.

    Both arms must also be the same unit of work: ``_run_direct_ingest``
    repeats the adapter's own ``ingest_files`` keywords, because
    ``whole_archive_convergence`` differing between the arms silently
    charged one of them archive-wide convergence.

    Anti-vacuity: the bound is live, not decorative. Warm, the honest ratio
    is ~1.03 (measured 1.01/1.05/1.03 idle), so adding ~0.5x of serial
    per-page cost to ``FileIntakeAdapter.admit_page`` or
    ``FairIntakeDispatcher.run_once`` (a sleep, an extra archive open, a
    second parse) makes this red: a 0.6 s-per-file page sleep was measured
    at 3.69x here. Replacing the dispatcher arm with a bare ``ingest_files``
    loop -- the deleted chunk route -- would instead make both arms
    identical and the assertion vacuous; the dispatcher arm must keep
    scheduling through ``run_once``.

    Rehearsal-4 and 09-15 receipts measured that deleted chunk route.
    """
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "polylogue.toml"))

    # Warm both routes before any timing. Each throwaway run uses its own
    # corpus and its own archive, so no measured pair inherits a populated
    # index or an already-advanced cursor from the warm-up.
    _run_dispatcher_ingest(
        _write_corpus(tmp_path / "warm-dispatcher", prefix="warm-dispatcher", keep_files=1),
        tmp_path / "warm-dispatcher-archive",
    )
    _run_direct_ingest(
        _write_corpus(tmp_path / "warm-direct", prefix="warm-direct", keep_files=1),
        tmp_path / "warm-direct-archive",
    )

    ratios: list[float] = []
    receipts: list[str] = []
    for pair in range(_MEASUREMENT_PAIRS):
        dispatcher_corpus = _write_corpus(
            tmp_path / f"dispatcher-{pair}", prefix=f"dispatcher-{pair}", keep_files=_MEASURED_PAGE_FILES
        )
        direct_corpus = _write_corpus(
            tmp_path / f"direct-{pair}", prefix=f"direct-{pair}", keep_files=_MEASURED_PAGE_FILES
        )
        dispatcher_archive = tmp_path / f"dispatcher-archive-{pair}"
        direct_archive = tmp_path / f"direct-archive-{pair}"
        # Alternate the arm order: neither route is structurally first.
        if pair % 2 == 0:
            dispatcher = _run_dispatcher_ingest(dispatcher_corpus, dispatcher_archive)
            direct = _run_direct_ingest(direct_corpus, direct_archive)
        else:
            direct = _run_direct_ingest(direct_corpus, direct_archive)
            dispatcher = _run_dispatcher_ingest(dispatcher_corpus, dispatcher_archive)

        assert dispatcher.files == direct["files"] > 0
        assert dispatcher.succeeded_files == dispatcher.files
        assert direct["succeeded_files"] == direct["files"]
        assert direct["failed_files"] == 0
        assert dispatcher.passes >= 1

        ratio = direct["end_to_end_mb_s"] / dispatcher.end_to_end_mb_s
        ratios.append(ratio)
        receipts.append(
            f"pair={pair} first={'dispatcher' if pair % 2 == 0 else 'direct'} ratio={ratio:.2f} "
            f"dispatcher_mb_s={dispatcher.end_to_end_mb_s:.4f} direct_mb_s={direct['end_to_end_mb_s']:.4f} "
            f"dispatcher_s={dispatcher.total_s:.4f} direct_s={direct['total_s']:.4f} "
            f"payload_bytes={dispatcher.payload_bytes} passes={dispatcher.passes}"
        )

    median_ratio = statistics.median(ratios)
    assert median_ratio <= _THROUGHPUT_BOUND, (
        f"dispatcher is {median_ratio:.2f}x slower than direct ingest_files at the median "
        f"of {_MEASUREMENT_PAIRS} warm alternating pairs (bound {_THROUGHPUT_BOUND}); " + "; ".join(receipts)
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


# Nine full dispatcher ingests across three archives: the 120 s default hang
# guard fired under host load before any assertion ran.
@pytest.mark.timeout(600)
def test_dispatcher_page_compaction_cost_is_one_scoped_hold_at_archive_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real dispatcher pages keep compaction bounded as unrelated archive rows grow.

    This runs the ordinary dispatcher, adapter, live materialization, and
    compaction callback under the daemon coordinator.  It records actual
    released writer holds rather than timing a local helper, and traces the
    retention authority's real index reads.  The three archive populations
    distinguish input-bounded work from a recurrence of archive-wide work.
    Anti-vacuity: restoring one retention call per file makes the authority
    call/query-count assertion red; dropping ``authority_source_paths=paths``
    makes the constrained-SQL assertion red while the page still completes.
    """
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "polylogue.toml"))

    observed_scopes: list[frozenset[Path] | None] = []
    retention_queries: list[tuple[frozenset[Path] | None, tuple[str, ...]]] = []
    original = raw_retention.active_raw_retention_authority
    original_connect = sqlite3.connect
    index_statements: list[str] = []

    def tracing_connect(database: object, *args: object, **kwargs: object) -> sqlite3.Connection:
        connection = cast(Callable[..., sqlite3.Connection], original_connect)(database, *args, **kwargs)
        # The retention authority opens the active index read-only.  Do not
        # trace every source-tier write in the production page: only the three
        # fixed-cost authority reads decide whether retention grew with archive
        # size.
        if str(database).endswith("/index.db?mode=ro"):
            connection.set_trace_callback(index_statements.append)
        return connection

    # ``raw_retention`` imports this standard-library module, so this traces
    # its real connection without reaching into a private production symbol.
    monkeypatch.setattr(sqlite3, "connect", tracing_connect)

    def recording_authority(*args: object, **kwargs: object) -> raw_retention.RawRetentionAuthority:
        paths = kwargs.get("authority_source_paths")
        scope = frozenset(cast(list[Path], paths)) if paths is not None else None
        started = len(index_statements)
        result = original(*args, **kwargs)  # type: ignore[arg-type]
        observed_scopes.append(scope)
        retention_queries.append((scope, tuple(index_statements[started:])))
        return result

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
    assert len(retention_queries) == len(expected_scopes)
    for scope, statements in retention_queries:
        assert scope is not None
        selects = tuple(
            " ".join(statement.lower().split())
            for statement in statements
            if statement.lstrip().upper().startswith("SELECT")
        )
        # Every validated read-only open checks the derived schema identity
        # (#5727).  That is a per-connection cost: exactly one here means the
        # page's authority reads share one connection.
        identity_checks = tuple(statement for statement in selects if "from schema_identity" in statement)
        assert len(identity_checks) == 1
        selects = tuple(statement for statement in selects if statement not in identity_checks)
        # One current page receives exactly its three authority reads: session
        # references, accepted heads, and eligible receipts.  A per-file/chunk
        # recurrence grows this count even when the archive's final content is
        # identical, which is why this is a count rather than a timing bound.
        assert len(selects) == 3
        assert sum("select distinct raw_id from sessions" in statement for statement in selects) == 1
        assert sum("raw_revision_heads" in statement for statement in selects) == 2
        assert all(" in (" in statement for statement in selects)
        assert all("where raw_id is not null" not in statement for statement in selects)
    assert all(measurement.succeeded_files == measurement.files > 0 for measurement in measurements)
    assert all(measurement.failed_files == 0 for measurement in measurements)
    assert all(measurement.writer_hold_s is not None and measurement.writer_hold_s > 0 for measurement in measurements)
    assert all(measurement.writer_hold_count > 1 for measurement in measurements)
    assert all(
        measurement.outside_writer_hold_s is not None and measurement.outside_writer_hold_s >= 0
        for measurement in measurements
    )
    assert all(measurement.raw_compaction_runs == 1 for measurement in measurements)
    assert all(
        measurement.raw_compaction_time_s is not None and measurement.raw_compaction_time_s > 0
        for measurement in measurements
    )
