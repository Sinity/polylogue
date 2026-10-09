"""Shared-compute session insights preserve order, rows, and writer custody."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

import polylogue.storage.derived.session.rebuild as rebuild_mod
from polylogue.core.compute import BoundedComputeAdapter, SubmittedOperation
from polylogue.storage.derived.session.rebuild import (
    SessionInsightRecordBundle,
    compute_session_insight_bundles,
    rebuild_session_insights_sync,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.connection_profile import open_connection
from tests.infra.storage_records import SessionBuilder

_BASE_TS = "2026-01-01T00:00:00+00:00"


@pytest.fixture
def insight_adapter(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[int], BoundedComputeAdapter]]:
    adapters: list[BoundedComputeAdapter] = []

    def install(workers: int) -> BoundedComputeAdapter:
        adapter = BoundedComputeAdapter(max_workers=workers, queue_units=8)
        adapters.append(adapter)
        monkeypatch.setattr(rebuild_mod, "compute_adapter", lambda: adapter)
        return adapter

    yield install
    for adapter in adapters:
        adapter.shutdown(wait=True)
        assert adapter.snapshot().used_units == 0


def _seed_corpus(archive_root: Path, *, session_count: int = 6) -> list[str]:
    """Seed a small, fully deterministic multi-session corpus.

    Every timestamp is pinned explicitly (never the builder's wall-clock
    default) so the only clock dependency left in the rebuilt insight rows is
    ``now_iso()``'s ``materialized_at`` stamp, which the ``frozen_clock``
    fixture pins for the equivalence test. Content varies deliberately
    (plain text only / tool_use+tool_result pair / thinking block) so the
    fanned-out compute exercises more than one code path per session.
    """
    initialize_active_archive_root(archive_root)
    db_path = archive_root / "index.db"
    session_ids: list[str] = []
    for index in range(session_count):
        builder = (
            SessionBuilder(db_path, f"fanout-{index}")
            .created_at(_BASE_TS)
            .updated_at(_BASE_TS)
            .add_message(
                "u1",
                role="user",
                text=f"question {index}: how do I run the build pipeline",
                timestamp=_BASE_TS,
            )
            .add_message(
                "a1",
                role="assistant",
                text=f"answer {index}: run devtools test then devtools verify",
                timestamp=_BASE_TS,
            )
        )
        if index % 2 == 0:
            builder = builder.add_message(
                "a2",
                role="assistant",
                text="running the test suite",
                timestamp=_BASE_TS,
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Bash",
                        "tool_id": f"tool-{index}",
                        "input": {"command": "pytest -q"},
                        "semantic_type": "shell",
                    },
                    {
                        "type": "tool_result",
                        "tool_id": f"tool-{index}",
                        "text": "5 passed",
                        "is_error": False,
                        "exit_code": 0,
                    },
                ],
            )
        if index % 3 == 0:
            builder = builder.add_message(
                "a3",
                role="assistant",
                text="thinking it through",
                timestamp=_BASE_TS,
                blocks=[{"type": "thinking", "text": f"considering approach {index}"}],
            )
        builder.save()
        session_ids.append(builder.native_session_id())
    return session_ids


def _dump_table(conn: sqlite3.Connection, table: str, *, order_by: str) -> list[dict[str, object]]:
    rows = conn.execute(f"SELECT * FROM {table} ORDER BY {order_by}").fetchall()
    return [dict(row) for row in rows]


def _rebuild_and_dump(archive_root: Path) -> dict[str, list[dict[str, object]]]:
    with closing(open_connection(archive_root / "index.db")) as conn:
        conn.row_factory = sqlite3.Row
        rebuild_session_insights_sync(conn, session_ids=None)
        return {
            "session_profiles": _dump_table(conn, "session_profiles", order_by="session_id"),
            "session_latency_profiles": _dump_table(conn, "session_latency_profiles", order_by="session_id"),
        }


# ---------------------------------------------------------------------------
# Equivalence: parallel vs sequential fan-out produce byte-identical rows.
# ---------------------------------------------------------------------------


@pytest.mark.frozen_clock_modules("polylogue.storage.derived.session.profiles")
def test_rebuild_session_insights_parallel_vs_sequential_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_clock: object,
    insight_adapter: Callable[[int], BoundedComputeAdapter],
) -> None:
    sequential_root = tmp_path / "sequential"
    parallel_root = tmp_path / "parallel"
    sequential_ids = _seed_corpus(sequential_root, session_count=6)
    parallel_ids = _seed_corpus(parallel_root, session_count=6)
    assert sequential_ids == parallel_ids  # sanity: identical seeded corpora

    insight_adapter(1)
    sequential_dump = _rebuild_and_dump(sequential_root)

    insight_adapter(4)
    parallel_dump = _rebuild_and_dump(parallel_root)

    assert len(sequential_dump["session_profiles"]) == 6
    for table_name, sequential_rows in sequential_dump.items():
        assert parallel_dump[table_name] == sequential_rows, f"{table_name} diverged between fan-out modes"


# ---------------------------------------------------------------------------
# Determinism: job-order results independent of completion order.
# ---------------------------------------------------------------------------


def test_compute_session_insight_bundles_one_worker_preserves_order(
    insight_adapter: Callable[[int], BoundedComputeAdapter],
) -> None:
    insight_adapter(1)
    calling_thread = threading.get_ident()
    seen_threads: list[int] = []

    def make_job(marker: int) -> Callable[[], int]:
        def job() -> int:
            seen_threads.append(threading.get_ident())
            return marker

        return job

    jobs = [make_job(marker) for marker in range(5)]
    results = compute_session_insight_bundles([(job, 64) for job in jobs])

    assert results == [0, 1, 2, 3, 4]
    assert len(set(seen_threads)) == 1
    assert seen_threads[0] != calling_thread


def test_compute_session_insight_bundles_fans_out_and_preserves_order(
    insight_adapter: Callable[[int], BoundedComputeAdapter],
) -> None:
    """Jobs finish in reverse-of-submission order; results still come back
    in submission order, and at least one job actually ran off the calling
    thread (proving the pool engaged, not just an accidental pass-through)."""
    insight_adapter(4)
    calling_thread = threading.get_ident()
    job_threads: list[int] = [-1] * 5

    def make_job(marker: int, *, delay_s: float) -> Callable[[], int]:
        def job() -> int:
            time.sleep(delay_s)
            job_threads[marker] = threading.get_ident()
            return marker

        return job

    # Job 0 sleeps longest, job 4 returns fastest: completion order is the
    # reverse of submission order, so an order bug (e.g. using
    # as_completed()'s own order) would be caught here.
    jobs = [make_job(marker, delay_s=(5 - marker) * 0.02) for marker in range(5)]
    results = compute_session_insight_bundles([(job, 64) for job in jobs])

    assert results == [0, 1, 2, 3, 4]
    assert any(thread_id != calling_thread for thread_id in job_threads), (
        "expected at least one job to run on a pool worker thread, not the calling thread"
    )


# ---------------------------------------------------------------------------
# Write boundary: worker threads compute only; writes stay on the caller.
# ---------------------------------------------------------------------------


def test_rebuild_session_insights_writes_only_happen_on_calling_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    insight_adapter: Callable[[int], BoundedComputeAdapter],
) -> None:
    """Anti-vacuity guard for the single-writer invariant.

    The actual shared adapter runs hydrated compute on worker threads while
    every bulk SQLite publication remains on the caller's writer thread.
    """
    archive_root = tmp_path / "write-boundary"
    _seed_corpus(archive_root, session_count=6)
    adapter = insight_adapter(4)
    admitted_bytes: list[int] = []
    original_submit = adapter.submit

    def recording_submit(function: Callable[[], object], **kwargs: Any) -> SubmittedOperation[object]:
        admitted_bytes.append(int(kwargs["estimated_bytes"]))
        return original_submit(function, **kwargs)

    monkeypatch.setattr(adapter, "submit", recording_submit)

    calling_thread = threading.get_ident()
    compute_threads: list[int] = []
    write_threads: list[int] = []

    original_build = rebuild_mod.build_session_insight_records

    def recording_build(*args: object, **kwargs: object) -> SessionInsightRecordBundle:
        compute_threads.append(threading.get_ident())
        return original_build(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(rebuild_mod, "build_session_insight_records", recording_build)

    write_fn_names = [
        "replace_session_profiles_bulk_sync",
        "replace_session_latency_profiles_bulk_sync",
    ]
    for name in write_fn_names:
        original_write = getattr(rebuild_mod, name)

        def make_recording_write(original: object, sink: list[int]) -> object:
            def recording_write(*args: object, **kwargs: object) -> object:
                sink.append(threading.get_ident())
                return original(*args, **kwargs)  # type: ignore[operator]

            return recording_write

        monkeypatch.setattr(rebuild_mod, name, make_recording_write(original_write, write_threads))

    with closing(open_connection(archive_root / "index.db")) as conn:
        conn.row_factory = sqlite3.Row
        counts = rebuild_session_insights_sync(conn, session_ids=None)

    assert counts.profiles == 6
    assert len(admitted_bytes) == 6
    assert all(weight > 1024 for weight in admitted_bytes)
    assert compute_threads, "expected build_session_insight_records to run at least once"
    assert any(thread_id != calling_thread for thread_id in compute_threads), (
        "fan-out did not engage a pool worker thread -- test would pass vacuously"
    )
    assert write_threads, "expected at least one bulk-write call"
    assert all(thread_id == calling_thread for thread_id in write_threads), (
        "a bulk SQLite writer ran off the calling thread -- single-writer invariant violated"
    )


def test_profile_rebuild_leaves_run_products_to_the_read_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    insight_adapter: Callable[[int], BoundedComputeAdapter],
) -> None:
    """A discarded read-product compilation must not gate profile publication."""
    import polylogue.analysis.transforms as transforms
    from polylogue.storage.derived.session.status import session_insight_status_sync

    archive_root = tmp_path / "profile-products"
    _seed_corpus(archive_root, session_count=3)
    insight_adapter(1)

    def refuse_unowned_projection(*args: object, **kwargs: object) -> object:
        raise AssertionError("profile publication requested a read-side run projection")

    monkeypatch.setattr(transforms, "compile_session_run_projection", refuse_unowned_projection)
    with closing(open_connection(archive_root / "index.db")) as conn:
        conn.row_factory = sqlite3.Row
        counts = rebuild_session_insights_sync(conn, session_ids=None)
        status = session_insight_status_sync(conn)
        assert counts.profiles == 3
        assert conn.execute("SELECT COUNT(*) FROM session_latency_profiles").fetchone()[0] == 3
        assert status.run_count == 3
        assert status.observed_event_count > 0
        assert status.context_snapshot_count >= 3
