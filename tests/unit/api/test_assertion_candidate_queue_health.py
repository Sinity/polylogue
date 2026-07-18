"""Queue-health contracts for the canonical assertion judgment lifecycle.

The tests exercise the real user/ops tier schemas and the production health
projection.  Anti-vacuity: treating every empty queue as healthy makes the
first test fail; deleting old candidates makes the retention test fail; and
ignoring producer debt makes the stalled-producer test fail.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue import Polylogue
from polylogue.api.archive import _archive_assertion_candidate_queue_health
from polylogue.config import Config
from polylogue.core.enums import AssertionKind
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.ops_write import (
    add_convergence_debt,
    record_daemon_lifecycle_start,
    record_daemon_stage_event,
)
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion
from polylogue.surfaces.payloads import AssertionCandidateQueueHealthPayload

_DAY_MS = 24 * 60 * 60 * 1000


def _config(root: Path) -> Config:
    return Config(
        archive_root=root,
        render_root=root / "render",
        sources=[],
        db_path=root / "index.db",
    )


def _initialize(root: Path) -> Config:
    initialize_active_archive_root(root)
    return _config(root)


def test_empty_queue_is_unverified_without_producer_and_scheduler_evidence(tmp_path: Path) -> None:
    config = _initialize(tmp_path)

    health = _archive_assertion_candidate_queue_health(config, now_ms=1_800_000_000_000)

    assert health.state == "empty-unverified"
    assert health.pending_count == 0
    assert health.scheduler_state == "unknown"
    assert any("producer" in caveat for caveat in health.caveats)
    assert any("heartbeat" in caveat for caveat in health.caveats)


def test_empty_queue_is_healthy_only_after_fresh_producer_and_heartbeat(tmp_path: Path) -> None:
    config = _initialize(tmp_path)
    now_ms = 1_800_000_000_000
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        record_daemon_stage_event(
            conn,
            stage="standing-queries",
            status="completed",
            observed_at_ms=now_ms - 60_000,
        )
        record_daemon_lifecycle_start(conn, run_id="queue-health", started_at_ms=now_ms - 30_000)

    health = _archive_assertion_candidate_queue_health(config, now_ms=now_ms)

    assert health.state == "healthy-empty"
    assert health.pending_count == 0
    assert health.producer_status == "completed"
    assert health.scheduler_state == "fresh"
    assert health.caveats == ()


def test_old_pending_candidates_remain_durable_and_visible(tmp_path: Path) -> None:
    config = _initialize(tmp_path)
    now_ms = 1_800_000_000_000
    created_at_ms = now_ms - 61 * _DAY_MS
    with sqlite3.connect(tmp_path / "user.db") as conn:
        upsert_assertion(
            conn,
            assertion_id="candidate-old",
            target_ref="session:queue-health",
            kind=AssertionKind.LESSON,
            body_text="A retained old judgment candidate",
            author_ref="agent:standing-queries",
            author_kind="agent",
            status="candidate",
            now_ms=created_at_ms,
        )

    health = _archive_assertion_candidate_queue_health(config, now_ms=now_ms)

    assert health.state == "stale-pending"
    assert health.pending_count == 1
    assert health.stale_pending_count == 1
    assert health.oldest_pending_at_ms == created_at_ms
    assert health.oldest_pending_age_ms == 61 * _DAY_MS
    assert health.retention_outcome == "retained-visible"
    assert health.kind_counts == {"lesson": 1}
    assert health.source_counts == {"agent:agent:standing-queries": 1}
    with sqlite3.connect(tmp_path / "user.db") as conn:
        assert conn.execute("SELECT status FROM assertions WHERE assertion_id = 'candidate-old'").fetchone() == (
            "candidate",
        )


def test_producer_failure_or_debt_overrides_empty_queue(tmp_path: Path) -> None:
    config = _initialize(tmp_path)
    now_ms = 1_800_000_000_000
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        record_daemon_stage_event(
            conn,
            stage="standing-queries",
            status="completed",
            observed_at_ms=now_ms - 60_000,
        )
        record_daemon_lifecycle_start(conn, run_id="queue-health", started_at_ms=now_ms - 30_000)
        add_convergence_debt(
            conn,
            stage="standing-queries",
            target_type="session",
            target_id="session:failed-producer",
            status="failed",
            last_error="candidate capture failed",
            created_at_ms=now_ms - 20_000,
        )

    health = _archive_assertion_candidate_queue_health(config, now_ms=now_ms)

    assert health.state == "producer-stalled"
    assert health.pending_count == 0
    assert health.producer_debt_count == 1


async def test_queue_health_is_queryable_through_polylogue_facade(tmp_path: Path) -> None:
    config = _initialize(tmp_path)
    archive = Polylogue(archive_root=config.archive_root, db_path=config.db_path)
    try:
        health = await archive.assertion_candidate_queue_health()
    finally:
        await archive.close()

    assert isinstance(health, AssertionCandidateQueueHealthPayload)
    assert health.mode == "assertion-candidate-queue-health"
