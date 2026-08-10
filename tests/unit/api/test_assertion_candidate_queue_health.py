"""Queue-health contracts for the canonical assertion judgment lifecycle.

The tests exercise the real user/ops tier schemas and the production health
projection.  Anti-vacuity: treating every empty queue as healthy makes the
first test fail; deleting old candidates makes the retention test fail; and
ignoring producer debt makes the stalled-producer test fail.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from polylogue import Polylogue
from polylogue.api.archive import _archive_assertion_candidate_queue_health
from polylogue.config import Config, resolve_runtime_config
from polylogue.core.enums import AssertionKind
from polylogue.daemon import status as status_module
from polylogue.daemon.events import emit_daemon_event
from polylogue.daemon.status import (
    daemon_status_payload,
    periodic_status_component_registry,
    reset_periodic_status_component_registry,
)
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


def test_pending_queue_without_judgment_scheduler_receipt_is_parked(tmp_path: Path) -> None:
    config = _initialize(tmp_path)
    now_ms = 1_800_000_000_000
    with sqlite3.connect(tmp_path / "user.db") as conn:
        upsert_assertion(
            conn,
            assertion_id="candidate-parked",
            target_ref="session:queue-health",
            kind=AssertionKind.LESSON,
            body_text="A pending judgment candidate",
            author_ref="agent:standing-queries",
            author_kind="agent",
            now_ms=now_ms - 25 * _DAY_MS,
        )

    health = _archive_assertion_candidate_queue_health(config, now_ms=now_ms)

    assert health.state == "parked-pending"
    assert health.pending_count == 1
    assert health.judgment_scheduler_receipt_status == "unknown"
    assert any("not converged" in caveat for caveat in health.caveats)


@pytest.mark.parametrize("payload_json", ["not-json", "[]", '{"status":"bogus"}', '{"status":"completed"}'])
def test_malformed_latest_scheduler_receipt_is_unknown_and_keeps_pending_parked(
    tmp_path: Path, payload_json: str
) -> None:
    config = _initialize(tmp_path)
    now_ms = 1_800_000_000_000
    with sqlite3.connect(tmp_path / "user.db") as conn:
        upsert_assertion(
            conn,
            assertion_id="candidate-malformed-receipt",
            target_ref="session:queue-health",
            kind=AssertionKind.LESSON,
            body_text="A pending judgment candidate",
            author_ref="agent:standing-queries",
            author_kind="agent",
            now_ms=now_ms - 25 * _DAY_MS,
        )
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute(
            "INSERT INTO daemon_events (ts_ms, kind, operation_id, payload_json) VALUES (?, ?, ?, ?)",
            (now_ms - 1_000, "judgment-automation", None, payload_json),
        )

    health = _archive_assertion_candidate_queue_health(config, now_ms=now_ms)

    assert health.state == "parked-pending"
    assert health.judgment_scheduler_receipt_status == "unknown"
    assert any("latest judgment scheduler receipt is malformed" in caveat for caveat in health.caveats)


def test_missing_scheduler_interval_authority_is_bounded_as_unavailable(tmp_path: Path) -> None:
    config = SimpleNamespace(archive_root=tmp_path, db_path=tmp_path / "index.db")

    health = _archive_assertion_candidate_queue_health(config, now_ms=1_800_000_000_000)  # type: ignore[arg-type]

    assert health.state == "unavailable"
    assert health.pending_count == 0
    assert any("interval authority is unavailable" in caveat for caveat in health.caveats)


def test_pending_queue_with_fresh_scheduler_receipt_is_active_pending(tmp_path: Path) -> None:
    config = _initialize(tmp_path)
    now_ms = 1_800_000_000_000
    with sqlite3.connect(tmp_path / "user.db") as conn:
        upsert_assertion(
            conn,
            assertion_id="candidate-active",
            target_ref="session:queue-health",
            kind=AssertionKind.LESSON,
            body_text="An actively draining judgment candidate",
            author_ref="agent:standing-queries",
            author_kind="agent",
            now_ms=now_ms - 25 * _DAY_MS,
        )
    emit_daemon_event(
        "judgment-automation",
        archive_root_path=tmp_path,
        observed_at_ms=now_ms - 60_000,
        payload={
            "status": "completed",
            "reason": "sweep_completed",
            "retryable": False,
            "retry_route": "next enabled judgment-automation tick",
            "batch_limit": 200,
            "receipt_persistence_degraded": False,
            "receipt_persistence_recovered": False,
        },
    )

    health = _archive_assertion_candidate_queue_health(config, now_ms=now_ms)

    assert health.state == "pending"
    assert health.judgment_scheduler_receipt_status == "completed"
    assert health.judgment_scheduler_receipt_age_ms == 60_000


def test_typed_scheduler_receipt_is_authoritative_over_newer_legacy_event(tmp_path: Path) -> None:
    config = _initialize(tmp_path)
    now_ms = 1_800_000_000_000
    with sqlite3.connect(tmp_path / "user.db") as conn:
        upsert_assertion(
            conn,
            assertion_id="candidate-typed-authority",
            target_ref="session:queue-health",
            kind=AssertionKind.LESSON,
            body_text="A pending judgment candidate",
            author_ref="agent:standing-queries",
            author_kind="agent",
            now_ms=now_ms - 25 * _DAY_MS,
        )
    emit_daemon_event(
        "judgment-automation",
        archive_root_path=tmp_path,
        operation_id="judgment-automation:typed-authority",
        observed_at_ms=now_ms - 60_000,
        payload={
            "status": "completed",
            "reason": "sweep_completed",
            "retryable": False,
            "retry_route": "next enabled judgment-automation tick",
            "batch_limit": 200,
            "receipt_persistence_degraded": False,
            "receipt_persistence_recovered": False,
        },
    )
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute(
            "INSERT INTO daemon_events (ts_ms, kind, operation_id, payload_json) VALUES (?, ?, ?, ?)",
            (now_ms, "judgment-automation", "legacy-newer", '{"status":"failed","reason":"stale-legacy"}'),
        )

    health = _archive_assertion_candidate_queue_health(config, now_ms=now_ms)

    assert health.state == "pending"
    assert health.judgment_scheduler_receipt_status == "completed"
    assert health.judgment_scheduler_receipt_reason == "sweep_completed"


def test_latest_scheduler_receipt_uses_ledger_order_when_clock_regresses(tmp_path: Path) -> None:
    config = _initialize(tmp_path)
    now_ms = 1_800_000_000_000
    with sqlite3.connect(tmp_path / "user.db") as conn:
        upsert_assertion(
            conn,
            assertion_id="candidate-clock-regression",
            target_ref="session:queue-health",
            kind=AssertionKind.LESSON,
            body_text="A pending judgment candidate",
            author_ref="agent:standing-queries",
            author_kind="agent",
            now_ms=now_ms - 25 * _DAY_MS,
        )
    emit_daemon_event(
        "judgment-automation",
        archive_root_path=tmp_path,
        observed_at_ms=now_ms - 60_000,
        payload={
            "status": "completed",
            "reason": "sweep_completed",
            "retryable": False,
            "retry_route": "next enabled judgment-automation tick",
            "batch_limit": 200,
            "receipt_persistence_degraded": False,
            "receipt_persistence_recovered": False,
        },
    )
    emit_daemon_event(
        "judgment-automation",
        archive_root_path=tmp_path,
        observed_at_ms=now_ms - 120_000,
        payload={
            "status": "failed",
            "reason": "clock_regressed",
            "retryable": True,
            "retry_route": "next enabled judgment-automation tick",
            "batch_limit": 200,
            "receipt_persistence_degraded": False,
            "receipt_persistence_recovered": False,
        },
    )

    health = _archive_assertion_candidate_queue_health(config, now_ms=now_ms)

    assert health.judgment_scheduler_receipt_status == "failed"
    assert health.judgment_scheduler_receipt_at_ms == now_ms - 120_000
    assert health.state == "scheduler-stalled"


def test_scheduler_receipt_freshness_uses_configured_interval_and_bounded_grace(tmp_path: Path) -> None:
    config = _initialize(tmp_path)
    now_ms = 1_800_000_000_000
    interval_s = 48 * 60 * 60
    config.judgment_automation_interval_s = interval_s
    with sqlite3.connect(tmp_path / "user.db") as conn:
        upsert_assertion(
            conn,
            assertion_id="candidate-long-interval",
            target_ref="session:queue-health",
            kind=AssertionKind.LESSON,
            body_text="A pending judgment candidate",
            author_ref="agent:standing-queries",
            author_kind="agent",
            now_ms=now_ms - 25 * _DAY_MS,
        )
    receipt_age_ms = (interval_s + 60 * 60) * 1000
    emit_daemon_event(
        "judgment-automation",
        archive_root_path=tmp_path,
        observed_at_ms=now_ms - receipt_age_ms,
        payload={
            "status": "completed",
            "reason": "sweep_completed",
            "retryable": False,
            "retry_route": "next enabled judgment-automation tick",
            "batch_limit": 200,
            "receipt_persistence_degraded": False,
            "receipt_persistence_recovered": False,
        },
    )

    at_boundary = _archive_assertion_candidate_queue_health(config, now_ms=now_ms)
    just_past_boundary = _archive_assertion_candidate_queue_health(config, now_ms=now_ms + 1)

    assert at_boundary.state == "pending"
    assert at_boundary.judgment_scheduler_receipt_age_ms == receipt_age_ms
    assert just_past_boundary.state == "scheduler-stalled"
    assert just_past_boundary.judgment_scheduler_receipt_age_ms == receipt_age_ms + 1


def test_scheduler_receipt_freshness_uses_explicit_runtime_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interval_s = 48 * 60 * 60
    runtime = resolve_runtime_config(
        cli_overrides={"archive_root": str(tmp_path), "judgment_automation_interval_s": interval_s},
        environment={"POLYLOGUE_SITE_CONFIG": ""},
        cwd=tmp_path,
        home=tmp_path / "home",
    )
    initialize_active_archive_root(runtime.paths.archive_root)
    config = runtime.as_config()
    now_ms = 1_800_000_000_000
    with sqlite3.connect(tmp_path / "user.db") as conn:
        upsert_assertion(
            conn,
            assertion_id="candidate-explicit-runtime",
            target_ref="session:queue-health",
            kind=AssertionKind.LESSON,
            body_text="A pending judgment candidate",
            author_ref="agent:standing-queries",
            author_kind="agent",
            now_ms=now_ms - 25 * _DAY_MS,
        )
    receipt_age_ms = (interval_s + 60 * 60) * 1000
    emit_daemon_event(
        "judgment-automation",
        archive_root_path=tmp_path,
        observed_at_ms=now_ms - receipt_age_ms,
        payload={
            "status": "completed",
            "reason": "sweep_completed",
            "retryable": False,
            "retry_route": "next enabled judgment-automation tick",
            "batch_limit": 200,
            "receipt_persistence_degraded": False,
            "receipt_persistence_recovered": False,
        },
    )

    health = _archive_assertion_candidate_queue_health(config, now_ms=now_ms)

    assert config.judgment_automation_interval_s == interval_s
    assert health.state == "pending"

    calls: list[Config | None] = []

    def queue_health(*, config: Config | None = None) -> dict[str, object]:
        calls.append(config)
        return {
            "state": "explicit-config" if config is not None else "ambient-config",
            "archive_root": str(config.archive_root) if config is not None else "ambient",
        }

    monkeypatch.setattr(status_module, "assertion_candidate_queue_status_summary", queue_health)
    reset_periodic_status_component_registry()
    try:
        registry = periodic_status_component_registry()
        status_payload = daemon_status_payload(
            config=config,
            sources=(),
            include_raw_replay_backlog=False,
            include_exact_raw_materialization_readiness=False,
            include_archive_debt=False,
            registry=registry,
        )
    finally:
        reset_periodic_status_component_registry()
    queue_health_payload = status_payload["assertion_candidate_queue"]
    assert isinstance(queue_health_payload, dict)
    assert queue_health_payload["state"] == "explicit-config"
    assert queue_health_payload["archive_root"] == str(tmp_path)
    assert calls[-1] is config


def test_failed_scheduler_receipt_is_not_converged(tmp_path: Path) -> None:
    config = _initialize(tmp_path)
    now_ms = 1_800_000_000_000
    with sqlite3.connect(tmp_path / "user.db") as conn:
        upsert_assertion(
            conn,
            assertion_id="candidate-failed",
            target_ref="session:queue-health",
            kind=AssertionKind.LESSON,
            body_text="A retryable judgment candidate",
            author_ref="agent:standing-queries",
            author_kind="agent",
            now_ms=now_ms - 25 * _DAY_MS,
        )
    emit_daemon_event(
        "judgment-automation",
        archive_root_path=tmp_path,
        observed_at_ms=now_ms - 60_000,
        payload={
            "status": "failed",
            "reason": "transient_sqlite_lock",
            "retryable": True,
            "retry_route": "next enabled judgment-automation tick",
            "batch_limit": 200,
            "receipt_persistence_degraded": False,
            "receipt_persistence_recovered": False,
        },
    )

    health = _archive_assertion_candidate_queue_health(config, now_ms=now_ms)

    assert health.state == "scheduler-stalled"
    assert health.judgment_scheduler_receipt_status == "failed"
    assert any("retry" in caveat for caveat in health.caveats)


def test_parked_scheduler_receipt_covers_coalesced_tick_before_expiring(tmp_path: Path) -> None:
    """A coalesced parked tick stays truthful until its next write window ends.

    Anti-vacuity: the production dependency is the queue-health projection's
    receipt-age calculation. Removing the parked-receipt freshness branch
    makes this real user/ops ledger fixture remain ``parked-pending`` instead
    of becoming ``scheduler-stalled`` after the cadence horizon.
    """
    config = _initialize(tmp_path)
    now_ms = 1_800_000_000_000
    with sqlite3.connect(tmp_path / "user.db") as conn:
        upsert_assertion(
            conn,
            assertion_id="candidate-expired-parked",
            target_ref="session:queue-health",
            kind=AssertionKind.LESSON,
            body_text="A pending judgment candidate",
            author_ref="agent:standing-queries",
            author_kind="agent",
            now_ms=now_ms - 25 * _DAY_MS,
        )
    grace_s = max(config.judgment_automation_interval_s // 10, 5 * 60)
    cadence_ms = (2 * config.judgment_automation_interval_s + grace_s) * 1000
    emit_daemon_event(
        "judgment-automation",
        archive_root_path=tmp_path,
        observed_at_ms=now_ms - cadence_ms,
        payload={
            "status": "parked",
            "reason": "capability_gate_disabled",
            "retryable": True,
            "retry_route": "next enabled judgment-automation tick",
            "batch_limit": 200,
            "receipt_persistence_degraded": False,
            "receipt_persistence_recovered": False,
        },
    )

    at_boundary = _archive_assertion_candidate_queue_health(config, now_ms=now_ms)
    health = _archive_assertion_candidate_queue_health(config, now_ms=now_ms + 1)

    assert at_boundary.state == "parked-pending"
    assert at_boundary.judgment_scheduler_receipt_age_ms == cadence_ms
    assert health.state == "scheduler-stalled"
    assert health.judgment_scheduler_receipt_status == "parked"
    assert health.judgment_scheduler_receipt_age_ms == cadence_ms + 1
    assert any("fresh parked receipt" in caveat for caveat in health.caveats)


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
