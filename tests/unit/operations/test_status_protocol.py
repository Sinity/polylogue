"""Tests for the shared budgeted component-snapshot protocol (polylogue-20d.17)."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from time import sleep

import pytest

from polylogue.operations.status_protocol import (
    ComponentUnavailableError,
    StatusComponentRegistry,
    StatusComponentSpec,
    _Attempt,
)


def test_healthy_component_reports_fresh() -> None:
    registry = StatusComponentRegistry(
        [StatusComponentSpec(name="a", scope="test", collector=lambda: 42, deadline_s=1.0)]
    )
    snap = registry.collect()["a"]
    assert snap.state == "fresh"
    assert snap.value == 42
    assert snap.error is None


@pytest.mark.parametrize("old_failure", [False, True])
def test_concurrent_fts_readers_finalize_one_attempt_and_preserve_its_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old_failure: bool
) -> None:
    """Waiters adopt their attempt while older completion cannot roll cache back.

    The tracked event gates both waiters after the original collector has
    completed. A newer refresh is then completed and observed through the
    production FTS reader before the older waiters finalize, exercising
    idempotence, exact-attempt ownership, and cache ordering without sleeps.
    """
    old_started = threading.Event()
    release_old = threading.Event()
    both_waiting = threading.Event()
    both_completed_waits = threading.Event()
    allow_finalize = threading.Event()
    new_started = threading.Event()
    release_new = threading.Event()
    wait_count_lock = threading.Lock()
    waiting_threads: set[int] = set()
    completed_wait_threads: set[int] = set()
    original_event = threading.Event

    class TrackedEvent:
        def __init__(self) -> None:
            self.inner = original_event()

        def set(self) -> None:
            self.inner.set()

        def is_set(self) -> bool:
            return self.inner.is_set()

        def wait(self, timeout: float | None = None) -> bool:
            thread_id = threading.get_ident()
            with wait_count_lock:
                waiting_threads.add(thread_id)
                if len(waiting_threads) == 2:
                    both_waiting.set()
            result = self.inner.wait(timeout)
            if result:
                with wait_count_lock:
                    completed_wait_threads.add(thread_id)
                    if len(completed_wait_threads) == 2:
                        both_completed_waits.set()
                assert allow_finalize.wait(2)
            return result

    calls = 0

    def collect() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            old_started.set()
            assert release_old.wait(2)
            if old_failure:
                raise RuntimeError("collector failed")
            return "old snapshot"
        new_started.set()
        assert release_new.wait(2)
        return "new snapshot"

    class TrackingRegistry(StatusComponentRegistry):
        attempts = 0

        def _start_attempt_locked(self, spec: StatusComponentSpec, fingerprint: str | None) -> _Attempt:
            attempt = super()._start_attempt_locked(spec, fingerprint)
            self.attempts += 1
            if self.attempts == 1:
                # The collector is gated above, so replacing only its signal
                # is safe; production registry state and synchronization are
                # unchanged. This exposes the precise waiter schedule.
                attempt.done = TrackedEvent()  # type: ignore[assignment]
            return attempt

    import polylogue.daemon.fts_status as fts_status

    dbf = tmp_path / f"index-{old_failure}.db"

    def collect_fts(_dbf: Path, *, exact: bool) -> dict[str, object]:
        assert exact is False
        value = collect()
        return {
            "payload": {"probe_value": value, "messages_ready": True, "invariant_ready": True},
            "source_fingerprint": fts_status._fts_readiness_fingerprint(dbf),
        }

    monkeypatch.setattr(fts_status, "StatusComponentRegistry", TrackingRegistry)
    monkeypatch.setattr(fts_status, "_collect_fts_readiness_component", collect_fts)
    registry = fts_status._fts_readiness_registry(dbf)
    results: list[dict[str, object]] = []
    failures: list[BaseException] = []

    def read() -> None:
        try:
            results.append(fts_status.fts_readiness_info(dbf))
        except BaseException as exc:  # retain the thread failure for assertion below
            failures.append(exc)

    first = threading.Thread(target=read)
    second = threading.Thread(target=read)
    first.start()
    assert old_started.wait(2)
    second.start()
    assert both_waiting.wait(2)
    release_old.set()
    assert both_completed_waits.wait(2)
    try:
        registry.request_refresh("fts_readiness")
        assert new_started.wait(2)
        release_new.set()
        assert registry._pending["fts_readiness"].done.wait(2)
        newer = fts_status.fts_readiness_info(dbf)
        assert newer["inspection_state"] == "fresh"
        assert newer["probe_value"] == "new snapshot"
    finally:
        allow_finalize.set()
    first.join(timeout=2)
    second.join(timeout=2)
    assert not first.is_alive() and not second.is_alive()
    assert failures == []
    assert len(results) == 2
    assert results[0] == results[1]
    if old_failure:
        assert results[0]["inspection_state"] == "degraded"
        assert results[0]["messages_ready"] is False
        assert results[0]["message_indexed_count"] is None
    else:
        assert results[0]["inspection_state"] == "fresh"
        assert results[0]["probe_value"] == "old snapshot"
    latest = fts_status.fts_readiness_info(dbf)
    assert latest["inspection_state"] == "fresh"
    assert latest["probe_value"] == "new snapshot"


def test_stalled_component_times_out_without_blocking_healthy_components() -> None:
    """Anti-vacuity: a collector that sleeps past its deadline must not delay a healthy sibling.

    This fails on a synchronous whole-payload collector (the pre-20d.17 shape):
    there, one slow component's cost is paid by every component in the same
    request. Here the two collectors run under independent deadlines and the
    slow one reports ``timed_out`` while the fast one still reports ``fresh``.
    """
    release = threading.Event()
    calls = 0

    def slow() -> str:
        nonlocal calls
        calls += 1
        release.wait(timeout=5.0)
        return "eventually done"

    registry = StatusComponentRegistry(
        [
            StatusComponentSpec(name="slow", scope="test", collector=slow, deadline_s=0.05),
            StatusComponentSpec(name="fast", scope="test", collector=lambda: "ok", deadline_s=1.0),
        ]
    )
    snapshots = registry.collect()
    assert snapshots["slow"].state == "timed_out"
    assert snapshots["slow"].error is not None
    assert snapshots["fast"].state == "fresh"
    assert snapshots["fast"].value == "ok"
    assert calls == 1

    # A second call while the same collector is still stuck reports
    # "refreshing" (a background attempt already in flight) instead of
    # spawning a duplicate thread or blocking again on the same deadline.
    snapshots2 = registry.collect()
    assert snapshots2["slow"].state == "refreshing"
    assert calls == 1

    release.set()
    # Give the background thread a moment to finish and record its result.
    for _ in range(200):
        if registry.last_good("slow") is not None:
            break
        sleep(0.01)
    snapshots3 = registry.collect()
    assert snapshots3["slow"].state == "fresh"
    assert snapshots3["slow"].value == "eventually done"


def test_failing_collector_reports_degraded_with_last_good_evidence() -> None:
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return "first value"
        raise RuntimeError("boom")

    registry = StatusComponentRegistry(
        [StatusComponentSpec(name="c", scope="test", collector=flaky, deadline_s=1.0, ttl_s=0.0)]
    )
    first = registry.collect()["c"]
    assert first.state == "fresh"
    assert first.value == "first value"

    # TTL already expired: this call kicks a background refresh (the failing
    # 2nd attempt) and serves the old value immediately, labeled stale.
    stale = registry.collect()["c"]
    assert stale.state == "stale"
    assert stale.value == "first value"

    for _ in range(200):
        if attempts["n"] >= 2:
            break
        sleep(0.01)

    # The failed background attempt has now finished; the next collect()
    # call finalizes it and reports degraded, retaining last-good evidence
    # instead of hiding the failure behind the previously-served value.
    second = registry.collect()["c"]
    assert second.state == "degraded"
    assert second.error is not None
    assert second.value == "first value"  # last-good evidence retained
    assert second.last_good_at == first.captured_at


def test_unavailable_collector_reports_unavailable_state() -> None:
    def absent() -> str:
        raise ComponentUnavailableError("not configured")

    registry = StatusComponentRegistry([StatusComponentSpec(name="c", scope="test", collector=absent, deadline_s=1.0)])
    snap = registry.collect()["c"]
    assert snap.state == "unavailable"
    assert snap.value is None


def test_ttl_reuse_avoids_recollection() -> None:
    calls = {"n": 0}

    def counted() -> int:
        calls["n"] += 1
        return calls["n"]

    registry = StatusComponentRegistry(
        [StatusComponentSpec(name="c", scope="test", collector=counted, deadline_s=1.0, ttl_s=10.0)]
    )
    first = registry.collect()["c"]
    second = registry.collect()["c"]
    assert first.value == second.value == 1
    assert second.state == "fresh"
    assert calls["n"] == 1


def test_ttl_expiry_serves_stale_value_and_kicks_background_refresh() -> None:
    calls = {"n": 0}
    release = threading.Event()

    def counted() -> int:
        calls["n"] += 1
        if calls["n"] > 1:
            release.wait(timeout=5.0)
        return calls["n"]

    registry = StatusComponentRegistry(
        [StatusComponentSpec(name="c", scope="test", collector=counted, deadline_s=1.0, ttl_s=0.0)]
    )
    first = registry.collect()["c"]
    assert first.state == "fresh"
    assert first.value == 1

    second = registry.collect()["c"]
    assert second.state == "stale"
    assert second.value == 1  # old value served immediately; refresh is in flight
    release.set()


def test_fingerprint_change_forces_refresh_inside_ttl() -> None:
    fingerprint_value = {"v": "v1"}
    calls = {"n": 0}

    def counted() -> int:
        calls["n"] += 1
        return calls["n"]

    registry = StatusComponentRegistry(
        [
            StatusComponentSpec(
                name="c",
                scope="test",
                collector=counted,
                deadline_s=1.0,
                ttl_s=1000.0,
                fingerprint=lambda: fingerprint_value["v"],
            )
        ]
    )
    first = registry.collect()["c"]
    assert first.value == 1

    # fingerprint unchanged, well within TTL -> cached value reused.
    second = registry.collect()["c"]
    assert second.value == 1
    assert calls["n"] == 1

    # fingerprint changes -> a changed source cannot hide behind the TTL.
    fingerprint_value["v"] = "v2"
    third = registry.collect()["c"]
    assert third.value == 1  # stale value served immediately (new value in flight)
    assert third.state == "stale"
    for _ in range(200):
        if calls["n"] >= 2:
            break
        sleep(0.01)
    assert calls["n"] == 2


def test_generation_switch_during_collection_never_relabels_old_result() -> None:
    generation = {"value": "A"}
    started = threading.Event()
    release = threading.Event()

    def collect() -> dict[str, str]:
        observed = generation["value"]
        started.set()
        assert release.wait(2)
        return {"observed_generation": observed}

    registry = StatusComponentRegistry(
        [StatusComponentSpec(name="probe", scope="test", collector=collect, fingerprint=lambda: generation["value"])]
    )
    registry.request_refresh("probe")
    assert started.wait(2)
    generation["value"] = "B"
    release.set()
    assert registry._pending["probe"].done.wait(2)
    old = registry.collect()["probe"]
    assert old.state == "unavailable"
    assert old.value is None
    assert old.fingerprint == "A"
    assert "fingerprint changed" in (old.error or "")
    newer = registry.collect()["probe"]
    assert newer.state == "fresh"
    assert newer.value == {"observed_generation": "B"}
    assert newer.fingerprint == "B"


def test_generation_switch_preserves_old_good_as_advisory() -> None:
    generation = {"value": "A"}
    started = threading.Event()
    release = threading.Event()
    calls = {"count": 0}

    def collect() -> str:
        calls["count"] += 1
        observed = generation["value"]
        if calls["count"] == 2:
            started.set()
            assert release.wait(2)
        return observed

    registry = StatusComponentRegistry(
        [StatusComponentSpec(name="probe", scope="test", collector=collect, fingerprint=lambda: generation["value"])]
    )
    first = registry.collect()["probe"]
    assert first.state == "fresh"
    registry.request_refresh("probe")
    assert started.wait(2)
    generation["value"] = "B"
    registry.request_refresh("probe")
    assert calls["count"] == 2
    release.set()
    assert registry._pending["probe"].done.wait(2)
    stale = registry.collect()["probe"]
    assert stale.state == "stale"
    assert stale.value == "A"
    assert stale.fingerprint == "A"
    assert stale.last_good_at == first.captured_at
    assert "fingerprint changed" in (stale.error or "")


def test_delayed_harvest_keeps_observation_age_and_collection_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    import polylogue.operations.status_protocol as protocol

    clock = {"value": 10.0}
    monkeypatch.setattr(protocol, "monotonic", lambda: clock["value"])
    started = threading.Event()
    release = threading.Event()

    def collect() -> str:
        started.set()
        assert release.wait(2)
        return "measured"

    registry = StatusComponentRegistry([StatusComponentSpec(name="probe", scope="test", collector=collect, ttl_s=5.0)])
    registry.request_refresh("probe")
    assert started.wait(2)
    clock["value"] = 12.0
    release.set()
    assert registry._pending["probe"].done.wait(2)
    completed_at = registry._pending["probe"].completed_at
    clock["value"] = 20.0
    snapshot = registry.collect()["probe"]
    assert snapshot.state == "stale"
    assert snapshot.age_s == 10.0
    assert snapshot.collection_duration_s == 2.0
    assert snapshot.completed_at == completed_at
    assert datetime.fromisoformat(snapshot.captured_at).tzinfo == UTC


def test_fingerprint_failure_does_not_start_or_certify_collection() -> None:
    generation = {"value": "A"}
    calls = {"count": 0}

    def fingerprint() -> str:
        if generation["value"] == "unreadable":
            raise OSError("identity missing")
        return generation["value"]

    def collect() -> str:
        calls["count"] += 1
        return generation["value"]

    registry = StatusComponentRegistry(
        [StatusComponentSpec(name="probe", scope="test", collector=collect, fingerprint=fingerprint)]
    )
    assert registry.collect()["probe"].state == "fresh"
    generation["value"] = "unreadable"
    registry.request_refresh("probe")
    failed = registry.collect()["probe"]
    assert failed.state == "unavailable"
    assert "fingerprint unavailable" in (failed.error or "")
    assert calls["count"] == 1


def test_detail_only_component_excluded_from_default_collection() -> None:
    registry = StatusComponentRegistry(
        [
            StatusComponentSpec(name="cheap", scope="test", collector=lambda: "ok", deadline_s=1.0),
            StatusComponentSpec(name="expensive", scope="test", collector=lambda: "detail", detail_only=True),
        ]
    )
    default = registry.collect()
    assert "cheap" in default
    assert "expensive" not in default

    detail = registry.collect(names=["expensive"])
    assert detail["expensive"].value == "detail"


def test_to_dict_is_json_serializable_shape() -> None:
    registry = StatusComponentRegistry([StatusComponentSpec(name="a", scope="test", collector=lambda: {"n": 1})])
    payload = registry.collect()["a"].to_dict()
    assert payload["component"] == "a"
    assert payload["state"] == "fresh"
    assert set(payload) == {
        "component",
        "scope",
        "state",
        "value",
        "captured_at",
        "age_s",
        "deadline_s",
        "fingerprint",
        "error",
        "last_good_at",
        "completed_at",
        "collection_duration_s",
    }


def test_registry_rejects_duplicate_component_publishers() -> None:
    """Anti-vacuity: silently replacing one declaration loses an observation."""
    specs = [
        StatusComponentSpec(name="same", scope="daemon", collector=lambda: 1),
        StatusComponentSpec(name="same", scope="archive", collector=lambda: 2),
    ]
    try:
        StatusComponentRegistry(specs)
    except ValueError as exc:
        assert "same" in str(exc)
    else:  # pragma: no cover - mutation target
        raise AssertionError("duplicate status component declarations must be rejected")


def test_component_budget_and_identity_metadata_are_validated() -> None:
    with pytest.raises(ValueError, match="deadline_s"):
        StatusComponentSpec(name="bad", scope="test", collector=lambda: None, deadline_s=0)
    with pytest.raises(ValueError, match="ttl_s"):
        StatusComponentSpec(name="bad", scope="test", collector=lambda: None, ttl_s=-1)


def test_collector_failure_diagnostic_is_redacted_on_serialization() -> None:
    """Removing ComponentSnapshot's privacy projection exposes the collector path."""
    from polylogue.operations.status_protocol import StatusComponentRegistry, StatusComponentSpec

    def fail() -> None:
        raise OSError("cannot read '/opt/private space/例.json'")

    registry = StatusComponentRegistry(
        [
            StatusComponentSpec(name="private_failure", scope="archive", collector=fail, deadline_s=1.0),
        ]
    )
    snapshot = registry.collect()["private_failure"]
    assert snapshot.state == "degraded"
    error = snapshot.to_dict()["error"]
    assert isinstance(error, str)
    assert "[redacted]" in error
    assert "private space" not in error
    assert "例.json" not in error
