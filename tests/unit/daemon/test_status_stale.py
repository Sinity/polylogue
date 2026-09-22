"""A stale status component is a last-good value, not a sixth refusal state.

``_periodic_status_snapshot_refresh`` polls the persistent registry every
``_STATUS_SNAPSHOT_REFRESH_INTERVAL_SECONDS`` plus jitter while its components
carry a 10s ``ttl_s``, so the TTL expires on essentially every tick: the tick
kicks a background refresh and returns ``stale`` *carrying the previous good
value*. Treating ``stale`` as unmeasured therefore discarded a ten-second-old
reading on half of all ticks and published an unknown raw frontier with
``ok: false`` instead (measured as ``fresh, stale, fresh, stale`` at a 10.5s
cadence).

What decides whether that value is a current measurement is the fingerprint it
was collected under, which is exactly the "frame that is no longer
authoritative" concern the blanket rule was standing in for.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue.daemon import status as status_module
from polylogue.daemon.health import DaemonHealth
from polylogue.daemon.status import (
    RawMaterializationReadiness,
    _component_is_unmeasured,
    _daemon_status_component_specs,
    build_daemon_status,
)
from polylogue.operations import status_protocol
from polylogue.operations.status_protocol import ComponentSnapshot, StatusComponentRegistry


def _snapshot(state: str, *, fingerprint: str | None) -> ComponentSnapshot:
    return ComponentSnapshot(
        name="raw_materialization",
        scope="archive",
        state=state,  # type: ignore[arg-type]
        value={"available": True},
        captured_at="2026-09-22T00:00:00+00:00",
        age_s=10.5,
        deadline_s=1.0,
        fingerprint=fingerprint,
    )


def test_stale_with_current_fingerprint_is_measured() -> None:
    assert _component_is_unmeasured(_snapshot("stale", fingerprint="frame-1"), current_fingerprint="frame-1") is False


def test_stale_with_changed_fingerprint_unmeasured() -> None:
    """The frame the value describes is gone; promoting it certifies nothing."""
    assert _component_is_unmeasured(_snapshot("stale", fingerprint="frame-1"), current_fingerprint="frame-2") is True


def test_stale_without_fingerprint_is_unmeasured() -> None:
    """No fingerprint establishes no currency, so the value stays advisory."""
    assert _component_is_unmeasured(_snapshot("stale", fingerprint=None), current_fingerprint="frame-1") is True


@pytest.mark.parametrize("state", ["refreshing", "timed_out", "unavailable", "degraded"])
def test_incomplete_states_stay_unmeasured(state: str) -> None:
    assert _component_is_unmeasured(_snapshot(state, fingerprint="frame-1"), current_fingerprint="frame-1") is True


def test_second_tick_keeps_the_raw_frontier_reading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The production build, over the production registry, across two ticks.

    Anti-vacuity: put ``"stale"`` back into the unmeasured set and the second
    tick reports ``RawMaterializationReadiness(available=False)`` -- the
    unknown-frontier payload -- while holding a good reading whose frame has
    not changed. Removing the fingerprint arm instead is caught by
    ``test_stale_with_changed_fingerprint_unmeasured``.
    """
    db = tmp_path / "index.db"
    reading = RawMaterializationReadiness(available=True, broken_head_count=3)
    clock = [1000.0]
    monkeypatch.setattr(status_protocol, "monotonic", lambda: clock[0])
    monkeypatch.setattr(status_module, "_daemon_status_fingerprint", lambda _db: "frame-1")

    specs = [
        dataclasses.replace(spec, collector=lambda: reading) if spec.name == "raw_materialization" else spec
        for spec in _daemon_status_component_specs(
            checked_health=lambda _tiers: DaemonHealth(),
            health_tiers=set,
            include_raw_replay_backlog=False,
            include_exact_raw_materialization_readiness=False,
            fingerprint=lambda: status_module._daemon_status_fingerprint(db),
        )
    ]
    registry = StatusComponentRegistry(specs)
    # The build's own collect must be the observation under test: a probing
    # ``collect`` here would consume the stale tick itself and leave the build
    # finalising a settled attempt as ``fresh``, which no mutation can fail.
    observed: list[str] = []
    real_collect = registry.collect

    def spy_collect(**kwargs: object) -> dict[str, ComponentSnapshot]:
        result = real_collect(**kwargs)  # type: ignore[arg-type]
        observed.append(result["raw_materialization"].state)
        return result

    monkeypatch.setattr(registry, "collect", spy_collect)

    def _build() -> RawMaterializationReadiness:
        with (
            patch("polylogue.daemon.status._active_status_db_path", return_value=db),
            patch("polylogue.daemon.status._check_daemon_liveness", return_value=False),
            patch("polylogue.daemon.status._blob_size_info", return_value=0),
            patch("polylogue.daemon.status._fts_readiness_info", return_value={}),
            patch("polylogue.daemon.status._insight_freshness_info", return_value={}),
        ):
            return build_daemon_status(sources=(), registry=registry).raw_materialization_readiness

    assert _build() == reading, "the first tick must actually collect, or nothing below is being tested"
    assert observed == ["fresh"]
    # One production polling cadence: 10s interval plus positive jitter, which
    # is past the component's 10s TTL by construction.
    clock[0] += 10.5
    second = _build()
    assert observed[-1] == "stale", "the tick under test must be the stale one"
    assert second == reading
