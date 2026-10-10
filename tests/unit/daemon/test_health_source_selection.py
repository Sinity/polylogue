"""Rich status health observes the daemon's selected watch sources."""

from pathlib import Path
from typing import Any

import pytest

from polylogue.daemon import status as status_module
from polylogue.daemon.health import DaemonHealth, HealthSeverity, HealthTier
from polylogue.daemon.status_snapshot import configure_runtime_components, refresh_status_snapshot
from polylogue.sources.live import WatchSource


@pytest.mark.parametrize("selected_count", [0, 1])
def test_rich_status_health_checks_only_explicit_sources(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selected_count: int
) -> None:
    conventional = (WatchSource(name="conventional", root=tmp_path / "omitted-missing-source"),)
    selected_root = tmp_path / "selected"
    selected_root.mkdir()
    selected = (WatchSource(name="selected", root=selected_root),) if selected_count else ()
    monkeypatch.setattr(status_module, "default_sources", lambda: conventional)
    monkeypatch.setattr("polylogue.sources.live.watcher.default_sources", lambda: conventional)
    monkeypatch.setattr(status_module, "_configured_health_tiers", lambda **_kwargs: {HealthTier.FAST})

    status = status_module.build_daemon_status(
        sources=selected, include_raw_replay_backlog=False, include_exact_raw_materialization_readiness=False
    )
    alert = next(alert for alert in status.health.alerts if alert.check_name == "source_availability")
    assert alert.severity == HealthSeverity.OK, alert.message
    assert alert.message == f"{selected_count} source(s) available"
    assert [item.name for item in status.source_lag] == [source.name for source in selected]


def test_rich_status_health_warns_for_missing_selected_source(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = (WatchSource(name="selected-missing", root=tmp_path / "missing"),)
    monkeypatch.setattr(status_module, "_configured_health_tiers", lambda **_kwargs: {HealthTier.FAST})
    status = status_module.build_daemon_status(
        sources=selected, include_raw_replay_backlog=False, include_exact_raw_materialization_readiness=False
    )
    alert = next(alert for alert in status.health.alerts if alert.check_name == "source_availability")
    assert alert.severity == HealthSeverity.WARNING
    assert alert.message == "0/1 source(s) available, missing: selected-missing"


def test_periodic_health_registry_cannot_reuse_another_source_selection(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected_root = tmp_path / "selected"
    selected_root.mkdir()
    selected = (WatchSource(name="selected", root=selected_root),)
    missing = (WatchSource(name="selected", root=tmp_path / "different-missing-root"),)
    monkeypatch.setattr(status_module, "_configured_health_tiers", lambda **_kwargs: {HealthTier.FAST})
    registry = status_module.periodic_status_component_registry(sources=selected)
    assert status_module.periodic_status_component_registry(sources=selected) is registry
    assert status_module.periodic_status_component_registry(sources=missing) is not registry
    for sources, expected in (
        (selected, HealthSeverity.OK),
        ((), HealthSeverity.OK),
        (missing, HealthSeverity.WARNING),
    ):
        status = status_module.build_daemon_status(
            sources=sources,
            include_raw_replay_backlog=False,
            include_exact_raw_materialization_readiness=False,
            registry=registry,
        )
        alert = next(alert for alert in status.health.alerts if alert.check_name == "source_availability")
        assert alert.severity == expected
        assert "different-missing-root" not in alert.message
        if sources == ():
            assert alert.message == "0 source(s) available"


def test_periodic_fast_health_fingerprint_tracks_selected_root_availability(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "selected"
    root.mkdir()
    selected = (WatchSource(name="selected", root=root),)
    monkeypatch.setattr(status_module, "_configured_health_tiers", lambda **_kwargs: {HealthTier.FAST})
    monkeypatch.setattr(status_module, "_daemon_status_fingerprint", lambda _db: "stable-archive")
    registry = status_module.periodic_status_component_registry(sources=selected)
    spec = next(spec for spec in registry.specs if spec.name == "health_fast")
    assert spec.fingerprint is not None
    before = spec.fingerprint()
    first = registry.collect(names=["health_fast"])["health_fast"]
    assert first.state == "fresh"
    assert isinstance(first.value, DaemonHealth)
    alert = next(alert for alert in first.value.alerts if alert.check_name == "source_availability")
    assert alert.severity == HealthSeverity.OK
    root.rmdir()
    assert spec.fingerprint() != before
    changed = registry.collect(names=["health_fast"])["health_fast"]
    assert changed.state == "stale"
    attempt = registry._pending["health_fast"]
    assert attempt.thread is not None
    attempt.thread.join()
    refreshed = registry.collect(names=["health_fast"])["health_fast"]
    assert refreshed.state == "fresh"
    assert isinstance(refreshed.value, DaemonHealth)
    alert = next(alert for alert in refreshed.value.alerts if alert.check_name == "source_availability")
    assert alert.severity == HealthSeverity.WARNING


@pytest.mark.parametrize("selected_count", [0, 1])
def test_periodic_rich_refresh_uses_runtime_source_selection(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selected_count: int
) -> None:
    selected_root = tmp_path / "selected"
    selected_root.mkdir()
    selected = (WatchSource(name="selected", root=selected_root),) if selected_count else ()
    conventional = (WatchSource(name="conventional", root=tmp_path / "omitted-missing"),)
    monkeypatch.setattr(status_module, "default_sources", lambda: conventional)
    monkeypatch.setattr("polylogue.sources.live.watcher.default_sources", lambda: conventional)
    monkeypatch.setattr(status_module, "_configured_health_tiers", lambda **_kwargs: {HealthTier.FAST})
    configure_runtime_components(watcher_enabled=bool(selected), watch_sources=selected)
    original_builder = status_module.build_daemon_status
    observed: list[status_module.DaemonStatus] = []

    def observe_status(**kwargs: Any) -> status_module.DaemonStatus:
        status = original_builder(**kwargs)
        observed.append(status)
        return status

    monkeypatch.setattr(status_module, "build_daemon_status", observe_status)
    payload = refresh_status_snapshot().payload
    assert len(observed) == 1
    health = observed[0].health
    alert = next(alert for alert in health.alerts if alert.check_name == "source_availability")
    assert alert.severity == HealthSeverity.OK, alert.message
    assert alert.message == f"{selected_count} source(s) available"
    assert payload["watcher_roots"] == [str(source.root) for source in selected]
    health_payload = payload["health"]
    assert isinstance(health_payload, dict)
    assert health_payload["checks_run"] == len(health.alerts)
