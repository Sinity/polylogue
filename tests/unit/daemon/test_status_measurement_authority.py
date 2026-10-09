"""Current policy survives refused measurements at the production status route."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from threading import Event
from typing import cast

import pytest

from polylogue.daemon import status as status_module
from polylogue.daemon.health import DaemonHealth
from polylogue.operations.status_protocol import ComponentSnapshot, ComponentState, StatusComponentRegistry
from polylogue.readiness.capability import CapabilityReadinessState, ComponentReadiness
from tests.infra.embedding_config import embedding_config

_REAL_EMBEDDING_FINGERPRINT = status_module._embedding_status_fingerprint


@pytest.fixture
def measured_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[str, ComponentState, bool, bool], status_module.DaemonStatus]:
    """Compose real status from neutral snapshots without running any collector."""
    values: dict[str, object] = {
        "blob_size": 0,
        "archive_storage": status_module.ArchiveStorageStatus(
            archive_schema_ready=True, archive_materialization_ready=True, archive_ready=True
        ),
        "raw_materialization": status_module.RawMaterializationReadiness(
            available=True,
            raw_artifact_count=4,
            materialized_raw_artifact_count=4,
            archive_session_count=4,
            raw_authority_parser_census={"available": True},
        ),
        "fts_readiness": {
            "messages_ready": True,
            "inspection_state": "fresh",
            "message_indexable_count": 4,
            "message_indexed_count": 4,
        },
        "insight_freshness": {"sessions_with_profiles": 4, "total_sessions": 4, "profile_ready": True},
        "session_summary": ComponentReadiness(
            component="session_summary", state=CapabilityReadinessState.READY, summary="ready"
        ),
        "live_cursor": status_module.LiveCursorSummary(),
        "live_ingest_attempts": status_module.LiveIngestAttemptSummary(),
        "convergence": status_module.ConvergenceDebtSummary(available=False, error="neutral debt unavailable"),
        "cursor_lag": status_module.CursorLagSummary(),
        "blob_publication_reservations": status_module.BlobPublicationReservationStatus(),
        "embedding_readiness": {
            "embedding_config_enabled": True,
            "embedding_has_voyage_key": True,
            "embedding_status": "ready",
            "embedding_freshness_status": "ready",
            "embedding_retrieval_ready": True,
            "embedding_pending_count": 0,
            "embedding_failure_count": 0,
        },
        "configured_source_readiness": {
            name: {"component": name, "state": "ready", "summary": "ready"}
            for name in ("configured_sources", "attachments")
        },
        "health_fast": DaemonHealth(),
        "health_medium": DaemonHealth(),
    }
    monkeypatch.setattr(status_module, "_active_status_db_path", lambda: tmp_path / "index.db")
    monkeypatch.setattr(status_module, "archive_root", lambda: tmp_path)
    monkeypatch.setattr(status_module, "_configured_health_tiers", lambda **_: set())
    monkeypatch.setattr(status_module, "_configured_health_check_interval_s", lambda: 30)
    monkeypatch.setattr(status_module, "browser_capture_status_payload", lambda: {})
    monkeypatch.setattr(status_module, "_daemon_status_fingerprint", lambda *_: "neutral")
    monkeypatch.setattr(status_module, "_embedding_status_fingerprint", lambda **_: "neutral")
    monkeypatch.setattr(status_module, "_configured_source_status_fingerprint", lambda: "neutral")
    monkeypatch.setattr(
        status_module,
        "_raw_frontier_integrity_info",
        lambda *_: status_module.RawFrontierIntegrity(available=True, overall_status="healthy"),
    )
    monkeypatch.setattr(status_module, "catchup_status_info", lambda *_a, **_k: status_module.CatchupStatus())
    monkeypatch.setattr(status_module, "_check_daemon_liveness", lambda *_: False)
    monkeypatch.setattr("polylogue.daemon.lifecycle.lifecycle_status", lambda: {})

    def build(
        component: str,
        state: ComponentState,
        enabled: bool = True,
        incomplete_fts: bool = False,
        *,
        model: str = "voyage-4",
        embedding_registry: StatusComponentRegistry | None = None,
    ) -> status_module.DaemonStatus:
        monkeypatch.setattr(
            "polylogue.config.load_polylogue_config",
            lambda: embedding_config(embedding_enabled=enabled, embedding_model=model),
        )
        if incomplete_fts:
            values["fts_readiness"] = {
                "messages_ready": False,
                "inspection_state": "fresh",
                "message_indexable_count": 4,
                "message_indexed_count": 0,
            }

        class Registry(StatusComponentRegistry):
            def collect(self, *, names: Sequence[str] | None = None) -> dict[str, ComponentSnapshot]:
                assert names is not None
                snapshots = {
                    name: ComponentSnapshot(
                        name=name,
                        scope="neutral",
                        state=state if name == component else "fresh",
                        value=values.get(name, {}),
                        captured_at="2026-01-01T00:00:00+00:00",
                        age_s=0,
                        deadline_s=1,
                        fingerprint="prior" if name == component and state == "stale" else "neutral",
                    )
                    for name in names
                }
                if embedding_registry is not None:
                    snapshots.update(embedding_registry.collect(names=("embedding_readiness",)))
                return snapshots

        return status_module.build_daemon_status(sources=(), registry=Registry([]))

    return build


@pytest.mark.parametrize("state", ["refreshing", "timed_out", "unavailable", "degraded", "stale"])
def test_enabled_embeddings_remain_required_when_registry_refuses_prior_ready(
    measured_status: Callable[..., status_module.DaemonStatus], state: ComponentState
) -> None:
    result = measured_status("embedding_readiness", state)
    embedding = result.embedding_readiness
    assert embedding.embedding_config_enabled is True
    assert embedding.embedding_enabled is True
    assert embedding.embedding_has_voyage_key is True
    assert embedding.embedding_model == "voyage-4"
    assert embedding.embedding_dimension == 1024
    assert embedding.embedding_status == "unknown"
    assert embedding.embedding_unmeasurable_reason == "readiness_unmeasured"
    assert embedding.embedding_pending_count is None
    assert embedding.embedding_failure_count is None
    assert embedding.embedding_retrieval_ready is False
    assert cast(dict[str, dict[str, object]], result.component_readiness)["embeddings"]["state"] == "unknown"
    assert cast(dict[str, dict[str, object]], result.claim_guard)["converged"]["value"] is None
    assert cast(dict[str, dict[str, object]], result.claim_guard)["converged"]["determinate"] is False
    assert (
        cast(dict[str, dict[str, object]], result.claim_guard)["converged"]["signal"]
        == "derived_domain_readiness.embeddings"
    )


@pytest.mark.parametrize("state", ["fresh", "timed_out"])
def test_current_disabled_policy_overrides_prior_enabled_registry_value(
    measured_status: Callable[..., status_module.DaemonStatus], state: ComponentState
) -> None:
    result = measured_status("embedding_readiness", state, False)
    assert result.embedding_readiness.embedding_config_enabled is False
    assert result.embedding_readiness.embedding_enabled is False
    assert cast(dict[str, dict[str, object]], result.claim_guard)["converged"]["value"] is True


def test_measured_enabled_embeddings_can_certify_convergence(
    measured_status: Callable[..., status_module.DaemonStatus],
) -> None:
    result = measured_status("embedding_readiness", "fresh")
    assert result.embedding_readiness.embedding_config_enabled is True
    assert cast(dict[str, dict[str, object]], result.component_readiness)["embeddings"]["state"] == "ready"
    assert cast(dict[str, dict[str, object]], result.claim_guard)["converged"]["value"] is True


@pytest.mark.parametrize("state", ["refreshing", "timed_out", "unavailable", "degraded", "stale"])
def test_refused_fts_measurement_withholds_search_and_convergence(
    measured_status: Callable[..., status_module.DaemonStatus], state: ComponentState
) -> None:
    result = measured_status("fts_readiness", state)
    assert cast(dict[str, dict[str, object]], result.component_readiness)["search"]["state"] == "unknown"
    for claim in ("search_ready", "converged"):
        assert cast(dict[str, dict[str, object]], result.claim_guard)[claim]["value"] is None
        assert cast(dict[str, dict[str, object]], result.claim_guard)[claim]["determinate"] is False
    assert (
        cast(dict[str, dict[str, object]], result.claim_guard)["converged"]["signal"] == "derived_domain_readiness.fts"
    )


def test_measured_incomplete_fts_refutes_search_and_convergence(
    measured_status: Callable[..., status_module.DaemonStatus],
) -> None:
    result = measured_status("fts_readiness", "fresh", True, True)
    assert cast(dict[str, dict[str, object]], result.component_readiness)["search"]["state"] == "missing"
    for claim in ("search_ready", "converged"):
        assert cast(dict[str, dict[str, object]], result.claim_guard)[claim]["value"] is False
        assert cast(dict[str, dict[str, object]], result.claim_guard)[claim]["determinate"] is True


def test_real_registry_recipe_change_invalidates_ready_before_ttl_and_resumes(
    measured_status: Callable[..., status_module.DaemonStatus], monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from polylogue.daemon.embedding_readiness import embedding_readiness_settings

    monkeypatch.setattr(status_module, "_embedding_status_fingerprint", _REAL_EMBEDDING_FINGERPRINT)
    clock = [100.0]
    monkeypatch.setattr("polylogue.operations.status_protocol.monotonic", lambda: clock[0])
    started = Event()
    release = Event()
    finished = Event()
    collected_models: list[str] = []

    def collect_embedding() -> dict[str, object]:
        settings = embedding_readiness_settings()
        collected_models.append(str(settings["embedding_model"]))
        if len(collected_models) > 1:
            started.set()
            assert release.wait(5)
            finished.set()
        return {
            **settings,
            "embedding_status": "ready",
            "embedding_freshness_status": "ready",
            "embedding_retrieval_ready": True,
        }

    spec = next(
        spec
        for spec in status_module._daemon_status_component_specs(
            checked_health=lambda _tiers: DaemonHealth(),
            health_tiers=lambda: set(),
            include_raw_replay_backlog=False,
            include_exact_raw_materialization_readiness=False,
        )
        if spec.name == "embedding_readiness"
    )
    registry = StatusComponentRegistry([replace(spec, collector=collect_embedding)])
    initial = measured_status("embedding_readiness", "fresh", embedding_registry=registry)
    assert cast(dict[str, dict[str, object]], initial.claim_guard)["converged"]["value"] is True
    clock[0] = 105.0
    cached = measured_status("embedding_readiness", "fresh", embedding_registry=registry)
    assert cached.embedding_readiness.embedding_retrieval_ready is True
    assert collected_models == ["voyage-4"]
    try:
        changed = measured_status("embedding_readiness", "fresh", model="voyage-4-lite", embedding_registry=registry)
        assert started.wait(1)
        assert changed.embedding_readiness.embedding_model == "voyage-4-lite"
        assert changed.embedding_readiness.embedding_status == "unknown"
        assert changed.embedding_readiness.embedding_retrieval_ready is False
        assert cast(dict[str, dict[str, object]], changed.claim_guard)["converged"]["value"] is None
    finally:
        release.set()
    assert finished.wait(1)
    resumed = measured_status("embedding_readiness", "fresh", model="voyage-4-lite", embedding_registry=registry)
    assert resumed.embedding_readiness.embedding_model == "voyage-4-lite"
    assert resumed.embedding_readiness.embedding_retrieval_ready is True
    assert cast(dict[str, dict[str, object]], resumed.claim_guard)["converged"]["value"] is True
    assert collected_models == ["voyage-4", "voyage-4-lite"]
