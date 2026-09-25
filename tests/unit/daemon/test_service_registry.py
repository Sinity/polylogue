"""The declared service registry and its selection rules.

Anti-vacuity: every assertion here is about a value the production
composition root consumes. Deleting a registration, widening a profile, or
dropping a capability requirement changes what
:func:`polylogue.daemon.services.select_service_specs` returns, and one of
these tests goes red.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from polylogue.daemon.services import (
    DaemonServiceSpec,
    FailurePolicy,
    ServiceCapability,
    ServiceProfile,
    ServiceTrigger,
    UnknownServiceError,
    select_service_specs,
    service_spec,
    service_specs,
)
from tests.infra.daemon_service_harness import ServiceHarness

ALL_CAPABILITIES = frozenset(ServiceCapability) - {ServiceCapability.SCHEMA_BLOCKED}


def test_every_spec_has_a_distinct_name_and_a_declared_owner() -> None:
    specs = service_specs()

    assert specs, "the registry must not be empty"
    assert len({spec.name for spec in specs}) == len(specs)
    for spec in specs:
        assert spec.owner, f"{spec.name} declares no owner"
        assert isinstance(spec.trigger, ServiceTrigger)
        assert isinstance(spec.failure_policy, FailurePolicy)
        assert spec.shutdown_deadline_s > 0


def test_periodic_specs_declare_a_cadence() -> None:
    """A cadence loop with no declared interval is a loop nobody can budget."""
    for spec in service_specs():
        if spec.trigger is ServiceTrigger.PERIODIC:
            assert spec.cadence_s is not None, f"{spec.name} is periodic with no cadence"


def test_unknown_name_names_itself() -> None:
    with pytest.raises(UnknownServiceError, match="no_such_service"):
        service_spec("no_such_service")


def test_dependencies_precede_dependents_in_start_order() -> None:
    ordered = select_service_specs(profile=ServiceProfile.PRODUCTION, capabilities=ALL_CAPABILITIES)
    positions = {spec.name: index for index, spec in enumerate(ordered)}

    for spec in ordered:
        for dependency in spec.depends_on:
            if dependency in positions:
                assert positions[dependency] < positions[spec.name], (
                    f"{spec.name} is ordered before its dependency {dependency}"
                )


def test_start_order_is_stable_across_calls() -> None:
    first = select_service_specs(profile=ServiceProfile.PRODUCTION, capabilities=ALL_CAPABILITIES)
    second = select_service_specs(profile=ServiceProfile.PRODUCTION, capabilities=ALL_CAPABILITIES)

    assert [spec.name for spec in first] == [spec.name for spec in second]


def test_a_missing_capability_removes_exactly_its_dependents() -> None:
    with_api = {spec.name for spec in select_service_specs(capabilities=ALL_CAPABILITIES)}
    without_api = {spec.name for spec in select_service_specs(capabilities=ALL_CAPABILITIES - {ServiceCapability.API})}

    assert with_api - without_api == {"api_server", "uds_server"}


def test_derived_writes_gate_every_index_writing_service() -> None:
    """Schema-blocked startup must not schedule derived-tier work."""
    blocked = {
        spec.name
        for spec in select_service_specs(
            capabilities={ServiceCapability.API, ServiceCapability.SCHEMA_BLOCKED},
        )
    }

    assert "raw_observation_convergence" not in blocked
    assert "convergence_check" not in blocked
    assert "status_snapshot_refresh" not in blocked
    # ... while process liveness stays observable precisely then.
    assert {"lifecycle_heartbeat", "health_check", "schema_preflight_recheck"} <= blocked


def test_resident_core_profile_starts_no_archive_work() -> None:
    """The fixture profile that focused daemon tests select.

    It must exclude raw observation convergence and every other archive-writing
    service even when every capability is present -- that exclusion is what
    keeps a focused daemon test from doing a real convergence pass.
    """
    selected = {
        spec.name for spec in select_service_specs(profile=ServiceProfile.RESIDENT_CORE, capabilities=ALL_CAPABILITIES)
    }

    assert selected == {"lifecycle_heartbeat", "health_check"}
    assert "raw_observation_convergence" not in selected


def test_full_profile_resolves_missing_source_prerequisite_with_attributable_state() -> None:
    """A selected raw service with no source tier is explicit and leaves no task.

    Anti-vacuity: bypass ``mark_unavailable`` or lose the reason publication,
    and the declared selected node is left pending or loses its source-tier
    attribution.
    """
    import asyncio

    harness = ServiceHarness(
        profile=ServiceProfile.PRODUCTION,
        capabilities=ALL_CAPABILITIES,
    )
    assert "raw_observation_convergence" in harness.selected_names

    harness.prerequisite_missing("raw_observation_convergence", "source.db is absent")

    assert harness.state("raw_observation_convergence").value == "unavailable"
    observation = harness.supervisor.board.get_or_unavailable("convergence")
    assert observation.state.value == "unavailable"
    assert observation.reason == "source.db is absent"
    assert not [task for task in harness.supervisor.tasks if not task.done()]
    report = asyncio.run(harness.close())
    assert report.clean


def test_socket_servers_fail_the_daemon_and_maintenance_loops_do_not() -> None:
    """A dead listener is not something to survive quietly; a swept sweep is."""
    assert service_spec("api_server").failure_policy is FailurePolicy.FAIL_DAEMON
    assert service_spec("browser_capture_server").failure_policy is FailurePolicy.FAIL_DAEMON
    assert service_spec("watcher").failure_policy is FailurePolicy.FAIL_DAEMON
    assert service_spec("secret_scan_sweep").failure_policy is FailurePolicy.ISOLATE
    assert service_spec("health_check").failure_policy is FailurePolicy.DEGRADE


def test_selected_for_is_the_only_selection_rule() -> None:
    spec = DaemonServiceSpec(
        name="example",
        owner="test",
        trigger=ServiceTrigger.PERIODIC,
        failure_policy=FailurePolicy.ISOLATE,
        requires=frozenset({ServiceCapability.API}),
        excluded_by=frozenset({ServiceCapability.SCHEMA_BLOCKED}),
        profiles=frozenset({ServiceProfile.PRODUCTION}),
        cadence_s=1.0,
    )

    assert spec.selected_for(profile=ServiceProfile.PRODUCTION, capabilities=frozenset({ServiceCapability.API}))
    assert not spec.selected_for(profile=ServiceProfile.RESIDENT_CORE, capabilities=frozenset({ServiceCapability.API}))
    assert not spec.selected_for(profile=ServiceProfile.PRODUCTION, capabilities=frozenset())
    assert not spec.selected_for(
        profile=ServiceProfile.PRODUCTION,
        capabilities=frozenset({ServiceCapability.API, ServiceCapability.SCHEMA_BLOCKED}),
    )


def test_unconfigured_embeddings_deselect_the_ingest_woken_backlog_loop() -> None:
    """A permanently deferred backlog loop must never be planned.

    With embeddings unconfigured, ``compose_embedding_convergence`` returns a
    constant policy deferral for the life of the process, and the backlog loop
    is woken by every committed ingest -- so selecting it buys one identical
    refusal per commit and never a pass that can progress. Orphan reconcile is
    deliberately still selected: stale embedding rows are debt to drain whether
    or not new embedding work can be computed.

    Anti-vacuity: drop ``ServiceCapability.EMBEDDINGS`` from the
    ``embedding_backlog`` spec's ``requires`` and the first assertion fails
    because the loop is selected again. Executed: with the requirement removed,
    ``embedding_backlog`` reappears in ``without_embeddings``.
    """
    without_embeddings = {
        spec.name for spec in select_service_specs(capabilities=ALL_CAPABILITIES - {ServiceCapability.EMBEDDINGS})
    }
    with_embeddings = {spec.name for spec in select_service_specs(capabilities=ALL_CAPABILITIES)}

    assert "embedding_backlog" not in without_embeddings
    assert "embedding_backlog" in with_embeddings
    assert with_embeddings - without_embeddings == {"embedding_backlog"}
    assert "embedding_orphan_reconcile" in without_embeddings


def test_the_capability_predicate_agrees_with_the_composed_callback() -> None:
    """One configuration decides the capability and the composed deferral.

    ``embedding_convergence_unavailable_reason`` lets the composition root
    withhold the capability *before* a task exists, but it is a mirror of a
    decision ``compose_embedding_convergence`` owns. If the two drift, the
    service is scheduled against a callback that can only refuse -- exactly the
    state the capability was added to prevent -- and nothing would say so.

    Anti-vacuity: invert either branch of the predicate and a row below
    disagrees with the composer's permanent deferral. Executed: swapping the
    ``embedding_enabled`` test to ``if bool(...)`` makes the first two rows
    fail.
    """
    import asyncio

    from polylogue.config import PolylogueConfig
    from polylogue.daemon.embedding_backlog import embedding_convergence_unavailable_reason
    from polylogue.daemon.embedding_owner import compose_embedding_convergence
    from tests.infra.embedding_config import embedding_config

    matrix = (
        (False, None, "disabled"),
        (False, "vk-synthetic", "disabled"),
        (True, None, "provider_unavailable"),
    )
    for enabled, key, expected in matrix:
        config = embedding_config(embedding_enabled=enabled, voyage_api_key=key)
        assert embedding_convergence_unavailable_reason(config) == expected

        async def compose(cfg: PolylogueConfig = config) -> str | None:
            with patch("polylogue.config.load_polylogue_config", return_value=cfg):
                composed = compose_embedding_convergence(
                    Path("/nonexistent/archive/index.db"),
                    compute_adapter=cast(Any, None),
                    write_bridge=cast(Any, None),
                )
            return (await composed(None)).deferred_reason

        assert asyncio.run(compose()) == expected, f"composer disagreed for enabled={enabled} key={key!r}"
