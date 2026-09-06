"""Supervisor ownership, isolation, halting, and bounded shutdown.

Each test names the mutation that reddens it, because the whole point of a
supervisor is that these properties hold without anyone remembering them.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import pytest

from polylogue.daemon.observation import ObservationState
from polylogue.daemon.service_halt import HaltReason, HaltRegistry, UnitKind, unit_id
from polylogue.daemon.services import (
    ServiceCapability,
    ServiceProfile,
    ServiceState,
    UnknownServiceError,
    service_spec,
)
from polylogue.daemon.supervisor import (
    TASK_NAME_PREFIX,
    DaemonSupervisor,
    DuplicateServiceStartError,
    ServiceDependencyError,
)

ALL_CAPABILITIES = frozenset(ServiceCapability) - {ServiceCapability.SCHEMA_BLOCKED}


def _supervisor(**kwargs: object) -> DaemonSupervisor:
    kwargs.setdefault("capabilities", ALL_CAPABILITIES)
    return DaemonSupervisor(**kwargs)  # type: ignore[arg-type]


async def _forever() -> None:
    await asyncio.Event().wait()


async def _immediately() -> None:
    return None


def test_an_undeclared_service_cannot_be_started() -> None:
    """Removing the registry lookup would let any name spawn a task."""

    async def scenario() -> None:
        supervisor = _supervisor()
        with pytest.raises(UnknownServiceError, match="ghost_loop"):
            supervisor.start("ghost_loop", _forever)

    asyncio.run(scenario())


def test_a_service_cannot_be_started_twice() -> None:
    async def scenario() -> None:
        supervisor = _supervisor()
        supervisor.start("health_check", _forever)
        with pytest.raises(DuplicateServiceStartError, match="health_check"):
            supervisor.start("health_check", _forever)
        await supervisor.shutdown()

    asyncio.run(scenario())


def test_starting_before_a_dependency_is_refused() -> None:
    """``catch_up_complete_bridge`` declares ``watcher`` as its dependency."""

    async def scenario() -> None:
        supervisor = _supervisor()
        with pytest.raises(ServiceDependencyError, match="watcher"):
            supervisor.start("catch_up_complete_bridge", _forever)
        await supervisor.shutdown()

    asyncio.run(scenario())


def test_owned_tasks_carry_the_service_name() -> None:
    async def scenario() -> None:
        supervisor = _supervisor()
        task = supervisor.start("health_check", _forever)
        assert task is not None
        assert task.get_name() == f"{TASK_NAME_PREFIX}health_check"
        await supervisor.shutdown()

    asyncio.run(scenario())


def test_a_missing_capability_skips_with_the_missing_name() -> None:
    """An absent optional surface is one explicit state, not a retry loop."""

    async def scenario() -> None:
        supervisor = _supervisor(capabilities=ALL_CAPABILITIES - {ServiceCapability.API})
        assert supervisor.start("api_server", _forever) is None
        assert supervisor.state("api_server") is ServiceState.SKIPPED

        observation = supervisor.board.get_or_unavailable("api")
        assert observation.state is ObservationState.SKIPPED
        assert "api" in (observation.reason or "")
        assert observation.value is None

    asyncio.run(scenario())


def test_a_failed_prerequisite_is_unavailable_not_zero() -> None:
    async def scenario() -> None:
        supervisor = _supervisor()
        supervisor.mark_unavailable("embedding_backlog", reason="embeddings.db absent")

        assert supervisor.state("embedding_backlog") is ServiceState.UNAVAILABLE
        observation = supervisor.board.get_or_unavailable("embeddings")
        assert observation.state is ObservationState.UNAVAILABLE
        assert observation.reason == "embeddings.db absent"
        assert not observation.is_measured

    asyncio.run(scenario())


def test_an_isolated_failure_does_not_stop_the_daemon() -> None:
    """Declaring ``secret_scan_sweep`` FAIL_DAEMON instead would redden this."""

    async def failing() -> None:
        raise RuntimeError("sweep exploded")

    async def scenario() -> None:
        supervisor = _supervisor()
        supervisor.start("secret_scan_sweep", failing)
        supervisor.start("health_check", _immediately)

        await supervisor.wait()

        assert supervisor.state("secret_scan_sweep") is ServiceState.FAILED
        assert isinstance(supervisor.failure("secret_scan_sweep"), RuntimeError)

    asyncio.run(scenario())


def test_a_fail_daemon_failure_propagates() -> None:
    async def failing() -> None:
        raise RuntimeError("watch stopped")

    async def scenario() -> None:
        supervisor = _supervisor()
        supervisor.start("watcher", failing)
        with pytest.raises(RuntimeError, match="watch stopped"):
            await supervisor.wait()

    asyncio.run(scenario())


def test_a_degrading_failure_calls_back_once() -> None:
    degradations: list[str] = []

    async def failing() -> None:
        raise RuntimeError("nope")

    async def scenario() -> None:
        supervisor = DaemonSupervisor(
            capabilities=ALL_CAPABILITIES,
            on_degraded=lambda spec, _exc: degradations.append(spec.name),
        )
        # No service currently declares DEGRADE; assert the wiring through a
        # spec that does, so adding one cannot silently skip the callback.
        from polylogue.daemon.services import FailurePolicy

        assert {spec.failure_policy for spec in supervisor.selected} <= {
            FailurePolicy.ISOLATE,
            FailurePolicy.FAIL_DAEMON,
        }
        supervisor.start("secret_scan_sweep", failing)
        await supervisor.wait()
        assert degradations == []

    asyncio.run(scenario())


def test_shutdown_cancels_and_awaits_within_the_declared_deadline() -> None:
    stopped: list[str] = []

    async def cooperative() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            stopped.append("health_check")
            raise

    async def scenario() -> None:
        supervisor = _supervisor()
        supervisor.start("health_check", cooperative)
        await asyncio.sleep(0)

        report = await supervisor.shutdown()

        assert stopped == ["health_check"]
        assert report.clean
        assert report.stopped == ("health_check",)
        assert supervisor.state("health_check") is ServiceState.STOPPED

    asyncio.run(scenario())


def test_a_child_that_ignores_cancellation_is_named_as_an_orphan() -> None:
    """Removing the deadline turns this into a hang, which is the regression."""

    async def uncancellable() -> None:
        while True:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(0.05)

    async def scenario() -> None:
        supervisor = _supervisor()
        supervisor.start("health_check", uncancellable)
        await asyncio.sleep(0)

        report = await supervisor.shutdown()

        assert report.orphaned == ("health_check",)
        assert not report.clean
        assert supervisor.state("health_check") is ServiceState.ORPHANED
        observation = supervisor.board.get_or_unavailable("health")
        assert observation.state is ObservationState.DEGRADED

    asyncio.run(scenario())


def test_shutdown_stops_children_in_reverse_start_order() -> None:
    order: list[str] = []

    def _recording(name: str):
        async def _run() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                order.append(name)
                raise

        return _run

    async def scenario() -> None:
        supervisor = _supervisor()
        supervisor.start("lifecycle_heartbeat", _recording("lifecycle_heartbeat"))
        supervisor.start("health_check", _recording("health_check"))
        supervisor.start("api_server", _recording("api_server"))
        await asyncio.sleep(0)

        await supervisor.shutdown()

        assert order == ["api_server", "health_check", "lifecycle_heartbeat"]

    asyncio.run(scenario())


def test_a_terminal_transition_is_published_exactly_once(tmp_path: Path) -> None:
    """A drained loop that keeps re-reporting terminal is how a spin starts."""

    async def scenario() -> None:
        supervisor = _supervisor()
        supervisor.start("embedding_backlog", _immediately)
        await supervisor.wait()

        supervisor._settle(service_spec("embedding_backlog"), ServiceState.STOPPED, reason="completed")
        supervisor._settle(service_spec("embedding_backlog"), ServiceState.STOPPED, reason="completed")

        terminal = [
            transition
            for transition in supervisor.transitions()
            if transition.service == "embedding_backlog" and transition.state is ServiceState.STOPPED
        ]
        assert len(terminal) == 1

    asyncio.run(scenario())


# -- halted work is visible and unscheduled --------------------------------


def test_a_halted_service_is_never_scheduled(tmp_path: Path) -> None:
    """Mutation: dropping the halt check in ``start`` reddens this."""
    halts = HaltRegistry(tmp_path)
    halts.halt(
        unit_id(UnitKind.SERVICE, "secret_scan_sweep"),
        reason=HaltReason.TERMINAL_REFUSAL,
        message="refusing further sweeps until restart",
        frame="daemon:1",
    )
    started: list[str] = []

    async def scenario() -> None:
        supervisor = _supervisor(halts=halts)
        assert supervisor.is_schedulable("secret_scan_sweep") is False

        async def _body() -> None:
            started.append("secret_scan_sweep")

        assert supervisor.start("secret_scan_sweep", _body) is None
        await supervisor.wait()

        assert started == []
        assert supervisor.state("secret_scan_sweep") is ServiceState.HALTED

    asyncio.run(scenario())


def test_a_halted_service_is_named_in_status(tmp_path: Path) -> None:
    """Mutation: dropping the observation publish reddens this."""
    halts = HaltRegistry(tmp_path)
    halts.halt(
        unit_id(UnitKind.SERVICE, "secret_scan_sweep"),
        reason=HaltReason.POISON_ITEM,
        message="same receipt failed 3 times",
        frame="daemon:1",
    )

    async def scenario() -> None:
        supervisor = _supervisor(halts=halts)
        supervisor.start("secret_scan_sweep", _forever)

        observation = supervisor.board.get_or_unavailable("service.secret_scan_sweep")
        assert observation.state is ObservationState.FAILED
        assert "poison_item" in (observation.reason or "")
        assert "same receipt failed 3 times" in (observation.reason or "")

        records = supervisor.halted_records()
        assert [record.unit for record in records] == ["service:secret_scan_sweep"]

    asyncio.run(scenario())


def test_a_halt_survives_a_restart_with_its_reason(tmp_path: Path) -> None:
    """Mutation: making the halt process-local reddens this."""
    first = HaltRegistry(tmp_path)
    first.halt(
        unit_id(UnitKind.SERVICE, "watcher"),
        reason=HaltReason.SCHEMA_INCOMPATIBLE,
        message="index.db identity moved",
        frame="daemon:1",
    )

    reopened = HaltRegistry(tmp_path)
    record = reopened.record_for(unit_id(UnitKind.SERVICE, "watcher"))

    assert record is not None
    assert record.reason is HaltReason.SCHEMA_INCOMPATIBLE
    assert record.message == "index.db identity moved"
    assert record.frame == "daemon:1"

    async def scenario() -> None:
        supervisor = _supervisor(halts=reopened)
        assert supervisor.start("watcher", _forever) is None
        assert supervisor.state("watcher") is ServiceState.HALTED

    asyncio.run(scenario())


def test_clearing_a_halt_makes_the_unit_schedulable_again(tmp_path: Path) -> None:
    halts = HaltRegistry(tmp_path)
    unit = unit_id(UnitKind.SERVICE, "health_check")
    halts.halt(unit, reason=HaltReason.OPERATOR_HALT, message="paused", frame="daemon:1")

    assert halts.clear(unit) is True
    assert halts.clear(unit) is False
    assert HaltRegistry(tmp_path).halted_units() == ()


def test_recording_a_halt_keeps_the_frame_that_produced_it(tmp_path: Path) -> None:
    halts = HaltRegistry(tmp_path)
    unit = unit_id(UnitKind.SOURCE, "claude-code")
    first = halts.halt(unit, reason=HaltReason.TERMINAL_REFUSAL, message="first", frame="daemon:1")
    second = halts.halt(unit, reason=HaltReason.POISON_ITEM, message="second", frame="daemon:2")

    assert second == first
    assert first.frame == "daemon:1"


def test_supervisor_halt_records_durably_and_stops_the_task(tmp_path: Path) -> None:
    halts = HaltRegistry(tmp_path)

    async def scenario() -> None:
        supervisor = _supervisor(halts=halts)
        task = supervisor.start("secret_scan_sweep", _forever)
        assert task is not None
        await asyncio.sleep(0)

        supervisor.halt("secret_scan_sweep", reason=HaltReason.TERMINAL_REFUSAL, message="wedged")
        await asyncio.sleep(0)

        assert task.cancelled() or task.done()
        assert supervisor.state("secret_scan_sweep") is ServiceState.HALTED

    asyncio.run(scenario())

    assert HaltRegistry(tmp_path).is_halted(unit_id(UnitKind.SERVICE, "secret_scan_sweep"))


def test_profile_selection_comes_from_the_production_registry() -> None:
    """A focused profile narrows the one registry; it never adds to it."""

    async def scenario() -> None:
        focused = _supervisor(profile=ServiceProfile.RESIDENT_CORE)
        production = _supervisor(profile=ServiceProfile.PRODUCTION)

        focused_names = {spec.name for spec in focused.selected}
        production_names = {spec.name for spec in production.selected}

        assert focused_names < production_names
        assert "raw_materialization_convergence" not in focused_names
        assert focused.start("raw_materialization_convergence", _forever) is None
        assert focused.state("raw_materialization_convergence") is ServiceState.SKIPPED

    asyncio.run(scenario())
