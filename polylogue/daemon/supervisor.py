"""The sole owner of every task a daemon process spawns.

The composition root hands the supervisor a factory per declared service.
The supervisor decides whether the service runs at all (profile,
capabilities, durable halt), owns the resulting task, applies the declared
failure policy, records every lifecycle transition, and on shutdown cancels
and awaits each child inside its declared deadline, naming whatever is
still running when the deadline expires.

Nothing else in the daemon creates a task. That is what makes the running
process' shape equal to :mod:`polylogue.daemon.services`.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable, Coroutine, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polylogue.daemon.observation import Observation, ObservationBoard, ObservationState
from polylogue.daemon.service_halt import HaltReason, HaltRecord, HaltRegistry, UnitKind, unit_id
from polylogue.daemon.services import (
    PRODUCTION_PROFILE,
    DaemonServiceSpec,
    FailurePolicy,
    ServiceCapability,
    ServiceProfile,
    ServiceState,
    capability_tokens,
    select_service_specs,
    service_spec,
)
from polylogue.logging import DEBUG, WARNING, emit

TERMINAL_SERVICE_STATES: frozenset[ServiceState] = frozenset(
    {ServiceState.HALTED, ServiceState.FAILED, ServiceState.ORPHANED}
)
"""States that record *why* a service stopped, which plain stopping cannot.

A task reaches ``STOPPED`` both when it finished its work and when it was
cancelled -- and cancellation is how a halt, a failure and a shutdown
deadline all end. So every one of these states is followed, one loop turn
later, by the cancelled child settling ``STOPPED`` over it. Letting that
land would turn a durable refusal, a crash and an abandoned task alike into
a clean ``stopped`` on the status board, which is the one thing an operator
reads to decide nothing is wrong. Ordering the set by "which is truer" is
not needed: none of these is reachable from another, so first-writer-wins
is the whole rule.
"""

__all__ = [
    "DaemonSupervisor",
    "DuplicateServiceStartError",
    "ServiceDependencyError",
    "ServiceTransition",
    "ShutdownReport",
    "TASK_NAME_PREFIX",
    "supervised_task_names",
]

TASK_NAME_PREFIX = "daemon.service."
"""Every supervisor-owned task carries this prefix plus the service name."""


class DuplicateServiceStartError(RuntimeError):
    """Raised when one service is started twice in one process."""


class ServiceDependencyError(RuntimeError):
    """Raised when a service starts before a declared dependency is resolved."""


@dataclass(frozen=True, slots=True)
class ServiceTransition:
    """One recorded lifecycle change."""

    service: str
    state: ServiceState
    at: float
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"service": self.service, "state": self.state.value, "at": self.at}
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True, slots=True)
class ShutdownReport:
    """What happened when the supervisor stopped."""

    stopped: tuple[str, ...]
    failed: tuple[str, ...]
    orphaned: tuple[str, ...]
    """Cancelled but still running when their declared deadline expired."""

    duration_s: float

    @property
    def clean(self) -> bool:
        return not self.orphaned and not self.failed

    @property
    def incomplete(self) -> bool:
        """Whether a cancelled child outlived its deadline and is still live.

        Distinct from :attr:`clean`, and the distinction is load-bearing for
        the caller's ownership decision. A ``failed`` service already
        terminated with an exception and cannot commit anything; an
        ``orphaned`` one is running code we stopped waiting for. Only the
        latter means the process must keep the archive lease, the pidfile and
        rebuild exclusion, because only the latter can still write.
        """
        return bool(self.orphaned)

    def as_dict(self) -> dict[str, object]:
        return {
            "stopped": list(self.stopped),
            "failed": list(self.failed),
            "orphaned": list(self.orphaned),
            "duration_s": self.duration_s,
        }


def supervised_task_names(tasks: Iterable[asyncio.Task[Any]]) -> tuple[str, ...]:
    """Return the service names of the supervisor-owned tasks in *tasks*."""
    names = []
    for task in tasks:
        name = task.get_name()
        if name.startswith(TASK_NAME_PREFIX):
            names.append(name[len(TASK_NAME_PREFIX) :])
    return tuple(names)


class DaemonSupervisor:
    """Owns, isolates, reports, and stops the daemon's background services."""

    def __init__(
        self,
        *,
        profile: ServiceProfile = PRODUCTION_PROFILE,
        capabilities: Iterable[ServiceCapability] = (),
        halts: HaltRegistry | None = None,
        board: ObservationBoard | None = None,
        frame: str = "",
        on_degraded: Callable[[DaemonServiceSpec, BaseException], None] | None = None,
    ) -> None:
        self._profile = profile
        self._capabilities = capability_tokens(capabilities)
        self._halts = halts
        self._board = board if board is not None else ObservationBoard()
        self._frame = frame
        self._on_degraded = on_degraded

        self._selected = select_service_specs(profile=profile, capabilities=self._capabilities)
        self._selected_names = {spec.name for spec in self._selected}
        self._states: dict[str, ServiceState] = {}
        self._transitions: list[ServiceTransition] = []
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._failures: dict[str, BaseException] = {}
        self._resolved: set[str] = set()

    # -- inspection ---------------------------------------------------

    @property
    def profile(self) -> ServiceProfile:
        return self._profile

    @property
    def capabilities(self) -> frozenset[ServiceCapability]:
        return self._capabilities

    @property
    def board(self) -> ObservationBoard:
        return self._board

    @property
    def selected(self) -> tuple[DaemonServiceSpec, ...]:
        """Declared services this profile/capability set runs, in start order."""
        return self._selected

    @property
    def tasks(self) -> tuple[asyncio.Task[None], ...]:
        return tuple(self._tasks.values())

    def state(self, name: str) -> ServiceState:
        return self._states.get(name, ServiceState.PENDING)

    def states(self) -> Mapping[str, ServiceState]:
        return dict(self._states)

    def transitions(self) -> tuple[ServiceTransition, ...]:
        return tuple(self._transitions)

    def failure(self, name: str) -> BaseException | None:
        return self._failures.get(name)

    def halted_records(self) -> tuple[HaltRecord, ...]:
        return self._halts.halted_units() if self._halts is not None else ()

    def observations(self) -> Mapping[str, Observation[object]]:
        """Every service's last published lifecycle observation."""
        return self._board.snapshot()

    # -- scheduling ---------------------------------------------------

    def is_schedulable(self, name: str) -> bool:
        """Whether *name* would be started, without starting it.

        Selection is the only place a halt is consulted. Work downstream of
        a halted unit is never planned, so it never reaches an execution
        path that would have to refuse it.
        """
        spec = service_spec(name)
        if name in self._resolved:
            return False
        if spec.name not in self._selected_names:
            return False
        return not self._is_halted(spec)

    def start(
        self,
        name: str,
        factory: Callable[[], Coroutine[Any, Any, None]],
    ) -> asyncio.Task[None] | None:
        """Start the declared service *name*, or resolve it without starting.

        Returns the owned task, or ``None`` when the service is skipped,
        halted, or excluded. An undeclared name raises
        :class:`~polylogue.daemon.services.UnknownServiceError` naming it,
        which is what makes an unregistered spawn impossible rather than
        merely discouraged.
        """
        spec = service_spec(name)
        if name in self._tasks or name in self._resolved:
            raise DuplicateServiceStartError(f"daemon service {name!r} was already started")

        halt = self._halt_record(spec)
        if halt is not None:
            self._resolve(spec, ServiceState.HALTED, reason=f"{halt.reason.value}: {halt.message}")
            return None
        if name not in self._selected_names:
            self._resolve(spec, ServiceState.SKIPPED, reason=self._exclusion_reason(spec))
            return None

        unresolved = [dep for dep in spec.depends_on if dep in self._selected_names and dep not in self._resolved]
        if unresolved:
            raise ServiceDependencyError(
                f"daemon service {name!r} started before its dependencies: {', '.join(sorted(unresolved))}"
            )

        task = asyncio.create_task(self._run(spec, factory()), name=f"{TASK_NAME_PREFIX}{name}")
        self._tasks[name] = task
        self._resolve(spec, ServiceState.RUNNING)
        return task

    def mark_unavailable(self, name: str, *, reason: str) -> None:
        """Resolve *name* as unavailable because a prerequisite failed.

        An absent optional tier or a failed prerequisite becomes one
        explicit state, not a background loop that keeps retrying past the
        point where anything can succeed.
        """
        spec = service_spec(name)
        if name in self._tasks:
            raise DuplicateServiceStartError(f"daemon service {name!r} is already running")
        if name in self._resolved:
            return
        self._resolve(spec, ServiceState.UNAVAILABLE, reason=reason)

    def halt(self, name: str, *, reason: HaltReason, message: str) -> None:
        """Record a durable halt for the service *name*.

        The halt outlives this process, so the next start also refuses to
        schedule it and status keeps naming it.
        """
        spec = service_spec(name)
        if self._halts is None:
            raise RuntimeError("supervisor has no halt registry; cannot record a durable halt")
        record = self._halts.halt(
            unit_id(UnitKind.SERVICE, spec.name),
            reason=reason,
            message=message,
            frame=self._frame,
        )
        task = self._tasks.pop(name, None)
        if task is not None and not task.done():
            task.cancel()
        self._states[name] = ServiceState.HALTED
        self._resolved.add(name)
        self._record(spec.name, ServiceState.HALTED, reason=f"{record.reason.value}: {record.message}")
        self._publish(spec, ServiceState.HALTED, reason=f"{record.reason.value}: {record.message}")

    # -- running ------------------------------------------------------

    async def wait(self) -> None:
        """Await every owned task.

        A service whose policy is ``FAIL_DAEMON`` propagates its failure
        here; ``ISOLATE`` and ``DEGRADE`` failures were already absorbed by
        the wrapper, so a maintenance loop dying does not take the process
        with it.
        """
        if not self._tasks:
            return
        await asyncio.gather(*self._tasks.values())

    async def _run(self, spec: DaemonServiceSpec, coro: Coroutine[Any, Any, None]) -> None:
        self._publish(spec, ServiceState.RUNNING)
        try:
            await coro
        except asyncio.CancelledError:
            self._settle(spec, ServiceState.STOPPED, reason="cancelled")
            raise
        except BaseException as exc:
            self._failures[spec.name] = exc
            self._settle(spec, ServiceState.FAILED, reason=f"{type(exc).__name__}: {exc}")
            if spec.failure_policy is FailurePolicy.FAIL_DAEMON:
                raise
            emit(
                "daemon.service.failed",
                level=WARNING,
                outcome="error",
                reason="service_raised",
                service=spec.name,
                policy=spec.failure_policy.value,
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            if spec.failure_policy is FailurePolicy.DEGRADE and self._on_degraded is not None:
                with contextlib.suppress(Exception):
                    self._on_degraded(spec, exc)
            return
        self._settle(spec, ServiceState.STOPPED, reason="completed")

    # -- shutdown -----------------------------------------------------

    async def shutdown(self) -> ShutdownReport:
        """Cancel every owned task and await it within its declared deadline.

        Tasks stop in reverse start order so a dependent never outlives what
        it depends on. Anything still running when its deadline expires is
        reported by name as an orphan rather than silently abandoned, and the
        report is :attr:`~ShutdownReport.incomplete` -- never ``stopped``. The
        supervisor keeps that child in :data:`ServiceState.ORPHANED` and first-
        writer-wins stops a later cancellation from settling ``stopped`` over
        it, so the composition root's ownership decision reads one honest
        state.
        """
        started = time.monotonic()
        stopped: list[str] = []
        orphaned: list[str] = []
        failed: list[str] = []

        for name in reversed(list(self._tasks)):
            task = self._tasks[name]
            spec = service_spec(name)
            if not task.done():
                task.cancel()
            _done, pending = await asyncio.wait({task}, timeout=spec.shutdown_deadline_s)
            if pending:
                orphaned.append(name)
                self._states[name] = ServiceState.ORPHANED
                self._record(name, ServiceState.ORPHANED, reason=f"deadline {spec.shutdown_deadline_s:g}s expired")
                self._publish(spec, ServiceState.ORPHANED, reason="shutdown deadline expired")
                emit(
                    "daemon.service.orphaned",
                    level=WARNING,
                    outcome="degraded",
                    reason="shutdown_deadline_expired",
                    service=name,
                    state=ServiceState.ORPHANED.value,
                    timeout_ms=round(spec.shutdown_deadline_s * 1000, 3),
                )
                continue
            if self._states.get(name) is ServiceState.FAILED:
                failed.append(name)
            else:
                stopped.append(name)

        return ShutdownReport(
            stopped=tuple(stopped),
            failed=tuple(failed),
            orphaned=tuple(orphaned),
            duration_s=time.monotonic() - started,
        )

    # -- internals ----------------------------------------------------

    def _is_halted(self, spec: DaemonServiceSpec) -> bool:
        return self._halt_record(spec) is not None

    def _halt_record(self, spec: DaemonServiceSpec) -> HaltRecord | None:
        if self._halts is None:
            return None
        return self._halts.record_for(unit_id(UnitKind.SERVICE, spec.name))

    def _exclusion_reason(self, spec: DaemonServiceSpec) -> str:
        if self._profile not in spec.profiles:
            return f"profile {self._profile.value} does not include this service"
        missing = sorted(capability.value for capability in spec.requires - self._capabilities)
        if missing:
            return f"missing capability: {', '.join(missing)}"
        excluded = sorted(capability.value for capability in spec.excluded_by & self._capabilities)
        if excluded:
            return f"excluded by capability: {', '.join(excluded)}"
        return "not selected"

    def _resolve(self, spec: DaemonServiceSpec, state: ServiceState, *, reason: str | None = None) -> None:
        self._states[spec.name] = state
        self._resolved.add(spec.name)
        self._record(spec.name, state, reason=reason)
        if state is not ServiceState.RUNNING:
            self._publish(spec, state, reason=reason)

    def _settle(self, spec: DaemonServiceSpec, state: ServiceState, *, reason: str | None) -> None:
        """Record a terminal state exactly once.

        A drained loop that keeps re-reporting the same terminal state is
        how an empty backlog turns into a spin; the dedup lives here so no
        producer has to remember it.
        """
        current = self._states.get(spec.name)
        if current is state:
            return
        if current in TERMINAL_SERVICE_STATES:
            # The child is settling its cancellation; the state that caused
            # that cancellation is the honest one. See TERMINAL_SERVICE_STATES.
            emit(
                "daemon.service.demotion_refused",
                level=DEBUG,
                outcome="skipped",
                reason="already_terminal",
                service=spec.name,
                state=current.value,
                phase=state.value,
                error_detail=reason,
            )
            return
        self._states[spec.name] = state
        self._record(spec.name, state, reason=reason)
        self._publish(spec, state, reason=reason)

    def _record(self, name: str, state: ServiceState, *, reason: str | None = None) -> None:
        previous = self._transitions[-1] if self._transitions else None
        if previous is not None and previous.service == name and previous.state is state and previous.reason == reason:
            return
        self._transitions.append(ServiceTransition(service=name, state=state, at=time.time(), reason=reason))

    def _publish(self, spec: DaemonServiceSpec, state: ServiceState, *, reason: str | None = None) -> None:
        component = spec.status_component or f"service.{spec.name}"
        if state in (ServiceState.RUNNING, ServiceState.STOPPED):
            observation: Observation[object] = Observation.measured(
                component,
                state.value,
                frame=self._frame or None,
            )
        else:
            observation = Observation.unmeasured(
                component,
                _OBSERVATION_STATE_FOR[state],
                reason=reason or state.value,
                frame=self._frame or None,
            )
        self._board.publish(observation)


_OBSERVATION_STATE_FOR: Mapping[ServiceState, ObservationState] = {
    ServiceState.PENDING: ObservationState.UNAVAILABLE,
    ServiceState.SKIPPED: ObservationState.SKIPPED,
    ServiceState.UNAVAILABLE: ObservationState.UNAVAILABLE,
    ServiceState.HALTED: ObservationState.FAILED,
    ServiceState.FAILED: ObservationState.FAILED,
    ServiceState.ORPHANED: ObservationState.DEGRADED,
    ServiceState.RUNNING: ObservationState.MEASURED,
    ServiceState.STOPPED: ObservationState.MEASURED,
}


def halt_registry_for(archive_root: Path | str) -> HaltRegistry:
    """Return the durable halt registry for *archive_root*."""
    return HaltRegistry(archive_root)
