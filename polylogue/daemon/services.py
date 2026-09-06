"""Declared inventory of every background service the daemon spawns.

One spec per spawned task. The composition root may only create a task
through the supervisor, and the supervisor only accepts a name declared
here, so the set of things running inside a daemon process is a value that
can be read, filtered, and asserted on rather than a shape recovered from
reading startup code.

A spec declares what the supervisor needs in order to own the task: who
owns it, which runtime capabilities it presupposes, which siblings must
already be started, how it is triggered, what its failure means for the
rest of the daemon, how long shutdown may wait for it, and which execution
profiles include it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "PRODUCTION_PROFILE",
    "DaemonServiceSpec",
    "FailurePolicy",
    "ServiceCapability",
    "ServiceProfile",
    "ServiceReadiness",
    "ServiceState",
    "ServiceTrigger",
    "UnknownServiceError",
    "capability_tokens",
    "select_service_specs",
    "service_spec",
    "service_specs",
]


class ServiceTrigger(str, Enum):
    """What makes the service do work."""

    PERIODIC = "periodic"
    """A cadence loop; ``cadence_s`` records the nominal interval."""

    EVENT_AND_RECONCILE = "event_and_reconcile"
    """Event-driven with a slow reconciliation heartbeat behind it."""

    SERVER = "server"
    """A socket server; the task exposes only the server's completion."""

    CONTINUOUS = "continuous"
    """A long-lived driver that runs until cancelled."""

    ONESHOT = "oneshot"
    """Runs once and completes; still owned so shutdown can await it."""


class FailurePolicy(str, Enum):
    """What an unhandled service failure means for the rest of the daemon."""

    ISOLATE = "isolate"
    """Record the failure; siblings and the daemon continue."""

    DEGRADE = "degrade"
    """Record the failure and mark the daemon degraded; siblings continue."""

    FAIL_DAEMON = "fail_daemon"
    """The daemon cannot honestly continue without this service."""


class ServiceReadiness(str, Enum):
    """When the service counts as ready."""

    ON_START = "on_start"
    """Ready as soon as its task is scheduled."""

    ON_FIRST_PASS = "on_first_pass"
    """Ready once it publishes its first observation."""

    NEVER_REPORTS = "never_reports"
    """Carries no readiness of its own (servers report through their socket)."""


class ServiceState(str, Enum):
    """Lifecycle state of one declared service in one daemon process."""

    PENDING = "pending"
    RUNNING = "running"
    SKIPPED = "skipped"
    """Not selected: profile excluded it or a capability is absent."""

    UNAVAILABLE = "unavailable"
    """Selected but a prerequisite could not be satisfied at start time."""

    HALTED = "halted"
    """Excluded from scheduling by a durable halt (see ``service_halt``)."""

    FAILED = "failed"
    STOPPED = "stopped"
    ORPHANED = "orphaned"
    """Cancelled but still running when the shutdown deadline expired."""


class ServiceCapability(str, Enum):
    """Runtime facts a service can presuppose.

    A capability is resolved once by the composition root from its own
    arguments and preflight results. Selection consumes the resolved set;
    no service re-derives one for itself.
    """

    WATCH = "watch"
    """Live source watching is enabled for this run."""

    SOURCE_CATCHUP = "source_catchup"
    """Configured (non-filesystem) source catch-up is enabled."""

    BROWSER_CAPTURE = "browser_capture"
    """The browser-capture receiver is enabled."""

    API = "api"
    """The machine API surfaces (TCP + UDS) are enabled."""

    DERIVED_WRITES = "derived_writes"
    """Derived tiers are usable, so index-writing work may be scheduled."""

    SCHEMA_BLOCKED = "schema_blocked"
    """Schema preflight was CRITICAL; recheck work is scheduled instead."""


class ServiceProfile(str, Enum):
    """Named subsets of the one registry.

    A focused test selects a profile; it never assembles a private startup
    chain, so it cannot start a service production does not declare, nor
    miss one production does.
    """

    PRODUCTION = "production"
    """Everything the daemon declares."""

    RESIDENT_CORE = "resident_core"
    """Process liveness and health only: no archive work, no sockets."""

    SURFACES = "surfaces"
    """Resident core plus the socket servers."""

    INTAKE = "intake"
    """Resident core plus acquisition; no derived materialization."""


PRODUCTION_PROFILE = ServiceProfile.PRODUCTION


class UnknownServiceError(LookupError):
    """Raised when a name that is not declared here is started or looked up."""


@dataclass(frozen=True, slots=True)
class DaemonServiceSpec:
    """One declared background service."""

    name: str
    owner: str
    trigger: ServiceTrigger
    failure_policy: FailurePolicy
    requires: frozenset[ServiceCapability] = frozenset()
    """Capabilities that must all be present for the service to be selected."""

    excluded_by: frozenset[ServiceCapability] = frozenset()
    """Capabilities whose presence excludes the service."""

    depends_on: tuple[str, ...] = ()
    """Sibling services that must be started or resolved first."""

    profiles: frozenset[ServiceProfile] = field(default_factory=lambda: frozenset({ServiceProfile.PRODUCTION}))
    readiness: ServiceReadiness = ServiceReadiness.ON_START
    shutdown_deadline_s: float = 5.0
    status_component: str | None = None
    cadence_s: float | None = None

    def selected_for(
        self,
        *,
        profile: ServiceProfile,
        capabilities: frozenset[ServiceCapability],
    ) -> bool:
        """Return whether this spec runs under *profile* with *capabilities*."""
        if profile not in self.profiles:
            return False
        if not self.requires <= capabilities:
            return False
        return not (self.excluded_by & capabilities)


def _spec(
    name: str,
    *,
    owner: str,
    trigger: ServiceTrigger,
    failure_policy: FailurePolicy = FailurePolicy.ISOLATE,
    requires: Iterable[ServiceCapability] = (),
    excluded_by: Iterable[ServiceCapability] = (),
    depends_on: tuple[str, ...] = (),
    profiles: Iterable[ServiceProfile] = (ServiceProfile.PRODUCTION,),
    readiness: ServiceReadiness = ServiceReadiness.ON_START,
    shutdown_deadline_s: float = 5.0,
    status_component: str | None = None,
    cadence_s: float | None = None,
) -> DaemonServiceSpec:
    return DaemonServiceSpec(
        name=name,
        owner=owner,
        trigger=trigger,
        failure_policy=failure_policy,
        requires=frozenset(requires),
        excluded_by=frozenset(excluded_by),
        depends_on=depends_on,
        profiles=frozenset(profiles),
        readiness=readiness,
        shutdown_deadline_s=shutdown_deadline_s,
        status_component=status_component,
        cadence_s=cadence_s,
    )


_ALL_PROFILES = tuple(ServiceProfile)
_RESIDENT = (
    ServiceProfile.PRODUCTION,
    ServiceProfile.RESIDENT_CORE,
    ServiceProfile.SURFACES,
    ServiceProfile.INTAKE,
)
_WITH_SURFACES = (ServiceProfile.PRODUCTION, ServiceProfile.SURFACES)
_WITH_INTAKE = (ServiceProfile.PRODUCTION, ServiceProfile.INTAKE)


_SPECS: tuple[DaemonServiceSpec, ...] = (
    # --- resident core: proves the process is alive even when archive work is withheld
    _spec(
        "lifecycle_heartbeat",
        owner="daemon.lifecycle",
        trigger=ServiceTrigger.PERIODIC,
        profiles=_RESIDENT,
        status_component="daemon_process",
        cadence_s=30.0,
    ),
    _spec(
        "health_check",
        owner="daemon.health",
        trigger=ServiceTrigger.PERIODIC,
        # A daemon that stopped checking its own health is not healthy; it is
        # a daemon with nothing left to report a problem with.
        failure_policy=FailurePolicy.DEGRADE,
        profiles=_RESIDENT,
        status_component="health",
        cadence_s=60.0,
    ),
    _spec(
        "schema_preflight_recheck",
        owner="daemon.schema",
        trigger=ServiceTrigger.PERIODIC,
        requires=(ServiceCapability.SCHEMA_BLOCKED,),
        profiles=_RESIDENT,
        status_component="schema",
        cadence_s=60.0,
    ),
    # --- sockets
    _spec(
        "browser_capture_server",
        owner="daemon.browser_capture",
        trigger=ServiceTrigger.SERVER,
        failure_policy=FailurePolicy.FAIL_DAEMON,
        requires=(ServiceCapability.BROWSER_CAPTURE,),
        profiles=_WITH_SURFACES,
        readiness=ServiceReadiness.NEVER_REPORTS,
        status_component="browser_capture",
    ),
    _spec(
        "api_server",
        owner="daemon.http",
        trigger=ServiceTrigger.SERVER,
        failure_policy=FailurePolicy.FAIL_DAEMON,
        requires=(ServiceCapability.API,),
        profiles=_WITH_SURFACES,
        readiness=ServiceReadiness.NEVER_REPORTS,
        status_component="api",
    ),
    _spec(
        "uds_server",
        owner="daemon.uds",
        trigger=ServiceTrigger.SERVER,
        failure_policy=FailurePolicy.FAIL_DAEMON,
        requires=(ServiceCapability.API,),
        depends_on=("api_server",),
        profiles=_WITH_SURFACES,
        readiness=ServiceReadiness.NEVER_REPORTS,
        status_component="uds",
    ),
    # --- derived convergence: everything below writes index/embedding tiers
    _spec(
        "raw_materialization_convergence",
        owner="daemon.convergence",
        trigger=ServiceTrigger.PERIODIC,
        requires=(ServiceCapability.DERIVED_WRITES,),
        profiles=(ServiceProfile.PRODUCTION,),
        readiness=ServiceReadiness.ON_FIRST_PASS,
        status_component="raw_materialization",
        cadence_s=60.0,
    ),
    _spec(
        "convergence_check",
        owner="daemon.convergence",
        trigger=ServiceTrigger.PERIODIC,
        requires=(ServiceCapability.DERIVED_WRITES,),
        profiles=(ServiceProfile.PRODUCTION,),
        readiness=ServiceReadiness.ON_FIRST_PASS,
        status_component="convergence",
        cadence_s=60.0,
    ),
    _spec(
        "wal_checkpoint",
        owner="daemon.storage",
        trigger=ServiceTrigger.PERIODIC,
        requires=(ServiceCapability.DERIVED_WRITES,),
        profiles=(ServiceProfile.PRODUCTION,),
        cadence_s=300.0,
    ),
    _spec(
        "fts_merge",
        owner="daemon.fts",
        trigger=ServiceTrigger.PERIODIC,
        requires=(ServiceCapability.DERIVED_WRITES,),
        profiles=(ServiceProfile.PRODUCTION,),
        status_component="fts",
        cadence_s=300.0,
    ),
    _spec(
        "heartbeat",
        owner="daemon.metrics",
        trigger=ServiceTrigger.PERIODIC,
        requires=(ServiceCapability.DERIVED_WRITES,),
        profiles=(ServiceProfile.PRODUCTION,),
        cadence_s=60.0,
    ),
    _spec(
        "embedding_backlog",
        owner="daemon.embeddings",
        trigger=ServiceTrigger.PERIODIC,
        requires=(ServiceCapability.DERIVED_WRITES,),
        profiles=(ServiceProfile.PRODUCTION,),
        readiness=ServiceReadiness.ON_FIRST_PASS,
        status_component="embeddings",
        cadence_s=60.0,
    ),
    _spec(
        "embedding_orphan_reconcile",
        owner="daemon.embeddings",
        trigger=ServiceTrigger.PERIODIC,
        requires=(ServiceCapability.DERIVED_WRITES,),
        profiles=(ServiceProfile.PRODUCTION,),
        cadence_s=600.0,
    ),
    _spec(
        "db_optimize",
        owner="daemon.storage",
        trigger=ServiceTrigger.PERIODIC,
        requires=(ServiceCapability.DERIVED_WRITES,),
        profiles=(ServiceProfile.PRODUCTION,),
        cadence_s=3600.0,
    ),
    _spec(
        "status_snapshot_refresh",
        owner="daemon.status",
        trigger=ServiceTrigger.PERIODIC,
        # Every surface reads the snapshot this publishes. Losing it silently
        # would leave stale numbers looking current.
        failure_policy=FailurePolicy.DEGRADE,
        requires=(ServiceCapability.DERIVED_WRITES,),
        profiles=(ServiceProfile.PRODUCTION,),
        readiness=ServiceReadiness.ON_FIRST_PASS,
        status_component="status_snapshot",
        cadence_s=10.0,
    ),
    _spec(
        "judgment_automation",
        owner="daemon.judgment",
        trigger=ServiceTrigger.PERIODIC,
        requires=(ServiceCapability.DERIVED_WRITES,),
        profiles=(ServiceProfile.PRODUCTION,),
        cadence_s=900.0,
    ),
    _spec(
        "blob_gc",
        owner="daemon.blobs",
        trigger=ServiceTrigger.PERIODIC,
        requires=(ServiceCapability.DERIVED_WRITES,),
        profiles=(ServiceProfile.PRODUCTION,),
        cadence_s=3600.0,
    ),
    _spec(
        "blob_publication_reconciliation",
        owner="daemon.blobs",
        trigger=ServiceTrigger.PERIODIC,
        requires=(ServiceCapability.DERIVED_WRITES,),
        profiles=(ServiceProfile.PRODUCTION,),
        cadence_s=3600.0,
    ),
    _spec(
        "secret_scan_sweep",
        owner="daemon.secrets",
        trigger=ServiceTrigger.PERIODIC,
        requires=(ServiceCapability.DERIVED_WRITES,),
        profiles=(ServiceProfile.PRODUCTION,),
        cadence_s=3600.0,
    ),
    _spec(
        "drive_source_catchup",
        owner="daemon.sources",
        trigger=ServiceTrigger.PERIODIC,
        requires=(ServiceCapability.DERIVED_WRITES, ServiceCapability.SOURCE_CATCHUP),
        profiles=_WITH_INTAKE,
        status_component="source_catchup",
        cadence_s=900.0,
    ),
    # --- acquisition
    _spec(
        "catch_up_complete_bridge",
        owner="daemon.watcher",
        trigger=ServiceTrigger.ONESHOT,
        requires=(ServiceCapability.WATCH, ServiceCapability.DERIVED_WRITES),
        depends_on=("watcher",),
        profiles=_WITH_INTAKE,
    ),
    _spec(
        "watcher",
        owner="daemon.watcher",
        trigger=ServiceTrigger.CONTINUOUS,
        failure_policy=FailurePolicy.FAIL_DAEMON,
        requires=(ServiceCapability.WATCH,),
        profiles=_WITH_INTAKE,
        readiness=ServiceReadiness.ON_FIRST_PASS,
        shutdown_deadline_s=5.0,
        status_component="watcher",
    ),
)


_BY_NAME: Mapping[str, DaemonServiceSpec] = {spec.name: spec for spec in _SPECS}

if len(_BY_NAME) != len(_SPECS):  # pragma: no cover - construction invariant
    raise RuntimeError("duplicate daemon service name in the registry")

for _spec_entry in _SPECS:  # pragma: no cover - construction invariant
    for _dependency in _spec_entry.depends_on:
        if _dependency not in _BY_NAME:
            raise RuntimeError(f"daemon service {_spec_entry.name!r} depends on undeclared {_dependency!r}")


def service_specs() -> tuple[DaemonServiceSpec, ...]:
    """Return every declared service, in declaration order."""
    return _SPECS


def service_spec(name: str) -> DaemonServiceSpec:
    """Return the spec named *name*, or raise :class:`UnknownServiceError`."""
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        raise UnknownServiceError(f"no daemon service named {name!r} is declared") from exc


def capability_tokens(capabilities: Iterable[ServiceCapability]) -> frozenset[ServiceCapability]:
    """Normalize an iterable of capabilities into the frozen set selection uses."""
    return frozenset(capabilities)


def select_service_specs(
    *,
    profile: ServiceProfile = PRODUCTION_PROFILE,
    capabilities: Iterable[ServiceCapability] = (),
) -> tuple[DaemonServiceSpec, ...]:
    """Return the specs *profile* runs given *capabilities*, in start order.

    Start order is the topological order of ``depends_on`` with declaration
    order as the tiebreak, so the same inputs always produce the same
    sequence.
    """
    resolved = capability_tokens(capabilities)
    selected = [spec for spec in _SPECS if spec.selected_for(profile=profile, capabilities=resolved)]
    return _dependency_order(selected)


def _dependency_order(specs: list[DaemonServiceSpec]) -> tuple[DaemonServiceSpec, ...]:
    """Order *specs* so every dependency precedes its dependent.

    Dependencies that are not themselves selected are ignored: a service
    whose dependency was excluded by profile or capability still has a
    well-defined position.
    """
    present = {spec.name for spec in specs}
    emitted: set[str] = set()
    ordered: list[DaemonServiceSpec] = []
    remaining = list(specs)
    while remaining:
        progressed = False
        deferred: list[DaemonServiceSpec] = []
        for spec in remaining:
            pending = [dep for dep in spec.depends_on if dep in present and dep not in emitted]
            if pending:
                deferred.append(spec)
                continue
            ordered.append(spec)
            emitted.add(spec.name)
            progressed = True
        if not progressed:
            cycle = ", ".join(sorted(spec.name for spec in deferred))
            raise RuntimeError(f"daemon service dependency cycle: {cycle}")
        remaining = deferred
    return tuple(ordered)
