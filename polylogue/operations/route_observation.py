"""Bounded route-latency observation (polylogue-jtwu / polylogue-20d.17 AC #4).

Covers the routes ``mcp_call_log`` (whole MCP tool calls, durably delivered
via an outbox) does not: CLI command invocations and MCP sub-route detail a
caller wants to time without routing through it. Best-effort telemetry, not
audit evidence -- a caller that cannot reach ``ops.db`` (no archive
configured, disposable tier missing, locked) drops the observation rather
than blocking or retrying the operation being observed.

:class:`RouteObservationSpec` and :class:`RouteObservationReceipt` are the one
declared contract every latency product derives from
(polylogue-jtwu.1); :meth:`RouteObservationReceipt.to_workload_receipt` is the
latency-focused adapter onto ``WorkloadReceipt``. A second surface that wants
route latency declares a spec here rather than inventing its own identifiers,
which is the fragmentation the parent bead exists to stop.

Dropped observations are counted, not merely logged (polylogue-jtwu.2). A
percentile over a sample that silently lost an unknown number of members is
not a measurement of the route, so :func:`compute_latency_percentiles` cannot
be called without stating the drop disposition and returns a
:class:`RouteLatencyReport` that carries it beside the p50/p95. Every drop
is recorded in the ops tier's ``route_observation_drops`` -- at once when the
observation's own write can carry it, otherwise by the process's next
successful write to that tier or its exit flush -- so a reader in any process
counts them. A drop that no write can record before the process exits is
reported as a typed ``route_observation.drops_unflushed`` event instead.
"""

from __future__ import annotations

import sqlite3
import subprocess
import time
import uuid
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, get_args

from polylogue.core.types import (
    RouteDaemonPath,
    RouteObservationDropReasonToken,
    RouteObservationStatus,
    require_literal,
)
from polylogue.logging import get_logger

if TYPE_CHECKING:
    from polylogue.scenarios.workload import WorkloadEnvelopeSpec, WorkloadReceipt, WorkloadRunStatus
    from polylogue.storage.sqlite.archive_tiers.ops_write import RouteObservationDropRow
    from polylogue.surfaces.outcome import OutcomeEnvelope

logger = get_logger(__name__)


def _recordable_daemon_path(value: str | None) -> str | None:
    """Map the CLI's unreachable marker to no selected runtime path."""
    if value is None or value == "unreachable":
        return None
    return require_literal(value, RouteDaemonPath, name="route daemon path")


_CONNECT_TIMEOUT_S = 2.0
#: ``ops.db`` is the disposable tier and a route observation is explicitly
#: best-effort telemetry, not audit evidence. The default ``synchronous=FULL``
#: made every observation pay a synchronous fsync on the hot path of the route
#: it was measuring -- 12.88 ms per insert+commit measured against a real
#: ops.db (polylogue-5lfcr), which is the opposite of this writer's declared
#: intent and is three orders of magnitude above the pruning work it does per
#: call. ``OFF`` keeps the row committed and immediately visible to every
#: reader of the file; what it drops is the durability guarantee across a host
#: crash, for a tier whose whole contract is that it can be discarded.
_OBSERVATION_SYNCHRONOUS = "OFF"
_GIT_HEAD_TIMEOUT_S = 1.0
LOW_CONFIDENCE_SAMPLE_FLOOR = 5

#: Phase every route observation records, whatever else it declares.
DEFAULT_ROUTE_PHASE = "total"

#: Family every route-observation workload spec belongs to, so receipts from
#: different surfaces land in one family rather than one per instrumented call
#: site.
ROUTE_OBSERVATION_WORKLOAD_FAMILY = "route-observation"

#: Reserved key under which the receipt projection rides in the ops-tier row's
#: freeform ``attributes`` document.
RECEIPT_ATTRIBUTE_KEY = "route_receipt"


# ---------------------------------------------------------------------------
# Drop accounting (polylogue-jtwu.2)
# ---------------------------------------------------------------------------


DROP_ACCOUNTING_INCOMPLETE = "drop_accounting_incomplete"
"""Outcome gap: the sample's lost observations are not countable from the reader."""

OBSERVATIONS_DROPPED = "observations_dropped"
"""Outcome gap: counted observations were lost, so some percentile covers part of its route."""

WINDOW_EXCEEDS_RETENTION = "window_exceeds_retention"
"""Outcome gap: the lookback window reaches past the retained telemetry horizon."""


class RouteObservationDropReason(str, Enum):
    """Why an observation never reached the sample a percentile is computed over."""

    NO_ARCHIVE_ROOT = "no_archive_root"
    """The observed caller had no archive configured at all."""

    OPS_DB_MISSING = "ops_db_missing"
    """The disposable tier that receives observations does not exist."""

    EMIT_FAILED = "emit_failed"
    """The write raised -- typically a locked ops.db -- and was swallowed."""

    PRUNED = "pruned"
    """The row landed but the writer's retention/row-cap prune removed rows in
    the same transaction. The parent bead calls this out: rows were pruned by
    retention and row cap without being counted or reported."""

    NOT_SAMPLED = "not_sampled"
    """The spec's sampling disposition excluded this invocation."""


# One vocabulary: the storage boundary validates reasons against the Literal.
if {reason.value for reason in RouteObservationDropReason} != set(get_args(RouteObservationDropReasonToken)):
    raise RuntimeError("RouteObservationDropReason and RouteObservationDropReasonToken must name the same reasons")


@dataclass(frozen=True, slots=True)
class RouteObservationDrops:
    """Observations missing from a latency sample, and how well that loss is known.

    ``accounting_complete`` is the honesty bit: zero drops and unknown drops
    are different answers and a percentile must not present the second as the
    first. The ops-tier reader counts ``route_observation_drops``, where every
    process records its drops, so its answer is complete; a process-local
    snapshot of unrecorded drops is not.
    """

    accounting_complete: bool
    by_reason: Mapping[str, int] = field(default_factory=dict)
    by_route: Mapping[str, int] = field(default_factory=dict)
    """Drops attributed to one ``"<surface>\\t<route>"`` key, for the buckets
    that carry them beside their own percentiles."""

    @property
    def total(self) -> int:
        return sum(self.by_reason.values())

    def attributed_to(self, surface: str, route: str) -> int:
        """Return drops known to belong to one (surface, route) group."""
        return self.by_route.get(route_key(surface, route), 0)

    @property
    def unattributed(self) -> int:
        """Drops that are real but cannot be charged to a single route group."""
        return max(0, self.total - sum(self.by_route.values()))

    def merged_with(self, other: RouteObservationDrops) -> RouteObservationDrops:
        """Combine two drop accounts; completeness is the conjunction."""
        reasons: dict[str, int] = dict(self.by_reason)
        for reason, count in other.by_reason.items():
            reasons[reason] = reasons.get(reason, 0) + count
        routes: dict[str, int] = dict(self.by_route)
        for key, count in other.by_route.items():
            routes[key] = routes.get(key, 0) + count
        return RouteObservationDrops(
            accounting_complete=self.accounting_complete and other.accounting_complete,
            by_reason=reasons,
            by_route=routes,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "total": self.total,
            "accounting_complete": self.accounting_complete,
            "by_reason": dict(sorted(self.by_reason.items())),
            "by_route": dict(sorted(self.by_route.items())),
            "unattributed": self.unattributed,
        }

    @classmethod
    def none_observed(cls) -> RouteObservationDrops:
        """No drops, and the caller can account for every producer of the sample.

        Only correct when the caller built the whole sample itself -- a test, a
        benchmark, or an in-memory aggregation.
        """
        return cls(accounting_complete=True)

    @classmethod
    def unaccounted(cls) -> RouteObservationDrops:
        """The loss is real-valued but not knowable from here."""
        return cls(accounting_complete=False)


def route_key(surface: str, route: str) -> str:
    """Return the attribution key for one (surface, route) group."""
    return f"{surface}\t{route}"


class RouteObservationDropLedger:
    """Drops this process counted and has not yet recorded in the ops tier.

    The dominant drop causes are "cannot reach ops.db" and "ops.db is locked",
    so a drop cannot always be written when it happens. It is held here, keyed
    by the ops tier it belongs to, and written into ``route_observation_drops``
    by that tier's next successful observation or by the process's exit flush
    (:func:`flush_route_observation_drops`). A drop with no archive at all
    belongs to no archive's sample and stays process-local.
    """

    __slots__ = ("_pending",)

    def __init__(self) -> None:
        # Aggregated per (tier, reason, route): bounded by the declared
        # reasons and routes, not by lifetime request volume.
        self._pending: dict[tuple[Path | None, str, str, str], list[int]] = {}

    def record(
        self,
        reason: RouteObservationDropReason,
        *,
        surface: str,
        route: str,
        ops_db: Path | None,
        observed_at_ms: int,
        count: int = 1,
    ) -> None:
        if count <= 0:
            return
        key = (ops_db, reason.value, surface, route)
        entry = self._pending.get(key)
        if entry is None:
            self._pending[key] = [count, observed_at_ms, observed_at_ms]
            _register_exit_flush()
            return
        entry[0] += count
        entry[1] = min(entry[1], observed_at_ms)
        entry[2] = max(entry[2], observed_at_ms)

    def drain(self, ops_db: Path) -> tuple[RouteObservationDropRow, ...]:
        """Remove and return the pending drops that belong to ``ops_db``."""
        from polylogue.storage.sqlite.archive_tiers.ops_write import RouteObservationDropRow

        rows: list[RouteObservationDropRow] = []
        for key in [key for key in self._pending if key[0] == ops_db]:
            count, first_ms, last_ms = self._pending.pop(key)
            rows.append(
                RouteObservationDropRow(
                    surface=key[2],
                    route=key[3],
                    reason=key[1],
                    first_observed_at_ms=first_ms,
                    last_observed_at_ms=last_ms,
                    drop_count=count,
                )
            )
        return tuple(rows)

    def restore(self, ops_db: Path, rows: Sequence[RouteObservationDropRow]) -> None:
        """Put drained drops back after the write that would have recorded them failed."""
        for row in rows:
            self.record(
                RouteObservationDropReason(row.reason),
                surface=row.surface,
                route=row.route,
                ops_db=ops_db,
                observed_at_ms=row.first_observed_at_ms,
                count=row.drop_count,
            )
            entry = self._pending[(ops_db, row.reason, row.surface, row.route)]
            entry[2] = max(entry[2], row.last_observed_at_ms)

    def recordable_count(self) -> int:
        """Pending drops that belong to an ops tier (and so to some archive's sample)."""
        return sum(entry[0] for key, entry in self._pending.items() if key[0] is not None)

    def pending_tiers(self) -> tuple[Path, ...]:
        return tuple(sorted({key[0] for key in self._pending if key[0] is not None}))

    def snapshot(self) -> RouteObservationDrops:
        """Return the drops this process holds unrecorded. Never claims completeness."""
        reasons: dict[str, int] = {}
        routes: dict[str, int] = {}
        for (_ops_db, reason, surface, route), (count, _first, _last) in self._pending.items():
            reasons[reason] = reasons.get(reason, 0) + count
            routes[route_key(surface, route)] = routes.get(route_key(surface, route), 0) + count
        return RouteObservationDrops(
            accounting_complete=False,
            by_reason=reasons,
            by_route=routes,
        )

    def reset(self) -> None:
        self._pending.clear()


_DROP_LEDGER = RouteObservationDropLedger()
_EXIT_FLUSH_REGISTERED = False


def _register_exit_flush() -> None:
    global _EXIT_FLUSH_REGISTERED
    if _EXIT_FLUSH_REGISTERED:
        return
    import atexit

    atexit.register(flush_route_observation_drops)
    _EXIT_FLUSH_REGISTERED = True


def flush_route_observation_drops() -> int:
    """Record every pending drop in its ops tier; return how many drops remain unrecorded.

    Runs at interpreter exit. A tier that still cannot be written keeps its
    drops pending and the loss is reported as a typed
    ``route_observation.drops_unflushed`` event, the one place those drops
    remain visible once the process is gone.
    """
    from polylogue.logging import WARNING, emit
    from polylogue.storage.sqlite.archive_tiers.ops_write import record_route_observation_drops

    for ops_db in _DROP_LEDGER.pending_tiers():
        rows = _DROP_LEDGER.drain(ops_db)
        if not ops_db.exists():
            _DROP_LEDGER.restore(ops_db, rows)
            continue
        try:
            conn = open_observation_connection(ops_db)
            try:
                record_route_observation_drops(conn, drops=rows, now_ms=int(time.time() * 1000))
            finally:
                conn.close()
        except Exception:
            _DROP_LEDGER.restore(ops_db, rows)
    remaining = _DROP_LEDGER.recordable_count()
    if remaining:
        emit(
            "route_observation.drops_unflushed",
            level=WARNING,
            outcome="degraded",
            reason="ops_tier_unwritable_at_exit",
            count=remaining,
        )
    return remaining


def route_observation_drops() -> RouteObservationDrops:
    """Return the drops this process has counted but not yet recorded in an ops tier."""
    return _DROP_LEDGER.snapshot()


def reset_route_observation_drops() -> None:
    """Clear the process-local pending drops."""
    _DROP_LEDGER.reset()


# ---------------------------------------------------------------------------
# The route-observation contract (polylogue-jtwu.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RouteObservationSpec:
    """The declared contract for one instrumented route.

    Names the route once, so every surface that wants its latency joins the
    same identifiers instead of inventing parallel ones.
    """

    surface: str
    route: str
    verb: str | None = None
    phases: tuple[str, ...] = (DEFAULT_ROUTE_PHASE,)
    version: int = 1
    sampled: bool = True
    """Whether invocations of this route are recorded at all. A spec that is
    not sampled still has a declared contract; its invocations are counted as
    ``NOT_SAMPLED`` drops rather than vanishing."""

    def __post_init__(self) -> None:
        if not self.surface or not self.route:
            raise ValueError("a route observation spec requires a surface and a route")
        if self.version <= 0:
            raise ValueError("a route observation spec version must be positive")
        if not self.phases or len(set(self.phases)) != len(self.phases):
            raise ValueError("route observation phases must be non-empty and unique")
        if any(not phase for phase in self.phases):
            raise ValueError("route observation phase names must be non-empty")
        if self.phases[0] != DEFAULT_ROUTE_PHASE:
            raise ValueError(f"the first declared route phase must be {DEFAULT_ROUTE_PHASE!r}")

    @property
    def workload_id(self) -> str:
        """Stable workload identity for this route, shared by every receipt."""
        suffix = f":{self.verb}" if self.verb else ""
        return f"route:{self.surface}:{self.route}{suffix}"

    @property
    def attribution_key(self) -> str:
        return route_key(self.surface, self.route)

    def to_payload(self) -> dict[str, object]:
        return {
            "surface": self.surface,
            "route": self.route,
            "verb": self.verb,
            "phases": list(self.phases),
            "version": self.version,
            "sampled": self.sampled,
        }

    def to_workload_spec(self) -> WorkloadEnvelopeSpec:
        """Return the ``WorkloadEnvelopeSpec`` this route's receipts observe."""
        from polylogue.scenarios.workload import WorkloadEnvelopeSpec, WorkloadInputRef

        return WorkloadEnvelopeSpec(
            workload_id=self.workload_id,
            family_id=ROUTE_OBSERVATION_WORKLOAD_FAMILY,
            version=self.version,
            inputs=(WorkloadInputRef(input_id=f"route-invocation:{self.surface}:{self.route}"),),
            phases=self.phases,
        )


@dataclass(frozen=True, slots=True)
class RoutePhaseObservation:
    """Wall and CPU evidence for one declared phase of a route invocation."""

    name: str
    wall_ms: float
    cpu_ms: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a route phase observation requires a name")
        if self.wall_ms < 0:
            raise ValueError("route phase wall time must be non-negative")
        if self.cpu_ms is not None and self.cpu_ms < 0:
            raise ValueError("route phase CPU time must be non-negative")

    @property
    def unavailable(self) -> tuple[str, ...]:
        """Declared measures this phase could not obtain."""
        return () if self.cpu_ms is not None else ("cpu_ms",)


@dataclass(frozen=True, slots=True)
class RouteObservationReceipt:
    """Evidence for one route invocation, against its declared spec.

    Carries build/archive/workload scope, request and run identity with its
    parent, surface/route/verb, daemon-versus-direct, named phases with wall
    and CPU time, response size and status, evidence refs, and which measures
    were unavailable.
    """

    spec: RouteObservationSpec
    trace_id: str
    """Request id: the correlation handle shared with structured logs and with
    any other projection of the same invocation."""
    run_id: str
    """Identity of this receipt's own run, distinct from the request it serves."""
    parent_run_id: str | None
    started_at_ms: int
    phases: tuple[RoutePhaseObservation, ...]
    status: RouteObservationStatus
    daemon_path: str | None = None
    build_id: str | None = None
    archive_id: str | None = None
    """Archive scope, supplied by the caller. Deliberately not derived from the
    archive root path: telemetry must not bake an operator filesystem location
    into every row it writes."""
    archive_epoch: str | None = None
    response_bytes: int | None = None
    evidence_refs: tuple[str, ...] = ()
    attributes: Mapping[str, object] = field(default_factory=dict)
    sampled: bool = True
    daemon_run_id: str | None = None
    """The daemon run this invocation executed in (``daemon.cli.daemon_run_id``),
    when it ran inside one: the id the run's lifecycle row, heartbeats, status
    and events carry (polylogue-peo), so a dead run's last workload joins to it."""

    def __post_init__(self) -> None:
        if not self.trace_id or not self.run_id:
            raise ValueError("a route observation receipt requires a trace id and a run id")
        observed = tuple(phase.name for phase in self.phases)
        if len(set(observed)) != len(observed):
            raise ValueError("a route observation receipt cannot repeat a phase")
        undeclared = set(observed) - set(self.spec.phases)
        if undeclared:
            raise ValueError(f"receipt observes undeclared route phases: {sorted(undeclared)}")
        if DEFAULT_ROUTE_PHASE not in observed:
            raise ValueError(f"every route observation receipt must observe the {DEFAULT_ROUTE_PHASE!r} phase")
        if RECEIPT_ATTRIBUTE_KEY in self.attributes:
            raise ValueError(f"{RECEIPT_ATTRIBUTE_KEY!r} is reserved for the receipt projection")

    @property
    def total_phase(self) -> RoutePhaseObservation:
        return next(phase for phase in self.phases if phase.name == DEFAULT_ROUTE_PHASE)

    @property
    def duration_ms(self) -> int:
        return max(0, round(self.total_phase.wall_ms))

    @property
    def correlation_refs(self) -> tuple[str, ...]:
        """The refs that let a second projection of this invocation be joined."""
        refs = [f"route-request:{self.trace_id}", f"route-run:{self.run_id}"]
        if self.parent_run_id is not None:
            refs.append(f"route-parent-run:{self.parent_run_id}")
        if self.daemon_run_id is not None:
            refs.append(f"daemon-run:{self.daemon_run_id}")
        return tuple(refs)

    @property
    def unavailable_measures(self) -> tuple[str, ...]:
        """Measures no observed phase obtained."""
        missing: set[str] = set()
        for phase in self.phases:
            missing.update(phase.unavailable)
        if self.response_bytes is None:
            missing.add("response_bytes")
        return tuple(sorted(missing))

    def to_attributes(self) -> dict[str, object]:
        """Project the receipt into the ops-tier row's ``attributes`` document.

        ``route_observations`` predates this contract and carries only
        trace/surface/route/timing columns, so the receipt's scope and
        correlation fields ride in the row's declared freeform attributes. The
        correlation refs are what make the persisted row joinable with the
        workload receipt built from the same invocation.
        """
        payload: dict[str, object] = dict(self.attributes)
        payload[RECEIPT_ATTRIBUTE_KEY] = {
            "spec": self.spec.to_payload(),
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "daemon_run_id": self.daemon_run_id,
            "build_id": self.build_id,
            "archive_id": self.archive_id,
            "workload_id": self.spec.workload_id,
            "correlation_refs": list(self.correlation_refs),
            "phases": [{"name": phase.name, "wall_ms": phase.wall_ms, "cpu_ms": phase.cpu_ms} for phase in self.phases],
            "response_bytes": self.response_bytes,
            "evidence_refs": list(self.evidence_refs),
            "unavailable": list(self.unavailable_measures),
        }
        return payload

    def to_workload_receipt(self) -> WorkloadReceipt:
        """Adapt this route observation onto the shared ``WorkloadReceipt``.

        The adapter polylogue-jtwu's DESIGN names: route latency becomes an
        ordinary workload receipt rather than a parallel measurement vocabulary.
        """
        from polylogue.scenarios.workload import (
            WorkloadPhaseObservation,
            WorkloadReceipt,
            WorkloadRunStatus,
        )

        workload_spec = self.spec.to_workload_spec()
        observations = tuple(
            WorkloadPhaseObservation(
                name=phase.name,
                wall_ms=phase.wall_ms,
                cpu_ms=phase.cpu_ms,
                response_bytes=self.response_bytes if phase.name == DEFAULT_ROUTE_PHASE else None,
                unavailable=phase.unavailable
                + (("response_bytes",) if phase.name == DEFAULT_ROUTE_PHASE and self.response_bytes is None else ()),
            )
            for phase in self.phases
        )
        status = _workload_status(self.status)
        if status is WorkloadRunStatus.SUCCEEDED and tuple(phase.name for phase in self.phases) != workload_spec.phases:
            # A declared phase was never entered: the invocation finished, but
            # its measurement did not cover the declared work.
            status = WorkloadRunStatus.INTERRUPTED
        return WorkloadReceipt.from_observations(
            spec=workload_spec,
            status=status,
            build_id=self.build_id,
            runtime_id=self.daemon_path,
            archive_id=self.archive_id,
            generation_id=self.archive_epoch,
            frame_id=None,
            phases=observations,
            evidence_refs=self.correlation_refs + self.evidence_refs,
            daemon_run_id=self.daemon_run_id,
        )


def _workload_status(status: str) -> WorkloadRunStatus:
    from polylogue.scenarios.workload import WorkloadRunStatus

    if status in ("error", "timed_out", "unavailable"):
        return WorkloadRunStatus.FAILED
    if status == "degraded":
        return WorkloadRunStatus.INTERRUPTED
    return WorkloadRunStatus.SUCCEEDED


@dataclass
class RouteObservationContext:
    """Mutable handle yielded by :func:`observe_route`.

    ``status`` defaults to ``"ok"`` and is set to ``"error"`` automatically
    if the observed block raises; callers may set it explicitly (e.g.
    ``"degraded"``) before the block exits. ``attributes`` is freeform
    caller-supplied detail (e.g. per-component states) recorded alongside
    the timing.
    """

    attributes: dict[str, object] = field(default_factory=dict)
    status: RouteObservationStatus = "ok"
    daemon_path: str | None = None
    """Set explicitly by the caller once known ('daemon' or 'direct'); the
    ``observe_route`` argument of the same name only seeds the initial
    value for callers that already know it when the block starts."""
    response_bytes: int | None = None
    evidence_refs: tuple[str, ...] = ()
    receipt: RouteObservationReceipt | None = None
    """The emitted receipt, available after the observed block exits."""

    _phases: list[RoutePhaseObservation] = field(default_factory=list, repr=False)
    _declared_phases: tuple[str, ...] = (DEFAULT_ROUTE_PHASE,)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time one declared sub-phase of the observed route."""
        if name not in self._declared_phases:
            raise ValueError(f"route phase {name!r} is not declared by this route's spec")
        if name == DEFAULT_ROUTE_PHASE:
            raise ValueError(f"the {DEFAULT_ROUTE_PHASE!r} phase is measured by observe_route itself")
        wall_start = time.monotonic()
        cpu_start = time.process_time()
        try:
            yield
        finally:
            self._phases.append(
                RoutePhaseObservation(
                    name=name,
                    wall_ms=max(0.0, (time.monotonic() - wall_start) * 1000.0),
                    cpu_ms=max(0.0, (time.process_time() - cpu_start) * 1000.0),
                )
            )


def _current_git_head(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_HEAD_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    head = result.stdout.strip()
    return head or None


@contextmanager
def observe_route(
    *,
    archive_root: Path | None,
    surface: str,
    route: str,
    verb: str | None = None,
    daemon_path: str | None = None,
    trace_id: str | None = None,
    git_head_cwd: Path | None = None,
    spec: RouteObservationSpec | None = None,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    archive_id: str | None = None,
    archive_epoch: str | None = None,
) -> Iterator[RouteObservationContext]:
    """Time one route invocation and record a best-effort receipt.

    Builds a :class:`RouteObservationReceipt` against ``spec`` (or a
    single-phase spec derived from ``surface``/``route``/``verb``) and persists
    its projection through the existing ``route_observations`` writer, so
    ``compute_latency_percentiles`` and ``polylogue analyze latency`` keep
    reading the same rows.

    Telemetry failures (no archive configured, locked ops.db, disposable
    tier missing) are logged at debug level, counted on the drop ledger, and
    swallowed -- they must never surface as an error in, or block, the
    operation being observed. Re-raises whatever the observed block raises,
    unchanged.
    """
    resolved_spec = spec if spec is not None else RouteObservationSpec(surface=surface, route=route, verb=verb)
    if spec is not None and (
        surface != spec.surface or route != spec.route or (verb is not None and verb != spec.verb)
    ):
        raise ValueError("route observation spec identity must match the call-site identity")
    ctx = RouteObservationContext(daemon_path=daemon_path)
    ctx._declared_phases = resolved_spec.phases
    started_at_ms = int(time.time() * 1000)
    started_monotonic = time.monotonic()
    started_cpu = time.process_time()
    try:
        yield ctx
    except Exception:
        ctx.status = "error"
        raise
    finally:
        total = RoutePhaseObservation(
            name=DEFAULT_ROUTE_PHASE,
            wall_ms=max(0.0, (time.monotonic() - started_monotonic) * 1000.0),
            cpu_ms=max(0.0, (time.process_time() - started_cpu) * 1000.0),
        )
        phases_by_name: dict[str, RoutePhaseObservation] = {DEFAULT_ROUTE_PHASE: total}
        for phase in ctx._phases:
            previous = phases_by_name.get(phase.name)
            phases_by_name[phase.name] = (
                phase
                if previous is None
                else RoutePhaseObservation(
                    phase.name,
                    previous.wall_ms + phase.wall_ms,
                    None if previous.cpu_ms is None or phase.cpu_ms is None else previous.cpu_ms + phase.cpu_ms,
                )
            )
        invalid_daemon_path = False
        try:
            recordable_path = _recordable_daemon_path(ctx.daemon_path)
        except Exception:
            _DROP_LEDGER.record(
                RouteObservationDropReason.EMIT_FAILED,
                surface=resolved_spec.surface,
                route=resolved_spec.route,
                ops_db=None if archive_root is None else Path(archive_root) / "ops.db",
                observed_at_ms=started_at_ms,
            )
            invalid_daemon_path = True
            recordable_path = None
        receipt = RouteObservationReceipt(
            spec=resolved_spec,
            trace_id=trace_id or str(uuid.uuid4()),
            run_id=run_id or str(uuid.uuid4()),
            parent_run_id=parent_run_id,
            started_at_ms=started_at_ms,
            phases=tuple(
                phases_by_name[name]
                for name in (DEFAULT_ROUTE_PHASE, *resolved_spec.phases[1:])
                if name in phases_by_name
            ),
            status=ctx.status,
            daemon_path=recordable_path,
            build_id=_current_git_head(git_head_cwd) if git_head_cwd is not None else None,
            archive_id=archive_id,
            archive_epoch=archive_epoch,
            response_bytes=ctx.response_bytes,
            evidence_refs=ctx.evidence_refs,
            attributes=dict(ctx.attributes),
            sampled=resolved_spec.sampled,
            daemon_run_id=_bound_daemon_run_id(),
        )
        ctx.receipt = receipt
        if not invalid_daemon_path:
            _emit_best_effort(archive_root=archive_root, receipt=receipt)


def _bound_daemon_run_id() -> str | None:
    """Return the daemon run id bound in this context, when running in a daemon.

    ``daemon.cli.daemon_run_id`` binds ``run_id`` with ``component="daemon"``
    into the process-wide logging context before the run's first event, so an
    invocation inside the daemon reads it here without importing the daemon.
    """
    from polylogue.logging import current_context

    context = current_context()
    run_id = context.get("run_id")
    if context.get("component") == "daemon" and isinstance(run_id, str) and run_id:
        return run_id
    return None


def open_observation_connection(ops_db: Path) -> sqlite3.Connection:
    """Open the best-effort route-observation writer for ``ops_db``."""
    conn = sqlite3.connect(ops_db, timeout=_CONNECT_TIMEOUT_S)
    try:
        conn.execute(f"PRAGMA synchronous = {_OBSERVATION_SYNCHRONOUS}")
    except sqlite3.Error:
        conn.close()
        raise
    return conn


def _emit_best_effort(*, archive_root: Path | None, receipt: RouteObservationReceipt) -> None:
    """Persist ``receipt``'s projection, counting every path that loses it.

    A successful write also records the drops this process was holding for the
    same tier, in the same transaction.
    """
    spec = receipt.spec
    ops_db = None if archive_root is None else Path(archive_root) / "ops.db"

    def drop(reason: RouteObservationDropReason) -> None:
        _DROP_LEDGER.record(
            reason, surface=spec.surface, route=spec.route, ops_db=ops_db, observed_at_ms=receipt.started_at_ms
        )

    if not receipt.sampled:
        drop(RouteObservationDropReason.NOT_SAMPLED)
        return
    if ops_db is None:
        drop(RouteObservationDropReason.NO_ARCHIVE_ROOT)
        return
    if not ops_db.exists():
        drop(RouteObservationDropReason.OPS_DB_MISSING)
        return
    pending = _DROP_LEDGER.drain(ops_db)
    try:
        from polylogue.storage.sqlite.archive_tiers.ops_write import record_route_observation

        conn = open_observation_connection(ops_db)
        try:
            record_route_observation(
                conn,
                trace_id=receipt.trace_id,
                surface=spec.surface,
                route=spec.route,
                verb=spec.verb,
                daemon_path=receipt.daemon_path,
                started_at_ms=receipt.started_at_ms,
                duration_ms=receipt.duration_ms,
                status=receipt.status,
                git_head=receipt.build_id,
                archive_epoch=receipt.archive_epoch,
                attributes=receipt.to_attributes(),
                sampled=receipt.sampled,
                drops=pending,
            )
        finally:
            conn.close()
    except Exception:
        _DROP_LEDGER.restore(ops_db, pending)
        drop(RouteObservationDropReason.EMIT_FAILED)
        logger.debug(
            "route observation emit failed (best-effort, dropped): surface=%s route=%s",
            spec.surface,
            spec.route,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Latency projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RouteLatencyBucket:
    """One (surface, route) group's latency distribution and its lost members."""

    surface: str
    route: str
    sample_count: int
    p50_ms: float | None
    p95_ms: float | None
    error_count: int
    error_rate: float
    oldest_at_ms: int | None
    newest_at_ms: int | None
    low_confidence: bool
    """True when ``sample_count`` is below :data:`LOW_CONFIDENCE_SAMPLE_FLOOR`
    -- a p95 over a handful of samples is not a reliable percentile; render
    it visibly caveated rather than as a confident budget number."""
    dropped_count: int
    """Observations of this route known to have been lost before the sample."""
    drop_accounting_complete: bool
    """False when drops for this route exist that nobody in this process could
    count. The percentile is then over an unknown fraction of reality."""

    @property
    def sample_completeness(self) -> float | None:
        """Fraction of known invocations this percentile actually saw.

        ``None`` when the drop accounting is incomplete: an unknown denominator
        has no fraction, and reporting 1.0 would be the exact lie this field
        exists to prevent.
        """
        if not self.drop_accounting_complete:
            return None
        total = self.sample_count + self.dropped_count
        if total == 0:
            return None
        return self.sample_count / total

    @property
    def is_qualified(self) -> bool:
        """True when this percentile must be rendered with a caveat."""
        return self.low_confidence or self.dropped_count > 0 or not self.drop_accounting_complete


@dataclass(frozen=True, slots=True)
class RouteLatencyReport:
    """Percentiles and the drop disposition they were computed under.

    The two cannot be separated: :func:`compute_latency_percentiles` returns
    this rather than bare buckets so no surface can render a p50/p95 without
    also holding the statement of what the sample lost.
    """

    buckets: tuple[RouteLatencyBucket, ...]
    drops: RouteObservationDrops
    window_exceeds_retention: bool = False
    """The lookback window starts before the writers' age horizon, so part of
    it can no longer be answered: its observations and its drop records were
    retired together."""

    @property
    def unattributed_drops(self) -> int:
        """Drops that belong to no rendered bucket, and would vanish if the
        report were flattened to its buckets."""
        visible = {route_key(bucket.surface, bucket.route) for bucket in self.buckets}
        return max(0, self.drops.total - sum(self.drops.by_route.get(key, 0) for key in visible))

    @property
    def is_complete(self) -> bool:
        return self.drops.accounting_complete and self.drops.total == 0

    @property
    def outcome(self) -> OutcomeEnvelope:
        """The report's terminal outcome: lost observations are a named gap.

        A percentile over a sample that lost members -- an unknown number, or a
        counted one -- is not a whole measurement of the route, so such an
        answer is ``degraded`` even when it holds buckets, and an empty window
        stays ``degraded`` too: the gap, not the absence of traffic, may be why
        it is empty. Complete accounting with nothing lost is ``ok``.
        """
        from polylogue.surfaces.outcome import decide_outcome

        gaps: list[str] = []
        if not self.drops.accounting_complete:
            gaps.append(DROP_ACCOUNTING_INCOMPLETE)
        elif self.drops.total > 0:
            gaps.append(OBSERVATIONS_DROPPED)
        if self.window_exceeds_retention:
            gaps.append(WINDOW_EXCEEDS_RETENTION)
        return decide_outcome(matched=len(self.buckets), degraded=tuple(gaps))

    def to_payload(self) -> dict[str, object]:
        drops = self.drops.to_payload()
        drops["unattributed"] = self.unattributed_drops
        return {
            "outcome": self.outcome.to_dict(),
            "buckets": [
                {
                    "surface": bucket.surface,
                    "route": bucket.route,
                    "sample_count": bucket.sample_count,
                    "p50_ms": bucket.p50_ms,
                    "p95_ms": bucket.p95_ms,
                    "error_count": bucket.error_count,
                    "error_rate": round(bucket.error_rate, 4),
                    "low_confidence": bucket.low_confidence,
                    "dropped_count": bucket.dropped_count,
                    "drop_accounting_complete": bucket.drop_accounting_complete,
                    "sample_completeness": bucket.sample_completeness,
                }
                for bucket in self.buckets
            ],
            "drops": drops,
            "window_exceeds_retention": self.window_exceeds_retention,
        }


def _percentile(sorted_values: Sequence[int], quantile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = quantile * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def compute_latency_percentiles(
    route_observations: Iterable[object],
    mcp_calls: Iterable[object] = (),
    *,
    drops: RouteObservationDrops,
) -> RouteLatencyReport:
    """Group ``route_observations`` and ``mcp_calls`` by (surface, route) and compute p50/p95.

    Accepts ``ArchiveRouteObservation``/``ArchiveMcpCallLogEntry`` rows
    (typed as ``object`` here to avoid importing the ops-tier module at
    call sites that only need the pure aggregation); duck-types on the
    attributes each row type actually has.

    ``drops`` is required, not defaulted. A caller that genuinely built the
    whole sample passes :meth:`RouteObservationDrops.none_observed`; a caller
    reading rows produced by other processes passes
    :meth:`RouteObservationDrops.unaccounted` or the emitter's own ledger. A
    default would let a percentile be computed as though nothing was ever lost,
    which is the defect this signature exists to make unexpressible.
    """
    grouped: dict[tuple[str, str], list[tuple[int, bool, int]]] = defaultdict(list)
    for obs in route_observations:
        grouped[(obs.surface, obs.route)].append(  # type: ignore[attr-defined]
            (obs.duration_ms, obs.status in ("error", "timed_out", "unavailable"), obs.started_at_ms)  # type: ignore[attr-defined]
        )
    for call in mcp_calls:
        grouped[("mcp", f"mcp.{call.tool_name}")].append(  # type: ignore[attr-defined]
            (call.duration_ms, not call.success, call.started_at_ms)  # type: ignore[attr-defined]
        )

    buckets: list[RouteLatencyBucket] = []
    for (surface, route), samples in sorted(grouped.items()):
        durations = sorted(duration for duration, _is_error, _started in samples)
        error_count = sum(1 for _duration, is_error, _started in samples if is_error)
        started_values = [started for _duration, _is_error, started in samples]
        buckets.append(
            RouteLatencyBucket(
                surface=surface,
                route=route,
                sample_count=len(samples),
                p50_ms=_percentile(durations, 0.50),
                p95_ms=_percentile(durations, 0.95),
                error_count=error_count,
                error_rate=error_count / len(samples) if samples else 0.0,
                oldest_at_ms=min(started_values) if started_values else None,
                newest_at_ms=max(started_values) if started_values else None,
                low_confidence=len(samples) < LOW_CONFIDENCE_SAMPLE_FLOOR,
                dropped_count=drops.attributed_to(surface, route),
                drop_accounting_complete=drops.accounting_complete,
            )
        )
    return RouteLatencyReport(buckets=tuple(buckets), drops=drops)


@dataclass(frozen=True, slots=True)
class _ObservationSample:
    surface: str
    route: str
    duration_ms: int
    status: str
    started_at_ms: int


@dataclass(frozen=True, slots=True)
class _McpCallSample:
    tool_name: str
    duration_ms: int
    success: bool
    started_at_ms: int


def read_latency_report(
    conn: sqlite3.Connection,
    *,
    since_ms: int,
    surface: str | None = None,
    now_ms: int | None = None,
) -> RouteLatencyReport:
    """Compute the latency report over every sample in the lookback window.

    The sample is the whole window, streamed from the cursor: a reader that
    kept only the newest N rows reported a p95 over "the newest N of an
    unknown total", which is not a measurement of the route. The window is
    bounded by the writers' own retention (``ROUTE_OBSERVATION_RETENTION_MS``
    and its row cap for observations, ``MCP_CALL_LOG_RETENTION_MS`` for MCP
    calls), not by this reader.

    Drops come from ``route_observation_drops``, where every emitting process
    records what it lost (polylogue-jtwu.2), so a reader in another process
    reports them beside the percentiles and an answer with nothing lost is
    ``ok``. A window that starts before the retention horizon is ``degraded``:
    that part of it was retired, observations and drop records alike.
    """
    observation_sql = (
        "SELECT surface, route, duration_ms, status, started_at_ms FROM route_observations WHERE started_at_ms >= ?"
    )
    observation_params: tuple[object, ...] = (since_ms,)
    if surface is not None:
        observation_sql += " AND surface = ?"
        observation_params = (since_ms, surface)
    observations = (
        _ObservationSample(
            surface=str(row[0]),
            route=str(row[1]),
            duration_ms=int(row[2]),
            status=str(row[3]),
            started_at_ms=int(row[4]),
        )
        for row in conn.execute(observation_sql, observation_params)
    )
    report_without_calls = surface not in (None, "mcp")
    calls = (
        ()
        if report_without_calls
        else (
            _McpCallSample(
                tool_name=str(row[0]),
                duration_ms=int(row[1]),
                success=bool(row[2]),
                started_at_ms=int(row[3]),
            )
            for row in conn.execute(
                "SELECT tool_name, duration_ms, success, started_at_ms FROM mcp_call_log WHERE started_at_ms >= ?",
                (since_ms,),
            )
        )
    )
    from polylogue.storage.sqlite.archive_tiers.ops_write import ROUTE_OBSERVATION_RETENTION_MS

    report = compute_latency_percentiles(observations, calls, drops=_recorded_drops(conn, since_ms, surface))
    current_ms = int(time.time() * 1000) if now_ms is None else now_ms
    if since_ms < current_ms - ROUTE_OBSERVATION_RETENTION_MS:
        return RouteLatencyReport(buckets=report.buckets, drops=report.drops, window_exceeds_retention=True)
    return report


def _recorded_drops(conn: sqlite3.Connection, since_ms: int, surface: str | None) -> RouteObservationDrops:
    """Sum the drops every process recorded for the window."""
    from polylogue.storage.sqlite.archive_tiers.ops_write import route_observation_drop_counts

    rows = route_observation_drop_counts(conn, since_ms=since_ms, surface=surface)
    reasons: dict[str, int] = {}
    routes: dict[str, int] = {}
    for row in rows:
        reasons[row.reason] = reasons.get(row.reason, 0) + row.drop_count
        key = route_key(row.surface, row.route)
        routes[key] = routes.get(key, 0) + row.drop_count
    return RouteObservationDrops(accounting_complete=True, by_reason=reasons, by_route=routes)


__all__ = [
    "DEFAULT_ROUTE_PHASE",
    "DROP_ACCOUNTING_INCOMPLETE",
    "OBSERVATIONS_DROPPED",
    "WINDOW_EXCEEDS_RETENTION",
    "LOW_CONFIDENCE_SAMPLE_FLOOR",
    "RECEIPT_ATTRIBUTE_KEY",
    "ROUTE_OBSERVATION_WORKLOAD_FAMILY",
    "RouteLatencyBucket",
    "RouteLatencyReport",
    "RouteObservationContext",
    "RouteObservationDropLedger",
    "RouteObservationDropReason",
    "RouteObservationDrops",
    "RouteObservationReceipt",
    "RouteObservationSpec",
    "RoutePhaseObservation",
    "compute_latency_percentiles",
    "flush_route_observation_drops",
    "observe_route",
    "open_observation_connection",
    "read_latency_report",
    "reset_route_observation_drops",
    "route_key",
    "route_observation_drops",
]
