"""Shared workload declarations and resource receipts.

The declaration describes logical work independently from its physical
execution budget.  A budget may measure, gate, or contain execution, but it
cannot redefine the requested result set.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from enum import Enum

from polylogue.core.json import JSONDocument, JSONValue, json_document
from polylogue.schemas.workload_tiers import WorkloadScaleTier, WorkloadSelectivityTier


class MeasurementScope(str, Enum):
    """Kernel accounting boundary used for a workload run."""

    PROCESS_TREE = "process-tree"
    CGROUP = "cgroup"


class BudgetSemantics(str, Enum):
    """What exceeding a physical budget means."""

    MEASURE_ONLY = "measure-only"
    REGRESSION_GATE = "regression-gate"
    CONTAINMENT = "containment"


class BudgetAggregation(str, Enum):
    """How like measurements from selected phases form one verdict."""

    MAXIMUM = "maximum"
    SUM = "sum"
    FINAL = "final"


class BudgetMeasure(str, Enum):
    """Physical measures that may be budgeted without limiting semantics."""

    WALL_MS = "wall_ms"
    CPU_MS = "cpu_ms"
    PEAK_RSS_BYTES = "peak_rss_bytes"
    PEAK_PSS_BYTES = "peak_pss_bytes"
    ANON_BYTES = "anon_bytes"
    FILE_CACHE_BYTES = "file_cache_bytes"
    SWAP_BYTES = "swap_bytes"
    TEMP_STORAGE_BYTES = "temp_storage_bytes"
    READ_IO_BYTES = "read_io_bytes"
    WRITE_IO_BYTES = "write_io_bytes"
    RESPONSE_BYTES = "response_bytes"
    CANCELLATION_LATENCY_MS = "cancellation_latency_ms"
    QUEUE_DEPTH = "queue_depth"
    BACKPRESSURE_MS = "backpressure_ms"
    SQLITE_VM_STEPS = "sqlite_vm_steps"


class BudgetVerdict(str, Enum):
    """Measured result of one declared physical budget."""

    PASS = "pass"
    EXCEEDED = "exceeded"
    MEASUREMENT_UNAVAILABLE = "measurement-unavailable"


class WorkloadRunStatus(str, Enum):
    """Terminal status of the logical workload."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class WorkloadInputRef:
    """Stable identity for the executable corpus/profile input."""

    input_id: str
    corpus_id: str | None = None
    profile_id: str | None = None
    package_ref: str | None = None
    scale_tier: str | None = None
    selectivity_tier: str | None = None
    seed: int | None = None
    distribution_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.input_id:
            raise ValueError("Workload input identity is required")

    def to_payload(self) -> JSONDocument:
        return json_document(
            {
                "input_id": self.input_id,
                "corpus_id": self.corpus_id,
                "profile_id": self.profile_id,
                "package_ref": self.package_ref,
                "scale_tier": self.scale_tier,
                "selectivity_tier": self.selectivity_tier,
                "seed": self.seed,
                "distribution_refs": list(self.distribution_refs),
            }
        )


@dataclass(frozen=True, slots=True)
class WorkloadBudget:
    """One physical constraint; logical result cardinality is not expressible."""

    measure: BudgetMeasure
    maximum: float
    semantics: BudgetSemantics = BudgetSemantics.MEASURE_ONLY
    aggregation: BudgetAggregation | None = None
    phase: str | None = None

    def __post_init__(self) -> None:
        if self.maximum < 0:
            raise ValueError("Workload budget maximum must be non-negative")

    def to_payload(self) -> JSONDocument:
        return {
            "measure": self.measure.value,
            "maximum": self.maximum,
            "semantics": self.semantics.value,
            "aggregation": self.resolved_aggregation.value,
            "phase": self.phase,
        }

    @property
    def resolved_aggregation(self) -> BudgetAggregation:
        if self.aggregation is not None:
            return self.aggregation
        return _DEFAULT_BUDGET_AGGREGATION[self.measure]


@dataclass(frozen=True, slots=True)
class WorkloadEnvelopeSpec:
    """One reusable logical workload plus its physical observation contract."""

    workload_id: str
    family_id: str
    version: int
    inputs: tuple[WorkloadInputRef, ...]
    phases: tuple[str, ...]
    measurement_scope: MeasurementScope = MeasurementScope.PROCESS_TREE
    concurrency: int = 1
    admission: str = "immediate"
    quiescence_ms: int = 0
    budgets: tuple[WorkloadBudget, ...] = ()
    semantic_result: str = "complete"

    def __post_init__(self) -> None:
        if not self.workload_id or not self.family_id:
            raise ValueError("Workload and family identities are required")
        if self.version <= 0:
            raise ValueError("Workload version must be positive")
        if self.concurrency <= 0:
            raise ValueError("Workload concurrency must be positive")
        if self.quiescence_ms < 0:
            raise ValueError("Workload quiescence must be non-negative")
        if len(set(self.phases)) != len(self.phases) or any(not phase for phase in self.phases):
            raise ValueError("Workload phase names must be unique and non-empty")
        undeclared_budget_phases = {
            budget.phase for budget in self.budgets if budget.phase is not None and budget.phase not in self.phases
        }
        if undeclared_budget_phases:
            raise ValueError(f"Workload budgets reference undeclared phases: {sorted(undeclared_budget_phases)}")
        if self.semantic_result != "complete":
            raise ValueError("Physical workload budgets cannot narrow logical result semantics")

    def to_payload(self) -> JSONDocument:
        return json_document(
            {
                "workload_id": self.workload_id,
                "family_id": self.family_id,
                "version": self.version,
                "inputs": [item.to_payload() for item in self.inputs],
                "phases": list(self.phases),
                "measurement_scope": self.measurement_scope.value,
                "concurrency": self.concurrency,
                "admission": self.admission,
                "quiescence_ms": self.quiescence_ms,
                "budgets": [budget.to_payload() for budget in self.budgets],
                "semantic_result": self.semantic_result,
            }
        )

    @property
    def spec_id(self) -> str:
        return _identity("workload-spec", self.to_payload())


_PHASE_MEASURES = frozenset(measure.value for measure in BudgetMeasure) | {
    "current_rss_bytes",
    "current_pss_bytes",
    "storage_bytes",
    "progress_completed",
    "progress_total",
    "cleanup_reclaimed_bytes",
    "sqlite_vm_steps",
}


@dataclass(frozen=True, slots=True)
class WorkloadPhaseObservation:
    """Resource and progress evidence for one declared phase."""

    name: str
    measurement_scope: MeasurementScope | None = None
    wall_ms: float | None = None
    cpu_ms: float | None = None
    current_rss_bytes: int | None = None
    peak_rss_bytes: int | None = None
    current_pss_bytes: int | None = None
    peak_pss_bytes: int | None = None
    anon_bytes: int | None = None
    file_cache_bytes: int | None = None
    swap_bytes: int | None = None
    temp_storage_bytes: int | None = None
    storage_bytes: int | None = None
    read_io_bytes: int | None = None
    write_io_bytes: int | None = None
    response_bytes: int | None = None
    cancellation_latency_ms: float | None = None
    progress_completed: int | None = None
    progress_total: int | None = None
    queue_depth: int | None = None
    backpressure_ms: float | None = None
    cleanup_reclaimed_bytes: int | None = None
    sqlite_vm_steps: int | None = None
    cleanup_complete: bool | None = None
    quiescent: bool = False
    unavailable: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Workload phase name is required")
        unknown = set(self.unavailable) - _PHASE_MEASURES
        if unknown:
            raise ValueError(f"Unknown unavailable workload measures: {sorted(unknown)}")
        for measure in self.unavailable:
            if getattr(self, measure) is not None:
                raise ValueError(f"Workload measure {measure} cannot be both observed and unavailable")

    def to_payload(self) -> JSONDocument:
        values: dict[str, JSONValue] = {
            "name": self.name,
            "measurement_scope": self.measurement_scope.value if self.measurement_scope is not None else None,
        }
        for field_name in sorted(_PHASE_MEASURES):
            value = getattr(self, field_name)
            if value is not None:
                values[field_name] = value
        values["cleanup_complete"] = self.cleanup_complete
        values["quiescent"] = self.quiescent
        values["unavailable"] = list(self.unavailable)
        return json_document(values)


@dataclass(frozen=True, slots=True)
class WorkloadBudgetResult:
    """Verdict for one budget after a workload run."""

    measure: BudgetMeasure
    semantics: BudgetSemantics
    aggregation: BudgetAggregation
    phase: str | None
    maximum: float
    observed: float | None
    verdict: BudgetVerdict

    def to_payload(self) -> JSONDocument:
        return {
            "measure": self.measure.value,
            "semantics": self.semantics.value,
            "aggregation": self.aggregation.value,
            "phase": self.phase,
            "maximum": self.maximum,
            "observed": self.observed,
            "verdict": self.verdict.value,
        }


@dataclass(frozen=True, slots=True)
class WorkloadReceipt:
    """Content-addressed evidence binding a workload to one archive/build run."""

    spec: WorkloadEnvelopeSpec
    status: WorkloadRunStatus
    build_id: str | None
    runtime_id: str | None
    archive_id: str | None
    generation_id: str | None
    frame_id: str | None
    phases: tuple[WorkloadPhaseObservation, ...]
    budget_results: tuple[WorkloadBudgetResult, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    cancellation_requested: bool = False
    cleanup_complete: bool | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        observed_phases = tuple(phase.name for phase in self.phases)
        if len(set(observed_phases)) != len(observed_phases):
            raise ValueError("A workload receipt cannot contain duplicate phase observations")
        undeclared = set(observed_phases) - set(self.spec.phases)
        if undeclared:
            raise ValueError(f"Receipt contains undeclared workload phases: {sorted(undeclared)}")
        if self.status is WorkloadRunStatus.SUCCEEDED and observed_phases != self.spec.phases:
            raise ValueError("A successful workload receipt must observe every declared phase in order")
        expected_budget_results = evaluate_budgets(self.spec, self.phases)
        if self.budget_results != expected_budget_results:
            raise ValueError("Workload receipt budget verdicts do not match its phase observations")

    @classmethod
    def from_observations(
        cls,
        *,
        spec: WorkloadEnvelopeSpec,
        status: WorkloadRunStatus,
        build_id: str | None,
        runtime_id: str | None,
        archive_id: str | None,
        generation_id: str | None,
        frame_id: str | None,
        phases: tuple[WorkloadPhaseObservation, ...],
        evidence_refs: tuple[str, ...] = (),
        cancellation_requested: bool = False,
        cleanup_complete: bool | None = None,
        notes: tuple[str, ...] = (),
    ) -> WorkloadReceipt:
        """Construct a receipt whose budget verdicts are derived, not asserted."""
        return cls(
            spec=spec,
            status=status,
            build_id=build_id,
            runtime_id=runtime_id,
            archive_id=archive_id,
            generation_id=generation_id,
            frame_id=frame_id,
            phases=phases,
            budget_results=evaluate_budgets(spec, phases),
            evidence_refs=evidence_refs,
            cancellation_requested=cancellation_requested,
            cleanup_complete=cleanup_complete,
            notes=notes,
        )

    def to_payload(self, *, include_receipt_id: bool = True) -> JSONDocument:
        payload = json_document(
            {
                "spec_id": self.spec.spec_id,
                "spec": self.spec.to_payload(),
                "status": self.status.value,
                "build_id": self.build_id,
                "runtime_id": self.runtime_id,
                "archive_id": self.archive_id,
                "generation_id": self.generation_id,
                "frame_id": self.frame_id,
                "phases": [phase.to_payload() for phase in self.phases],
                "budget_results": [result.to_payload() for result in self.budget_results],
                "evidence_refs": list(self.evidence_refs),
                "cancellation_requested": self.cancellation_requested,
                "cleanup_complete": self.cleanup_complete,
                "notes": list(self.notes),
            }
        )
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload

    @property
    def receipt_id(self) -> str:
        return _identity("workload-receipt", self.to_payload(include_receipt_id=False))


def evaluate_budgets(
    spec: WorkloadEnvelopeSpec,
    phases: tuple[WorkloadPhaseObservation, ...],
) -> tuple[WorkloadBudgetResult, ...]:
    """Compare like physical measures without inventing unavailable passes."""
    results: list[WorkloadBudgetResult] = []
    for budget in spec.budgets:
        selected_phases = (phase for phase in phases if budget.phase is None or phase.name == budget.phase)
        observed_values = [
            float(value) for phase in selected_phases if (value := getattr(phase, budget.measure.value)) is not None
        ]
        aggregation = budget.resolved_aggregation
        observed = (
            None
            if not observed_values
            else max(observed_values)
            if aggregation is BudgetAggregation.MAXIMUM
            else sum(observed_values)
            if aggregation is BudgetAggregation.SUM
            else observed_values[-1]
        )
        verdict = (
            BudgetVerdict.MEASUREMENT_UNAVAILABLE
            if observed is None
            else BudgetVerdict.PASS
            if observed <= budget.maximum
            else BudgetVerdict.EXCEEDED
        )
        results.append(
            WorkloadBudgetResult(
                measure=budget.measure,
                semantics=budget.semantics,
                aggregation=aggregation,
                phase=budget.phase,
                maximum=budget.maximum,
                observed=observed,
                verdict=verdict,
            )
        )
    return tuple(results)


_DEFAULT_BUDGET_AGGREGATION: dict[BudgetMeasure, BudgetAggregation] = {
    BudgetMeasure.WALL_MS: BudgetAggregation.SUM,
    BudgetMeasure.CPU_MS: BudgetAggregation.SUM,
    BudgetMeasure.PEAK_RSS_BYTES: BudgetAggregation.MAXIMUM,
    BudgetMeasure.PEAK_PSS_BYTES: BudgetAggregation.MAXIMUM,
    BudgetMeasure.ANON_BYTES: BudgetAggregation.MAXIMUM,
    BudgetMeasure.FILE_CACHE_BYTES: BudgetAggregation.MAXIMUM,
    BudgetMeasure.SWAP_BYTES: BudgetAggregation.MAXIMUM,
    BudgetMeasure.TEMP_STORAGE_BYTES: BudgetAggregation.MAXIMUM,
    BudgetMeasure.READ_IO_BYTES: BudgetAggregation.SUM,
    BudgetMeasure.WRITE_IO_BYTES: BudgetAggregation.SUM,
    BudgetMeasure.RESPONSE_BYTES: BudgetAggregation.SUM,
    BudgetMeasure.CANCELLATION_LATENCY_MS: BudgetAggregation.MAXIMUM,
    BudgetMeasure.QUEUE_DEPTH: BudgetAggregation.MAXIMUM,
    BudgetMeasure.BACKPRESSURE_MS: BudgetAggregation.SUM,
    BudgetMeasure.SQLITE_VM_STEPS: BudgetAggregation.SUM,
}


def exact_session_actions_canary_spec(
    *,
    profile_id: str,
    archive_id: str,
    scale_tier: WorkloadScaleTier = WorkloadScaleTier.CI_ACTIVATION,
    maximum_vm_steps: int = 50_000,
) -> WorkloadEnvelopeSpec:
    """Declare the C-03 exact-session action query against a mixed archive."""
    selectivity_tier = WorkloadSelectivityTier.EXACT_ONE
    input_id = _identity(
        "workload-input",
        {
            "archive_id": archive_id,
            "profile_id": profile_id,
            "scale_tier": scale_tier.value,
            "selectivity_tier": selectivity_tier.value,
        },
    )
    return WorkloadEnvelopeSpec(
        workload_id="canary:c03:exact-session-actions",
        family_id="schema-profile-query-canary",
        version=1,
        inputs=(
            WorkloadInputRef(
                input_id=input_id,
                corpus_id=archive_id,
                profile_id=profile_id,
                scale_tier=scale_tier.value,
                selectivity_tier=selectivity_tier.value,
                distribution_refs=(
                    "index.action_shapes.tool_pairing",
                    "index.predicate_selectivity.exact_existing_session_cardinality",
                ),
            ),
        ),
        phases=("seed", "query", "quiescent"),
        budgets=(
            WorkloadBudget(
                measure=BudgetMeasure.SQLITE_VM_STEPS,
                maximum=maximum_vm_steps,
                semantics=BudgetSemantics.REGRESSION_GATE,
                phase="query",
            ),
        ),
    )


def _schema_profile_canary_spec(
    *,
    workload_id: str,
    profile_id: str,
    archive_id: str,
    phases: tuple[str, ...],
    distribution_refs: tuple[str, ...],
    scale_tier: WorkloadScaleTier = WorkloadScaleTier.CI_ACTIVATION,
) -> WorkloadEnvelopeSpec:
    input_id = _identity(
        "workload-input",
        {
            "archive_id": archive_id,
            "profile_id": profile_id,
            "scale_tier": scale_tier.value,
            "distribution_refs": list(distribution_refs),
        },
    )
    return WorkloadEnvelopeSpec(
        workload_id=workload_id,
        family_id="schema-profile-production-canary",
        version=1,
        inputs=(
            WorkloadInputRef(
                input_id=input_id,
                corpus_id=archive_id,
                profile_id=profile_id,
                scale_tier=scale_tier.value,
                distribution_refs=distribution_refs,
            ),
        ),
        phases=phases,
    )


def tool_pairing_canary_spec(
    *,
    profile_id: str,
    archive_id: str,
    scale_tier: WorkloadScaleTier = WorkloadScaleTier.CI_ACTIVATION,
) -> WorkloadEnvelopeSpec:
    """Declare profile-generated tool call/result pairing through actions."""
    return _schema_profile_canary_spec(
        workload_id="canary:tool-pairing-actions",
        profile_id=profile_id,
        archive_id=archive_id,
        phases=("generate", "parse", "materialize", "query", "quiescent"),
        distribution_refs=("relationships.tool_results", "index.action_shapes.tool_pairing"),
        scale_tier=scale_tier,
    )


def lineage_replay_canary_spec(
    *,
    profile_id: str,
    archive_id: str,
    scale_tier: WorkloadScaleTier = WorkloadScaleTier.CI_ACTIVATION,
) -> WorkloadEnvelopeSpec:
    """Declare profile-generated parent/child replay through composed reads."""
    return _schema_profile_canary_spec(
        workload_id="canary:lineage-replay-composition",
        profile_id=profile_id,
        archive_id=archive_id,
        phases=("generate", "parse", "materialize", "compose", "quiescent"),
        distribution_refs=("relationships.lineage", "index.topology"),
        scale_tier=scale_tier,
    )


def watcher_append_cohort_canary_spec(
    *,
    profile_id: str,
    archive_id: str,
    scale_tier: WorkloadScaleTier = WorkloadScaleTier.CI_ACTIVATION,
) -> WorkloadEnvelopeSpec:
    """Declare the watcher append/cohort memory incident production route."""
    route_phases = (
        "watcher_append:before",
        "raw_revision_replay_plan:before",
        "raw_revision_replay_plan:after",
        "watcher_append:after",
        "quiescent",
    )
    phases = tuple(f"{phase}:{scope.value}" for phase in route_phases for scope in MeasurementScope)
    spec = _schema_profile_canary_spec(
        workload_id="canary:watcher-append-cohort",
        profile_id=profile_id,
        archive_id=archive_id,
        phases=phases,
        distribution_refs=(
            "source.revisions_per_logical_source",
            "source.append_span",
            "operations.growing_sources",
        ),
        scale_tier=scale_tier,
    )
    return WorkloadEnvelopeSpec(
        workload_id=spec.workload_id,
        family_id=spec.family_id,
        version=spec.version,
        inputs=spec.inputs,
        phases=spec.phases,
        measurement_scope=MeasurementScope.CGROUP,
        quiescence_ms=0,
    )


def partial_convergence_canary_spec(
    *,
    profile_id: str,
    archive_id: str,
    scale_tier: WorkloadScaleTier = WorkloadScaleTier.CI_ACTIVATION,
) -> WorkloadEnvelopeSpec:
    """Declare ingest that leaves and then drains typed convergence debt."""
    return _schema_profile_canary_spec(
        workload_id="canary:partial-convergence-drain",
        profile_id=profile_id,
        archive_id=archive_id,
        phases=("generate", "ingest", "observe-debt", "converge", "query", "quiescent"),
        distribution_refs=("operations.convergence_debt", "operations.cursor_lag"),
        scale_tier=scale_tier,
    )


def raw_authority_fixed_point_spec(
    *,
    profile_id: str,
    archive_id: str,
    scale_tier: WorkloadScaleTier = WorkloadScaleTier.CI_ACTIVATION,
) -> WorkloadEnvelopeSpec:
    """Declare the bounded raw-authority census and replay fixed-point route.

    This is intentionally an envelope over the shared schema-derived corpus,
    rather than another synthetic archive format.  A production-shaped run
    binds its revision skew, component closure, and cursor/head conflict
    distributions to the same profile and receipt as the input corpus.
    """
    return _schema_profile_canary_spec(
        workload_id="canary:raw-authority-fixed-point",
        profile_id=profile_id,
        archive_id=archive_id,
        phases=("generate", "acquire", "census", "replay", "postflight", "quiescent"),
        distribution_refs=(
            "archive.raw_revision_shapes",
            "archive.authority_component_sizes",
            "operations.cursor_lag",
            "operations.convergence_debt",
        ),
        scale_tier=scale_tier,
    )


def _identity(namespace: str, payload: JSONDocument) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    normalized = unicodedata.normalize("NFC", encoded).encode("utf-8")
    return f"{namespace}:sha256:{hashlib.sha256(normalized).hexdigest()}"


__all__ = [
    "BudgetAggregation",
    "BudgetMeasure",
    "BudgetSemantics",
    "BudgetVerdict",
    "MeasurementScope",
    "WorkloadBudget",
    "WorkloadBudgetResult",
    "WorkloadEnvelopeSpec",
    "WorkloadInputRef",
    "WorkloadPhaseObservation",
    "WorkloadReceipt",
    "WorkloadRunStatus",
    "evaluate_budgets",
    "exact_session_actions_canary_spec",
    "lineage_replay_canary_spec",
    "partial_convergence_canary_spec",
    "raw_authority_fixed_point_spec",
    "tool_pairing_canary_spec",
    "watcher_append_cohort_canary_spec",
]
