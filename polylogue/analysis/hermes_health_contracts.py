"""Closed evidence report types for the Hermes integration diagnostic."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import ClassVar, Literal, cast

from pydantic import ConfigDict

HermesHealthVerdict = Literal["disabled", "healthy", "degraded", "unavailable"]


@dataclass(frozen=True, slots=True)
class HermesSourceStatus:
    """Freshness/cursor evidence for one discovered Hermes source file."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    source_ref: str
    source_class: Literal["state_db", "verification_evidence_db", "atof_stream", "atif_document", "other"]
    stage: str
    operational_state: str
    operational_reason: str
    parse_state: str
    byte_lag_bytes: int | None
    fts_converged: bool
    insights_converged: bool
    projection_error_count: int
    session_ref: str | None


@dataclass(frozen=True, slots=True)
class HermesParserFailure:
    """One file the dry-run explain pass could not parse or read."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    source_ref: str
    reason: str


@dataclass(frozen=True, slots=True)
class HermesFidelityCapabilityStatus:
    """Aggregated fidelity-capability status across discovered Hermes sources."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    capability: str
    status: str
    observed: int
    expected: int
    detail: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HermesLifecycleDebtSummary:
    """Lifecycle-event pairing debt (fs1.7) across sampled Hermes sessions."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    sessions_checked: int
    total_events: int
    unpaired_event_count: int
    unknown_message_reference_count: int
    caveats: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HermesDeliveryCorrelationSummary:
    """Context-delivery correlation state (fs1.11) across sampled Hermes sessions."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    sessions_checked: int
    events_checked: int
    available_count: int
    unavailable_count: int
    caveats: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HermesMeasurementCoverage:
    """What this rollup actually measured, and what it tried to and could not.

    The verdict is only as good as its inputs, so the inputs are reported
    beside it. Each probe contributes two distinct counters -- what it
    measured and what it could not -- rather than one number that folds an
    unreadable tier into the same zero as a genuinely empty one. This is the
    same discipline
    :class:`polylogue.analysis.measurement.outcome_coverage.ToolOutcomeAggregate`
    applies to tool outcomes, where ``unknown_n`` and ``no_result_n`` stay
    separate uncounted buckets and are never folded into the numerator.

    ``unmeasured_reasons`` names every probe that was attempted and did not
    produce a measurement. "Nothing in scope" is not an entry here: a Hermes
    root with no files and an archive with no Hermes hook events were both
    fully measured and found empty. Only a failure to look is recorded.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    sources_projected: int = 0
    sources_unprojected: int = 0
    lifecycle_sessions_sampled: int = 0
    lifecycle_sessions_unsampled: int = 0
    delivery_sessions_sampled: int = 0
    delivery_sessions_unsampled: int = 0
    unmeasured_reasons: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """True when every probe the rollup attempted returned a measurement."""

        return not self.unmeasured_reasons

    def merge(self, other: HermesMeasurementCoverage) -> HermesMeasurementCoverage:
        """Combine two probes' coverage, keeping distinct reasons in order."""

        return HermesMeasurementCoverage(
            sources_projected=self.sources_projected + other.sources_projected,
            sources_unprojected=self.sources_unprojected + other.sources_unprojected,
            lifecycle_sessions_sampled=self.lifecycle_sessions_sampled + other.lifecycle_sessions_sampled,
            lifecycle_sessions_unsampled=self.lifecycle_sessions_unsampled + other.lifecycle_sessions_unsampled,
            delivery_sessions_sampled=self.delivery_sessions_sampled + other.delivery_sessions_sampled,
            delivery_sessions_unsampled=self.delivery_sessions_unsampled + other.delivery_sessions_unsampled,
            unmeasured_reasons=tuple(dict.fromkeys((*self.unmeasured_reasons, *other.unmeasured_reasons))),
        )


@dataclass(frozen=True, slots=True)
class HermesIntegrationHealth:
    """One bounded, composed Hermes integration health rollup."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    checked_at: str
    enabled: bool
    enabled_reason: str
    verdict: HermesHealthVerdict
    sources: tuple[HermesSourceStatus, ...] = ()
    parser_failures: tuple[HermesParserFailure, ...] = ()
    fidelity_capabilities: tuple[HermesFidelityCapabilityStatus, ...] = ()
    convergence_debt_failed_count: int = 0
    convergence_debt_deferred_count: int = 0
    convergence_debt_retry_due_count: int = 0
    lifecycle_debt: HermesLifecycleDebtSummary = field(
        default_factory=lambda: HermesLifecycleDebtSummary(0, 0, 0, 0, ())
    )
    delivery_correlation: HermesDeliveryCorrelationSummary = field(
        default_factory=lambda: HermesDeliveryCorrelationSummary(0, 0, 0, 0, ())
    )
    measurement_coverage: HermesMeasurementCoverage = field(default_factory=lambda: HermesMeasurementCoverage())
    caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return cast("dict[str, object]", _jsonable(self))


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (list, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value
