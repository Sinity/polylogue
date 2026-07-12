"""Private, descriptive delegation packet compilation for the Fable campaign.

The compiler is intentionally evidence-first and emits ``not_supported`` when
the supplied structural/annotation material cannot support a private descriptive
packet.  It produces no comparative, utility, routing-quality, or sentiment
claim.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from polylogue.insights.cohorts import CohortManifest

PacketStatus = Literal["complete", "not_supported"]


@dataclass(frozen=True)
class DelegationPacketRow:
    """Bounded structural evidence needed by the descriptive packet."""

    delegation_ref: str
    evidence_basis: Literal["action", "edge"]
    mapping_state: str
    instruction_sha256: str | None


@dataclass(frozen=True)
class DelegationPacketLabel:
    """One accepted or candidate descriptive annotation with evidence spans."""

    delegation_ref: str
    field: str
    value: str | None
    batch_id: str
    accepted: bool
    applicable: bool | None
    confidence: float | None
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class DescriptiveDistribution:
    """One accepted-label distribution with explicit denominator/missingness."""

    field: str
    value: str
    count: int
    proportion: float
    denominator_n: int
    missing_n: int


@dataclass(frozen=True)
class FableDelegationPacket:
    """A private descriptive packet or a concrete fail-closed explanation."""

    status: PacketStatus
    manifest_id: str
    population_count: int
    action_observed_count: int
    edge_only_count: int
    unresolved_count: int
    selected_refs: tuple[str, ...]
    annotation_schema_id: str | None
    annotation_batches: tuple[str, ...]
    distributions: tuple[DescriptiveDistribution, ...]
    disagreement_count: int
    specimen_refs: tuple[str, ...]
    counterexample_refs: tuple[str, ...]
    limits: tuple[str, ...]
    not_supported_reasons: tuple[str, ...] = ()


def _unsupported(
    manifest: CohortManifest,
    rows: Sequence[DelegationPacketRow],
    reasons: Sequence[str],
) -> FableDelegationPacket:
    return FableDelegationPacket(
        status="not_supported",
        manifest_id=manifest.manifest_id,
        population_count=len(rows),
        action_observed_count=sum(row.evidence_basis == "action" for row in rows),
        edge_only_count=sum(row.evidence_basis == "edge" for row in rows),
        unresolved_count=sum(row.mapping_state == "unresolved" for row in rows),
        selected_refs=manifest.selected_refs,
        annotation_schema_id=None,
        annotation_batches=(),
        distributions=(),
        disagreement_count=0,
        specimen_refs=(),
        counterexample_refs=(),
        limits=("private_descriptive_only",),
        not_supported_reasons=tuple(sorted(set(reasons))),
    )


def compile_private_fable_packet(
    *,
    manifest: CohortManifest,
    rows: Sequence[DelegationPacketRow],
    annotation_schema_id: str | None,
    labels: Sequence[DelegationPacketLabel],
) -> FableDelegationPacket:
    """Compile a private descriptive packet or fail closed with named gaps.

    Accepted labels must target sampled, action-observed rows and retain at
    least one evidence ref. Edge-only and unresolved rows are coverage facts,
    never rhetorical evidence.  Distributions are per field over applicable
    accepted labels, retaining their denominator and missing label count.
    """

    by_ref = {row.delegation_ref: row for row in rows}
    reasons: list[str] = []
    if annotation_schema_id is None:
        reasons.append("missing_annotation_schema")
    if not manifest.selected_refs:
        reasons.append("empty_deterministic_sample")
    missing_sample_refs = sorted(set(manifest.selected_refs) - by_ref.keys())
    if missing_sample_refs:
        reasons.append("selected_refs_missing_from_structural_population")
    action_rows = {ref: row for ref, row in by_ref.items() if row.evidence_basis == "action"}
    if not action_rows:
        reasons.append("no_action_observed_delegation_attempts")

    accepted = [label for label in labels if label.accepted]
    if not accepted:
        reasons.append("no_accepted_labels")
    for label in accepted:
        if label.delegation_ref not in action_rows:
            reasons.append("accepted_label_not_action_observed")
        if label.delegation_ref not in manifest.selected_refs:
            reasons.append("accepted_label_outside_deterministic_sample")
        if not label.evidence_refs:
            reasons.append("accepted_label_missing_evidence")
    if reasons:
        return _unsupported(manifest, rows, reasons)

    labels_by_field: dict[str, list[DelegationPacketLabel]] = defaultdict(list)
    for label in accepted:
        labels_by_field[label.field].append(label)
    distributions: list[DescriptiveDistribution] = []
    disagreement_count = 0
    specimen_refs: set[str] = set()
    counterexample_refs: set[str] = set()
    for field, field_labels in sorted(labels_by_field.items()):
        counterexample_refs.update(label.delegation_ref for label in field_labels if label.applicable is False)
        applicable = [label for label in field_labels if label.applicable is not False]
        denominator = len(applicable)
        missing = sum(label.value is None for label in applicable)
        values = Counter(label.value for label in applicable if label.value is not None)
        for value, count in sorted(values.items()):
            assert value is not None
            distributions.append(
                DescriptiveDistribution(
                    field=field,
                    value=value,
                    count=count,
                    proportion=count / denominator if denominator else 0.0,
                    denominator_n=denominator,
                    missing_n=missing,
                )
            )
        labels_by_ref: dict[str, set[str]] = defaultdict(set)
        for label in applicable:
            if label.value is not None:
                labels_by_ref[label.delegation_ref].add(label.value)
                specimen_refs.add(label.delegation_ref)
        disagreement_count += sum(len(values) > 1 for values in labels_by_ref.values())

    return FableDelegationPacket(
        status="complete",
        manifest_id=manifest.manifest_id,
        population_count=len(rows),
        action_observed_count=len(action_rows),
        edge_only_count=sum(row.evidence_basis == "edge" for row in rows),
        unresolved_count=sum(row.mapping_state == "unresolved" for row in rows),
        selected_refs=manifest.selected_refs,
        annotation_schema_id=annotation_schema_id,
        annotation_batches=tuple(sorted({label.batch_id for label in accepted})),
        distributions=tuple(distributions),
        disagreement_count=disagreement_count,
        specimen_refs=tuple(sorted(specimen_refs)),
        counterexample_refs=tuple(sorted(counterexample_refs)),
        limits=(
            "private_descriptive_only",
            "no_comparative_authoritarianism_success_utility_or_routing_quality_claims",
            "edge_only_and_unresolved_rows_excluded_from_rhetoric_denominators",
        ),
    )


__all__ = [
    "DelegationPacketLabel",
    "DelegationPacketRow",
    "DescriptiveDistribution",
    "FableDelegationPacket",
    "compile_private_fable_packet",
]
