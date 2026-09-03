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
from hashlib import sha256
from typing import Literal

from polylogue.analysis.cohorts import CohortCandidate, CohortManifest, CohortSpec, compile_cohort_manifest
from polylogue.archive.query.predicate import QueryBoolPredicate, QueryFieldPredicate, QueryFieldRef
from polylogue.core.refs import ObjectRef, delegation_edge_object_id
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveDelegationQueryRow, ArchiveStore

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
    adjudication_counts: tuple[tuple[str, int], ...]
    specimen_refs: tuple[str, ...]
    counterexample_refs: tuple[str, ...]
    limits: tuple[str, ...]
    not_supported_reasons: tuple[str, ...] = ()
    manifest: CohortManifest | None = None
    label_evidence_refs: tuple[tuple[str, tuple[str, ...]], ...] = ()
    aggregate_evidence_refs: tuple[str, ...] = ()


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
        adjudication_counts=(),
        specimen_refs=(),
        counterexample_refs=(),
        limits=("private_descriptive_only",),
        not_supported_reasons=tuple(sorted(set(reasons))),
        manifest=manifest,
    )


def compile_private_fable_packet(
    *,
    manifest: CohortManifest,
    rows: Sequence[DelegationPacketRow],
    annotation_schema_id: str | None,
    labels: Sequence[DelegationPacketLabel],
    resolved_evidence_refs: frozenset[str] | None = None,
    adjudication_counts: tuple[tuple[str, int], ...] = (),
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
        if resolved_evidence_refs is not None:
            unresolved = sorted(set(label.evidence_refs) - resolved_evidence_refs)
            if unresolved:
                reasons.append("accepted_label_evidence_not_resolved")
    if reasons:
        return _unsupported(manifest, rows, reasons)

    labels_by_field: dict[str, list[DelegationPacketLabel]] = defaultdict(list)
    for label in accepted:
        labels_by_field[label.field].append(label)
    distributions: list[DescriptiveDistribution] = []
    disagreement_count = 0
    specimen_refs: set[str] = set()
    counterexample_refs: set[str] = set()
    label_evidence_refs: list[tuple[str, tuple[str, ...]]] = []
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
            label_evidence_refs.append((f"{label.delegation_ref}:{field}", tuple(sorted(label.evidence_refs))))
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
        adjudication_counts=adjudication_counts,
        specimen_refs=tuple(sorted(specimen_refs)),
        counterexample_refs=tuple(sorted(counterexample_refs)),
        limits=(
            "private_descriptive_only",
            "no_comparative_authoritarianism_success_utility_or_routing_quality_claims",
            "edge_only_and_unresolved_rows_excluded_from_rhetoric_denominators",
        ),
        manifest=manifest,
        label_evidence_refs=tuple(sorted(label_evidence_refs)),
        aggregate_evidence_refs=tuple(sorted(specimen_refs)),
    )


def _delegation_ref(row: ArchiveDelegationQueryRow) -> str:
    instruction_block_id = row.instruction_tool_use_block_id
    if instruction_block_id is not None:
        return f"delegation:{instruction_block_id}"
    if row.child_session_id is None:
        raise ValueError("edge-only delegation rows require a child session id")
    return f"delegation:{delegation_edge_object_id(row.parent_session_id, row.child_session_id)}"


def regenerate_private_fable_packet(
    archive: ArchiveStore,
    *,
    seed: str,
    requested_size: int,
    schema_id: str = "delegation.discourse",
    schema_version: int = 1,
    exact_template_cap: int = 1,
) -> FableDelegationPacket:
    """Cold-regenerate a private packet from canonical archive evidence.

    This is intentionally a read-only composition of the canonical delegation
    relation and the durable annotation substrate. Missing schema, batches, or
    active labels flows into the compiler's explicit ``not_supported`` result.
    """

    max_population = 100_000
    all_rows = archive.query_delegations(QueryBoolPredicate("and", ()), limit=max_population + 1)
    population_truncated = len(all_rows) > max_population
    all_rows = all_rows[:max_population]
    packet_rows = tuple(
        DelegationPacketRow(
            delegation_ref=_delegation_ref(row),
            evidence_basis="action" if row.instruction_tool_use_block_id is not None else "edge",
            mapping_state=row.mapping_state,
            instruction_sha256=(
                sha256(row.instruction_payload.encode("utf-8")).hexdigest()
                if row.instruction_payload is not None
                else None
            ),
        )
        for row in all_rows
    )
    cursor = f"index:{archive.index_db_path.stat().st_mtime_ns}"
    manifest = compile_cohort_manifest(
        CohortSpec(
            population_query="delegations where basis:action",
            archive_cursor=cursor,
            seed=seed,
            requested_size=requested_size,
            strata=("origin", "dispatch_model"),
            exact_template_cap=exact_template_cap,
        ),
        tuple(
            CohortCandidate(
                object_ref=_delegation_ref(row),
                dimensions={"origin": row.parent_origin, "dispatch_model": row.dispatch_turn_model},
                template_key=(
                    sha256(row.instruction_payload.encode("utf-8")).hexdigest() if row.instruction_payload else None
                ),
                exclusion_reason=None if row.instruction_tool_use_block_id is not None else "edge_only",
            )
            for row in all_rows
        ),
    )
    schema = archive.get_annotation_schema(schema_id, schema_version)
    assertions = archive.query_assertions(
        QueryFieldPredicate(
            field="kind",
            values=("annotation",),
            field_ref=QueryFieldRef(scope="unit", name="kind", source_name="assertions", unit="assertion"),
        ),
        limit=100_000,
    )
    labels: list[DelegationPacketLabel] = []
    referenced_batch_ids: set[str] = set()
    evidence_refs: set[str] = set()
    adjudication_counts: Counter[str] = Counter()
    qualified_schema_id = f"{schema_id}@v{schema_version}"
    for assertion in assertions:
        value = assertion.value
        if not isinstance(value, dict) or value.get("_schema") != qualified_schema_id:
            continue
        status = getattr(assertion.status, "value", assertion.status)
        adjudication_counts[str(status)] += 1
        if status != "active":
            continue
        batch_id = assertion.scope_ref.removeprefix("annotation-batch:") if assertion.scope_ref else "unbatched"
        referenced_batch_ids.add(batch_id)
        evidence_refs.update(assertion.evidence_refs)
        applicable_value = value.get("applicable")
        confidence_value = value.get("confidence")
        for field, field_value in value.items():
            if field.startswith("_") or field in {"applicable", "confidence", "abstain"}:
                continue
            labels.append(
                DelegationPacketLabel(
                    delegation_ref=assertion.target_ref,
                    field=field,
                    value=field_value if isinstance(field_value, str) else None,
                    batch_id=batch_id,
                    accepted=True,
                    applicable=applicable_value if isinstance(applicable_value, bool) else None,
                    confidence=float(confidence_value) if isinstance(confidence_value, (int, float)) else None,
                    evidence_refs=assertion.evidence_refs,
                )
            )
    reasons: list[str] = []
    if population_truncated:
        reasons.append("population_scan_truncated")
    for batch_id in sorted(referenced_batch_ids):
        batch = None if batch_id == "unbatched" else archive.get_annotation_batch(batch_id)
        if batch is None:
            reasons.append("annotation_batch_metadata_missing")
        elif (
            batch.schema_id != schema_id
            or batch.schema_version != schema_version
            or batch.target_ref not in {label.delegation_ref for label in labels if label.batch_id == batch_id}
        ):
            reasons.append("annotation_batch_metadata_mismatch")
    if reasons:
        return _unsupported(manifest, packet_rows, reasons)

    resolved_evidence = frozenset(ref for ref in evidence_refs if _evidence_ref_resolves(archive, ref))
    return compile_private_fable_packet(
        manifest=manifest,
        rows=packet_rows,
        annotation_schema_id=schema.schema.qualified_id if schema is not None else None,
        labels=labels,
        resolved_evidence_refs=resolved_evidence,
        adjudication_counts=tuple(sorted(adjudication_counts.items())),
    )


def _evidence_ref_resolves(archive: ArchiveStore, ref: str) -> bool:
    """Resolve a bounded label citation against canonical index evidence."""
    try:
        parsed = ObjectRef.parse(ref)
    except ValueError:
        # EvidenceRef is the compact session::message::position form.
        try:
            parts = ref.split("::")
            if len(parts) == 1:
                return (
                    archive._conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", parts).fetchone() is not None
                )
            if len(parts) == 2:
                return (
                    archive._conn.execute(
                        "SELECT 1 FROM messages WHERE message_id = ? AND session_id = ?", (parts[1], parts[0])
                    ).fetchone()
                    is not None
                )
            if len(parts) == 3:
                return (
                    archive._conn.execute(
                        "SELECT 1 FROM blocks WHERE message_id = ? AND position = ?", (parts[1], int(parts[2]))
                    ).fetchone()
                    is not None
                )
        except (ValueError, TypeError):
            return False
        return False
    if parsed.kind == "session":
        return (
            archive._conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (parsed.object_id,)).fetchone()
            is not None
        )
    if parsed.kind == "message":
        return (
            archive._conn.execute("SELECT 1 FROM messages WHERE message_id = ?", (parsed.object_id,)).fetchone()
            is not None
        )
    if parsed.kind == "block":
        if parsed.qualifiers:
            try:
                position = int(parsed.qualifiers[0])
            except ValueError:
                return False
            return (
                archive._conn.execute(
                    "SELECT 1 FROM blocks WHERE message_id = ? AND position = ?", (parsed.object_id, position)
                ).fetchone()
                is not None
            )
        return (
            archive._conn.execute("SELECT 1 FROM blocks WHERE block_id = ?", (parsed.object_id,)).fetchone() is not None
        )
    return False


__all__ = [
    "DelegationPacketLabel",
    "DelegationPacketRow",
    "DescriptiveDistribution",
    "FableDelegationPacket",
    "compile_private_fable_packet",
    "regenerate_private_fable_packet",
]
