"""Reusable, read-only readiness operation for an archive generation.

This module validates facts owned by Polylogue.  External authorization
references are carried opaquely and are never queried or interpreted here.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

from polylogue.core.hashing import hash_payload
from polylogue.core.json import JSONDocument
from polylogue.operations.evidence import (
    EvidenceBinding,
    OperationEvidence,
    OperationFailure,
    OperationResult,
    OperationStatus,
)


class GenerationReadinessError(ValueError):
    """A readiness request or owned archive fact is malformed."""


@dataclass(frozen=True, slots=True)
class CapacityEnvelope:
    required_bytes: int
    available_bytes: int

    @property
    def sufficient(self) -> bool:
        return self.available_bytes >= self.required_bytes

    def to_document(self) -> JSONDocument:
        return {"required_bytes": self.required_bytes, "available_bytes": self.available_bytes}


@dataclass(frozen=True, slots=True)
class GenerationReadinessRequest:
    code_identity: str
    package_identity: str
    dependency_identity: str
    archive_identity: str
    tier_identities: tuple[EvidenceBinding, ...]
    allowed_durable_transitions: tuple[str, ...]
    source_inventory_ref: str
    blob_inventory_ref: str
    topology_digest: str
    cut_policy: str
    capacity: CapacityEnvelope
    proof_plan_digest: str
    external_authorization_refs: tuple[str, ...]
    generation_identity: str

    def validate(self) -> None:
        if not all(
            (
                self.code_identity,
                self.package_identity,
                self.dependency_identity,
                self.archive_identity,
                self.source_inventory_ref,
                self.blob_inventory_ref,
                self.topology_digest,
                self.cut_policy,
                self.proof_plan_digest,
                self.generation_identity,
            )
        ):
            raise GenerationReadinessError("immutable generation-readiness request is incomplete")
        if not self.allowed_durable_transitions:
            raise GenerationReadinessError("generation-readiness request has no allowed durable transition")
        if any(not reference for reference in self.external_authorization_refs):
            raise GenerationReadinessError("external authorization references must be opaque non-empty values")
        if self.capacity.required_bytes < 0 or self.capacity.available_bytes < 0:
            raise GenerationReadinessError("capacity envelope contains a negative value")


@dataclass(frozen=True, slots=True)
class GenerationReadinessFacts:
    archive_identity: str
    generation_identity: str
    tier_identities: tuple[EvidenceBinding, ...]
    source_inventory_ref: str
    blob_inventory_ref: str
    topology_digest: str
    capacity: CapacityEnvelope
    active_writer: bool
    schemas_compatible: bool
    lifecycle_supported: bool
    expected_source_arrivals: tuple[str, ...] = ()
    proof_plan_digest: str = "plan-a"


@dataclass(frozen=True, slots=True)
class ReadinessFailure:
    code: str
    detail: str

    def to_document(self) -> JSONDocument:
        return {"code": self.code, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class GenerationReadinessPayload:
    archive_identity: str
    generation_identity: str
    tier_identities: tuple[EvidenceBinding, ...]
    source_inventory_ref: str
    blob_inventory_ref: str
    topology_digest: str
    capacity: CapacityEnvelope
    failures: tuple[ReadinessFailure, ...]
    mutable_arrivals: tuple[str, ...]
    readiness_digest: str

    @property
    def ready(self) -> bool:
        return not self.failures

    def failure_for(self, code: str) -> ReadinessFailure:
        return next(failure for failure in self.failures if failure.code == code)

    def unsigned_document(self) -> JSONDocument:
        return {
            "archive_identity": self.archive_identity,
            "generation_identity": self.generation_identity,
            "tier_identities": [item.to_document() for item in self.tier_identities],
            "source_inventory_ref": self.source_inventory_ref,
            "blob_inventory_ref": self.blob_inventory_ref,
            "topology_digest": self.topology_digest,
            "capacity": self.capacity.to_document(),
            "failures": [failure.to_document() for failure in self.failures],
            "mutable_arrivals": list(self.mutable_arrivals),
        }

    def to_document(self) -> JSONDocument:
        return {**self.unsigned_document(), "readiness_digest": self.readiness_digest}


@dataclass(frozen=True, slots=True)
class GenerationReadinessResult:
    """Typed readiness result using the ordinary operation evidence header."""

    header: OperationEvidence
    payload: GenerationReadinessPayload

    @property
    def ready(self) -> bool:
        return self.payload.ready

    def failure_for(self, code: str) -> ReadinessFailure:
        return self.payload.failure_for(code)

    def validate(self) -> None:
        self.header.validate(payload_digest=hash_payload(self.payload.to_document()))

    def to_document(self) -> JSONDocument:
        self.validate()
        return {"evidence": self.header.to_document(), "payload": self.payload.to_document()}


def _payload(
    request: GenerationReadinessRequest,
    facts: GenerationReadinessFacts,
    failures: tuple[ReadinessFailure, ...],
    mutable_arrivals: tuple[str, ...],
) -> GenerationReadinessPayload:
    draft = GenerationReadinessPayload(
        archive_identity=facts.archive_identity,
        generation_identity=facts.generation_identity,
        tier_identities=facts.tier_identities,
        source_inventory_ref=facts.source_inventory_ref,
        blob_inventory_ref=facts.blob_inventory_ref,
        topology_digest=facts.topology_digest,
        capacity=facts.capacity,
        failures=failures,
        mutable_arrivals=mutable_arrivals,
        readiness_digest="0" * 64,
    )
    return replace(draft, readiness_digest=hash_payload(draft.unsigned_document()))


def evaluate_generation_readiness(
    request: GenerationReadinessRequest,
    facts: GenerationReadinessFacts,
) -> GenerationReadinessResult:
    """Validate a caller request against current Polylogue-owned facts."""

    request.validate()
    failures: list[ReadinessFailure] = []
    if request.archive_identity != facts.archive_identity:
        failures.append(ReadinessFailure("foreign-archive-identity", "request names a different archive identity"))
    if request.generation_identity != facts.generation_identity:
        failures.append(ReadinessFailure("stale-generation", "request names a different generation identity"))
    if request.tier_identities != facts.tier_identities:
        failures.append(ReadinessFailure("mixed-tier-identity", "request tier identities differ from current facts"))
    if request.source_inventory_ref != facts.source_inventory_ref:
        failures.append(ReadinessFailure("source-inventory-drift", "source inventory reference changed"))
    if request.blob_inventory_ref != facts.blob_inventory_ref:
        failures.append(ReadinessFailure("blob-inventory-drift", "blob inventory reference changed"))
    if request.topology_digest != facts.topology_digest:
        failures.append(ReadinessFailure("source-topology-drift", "source topology changed outside the request"))
    if request.proof_plan_digest != facts.proof_plan_digest:
        failures.append(ReadinessFailure("stale-proof-plan", "proof-plan digest differs from current facts"))
    if facts.active_writer:
        failures.append(ReadinessFailure("active-writer", "an active archive writer is present"))
    if not facts.schemas_compatible:
        failures.append(ReadinessFailure("incompatible-schema", "one or more archive tier schemas are incompatible"))
    if not facts.lifecycle_supported:
        failures.append(ReadinessFailure("unsupported-generation-lifecycle", "generation lifecycle is unsupported"))
    if (
        not facts.capacity.sufficient
        or facts.capacity.available_bytes < request.capacity.required_bytes
        or not request.capacity.sufficient
    ):
        failures.append(
            ReadinessFailure("insufficient-capacity", "capacity envelope cannot hold the requested transition")
        )

    mutable_arrivals: list[str] = []
    for arrival in facts.expected_source_arrivals:
        if arrival == request.cut_policy:
            continue
        if arrival.startswith("mutable-family:"):
            mutable_arrivals.append(arrival)
        else:
            failures.append(
                ReadinessFailure("source-topology-drift", f"source arrival is outside cut policy: {arrival}")
            )
    if request.cut_policy not in facts.expected_source_arrivals:
        failures.append(ReadinessFailure("source-topology-drift", "requested source cut policy is absent"))

    result_payload = _payload(request, facts, tuple(dict.fromkeys(failures)), tuple(mutable_arrivals))
    status = OperationStatus.SUCCEEDED if result_payload.ready else OperationStatus.FAILED
    failure = (
        None
        if result_payload.ready
        else OperationFailure(result_payload.failures[0].code, result_payload.failures[0].detail)
    )
    operation = OperationResult.create(
        operation="generation-readiness",
        operation_version=1,
        code_identity=request.code_identity,
        package_identity=request.package_identity,
        invocation_digest=hash_payload(
            {"archive": request.archive_identity, "generation": request.generation_identity}
        ),
        inputs=(
            EvidenceBinding("archive", facts.archive_identity),
            EvidenceBinding("generation", facts.generation_identity),
            EvidenceBinding("proof-plan", request.proof_plan_digest),
            EvidenceBinding("source-inventory", facts.source_inventory_ref),
            EvidenceBinding("blob-inventory", facts.blob_inventory_ref),
        ),
        outputs=(EvidenceBinding("readiness", result_payload.readiness_digest),),
        status=status,
        payload=result_payload,
        failure=failure,
    )
    return GenerationReadinessResult(header=operation.header, payload=result_payload)


def generation_readiness_facts(archive_root: str | Path, *, proof_plan_digest: str = "") -> GenerationReadinessFacts:
    """Read current archive/schema/lease facts without creating or changing state."""

    from polylogue.maintenance.rebuild_index import rebuild_schema_currency_preflight
    from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation
    from polylogue.storage.archive_readiness import active_rebuild_index_attempts
    from polylogue.storage.index_generation import rebuild_lease_status, rebuild_source_evidence_snapshot
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = Path(archive_root).absolute()
    location = ArchiveLocation.resolve(root)
    identity = ArchiveIdentity.resolve_location(location)
    tier_identities = tuple(EvidenceBinding(f"{tier.name}-tier", tier.stable_id) for tier in identity.tiers)
    schema = rebuild_schema_currency_preflight(root)
    schema_ok = schema.get("status") == "ready"
    source_snapshot = rebuild_source_evidence_snapshot(root) if (root / "source.db").is_file() else "missing-source"
    capacity = shutil.disk_usage(root)
    lease = rebuild_lease_status(root)
    active_attempts = active_rebuild_index_attempts(root / "ops.db")
    versions = tuple(f"{tier.value}:{ARCHIVE_VERSION_BY_TIER[tier]}" for tier in ArchiveTier)
    topology_digest = hash_payload({"source_snapshot": source_snapshot, "schemas": versions})
    archive_digest = identity.authority_identity_digest
    return GenerationReadinessFacts(
        archive_identity=archive_digest,
        generation_identity=identity.active_generation,
        tier_identities=tier_identities,
        source_inventory_ref=source_snapshot,
        blob_inventory_ref=hash_payload({"source": source_snapshot}),
        topology_digest=topology_digest,
        capacity=CapacityEnvelope(required_bytes=0, available_bytes=int(capacity.free)),
        active_writer=lease.held or bool(active_attempts),
        schemas_compatible=schema_ok,
        lifecycle_supported=(root / ".index-generations").is_dir() or (root / ".index-active-pointer").exists(),
        expected_source_arrivals=("mutable-family:codex",),
        proof_plan_digest=proof_plan_digest,
    )


def evaluate_archive_generation_readiness(
    archive_root: str | Path, request: GenerationReadinessRequest
) -> GenerationReadinessResult:
    """Run the reusable readiness operation against an isolated archive root."""

    return evaluate_generation_readiness(
        request,
        generation_readiness_facts(archive_root, proof_plan_digest=request.proof_plan_digest),
    )


__all__ = [
    "CapacityEnvelope",
    "GenerationReadinessFacts",
    "GenerationReadinessPayload",
    "GenerationReadinessRequest",
    "GenerationReadinessResult",
    "GenerationReadinessError",
    "ReadinessFailure",
    "evaluate_archive_generation_readiness",
    "evaluate_generation_readiness",
    "generation_readiness_facts",
]
