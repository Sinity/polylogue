"""Domain-owned source cut/seal operation result."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.core.json import JSONDocument, json_document
from polylogue.operations.evidence import (
    EvidenceBinding,
    EvidenceValidationError,
    OperationEvidence,
    OperationResult,
    OperationStatus,
)


@dataclass(frozen=True, slots=True)
class SourceSealPayload:
    archive_identity: str
    source_snapshot: str
    source_inventory_ref: str
    topology_digest: str
    cut_policy: str

    def validate(self) -> None:
        if not all(
            (
                self.archive_identity,
                self.source_snapshot,
                self.source_inventory_ref,
                self.topology_digest,
                self.cut_policy,
            )
        ):
            raise EvidenceValidationError("source seal payload is incomplete")

    def validate_against_evidence(self, evidence: OperationEvidence) -> None:
        bindings = {binding.name: binding.value for binding in evidence.inputs}
        if bindings.get("archive") != self.archive_identity:
            raise EvidenceValidationError("source seal archive binding does not match its domain payload")
        if bindings.get("source-inventory") not in {None, self.source_inventory_ref}:
            raise EvidenceValidationError("source seal inventory binding does not match its domain payload")

    def to_document(self) -> JSONDocument:
        return json_document(
            {
                "archive_identity": self.archive_identity,
                "source_snapshot": self.source_snapshot,
                "source_inventory_ref": self.source_inventory_ref,
                "topology_digest": self.topology_digest,
                "cut_policy": self.cut_policy,
            }
        )


def source_seal_result(
    payload: SourceSealPayload,
    *,
    code_identity: str,
    package_identity: str,
    invocation_digest: str,
    inputs: tuple[EvidenceBinding, ...],
    outputs: tuple[EvidenceBinding, ...],
    status: OperationStatus = OperationStatus.SUCCEEDED,
) -> OperationResult[SourceSealPayload]:
    """Build the source owner's typed result through the common evidence seam."""

    return OperationResult.create(
        operation="source-seal",
        operation_version=1,
        code_identity=code_identity,
        package_identity=package_identity,
        invocation_digest=invocation_digest,
        inputs=inputs,
        outputs=outputs,
        status=status,
        payload=payload,
        failure=None
        if status is OperationStatus.SUCCEEDED
        else {"code": "source-seal-failed", "message": "source seal failed"},
    )


__all__ = ["SourceSealPayload", "source_seal_result"]
