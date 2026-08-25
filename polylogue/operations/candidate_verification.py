"""Domain-owned inactive-candidate verification result."""

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
class CandidateVerificationPayload:
    archive_identity: str
    generation_identity: str
    candidate_digest: str
    semantic_digest: str
    publication_state: str

    def validate(self) -> None:
        if self.publication_state != "inactive":
            raise EvidenceValidationError("candidate verification must bind an inactive candidate")
        if not all((self.archive_identity, self.generation_identity, self.candidate_digest, self.semantic_digest)):
            raise EvidenceValidationError("candidate verification payload is incomplete")

    def validate_against_evidence(self, evidence: OperationEvidence) -> None:
        bindings = {binding.name: binding.value for binding in evidence.inputs}
        outputs = {binding.name: binding.value for binding in evidence.outputs}
        if bindings.get("archive") != self.archive_identity:
            raise EvidenceValidationError("candidate verification archive binding does not match its domain payload")
        if bindings.get("generation") != self.generation_identity:
            raise EvidenceValidationError("candidate verification generation binding does not match its domain payload")
        if outputs.get("candidate") not in {None, self.candidate_digest}:
            raise EvidenceValidationError("candidate verification output does not match its domain payload")

    def to_document(self) -> JSONDocument:
        return json_document(
            {
                "archive_identity": self.archive_identity,
                "generation_identity": self.generation_identity,
                "candidate_digest": self.candidate_digest,
                "semantic_digest": self.semantic_digest,
                "publication_state": self.publication_state,
            }
        )


def candidate_verification_result(
    payload: CandidateVerificationPayload,
    *,
    code_identity: str,
    package_identity: str,
    invocation_digest: str,
    inputs: tuple[EvidenceBinding, ...],
    outputs: tuple[EvidenceBinding, ...],
) -> OperationResult[CandidateVerificationPayload]:
    return OperationResult.create(
        operation="candidate-verification",
        operation_version=1,
        code_identity=code_identity,
        package_identity=package_identity,
        invocation_digest=invocation_digest,
        inputs=inputs,
        outputs=outputs,
        status=OperationStatus.SUCCEEDED,
        payload=payload,
    )


__all__ = ["CandidateVerificationPayload", "candidate_verification_result"]
