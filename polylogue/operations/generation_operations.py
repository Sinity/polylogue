"""Typed results for the generation lifecycle operation families.

Each payload below is owned by its operation family.  Only the immutable
evidence header is shared with source sealing and candidate verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from polylogue.core.json import JSONDocument, json_document
from polylogue.operations.evidence import EvidenceBinding, EvidenceValidationError, OperationResult, OperationStatus


def _require(values: tuple[str, ...], label: str) -> None:
    if not all(values):
        raise EvidenceValidationError(f"{label} result payload is incomplete")


@dataclass(frozen=True, slots=True)
class CandidateBuildPayload:
    archive_identity: str
    generation_identity: str
    source_snapshot: str
    candidate_digest: str
    publication_state: str

    def validate(self) -> None:
        _require(
            (self.archive_identity, self.generation_identity, self.source_snapshot, self.candidate_digest),
            "candidate build",
        )
        if self.publication_state != "inactive":
            raise EvidenceValidationError("candidate build must produce an inactive generation")

    def to_document(self) -> JSONDocument:
        return json_document(
            {
                "archive_identity": self.archive_identity,
                "generation_identity": self.generation_identity,
                "source_snapshot": self.source_snapshot,
                "candidate_digest": self.candidate_digest,
                "publication_state": self.publication_state,
            }
        )


@dataclass(frozen=True, slots=True)
class SemanticVerificationPayload:
    archive_identity: str
    generation_identity: str
    candidate_digest: str
    comparison_digest: str
    unexpected_difference_count: int

    def validate(self) -> None:
        _require(
            (self.archive_identity, self.generation_identity, self.candidate_digest, self.comparison_digest),
            "semantic verification",
        )
        if self.unexpected_difference_count < 0:
            raise EvidenceValidationError("semantic verification difference count is negative")

    def to_document(self) -> JSONDocument:
        return json_document(
            {
                "archive_identity": self.archive_identity,
                "generation_identity": self.generation_identity,
                "candidate_digest": self.candidate_digest,
                "comparison_digest": self.comparison_digest,
                "unexpected_difference_count": self.unexpected_difference_count,
            }
        )


@dataclass(frozen=True, slots=True)
class DurableTransitionPayload:
    archive_identity: str
    generation_identity: str
    allowed_transition: str
    precondition_digest: str
    transition_digest: str

    def validate(self) -> None:
        _require(
            (
                self.archive_identity,
                self.generation_identity,
                self.allowed_transition,
                self.precondition_digest,
                self.transition_digest,
            ),
            "durable transition",
        )

    def to_document(self) -> JSONDocument:
        return json_document(
            {
                "archive_identity": self.archive_identity,
                "generation_identity": self.generation_identity,
                "allowed_transition": self.allowed_transition,
                "precondition_digest": self.precondition_digest,
                "transition_digest": self.transition_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class ActivationPayload:
    archive_identity: str
    generation_identity: str
    active_generation_identity: str
    publication_digest: str

    def validate(self) -> None:
        _require(
            (self.archive_identity, self.generation_identity, self.active_generation_identity, self.publication_digest),
            "activation",
        )
        if self.generation_identity != self.active_generation_identity:
            raise EvidenceValidationError("activation output is not the requested generation")

    def to_document(self) -> JSONDocument:
        return json_document(
            {
                "archive_identity": self.archive_identity,
                "generation_identity": self.generation_identity,
                "active_generation_identity": self.active_generation_identity,
                "publication_digest": self.publication_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class PostflightPayload:
    archive_identity: str
    generation_identity: str
    active_generation_identity: str
    query_digest: str
    restart_digest: str

    def validate(self) -> None:
        _require(
            (
                self.archive_identity,
                self.generation_identity,
                self.active_generation_identity,
                self.query_digest,
                self.restart_digest,
            ),
            "postflight",
        )
        if self.generation_identity != self.active_generation_identity:
            raise EvidenceValidationError("postflight observes a different active generation")

    def to_document(self) -> JSONDocument:
        return json_document(
            {
                "archive_identity": self.archive_identity,
                "generation_identity": self.generation_identity,
                "active_generation_identity": self.active_generation_identity,
                "query_digest": self.query_digest,
                "restart_digest": self.restart_digest,
            }
        )


def _result(
    operation: str,
    payload: object,
    *,
    code_identity: str,
    package_identity: str,
    invocation_digest: str,
    inputs: tuple[EvidenceBinding, ...],
    outputs: tuple[EvidenceBinding, ...],
    status: OperationStatus = OperationStatus.SUCCEEDED,
) -> OperationResult[object]:
    return OperationResult.create(
        operation=operation,
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
        else {"code": f"{operation}-failed", "message": f"{operation} failed"},
    )


def candidate_build_result(payload: CandidateBuildPayload, **kwargs: Any) -> OperationResult[CandidateBuildPayload]:
    return cast(OperationResult[CandidateBuildPayload], _result("candidate-build", payload, **kwargs))


def semantic_verification_result(
    payload: SemanticVerificationPayload, **kwargs: Any
) -> OperationResult[SemanticVerificationPayload]:
    return cast(OperationResult[SemanticVerificationPayload], _result("semantic-verification", payload, **kwargs))


def durable_transition_result(
    payload: DurableTransitionPayload, **kwargs: Any
) -> OperationResult[DurableTransitionPayload]:
    return cast(OperationResult[DurableTransitionPayload], _result("durable-transition", payload, **kwargs))


def activation_result(payload: ActivationPayload, **kwargs: Any) -> OperationResult[ActivationPayload]:
    return cast(OperationResult[ActivationPayload], _result("activation", payload, **kwargs))


def postflight_result(payload: PostflightPayload, **kwargs: Any) -> OperationResult[PostflightPayload]:
    return cast(OperationResult[PostflightPayload], _result("postflight", payload, **kwargs))


__all__ = [
    "ActivationPayload",
    "CandidateBuildPayload",
    "DurableTransitionPayload",
    "PostflightPayload",
    "SemanticVerificationPayload",
    "activation_result",
    "candidate_build_result",
    "durable_transition_result",
    "postflight_result",
    "semantic_verification_result",
]
