"""Small shared evidence header for ordinary Polylogue operation results.

The header binds an operation's execution to its code, package, invocation,
inputs, outputs, terminal outcome, and self digest.  It deliberately knows
nothing about campaigns, task trackers, phases, or domain payloads.  Each
operation family supplies a typed payload and owns its payload validation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Generic, TypeVar

from polylogue.core.hashing import hash_payload
from polylogue.core.json import JSONDocument, json_document

_DIGEST = re.compile(r"^[0-9a-f]{64}$")

PayloadT = TypeVar("PayloadT")


class EvidenceValidationError(ValueError):
    """An operation result or caller-supplied evidence set is invalid."""


class OperationStatus(StrEnum):
    """Terminal state carried by an operation result."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class EvidenceBinding:
    """One named authoritative identity bound to an operation result."""

    name: str
    value: str

    def __post_init__(self) -> None:
        if not self.name or not self.value:
            raise EvidenceValidationError("evidence bindings require non-empty names and values")

    def to_document(self) -> JSONDocument:
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True, slots=True)
class OperationFailure:
    """Structured terminal failure owned by the operation that produced it."""

    code: str
    message: str
    details: tuple[EvidenceBinding, ...] = ()

    def to_document(self) -> JSONDocument:
        return {
            "code": self.code,
            "message": self.message,
            "details": [item.to_document() for item in self.details],
        }


@dataclass(frozen=True, slots=True)
class OperationEvidence:
    """Common immutable evidence header shared by ordinary operations."""

    operation: str
    operation_version: int
    code_identity: str
    package_identity: str
    invocation_digest: str
    inputs: tuple[EvidenceBinding, ...]
    outputs: tuple[EvidenceBinding, ...]
    status: OperationStatus
    failure: OperationFailure | None
    started_at: str
    finished_at: str
    payload_digest: str
    self_digest: str

    def unsigned_document(self) -> JSONDocument:
        return {
            "operation": self.operation,
            "operation_version": self.operation_version,
            "code_identity": self.code_identity,
            "package_identity": self.package_identity,
            "invocation_digest": self.invocation_digest,
            "inputs": [item.to_document() for item in self.inputs],
            "outputs": [item.to_document() for item in self.outputs],
            "status": self.status.value,
            "failure": self.failure.to_document() if self.failure is not None else None,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "payload_digest": self.payload_digest,
        }

    def to_document(self) -> JSONDocument:
        return {**self.unsigned_document(), "self_digest": self.self_digest}

    def validate(self, *, payload_digest: str) -> None:
        if not self.operation or self.operation_version < 1:
            raise EvidenceValidationError("operation identity or version is invalid")
        if not self.code_identity or not self.package_identity or not self.invocation_digest:
            raise EvidenceValidationError("operation code, package, and invocation identities are required")
        if not _DIGEST.fullmatch(self.payload_digest) or self.payload_digest != payload_digest:
            raise EvidenceValidationError("payload digest is invalid or does not match the domain payload")
        if not _DIGEST.fullmatch(self.self_digest):
            raise EvidenceValidationError("self digest is malformed")
        if hash_payload(self.unsigned_document()) != self.self_digest:
            raise EvidenceValidationError("self digest does not match the operation evidence header")
        if not self.started_at or not self.finished_at:
            raise EvidenceValidationError("operation result timestamps are required")
        if self.status is OperationStatus.SUCCEEDED and self.failure is not None:
            raise EvidenceValidationError("successful operation result cannot carry a failure")
        if self.status in {OperationStatus.FAILED, OperationStatus.UNKNOWN} and self.failure is None:
            raise EvidenceValidationError("failed or unknown operation result requires structured failure")

    @classmethod
    def create(
        cls,
        *,
        operation: str,
        operation_version: int,
        code_identity: str,
        package_identity: str,
        invocation_digest: str,
        inputs: tuple[EvidenceBinding, ...],
        outputs: tuple[EvidenceBinding, ...],
        status: OperationStatus,
        failure: OperationFailure | None,
        payload_digest: str,
        started_at: str,
        finished_at: str,
    ) -> OperationEvidence:
        draft = cls(
            operation=operation,
            operation_version=operation_version,
            code_identity=code_identity,
            package_identity=package_identity,
            invocation_digest=invocation_digest,
            inputs=inputs,
            outputs=outputs,
            status=status,
            failure=failure,
            started_at=started_at,
            finished_at=finished_at,
            payload_digest=payload_digest,
            self_digest="0" * 64,
        )
        return replace(draft, self_digest=hash_payload(draft.unsigned_document()))


def _payload_document(payload: object) -> JSONDocument:
    to_document = getattr(payload, "to_document", None)
    if not callable(to_document):
        raise EvidenceValidationError("operation payload must be a domain-owned typed document")
    document = to_document()
    if not isinstance(document, dict):
        raise EvidenceValidationError("operation payload document must be an object")
    return json_document(document)


@dataclass(frozen=True, slots=True)
class OperationResult(Generic[PayloadT]):
    """A shared header paired with one domain-owned typed payload."""

    header: OperationEvidence
    payload: PayloadT

    @classmethod
    def create(
        cls,
        *,
        operation: str,
        operation_version: int,
        code_identity: str,
        package_identity: str,
        invocation_digest: str,
        inputs: tuple[EvidenceBinding, ...],
        outputs: tuple[EvidenceBinding, ...],
        status: OperationStatus,
        payload: PayloadT,
        failure: Mapping[str, object] | OperationFailure | None = None,
        started_at: str = "1970-01-01T00:00:00+00:00",
        finished_at: str = "1970-01-01T00:00:00+00:00",
    ) -> OperationResult[PayloadT]:
        if isinstance(failure, Mapping):
            raw_details_value = failure.get("details", ())
            raw_details = raw_details_value if isinstance(raw_details_value, (list, tuple)) else ()
            details = tuple(
                EvidenceBinding(str(item["name"]), str(item["value"]))
                for item in raw_details
                if isinstance(item, Mapping) and "name" in item and "value" in item
            )
            failure = OperationFailure(
                str(failure.get("code", "operation-failed")), str(failure.get("message", "")), details
            )
        payload_digest = hash_payload(_payload_document(payload))
        header = OperationEvidence.create(
            operation=operation,
            operation_version=operation_version,
            code_identity=code_identity,
            package_identity=package_identity,
            invocation_digest=invocation_digest,
            inputs=inputs,
            outputs=outputs,
            status=status,
            failure=failure,
            payload_digest=payload_digest,
            started_at=started_at,
            finished_at=finished_at,
        )
        return cls(header=header, payload=payload)

    def validate(self) -> None:
        self.header.validate(payload_digest=hash_payload(_payload_document(self.payload)))
        validate_payload = getattr(self.payload, "validate", None)
        if callable(validate_payload):
            validate_payload()
        validate_bindings = getattr(self.payload, "validate_against_evidence", None)
        if callable(validate_bindings):
            validate_bindings(self.header)

    def to_document(self) -> JSONDocument:
        self.validate()
        return {"evidence": self.header.to_document(), "payload": _payload_document(self.payload)}


def _bindings_by_name(bindings: Sequence[EvidenceBinding]) -> dict[str, str]:
    result: dict[str, str] = {}
    for binding in bindings:
        if binding.name in result:
            raise EvidenceValidationError(f"duplicate evidence binding: {binding.name}")
        result[binding.name] = binding.value
    return result


def validate_evidence_set(
    results: Sequence[OperationResult[Any]],
    *,
    required_operations: tuple[str, ...],
    expected_bindings: tuple[EvidenceBinding, ...],
) -> tuple[OperationResult[Any], ...]:
    """Validate a caller-supplied result set without discovering required work."""

    if not results:
        raise EvidenceValidationError("evidence set is empty")
    expected = _bindings_by_name(expected_bindings)
    if len(required_operations) != len(set(required_operations)):
        raise EvidenceValidationError("required operation references are duplicated")
    seen_digests: set[str] = set()
    seen_operations: set[str] = set()
    validated: list[OperationResult[Any]] = []
    for result in results:
        try:
            result.validate()
        except EvidenceValidationError:
            raise
        digest = result.header.self_digest
        if digest in seen_digests:
            raise EvidenceValidationError("duplicate evidence reference")
        seen_digests.add(digest)
        if result.header.operation in seen_operations:
            raise EvidenceValidationError("duplicate evidence reference")
        seen_operations.add(result.header.operation)
        if result.header.status is not OperationStatus.SUCCEEDED:
            raise EvidenceValidationError("evidence result terminal status is not successful")
        actual = _bindings_by_name(result.header.inputs)
        for name, value in expected.items():
            if actual.get(name) != value:
                if name == "archive":
                    raise EvidenceValidationError("mixed archive identity")
                if name == "generation":
                    raise EvidenceValidationError("mixed generation identity")
                raise EvidenceValidationError("invocation or proof-plan binding mismatch")
        validated.append(result)
    missing = set(required_operations) - seen_operations
    if missing:
        raise EvidenceValidationError(f"missing required operation: {sorted(missing)[0]}")
    return tuple(validated)


__all__ = [
    "EvidenceBinding",
    "EvidenceValidationError",
    "OperationEvidence",
    "OperationFailure",
    "OperationResult",
    "OperationStatus",
    "validate_evidence_set",
]
