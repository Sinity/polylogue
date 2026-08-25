from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from polylogue.core.hashing import hash_payload
from polylogue.operations.candidate_verification import CandidateVerificationPayload, candidate_verification_result
from polylogue.operations.evidence import (
    EvidenceBinding,
    EvidenceValidationError,
    OperationResult,
    OperationStatus,
    validate_evidence_set,
)
from polylogue.operations.generation_operations import (
    ActivationPayload,
    CandidateBuildPayload,
    DurableTransitionPayload,
    PostflightPayload,
    SemanticVerificationPayload,
    activation_result,
    candidate_build_result,
    durable_transition_result,
    postflight_result,
    semantic_verification_result,
)
from polylogue.operations.source_seal import SourceSealPayload, source_seal_result


def _bindings(*, generation: str = "generation-a", proof_plan: str = "plan-a") -> tuple[EvidenceBinding, ...]:
    return (
        EvidenceBinding("archive", "archive-a"),
        EvidenceBinding("generation", generation),
        EvidenceBinding("invocation", "invocation-a"),
        EvidenceBinding("proof-plan", proof_plan),
    )


def _source_result(*, status: OperationStatus = OperationStatus.SUCCEEDED) -> OperationResult[SourceSealPayload]:
    return source_seal_result(
        payload=SourceSealPayload(
            archive_identity="archive-a",
            source_snapshot="source-a",
            source_inventory_ref="source-inventory-a",
            topology_digest="topology-a",
            cut_policy="mutable-family:codex",
        ),
        code_identity="git:" + "a" * 40,
        package_identity="polylogue:1",
        invocation_digest="invocation-a",
        inputs=_bindings(),
        outputs=(EvidenceBinding("source-seal", "seal-a"),),
        status=status,
    )


def _candidate_result() -> OperationResult[CandidateVerificationPayload]:
    return candidate_verification_result(
        payload=CandidateVerificationPayload(
            archive_identity="archive-a",
            generation_identity="generation-a",
            candidate_digest="candidate-a",
            semantic_digest="semantic-a",
            publication_state="inactive",
        ),
        code_identity="git:" + "a" * 40,
        package_identity="polylogue:1",
        invocation_digest="invocation-a",
        inputs=_bindings(),
        outputs=(EvidenceBinding("candidate", "candidate-a"),),
    )


def test_two_ordinary_operation_families_share_only_the_evidence_header() -> None:
    source = _source_result()
    candidate = _candidate_result()

    validated = validate_evidence_set(
        (source, candidate),
        required_operations=("source-seal", "candidate-verification"),
        expected_bindings=_bindings(),
    )

    assert tuple(item.header.operation for item in validated) == ("source-seal", "candidate-verification")
    assert source.payload.source_snapshot == "source-a"
    assert candidate.payload.semantic_digest == "semantic-a"


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("archive", "mixed archive identity"),
        ("generation", "mixed generation identity"),
        ("proof-plan", "invocation or proof-plan binding mismatch"),
    ),
)
def test_aggregate_rejects_mutated_shared_bindings(mutation: str, message: str) -> None:
    source = _source_result()
    candidate = _candidate_result()
    bindings = {binding.name: binding.value for binding in _bindings()}
    bindings[mutation] = "foreign"

    with pytest.raises(EvidenceValidationError, match=message):
        validate_evidence_set(
            (source, candidate),
            required_operations=("source-seal", "candidate-verification"),
            expected_bindings=tuple(EvidenceBinding(key, value) for key, value in bindings.items()),
        )


def test_aggregate_rejects_duplicate_missing_failed_unknown_and_bad_digest() -> None:
    source = _source_result()
    candidate = _candidate_result()

    with pytest.raises(EvidenceValidationError, match="duplicate evidence reference"):
        validate_evidence_set(
            (source, source),
            required_operations=("source-seal",),
            expected_bindings=_bindings(),
        )
    with pytest.raises(EvidenceValidationError, match="missing required operation"):
        validate_evidence_set(
            (source,), required_operations=("source-seal", "candidate-verification"), expected_bindings=_bindings()
        )
    with pytest.raises(EvidenceValidationError, match="terminal status"):
        validate_evidence_set(
            (_source_result(status=OperationStatus.FAILED), candidate),
            required_operations=("source-seal", "candidate-verification"),
            expected_bindings=_bindings(),
        )
    with pytest.raises(EvidenceValidationError, match="terminal status"):
        validate_evidence_set(
            (_source_result(status=OperationStatus.UNKNOWN), candidate),
            required_operations=("source-seal", "candidate-verification"),
            expected_bindings=_bindings(),
        )
    with pytest.raises(EvidenceValidationError, match="self digest"):
        validate_evidence_set(
            (replace(source, header=replace(source.header, self_digest="0" * 64)), candidate),
            required_operations=("source-seal", "candidate-verification"),
            expected_bindings=_bindings(),
        )


def test_domain_payload_mutation_invalidates_the_real_result() -> None:
    result = _source_result()
    mutated = replace(result, payload=replace(result.payload, topology_digest="foreign-topology"))

    with pytest.raises(EvidenceValidationError, match="payload digest"):
        mutated.validate()


def test_domain_binding_mutation_invalidates_the_real_result() -> None:
    result = _source_result()
    altered_header = replace(result.header, inputs=(EvidenceBinding("archive", "foreign"),), self_digest="0" * 64)
    altered_header = replace(altered_header, self_digest=hash_payload(altered_header.unsigned_document()))
    mutated = replace(result, header=altered_header)

    with pytest.raises(EvidenceValidationError, match="archive binding"):
        mutated.validate()


@pytest.mark.parametrize(
    ("factory", "payload", "operation"),
    (
        (
            candidate_build_result,
            CandidateBuildPayload("archive-a", "generation-a", "source-a", "candidate-a", "inactive"),
            "candidate-build",
        ),
        (
            semantic_verification_result,
            SemanticVerificationPayload("archive-a", "generation-a", "candidate-a", "comparison-a", 0),
            "semantic-verification",
        ),
        (
            durable_transition_result,
            DurableTransitionPayload("archive-a", "generation-a", "index:inactive->active", "pre-a", "transition-a"),
            "durable-transition",
        ),
        (
            activation_result,
            ActivationPayload("archive-a", "generation-a", "generation-a", "publication-a"),
            "activation",
        ),
        (
            postflight_result,
            PostflightPayload("archive-a", "generation-a", "generation-a", "query-a", "restart-a"),
            "postflight",
        ),
    ),
)
def test_generation_lifecycle_families_emit_validated_typed_results(factory: Any, payload: Any, operation: str) -> None:
    result = factory(
        payload,
        code_identity="git:" + "a" * 40,
        package_identity="polylogue:1",
        invocation_digest="invocation-a",
        inputs=_bindings(),
        outputs=(EvidenceBinding(operation, operation + "-output"),),
    )

    result.validate()
    assert result.header.operation == operation
