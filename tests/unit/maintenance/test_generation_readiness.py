from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.operations.evidence import EvidenceBinding
from polylogue.operations.generation_readiness import (
    CapacityEnvelope,
    GenerationReadinessFacts,
    GenerationReadinessRequest,
    ReadinessFailure,
    evaluate_generation_readiness,
    generation_readiness_facts,
)


def _request() -> GenerationReadinessRequest:
    return GenerationReadinessRequest(
        code_identity="git:" + "a" * 40,
        package_identity="polylogue:1",
        dependency_identity="deps-a",
        archive_identity="archive-a",
        tier_identities=(EvidenceBinding("source-tier", "source-tier-a"), EvidenceBinding("user-tier", "user-tier-a")),
        allowed_durable_transitions=("index:inactive->active",),
        source_inventory_ref="source-inventory-a",
        blob_inventory_ref="blob-inventory-a",
        topology_digest="topology-a",
        cut_policy="mutable-family:codex",
        capacity=CapacityEnvelope(required_bytes=10, available_bytes=100),
        proof_plan_digest="plan-a",
        external_authorization_refs=("opaque-coordinator-ref",),
        generation_identity="generation-a",
    )


def _facts() -> GenerationReadinessFacts:
    return GenerationReadinessFacts(
        archive_identity="archive-a",
        generation_identity="generation-a",
        tier_identities=(EvidenceBinding("source-tier", "source-tier-a"), EvidenceBinding("user-tier", "user-tier-a")),
        source_inventory_ref="source-inventory-a",
        blob_inventory_ref="blob-inventory-a",
        topology_digest="topology-a",
        capacity=CapacityEnvelope(required_bytes=10, available_bytes=100),
        active_writer=False,
        schemas_compatible=True,
        lifecycle_supported=True,
        expected_source_arrivals=("mutable-family:codex",),
    )


def test_readiness_is_reusable_and_read_only() -> None:
    result = evaluate_generation_readiness(_request(), _facts())

    assert result.ready
    assert result.payload.archive_identity == "archive-a"
    assert result.header.operation == "generation-readiness"
    assert result.header.outputs == (EvidenceBinding("readiness", result.payload.readiness_digest),)


@pytest.mark.parametrize(
    ("request_mutation", "facts_mutation", "failure_code"),
    (
        ("archive_identity", None, "foreign-archive-identity"),
        (None, {"active_writer": True}, "active-writer"),
        (None, {"schemas_compatible": False}, "incompatible-schema"),
        (None, {"lifecycle_supported": False}, "unsupported-generation-lifecycle"),
        (None, {"capacity": CapacityEnvelope(required_bytes=101, available_bytes=100)}, "insufficient-capacity"),
        ("proof_plan_digest", None, "stale-proof-plan"),
        ("topology_digest", None, "source-topology-drift"),
    ),
)
def test_readiness_returns_structured_failures_for_each_correctness_input(
    request_mutation: str | None,
    facts_mutation: dict[str, object] | None,
    failure_code: str,
) -> None:
    request = _request()
    facts = _facts()
    if request_mutation is not None:
        value = "foreign" if request_mutation != "proof_plan_digest" else "stale-plan"
        request = replace(request, **{request_mutation: value})  # type: ignore[arg-type]
    if facts_mutation:
        facts = replace(facts, **facts_mutation)  # type: ignore[arg-type]

    result = evaluate_generation_readiness(request, facts)

    assert not result.ready
    assert (
        ReadinessFailure(code=failure_code, detail=result.failure_for(failure_code).detail) in result.payload.failures
    )


def test_readiness_classifies_authorized_mutable_arrival_without_accepting_topology_drift() -> None:
    facts = replace(_facts(), expected_source_arrivals=("mutable-family:codex", "mutable-family:claude"))
    result = evaluate_generation_readiness(_request(), facts)

    assert result.ready
    assert result.payload.mutable_arrivals == ("mutable-family:claude",)


def test_archive_readiness_facts_are_read_only(tmp_path: Path) -> None:
    root = tmp_path / "isolated-archive"
    root.mkdir()
    before = tuple(root.iterdir())

    generation_readiness_facts(root, proof_plan_digest="plan-a")

    assert tuple(root.iterdir()) == before
