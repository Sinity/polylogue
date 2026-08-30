"""Contract and capability-negative tests for candidate construction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import ValidationError

from polylogue.core.enums import OperationStatus
from polylogue.core.json import JSONValue
from polylogue.operations import (
    CandidateBuildBudget,
    CandidateBuildError,
    CandidateBuildGeneration,
    CandidateBuildObligation,
    CandidateBuildPlanningContext,
    CandidateBuildProgress,
    CandidateBuildReceipt,
    CandidateBuildRequest,
    CandidateBuildResult,
    CandidateBuildWireRequest,
    SourceSeal,
    build_runtime_operation_catalog,
    lower_candidate_build_wire,
    plan_candidate_build,
)
from polylogue.storage.index_generation import IndexGeneration


def _seal() -> SourceSeal:
    return SourceSeal(
        archive_identity="archive-identity",
        source_identity="source-identity",
        source_snapshot="source-snapshot",
        source_schema_version=35,
        cut_identity="0" * 64,
        candidate_manifest_digest="1" * 64,
        carry_forward_manifest_digest="2" * 64,
    )


def _request() -> CandidateBuildRequest:
    return CandidateBuildRequest(
        source_seal=_seal(),
        package="polylogue-index",
        code="git:abc123",
        schemas=("index:35", "source:35"),
        parser_declarations=("parser:codex:v1",),
        lowering_declarations=("lowering:archive:v2",),
        origin_declarations=("origin:codex",),
        recipe_version="recipe-v3",
        semantic_version="semantic-v4",
    )


def _generation(*, state: str = "inactive") -> IndexGeneration:
    return IndexGeneration(
        generation_id="gen-1",
        owner_id="owner-1",
        archive_root="/archive",
        index_path="/archive/.index-generations/gen-1/index.db",
        state=state,
        created_at_ms=1,
    )


def test_candidate_plan_refuses_a_recomputed_cut_mismatch() -> None:
    """Mutation: changing the cut after request construction must reject the candidate."""
    request = _request()
    stale = request.source_seal.model_copy(update={"cut_identity": "f" * 64})
    context = CandidateBuildPlanningContext(
        archive_root=Path("/archive"),
        source_seal=request.source_seal,
        generation=_generation(),
        budget=CandidateBuildBudget(source_bytes=1, work_units=1, memory_bytes=1),
        recompute_source_seal=lambda: stale,
    )

    with pytest.raises(ValueError, match="source cut is stale"):
        plan_candidate_build(request, context)


def test_catalog_names_every_candidate_contract_type() -> None:
    spec = build_runtime_operation_catalog().by_name()["candidate-build"]
    contracts: tuple[tuple[str | None, str], ...] = (
        (spec.request_contract, "CandidateBuildRequest"),
        (spec.plan_contract, "CandidateBuildPlan"),
        (spec.progress_contract, "CandidateBuildProgress"),
        (spec.result_contract, "CandidateBuildResult"),
        (spec.error_contract, "CandidateBuildError"),
        (spec.receipt_contract, "CandidateBuildReceipt"),
    )
    for contract, suffix in contracts:
        assert contract is not None
        assert contract.endswith(suffix)
    assert spec.surfaces == ("daemon", "cli", "mcp", "api")


def test_request_identity_is_order_independent_and_binds_every_authority_field() -> None:
    request = _request()
    reordered = request.model_copy(
        update={
            "schemas": ("source:35", "index:35"),
            "parser_declarations": ("parser:codex:v1",),
        }
    )
    assert reordered.identity_digest == request.identity_digest

    for field in (
        "source_seal",
        "package",
        "code",
        "schemas",
        "parser_declarations",
        "lowering_declarations",
        "origin_declarations",
        "recipe_version",
        "semantic_version",
    ):
        value = getattr(request, field)
        if field == "source_seal":
            value = value.model_copy(update={"source_snapshot": "changed"})
        elif isinstance(value, tuple):
            value = (*value, "changed")
        else:
            value = f"{value}-changed"
        assert request.model_copy(update={field: value}).identity_digest != request.identity_digest, field


@pytest.mark.parametrize(
    "forbidden",
    [
        {"archive_root": "/tmp/archive"},
        {"generation_id": "gen-1"},
        {"batch_size": 10},
        {"concurrency": 4},
        {"resource_budget": {"bytes": 10}},
        {"checks": ["anything"]},
        {"callback": "callable"},
        {"promote": True},
        {"activate": True},
        {"cleanup": True},
        {"source_mutation": True},
    ],
)
def test_request_rejects_forbidden_capabilities(forbidden: Mapping[str, JSONValue]) -> None:
    payload = _request().to_dict()
    payload.update(forbidden)
    with pytest.raises(ValidationError):
        CandidateBuildRequest.model_validate(payload)


def test_wire_protocol_rejects_drift_and_unknown_fields() -> None:
    wire = CandidateBuildWireRequest(protocol="polylogue.candidate-build/v1", request=_request())
    payload = wire.to_dict()
    assert lower_candidate_build_wire(payload, surface="cli") == _request()
    with pytest.raises(ValidationError):
        CandidateBuildWireRequest.model_validate({**payload, "protocol": "polylogue.candidate-build/v2"})
    with pytest.raises(ValidationError):
        CandidateBuildWireRequest.model_validate({**payload, "unexpected": True})
    with pytest.raises(ValidationError):
        CandidateBuildRequest.model_validate({**_request().to_dict(), "unexpected": True})


def test_plan_is_server_resolved_and_preview_does_not_grant_lifecycle_authority() -> None:
    context = CandidateBuildPlanningContext(
        archive_root=Path("/archive"),
        source_seal=_seal(),
        generation=_generation(),
        budget=CandidateBuildBudget(source_bytes=100, work_units=4, memory_bytes=1024),
        obligations=(CandidateBuildObligation(name="lineage"), CandidateBuildObligation(name="source")),
    )
    preview = plan_candidate_build(_request(), context, preview=True)
    execute = plan_candidate_build(_request(), context)
    assert preview.model_dump() | {"preview": False} == execute.model_dump()
    assert preview.target_generation.state == "inactive"
    assert preview.authority == "inactive-only"
    assert preview.target_generation.index_path.startswith("/")


def test_plan_rejects_stale_seal_and_active_target() -> None:
    context = CandidateBuildPlanningContext(
        archive_root=Path("/archive"),
        source_seal=_seal(),
        generation=_generation(),
        budget=CandidateBuildBudget(source_bytes=1, work_units=1, memory_bytes=1),
    )
    with pytest.raises(ValueError, match="source seal"):
        plan_candidate_build(
            _request().model_copy(update={"source_seal": _seal().model_copy(update={"source_snapshot": "new"})}),
            context,
        )
    with pytest.raises(ValueError, match="inactive"):
        plan_candidate_build(
            _request(),
            CandidateBuildPlanningContext(
                archive_root=context.archive_root,
                source_seal=context.source_seal,
                generation=_generation(state="active"),
                budget=context.budget,
            ),
        )


def test_progress_and_receipt_cannot_attest_activation() -> None:
    request = _request()
    generation = CandidateBuildGeneration(
        generation_id="gen-1",
        owner_id="owner-1",
        archive_root="/archive",
        index_path="/archive/gen-1/index.db",
    )
    progress = CandidateBuildProgress(
        operation_id="operation-1",
        request_digest=request.identity_digest,
        generation_id=generation.generation_id,
        status=OperationStatus.RUNNING,
        processed_units=0,
        total_units=1,
        processed_bytes=0,
        total_bytes=1,
    )
    assert progress.status is OperationStatus.RUNNING
    result = CandidateBuildResult(
        request_digest=request.identity_digest,
        plan_digest="a" * 64,
        generation=generation,
        source_seal_digest=request.source_seal.digest,
        processed_units=1,
        processed_bytes=1,
    )
    receipt = CandidateBuildReceipt(
        operation_id="operation-1",
        request_digest=request.identity_digest,
        plan_digest=result.plan_digest,
        status="completed",
        result=result,
        source_seal_digest=result.source_seal_digest,
        generation_id=generation.generation_id,
    )
    assert receipt.result is not None and receipt.result.lifecycle == "inactive"
    with pytest.raises(ValidationError):
        CandidateBuildResult.model_validate(
            {**result.to_dict(), "generation": {**generation.model_dump(), "state": "active"}}
        )
    with pytest.raises(ValidationError):
        CandidateBuildReceipt.model_validate({**receipt.to_dict(), "promoted": True})


def test_typed_failure_is_not_a_success_receipt() -> None:
    request = _request()
    error = CandidateBuildError(
        status="failed",
        code="source_changed",
        message="source seal changed",
        request_digest=request.identity_digest,
        retryable=True,
    )
    with pytest.raises(ValidationError):
        CandidateBuildReceipt(
            operation_id="operation-1",
            request_digest=request.identity_digest,
            plan_digest="b" * 64,
            status="completed",
            error=error,
            source_seal_digest=request.source_seal.digest,
            generation_id="gen-1",
        )
