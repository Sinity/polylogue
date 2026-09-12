"""Typed daemon operation protocol contracts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from polylogue.operations.daemon_protocol import (
    DAEMON_OPERATION_PROTOCOL,
    DAEMON_OPERATION_SPECS,
    DaemonOperationRequest,
    OperationResultContractError,
    StatusRequest,
    archive_identity,
    validate_operation_result,
)


def test_operation_request_rejects_untyped_payloads() -> None:
    with pytest.raises(ValueError, match="payload must be an object"):
        DaemonOperationRequest.from_dict({"operation": "status", "payload": []})


def test_operation_request_requires_the_negotiated_protocol() -> None:
    with pytest.raises(ValueError, match="unsupported daemon operation protocol"):
        DaemonOperationRequest.from_dict({"operation": "status", "payload": {}, "protocol": "v0"})

    with pytest.raises(ValueError, match="unsupported daemon operation protocol"):
        DaemonOperationRequest.from_dict({"operation": "status", "payload": {}})


def test_operation_request_requires_a_declared_operation_and_exchange_identity() -> None:
    request = {
        "protocol": DAEMON_OPERATION_PROTOCOL,
        "operation": "status",
        "payload": {},
        "request_id": "request-1",
    }

    parsed = DaemonOperationRequest.from_dict(request)

    assert parsed.operation == "status"
    assert parsed.request_id == "request-1"
    with pytest.raises(ValueError, match="operation is not declared"):
        DaemonOperationRequest.from_dict({**request, "operation": "unknown"})
    with pytest.raises(ValueError, match="request_id must be a non-empty string"):
        DaemonOperationRequest.from_dict({key: value for key, value in request.items() if key != "request_id"})
    with pytest.raises(ValueError, match="index_schema_version must be an integer"):
        DaemonOperationRequest.from_dict({**request, "index_schema_version": True})
    with pytest.raises(ValueError, match="deadline_ms must be a positive integer"):
        DaemonOperationRequest.from_dict({**request, "deadline_ms": True})


def test_operation_specs_bind_concrete_payload_models() -> None:
    """A string label alone cannot be the machine contract."""

    status = next(spec for spec in DAEMON_OPERATION_SPECS if spec.name == "status")

    assert status.request_model is StatusRequest
    assert status.request_model.__name__ == status.request_type
    assert status.result_model.__name__ == status.result_type
    with pytest.raises(ValueError, match="invalid StatusRequest payload"):
        DaemonOperationRequest.from_dict(
            {
                "protocol": DAEMON_OPERATION_PROTOCOL,
                "operation": "status",
                "payload": {"unexpected": True},
                "request_id": "bad-status",
            }
        )


@pytest.mark.parametrize(
    ("operation", "payload"),
    [
        ("status", {"include_archive_readiness": "true"}),
        ("mutation.session.metadata", {"session_ids": ["codex:one"], "pairs": [["key"]]}),
        ("mutation.session.metadata", {"session_ids": ["codex:one"], "pairs": [[" ", "value"]]}),
        ("mutation.session.delete.execute", {"authorization_refs": ["same", "same"]}),
    ],
)
def test_request_contract_rejects_coercion_and_ambiguous_mutation_inputs(
    operation: str,
    payload: dict[str, object],
) -> None:
    """Removing strict validation or reference uniqueness admits these inputs."""
    with pytest.raises(ValueError):
        DaemonOperationRequest.from_dict(
            {
                "protocol": DAEMON_OPERATION_PROTOCOL,
                "request_id": "contract-request",
                "operation": operation,
                "payload": payload,
            }
        )


@pytest.mark.parametrize(
    "operation", ["cli.query", "query.units", "completion", "facets", "status", "operation.status"]
)
def test_declared_result_contracts_reject_empty_placeholders(operation: str) -> None:
    """An empty permissive model would accept a result with no product evidence."""
    with pytest.raises(OperationResultContractError):
        validate_operation_result(operation, {})


def test_mutation_result_requires_exact_authorization_reference_shape() -> None:
    validate_operation_result(
        "mutation.session.delete.authorize",
        {
            "status": "authorized",
            "authorization_ref": "one",
            "authorization_refs": ["one", "two"],
        },
    )
    with pytest.raises(OperationResultContractError):
        validate_operation_result(
            "mutation.session.delete.authorize",
            {
                "status": "authorized",
                "authorization_ref": "other",
                "authorization_refs": ["one", "two"],
            },
        )


def test_operation_request_binds_an_optional_prior_authority_snapshot() -> None:
    request = DaemonOperationRequest.from_dict(
        {
            "protocol": DAEMON_OPERATION_PROTOCOL,
            "operation": "status",
            "payload": {},
            "request_id": "request-1",
            "expected_archive_identity": "archive-epoch",
            "expected_generation_id": "generation-1",
        }
    )

    assert request.expected_archive_identity == "archive-epoch"
    assert request.expected_generation_id == "generation-1"
    assert DaemonOperationRequest.from_dict(request.to_dict()) == request
    with pytest.raises(ValueError, match="expected_generation_id must be a non-empty string"):
        DaemonOperationRequest.from_dict({**request.to_dict(), "expected_generation_id": ""})


def test_archive_identity_contains_readiness_and_generation(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "index.db").write_bytes(b"fixture")
    identity, generation, readiness = archive_identity(archive, schema_version=24, daemon_version="test")

    assert identity["root"] == str(archive)
    assert isinstance(identity["archive_identity"], str)
    tiers = cast(dict[str, int], identity["tier_schema_versions"])
    generation_id = cast(str, generation["id"])
    assert generation["index_schema_version"] == tiers["index"]
    assert generation_id.startswith("dev:")
    assert identity["tier_schema_versions"] == generation["tier_schema_versions"]
    assert {"source", "index", "embeddings", "user", "audit", "ops"} <= set(tiers)
    assert readiness == {"state": "ready", "ready": True, "reason": None, "degraded_components": []}
    assert DAEMON_OPERATION_PROTOCOL == "polylogue.daemon-operation/v1"
