from __future__ import annotations

from devtools.render_openapi import _build_openapi_document


def test_openapi_publishes_stable_evidence_routes() -> None:
    document = _build_openapi_document()
    paths = document["paths"]
    schemas = document["components"]["schemas"]

    recovery = paths["/api/sessions/{session_id}/recovery"]["get"]
    assertions = paths["/api/assertions"]["get"]

    assert recovery["operationId"] == "readSessionRecovery"
    assert recovery["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/RecoveryReadPayload"
    }
    assert assertions["operationId"] == "listAssertionClaims"
    assert assertions["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AssertionClaimListPayload"
    }
    assert "RecoveryReadPayload" in schemas
    assert "AssertionClaimListPayload" in schemas


def test_openapi_publishes_route_contract_extension() -> None:
    document = _build_openapi_document()
    contracts = document["x-polylogue-route-contracts"]

    contract_by_pattern = {(contract["method"], contract["pattern"]): contract for contract in contracts}
    query_units = contract_by_pattern[("GET", "/api/query-units")]
    recovery = contract_by_pattern[("GET", "/api/sessions/:id/recovery")]

    assert query_units["kind"] == "read_query"
    assert query_units["stability"] == "stable"
    assert query_units["response_contract"] == "QueryUnitResultEnvelope"
    assert recovery["kind"] == "read_detail"
    assert recovery["auth_policy"] == "bearer_if_configured"
    assert recovery["response_contract"] == "RecoveryReadPayload"
