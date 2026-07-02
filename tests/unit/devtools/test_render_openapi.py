from __future__ import annotations

from devtools.render_openapi import _build_openapi_document
from polylogue.archive.viewport import (
    read_view_http_choices,
    read_view_http_format_choices,
    read_view_http_query_params,
)


def test_openapi_publishes_stable_evidence_routes() -> None:
    document = _build_openapi_document()
    paths = document["paths"]
    schemas = document["components"]["schemas"]

    assertions = paths["/api/assertions"]["get"]

    assert assertions["operationId"] == "listAssertionClaims"
    assert assertions["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AssertionClaimListPayload"
    }
    assert "/api/sessions/{session_id}/recovery" not in paths
    assert "RecoveryReadPayload" not in schemas
    assert "AssertionClaimListPayload" in schemas


def test_openapi_publishes_route_contract_extension() -> None:
    document = _build_openapi_document()
    contracts = document["x-polylogue-route-contracts"]

    contract_by_pattern = {(contract["method"], contract["pattern"]): contract for contract in contracts}
    query_units = contract_by_pattern[("GET", "/api/query-units")]

    assert query_units["kind"] == "read_query"
    assert query_units["stability"] == "stable"
    assert query_units["response_contract"] == "QueryUnitResultEnvelope"
    assert ("GET", "/api/sessions/:id/recovery") not in contract_by_pattern


def test_openapi_read_view_route_uses_shared_http_capability_contract() -> None:
    document = _build_openapi_document()
    read_route = document["paths"]["/api/sessions/{session_id}/read"]["get"]
    parameters = {parameter["name"]: parameter for parameter in read_route["parameters"]}

    assert parameters["view"]["schema"]["enum"] == list(read_view_http_choices())
    assert parameters["format"]["schema"]["enum"] == list(read_view_http_format_choices())
    assert set(read_view_http_query_params()).issubset(parameters)
    assert "context-image" in read_route["description"]
    assert "max_tokens" in parameters
    assert "message_role" not in parameters
