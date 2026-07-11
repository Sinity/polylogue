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


def test_openapi_publishes_typed_first_party_credential_contract() -> None:
    """A generated browser client can bootstrap without shell-owned globals."""

    document = _build_openapi_document()
    lifecycle = document["paths"]["/api/web-auth/session"]
    schemas = document["components"]["schemas"]

    assert lifecycle["post"]["operationId"] == "bootstrapWebCredential"
    assert lifecycle["post"]["security"] == []
    assert lifecycle["post"]["responses"]["201"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/WebCredentialBootstrapPayload"
    }
    assert lifecycle["post"]["responses"]["400"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/QueryErrorPayload"
    }
    assert lifecycle["delete"]["operationId"] == "revokeWebCredential"
    assert lifecycle["delete"]["security"] == [{"webCredentialCookie": []}]
    assert lifecycle["delete"]["responses"]["400"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/QueryErrorPayload"
    }
    assert lifecycle["delete"]["responses"]["200"]["headers"]["Set-Cookie"]["schema"]["writeOnly"] is True
    assert document["components"]["securitySchemes"]["machineBearer"] == {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "opaque",
        "description": "Machine-client token configured by --api-auth-token.",
    }
    assert document["components"]["securitySchemes"]["webCredentialCookie"] == {
        "type": "apiKey",
        "in": "cookie",
        "name": "polylogue_web_credential",
        "description": (
            "Short-lived, origin-bound HttpOnly credential issued by bootstrapWebCredential. "
            "Browser clients use credentials: same-origin and never read the cookie value."
        ),
    }
    assert {
        "WebCredentialBootstrapPayload",
        "WebCredentialRevocationPayload",
        "WebCredentialFailurePayload",
        "WebCredentialReadyPayload",
        "QueryErrorPayload",
    }.issubset(schemas)
    failure_states = schemas["WebCredentialFailurePayload"]["properties"]["error"]["enum"]
    assert "web_credential_expired" in failure_states
    assert "web_credential_wrong_origin" in failure_states

    expected_read_security: list[dict[str, list[str]]] = [
        {"machineBearer": []},
        {"webCredentialCookie": []},
    ]
    protected_operations = [
        document["paths"]["/api/sessions"]["get"],
        document["paths"]["/api/query-units"]["get"],
        document["paths"]["/api/sessions/{session_id}/read"]["get"],
        document["paths"]["/api/assertions"]["get"],
    ]
    assert all(operation["security"] == expected_read_security for operation in protected_operations)
    assert all(
        operation["responses"]["400"]["content"]["application/json"]["schema"]
        == {"$ref": "#/components/schemas/QueryErrorPayload"}
        for operation in protected_operations
    )
    expected_auth_errors = [
        {"$ref": "#/components/schemas/QueryErrorPayload"},
        {"$ref": "#/components/schemas/WebCredentialFailurePayload"},
    ]
    for operation in protected_operations:
        for status in ("401", "403"):
            response = operation["responses"][status]
            assert response["content"]["application/json"]["schema"]["anyOf"] == expected_auth_errors
            assert (
                "web_credential_wrong_origin"
                in response["headers"]["X-Polylogue-Web-Credential-State"]["schema"]["enum"]
            )
