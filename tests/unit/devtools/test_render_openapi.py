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
