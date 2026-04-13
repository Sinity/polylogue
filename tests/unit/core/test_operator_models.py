"""Operator model serialization contracts."""

from __future__ import annotations

from polylogue.schemas.operator_models import (
    SchemaAnnotationSummary,
    SchemaCompareResult,
    SchemaCoverageSummary,
    SchemaExplainResult,
    SchemaListResult,
    SchemaPayloadResolveRequest,
    SchemaPayloadResolveResult,
    SchemaPromoteResult,
    SchemaProviderSnapshot,
    SchemaRoleAssignment,
)
from polylogue.schemas.packages import SchemaPackageCatalog, SchemaResolution, SchemaVersionPackage
from polylogue.schemas.runtime_registry import canonical_schema_provider
from polylogue.schemas.tooling_registry import ClusterManifest, PropertyChange, SchemaDiff


def test_schema_list_result_to_dict_for_provider() -> None:
    result = SchemaListResult(
        provider="chatgpt",
        selected=SchemaProviderSnapshot(
            provider="chatgpt",
            versions=["v1", "v2"],
            catalog=None,
            manifest=None,
            latest_age_days=3,
        ),
    )

    assert result.to_dict() == {"provider": "chatgpt", "versions": ["v1", "v2"]}


def test_schema_list_result_to_dict_for_provider_with_components() -> None:
    package = SchemaVersionPackage(
        provider="chatgpt",
        version="v1",
        anchor_kind="message",
        default_element_kind="message",
        first_seen="2026-01-01T00:00:00Z",
        last_seen="2026-01-01T01:00:00Z",
        bundle_scope_count=0,
        sample_count=12,
    )
    catalog = SchemaPackageCatalog(
        provider="chatgpt",
        packages=[package],
        latest_version="v1",
        default_version="v1",
    )
    manifest = ClusterManifest(provider=canonical_schema_provider("chatgpt"))
    result = SchemaListResult(
        provider=None,
        providers=[
            SchemaProviderSnapshot(
                provider="chatgpt",
                versions=["v1"],
                catalog=catalog,
                manifest=manifest,
            )
        ],
    )

    assert result.to_dict() == [
        {
            "provider": "chatgpt",
            "versions": ["v1"],
            "package_count": 1,
            "default_version": "v1",
            "latest_version": "v1",
            "cluster_count": 0,
            "corpus_spec_count": 0,
        }
    ]


def test_schema_compare_result_to_dict() -> None:
    diff = SchemaDiff(
        provider=canonical_schema_provider("chatgpt"),
        version_a="v1",
        version_b="v2",
        added_properties=["status"],
        removed_properties=[],
        changed_properties=[],
        classified_changes=[PropertyChange(path="status", kind="added", detail="new field")],
    )

    assert SchemaCompareResult(diff=diff).to_dict() == {
        "provider": "chatgpt",
        "version_a": "v1",
        "version_b": "v2",
        "summary": "+1 properties",
        "has_changes": True,
        "added_properties": ["status"],
        "removed_properties": [],
        "changed_properties": [],
        "classified_changes": [{"path": "status", "kind": "added", "detail": "new field"}],
    }


def test_schema_promote_result_to_dict() -> None:
    package = SchemaVersionPackage(
        provider="chatgpt",
        version="v1",
        anchor_kind="message",
        default_element_kind="message",
        first_seen="2026-01-01T00:00:00Z",
        last_seen="2026-01-01T01:00:00Z",
        bundle_scope_count=0,
        sample_count=1,
    )

    result = SchemaPromoteResult(
        provider="chatgpt",
        cluster_id="abc",
        package_version="v1",
        package=package,
        schema={"type": "object"},
        versions=["v1"],
    )

    assert result.to_dict() == {
        "provider": "chatgpt",
        "cluster_id": "abc",
        "package_version": "v1",
        "package": package.to_dict(),
        "schema": {"type": "object"},
    }


def test_schema_explain_result_to_dict() -> None:
    package = SchemaVersionPackage(
        provider="chatgpt",
        version="v1",
        anchor_kind="message",
        default_element_kind="message",
        first_seen="2026-01-01T00:00:00Z",
        last_seen="2026-01-01T01:00:00Z",
        bundle_scope_count=0,
        sample_count=1,
    )
    annotations = SchemaAnnotationSummary(
        semantic_count=1,
        format_count=0,
        values_count=0,
        total_enum_values=0,
        roles=[
            SchemaRoleAssignment(
                path="$.id",
                role="message_id",
                confidence=1.0,
                evidence={"source": "generated"},
            )
        ],
        coverage=SchemaCoverageSummary(
            total_fields=1,
            with_format=0,
            with_values=0,
            with_role=1,
        ),
    )
    result = SchemaExplainResult(
        provider="chatgpt",
        version="v1",
        element_kind=None,
        package=package,
        schema={"type": "object"},
        annotations=annotations,
    )

    payload = result.to_dict()
    assert payload["schema"] == {"type": "object"}
    assert payload["package"] == package.to_dict()
    assert payload["annotations"]["semantic_count"] == 1
    assert payload["annotations"]["roles"][0]["role"] == "message_id"


def test_schema_payload_resolve_result_resolved_property() -> None:
    resolved = SchemaPayloadResolveResult(
        provider="chatgpt",
        source_path="path",
        resolution=SchemaResolution(
            provider="chatgpt",
            package_version="v1",
            element_kind="message",
            exact_structure_id=None,
            bundle_scope=None,
            reason="exact_structure",
        ),
    )
    missing = SchemaPayloadResolveResult(
        provider="chatgpt",
        source_path=None,
        resolution=None,
    )

    assert resolved.is_resolved is True
    assert missing.is_resolved is False


def test_schema_payload_resolve_request_roundtrip() -> None:
    request = SchemaPayloadResolveRequest(
        provider="chatgpt",
        payload={"foo": "bar"},
        source_path=None,
    )

    assert request.provider == "chatgpt"
    assert request.payload == {"foo": "bar"}
    assert request.source_path is None
