"""Resolution/explanation helpers for schema operator workflows."""

from __future__ import annotations

from polylogue.schemas.operator_annotations import collect_annotation_summary
from polylogue.schemas.operator_models import (
    SchemaExplainRequest,
    SchemaExplainResult,
    SchemaPayloadResolveRequest,
    SchemaPayloadResolveResult,
)
from polylogue.schemas.operator_registry import runtime_schema_registry, schema_registry


def explain_schema(request: SchemaExplainRequest) -> SchemaExplainResult:
    registry = schema_registry()
    package = registry.get_package(request.provider, version=request.version)
    schema = registry.get_element_schema(
        request.provider,
        version=request.version,
        element_kind=request.element_kind,
    )
    if schema is None:
        raise ValueError(
            f"No schema found for {request.provider} version={request.version}"
            + (f" element={request.element_kind}" if request.element_kind else "")
        )
    return SchemaExplainResult(
        provider=request.provider,
        version=request.version,
        element_kind=request.element_kind,
        package=package,
        schema=schema,
        annotations=collect_annotation_summary(schema),
    )


def resolve_schema_payload(request: SchemaPayloadResolveRequest) -> SchemaPayloadResolveResult:
    registry = runtime_schema_registry()
    return SchemaPayloadResolveResult(
        provider=request.provider,
        source_path=request.source_path,
        resolution=registry.resolve_payload(
            request.provider,
            request.payload,
            source_path=request.source_path,
        ),
    )


__all__ = [
    "explain_schema",
    "resolve_schema_payload",
]
