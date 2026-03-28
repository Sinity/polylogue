"""Schema-annotation summarization helpers for operator workflows."""

from __future__ import annotations

from polylogue.schemas.operator_models import (
    SchemaAnnotationSummary,
    SchemaCoverageSummary,
    SchemaRoleAssignment,
)


def collect_annotation_summary(schema: dict) -> SchemaAnnotationSummary:
    """Collect format/value/semantic coverage from a schema document."""
    semantic_count = 0
    format_count = 0
    values_count = 0
    total_enum_values = 0
    roles: list[SchemaRoleAssignment] = []
    total_fields = 0
    with_format = 0
    with_values = 0
    with_role = 0

    def visit(node: dict, *, path: str) -> None:
        nonlocal semantic_count, format_count, values_count, total_enum_values
        nonlocal total_fields, with_format, with_values, with_role
        if not isinstance(node, dict):
            return
        role = node.get("x-polylogue-semantic-role")
        if role:
            semantic_count += 1
            roles.append(
                SchemaRoleAssignment(
                    path=path,
                    role=str(role),
                    confidence=float(node.get("x-polylogue-confidence", 0.0) or 0.0),
                    evidence=dict(node.get("x-polylogue-evidence", {})),
                )
            )
        if "x-polylogue-format" in node:
            format_count += 1
        if "x-polylogue-values" in node:
            values_count += 1
            total_enum_values += len(node["x-polylogue-values"])

        for name, child in node.get("properties", {}).items():
            if isinstance(child, dict):
                total_fields += 1
                if "x-polylogue-format" in child:
                    with_format += 1
                if "x-polylogue-values" in child:
                    with_values += 1
                if "x-polylogue-semantic-role" in child:
                    with_role += 1
                visit(child, path=f"{path}.{name}")
        if isinstance(node.get("items"), dict):
            visit(node["items"], path=f"{path}[*]")
        if isinstance(node.get("additionalProperties"), dict):
            visit(node["additionalProperties"], path=f"{path}.*")
        for keyword in ("anyOf", "oneOf", "allOf"):
            for child in node.get(keyword, []):
                if isinstance(child, dict):
                    visit(child, path=path)

    visit(schema, path="$")
    return SchemaAnnotationSummary(
        semantic_count=semantic_count,
        format_count=format_count,
        values_count=values_count,
        total_enum_values=total_enum_values,
        roles=sorted(roles, key=lambda item: (-item.confidence, item.path, item.role)),
        coverage=SchemaCoverageSummary(
            total_fields=total_fields,
            with_format=with_format,
            with_values=with_values,
            with_role=with_role,
        ),
    )


__all__ = ["collect_annotation_summary"]
