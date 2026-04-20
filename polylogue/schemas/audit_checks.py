"""Atomic schema audit checks."""

from __future__ import annotations

from collections.abc import Mapping

from polylogue.lib.outcomes import OutcomeCheck as CheckResult
from polylogue.lib.outcomes import OutcomeStatus
from polylogue.schemas.audit_walkers import _HEX_RE, _UUID_RE, SchemaNode, _walk_semantic_roles, _walk_values
from polylogue.schemas.json_types import json_document, json_document_list
from polylogue.schemas.privacy import _is_safe_enum_value, _looks_high_entropy_token


def check_privacy_guards(schema: Mapping[str, object] | SchemaNode) -> CheckResult:
    """Check that no UUIDs, hashes, or PII leak through enum values."""
    violations: list[str] = []
    root = json_document(schema)

    for path, values in _walk_values(root):
        for v in values:
            if _UUID_RE.match(v):
                violations.append(f"{path}: UUID leak {v!r}")
            elif _HEX_RE.match(v):
                violations.append(f"{path}: hex-id leak {v!r}")
            elif _looks_high_entropy_token(v):
                violations.append(f"{path}: high-entropy token {v!r}")
            elif not _is_safe_enum_value(v, path=path):
                violations.append(f"{path}: unsafe value {v!r}")

    if violations:
        return CheckResult(
            name="privacy_guards",
            status=OutcomeStatus.ERROR,
            summary=f"{len(violations)} unsafe enum value(s) found",
            details=violations[:20],
        )
    return CheckResult(
        name="privacy_guards",
        status=OutcomeStatus.OK,
        summary="All enum values pass privacy checks",
    )


def _schema_node_at_path(schema: Mapping[str, object] | SchemaNode, path: str) -> SchemaNode:
    current = json_document(schema)
    for part in path.split(".")[1:]:
        if part == "*":
            current = json_document(current.get("additionalProperties"))
            continue
        if part.endswith("[*]"):
            name = part[:-3]
            if name:
                current = json_document(json_document(current.get("properties")).get(name))
            current = json_document(current.get("items"))
            continue
        current = json_document(json_document(current.get("properties")).get(part))
    return current


def check_semantic_roles(schema: Mapping[str, object] | SchemaNode) -> CheckResult:
    """Check semantic role assignments for sanity."""
    issues: list[str] = []
    root = json_document(schema)
    roles = _walk_semantic_roles(root)

    for path, role, _confidence in roles:
        if role == "conversation_title":
            current = _schema_node_at_path(root, path)
            fmt = current.get("x-polylogue-format")

            if fmt in ("uuid4", "uuid", "hex-id"):
                issues.append(f"{path}: {fmt}-format field assigned as {role}")

            terminal = path.rsplit(".", 1)[-1].lower()
            if terminal.endswith(("id", "_id", "uuid")):
                issues.append(f"{path}: ID-like field assigned as {role}")

    if issues:
        return CheckResult(
            name="semantic_roles",
            status=OutcomeStatus.ERROR,
            summary=f"{len(issues)} misclassified semantic role(s)",
            details=issues[:10],
        )

    if not roles:
        return CheckResult(
            name="semantic_roles",
            status=OutcomeStatus.WARNING,
            summary="No semantic roles detected",
        )

    return CheckResult(
        name="semantic_roles",
        status=OutcomeStatus.OK,
        summary=f"{len(roles)} role(s) assigned correctly",
    )


def check_annotation_coverage(schema: Mapping[str, object] | SchemaNode) -> CheckResult:
    """Check that schema has adequate annotation coverage."""
    total_fields = 0
    annotated_fields = 0
    annotation_keys = {
        "x-polylogue-format",
        "x-polylogue-values",
        "x-polylogue-semantic-role",
        "x-polylogue-frequency",
        "x-polylogue-range",
        "x-polylogue-multiline",
    }

    def _count(node: SchemaNode) -> None:
        nonlocal total_fields, annotated_fields
        properties = json_document(node.get("properties"))
        for prop in properties.values():
            child = json_document(prop)
            if not child:
                continue
            total_fields += 1
            if any(key in child for key in annotation_keys):
                annotated_fields += 1
            _count(child)
        items = json_document(node.get("items"))
        if items:
            _count(items)
        additional_properties = json_document(node.get("additionalProperties"))
        if additional_properties:
            _count(additional_properties)
        for keyword in ("anyOf", "oneOf", "allOf"):
            for child in json_document_list(node.get(keyword)):
                _count(child)

    _count(json_document(schema))

    if total_fields == 0:
        return CheckResult(
            name="annotation_coverage",
            status=OutcomeStatus.WARNING,
            summary="No properties found in schema",
        )

    pct = (annotated_fields / total_fields) * 100
    if total_fields > 500:
        pass_pct, warn_pct = 5, 2
    else:
        pass_pct, warn_pct = 30, 10
    if pct >= pass_pct:
        status = OutcomeStatus.OK
    elif pct >= warn_pct:
        status = OutcomeStatus.WARNING
    else:
        status = OutcomeStatus.ERROR

    return CheckResult(
        name="annotation_coverage",
        status=status,
        summary=f"{annotated_fields}/{total_fields} fields annotated ({pct:.0f}%)",
    )


def check_cross_provider_consistency(schemas: Mapping[str, Mapping[str, object] | SchemaNode]) -> CheckResult:
    """Check consistency across all provider schemas."""
    issues: list[str] = []

    for provider, schema in schemas.items():
        root = json_document(schema)
        roles = _walk_semantic_roles(root)
        if not roles:
            issues.append(f"{provider}: no semantic roles")

        if not root.get("x-polylogue-sample-count"):
            issues.append(f"{provider}: missing sample count")

    if issues:
        return CheckResult(
            name="cross_provider_consistency",
            status=OutcomeStatus.WARNING,
            summary=f"{len(issues)} consistency issue(s)",
            details=issues,
        )

    return CheckResult(
        name="cross_provider_consistency",
        status=OutcomeStatus.OK,
        summary=f"All {len(schemas)} provider schemas consistent",
    )


__all__ = [
    "CheckResult",
    "check_annotation_coverage",
    "check_cross_provider_consistency",
    "check_privacy_guards",
    "check_semantic_roles",
]
