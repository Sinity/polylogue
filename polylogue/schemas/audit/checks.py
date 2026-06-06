"""Atomic schema audit checks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from polylogue.core.json import json_document, json_document_list
from polylogue.core.outcomes import OutcomeCheck as CheckResult
from polylogue.core.outcomes import OutcomeStatus
from polylogue.schemas.audit.walkers import _HEX_RE, _UUID_RE, SchemaNode, _walk_semantic_roles, _walk_values
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
        if role == "session_title":
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
        "x-polylogue-high-cardinality-keys",
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
            if child.get("x-polylogue-high-cardinality-keys") is True:
                continue
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


def check_schema_staleness(schema: Mapping[str, object] | SchemaNode) -> CheckResult:
    """Check whether the committed schema is stale via its generated-at timestamp."""
    from datetime import UTC, datetime

    root = json_document(schema)
    generated_at = root.get("x-polylogue-generated-at")
    if not generated_at or not isinstance(generated_at, str):
        return CheckResult(
            name="schema_staleness",
            status=OutcomeStatus.WARNING,
            summary="Schema has no x-polylogue-generated-at timestamp",
        )
    try:
        gen_time = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        age_days = (datetime.now(UTC) - gen_time).days
    except ValueError:
        return CheckResult(
            name="schema_staleness",
            status=OutcomeStatus.WARNING,
            summary=f"Unparseable generated-at timestamp: {generated_at!r}",
        )
    if age_days > 90:
        return CheckResult(
            name="schema_staleness",
            status=OutcomeStatus.WARNING,
            summary=f"Schema may be stale — generated {age_days} days ago",
        )
    return CheckResult(
        name="schema_staleness",
        status=OutcomeStatus.OK,
        summary=f"Schema generated {age_days} days ago",
    )


def check_schema_drift(
    schema: Mapping[str, object] | SchemaNode,
    *,
    db_path: Path | None = None,
    provider: str = "",
    max_samples: int = 50,
) -> CheckResult:
    """Check whether committed schema has drifted from live archive data.

    Loads samples from the database and runs detect_drift() against the
    committed schema.  Reports fields present in live data but missing from
    the schema.
    """
    from polylogue.core.json import json_document
    from polylogue.schemas.sampling import load_samples_from_db
    from polylogue.schemas.validator import detect_drift

    if db_path is None or not Path(db_path).exists():
        return CheckResult(
            name="schema_drift",
            status=OutcomeStatus.SKIP,
            summary="No database available for drift detection",
        )

    try:
        samples = load_samples_from_db(provider, db_path=Path(db_path), max_samples=max_samples)
    except Exception as exc:
        return CheckResult(
            name="schema_drift",
            status=OutcomeStatus.WARNING,
            summary=f"Failed to load samples for drift detection: {exc}",
        )

    if not samples:
        return CheckResult(
            name="schema_drift",
            status=OutcomeStatus.SKIP,
            summary="No samples available for drift detection",
        )

    root = json_document(schema)
    all_drift: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        for warning in detect_drift(sample, root, ""):
            if warning not in seen:
                seen.add(warning)
                all_drift.append(warning)

    if not all_drift:
        return CheckResult(
            name="schema_drift",
            status=OutcomeStatus.OK,
            summary=f"No drift detected across {len(samples)} sample(s)",
            count=len(samples),
        )

    unique_drift = list(dict.fromkeys(all_drift))
    return CheckResult(
        name="schema_drift",
        status=OutcomeStatus.WARNING,
        summary=f"Schema drift detected: {len(unique_drift)} unexpected field(s) in live data",
        details=unique_drift[:50],
        count=len(samples),
    )


__all__ = [
    "CheckResult",
    "check_annotation_coverage",
    "check_cross_provider_consistency",
    "check_privacy_guards",
    "check_schema_drift",
    "check_schema_staleness",
    "check_semantic_roles",
]
