"""Atomic schema audit checks."""

from __future__ import annotations

from typing import Any

from polylogue.lib.outcomes import OutcomeCheck as CheckResult
from polylogue.lib.outcomes import OutcomeStatus
from polylogue.schemas.audit_walkers import _HEX_RE, _UUID_RE, _walk_semantic_roles, _walk_values
from polylogue.schemas.privacy import _is_safe_enum_value, _looks_high_entropy_token


def check_privacy_guards(schema: dict[str, Any]) -> CheckResult:
    """Check that no UUIDs, hashes, or PII leak through enum values."""
    violations: list[str] = []

    for path, values in _walk_values(schema):
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


def check_semantic_roles(schema: dict[str, Any]) -> CheckResult:
    """Check semantic role assignments for sanity."""
    issues: list[str] = []
    roles = _walk_semantic_roles(schema)

    for path, role, _confidence in roles:
        if role == "conversation_title":
            fmt = None
            parts = path.split(".")
            current = schema
            for part in parts[1:]:
                if part == "*":
                    current = current.get("additionalProperties", {})
                elif "[*]" in part:
                    name = part.replace("[*]", "")
                    if name:
                        current = current.get("properties", {}).get(name, {})
                    current = current.get("items", {})
                else:
                    current = current.get("properties", {}).get(part, {})
            if isinstance(current, dict):
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


def check_annotation_coverage(schema: dict[str, Any]) -> CheckResult:
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

    def _count(s: dict[str, Any]) -> None:
        nonlocal total_fields, annotated_fields
        if not isinstance(s, dict):
            return
        if "properties" in s:
            for prop in s["properties"].values():
                if isinstance(prop, dict):
                    total_fields += 1
                    if any(k in prop for k in annotation_keys):
                        annotated_fields += 1
                    _count(prop)
        if isinstance(s.get("items"), dict):
            _count(s["items"])
        if isinstance(s.get("additionalProperties"), dict):
            _count(s["additionalProperties"])
        for kw in ("anyOf", "oneOf", "allOf"):
            if kw in s:
                for sub in s[kw]:
                    if isinstance(sub, dict):
                        _count(sub)

    _count(schema)

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


def check_cross_provider_consistency(schemas: dict[str, dict[str, Any]]) -> CheckResult:
    """Check consistency across all provider schemas."""
    issues: list[str] = []

    for provider, schema in schemas.items():
        roles = _walk_semantic_roles(schema)
        if not roles:
            issues.append(f"{provider}: no semantic roles")

        if not schema.get("x-polylogue-sample-count"):
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
