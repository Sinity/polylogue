"""Automated schema quality audit checks.

Provides check functions that examine provider schemas for privacy leaks,
semantic misclassifications, and annotation quality.  Each check returns
a structured result with PASS/WARN/FAIL status.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polylogue.lib.outcomes import OutcomeCheck as CheckResult
from polylogue.lib.outcomes import OutcomeReport, OutcomeStatus
from polylogue.schemas.privacy import (
    _is_safe_enum_value,
    _looks_high_entropy_token,
)
from polylogue.schemas.runtime_registry import SchemaRegistry

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_HEX_RE = re.compile(r"^[0-9a-f]{16,}$", re.IGNORECASE)


@dataclass
class AuditReport(OutcomeReport):
    """Full audit report across all checks."""

    provider: str | None = None

    _LABELS = {
        OutcomeStatus.OK: "PASS",
        OutcomeStatus.WARNING: "WARN",
        OutcomeStatus.ERROR: "FAIL",
        OutcomeStatus.SKIP: "SKIP",
    }
    _ICONS = {
        OutcomeStatus.OK: "✓",
        OutcomeStatus.WARNING: "⚠",
        OutcomeStatus.ERROR: "✗",
        OutcomeStatus.SKIP: "◌",
    }

    @property
    def passed(self) -> int:
        return self.ok_count

    @property
    def warned(self) -> int:
        return self.warning_count

    @property
    def failed(self) -> int:
        return self.error_count

    @property
    def all_passed(self) -> bool:
        return self.all_ok

    def format_text(self) -> str:
        lines = []
        scope = f" ({self.provider})" if self.provider else ""
        lines.append(f"Schema Audit{scope}: {self.passed} pass, {self.warned} warn, {self.failed} fail")
        lines.append("")
        for c in self.checks:
            lines.append(f"  {c.format_line(labels=self._LABELS, icons=self._ICONS)}")
            for d in c.details[:5]:
                lines.append(f"      {d}")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "summary": {
                "passed": self.passed,
                "warned": self.warned,
                "failed": self.failed,
            },
            "checks": [
                {
                    "name": c.name,
                    "status": self._LABELS[c.status],
                    "message": c.summary,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


def _load_committed_schema(provider: str) -> dict[str, Any] | None:
    """Load a committed provider schema."""
    schema_root = Path(__file__).resolve().parent / "providers"
    return SchemaRegistry(storage_root=schema_root).get_schema(provider, version="default")


def _walk_values(schema: dict[str, Any], path: str = "$") -> list[tuple[str, list[str]]]:
    """Walk schema tree and collect all x-polylogue-values entries."""
    results: list[tuple[str, list[str]]] = []
    if not isinstance(schema, dict):
        return results

    values = schema.get("x-polylogue-values")
    if isinstance(values, list):
        results.append((path, [str(v) for v in values]))

    if "properties" in schema:
        for name, prop in schema["properties"].items():
            results.extend(_walk_values(prop, f"{path}.{name}"))
    if isinstance(schema.get("additionalProperties"), dict):
        results.extend(_walk_values(schema["additionalProperties"], f"{path}.*"))
    if isinstance(schema.get("items"), dict):
        results.extend(_walk_values(schema["items"], f"{path}[*]"))
    for kw in ("anyOf", "oneOf", "allOf"):
        if kw in schema:
            for sub in schema[kw]:
                if isinstance(sub, dict):
                    results.extend(_walk_values(sub, path))

    return results


def _walk_semantic_roles(schema: dict[str, Any], path: str = "$") -> list[tuple[str, str, float]]:
    """Walk schema and collect (path, role, confidence) tuples."""
    results: list[tuple[str, str, float]] = []
    if not isinstance(schema, dict):
        return results

    role = schema.get("x-polylogue-semantic-role")
    confidence = schema.get("x-polylogue-confidence", 0.0)
    if role:
        results.append((path, role, confidence))

    if "properties" in schema:
        for name, prop in schema["properties"].items():
            results.extend(_walk_semantic_roles(prop, f"{path}.{name}"))
    if isinstance(schema.get("additionalProperties"), dict):
        results.extend(_walk_semantic_roles(schema["additionalProperties"], f"{path}.*"))
    if isinstance(schema.get("items"), dict):
        results.extend(_walk_semantic_roles(schema["items"], f"{path}[*]"))
    for kw in ("anyOf", "oneOf", "allOf"):
        if kw in schema:
            for sub in schema[kw]:
                if isinstance(sub, dict):
                    results.extend(_walk_semantic_roles(sub, path))

    return results


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
        # UUID-format field should never be a title
        if role == "conversation_title":
            fmt = None
            # Walk to the path and check format
            parts = path.split(".")
            current = schema
            for part in parts[1:]:  # skip $
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

            # Check for ID-like field names
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
        "x-polylogue-format", "x-polylogue-values",
        "x-polylogue-semantic-role", "x-polylogue-frequency",
        "x-polylogue-range", "x-polylogue-multiline",
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
    status: OutcomeStatus
    # Large schemas with hundreds of auto-discovered nested properties
    # naturally have lower annotation percentages than small schemas.
    # Scale thresholds by schema size: >500 props → 5%/2%, else 30%/10%.
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

        # Check for basic metadata
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


def audit_provider(provider: str) -> AuditReport:
    """Run all audit checks on a single provider's committed schema."""
    report = AuditReport(provider=provider)

    schema = _load_committed_schema(provider)
    if schema is None:
        report.checks.append(CheckResult(
            name="schema_exists",
            status=OutcomeStatus.ERROR,
            summary=f"No committed schema found for {provider}",
        ))
        return report

    report.checks.append(CheckResult(
        name="schema_exists",
        status=OutcomeStatus.OK,
        summary="Committed schema loaded",
    ))
    report.checks.append(check_privacy_guards(schema))
    report.checks.append(check_semantic_roles(schema))
    report.checks.append(check_annotation_coverage(schema))

    return report


def audit_all_providers(providers: list[str] | None = None) -> AuditReport:
    """Run audit checks across all (or specified) providers."""
    from polylogue.schemas.sampling import PROVIDERS

    provider_list = providers or list(PROVIDERS.keys())
    report = AuditReport()

    schemas: dict[str, dict[str, Any]] = {}
    for provider in provider_list:
        provider_report = audit_provider(provider)
        report.checks.extend(provider_report.checks)
        schema = _load_committed_schema(provider)
        if schema:
            schemas[provider] = schema

    if len(schemas) >= 2:
        report.checks.append(check_cross_provider_consistency(schemas))

    return report


__all__ = [
    "AuditReport",
    "CheckResult",
    "audit_all_providers",
    "audit_provider",
    "check_annotation_coverage",
    "check_cross_provider_consistency",
    "check_privacy_guards",
    "check_semantic_roles",
]
