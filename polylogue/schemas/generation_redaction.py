"""Privacy redaction reporting for generated schemas."""

from __future__ import annotations

from typing import Any

from polylogue.schemas.field_stats import FieldStats
from polylogue.schemas.privacy import (
    _is_content_field,
    _is_safe_enum_value,
    _looks_high_entropy_token,
)
from polylogue.schemas.redaction_report import (
    FieldReport,
    RedactionDecision,
    SchemaReport,
)


def _build_redaction_report(
    provider: str,
    stats: dict[str, FieldStats],
    schema: dict[str, Any],
    *,
    privacy_config: Any | None = None,
    privacy_level: str = "standard",
) -> SchemaReport:
    report = SchemaReport(provider=provider, privacy_level=privacy_level)
    schema_values: dict[str, set[str]] = {}

    def _collect_schema_values(s: dict[str, Any], path: str = "$") -> None:
        if not isinstance(s, dict):
            return
        vals = s.get("x-polylogue-values")
        if isinstance(vals, list):
            schema_values[path] = {str(v) for v in vals}
        if "properties" in s:
            for name, prop in s["properties"].items():
                _collect_schema_values(prop, f"{path}.{name}")
        if isinstance(s.get("additionalProperties"), dict):
            _collect_schema_values(s["additionalProperties"], f"{path}.*")
        if isinstance(s.get("items"), dict):
            _collect_schema_values(s["items"], f"{path}[*]")
        for kw in ("anyOf", "oneOf", "allOf"):
            for sub in s.get(kw, []):
                if isinstance(sub, dict):
                    _collect_schema_values(sub, path)

    _collect_schema_values(schema)

    for path, fs in stats.items():
        if not fs.is_enum_like or not fs.observed_values:
            continue

        report.total_fields += 1
        report.fields_with_enums += 1
        included_in_schema = schema_values.get(path, set())
        field_report = FieldReport(path=path)

        if _is_content_field(path):
            field_report.content_field_blocked = True

        for value, count in fs.observed_values.most_common():
            if not isinstance(value, str):
                continue
            report.total_values_considered += 1

            if value in included_in_schema:
                decision = RedactionDecision(path=path, value=value, action="included", count=count)
                field_report.included_values.append(value)
            else:
                reason = "unknown"
                if _is_content_field(path):
                    reason = "content_field"
                elif _looks_high_entropy_token(value):
                    reason = "high_entropy"
                elif not _is_safe_enum_value(value, path=path, config=privacy_config):
                    reason = "unsafe_value"
                else:
                    reason = "threshold"

                decision = RedactionDecision(
                    path=path,
                    value=value,
                    action="rejected",
                    reason=reason,
                    count=count,
                )
                field_report.rejected.append(decision)

                if count >= 100:
                    decision.risk = "medium"
                    report.borderline_decisions.append(decision)

            report.add_decision(decision)

        if field_report.included_values or field_report.rejected:
            report.field_reports.append(field_report)

    return report


__all__ = ["_build_redaction_report"]
