"""Privacy redaction reporting for generated schemas."""

from __future__ import annotations

from polylogue.schemas.audit.walkers import SchemaNode, _walk_values
from polylogue.schemas.field_stats.stats import FieldStats
from polylogue.schemas.privacy import (
    _is_content_field,
    _is_safe_enum_value,
    _looks_high_entropy_token,
)
from polylogue.schemas.privacy_config import SchemaPrivacyConfig
from polylogue.schemas.redaction_report import (
    FieldReport,
    RedactionDecision,
    SchemaReport,
)


def _build_redaction_report(
    provider: str,
    stats: dict[str, FieldStats],
    schema: SchemaNode,
    *,
    privacy_config: SchemaPrivacyConfig | None = None,
    privacy_level: str = "standard",
) -> SchemaReport:
    report = SchemaReport(provider=provider, privacy_level=privacy_level)
    schema_values = {path: set(values) for path, values in _walk_values(schema)}

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
