"""Redaction report for schema generation.

Collects every filtering decision made during schema annotation and
provides structured + human-readable output for QA auditing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from polylogue.schemas.json_types import JSONDocument, json_document


@dataclass
class RedactionDecision:
    """A single filtering decision for a value in a field."""

    path: str
    value: str
    action: Literal["included", "rejected", "overridden_allow", "overridden_deny"]
    reason: str | None = None  # "high_entropy", "identifier_field", etc.
    count: int = 0  # occurrences in corpus
    conversation_count: int | None = None
    risk: Literal["none", "low", "medium", "high"] | None = None


@dataclass
class FieldReport:
    """Per-field redaction summary."""

    path: str
    included_values: list[str] = field(default_factory=list)
    rejected: list[RedactionDecision] = field(default_factory=list)
    content_field_blocked: bool = False
    identifier_field_blocked: bool = False


@dataclass
class SchemaReport:
    """Full redaction report for a provider schema generation run."""

    provider: str
    timestamp: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    privacy_level: str = "standard"
    total_fields: int = 0
    fields_with_enums: int = 0
    total_values_considered: int = 0
    total_included: int = 0
    total_rejected: int = 0
    field_reports: list[FieldReport] = field(default_factory=list)
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    borderline_decisions: list[RedactionDecision] = field(default_factory=list)

    def add_decision(self, decision: RedactionDecision) -> None:
        """Track a single redaction decision."""
        if decision.action == "rejected":
            self.total_rejected += 1
            reason = decision.reason or "unknown"
            self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
        elif decision.action == "included":
            self.total_included += 1

    def format_summary(self) -> str:
        """Compact stderr summary for display during generation."""
        lines = [
            f"schema[{self.provider}]: {self.total_fields} fields, "
            f"{self.total_values_considered} samples | privacy: {self.privacy_level}",
        ]

        if self.total_included or self.total_rejected:
            reason_parts = " ".join(
                f"{reason}:{count}" for reason, count in sorted(self.rejection_reasons.items(), key=lambda kv: -kv[1])
            )
            lines.append(f"  enums: {self.total_included} included, {self.total_rejected} rejected ({reason_parts})")

        if self.borderline_decisions:
            lines.append(
                f"  ⚠ {len(self.borderline_decisions)} borderline rejections "
                f"(>100 occurrences) — see report for details"
            )

        return "\n".join(lines)

    def format_markdown(self) -> str:
        """Full Markdown report for file output."""
        lines = [
            f"# Schema Redaction Report: {self.provider}",
            "",
            f"**Generated**: {self.timestamp}",
            f"**Privacy level**: {self.privacy_level}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Total fields | {self.total_fields} |",
            f"| Fields with enums | {self.fields_with_enums} |",
            f"| Values considered | {self.total_values_considered} |",
            f"| Values included | {self.total_included} |",
            f"| Values rejected | {self.total_rejected} |",
            "",
        ]

        if self.rejection_reasons:
            lines.append("## Rejection Breakdown")
            lines.append("")
            lines.append("| Reason | Count |")
            lines.append("| --- | ---: |")
            for reason, count in sorted(self.rejection_reasons.items(), key=lambda kv: -kv[1]):
                lines.append(f"| {reason} | {count} |")
            lines.append("")

        if self.borderline_decisions:
            lines.append("## Borderline Decisions")
            lines.append("")
            lines.append("Values that were rejected but have high occurrence counts (potential false positives):")
            lines.append("")
            lines.append("| Path | Value | Count | Reason |")
            lines.append("| --- | --- | ---: | --- |")
            for d in sorted(self.borderline_decisions, key=lambda d: -d.count):
                lines.append(f"| `{d.path}` | `{d.value}` | {d.count} | {d.reason or '?'} |")
            lines.append("")

            # Suggest overrides
            lines.append("### Suggested Overrides")
            lines.append("")
            lines.append("```toml")
            lines.append("[schema.privacy.field_overrides]")
            for d in self.borderline_decisions[:10]:
                lines.append(f'"{d.path}" = "allow"        # {d.value!r} (n={d.count}, rejected: {d.reason})')
            lines.append("```")
            lines.append("")

        if self.field_reports:
            lines.append("## Field Details")
            lines.append("")
            for fr in self.field_reports:
                if not fr.included_values and not fr.rejected:
                    continue
                lines.append(f"### `{fr.path}`")
                lines.append("")
                if fr.content_field_blocked:
                    lines.append("*Content field — all values suppressed*")
                elif fr.identifier_field_blocked:
                    lines.append("*Identifier field — non-structural values suppressed*")
                if fr.included_values:
                    vals = ", ".join(f"`{v}`" for v in fr.included_values[:20])
                    lines.append(f"**Included**: {vals}")
                if fr.rejected:
                    for rd in fr.rejected[:10]:
                        lines.append(f"- Rejected `{rd.value}` (n={rd.count}, {rd.reason})")
                lines.append("")

        return "\n".join(lines)

    def to_json(self) -> JSONDocument:
        """Structured JSON representation."""
        return json_document(
            {
                "provider": self.provider,
                "timestamp": self.timestamp,
                "privacy_level": self.privacy_level,
                "summary": {
                    "total_fields": self.total_fields,
                    "fields_with_enums": self.fields_with_enums,
                    "total_values_considered": self.total_values_considered,
                    "total_included": self.total_included,
                    "total_rejected": self.total_rejected,
                },
                "rejection_reasons": self.rejection_reasons,
                "borderline_decisions": [
                    {
                        "path": d.path,
                        "value": d.value,
                        "count": d.count,
                        "reason": d.reason,
                    }
                    for d in self.borderline_decisions
                ],
                "field_reports": [
                    {
                        "path": fr.path,
                        "included_count": len(fr.included_values),
                        "rejected_count": len(fr.rejected),
                        "content_field_blocked": fr.content_field_blocked,
                        "identifier_field_blocked": fr.identifier_field_blocked,
                    }
                    for fr in self.field_reports
                ],
            }
        )


__all__ = [
    "FieldReport",
    "RedactionDecision",
    "SchemaReport",
]
