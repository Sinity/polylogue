"""Models for schema audit reporting."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.lib.outcomes import OutcomeCheck, OutcomeReport, OutcomeStatus
from polylogue.schemas.json_types import JSONDocument, json_document


@dataclass
class AuditCheck(OutcomeCheck):
    """Schema-audit check annotated with its provider scope."""

    provider: str | None = None


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
            check_provider = getattr(c, "provider", None)
            prefix = f"[{check_provider}] " if check_provider and not self.provider else ""
            lines.append(f"  {prefix}{c.format_line(labels=self._LABELS, icons=self._ICONS)}")
            for d in c.details[:5]:
                lines.append(f"      {d}")
        return "\n".join(lines)

    def to_json(self) -> JSONDocument:
        return json_document(
            {
                "provider": self.provider,
                "summary": {
                    "passed": self.passed,
                    "warned": self.warned,
                    "failed": self.failed,
                },
                "checks": [
                    {
                        "name": c.name,
                        "provider": getattr(c, "provider", None),
                        "status": self._LABELS[c.status],
                        "message": c.summary,
                        "details": c.details,
                    }
                    for c in self.checks
                ],
            }
        )


__all__ = ["AuditCheck", "AuditReport"]
