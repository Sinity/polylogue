"""Models for schema audit reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from polylogue.lib.outcomes import OutcomeReport, OutcomeStatus


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


__all__ = ["AuditReport"]
