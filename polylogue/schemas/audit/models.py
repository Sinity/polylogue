"""Models for schema audit reporting."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.lib.json import JSONDocument, json_document
from polylogue.lib.outcomes import OutcomeCheck, OutcomeReport, OutcomeStatus


@dataclass
class AuditCheck(OutcomeCheck):
    """Schema-audit check annotated with its provider scope."""

    provider: str | None = None

    def to_json(self, *, label: str | None = None) -> JSONDocument:
        """Return the machine payload for this audit check."""
        return audit_check_json(self, provider=self.provider, label=label)


@dataclass(frozen=True)
class AuditSummary:
    """Machine-readable schema-audit summary counts."""

    passed: int
    warned: int
    failed: int

    def to_json(self) -> JSONDocument:
        return {
            "passed": self.passed,
            "warned": self.warned,
            "failed": self.failed,
        }


def audit_check_json(check: OutcomeCheck, *, provider: str | None, label: str | None) -> JSONDocument:
    """Return the schema-audit JSON payload for one check."""
    return {
        "name": check.name,
        "provider": provider,
        "status": label or check.status.value,
        "message": check.summary,
        "details": list(check.details),
    }


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

    @property
    def summary(self) -> AuditSummary:
        return AuditSummary(passed=self.passed, warned=self.warned, failed=self.failed)

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
                "summary": self.summary.to_json(),
                "checks": [
                    audit_check_json(
                        c,
                        provider=getattr(c, "provider", None),
                        label=self._LABELS[c.status],
                    )
                    for c in self.checks
                ],
            }
        )


__all__ = ["AuditCheck", "AuditReport", "AuditSummary", "audit_check_json"]
