"""Shared health reporting types."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .lib.outcomes import OutcomeCheck, OutcomeReport, OutcomeStatus
from .maintenance_models import ArchiveDebtStatus, DerivedModelStatus, ReportProvenance

HealthCheck = OutcomeCheck
VerifyStatus = OutcomeStatus

HEALTH_TTL_SECONDS = 600


@dataclass
class HealthReport(OutcomeReport):
    """Comprehensive health and verification report."""

    timestamp: int = field(default_factory=lambda: int(time.time()))
    provenance: ReportProvenance = field(default_factory=ReportProvenance)
    derived_models: dict[str, DerivedModelStatus] = field(default_factory=dict)
    archive_debt: dict[str, ArchiveDebtStatus] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, int]:
        return self.summary_counts()

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "provenance": self.provenance.to_dict(),
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "count": check.count,
                    "detail": check.summary,
                    "breakdown": check.breakdown,
                }
                for check in self.checks
            ],
            "derived_models": {
                name: status.to_dict()
                for name, status in sorted(self.derived_models.items())
            },
            "archive_debt": {
                name: status.to_dict()
                for name, status in sorted(self.archive_debt.items())
            },
            "summary": self.summary,
        }


__all__ = [
    "HEALTH_TTL_SECONDS",
    "HealthCheck",
    "HealthReport",
    "VerifyStatus",
]
