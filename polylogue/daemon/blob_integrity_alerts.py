"""Daemon-health alert rendering for blob-integrity reports."""

from __future__ import annotations

from collections.abc import Callable

from polylogue.daemon.health import HealthAlert, HealthSeverity, HealthTier
from polylogue.storage.blob_integrity import BlobIntegrityReport


def blob_integrity_alerts_from_report(
    report: BlobIntegrityReport,
    checked_at: str,
    record_failure: Callable[[str, bool], int],
) -> list[HealthAlert]:
    def alert(name: str, severity: HealthSeverity, message: str, *, healthy: bool) -> HealthAlert:
        return HealthAlert(
            check_name=name,
            tier=HealthTier.EXPENSIVE,
            severity=severity,
            message=message,
            checked_at=checked_at,
            consecutive_failures=record_failure(name, healthy),
        )

    if report.ok:
        message = (
            f"blob integrity ok ({report.scanned_blobs}/{report.total_blobs_seen} blobs, "
            f"{report.scanned_references}/{report.total_references_seen} references checked)"
        )
        return [alert("blob_integrity", HealthSeverity.OK, message, healthy=True)]

    alerts: list[HealthAlert] = []
    for finding in report.findings:
        severity = HealthSeverity.CRITICAL if finding.severity == "critical" else HealthSeverity.WARNING
        sample = ", ".join(f"{item[:12]}..." for item in finding.sample)
        suffix = f"; sample={sample}" if sample else ""
        if finding.bytes_total:
            suffix += f"; bytes={finding.bytes_total:,}"
        alerts.append(
            alert(
                f"blob_integrity.{finding.kind}",
                severity,
                f"{finding.kind}: {finding.count} finding(s){suffix}; action={finding.suggested_action}",
                healthy=False,
            )
        )
    return alerts


__all__ = ["blob_integrity_alerts_from_report"]
