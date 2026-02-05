"""Integration tests for the check command with health reports.

These tests verify that HealthReport construction includes the required summary field.
Bug: HealthReport constructor was being called without the summary argument.
"""

from __future__ import annotations

import pytest

from polylogue.health import HealthCheck, HealthReport, VerifyStatus


class TestHealthReportConstruction:
    """Tests for proper HealthReport instantiation."""

    def test_health_report_requires_summary(self):
        """HealthReport must include summary dict with ok/warning/error counts."""
        checks = [
            HealthCheck("database", VerifyStatus.OK, detail="DB reachable"),
            HealthCheck("archive", VerifyStatus.WARNING, detail="Not found"),
        ]

        # Correct: include summary
        report = HealthReport(
            checks=checks,
            summary={"ok": 1, "warning": 1, "error": 0},
        )

        assert len(report.checks) == 2
        assert report.summary == {"ok": 1, "warning": 1, "error": 0}

    def test_health_report_summary_counts(self):
        """Summary should accurately reflect check status counts."""
        checks = [
            HealthCheck("check1", VerifyStatus.OK),
            HealthCheck("check2", VerifyStatus.OK),
            HealthCheck("check3", VerifyStatus.WARNING),
            HealthCheck("check4", VerifyStatus.ERROR),
        ]

        report = HealthReport(
            checks=checks,
            summary={"ok": 2, "warning": 1, "error": 1},
        )

        # Verify counts match
        assert report.summary["ok"] == 2
        assert report.summary["warning"] == 1
        assert report.summary["error"] == 1

    def test_health_report_to_dict_serialization(self):
        """HealthReport should serialize to dict with all required fields."""
        checks = [HealthCheck("test", VerifyStatus.OK, detail="OK")]
        report = HealthReport(checks=checks, summary={"ok": 1, "warning": 0, "error": 0})

        data = report.to_dict()

        assert "checks" in data
        assert "summary" in data
        assert "timestamp" in data
        assert data["summary"] == {"ok": 1, "warning": 0, "error": 0}

    def test_health_report_empty_checks(self):
        """HealthReport with no checks should still have summary."""
        report = HealthReport(checks=[], summary={"ok": 0, "warning": 0, "error": 0})

        assert len(report.checks) == 0
        assert report.summary == {"ok": 0, "warning": 0, "error": 0}
