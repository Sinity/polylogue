"""Direct tests for polylogue.schemas.redaction_report module.

Covers SchemaReport accumulation, format methods, and serialization.
"""
from __future__ import annotations

import json

from polylogue.schemas.redaction_report import (
    FieldReport,
    RedactionDecision,
    SchemaReport,
)


class TestSchemaReportAccumulation:
    def test_add_rejected_decision(self) -> None:
        report = SchemaReport(provider="chatgpt")
        report.add_decision(RedactionDecision(
            path="$.type", value="secret", action="rejected", reason="high_entropy",
        ))
        assert report.total_rejected == 1
        assert report.total_included == 0
        assert report.rejection_reasons == {"high_entropy": 1}

    def test_add_included_decision(self) -> None:
        report = SchemaReport(provider="chatgpt")
        report.add_decision(RedactionDecision(
            path="$.type", value="user", action="included",
        ))
        assert report.total_included == 1
        assert report.total_rejected == 0

    def test_multiple_rejections_accumulate(self) -> None:
        report = SchemaReport(provider="chatgpt")
        for reason in ["high_entropy", "high_entropy", "identifier_field"]:
            report.add_decision(RedactionDecision(
                path="$.x", value="v", action="rejected", reason=reason,
            ))
        assert report.total_rejected == 3
        assert report.rejection_reasons == {"high_entropy": 2, "identifier_field": 1}

    def test_overridden_decisions_not_counted(self) -> None:
        report = SchemaReport(provider="chatgpt")
        report.add_decision(RedactionDecision(
            path="$.x", value="v", action="overridden_allow",
        ))
        assert report.total_included == 0
        assert report.total_rejected == 0


class TestFormatSummary:
    def test_summary_with_data(self) -> None:
        report = SchemaReport(provider="chatgpt", total_fields=10)
        report.add_decision(RedactionDecision(
            path="$.x", value="v", action="rejected", reason="pii",
        ))
        summary = report.format_summary()
        assert "chatgpt" in summary
        assert "pii" in summary

    def test_summary_empty_report(self) -> None:
        report = SchemaReport(provider="test")
        summary = report.format_summary()
        assert "test" in summary


class TestFormatMarkdown:
    def test_markdown_has_header(self) -> None:
        report = SchemaReport(provider="claude-ai", total_fields=5)
        md = report.format_markdown()
        assert "# Schema Redaction Report: claude-ai" in md
        assert "## Summary" in md

    def test_markdown_includes_borderline(self) -> None:
        report = SchemaReport(provider="test")
        report.borderline_decisions = [
            RedactionDecision(path="$.x", value="borderline_val", action="rejected",
                            reason="high_entropy", count=150),
        ]
        md = report.format_markdown()
        assert "borderline_val" in md
        assert "Suggested Overrides" in md

    def test_markdown_includes_field_reports(self) -> None:
        report = SchemaReport(provider="test")
        report.field_reports = [
            FieldReport(
                path="$.status",
                included_values=["active", "pending"],
                content_field_blocked=False,
            ),
        ]
        md = report.format_markdown()
        assert "`$.status`" in md
        assert "active" in md


class TestToJson:
    def test_round_trip(self) -> None:
        report = SchemaReport(provider="chatgpt", total_fields=5, total_included=3, total_rejected=2)
        report.rejection_reasons = {"high_entropy": 2}
        report.field_reports = [
            FieldReport(path="$.x", included_values=["a"], rejected=[]),
        ]
        data = report.to_json()
        # Verify structure
        assert data["provider"] == "chatgpt"
        assert data["summary"]["total_fields"] == 5
        assert data["summary"]["total_included"] == 3
        assert data["summary"]["total_rejected"] == 2
        assert data["rejection_reasons"] == {"high_entropy": 2}
        assert len(data["field_reports"]) == 1
        # Should be JSON-serializable
        json.dumps(data)
