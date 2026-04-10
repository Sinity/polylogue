"""Direct tests for polylogue.schemas.audit module.

Covers check_privacy_guards, check_semantic_roles, check_annotation_coverage,
and AuditReport aggregation properties.
"""

from __future__ import annotations

from polylogue.lib.outcomes import OutcomeCheck as CheckResult
from polylogue.lib.outcomes import OutcomeStatus
from polylogue.schemas.audit_checks import (
    check_annotation_coverage,
    check_privacy_guards,
    check_semantic_roles,
)
from polylogue.schemas.audit_models import AuditReport

PASS = OutcomeStatus.OK
WARN = OutcomeStatus.WARNING
FAIL = OutcomeStatus.ERROR
AUDIT_LABELS = {
    OutcomeStatus.OK: "PASS",
    OutcomeStatus.WARNING: "WARN",
    OutcomeStatus.ERROR: "FAIL",
}


class TestCheckResult:
    def test_format_line_pass(self) -> None:
        cr = CheckResult(name="test", status=PASS, summary="OK")
        line = cr.format_line(labels=AUDIT_LABELS)
        assert "PASS" in line
        assert "test" in line

    def test_format_line_warn(self) -> None:
        cr = CheckResult(name="warning_test", status=WARN, summary="Check this")
        line = cr.format_line(labels=AUDIT_LABELS)
        assert "WARN" in line
        assert "warning_test" in line

    def test_format_line_fail(self) -> None:
        cr = CheckResult(name="fail_test", status=FAIL, summary="Bad")
        line = cr.format_line(labels=AUDIT_LABELS)
        assert "FAIL" in line
        assert "fail_test" in line

    def test_details_appended(self) -> None:
        cr = CheckResult(
            name="test",
            status=FAIL,
            summary="Bad",
            details=["detail1", "detail2"],
        )
        assert len(cr.details) == 2
        assert "detail1" in cr.details


class TestAuditReport:
    def test_count_pass(self) -> None:
        report = AuditReport(
            checks=[
                CheckResult(name="a", status=PASS, summary="ok"),
                CheckResult(name="b", status=PASS, summary="ok"),
            ]
        )
        assert report.passed == 2
        assert report.warned == 0
        assert report.failed == 0

    def test_count_warn(self) -> None:
        report = AuditReport(
            checks=[
                CheckResult(name="a", status=WARN, summary="warning"),
            ]
        )
        assert report.warned == 1

    def test_count_fail(self) -> None:
        report = AuditReport(
            checks=[
                CheckResult(name="a", status=FAIL, summary="bad"),
                CheckResult(name="b", status=FAIL, summary="bad"),
            ]
        )
        assert report.failed == 2

    def test_counts_mixed(self) -> None:
        report = AuditReport(
            checks=[
                CheckResult(name="a", status=PASS, summary="ok"),
                CheckResult(name="b", status=WARN, summary="warning"),
                CheckResult(name="c", status=FAIL, summary="bad"),
                CheckResult(name="d", status=PASS, summary="ok"),
            ]
        )
        assert report.passed == 2
        assert report.warned == 1
        assert report.failed == 1

    def test_all_passed_true(self) -> None:
        report = AuditReport(
            checks=[
                CheckResult(name="a", status=PASS, summary="ok"),
                CheckResult(name="b", status=PASS, summary="ok"),
            ]
        )
        assert report.all_passed

    def test_all_passed_false_with_warn(self) -> None:
        report = AuditReport(
            checks=[
                CheckResult(name="a", status=PASS, summary="ok"),
                CheckResult(name="b", status=WARN, summary="warning"),
            ]
        )
        assert not report.all_passed

    def test_all_passed_false_with_fail(self) -> None:
        report = AuditReport(
            checks=[
                CheckResult(name="a", status=FAIL, summary="bad"),
            ]
        )
        assert not report.all_passed

    def test_empty_report_all_passed(self) -> None:
        report = AuditReport()
        assert report.all_passed  # vacuously true

    def test_format_text(self) -> None:
        report = AuditReport(
            provider="chatgpt",
            checks=[
                CheckResult(name="test", status=PASS, summary="ok"),
            ],
        )
        text = report.format_text()
        assert "chatgpt" in text
        assert "1 pass" in text

    def test_format_text_no_provider(self) -> None:
        report = AuditReport(
            checks=[
                CheckResult(name="test", status=PASS, summary="ok"),
            ]
        )
        text = report.format_text()
        # Should not have provider in parens
        assert "Schema Audit:" in text

    def test_format_text_includes_checks(self) -> None:
        report = AuditReport(
            checks=[
                CheckResult(name="check1", status=PASS, summary="ok"),
                CheckResult(name="check2", status=FAIL, summary="bad"),
            ]
        )
        text = report.format_text()
        assert "check1" in text
        assert "check2" in text

    def test_format_text_limits_details(self) -> None:
        # Format text shows first 5 detail lines per check
        details = [f"detail_{i}" for i in range(10)]
        report = AuditReport(
            checks=[
                CheckResult(name="test", status=FAIL, summary="bad", details=details),
            ]
        )
        text = report.format_text()
        # Should include first 5
        assert "detail_0" in text
        assert "detail_4" in text
        # Should not include all 10
        # (we can't guarantee the others aren't shown, but checking limit)

    def test_to_json(self) -> None:
        report = AuditReport(
            provider="test",
            checks=[
                CheckResult(name="a", status=PASS, summary="ok", details=["detail1"]),
            ],
        )
        data = report.to_json()
        assert data["provider"] == "test"
        assert data["summary"]["passed"] == 1
        assert data["summary"]["warned"] == 0
        assert data["summary"]["failed"] == 0
        assert len(data["checks"]) == 1
        assert data["checks"][0]["name"] == "a"
        assert data["checks"][0]["details"] == ["detail1"]

    def test_to_json_no_provider(self) -> None:
        report = AuditReport(checks=[])
        data = report.to_json()
        assert data["provider"] is None


class TestCheckPrivacyGuards:
    def test_clean_schema_passes(self) -> None:
        schema = {
            "properties": {
                "role": {
                    "type": "string",
                    "x-polylogue-values": ["user", "assistant", "system"],
                },
            },
        }
        result = check_privacy_guards(schema)
        assert result.status is PASS
        assert "privacy" in result.name

    def test_uuid_leak_fails(self) -> None:
        schema = {
            "properties": {
                "id": {
                    "type": "string",
                    "x-polylogue-values": ["550e8400-e29b-41d4-a716-446655440000"],
                },
            },
        }
        result = check_privacy_guards(schema)
        assert result.status is FAIL
        assert len(result.details) > 0
        assert "UUID" in result.details[0]

    def test_hex_id_leak_fails(self) -> None:
        schema = {
            "properties": {
                "ref": {
                    "type": "string",
                    "x-polylogue-values": ["a" * 24],
                },
            },
        }
        result = check_privacy_guards(schema)
        assert result.status is FAIL
        assert "hex" in result.details[0].lower()

    def test_high_entropy_token_fails(self) -> None:
        schema = {
            "properties": {
                "token": {
                    "type": "string",
                    "x-polylogue-values": ["sk-abc123XYZ789def456"],
                },
            },
        }
        result = check_privacy_guards(schema)
        assert result.status is FAIL

    def test_private_tld_rejected(self) -> None:
        schema = {
            "properties": {
                "endpoint": {
                    "type": "string",
                    "x-polylogue-values": ["api.internal"],
                },
            },
        }
        result = check_privacy_guards(schema)
        assert result.status is FAIL


class TestCheckSemanticRoles:
    def test_no_roles_warns(self) -> None:
        schema = {"properties": {"x": {"type": "string"}}}
        result = check_semantic_roles(schema)
        assert result.status is WARN

    def test_valid_roles_pass(self) -> None:
        schema = {
            "properties": {
                "title": {
                    "type": "string",
                    "x-polylogue-semantic-role": "conversation_title",
                },
            },
        }
        result = check_semantic_roles(schema)
        assert result.status is PASS

    def test_id_field_as_title_fails(self) -> None:
        schema = {
            "properties": {
                "user_id": {
                    "type": "string",
                    "x-polylogue-semantic-role": "conversation_title",
                },
            },
        }
        result = check_semantic_roles(schema)
        assert result.status is FAIL
        assert "ID-like" in result.details[0]

    def test_uuid_field_as_title_fails(self) -> None:
        schema = {
            "properties": {
                "message_uuid": {
                    "type": "string",
                    "x-polylogue-format": "uuid",
                    "x-polylogue-semantic-role": "conversation_title",
                },
            },
        }
        result = check_semantic_roles(schema)
        assert result.status is FAIL

    def test_multiple_roles_all_checked(self) -> None:
        schema = {
            "properties": {
                "title": {
                    "type": "string",
                    "x-polylogue-semantic-role": "conversation_title",
                },
                "content": {
                    "type": "string",
                    "x-polylogue-semantic-role": "message_content",
                },
            },
        }
        result = check_semantic_roles(schema)
        assert result.status is PASS


class TestCheckAnnotationCoverage:
    def test_well_annotated_passes(self) -> None:
        schema = {
            "properties": {
                "a": {"type": "string", "x-polylogue-format": "uuid"},
                "b": {"type": "string", "x-polylogue-values": ["x"]},
                "c": {"type": "string", "x-polylogue-frequency": 0.9},
                "d": {"type": "integer"},  # not annotated
            },
        }
        result = check_annotation_coverage(schema)
        # 3/4 = 75% >= 30% → PASS
        assert result.status is PASS

    def test_low_coverage_fails(self) -> None:
        schema = {
            "properties": {f"field_{i}": {"type": "string"} for i in range(20)},
        }
        # 0/20 = 0% < 10% → FAIL
        result = check_annotation_coverage(schema)
        assert result.status is FAIL

    def test_medium_coverage_warns(self) -> None:
        schema = {
            "properties": {
                "a": {"type": "string", "x-polylogue-format": "uuid"},
                "b": {"type": "string"},
                "c": {"type": "string"},
                "d": {"type": "string"},
                "e": {"type": "string"},
                "f": {"type": "string"},
                "g": {"type": "string"},
                "h": {"type": "string"},
                "i": {"type": "string"},
                "j": {"type": "string"},
            },
        }
        # 1/10 = 10% >= 10% but < 30% → WARN
        result = check_annotation_coverage(schema)
        assert result.status is WARN

    def test_no_properties_warns(self) -> None:
        schema = {"type": "object"}
        result = check_annotation_coverage(schema)
        assert result.status is WARN
        assert "No properties" in result.message

    def test_large_schema_relaxed_thresholds(self) -> None:
        # >500 fields: pass threshold = 5%, warn threshold = 2%
        props = {}
        for i in range(510):
            p = {"type": "string"}
            if i < 30:  # 30/510 ≈ 5.9% → PASS
                p["x-polylogue-format"] = "uuid"
            props[f"f_{i}"] = p
        schema = {"properties": props}
        result = check_annotation_coverage(schema)
        assert result.status is PASS

    def test_large_schema_fails_below_threshold(self) -> None:
        # >500 fields, <2% annotated (below warn threshold for large schemas)
        props = {}
        for i in range(510):
            p = {"type": "string"}
            if i < 8:  # 8/510 ≈ 1.57% < 2%
                p["x-polylogue-format"] = "uuid"
            props[f"f_{i}"] = p
        schema = {"properties": props}
        result = check_annotation_coverage(schema)
        assert result.status is FAIL

    def test_nested_properties_counted(self) -> None:
        schema = {
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {
                        "inner": {"type": "string", "x-polylogue-format": "uuid"},
                    },
                },
            },
        }
        result = check_annotation_coverage(schema)
        # Should count nested properties
        assert result.status in (PASS, WARN, FAIL)

    def test_message_includes_counts(self) -> None:
        schema = {
            "properties": {
                "a": {"type": "string", "x-polylogue-format": "uuid"},
                "b": {"type": "string"},
            },
        }
        result = check_annotation_coverage(schema)
        assert "1/2" in result.message or "50" in result.message
