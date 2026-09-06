# mypy: disable-error-code="no-untyped-def,arg-type,call-arg,attr-defined,dict-item,list-item"

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from polylogue.cli.shared.check_models import CheckCommandResult
from polylogue.cli.shared.check_rendering_plain import (
    build_report_lines,
    render_plain_output,
    status_icon,
)
from polylogue.cli.shared.check_workflow import CheckCommandOptions
from polylogue.core.outcomes import OutcomeCheck, OutcomeStatus
from polylogue.readiness import ReadinessReport


def _env(*, plain: bool = True) -> SimpleNamespace:
    console = SimpleNamespace(print=MagicMock())
    ui = SimpleNamespace(
        plain=plain,
        console=console,
        summary=MagicMock(),
    )
    return SimpleNamespace(ui=ui)


def _options(**overrides: object) -> CheckCommandOptions:
    values = {
        "json_output": False,
        "verbose": False,
        "deep": False,
        "runtime": False,
        "check_daemon": False,
        "check_blob": False,
        "blob_integrity_full": False,
        "check_schemas": False,
        "check_artifact_coverage": False,
        "check_artifacts": False,
        "check_cohorts": False,
        "schema_providers": (),
        "artifact_providers": (),
        "artifact_statuses": (),
        "artifact_kinds": (),
        "artifact_limit": None,
        "artifact_offset": 0,
        "schema_samples": "auto",
        "schema_record_limit": None,
        "schema_record_offset": 0,
        "schema_quarantine_malformed": False,
    }
    values.update(overrides)
    return CheckCommandOptions(**values)


def test_status_icon_handles_unknown_status_in_plain_and_rich_modes() -> None:
    assert status_icon(OutcomeStatus.SKIP, plain=True) == "?"
    assert status_icon(OutcomeStatus.SKIP, plain=False) == "?"


def test_build_report_lines_renders_all_sections_and_breakdowns() -> None:
    env = _env(plain=True)
    report = ReadinessReport(
        checks=[
            OutcomeCheck("db", OutcomeStatus.WARNING, summary="busy", breakdown={"chatgpt": 2, "codex": 5}),
            OutcomeCheck("index", OutcomeStatus.OK, summary="ready"),
        ],
        derived_models={
            "session_profiles": SimpleNamespace(
                ready=True,
                materialized_documents=2,
                source_documents=3,
                materialized_rows=20,
                source_rows=30,
                pending_documents=1,
                pending_rows=10,
                stale_rows=0,
                orphan_rows=1,
                missing_provenance_rows=2,
            )
        },
    )
    runtime_report = ReadinessReport(checks=[OutcomeCheck("sqlite", OutcomeStatus.ERROR, summary="missing")])
    schema_report = SimpleNamespace(
        total_records=42,
        max_samples=None,
        record_limit=5,
        record_offset=2,
        providers={
            "claude-code": SimpleNamespace(
                valid_records=4,
                invalid_records=1,
                drift_records=2,
                skipped_no_schema=3,
                decode_errors=4,
                quarantined_records=5,
            )
        },
    )
    coverage_report = SimpleNamespace(
        total_records=12,
        contract_backed_records=9,
        unsupported_parseable_records=1,
        recognized_non_parseable_records=1,
        unknown_records=1,
        decode_errors=0,
        subagent_streams=2,
        linked_sidecars=3,
        orphan_sidecars=4,
        package_versions={"v1": 2},
        element_kinds={"tool_use": 5},
        resolution_reasons={"inferred": 1},
        providers={
            "claude-code": SimpleNamespace(
                contract_backed_records=5,
                unsupported_parseable_records=1,
                recognized_non_parseable_records=0,
                unknown_records=0,
                decode_errors=0,
                package_versions={"v1": 2},
                element_kinds={"tool_use": 5},
                resolution_reasons={"inferred": 1},
            )
        },
    )
    artifact_rows = [
        SimpleNamespace(
            support_status="contract_backed",
            payload_provider="claude-code",
            source_name="fallback-provider",
            artifact_kind="tool_use",
            source_path="payload.json",
            resolved_package_version="v1",
            resolved_element_kind="tool_use",
            resolution_reason="schema",
        ),
        SimpleNamespace(
            support_status="unknown",
            payload_provider=None,
            source_name="codex",
            artifact_kind="response",
            source_path="other.json",
            resolved_package_version=None,
            resolved_element_kind=None,
            resolution_reason=None,
        ),
    ]
    cohort_rows = [
        SimpleNamespace(
            source_name="claude-code",
            artifact_kind="tool_use",
            support_status="contract_backed",
            observation_count=7,
            cohort_id="cohort-1",
            resolved_package_version="v1",
            resolved_element_kind="tool_use",
        )
    ]
    result = CheckCommandResult(
        report=report,
        runtime_report=runtime_report,
        schema_report=schema_report,
        coverage_report=coverage_report,
        artifact_rows=artifact_rows,
        cohort_rows=cohort_rows,
        daemon_report={
            "live": {
                "source_count": 2,
                "existing_source_count": 1,
                "sources": [
                    {"name": "codex", "root": "/tmp/codex", "exists": True},
                    {"name": "claude-code", "root": "/tmp/claude", "exists": False},
                ],
            },
            "browser_capture": {"spool_ready": True},
        },
    )

    lines = build_report_lines(env, result, _options(verbose=True))
    rendered = "\n".join(lines)

    assert "db: busy" in rendered
    assert "codex: 5" in rendered
    assert "Summary: 1 ok, 1 warnings, 0 errors (source=live)" in rendered
    assert "Derived Models:" in rendered
    assert "Schema verification: 42 raw records" in rendered
    assert "Artifact coverage: 12 artifact observations" in rendered
    assert "Claude subagents: linked_sidecars=3 orphan_sidecars=4 streams=2" in rendered
    assert "Artifact observations: 2 rows" in rendered
    assert "payload.json -> v1/tool_use [schema]" in rendered
    assert "Artifact cohorts: 1 cohorts" in rendered
    assert "Runtime Environment:" in rendered
    assert "Daemon Components:" in rendered
    assert "Live sources: 1/2 available" in rendered
    assert "codex: /tmp/codex (available)" in rendered
    assert "Browser capture spool: ready" in rendered


def test_render_plain_output_delegates_to_summary() -> None:
    env = _env(plain=True)
    result = CheckCommandResult(report=ReadinessReport())
    options = _options()

    with patch("polylogue.cli.shared.check_rendering_plain.build_report_lines", return_value=["alpha"]) as build_lines:
        render_plain_output(env, result, options)

    build_lines.assert_called_once_with(env, result, options)
    env.ui.summary.assert_called_once_with("Health Check", ["alpha"])
