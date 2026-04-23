# mypy: disable-error-code="no-untyped-def,arg-type,call-arg,attr-defined,dict-item,list-item"

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from polylogue.cli.check_models import CheckCommandResult
from polylogue.cli.check_rendering_plain import (
    build_report_lines,
    emit_maintenance_output,
    render_plain_output,
    status_icon,
)
from polylogue.cli.check_workflow import CheckCommandOptions
from polylogue.lib.outcomes import OutcomeCheck, OutcomeStatus
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
        "repair": False,
        "cleanup": False,
        "preview": False,
        "vacuum": False,
        "deep": False,
        "runtime": False,
        "check_blob": False,
        "check_schemas": False,
        "check_proof": False,
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
        "maintenance_targets": (),
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
    proof_report = SimpleNamespace(
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
            provider_name="fallback-provider",
            artifact_kind="tool_use",
            source_path="payload.json",
            resolved_package_version="v1",
            resolved_element_kind="tool_use",
            resolution_reason="schema",
        ),
        SimpleNamespace(
            support_status="unknown",
            payload_provider=None,
            provider_name="codex",
            artifact_kind="response",
            source_path="other.json",
            resolved_package_version=None,
            resolved_element_kind=None,
            resolution_reason=None,
        ),
    ]
    cohort_rows = [
        SimpleNamespace(
            provider_name="claude-code",
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
        proof_report=proof_report,
        artifact_rows=artifact_rows,
        cohort_rows=cohort_rows,
    )

    lines = build_report_lines(env, result, _options(verbose=True))
    rendered = "\n".join(lines)

    assert "db: busy" in rendered
    assert "codex: 5" in rendered
    assert "Summary: 1 ok, 1 warnings, 0 errors (source=live)" in rendered
    assert "Derived Models:" in rendered
    assert "Schema verification: 42 raw records" in rendered
    assert "Artifact proof: 12 artifact observations" in rendered
    assert "Claude subagents: linked_sidecars=3 orphan_sidecars=4 streams=2" in rendered
    assert "Artifact observations: 2 rows" in rendered
    assert "payload.json -> v1/tool_use [schema]" in rendered
    assert "Artifact cohorts: 1 cohorts" in rendered
    assert "Runtime Environment:" in rendered


def test_emit_maintenance_output_handles_preview_empty_selection_and_vacuum_modes() -> None:
    env = _env(plain=True)
    result = CheckCommandResult(
        report=ReadinessReport(),
        maintenance_results=[
            SimpleNamespace(
                repaired_count=2,
                success=True,
                category=SimpleNamespace(value="repair"),
                destructive=False,
                name="fts",
                detail="rebuilt",
            ),
            SimpleNamespace(
                repaired_count=0,
                success=False,
                category=SimpleNamespace(value="cleanup"),
                destructive=True,
                name="orphans",
                detail="failed",
            ),
        ],
        maintenance_targets=("session_products",),
    )
    with patch("polylogue.cli.check_rendering_plain.run_vacuum") as run_vacuum:
        emit_maintenance_output(env, result, _options(repair=True, preview=True, vacuum=True))
        run_vacuum.assert_not_called()

    printed = "\n".join(call.args[0] for call in env.ui.console.print.call_args_list)
    assert "OK fts [repair]: rebuilt" in printed
    assert "FAIL orphans [cleanup destructive]: failed" in printed
    assert "Preview mode: VACUUM skipped." in printed

    env_no_selection = _env(plain=False)
    with patch("polylogue.cli.check_rendering_plain.run_vacuum") as run_vacuum:
        emit_maintenance_output(
            env_no_selection,
            CheckCommandResult(report=ReadinessReport(), maintenance_results=None),
            _options(cleanup=True, vacuum=True),
        )
        run_vacuum.assert_called_once_with(env_no_selection)

    assert any(
        "No maintenance operations were selected." in call.args[0]
        for call in env_no_selection.ui.console.print.call_args_list
    )


def test_render_plain_output_delegates_to_summary_and_maintenance() -> None:
    env = _env(plain=True)
    result = CheckCommandResult(report=ReadinessReport())
    options = _options()

    with patch("polylogue.cli.check_rendering_plain.build_report_lines", return_value=["alpha"]) as build_lines:
        with patch("polylogue.cli.check_rendering_plain.emit_maintenance_output") as emit_maintenance:
            render_plain_output(env, result, options)

    build_lines.assert_called_once_with(env, result, options)
    env.ui.summary.assert_called_once_with("Health Check", ["alpha"])
    emit_maintenance.assert_called_once_with(env, result, options)
