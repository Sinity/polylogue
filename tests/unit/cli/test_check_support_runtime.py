from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from polylogue.cli import check_workflow, formatting
from polylogue.cli.check_models import CheckCommandResult, VacuumResult
from polylogue.cli.check_workflow import CheckCommandOptions
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.maintenance.targets import MaintenanceTargetMode
from polylogue.readiness import ReadinessCheck, ReadinessReport, VerifyStatus
from polylogue.schemas.verification_models import SchemaVerificationReport
from polylogue.storage.repair import RepairResult


def _env() -> AppEnv:
    ui = MagicMock()
    ui.console = MagicMock()
    return cast(AppEnv, SimpleNamespace(ui=ui, config=None))


def _config() -> Config:
    return Config(
        archive_root=Path("/tmp/archive"),
        render_root=Path("/tmp/render"),
        db_path=Path("/tmp/archive/polylogue.sqlite"),
        sources=[],
    )


def _report() -> ReadinessReport:
    return ReadinessReport(checks=[ReadinessCheck("database", VerifyStatus.OK, summary="ok")])


def _options(**overrides: object) -> CheckCommandOptions:
    payload = {
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
        "schema_samples": "all",
        "schema_record_limit": None,
        "schema_record_offset": 0,
        "schema_quarantine_malformed": False,
        "maintenance_targets": (),
    }
    payload.update(overrides)
    return CheckCommandOptions(**cast(Any, payload))


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"vacuum": True}, "--vacuum requires --repair or --cleanup"),
        ({"preview": True}, "--preview requires --repair or --cleanup"),
        ({"maintenance_targets": ("session_products",)}, "--target requires --repair or --cleanup"),
        ({"schema_providers": ("claude-code",)}, "--schema-provider requires --schemas"),
        ({"schema_samples": "10"}, "--schema-samples requires --schemas"),
        ({"schema_record_limit": 5}, "--schema-record-limit requires --schemas"),
        ({"schema_record_offset": 2}, "--schema-record-offset requires --schemas"),
        ({"schema_quarantine_malformed": True}, "--schema-quarantine-malformed requires --schemas"),
        ({"artifact_providers": ("claude-code",)}, "--artifact-provider requires --proof, --artifacts, or --cohorts"),
        ({"artifact_statuses": ("supported",)}, "--artifact-status requires --artifacts or --cohorts"),
        ({"artifact_kinds": ("schema",)}, "--artifact-kind requires --artifacts or --cohorts"),
        ({"artifact_limit": 5}, "--artifact-limit requires --proof, --artifacts, or --cohorts"),
        ({"artifact_offset": 2}, "--artifact-offset requires --proof, --artifacts, or --cohorts"),
        ({"check_schemas": True, "schema_record_limit": 0}, "--schema-record-limit must be a positive integer"),
        ({"check_schemas": True, "schema_record_offset": -1}, "--schema-record-offset must be >= 0"),
        ({"check_proof": True, "artifact_limit": 0}, "--artifact-limit must be a positive integer"),
        ({"check_proof": True, "artifact_offset": -1}, "--artifact-offset must be >= 0"),
    ],
)
def test_validate_check_options_rejects_invalid_flag_combinations(
    overrides: dict[str, object],
    expected: str,
) -> None:
    with pytest.raises(SystemExit, match=expected):
        check_workflow.validate_check_options(_options(**overrides))


def test_validate_check_options_rejects_target_mode_mismatches() -> None:
    cleanup_spec = SimpleNamespace(mode=MaintenanceTargetMode.CLEANUP)
    repair_spec = SimpleNamespace(mode=MaintenanceTargetMode.REPAIR)
    catalog = SimpleNamespace(resolve=lambda names: [cleanup_spec] if names == ("cleanup_only",) else [repair_spec])

    with patch("polylogue.cli.check_validation.build_maintenance_target_catalog", return_value=catalog):
        with pytest.raises(SystemExit, match="only selected cleanup targets"):
            check_workflow.validate_check_options(_options(repair=True, maintenance_targets=("cleanup_only",)))

        with pytest.raises(SystemExit, match="only selected repair targets"):
            check_workflow.validate_check_options(_options(cleanup=True, maintenance_targets=("repair_only",)))


def test_formatting_helpers_cover_plan_counts_details_and_run_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYLOGUE_FORCE_PLAIN", raising=False)
    assert formatting.plain_forced_by_env() is False
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "yes")
    assert formatting.plain_forced_by_env() is True

    counts = {
        "conversations": 4,
        "new_conversations": 2,
        "changed_conversations": 1,
        "acquired": 4,
        "skipped": 1,
        "acquire_errors": 1,
        "validated": 3,
        "validation_invalid": 1,
        "validation_drift": 1,
        "validation_skipped_no_schema": 2,
        "validation_errors": 1,
        "parse_failures": 2,
        "materialized": 3,
        "rendered": 2,
        "render_failures": 1,
        "schemas_generated": 1,
        "schemas_failed": 1,
    }
    assert formatting.format_counts({"conversations": 4, "messages": 0}) == "4 conv (4 new)"
    assert formatting.format_run_details(counts) == [
        "Acquire: 4 acquired, 1 skipped, 1 errors",
        "Validate: 3 passed, 1 invalid, 1 drift, 2 no-schema, 1 errors",
        "Conversations: 2 new, 1 changed",
        "Parse: 2 failures",
        "Materialize: 3 conversations",
        "Render: 2 rendered, 1 failures",
        "Schemas: 1 generated, 1 failed",
    ]
    assert formatting.format_run_details({"conversations": 2}) == ["Conversations: 2 new"]

    assert formatting.format_plan_counts({"scan": 2, "render": 1}) == "2 scan, 1 render"
    assert formatting.format_plan_counts({}) == "no pipeline actions"
    assert formatting.format_plan_details({"new_raw": 2, "preview_invalid": 1}) == (
        "2 new raw, 1 would fail validation"
    )
    assert formatting.format_plan_details({}) is None
    assert formatting.format_sources_summary([]) == "none"


def test_run_blob_store_check_reports_missing_orphaned_and_verified_states() -> None:
    env = _env()
    config = _config()

    rows = [("raw-a",), ("raw-b",)]

    @contextmanager
    def connection() -> Iterator[object]:
        yield SimpleNamespace(execute=lambda sql: rows)

    blob_store = SimpleNamespace(iter_all=lambda: ["raw-a", "orphan-c"])

    with (
        patch("polylogue.cli.check_workflow.connection_context", return_value=connection()),
        patch("polylogue.storage.blob_store.get_blob_store", return_value=blob_store),
    ):
        check_workflow._run_blob_store_check(env, config)

    console_print = cast(MagicMock, env.ui.console.print)
    printed = [call.args[0] for call in console_print.call_args_list if call.args]
    assert "Blob store: 2 blobs on disk, 2 raw records in DB" in printed
    assert "  MISSING: 1 blobs referenced in DB but not on disk" in printed
    assert "  Orphaned: 1 blobs on disk not in DB" in printed

    env = _env()
    verified_store = SimpleNamespace(iter_all=lambda: ["raw-a", "raw-b"])
    with (
        patch("polylogue.cli.check_workflow.connection_context", return_value=connection()),
        patch("polylogue.storage.blob_store.get_blob_store", return_value=verified_store),
    ):
        check_workflow._run_blob_store_check(env, config)

    verified_console_print = cast(MagicMock, env.ui.console.print)
    assert verified_console_print.call_args_list[-1].args[0] == "  All blobs verified."


def test_schema_verification_and_maintenance_helpers_cover_runtime_paths() -> None:
    config = _config()
    options = _options(
        check_schemas=True,
        schema_providers=("claude-code",),
        schema_samples="25",
        schema_record_limit=10,
        schema_record_offset=2,
        schema_quarantine_malformed=True,
        repair=True,
    )
    report = _report()

    schema_report = cast(SchemaVerificationReport, SimpleNamespace())
    session_progress_callback = cast(Any, lambda: None)
    with (
        patch("polylogue.cli.check_workflow.run_schema_verification", return_value=schema_report) as run_verify,
        patch("polylogue.cli.check_workflow.parse_schema_samples", return_value=25) as parse_samples,
        patch("polylogue.cli.check_workflow.make_schema_progress_callback", return_value=session_progress_callback),
        patch("builtins.print") as builtins_print,
    ):
        assert check_workflow._run_schema_verification(options, config) is schema_report

    request = run_verify.call_args.args[0]
    assert request.providers == ["claude-code"]
    assert request.max_samples == 25
    assert request.record_limit == 10
    assert request.record_offset == 2
    assert request.quarantine_malformed is True
    assert request.progress_callback is session_progress_callback
    assert run_verify.call_args.kwargs["db_path"] == config.db_path
    parse_samples.assert_called_once_with("25")
    builtins_print.assert_called_once()

    with patch(
        "polylogue.cli.check_workflow.make_session_product_progress_callback",
        return_value=session_progress_callback,
    ):
        assert (
            check_workflow._session_product_progress_callback(options, ("session_products",))
            is session_progress_callback
        )
    assert check_workflow._session_product_progress_callback(_options(repair=True, preview=True), ()) is None
    assert check_workflow._session_product_progress_callback(_options(repair=True, json_output=True), ()) is None

    with (
        patch("polylogue.cli.check_workflow._resolve_selected_maintenance_targets", return_value=("session_products",)),
        patch("polylogue.cli.check_workflow._build_preview_counts", return_value={"session_products": 2}),
    ):
        preview_inputs = check_workflow._maintenance_run_inputs(_options(repair=True, preview=True), report)
        assert preview_inputs.selected_targets == ("session_products",)
        assert preview_inputs.preview_counts == {"session_products": 2}

    result = CheckCommandResult(report=report)
    inputs = check_workflow._MaintenanceRunInputs(selected_targets=("session_products",), preview_counts={"x": 1})
    repair_result = cast(RepairResult, SimpleNamespace())
    with patch("polylogue.cli.check_workflow.run_selected_maintenance", return_value=[repair_result]) as run_selected:
        check_workflow._run_maintenance(config, result, _options(repair=True), inputs)
    assert result.maintenance_targets == ("session_products",)
    assert result.maintenance_results == [repair_result]
    assert run_selected.call_args.kwargs["targets"] == ("session_products",)

    env = _env()
    result.vacuum_result = VacuumResult(ok=True, detail="done")
    with patch("polylogue.cli.check_workflow.persist_maintenance_run") as persist_run:
        check_workflow._persist_maintenance_run(
            env,
            report=report,
            result=result,
            options=_options(repair=True),
            inputs=inputs,
        )
    persist_run.assert_called_once()


def test_run_check_workflow_covers_runtime_blob_vacuum_and_persist_paths() -> None:
    env = _env()
    config = _config()
    object.__setattr__(env, "config", config)
    report = _report()
    runtime_report = ReadinessReport(checks=[ReadinessCheck("runtime", VerifyStatus.OK, summary="ok")])
    options = _options(
        repair=True,
        runtime=True,
        check_blob=True,
        vacuum=True,
        json_output=True,
    )

    repair_result = cast(RepairResult, SimpleNamespace())
    with (
        patch("polylogue.cli.check_workflow.load_effective_config", return_value=config),
        patch("polylogue.cli.check_workflow.get_readiness", return_value=report),
        patch("polylogue.cli.check_workflow.run_runtime_readiness", return_value=runtime_report),
        patch("polylogue.cli.check_workflow._run_blob_store_check") as run_blob_check,
        patch(
            "polylogue.cli.check_workflow._maintenance_run_inputs",
            return_value=check_workflow._MaintenanceRunInputs(
                selected_targets=("session_products",), preview_counts=None
            ),
        ),
        patch("polylogue.cli.check_workflow.run_selected_maintenance", return_value=[repair_result]),
        patch("polylogue.cli.check_workflow.make_session_product_progress_callback", return_value="progress"),
        patch("polylogue.cli.check_workflow.vacuum_database", return_value=VacuumResult(ok=True, detail="vacuumed")),
        patch("polylogue.cli.check_workflow.persist_maintenance_run") as persist_run,
    ):
        result = check_workflow.run_check_workflow(env, options)

    assert result.report is report
    assert result.runtime_report is runtime_report
    assert result.maintenance_results == [repair_result]
    assert result.vacuum_result == VacuumResult(ok=True, detail="vacuumed")
    run_blob_check.assert_called_once_with(env, config)
    persist_run.assert_called_once()
