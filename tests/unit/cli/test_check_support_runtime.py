from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from polylogue.cli.shared import check_workflow
from polylogue.cli.shared.check_workflow import CheckCommandOptions
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config
from polylogue.core.json import JSONDocument
from polylogue.readiness import ReadinessCheck, ReadinessReport, VerifyStatus
from polylogue.schemas.validation.models import SchemaVerificationReport


def _env() -> AppEnv:
    ui = MagicMock()
    ui.console = MagicMock()
    return cast(AppEnv, SimpleNamespace(ui=ui, config=None))


def _config() -> Config:
    return Config(
        archive_root=Path("/tmp/archive"),
        render_root=Path("/tmp/render"),
        db_path=Path("/tmp/archive/index.db"),
        sources=[],
    )


def _report() -> ReadinessReport:
    return ReadinessReport(checks=[ReadinessCheck("database", VerifyStatus.OK, summary="ok")])


def _options(**overrides: object) -> CheckCommandOptions:
    payload = {
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
        "schema_samples": "all",
        "schema_record_limit": None,
        "schema_record_offset": 0,
        "schema_quarantine_malformed": False,
    }
    payload.update(overrides)
    return CheckCommandOptions(**cast(Any, payload))


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"schema_providers": ("claude-code",)}, "--schema-origin requires --schemas"),
        ({"schema_samples": "10"}, "--schema-samples requires --schemas"),
        ({"schema_record_limit": 5}, "--schema-record-limit requires --schemas"),
        ({"schema_record_offset": 2}, "--schema-record-offset requires --schemas"),
        ({"schema_quarantine_malformed": True}, "--schema-quarantine-malformed requires --schemas"),
        (
            {"artifact_providers": ("claude-code",)},
            "--artifact-origin requires --artifact-coverage, --artifacts, or --cohorts",
        ),
        ({"artifact_statuses": ("supported",)}, "--artifact-status requires --artifacts or --cohorts"),
        ({"artifact_kinds": ("schema",)}, "--artifact-kind requires --artifacts or --cohorts"),
        ({"artifact_limit": 5}, "--artifact-limit requires --artifact-coverage, --artifacts, or --cohorts"),
        ({"artifact_offset": 2}, "--artifact-offset requires --artifact-coverage, --artifacts, or --cohorts"),
        ({"check_schemas": True, "schema_record_limit": 0}, "--schema-record-limit must be a positive integer"),
        ({"check_schemas": True, "schema_record_offset": -1}, "--schema-record-offset must be >= 0"),
        ({"check_artifact_coverage": True, "artifact_limit": 0}, "--artifact-limit must be a positive integer"),
        ({"check_artifact_coverage": True, "artifact_offset": -1}, "--artifact-offset must be >= 0"),
    ],
)
def test_validate_check_options_rejects_invalid_flag_combinations(
    overrides: dict[str, object],
    expected: str,
) -> None:
    with pytest.raises(SystemExit, match=expected):
        check_workflow.validate_check_options(_options(**overrides))


def test_run_blob_store_check_reports_missing_orphaned_and_verified_states() -> None:
    config = _config()
    with patch("polylogue.storage.blob_integrity.scan_blob_integrity") as scan:
        scan.return_value.to_dict.return_value = {
            "ok": False,
            "full_scan": True,
            "sample_size": 100,
            "scanned_blobs": 2,
            "scanned_references": 2,
            "total_blobs_seen": 2,
            "total_references_seen": 2,
            "findings": [{"kind": "orphan_blobs", "severity": "warning", "count": 1, "sample": ["orphan-c"]}],
        }
        payload = check_workflow._run_blob_store_check(config, full=True)

    scan.assert_called_once_with(config.db_path, full=True, configured_root=config.archive_root)
    assert payload["ok"] is False
    findings = cast(list[JSONDocument], payload["findings"])
    assert findings[0]["kind"] == "orphan_blobs"


def test_run_blob_store_check_returns_json_payload_without_emitting() -> None:
    config = _config()
    with patch("polylogue.storage.blob_integrity.scan_blob_integrity") as scan:
        scan.return_value.to_dict.return_value = {"ok": True, "findings": []}
        payload = check_workflow._run_blob_store_check(config)

    assert payload == {"ok": True, "findings": []}


def test_schema_verification_helpers_cover_runtime_paths() -> None:
    config = _config()
    options = _options(
        check_schemas=True,
        schema_providers=("claude-code",),
        schema_samples="25",
        schema_record_limit=10,
        schema_record_offset=2,
        schema_quarantine_malformed=True,
    )

    schema_report = cast(SchemaVerificationReport, SimpleNamespace())
    session_progress_callback = cast(Any, lambda: None)
    with (
        patch("polylogue.cli.shared.check_workflow.run_schema_verification", return_value=schema_report) as run_verify,
        patch("polylogue.cli.shared.check_workflow.parse_schema_samples", return_value=25) as parse_samples,
        patch(
            "polylogue.cli.shared.check_workflow.make_schema_progress_callback", return_value=session_progress_callback
        ),
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


def test_run_check_workflow_covers_runtime_and_blob_paths() -> None:
    env = _env()
    config = _config()
    object.__setattr__(env, "config", config)
    report = _report()
    runtime_report = ReadinessReport(checks=[ReadinessCheck("runtime", VerifyStatus.OK, summary="ok")])
    options = _options(
        runtime=True,
        check_blob=True,
        json_output=True,
    )

    with (
        patch("polylogue.cli.shared.check_workflow.load_effective_config", return_value=config),
        patch("polylogue.cli.shared.check_workflow.get_readiness", return_value=report) as get_readiness,
        patch("polylogue.cli.shared.check_workflow.run_runtime_readiness", return_value=runtime_report),
        patch("polylogue.cli.shared.check_workflow._run_blob_store_check") as run_blob_check,
    ):
        result = check_workflow.run_check_workflow(env, options)

    assert result.report is report
    assert result.runtime_report is runtime_report
    assert result.blob_report is run_blob_check.return_value
    run_blob_check.assert_called_once_with(config, full=False)
    get_readiness.assert_called_once_with(config, deep=False, probe_only=True)


def test_run_check_workflow_includes_daemon_status_when_requested() -> None:
    env = _env()
    config = _config()
    report = _report()
    daemon_report = {"ok": True, "daemon": "polylogued"}

    with (
        patch("polylogue.cli.shared.check_workflow.load_effective_config", return_value=config),
        patch("polylogue.cli.shared.check_workflow.get_readiness", return_value=report),
        patch("polylogue.cli.shared.check_workflow.daemon_status_payload", return_value=daemon_report) as daemon_status,
    ):
        result = check_workflow.run_check_workflow(env, _options(check_daemon=True))

    assert result.report is report
    assert result.daemon_report is daemon_report
    daemon_status.assert_called_once_with()
