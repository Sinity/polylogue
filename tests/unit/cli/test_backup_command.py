from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from polylogue.cli.commands.backup import backup_command
from polylogue.cli.operation_kernel import OperationFailedError
from polylogue.daemon.backup import BackupResult


def test_backup_command_passes_profile_to_archive_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_submit(_env: object, operation: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append({"operation": operation, **payload})
        return {
            "result": BackupResult(
                ok=True, output_path=str(tmp_path / "out"), backup_profile=str(payload["profile"])
            ).model_dump(mode="json")
        }

    monkeypatch.setattr("polylogue.cli.archive_query.submit_cli_mutation", fake_submit)

    result = CliRunner().invoke(
        backup_command,
        ["--output-dir", str(tmp_path), "--profile", "full_evidence"],
        obj=object(),
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert calls == [
        {
            "operation": "maintenance.backup",
            "output_dir": str(tmp_path),
            "check_only": False,
            "verify": False,
            "profile": "full_evidence",
        }
    ]
    assert "Profile: full_evidence" in result.output


def test_backup_check_runs_locally_without_daemon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_backup_archive(**kwargs: object) -> BackupResult:
        calls.append(kwargs)
        return BackupResult(ok=True, check_only=True, backup_profile=str(kwargs["profile"]))

    monkeypatch.setattr("polylogue.cli.commands.backup.backup_archive", fake_backup_archive)

    def unexpected_submit(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("read-only preflight contacted the daemon")

    monkeypatch.setattr("polylogue.cli.archive_query.submit_cli_mutation", unexpected_submit)
    env = SimpleNamespace(config=SimpleNamespace(archive_root=tmp_path / "archive"))
    result = CliRunner().invoke(
        backup_command,
        ["--output-dir", str(tmp_path / "out"), "--check", "--profile", "user_overlays"],
        obj=env,
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert calls == [
        {
            "output_dir": tmp_path / "out",
            "check_only": True,
            "verify": False,
            "profile": "user_overlays",
            "archive_root_path": tmp_path / "archive",
        }
    ]
    assert "Backup prerequisites: OK" in result.output


@pytest.mark.parametrize("output_format", ["plain", "json"])
def test_backup_failure_preserves_partial_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_format: str
) -> None:
    partial = BackupResult(
        ok=False,
        output_path=str(tmp_path / "partial"),
        error="verification failed",
        warnings=["source.db could not be verified"],
    )

    def failed_submit(*_args: object, **_kwargs: object) -> None:
        failure = OperationFailedError(
            "backup_failed", "verification failed", {"backup_result": partial.model_dump(mode="json")}
        )
        raise click.ClickException("daemon refused backup") from failure

    monkeypatch.setattr("polylogue.cli.archive_query.submit_cli_mutation", failed_submit)
    result = CliRunner().invoke(
        backup_command,
        ["--output-dir", str(tmp_path), "--format", output_format],
        obj=object(),
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    if output_format == "json":
        assert json.loads(result.output) == partial.model_dump(mode="json")
    else:
        assert f"Partial output: {partial.output_path}" in result.output
        assert "Warning: source.db could not be verified" in result.output
