from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.backup import backup_command
from polylogue.daemon.backup import BackupResult


def test_backup_command_passes_profile_to_archive_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_backup_archive(**kwargs: object) -> BackupResult:
        calls.append(kwargs)
        return BackupResult(ok=True, output_path=str(tmp_path / "out"), backup_profile=str(kwargs["profile"]))

    monkeypatch.setattr("polylogue.cli.commands.backup.backup_archive", fake_backup_archive)

    result = CliRunner().invoke(
        backup_command,
        ["--output-dir", str(tmp_path), "--profile", "full_evidence"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert calls == [
        {
            "output_dir": tmp_path,
            "check_only": False,
            "include_blobs": False,
            "verify": False,
            "profile": "full_evidence",
        }
    ]
    assert "Profile: full_evidence" in result.output
