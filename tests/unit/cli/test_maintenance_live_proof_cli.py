"""Real Click dispatch tests for the fixed maintenance live-proof command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.maintenance import live_proof


def test_live_proof_cli_dispatches_registered_read_only_proof(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POLYLOGUE_CODE_SHA", "b" * 40)
    output = cli_workspace["archive_root"].parent / "live-proof.json"

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "live-proof",
            "--proof-id",
            "archive-verification",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["proof_id"] == "archive-verification"
    assert receipt["mode"] == "read_only"
    assert str(cli_workspace["archive_root"]) not in output.read_text(encoding="utf-8")


def test_live_proof_cli_rejects_unknown_route_without_creating_output(
    cli_workspace: dict[str, Path], cli_runner: CliRunner
) -> None:
    output = cli_workspace["archive_root"].parent / "unknown-live-proof.json"

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "live-proof",
            "--proof-id",
            "arbitrary-shell-command",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert "unknown live-proof id" in result.output
    assert not output.exists()


def test_live_proof_cli_translates_output_os_error(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POLYLOGUE_CODE_SHA", "c" * 40)
    output = cli_workspace["archive_root"].parent / "live-proof-output-error.json"

    def fail_write(_path: Path, _receipt: object) -> None:
        raise OSError("read-only filesystem")

    monkeypatch.setattr(live_proof, "write_live_proof_receipt", fail_write)
    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "live-proof",
            "--proof-id",
            "archive-verification",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert "live-proof receipt output could not be written" in result.output
    assert not output.exists()
