"""CLI tests for ``polylogue ops maintenance verify-archive``."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.maintenance.archive_verification import ARCHIVE_VERIFICATION_CHECK_NAMES


def test_verify_archive_cli_plain_exits_zero_on_empty_archive(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    result = cli_runner.invoke(cli, ["--plain", "ops", "maintenance", "verify-archive"])

    assert result.exit_code == 0, result.output
    assert "Archive verification:" in result.output
    assert "clear" in result.output


def test_verify_archive_cli_json_reports_every_registered_check(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "verify-archive", "--output-format", "json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["blocking"] is False
    names = {check["name"] for check in payload["checks"]}
    # Sourced from the registry itself (not a hand-maintained literal set)
    # so a new check added to ARCHIVE_VERIFICATION_CHECKS is caught by this
    # test automatically instead of the CLI JSON surface silently drifting
    # out of sync with the registry -- exactly the staleness this assertion
    # itself suffered from before polylogue-t0m73 (missing five checks
    # landed by an earlier PR that never updated this literal).
    assert names == set(ARCHIVE_VERIFICATION_CHECK_NAMES)


def test_verify_archive_cli_exits_nonzero_on_schema_drift(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    root = cli_workspace["archive_root"]
    conn = sqlite3.connect(root / "user.db")
    try:
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
    finally:
        conn.close()

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "verify-archive", "--output-format", "json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["blocking"] is True
    tier_check = next(check for check in payload["checks"] if check["name"] == "tier-schema")
    assert tier_check["status"] == "error"


def test_verify_archive_cli_restricts_to_selected_checks(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "verify-archive",
            "--check",
            "tier-schema",
            "--check",
            "counts-summary",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    names = {check["name"] for check in payload["checks"]}
    assert names == {"tier-schema", "counts-summary"}


def test_verify_archive_cli_rejects_unknown_check_name(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "verify-archive", "--check", "not-a-real-check"],
    )

    assert result.exit_code != 0
    assert "unknown archive verification check" in result.output


def test_verify_archive_cli_strict_fails_on_warning(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    # A freshly-templated empty archive has no sqlite_stat1 rows yet, which
    # is a warning-level (not error-level) planner-stats finding -- prove
    # --strict promotes that warning to a blocking exit code.
    default_result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "verify-archive", "--check", "planner-stats", "--output-format", "json"],
    )
    assert default_result.exit_code == 0, default_result.output
    payload = json.loads(default_result.stdout)
    assert payload["checks"][0]["status"] == "warning"

    strict_result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "verify-archive", "--check", "planner-stats", "--strict"],
    )
    assert strict_result.exit_code == 1
