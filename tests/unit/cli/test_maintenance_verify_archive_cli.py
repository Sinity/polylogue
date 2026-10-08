"""CLI tests for ``polylogue ops maintenance verify-archive``."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.maintenance.archive_verification import archive_verification_names_for_route
from polylogue.maintenance.source_manifest_continuity import (
    SourceDeclaration,
    SourceRole,
    build_source_frontier,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers import USER_TIER_VERSION


@pytest.fixture(autouse=True)
def _isolated_source_frontier(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """CLI tests never enumerate the operator's real provider directories."""
    from polylogue.maintenance import source_manifest_continuity

    empty_source = tmp_path / "inputs"
    empty_source.mkdir()
    declaration = SourceDeclaration("test-empty", SourceRole.DIRECTORY, empty_source, True)
    monkeypatch.setattr(
        source_manifest_continuity,
        "configured_source_frontier",
        lambda _archive_root: build_source_frontier((declaration,)),
    )


def test_verify_archive_cli_plain_exits_zero_on_empty_archive(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    result = cli_runner.invoke(cli, ["--plain", "ops", "maintenance", "verify-archive"])

    assert result.exit_code == 0, result.output
    assert "Archive verification:" in result.output
    assert "clear" in result.output


def test_verify_archive_cli_json_reports_every_declared_check(
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
    assert names == set(archive_verification_names_for_route("live-archive"))


def test_verify_archive_cli_exits_nonzero_on_schema_drift(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    root = cli_workspace["archive_root"]
    conn = sqlite3.connect(root / "user.db")
    try:
        # A tier newer than the runtime is drift the check must block on. The
        # fresh-v1 floor is version 1, so writing 1 would describe no drift.
        conn.execute(f"PRAGMA user_version = {USER_TIER_VERSION + 1}")
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
            "planner-stats",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    names = {check["name"] for check in payload["checks"]}
    assert names == {"tier-schema", "planner-stats"}


def test_selected_non_source_check_does_not_construct_source_frontier(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.maintenance import source_manifest_continuity

    def unexpected_frontier(_root: Path) -> object:
        pytest.fail("an unrelated selected check must not observe provider roots")

    monkeypatch.setattr(source_manifest_continuity, "configured_source_frontier", unexpected_frontier)

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "verify-archive", "--check", "tier-schema", "--output-format", "json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert {check["name"] for check in payload["checks"]} == {"tier-schema"}


def test_verify_archive_cli_uses_independent_source_frontier_for_missing_input(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from polylogue.maintenance import source_manifest_continuity

    source_root = tmp_path / "provider-inputs"
    source_root.mkdir()
    (source_root / "unretained.json").write_text('{"synthetic": true}', encoding="utf-8")
    frontier = build_source_frontier((SourceDeclaration("test-provider", SourceRole.DIRECTORY, source_root, True),))
    monkeypatch.setattr(source_manifest_continuity, "configured_source_frontier", lambda _root: frontier)

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "verify-archive",
            "--check",
            "source-conservation",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    check = next(row for row in payload["checks"] if row["name"] == "source-conservation")
    assert check["status"] == "error"
    assert check["evidence"]["terms"]["frontier_unacquired"]["count"] == 1


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


def test_verify_archive_cli_blocks_on_physical_blob_orphan(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    BlobStore(cli_workspace["archive_root"] / "blob").write_from_bytes(b"orphan physical bytes")

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "verify-archive",
            "--check",
            "blob-integrity",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    check = payload["checks"][0]
    assert check["status"] == "error"
    assert payload["blocking"] is True


def test_verify_archive_cli_strict_fails_on_required_skip(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    (cli_workspace["archive_root"] / "embeddings.db").unlink()

    ordinary = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "verify-archive",
            "--check",
            "embeddings-refs-liveness",
            "--output-format",
            "json",
        ],
    )
    assert ordinary.exit_code == 0, ordinary.output
    assert json.loads(ordinary.stdout)["checks"][0]["status"] == "skip"

    strict = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "verify-archive",
            "--check",
            "embeddings-refs-liveness",
            "--strict",
        ],
    )
    assert strict.exit_code == 1, strict.output
