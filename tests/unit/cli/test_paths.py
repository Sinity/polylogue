"""CLI tests for the paths command (#1627)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.paths import paths_command

_ARCHIVE_TIERS = ("source.db", "index.db", "embeddings.db", "ops.db", "user.db")


def _clear_archive_tiers(archive_root: Path) -> None:
    """Remove the archive tier databases the workspace fixture pre-creates.

    ``cli_workspace`` bootstraps a complete archive (all five tiers).
    Tests that assert a *partial* or *missing* storage layout clear those tier
    files first so they can stage the exact archive state under inspection.
    """
    for name in _ARCHIVE_TIERS:
        (archive_root / name).unlink(missing_ok=True)


def test_paths_command_text_output(cli_workspace: dict[str, Path], cli_runner: CliRunner) -> None:
    """The paths command produces human-readable text output."""
    result = cli_runner.invoke(paths_command, [], catch_exceptions=False)

    assert result.exit_code == 0
    output = result.output
    assert "Archive root" in output
    assert "Source DB" in output
    assert "Index DB" in output
    assert "Embeddings DB" in output
    assert "Ops DB" in output
    assert "User DB" in output
    assert "Config file" in output
    assert "Blob store" in output


def test_paths_command_json_output(cli_workspace: dict[str, Path], cli_runner: CliRunner) -> None:
    """The paths --json flag produces valid JSON with expected keys."""
    result = cli_runner.invoke(
        paths_command,
        ["--format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)

    assert "archive_root" in payload
    assert "active_archive_root" in payload
    assert "active_archive_root_matches_configured" in payload
    assert "database_path" in payload
    assert "active_database_path" in payload
    assert "active_index_database_path" in payload
    assert "active_database_role" in payload
    assert "storage_layout" in payload
    assert "archive_ready" in payload
    assert "final_shape_ready" in payload
    assert "archive_layout_ready" in payload
    assert "archive_layout_blockers" in payload
    assert "present_tiers" in payload
    assert "missing_tiers" in payload
    assert "source_database_path" in payload
    assert "index_database_path" in payload
    assert "embeddings_database_path" in payload
    assert "ops_database_path" in payload
    assert "user_database_path" in payload
    assert "config_file_path" in payload
    assert "blob_store_root" in payload
    assert "bind_mounts" in payload

    # All path values should be strings.
    assert isinstance(payload["archive_root"], str)
    assert isinstance(payload["active_archive_root"], str)
    assert isinstance(payload["database_path"], str)
    assert isinstance(payload["config_file_path"], str)
    assert isinstance(payload["blob_store_root"], str)


def test_paths_json_paths_are_absolute(cli_workspace: dict[str, Path], cli_runner: CliRunner) -> None:
    """The paths JSON output contains absolute paths."""
    result = cli_runner.invoke(
        paths_command,
        ["--format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)

    for key in (
        "archive_root",
        "active_archive_root",
        "database_path",
        "active_database_path",
        "active_index_database_path",
        "source_database_path",
        "index_database_path",
        "embeddings_database_path",
        "ops_database_path",
        "user_database_path",
        "config_file_path",
        "blob_store_root",
    ):
        path_str = payload[key]
        if path_str is not None:
            assert path_str.startswith("/"), f"{key} is not absolute: {path_str!r}"


def test_paths_json_reports_database_existence(cli_workspace: dict[str, Path], cli_runner: CliRunner) -> None:
    """The paths JSON output reports a missing archive."""
    _clear_archive_tiers(cli_workspace["archive_root"])
    (cli_workspace["archive_root"] / "polylogue.db").write_text("unrelated file", encoding="utf-8")

    result = cli_runner.invoke(
        paths_command,
        ["--format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)

    assert payload["database_exists"] is False
    assert payload["database_size_bytes"] is None
    assert payload["active_database_exists"] is False
    assert payload["active_database_path"].endswith("index.db")
    assert payload["active_archive_root"] == payload["archive_root"]
    assert payload["active_archive_root_matches_configured"] is True
    assert payload["active_database_role"] == "index"
    assert payload["storage_layout"] == "archive_missing"
    assert payload["archive_ready"] is False
    assert payload["final_shape_ready"] is False
    assert payload["archive_layout_ready"] is False
    assert payload["archive_layout_blockers"] == [
        "no_archive_tiers_present",
        "missing_archive_tiers",
        "missing_backup_required_tier:source",
        "missing_backup_required_tier:embeddings",
        "missing_backup_required_tier:user",
    ]
    assert payload["present_tiers"] == []
    assert payload["missing_tiers"] == ["source", "index", "embeddings", "ops", "user"]


def test_paths_json_reports_archive_final_shape(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    """The paths JSON output classifies a complete archive."""
    archive = cli_workspace["archive_root"]
    for name in _ARCHIVE_TIERS:
        (archive / name).write_text("fake database")
    (archive / "polylogue.db").write_text("unrelated file", encoding="utf-8")

    result = cli_runner.invoke(
        paths_command,
        ["--format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)

    assert payload["active_database_path"].endswith("index.db")
    assert payload["active_index_database_path"].endswith("index.db")
    assert payload["active_archive_root"] == payload["archive_root"]
    assert payload["active_archive_root_matches_configured"] is True
    assert payload["active_database_role"] == "index"
    assert payload["storage_layout"] == "archive_complete"
    assert payload["archive_ready"] is True
    assert payload["final_shape_ready"] is True
    assert payload["archive_layout_ready"] is True
    assert payload["archive_layout_blockers"] == []
    assert payload["present_tiers"] == ["source", "index", "embeddings", "ops", "user"]
    assert payload["missing_tiers"] == []


def test_paths_json_ignores_sibling_index_outside_configured_archive_root(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    """The storage layout follows the configured archive root.

    Tier databases sitting in a sibling directory outside the configured
    archive root must be ignored: the layout still classifies as missing.
    """
    _clear_archive_tiers(cli_workspace["archive_root"])
    sibling_root = cli_workspace["archive_root"].parent / "sibling-archive"
    sibling_root.mkdir(parents=True, exist_ok=True)
    for name in _ARCHIVE_TIERS:
        (sibling_root / name).write_text("fake database")

    result = cli_runner.invoke(
        paths_command,
        ["--format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)

    assert payload["archive_root"] == str(cli_workspace["archive_root"])
    assert payload["active_archive_root"] == str(cli_workspace["archive_root"])
    assert payload["active_archive_root_matches_configured"] is True
    assert payload["active_database_path"] == str(cli_workspace["archive_root"] / "index.db")
    assert payload["index_database_path"] == str(cli_workspace["archive_root"] / "index.db")
    assert payload["storage_layout"] == "archive_missing"
    assert payload["present_tiers"] == []


def test_paths_json_bind_mounts_is_list(cli_workspace: dict[str, Path], cli_runner: CliRunner) -> None:
    """The bind_mounts field is a list."""
    result = cli_runner.invoke(
        paths_command,
        ["--format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert isinstance(payload["bind_mounts"], list)


def test_paths_text_default_format(cli_workspace: dict[str, Path], cli_runner: CliRunner) -> None:
    """Default output format is human-readable text."""
    result = cli_runner.invoke(paths_command, [], catch_exceptions=False)

    assert result.exit_code == 0
    # The text output should NOT be valid JSON.
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.output)


def test_paths_text_reports_archive_layout_blockers(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    """The text output exposes why the archive layout is not ready."""
    _clear_archive_tiers(cli_workspace["archive_root"])
    (cli_workspace["archive_root"] / "polylogue.db").write_text("unrelated file", encoding="utf-8")

    result = cli_runner.invoke(paths_command, [], catch_exceptions=False)

    assert result.exit_code == 0
    assert "Archive layout" in result.output
    assert "not ready" in result.output
    assert "missing_archive_tiers" in result.output
    assert "polylogue.db" not in result.output
