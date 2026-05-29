"""CLI tests for the paths command (#1627)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.paths import paths_command


def test_paths_command_text_output(cli_workspace: dict[str, Path], cli_runner: CliRunner) -> None:
    """The paths command produces human-readable text output."""
    result = cli_runner.invoke(paths_command, [], catch_exceptions=False)

    assert result.exit_code == 0
    output = result.output
    assert "Archive root" in output
    assert "Database" in output
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
    assert "database_path" in payload
    assert "config_file_path" in payload
    assert "blob_store_root" in payload
    assert "bind_mounts" in payload
    assert "non_canonical_files" in payload

    # All path values should be strings.
    assert isinstance(payload["archive_root"], str)
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

    for key in ("archive_root", "database_path", "config_file_path", "blob_store_root"):
        path_str = payload[key]
        if path_str is not None:
            assert path_str.startswith("/"), f"{key} is not absolute: {path_str!r}"


def test_paths_json_reports_database_existence(cli_workspace: dict[str, Path], cli_runner: CliRunner) -> None:
    """The paths JSON output reports whether the database exists."""
    result = cli_runner.invoke(
        paths_command,
        ["--format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)

    # The cli_workspace fixture creates the database.
    assert payload["database_exists"] is True, "database should exist in cli_workspace"
    assert payload["database_size_bytes"] is not None
    assert isinstance(payload["database_size_bytes"], int)


def test_paths_json_non_canonical_is_list(cli_workspace: dict[str, Path], cli_runner: CliRunner) -> None:
    """The non_canonical_files field is a list."""
    result = cli_runner.invoke(
        paths_command,
        ["--format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert isinstance(payload["non_canonical_files"], list)


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


# ── archive_db_stub tests ────────────────────────────────────────


def test_archive_db_stub_zero_byte_replaced_by_symlink(tmp_path: Path) -> None:
    """A 0-byte archive.db is replaced by a symlink to polylogue.db."""
    from polylogue.paths.archive_db_stub import ensure_canonical_archive_db_name

    db_path = tmp_path / "polylogue.db"
    db_path.write_text("fake database")

    archive_db = tmp_path / "archive.db"
    archive_db.write_text("")  # 0-byte stub

    ensure_canonical_archive_db_name(db_path)

    assert not archive_db.exists() or archive_db.is_symlink()
    if archive_db.is_symlink():
        assert archive_db.readlink() == Path("polylogue.db")


def test_archive_db_stub_non_empty_left_alone(tmp_path: Path) -> None:
    """A non-empty archive.db is left untouched."""
    from polylogue.paths.archive_db_stub import ensure_canonical_archive_db_name

    db_path = tmp_path / "polylogue.db"
    db_path.write_text("fake database")

    archive_db = tmp_path / "archive.db"
    archive_db.write_text("real legacy data")

    ensure_canonical_archive_db_name(db_path)

    assert archive_db.read_text() == "real legacy data"
    assert not archive_db.is_symlink()


def test_archive_db_stub_noop_when_absent(tmp_path: Path) -> None:
    """No-op when archive.db does not exist."""
    from polylogue.paths.archive_db_stub import ensure_canonical_archive_db_name

    db_path = tmp_path / "polylogue.db"
    db_path.write_text("fake database")

    # No archive.db present.
    ensure_canonical_archive_db_name(db_path)

    assert not (tmp_path / "archive.db").exists()


def test_archive_db_stub_noop_when_already_symlink(tmp_path: Path) -> None:
    """No-op when archive.db is already a symlink."""
    from polylogue.paths.archive_db_stub import ensure_canonical_archive_db_name

    db_path = tmp_path / "polylogue.db"
    db_path.write_text("fake database")

    archive_db = tmp_path / "archive.db"
    archive_db.symlink_to("polylogue.db")

    ensure_canonical_archive_db_name(db_path)

    assert archive_db.is_symlink()
    assert archive_db.readlink() == Path("polylogue.db")
