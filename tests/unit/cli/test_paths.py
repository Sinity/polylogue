"""CLI tests for the paths command (#1627)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.cli.commands.maintenance import _backup_plan as maintenance_backup_plan
from polylogue.cli.commands.paths import paths_command
from polylogue.daemon.provenance import _build_raw_preview
from polylogue.paths import blob_store_root
from polylogue.storage.archive_layout import ARCHIVE_TIER_ORDER
from polylogue.storage.blob_store import get_blob_store

#: Derived from the canonical tier registry, never restated: a tier added to
#: ``ARCHIVE_TIER_SPECS`` must appear in this fixture (and in the assertions
#: below) without an edit here. A hand-written copy of this tuple previously
#: omitted ``audit``, which is what let the five-tier drift in the command go
#: unnoticed.
_ARCHIVE_TIERS = tuple(f"{name}.db" for name in ARCHIVE_TIER_ORDER)


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
    assert "archive_schema_ready" in payload
    assert "archive_layout_ready" in payload
    assert "archive_layout_blockers" in payload
    assert "archive_tier_versions" in payload
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


def test_archive_override_unifies_blob_write_read_maintenance_report_and_reset(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    archive = cli_workspace["archive_root"]
    expected_blob_root = archive / "blob"
    assert blob_store_root() == expected_blob_root

    payload = b"override-authoritative blob"
    blob_hash, _ = get_blob_store().write_from_bytes(payload)
    assert (expected_blob_root / blob_hash[:2] / blob_hash[2:]).read_bytes() == payload
    assert _build_raw_preview(blob_hash, len(payload), requested_bytes=None)["text"] == payload.decode()

    maintenance_plan = maintenance_backup_plan._backup_plan_payload(archive)
    assert maintenance_plan["blob_store"]["path"] == str(expected_blob_root)
    paths_result = cli_runner.invoke(paths_command, ["--format", "json"], catch_exceptions=False)
    assert json.loads(paths_result.output)["blob_store_root"] == str(expected_blob_root)

    # `ops reset` no longer deletes in-process: #4955 ("Delete CLI direct-writer
    # bypasses and prove daemon sole write authority") moved the deletion behind
    # `maintenance.reset`, so with no daemon the verb refuses. What this test
    # owns is the archive-root override, and the override still has to reach the
    # read-only preview the CLI renders before lowering the intent -- that
    # preview must name the overridden blob root, not a default one.
    reset_result = cli_runner.invoke(cli, ["--plain", "ops", "reset", "--blob", "--yes"])
    assert reset_result.exit_code == 1, reset_result.output
    assert "daemon is unavailable; it must execute maintenance.reset" in reset_result.output
    assert f"blob store: {expected_blob_root}" in reset_result.output
    # The refusal is a refusal: nothing was deleted behind it.
    assert expected_blob_root.exists()


def test_paths_json_reports_raw_materialization_debt_as_not_ready(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.surfaces.payloads import ArchiveDebtListPayload, ArchiveDebtRowPayload, ArchiveDebtTotalsPayload

    payload = ArchiveDebtListPayload(
        generated_at="2026-07-03T00:00:00+00:00",
        archive_root=str(cli_workspace["archive_root"]),
        rows=(
            ArchiveDebtRowPayload(
                debt_ref="debt:raw-materialization:chatgpt-export:parsed-without-session",
                kind="raw-materialization",
                category="parsed-without-session",
                stage="parse",
                subject_ref="raw-origin:chatgpt-export",
                severity="warning",
                status="actionable",
                owner="daemon",
                summary="4 chatgpt-export raw artifact(s) parsed but have no materialized session",
                affected_count=4,
                source_family="chatgpt-export",
            ),
        ),
        totals=ArchiveDebtTotalsPayload(total=1, warning=1, actionable=1, affected_total=4, affected_actionable=4),
    )
    monkeypatch.setattr("polylogue.operations.archive_debt.archive_debt_list", lambda **_kwargs: payload)

    result = cli_runner.invoke(
        paths_command,
        ["--format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    result_payload = json.loads(result.output)
    assert result_payload["archive_schema_ready"] is True
    assert result_payload["archive_ready"] is False
    assert result_payload["archive_materialization_ready"] is False
    readiness = result_payload["raw_materialization_readiness"]
    assert readiness["available"] is True
    assert readiness["actionable"] == 1
    assert readiness["affected_actionable"] == 4


def test_paths_readiness_resolves_active_generation_with_root_source_tier(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    import shutil

    archive_root = cli_workspace["archive_root"]
    generation_index = archive_root / ".index-generations" / "gen-test" / "index.db"
    generation_index.parent.mkdir(parents=True)
    shutil.copy2(archive_root / "index.db", generation_index)
    (archive_root / ".index-active-pointer").write_text(str(generation_index.resolve()), encoding="utf-8")

    result = cli_runner.invoke(paths_command, ["--format", "json"], catch_exceptions=False)

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["active_index_database_path"] == str(generation_index.resolve())
    readiness = payload["raw_materialization_readiness"]
    assert readiness["available"] is True
    assert "source.db or index.db missing" not in str(readiness.get("error"))


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
    (cli_workspace["archive_root"] / "stray.sqlite").write_text("unrelated file", encoding="utf-8")

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
    assert payload["archive_schema_ready"] is False
    assert payload["archive_layout_ready"] is False
    assert payload["archive_layout_blockers"] == [
        "no_archive_tiers_present",
        "missing_archive_tiers",
        "missing_backup_required_tier:source",
        "missing_backup_required_tier:embeddings",
        "missing_backup_required_tier:user",
        "missing_backup_required_tier:audit",
    ]
    assert payload["present_tiers"] == []
    assert payload["missing_tiers"] == list(ARCHIVE_TIER_ORDER)
    assert "audit" in payload["missing_tiers"]


def test_paths_json_reports_archive_final_shape(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    """The paths JSON output classifies a complete archive."""
    archive = cli_workspace["archive_root"]
    for name in _ARCHIVE_TIERS:
        (archive / name).write_text("fake database")
    (archive / "stray.sqlite").write_text("unrelated file", encoding="utf-8")

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
    assert payload["archive_ready"] is False
    assert payload["final_shape_ready"] is True
    assert payload["archive_schema_ready"] is False
    assert payload["archive_layout_ready"] is True
    assert payload["archive_layout_blockers"] == []
    assert payload["archive_tier_versions"]["index"]["version_status"] == "invalid"
    assert payload["present_tiers"] == list(ARCHIVE_TIER_ORDER)
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
    (cli_workspace["archive_root"] / "stray.sqlite").write_text("unrelated file", encoding="utf-8")

    result = cli_runner.invoke(paths_command, [], catch_exceptions=False)

    assert result.exit_code == 0
    assert "Archive layout" in result.output
    assert "not ready" in result.output
    assert "missing_archive_tiers" in result.output
    assert "stray.sqlite" not in result.output
