"""Tests for the resume command surface."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from polylogue.api.archive import _rebuild_archive_session_insights
from polylogue.cli.click_app import cli
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_scenarios import native_session_id_for
from tests.infra.storage_records import SessionBuilder

NID_RESUME_ROOT = native_session_id_for("codex", "cli-resume-root")


def _seed_resume_session(db_path: Path) -> None:
    (
        SessionBuilder(db_path, "cli-resume-root")
        .provider("codex")
        .title("CLI Resume")
        .created_at("2026-04-21T09:00:00+00:00")
        .updated_at("2026-04-21T09:30:00+00:00")
        .add_message(
            "u1",
            role="user",
            text="Add resume JSON output.",
            timestamp="2026-04-21T09:00:00+00:00",
        )
        .add_message(
            "a1",
            role="assistant",
            text="Implemented JSON output and ready to verify.",
            timestamp="2026-04-21T09:25:00+00:00",
        )
        .save()
    )
    with ArchiveStore.open_existing(db_path.parent, read_only=False) as archive:
        _rebuild_archive_session_insights(archive)


def test_resume_json_by_root_format(cli_workspace: dict[str, Path]) -> None:
    _seed_resume_session(cli_workspace["db_path"])

    result = CliRunner().invoke(
        cli,
        ["--format", "json", "resume", NID_RESUME_ROOT],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    brief = payload["result"]
    assert brief["session_id"] == NID_RESUME_ROOT
    assert brief["facts"]["title"] == "CLI Resume"
    assert set(brief) >= {"facts", "inferences", "related_sessions", "uncertainties", "next_steps"}


def test_resume_plain_names_evidence_sections(cli_workspace: dict[str, Path]) -> None:
    _seed_resume_session(cli_workspace["db_path"])

    result = CliRunner().invoke(cli, ["resume", NID_RESUME_ROOT], catch_exceptions=False)

    assert result.exit_code == 0
    assert "Resume Brief" in result.output
    assert "Facts" in result.output
    assert "Inferred State" in result.output
    assert "Next Steps" in result.output


def test_resume_missing_session_exits_with_clear_message(cli_workspace: dict[str, Path]) -> None:
    result = CliRunner().invoke(cli, ["resume", "missing-session"], catch_exceptions=False)

    assert result.exit_code == 1
    assert "Session not found: missing-session" in str(result.exception)


def test_resume_candidates_json_repeats_recent_files(cli_workspace: dict[str, Path]) -> None:
    _seed_resume_session(cli_workspace["db_path"])

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "resume-candidates",
            "--repo",
            "/workspace/polylogue",
            "--cwd",
            "/workspace/polylogue",
            "--recent",
            "/workspace/polylogue/polylogue/cli/click_app.py",
            "--recent",
            "/workspace/polylogue/polylogue/cli/commands/resume.py",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    result_payload = payload["result"]
    assert result_payload["total"] >= 1
    assert result_payload["candidates"][0]["logical_session_id"] == NID_RESUME_ROOT
    assert "score_breakdown" in result_payload["candidates"][0]
