"""CLI contracts for static site publication output via `run site` stage."""

from __future__ import annotations

import json

from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.store import RunRecord
from tests.infra.storage_records import ConversationBuilder, record_run


def _seed_latest_run(db_path) -> None:
    with open_connection(db_path) as conn:
        record_run(
            conn,
            RunRecord(
                run_id="run-cli-001",
                timestamp="2026-03-22T12:30:00+00:00",
                counts={"conversations": 1},
                indexed=False,
                duration_ms=321,
            ),
        )
        conn.commit()


def test_run_site_builds_manifest(cli_workspace) -> None:
    """`polylogue run site` builds the site and writes the manifest."""
    (
        ConversationBuilder(cli_workspace["db_path"], "site-cli-1")
        .provider("chatgpt")
        .title("CLI Site")
        .updated_at("2026-03-22T12:00:00+00:00")
        .add_message("m1", role="user", text="hello")
        .save()
    )
    _seed_latest_run(cli_workspace["db_path"])

    output_dir = cli_workspace["archive_root"] / "site-json"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--plain", "run", "site", "-o", str(output_dir)],
    )

    assert result.exit_code == 0, f"run site failed: {result.output}"
    assert (output_dir / "site-manifest.json").exists()
    manifest = json.loads((output_dir / "site-manifest.json").read_text())
    assert manifest["publication_kind"] == "site"
    assert manifest["archive"]["total_conversations"] == 1


def test_run_site_help(cli_workspace) -> None:
    """`polylogue run site --help` shows site-specific options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "site", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.output
    assert "--title" in result.output
    assert "--search-provider" in result.output
    assert "--dashboard" in result.output
