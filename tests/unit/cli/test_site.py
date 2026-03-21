"""CLI contracts for static site publication output."""

from __future__ import annotations

import json

from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.store import RunRecord
from tests.infra.storage_records import ConversationBuilder, record_run


def _unwrap_success(output: str) -> dict:
    payload = json.loads(output)
    assert payload["status"] == "ok"
    return payload["result"]


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


def test_site_command_json_emits_manifest(cli_workspace) -> None:
    """`polylogue site --json` emits the typed publication manifest envelope."""
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
        ["--plain", "site", "-o", str(output_dir), "--json"],
    )

    assert result.exit_code == 0
    payload = _unwrap_success(result.output)
    assert payload["publication_kind"] == "site"
    assert payload["archive"]["total_conversations"] == 1
    assert payload["latest_run"]["run_id"] == "run-cli-001"
    assert payload["outputs"]["total_index_pages"] >= 1
    assert payload["artifacts"]["entry_count"] >= 1
    assert "site-manifest.json" not in {
        entry["relative_path"] for entry in payload["artifacts"]["entries"]
    }
    assert (output_dir / "site-manifest.json").exists()
