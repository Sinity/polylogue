"""CLI wiring smoke tests for ``ops maintenance embeddings-rescue`` (polylogue-04kl).

Deep classification/rescue correctness is covered at the storage layer in
``tests/unit/storage/test_embedding_rescue.py``; these tests only prove the
CLI is registered, dispatches plan vs. apply correctly, and honors the
offline daemon guard -- using the real full-schema empty archive template so
argument/path resolution is exercised against production shapes.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _build_empty_retired_source(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        initialize_archive_tier(conn, ArchiveTier.EMBEDDINGS)
    except sqlite3.OperationalError as exc:
        if "vec0" in str(exc) or "sqlite-vec" in str(exc):
            pytest.skip("sqlite-vec extension is unavailable")
        raise
    finally:
        conn.close()


def test_embeddings_rescue_cli_plan_mode_is_read_only_on_empty_archive(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    source = tmp_path / "retired-embeddings.db"
    _build_empty_retired_source(source)

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "embeddings-rescue", "--source", str(source), "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "plan"
    assert payload["mutates"] is False
    assert payload["eligible_sessions"] == 0
    assert payload["fully_rescuable_sessions"] == 0
    assert payload["rescuable_messages"] == 0


def test_embeddings_rescue_cli_apply_mode_no_op_on_empty_archive(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    source = tmp_path / "retired-embeddings.db"
    _build_empty_retired_source(source)

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "embeddings-rescue",
            "--source",
            str(source),
            "--yes",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "execute"
    assert payload["mutates"] is True
    assert payload["rescued_sessions"] == 0
    assert payload["ok"] is True


def test_embeddings_rescue_cli_apply_refuses_while_daemon_runs(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    source = tmp_path / "retired-embeddings.db"
    _build_empty_retired_source(source)

    with patch("polylogue.maintenance.offline_guard.running_daemon_pid", return_value=123):
        result = cli_runner.invoke(
            cli,
            ["--plain", "ops", "maintenance", "embeddings-rescue", "--source", str(source), "--yes"],
        )

    assert result.exit_code == 1
    assert "refused while polylogued PID 123 is running" in result.output


def test_embeddings_rescue_cli_requires_existing_source(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does-not-exist.db"
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "embeddings-rescue", "--source", str(missing)],
    )
    assert result.exit_code != 0
