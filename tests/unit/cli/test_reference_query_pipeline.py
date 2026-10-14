"""Real CLI coverage for durable reference query roots."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.config import Config
from polylogue.storage.sqlite.query_objects import QueryObject
from tests.infra.daemon_operations import running_daemon_operations
from tests.unit.mcp.test_reference_query_pipeline import _origin_query, _seed_archive


def test_find_from_query_uses_canonical_planner_and_emits_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``find from query:<hash>`` resolves through the resident session.reference route.

    The CLI only lowers the reference; the daemon resolves its lineage. With
    ``--no-daemon`` the operation is refused (every kernel read is
    resident-only), so this runs against a real operation stack.
    """
    archive_root = tmp_path / "archive"
    queries: list[QueryObject] = []

    def seed(root: Path) -> None:
        _seed_archive(root)
        with sqlite3.connect(root / "user.db") as conn:
            queries.append(_origin_query(conn, origin="codex-session"))
            conn.commit()

    monkeypatch.setattr(
        "polylogue.cli.archive_query.load_effective_config",
        lambda _env: Config(
            archive_root=archive_root,
            render_root=tmp_path / "render",
            sources=[],
        ),
    )
    with running_daemon_operations(archive_root, seed_archive=seed) as stack:
        (query,) = queries
        with patch("polylogue.daemon.socket_path.daemon_socket_path", lambda _root: stack.socket_path):
            result = CliRunner().invoke(
                cli,
                ["--plain", "find", f"from query:{query.query_hash}", "--format", "json"],
            )
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["source"] == f"query:{query.query_hash}"
    assert body["lineage"] == [f"query:{query.query_hash}"]
    assert body["member_count"] == 1
    assert body["members"][0].startswith("session:codex-session:")


def test_find_from_query_without_a_daemon_is_a_typed_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--no-daemon`` names the refused resident operation instead of reading locally."""
    archive_root = tmp_path / "archive"
    _seed_archive(archive_root)
    with sqlite3.connect(archive_root / "user.db") as conn:
        query = _origin_query(conn, origin="codex-session")
        conn.commit()
    monkeypatch.setattr(
        "polylogue.cli.archive_query.load_effective_config",
        lambda _env: Config(archive_root=archive_root, render_root=tmp_path / "render", sources=[]),
    )
    result = CliRunner().invoke(
        cli, ["--plain", "--no-daemon", "find", f"from query:{query.query_hash}", "--format", "json"]
    )
    assert result.exit_code != 0
    assert "session.reference" in result.output


@pytest.mark.parametrize("root_options", [("--tag", "absent"), ("--id", "missing"), ("--origin", "claude-ai-export")])
def test_reference_member_query_refuses_root_predicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root_options: tuple[str, ...]
) -> None:
    """A valid durable reference cannot silently discard a root selection predicate."""
    archive_root = tmp_path / "archive"
    queries: list[QueryObject] = []

    def seed(root: Path) -> None:
        _seed_archive(root)
        with sqlite3.connect(root / "user.db") as conn:
            queries.append(_origin_query(conn, origin="codex-session"))
            conn.commit()

    monkeypatch.setattr(
        "polylogue.cli.archive_query.load_effective_config",
        lambda _env: Config(archive_root=archive_root, render_root=tmp_path / "render", sources=[]),
    )
    with running_daemon_operations(archive_root, seed_archive=seed) as stack:
        (query,) = queries
        with patch("polylogue.daemon.socket_path.daemon_socket_path", lambda _root: stack.socket_path):
            runner = CliRunner()
            arguments = ["find", f"from query:{query.query_hash}", "--format", "json"]
            admitted = runner.invoke(cli, ["--plain", "--limit", "1", *arguments])
            assert admitted.exit_code == 0, admitted.output
            payload = json.loads(admitted.output)
            assert payload["source"] == f"query:{query.query_hash}"
            assert payload["member_count"] == 1
            assert payload["members"][0].startswith("session:codex-session:")

            refused = runner.invoke(cli, ["--plain", *root_options, *arguments])
            assert refused.exit_code == 2, refused.output
            assert "root predicates" in refused.output

            selected = runner.invoke(
                cli, ["--plain", "--origin", "codex-session", "--limit", "1", "find", "--format", "json"]
            )
            assert selected.exit_code == 0, selected.output
            selected_payload = json.loads(selected.output)
            assert selected_payload["total"] == 1
            assert len(selected_payload["items"]) == 1
            assert selected_payload["items"][0]["origin"] == "codex-session"
