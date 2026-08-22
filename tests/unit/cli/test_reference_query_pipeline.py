"""Real CLI coverage for durable reference query roots."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.config import Config
from polylogue.storage.sqlite.query_objects import QueryObject
from tests.unit.mcp.test_reference_query_pipeline import _origin_query, _seed_archive


def test_find_from_query_uses_canonical_planner_and_emits_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = tmp_path / "archive"
    _seed_archive(archive_root)
    with sqlite3.connect(archive_root / "user.db") as conn:
        query: QueryObject = _origin_query(conn, origin="codex-session")
        conn.commit()

    monkeypatch.setattr(
        "polylogue.cli.archive_query.load_effective_config",
        lambda _env: Config(
            archive_root=archive_root,
            render_root=tmp_path / "render",
            sources=[],
        ),
    )
    result = CliRunner().invoke(
        cli,
        [
            "--plain",
            "--no-daemon",
            "find",
            f"from query:{query.query_hash}",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["source"] == f"query:{query.query_hash}"
    assert body["lineage"] == [f"query:{query.query_hash}"]
    assert body["member_count"] == 1
    assert body["members"][0].startswith("session:codex-session:")
