from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from click.testing import CliRunner

from polylogue.cli.click_app import cli


def test_blob_reference_liveness_cli_defaults_to_read_only(
    cli_workspace: dict[str, Path], cli_runner: CliRunner
) -> None:
    archive_root = cli_workspace["archive_root"]
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'historical-raw', 'raw_payload', '/historical', 1, 1)
            """,
            (b"h" * 32,),
        )

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "blob-reference-liveness",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] == "blob_reference_liveness"
    assert payload["mutates"] is False
    assert payload["dry_run"] is True
    assert payload["orphaned_by_ref_type"] == {"raw_payload": 1}
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM blob_refs").fetchone() == (1,)
