from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.storage.blob_gc import OrphanedBlobRefCensus


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


def test_blob_reference_liveness_cli_census_is_counts_only(
    cli_workspace: dict[str, Path], cli_runner: CliRunner
) -> None:
    archive_root = cli_workspace["archive_root"]
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'historical-raw', 'raw_payload', '/private/path', 1, 1)
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
            "--census-only",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == {
        "by_ref_type": {"raw_payload": 1},
        "deferred_by_ref_type": {},
        "mode": "blob_reference_census",
        "mutates": False,
        "ref_type_counts": {"raw_payload": 1},
        "scanned_count": 1,
        "schema_unavailable_count": 0,
        "total": 1,
        "unavailable_ref_types": {},
        "unknown_ref_type_count": 0,
    }
    assert "/private/path" not in result.stdout
    assert "historical-raw" not in result.stdout
    assert "686868" not in result.stdout


def test_blob_reference_liveness_cli_census_redacts_unknown_ref_type_names(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "polylogue.maintenance.blob_ref_liveness_reconciliation.census_blob_ref_liveness",
        lambda _archive_root: OrphanedBlobRefCensus(
            total=0,
            by_ref_type={},
            scanned_count=1,
            ref_type_counts={"future:/private/secret/hash": 1},
            unknown_ref_types={"future:/private/secret/hash": 1},
        ),
    )

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "blob-reference-liveness", "--census-only", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["unknown_ref_type_count"] == 1
    assert payload["ref_type_counts"] == {}
    assert "future:/private/secret/hash" not in result.stdout
    assert "secret-ref" not in result.stdout
    assert "/private/path" not in result.stdout
