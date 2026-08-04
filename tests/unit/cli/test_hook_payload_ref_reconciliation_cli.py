"""Real CLI coverage for guarded hook-payload reference reconciliation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.storage.sqlite.archive_tiers.source_write import deterministic_blob_hash, deterministic_raw_session_id


def _seed_exact_hook_ref(archive_root: Path) -> None:
    payload = b'{"event":"PostToolUse"}'
    blob_hash = deterministic_blob_hash(payload)
    source_path = "/hooks/cli.jsonl"
    native_id = "cli-native"
    ref_id = deterministic_raw_session_id("codex-session", source_path, 0, blob_hash, native_id)
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_hook_events (
                hook_event_id, origin, native_id, session_native_id, source_path, event_type,
                payload_json, observed_at_ms
            ) VALUES ('cli-hook', 'codex-session', ?, 'session-1', ?, 'PostToolUse', '{}', 1)
            """,
            (native_id, source_path),
        )
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, ?, 'raw_payload', ?, ?, 1)
            """,
            (blob_hash, ref_id, source_path, len(payload)),
        )
        conn.commit()


def test_hook_payload_reconcile_cli_defaults_to_read_only_json(
    cli_workspace: dict[str, Path], cli_runner: CliRunner
) -> None:
    archive_root = cli_workspace["archive_root"]
    _seed_exact_hook_ref(archive_root)

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "hook-payload-ref-reconcile", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] == "hook_payload_ref_reconciliation"
    assert payload["mutates"] is False
    assert payload["applied"] is False
    assert payload["matched_count"] == 1
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT ref_type FROM blob_refs").fetchone() == ("raw_payload",)


def test_hook_payload_reconcile_cli_apply_requires_receipt_file(
    cli_workspace: dict[str, Path], cli_runner: CliRunner
) -> None:
    _seed_exact_hook_ref(cli_workspace["archive_root"])

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "hook-payload-ref-reconcile",
            "--apply",
            "--backup-manifest",
            str(cli_workspace["archive_root"] / "not-used-manifest.json"),
        ],
    )

    assert result.exit_code != 0
    assert result.exception is not None
    assert "receipt output" in str(result.exception)
