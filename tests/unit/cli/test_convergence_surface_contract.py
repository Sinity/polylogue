from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from polylogue.config import Config
from polylogue.mcp.payloads import MCPReadinessReportPayload
from polylogue.readiness import get_readiness
from tests.infra.cli_subprocess import IsolatedWorkspace, run_cli, setup_isolated_workspace
from tests.infra.storage_records import SessionBuilder


def _seed_converging_workspace(tmp_path: Path) -> IsolatedWorkspace:
    workspace = setup_isolated_workspace(tmp_path)
    archive_root = workspace["paths"]["archive_root"]
    index_db = archive_root / "index.db"
    source_db = archive_root / "source.db"
    SessionBuilder(index_db, "converged").provider("codex").title("Converged Session").add_message(
        text="needle materialized"
    ).save()
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            "UPDATE sessions SET raw_id = ? WHERE native_id = ? AND origin = ?",
            ("raw-converged", "ext-converged", "codex-session"),
        )
    with sqlite3.connect(source_db) as conn:
        conn.executemany(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms, parsed_at_ms, validation_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "raw-converged",
                    "codex-session",
                    "ext-converged",
                    str(archive_root / "converged.json"),
                    bytes.fromhex("11" * 32),
                    10,
                    1,
                    1,
                    "passed",
                ),
                (
                    "raw-gap",
                    "chatgpt-export",
                    "gap-native",
                    str(archive_root / "gap.json"),
                    bytes.fromhex("22" * 32),
                    10,
                    1,
                    1,
                    "passed",
                ),
            ],
        )
    return workspace


def _load_stdout_json(result_stdout: str) -> dict[str, Any]:
    payload = json.loads(result_stdout)
    assert isinstance(payload, dict)
    return payload


def test_converging_archive_surfaces_share_materialization_counts(tmp_path: Path) -> None:
    workspace = _seed_converging_workspace(tmp_path)
    env = workspace["env"]
    archive_root = workspace["paths"]["archive_root"]
    expected_counts = {
        "raw_artifact_count": 2,
        "materialized_raw_artifact_count": 1,
        "archive_session_count": 1,
        "join_gap_count": 1,
    }
    expected_warning = (
        "Archive materialization needs classification: 1/2 raw artifact(s) materialized; "
        "1 raw/index join gap(s) found; results may be partial until daemon convergence classifies them."
    )

    status = run_cli(["--plain", "status", "--format", "json"], env=env, timeout=30)
    assert status.exit_code == 0, status.output
    status_payload = _load_stdout_json(status.stdout)
    status_counts = status_payload["component_readiness"]["raw_materialization"]["counts"]
    assert {key: status_counts[key] for key in expected_counts} == expected_counts

    no_results = run_cli(["--plain", "find", "absenttoken", "--format", "json"], env=env, timeout=30)
    assert no_results.exit_code == 2, no_results.output
    no_results_payload = _load_stdout_json(no_results.stdout)
    assert no_results_payload["archive_converging"] is True
    assert no_results_payload["convergence_warning"] == expected_warning
    assert no_results_payload["items"] == []

    analyze = run_cli(["--plain", "analyze", "--format", "json"], env=env, timeout=30)
    assert analyze.exit_code == 0, analyze.output
    analyze_payload = _load_stdout_json(analyze.stdout)
    assert analyze_payload["archive_converging"] is True
    assert analyze_payload["convergence_warning"] == expected_warning

    report = get_readiness(
        Config(
            archive_root=archive_root,
            render_root=workspace["paths"]["render_root"],
            sources=[],
            db_path=workspace["paths"]["db_path"],
        )
    )
    mcp_payload = MCPReadinessReportPayload.from_report(
        report,
        include_counts=True,
        include_detail=True,
        include_cached=True,
    ).model_dump()
    assert mcp_payload["archive_convergence"]["materialization_progress"] == expected_counts
    mcp_counts = mcp_payload["component_readiness"]["raw_materialization"]["counts"]
    assert {key: mcp_counts[key] for key in expected_counts} == expected_counts
