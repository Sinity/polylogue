"""Measured completion-claim receipts over the real deterministic demo archive."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.demo import inspect_completion_claims, inspect_demo_receipts, seed_demo_archive
from polylogue.scenarios import DEMO_CODEX_RECEIPTS_SESSION_ID


@pytest.mark.asyncio
async def test_completion_claim_experiment_has_a_stable_manifest_and_structural_receipts(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    await seed_demo_archive(archive_root, force=True, with_overlays=False)

    result = inspect_completion_claims(archive_root, sample_size=10)

    assert result.manifest.population_count >= 1
    assert result.manifest.selected_refs
    assert result.unsupported_count == sum(
        item.classification == "unsupported_by_structural_tool_evidence" for item in result.evidence
    )
    assert result.contradicted_then_repaired_count == 1
    (evidence,) = [item for item in result.evidence if item.session_ref == f"session:{DEMO_CODEX_RECEIPTS_SESSION_ID}"]
    assert evidence.classification == "contradicted_then_repaired"
    assert evidence.prior_action_ref is not None
    assert evidence.repair_action_ref is not None
    assert evidence.prior_is_error is True
    assert evidence.prior_exit_code == 1
    assert evidence.repair_is_error is False
    assert evidence.repair_exit_code == 0
    assert result.evidence_fingerprint.startswith("sha256:")


@pytest.mark.asyncio
async def test_completion_claim_experiment_goes_red_when_structural_failure_or_repair_is_withheld(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    await seed_demo_archive(archive_root, force=True, with_overlays=False)

    with sqlite3.connect(archive_root / "index.db") as conn:
        deleted = conn.execute(
            """
            DELETE FROM blocks
            WHERE session_id = ?
              AND block_type = 'tool_result'
              AND tool_result_exit_code = 0
            """,
            (DEMO_CODEX_RECEIPTS_SESSION_ID,),
        ).rowcount
        conn.commit()
    assert deleted >= 1

    result = inspect_completion_claims(archive_root, sample_size=10)
    (evidence,) = [item for item in result.evidence if item.session_ref == f"session:{DEMO_CODEX_RECEIPTS_SESSION_ID}"]
    assert evidence.classification == "contradicted_without_recorded_repair"
    assert evidence.repair_action_ref is None


@pytest.mark.asyncio
async def test_completion_claim_experiment_keeps_is_error_only_outcomes(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    await seed_demo_archive(archive_root, force=True, with_overlays=False)

    with sqlite3.connect(archive_root / "index.db") as conn:
        updated = conn.execute(
            """
            UPDATE blocks
            SET tool_result_exit_code = NULL
            WHERE session_id = ?
              AND block_type = 'tool_result'
              AND tool_result_is_error IS NOT NULL
            """,
            (DEMO_CODEX_RECEIPTS_SESSION_ID,),
        ).rowcount
        conn.commit()
    assert updated >= 2

    result = inspect_completion_claims(archive_root, sample_size=10)
    (evidence,) = [item for item in result.evidence if item.session_ref == f"session:{DEMO_CODEX_RECEIPTS_SESSION_ID}"]
    assert evidence.classification == "contradicted_then_repaired"
    assert evidence.prior_action_ref is not None
    assert evidence.repair_action_ref is not None


@pytest.mark.asyncio
async def test_completion_claim_experiment_excludes_runtime_protocol_material(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    await seed_demo_archive(archive_root, force=True, with_overlays=False)

    with sqlite3.connect(archive_root / "index.db") as conn:
        updated = conn.execute(
            """
            UPDATE messages
            SET material_origin = 'runtime_protocol'
            WHERE session_id = ?
              AND role = 'assistant'
            """,
            (DEMO_CODEX_RECEIPTS_SESSION_ID,),
        ).rowcount
        conn.commit()
    assert updated >= 1

    result = inspect_completion_claims(archive_root, sample_size=10)
    assert all(item.session_ref != f"session:{DEMO_CODEX_RECEIPTS_SESSION_ID}" for item in result.evidence)


@pytest.mark.asyncio
async def test_demo_receipts_reports_missing_fts_instead_of_publishing_a_headline(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    await seed_demo_archive(archive_root, force=True, with_overlays=False)

    with sqlite3.connect(archive_root / "index.db") as conn:
        conn.execute("DROP TABLE messages_fts")
        conn.commit()

    result = inspect_demo_receipts(archive_root)
    assert result.ok is False
    assert result.completion_claims is None
    assert any(
        problem.startswith("completion-claim evidence unreadable: Search index not built")
        for problem in result.problems
    )


def test_demo_receipts_returns_a_failed_result_for_an_unreadable_archive(tmp_path: Path) -> None:
    result = inspect_demo_receipts(tmp_path / "missing")

    assert result.ok is False
    assert result.completion_claims is None
    assert any(problem.startswith("archive evidence unreadable:") for problem in result.problems)
    assert any(problem.startswith("completion-claim evidence unreadable:") for problem in result.problems)


@pytest.mark.asyncio
async def test_completion_claims_only_cli_runs_without_fixture_receipts(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    await seed_demo_archive(archive_root, force=True, with_overlays=False)

    result = CliRunner().invoke(
        cli,
        [
            "demo",
            "receipts",
            "--root",
            str(archive_root),
            "--no-seed",
            "--completion-claims-only",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["headline"]["denominator"] >= 1
    assert payload["manifest"]["selected_refs"]
