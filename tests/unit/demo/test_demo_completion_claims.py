"""Measured completion-claim receipts over the real deterministic demo archive."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.demo import inspect_completion_claims, seed_demo_archive
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
