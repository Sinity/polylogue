"""Measured completion-claim receipts over the real deterministic demo archive."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.demo import (
    inspect_completion_claims,
    inspect_demo_receipts,
    render_completion_claims,
    seed_demo_archive,
)
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
    headline = result.to_payload()["headline"]
    assert isinstance(headline, dict)
    assert (
        sum(
            int(headline[key])
            for key in (
                "unsupported_count",
                "neutral_prior_count",
                "contradicted_then_repaired_count",
                "contradicted_without_repair_count",
            )
        )
        == result.sample_size
    )
    assert headline["unsupported_percent"] == 100 * float(headline["unsupported_rate"])
    assert headline["contradicted_then_repaired_percent"] == 100 * float(headline["contradicted_then_repaired_rate"])
    assert f"unsupported by structural evidence: {result.unsupported_count} (" in render_completion_claims(result)


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
async def test_completion_claim_experiment_preserves_unknown_is_error_values(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    await seed_demo_archive(archive_root, force=True, with_overlays=False)

    with sqlite3.connect(archive_root / "index.db") as conn:
        updated = conn.execute(
            """
            UPDATE blocks
            SET tool_result_is_error = NULL
            WHERE session_id = ?
              AND block_type = 'tool_result'
              AND tool_result_exit_code IS NOT NULL
            """,
            (DEMO_CODEX_RECEIPTS_SESSION_ID,),
        ).rowcount
        conn.commit()
    assert updated >= 2

    result = inspect_completion_claims(archive_root, sample_size=10)
    (evidence,) = [item for item in result.evidence if item.session_ref == f"session:{DEMO_CODEX_RECEIPTS_SESSION_ID}"]
    assert evidence.classification == "contradicted_then_repaired"
    assert evidence.prior_is_error is None
    assert evidence.repair_is_error is None


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
async def test_completion_claim_experiment_uses_tool_result_time_not_tool_use_time(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    await seed_demo_archive(archive_root, force=True, with_overlays=False)

    with sqlite3.connect(archive_root / "index.db") as conn:
        failed = conn.execute(
            """
            SELECT tool_id, text, tool_result_is_error, tool_result_exit_code
            FROM blocks
            WHERE session_id = ?
              AND block_type = 'tool_result'
              AND tool_result_is_error = 1
            LIMIT 1
            """,
            (DEMO_CODEX_RECEIPTS_SESSION_ID,),
        ).fetchone()
        assert failed is not None
        late_position = conn.execute(
            "SELECT MAX(position) + 1 FROM messages WHERE session_id = ?",
            (DEMO_CODEX_RECEIPTS_SESSION_ID,),
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO messages(session_id, native_id, position, role, message_type, material_origin, content_hash)
            VALUES (?, 'late-tool-result', ?, 'tool', 'tool_result', 'tool_result', randomblob(32))
            """,
            (DEMO_CODEX_RECEIPTS_SESSION_ID, late_position),
        )
        late_message_id = conn.execute(
            "SELECT message_id FROM messages WHERE session_id = ? AND native_id = 'late-tool-result'",
            (DEMO_CODEX_RECEIPTS_SESSION_ID,),
        ).fetchone()[0]
        conn.execute(
            """
            DELETE FROM blocks
            WHERE session_id = ?
              AND block_type = 'tool_result'
              AND tool_id = ?
              AND tool_result_is_error = 1
            """,
            (DEMO_CODEX_RECEIPTS_SESSION_ID, failed[0]),
        )
        conn.execute(
            """
            INSERT INTO blocks(message_id, session_id, position, block_type, text, tool_id, tool_result_is_error, tool_result_exit_code)
            VALUES (?, ?, 0, 'tool_result', ?, ?, ?, ?)
            """,
            (late_message_id, DEMO_CODEX_RECEIPTS_SESSION_ID, failed[1], failed[0], failed[2], failed[3]),
        )
        conn.commit()

    result = inspect_completion_claims(archive_root, sample_size=10)
    (evidence,) = [item for item in result.evidence if item.session_ref == f"session:{DEMO_CODEX_RECEIPTS_SESSION_ID}"]
    assert evidence.classification == "unsupported_by_structural_tool_evidence"
    assert evidence.prior_action_ref is None


@pytest.mark.asyncio
async def test_completion_claim_experiment_orders_same_position_variants(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    await seed_demo_archive(archive_root, force=True, with_overlays=False)

    with sqlite3.connect(archive_root / "index.db") as conn:
        claim_position, claim_variant_index = conn.execute(
            """
            SELECT m.position, m.variant_index
            FROM blocks AS b
            JOIN messages AS m ON m.message_id = b.message_id
            WHERE b.session_id = ? AND b.block_type = 'text' AND b.text LIKE 'All tests pass.%'
            """,
            (DEMO_CODEX_RECEIPTS_SESSION_ID,),
        ).fetchone()
        failed_result_message_id = conn.execute(
            """
            SELECT message_id
            FROM blocks
            WHERE session_id = ? AND block_type = 'tool_result' AND tool_result_is_error = 1
            LIMIT 1
            """,
            (DEMO_CODEX_RECEIPTS_SESSION_ID,),
        ).fetchone()[0]
        conn.execute(
            "UPDATE messages SET position = ?, variant_index = ? WHERE message_id = ?",
            (claim_position, claim_variant_index + 1, failed_result_message_id),
        )
        conn.commit()

    result = inspect_completion_claims(archive_root, sample_size=10)
    (evidence,) = [item for item in result.evidence if item.session_ref == f"session:{DEMO_CODEX_RECEIPTS_SESSION_ID}"]
    assert evidence.classification == "unsupported_by_structural_tool_evidence"
    assert evidence.prior_action_ref is None


@pytest.mark.asyncio
async def test_completion_claim_experiment_requires_command_evidence_for_a_repair(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    await seed_demo_archive(archive_root, force=True, with_overlays=False)

    with sqlite3.connect(archive_root / "index.db") as conn:
        updated = conn.execute(
            """
            UPDATE blocks
            SET tool_input = NULL
            WHERE session_id = ? AND block_type = 'tool_use'
            """,
            (DEMO_CODEX_RECEIPTS_SESSION_ID,),
        ).rowcount
        conn.commit()
    assert updated >= 2

    result = inspect_completion_claims(archive_root, sample_size=10)
    (evidence,) = [item for item in result.evidence if item.session_ref == f"session:{DEMO_CODEX_RECEIPTS_SESSION_ID}"]
    assert evidence.classification == "contradicted_without_recorded_repair"
    assert evidence.repair_action_ref is None


@pytest.mark.asyncio
async def test_completion_claim_experiment_requires_tool_identity_for_a_repair(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    await seed_demo_archive(archive_root, force=True, with_overlays=False)

    with sqlite3.connect(archive_root / "index.db") as conn:
        updated = conn.execute(
            """
            UPDATE blocks
            SET tool_name = NULL
            WHERE session_id = ? AND block_type = 'tool_use'
            """,
            (DEMO_CODEX_RECEIPTS_SESSION_ID,),
        ).rowcount
        conn.commit()
    assert updated >= 2

    result = inspect_completion_claims(archive_root, sample_size=10)
    (evidence,) = [item for item in result.evidence if item.session_ref == f"session:{DEMO_CODEX_RECEIPTS_SESSION_ID}"]
    assert evidence.classification == "contradicted_without_recorded_repair"
    assert evidence.repair_action_ref is None


@pytest.mark.asyncio
async def test_completion_claim_manifest_and_evidence_fingerprint_drift_independently(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    await seed_demo_archive(archive_root, force=True, with_overlays=False)
    baseline = inspect_completion_claims(archive_root, sample_size=10)

    with sqlite3.connect(archive_root / "index.db") as conn:
        updated = conn.execute(
            """
            UPDATE blocks
            SET tool_result_exit_code = 2
            WHERE session_id = ?
              AND block_type = 'tool_result'
              AND tool_result_is_error = 1
            """,
            (DEMO_CODEX_RECEIPTS_SESSION_ID,),
        ).rowcount
        conn.commit()
    assert updated >= 1

    changed = inspect_completion_claims(archive_root, sample_size=10)
    assert changed.manifest.manifest_id == baseline.manifest.manifest_id
    assert changed.evidence_fingerprint != baseline.evidence_fingerprint


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
