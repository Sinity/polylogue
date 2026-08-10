"""Regression coverage for the derived-tier rebuild safety scenario."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest


def _seeded_archive(tmp_path: Path) -> tuple[Path, list[str]]:
    from devtools import rebuild_safety_scenario as scenario
    from polylogue.sources.revision_backfill import backfill_historical_revision_evidence
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    raw_ids = scenario._seed_demo_corpus(archive_root)
    backfill_historical_revision_evidence(archive_root, ingest_workers=1)
    return archive_root, raw_ids


def test_seeded_corpus_populates_content_bearing_derived_relations(tmp_path: Path) -> None:
    """The differential must compare non-empty rows from real parser input."""
    from devtools import rebuild_safety_scenario as scenario

    archive_root, _raw_ids = _seeded_archive(tmp_path)
    scenario._full_rebuild(archive_root)
    scenario._write_attachment_witness(archive_root)

    with sqlite3.connect(archive_root / "index.db") as conn:
        counts = {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "actions",
                "attachments",
                "session_links",
                "session_model_usage",
                "session_provider_usage_events",
                "blocks_command_trigram_docsize",
            )
        }

    assert all(count > 0 for count in counts.values()), counts


def test_differential_detects_logical_fts_drift(tmp_path: Path) -> None:
    """A contentless FTS deletion must fail through its logical query surface."""
    from devtools import rebuild_safety_scenario as scenario

    archive_root, _raw_ids = _seeded_archive(tmp_path)
    scenario._full_rebuild(archive_root)
    scenario._write_attachment_witness(archive_root)
    expected = tmp_path / "expected-index.db"
    actual = tmp_path / "actual-index.db"
    shutil.copy2(archive_root / "index.db", expected)
    shutil.copy2(expected, actual)

    with sqlite3.connect(actual) as conn:
        conn.execute("DELETE FROM messages_fts")
        conn.execute(
            "UPDATE messages_fts_identity SET source_hash = X'00' "
            "WHERE rowid = (SELECT MIN(rowid) FROM messages_fts_identity)"
        )
        conn.execute(
            "UPDATE insight_materialization SET input_row_count = input_row_count + 1 "
            "WHERE rowid = (SELECT MIN(rowid) FROM insight_materialization)"
        )
        conn.commit()

    result = scenario._diff_index_databases(expected, actual, scenario_name="fts-drift")

    assert result.all_passed is False
    assert {diff.table for diff in result.diverging_tables} >= {
        "messages_fts logical query",
        "messages_fts_identity",
        "insight_materialization",
    }
    assert result.extra_checks["messages_fts_identity_b_is_consistent"] is False


def test_rebuild_safety_rejects_complete_user_tier_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reset that changes a durable assertion field must not look safe."""
    from devtools import rebuild_safety_scenario as scenario

    original = scenario._full_rebuild

    def mutate_user_tier(archive_root: Path) -> None:
        original(archive_root)
        with sqlite3.connect(archive_root / "user.db") as conn:
            conn.execute(
                "UPDATE assertions SET context_policy_json = '{\"inject\":true}' WHERE assertion_id = ?",
                ("rebuild-safety-canary",),
            )
            conn.commit()

    monkeypatch.setattr(scenario, "_full_rebuild", mutate_user_tier)

    assert scenario.run_rebuild_safety().all_passed is False


def test_incremental_path_rejects_pending_insights_convergence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The production convergence contract returning False remains a failure."""
    from devtools import rebuild_safety_scenario as scenario

    archive_root, raw_ids = _seeded_archive(tmp_path)
    monkeypatch.setattr(
        scenario,
        "make_insights_stage",
        lambda _index_db: SimpleNamespace(execute_sessions=lambda _session_ids: False),
    )

    with pytest.raises(RuntimeError, match="pending"):
        scenario._incremental_ingest_and_converge(archive_root, raw_ids)


def test_incremental_path_writes_every_session_from_a_multi_session_raw(tmp_path: Path) -> None:
    """Incremental parsing preserves the same multi-session raw expansion as replay."""
    from devtools import rebuild_safety_scenario as scenario

    archive_root, raw_ids = _seeded_archive(tmp_path)
    scenario._incremental_ingest_and_converge(archive_root, raw_ids)

    with sqlite3.connect(archive_root / "index.db") as conn:
        count = int(
            conn.execute("SELECT COUNT(*) FROM sessions WHERE native_id = 'claude-normalization-other'").fetchone()[0]
        )

    assert count == 1


def test_lab_run_writes_rebuild_report_without_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from devtools import __main__ as devtools_main
    from devtools import lab_scenario
    from devtools.rebuild_safety_scenario import RebuildComparisonResult

    result = RebuildComparisonResult(
        scenario_name="rebuild-safety",
        diffs=(),
        covered_tables=frozenset(),
        census_tables=frozenset(),
    )
    monkeypatch.setattr(lab_scenario, "run_rebuild_safety", lambda: result)
    monkeypatch.setattr(lab_scenario, "run_rebuild_differential", lambda: result)
    report_dir = tmp_path / "report"

    assert devtools_main.main(["lab", "run", "rebuild-safety", "--report-dir", str(report_dir)]) == 0
    assert (report_dir / "rebuild-safety.txt").read_text(encoding="utf-8")


def test_lab_run_serializes_each_rebuild_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from devtools import lab_scenario
    from devtools.rebuild_safety_scenario import RebuildComparisonResult

    differential = RebuildComparisonResult(
        scenario_name="rebuild-differential",
        diffs=(),
        covered_tables=frozenset(),
        census_tables=frozenset(),
    )
    monkeypatch.setattr(lab_scenario, "run_rebuild_safety", lambda: (_ for _ in ()).throw(RuntimeError("safety boom")))
    monkeypatch.setattr(lab_scenario, "run_rebuild_differential", lambda: differential)

    assert lab_scenario.main(["run", "rebuild-safety", "--json", "--report-dir", str(tmp_path)]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["stages"] == {"rebuild-safety": "error", "rebuild-differential": "ok"}
    assert "safety boom" in payload["safety_report"]
    assert (tmp_path / "rebuild-safety.txt").exists()
