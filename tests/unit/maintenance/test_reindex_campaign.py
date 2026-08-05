"""Executable reindex campaign corpus and convergence differential tests.

These tests intentionally call the production ingest, daemon convergence,
inactive rebuild, canary, and debt-retry routes.  No test-local parser,
rebuild, promotion, or comparison implementation is used.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from polylogue.daemon.convergence import DaemonConverger
from polylogue.daemon.convergence_stages import _HOT_INSIGHT_SOURCE_BYTES, make_fts_stage, make_insights_stage
from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
from polylogue.maintenance.reindex_canary import run_reindex_canary
from polylogue.sources.live.convergence_debt import convergence_debt_from_states
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.index_generation import IndexGenerationStore
from tests.infra.convergence_harness import (
    debt_ledger_row,
    make_messages_fts_stale,
    set_debt_retry_at,
)
from tests.infra.reindex_campaign import ReindexCampaignCorpus, build_reindex_campaign_corpus
from tests.infra.reindex_differential import (
    DerivedModelSnapshot,
    assert_derived_model_ready,
    assert_derived_models_equivalent,
    snapshot_derived_model,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_path_for_session(corpus: ReindexCampaignCorpus, session_id: str) -> Path:
    with sqlite3.connect(corpus.root / "index.db") as index_conn:
        raw_id = str(
            index_conn.execute("SELECT raw_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0]
        )
    with sqlite3.connect(corpus.root / "source.db") as source_conn:
        return Path(
            str(source_conn.execute("SELECT source_path FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0])
        )


def _retry_debt_in_fresh_process(index_db: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    script = (
        "from pathlib import Path\n"
        "from polylogue.daemon.cli import _drain_convergence_debt_once\n"
        f"result = _drain_convergence_debt_once(Path({str(index_db)!r}))\n"
        "assert result >= 1, result\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, f"fresh convergence retry failed:\n{completed.stdout}\n{completed.stderr}"


def _snapshot(corpus: ReindexCampaignCorpus) -> DerivedModelSnapshot:
    return snapshot_derived_model(
        corpus.root,
        corpus.root / "index.db",
        session_ids=corpus.manifest.session_ids,
        search_queries=corpus.manifest.fts_queries,
    )


def _clone_campaign_corpus(source: ReindexCampaignCorpus, target_root: Path) -> ReindexCampaignCorpus:
    """Clone one converged archive so differential rows retain source identity."""

    shutil.copytree(source.root, target_root)
    with sqlite3.connect(target_root / "source.db") as conn:
        conn.execute(
            """
            UPDATE raw_sessions
            SET source_path = replace(source_path, ?, ?)
            WHERE source_path LIKE ?
            """,
            (str(source.root), str(target_root), f"{source.root}%"),
        )
        conn.commit()
    return ReindexCampaignCorpus(root=target_root, manifest=source.manifest)


def test_reindex_campaign_manifest_has_positive_denominators(tmp_path: Path) -> None:
    """Every campaign edge class has a real production-ingested witness."""

    corpus = build_reindex_campaign_corpus(tmp_path / "campaign")
    corpus.manifest.assert_positive()
    assert corpus.manifest.lineage_session_ids
    assert corpus.manifest.attachment_session_ids
    assert corpus.manifest.parser_failure_raw_ids
    assert corpus.manifest.duplicate_raw_ids
    assert corpus.manifest.restart_session_ids
    assert set(corpus.manifest.parser_failure_raw_ids).isdisjoint(corpus.manifest.duplicate_raw_ids)


def test_real_inactive_rebuild_and_canary_preserve_active_and_reject_parser_as_duplicate(tmp_path: Path) -> None:
    """Full replay and the canary use inactive generations and never promote.

    Mutations killed by this test include promoting a no-promote candidate,
    omitting structured tool outcomes or attachments during replay, allowing
    a parser-failed raw into duplicate-byte supersession, and comparing only
    a fabricated summary instead of the candidate's real read model.
    """

    corpus = build_reindex_campaign_corpus(tmp_path / "campaign")
    root = corpus.root
    baseline = rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root, promote=True))
    assert baseline.status == "replayed"
    active_before = _digest(root / "index.db")

    receipt = rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root, promote=False))
    assert receipt.status == "replayed"
    assert receipt.generation["state"] == "inactive"
    assert receipt.generation["index_path"] != str((root / "index.db").resolve())
    replayed_count = receipt.replay["replayed_logical_source_count"]
    assert isinstance(replayed_count, int) and replayed_count >= 1

    store = IndexGenerationStore.for_archive_root(root)
    candidate = Path(str(receipt.generation["index_path"]))
    assert store.active_pointer.resolve(strict=True) == (root / "index.db").resolve()
    assert _digest(root / "index.db") == active_before
    assert candidate.is_file()

    with sqlite3.connect(root / "source.db") as source_conn:
        for raw_id in corpus.manifest.parser_failure_raw_ids:
            residual = source_conn.execute(
                "SELECT parsed_at_ms, parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)
            ).fetchone()
            assert residual is not None and residual[0] is None and residual[1]
            assert (
                source_conn.execute(
                    "SELECT 1 FROM raw_byte_duplicate_supersession_receipts WHERE raw_id = ?", (raw_id,)
                ).fetchone()
                is None
            )

    canary = run_reindex_canary(root, sessions_per_origin=100, no_promote=True)
    assert canary.comparison.unexpected_count > 0
    assert set(canary.comparison.counts_by_table) == {"raw_revision_applications", "raw_revision_heads"}
    canary_generation = canary.rebuild_receipt["generation"]
    assert isinstance(canary_generation, dict) and canary_generation["state"] == "inactive"
    assert _digest(root / "index.db") == active_before
    assert canary.selection.selected_session_ids


@pytest.mark.uses_real_clock("the restart differential deliberately controls source mtimes around the quiet window")
def test_restart_debt_converges_to_uninterrupted_campaign_state(tmp_path: Path) -> None:
    """A fresh-process debt retry reaches the same state as uninterrupted work."""

    template = build_reindex_campaign_corpus(tmp_path / "template")
    uninterrupted = _clone_campaign_corpus(template, tmp_path / "uninterrupted")
    restarted = _clone_campaign_corpus(template, tmp_path / "restarted")
    session_id = uninterrupted.manifest.restart_session_ids[0]
    restarted_session_id = restarted.manifest.restart_session_ids[0]

    uninterrupted_removed_fts = make_messages_fts_stale(uninterrupted.root / "index.db", session_id=session_id)
    assert uninterrupted_removed_fts > 0
    with sqlite3.connect(uninterrupted.root / "index.db") as conn:
        conn.execute("DELETE FROM session_profiles WHERE session_id = ?", (session_id,))
        conn.commit()
    uninterrupted_states, _ = DaemonConverger(
        (make_fts_stage(uninterrupted.root / "index.db"), make_insights_stage(uninterrupted.root / "index.db"))
    ).converge_sessions((session_id,))
    assert uninterrupted_states[session_id].converged

    source_path = _source_path_for_session(restarted, restarted_session_id)
    assert source_path.is_file()
    restarted_removed_fts = make_messages_fts_stale(restarted.root / "index.db", session_id=restarted_session_id)
    assert restarted_removed_fts > 0
    with sqlite3.connect(restarted.root / "index.db") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM messages_fts_identity AS f "
            "JOIN blocks AS b ON b.block_id = f.block_id WHERE b.session_id = ?",
            (restarted_session_id,),
        ).fetchone() == (0,)
    with sqlite3.connect(restarted.root / "index.db") as conn:
        conn.execute("DELETE FROM session_profiles WHERE session_id = ?", (restarted_session_id,))
        conn.commit()
    with source_path.open("ab") as restart_source:
        restart_source.truncate(_HOT_INSIGHT_SOURCE_BYTES)
    future = time.time() + 100_000
    os.utime(source_path, (future, future))
    stages = (make_fts_stage(restarted.root / "index.db"), make_insights_stage(restarted.root / "index.db"))
    deferred_states, _ = DaemonConverger(stages).converge_batch((source_path,))
    assert deferred_states[source_path].converged is False
    assert deferred_states[source_path].last_error == "insights deferred until source quiet"
    record = CursorStore(restarted.root / "index.db")
    record_convergence = convergence_debt_from_states((source_path,), deferred_states)
    assert record_convergence
    from polylogue.sources.live.convergence_outcome import record_convergence_outcome

    record_convergence_outcome(record, source_path, record_convergence, archive_root=restarted.root)
    record.record_convergence_debt(
        stage="fts",
        subject_type="session_id",
        subject_id=restarted_session_id,
        error="deliberate restart FTS backlog",
    )
    debt = debt_ledger_row(
        restarted.root / "ops.db",
        stage="insights",
        subject_type="session_id",
        subject_id=restarted_session_id,
    )
    assert debt is not None and debt.status == "deferred"
    set_debt_retry_at(
        restarted.root / "ops.db",
        stage="insights",
        subject_type="session_id",
        subject_id=restarted_session_id,
        retry_at="1970-01-01T00:00:00+00:00",
    )
    set_debt_retry_at(
        restarted.root / "ops.db",
        stage="fts",
        subject_type="session_id",
        subject_id=restarted_session_id,
        retry_at="9999-01-01T00:00:00+00:00",
    )
    old = time.time() - 100_000
    os.utime(source_path, (old, old))

    _retry_debt_in_fresh_process(restarted.root / "index.db")
    assert (
        debt_ledger_row(
            restarted.root / "ops.db",
            stage="insights",
            subject_type="session_id",
            subject_id=restarted_session_id,
        )
        is None
    )
    assert (
        debt_ledger_row(
            restarted.root / "ops.db",
            stage="fts",
            subject_type="session_id",
            subject_id=restarted_session_id,
        )
        is not None
    )

    set_debt_retry_at(
        restarted.root / "ops.db",
        stage="fts",
        subject_type="session_id",
        subject_id=restarted_session_id,
        retry_at="1970-01-01T00:00:00+00:00",
    )
    _retry_debt_in_fresh_process(restarted.root / "index.db")
    assert (
        debt_ledger_row(
            restarted.root / "ops.db",
            stage="fts",
            subject_type="session_id",
            subject_id=restarted_session_id,
        )
        is None
    )
    with sqlite3.connect(restarted.root / "index.db") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM messages_fts_identity AS f "
            "JOIN blocks AS b ON b.block_id = f.block_id WHERE b.session_id = ?",
            (restarted_session_id,),
        ).fetchone() == (restarted_removed_fts,)

    uninterrupted_snapshot = _snapshot(uninterrupted)
    restarted_snapshot = _snapshot(restarted)
    assert_derived_model_ready(uninterrupted_snapshot)
    assert_derived_model_ready(restarted_snapshot)
    assert_derived_models_equivalent(uninterrupted_snapshot, restarted_snapshot)
