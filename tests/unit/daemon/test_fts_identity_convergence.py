"""Periodic exact FTS identity-ledger drift recompute (polylogue-miwv).

Exercises ``run_fts_identity_drift_recompute_once_sync`` -- the sync body the
periodic daemon loop (``periodic_fts_identity_drift_recompute``) schedules on
a quiet cadence -- directly against a real archive fixture (never a
mock/stub of the reconciliation SQL). Proves it is the recompute AUTHORITY
for ``identity_mismatch_rows``: a hand-corrupted ledger row makes the next
recompute pass record a nonzero count, and repairing the row makes the pass
after that record 0 again. A second test reproduces the exact hazard the
bead names -- ``daemon/fts_startup.py``'s bounded STALE-write path can reset
a recorded nonzero ``identity_mismatch_rows`` to 0 without recomputing --
and proves the periodic recompute corrects that stale zero back to the true
count on its next tick.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.daemon.fts_identity_convergence import (
    FtsIdentityDriftRecomputeResult,
    run_fts_identity_drift_recompute_once_sync,
)
from polylogue.storage.fts.freshness import MESSAGE_SURFACE, READY, record_fts_surface_state_sync


def _write_seed_session(db_path: Path) -> str:
    """Write one real session through the production writer and return its block's rowid-bearing block_id."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
    from polylogue.storage.sqlite.connection import open_connection

    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="fts-identity-drift-recompute",
        title="Periodic drift recompute test",
        messages=[
            ParsedMessage(
                provider_message_id="m0",
                role=Role.USER,
                text="a needle for the periodic identity drift recompute stage",
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TEXT,
                        text="a needle for the periodic identity drift recompute stage",
                    )
                ],
            )
        ],
    )
    with open_connection(db_path) as conn:
        write_parsed_session_to_archive(conn, session)
        conn.commit()
        row = conn.execute("SELECT block_id FROM blocks LIMIT 1").fetchone()
        assert row is not None
        return str(row[0])


def _corrupt_identity_row(db_path: Path, real_block_id: str) -> None:
    """Hand-corrupt the ledger row for ``real_block_id`` -- the rowid-reuse-gone-wrong shape."""
    from polylogue.storage.sqlite.connection import open_connection

    with open_connection(db_path) as conn:
        conn.execute(
            "UPDATE messages_fts_identity SET block_id = 'stale:ghost:0' WHERE block_id = ?",
            (real_block_id,),
        )
        conn.commit()


def _repair_identity_row(db_path: Path, real_block_id: str) -> None:
    """Restore the corrupted ledger row back to its correct binding."""
    from polylogue.storage.sqlite.connection import open_connection

    with open_connection(db_path) as conn:
        conn.execute(
            "UPDATE messages_fts_identity SET block_id = ? WHERE block_id = 'stale:ghost:0'",
            (real_block_id,),
        )
        conn.commit()


def _freshness_identity_mismatch_rows(db_path: Path) -> int | None:
    from polylogue.storage.sqlite.connection import open_connection

    with open_connection(db_path) as conn:
        row = conn.execute(
            "SELECT identity_mismatch_rows FROM fts_freshness_state WHERE surface = ?",
            (MESSAGE_SURFACE,),
        ).fetchone()
        return None if row is None else int(row[0])


class TestFtsIdentityDriftRecomputeStage:
    def test_missing_db_is_a_bounded_noop(self, tmp_path: Path) -> None:
        result = run_fts_identity_drift_recompute_once_sync(tmp_path / "does-not-exist.db")
        assert result == FtsIdentityDriftRecomputeResult()

    def test_corrupted_ledger_row_recomputes_nonzero_then_repair_recomputes_zero(self, test_db: Path) -> None:
        real_block_id = _write_seed_session(test_db)

        # Sanity: a fresh write through the real writer is ledger-clean.
        clean = run_fts_identity_drift_recompute_once_sync(test_db)
        assert clean.ran
        assert clean.identity_mismatch_rows == 0
        assert clean.ready
        assert _freshness_identity_mismatch_rows(test_db) == 0

        _corrupt_identity_row(test_db, real_block_id)

        drifted = run_fts_identity_drift_recompute_once_sync(test_db)
        assert drifted.ran
        assert drifted.identity_mismatch_rows == 1, (
            "periodic recompute must observe the hand-corrupted ledger row as a real mismatch"
        )
        assert not drifted.ready
        assert _freshness_identity_mismatch_rows(test_db) == 1, (
            "the recorded fts_freshness_state row must reflect the recomputed (nonzero) count, "
            "not a stale value from an earlier pass"
        )

        _repair_identity_row(test_db, real_block_id)

        repaired = run_fts_identity_drift_recompute_once_sync(test_db)
        assert repaired.ran
        assert repaired.identity_mismatch_rows == 0
        assert repaired.ready
        assert _freshness_identity_mismatch_rows(test_db) == 0

    def test_recompute_corrects_a_stale_zero_left_by_bounded_startup_write(self, test_db: Path) -> None:
        """Reproduces the documented hazard: fts_startup.py's bounded
        STALE-write path (``_ensure_archive_messages_fts_startup_readiness_sync``)
        calls ``record_fts_surface_state_sync`` without
        ``identity_mismatch_rows`` (defaults to 0), which the ``ON CONFLICT``
        upsert then writes over any previously recorded nonzero value. This
        proves the periodic recompute is the authority that corrects that
        stale zero on its next tick, rather than trusting the last writer.
        """
        real_block_id = _write_seed_session(test_db)
        _corrupt_identity_row(test_db, real_block_id)

        first_pass = run_fts_identity_drift_recompute_once_sync(test_db)
        assert first_pass.identity_mismatch_rows == 1
        assert _freshness_identity_mismatch_rows(test_db) == 1

        # Simulate the bounded STALE-write hazard: some other bounded caller
        # (e.g. fts_startup.py) re-records the surface state without ever
        # computing identity_mismatch_rows, silently resetting it to 0 even
        # though the underlying corruption is still there.
        from polylogue.storage.sqlite.connection import open_connection

        with open_connection(test_db) as conn:
            record_fts_surface_state_sync(
                conn,
                surface=MESSAGE_SURFACE,
                state=READY,
                source_rows=1,
                indexed_rows=1,
                missing_rows=0,
                excess_rows=0,
                duplicate_rows=0,
                # identity_mismatch_rows omitted -- defaults to 0, the hazard.
            )
            conn.commit()
        assert _freshness_identity_mismatch_rows(test_db) == 0, "test setup must reproduce the stale-zero hazard"

        second_pass = run_fts_identity_drift_recompute_once_sync(test_db)
        assert second_pass.identity_mismatch_rows == 1, (
            "the periodic recompute must recompute from the real ledger state, not trust the "
            "stale zero a bounded caller wrote in between passes"
        )
        assert _freshness_identity_mismatch_rows(test_db) == 1


class TestFtsIdentityDriftRecomputeAntiVacuity:
    """Prove the stage actually calls the exact reconciliation, not a stub."""

    def test_drift_samples_are_appended_to_ops_db(self, test_db: Path) -> None:
        """A real recompute pass also writes an ops.db drift-magnitude sample.

        Sanity-checks the stage composes all three functions its docstring
        claims (fts_invariant_snapshot_sync + record_fts_invariant_snapshot_sync
        + sample_fts_drift_to_ops_sync), not just the freshness-row write.
        """
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
        from polylogue.storage.sqlite.archive_tiers.ops_write import list_fts_drift_samples
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        _write_seed_session(test_db)
        ops_db = test_db.with_name("ops.db")
        ops_conn = sqlite3.connect(str(ops_db))
        try:
            initialize_archive_tier(ops_conn, ArchiveTier.OPS)
        finally:
            ops_conn.close()

        result = run_fts_identity_drift_recompute_once_sync(test_db)
        assert result.ran
        assert result.drift_samples_written > 0

        ops_conn = sqlite3.connect(str(ops_db))
        try:
            samples = list_fts_drift_samples(ops_conn, surface=MESSAGE_SURFACE)
        finally:
            ops_conn.close()
        assert samples, "recompute pass must append at least one ops.db drift sample"
