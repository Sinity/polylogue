"""Periodic archive-wide FTS orphan audit (polylogue-5vbs).

Exercises ``find_orphaned_fts_sessions_sync``/``run_fts_orphan_audit_once_sync``
-- the sync bodies the periodic daemon loop (``periodic_fts_orphan_audit``)
schedules on a quiet cadence -- directly against a real archive fixture.
Proves the audit is a genuine feeder for the already-existing (and
already-tested) session-scoped FTS convergence-debt retry path: a
hand-orphaned block (search_text populated, no messages_fts row -- the
external-DB-surgery/partial-migration shape the write path's own
in-transaction repair cannot see) is found, recorded as retryable
``stage="fts", subject_type="session_id"`` convergence debt, and the fed
debt is then genuinely repaired by ``make_fts_stage``'s
``check_sessions``/``execute_sessions``, closing the loop end to end.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.daemon.fts_orphan_audit import (
    FtsOrphanAuditResult,
    find_orphaned_fts_sessions_sync,
    run_fts_orphan_audit_once_sync,
)


def _write_seed_session(db_path: Path) -> tuple[str, int]:
    """Write one real session through the production writer.

    Returns ``(session_id, block_rowid)``.
    """
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
    from polylogue.storage.sqlite.connection import open_connection

    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="fts-orphan-audit",
        title="Periodic FTS orphan audit test",
        messages=[
            ParsedMessage(
                provider_message_id="m0",
                role=Role.USER,
                text="a needle for the periodic fts orphan audit",
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TEXT,
                        text="a needle for the periodic fts orphan audit",
                    )
                ],
            )
        ],
    )
    with open_connection(db_path) as conn:
        session_id = write_parsed_session_to_archive(conn, session)
        conn.commit()
        row = conn.execute("SELECT rowid FROM blocks LIMIT 1").fetchone()
        assert row is not None
        return session_id, int(row[0])


def _orphan_the_fts_row(db_path: Path, block_rowid: int) -> None:
    """Delete a block's ``messages_fts``/``messages_fts_identity`` row directly.

    Simulates FTS drift introduced OUTSIDE the write path (external DB
    surgery, a partial migration) -- the block itself, and its populated
    ``search_text``, is left untouched. This is exactly the shape
    ``make_fts_stage``'s path-scoped ``check``/``check_many`` are documented
    as unable to see in archive/split-file mode.
    """
    from polylogue.storage.sqlite.connection import open_connection

    with open_connection(db_path) as conn:
        conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (block_rowid,))
        conn.execute("DELETE FROM messages_fts_identity WHERE rowid = ?", (block_rowid,))
        conn.commit()


class TestFindOrphanedFtsSessionsSync:
    def test_missing_db_is_a_bounded_noop(self, tmp_path: Path) -> None:
        assert find_orphaned_fts_sessions_sync(tmp_path / "does-not-exist.db") == []

    def test_clean_write_has_no_orphans(self, test_db: Path) -> None:
        _write_seed_session(test_db)

        assert find_orphaned_fts_sessions_sync(test_db) == []

    def test_detects_session_with_fts_orphaned_block(self, test_db: Path) -> None:
        session_id, block_rowid = _write_seed_session(test_db)
        _orphan_the_fts_row(test_db, block_rowid)

        found = find_orphaned_fts_sessions_sync(test_db)

        assert found == [session_id]


class TestRunFtsOrphanAuditOnceSync:
    def test_missing_db_is_a_bounded_noop(self, tmp_path: Path) -> None:
        result = run_fts_orphan_audit_once_sync(tmp_path / "does-not-exist.db")

        assert result == FtsOrphanAuditResult()

    def test_clean_write_records_no_debt(self, test_db: Path) -> None:
        _write_seed_session(test_db)

        result = run_fts_orphan_audit_once_sync(test_db)

        assert result == FtsOrphanAuditResult(ran=True, orphaned_sessions_found=0)

    def test_records_convergence_debt_for_orphaned_session(self, test_db: Path) -> None:
        from polylogue.sources.live.cursor import CursorStore

        session_id, block_rowid = _write_seed_session(test_db)
        _orphan_the_fts_row(test_db, block_rowid)

        result = run_fts_orphan_audit_once_sync(test_db)

        assert result.ran
        assert result.orphaned_sessions_found == 1

        debt = CursorStore(test_db).list_convergence_debt()
        matching = [
            d for d in debt if d.stage == "fts" and d.subject_type == "session_id" and d.subject_id == session_id
        ]
        assert matching, "audit must record retryable stage=fts/subject_type=session_id debt for the orphaned session"

    def test_recorded_debt_is_actually_repaired_by_the_existing_session_scoped_retry(self, test_db: Path) -> None:
        """Anti-vacuity: prove the fed debt is genuinely actionable by the
        already-existing (and already-tested) session-scoped FTS retry
        primitive, not just recorded and forgotten -- closing the exact
        feeder gap polylogue-5vbs names.
        """
        from polylogue.daemon.convergence_stages import make_fts_stage

        session_id, block_rowid = _write_seed_session(test_db)
        _orphan_the_fts_row(test_db, block_rowid)

        audit_result = run_fts_orphan_audit_once_sync(test_db)
        assert audit_result.orphaned_sessions_found == 1

        stage = make_fts_stage(test_db)
        assert stage.check_sessions is not None
        assert stage.check_sessions([session_id]) == {session_id}, (
            "the fed session must be flagged as needing repair by the stage's own check"
        )
        assert stage.execute_sessions is not None
        assert stage.execute_sessions([session_id]) is True

        assert find_orphaned_fts_sessions_sync(test_db) == [], "the orphan must be gone after the retry repairs it"
