"""Insight materialization laws: session insights agree with source sessions.

Proves that materialized session insights (profiles, work events, phases)
reflect the sessions they were derived from — counts match, provider
agrees, no phantom insights for non-existent sessions.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path

import pytest

from tests.infra.storage_records import SessionBuilder, db_setup


def _open_archive(db_path: Path) -> sqlite3.Connection:
    """Open a raw read connection to the index.db.

    The old single-file ``open_connection`` enforces the v22 schema guard and would
    reject the archive file; these laws assert against the archive tables
    directly.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


@pytest.fixture()
def materialized_db(workspace_env: Mapping[str, Path]) -> Path:
    """Create a DB with sessions and run session insight materialization."""
    db_path = db_setup(workspace_env)

    SessionBuilder(db_path, "mat-gpt-1").provider("chatgpt").title("GPT session").add_message(
        role="user", text="Write a function"
    ).add_message(role="assistant", text="def hello(): pass").add_message(
        role="user", text="Add error handling"
    ).add_message(role="assistant", text="def hello(): try: pass except: pass").save()

    SessionBuilder(db_path, "mat-claude-1").provider("claude-code").title("Claude session").add_message(
        role="user", text="Refactor storage"
    ).add_message(role="assistant", text="I will restructure the module").save()

    SessionBuilder(db_path, "mat-codex-1").provider("codex").title("Codex session").add_message(
        role="user", text="Generate tests"
    ).add_message(role="assistant", text="Here are the tests").add_message(role="user", text="Add edge cases").save()

    from polylogue.api import Polylogue

    async def _rebuild() -> None:
        archive = Polylogue(archive_root=db_path.parent, db_path=db_path)
        try:
            await archive.rebuild_insights()
        finally:
            await archive.close()

    asyncio.run(_rebuild())

    return db_path


class TestProfileSessionAgreement:
    """Every session profile must correspond to exactly one real session."""

    def test_profile_count_matches_session_count(self, materialized_db: Path) -> None:
        with closing(_open_archive(materialized_db)) as conn:
            conv_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            has_profiles = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_profiles'"
            ).fetchone()
            if has_profiles is None:
                pytest.skip("session_profiles table not present")
            profile_count = conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0]
            assert profile_count == conv_count, f"Profile count ({profile_count}) != session count ({conv_count})"

    def test_no_phantom_profiles(self, materialized_db: Path) -> None:
        """No profile should reference a non-existent session."""
        with closing(_open_archive(materialized_db)) as conn:
            has_profiles = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_profiles'"
            ).fetchone()
            if has_profiles is None:
                pytest.skip("session_profiles table not present")

            phantom_count = conn.execute(
                "SELECT COUNT(*) FROM session_profiles sp "
                "WHERE NOT EXISTS (SELECT 1 FROM sessions c WHERE c.session_id = sp.session_id)"
            ).fetchone()[0]
            assert phantom_count == 0, f"Found {phantom_count} phantom profiles"

    def test_profile_every_profile_has_a_session(self, materialized_db: Path) -> None:
        """Every profile's session must exist with a resolvable origin.

        In the store the profile does not carry a
        ``source_name`` column; provider identity lives on ``sessions.origin``.
        The agreement law is therefore: no profile may reference a session
        whose origin is missing.
        """
        with closing(_open_archive(materialized_db)) as conn:
            has_profiles = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_profiles'"
            ).fetchone()
            if has_profiles is None:
                pytest.skip("session_profiles table not present")

            orphans = conn.execute(
                "SELECT sp.session_id "
                "FROM session_profiles sp "
                "LEFT JOIN sessions s ON s.session_id = sp.session_id "
                "WHERE s.session_id IS NULL OR s.origin IS NULL OR s.origin = ''"
            ).fetchall()
            assert len(orphans) == 0, f"Profiles without a session/origin: {[dict(r) for r in orphans]}"


class TestInsightMaterializationIdempotence:
    """Running materialization twice produces the same profile set."""

    def test_rebuild_is_idempotent(self, materialized_db: Path) -> None:
        import asyncio

        from polylogue.api import Polylogue

        with closing(_open_archive(materialized_db)) as conn:
            has_profiles = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_profiles'"
            ).fetchone()
            if has_profiles is None:
                pytest.skip("session_profiles table not present")

            ids_before = {r["session_id"] for r in conn.execute("SELECT session_id FROM session_profiles").fetchall()}

        async def _rebuild() -> None:
            archive = Polylogue(archive_root=materialized_db.parent, db_path=materialized_db)
            try:
                await archive.rebuild_insights()
            finally:
                await archive.close()

        asyncio.run(_rebuild())

        with closing(_open_archive(materialized_db)) as conn:
            ids_after = {r["session_id"] for r in conn.execute("SELECT session_id FROM session_profiles").fetchall()}

        assert ids_before == ids_after, "Rebuild changed profile set"


class TestWorkEventAgreement:
    """Work events must reference valid profiles."""

    def test_no_orphan_work_events(self, materialized_db: Path) -> None:
        with closing(_open_archive(materialized_db)) as conn:
            has_events = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_work_events'"
            ).fetchone()
            if has_events is None:
                pytest.skip("session_work_events table not present")

            orphans = conn.execute(
                "SELECT COUNT(*) FROM session_work_events we "
                "WHERE NOT EXISTS (SELECT 1 FROM session_profiles sp WHERE sp.session_id = we.session_id)"
            ).fetchone()[0]
            assert orphans == 0, f"Found {orphans} orphan work events"


class TestPhaseAgreement:
    """Phases must reference valid profiles."""

    def test_no_orphan_phases(self, materialized_db: Path) -> None:
        with closing(_open_archive(materialized_db)) as conn:
            has_phases = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_phases'"
            ).fetchone()
            if has_phases is None:
                pytest.skip("session_phases table not present")

            orphans = conn.execute(
                "SELECT COUNT(*) FROM session_phases sp2 "
                "WHERE NOT EXISTS (SELECT 1 FROM session_profiles sp WHERE sp.session_id = sp2.session_id)"
            ).fetchone()[0]
            assert orphans == 0, f"Found {orphans} orphan phases"
