"""Product materialization laws: session products agree with source conversations.

Proves that materialized session products (profiles, work events, phases)
reflect the conversations they were derived from — counts match, provider
agrees, no phantom products for non-existent conversations.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from tests.infra.storage_records import ConversationBuilder, db_setup


@pytest.fixture()
def materialized_db(workspace_env: Mapping[str, Path]) -> Path:
    """Create a DB with conversations and run session product materialization."""
    db_path = db_setup(workspace_env)

    ConversationBuilder(db_path, "mat-gpt-1").provider("chatgpt").title("GPT session").add_message(
        role="user", text="Write a function"
    ).add_message(role="assistant", text="def hello(): pass").add_message(
        role="user", text="Add error handling"
    ).add_message(role="assistant", text="def hello(): try: pass except: pass").save()

    ConversationBuilder(db_path, "mat-claude-1").provider("claude-code").title("Claude session").add_message(
        role="user", text="Refactor storage"
    ).add_message(role="assistant", text="I will restructure the module").save()

    ConversationBuilder(db_path, "mat-codex-1").provider("codex").title("Codex session").add_message(
        role="user", text="Generate tests"
    ).add_message(role="assistant", text="Here are the tests").add_message(role="user", text="Add edge cases").save()

    from polylogue.storage.backends.connection import open_connection
    from polylogue.storage.products.session.rebuild import rebuild_session_products_sync

    with open_connection(db_path) as conn:
        rebuild_session_products_sync(conn)
        conn.commit()

    return db_path


class TestProfileConversationAgreement:
    """Every session profile must correspond to exactly one real conversation."""

    def test_profile_count_matches_conversation_count(self, materialized_db: Path) -> None:
        from polylogue.storage.backends.connection import open_connection

        with open_connection(materialized_db) as conn:
            conv_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            has_profiles = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_profiles'"
            ).fetchone()
            if has_profiles is None:
                pytest.skip("session_profiles table not present")
            profile_count = conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0]
            assert profile_count == conv_count, f"Profile count ({profile_count}) != conversation count ({conv_count})"

    def test_no_phantom_profiles(self, materialized_db: Path) -> None:
        """No profile should reference a non-existent conversation."""
        from polylogue.storage.backends.connection import open_connection

        with open_connection(materialized_db) as conn:
            has_profiles = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_profiles'"
            ).fetchone()
            if has_profiles is None:
                pytest.skip("session_profiles table not present")

            phantom_count = conn.execute(
                "SELECT COUNT(*) FROM session_profiles sp "
                "WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = sp.conversation_id)"
            ).fetchone()[0]
            assert phantom_count == 0, f"Found {phantom_count} phantom profiles"

    def test_profile_provider_matches_conversation(self, materialized_db: Path) -> None:
        """Profile provider_name must match source conversation provider_name."""
        from polylogue.storage.backends.connection import open_connection

        with open_connection(materialized_db) as conn:
            has_profiles = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_profiles'"
            ).fetchone()
            if has_profiles is None:
                pytest.skip("session_profiles table not present")

            mismatches = conn.execute(
                "SELECT sp.conversation_id, sp.provider_name AS sp_provider, c.provider_name AS c_provider "
                "FROM session_profiles sp "
                "JOIN conversations c ON c.conversation_id = sp.conversation_id "
                "WHERE sp.provider_name != c.provider_name"
            ).fetchall()
            assert len(mismatches) == 0, f"Provider mismatches: {[dict(r) for r in mismatches]}"


class TestProductMaterializationIdempotence:
    """Running materialization twice produces the same profile set."""

    def test_rebuild_is_idempotent(self, materialized_db: Path) -> None:
        from polylogue.storage.backends.connection import open_connection
        from polylogue.storage.products.session.rebuild import rebuild_session_products_sync

        with open_connection(materialized_db) as conn:
            has_profiles = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_profiles'"
            ).fetchone()
            if has_profiles is None:
                pytest.skip("session_profiles table not present")

            ids_before = {
                r["conversation_id"] for r in conn.execute("SELECT conversation_id FROM session_profiles").fetchall()
            }

            rebuild_session_products_sync(conn)
            conn.commit()

            ids_after = {
                r["conversation_id"] for r in conn.execute("SELECT conversation_id FROM session_profiles").fetchall()
            }

            assert ids_before == ids_after, "Rebuild changed profile set"


class TestWorkEventAgreement:
    """Work events must reference valid profiles."""

    def test_no_orphan_work_events(self, materialized_db: Path) -> None:
        from polylogue.storage.backends.connection import open_connection

        with open_connection(materialized_db) as conn:
            has_events = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_work_events'"
            ).fetchone()
            if has_events is None:
                pytest.skip("session_work_events table not present")

            orphans = conn.execute(
                "SELECT COUNT(*) FROM session_work_events we "
                "WHERE NOT EXISTS (SELECT 1 FROM session_profiles sp WHERE sp.conversation_id = we.conversation_id)"
            ).fetchone()[0]
            assert orphans == 0, f"Found {orphans} orphan work events"


class TestPhaseAgreement:
    """Phases must reference valid profiles."""

    def test_no_orphan_phases(self, materialized_db: Path) -> None:
        from polylogue.storage.backends.connection import open_connection

        with open_connection(materialized_db) as conn:
            has_phases = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_phases'"
            ).fetchone()
            if has_phases is None:
                pytest.skip("session_phases table not present")

            orphans = conn.execute(
                "SELECT COUNT(*) FROM session_phases sp2 "
                "WHERE NOT EXISTS (SELECT 1 FROM session_profiles sp WHERE sp.conversation_id = sp2.conversation_id)"
            ).fetchone()[0]
            assert orphans == 0, f"Found {orphans} orphan phases"
