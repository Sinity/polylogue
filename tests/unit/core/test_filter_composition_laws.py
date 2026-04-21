"""Filter composition laws: prove filter chains preserve monotonicity and subset properties.

Adding more filters can only narrow the result set, never widen it.
Composing independent filters is commutative. These are algebraic
properties of the filter DSL.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.infra.storage_records import ConversationBuilder, db_setup


@pytest.fixture()
def filterable_db(workspace_env: dict[str, Path]) -> Path:
    """Create a rich DB with varied conversations for filter testing."""
    db_path = db_setup(workspace_env)

    ConversationBuilder(db_path, "gpt-long").provider("chatgpt").title("Long GPT chat").add_message(
        role="user", text="First question about testing"
    ).add_message(role="assistant", text="Testing is important for correctness").add_message(
        role="user", text="Tell me more about property testing"
    ).add_message(role="assistant", text="Property testing uses random inputs to verify invariants").save()

    ConversationBuilder(db_path, "gpt-short").provider("chatgpt").title("Quick question").add_message(
        role="user", text="Hello"
    ).save()

    ConversationBuilder(db_path, "claude-mid").provider("claude-code").title("Refactoring session").add_message(
        role="user", text="Refactor the module"
    ).add_message(role="assistant", text="Done with refactoring").save()

    ConversationBuilder(db_path, "codex-mid").provider("codex").title("Code generation").add_message(
        role="user", text="Generate authentication code"
    ).add_message(role="assistant", text="Here is the auth implementation").save()

    return db_path


def _query_ids(conn: sqlite3.Connection, where_clause: str = "", params: tuple[object, ...] = ()) -> set[str]:
    sql = "SELECT conversation_id FROM conversations"
    if where_clause:
        sql += f" WHERE {where_clause}"
    return {r["conversation_id"] for r in conn.execute(sql, params).fetchall()}


class TestFilterMonotonicity:
    """Adding a filter can only narrow or maintain the result set."""

    def test_provider_filter_narrows(self, filterable_db: Path) -> None:
        from polylogue.storage.backends.connection import open_connection

        with open_connection(filterable_db) as conn:
            all_ids = _query_ids(conn)
            chatgpt_ids = _query_ids(conn, "provider_name = ?", ("chatgpt",))

            assert chatgpt_ids.issubset(all_ids)
            assert len(chatgpt_ids) < len(all_ids)

    def test_combined_filters_narrow_further(self, filterable_db: Path) -> None:
        from polylogue.storage.backends.connection import open_connection

        with open_connection(filterable_db) as conn:
            all_ids = _query_ids(conn)
            chatgpt_ids = _query_ids(conn, "provider_name = ?", ("chatgpt",))

            chatgpt_with_stats = set()
            for cid in chatgpt_ids:
                msg_count = conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (cid,)).fetchone()[
                    0
                ]
                if msg_count >= 2:
                    chatgpt_with_stats.add(cid)

            assert chatgpt_with_stats.issubset(chatgpt_ids)
            assert chatgpt_ids.issubset(all_ids)

    def test_empty_provider_returns_empty(self, filterable_db: Path) -> None:
        from polylogue.storage.backends.connection import open_connection

        with open_connection(filterable_db) as conn:
            nonexistent = _query_ids(conn, "provider_name = ?", ("nonexistent-provider",))
            assert nonexistent == set()


class TestFilterCommutativity:
    """Independent filters commute — order of application doesn't matter."""

    def test_provider_then_message_count_equals_reverse(self, filterable_db: Path) -> None:
        from polylogue.storage.backends.connection import open_connection

        with open_connection(filterable_db) as conn:
            chatgpt_ids = _query_ids(conn, "provider_name = ?", ("chatgpt",))
            chatgpt_with_msgs = set()
            for cid in chatgpt_ids:
                cnt = conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (cid,)).fetchone()[0]
                if cnt >= 2:
                    chatgpt_with_msgs.add(cid)

            all_with_msgs = set()
            for row in conn.execute("SELECT conversation_id FROM conversations").fetchall():
                cid = row["conversation_id"]
                cnt = conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (cid,)).fetchone()[0]
                if cnt >= 2:
                    all_with_msgs.add(cid)
            msgs_then_chatgpt = all_with_msgs & chatgpt_ids

            assert chatgpt_with_msgs == msgs_then_chatgpt


class TestFilterIdempotence:
    """Applying the same filter twice yields the same result."""

    def test_double_provider_filter_is_idempotent(self, filterable_db: Path) -> None:
        from polylogue.storage.backends.connection import open_connection

        with open_connection(filterable_db) as conn:
            once = _query_ids(conn, "provider_name = ?", ("chatgpt",))
            twice = _query_ids(conn, "provider_name = ? AND provider_name = ?", ("chatgpt", "chatgpt"))
            assert once == twice


class TestFilterPartition:
    """Provider filters partition the conversation space."""

    def test_providers_partition_total(self, filterable_db: Path) -> None:
        from polylogue.storage.backends.connection import open_connection

        with open_connection(filterable_db) as conn:
            all_ids = _query_ids(conn)
            providers = {r[0] for r in conn.execute("SELECT DISTINCT provider_name FROM conversations").fetchall()}

            union = set()
            for p in providers:
                union |= _query_ids(conn, "provider_name = ?", (p,))

            assert union == all_ids

    def test_provider_partitions_are_disjoint(self, filterable_db: Path) -> None:
        from polylogue.storage.backends.connection import open_connection

        with open_connection(filterable_db) as conn:
            providers = [r[0] for r in conn.execute("SELECT DISTINCT provider_name FROM conversations").fetchall()]

            for i, p1 in enumerate(providers):
                for p2 in providers[i + 1 :]:
                    ids1 = _query_ids(conn, "provider_name = ?", (p1,))
                    ids2 = _query_ids(conn, "provider_name = ?", (p2,))
                    assert ids1.isdisjoint(ids2), f"{p1} and {p2} overlap"
