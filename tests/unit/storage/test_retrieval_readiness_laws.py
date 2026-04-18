"""Retrieval readiness laws: prove index/query semantics are consistent.

These laws verify that FTS indexes, provider filters, and query surfaces
agree with each other and with the underlying stored data.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.infra.storage_records import ConversationBuilder, db_setup


@pytest.fixture()
def populated_db(workspace_env: Any) -> Any:
    """Create a DB with conversations across multiple providers, with FTS indexed."""
    db_path = db_setup(workspace_env)

    (
        ConversationBuilder(db_path, "chatgpt-1")
        .provider("chatgpt")
        .title("ChatGPT conversation about testing")
        .add_message(role="user", text="How do I write property tests?")
        .add_message(role="assistant", text="Property tests verify invariants using random inputs")
        .save()
    )
    (
        ConversationBuilder(db_path, "claude-1")
        .provider("claude-code")
        .title("Claude Code session on refactoring")
        .add_message(role="user", text="Refactor the storage module")
        .add_message(role="assistant", text="I will restructure the query layer")
        .add_message(role="user", text="Also fix the property tests")
        .save()
    )
    (
        ConversationBuilder(db_path, "claude-2")
        .provider("claude-code")
        .title("Claude debugging memory leak")
        .add_message(role="user", text="Memory keeps growing during ingest")
        .add_message(role="assistant", text="The blob store path has a leak")
        .save()
    )
    (
        ConversationBuilder(db_path, "codex-1")
        .provider("codex")
        .title("Codex adding authentication")
        .add_message(role="user", text="Add OAuth2 authentication")
        .save()
    )

    return db_path


# ---------------------------------------------------------------------------
# Law 1: Provider filter results are strict subsets of unfiltered
# ---------------------------------------------------------------------------


class TestProviderFilterSubset:
    def test_provider_filter_returns_subset(self: Any, populated_db: Any) -> None:
        from polylogue.storage.backends.connection import open_connection

        with open_connection(populated_db) as conn:
            all_ids = {
                r["conversation_id"] for r in conn.execute("SELECT conversation_id FROM conversations").fetchall()
            }
            claude_ids = {
                r["conversation_id"]
                for r in conn.execute(
                    "SELECT conversation_id FROM conversations WHERE provider_name = ?",
                    ("claude-code",),
                ).fetchall()
            }

            assert claude_ids.issubset(all_ids), "Provider-filtered results must be a subset of all"
            assert len(claude_ids) == 2
            assert len(all_ids) == 4

    def test_all_providers_partition_total(self: Any, populated_db: Any) -> None:
        """Union of all per-provider sets equals the full set."""
        from polylogue.storage.backends.connection import open_connection

        with open_connection(populated_db) as conn:
            all_ids = {
                r["conversation_id"] for r in conn.execute("SELECT conversation_id FROM conversations").fetchall()
            }

            providers = [
                r["provider_name"] for r in conn.execute("SELECT DISTINCT provider_name FROM conversations").fetchall()
            ]

            union = set()
            for p in providers:
                ids = {
                    r["conversation_id"]
                    for r in conn.execute(
                        "SELECT conversation_id FROM conversations WHERE provider_name = ?", (p,)
                    ).fetchall()
                }
                union |= ids

            assert union == all_ids, "Union of provider partitions must equal total"


# ---------------------------------------------------------------------------
# Law 2: FTS index contains exactly the indexable messages
# ---------------------------------------------------------------------------


class TestFTSIndexCompleteness:
    def test_fts_message_count_matches_messages_table(self: Any, populated_db: Any) -> None:
        from polylogue.storage.backends.connection import open_connection

        with open_connection(populated_db) as conn:
            msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

            fts_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
            ).fetchone()
            if fts_exists is None:
                pytest.skip("FTS table not present")

            fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]

            assert fts_count == msg_count, f"FTS row count ({fts_count}) != messages table count ({msg_count})"

    def test_fts_search_returns_subset_of_messages(self: Any, populated_db: Any) -> None:
        from polylogue.storage.backends.connection import open_connection

        with open_connection(populated_db) as conn:
            fts_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
            ).fetchone()
            if fts_exists is None:
                pytest.skip("FTS table not present")

            all_msg_ids = {r["message_id"] for r in conn.execute("SELECT message_id FROM messages").fetchall()}

            fts_hits = conn.execute("SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'property'").fetchall()
            fts_msg_ids = set()
            for hit in fts_hits:
                row = conn.execute("SELECT message_id FROM messages WHERE rowid = ?", (hit[0],)).fetchone()
                if row:
                    fts_msg_ids.add(row["message_id"])

            assert fts_msg_ids.issubset(all_msg_ids), "FTS hits must reference existing messages"


# ---------------------------------------------------------------------------
# Law 3: List count agrees with actual list length
# ---------------------------------------------------------------------------


class TestCountListAgreement:
    def test_count_matches_list_length(self: Any, populated_db: Any) -> None:
        from polylogue.storage.backends.connection import open_connection

        with open_connection(populated_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            rows = conn.execute("SELECT conversation_id FROM conversations").fetchall()

            assert count == len(rows), f"COUNT(*) ({count}) != len(SELECT) ({len(rows)})"

    def test_per_provider_count_matches_filtered_list(self: Any, populated_db: Any) -> None:
        from polylogue.storage.backends.connection import open_connection

        with open_connection(populated_db) as conn:
            for provider in ("chatgpt", "claude-code", "codex"):
                count = conn.execute(
                    "SELECT COUNT(*) FROM conversations WHERE provider_name = ?", (provider,)
                ).fetchone()[0]
                rows = conn.execute(
                    "SELECT conversation_id FROM conversations WHERE provider_name = ?", (provider,)
                ).fetchall()

                assert count == len(rows), f"Provider {provider}: COUNT={count}, len(rows)={len(rows)}"


# ---------------------------------------------------------------------------
# Law 4: Message stats agree with stored messages
# ---------------------------------------------------------------------------


class TestMessageStatsConsistency:
    def test_conversation_stats_match_actual_messages(self: Any, populated_db: Any) -> None:
        from polylogue.storage.backends.connection import open_connection

        with open_connection(populated_db) as conn:
            stats_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_stats'"
            ).fetchone()
            if stats_exists is None:
                pytest.skip("conversation_stats table not present")

            for row in conn.execute(
                "SELECT cs.conversation_id, cs.message_count, "
                "(SELECT COUNT(*) FROM messages m WHERE m.conversation_id = cs.conversation_id) AS actual_count "
                "FROM conversation_stats cs"
            ).fetchall():
                assert row["message_count"] == row["actual_count"], (
                    f"Stats message_count ({row['message_count']}) != actual ({row['actual_count']}) "
                    f"for {row['conversation_id']}"
                )
