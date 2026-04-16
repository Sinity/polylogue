"""Cross-surface agreement: prove repository, hydration, and records agree.

When the same conversation is viewed through different surfaces, the
semantic facts must be identical. A failure here means two surfaces
disagree about what's in the archive.
"""

from __future__ import annotations

import pytest

from tests.infra.semantic_facts import ArchiveFacts, ConversationFacts
from tests.infra.storage_records import ConversationBuilder, db_setup


@pytest.fixture()
def multi_provider_db(workspace_env):
    """Populate a DB with conversations across providers."""
    db_path = db_setup(workspace_env)

    ConversationBuilder(db_path, "chatgpt-xsurf-1").provider("chatgpt").title("GPT chat").add_message(
        role="user", text="Hello GPT"
    ).add_message(role="assistant", text="Hello user").save()

    ConversationBuilder(db_path, "claude-xsurf-1").provider("claude-code").title("Claude session").add_message(
        role="user", text="Refactor this"
    ).add_message(role="assistant", text="Done").add_message(role="user", text="Thanks").save()

    ConversationBuilder(db_path, "codex-xsurf-1").provider("codex").title("Codex work").add_message(
        role="user", text="Generate code"
    ).save()

    return db_path


# ---------------------------------------------------------------------------
# Record-level vs hydration-level facts agree
# ---------------------------------------------------------------------------


class TestRecordVsHydrationAgreement:
    def test_facts_agree_for_each_conversation(self, multi_provider_db):
        from polylogue.storage.backends.connection import open_connection
        from polylogue.storage.backends.queries.mappers_archive import (
            _row_to_conversation,
            _row_to_message,
        )
        from polylogue.storage.hydrators import conversation_from_records

        with open_connection(multi_provider_db) as conn:
            for conv_row in conn.execute("SELECT * FROM conversations").fetchall():
                conv_record = _row_to_conversation(conv_row)
                cid = conv_record.conversation_id

                msg_rows = conn.execute(
                    "SELECT * FROM messages WHERE conversation_id = ? ORDER BY sort_key", (cid,)
                ).fetchall()
                msg_records = [_row_to_message(r) for r in msg_rows]

                record_facts = ConversationFacts.from_records(conv_record, msg_records)

                hydrated = conversation_from_records(conv_record, msg_records, [])
                hydrated_facts = ConversationFacts.from_domain_conversation(hydrated)

                assert record_facts.conversation_id == hydrated_facts.conversation_id
                assert record_facts.provider == hydrated_facts.provider
                assert record_facts.title == hydrated_facts.title
                assert record_facts.message_count == hydrated_facts.message_count
                assert record_facts.role_multiset == hydrated_facts.role_multiset


# ---------------------------------------------------------------------------
# Archive-level facts: direct SQL vs count queries agree
# ---------------------------------------------------------------------------


class TestArchiveFactsConsistency:
    def test_archive_facts_internally_consistent(self, multi_provider_db):
        from polylogue.storage.backends.connection import open_connection

        with open_connection(multi_provider_db) as conn:
            facts = ArchiveFacts.from_db_connection(conn)

            assert facts.total_conversations == 3
            assert sum(facts.provider_counts.values()) == facts.total_conversations
            assert facts.total_messages > 0
            assert facts.provider_counts.get("chatgpt") == 1
            assert facts.provider_counts.get("claude-code") == 1
            assert facts.provider_counts.get("codex") == 1

    def test_provider_partition_exhaustive(self, multi_provider_db):
        """Every conversation belongs to exactly one provider in the facts."""
        from polylogue.storage.backends.connection import open_connection

        with open_connection(multi_provider_db) as conn:
            facts = ArchiveFacts.from_db_connection(conn)

            all_ids = {
                r["conversation_id"] for r in conn.execute("SELECT conversation_id FROM conversations").fetchall()
            }
            per_provider_ids = set()
            for provider in facts.provider_counts:
                ids = {
                    r["conversation_id"]
                    for r in conn.execute(
                        "SELECT conversation_id FROM conversations WHERE provider_name = ?", (provider,)
                    ).fetchall()
                }
                per_provider_ids |= ids

            assert per_provider_ids == all_ids


# ---------------------------------------------------------------------------
# Synthetic roundtrip: generated payload facts match hydrated facts
# ---------------------------------------------------------------------------


class TestSyntheticRoundtripFactAgreement:
    @pytest.mark.parametrize("provider_name", ["chatgpt", "claude-code", "claude-ai", "codex", "gemini"])
    def test_parsed_vs_hydrated_facts_agree(self, provider_name, workspace_env):
        import json

        from polylogue.pipeline.prepare_transform import transform_to_records
        from polylogue.schemas.synthetic.core import SyntheticCorpus
        from polylogue.sources.dispatch import detect_provider, parse_payload
        from polylogue.storage.backends.connection import open_connection
        from polylogue.storage.backends.queries.mappers_archive import (
            _row_to_conversation,
            _row_to_message,
        )
        from polylogue.storage.hydrators import conversation_from_records
        from tests.infra.storage_records import db_setup, store_records

        corpus = SyntheticCorpus.for_provider(provider_name)
        raw_bytes = corpus.generate(count=1, seed=99)[0]

        text = raw_bytes.decode("utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = [json.loads(line) for line in text.strip().splitlines() if line.strip()]

        detected = detect_provider(payload)
        assert detected is not None
        parsed_list = parse_payload(detected, payload, f"xsurf-{provider_name}")
        parsed = parsed_list[0]

        archive_root = workspace_env["archive_root"]
        result = transform_to_records(parsed, f"test-{provider_name}", archive_root=archive_root)

        db_path = db_setup(workspace_env)
        with open_connection(db_path) as conn:
            bundle = result.bundle
            store_records(
                conversation=bundle.conversation,
                messages=bundle.messages,
                attachments=bundle.attachments,
                conn=conn,
            )

            cid = bundle.conversation.conversation_id
            conv_row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?", (cid,)).fetchone()
            conv_record = _row_to_conversation(conv_row)
            msg_rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY sort_key", (cid,)
            ).fetchall()
            msg_records = [_row_to_message(r) for r in msg_rows]

            hydrated = conversation_from_records(conv_record, msg_records, [])
            hydrated_facts = ConversationFacts.from_domain_conversation(hydrated)

            assert hydrated_facts.message_count == len(parsed.messages)
            assert hydrated_facts.provider == str(parsed.provider_name)
            assert hydrated_facts.title == parsed.title
