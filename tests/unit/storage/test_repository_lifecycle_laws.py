"""Repository lifecycle laws over declarative archive scenarios."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from polylogue.storage.backends.connection import open_connection
from polylogue.storage.hydrators import conversation_from_records
from tests.infra.archive_scenarios import (
    ArchiveScenario,
    ScenarioAttachment,
    ScenarioMessage,
    repository_for_scenario_db,
    seed_workspace_scenarios,
)
from tests.infra.oracles import assert_archive_surfaces_agree, assert_conversation_surfaces_agree
from tests.infra.semantic_facts import ArchiveFacts, ConversationFacts


@pytest.mark.asyncio()
async def test_repository_metadata_tag_and_delete_lifecycle_laws(workspace_env: Mapping[str, Path]) -> None:
    """Repository writes converge and keep archive surfaces internally consistent."""
    scenario = ArchiveScenario(
        name="repo-lifecycle",
        provider="claude-code",
        title="Lifecycle law",
        messages=(
            ScenarioMessage(role="user", text="Inspect lifecycle", message_id="m-user"),
            ScenarioMessage(
                role="assistant",
                text="I will call a tool",
                message_id="m-tool",
                content_blocks=(
                    {
                        "type": "tool_use",
                        "tool_name": "shell",
                        "tool_input": {"command": "polylogue doctor --json"},
                    },
                ),
                attachments=(ScenarioAttachment(attachment_id="att-lifecycle", mime_type="application/json"),),
            ),
        ),
        metadata={"tags": ["initial"], "summary": "before"},
    )
    db_path, _ = seed_workspace_scenarios(workspace_env, [scenario])
    repository = repository_for_scenario_db(db_path)
    try:
        with open_connection(db_path) as conn:
            assert_conversation_surfaces_agree(
                scenario.facts_from_connection(conn),
                scenario.hydrated_facts_from_connection(conn),
                await scenario.facts_from_repository(repository),
            )

        await repository.add_tag(scenario.resolved_conversation_id, "review")
        await repository.add_tag(scenario.resolved_conversation_id, "review")
        await repository.update_metadata(scenario.resolved_conversation_id, "summary", "after")
        await repository.update_metadata(
            scenario.resolved_conversation_id,
            "audit",
            {"status": "checked", "tooling": ["repository", "scenario"]},
        )
        await repository.delete_metadata(scenario.resolved_conversation_id, "missing")

        metadata = await repository.get_metadata(scenario.resolved_conversation_id)
        assert metadata["tags"] == ["initial", "review"]
        assert metadata["summary"] == "after"
        assert metadata["audit"] == {"status": "checked", "tooling": ["repository", "scenario"]}
        assert await repository.list_tags() == {"initial": 1, "review": 1}

        await repository.remove_tag(scenario.resolved_conversation_id, "initial")
        await repository.remove_tag(scenario.resolved_conversation_id, "initial")
        metadata_after_remove = await repository.get_metadata(scenario.resolved_conversation_id)
        assert metadata_after_remove["tags"] == ["review"]
        assert await repository.list_tags(provider="claude-code") == {"review": 1}

        with open_connection(db_path) as conn:
            db_facts_before_delete = ArchiveFacts.from_db_connection(conn)
        repo_facts_before_delete = ArchiveFacts.from_conversations(await repository.list(limit=None))
        assert_archive_surfaces_agree(db_facts_before_delete, repo_facts_before_delete)
        assert db_facts_before_delete.total_conversations == 1

        assert await repository.delete_conversation(scenario.resolved_conversation_id) is True
        assert await repository.delete_conversation(scenario.resolved_conversation_id) is False
        assert await repository.get(scenario.resolved_conversation_id) is None
        assert await repository.count() == 0
        assert await repository.list_tags() == {}

        with open_connection(db_path) as conn:
            assert ArchiveFacts.from_db_connection(conn) == ArchiveFacts(
                total_conversations=0,
                provider_counts={},
                total_messages=0,
                conversation_ids=(),
            )
            for table_name in ("messages", "content_blocks", "attachment_refs"):
                count = int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
                assert count == 0, f"{table_name} retained rows after repository delete"
    finally:
        await repository.close()


@pytest.mark.asyncio()
async def test_repository_resave_existing_conversation_is_idempotent(workspace_env: Mapping[str, Path]) -> None:
    """Saving existing records does not create duplicate archive facts."""
    scenario = ArchiveScenario(
        name="repo-resave",
        provider="chatgpt",
        title="Resave law",
        messages=(
            ScenarioMessage(role="user", text="Question", message_id="m1"),
            ScenarioMessage(role="assistant", text="Answer", message_id="m2"),
        ),
    )
    db_path, _ = seed_workspace_scenarios(workspace_env, [scenario])
    repository = repository_for_scenario_db(db_path)
    try:
        with open_connection(db_path) as conn:
            conv_record, msg_records, attachment_records = scenario.records_from_connection(conn)
            before = scenario.facts_from_connection(conn)

        counts = await repository.save_conversation(conv_record, msg_records, attachment_records)
        assert counts["conversations"] == 0
        assert counts["messages"] == 0
        assert counts["skipped_conversations"] == 1
        assert counts["skipped_messages"] == 2

        with open_connection(db_path) as conn:
            after = scenario.facts_from_connection(conn)
        conversation = await repository.get(scenario.resolved_conversation_id)
        assert conversation is not None
        assert_conversation_surfaces_agree(before, after, ConversationFacts.from_domain_conversation(conversation))
    finally:
        await repository.close()


def test_domain_conversation_to_record_generates_content_hash(workspace_env: Mapping[str, Path]) -> None:
    """Hydrated domain conversations remain convertible even without stored metadata hashes."""
    from polylogue.storage.repository_write_conversations import conversation_to_record

    scenario = ArchiveScenario(
        name="domain-record",
        provider="chatgpt",
        title="Domain conversion",
        messages=(ScenarioMessage(role="user", text="Question", message_id="m1"),),
    )
    db_path, _ = seed_workspace_scenarios(workspace_env, [scenario])
    with open_connection(db_path) as conn:
        conv_record, msg_records, attachment_records = scenario.records_from_connection(conn)

    domain_conversation = conversation_from_records(conv_record, msg_records, attachment_records)
    record = conversation_to_record(domain_conversation)
    assert record.content_hash
