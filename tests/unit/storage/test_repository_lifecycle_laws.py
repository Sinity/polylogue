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
from tests.infra.oracles import assert_conversation_surfaces_agree
from tests.infra.semantic_facts import ArchiveFacts, ConversationFacts
from tests.infra.state_machines import RepositoryLifecycleHarness


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
    harness = RepositoryLifecycleHarness.create(
        db_path=db_path,
        archive_root=workspace_env["archive_root"],
        scenarios=(scenario,),
    )
    try:
        await harness.assert_conversation_visible(scenario)

        await harness.add_tag_and_assert_visible(scenario, "review")
        await harness.add_tag_and_assert_visible(scenario, "review")
        await harness.update_metadata_and_assert_visible(scenario, "summary", "after")
        await harness.update_metadata_and_assert_visible(
            scenario,
            "audit",
            {"status": "checked", "tooling": ["repository", "scenario"]},
        )
        await harness.delete_metadata_and_assert_visible(scenario, "missing")

        metadata = await harness.repository.get_metadata(scenario.resolved_conversation_id)
        assert metadata["tags"] == ["initial", "review"]
        assert metadata["summary"] == "after"
        assert metadata["audit"] == {"status": "checked", "tooling": ["repository", "scenario"]}
        await harness.assert_tags({"initial": 1, "review": 1})

        await harness.remove_tag_and_assert_visible(scenario, "initial")
        await harness.remove_tag_and_assert_visible(scenario, "initial")
        metadata_after_remove = await harness.repository.get_metadata(scenario.resolved_conversation_id)
        assert metadata_after_remove["tags"] == ["review"]
        await harness.assert_tags({"review": 1}, provider="claude-code")
        await harness.assert_archive_agrees(total_conversations=1)

        await harness.delete_conversation_and_assert_absent(scenario)
        assert await harness.repository.count() == 0
        await harness.assert_tags({})

        with open_connection(db_path) as conn:
            assert ArchiveFacts.from_db_connection(conn) == ArchiveFacts(
                total_conversations=0,
                provider_counts={},
                total_messages=0,
                conversation_ids=(),
            )
    finally:
        await harness.close()


@pytest.mark.asyncio()
async def test_repository_lifecycle_state_keeps_neighbor_conversations_visible(
    workspace_env: Mapping[str, Path],
) -> None:
    """State transitions for one conversation must not disturb its archive neighbors."""
    target = ArchiveScenario(
        name="repo-state-target",
        provider="claude-code",
        title="Target lifecycle",
        messages=(ScenarioMessage(role="user", text="Mutate me", message_id="target-m1"),),
        metadata={"tags": ["target"]},
    )
    neighbor = ArchiveScenario(
        name="repo-state-neighbor",
        provider="chatgpt",
        title="Neighbor lifecycle",
        messages=(ScenarioMessage(role="assistant", text="Keep me", message_id="neighbor-m1"),),
        metadata={"tags": ["neighbor"]},
    )
    db_path, _ = seed_workspace_scenarios(workspace_env, [target, neighbor])
    harness = RepositoryLifecycleHarness.create(
        db_path=db_path,
        archive_root=workspace_env["archive_root"],
        scenarios=(target, neighbor),
    )
    try:
        await harness.assert_archive_agrees(total_conversations=2)
        await harness.add_tag_and_assert_visible(target, "review")
        await harness.assert_conversation_visible(neighbor)
        await harness.update_metadata_and_assert_visible(target, "summary", "reviewed")
        await harness.assert_metadata_contains(
            target.resolved_conversation_id,
            {"tags": ["target", "review"], "summary": "reviewed"},
        )
        await harness.assert_metadata_contains(neighbor.resolved_conversation_id, {"tags": ["neighbor"]})
        await harness.delete_conversation_and_assert_absent(target)

        await harness.assert_conversation_visible(neighbor)
        archive_facts = await harness.assert_archive_agrees(total_conversations=1)
        assert archive_facts.conversation_ids == (neighbor.resolved_conversation_id,)
        await harness.assert_tags({"neighbor": 1})
    finally:
        await harness.close()


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
