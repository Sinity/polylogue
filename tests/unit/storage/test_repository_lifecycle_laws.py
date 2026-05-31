"""Repository lifecycle laws over declarative archive scenarios."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from polylogue.storage.hydrators import conversation_from_records
from polylogue.storage.runtime import PROVIDER_EVENT_MATERIALIZER_VERSION, ProviderEventRecord
from polylogue.storage.sqlite.connection import open_connection
from polylogue.types import ProviderEventId
from tests.infra.archive_scenarios import (
    ArchiveScenario,
    ScenarioAttachment,
    ScenarioContentBlock,
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
                    ScenarioContentBlock.tool_use(
                        tool_name="shell",
                        tool_input={"command": "polylogue doctor --format json"},
                    ),
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
        # #1240: tags are M2M-only; conversations.metadata['tags'] is no
        # longer dual-written by add_tag, so the seeded payload is unchanged
        # after add_tag("review").
        assert metadata["tags"] == ["initial"]
        assert metadata["summary"] == "after"
        assert metadata["audit"] == {"status": "checked", "tooling": ["repository", "scenario"]}
        # M2M reflects the seeded "initial" (written into conversation_tags by
        # the scenario seeder) plus the "review" added through add_tag.
        await harness.assert_tags({"initial": 1, "review": 1})

        await harness.remove_tag_and_assert_visible(scenario, "initial")
        await harness.remove_tag_and_assert_visible(scenario, "initial")
        metadata_after_remove = await harness.repository.get_metadata(scenario.resolved_conversation_id)
        # remove_tag only touches M2M; the seeded metadata payload is untouched.
        assert metadata_after_remove["tags"] == ["initial"]
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
        # #1240: add_tag no longer dual-writes into JSON metadata; the seeded
        # ["target"] payload is unchanged by add_tag("review").
        await harness.assert_metadata_contains(
            target.resolved_conversation_id,
            {"tags": ["target"], "summary": "reviewed"},
        )
        await harness.assert_metadata_contains(neighbor.resolved_conversation_id, {"tags": ["neighbor"]})
        await harness.delete_conversation_and_assert_absent(target)

        await harness.assert_conversation_visible(neighbor)
        archive_facts = await harness.assert_archive_agrees(total_conversations=1)
        assert archive_facts.conversation_ids == (neighbor.resolved_conversation_id,)
        # Target's M2M rows cascaded out with delete_conversation; neighbor's
        # seeded "neighbor" tag remains in M2M.
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

        # First write: no prior DB entry, so this stores the record with
        # ConversationBuilder's hash (any stable hash value works for
        # idempotency — the key invariant is DB hash == record hash).
        await repository.save_conversation(conv_record, msg_records, attachment_records)

        # Second write: DB hash == record hash → idempotency skip fires.
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
    from polylogue.storage.repository.archive.writes.conversations import conversation_to_record

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


@pytest.mark.asyncio()
async def test_repository_rejects_domain_conversation_without_message_records(
    workspace_env: Mapping[str, Path],
) -> None:
    """A hydrated domain object must not be re-saved as an empty runtime graph."""
    scenario = ArchiveScenario(
        name="domain-empty-records",
        provider="claude-code",
        title="Domain empty records",
        messages=(ScenarioMessage(role="user", text="Question", message_id="m1"),),
    )
    db_path, _ = seed_workspace_scenarios(workspace_env, [scenario])
    repository = repository_for_scenario_db(db_path)
    try:
        conversation = await repository.get(scenario.resolved_conversation_id)
        assert conversation is not None

        with pytest.raises(ValueError, match="domain Conversation with messages but no MessageRecord"):
            await repository.save_conversation(conversation, [], [])

        with open_connection(db_path) as conn:
            assert_conversation_surfaces_agree(
                scenario.facts_from_connection(conn),
                await scenario.facts_from_repository(repository),
            )
    finally:
        await repository.close()


@pytest.mark.asyncio()
async def test_repository_resaves_hydrated_domain_provider_events(
    workspace_env: Mapping[str, Path],
) -> None:
    """Domain Conversation writes preserve first-class provider events."""
    scenario = ArchiveScenario(
        name="domain-provider-events",
        provider="claude-code",
        title="Domain provider events",
        messages=(ScenarioMessage(role="user", text="Question", message_id="m1"),),
    )
    db_path, _ = seed_workspace_scenarios(workspace_env, [scenario])
    with open_connection(db_path) as conn:
        conv_record, msg_records, attachment_records = scenario.records_from_connection(conn)
        conn.execute(
            """
            INSERT INTO provider_events (
                event_id, conversation_id, source_name, event_index, event_type,
                timestamp, sort_key, materializer_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"{conv_record.conversation_id}:provider-event:000000",
                conv_record.conversation_id,
                conv_record.source_name,
                0,
                "compaction",
                conv_record.updated_at,
                0.5,
                PROVIDER_EVENT_MATERIALIZER_VERSION,
            ),
        )
        conn.execute(
            "INSERT INTO provider_event_compactions (event_id, summary) VALUES (?, ?)",
            (f"{conv_record.conversation_id}:provider-event:000000", "compact"),
        )
        conn.commit()

    repository = repository_for_scenario_db(db_path)
    try:
        conversation = await repository.get(scenario.resolved_conversation_id)
        assert conversation is not None
        assert [event.event_type for event in conversation.provider_events] == ["compaction"]

        counts = await repository.save_conversation(conversation, msg_records, attachment_records)
        assert counts["provider_events"] == 1

        refreshed = await repository.get(scenario.resolved_conversation_id)
        assert refreshed is not None
        assert [event.payload for event in refreshed.provider_events] == [{"summary": "compact"}]
    finally:
        await repository.close()


@pytest.mark.asyncio()
async def test_repository_record_resave_preserves_provider_events_when_omitted(
    workspace_env: Mapping[str, Path],
) -> None:
    """Record writes without provider events must not erase existing event rows."""
    scenario = ArchiveScenario(
        name="record-provider-events-preserve",
        provider="claude-code",
        title="Preserve provider events",
        messages=(ScenarioMessage(role="user", text="Question", message_id="m1"),),
    )
    db_path, _ = seed_workspace_scenarios(workspace_env, [scenario])
    with open_connection(db_path) as conn:
        conv_record, msg_records, attachment_records = scenario.records_from_connection(conn)
        conn.execute(
            """
            INSERT INTO provider_events (
                event_id, conversation_id, source_name, event_index, event_type,
                timestamp, sort_key, materializer_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"{conv_record.conversation_id}:provider-event:000000",
                conv_record.conversation_id,
                conv_record.source_name,
                0,
                "compaction",
                conv_record.updated_at,
                0.5,
                PROVIDER_EVENT_MATERIALIZER_VERSION,
            ),
        )
        conn.execute(
            "INSERT INTO provider_event_compactions (event_id, summary) VALUES (?, ?)",
            (f"{conv_record.conversation_id}:provider-event:000000", "keep"),
        )
        conn.commit()

    repository = repository_for_scenario_db(db_path)
    try:
        counts = await repository.save_conversation(conv_record, msg_records, attachment_records)
        assert counts["provider_events"] == 0

        refreshed = await repository.get(scenario.resolved_conversation_id)
        assert refreshed is not None
        assert [event.payload for event in refreshed.provider_events] == [{"summary": "keep"}]
    finally:
        await repository.close()


@pytest.mark.asyncio()
async def test_repository_record_resave_replaces_explicit_provider_events_on_hash_match(
    workspace_env: Mapping[str, Path],
) -> None:
    """Explicit provider-event records remain authoritative even when content hash is unchanged."""
    scenario = ArchiveScenario(
        name="record-provider-events-replace",
        provider="claude-code",
        title="Replace provider events",
        messages=(ScenarioMessage(role="user", text="Question", message_id="m1"),),
    )
    db_path, _ = seed_workspace_scenarios(workspace_env, [scenario])
    with open_connection(db_path) as conn:
        conv_record, msg_records, attachment_records = scenario.records_from_connection(conn)
        conn.execute(
            """
            INSERT INTO provider_events (
                event_id, conversation_id, source_name, event_index, event_type,
                timestamp, sort_key, materializer_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"{conv_record.conversation_id}:provider-event:000000",
                conv_record.conversation_id,
                conv_record.source_name,
                0,
                "compaction",
                conv_record.updated_at,
                0.5,
                PROVIDER_EVENT_MATERIALIZER_VERSION,
            ),
        )
        conn.execute(
            "INSERT INTO provider_event_compactions (event_id, summary) VALUES (?, ?)",
            (f"{conv_record.conversation_id}:provider-event:000000", "old"),
        )
        conn.commit()

    replacement = ProviderEventRecord(
        event_id=ProviderEventId(f"{conv_record.conversation_id}:provider-event:000000"),
        conversation_id=conv_record.conversation_id,
        source_name=conv_record.source_name,
        event_index=0,
        event_type="compaction",
        timestamp=conv_record.updated_at,
        sort_key=0.5,
        payload={"summary": "new"},
        materializer_version=PROVIDER_EVENT_MATERIALIZER_VERSION,
    )
    repository = repository_for_scenario_db(db_path)
    try:
        counts = await repository.save_conversation(
            conv_record,
            msg_records,
            attachment_records,
            provider_events=[replacement],
        )
        # First write with replacement provider events — always written since
        # there is no prior DB entry.
        assert counts["provider_events"] == 1

        refreshed = await repository.get(scenario.resolved_conversation_id)
        assert refreshed is not None
        assert [event.payload for event in refreshed.provider_events] == [{"summary": "new"}]
    finally:
        await repository.close()


# ---------------------------------------------------------------------------
# Hash idempotency tests -- prove that the write path correctly skips on
# hash match and that action events are correctly derived from content.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_hash_match_skips_writes_when_row_graph_agrees(workspace_env: Mapping[str, Path]) -> None:
    """When the stored hash matches the record hash, writes are correctly skipped."""
    scenario = ArchiveScenario(
        name="hash-match-skip",
        provider="chatgpt",
        title="Hash match skip",
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

        # First write stores records with ConversationBuilder's hash.
        await repository.save_conversation(conv_record, msg_records, attachment_records)

        # Re-save identical records -- DB hash == record hash → everything skipped.
        counts = await repository.save_conversation(conv_record, msg_records, attachment_records)
        assert counts["skipped_conversations"] == 1
        assert counts["skipped_messages"] == 2
        assert counts["messages"] == 0
    finally:
        await repository.close()


@pytest.mark.asyncio()
async def test_action_events_are_derived_and_replaced_on_content_change(
    workspace_env: Mapping[str, Path],
) -> None:
    """Action events are derived from content and rebuilt on every content change.

    They are NOT in the content hash -- they are derived from messages +
    content_blocks at write time by build_action_event_records().
    When the hash matches (content unchanged), they are skipped because
    the messages that would produce them haven't changed either.
    """
    from polylogue.storage.sqlite.queries import action_events as action_events_q

    scenario = ArchiveScenario(
        name="action-events-derived",
        provider="claude-code",
        title="Action events derived",
        messages=(
            ScenarioMessage(role="user", text="Run a command", message_id="m1"),
            ScenarioMessage(
                role="assistant",
                text="I will run it",
                message_id="m2",
                content_blocks=(
                    ScenarioContentBlock.tool_use(
                        tool_name="bash",
                        tool_input={"command": "echo hello"},
                    ),
                ),
            ),
        ),
    )
    db_path, _ = seed_workspace_scenarios(workspace_env, [scenario])
    with open_connection(db_path) as conn:
        conv_record, msg_records, attachment_records = scenario.records_from_connection(conn)

    # seed_workspace_scenarios already persisted the scenario into db_path, so
    # exercise save_conversation against a FRESH archive: a genuine first write
    # (which derives action events), not an idempotent re-save of seeded rows.
    fresh_db = workspace_env["data_root"] / "polylogue" / "lifecycle-fresh.db"
    fresh_db.parent.mkdir(parents=True, exist_ok=True)
    repository = repository_for_scenario_db(fresh_db)
    try:
        # First write creates action events from the tool_use content block.
        counts_init = await repository.save_conversation(conv_record, msg_records, attachment_records)
        assert counts_init["messages"] == 2, "Initial write should persist messages"

        async with repository._backend.connection() as aconn:
            initial_events = await action_events_q.get_action_events(aconn, conv_record.conversation_id)
        assert len(initial_events) >= 1, "Expected at least one action event from tool_use"

        # Re-save identical content -- hash matches, action events are skipped.
        counts = await repository.save_conversation(conv_record, msg_records, attachment_records)
        assert counts["skipped_messages"] == 2, "Identical content should be skipped"

        # Now change a message and provide a new content_hash (mirrors what the
        # pipeline does when content changes: the hash is recomputed from the new
        # payload).  The hash change signals content_unchanged=False, so messages
        # and action events are rebuilt.
        modified = [
            msg_records[0],
            msg_records[1].model_copy(update={"text": "Running it differently"}),
        ]
        changed_conv = conv_record.model_copy(update={"content_hash": "b" * 64})
        counts2 = await repository.save_conversation(changed_conv, modified, attachment_records)
        assert counts2["messages"] == 2, "Changed message should trigger full write"

        async with repository._backend.connection() as aconn:
            refreshed_events = await action_events_q.get_action_events(aconn, conv_record.conversation_id)
        assert len(refreshed_events) >= 1, "Action events should still exist after update"
    finally:
        await repository.close()
