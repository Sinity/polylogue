"""Repository lifecycle laws over declarative archive scenarios."""

from __future__ import annotations

import json
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

        # Fix up the UUID hash from ConversationBuilder with a real SHA-256
        # hash so save_via_backend's row-graph validation can function.
        from polylogue.storage.repository.archive.writes.conversations import _compute_hash_from_row_graph

        real_hash = _compute_hash_from_row_graph(
            conversation=conv_record,
            messages=msg_records,
            attachments=attachment_records,
        )
        conv_record = conv_record.model_copy(update={"content_hash": real_hash})

        # First write: the DB still has the seed's UUID hash, so this
        # updates the conversation row with the real SHA-256 hash.
        await repository.save_conversation(conv_record, msg_records, attachment_records)

        # Second write: now DB hash == record hash == row-graph hash, so
        # the idempotency skip path should be clean.
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
                event_id, conversation_id, provider_name, event_index, event_type,
                timestamp, sort_key, payload_json, materializer_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"{conv_record.conversation_id}:provider-event:000000",
                conv_record.conversation_id,
                conv_record.provider_name,
                0,
                "compaction",
                conv_record.updated_at,
                0.5,
                json.dumps({"summary": "compact"}),
                PROVIDER_EVENT_MATERIALIZER_VERSION,
            ),
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
                event_id, conversation_id, provider_name, event_index, event_type,
                timestamp, sort_key, payload_json, materializer_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"{conv_record.conversation_id}:provider-event:000000",
                conv_record.conversation_id,
                conv_record.provider_name,
                0,
                "compaction",
                conv_record.updated_at,
                0.5,
                json.dumps({"summary": "keep"}),
                PROVIDER_EVENT_MATERIALIZER_VERSION,
            ),
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
                event_id, conversation_id, provider_name, event_index, event_type,
                timestamp, sort_key, payload_json, materializer_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"{conv_record.conversation_id}:provider-event:000000",
                conv_record.conversation_id,
                conv_record.provider_name,
                0,
                "compaction",
                conv_record.updated_at,
                0.5,
                json.dumps({"summary": "old"}),
                PROVIDER_EVENT_MATERIALIZER_VERSION,
            ),
        )
        conn.commit()

    replacement = ProviderEventRecord(
        event_id=ProviderEventId(f"{conv_record.conversation_id}:provider-event:000000"),
        conversation_id=conv_record.conversation_id,
        provider_name=conv_record.provider_name,
        event_index=0,
        event_type="compaction",
        timestamp=conv_record.updated_at,
        sort_key=0.5,
        payload={"summary": "new"},
        materializer_version=PROVIDER_EVENT_MATERIALIZER_VERSION,
    )
    repository = repository_for_scenario_db(db_path)
    try:
        from polylogue.storage.repository.archive.writes.conversations import _compute_hash_from_row_graph

        # Fix up the UUID hash from ConversationBuilder with a real SHA-256
        # hash so that save_via_backend's row-graph validation works correctly.
        real_hash_pe = _compute_hash_from_row_graph(
            conversation=conv_record,
            messages=msg_records,
            attachments=attachment_records,
        )
        conv_record = conv_record.model_copy(update={"content_hash": real_hash_pe})

        counts = await repository.save_conversation(
            conv_record,
            msg_records,
            attachment_records,
            provider_events=[replacement],
        )
        # The row-graph hash (with replacement provider events) does not match
        # the record hash (computed without provider events).  The validation
        # detects the mismatch and forces a full write, which includes the
        # replacement provider events.
        assert counts["provider_events"] == 1

        refreshed = await repository.get(scenario.resolved_conversation_id)
        assert refreshed is not None
        assert [event.payload for event in refreshed.provider_events] == [{"summary": "new"}]
    finally:
        await repository.close()


# ---------------------------------------------------------------------------
# Hash honesty tests -- prove that the write path validates row-graph / hash
# consistency and that action events are correctly derived from content.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_compute_hash_from_row_graph_matches_domain_hash(workspace_env: Mapping[str, Path]) -> None:
    """_compute_hash_from_row_graph produces the same hash as the canonical domain path."""
    from polylogue.storage.hydrators import conversation_from_records
    from polylogue.storage.repository.archive.writes.conversations import (
        _compute_hash_from_row_graph,
        _content_hash_from_metadata_or_domain,
    )

    scenario = ArchiveScenario(
        name="hash-roundtrip",
        provider="claude-code",
        title="Hash roundtrip",
        messages=(
            ScenarioMessage(role="user", text="Roundtrip question", message_id="m1"),
            ScenarioMessage(role="assistant", text="Roundtrip answer", message_id="m2"),
        ),
    )
    db_path, _ = seed_workspace_scenarios(workspace_env, [scenario])
    with open_connection(db_path) as conn:
        conv_record, msg_records, attachment_records = scenario.records_from_connection(conn)

    domain = conversation_from_records(conv_record, msg_records, attachment_records)
    domain_hash = _content_hash_from_metadata_or_domain(domain, metadata={})
    row_graph_hash = _compute_hash_from_row_graph(
        conversation=conv_record,
        messages=msg_records,
        attachments=attachment_records,
    )
    assert domain_hash == row_graph_hash, f"domain={domain_hash[:16]}... row_graph={row_graph_hash[:16]}..."


@pytest.mark.asyncio()
async def test_hash_consistency_forces_write_on_stale_hash(workspace_env: Mapping[str, Path]) -> None:
    """When the row graph diverges from the record hash, the write path must not skip."""
    from polylogue.storage.repository.archive.writes.conversations import _compute_hash_from_row_graph

    scenario = ArchiveScenario(
        name="stale-hash-fix",
        provider="chatgpt",
        title="Stale hash",
        messages=(
            ScenarioMessage(role="user", text="Original text", message_id="m1"),
            ScenarioMessage(role="assistant", text="Original response", message_id="m2"),
        ),
    )
    db_path, _ = seed_workspace_scenarios(workspace_env, [scenario])
    repository = repository_for_scenario_db(db_path)
    try:
        with open_connection(db_path) as conn:
            conv_record, msg_records, attachment_records = scenario.records_from_connection(conn)

        # Compute the real content hash so the validation sees a consistent
        # row-graph-to-hash mapping.  ConversationBuilder generates UUID hashes.
        real_hash = _compute_hash_from_row_graph(
            conversation=conv_record,
            messages=msg_records,
            attachments=attachment_records,
        )
        conv_record = conv_record.model_copy(update={"content_hash": real_hash})

        # First write: hash won't match DB (or no prior entry) so full write.
        counts_init = await repository.save_conversation(conv_record, msg_records, attachment_records)
        assert counts_init["messages"] == 2

        # Re-save identical records -- hash + row graph both agree -> skip.
        counts_skip = await repository.save_conversation(conv_record, msg_records, attachment_records)
        assert counts_skip["skipped_messages"] == 2

        # Modify a message but keep the OLD (now stale) content_hash.
        modified_messages = [
            msg_records[0].model_copy(update={"text": "Changed text"}),
            msg_records[1],
        ]
        counts = await repository.save_conversation(
            conv_record,  # still has old content_hash
            modified_messages,
            attachment_records,
        )
        # Validation detected the mismatch and forced a full write.
        assert counts["messages"] == 2, f"Expected 2 messages written, got {counts}"
        assert counts["skipped_messages"] == 0

        refreshed = await repository.get(scenario.resolved_conversation_id)
        assert refreshed is not None
        texts = [msg.text for msg in refreshed.messages]
        assert "Changed text" in texts
    finally:
        await repository.close()


@pytest.mark.asyncio()
async def test_hash_match_skips_writes_when_row_graph_agrees(workspace_env: Mapping[str, Path]) -> None:
    """When both the stored hash and the row graph agree, writes are correctly skipped."""
    from polylogue.storage.repository.archive.writes.conversations import _compute_hash_from_row_graph

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

        # Compute the real content hash and set it.
        real_hash = _compute_hash_from_row_graph(
            conversation=conv_record,
            messages=msg_records,
            attachments=attachment_records,
        )
        conv_record = conv_record.model_copy(update={"content_hash": real_hash})

        # First write stores records with the correct hash.
        await repository.save_conversation(conv_record, msg_records, attachment_records)

        # Re-save identical records -- everything should be skipped.
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
    from polylogue.storage.repository.archive.writes.conversations import _compute_hash_from_row_graph
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
    repository = repository_for_scenario_db(db_path)
    try:
        with open_connection(db_path) as conn:
            conv_record, msg_records, attachment_records = scenario.records_from_connection(conn)

        # Compute the real content hash.
        real_hash = _compute_hash_from_row_graph(
            conversation=conv_record,
            messages=msg_records,
            attachments=attachment_records,
        )
        conv_record = conv_record.model_copy(update={"content_hash": real_hash})

        # First write creates action events from the tool_use content block.
        counts_init = await repository.save_conversation(conv_record, msg_records, attachment_records)
        assert counts_init["messages"] == 2, "Initial write should persist messages"

        async with repository._backend.connection() as aconn:
            initial_events = await action_events_q.get_action_events(aconn, conv_record.conversation_id)
        assert len(initial_events) >= 1, "Expected at least one action event from tool_use"

        # Re-save identical content -- hash matches, action events are skipped.
        counts = await repository.save_conversation(conv_record, msg_records, attachment_records)
        assert counts["skipped_messages"] == 2, "Identical content should be skipped"

        # Now change a message -- action events must be rebuilt.
        modified = [
            msg_records[0],
            msg_records[1].model_copy(update={"text": "Running it differently"}),
        ]
        counts2 = await repository.save_conversation(conv_record, modified, attachment_records)
        assert counts2["messages"] == 2, "Changed message should trigger full write"

        async with repository._backend.connection() as aconn:
            refreshed_events = await action_events_q.get_action_events(aconn, conv_record.conversation_id)
        assert len(refreshed_events) >= 1, "Action events should still exist after update"
    finally:
        await repository.close()
