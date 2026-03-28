"""Law-based contracts for SQLiteBackend CRUD behavior."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.store import ContentBlockRecord, ConversationRecord, MessageRecord
from tests.infra.storage_records import make_attachment, make_conversation, make_message
from tests.infra.strategies import (
    conversation_graph_strategy,
    expected_sorted_ids,
    expected_tag_counts,
    literal_title_search_strategy,
    seed_conversation_graph,
    shortest_unique_prefix,
    tag_assignment_strategy,
)


def _seed_backend(specs) -> tuple[TemporaryDirectory[str], SQLiteBackend]:
    tempdir = TemporaryDirectory()
    db_path = Path(tempdir.name) / "backend.db"
    seed_conversation_graph(db_path, specs)
    return tempdir, SQLiteBackend(db_path=db_path)


def _new_backend(name: str) -> tuple[TemporaryDirectory[str], SQLiteBackend]:
    tempdir = TemporaryDirectory()
    return tempdir, SQLiteBackend(db_path=Path(tempdir.name) / name)


@settings(
    deadline=None,
    max_examples=25,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(conversation_graph_strategy(), st.integers(min_value=1, max_value=4), st.integers(min_value=0, max_value=5))
async def test_backend_list_windows_are_ordered_slices(specs, limit: int, offset: int) -> None:
    """list_conversations() paging must be a slice of the unbounded ordered result."""
    tempdir, backend = _seed_backend(specs)
    try:
        full_ids = expected_sorted_ids(specs)
        page = await backend.list_conversations(limit=limit, offset=offset)
        suffix = await backend.list_conversations(offset=offset)

        assert [record.conversation_id for record in page] == full_ids[offset:offset + limit]
        assert [record.conversation_id for record in suffix] == full_ids[offset:]
    finally:
        await backend.close()
        tempdir.cleanup()


@settings(
    deadline=None,
    max_examples=25,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(conversation_graph_strategy())
async def test_backend_provider_filters_match_expected_subset(specs) -> None:
    """Provider filtering must preserve count/order agreement at the backend layer."""
    tempdir, backend = _seed_backend(specs)
    try:
        for provider in sorted({spec.provider for spec in specs}):
            expected_ids = expected_sorted_ids(tuple(spec for spec in specs if spec.provider == provider))
            records = await backend.list_conversations(provider=provider)
            count = await backend.count_conversations(provider=provider)

            assert [record.conversation_id for record in records] == expected_ids
            assert count == len(expected_ids)
    finally:
        await backend.close()
        tempdir.cleanup()


@settings(max_examples=20, deadline=None)
@given(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789 -_", min_size=1, max_size=30).filter(lambda value: value.strip() != ""),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789 -_", min_size=1, max_size=30).filter(lambda value: value.strip() != ""),
)
async def test_backend_upsert_keeps_one_visible_record(initial_title: str, updated_title: str) -> None:
    """Saving the same conversation ID twice must update in place, not duplicate."""
    tempdir = TemporaryDirectory()
    backend = SQLiteBackend(db_path=Path(tempdir.name) / "upsert.db")
    try:
        await backend.save_conversation_record(make_conversation("conv-upsert", title=initial_title))
        await backend.save_conversation_record(make_conversation("conv-upsert", title=updated_title))

        retrieved = await backend.get_conversation("conv-upsert")
        listed = await backend.list_conversations()

        assert retrieved is not None
        assert retrieved.title == updated_title
        assert [record.conversation_id for record in listed] == ["conv-upsert"]
    finally:
        await backend.close()
        tempdir.cleanup()


@settings(max_examples=20, deadline=None)
@given(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_", min_size=3, max_size=12),
    st.sampled_from(("claude", "chatgpt", "codex", "claude-code")),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_", min_size=1, max_size=30).filter(
        lambda value: value.strip() != ""
    ),
)
async def test_backend_roundtrip_and_missing_lookup_contract(
    conversation_id: str,
    provider: str,
    title: str,
) -> None:
    """Saving one record must make it visible while unrelated IDs stay missing."""
    tempdir = TemporaryDirectory()
    backend = SQLiteBackend(db_path=Path(tempdir.name) / "roundtrip.db")
    try:
        full_id = f"conv-{conversation_id}"
        assert await backend.get_conversation("missing-conversation") is None

        await backend.save_conversation_record(
            make_conversation(full_id, provider_name=provider, title=title)
        )

        retrieved = await backend.get_conversation(full_id)
        assert retrieved is not None
        assert retrieved.conversation_id == full_id
        assert retrieved.provider_name == provider
        assert retrieved.title == title
    finally:
        await backend.close()
        tempdir.cleanup()


@settings(max_examples=30, deadline=None)
@given(literal_title_search_strategy())
async def test_backend_title_search_treats_wildcards_as_literals(case) -> None:
    """Wildcard-sensitive characters in title queries must be matched literally."""
    tempdir = TemporaryDirectory()
    backend = SQLiteBackend(db_path=Path(tempdir.name) / "title-search.db")
    try:
        await backend.save_conversation_record(
            make_conversation("conv-match", title=case.matching_title, provider_name="test")
        )
        await backend.save_conversation_record(
            make_conversation("conv-decoy", title=case.decoy_title, provider_name="test")
        )

        results = await backend.list_conversations(title_contains=case.needle)

        assert [record.conversation_id for record in results] == ["conv-match"]
        assert results[0].title == case.matching_title
    finally:
        await backend.close()
        tempdir.cleanup()


@settings(
    deadline=None,
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(tag_assignment_strategy())
async def test_backend_list_tags_matches_generated_tag_distribution(spec) -> None:
    """Tag counts must dedupe per conversation and respect provider filters."""
    tempdir, backend = _seed_backend(spec.conversations)
    try:
        for conversation, tag_sequence in zip(spec.conversations, spec.tag_sequences, strict=True):
            for tag in tag_sequence:
                await backend.add_tag(conversation.conversation_id, tag)

        assert await backend.list_tags() == expected_tag_counts(spec)
        for provider in sorted({conversation.provider for conversation in spec.conversations}):
            assert await backend.list_tags(provider=provider) == expected_tag_counts(spec, provider=provider)
    finally:
        await backend.close()
        tempdir.cleanup()


@settings(
    deadline=None,
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(conversation_graph_strategy(min_conversations=2, max_conversations=5), st.integers(min_value=0, max_value=4))
async def test_backend_resolve_id_obeys_exact_unique_and_ambiguous_prefixes(specs, candidate_index: int) -> None:
    """resolve_id() must accept exact IDs, shortest unique prefixes, and reject shared prefixes."""
    tempdir, backend = _seed_backend(specs)
    try:
        index = candidate_index % len(specs)
        target_id = specs[index].conversation_id
        ids = tuple(spec.conversation_id for spec in specs)

        assert await backend.resolve_id(target_id) == target_id
        assert await backend.resolve_id(shortest_unique_prefix(ids, target_id)) == target_id
        assert await backend.resolve_id("conv-") is None
    finally:
        await backend.close()
        tempdir.cleanup()


@settings(
    deadline=None,
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(conversation_graph_strategy(min_conversations=1, max_conversations=5), st.integers(min_value=0, max_value=4))
async def test_backend_delete_is_persistent(specs, candidate_index: int) -> None:
    """Deleting an existing conversation removes it exactly once."""
    tempdir, backend = _seed_backend(specs)
    try:
        index = candidate_index % len(specs)
        target_id = specs[index].conversation_id

        assert await backend.delete_conversation(target_id) is True
        assert await backend.get_conversation(target_id) is None
        assert await backend.delete_conversation(target_id) is False
    finally:
        await backend.close()
        tempdir.cleanup()


async def test_backend_get_conversations_batch_contract() -> None:
    """Batch conversation lookup must preserve requested order, duplicates, and skip missing IDs."""
    tempdir, backend = _new_backend("conversation-batch.db")
    try:
        await backend.save_conversation_record(make_conversation("conv-1", title="First"))
        await backend.save_conversation_record(make_conversation("conv-2", title="Second"))

        assert await backend.get_conversations_batch([]) == []

        batch = await backend.get_conversations_batch(["conv-2", "missing", "conv-1", "conv-1"])
        assert [record.conversation_id for record in batch] == ["conv-2", "conv-1", "conv-1"]
        assert [record.title for record in batch] == ["Second", "First", "First"]
    finally:
        await backend.close()
        tempdir.cleanup()


async def test_backend_conversation_roundtrip_preserves_nulls_branching_and_metadata_contract() -> None:
    """Conversation save/get must preserve null optionals, branch links, and user metadata on upsert."""
    tempdir, backend = _new_backend("conversation-roundtrip.db")
    try:
        await backend.save_conversation_record(
            ConversationRecord(
                conversation_id="conv-root",
                provider_name="claude",
                provider_conversation_id="prov-root",
                title=None,
                created_at=None,
                updated_at=None,
                content_hash="hash-root",
                provider_meta=None,
                version=1,
            )
        )
        await backend.save_conversation_record(
            make_conversation(
                "conv-child",
                provider_name="claude",
                title="Child",
                parent_conversation_id="conv-root",
                branch_type="continuation",
                content_hash="hash-child-v1",
            )
        )
        await backend.update_metadata("conv-child", "custom_key", "custom_value")
        await backend.save_conversation_record(
            make_conversation(
                "conv-child",
                provider_name="claude",
                title="Child Updated",
                parent_conversation_id="conv-root",
                branch_type="continuation",
                content_hash="hash-child-v2",
                metadata=None,
            )
        )

        root = await backend.get_conversation("conv-root")
        child = await backend.get_conversation("conv-child")

        assert root is not None
        assert root.title is None
        assert root.created_at is None
        assert root.provider_meta is None

        assert child is not None
        assert child.title == "Child Updated"
        assert child.parent_conversation_id == "conv-root"
        assert child.branch_type == "continuation"
        metadata = await backend.get_metadata("conv-child")
        assert metadata["custom_key"] == "custom_value"
    finally:
        await backend.close()
        tempdir.cleanup()


async def test_backend_message_roundtrip_contract() -> None:
    """Message save/get must preserve ordering, text, and branching fields."""
    tempdir, backend = _new_backend("message-roundtrip.db")
    try:
        await backend.save_conversation_record(make_conversation("conv-1", title="Messages"))
        parent = MessageRecord(
            message_id="msg-parent",
            conversation_id="conv-1",
            role="user",
            text="Parent",
            sort_key=1.0,
            content_hash="hash-parent",
            version=1,
        )
        child = MessageRecord(
            message_id="msg-child",
            conversation_id="conv-1",
            role="assistant",
            text="Child",
            sort_key=2.0,
            content_hash="hash-child",
            version=1,
            parent_message_id="msg-parent",
            branch_index=2,
        )
        await backend.save_messages([child, parent])

        assert await backend.get_messages("missing") == []

        messages = await backend.get_messages("conv-1")
        assert [message.message_id for message in messages] == ["msg-parent", "msg-child"]
        assert messages[0].text == "Parent"
        assert messages[1].text == "Child"
        assert messages[1].parent_message_id == "msg-parent"
        assert messages[1].branch_index == 2
    finally:
        await backend.close()
        tempdir.cleanup()


async def test_backend_get_messages_batch_contract() -> None:
    """Batch message lookup must return requested keys with sorted groups and empty misses."""
    tempdir, backend = _new_backend("messages-batch.db")
    try:
        await backend.save_conversation_record(make_conversation("conv-1"))
        await backend.save_conversation_record(make_conversation("conv-2"))
        await backend.save_messages(
            [
                MessageRecord(
                    message_id="m3",
                    conversation_id="conv-1",
                    role="assistant",
                    text="third",
                    sort_key=3.0,
                    content_hash="hash-3",
                    version=1,
                ),
                MessageRecord(
                    message_id="m1",
                    conversation_id="conv-1",
                    role="user",
                    text="first",
                    sort_key=1.0,
                    content_hash="hash-1",
                    version=1,
                ),
                MessageRecord(
                    message_id="m2",
                    conversation_id="conv-1",
                    role="assistant",
                    text="second",
                    sort_key=2.0,
                    content_hash="hash-2",
                    version=1,
                ),
                MessageRecord(
                    message_id="m4",
                    conversation_id="conv-2",
                    role="user",
                    text="other",
                    sort_key=4.0,
                    content_hash="hash-4",
                    version=1,
                ),
            ]
        )

        assert await backend.get_messages_batch([]) == {}

        result = await backend.get_messages_batch(["missing-1", "conv-1", "conv-2", "missing-2"])
        assert list(result) == ["missing-1", "conv-1", "conv-2", "missing-2"]
        assert result["missing-1"] == []
        assert result["missing-2"] == []
        assert [message.message_id for message in result["conv-1"]] == ["m1", "m2", "m3"]
        assert [message.message_id for message in result["conv-2"]] == ["m4"]
    finally:
        await backend.close()
        tempdir.cleanup()


async def test_backend_get_attachments_batch_contract() -> None:
    """Batch attachment lookup must preserve fields and return empty groups for missing IDs."""
    tempdir, backend = _new_backend("attachments-batch.db")
    try:
        await backend.save_conversation_record(make_conversation("conv-1"))
        await backend.save_conversation_record(make_conversation("conv-2"))
        await backend.save_messages([make_message("m1", "conv-1"), make_message("m2", "conv-2")])
        await backend.save_attachments(
            [
                make_attachment(
                    "att-1",
                    "conv-1",
                    "m1",
                    mime_type="application/pdf",
                    size_bytes=2048,
                    path="/tmp/test.pdf",
                ),
                make_attachment("att-2", "conv-2", "m2", mime_type="image/png", size_bytes=512),
            ]
        )

        assert await backend.get_attachments_batch([]) == {}

        result = await backend.get_attachments_batch(["missing", "conv-1", "conv-2"])
        assert list(result) == ["missing", "conv-1", "conv-2"]
        assert result["missing"] == []
        assert [attachment.attachment_id for attachment in result["conv-1"]] == ["att-1"]
        assert [attachment.attachment_id for attachment in result["conv-2"]] == ["att-2"]
        assert result["conv-1"][0].mime_type == "application/pdf"
        assert result["conv-1"][0].size_bytes == 2048
        assert result["conv-1"][0].path == "/tmp/test.pdf"
        assert result["conv-1"][0].message_id == "m1"
    finally:
        await backend.close()
        tempdir.cleanup()


async def test_backend_metadata_and_tag_contract() -> None:
    """Metadata and tag mutations must compose consistently through the backend API."""
    tempdir, backend = _new_backend("metadata-tags.db")
    try:
        await backend.save_conversation_record(make_conversation("conv-1"))

        assert await backend.list_tags() == {}

        await backend.update_metadata("conv-1", "rating", 5)
        await backend.update_metadata("conv-1", "reviewed", True)
        await backend.update_metadata("conv-1", "temp", "value")
        await backend.delete_metadata("conv-1", "temp")
        await backend.add_tag("conv-1", "important")
        await backend.add_tag("conv-1", "work")
        await backend.remove_tag("conv-1", "work")

        metadata = await backend.get_metadata("conv-1")
        assert metadata["rating"] == 5
        assert metadata["reviewed"] is True
        assert "temp" not in metadata
        assert metadata["tags"] == ["important"]
        assert await backend.list_tags() == {"important": 1}
    finally:
        await backend.close()
        tempdir.cleanup()


async def test_backend_delete_reparents_children_contract() -> None:
    """Deleting an interior conversation must reparent its descendants to preserve the tree."""
    tempdir, backend = _new_backend("delete-reparent.db")
    try:
        await backend.save_conversation_record(make_conversation("root"))
        await backend.save_conversation_record(make_conversation("child", parent_conversation_id="root"))
        await backend.save_conversation_record(make_conversation("grandchild", parent_conversation_id="child"))

        assert await backend.delete_conversation("child") is True

        assert await backend.get_conversation("child") is None
        grandchild = await backend.get_conversation("grandchild")
        assert grandchild is not None
        assert grandchild.parent_conversation_id == "root"
    finally:
        await backend.close()
        tempdir.cleanup()


async def test_backend_prune_attachments_contract() -> None:
    """Pruning must keep requested refs, drop local-only refs, and preserve shared attachments."""
    tempdir, backend = _new_backend("prune-attachments.db")
    try:
        await backend.save_conversation_record(make_conversation("conv-1"))
        await backend.save_conversation_record(make_conversation("conv-2"))
        await backend.save_messages([make_message("m1", "conv-1"), make_message("m2", "conv-2")])
        await backend.save_attachments(
            [
                make_attachment("att-1", "conv-1", "m1"),
                make_attachment("att-2", "conv-1", "m1"),
                make_attachment("shared-att", "conv-1", "m1"),
                make_attachment("shared-att", "conv-2", "m2"),
            ]
        )

        await backend.prune_attachments("conv-1", {"att-1"})
        conv1_ids = {attachment.attachment_id for attachment in await backend.get_attachments("conv-1")}
        conv2_ids = {attachment.attachment_id for attachment in await backend.get_attachments("conv-2")}
        assert conv1_ids == {"att-1"}
        assert conv2_ids == {"shared-att"}

        async with backend.connection() as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM attachments WHERE attachment_id = 'shared-att'")
            shared_count = (await cursor.fetchone())[0]
            assert shared_count == 1

        await backend.prune_attachments("conv-1", set())
        assert await backend.get_attachments("conv-1") == []
    finally:
        await backend.close()
        tempdir.cleanup()


async def test_backend_nested_transaction_and_control_contract() -> None:
    """Transaction controls must enforce active state and nested savepoints correctly."""
    tempdir, backend = _new_backend("transactions.db")
    try:
        with pytest.raises(Exception, match="No active transaction to commit"):
            await backend.commit()
        with pytest.raises(Exception, match="No active transaction to rollback"):
            await backend.rollback()

        await backend.begin()
        await backend.save_conversation_record(make_conversation("conv-1", title="First"))
        await backend.begin()
        await backend.save_conversation_record(make_conversation("conv-2", title="Second"))
        await backend.rollback()
        await backend.commit()

        assert await backend.get_conversation("conv-1") is not None
        assert await backend.get_conversation("conv-2") is None
    finally:
        await backend.close()
        tempdir.cleanup()


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        ([], {"total_messages": 0, "dialogue_messages": 0, "tool_messages": 0}),
        (
            [
                make_message("m1", "conv-1", role="user", text="hello", sort_key=1.0, content_hash="hash-1"),
                make_message("m2", "conv-1", role="assistant", text="hi", sort_key=2.0, content_hash="hash-2"),
                make_message("m3", "conv-1", role="tool", text="output", sort_key=3.0, content_hash="hash-3"),
            ],
            {"total_messages": 3, "dialogue_messages": 2, "tool_messages": 1},
        ),
    ],
    ids=["empty", "with-tool"],
)
async def test_backend_get_conversation_stats_contract(messages, expected) -> None:
    """Conversation stats must count total, dialogue, and tool messages consistently."""
    tempdir, backend = _new_backend("conversation-stats.db")
    try:
        await backend.save_conversation_record(make_conversation("conv-1"))
        if messages:
            await backend.save_messages(messages)

        assert await backend.get_conversation_stats("conv-1") == expected
    finally:
        await backend.close()
        tempdir.cleanup()


async def test_backend_save_conversation_counts_contract() -> None:
    """save_conversation() must report created counts for message and attachment payloads."""
    tempdir, backend = _new_backend("save-conversation.db")
    try:
        conversation = make_conversation("conv-1", content_hash="hash-1")
        messages = [
            make_message("m1", "conv-1", text="one", content_hash="msg-hash-1"),
            make_message("m2", "conv-1", role="assistant", text="two", content_hash="msg-hash-2"),
        ]
        attachments = [
            make_attachment("att-1", "conv-1", "m1"),
            make_attachment("att-2", "conv-1", "m1"),
        ]

        counts = await backend.save_conversation(conversation, messages, attachments)
        assert counts == {
            "conversations_created": 1,
            "messages_created": 2,
            "attachments_created": 2,
        }

        empty_counts = await backend.save_conversation(
            make_conversation("conv-2", content_hash="hash-2"),
            [],
            [],
        )
        assert empty_counts == {
            "conversations_created": 1,
            "messages_created": 0,
            "attachments_created": 0,
        }
    finally:
        await backend.close()
        tempdir.cleanup()


async def test_backend_content_blocks_roundtrip_preserves_semantic_type() -> None:
    """Content blocks loaded through the backend must preserve semantic_type and ordering."""
    tempdir, backend = _new_backend("content-blocks.db")
    try:
        await backend.save_conversation_record(make_conversation("conv-sem"))
        await backend.save_messages([make_message("msg-sem", "conv-sem", role="assistant", text="")])
        await backend.save_content_blocks(
            [
                ContentBlockRecord(
                    block_id="block-1",
                    message_id="msg-sem",
                    conversation_id="conv-sem",
                    block_index=0,
                    type="tool_use",
                    tool_name="Bash",
                    semantic_type="git",
                    metadata='{"command": "status"}',
                ),
                ContentBlockRecord(
                    block_id="block-2",
                    message_id="msg-sem",
                    conversation_id="conv-sem",
                    block_index=1,
                    type="thinking",
                    text="Let me think",
                    semantic_type="thinking",
                ),
                ContentBlockRecord(
                    block_id="block-3",
                    message_id="msg-sem",
                    conversation_id="conv-sem",
                    block_index=2,
                    type="text",
                    text="plain text",
                    semantic_type=None,
                ),
            ]
        )

        blocks = (await backend.get_content_blocks(["msg-sem"]))["msg-sem"]
        assert [block.block_index for block in blocks] == [0, 1, 2]
        assert [block.semantic_type for block in blocks] == ["git", "thinking", None]
    finally:
        await backend.close()
        tempdir.cleanup()
