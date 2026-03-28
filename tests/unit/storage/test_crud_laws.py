"""Law-based contracts for SQLiteBackend CRUD behavior."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from tests.infra.helpers import make_conversation
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
