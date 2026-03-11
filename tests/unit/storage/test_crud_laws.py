"""Law-based contracts for SQLiteBackend CRUD behavior."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from tests.infra.helpers import make_conversation
from tests.infra.strategies import conversation_graph_strategy, expected_sorted_ids, seed_conversation_graph


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
