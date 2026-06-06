"""Filter composition laws: prove filter chains preserve monotonicity and subset properties.

Adding more filters can only narrow the result set, never widen it.
Composing independent filters is commutative. These are algebraic
properties of the filter DSL, exercised directly through the
:class:`~polylogue.archive.filter.filters.SessionFilter` over a
archive ``ArchiveStore`` (no archive repository or raw-SQL column access).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.filter.filters import SessionFilter
from tests.infra.storage_records import SessionBuilder, db_setup


@pytest.fixture()
def filterable_root(workspace_env: dict[str, Path]) -> Path:
    """Create a rich archive with varied sessions for filter laws.

    Returns the archive root the native ``SessionFilter`` reads from.
    """
    db_path = db_setup(workspace_env)

    SessionBuilder(db_path, "gpt-long").provider("chatgpt").title("Long GPT chat").add_message(
        role="user", text="First question about testing"
    ).add_message(role="assistant", text="Testing is important for correctness").add_message(
        role="user", text="Tell me more about property testing"
    ).add_message(role="assistant", text="Property testing uses random inputs to verify invariants").save()

    SessionBuilder(db_path, "gpt-short").provider("chatgpt").title("Quick question").add_message(
        role="user", text="Hello"
    ).save()

    SessionBuilder(db_path, "claude-mid").provider("claude-code").title("Refactoring session").add_message(
        role="user", text="Refactor the module"
    ).add_message(role="assistant", text="Done with refactoring").save()

    SessionBuilder(db_path, "codex-mid").provider("codex").title("Code generation").add_message(
        role="user", text="Generate authentication code"
    ).add_message(role="assistant", text="Here is the auth implementation").save()

    return db_path.parent


async def _ids(flt: SessionFilter) -> set[str]:
    return {str(c.id) for c in await flt.list()}


def _cf(root: Path) -> SessionFilter:
    return SessionFilter(archive_root=root)


class TestFilterMonotonicity:
    """Adding a filter can only narrow or maintain the result set."""

    @pytest.mark.asyncio
    async def test_provider_filter_narrows(self, filterable_root: Path) -> None:
        all_ids = await _ids(_cf(filterable_root))
        chatgpt_ids = await _ids(_cf(filterable_root).origin("chatgpt-export"))

        assert chatgpt_ids.issubset(all_ids)
        assert len(chatgpt_ids) < len(all_ids)

    @pytest.mark.asyncio
    async def test_combined_filters_narrow_further(self, filterable_root: Path) -> None:
        all_ids = await _ids(_cf(filterable_root))
        chatgpt_ids = await _ids(_cf(filterable_root).origin("chatgpt-export"))
        chatgpt_with_msgs = await _ids(_cf(filterable_root).origin("chatgpt-export").min_messages(2))

        assert chatgpt_with_msgs.issubset(chatgpt_ids)
        assert chatgpt_ids.issubset(all_ids)

    @pytest.mark.asyncio
    async def test_empty_provider_returns_empty(self, filterable_root: Path) -> None:
        nonexistent = await _ids(_cf(filterable_root).origin("nonexistent-provider"))
        assert nonexistent == set()


class TestFilterCommutativity:
    """Independent filters commute — order of application doesn't matter."""

    @pytest.mark.asyncio
    async def test_provider_then_message_count_equals_reverse(self, filterable_root: Path) -> None:
        provider_then_msgs = await _ids(_cf(filterable_root).origin("chatgpt-export").min_messages(2))
        msgs_then_provider = await _ids(_cf(filterable_root).min_messages(2).origin("chatgpt-export"))
        assert provider_then_msgs == msgs_then_provider


class TestFilterIdempotence:
    """Applying the same filter twice yields the same result."""

    @pytest.mark.asyncio
    async def test_double_provider_filter_is_idempotent(self, filterable_root: Path) -> None:
        once = await _ids(_cf(filterable_root).origin("chatgpt-export"))
        twice = await _ids(_cf(filterable_root).origin("chatgpt-export").origin("chatgpt-export"))
        assert once == twice


class TestFilterPartition:
    """Provider filters partition the session space."""

    @pytest.mark.asyncio
    async def test_providers_partition_total(self, filterable_root: Path) -> None:
        all_convs = await _cf(filterable_root).list()
        all_ids = {str(c.id) for c in all_convs}
        providers = {str(c.origin) for c in all_convs}

        union: set[str] = set()
        for p in providers:
            union |= await _ids(_cf(filterable_root).origin(p))

        assert union == all_ids

    @pytest.mark.asyncio
    async def test_provider_partitions_are_disjoint(self, filterable_root: Path) -> None:
        all_convs = await _cf(filterable_root).list()
        providers = sorted({str(c.origin) for c in all_convs})

        for i, p1 in enumerate(providers):
            for p2 in providers[i + 1 :]:
                ids1 = await _ids(_cf(filterable_root).origin(p1))
                ids2 = await _ids(_cf(filterable_root).origin(p2))
                assert ids1.isdisjoint(ids2), f"{p1} and {p2} overlap"
