"""Reusable semantic oracles for archive verification tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from tests.infra.semantic_facts import (
    ArchiveFacts,
    SessionFacts,
    assert_same_archive_facts,
    assert_same_session_facts,
)


def assert_session_surfaces_agree(*facts: SessionFacts) -> None:
    """Assert record, hydrated, repository, and surface facts agree."""
    assert_same_session_facts(*facts)


def assert_archive_surfaces_agree(*facts: ArchiveFacts) -> None:
    """Assert aggregate archive facts agree across surfaces."""
    assert_same_archive_facts(*facts)


def assert_provider_partition_exhaustive(
    *,
    all_session_ids: Iterable[str],
    ids_by_provider: Mapping[str, Iterable[str]],
) -> None:
    """Assert provider buckets partition the archive without gaps or overlap."""
    expected = set(all_session_ids)
    seen: set[str] = set()
    for provider, provider_ids in ids_by_provider.items():
        ids = set(provider_ids)
        overlap = seen & ids
        assert not overlap, f"Provider bucket {provider!r} overlaps earlier buckets: {sorted(overlap)}"
        seen.update(ids)
    assert seen == expected, f"Provider partition mismatch: expected={sorted(expected)} seen={sorted(seen)}"


__all__ = [
    "assert_archive_surfaces_agree",
    "assert_session_surfaces_agree",
    "assert_provider_partition_exhaustive",
]
