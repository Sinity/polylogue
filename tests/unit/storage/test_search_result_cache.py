"""Functional laws for the daemon's bounded read-result cache."""

from __future__ import annotations

from pathlib import Path

from polylogue.storage.search import cache


def test_result_cache_is_payload_and_generation_scoped(tmp_path: Path) -> None:
    cache.invalidate_search_cache()
    result = {"items": [{"id": "codex:one"}], "total": 1}

    cache.put_cached_result(
        "cli.query",
        {"limit": 1},
        result,
        archive_root=tmp_path,
        generation="generation-a",
    )

    hit = cache.get_cached_result(
        "cli.query",
        {"limit": 1},
        archive_root=tmp_path,
        generation="generation-a",
    )
    assert hit == result
    assert hit is not result

    # A caller cannot mutate the resident value, and neither a changed query
    # nor a promoted active generation can reuse it.
    assert isinstance(hit, dict)
    hit["total"] = 99
    assert (
        cache.get_cached_result(
            "cli.query",
            {"limit": 1},
            archive_root=tmp_path,
            generation="generation-a",
        )
        == result
    )
    assert (
        cache.get_cached_result(
            "cli.query",
            {"limit": 2},
            archive_root=tmp_path,
            generation="generation-a",
        )
        is None
    )
    assert (
        cache.get_cached_result(
            "cli.query",
            {"limit": 1},
            archive_root=tmp_path,
            generation="generation-b",
        )
        is None
    )


def test_result_cache_invalidation_is_a_hard_freshness_boundary(tmp_path: Path) -> None:
    cache.invalidate_search_cache()
    cache.put_cached_result(
        "facets",
        {"include_deferred": True},
        {"total_sessions": 1},
        archive_root=tmp_path,
        generation="generation-a",
    )
    before = cache.get_cache_stats()
    cache.invalidate_search_cache()
    after = cache.get_cache_stats()

    assert after["cache_version"] == before["cache_version"] + 1
    assert after["result_cache_entries"] == 0
    assert (
        cache.get_cached_result(
            "facets",
            {"include_deferred": True},
            archive_root=tmp_path,
            generation="generation-a",
        )
        is None
    )


def test_result_cache_evicts_oldest_entry_at_bounded_capacity(tmp_path: Path) -> None:
    cache.invalidate_search_cache()
    for number in range(cache.RESULT_CACHE_MAX_ENTRIES + 1):
        cache.put_cached_result(
            "cli.query",
            {"offset": number},
            {"offset": number},
            archive_root=tmp_path,
            generation="generation-a",
        )

    stats = cache.get_cache_stats()
    assert stats["result_cache_entries"] == cache.RESULT_CACHE_MAX_ENTRIES
    assert stats["result_cache_evictions"] >= 1
    assert (
        cache.get_cached_result(
            "cli.query",
            {"offset": 0},
            archive_root=tmp_path,
            generation="generation-a",
        )
        is None
    )
