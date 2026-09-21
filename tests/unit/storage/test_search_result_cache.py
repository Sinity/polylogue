"""Functional laws for the daemon's bounded read-result cache.

The cache's correctness question is not "does it hit often enough" but "does
every entry describe the view that computed it".  A result relabelled with a
newer epoch is served to a later read *without executing that read's query
body*, so the failure is a wrong answer wearing a performance costume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polylogue.operations import daemon_reads
from polylogue.operations.daemon_reads import execute_read_operation
from polylogue.operations.operation_context import open_operation_read
from polylogue.storage.search import cache
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root


def _view(tmp_path: Path, generation: str = "generation-a") -> cache.ReadViewIdentity:
    return cache.capture_read_view(archive_root=tmp_path, generation=generation)


def test_result_cache_is_payload_and_generation_scoped(tmp_path: Path) -> None:
    cache.invalidate_search_cache()
    result = {"items": [{"id": "codex:one"}], "total": 1}
    view_a = _view(tmp_path)

    cache.put_cached_result("cli.query", {"limit": 1}, result, view=view_a)

    hit = cache.get_cached_result("cli.query", {"limit": 1}, view=view_a)
    assert hit == result
    assert hit is not result

    # A caller cannot mutate the resident value, and neither a changed query
    # nor a promoted active generation can reuse it.
    assert isinstance(hit, dict)
    hit["total"] = 99
    assert cache.get_cached_result("cli.query", {"limit": 1}, view=view_a) == result
    assert cache.get_cached_result("cli.query", {"limit": 2}, view=view_a) is None
    assert cache.get_cached_result("cli.query", {"limit": 1}, view=_view(tmp_path, "generation-b")) is None


def test_result_cache_invalidation_is_a_hard_freshness_boundary(tmp_path: Path) -> None:
    cache.invalidate_search_cache()
    cache.put_cached_result("facets", {"include_deferred": True}, {"total_sessions": 1}, view=_view(tmp_path))
    before = cache.get_cache_stats()
    cache.invalidate_search_cache()
    after = cache.get_cache_stats()

    assert after["cache_version"] == before["cache_version"] + 1
    assert after["result_cache_entries"] == 0
    assert cache.get_cached_result("facets", {"include_deferred": True}, view=_view(tmp_path)) is None


def test_result_cache_evicts_oldest_entry_at_bounded_capacity(tmp_path: Path) -> None:
    cache.invalidate_search_cache()
    view = _view(tmp_path)
    for number in range(cache.RESULT_CACHE_MAX_ENTRIES + 1):
        cache.put_cached_result("cli.query", {"offset": number}, {"offset": number}, view=view)

    stats = cache.get_cache_stats()
    assert stats["result_cache_entries"] == cache.RESULT_CACHE_MAX_ENTRIES
    assert stats["result_cache_evictions"] >= 1
    assert cache.get_cached_result("cli.query", {"offset": 0}, view=view) is None


def test_insert_under_a_superseded_view_is_declined_not_relabelled(tmp_path: Path) -> None:
    """Anti-vacuity: re-read the epoch at insert and this stores the answer.

    A read captures its view, its query body runs, an ingest write invalidates
    the cache while it runs, and only then does the read insert.  The answer
    describes the pre-invalidation view, so it must not become the resident
    value for the post-invalidation one.
    """
    cache.invalidate_search_cache()
    pinned_view = _view(tmp_path)

    # The query body runs here; an ingest write lands while it does.
    cache.invalidate_search_cache()

    cache.put_cached_result("facets", {"include_deferred": True}, {"total_sessions": 1}, view=pinned_view)

    assert cache.get_cache_stats()["result_cache_entries"] == 0
    # The next read pins the current view and must find nothing to serve.
    assert cache.get_cached_result("facets", {"include_deferred": True}, view=_view(tmp_path)) is None


def test_an_older_view_cannot_consume_a_newer_cached_answer(tmp_path: Path) -> None:
    """The symmetric direction: a stale pin must not read a fresher answer."""
    cache.invalidate_search_cache()
    older_view = _view(tmp_path)
    cache.invalidate_search_cache()
    newer_view = _view(tmp_path)
    assert newer_view.epoch != older_view.epoch

    cache.put_cached_result("cli.query", {"limit": 1}, {"total": 2}, view=newer_view)

    assert cache.get_cached_result("cli.query", {"limit": 1}, view=newer_view) == {"total": 2}
    assert cache.get_cached_result("cli.query", {"limit": 1}, view=older_view) is None


def test_pinned_read_names_a_view_and_serves_a_warm_repeat(tmp_path: Path, monkeypatch: Any) -> None:
    """Control: with no intervening write the route really does use the cache.

    Without this the interleaving law below could pass on a route that never
    caches at all.
    """
    bootstrap_archive_root(tmp_path)
    cache.invalidate_search_cache()
    executions: list[int] = []

    def _facets(params: Any, *, archive: Any) -> dict[str, object]:
        executions.append(1)
        return {"total_sessions": len(executions)}

    monkeypatch.setattr(daemon_reads, "_facets_payload", _facets)

    bodies: list[dict[str, object]] = []
    for _ in range(2):
        with open_operation_read(tmp_path) as pinned:
            assert pinned.read_view is not None
            bodies.append(
                execute_read_operation(
                    "facets",
                    {"params": {"include_deferred": True}},
                    archive=pinned.archive,
                    serving_identity="daemon",
                    read_view=pinned.read_view,
                )
            )

    assert executions == [1]
    assert bodies == [{"total_sessions": 1}, {"total_sessions": 1}]


def test_a_read_completing_across_an_invalidation_is_not_served_to_the_next_read(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The interleaving the cache existed through without covering.

    Anti-vacuity: make ``put_cached_result`` re-read the global epoch instead
    of using the view it was handed, and the second read is served the first
    read's pre-invalidation rows without executing its own query body.
    """
    bootstrap_archive_root(tmp_path)
    cache.invalidate_search_cache()
    answers: list[dict[str, object]] = [{"total_sessions": 1}, {"total_sessions": 2}]
    executions: list[dict[str, object]] = []

    def _facets(params: Any, *, archive: Any) -> dict[str, object]:
        answer = answers[len(executions)]
        executions.append(answer)
        if len(executions) == 1:
            # An ingest write commits and announces itself while the first
            # read's query body is still running.
            cache.invalidate_search_cache()
        return answer

    monkeypatch.setattr(daemon_reads, "_facets_payload", _facets)

    def _read() -> dict[str, object]:
        with open_operation_read(tmp_path) as pinned:
            return execute_read_operation(
                "facets",
                {"params": {"include_deferred": True}},
                archive=pinned.archive,
                serving_identity="daemon",
                read_view=pinned.read_view,
            )

    first = _read()
    second = _read()

    assert first == {"total_sessions": 1}
    # The second read must see the new state, which means it ran its own query
    # body rather than being handed the relabelled pre-invalidation answer.
    assert second == {"total_sessions": 2}
    assert len(executions) == 2


def test_a_read_whose_pin_straddles_an_invalidation_names_no_view(tmp_path: Path, monkeypatch: Any) -> None:
    """A cache token captured after the snapshot can also be too new.

    When the epoch moves across the pin itself, no single view describes the
    snapshot, so the read is named uncacheable instead of being labelled with
    a guess.
    """
    bootstrap_archive_root(tmp_path)
    cache.invalidate_search_cache()
    pin = ArchiveStore.pin_operation_snapshot

    def _pin_then_invalidate(self: Any) -> Any:
        versions = pin(self)
        cache.invalidate_search_cache()
        return versions

    monkeypatch.setattr(ArchiveStore, "pin_operation_snapshot", _pin_then_invalidate)

    with open_operation_read(tmp_path) as pinned:
        assert pinned.read_view is None
        executions: list[int] = []

        def _facets(params: Any, *, archive: Any) -> dict[str, object]:
            executions.append(1)
            return {"total_sessions": len(executions)}

        monkeypatch.setattr(daemon_reads, "_facets_payload", _facets)
        execute_read_operation(
            "facets",
            {"params": {"include_deferred": True}},
            archive=pinned.archive,
            serving_identity="daemon",
            read_view=pinned.read_view,
        )

    assert cache.get_cache_stats()["result_cache_entries"] == 0
