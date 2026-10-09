"""Functional laws for the daemon's bounded read-result cache.

The cache's correctness question is not "does it hit often enough" but "does
every entry describe the view that computed it".  A result relabelled with a
newer epoch is served to a later read *without executing that read's query
body*, so the failure is a wrong answer wearing a performance costume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from polylogue.operations import daemon_reads
from polylogue.operations.daemon_reads import execute_read_operation
from polylogue.operations.operation_context import open_operation_read
from polylogue.storage.search import cache
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.frozen_clock import FrozenClock


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


def test_facets_cache_follows_user_commit_before_epoch_invalidation(tmp_path: Path, monkeypatch: Any) -> None:
    """An unchanged frame hits; a committed assertion with the same epoch does not."""
    bootstrap_archive_root(tmp_path)
    with ArchiveStore(tmp_path) as writer:
        writer._conn.execute(
            "INSERT INTO sessions (native_id, origin, content_hash) VALUES ('one', 'unknown-export', ?)",
            (bytes(32),),
        )
        writer._conn.commit()
    cache.invalidate_search_cache()
    params = {"include_deferred": False}

    def read() -> dict[str, object]:
        with open_operation_read(tmp_path) as pinned:
            return execute_read_operation(
                "facets",
                {"params": params},
                archive=pinned.archive,
                serving_identity="daemon",
                read_view=pinned.read_view,
            )

    first = read()
    hits_before = cache.get_cache_stats()["result_cache_hits"]
    assert read() == first
    assert cache.get_cache_stats()["result_cache_hits"] == hits_before + 1
    assert first["tags"] == {}
    original_epoch = cache.current_cache_epoch()
    invalidate = cache.invalidate_search_cache
    interleaved: list[dict[str, object]] = []

    def read_after_commit_then_invalidate() -> None:
        assert cache.current_cache_epoch() == original_epoch
        interleaved.append(read())
        invalidate()

    # The actual tag writer commits user.db before this callback advances the
    # process counter. Schedule the fresh reader in that existing interval.
    with ArchiveStore(tmp_path) as writer:
        with monkeypatch.context() as patch:
            patch.setattr(cache, "invalidate_search_cache", read_after_commit_then_invalidate)
            assert writer.add_user_tags(("unknown-export:one",), ("review",)) == 1
    assert len(interleaved) == 1
    assert interleaved[0]["tags"] == {"review": 1}
    assert read()["tags"] == {"review": 1}


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


@pytest.mark.frozen_clock_modules("polylogue.surfaces.query_rows")
@pytest.mark.parametrize("ranked", [False, True], ids=["list", "ranked"])
def test_warm_query_refreshes_relative_time_without_reexecuting_selection(
    tmp_path: Path, monkeypatch: Any, frozen_clock: FrozenClock, ranked: bool
) -> None:
    bootstrap_archive_root(tmp_path)
    cache.invalidate_search_cache()
    timestamp = frozen_clock.now().isoformat()
    row = {"id": "codex:one", "updated_at": timestamp, "relative_time": "just now"}
    undated = {"id": "codex:undated", "relative_time": "unknown"}
    unprojected = {"id": "codex:unprojected"}
    rows = [row, undated, unprojected]
    result: dict[str, object] = {
        "hits" if ranked else "items": [{"session": value, "match": {"rank": i}} for i, value in enumerate(rows)]
        if ranked
        else rows,
        "snapshot_epoch": "original-snapshot",
        "outcome": {"state": "ok"},
        "next_offset": 3,
    }
    executions: list[int] = []

    def query(*args: Any, **kwargs: Any) -> dict[str, object]:
        executions.append(1)
        return result

    monkeypatch.setattr(daemon_reads, "_query_payload", query)

    def read() -> dict[str, object]:
        with open_operation_read(tmp_path) as pinned:
            return execute_read_operation(
                "cli.query",
                {},
                archive=pinned.archive,
                serving_identity="daemon",
                read_view=pinned.read_view,
            )

    first = read()
    before_hits = cache.get_cache_stats()["result_cache_hits"]
    frozen_clock.advance(3600)
    second = read()
    delivered = second["hits" if ranked else "items"]
    assert isinstance(delivered, list)
    actual_rows = [hit["session"] for hit in delivered] if ranked else delivered
    assert actual_rows == [
        {**row, "relative_time": "1h ago"},
        undated,
        unprojected,
    ]
    assert executions == [1]
    assert cache.get_cache_stats()["result_cache_hits"] == before_hits + 1
    assert second["snapshot_epoch"] == first["snapshot_epoch"]
    assert second["outcome"] == first["outcome"]
    assert second["next_offset"] == first["next_offset"]
    assert row["relative_time"] == "just now"


def test_ranked_query_cache_preserves_the_actual_pinned_selection_frame(tmp_path: Path, monkeypatch: Any) -> None:
    """A real fresh query and its cache hit carry the same authoritative frame."""
    bootstrap_archive_root(tmp_path)
    cache.invalidate_search_cache()
    original = daemon_reads._query_payload
    executions: list[int] = []

    def query(*args: Any, **kwargs: Any) -> dict[str, object]:
        executions.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(daemon_reads, "_query_payload", query)
    with open_operation_read(tmp_path) as pinned:
        values = [
            execute_read_operation(
                "cli.query",
                {"params": {"query": "needle", "limit": 5}},
                archive=pinned.archive,
                serving_identity="daemon",
                read_view=pinned.read_view,
            )
            for _ in range(2)
        ]
    from polylogue.surfaces.payloads import SearchEnvelope

    assert executions == [1]
    assert values[0]["snapshot_epoch"] == values[1]["snapshot_epoch"]
    assert SearchEnvelope.model_validate(values[0]).snapshot_epoch
    assert SearchEnvelope.model_validate(values[1]).snapshot_epoch
    import json

    import jsonschema
    from pydantic import ValidationError

    from polylogue.operations.daemon_protocol import QueryResult

    schema = json.loads((Path(__file__).parents[3] / "docs/schemas/cli-output/search-envelope.schema.json").read_text())
    for value in values:
        jsonschema.validate(value, schema)
        QueryResult.model_validate(value)
    with pytest.raises(ValidationError):
        QueryResult.model_validate({key: value for key, value in values[0].items() if key != "snapshot_epoch"})


def test_ranked_query_declines_missing_or_stale_cached_frames(tmp_path: Path, monkeypatch: Any) -> None:
    """Neither absent authority nor an old epoch can reach cached decoration."""
    import pytest

    from polylogue.cli.operation_kernel import OperationEnvelopeError, OperationFailedError
    from polylogue.cli.session_rows import _selection_frame

    bootstrap_archive_root(tmp_path)
    cache.invalidate_search_cache()
    params = {"query": "needle", "limit": 5}
    with open_operation_read(tmp_path) as pinned:
        assert pinned.read_view is not None
        current = execute_read_operation(
            "cli.query",
            {"params": params},
            archive=pinned.archive,
            serving_identity="daemon",
            read_view=pinned.read_view,
        )
        epoch = current["snapshot_epoch"]
        assert isinstance(epoch, str)
        for bad_epoch in (None, epoch + ":stale-generation"):
            poisoned = {**current, "snapshot_epoch": bad_epoch, "query": "poisoned-cache"}
            cache.put_cached_result("cli.query", params, poisoned, view=pinned.read_view)
            value = execute_read_operation(
                "cli.query",
                {"params": params},
                archive=pinned.archive,
                serving_identity="daemon",
                read_view=pinned.read_view,
            )
            assert value["query"] == "needle"
            assert value["snapshot_epoch"] == epoch
        with pytest.raises(OperationEnvelopeError):
            _selection_frame(None, {"snapshot_epoch": None})
        with pytest.raises(OperationFailedError) as refusal:
            _selection_frame(epoch, {"snapshot_epoch": epoch + ":stale-generation"})
        assert refusal.value.code == "query_continuation_stale"
