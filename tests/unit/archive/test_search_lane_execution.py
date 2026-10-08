"""Codex search-lane regressions through production archive and surface routes."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Iterator
from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import pytest

from polylogue.api import Polylogue
from polylogue.api.search_envelope_builder import build_search_envelope_for_spec
from polylogue.archive.query.archive_execution import archive_search_hits, count_archive
from polylogue.archive.query.execution_control import (
    QueryCancelledError,
    QueryTimeoutError,
    QueryWorkBudgetExceededError,
)
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.archive.query.search_contract import LaneFailure
from polylogue.archive.query.search_hits import project_search_hits
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.cli.query_output import format_search_envelope
from polylogue.config import Config, Source
from polylogue.core.errors import EmbeddingRetrievalNotReadyError
from polylogue.core.protocols import ScopedVectorQuery, VectorProvider
from polylogue.mcp.archive_support import archive_search_payload
from polylogue.mcp.payloads import session_search_result_payload
from polylogue.operations.daemon_reads import execute_read_operation
from polylogue.operations.operation_context import open_operation_read
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSummary
from polylogue.storage.sqlite.connection_profile import ReadFrameCancelledError, ReadFrameExpiredError
from polylogue.surfaces.payloads import decode_search_cursor
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.storage_records import SessionBuilder

LaneArchive = tuple[Path, Config, dict[str, str]]


class _VectorReply:
    """Only the external vector service is substituted; SQL/hydration are real."""

    def __init__(self, failure: Exception | None = None, *, hits: tuple[tuple[str, float], ...] = ()) -> None:
        self.hits = hits
        self.failure = failure
        self.calls = 0

    @contextmanager
    def scoped_query(
        self,
        session_ids: Iterable[str],
        *,
        text: str | None = None,
        seed_session_id: str | None = None,
        index_connection: sqlite3.Connection,
        configure_connection: Callable[[sqlite3.Connection], None],
        check_cancelled: Callable[[], None],
    ) -> Iterator[ScopedVectorQuery]:
        del session_ids, text, seed_session_id, index_connection, configure_connection
        check_cancelled()
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        yield ScopedVectorQuery(rows=iter(self.hits))


@pytest.fixture
def lane_archive(tmp_path: Path) -> tuple[Path, Config, dict[str, str]]:
    root = tmp_path / "archive"
    bootstrap_archive_root(root)
    ids: dict[str, str] = {}
    for name in ("dialogue", "action-one", "action-two"):
        builder = SessionBuilder(root / "index.db", name).provider("codex").title(name)
        if name == "dialogue":
            builder.add_message("plain", role="user", text="needle in ordinary conversation")
        else:
            # Repeated action hits in one session must not starve a distinct
            # session from an action page. The storage lowering owns these IDs.
            for index in range(4):
                builder.add_message(
                    f"tool-{index}",
                    role="assistant",
                    text="",
                    blocks=[
                        {
                            "type": "tool_use",
                            "tool_id": f"call-{index}",
                            "tool_name": "shell",
                            "tool_input": {"command": "needle"},
                            "text": "needle in a command",
                        }
                    ],
                )
        builder.save()
        ids[name] = builder.native_session_id()
    config = Config(
        archive_root=root,
        render_root=tmp_path / "render",
        db_path=root / "index.db",
        sources=[Source(name="fixture", path=tmp_path / "inbox")],
    )
    return root, config, ids


def test_actions_without_embeddings_return_only_action_evidence(lane_archive: LaneArchive) -> None:
    """F871: treating actions as semantic refuses; generic FTS admits dialogue."""
    root, _config, ids = lane_archive
    with open_operation_read(root) as pinned:
        result = archive_search_hits(
            SessionQueryPlan(query_terms=("needle",), retrieval_lane="actions", limit=2),
            archive_root=root,
            config=None,
            archive=pinned.archive,
        )
    assert {hit.session_id for hit, _summary in result.hits} == {ids["action-one"], ids["action-two"]}
    assert result.retrieval_lane == "actions"
    assert result.execution.requested_lanes == result.execution.executed_lanes == ("action",)
    assert not result.execution.degraded


@pytest.mark.asyncio
async def test_actions_lane_count_counts_only_sessions_its_search_returns(lane_archive: LaneArchive) -> None:
    """The actions-lane count and the actions-lane search read one relation.

    Anti-vacuity: drop ``actions_only`` from the count route and the dialogue
    session, which the actions search never returns, is counted too (3).
    """
    root, _config, _ids = lane_archive
    plan = SessionQueryPlan(query_terms=("needle",), retrieval_lane="actions")
    assert await count_archive(plan, archive_root=root, config=None) == 2
    dialogue = replace(plan, retrieval_lane="dialogue")
    assert await count_archive(dialogue, archive_root=root, config=None) == 3


def test_spec_count_counts_only_sessions_its_actions_search_returns(lane_archive: LaneArchive) -> None:
    """The facade and daemon count seam applies the same actions lane.

    Anti-vacuity: drop ``actions_only`` from ``_archive_count_sessions_for_spec``
    and the dialogue session is counted for the actions lane too (3).
    """
    from polylogue.api.archive import _archive_count_sessions_for_spec
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root, _config, _ids = lane_archive
    actions = SessionQuerySpec(query_terms=("needle",), retrieval_lane="actions")
    with ArchiveStore.open_existing(root) as archive:
        assert _archive_count_sessions_for_spec(archive, actions) == 2
        assert _archive_count_sessions_for_spec(archive, replace(actions, retrieval_lane="dialogue")) == 3


def test_vector_execution_failure_degrades_hybrid_and_refuses_semantic(lane_archive: LaneArchive) -> None:
    """F872: an uncaught query failure loses real lexical hits and leaks its detail."""
    root, config, ids = lane_archive
    backend = _VectorReply(RuntimeError("sensitive provider detail must not escape"))
    with open_operation_read(root) as pinned:
        hybrid = archive_search_hits(
            SessionQueryPlan(
                query_terms=("needle",),
                retrieval_lane="hybrid",
                limit=10,
                vector_provider=cast(VectorProvider, backend),
            ),
            archive_root=root,
            config=config,
            archive=pinned.archive,
        )
        with pytest.raises(EmbeddingRetrievalNotReadyError) as refusal:
            archive_search_hits(
                SessionQueryPlan(
                    similar_text="needle",
                    retrieval_lane="semantic",
                    vector_provider=cast(VectorProvider, backend),
                ),
                archive_root=root,
                config=config,
                archive=pinned.archive,
            )
    assert {hit.session_id for hit, _summary in hybrid.hits} == set(ids.values())
    assert hybrid.execution.failed_lanes[0].kind == "execution_failed"
    assert hybrid.execution.failed_lanes[0].reason == "RuntimeError"
    assert hybrid.execution.executed_lanes == ("text", "action")
    assert refusal.value.readiness_status == "failed"
    assert "sensitive provider detail" not in str(refusal.value)
    assert backend.calls == 2


@pytest.mark.parametrize(
    "error_type",
    [
        QueryCancelledError,
        QueryTimeoutError,
        QueryWorkBudgetExceededError,
        ReadFrameCancelledError,
        ReadFrameExpiredError,
    ],
)
def test_vector_read_cancellation_is_not_a_successful_hybrid_degradation(
    lane_archive: LaneArchive, error_type: type[Exception]
) -> None:
    """F872 guard: catching cancellation as an optional-lane error conceals aborted reads."""
    root, config, _ids = lane_archive
    backend = _VectorReply(error_type("cancelled read"))
    with open_operation_read(root) as pinned, pytest.raises(error_type):
        archive_search_hits(
            SessionQueryPlan(
                query_terms=("needle",),
                retrieval_lane="hybrid",
                vector_provider=cast(VectorProvider, backend),
            ),
            archive_root=root,
            config=config,
            archive=pinned.archive,
        )


@pytest.mark.parametrize("query", ["needle", "nomatchingword"])
def test_cli_and_both_mcp_builders_preserve_degraded_execution(lane_archive: LaneArchive, query: str) -> None:
    """F873: omitting execution reports an unavailable vector lane as healthy, even on empty pages."""
    root, _config, _ids = lane_archive
    spec = SessionQuerySpec.from_params(
        {"query": (query,), "retrieval_lane": "hybrid", "limit": 10},
        strict=True,
    )
    with open_operation_read(root) as pinned:
        plan = spec.to_plan()
        native = archive_search_hits(plan, archive_root=root, config=None, archive=pinned.archive)
        hits = project_search_hits(plan, native)
        mcp_native = archive_search_payload(
            pinned.archive,
            spec,
            query=query,
            limit=10,
            offset=0,
            sort=None,
            archive_root=root,
        )
    cli = json.loads(
        format_search_envelope(
            hits,
            query=query,
            retrieval_lane="hybrid",
            limit=10,
            offset=0,
            sort=None,
        )
    )
    mcp_hits = session_search_result_payload(
        hits,
        query=query,
        retrieval_lane="hybrid",
        total=None,
        limit=10,
        offset=0,
    )
    for envelope in (cli, mcp_hits.model_dump(mode="json"), mcp_native.model_dump(mode="json")):
        assert envelope["requested_lanes"] == ["text", "action", "vector"]
        assert envelope["executed_lanes"] == ["text", "action"]
        assert envelope["unavailable_lanes"] == ["vector"]
        assert envelope["advisories"]
        assert envelope["outcome"]["state"] == "degraded"
        assert bool(envelope["hits"]) is (query == "needle")


@pytest.mark.asyncio
async def test_hybrid_api_does_not_publish_semantic_only_total(
    lane_archive: LaneArchive, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F874: the semantic-only count returns zero despite a full lexical union page."""
    root, _config, _ids = lane_archive
    backend = _VectorReply()
    monkeypatch.setattr(
        "polylogue.storage.search_providers.create_vector_provider",
        lambda *args, **kwargs: backend,
    )
    spec = SessionQuerySpec.from_params(
        {"query": ("needle",), "retrieval_lane": "hybrid", "limit": 1},
        strict=True,
    )
    async with Polylogue(archive_root=root, db_path=root / "index.db") as facade:
        first = await build_search_envelope_for_spec(facade, spec)
        assert len(first.hits) == 1
        assert first.total is None  # Qualified unknown, never the unrelated vector count.
        assert first.next_offset == 1
        assert first.next_cursor is not None
        second = await build_search_envelope_for_spec(facade, replace(spec, cursor=first.next_cursor))
    assert len(second.hits) == 1
    assert second.hits[0].session.id != first.hits[0].session.id
    assert backend.calls == 2  # Counts must not rerun a different vector-only relation.


@pytest.mark.asyncio
async def test_degraded_hybrid_api_cursor_keeps_request_lane(
    lane_archive: LaneArchive, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F877: a dialogue-labelled cursor is rejected by its originating hybrid request."""
    root, _config, _ids = lane_archive
    monkeypatch.setattr("polylogue.storage.search_providers.create_vector_provider", lambda *a, **k: None)
    spec = SessionQuerySpec.from_params(
        {"query": ("needle",), "retrieval_lane": "hybrid", "limit": 1},
        strict=True,
    )
    async with Polylogue(archive_root=root, db_path=root / "index.db") as facade:
        first = await build_search_envelope_for_spec(facade, spec)
        assert first.next_cursor is not None
        assert decode_search_cursor(first.next_cursor).lane == "hybrid"
        second = await build_search_envelope_for_spec(facade, replace(spec, cursor=first.next_cursor))
    assert second.offset == 1
    assert second.hits and second.hits[0].session.id != first.hits[0].session.id
    assert second.executed_lanes == ("text", "action")
    assert second.unavailable_lanes == ("vector",)


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.asyncio
async def test_hybrid_explicit_date_cursor_follows_selected_order(
    lane_archive: LaneArchive, reverse: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Date order may oppose RRF order, including on reverse pages."""
    root, _config, ids = lane_archive
    # Relevance is action-one, action-two, dialogue. Date order is deliberately
    # dialogue, action-one, action-two; reverse flips only the requested order.
    with closing(sqlite3.connect(root / "index.db")) as connection:
        for name, updated in (("dialogue", 300), ("action-one", 200), ("action-two", 100)):
            connection.execute(
                "UPDATE sessions SET updated_at_ms = ? WHERE session_id = ?",
                (1_700_000_000_000 + updated * 1000, ids[name]),
            )
        connection.commit()
    monkeypatch.setattr("polylogue.storage.search_providers.create_vector_provider", lambda *a, **k: None)
    spec = SessionQuerySpec.from_params(
        {"query": ("needle",), "retrieval_lane": "hybrid", "sort": "date", "reverse": reverse, "limit": 1},
        strict=True,
    )
    names = ("dialogue", "action-one", "action-two") if not reverse else ("action-two", "action-one", "dialogue")
    expected = [ids[name] for name in names]
    async with Polylogue(archive_root=root, db_path=root / "index.db") as facade:
        page = await build_search_envelope_for_spec(facade, spec)
        actual = [str(page.hits[0].session.id)]
        while page.next_cursor is not None:
            page = await build_search_envelope_for_spec(facade, replace(spec, cursor=page.next_cursor))
            actual.extend(str(hit.session.id) for hit in page.hits)
    assert actual == expected


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("query_text", (False, True))
def test_random_post_filter_consumes_one_controlled_permutation(
    lane_archive: LaneArchive, reverse: bool, query_text: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One random traversal sees every candidate once, even for one survivor."""
    from polylogue.archive.query.archive_execution import _archive_summaries

    root, config, ids = lane_archive
    with open_operation_read(root) as pinned:
        archive = pinned.archive
        names = ("dialogue", "action-one", "action-two") if not reverse else ("action-two", "action-one", "dialogue")
        expected_order = [ids[name] for name in names]
        observed: list[str] = []
        plan = SessionQueryPlan(sort="random", query_terms=("needle",) if query_text else ())
        if query_text:
            hits = archive.iter_search_summaries("needle", limit=None, sort="date")
            search_by_id = {hit.session_id: hit for hit in hits}
            controlled_search_hits = [search_by_id[session_id] for session_id in expected_order]

            def one_permutation(*_args: object, **_kwargs: object) -> Iterator[object]:
                for hit in controlled_search_hits:
                    observed.append(hit.session_id)
                    yield hit

            monkeypatch.setattr(archive, "iter_search_summaries", one_permutation)
        else:
            all_rows = list(archive.iter_summaries(limit=None, offset=0, sort="date"))
            summary_by_id = {row.session_id: row for row in all_rows}
            controlled_summaries = [summary_by_id[session_id] for session_id in expected_order]

            def one_permutation(*_args: object, **_kwargs: object) -> Iterator[object]:
                for row in controlled_summaries:
                    observed.append(row.session_id)
                    yield row

            monkeypatch.setattr(archive, "iter_summaries", one_permutation)

        def reject_offset_pages(*_args: object, **_kwargs: object) -> list[object]:
            raise AssertionError("random post-filter traversal must not issue fresh random OFFSET pages")

        monkeypatch.setattr(archive, "list_summaries", reject_offset_pages)
        monkeypatch.setattr(archive, "search_summaries", reject_offset_pages)
        rows = _archive_summaries(
            plan,
            archive,
            config=config,
            archive_root=root,
            default_limit=1,
            complete=True,
            keep=lambda batch: [row for row in batch if row.session_id == ids["action-one"]],
        )
    assert observed == expected_order
    assert [row.session_id for row in rows] == [ids["action-one"]]


def test_random_post_filter_deduplicates_sessions_across_fts_batches(
    lane_archive: LaneArchive, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session with >500 matching blocks cannot consume a two-session page."""
    from polylogue.archive.query.archive_execution import _archive_summaries

    root, config, _ids = lane_archive
    first = SessionBuilder(root / "index.db", "many-blocks").provider("codex").title("many blocks")
    for index in range(501):
        first.add_message(f"many-{index}", role="user", text="batchmarker")
    first.save()
    first_id = first.native_session_id()
    second = SessionBuilder(root / "index.db", "last-block").provider("codex").title("last block")
    second.add_message("one", role="user", text="batchmarker")
    second.save()
    second_id = second.native_session_id()

    with open_operation_read(root) as pinned:
        archive = pinned.archive
        actual_matches = list(archive.iter_search_summaries("batchmarker", limit=None, sort="date"))
        first_hits = [hit for hit in actual_matches if hit.session_id == first_id]
        second_hits = [hit for hit in actual_matches if hit.session_id == second_id]
        assert len(first_hits) == 501
        assert len(second_hits) == 1
        controlled = [*first_hits, *second_hits]

        def one_random_permutation(*_args: object, **_kwargs: object) -> Iterator[object]:
            yield from controlled

        monkeypatch.setattr(archive, "iter_search_summaries", one_random_permutation)
        plan = SessionQueryPlan(sort="random", query_terms=("batchmarker",), negative_terms=("unused",), limit=2)
        rows = _archive_summaries(
            plan,
            archive,
            config=config,
            archive_root=root,
            default_limit=2,
            keep=lambda batch: batch,
        )

    assert [row.session_id for row in rows] == [first_id, second_id]


@pytest.mark.asyncio
async def test_session_list_page_and_total_share_one_snapshot(
    lane_archive: LaneArchive, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent delete after page selection cannot make its total disagree."""
    from polylogue.api import archive as archive_api

    root, _config, ids = lane_archive
    real_list = archive_api._archive_list_summaries_for_spec

    def list_then_delete(archive: object, spec: object, **kwargs: object) -> list[ArchiveSessionSummary]:
        summaries = real_list(archive, spec, **kwargs)  # type: ignore[arg-type]
        with closing(sqlite3.connect(root / "index.db")) as writer:
            writer.execute("DELETE FROM sessions WHERE session_id = ?", (ids["dialogue"],))
            writer.commit()
        return summaries

    monkeypatch.setattr(archive_api, "_archive_list_summaries_for_spec", list_then_delete)
    spec = SessionQuerySpec.from_params({"limit": 10}, strict=True)
    async with Polylogue(archive_root=root, db_path=root / "index.db") as facade:
        summaries, total = await facade.list_session_summaries_with_count(spec)
        current_total = await facade.count_sessions()
    assert len(summaries) == 3
    assert total == 3
    assert current_total == 2


@pytest.mark.parametrize("sort", ("messages", "tokens", "words", "longest"))
@pytest.mark.parametrize("reverse", (False, True))
def test_ranked_hybrid_numeric_sort_matches_full_session_metrics(
    lane_archive: LaneArchive, sort: Literal["messages", "tokens", "words", "longest"], reverse: bool
) -> None:
    """Ranked summary ordering uses transcript metrics, even without vectors."""
    from polylogue.archive.hydration import archive_envelope_to_session
    from polylogue.archive.query.sorting import sort_sessions

    root, _config, ids = lane_archive
    with closing(sqlite3.connect(root / "index.db")) as connection:
        for name, token_count in (("dialogue", 30), ("action-one", 5), ("action-two", 5)):
            connection.execute("UPDATE messages SET input_tokens = ? WHERE session_id = ?", (token_count, ids[name]))
        connection.commit()
    long_session = (
        SessionBuilder(root / "index.db", "many-old").provider("codex").updated_at("2020-01-01T00:00:00+00:00")
    )
    for ordinal in range(7):
        long_session.add_message(f"long-{ordinal}", text="needle " + "word " * 20, input_tokens=100)
    long_session_id = long_session.native_session_id()
    long_session.save()
    plan = SessionQueryPlan(query_terms=("needle",), retrieval_lane="hybrid", sort=sort, reverse=reverse)
    with open_operation_read(root) as pinned:
        all_sessions = [
            archive_envelope_to_session(pinned.archive.read_session(session_id))
            for session_id in (*ids.values(), long_session_id)
        ]
        for session in all_sessions:
            messages = list(session.messages)
            measured = any(
                message.input_tokens is not None
                or message.output_tokens is not None
                or message.cache_read_tokens is not None
                or message.cache_write_tokens is not None
                for message in messages
            )
            tokens = sum(
                (message.input_tokens or 0)
                + (message.output_tokens or 0)
                + (message.cache_read_tokens or 0)
                + (message.cache_write_tokens or 0)
                for message in messages
            )
            actual_metrics = pinned.archive.read_session_sort_metrics(str(session.id), sort="words")
            expected_metrics = (
                len(messages),
                sum(message.word_count for message in messages),
                max((message.word_count for message in messages), default=0),
                measured,
                tokens,
            )
            assert actual_metrics == expected_metrics, (session.id, actual_metrics, expected_metrics)
        expected = [str(session.id) for session in sort_sessions(plan, all_sessions)]
        result = archive_search_hits(
            plan,
            archive_root=root,
            config=None,
            archive=pinned.archive,
            vector_failure=LaneFailure("vector", "unavailable", "test", "synthetic lexical-only case"),
        )
    assert [hit.session_id for hit, _summary in result.hits] == expected
    assert result.execution.executed_lanes == ("text", "action")


@pytest.mark.parametrize("reverse", (False, True))
def test_ranked_numeric_sort_uses_composed_lineage_messages(lane_archive: LaneArchive, reverse: bool) -> None:
    """A child's inherited prefix contributes to its ranked message count."""
    from polylogue.archive.hydration import archive_envelope_to_session
    from tests.infra.storage_records import SessionBuilder

    root, _config, _ids = lane_archive
    index = root / "index.db"
    parent = SessionBuilder(index, "lineage-root").provider("codex").updated_at("2020-01-01T00:00:00+00:00")
    for ordinal in range(5):
        parent.add_message(f"parent-{ordinal}", text=f"inherited prefix {ordinal}")
    parent.save()
    child = (
        SessionBuilder(index, "lineage-child")
        .provider("codex")
        .parent_session("ext-lineage-root")
        .branch_type("continuation")
        .updated_at("2021-01-01T00:00:00+00:00")
    )
    # Replaying the same native prefix is how the writer records a
    # prefix-sharing edge and stores only the divergent tail.
    for ordinal in range(5):
        child.add_message(f"parent-{ordinal}", text=f"inherited prefix {ordinal}")
    child.add_message("child-tail", text="needle child tail")
    child_id = child.native_session_id()
    child.save()
    small_child = (
        SessionBuilder(index, "lineage-small")
        .provider("codex")
        .parent_session("ext-lineage-root")
        .branch_type("continuation")
        .updated_at("2022-01-01T00:00:00+00:00")
        .add_message("parent-0", text="inherited prefix 0")
        .add_message("small-tail", text="needle small tail")
    )
    small_child_id = small_child.native_session_id()
    small_child.save()
    plan = SessionQueryPlan(
        query_terms=("needle",), retrieval_lane="hybrid", sort="messages", reverse=reverse, root=False
    )
    with open_operation_read(root) as pinned:
        composed = archive_envelope_to_session(pinned.archive.read_session(child_id))
        messages = list(composed.messages)
        assert len(messages) == 6
        assert pinned.archive.read_session_sort_metrics(child_id, sort="words") == (
            len(messages),
            sum(message.word_count for message in messages),
            max((message.word_count for message in messages), default=0),
            any(
                message.input_tokens is not None
                or message.output_tokens is not None
                or message.cache_read_tokens is not None
                or message.cache_write_tokens is not None
                for message in messages
            ),
            sum(
                (message.input_tokens or 0)
                + (message.output_tokens or 0)
                + (message.cache_read_tokens or 0)
                + (message.cache_write_tokens or 0)
                for message in messages
            ),
        )
        small_composed = archive_envelope_to_session(pinned.archive.read_session(small_child_id))
        assert len(small_composed.messages) == 2
        result = archive_search_hits(
            plan,
            archive_root=root,
            config=None,
            archive=pinned.archive,
            vector_failure=LaneFailure("vector", "unavailable", "test", "synthetic lexical-only case"),
        )
    ordered = [hit.session_id for hit, _summary in result.hits]
    assert ordered == ([child_id, small_child_id] if not reverse else [small_child_id, child_id]), ordered


def test_ranked_word_sort_streams_chunk_boundaries_and_honors_cancellation(
    lane_archive: LaneArchive, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ranked production route counts chunk-spanning words and cancels mid-block."""
    from polylogue.archive.hydration import archive_envelope_to_session
    from polylogue.archive.query.execution_control import QueryCancelledError, QueryExecutionContext
    from polylogue.archive.query.sorting import sort_sessions
    from polylogue.storage.sqlite.archive_tiers.archive import SORT_METRIC_TEXT_CHUNK

    root, _config, ids = lane_archive
    long_word = "x" * (SORT_METRIC_TEXT_CHUNK + 7)
    whale = (
        SessionBuilder(root / "index.db", "chunk-boundary")
        .provider("codex")
        .updated_at("2020-01-01T00:00:00+00:00")
        .add_message("large-block", text=f"needle {long_word} y")
    )
    whale_id = whale.native_session_id()
    whale.save()
    plan = SessionQueryPlan(query_terms=("needle",), retrieval_lane="hybrid", sort="words")

    with open_operation_read(root) as pinned:
        actual_metrics = pinned.archive.read_session_sort_metrics(whale_id, sort="words")
        assert actual_metrics[1:3] == (3, 3)
        all_sessions = [
            archive_envelope_to_session(pinned.archive.read_session(session_id))
            for session_id in (*ids.values(), whale_id)
        ]
        expected = [str(session.id) for session in sort_sessions(plan, all_sessions)]
        result = archive_search_hits(
            plan,
            archive_root=root,
            config=None,
            archive=pinned.archive,
            vector_failure=LaneFailure("vector", "unavailable", "test", "synthetic lexical-only case"),
        )
    assert [hit.session_id for hit, _summary in result.hits] == expected

    context = QueryExecutionContext(call_id="sort-metric-cancel", query_ref="synthetic-large-block")
    checkpoints_inside_target_metrics = 0
    with pytest.raises(QueryCancelledError):
        with open_operation_read(root, execution_context=context) as pinned:
            archive = pinned.archive
            original_check = archive.check_operation_read
            original_metrics = archive.read_session_sort_metrics
            metric_session_active = False

            def monitored_check() -> None:
                nonlocal checkpoints_inside_target_metrics
                original_check()
                if metric_session_active:
                    checkpoints_inside_target_metrics += 1
                    if checkpoints_inside_target_metrics == 2:
                        context.cancel()

            def monitored_metrics(
                session_id: str, *, sort: Literal["messages", "words", "longest", "tokens"]
            ) -> tuple[int, int, int, bool, int]:
                nonlocal metric_session_active
                if session_id != whale_id:
                    return original_metrics(session_id, sort=sort)
                metric_session_active = True
                try:
                    return original_metrics(session_id, sort=sort)
                finally:
                    metric_session_active = False

            monkeypatch.setattr(archive, "check_operation_read", monitored_check)
            monkeypatch.setattr(archive, "read_session_sort_metrics", monitored_metrics)
            archive_search_hits(
                plan,
                archive_root=root,
                config=None,
                archive=archive,
                vector_failure=LaneFailure("vector", "unavailable", "test", "synthetic lexical-only case"),
            )
    assert checkpoints_inside_target_metrics >= 2


def test_degraded_hybrid_daemon_cursor_keeps_request_lane(lane_archive: LaneArchive) -> None:
    """F877: the pinned daemon path must also consume its own degraded cursor."""
    root, _config, _ids = lane_archive
    params = {"query": ("needle",), "retrieval_lane": "hybrid", "limit": 1}
    with open_operation_read(root) as pinned:
        first = execute_read_operation(
            "cli.query",
            {"params": params},
            archive=pinned.archive,
            serving_identity="daemon",
        )
        assert first["next_cursor"]
        second = execute_read_operation(
            "cli.query",
            {"params": {**params, "cursor": first["next_cursor"]}},
            archive=pinned.archive,
            serving_identity="daemon",
        )
    assert second["offset"] == 1
    assert second["retrieval_lane"] == "hybrid"
    assert second["executed_lanes"] == ["text", "action"]


def test_degraded_hybrid_fuses_action_rank_contributions(lane_archive: LaneArchive) -> None:
    """F879: returning generic FTS directly omits action membership and its RRF contribution."""
    root, _config, ids = lane_archive
    with open_operation_read(root) as pinned:
        result = archive_search_hits(
            SessionQueryPlan(query_terms=("needle",), retrieval_lane="hybrid", limit=10),
            archive_root=root,
            config=None,
            archive=pinned.archive,
        )
    by_id = {hit.session_id: hit for hit, _summary in result.hits}
    assert set(by_id) == set(ids.values())
    assert result.execution.requested_lanes == ("text", "action", "vector")
    assert result.execution.executed_lanes == ("text", "action")
    for name, hit in ((name, by_id[session_id]) for name, session_id in ids.items()):
        assert hit.lane_ranks is not None
        assert (hit.lane_ranks["action"] is None) is (name == "dialogue")
    # With only three candidates, an extra positive action contribution beats
    # any text-only contribution, independently of FTS's ordering of the three.
    assert {hit.session_id for hit, _summary in result.hits[:2]} == {ids["action-one"], ids["action-two"]}


def test_archive_read_failure_is_not_relabelled_as_a_degraded_vector_lane(
    lane_archive: LaneArchive, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F872 guard: wrapping ``semantic_summaries`` in the vector-leg handler hides index faults.

    The vector service answered; the archive's own hydration read failed. That
    is not an optional-lane outcome, so hybrid must raise rather than report a
    lexical page with a ``failed_lanes`` entry.
    """
    import sqlite3

    root, config, _ids = lane_archive
    backend = _VectorReply(hits=(("actual-witness", 0.1),))
    with open_operation_read(root) as pinned:

        def broken_semantic_summaries(*_args: object, **_kwargs: object) -> list[object]:
            raise sqlite3.DatabaseError("index page corrupt")

        monkeypatch.setattr(pinned.archive, "semantic_summaries", broken_semantic_summaries)
        with pytest.raises(sqlite3.DatabaseError, match="index page corrupt"):
            archive_search_hits(
                SessionQueryPlan(
                    query_terms=("needle",),
                    retrieval_lane="hybrid",
                    vector_provider=cast(VectorProvider, backend),
                ),
                archive_root=root,
                config=config,
                archive=pinned.archive,
            )
    assert backend.calls == 1
