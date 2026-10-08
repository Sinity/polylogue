"""Scope and requested grain precede exact semantic and hybrid result windows."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Any, Never, cast

import pytest

from polylogue.archive.query.archive_execution import (
    archive_search_hits,
    count_archive,
    list_archive,
    list_summaries_archive,
)
from polylogue.archive.query.execution_control import QueryCancelledError
from polylogue.archive.query.expression import ExpressionCompileError, compile_expression
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.archive.session.domain_models import Session
from polylogue.config import Config
from polylogue.core.errors import EmbeddingRetrievalNotReadyError
from polylogue.operations.operation_context import open_operation_read
from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.scoped_semantic import declare_ranking_repository
from tests.infra.scoped_semantic import ranking_archive as _ranking_archive


def ranking_archive(
    *args: Any, **kwargs: Any
) -> tuple[Config, SqliteVecProvider, dict[tuple[str, str], tuple[str, str]], list[dict[str, object]]]:
    """Seed off the event loop: a synchronous write lease may not block it."""
    return run_off_event_loop(lambda: _ranking_archive(*args, **kwargs))


def test_rare_scope_precedes_vector_ranking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "archive"
    samples = [(f"outside-{i}", "m", f"Outside population has sufficient prose {i}", i / 100) for i in range(15)]
    samples += [
        ("eligible-a", "m", "Scoped eligible first result has enough prose", 1.0),
        ("eligible-b", "m", "Scoped eligible second result has enough prose", 2.0),
    ]
    config, provider, ids, requests = ranking_archive(root, samples, query_axis=0.0, monkeypatch=monkeypatch)
    declare_ranking_repository(root, [ids[(sid, "m")][0] for sid in ("eligible-a", "eligible-b")])
    plan = SessionQueryPlan(similar_text="question", repo_names=("target",), limit=2, vector_provider=provider)
    with open_operation_read(root) as frame:
        result = archive_search_hits(plan, archive_root=root, config=config, archive=frame.archive)
    assert [hit.session_id for hit, _ in result.hits] == [ids[(sid, "m")][0] for sid in ("eligible-a", "eligible-b")]
    assert result.execution.completed_lanes == ("vector",)
    assert result.execution.exactness == "exact"
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_semantic_count_applies_content_postfilters_like_the_ranked_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    config, provider, _ids, _requests = ranking_archive(
        root,
        [
            ("kept", "m", "Candidate has sufficient semantic prose", 0.0),
            ("excluded", "m", "Candidate has excluded semantic prose", 1.0),
        ],
        query_axis=0.0,
        monkeypatch=monkeypatch,
    )
    plan = SessionQueryPlan(similar_text="question", negative_terms=("excluded",), vector_provider=provider)
    page = await list_summaries_archive(plan, archive_root=root, config=config)
    total = await count_archive(plan, archive_root=root, config=config)
    assert len(page) == 1
    assert total == 1


@pytest.mark.parametrize("full", (False, True))
async def test_dominant_session_does_not_consume_a_session_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    full: bool,
) -> None:
    root = tmp_path / "archive"
    samples = [
        ("dominant", f"m-{i:02}", f"Dominant occurrence has distinct adequate prose {i}", i / 100) for i in range(20)
    ]
    samples += [
        ("runner", "m", "Runner session has adequate purchased prose", 1.0),
        ("later", "m", "Later session has adequate purchased prose", 2.0),
    ]
    config, provider, ids, requests = ranking_archive(root, samples, query_axis=0.0, monkeypatch=monkeypatch)
    plan = SessionQueryPlan(similar_text="question", limit=2, vector_provider=provider)
    reader = list_archive if full else list_summaries_archive
    rows = await reader(plan, archive_root=root, config=config)
    assert [str(row.id) for row in rows] == [ids[("dominant", "m-00")][0], ids[("runner", "m")][0]]
    assert len(requests) == 1


def test_shared_output_occurrences_and_tied_witnesses_are_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    text = "Exactly identical purchased prose shared by real occurrences"
    samples = [(sid, mid, text, 0.0) for sid, mid in (("a", "second"), ("a", "first"), ("b", "same"))]
    config, provider, ids, _ = ranking_archive(root, samples, query_axis=0.0, monkeypatch=monkeypatch)
    with open_operation_read(root) as frame:
        result = archive_search_hits(
            SessionQueryPlan(similar_text="question", limit=2, vector_provider=provider),
            archive_root=root,
            config=config,
            archive=frame.archive,
        )
    assert [(hit.session_id, hit.message_id) for hit, _ in result.hits] == [ids[("a", "first")], ids[("b", "same")]]
    assert all(hit.block_id.startswith(hit.message_id) for hit, _ in result.hits)


def test_near_uses_all_seeds_and_preserves_minimum_distance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    samples = [
        ("seed", f"m-{i:02}", f"Seed has distinct retained purchased prose number {i}", 100.0 + i) for i in range(20)
    ]
    samples += [
        ("seed", "m-20", "Last seed supplies the omitted nearest vector", 0.0),
        ("a-late", "m", "Candidate closest to the twenty first seed", 0.0),
        ("b-first", "m", "Candidate closest to the original first seed", 100.0),
        ("c-middle", "m", "Candidate between both ends of the seed range", 50.0),
    ]
    config, provider, ids, requests = ranking_archive(root, samples, query_axis=0.0, monkeypatch=monkeypatch)
    provider.voyage_key = None
    with open_operation_read(root) as frame:
        result = archive_search_hits(
            SessionQueryPlan(similar_session_id=ids[("seed", "m-00")][0], limit=3, vector_provider=provider),
            archive_root=root,
            config=config,
            archive=frame.archive,
        )
    assert [hit.session_id for hit, _ in result.hits] == [
        ids[(sid, "m")][0] for sid in ("a-late", "b-first", "c-middle")
    ]
    assert requests == []
    scored = provider.query_by_session(ids[("seed", "m-00")][0], limit=3)
    assert [distance for _, distance in scored] == [0.0, 0.0, 50.0]


def test_session_existential_scope_does_not_rebind_semantic_witness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    samples = [
        ("positive", "user", "The user independently supplies the gate token", 20.0),
        ("positive", "assistant", "Assistant is the actual closest semantic witness", 0.0),
        ("negative", "user", "The user lacks the required correlated condition", 0.0),
        ("negative", "assistant", "Assistant contains gate but does not satisfy user scope", 0.0),
    ]
    config, provider, ids, _ = ranking_archive(root, samples, query_axis=0.0, monkeypatch=monkeypatch)
    with closing(sqlite3.connect(root / "index.db")) as connection:
        connection.execute(
            "UPDATE messages SET role='assistant', material_origin='assistant_authored' WHERE native_id='assistant'"
        )
        connection.commit()
    spec = compile_expression('semantic:"question" AND exists message(role:user AND text:gate)')
    plan = spec.to_plan(vector_provider=provider)
    with open_operation_read(root) as frame:
        result = archive_search_hits(plan, archive_root=root, config=config, archive=frame.archive)
    assert [(hit.session_id, hit.message_id) for hit, _ in result.hits] == [ids[("positive", "assistant")]]


@pytest.mark.parametrize("expression", ['exists message(semantic:"question")', 'messages where semantic:"question"'])
def test_bound_unit_semantic_queries_remain_typed_refusals(expression: str) -> None:
    with pytest.raises(ExpressionCompileError):
        compile_expression(expression)


def test_cancellation_before_opcode_interval_prevents_query_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    config, provider, _ids, requests = ranking_archive(
        root,
        [("only", "m", "One current purchased occurrence with sufficient prose", 0.0)],
        query_axis=0.0,
        monkeypatch=monkeypatch,
    )
    with open_operation_read(root) as frame:

        def cancelled() -> None:
            raise QueryCancelledError("synthetic cancellation")

        frame.archive.set_read_progress_guard(lambda: 0, n_opcodes=1_000_000, check_cancelled=cancelled)
        with pytest.raises(QueryCancelledError):
            archive_search_hits(
                SessionQueryPlan(similar_text="question", vector_provider=provider),
                archive_root=root,
                config=config,
                archive=frame.archive,
            )
        assert tuple(frame.archive._conn.execute("SELECT 1").fetchone()) == (1,)
        frame.archive.clear_read_progress_guard()
    assert requests == []


def test_post_pin_edit_does_not_replace_the_original_vector_witness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    config, provider, ids, requests = ranking_archive(
        root,
        [("only", "m", "Original pinned purchased occurrence has enough prose", 0.0)],
        query_axis=0.0,
        monkeypatch=monkeypatch,
        concurrent_writes=True,
    )
    with open_operation_read(root) as frame:
        with closing(sqlite3.connect(root / "index.db")) as writer:
            writer.execute("UPDATE blocks SET text='Replacement unembedded content with enough prose'")
            writer.commit()
        result = archive_search_hits(
            SessionQueryPlan(similar_text="question", vector_provider=provider),
            archive_root=root,
            config=config,
            archive=frame.archive,
        )
    assert [(hit.session_id, hit.message_id) for hit, _ in result.hits] == [ids[("only", "m")]]
    assert result.hits[0][0].snippet == "Original pinned purchased occurrence has enough prose"
    assert len(requests) == 1
    with pytest.raises(EmbeddingRetrievalNotReadyError) as refusal:
        archive_search_hits(
            SessionQueryPlan(similar_text="question", vector_provider=provider), archive_root=root, config=config
        )
    assert refusal.value.readiness_status == "empty"
    assert len(requests) == 1


def test_later_hybrid_winner_uses_complete_original_lane_ranks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    samples: list[tuple[str, str, str, float | None]] = [
        (f"text-{i}", "m", "needle " * 12 + f"lexical candidate number {i}", None) for i in range(3)
    ]
    samples += [
        (f"vector-{i}", "m", f"Retained vector candidate without lexical term number {i}", i / 10) for i in range(3)
    ]
    samples += [
        ("winner", "m", "The needle occurs once in this deliberately longer later lexical candidate prose", 1.0)
    ]
    config, provider, ids, requests = ranking_archive(root, samples, query_axis=0.0, monkeypatch=monkeypatch)
    plan = SessionQueryPlan(query_terms=("needle",), retrieval_lane="hybrid", limit=1, vector_provider=provider)
    with open_operation_read(root) as frame:
        result = archive_search_hits(plan, archive_root=root, config=config, archive=frame.archive)
        # A later lexical/vector overlap wins over every one-lane prefix.
        assert result.hits[0][0].session_id == ids[("winner", "m")][0]
        assert result.hits[0][0].lane_ranks == {"text": 4, "action": None, "vector": 4}
        assert result.execution.completed_lanes == ("text", "action", "vector")
        assert result.execution.exactness == "exact"
    assert len(requests) == 1


def test_residual_predicate_precedes_all_hybrid_lane_ranks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    samples = [
        (f"excluded-{i}", "m", f"needle excluded closer purchased occurrence with adequate prose {i}", i / 100)
        for i in range(12)
    ]
    samples += [("eligible", "m", "needle eligible later purchased occurrence with adequate prose", 3.0)]
    config, provider, ids, _requests = ranking_archive(root, samples, query_axis=0.0, monkeypatch=monkeypatch)
    visited: list[str] = []

    def qualifies(session: Session) -> bool:
        visited.append(str(session.id))
        return session.title == "eligible"

    plan = SessionQueryPlan(
        query_terms=("needle",), retrieval_lane="hybrid", predicates=(qualifies,), limit=1, vector_provider=provider
    )
    with open_operation_read(root) as frame:
        result = archive_search_hits(plan, archive_root=root, config=config, archive=frame.archive)
    assert result.hits[0][0].session_id == ids[("eligible", "m")][0]
    assert result.hits[0][0].lane_ranks == {"text": 1, "action": None, "vector": 1}
    assert len(visited) == len(samples)
    assert len(set(visited)) == len(samples)


@pytest.mark.parametrize("exit_route", ("abandon", "cancel", "error", "refuse"))
def test_scoped_cursor_exits_settle_owned_handle_and_preserve_borrowed_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_route: str,
) -> None:
    from polylogue.core.errors import SchemaSkewError
    from polylogue.storage.search_providers.sqlite_vec_support import SqliteVecError
    from tests.infra.embedding_reader_probe import EmbeddingReadProbe, TrackedCursor

    root = tmp_path / "archive"
    config, provider, ids, _requests = ranking_archive(
        root,
        [
            (f"session-{i}", "m", f"Purchased retained occurrence for cursor settlement number {i}", float(i))
            for i in range(5)
        ],
        query_axis=0.0,
        monkeypatch=monkeypatch,
    )
    probe = EmbeddingReadProbe(root, monkeypatch)
    original_open = probe.open

    def capture(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = original_open(*args, **kwargs)
        creator = connection.cursor

        def cursor(*args: Any, **kwargs: Any) -> TrackedCursor:
            kwargs.setdefault("factory", TrackedCursor)
            result = cast(TrackedCursor, creator(*args, **kwargs))
            probe.cursors.append(result)
            return result

        monkeypatch.setattr(connection, "cursor", cursor)
        return connection

    monkeypatch.setattr("polylogue.storage.search_providers.sqlite_vec_runtime.open_readonly_connection", capture)
    with open_operation_read(root) as frame:
        creator = frame.archive._conn.cursor

        def canonical_cursor(*args: Any, **kwargs: Any) -> TrackedCursor:
            kwargs.setdefault("factory", TrackedCursor)
            result = cast(TrackedCursor, creator(*args, **kwargs))
            probe.cursors.append(result)
            return result

        monkeypatch.setattr(frame.archive._conn, "cursor", canonical_cursor)
        checkpoints = 0

        def checkpoint() -> None:
            nonlocal checkpoints
            checkpoints += 1
            if exit_route == "cancel" and checkpoints == 3:
                raise QueryCancelledError("operation_cancelled")

        def failed_embedding(*args: Any, **kwargs: Any) -> Never:
            raise SqliteVecError("synthetic_transport_error")

        if exit_route == "error":
            monkeypatch.setattr(provider, "_get_embeddings", failed_embedding)
        if exit_route == "refuse":
            provider.dimension = 3

        def read_one() -> None:
            with provider.scoped_query(
                (sid for sid, _mid in ids.values()),
                index_connection=frame.archive._conn,
                configure_connection=frame.archive.configure_operation_read_connection,
                check_cancelled=checkpoint,
                text="question",
            ) as traversal:
                assert next(traversal.rows)
                assert traversal.exact

        if exit_route == "abandon":
            read_one()
        else:
            expected = {"cancel": QueryCancelledError, "error": SqliteVecError, "refuse": SchemaSkewError}[exit_route]
            with pytest.raises(expected):
                read_one()
        assert all(cursor.settled for cursor in probe.cursors)
        with closing(frame.archive._conn.cursor()) as cursor:
            assert tuple(cursor.execute("SELECT 1").fetchone()) == (1,)
    probe.assert_settled()


def test_sql_fusion_settles_three_lanes_once_with_stable_ties(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSearchHit

    root = tmp_path / "archive"
    _config, _provider, ids, _requests = ranking_archive(
        root,
        [(sid, "m", f"Real canonical witness occurrence for SQL fusion session {sid}", 0.0) for sid in ("a", "b", "c")],
        query_axis=0.0,
        monkeypatch=monkeypatch,
    )
    with open_operation_read(root) as frame:

        def witness(name: str) -> ArchiveSessionSearchHit:
            sid, mid = ids[(name, "m")]
            return frame.archive.semantic_summaries([(mid, 0.0)], limit=1)[0]

        with frame.archive.scoped_search_population(sid for sid, _mid in ids.values()):
            frame.archive.settle_scoped_search_lane(
                "text", iter([witness("a"), witness("a"), witness("b"), witness("c")])
            )
            frame.archive.settle_scoped_search_lane("action", iter([witness("b"), witness("a"), witness("c")]))
            frame.archive.settle_scoped_search_lane("vector", iter([witness("c"), witness("a"), witness("b")]))
            hits = list(frame.archive.iter_scoped_search_hits(hybrid=True))
        with frame.archive.scoped_search_population(sid for sid, _mid in ids.values()):
            frame.archive.settle_scoped_search_lane("text", iter([witness("b"), witness("a")]))
            frame.archive.settle_scoped_search_lane("vector", iter([witness("a"), witness("b")]))
            tied = list(frame.archive.iter_scoped_search_hits(hybrid=True))
        assert [hit.session_id for hit in tied] == [ids[(sid, "m")][0] for sid in ("a", "b")]
        assert tied[0].lane_ranks == {"text": 2, "action": None, "vector": 1}
        assert tied[1].lane_ranks == {"text": 1, "action": None, "vector": 2}
    assert [hit.session_id for hit in hits] == [ids[(sid, "m")][0] for sid in ("a", "b", "c")]
    assert hits[0].lane_ranks == {"text": 1, "action": 2, "vector": 2}
    assert hits[1].lane_ranks == {"text": 2, "action": 1, "vector": 3}
    assert hits[2].lane_ranks == {"text": 3, "action": 3, "vector": 1}


async def test_repository_session_grain_text_and_near_use_complete_same_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.repository import SessionRepository

    root = tmp_path / "archive"
    samples = [("dominant", f"m-{i:02}", f"Dominant repository prose occurrence {i}", i / 100) for i in range(20)]
    samples += [
        ("runner", "m", "Repository runner is the second session", 1.0),
        ("seed", "m", "Repository seed has a retained vector", 0.0),
    ]
    _config, provider, ids, requests = ranking_archive(
        root, samples, query_axis=0.0, monkeypatch=monkeypatch, concurrent_writes=True
    )
    with closing(sqlite3.connect(root / "index.db")) as connection:
        connection.execute(
            "INSERT INTO session_events (session_id, position, event_type, payload_json) VALUES (?, 0, 'notice', ?)",
            (ids[("runner", "m")][0], '{"text":"original retained event"}'),
        )
        connection.commit()
    original_query = provider._query_vector
    caller_thread = threading.get_ident()

    def after_pin(connection: sqlite3.Connection, text: str) -> bytes | None:
        assert threading.get_ident() != caller_thread
        # A concurrent replacement after projection must not leak into hydration.
        with closing(sqlite3.connect(root / "index.db")) as writer:
            writer.execute("UPDATE sessions SET title = 'changed' WHERE session_id = ?", (ids[("runner", "m")][0],))
            writer.execute(
                "UPDATE session_events SET payload_json = ? WHERE session_id = ?",
                ('{"text":"changed event"}', ids[("runner", "m")][0]),
            )
            writer.commit()
        return original_query(connection, text)

    monkeypatch.setattr(provider, "_query_vector", after_pin)
    async with SessionRepository(db_path=root / "index.db") as repository:

        async def forbidden_get_many(*args: object, **kwargs: object) -> None:
            raise AssertionError("ranking and full hydration must share the held frame")

        monkeypatch.setattr(repository, "get_many", forbidden_get_many)
        rows = await repository.search_similar("question", limit=3, vector_provider=provider)
        assert [str(row.id) for row in rows] == [
            ids[(sid, "m" if sid != "dominant" else "m-00")][0] for sid in ("dominant", "seed", "runner")
        ]
        assert rows[-1].title == "runner"
        assert len(rows[0].messages) == 20
        assert rows[-1].session_events[0].payload == {"text": "original retained event"}
        assert len(requests) == 1
        provider.voyage_key = None
        near = await repository.search_similar_sessions(ids[("seed", "m")][0], limit=2, vector_provider=provider)
    assert near["source_embedded_messages"] == 1
    near_hits = cast(list[dict[str, object]], near["results"])
    assert [hit["session_id"] for hit in near_hits] == [
        ids[(sid, "m" if sid != "dominant" else "m-00")][0] for sid in ("dominant", "runner")
    ]
    assert all(hit["matched_message_count"] == 1 for hit in near_hits)
    assert len(requests) == 1


async def test_ranked_count_streams_the_complete_qualified_relation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.archive.query import archive_execution

    root = tmp_path / "archive"
    config, provider, _ids, requests = ranking_archive(
        root,
        [(f"s-{i:03}", "m", f"Distinct count witness with adequate prose {i}", float(i)) for i in range(12)],
        query_axis=0.0,
        monkeypatch=monkeypatch,
    )

    async def forbidden_list(*args: object, **kwargs: object) -> None:
        raise AssertionError("count must not retain full result objects")

    monkeypatch.setattr(archive_execution, "list_archive", forbidden_list)
    monkeypatch.setattr(archive_execution, "list_summaries_archive", forbidden_list)
    total = await archive_execution.count_archive(
        SessionQueryPlan(similar_text="question", limit=1, offset=10, vector_provider=provider),
        archive_root=root,
        config=config,
    )
    assert total == 12
    assert len(requests) == 1


@pytest.mark.parametrize("full", [False, True])
async def test_unlimited_explicit_ranked_sort_has_no_default_candidate_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    full: bool,
) -> None:
    root = tmp_path / "archive"
    config, provider, ids, requests = ranking_archive(
        root,
        [(f"s-{i:03}", "m", f"Unlimited exact sorted population witness number {i}", float(i)) for i in range(65)],
        query_axis=0.0,
        monkeypatch=monkeypatch,
    )
    with closing(sqlite3.connect(root / "index.db")) as connection:
        for i in range(65):
            connection.execute(
                "UPDATE sessions SET updated_at_ms = ? WHERE session_id = ?",
                (1_700_000_000_000 + i * 1000, ids[(f"s-{i:03}", "m")][0]),
            )
        connection.commit()
    reader = list_archive if full else list_summaries_archive
    rows = await reader(
        SessionQueryPlan(similar_text="question", sort="date", limit=None, offset=2, vector_provider=provider),
        archive_root=root,
        config=config,
    )
    expected = [ids[(f"s-{i:03}", "m")][0] for i in reversed(range(63))]
    assert [str(row.id) for row in rows] == expected
    assert len(requests) == 1


def test_sql_comparison_keys_match_current_python_owner_on_held_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.archive.hydration import archive_envelope_to_session, archive_summary_to_domain
    from polylogue.archive.query.sorting import session_order_values, summary_order_values

    root = tmp_path / "archive"
    samples = [
        (sid, mid, f"Purchased comparison prose {sid} {mid} " + "word " * words, float(i))
        for i, (sid, mid, words) in enumerate(
            [("a", "m0", 4), ("a", "m1", 1), ("b", "m0", 10), ("c", "m0", 2), ("d", "m0", 3)]
        )
    ]
    _config, _provider, ids, _requests = ranking_archive(root, samples, query_axis=0.0, monkeypatch=monkeypatch)
    with closing(sqlite3.connect(root / "index.db")) as connection:
        connection.execute("UPDATE sessions SET updated_at_ms = 1700000000000 WHERE title IN ('a','b')")
        connection.execute("UPDATE sessions SET created_at_ms = 1600000000000 WHERE title = 'c'")
        connection.execute(
            "UPDATE messages SET input_tokens = ? WHERE session_id = ?", (2**63 - 1, ids[("a", "m0")][0])
        )
        connection.execute("UPDATE messages SET input_tokens = 10 WHERE session_id = ?", (ids[("b", "m0")][0],))
        connection.commit()
    selected_ids = [ids[(sid, "m0")][0] for sid in ("d", "b", "c", "a")]
    with open_operation_read(root) as frame:
        sessions = [archive_envelope_to_session(frame.archive.read_session(sid)) for sid in selected_ids]
        summaries = [archive_summary_to_domain(frame.archive.read_summary(sid)) for sid in selected_ids]
        for full in (False, True):
            for sort in ("date", "messages", "words", "longest", "tokens"):
                for reverse in (False, True):
                    plan = SessionQueryPlan(sort=sort, reverse=reverse)
                    # Numeric order keys are transcript metrics, including for
                    # summary results. Their independent oracle is the full
                    # composed session, not the date-only summary comparator.
                    numeric = sort in {"messages", "tokens", "words", "longest"}
                    use_full = full or numeric
                    expected = plan._sort_sessions(sessions) if use_full else plan._sort_summaries(summaries)
                    with frame.archive.scoped_search_population(selected_ids):
                        hits = [
                            frame.archive.semantic_summaries([(ids[(sid, "m0")][1], 0.0)], limit=1)[0]
                            for sid in ("d", "b", "c", "a")
                        ]
                        frame.archive.settle_scoped_search_lane("vector", hits)
                        keys = (
                            (
                                (str(row.id), *session_order_values(plan, row), ordinal)
                                for ordinal, row in enumerate(sessions, start=1)
                            )
                            if use_full
                            else (
                                (str(row.id), *summary_order_values(plan, row), ordinal)
                                for ordinal, row in enumerate(summaries, start=1)
                            )
                        )
                        frame.archive.settle_scoped_search_order(keys)
                        actual = list(
                            frame.archive.iter_scoped_search_hits(hybrid=False, explicit_sort=True, reverse=reverse)
                        )
                        page = list(
                            frame.archive.iter_scoped_search_hits(
                                hybrid=False, explicit_sort=True, reverse=reverse, limit=1, offset=1
                            )
                        )
                    assert [hit.session_id for hit in actual] == [str(row.id) for row in expected]
                    assert page[0].session_id == str(expected[1].id)
                    if full and sort == "tokens":
                        assert {hit.session_id for hit in actual[-2:]} == {ids[(sid, "m0")][0] for sid in ("c", "d")}


def test_scope_with_no_current_vectors_refuses_before_query_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    config, provider, ids, requests = ranking_archive(
        root,
        [
            ("outside", "m", "Outside scope has a valid retained vector", 0.0),
            ("inside", "m", "Scoped input initially has a retained vector", 1.0),
        ],
        query_axis=0.0,
        monkeypatch=monkeypatch,
    )
    declare_ranking_repository(root, [ids[("inside", "m")][0]])
    with closing(sqlite3.connect(root / "index.db")) as connection:
        connection.execute(
            "UPDATE blocks SET text = 'Changed scoped prose without a purchased output' WHERE message_id = ?",
            (ids[("inside", "m")][1],),
        )
        connection.execute(
            "UPDATE messages SET content_hash = ? WHERE message_id = ?", (b"x" * 32, ids[("inside", "m")][1])
        )
        connection.commit()
    with open_operation_read(root) as frame, pytest.raises(EmbeddingRetrievalNotReadyError) as refusal:
        archive_search_hits(
            SessionQueryPlan(similar_text="question", repo_names=("target",), vector_provider=provider),
            archive_root=root,
            config=config,
            archive=frame.archive,
        )
    assert refusal.value.readiness_status == "empty"
    assert requests == []


@pytest.mark.parametrize("view", ("chronicle", "temporal"))
def test_ranked_read_views_sort_the_full_session_population_before_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, view: str
) -> None:
    from polylogue.operations.read_view_chronicle import execute_chronicle_read
    from polylogue.operations.read_view_dialogue_temporal import execute_temporal_read

    root = tmp_path / "archive"
    samples = [("dominant", f"m-{i}", f"Closest short prose occurrence number {i}", i / 100) for i in range(20)]
    samples += [("runner", f"m-{i}", f"Runner occurrence {i} " + "neutral " * 60, 10.0 + i) for i in range(3)]
    samples += [("winner", f"m-{i}", f"Winner occurrence {i} " + "synthetic " * 100, 20.0 + i) for i in range(4)]
    _, provider, ids, requests = ranking_archive(root, samples, query_axis=0.0, monkeypatch=monkeypatch)
    with closing(sqlite3.connect(root / "index.db")) as connection:
        connection.executemany(
            "UPDATE sessions SET created_at_ms = ? WHERE session_id = ?",
            [
                (1_767_225_600_000 + day * 86_400_000, ids[(sid, "m-0")][0])
                for day, sid in enumerate(("dominant", "runner", "winner"), start=1)
            ],
        )
        connection.commit()
    payload = {
        "params": {
            "similar_text": "question",
            "sort": "words" if view == "chronicle" else "date",
            "offset": 1,
            "limit": 1,
        }
    }
    with open_operation_read(root) as frame:
        if view == "chronicle":
            result = execute_chronicle_read(payload, archive=frame.archive, vector_provider=provider)
            selected = [row["session_id"] for row in cast(dict[str, Any], result["payload"])["sessions"]]
        else:
            result = execute_temporal_read(payload, archive=frame.archive, vector_provider=provider)
            selected = [
                event["source_ref"].removeprefix("session:")
                for event in cast(dict[str, Any], result["payload"])["temporal_window"]["events"]
                if event["family"] == "archive-session"
            ]
    assert selected == [ids[("runner", "m-0")][0]]
    assert len(requests) == 1


def test_numeric_sort_collation_observes_python_cancellation_below_sql_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    ranking_archive(
        root,
        [("one", "m", "Neutral prose to bootstrap the actual read frame", 0.0)],
        query_axis=0.0,
        monkeypatch=monkeypatch,
    )

    def cancelled() -> None:
        raise QueryCancelledError("operation_cancelled")

    with open_operation_read(root) as frame:
        frame.archive.set_read_progress_guard(lambda: 0, n_opcodes=1_000_000, check_cancelled=cancelled)
        with closing(frame.archive._conn.cursor()) as cursor, pytest.raises(QueryCancelledError):
            cursor.execute("SELECT '9223372036854775808' COLLATE polylogue_result_number < '9223372036854775809'")
        frame.archive.clear_read_progress_guard()


def test_scoped_ranking_keeps_borrowed_index_after_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
    from polylogue.storage.search_providers.sqlite_vec_support import SqliteVecError

    root = tmp_path / "archive"
    config, provider, ids, requests = ranking_archive(
        root,
        [
            ("seed", "m", "Original seed has a retained purchased output", 1.0),
            ("neighbor", "m", "Original neighbor has a retained purchased output", 0.0),
        ],
        query_axis=0.0,
        monkeypatch=monkeypatch,
    )
    with open_operation_read(root) as frame:
        replacement = root / "replacement.db"
        with closing(sqlite3.connect(replacement)) as writer, closing(writer.cursor()) as cursor:
            frame.archive._conn.backup(writer)
            cursor.execute("DELETE FROM blocks")
            cursor.execute("DELETE FROM messages")
            cursor.execute("DELETE FROM sessions")
            writer.commit()
        replacement.replace(root / "index.db")
        original_open = provider._get_read_connection

        def open_scoped(**kwargs: Any) -> sqlite3.Connection:
            connection = original_open(**kwargs)
            with closing(connection.cursor()) as cursor:
                assert "archive_index" not in {row[1] for row in cursor.execute("PRAGMA database_list")}
            with pytest.raises(SqliteVecError):
                SqliteVecProvider(None, snapshot_connection=connection)
            return connection

        monkeypatch.setattr(provider, "_get_read_connection", open_scoped)
        for plan in (
            SessionQueryPlan(similar_text="question", vector_provider=provider),
            SessionQueryPlan(similar_session_id=ids[("seed", "m")][0], vector_provider=provider),
        ):
            result = archive_search_hits(plan, archive_root=root, config=config, archive=frame.archive)
            assert result.hits[0][0].message_id == ids[("neighbor", "m")][1]
            assert result.hits[0][0].snippet == "Original neighbor has a retained purchased output"
        with closing(frame.archive._conn.cursor()) as cursor:
            assert cursor.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2
        with closing(sqlite3.connect(root / "index.db")) as observer, closing(observer.cursor()) as cursor:
            assert cursor.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    assert len(requests) == 1


def test_session_embedding_preserves_index_hash_for_compatible_scoped_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The session embedding route keys its vector by the index content hash,
    and a compatible model's scoped query reads it without a new purchase."""
    from polylogue.storage.embeddings.materialization import embed_archive_session_sync
    from tests.infra.scoped_semantic import axis_vector

    root = tmp_path / "archive"
    text = "Actual public producer sends this exact canonical authored prose"
    config, provider, ids, requests = ranking_archive(
        root, [("only", "m", text, None)], query_axis=0.0, monkeypatch=monkeypatch
    )
    sid, mid = ids[("only", "m")]
    calls: list[str] = []

    def embed(texts: list[str], input_type: str = "document") -> list[list[float]]:
        calls.append(input_type)
        return [axis_vector(2.0 if input_type == "document" else 0.0) for _ in texts]

    monkeypatch.setattr(provider, "_get_embeddings", embed)
    provider.model = "voyage-4"
    outcome = embed_archive_session_sync(root / "index.db", provider, sid)
    assert (outcome.status, outcome.embedded_message_count) == ("embedded", 1)
    with closing(sqlite3.connect(root / "embeddings.db")) as observer, closing(observer.cursor()) as cursor:
        assert (
            cursor.execute(
                "SELECT message_content_hash FROM message_embedding_refs WHERE message_id=?", (mid,)
            ).fetchone()[0]
            == b"m" * 32
        )
        original_meta = cursor.execute("SELECT * FROM message_embeddings_meta").fetchall()
    provider.model = "voyage-4-lite"
    with open_operation_read(root) as frame:
        result = archive_search_hits(
            SessionQueryPlan(similar_text="question", vector_provider=provider),
            archive_root=root,
            config=config,
            archive=frame.archive,
        )
    assert result.hits[0][0].message_id == mid
    assert calls == ["document", "query"]
    assert requests == []
    with closing(sqlite3.connect(root / "embeddings.db")) as observer, closing(observer.cursor()) as cursor:
        assert cursor.execute("SELECT * FROM message_embeddings_meta").fetchall() == original_meta


@pytest.mark.parametrize("supplied", (False, True))
def test_scoped_provider_refuses_a_borrowed_frame_outside_its_archive_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    supplied: bool,
) -> None:
    from contextlib import nullcontext

    from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
    from polylogue.storage.search_providers.sqlite_vec_runtime import open_vector_read_snapshot
    from polylogue.storage.search_providers.sqlite_vec_support import SqliteVecError
    from tests.infra.archive_templates import bootstrap_archive_root
    from tests.infra.scoped_semantic import axis_vector
    from tests.infra.vector_archive import seed_vector_archive

    root_a, root_b = tmp_path / "a", tmp_path / "b"
    _, _, identities_a, requests_a = ranking_archive(
        root_a,
        [("a", "m", "Original archive A has exact purchased prose", 0.0)],
        query_axis=0.0,
        monkeypatch=monkeypatch,
    )
    bootstrap_archive_root(root_b)
    seed_vector_archive(root_b, [("b", "m", "Different archive B has exact purchased prose", axis_vector(0.0))])
    ordinary = SqliteVecProvider("synthetic-key", db_path=root_b / "embeddings.db", archive_root=root_b)
    snapshot = (
        open_vector_read_snapshot(
            embeddings_path=root_b / "embeddings.db", index_path=root_b / "index.db", recipe=ordinary.document_recipe
        )
        if supplied
        else None
    )
    with closing(snapshot) if snapshot is not None else nullcontext():
        provider_b = SqliteVecProvider("synthetic-key", snapshot_connection=snapshot) if supplied else ordinary
        with open_operation_read(root_a) as frame:
            with (
                pytest.raises(SqliteVecError),
                provider_b.scoped_query(
                    (sid for sid, _mid in identities_a.values()),
                    index_connection=frame.archive._conn,
                    configure_connection=frame.archive.configure_operation_read_connection,
                    check_cancelled=frame.archive.check_operation_read,
                    text="question",
                ),
            ):
                raise AssertionError("cross-archive scope must refuse before scoring")
            with closing(frame.archive._conn.cursor()) as cursor:
                assert cursor.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        assert requests_a == []


def test_joint_operation_owner_admits_only_its_original_canonical_lender(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider

    root = tmp_path / "archive"
    config, ordinary, identities, requests = ranking_archive(
        root,
        [("only", "m", "Joint operation retains original canonical purchased prose", 0.0)],
        query_axis=0.0,
        monkeypatch=monkeypatch,
    )
    with open_operation_read(root, vector_recipe=ordinary.document_recipe) as frame:
        vector = frame.archive.operation_vector_connection
        assert vector is not None
        supplied = SqliteVecProvider("synthetic-key", snapshot_connection=vector)
        result = archive_search_hits(
            SessionQueryPlan(similar_text="question", vector_provider=supplied),
            archive_root=root,
            config=config,
            archive=frame.archive,
        )
        assert result.hits[0][0].message_id == identities[("only", "m")][1]
        with closing(vector.cursor()) as cursor:
            assert cursor.execute("SELECT 1").fetchone()[0] == 1
        with closing(frame.archive._conn.cursor()) as cursor:
            assert cursor.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
    assert len(requests) == 1


def test_supplied_old_snapshot_refuses_new_lender_at_the_same_index_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
    from polylogue.storage.search_providers.sqlite_vec_support import SqliteVecError

    root = tmp_path / "archive"
    _, ordinary, identities, requests = ranking_archive(
        root,
        [("only", "m", "Old operation owns exactly this canonical purchased prose", 0.0)],
        query_axis=0.0,
        monkeypatch=monkeypatch,
    )
    with open_operation_read(root, vector_recipe=ordinary.document_recipe) as original:
        vector = original.archive.operation_vector_connection
        assert vector is not None
        supplied = SqliteVecProvider("synthetic-key", snapshot_connection=vector)
        replacement = root / "replacement.db"
        with closing(sqlite3.connect(replacement)) as writer, closing(writer.cursor()) as cursor:
            original.archive._conn.backup(writer)
            cursor.execute("UPDATE sessions SET title = 'Replacement frame'")
            writer.commit()
        replacement.replace(root / "index.db")
        with open_operation_read(root) as current:
            with (
                pytest.raises(SqliteVecError),
                supplied.scoped_query(
                    (sid for sid, _mid in identities.values()),
                    index_connection=current.archive._conn,
                    configure_connection=current.archive.configure_operation_read_connection,
                    check_cancelled=current.archive.check_operation_read,
                    text="question",
                ),
            ):
                raise AssertionError("a different lender cannot inherit original owner admission")
            with closing(current.archive._conn.cursor()) as cursor:
                assert cursor.execute("SELECT title FROM sessions").fetchone()[0] == "Replacement frame"
        with closing(original.archive._conn.cursor()) as cursor:
            assert cursor.execute("SELECT title FROM sessions").fetchone()[0] == "only"
        with closing(vector.cursor()) as cursor:
            assert cursor.execute("SELECT 1").fetchone()[0] == 1
    assert requests == []
