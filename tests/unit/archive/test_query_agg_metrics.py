"""Tests for the ``| agg ...`` named-metric aggregate pipeline stage.

``| agg ...`` adds named sum/avg/min/max/percentile reducers over the small
set of numeric fields declared per unit
(``QueryUnitDescriptor.aggregate_metric_fields``). Every reducer is evaluated
by SQLite over the complete predicate-matching relation
(``ArchiveStore.query_unit_agg_metrics``), so there is no row cap and no
bounded-sample regime -- these tests pin the parser-level validation, the
executed numeric results against hand-computed expectations, and the absence
of any population bound.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from polylogue.archive.query.execution_control import (
    InterruptibleSQLiteRead,
    QueryExecutionContext,
    QueryWorkBudgetExceededError,
)
from polylogue.archive.query.expression import (
    ExpressionCompileError,
    QueryUnitGroupStage,
    QueryUnitSource,
    parse_unit_source_expression,
)
from polylogue.archive.query.unit_results import query_unit_rows
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.surfaces.payloads import QueryUnitAggregateEnvelope
from tests.infra.storage_records import SessionBuilder


def _seed_messages(index_db: Path) -> None:
    (
        SessionBuilder(index_db, "agg-metrics")
        .provider("claude-code")
        .add_message("m1", role="assistant", text="one two three")  # word_count=3
        .add_message("m2", role="assistant", text="one two three four five")  # word_count=5
        .add_message("m3", role="assistant", text="one two three four five six seven")  # word_count=7
        .add_message("m4", role="user", text="short")  # word_count=1, excluded by role filter
        .save()
    )


def test_agg_stage_computes_sum_avg_min_max_percentile_over_matched_rows(workspace_env: dict[str, Path]) -> None:
    index_db = workspace_env["archive_root"] / "index.db"
    _seed_messages(index_db)

    source = parse_unit_source_expression(
        "messages where role:assistant | agg count, sum:word_count, avg:word_count, min:word_count, "
        "max:word_count, p50:word_count"
    )
    assert source is not None

    with ArchiveStore.open_existing(index_db.parent) as archive:
        envelope = query_unit_rows(archive, source, query="agg-metrics", limit=20)

    assert isinstance(envelope, QueryUnitAggregateEnvelope)
    assert len(envelope.items) == 1
    row = envelope.items[0]
    assert row.group_key is None
    assert row.count == 3
    assert row.metrics == {
        "count": 3,
        "sum_word_count": 15.0,
        "avg_word_count": 5.0,
        "min_word_count": 3.0,
        "max_word_count": 7.0,
        # nearest-rank p50 over sorted [3, 5, 7] -> ceil(0.5*3)=2 -> index 1 -> 5
        "p50_word_count": 5.0,
    }
    assert envelope.pipeline is not None
    # The truncation regime is gone: the payload carries no exactness flag and
    # no sampled-row count because every reducer sees the whole relation.
    result = cast(dict[str, object], envelope.pipeline.get("result", {}))
    assert "exact" not in result
    assert "sampled_rows" not in result


def test_agg_stage_grouped_by_role_reports_per_group_metrics(workspace_env: dict[str, Path]) -> None:
    index_db = workspace_env["archive_root"] / "index.db"
    _seed_messages(index_db)

    source = parse_unit_source_expression(
        "messages where text:one OR text:short | group by role | agg count, avg:word_count | sort by key asc"
    )
    assert source is not None

    with ArchiveStore.open_existing(index_db.parent) as archive:
        envelope = query_unit_rows(archive, source, query="agg-grouped", limit=20)

    assert isinstance(envelope, QueryUnitAggregateEnvelope)
    by_group = {row.group_key: row.metrics for row in envelope.items}
    assert by_group["assistant"] == {"count": 3, "avg_word_count": 5.0}
    assert by_group["user"] == {"count": 1, "avg_word_count": 1.0}


def test_agg_unsupported_field_error_names_unit_metric_and_supported_set() -> None:
    with pytest.raises(
        ExpressionCompileError, match=r"agg avg:nope.*message rows.*supported metric fields: word_count"
    ):
        parse_unit_source_expression("messages where role:assistant | agg avg:nope")


def test_agg_unsupported_function_error_names_supported_functions() -> None:
    with pytest.raises(ExpressionCompileError, match=r"unsupported .agg. function 'median'"):
        parse_unit_source_expression("messages where role:assistant | agg median:word_count")


def test_agg_unit_without_metric_fields_rejects_field_bearing_metric() -> None:
    with pytest.raises(ExpressionCompileError, match=r"assertion has no numeric metric fields, only .agg count."):
        parse_unit_source_expression("assertions where kind:decision | agg avg:nope")


def test_agg_stage_cannot_follow_count_or_another_agg_stage() -> None:
    with pytest.raises(ExpressionCompileError, match=r"agg.*cannot follow"):
        parse_unit_source_expression("messages where role:assistant | count | agg avg:word_count")
    with pytest.raises(ExpressionCompileError, match=r"agg.*cannot follow"):
        parse_unit_source_expression("messages where role:assistant | agg count | agg avg:word_count")


def test_agg_stage_must_precede_limit_and_offset() -> None:
    with pytest.raises(ExpressionCompileError, match=r"agg.*must appear before .limit. and .offset."):
        parse_unit_source_expression("messages where role:assistant | limit 5 | agg avg:word_count")


def test_agg_action_error_rate_metrics_over_is_error(workspace_env: dict[str, Path]) -> None:
    """``is_error`` is 0/1 on action rows, so sum/avg over it are error count/rate."""

    index_db = workspace_env["archive_root"] / "index.db"
    (
        SessionBuilder(index_db, "agg-actions")
        .provider("claude-code")
        .git_repository_url("polylogue")
        .add_message(
            "turn",
            role="assistant",
            text="run tools",
            blocks=[
                {
                    "type": "tool_use",
                    "tool_name": "Bash",
                    "tool_id": "bash-ok",
                    "input": {"command": "pytest -q"},
                    "semantic_type": "shell",
                },
                {
                    "type": "tool_result",
                    "tool_id": "bash-ok",
                    "text": "passed",
                    "tool_result_is_error": 0,
                    "tool_result_exit_code": 0,
                },
                {
                    "type": "tool_use",
                    "tool_name": "Bash",
                    "tool_id": "bash-fail",
                    "input": {"command": "pytest -q broken"},
                    "semantic_type": "shell",
                },
                {
                    "type": "tool_result",
                    "tool_id": "bash-fail",
                    "text": "failed",
                    "tool_result_is_error": 1,
                    "tool_result_exit_code": 1,
                },
            ],
        )
        .save()
    )

    source = parse_unit_source_expression(
        "actions where tool:Bash | agg count, sum:is_error, avg:is_error, max:exit_code"
    )
    assert source is not None

    with ArchiveStore.open_existing(index_db.parent) as archive:
        envelope = query_unit_rows(archive, source, query="agg-actions", limit=10)

    assert isinstance(envelope, QueryUnitAggregateEnvelope)
    assert len(envelope.items) == 1
    metrics = envelope.items[0].metrics
    assert metrics is not None
    assert metrics["count"] == 2
    assert metrics["sum_is_error"] == 1.0
    assert metrics["avg_is_error"] == 0.5
    assert metrics["max_exit_code"] == 1.0


def test_group_by_dedupes_repeated_fields_preserving_order() -> None:
    """A repeated `group by` field collapses instead of widening the group key.

    Anti-vacuity: without the order-preserving dedupe in `_parse_group_stage`,
    `group by role,role,type,role` compiles to a four-field group, so the
    asserted `("role", "type")` stage fields and `"role,type"` group_by are red.
    """

    source = parse_unit_source_expression("messages where role:assistant | group by role,role,type,role | count")
    assert source is not None

    assert source.group_by == "role,type"
    group_stages = [s for s in source.pipeline_stages if isinstance(s, QueryUnitGroupStage)]
    assert [s.fields for s in group_stages] == [("role", "type")]


def test_group_by_dedupes_origin_alias_against_normalized_field() -> None:
    """`origin` and `session.origin` name one field and must collapse to one.

    Anti-vacuity: deduping before the `origin`/`repo` -> `session.*` rewrite (or
    not deduping at all) leaves two group fields, so the single-field
    `"session.origin"` assertion fails.
    """

    source = parse_unit_source_expression("messages where role:assistant | group by origin,session.origin | count")
    assert source is not None

    assert source.group_by == "session.origin"


def test_group_by_refuses_more_distinct_fields_than_the_declared_cap() -> None:
    """An adversarially wide `group by` is refused explicitly, never truncated.

    Anti-vacuity: without the `_MAX_GROUP_FIELDS` check, 10k distinct fields
    reach per-field validation and the raised error names an unsupported field
    rather than the cap, so the `at most 16 distinct fields` match fails. A
    truncating fix would raise nothing at all and fail on `pytest.raises`.
    """

    wide = ",".join(f"f{i}" for i in range(10_000))
    with pytest.raises(ExpressionCompileError, match=r"at most 16 distinct fields; got 10000"):
        parse_unit_source_expression(f"messages where role:assistant | group by {wide} | count")


def test_group_by_repeated_field_flood_is_bounded_by_dedupe() -> None:
    """10k repetitions of one real field compile to a single group field.

    Anti-vacuity: without dedupe this input passes per-field validation
    (every repetition is the supported `role`) and yields a 10k-wide group key,
    so the `"role"` equality assertion is red.
    """

    flood = ",".join(["role"] * 10_000)
    source = parse_unit_source_expression(f"messages where role:assistant | group by {flood} | count")
    assert source is not None

    assert source.group_by == "role"
    group_stages = [s for s in source.pipeline_stages if isinstance(s, QueryUnitGroupStage)]
    assert [s.fields for s in group_stages] == [("role",)]


#: Population used by the exactness tests. Strictly larger than the 50,000-row
#: fetch cap this lowering deleted, so any reintroduced bound at or below that
#: size changes the asserted numbers instead of leaving them green.
_LARGE_POPULATION = 50_001


def _seed_bulk_assistant_messages(index_db: Path, count: int) -> list[int]:
    """Insert ``count`` assistant message rows with word counts cycling 1..100.

    Returns the seeded word counts so the expectations below can be computed
    independently of the production reducer.
    """

    conn = sqlite3.connect(index_db)
    try:
        session_id = str(conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()[0])
        conn.execute(
            """
            WITH RECURSIVE seq(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM seq WHERE i < ?)
            INSERT INTO messages (
                session_id, native_id, position, role, message_type, material_origin,
                word_count, content_hash, occurred_at_ms
            )
            SELECT ?, 'bulk-' || i, i + 1000, 'assistant', 'message', 'assistant_authored',
                   (i % 100) + 1,
                   CAST(substr('bulk' || i || '________________________________', 1, 32) AS BLOB),
                   1700000000000 + i
            FROM seq
            """,
            (count, session_id),
        )
        conn.commit()
    finally:
        conn.close()
    return [(i % 100) + 1 for i in range(1, count + 1)]


def _nearest_rank(values: Sequence[int], rank: int) -> float:
    """Independent nearest-rank percentile: the ceil(rank*n/100)-th smallest value."""

    ordered = sorted(values)
    index = max(1, min(len(ordered), -((-rank * len(ordered)) // 100)))
    return float(ordered[index - 1])


def test_agg_metrics_stay_exact_over_a_population_larger_than_the_deleted_row_cap(
    workspace_env: dict[str, Path],
) -> None:
    """Every reducer sees the whole match set, with no bounded-sample regime.

    Anti-vacuity: reintroducing any row bound at or below 50,000 on the
    aggregate relation (the deleted ``_AGG_ROW_FETCH_CAP``, or a ``LIMIT`` in
    the ``selected`` CTE) changes ``count``, ``sum`` and ``avg`` here. The
    expected values are computed from the seeded word counts in Python, not
    from the production reducer.
    """

    index_db = workspace_env["archive_root"] / "index.db"
    (SessionBuilder(index_db, "agg-large").provider("claude-code").add_message("seed", role="user", text="seed").save())
    word_counts = _seed_bulk_assistant_messages(index_db, _LARGE_POPULATION)

    source = parse_unit_source_expression(
        "messages where role:assistant | agg count, sum:word_count, avg:word_count, "
        "min:word_count, max:word_count, p50:word_count, p90:word_count"
    )
    assert source is not None

    with ArchiveStore.open_existing(index_db.parent) as archive:
        envelope = query_unit_rows(archive, source, query="agg-large", limit=10)

    assert isinstance(envelope, QueryUnitAggregateEnvelope)
    assert len(envelope.items) == 1
    assert envelope.items[0].count == _LARGE_POPULATION
    assert envelope.items[0].metrics == {
        "count": len(word_counts),
        "sum_word_count": float(sum(word_counts)),
        "avg_word_count": sum(word_counts) / len(word_counts),
        "min_word_count": float(min(word_counts)),
        "max_word_count": float(max(word_counts)),
        "p50_word_count": _nearest_rank(word_counts, 50),
        "p90_word_count": _nearest_rank(word_counts, 90),
    }


def test_agg_percentile_uses_exact_integer_nearest_rank(workspace_env: dict[str, Path]) -> None:
    """``p7`` over 100 distinct values is the 7th smallest, not the 8th.

    ``rank / 100 * n`` in binary floating point returns 7.000000000000001 for
    ``rank=7, n=100``, so a float ``ceil`` selects rank 8. Nearest rank is an
    integer definition: ``ceil(rank * n / 100)`` computed in integer
    arithmetic.

    Anti-vacuity: computing the rank in floating point -- the reducer this
    lowering replaced -- returns 8.0 here.
    """

    index_db = workspace_env["archive_root"] / "index.db"
    (SessionBuilder(index_db, "agg-p7").provider("claude-code").add_message("seed", role="user", text="seed").save())
    word_counts = _seed_bulk_assistant_messages(index_db, 100)
    assert sorted(word_counts) == list(range(1, 101))

    source = parse_unit_source_expression("messages where role:assistant | agg p7:word_count, p10:word_count")
    assert source is not None

    with ArchiveStore.open_existing(index_db.parent) as archive:
        envelope = query_unit_rows(archive, source, query="agg-p7", limit=10)

    assert isinstance(envelope, QueryUnitAggregateEnvelope)
    assert envelope.items[0].metrics == {"p7_word_count": 7.0, "p10_word_count": 10.0}


def test_agg_group_by_session_repo_uses_the_owning_repository(workspace_env: dict[str, Path]) -> None:
    """``group by session.repo`` groups an ``agg`` page exactly as ``count`` does.

    Anti-vacuity: resolving group fields through a hand-maintained DSL-field ->
    row-payload-attribute map (the deleted ``_AGGREGATE_ROW_FIELDS``) has no
    ``session.repo`` entry for any unit, so every row falls into one
    ``[missing]`` group and both assertions below fail.
    """

    index_db = workspace_env["archive_root"] / "index.db"
    (
        SessionBuilder(index_db, "agg-repo-alpha")
        .provider("claude-code")
        .git_repository_url("https://example.invalid/alpha")
        .add_message("a1", role="assistant", text="one two three")
        .add_message("a2", role="assistant", text="one two three four five")
        .save()
    )
    (
        SessionBuilder(index_db, "agg-repo-beta")
        .provider("codex")
        .git_repository_url("https://example.invalid/beta")
        .add_message("b1", role="assistant", text="one two")
        .save()
    )

    agg_source = parse_unit_source_expression(
        "messages where role:assistant | group by session.repo | agg count, avg:word_count"
    )
    count_source = parse_unit_source_expression("messages where role:assistant | group by session.repo | count")
    assert agg_source is not None
    assert count_source is not None

    with ArchiveStore.open_existing(index_db.parent) as archive:
        agg_envelope = query_unit_rows(archive, agg_source, query="agg-repo", limit=20)
        count_envelope = query_unit_rows(archive, count_source, query="count-repo", limit=20)

    assert isinstance(agg_envelope, QueryUnitAggregateEnvelope)
    assert isinstance(count_envelope, QueryUnitAggregateEnvelope)
    assert {row.group_key: row.metrics for row in agg_envelope.items} == {
        "https://example.invalid/alpha": {"count": 2, "avg_word_count": 4.0},
        "https://example.invalid/beta": {"count": 1, "avg_word_count": 2.0},
    }
    assert {row.group_key: row.count for row in agg_envelope.items} == {
        row.group_key: row.count for row in count_envelope.items
    }


def test_agg_reports_none_for_a_group_with_no_values_and_counts_every_row(
    workspace_env: dict[str, Path],
) -> None:
    """A group whose metric column is entirely NULL reports ``None``, not zero.

    ``count`` still counts every row in the group, including the rows whose
    metric field is NULL.

    Anti-vacuity: folding SQL ``NULL`` into ``0.0`` (or counting only rows with
    a non-NULL metric) makes the ``ls`` group's assertions fail.
    """

    index_db = workspace_env["archive_root"] / "index.db"
    (
        SessionBuilder(index_db, "agg-null")
        .provider("claude-code")
        .add_message(
            "turn",
            role="assistant",
            text="run tools",
            blocks=[
                {
                    "type": "tool_use",
                    "tool_name": "ls",
                    "tool_id": "ls-1",
                    "input": {"command": "ls"},
                    "semantic_type": "shell",
                },
                {"type": "tool_result", "tool_id": "ls-1", "text": "a b"},
                {
                    "type": "tool_use",
                    "tool_name": "bash",
                    "tool_id": "bash-1",
                    "input": {"command": "false"},
                    "semantic_type": "shell",
                },
                {
                    "type": "tool_result",
                    "tool_id": "bash-1",
                    "text": "nope",
                    "tool_result_is_error": 1,
                    "tool_result_exit_code": 3,
                },
            ],
        )
        .save()
    )

    source = parse_unit_source_expression(
        "actions where session.origin:claude-code-session | group by tool | agg count, max:exit_code, avg:exit_code"
    )
    assert source is not None

    with ArchiveStore.open_existing(index_db.parent) as archive:
        envelope = query_unit_rows(archive, source, query="agg-null", limit=20)

    assert isinstance(envelope, QueryUnitAggregateEnvelope)
    by_group = {row.group_key: row.metrics for row in envelope.items}
    assert by_group["ls"] == {"count": 1, "max_exit_code": None, "avg_exit_code": None}
    assert by_group["bash"] == {"count": 1, "max_exit_code": 3.0, "avg_exit_code": 3.0}


def test_agg_over_an_empty_match_set_emits_no_group(workspace_env: dict[str, Path]) -> None:
    """An ungrouped ``agg`` over zero matching rows returns no rows at all.

    Anti-vacuity: an ungrouped SQL aggregate without the group-by guard emits
    one synthetic ``count: 0`` row, so ``items`` is length 1 and the outcome
    reads ``ok`` instead of ``empty``.
    """

    index_db = workspace_env["archive_root"] / "index.db"
    _seed_messages(index_db)

    source = parse_unit_source_expression("messages where text:nothingmatchesthis | agg count, avg:word_count")
    assert source is not None

    with ArchiveStore.open_existing(index_db.parent) as archive:
        envelope = query_unit_rows(archive, source, query="agg-empty", limit=20)

    assert isinstance(envelope, QueryUnitAggregateEnvelope)
    assert envelope.items == ()
    assert envelope.outcome.state == "empty"


def test_agg_pages_groups_and_reports_the_exact_total_group_count(workspace_env: dict[str, Path]) -> None:
    """Paging an ``agg`` page walks group keys in ascending order without gaps.

    The execution receipt's ``selected_rows_exact`` reports every group the
    predicate produced, not the page.

    Anti-vacuity: paging the underlying rows instead of the grouped relation,
    or reporting the page size as the selection, changes the walked keys or the
    reported total.
    """

    index_db = workspace_env["archive_root"] / "index.db"
    for index in range(5):
        (
            SessionBuilder(index_db, f"agg-page-{index}")
            .provider("claude-code")
            .git_repository_url(f"https://example.invalid/repo-{index}")
            .add_message(f"p{index}", role="assistant", text="one two three")
            .save()
        )

    walked: list[str | None] = []
    totals: list[int | None] = []
    with ArchiveStore.open_existing(index_db.parent) as archive:
        for offset in (0, 2, 4):
            source = parse_unit_source_expression(
                "messages where role:assistant | group by session.repo | agg count, avg:word_count"
            )
            assert source is not None
            context = QueryExecutionContext.create(query_text="agg-page")
            envelope = query_unit_rows(
                archive, source, query="agg-page", limit=2, offset=offset, execution_context=context
            )
            assert isinstance(envelope, QueryUnitAggregateEnvelope)
            walked.extend(row.group_key for row in envelope.items)
            totals.append(context.receipt.selected_rows_exact)

    assert walked == [f"https://example.invalid/repo-{index}" for index in range(5)]
    assert totals == [5, 5, 5]


def test_agg_terminal_is_interruptible_exactly_like_the_rows_terminal(workspace_env: dict[str, Path]) -> None:
    """A VM-work budget aborts the aggregate mid-statement, as it does for rows.

    The whole reduction runs inside one statement on the reader's own
    connection, so the production progress handler governs the aggregate
    terminal exactly as it governs the row terminal.

    Anti-vacuity: removing the reader's SQLite progress handler (or running
    the aggregate off the reader's connection) lets both branches complete
    instead of raising, and leaves ``receipt.interrupted`` false.
    """

    index_db = workspace_env["archive_root"] / "index.db"
    (
        SessionBuilder(index_db, "agg-budget")
        .provider("claude-code")
        .add_message("seed", role="user", text="seed")
        .save()
    )
    _seed_bulk_assistant_messages(index_db, 20_000)

    for expression in (
        "messages where role:assistant | agg count, avg:word_count, p90:word_count",
        "messages where role:assistant",
    ):
        source = parse_unit_source_expression(expression)
        assert source is not None
        context = QueryExecutionContext.create(query_text=expression, timeout_s=30.0, sqlite_vm_step_budget=10_000)
        reader = InterruptibleSQLiteRead(context)

        def _read(
            archive: ArchiveStore, source: QueryUnitSource = source, context: QueryExecutionContext = context
        ) -> object:
            return query_unit_rows(archive, source, query="agg-budget", limit=10, execution_context=context)

        with pytest.raises(QueryWorkBudgetExceededError):
            reader.run(index_db.parent, _read)
        assert context.receipt.interrupted is True
