"""Tests for the ``| agg ...`` named-metric aggregate pipeline stage (polylogue-fnm.1).

``| group by ... | count`` (tested in ``test_query_multi_aggregate.py``) stays
the exact, fully SQL-pushed aggregate lowerer. ``| agg ...`` adds named
sum/avg/min/max/percentile reducers over the small set of numeric fields
declared per unit (``QueryUnitDescriptor.aggregate_metric_fields``), computed
by fetching predicate-matching rows through the unit's existing row query and
reducing them in Python -- these tests pin both the parser-level validation
and the executed numeric results against hand-computed expectations.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from polylogue.archive.query.expression import ExpressionCompileError, parse_unit_source_expression
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
    result = cast(dict[str, object], envelope.pipeline["result"])
    assert result["exact"] is True
    assert result["sampled_rows"] == 3


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
