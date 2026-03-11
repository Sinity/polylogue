"""Tests for typed query execution planning."""

from __future__ import annotations

import pytest

from polylogue.cli.query_plan import QueryAction, QueryPlanError, build_query_execution_plan


class TestBuildQueryExecutionPlan:
    def test_delete_without_filters_raises(self) -> None:
        with pytest.raises(QueryPlanError, match="--delete requires at least one filter"):
            build_query_execution_plan({"delete_matched": True, "query": ()})

    @pytest.mark.parametrize(
        ("params", "expected_action"),
        [
            ({"count_only": True, "query": ()}, QueryAction.COUNT),
            ({"stream": True, "query": ("abc",)}, QueryAction.STREAM),
            ({"stats_only": True, "query": ()}, QueryAction.STATS),
            ({"stats_by": "provider", "query": ()}, QueryAction.STATS_BY),
            ({"add_tag": ["x"], "query": ()}, QueryAction.MODIFY),
            ({"delete_matched": True, "provider": "claude", "query": ()}, QueryAction.DELETE),
            ({"open_result": True, "query": ("abc",)}, QueryAction.OPEN),
            ({"query": ("abc",)}, QueryAction.SHOW),
        ],
    )
    def test_action_selection(self, params: dict[str, object], expected_action: QueryAction) -> None:
        plan = build_query_execution_plan(params)
        assert plan.action == expected_action

    def test_stream_format_converts_json_to_json_lines(self) -> None:
        plan = build_query_execution_plan({"stream": True, "output_format": "json", "query": ("abc",)})
        assert plan.output.stream_format() == "json-lines"

    def test_summary_list_preference_requires_plain_listing_shape(self) -> None:
        plan = build_query_execution_plan({"list_mode": True, "query": ("abc",)})
        assert plan.prefers_summary_list() is True

        transformed = build_query_execution_plan({"list_mode": True, "transform": "strip-tools", "query": ("abc",)})
        assert transformed.prefers_summary_list() is False

    def test_mutation_fields_are_normalized(self) -> None:
        plan = build_query_execution_plan(
            {
                "set_meta": [("priority", 3)],
                "add_tag": ["todo", "review"],
                "force": True,
                "dry_run": True,
                "provider": "claude",
                "query": (),
            }
        )
        assert plan.mutation.set_meta == (("priority", "3"),)
        assert plan.mutation.add_tags == ("todo", "review")
        assert plan.mutation.force is True
        assert plan.mutation.dry_run is True
