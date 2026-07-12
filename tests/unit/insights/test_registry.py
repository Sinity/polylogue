"""Tests for polylogue.insights.registry — pure function coverage."""

from __future__ import annotations

import types
from typing import Any, cast

import pytest

from polylogue.insights.archive import (
    ArchiveCoverageInsightQuery,
    SessionProfileInsight,
    SessionProfileInsightQuery,
)
from polylogue.insights.archive_models import (
    ArchiveInsightProvenance,
)
from polylogue.insights.registry import (
    INSIGHT_REGISTRY,
    InsightQueryError,
    InsightType,
    _attr,
    _build_query,
    _count_with_percentage,
    _formatted_float,
    _id_with_origin,
    _list_preview,
    _nested,
    _nested_ms_as_seconds,
    _source_name_origin,
    _stringify,
    get_insight_type,
    insight_items_payload,
    list_insight_types,
    project_origin_payload,
    register,
    render_insight_items,
)


class TestProjectOriginPayload:
    """Test project_origin_payload() — origin vocabulary projection."""

    def test_scalar_passthrough(self) -> None:
        """Non-dict/list/tuple scalars pass through unchanged."""
        assert project_origin_payload(42) == 42
        assert project_origin_payload("test") == "test"
        assert project_origin_payload(None) is None
        assert project_origin_payload(3.14) == 3.14

    def test_list_recursion(self) -> None:
        """Lists are recursed element-wise."""
        result_list = project_origin_payload([{"source_name": "claude-code-session"}, 123])
        assert isinstance(result_list, list)
        assert len(result_list) == 2
        assert result_list[0] == {"origin": "claude-code-session"}
        assert result_list[1] == 123

    def test_tuple_becomes_list(self) -> None:
        """Tuples become lists after recursion."""
        result = project_origin_payload(({"provider": "CLAUDE"}, "scalar"))
        assert isinstance(result, list)
        assert len(result) == 2

    def test_source_name_to_origin(self) -> None:
        """Dict with source_name key → origin key."""
        result = project_origin_payload({"source_name": "claude-code-session", "x": 1})
        assert result == {"origin": "claude-code-session", "x": 1}

    def test_provider_to_origin(self) -> None:
        """Dict with provider key → origin key."""
        result = project_origin_payload({"provider": "CLAUDE", "count": 5})
        assert result == {"origin": "claude-ai-export", "count": 5}

    def test_provider_coverage_projects(self) -> None:
        """Dict with provider_coverage → origin_coverage, recursed."""
        result = project_origin_payload({"provider_coverage": {"provider": "CLAUDE", "count": 3}})
        assert result == {"origin_coverage": {"origin": "claude-ai-export", "count": 3}}

    def test_provider_breakdown_dict_projects(self) -> None:
        """provider_breakdown dict keys are origin-projected."""
        result = project_origin_payload({"provider_breakdown": {"CLAUDE": 10, "CHATGPT": 5}})
        result_dict = cast(dict[str, object], result)
        breakdown = cast(dict[str, object], result_dict["origin_breakdown"])
        assert isinstance(breakdown, dict)
        assert breakdown["claude-ai-export"] == 10
        assert breakdown["chatgpt-export"] == 5

    def test_origin_breakdown_dict_passes_through(self) -> None:
        """origin_breakdown keeps canonical origin tokens."""
        result = project_origin_payload({"origin_breakdown": {"claude-code-session": 10}})
        assert result == {"origin_breakdown": {"claude-code-session": 10}}

    def test_providers_with_data_to_origins_with_data(self) -> None:
        """providers_with_data → origins_with_data."""
        result = project_origin_payload({"providers_with_data": ["CLAUDE", "CHATGPT"]})
        assert result == {"origins_with_data": ["CLAUDE", "CHATGPT"]}

    def test_providers_without_data_to_origins_without_data(self) -> None:
        """providers_without_data → origins_without_data."""
        result = project_origin_payload({"providers_without_data": ["GEMINI"]})
        assert result == {"origins_without_data": ["GEMINI"]}

    def test_group_by_provider_to_origin(self) -> None:
        """group_by == 'provider' becomes 'origin'."""
        result = project_origin_payload({"group_by": "provider", "other": "data"})
        result_dict = cast(dict[str, object], result)
        assert result_dict["group_by"] == "origin"
        assert result_dict["other"] == "data"

    def test_group_by_provider_also_projects_bucket(self) -> None:
        """When group_by was 'provider', bucket key is also origin-projected."""
        result = project_origin_payload({"group_by": "provider", "bucket": "CLAUDE"})
        result_dict = cast(dict[str, object], result)
        assert result_dict["group_by"] == "origin"
        assert result_dict["bucket"] == "claude-ai-export"

    def test_group_by_other_values_unchanged(self) -> None:
        """group_by values other than 'provider' pass through."""
        result = project_origin_payload({"group_by": "day"})
        result_dict = cast(dict[str, object], result)
        assert result_dict["group_by"] == "day"

    def test_nested_recursion(self) -> None:
        """Nested structures are recursed."""
        result = project_origin_payload(
            {
                "data": [{"provider": "CLAUDE"}],
                "meta": {"source_name": "codex-session"},
            }
        )
        result_dict = cast(dict[str, object], result)
        assert result_dict["data"] == [{"origin": "claude-ai-export"}]
        assert result_dict["meta"] == {"origin": "codex-session"}


class TestInsightItemsPayload:
    """Test insight_items_payload() — payload envelope contract."""

    def test_empty_items_envelope(self) -> None:
        """Empty items list returns {total: 0, json_key: []}."""
        insight_type = get_insight_type("session_profiles")
        result = insight_items_payload([], insight_type)
        assert result == {"total": 0, "session_profiles": []}

    def test_total_field_not_count(self) -> None:
        """Critical contract: field is 'total', NOT 'count' (per #1007)."""
        insight_type = get_insight_type("threads")
        result = insight_items_payload([], insight_type)
        assert "total" in result
        assert "count" not in result

    def test_custom_item_key(self) -> None:
        """item_key parameter overrides default json_key."""
        insight_type = get_insight_type("session_profiles")
        result = insight_items_payload([], insight_type, item_key="custom_items")
        assert "custom_items" in result
        assert "session_profiles" not in result
        assert result["total"] == 0

    def test_items_are_serialized(self) -> None:
        """Items are serialized via model_dump(mode='json') and origin-projected."""
        # Build a minimal SessionProfileInsight
        insight = SessionProfileInsight(
            session_id="sess-123",
            logical_session_id="logical-456",
            source_name="claude-code-session",
            title="Test Session",
            provenance=ArchiveInsightProvenance(
                materializer_version=1,
                materialized_at="2026-01-01T00:00:00Z",
                input_high_water_mark_source="sort_key",
                time_confidence="estimated",
            ),
        )
        insight_type = get_insight_type("session_profiles")
        result = insight_items_payload([insight], insight_type)
        assert result["total"] == 1
        items_list = cast(list[dict[str, object]], result["session_profiles"])
        assert len(items_list) == 1
        item_dict = items_list[0]
        # Verify origin projection happened
        assert item_dict["origin"] == "claude-code-session"
        provenance = cast(dict[str, object], item_dict["provenance"])
        assert provenance["time_confidence"] == "estimated"


class TestRegistryOperations:
    """Test register(), get_insight_type(), list_insight_types()."""

    def test_list_insight_types_sorted(self) -> None:
        """list_insight_types() returns sorted list of registered names."""
        names = list_insight_types()
        assert isinstance(names, list)
        assert names == sorted(names)
        # Verify known types are present
        assert "session_profiles" in names
        assert "threads" in names
        assert "archive_coverage" in names

    def test_get_insight_type_success(self) -> None:
        """get_insight_type() returns registered InsightType by name."""
        insight = get_insight_type("session_profiles")
        assert insight.name == "session_profiles"
        assert insight.display_name == "Session Profiles"
        assert insight.json_key == "session_profiles"

    def test_get_insight_type_unknown_raises_keyerror(self) -> None:
        """get_insight_type() raises KeyError for unknown name, listing available."""
        with pytest.raises(KeyError) as exc_info:
            get_insight_type("nonexistent-type")
        error_msg = str(exc_info.value)
        assert "Unknown insight type" in error_msg
        assert "nonexistent-type" in error_msg
        assert "Available" in error_msg

    def test_register_new_type_and_retrieve(self) -> None:
        """register() adds to INSIGHT_REGISTRY; get_insight_type() retrieves it."""
        # Use a unique name to avoid conflicts
        unique_name = "test_unique_registry_type_xyz"
        new_type = InsightType(
            name=unique_name,
            display_name="Unique Test Type",
            json_key=unique_name,
        )
        # Clean up if it somehow exists
        INSIGHT_REGISTRY.pop(unique_name, None)

        # Register and retrieve
        returned = register(new_type)
        assert returned is new_type
        assert get_insight_type(unique_name) is new_type
        assert unique_name in list_insight_types()

        # Clean up
        INSIGHT_REGISTRY.pop(unique_name, None)


class TestInsightTypeResolvedCliCommandName:
    """Test InsightType.resolved_cli_command_name property."""

    def test_when_cli_command_name_set(self) -> None:
        """When cli_command_name is set, returns it."""
        insight = InsightType(
            name="my_type",
            display_name="My Type",
            json_key="my_type",
            cli_command_name="custom-command",
        )
        assert insight.resolved_cli_command_name == "custom-command"

    def test_when_cli_command_name_empty(self) -> None:
        """When cli_command_name is empty, returns name with underscores → dashes."""
        insight = InsightType(
            name="my_insight_type",
            display_name="My Insight Type",
            json_key="my_insight_type",
            cli_command_name="",
        )
        assert insight.resolved_cli_command_name == "my-insight-type"

    def test_when_cli_command_name_not_set(self) -> None:
        """When cli_command_name is not set (default), returns name with conversion."""
        insight = InsightType(
            name="another_test_name",
            display_name="Another Test",
            json_key="another",
        )
        assert insight.resolved_cli_command_name == "another-test-name"


class TestFieldAccessors:
    """Test field accessor factory functions."""

    def _make_item(self, **attrs: object) -> object:
        """Helper to create a test item with arbitrary attributes."""
        return types.SimpleNamespace(**attrs)

    def test_attr_basic(self) -> None:
        """_attr() reads attribute by name."""
        accessor = _attr("name")
        item = self._make_item(name="Alice")
        assert accessor(item) == "Alice"  # type: ignore[arg-type]

    def test_attr_missing_uses_default(self) -> None:
        """_attr() returns default when attribute is missing."""
        accessor = _attr("name", default="MISSING")
        item = self._make_item(other="value")
        assert accessor(item) == "MISSING"  # type: ignore[arg-type]

    def test_attr_none_uses_default(self) -> None:
        """_attr() returns default when attribute is None."""
        accessor = _attr("name", default="NULL")
        item = self._make_item(name=None)
        assert accessor(item) == "NULL"  # type: ignore[arg-type]

    def test_attr_empty_string_uses_default(self) -> None:
        """_attr() returns default when attribute is empty string."""
        accessor = _attr("name", default="EMPTY")
        item = self._make_item(name="")
        assert accessor(item) == "EMPTY"  # type: ignore[arg-type]

    def test_nested_reads_nested_attribute(self) -> None:
        """_nested() reads item.outer.inner."""
        accessor = _nested("data", "value")
        data = types.SimpleNamespace(value="nested_val")
        item = self._make_item(data=data)
        assert accessor(item) == "nested_val"  # type: ignore[arg-type]

    def test_nested_outer_none_uses_default(self) -> None:
        """_nested() returns default when outer is None."""
        accessor = _nested("data", "value", default="NONE")
        item = self._make_item(data=None)
        assert accessor(item) == "NONE"  # type: ignore[arg-type]

    def test_nested_inner_missing_uses_default(self) -> None:
        """_nested() returns default when inner is missing."""
        accessor = _nested("data", "value", default="MISSING")
        data = types.SimpleNamespace(other="val")
        item = self._make_item(data=data)
        assert accessor(item) == "MISSING"  # type: ignore[arg-type]

    def test_nested_ms_as_seconds_converts(self) -> None:
        """_nested_ms_as_seconds() converts ms to seconds (floor div 1000)."""
        accessor = _nested_ms_as_seconds("timing", "duration_ms")
        timing = types.SimpleNamespace(duration_ms=5000)
        item = self._make_item(timing=timing)
        assert accessor(item) == "5"  # type: ignore[arg-type]

    def test_nested_ms_as_seconds_floor(self) -> None:
        """_nested_ms_as_seconds() floors division."""
        accessor = _nested_ms_as_seconds("timing", "duration_ms")
        timing = types.SimpleNamespace(duration_ms=1234)
        item = self._make_item(timing=timing)
        assert accessor(item) == "1"  # type: ignore[arg-type]

    def test_nested_ms_as_seconds_negative_clamped(self) -> None:
        """_nested_ms_as_seconds() clamps negative values to 0."""
        accessor = _nested_ms_as_seconds("timing", "duration_ms")
        timing = types.SimpleNamespace(duration_ms=-100)
        item = self._make_item(timing=timing)
        assert accessor(item) == "0"  # type: ignore[arg-type]

    def test_nested_ms_as_seconds_non_int_returns_default(self) -> None:
        """_nested_ms_as_seconds() returns default for non-int values."""
        accessor = _nested_ms_as_seconds("timing", "duration_ms", default="ERROR")
        timing = types.SimpleNamespace(duration_ms="not_an_int")
        item = self._make_item(timing=timing)
        assert accessor(item) == "ERROR"  # type: ignore[arg-type]

    def test_id_with_origin_canonical(self) -> None:
        """_id_with_origin() formats 'id [origin]' with canonical origin."""
        accessor = _id_with_origin("session_id")
        item = self._make_item(session_id="sess-123", source_name="claude-code-session")
        assert accessor(item) == "sess-123 [claude-code-session]"  # type: ignore[arg-type]

    def test_id_with_origin_projected(self) -> None:
        """_id_with_origin() projects provider to origin."""
        accessor = _id_with_origin("event_id")
        item = self._make_item(event_id="ev-456", source_name="CLAUDE")
        result = accessor(item)  # type: ignore[arg-type]
        assert "ev-456" in result
        assert "claude-ai-export" in result

    def test_id_with_origin_unknown_source(self) -> None:
        """_id_with_origin() uses 'unknown-export' when source_name is unmapped."""
        accessor = _id_with_origin("id")
        item = self._make_item(id="test-id", source_name="nonexistent")
        assert accessor(item) == "test-id [unknown-export]"  # type: ignore[arg-type]

    def test_list_preview_basic(self) -> None:
        """_list_preview() shows first N items joined by ', '."""
        accessor = _list_preview("tags", limit=2)
        item = self._make_item(tags=["tag1", "tag2", "tag3"])
        assert accessor(item) == "tag1, tag2"  # type: ignore[arg-type]

    def test_list_preview_fewer_than_limit(self) -> None:
        """_list_preview() joins all items when fewer than limit."""
        accessor = _list_preview("tags", limit=5)
        item = self._make_item(tags=["a", "b"])
        assert accessor(item) == "a, b"  # type: ignore[arg-type]

    def test_list_preview_empty_list_returns_dash(self) -> None:
        """_list_preview() returns '-' for empty list."""
        accessor = _list_preview("tags")
        item = self._make_item(tags=[])
        assert accessor(item) == "-"  # type: ignore[arg-type]

    def test_list_preview_tuple_works(self) -> None:
        """_list_preview() works with tuples too."""
        accessor = _list_preview("items", limit=2)
        item = self._make_item(items=("x", "y", "z"))
        assert accessor(item) == "x, y"  # type: ignore[arg-type]

    def test_list_preview_non_list_stringified(self) -> None:
        """_list_preview() stringifies non-list/tuple values."""
        accessor = _list_preview("value")
        item = self._make_item(value=42)
        assert accessor(item) == "42"  # type: ignore[arg-type]

    def test_formatted_float_basic(self) -> None:
        """_formatted_float() formats with precision."""
        accessor = _formatted_float("ratio", precision=2)
        item = self._make_item(ratio=3.14159)
        assert accessor(item) == "3.14"  # type: ignore[arg-type]

    def test_formatted_float_int_converted(self) -> None:
        """_formatted_float() works with int values too."""
        accessor = _formatted_float("count", precision=1)
        item = self._make_item(count=42)
        assert accessor(item) == "42.0"  # type: ignore[arg-type]

    def test_formatted_float_bool_returns_default(self) -> None:
        """_formatted_float() returns default for bool values."""
        accessor = _formatted_float("flag", default="BOOL")
        item = self._make_item(flag=True)
        assert accessor(item) == "BOOL"  # type: ignore[arg-type]

    def test_formatted_float_non_numeric_returns_default(self) -> None:
        """_formatted_float() returns default for non-numeric."""
        accessor = _formatted_float("value", default="NAN")
        item = self._make_item(value="not_a_number")
        assert accessor(item) == "NAN"  # type: ignore[arg-type]

    def test_count_with_percentage_both_present(self) -> None:
        """_count_with_percentage() formats 'count (pct%)' when both present."""
        accessor = _count_with_percentage("count", "percent")
        item = self._make_item(count=50, percent=75.5)
        assert accessor(item) == "50 (75.5%)"  # type: ignore[arg-type]

    def test_count_with_percentage_count_only(self) -> None:
        """_count_with_percentage() shows count alone if percentage missing/non-numeric."""
        accessor = _count_with_percentage("count", "percent")
        item = self._make_item(count=30, percent=None)
        assert accessor(item) == "30"  # type: ignore[arg-type]

    def test_count_with_percentage_count_invalid_returns_dash(self) -> None:
        """_count_with_percentage() returns '-' if count is invalid."""
        accessor = _count_with_percentage("count", "percent")
        item = self._make_item(count="not_int", percent=50.0)
        assert accessor(item) == "-"  # type: ignore[arg-type]

    def test_count_with_percentage_bool_count_invalid(self) -> None:
        """_count_with_percentage() treats bool count as invalid."""
        accessor = _count_with_percentage("count", "percent")
        item = self._make_item(count=True, percent=50.0)
        assert accessor(item) == "-"  # type: ignore[arg-type]


class TestStringify:
    """Test _stringify() helper."""

    def test_none_returns_default(self) -> None:
        """_stringify(None) returns default."""
        assert _stringify(None) == "-"
        assert _stringify(None, "NULL") == "NULL"

    def test_empty_string_returns_default(self) -> None:
        """_stringify('') returns default."""
        assert _stringify("") == "-"
        assert _stringify("", "EMPTY") == "EMPTY"

    def test_non_empty_string_returned(self) -> None:
        """_stringify(str) returns the string."""
        assert _stringify("hello") == "hello"

    def test_other_types_stringified(self) -> None:
        """_stringify(non-str) converts to string."""
        assert _stringify(42) == "42"
        assert _stringify(3.14) == "3.14"


class TestSourceNameOrigin:
    """Test _source_name_origin() helper."""

    def test_canonical_origin_values_pass_through(self) -> None:
        """Canonical Origin enum values pass through unchanged."""
        assert _source_name_origin("claude-code-session") == "claude-code-session"
        assert _source_name_origin("chatgpt-export") == "chatgpt-export"

    def test_provider_enum_mapped(self) -> None:
        """Provider enum values are mapped to origins."""
        result = _source_name_origin("CLAUDE")
        assert result == "claude-ai-export"

    def test_empty_string_returns_unknown(self) -> None:
        """_source_name_origin('') returns 'unknown'."""
        assert _source_name_origin("") == "unknown"

    def test_none_returns_unknown(self) -> None:
        """_source_name_origin(None) returns 'unknown'."""
        assert _source_name_origin(None) == "unknown"

    def test_unmapped_value_returns_unknown_export(self) -> None:
        """Unmappable values canonicalize to Provider.UNKNOWN → Origin.UNKNOWN_EXPORT."""
        assert _source_name_origin("totally-invalid-provider") == "unknown-export"

    def test_provider_string_case_insensitive(self) -> None:
        """Provider strings are normalized (case-insensitive, alias-aware)."""
        # "Claude" normalizes to "claude" → aliased to "claude-ai" → Origin.CLAUDE_AI_EXPORT
        result = _source_name_origin("Claude")
        assert result == "claude-ai-export"


class TestRenderInsightItems:
    """Test render_insight_items() output."""

    def test_json_mode_with_empty_items(self, capsys: Any) -> None:
        """json_mode=True with empty items emits JSON success envelope."""
        insight_type = get_insight_type("session_profiles")
        render_insight_items([], insight_type, json_mode=True)
        captured = capsys.readouterr()
        assert captured.out  # Should have output
        # Should contain the structure from emit_success
        assert "session_profiles" in captured.out or "total" in captured.out

    def test_plain_mode_empty_items_shows_message(self, capsys: Any) -> None:
        """json_mode=False with empty items echoes empty_message."""
        insight_type = get_insight_type("threads")
        render_insight_items([], insight_type, json_mode=False)
        captured = capsys.readouterr()
        assert insight_type.empty_message in captured.out

    def test_plain_mode_with_items_shows_header_and_fields(self, capsys: Any) -> None:
        """json_mode=False with items shows display_name, count, and field lines."""
        # Create minimal SessionProfileInsight
        insight = SessionProfileInsight(
            session_id="sess-001",
            logical_session_id="logical-001",
            source_name="claude-code-session",
            title="Test Session",
            provenance=ArchiveInsightProvenance(
                materializer_version=1,
                materialized_at="2026-01-01T00:00:00Z",
            ),
        )
        insight_type = get_insight_type("session_profiles")
        render_insight_items([insight], insight_type, json_mode=False)
        captured = capsys.readouterr()

        # Should show display name and count
        assert "Session Profiles: 1" in captured.out
        # Should include some field output
        assert "sess-001" in captured.out


class TestBuildQuery:
    """Test _build_query() query validation."""

    def test_build_query_no_query_model_raises(self) -> None:
        """_build_query() raises InsightQueryError when query_model is None."""
        insight_type = InsightType(
            name="no_query",
            display_name="No Query",
            json_key="no_query",
            query_model=None,
        )
        with pytest.raises(InsightQueryError) as exc_info:
            _build_query(insight_type)
        assert "does not declare a query model" in str(exc_info.value)

    def test_build_query_unknown_field_raises(self) -> None:
        """_build_query() raises InsightQueryError for unknown query fields."""
        insight_type = get_insight_type("session_profiles")
        with pytest.raises(InsightQueryError) as exc_info:
            _build_query(insight_type, unknown_field="value")
        error_msg = str(exc_info.value)
        assert "Unknown query field" in error_msg
        assert "unknown_field" in error_msg
        assert "Accepted fields" in error_msg

    def test_build_query_valid_fields_succeed(self) -> None:
        """_build_query() succeeds with valid fields."""

        insight_type = get_insight_type("session_profiles")
        query = _build_query(
            insight_type,
            tier="merged",
            limit=10,
        )
        query_obj = cast(SessionProfileInsightQuery, query)
        assert query_obj.tier == "merged"
        assert query_obj.limit == 10

    def test_build_query_returns_correct_type(self) -> None:
        """_build_query() returns instance of query_model."""
        insight_type = get_insight_type("archive_coverage")
        query = _build_query(insight_type, group_by="day")
        assert isinstance(query, ArchiveCoverageInsightQuery)
