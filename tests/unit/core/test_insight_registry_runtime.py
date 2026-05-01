# mypy: disable-error-code="arg-type,comparison-overlap,attr-defined,index,list-item"

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from polylogue.insights import registry as product_registry
from polylogue.insights.archive import ProviderAnalyticsInsight
from polylogue.insights.registry import (
    CliOption,
    InsightField,
    InsightQueryError,
    InsightType,
    fetch_insights,
    fetch_insights_async,
    get_insight_type,
    insight_items_payload,
    list_insight_types,
    register,
    render_insight_items,
)


def _provider_analytics_product() -> ProviderAnalyticsInsight:
    return ProviderAnalyticsInsight(
        provider_name="claude-code",
        conversation_count=1,
        message_count=2,
        user_message_count=1,
        assistant_message_count=1,
        avg_messages_per_conversation=2.0,
        avg_user_words=3.0,
        avg_assistant_words=4.0,
        tool_use_count=1,
        thinking_count=0,
        total_conversations_with_tools=1,
        total_conversations_with_thinking=0,
        tool_use_percentage=100.0,
        thinking_percentage=0.0,
    )


def test_registry_accessors_format_values_and_defaults() -> None:
    item = SimpleNamespace(
        provider_name="claude-code",
        thread_id="thread-1",
        nested=SimpleNamespace(value="nested-value"),
        values=("a", "b", "c", "d"),
        ratio=12.345,
        count=7,
        percentage=82.5,
    )

    assert product_registry._stringify(None) == "-"
    assert product_registry._stringify("") == "-"
    assert product_registry._attr("thread_id")(item) == "thread-1"
    assert product_registry._nested("nested", "value")(item) == "nested-value"
    assert product_registry._nested("missing", "value")(item) == "-"
    assert product_registry._id_with_provider("thread_id")(item) == "thread-1 [claude-code]"
    assert product_registry._list_preview("values", limit=2)(item) == "a, b"
    assert product_registry._formatted_float("ratio", precision=2)(item) == "12.35"
    assert product_registry._formatted_float("missing")(item) == "-"
    assert product_registry._count_with_percentage("count", "percentage")(item) == "7 (82.5%)"
    assert product_registry._count_with_percentage("missing", "percentage")(item) == "-"


def test_insight_type_registry_helpers_cover_register_lookup_and_sorting() -> None:
    name = "runtime_dummy_product"
    insight_type = InsightType(
        name=name,
        display_name="Runtime Dummy",
        json_key="items",
        cli_options=(CliOption("flag", ("--flag",), help="flag"),),
    )

    try:
        returned = register(insight_type)
        assert returned is insight_type
        assert get_insight_type(name) is insight_type
        assert name in list_insight_types()
        assert insight_type.resolved_cli_command_name == "runtime-dummy-product"
    finally:
        product_registry.INSIGHT_REGISTRY.pop(name, None)


def test_get_insight_type_and_build_query_raise_useful_errors() -> None:
    with pytest.raises(KeyError, match="Unknown product type"):
        get_insight_type("missing-product")

    with pytest.raises(InsightQueryError, match="does not declare a query model"):
        product_registry._build_query(
            InsightType(name="dummy", display_name="Dummy", json_key="items"),
            query="value",
        )

    with pytest.raises(InsightQueryError, match="Unknown query field\\(s\\) for session_profiles: refined_work_kind"):
        product_registry._build_query(get_insight_type("session_profiles"), refined_work_kind="planning")


def test_insight_items_payload_and_rendering_cover_json_plain_and_empty_paths() -> None:
    product = _provider_analytics_product()
    insight_type = get_insight_type("provider_analytics")

    payload = insight_items_payload([product], insight_type, item_key="items")
    assert payload["count"] == 1
    assert payload["items"][0]["provider_name"] == "claude-code"

    with patch("polylogue.cli.shared.machine_errors.emit_success") as mock_emit:
        render_insight_items([product], insight_type, json_mode=True)
    mock_emit.assert_called_once_with(insight_items_payload([product], insight_type))

    custom_type = InsightType(
        name="custom",
        display_name="Custom",
        json_key="items",
        empty_message="No custom items.",
        fields=(
            InsightField("", lambda item: item.name),
            InsightField("missing", lambda _item: (_ for _ in ()).throw(AttributeError("boom")), group=1),
        ),
    )
    item = SimpleNamespace(name="item-1")

    with patch("click.echo") as mock_echo:
        render_insight_items([], custom_type)
        render_insight_items([item], custom_type)

    assert mock_echo.call_args_list[0].args == ("No custom items.",)
    assert mock_echo.call_args_list[1].args == ("Custom: 1\n",)
    assert mock_echo.call_args_list[2].args == ("  item-1",)
    assert mock_echo.call_args_list[3].args == ("    missing=-",)


def test_fetch_insights_sync_uses_registry_dispatch() -> None:
    insight_type = get_insight_type("provider_analytics")

    class _Operations:
        def list_provider_analytics_products(self, query: object) -> str:
            return f"sync:{query.provider}"

    with patch("polylogue.api.sync.bridge.run_coroutine_sync", side_effect=lambda value: [value]):
        assert fetch_insights(insight_type, _Operations(), provider="claude-code") == ["sync:claude-code"]


@pytest.mark.asyncio
async def test_fetch_insights_async_uses_registry_dispatch() -> None:
    insight_type = get_insight_type("provider_analytics")

    class _AsyncOperations:
        async def list_provider_analytics_products(self, query: object) -> list[str]:
            return [f"async:{query.provider}"]

    assert await fetch_insights_async(insight_type, _AsyncOperations(), provider="claude-code") == ["async:claude-code"]
