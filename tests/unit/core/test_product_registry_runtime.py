# mypy: disable-error-code="arg-type,comparison-overlap,attr-defined,index,list-item"

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from polylogue.products import registry as product_registry
from polylogue.products.archive import ProviderAnalyticsProduct
from polylogue.products.registry import (
    CliOption,
    ProductField,
    ProductQueryError,
    ProductType,
    fetch_products,
    fetch_products_async,
    get_product_type,
    list_product_types,
    product_items_payload,
    register,
    render_product_items,
)


def _provider_analytics_product() -> ProviderAnalyticsProduct:
    return ProviderAnalyticsProduct(
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


def test_product_type_registry_helpers_cover_register_lookup_and_sorting() -> None:
    name = "runtime_dummy_product"
    product_type = ProductType(
        name=name,
        display_name="Runtime Dummy",
        json_key="items",
        cli_options=(CliOption("flag", ("--flag",), help="flag"),),
    )

    try:
        returned = register(product_type)
        assert returned is product_type
        assert get_product_type(name) is product_type
        assert name in list_product_types()
        assert product_type.resolved_cli_command_name == "runtime-dummy-product"
    finally:
        product_registry.PRODUCT_REGISTRY.pop(name, None)


def test_get_product_type_and_build_query_raise_useful_errors() -> None:
    with pytest.raises(KeyError, match="Unknown product type"):
        get_product_type("missing-product")

    with pytest.raises(ProductQueryError, match="does not declare a query model"):
        product_registry._build_query(
            ProductType(name="dummy", display_name="Dummy", json_key="items"),
            query="value",
        )

    with pytest.raises(ProductQueryError, match="Unknown query field\\(s\\) for session_profiles: refined_work_kind"):
        product_registry._build_query(get_product_type("session_profiles"), refined_work_kind="planning")


def test_product_items_payload_and_rendering_cover_json_plain_and_empty_paths() -> None:
    product = _provider_analytics_product()
    product_type = get_product_type("provider_analytics")

    payload = product_items_payload([product], product_type, item_key="items")
    assert payload["count"] == 1
    assert payload["items"][0]["provider_name"] == "claude-code"

    with patch("polylogue.cli.machine_errors.emit_success") as mock_emit:
        render_product_items([product], product_type, json_mode=True)
    mock_emit.assert_called_once_with(product_items_payload([product], product_type))

    custom_type = ProductType(
        name="custom",
        display_name="Custom",
        json_key="items",
        empty_message="No custom items.",
        fields=(
            ProductField("", lambda item: item.name),
            ProductField("missing", lambda _item: (_ for _ in ()).throw(AttributeError("boom")), group=1),
        ),
    )
    item = SimpleNamespace(name="item-1")

    with patch("click.echo") as mock_echo:
        render_product_items([], custom_type)
        render_product_items([item], custom_type)

    assert mock_echo.call_args_list[0].args == ("No custom items.",)
    assert mock_echo.call_args_list[1].args == ("Custom: 1\n",)
    assert mock_echo.call_args_list[2].args == ("  item-1",)
    assert mock_echo.call_args_list[3].args == ("    missing=-",)


def test_fetch_products_sync_uses_registry_dispatch() -> None:
    product_type = get_product_type("provider_analytics")

    class _Operations:
        def list_provider_analytics_products(self, query: object) -> str:
            return f"sync:{query.provider}"

    with patch("polylogue.api.sync.bridge.run_coroutine_sync", side_effect=lambda value: [value]):
        assert fetch_products(product_type, _Operations(), provider="claude-code") == ["sync:claude-code"]


@pytest.mark.asyncio
async def test_fetch_products_async_uses_registry_dispatch() -> None:
    product_type = get_product_type("provider_analytics")

    class _AsyncOperations:
        async def list_provider_analytics_products(self, query: object) -> list[str]:
            return [f"async:{query.provider}"]

    assert await fetch_products_async(product_type, _AsyncOperations(), provider="claude-code") == ["async:claude-code"]
