# mypy: disable-error-code="no-untyped-def,call-arg,arg-type,attr-defined"

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import click
import pytest

from polylogue.cli.commands import products as products_module
from polylogue.product_export_bundles import (
    ProductExportBundleError,
    ProductExportBundleManifest,
    ProductExportBundleResult,
    ProductExportFileSummary,
)
from polylogue.product_readiness import (
    ProductProviderCoverage,
    ProductReadinessEntry,
    ProductReadinessReport,
    ProductVersionCoverage,
)
from polylogue.products.registry import CliOption, ProductQueryError, ProductType, get_product_type


def _root_context(*, output_format: str | None = None, provider: str | None = None) -> click.Context:
    root = click.Context(click.Command("polylogue"))
    root.params = {
        "output_format": output_format,
        "provider": provider,
        "since": "2026-04-01",
        "until": "2026-04-30",
    }
    return root


def _status_context(env: object, *, output_format: str | None = None, provider: str | None = None) -> click.Context:
    ctx = click.Context(
        products_module.products_status_command, parent=_root_context(output_format=output_format, provider=provider)
    )
    ctx.obj = env
    return ctx


def _export_context(env: object, *, output_format: str | None = None, provider: str | None = None) -> click.Context:
    ctx = click.Context(
        products_module.products_export_command, parent=_root_context(output_format=output_format, provider=provider)
    )
    ctx.obj = env
    return ctx


def _command_callback(command: click.Command) -> Callable[..., object]:
    callback = getattr(command.callback, "__wrapped__", command.callback)
    assert callback is not None
    return callback


def _status_report() -> ProductReadinessReport:
    return ProductReadinessReport(
        checked_at="2026-04-23T00:00:00+00:00",
        aggregate_verdict="partial",
        total_conversations=10,
        provider="codex",
        since="2026-04-01",
        until="2026-04-30",
        products=(
            ProductReadinessEntry(
                product_name="session_profiles",
                display_name="Session Profiles",
                verdict="partial",
                row_count=7,
                expected_row_count=10,
                missing_count=1,
                stale_count=2,
                orphan_count=3,
                legacy_incompatible_count=4,
                ready_flags={"fts": True},
                provider_coverage=(ProductProviderCoverage(provider_name="codex", row_count=7),),
                version_coverage=(
                    ProductVersionCoverage(field="materializer_version", current_version=4, versions={"4": 7}),
                ),
                schema_contract_issues=("missing field",),
            ),
        ),
    )


def _export_result(tmp_path: Path) -> ProductExportBundleResult:
    return ProductExportBundleResult(
        output_path=tmp_path / "bundle",
        manifest_path=tmp_path / "bundle" / "manifest.json",
        coverage_path=tmp_path / "bundle" / "coverage.json",
        manifest=ProductExportBundleManifest(
            generated_at="2026-04-23T00:00:00+00:00",
            polylogue_version="1.0.0",
            archive_root="/tmp/archive",
            database_path="/tmp/archive/polylogue.db",
            query={"provider": "codex"},
            products=(
                ProductExportFileSummary(
                    product_name="session_profiles",
                    file="products/session_profiles.jsonl",
                    schema_file="schemas/session_profiles.schema.json",
                    row_count=7,
                    readiness_verdict="partial",
                    warnings=("stale rows",),
                    errors=("schema drift",),
                ),
            ),
        ),
    )


def test_build_click_params_and_product_command_cover_dynamic_registration() -> None:
    product_type = ProductType(
        name="test_product",
        display_name="Test Product",
        json_key="items",
        cli_help="List test products.",
        cli_options=(CliOption("provider", ("--provider",), help="Provider", type=str, default=None),),
        mcp_default_limit=25,
    )

    params = products_module._build_click_params(product_type)
    command = products_module._build_product_command(product_type)

    assert [param.name for param in params] == ["provider", "limit", "offset", "json_mode", "output_format"]
    assert command.name == "test-product"
    assert command.help == "List test products."


def test_make_callback_renders_products_and_surfaces_query_errors() -> None:
    callback = products_module._make_callback(get_product_type("session_profiles"))
    raw_callback = getattr(callback, "__wrapped__", callback)
    env = SimpleNamespace(operations=MagicMock())
    ctx = click.Context(click.Command("profiles"))
    ctx.obj = env

    request = SimpleNamespace(query_kwargs={"limit": 1}, wants_json=True)
    with patch("polylogue.cli.commands.products.ProductCommandRequest.from_context", return_value=request):
        with patch("polylogue.cli.commands.products.fetch_products", return_value=["row"]) as fetch_products:
            with patch("polylogue.cli.commands.products.render_product_items") as render_items:
                raw_callback(ctx, json_mode=True, output_format=None)

    fetch_products.assert_called_once()
    render_items.assert_called_once_with(["row"], get_product_type("session_profiles"), json_mode=True)

    with patch("polylogue.cli.commands.products.ProductCommandRequest.from_context", return_value=request):
        with patch("polylogue.cli.commands.products.fetch_products", side_effect=ProductQueryError("bad query")):
            with pytest.raises(SystemExit, match="products profiles: bad query"):
                raw_callback(ctx, json_mode=False, output_format=None)


def test_status_wants_json_checks_command_and_root_flags() -> None:
    ctx = click.Context(click.Command("status"), parent=_root_context(output_format="json"))

    assert products_module._status_wants_json(ctx, json_mode=False, output_format=None) is True
    assert products_module._status_wants_json(ctx, json_mode=True, output_format=None) is True
    assert products_module._status_wants_json(ctx, json_mode=False, output_format="json") is True


def test_render_status_plain_and_export_plain_cover_optional_sections(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    products_module._render_status_plain(_status_report())
    products_module._render_export_plain(_export_result(tmp_path))

    output = capsys.readouterr().out
    assert "Product Readiness: partial" in output
    assert "Scope: provider=codex since=2026-04-01 until=2026-04-30" in output
    assert "session_profiles: partial rows=7 expected=10" in output
    assert "missing=1 stale=2 orphan=3 legacy=4" in output
    assert "flags: fts=True" in output
    assert "providers: codex=7" in output
    assert "versions: materializer_version={'4': 7}" in output
    assert "schema: missing field" in output
    assert "Product export bundle:" in output
    assert "warning: stale rows" in output
    assert "error: schema drift" in output


def test_products_status_command_emits_json_and_inherits_root_filters(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    async def get_report(query: object) -> ProductReadinessReport:
        captured["query"] = query
        return _status_report()

    env = SimpleNamespace(operations=SimpleNamespace(get_product_readiness_report=get_report))
    raw_callback = _command_callback(products_module.products_status_command)
    with patch("polylogue.cli.commands.products.run_coroutine_sync", side_effect=lambda coro: asyncio.run(coro)):
        with patch("polylogue.cli.commands.products.emit_success") as emit_success:
            raw_callback(
                _status_context(env, output_format="json", provider="codex"),
                products=("profiles",),
                provider=None,
                since=None,
                until=None,
                json_mode=False,
                output_format=None,
            )

    query = captured["query"]
    assert query.products == ("profiles",)
    assert query.provider == "codex"
    assert query.since == "2026-04-01"
    assert query.until == "2026-04-30"
    emit_success.assert_called_once()


def test_products_status_command_reports_invalid_product_names() -> None:
    env = SimpleNamespace(operations=SimpleNamespace(get_product_readiness_report=MagicMock()))
    raw_callback = _command_callback(products_module.products_status_command)

    with patch("polylogue.cli.commands.products.run_coroutine_sync", side_effect=ValueError("Unknown product")):
        with pytest.raises(SystemExit, match="products status: .*Known products:"):
            raw_callback(
                _status_context(env),
                products=("not-a-product",),
                provider=None,
                since=None,
                until=None,
                json_mode=False,
                output_format=None,
            )


def test_products_export_command_covers_json_plain_and_error_paths(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    async def export_bundle(request: object) -> ProductExportBundleResult:
        captured["request"] = request
        return _export_result(tmp_path)

    env = SimpleNamespace(operations=SimpleNamespace(export_product_bundle=export_bundle))
    raw_callback = _command_callback(products_module.products_export_command)

    with pytest.raises(SystemExit, match="products export: unsupported export format: csv"):
        raw_callback(
            _export_context(env),
            output_path=tmp_path / "bundle",
            products=("profiles",),
            provider=None,
            since=None,
            until=None,
            output_format="csv",
            overwrite=False,
            json_mode=False,
        )

    with patch("polylogue.cli.commands.products.run_coroutine_sync", side_effect=lambda coro: asyncio.run(coro)):
        with patch("polylogue.cli.commands.products.emit_success") as emit_success:
            raw_callback(
                _export_context(env, output_format="json", provider="codex"),
                output_path=tmp_path / "bundle",
                products=("profiles",),
                provider=None,
                since=None,
                until=None,
                output_format="jsonl",
                overwrite=True,
                json_mode=False,
            )

    request = captured["request"]
    assert request.output_path == tmp_path / "bundle"
    assert request.products == ("profiles",)
    assert request.provider == "codex"
    assert request.overwrite is True
    emit_success.assert_called_once()

    async def broken_export(request: object) -> ProductExportBundleResult:
        del request
        raise ProductExportBundleError("cannot write bundle")

    env = SimpleNamespace(operations=SimpleNamespace(export_product_bundle=broken_export))
    with patch("polylogue.cli.commands.products.run_coroutine_sync", side_effect=lambda coro: asyncio.run(coro)):
        with pytest.raises(SystemExit, match="products export: cannot write bundle"):
            raw_callback(
                _export_context(env),
                output_path=tmp_path / "bundle",
                products=("profiles",),
                provider=None,
                since=None,
                until=None,
                output_format="jsonl",
                overwrite=False,
                json_mode=False,
            )
