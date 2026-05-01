# mypy: disable-error-code="no-untyped-def,call-arg,arg-type,attr-defined"

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import click
import pytest

from polylogue.cli.commands import insights as insights_module
from polylogue.insights.export_bundles import (
    InsightExportBundleError,
    InsightExportBundleManifest,
    InsightExportBundleResult,
    InsightExportFileSummary,
)
from polylogue.insights.readiness import (
    InsightProviderCoverage,
    InsightReadinessEntry,
    InsightReadinessReport,
    InsightVersionCoverage,
)
from polylogue.insights.registry import CliOption, InsightQueryError, InsightType, get_insight_type


def _root_context(
    *,
    output_format: str | None = None,
    provider: str | None = None,
    since: str | None = "2026-04-01",
    until: str | None = "2026-04-30",
) -> click.Context:
    root = click.Context(click.Command("polylogue"))
    root.params = {
        "output_format": output_format,
        "provider": provider,
        "since": since,
        "until": until,
    }
    return root


def _status_context(
    env: object,
    *,
    output_format: str | None = None,
    provider: str | None = None,
    since: str | None = "2026-04-01",
    until: str | None = "2026-04-30",
) -> click.Context:
    ctx = click.Context(
        insights_module.insights_status_command,
        parent=_root_context(output_format=output_format, provider=provider, since=since, until=until),
    )
    ctx.obj = env
    return ctx


def _export_context(
    env: object,
    *,
    output_format: str | None = None,
    provider: str | None = None,
    since: str | None = "2026-04-01",
    until: str | None = "2026-04-30",
) -> click.Context:
    ctx = click.Context(
        insights_module.insights_export_command,
        parent=_root_context(output_format=output_format, provider=provider, since=since, until=until),
    )
    ctx.obj = env
    return ctx


def _command_callback(command: click.Command) -> Callable[..., object]:
    callback = getattr(command.callback, "__wrapped__", command.callback)
    assert callback is not None
    return callback


def _status_report() -> InsightReadinessReport:
    return InsightReadinessReport(
        checked_at="2026-04-23T00:00:00+00:00",
        aggregate_verdict="partial",
        total_conversations=10,
        provider="codex",
        since="2026-04-01",
        until="2026-04-30",
        insights=(
            InsightReadinessEntry(
                insight_name="session_profiles",
                display_name="Session Profiles",
                verdict="partial",
                row_count=7,
                expected_row_count=10,
                missing_count=1,
                stale_count=2,
                orphan_count=3,
                legacy_incompatible_count=4,
                ready_flags={"fts": True},
                provider_coverage=(InsightProviderCoverage(provider_name="codex", row_count=7),),
                version_coverage=(
                    InsightVersionCoverage(field="materializer_version", current_version=4, versions={"4": 7}),
                ),
                schema_contract_issues=("missing field",),
            ),
        ),
    )


def _export_result(tmp_path: Path) -> InsightExportBundleResult:
    return InsightExportBundleResult(
        output_path=tmp_path / "bundle",
        manifest_path=tmp_path / "bundle" / "manifest.json",
        coverage_path=tmp_path / "bundle" / "coverage.json",
        manifest=InsightExportBundleManifest(
            generated_at="2026-04-23T00:00:00+00:00",
            polylogue_version="1.0.0",
            archive_root="/tmp/archive",
            database_path="/tmp/archive/polylogue.db",
            query={"provider": "codex"},
            insights=(
                InsightExportFileSummary(
                    insight_name="session_profiles",
                    file="insights/session_profiles.jsonl",
                    schema_file="schemas/session_profiles.schema.json",
                    row_count=7,
                    readiness_verdict="partial",
                    warnings=("stale rows",),
                    errors=("schema drift",),
                ),
            ),
        ),
    )


def test_build_click_params_and_insight_command_cover_dynamic_registration() -> None:
    insight_type = InsightType(
        name="test_insight",
        display_name="Test Insight",
        json_key="items",
        cli_help="List test insights.",
        cli_options=(CliOption("provider", ("--provider",), help="Provider", type=str, default=None),),
        mcp_default_limit=25,
    )

    params = insights_module._build_click_params(insight_type)
    command = insights_module._build_insight_command(insight_type)

    assert [param.name for param in params] == ["provider", "limit", "offset", "json_mode", "output_format"]
    assert command.name == "test-insight"
    assert command.help == "List test insights."


def test_make_callback_renders_insights_and_surfaces_query_errors() -> None:
    callback = insights_module._make_callback(get_insight_type("session_profiles"))
    raw_callback = getattr(callback, "__wrapped__", callback)
    env = SimpleNamespace(operations=MagicMock())
    ctx = click.Context(click.Command("profiles"))
    ctx.obj = env

    request = SimpleNamespace(query_kwargs={"limit": 1}, wants_json=True)
    with patch("polylogue.cli.commands.insights.InsightCommandRequest.from_context", return_value=request):
        with patch("polylogue.cli.commands.insights.fetch_insights", return_value=["row"]) as fetch_insights:
            with patch("polylogue.cli.commands.insights.render_insight_items") as render_items:
                raw_callback(ctx, json_mode=True, output_format=None)

    fetch_insights.assert_called_once()
    render_items.assert_called_once_with(["row"], get_insight_type("session_profiles"), json_mode=True)

    with patch("polylogue.cli.commands.insights.InsightCommandRequest.from_context", return_value=request):
        with patch("polylogue.cli.commands.insights.fetch_insights", side_effect=InsightQueryError("bad query")):
            with pytest.raises(SystemExit, match="insights profiles: bad query"):
                raw_callback(ctx, json_mode=False, output_format=None)


def test_status_wants_json_checks_command_and_root_flags() -> None:
    ctx = click.Context(click.Command("status"), parent=_root_context(output_format="json"))

    assert insights_module._status_wants_json(ctx, json_mode=False, output_format=None) is True
    assert insights_module._status_wants_json(ctx, json_mode=True, output_format=None) is True
    assert insights_module._status_wants_json(ctx, json_mode=False, output_format="json") is True


def test_render_status_plain_and_export_plain_cover_optional_sections(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    insights_module._render_status_plain(_status_report())
    insights_module._render_export_plain(_export_result(tmp_path))

    output = capsys.readouterr().out
    assert "Insight Readiness: partial" in output
    assert "Scope: provider=codex since=2026-04-01 until=2026-04-30" in output
    assert "session_profiles: partial rows=7 expected=10" in output
    assert "missing=1 stale=2 orphan=3 legacy=4" in output
    assert "flags: fts=True" in output
    assert "providers: codex=7" in output
    assert "versions: materializer_version={'4': 7}" in output
    assert "schema: missing field" in output
    assert "Insight export bundle:" in output
    assert "warning: stale rows" in output
    assert "error: schema drift" in output


def test_insights_status_command_emits_json_and_inherits_root_filters(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    async def get_report(query: object) -> InsightReadinessReport:
        captured["query"] = query
        return _status_report()

    env = SimpleNamespace(operations=SimpleNamespace(get_product_readiness_report=get_report))
    raw_callback = _command_callback(insights_module.insights_status_command)
    with patch("polylogue.cli.commands.insights.run_coroutine_sync", side_effect=lambda coro: asyncio.run(coro)):
        with patch("polylogue.cli.commands.insights.emit_success") as emit_success:
            raw_callback(
                _status_context(env, output_format="json", provider="codex"),
                insights=("profiles",),
                provider=None,
                since=None,
                until=None,
                json_mode=False,
                output_format=None,
            )

    query = captured["query"]
    assert query.insights == ("profiles",)
    assert query.provider == "codex"
    assert query.since == "2026-04-01T00:00:00+00:00"
    assert query.until == "2026-04-30T00:00:00+00:00"
    emit_success.assert_called_once()


def test_insights_status_command_rejects_inherited_provider_csv() -> None:
    env = SimpleNamespace(operations=SimpleNamespace(get_product_readiness_report=MagicMock()))
    raw_callback = _command_callback(insights_module.insights_status_command)

    with pytest.raises(SystemExit, match="insights commands accept one provider"):
        raw_callback(
            _status_context(env, provider="codex,chatgpt"),
            insights=(),
            provider=None,
            since=None,
            until=None,
            json_mode=False,
            output_format=None,
        )


def test_insights_status_command_reports_invalid_insight_names() -> None:
    env = SimpleNamespace(operations=SimpleNamespace(get_product_readiness_report=MagicMock()))
    raw_callback = _command_callback(insights_module.insights_status_command)

    with patch("polylogue.cli.commands.insights.run_coroutine_sync", side_effect=ValueError("Unknown insight")):
        with pytest.raises(SystemExit, match="insights status: .*Known insights:"):
            raw_callback(
                _status_context(env),
                insights=("not-an-insight",),
                provider=None,
                since=None,
                until=None,
                json_mode=False,
                output_format=None,
            )


def test_insights_export_command_covers_json_plain_and_error_paths(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    async def export_bundle(request: object) -> InsightExportBundleResult:
        captured["request"] = request
        return _export_result(tmp_path)

    env = SimpleNamespace(operations=SimpleNamespace(export_product_bundle=export_bundle))
    raw_callback = _command_callback(insights_module.insights_export_command)

    with pytest.raises(SystemExit, match="insights export: unsupported export format: csv"):
        raw_callback(
            _export_context(env),
            output_path=tmp_path / "bundle",
            insights=("profiles",),
            provider=None,
            since=None,
            until=None,
            output_format="csv",
            overwrite=False,
            json_mode=False,
        )

    with patch("polylogue.cli.commands.insights.run_coroutine_sync", side_effect=lambda coro: asyncio.run(coro)):
        with patch("polylogue.cli.commands.insights.emit_success") as emit_success:
            raw_callback(
                _export_context(env, output_format="json", provider="codex"),
                output_path=tmp_path / "bundle",
                insights=("profiles",),
                provider=None,
                since=None,
                until=None,
                output_format="jsonl",
                overwrite=True,
                json_mode=False,
            )

    request = captured["request"]
    assert request.output_path == tmp_path / "bundle"
    assert request.insights == ("profiles",)
    assert request.provider == "codex"
    assert request.since == "2026-04-01T00:00:00+00:00"
    assert request.until == "2026-04-30T00:00:00+00:00"
    assert request.overwrite is True
    emit_success.assert_called_once()

    async def broken_export(request: object) -> InsightExportBundleResult:
        del request
        raise InsightExportBundleError("cannot write bundle")

    env = SimpleNamespace(operations=SimpleNamespace(export_product_bundle=broken_export))
    with patch("polylogue.cli.commands.insights.run_coroutine_sync", side_effect=lambda coro: asyncio.run(coro)):
        with pytest.raises(SystemExit, match="insights export: cannot write bundle"):
            raw_callback(
                _export_context(env),
                output_path=tmp_path / "bundle",
                insights=("profiles",),
                provider=None,
                since=None,
                until=None,
                output_format="jsonl",
                overwrite=False,
                json_mode=False,
            )
