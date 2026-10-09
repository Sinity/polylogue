# mypy: disable-error-code="no-untyped-def,call-arg,arg-type,attr-defined"

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import click
import pytest

from polylogue.analysis.export_bundle_contracts import (
    InsightExportBundleError,
    InsightExportBundleManifest,
    InsightExportBundleResult,
    InsightExportFileSummary,
)
from polylogue.analysis.readiness import (
    InsightOriginCoverage,
    InsightReadinessEntry,
    InsightReadinessReport,
    InsightVersionCoverage,
)
from polylogue.analysis.registry import CliOption, InsightQueryError, InsightType, get_insight_type
from polylogue.cli.commands import insights as insights_module
from polylogue.surfaces.outcome import decide_outcome


def _root_context(
    *,
    output_format: str | None = None,
    origin: str | None = None,
    since: str | None = "2026-04-01",
    until: str | None = "2026-04-30",
) -> click.Context:
    root = click.Context(click.Command("polylogue"))
    root.params = {
        "output_format": output_format,
        "origin": origin,
        "since": since,
        "until": until,
    }
    return root


def _status_context(
    env: object,
    *,
    output_format: str | None = None,
    origin: str | None = None,
    since: str | None = "2026-04-01",
    until: str | None = "2026-04-30",
) -> click.Context:
    ctx = click.Context(
        insights_module.insights_status_command,
        parent=_root_context(output_format=output_format, origin=origin, since=since, until=until),
    )
    ctx.obj = env
    return ctx


def _export_context(
    env: object,
    *,
    output_format: str | None = None,
    origin: str | None = None,
    since: str | None = "2026-04-01",
    until: str | None = "2026-04-30",
) -> click.Context:
    ctx = click.Context(
        insights_module.insights_export_command,
        parent=_root_context(output_format=output_format, origin=origin, since=since, until=until),
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
        converged=False,
        debt_stages=("derived",),
        total_sessions=10,
        origin="codex-session",
        since="2026-04-01",
        until="2026-04-30",
        insights=(
            InsightReadinessEntry(
                insight_name="session_profiles",
                display_name="Session Profiles",
                row_count=7,
                expected_row_count=10,
                missing_count=1,
                stale_count=2,
                orphan_count=3,
                incompatible_count=4,
                origin_coverage=(InsightOriginCoverage(origin="codex-session", row_count=7),),
                version_coverage=(
                    InsightVersionCoverage(field="materializer_version", current_version=4, versions={"4": 7}),
                ),
                schema_contract_issues=("missing field",),
            ),
        ),
    )


def _export_result(tmp_path: Path) -> InsightExportBundleResult:
    return InsightExportBundleResult(
        outcome=decide_outcome(matched=7),
        output_path=tmp_path / "bundle",
        manifest_path=tmp_path / "bundle" / "manifest.json",
        coverage_path=tmp_path / "bundle" / "coverage.json",
        manifest=InsightExportBundleManifest(
            generated_at="2026-04-23T00:00:00+00:00",
            polylogue_version="1.0.0",
            archive_root="/tmp/archive",
            database_path="/tmp/archive/index.db",
            query={"provider": "codex"},
            insights=(
                InsightExportFileSummary(
                    insight_name="session_profiles",
                    file="insights/session_profiles.jsonl",
                    schema_file="schemas/session_profiles.schema.json",
                    row_count=7,
                    withheld_reason=None,
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

    assert [param.name for param in params] == ["provider", "limit", "output_format", "offset", "output_format"]
    assert command.name == "test-insight"
    assert command.help == "List test insights."


def test_make_callback_renders_insights_and_surfaces_query_errors() -> None:
    callback = insights_module._make_callback(get_insight_type("session_profiles"))
    raw_callback = getattr(callback, "__wrapped__", callback)
    env = SimpleNamespace(config=MagicMock())
    ctx = click.Context(click.Command("profiles"))
    ctx.obj = env

    request = SimpleNamespace(query_kwargs={"limit": 1}, wants_json=True)
    result = {
        "page": {"insight": "session_profiles", "items": [], "total": 0},
        "outcome": {"state": "empty", "reason": "no_rows_in_scope", "detail": {}},
    }
    with patch("polylogue.cli.commands.insights.InsightCommandRequest.from_context", return_value=request):
        with patch("polylogue.cli.commands.insights.dispatch_read", return_value=(result, "daemon")) as dispatch:
            with patch("polylogue.cli.commands.insights.render_insight_items") as render_items:
                raw_callback(ctx, output_format="json")

    dispatch.assert_called_once()
    assert dispatch.call_args.args[0] is env.config
    operation = dispatch.call_args.args[1]
    assert operation.operation == "insights.list"
    assert operation.payload == {"page": {"insight": "session_profiles", "query": {"limit": 1}}}
    render_items.assert_called_once()
    assert render_items.call_args.args == ([], get_insight_type("session_profiles"))
    assert render_items.call_args.kwargs["json_mode"] is True
    assert render_items.call_args.kwargs["outcome"].state == "empty"

    with patch("polylogue.cli.commands.insights.InsightCommandRequest.from_context", return_value=request):
        with patch("polylogue.cli.commands.insights.build_insight_query", side_effect=InsightQueryError("bad query")):
            with patch("polylogue.cli.commands.insights.dispatch_read") as dispatch:
                with pytest.raises(SystemExit, match="insights profiles: bad query"):
                    raw_callback(ctx, output_format=None)
                dispatch.assert_not_called()


def test_status_wants_json_checks_command_and_root_flags() -> None:
    ctx = click.Context(click.Command("status"), parent=_root_context(output_format="json"))

    assert insights_module._status_wants_json(ctx, output_format=None) is True
    assert insights_module._status_wants_json(ctx, output_format="json") is True


def test_render_status_plain_and_export_plain_cover_optional_sections(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    insights_module._render_status_plain(_status_report())
    insights_module._render_export_plain(_export_result(tmp_path))

    output = capsys.readouterr().out
    assert "Readiness: derived domains incomplete" in output
    assert "Operation debt: derived" in output
    assert "Scope: origin=codex-session since=2026-04-01 until=2026-04-30" in output
    assert "session_profiles: rows=7 expected=10" in output
    assert "missing=1 stale=2 orphan=3 incompatible=4" in output
    assert "origins: codex-session=7" in output
    assert "versions: materializer_version={'4': 7}" in output
    assert "schema: missing field" in output
    assert "Insight export bundle:" in output
    assert "warning: stale rows" in output
    assert "error: schema drift" in output


def test_insights_status_command_emits_json_and_inherits_root_filters(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def get_report(config: object, operation: object) -> tuple[dict[str, object], str]:
        from polylogue.operations.insight_contracts import InsightReadinessRequest
        from polylogue.surfaces.outcome import decide_outcome

        assert operation.operation == "insights.readiness"
        captured["query"] = InsightReadinessRequest.model_validate(operation.payload).query
        return {
            "report": _status_report().model_dump(mode="json"),
            "outcome": decide_outcome(matched=1, degraded=("insight_convergence_pending",)).to_dict(),
        }, "daemon"

    env = SimpleNamespace(config=object())
    raw_callback = _command_callback(insights_module.insights_status_command)
    with patch("polylogue.cli.commands.insights.dispatch_read", side_effect=get_report) as dispatch:
        with patch("polylogue.cli.commands.insights.emit_success") as emit_success:
            raw_callback(
                _status_context(env, output_format="json", origin="codex-session"),
                insights=("profiles",),
                origin=None,
                since=None,
                until=None,
                output_format=None,
            )
    dispatch.assert_called_once()
    assert dispatch.call_args.args[0] is env.config
    query = captured["query"]
    assert query.insights == ("profiles",)
    assert query.origin == "codex-session"
    assert query.since == "2026-04-01T00:00:00+00:00"
    assert query.until == "2026-04-30T00:00:00+00:00"
    emit_success.assert_called_once()
    assert emit_success.call_args.args[0]["outcome"]["state"] == "degraded"


def test_insights_status_command_rejects_inherited_provider_csv() -> None:
    env = SimpleNamespace(config=object())
    raw_callback = _command_callback(insights_module.insights_status_command)
    with patch("polylogue.cli.commands.insights.dispatch_read") as dispatch:
        with pytest.raises(SystemExit, match="insights commands accept one origin"):
            raw_callback(
                _status_context(env, origin="codex-session,chatgpt-export"),
                insights=(),
                origin=None,
                since=None,
                until=None,
                output_format=None,
            )
        dispatch.assert_not_called()


def test_insights_status_command_reports_invalid_insight_names() -> None:
    env = SimpleNamespace(config=object())
    raw_callback = _command_callback(insights_module.insights_status_command)
    with patch("polylogue.cli.commands.insights.dispatch_read") as dispatch:
        with pytest.raises(SystemExit, match="insights status: .*Known insights:"):
            raw_callback(
                _status_context(env),
                insights=("not-an-insight",),
                origin=None,
                since=None,
                until=None,
                output_format=None,
            )
        dispatch.assert_not_called()


def test_insights_export_command_covers_json_plain_and_error_paths(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def export_bundle(config: object, operation: object) -> tuple[dict[str, object], str]:
        from polylogue.operations.insight_export_contracts import decode_insight_export_request

        captured["request"] = decode_insight_export_request(operation.payload).request
        return {"bundle": _export_result(tmp_path).model_dump(mode="json"), "outcome": {"state": "ok"}}, "daemon"

    env = SimpleNamespace(config=SimpleNamespace())
    raw_callback = _command_callback(insights_module.insights_export_command)

    with pytest.raises(SystemExit, match="insights export: unsupported export format: csv"):
        raw_callback(
            _export_context(env),
            output_path=tmp_path / "bundle",
            insights=("profiles",),
            origin=None,
            since=None,
            until=None,
            bundle_format="csv",
            output_format=None,
            overwrite=False,
        )

    with patch("polylogue.cli.commands.insights.dispatch_read", side_effect=export_bundle):
        with patch("polylogue.cli.commands.insights.emit_success") as emit_success:
            raw_callback(
                _export_context(env, output_format="json", origin="codex-session"),
                output_path=tmp_path / "bundle",
                insights=("profiles",),
                origin=None,
                since=None,
                until=None,
                bundle_format="jsonl",
                output_format=None,
                overwrite=True,
            )

    request = captured["request"]
    assert request.output_path == tmp_path / "bundle"
    assert request.insights == ("profiles",)
    assert request.origin == "codex-session"
    assert request.since == "2026-04-01T00:00:00+00:00"
    assert request.until == "2026-04-30T00:00:00+00:00"
    assert request.overwrite is True
    emit_success.assert_called_once()

    def broken_export(config: object, request: object) -> object:
        raise InsightExportBundleError("cannot write bundle")

    env = SimpleNamespace(config=SimpleNamespace())
    with patch("polylogue.cli.commands.insights.dispatch_read", side_effect=broken_export):
        with pytest.raises(SystemExit, match="insights export: cannot write bundle"):
            raw_callback(
                _export_context(env),
                output_path=tmp_path / "bundle",
                insights=("profiles",),
                origin=None,
                since=None,
                until=None,
                bundle_format="jsonl",
                output_format=None,
                overwrite=False,
            )


@pytest.mark.parametrize("state,expected", [("ok", 0), ("empty", 2), ("degraded", 1), ("error", 1)])
@pytest.mark.parametrize("output_format", [None, "json"])
def test_export_finishes_with_the_resident_outcome(tmp_path: Path, state: str, expected: int, output_format) -> None:
    from polylogue.surfaces.outcome import OutcomeEnvelope

    outcome = OutcomeEnvelope(state=state, reason="insight_output_incomplete" if state == "degraded" else None)
    bundle = _export_result(tmp_path).model_copy(update={"outcome": outcome})
    payload = {"bundle": bundle.model_dump(mode="json"), "outcome": outcome.to_dict()}
    callback = _command_callback(insights_module.insights_export_command)
    with patch("polylogue.cli.commands.insights.dispatch_read", return_value=(payload, "daemon")):
        if expected:
            with pytest.raises(SystemExit) as caught:
                callback(
                    _export_context(SimpleNamespace(config=object())),
                    output_path=tmp_path / "bundle",
                    insights=("profiles",),
                    origin=None,
                    since=None,
                    until=None,
                    bundle_format="jsonl",
                    output_format=output_format,
                    overwrite=False,
                )
            assert caught.value.code == expected
        else:
            callback(
                _export_context(SimpleNamespace(config=object())),
                output_path=tmp_path / "bundle",
                insights=("profiles",),
                origin=None,
                since=None,
                until=None,
                bundle_format="jsonl",
                output_format=output_format,
                overwrite=False,
            )
