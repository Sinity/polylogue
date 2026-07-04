from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import click
import pytest
from click.testing import CliRunner

from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.cli.commands import diagnostics
from polylogue.cli.shared.types import AppEnv
from polylogue.insights.tool_usage import ToolUsageInsightQuery
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _env() -> AppEnv:
    ui = MagicMock()
    ui.console = MagicMock()
    return AppEnv(ui=ui, services=MagicMock())


def _console_print(env: AppEnv) -> MagicMock:
    return cast(MagicMock, env.ui.console.print)


def _session(*messages: object) -> SimpleNamespace:
    return SimpleNamespace(
        id="conv-diagnostics",
        display_title="Diagnostics",
        messages=list(messages),
    )


def _message(
    role: str,
    timestamp: datetime | None,
    text: str = "hello",
    *,
    blocks: list[dict[str, object]] | None = None,
    duration_ms: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        role=role,
        timestamp=timestamp,
        text=text,
        blocks=blocks or [],
        duration_ms=duration_ms,
    )


def test_usage_command_passes_headline_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeReport:
        def to_dict(self) -> dict[str, object]:
            return {"detail_level": "headline"}

    class FakePolylogue:
        async def provider_usage_report(
            self,
            *,
            origin: str | None = None,
            limit: int | None = None,
            detail: str = "full",
        ) -> FakeReport:
            captured.update({"origin": origin, "limit": limit, "detail": detail})
            return FakeReport()

    monkeypatch.setattr(AppEnv, "polylogue", property(lambda self: FakePolylogue()))
    result = CliRunner().invoke(
        diagnostics.usage_command,
        ["--origin", "codex-session", "--limit", "0", "--detail", "headline", "--format", "json"],
        obj=_env(),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"detail_level": "headline"}
    assert captured == {"origin": "codex-session", "limit": 0, "detail": "headline"}


def test_usage_report_text_renders_pricing_lanes() -> None:
    env = _env()
    report = SimpleNamespace(
        archive_root="/archive",
        caveats=(),
        origins=(),
        model_rollup_grain="physical_session",
        model_rollup_usage=SimpleNamespace(to_dict=lambda: {}),
        logical_model_rollup_grain="logical_session_model_high_water",
        logical_model_rollup_usage=SimpleNamespace(to_dict=lambda: {}),
        stored_provider_priced_usd=12.5,
        catalog_api_equivalent_usd=18.75,
        logical_pricing_grain="logical_session_model_high_water",
        logical_catalog_api_equivalent_usd=14.25,
        pricing_lanes=(
            SimpleNamespace(
                provenance="priced",
                row_count=2,
                matched_model_row_count=2,
                unmatched_model_row_count=0,
                stored_cost_usd=12.5,
                catalog_api_equivalent_usd=12.5,
            ),
            SimpleNamespace(
                provenance="origin_reported",
                row_count=1,
                matched_model_row_count=1,
                unmatched_model_row_count=0,
                stored_cost_usd=0.0,
                catalog_api_equivalent_usd=6.25,
            ),
        ),
        logical_pricing_lanes=(
            SimpleNamespace(
                provenance="origin_reported",
                row_count=1,
                matched_model_row_count=1,
                unmatched_model_row_count=0,
                catalog_api_equivalent_usd=4.25,
            ),
        ),
    )

    diagnostics._render_usage_report(env, report)

    rendered = "\n".join(call.args[0] for call in _console_print(env).call_args_list if call.args)
    assert "stored/provider-priced cost: $12.50" in rendered
    assert "catalog API-equivalent cost: $18.75" in rendered
    assert "catalog API-equivalent cost (logical_session_model_high_water): $14.25" in rendered
    assert "pricing lane origin_reported" in rendered
    assert "logical pricing lane origin_reported" in rendered


@pytest.mark.asyncio
async def test_pace_reports_active_model_and_idle_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    conv = _session(
        _message("user", start),
        _message("assistant", start + timedelta(seconds=30)),
        _message("user", start + timedelta(seconds=120)),
    )

    async def fake_list(self: object, repository: object) -> list[SimpleNamespace]:
        return [conv]

    monkeypatch.setattr(SessionQuerySpec, "list", fake_list)
    env = _env()

    await diagnostics._pace(env, None, limit=5, threshold=60)

    rendered = "\n".join(call.args[0] for call in _console_print(env).call_args_list if call.args)
    assert "Turns: 3" in rendered
    assert "[active]" in rendered
    assert "[user_idle]" in rendered


@pytest.mark.asyncio
async def test_pace_reports_sparse_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    conv = _session(_message("user", None))

    async def fake_list(self: object, repository: object) -> list[SimpleNamespace]:
        return [conv]

    monkeypatch.setattr(SessionQuerySpec, "list", fake_list)
    env = _env()

    await diagnostics._pace(env, "conv-diagnostics", limit=5, threshold=60)

    assert "not enough messages" in _console_print(env).call_args.args[0]


@pytest.mark.asyncio
async def test_turns_reports_duration_thinking_tools_and_characters(monkeypatch: pytest.MonkeyPatch) -> None:
    conv = _session(
        _message(
            "assistant",
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            "assistant text",
            blocks=[
                {"type": "thinking", "text": "reason"},
                {"type": "tool_use", "name": "bash"},
                {"type": "tool_result", "text": "ok"},
            ],
            duration_ms=1250,
        )
    )

    async def fake_list(self: object, repository: object) -> list[SimpleNamespace]:
        return [conv]

    monkeypatch.setattr(SessionQuerySpec, "list", fake_list)
    env = _env()

    await diagnostics._turns(env, "conv-diagnostics", limit=3)

    rendered = "\n".join(call.args[0] for call in _console_print(env).call_args_list if call.args)
    assert "1.2s" in rendered
    row = _console_print(env).call_args_list[-1].args[0]
    assert re.search(r"\b6\b", row)
    assert re.search(r"\b2\b", row)


class _FakeToolCountStore:
    def __init__(
        self,
        call_rows: list[dict[str, object]],
        event_rows: list[dict[str, object]] | None = None,
        action_rows: list[dict[str, object]] | None = None,
    ) -> None:
        self.call_rows = call_rows
        self.event_rows = event_rows or []
        self.action_rows = action_rows or []
        self.queries: list[ToolUsageInsightQuery] = []
        self.event_queries: list[ToolUsageInsightQuery] = []
        self.action_queries: list[ToolUsageInsightQuery] = []
        self.detail_patterns: list[tuple[str, ...]] = []
        self.since_ms_values: list[int | None] = []

    def __enter__(self) -> _FakeToolCountStore:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def list_tool_call_count_rows(self, query: ToolUsageInsightQuery | None = None) -> list[dict[str, object]]:
        self.queries.append(query or ToolUsageInsightQuery())
        return self.call_rows

    def list_tool_observed_event_count_rows(
        self,
        query: ToolUsageInsightQuery | None = None,
    ) -> list[dict[str, object]]:
        self.event_queries.append(query or ToolUsageInsightQuery())
        return self.event_rows

    def list_tool_action_evidence_count_rows(
        self,
        query: ToolUsageInsightQuery | None = None,
        *,
        detail_patterns: tuple[str, ...] = (),
        since_ms: int | None = None,
    ) -> list[dict[str, object]]:
        self.action_queries.append(query or ToolUsageInsightQuery())
        self.detail_patterns.append(detail_patterns)
        self.since_ms_values.append(since_ms)
        return self.action_rows


def _patch_tool_count_store(
    monkeypatch: pytest.MonkeyPatch,
    call_rows: list[dict[str, object]],
    event_rows: list[dict[str, object]] | None = None,
    action_rows: list[dict[str, object]] | None = None,
) -> _FakeToolCountStore:
    store = _FakeToolCountStore(call_rows, event_rows, action_rows)
    monkeypatch.setattr(ArchiveStore, "open_existing", classmethod(lambda cls, archive_root: store))
    return store


def test_tools_help_explains_basis_and_mcp_server_filters() -> None:
    result = CliRunner().invoke(diagnostics.tools_command, ["--help"])
    output = " ".join(result.output.split())

    assert result.exit_code == 0
    assert "observed-events counts finished tool outcomes" in output
    assert "actions counts canonical action evidence" in output
    assert "tool-use-blocks counts calls" in output
    assert "--detail-pattern" in output
    assert "--days" in output
    assert "--compare-family" in output
    assert "serena -> mcp__serena__*" in output
    assert "mcp__serena__find_symbol" in output


@pytest.mark.asyncio
async def test_tools_renders_tool_usage_insight(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _env()
    _patch_tool_count_store(
        monkeypatch,
        [
            {
                "source_name": "claude-code",
                "origin": "claude-code-session",
                "normalized_tool_name": "bash",
                "action_kind": "shell",
                "call_count": 3,
            },
            {
                "source_name": "codex",
                "origin": "codex-session",
                "normalized_tool_name": "read",
                "action_kind": "file_read",
                "call_count": 2,
            },
        ],
    )

    await diagnostics._tools(
        env,
        origin=None,
        tool=None,
        mcp_server=None,
        action_kind=None,
        detail_patterns=(),
        days=None,
        basis="tool-use-blocks",
        limit=5,
    )

    rendered = "\n".join(call.args[0] for call in _console_print(env).call_args_list if call.args)
    assert "Tool call counts" in rendered
    assert "detail: tool_use block call counts" in rendered
    assert "bash" in rendered
    assert "read" in rendered


@pytest.mark.asyncio
async def test_tools_renders_observed_event_basis(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _env()
    _patch_tool_count_store(
        monkeypatch,
        [],
        [
            {
                "source_name": "claude-code",
                "origin": "claude-code-session",
                "normalized_tool_name": "mcp__serena__find_symbol",
                "action_kind": "mcp",
                "status": "ok",
                "event_count": 4,
            },
            {
                "source_name": "claude-code",
                "origin": "claude-code-session",
                "normalized_tool_name": "mcp__serena__find_symbol",
                "action_kind": "mcp",
                "status": "failed",
                "event_count": 1,
            },
        ],
    )

    await diagnostics._tools(
        env,
        origin=None,
        tool=None,
        mcp_server="serena",
        action_kind=None,
        detail_patterns=(),
        days=None,
        basis="observed-events",
        limit=5,
    )

    rendered = "\n".join(call.args[0] for call in _console_print(env).call_args_list if call.args)
    assert "Tool observed-event counts" in rendered
    assert "tool_finished observed events" in rendered
    assert "mcp__serena__find_symbol" in rendered
    assert "failed" in rendered


@pytest.mark.asyncio
async def test_tools_json_declares_tool_use_basis(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env = _env()
    _patch_tool_count_store(
        monkeypatch,
        [
            {
                "source_name": "codex",
                "origin": "codex-session",
                "normalized_tool_name": "exec_command",
                "action_kind": "shell",
                "call_count": 2,
            },
        ],
    )

    await diagnostics._tools(
        env,
        origin="codex-session",
        tool=None,
        mcp_server=None,
        action_kind=None,
        detail_patterns=(),
        days=None,
        basis="tool-use-blocks",
        limit=5,
        output_format="json",
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "tool_call_counts"
    assert payload["detail_level"] == "tool_use_block_call_counts"
    assert payload["filters"]["basis"] == "tool-use-blocks"
    assert payload["items"] == [
        {
            "source_name": "codex",
            "origin": "codex-session",
            "normalized_tool_name": "exec_command",
            "action_kind": "shell",
            "call_count": 2,
        }
    ]


@pytest.mark.asyncio
async def test_tools_json_declares_observed_event_basis(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = _env()
    _patch_tool_count_store(
        monkeypatch,
        [],
        [
            {
                "source_name": "claude-code",
                "origin": "claude-code-session",
                "normalized_tool_name": "mcp__serena__find_symbol",
                "action_kind": "mcp",
                "status": "failed",
                "event_count": 1,
            },
        ],
    )

    await diagnostics._tools(
        env,
        origin=None,
        tool=None,
        mcp_server="serena",
        action_kind=None,
        detail_patterns=(),
        days=None,
        basis="observed-events",
        limit=5,
        output_format="json",
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "tool_observed_event_counts"
    assert payload["detail_level"] == "tool_finished_observed_events"
    assert payload["filters"]["basis"] == "observed-events"
    assert payload["filters"]["mcp_server"] == "serena"
    assert payload["items"] == [
        {
            "source_name": "claude-code",
            "origin": "claude-code-session",
            "normalized_tool_name": "mcp__serena__find_symbol",
            "action_kind": "mcp",
            "status": "failed",
            "event_count": 1,
        }
    ]


@pytest.mark.asyncio
async def test_tools_json_declares_action_evidence_basis(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = _env()
    store = _patch_tool_count_store(
        monkeypatch,
        [],
        action_rows=[
            {
                "source_name": "codex",
                "origin": "codex-session",
                "normalized_tool_name": "codebase-memory/command-detail",
                "action_kind": "tool_use",
                "evidence_kind": "command_detail",
                "matched_by": "detail",
                "call_count": 2,
                "session_count": 1,
                "error_count": 0,
                "nonzero_exit_count": 0,
            },
        ],
    )

    await diagnostics._tools(
        env,
        origin=None,
        tool=None,
        mcp_server=None,
        action_kind=None,
        detail_patterns=("codebase-memory",),
        days=30,
        basis="actions",
        limit=5,
        output_format="json",
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "tool_action_evidence_counts"
    assert payload["detail_level"] == "canonical_action_evidence_counts"
    assert payload["filters"]["basis"] == "actions"
    assert payload["filters"]["detail_patterns"] == ["codebase-memory"]
    assert payload["filters"]["days"] == 30
    assert payload["items"] == [
        {
            "source_name": "codex",
            "origin": "codex-session",
            "normalized_tool_name": "codebase-memory/command-detail",
            "action_kind": "tool_use",
            "evidence_kind": "command_detail",
            "matched_by": "detail",
            "call_count": 2,
            "session_count": 1,
            "error_count": 0,
            "nonzero_exit_count": 0,
        }
    ]
    assert store.detail_patterns == [("codebase-memory",)]
    assert len(store.since_ms_values) == 1
    assert store.since_ms_values[0] is not None


@pytest.mark.asyncio
async def test_tools_json_compares_family_across_evidence_bases(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = _env()
    store = _patch_tool_count_store(
        monkeypatch,
        call_rows=[
            {
                "source_name": "claude-code",
                "origin": "claude-code-session",
                "normalized_tool_name": "mcp__serena__find_symbol",
                "action_kind": "mcp",
                "call_count": 3,
            },
        ],
        event_rows=[
            {
                "source_name": "claude-code",
                "origin": "claude-code-session",
                "normalized_tool_name": "mcp__serena__find_symbol",
                "action_kind": "mcp",
                "status": "ok",
                "event_count": 2,
            },
        ],
        action_rows=[
            {
                "source_name": "claude-code",
                "origin": "claude-code-session",
                "normalized_tool_name": "serena/find_symbol",
                "action_kind": "tool_use",
                "evidence_kind": "mcp_tool_call",
                "matched_by": "detail",
                "call_count": 2,
                "session_count": 1,
                "error_count": 0,
                "nonzero_exit_count": 0,
            },
        ],
    )

    await diagnostics._tools(
        env,
        origin=None,
        tool=None,
        mcp_server=None,
        action_kind=None,
        detail_patterns=(),
        days=30,
        basis="tool-use-blocks",
        limit=5,
        output_format="json",
        compare_family="serena",
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "tool_family_evidence_comparison"
    assert payload["family"] == "serena"
    assert [basis["filters"]["basis"] for basis in payload["bases"]] == [
        "tool-use-blocks",
        "observed-events",
        "actions",
    ]
    assert payload["bases"][0]["filters"]["mcp_server"] == "serena"
    assert payload["bases"][0]["filters"]["days"] == 30
    assert payload["bases"][1]["filters"]["mcp_server"] == "serena"
    assert payload["bases"][1]["filters"]["days"] == 30
    assert payload["bases"][2]["filters"]["detail_patterns"] == ["serena"]
    assert payload["bases"][2]["filters"]["days"] == 30
    assert payload["bases"][0]["items"][0]["call_count"] == 3
    assert payload["bases"][1]["items"][0]["event_count"] == 2
    assert payload["bases"][2]["items"][0]["evidence_kind"] == "mcp_tool_call"
    assert "source-derived tool_finished outcomes" in payload["caveats"][1]
    assert len(store.queries) == 1
    assert store.queries[0].since_ms is not None
    assert len(store.event_queries) == 1
    assert store.event_queries[0].since_ms is not None
    assert store.detail_patterns == [("serena",)]
    assert len(store.since_ms_values) == 1
    assert store.since_ms_values[0] is not None


@pytest.mark.asyncio
async def test_tools_compare_family_rejects_overlapping_tool_filters() -> None:
    with pytest.raises(click.UsageError):
        await diagnostics._tools(
            _env(),
            origin=None,
            tool="mcp__serena__find_symbol",
            mcp_server=None,
            action_kind=None,
            detail_patterns=(),
            days=None,
            basis="tool-use-blocks",
            limit=5,
            output_format="json",
            compare_family="serena",
        )


@pytest.mark.asyncio
async def test_tools_passes_filters_to_tool_usage_insight(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _env()
    store = _patch_tool_count_store(monkeypatch, [])

    await diagnostics._tools(
        env,
        origin="claude-code-session",
        tool="mcp__serena__find_symbol",
        mcp_server="serena",
        action_kind="tool_use",
        detail_patterns=(),
        days=None,
        basis="tool-use-blocks",
        limit=5,
    )

    query = store.queries[0]
    assert isinstance(query, ToolUsageInsightQuery)
    assert query.provider == "claude-code-session"
    assert query.tool == "mcp__serena__find_symbol"
    assert query.mcp_server == "serena"
    assert query.action_kind == "tool_use"
    assert query.limit == 5


@pytest.mark.asyncio
async def test_tools_reports_empty_archives(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _env()
    _patch_tool_count_store(monkeypatch, [])

    await diagnostics._tools(
        env,
        origin=None,
        tool=None,
        mcp_server=None,
        action_kind=None,
        detail_patterns=(),
        days=None,
        basis="tool-use-blocks",
        limit=5,
    )

    assert _console_print(env).call_args.args[0] == "No tool invocations found."
