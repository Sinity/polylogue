from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest

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
    ) -> None:
        self.call_rows = call_rows
        self.event_rows = event_rows or []
        self.queries: list[ToolUsageInsightQuery] = []
        self.event_queries: list[ToolUsageInsightQuery] = []

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


def _patch_tool_count_store(
    monkeypatch: pytest.MonkeyPatch,
    call_rows: list[dict[str, object]],
    event_rows: list[dict[str, object]] | None = None,
) -> _FakeToolCountStore:
    store = _FakeToolCountStore(call_rows, event_rows)
    monkeypatch.setattr(ArchiveStore, "open_existing", classmethod(lambda cls, archive_root: store))
    return store


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
async def test_tools_passes_filters_to_tool_usage_insight(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _env()
    store = _patch_tool_count_store(monkeypatch, [])

    await diagnostics._tools(
        env,
        origin="claude-code-session",
        tool="mcp__serena__find_symbol",
        mcp_server="serena",
        action_kind="tool_use",
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
        basis="tool-use-blocks",
        limit=5,
    )

    assert _console_print(env).call_args.args[0] == "No tool invocations found."
