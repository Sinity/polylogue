from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from polylogue.cli.commands import diagnostics
from polylogue.cli.shared.types import AppEnv
from polylogue.lib.query.spec import ConversationQuerySpec


def _env() -> AppEnv:
    ui = MagicMock()
    ui.console = MagicMock()
    return AppEnv(ui=ui, services=MagicMock())


def _console_print(env: AppEnv) -> MagicMock:
    return cast(MagicMock, env.ui.console.print)


def _get_repository_factory(env: AppEnv) -> MagicMock:
    return cast(MagicMock, env.services.get_repository)


def _conversation(*messages: object) -> SimpleNamespace:
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
    content_blocks: list[dict[str, object]] | None = None,
    provider_meta: dict[str, object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        role=role,
        timestamp=timestamp,
        text=text,
        content_blocks=content_blocks or [],
        provider_meta=provider_meta or {},
    )


@pytest.mark.asyncio
async def test_pace_reports_active_model_and_idle_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    conv = _conversation(
        _message("user", start),
        _message("assistant", start + timedelta(seconds=30)),
        _message("user", start + timedelta(seconds=120)),
    )

    async def fake_list(self: object, repository: object) -> list[SimpleNamespace]:
        return [conv]

    monkeypatch.setattr(ConversationQuerySpec, "list", fake_list)
    env = _env()

    await diagnostics._pace(env, None, limit=5, threshold=60)

    rendered = "\n".join(call.args[0] for call in _console_print(env).call_args_list if call.args)
    assert "Turns: 3" in rendered
    assert "[active]" in rendered
    assert "[user_idle]" in rendered


@pytest.mark.asyncio
async def test_pace_reports_sparse_conversations(monkeypatch: pytest.MonkeyPatch) -> None:
    conv = _conversation(_message("user", None))

    async def fake_list(self: object, repository: object) -> list[SimpleNamespace]:
        return [conv]

    monkeypatch.setattr(ConversationQuerySpec, "list", fake_list)
    env = _env()

    await diagnostics._pace(env, "conv-diagnostics", limit=5, threshold=60)

    assert "not enough messages" in _console_print(env).call_args.args[0]


@pytest.mark.asyncio
async def test_turns_reports_duration_thinking_tools_and_characters(monkeypatch: pytest.MonkeyPatch) -> None:
    conv = _conversation(
        _message(
            "assistant",
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            "assistant text",
            content_blocks=[
                {"type": "thinking", "text": "reason"},
                {"type": "tool_use", "name": "bash"},
                {"type": "tool_result", "text": "ok"},
            ],
            provider_meta={"durationMs": 1250},
        )
    )

    async def fake_list(self: object, repository: object) -> list[SimpleNamespace]:
        return [conv]

    monkeypatch.setattr(ConversationQuerySpec, "list", fake_list)
    env = _env()

    await diagnostics._turns(env, "conv-diagnostics", limit=3)

    rendered = "\n".join(call.args[0] for call in _console_print(env).call_args_list if call.args)
    assert "1.2s" in rendered
    row = _console_print(env).call_args_list[-1].args[0]
    assert re.search(r"\b6\b", row)
    assert re.search(r"\b2\b", row)


@pytest.mark.asyncio
async def test_tools_aggregates_action_events(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = SimpleNamespace(id="conv-tools")

    async def fake_list_summaries(self: object, repository: object) -> list[SimpleNamespace]:
        return [summary]

    monkeypatch.setattr(ConversationQuerySpec, "list_summaries", fake_list_summaries)
    env = _env()
    _get_repository_factory(env).return_value.get_action_events = AsyncMock(
        return_value=[
            SimpleNamespace(normalized_tool_name="bash", tool_name=None),
            SimpleNamespace(normalized_tool_name=None, tool_name="read"),
            SimpleNamespace(normalized_tool_name="bash", tool_name=None),
        ]
    )

    await diagnostics._tools(env, provider=None, limit=5)

    rendered = "\n".join(call.args[0] for call in _console_print(env).call_args_list if call.args)
    assert "Top tools" in rendered
    assert "bash" in rendered
    assert "read" in rendered


@pytest.mark.asyncio
async def test_tools_reports_empty_archives(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_list_summaries(self: object, repository: object) -> list[SimpleNamespace]:
        return []

    monkeypatch.setattr(ConversationQuerySpec, "list_summaries", fake_list_summaries)
    env = _env()

    await diagnostics._tools(env, provider=None, limit=5)

    assert _console_print(env).call_args.args[0] == "No tool invocations found."
