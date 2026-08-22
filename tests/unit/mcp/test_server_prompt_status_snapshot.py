"""The coordination MCP prompt consumes the shared status snapshot path."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from polylogue.mcp.server_prompts import register_prompts


class _PromptServer:
    def __init__(self) -> None:
        self.prompts: dict[str, object] = {}

    def prompt(self) -> Any:
        def decorate(function: Any) -> Any:
            self.prompts[function.__name__] = function
            return function

        return decorate


def test_coordination_prompt_uses_persistent_snapshot_for_compact_requests() -> None:
    server = _PromptServer()
    hooks = SimpleNamespace(clamp_limit=lambda limit: min(limit, 10))
    cached_payload = MagicMock()
    cached_payload.to_json.return_value = "{}"
    cache = MagicMock()
    cache.get_or_build.return_value = cached_payload

    with patch("polylogue.coordination.CoordinationEnvelopeCache", return_value=cache):
        register_prompts(server, hooks)  # type: ignore[arg-type]

    prompt = server.prompts["agent_coordination_brief"]
    asyncio.run(prompt(view="status", limit=100, detail=False))  # type: ignore[operator]

    cache.get_or_build.assert_called_once_with(view="status", cwd=None, limit=10)


def test_coordination_prompt_keeps_detail_as_explicit_uncached_query() -> None:
    server = _PromptServer()
    hooks = SimpleNamespace(clamp_limit=lambda limit: limit)
    detail_payload = MagicMock()
    detail_payload.to_json.return_value = "{}"
    cache = MagicMock()

    with (
        patch("polylogue.coordination.CoordinationEnvelopeCache", return_value=cache),
        patch("polylogue.coordination.build_coordination_envelope", return_value=detail_payload) as build,
    ):
        register_prompts(server, hooks)  # type: ignore[arg-type]
        prompt = server.prompts["agent_coordination_brief"]
        asyncio.run(prompt(view="handoff", limit=3, detail=True))  # type: ignore[operator]

    build.assert_called_once_with(view="handoff", limit=3, detail=True)
    cache.get_or_build.assert_not_called()
