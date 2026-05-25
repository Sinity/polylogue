"""Runtime contract for the MCP ``embedding_status`` tool."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from tests.infra.mcp import MCPServerUnderTest, invoke_surface


def test_embedding_status_returns_canonical_payload(mcp_server: MCPServerUnderTest) -> None:
    payload = {
        "config_enabled": False,
        "has_voyage_api_key": True,
        "retrieval_ready": False,
        "status": "none",
        "next_action": {
            "code": "enable_embeddings",
            "command": "polylogue embed enable --yes",
            "reason": "A Voyage key is available, but embedding convergence is disabled in config.",
        },
    }
    with (
        patch("polylogue.mcp.server._get_config", return_value=MagicMock(name="config")) as mock_get_config,
        patch(
            "polylogue.storage.embeddings.status_payload.embedding_status_payload", return_value=payload
        ) as mock_status,
    ):
        raw = invoke_surface(mcp_server._tool_manager._tools["embedding_status"].fn)

    parsed = json.loads(raw)
    assert parsed == payload
    mock_get_config.assert_called_once()
    mock_status.assert_called_once()
    assert mock_status.call_args.kwargs == {
        "include_retrieval_bands": False,
        "include_detail": False,
    }


def test_embedding_status_detail_requests_exact_readiness_bands(mcp_server: MCPServerUnderTest) -> None:
    payload = {
        "config_enabled": True,
        "has_voyage_api_key": True,
        "retrieval_ready": True,
        "status": "complete",
        "retrieval_bands": {"message_embeddings": {"ready": True}},
        "next_action": {"code": "ready", "command": "polylogue --semantic <query>", "reason": "ready"},
    }
    with patch(
        "polylogue.storage.embeddings.status_payload.embedding_status_payload", return_value=payload
    ) as mock_status:
        raw = invoke_surface(mcp_server._tool_manager._tools["embedding_status"].fn, detail=True)

    parsed = json.loads(raw)
    assert parsed["retrieval_bands"] == {"message_embeddings": {"ready": True}}
    mock_status.assert_called_once()
    assert mock_status.call_args.kwargs == {
        "include_retrieval_bands": True,
        "include_detail": True,
    }
