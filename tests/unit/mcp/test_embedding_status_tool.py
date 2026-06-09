"""Runtime contract for MCP embedding readiness tools."""

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


def test_embedding_preflight_returns_canonical_payload(mcp_server: MCPServerUnderTest) -> None:
    report = MagicMock(name="report")
    payload = {
        "total_sessions": 10,
        "pending_sessions": 3,
        "pending_messages": 120,
        "estimated_tokens": 60_000,
        "estimated_cost_usd": 0.006,
        "model": "voyage-4",
        "dimension": 1024,
        "monthly_cost_cap_usd": 5.0,
        "effective_cost_cap_usd": 0.1,
        "windowed": True,
        "max_sessions": 3,
        "max_messages": 2000,
        "max_cost_usd": 0.1,
        "pricing": {
            "estimated_tokens_per_message": 500,
            "cost_usd_per_1m_tokens": 0.10,
            "approximate": True,
        },
        "backfill_args": ["embed", "backfill", "--yes", "--max-sessions", "3"],
        "backfill_command": "polylogue embed backfill --yes --max-sessions 3",
    }
    with (
        patch("polylogue.mcp.server._get_config", return_value=MagicMock(db_path="/tmp/archive.db")),
        patch("polylogue.storage.embeddings.preflight.build_preflight_report", return_value=report) as mock_build,
        patch("polylogue.storage.embeddings.preflight.preflight_payload", return_value=payload) as mock_payload,
    ):
        raw = invoke_surface(
            mcp_server._tool_manager._tools["embedding_preflight"].fn,
            max_sessions=3,
            max_messages=2000,
            max_cost_usd=0.1,
        )

    parsed = json.loads(raw)
    assert parsed == payload
    mock_build.assert_called_once_with(
        "/tmp/archive.db",
        rebuild=False,
        max_sessions=3,
        max_messages=2000,
        max_cost_usd=0.1,
    )
    mock_payload.assert_called_once_with(report)


def test_embedding_preflight_forwards_rebuild_flag(mcp_server: MCPServerUnderTest) -> None:
    with (
        patch("polylogue.mcp.server._get_config", return_value=MagicMock(db_path="/tmp/archive.db")),
        patch("polylogue.storage.embeddings.preflight.build_preflight_report", return_value=MagicMock()) as mock_build,
        patch(
            "polylogue.storage.embeddings.preflight.preflight_payload",
            return_value={"pending_sessions": 0, "backfill_command": None},
        ),
    ):
        invoke_surface(mcp_server._tool_manager._tools["embedding_preflight"].fn, rebuild=True)

    assert mock_build.call_args.kwargs["rebuild"] is True
