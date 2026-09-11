"""Structured evidence must reach API and MCP without inventing accounting."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from polylogue import Polylogue
from polylogue.core.enums import BlockType, Provider, Role
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async


def _seed(root: Path, *, reset: bool = False) -> str:
    with ArchiveStore(root) as archive:
        parsed = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="orchestration-example",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.ASSISTANT,
                    model_name="recorded-model",
                    timestamp="2026-01-01T10:00:00+00:00",
                    blocks=[
                        ParsedContentBlock(
                            type=BlockType.TOOL_USE,
                            tool_name="Agent",
                            tool_id="launch-1",
                            tool_input={"model": "requested-model", "prompt": "private instructions"},
                        ),
                        ParsedContentBlock(
                            type=BlockType.TOOL_USE,
                            tool_name="exec_command",
                            tool_id="bead-1",
                            tool_input={"cmd": "bd show example-a12.3"},
                        ),
                    ],
                ),
                ParsedMessage(
                    provider_message_id="m2",
                    role=Role.USER,
                    text="Agent named requested-model finished example-fake.1",
                ),
            ],
            session_events=[
                ParsedSessionEvent(
                    event_type="turn_context",
                    timestamp="2026-01-01T09:59:00+00:00",
                    payload={"model": "configured-model"},
                ),
                *[
                    ParsedSessionEvent(
                        event_type="token_count",
                        timestamp=f"2026-01-01T10:0{index}:00+00:00",
                        payload={
                            "source_index": index,
                            "total_token_usage": {"input_tokens": count, "output_tokens": 10},
                            "last_token_usage": {"input_tokens": 5},
                            "rate_limits": {"primary": {"used_percent": 25, "window_minutes": 300}},
                        },
                    )
                    for index, count in enumerate([100, 100, 20 if reset else 120], start=1)
                ],
                ParsedSessionEvent(
                    event_type="rate_limits",
                    timestamp="2026-01-01T10:01:00+00:00",
                    payload={
                        "source_index": 1,
                        "rate_limits": {"primary": {"used_percent": 25, "window_minutes": 300}},
                    },
                ),
                ParsedSessionEvent(event_type="collab_agent_spawn_end", payload={"status": "completed"}),
            ],
        )
        archive.write_raw_and_parsed(
            parsed,
            payload=b'{"synthetic":true}',
            source_path="/private/example.jsonl",
            acquired_at_ms=1767265200000,
        )
    return "codex-session:orchestration-example"


@pytest.mark.asyncio
async def test_api_and_mcp_preserve_counter_and_native_evidence(tmp_path: Path) -> None:
    """Summing cumulative samples, guessing a model, or bypassing the owner fails."""
    from polylogue.mcp.server import build_server

    root = tmp_path / "archive"
    session_id = _seed(root)
    owner = Polylogue(archive_root=root)
    evidence = await owner.get_session_orchestration(session_id)
    assert evidence is not None
    payload = evidence.model_dump(mode="json")
    assert payload["usage"]["tokens"] == {"input_tokens": 120, "output_tokens": 10}
    assert len(payload["usage"]["observations"]) == 3
    assert payload["usage"]["quota_consumed_tokens"] is None
    assert payload["rate_limits"][0]["windows"]["primary"]["used_percent"] == 25
    assert payload["rate_limits"][0]["raw_id"] is None
    assert payload["rate_limits"][0]["event_id"]
    assert payload["rate_limits"][0]["source_index"] == 1
    assert {row["bead_id"] for row in payload["bead_mentions"]} == {"example-a12.3"}
    request = next(row for row in payload["launches"] if row["basis"] == "tool_request")
    assert request["requested_model"] == "requested-model"
    assert request["actual_model"] is None
    assert request["child_session_id"] is None
    assert request["block_id"]
    assert {row["basis"] for row in payload["model_segments"]} == {"recorded_message_model", "configured_turn_model"}
    assert payload["coverage"]["ingestion_watermark"]["raw_id"]
    assert payload["coverage"]["observed_at"]
    assert payload["coverage"]["complete"] is False
    assert "native_spawn_child_identity_not_retained" in payload["gaps"]
    assert "private instructions" not in json.dumps(payload)
    assert "/private/example.jsonl" not in json.dumps(payload)

    server = cast(MCPServerUnderTest, build_server())
    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=owner),
    ):
        result = json.loads(
            await invoke_surface_async(
                server._tool_manager._tools["get"].fn,
                ref=f"session:{session_id}",
                projection="orchestration",
            )
        )
        missing = json.loads(
            await invoke_surface_async(
                server._tool_manager._tools["get"].fn,
                ref="session:codex-session:missing",
                projection="orchestration",
            )
        )
    assert result == payload
    assert missing["code"] == "not_found"


@pytest.mark.asyncio
async def test_reset_does_not_claim_a_session_total(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    session_id = _seed(root, reset=True)
    evidence = await Polylogue(archive_root=root).get_session_orchestration(session_id)
    assert evidence is not None
    assert evidence.usage["tokens"] is None
    assert evidence.usage["latest_cumulative"] == {"input_tokens": 20, "output_tokens": 10}
    assert "cumulative_usage_counter_reset" in evidence.gaps


@pytest.mark.asyncio
async def test_absent_measurements_are_null_not_zero(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    with ArchiveStore(root) as archive:
        archive.write_raw_and_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="empty-evidence",
                messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="hello")],
            ),
            payload=b"{}",
            source_path="/example.jsonl",
            acquired_at_ms=1767265200000,
        )
    evidence = await Polylogue(archive_root=root).get_session_orchestration("codex-session:empty-evidence")
    assert evidence is not None
    assert evidence.usage["tokens"] is None
    assert evidence.model_segments == []
    assert evidence.rate_limits == []
    assert evidence.outcome == "degraded"


def test_parser_retains_quota_windows_and_native_spawn_identity_through_storage(tmp_path: Path) -> None:
    """The real parser/write/read route must retain quota independently of usage lowering."""
    from polylogue.api.sync import SyncPolylogue
    from polylogue.sources.parsers.codex import parse

    root = tmp_path / "archive"
    parsed = parse(
        [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "start"}],
                },
            },
            {
                "type": "event_msg",
                "timestamp": "2026-01-01T10:01:00Z",
                "payload": {
                    "type": "token_count",
                    "info": {"total_token_usage": {"input_tokens": 100, "output_tokens": 10}},
                    "rate_limits": {"primary": {"used_percent": 0, "window_minutes": 300, "resets_in_seconds": 90}},
                },
            },
            {
                "type": "event_msg",
                "timestamp": "2026-01-01T10:02:00Z",
                "payload": {
                    "type": "collab_agent_spawn_end",
                    "new_thread_id": "native-child",
                    "new_agent_nickname": "display-only",
                },
            },
        ],
        "native-evidence",
    )
    with ArchiveStore(root) as archive:
        archive.write_raw_and_parsed(parsed, payload=b"{}", source_path="/example.jsonl", acquired_at_ms=1767265200000)
    with SyncPolylogue(archive_root=root) as owner:
        evidence = owner.get_session_orchestration("codex-session:native-evidence")
    assert evidence is not None
    assert evidence.usage["tokens"] == {"input_tokens": 100, "output_tokens": 10}
    assert evidence.rate_limits[0]["windows"] == {
        "primary": {"used_percent": 0, "window_minutes": 300, "resets_in_seconds": 90}
    }
    spawn = next(row for row in evidence.launches if row["basis"] == "native_spawn_event")
    assert spawn["child_native_id"] == "native-child"
    assert spawn["child_session_id"] is None
    assert spawn["actual_model"] is None


def test_projection_uses_stored_edges_and_excludes_inherited_calls() -> None:
    """Inherited launch calls and human prose must not create extra children or task evidence."""
    from polylogue.analysis.orchestration_evidence import build_session_orchestration
    from polylogue.analysis.topology import SessionTopology, TopologyEdge, TopologyNode
    from polylogue.archive.message.models import Message
    from polylogue.archive.session.domain_models import Session

    session = Session(
        id="codex-session:parent",
        origin="codex-session",
        messages=[
            Message(
                id="codex-session:ancestor:n:m1",
                role="assistant",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "Agent",
                        "tool_input": {"model": "inherited"},
                    }
                ],
            ),
            Message(
                id="codex-session:parent:n:m2",
                role="assistant",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_name": "exec_command",
                        "tool_input": {"cmd": "bd create --title example-title"},
                    }
                ],
            ),
        ],
    )
    topology = SessionTopology(
        target_id=session.id,
        root_id=session.id,
        nodes=[TopologyNode(session_id=session.id), TopologyNode(session_id="codex-session:child")],
        edges=[TopologyEdge(parent_id=session.id, child_id="codex-session:child", kind="subagent")],
    )
    evidence = build_session_orchestration(session, topology)
    assert [row["session_id"] for row in evidence.children] == ["codex-session:child"]
    assert evidence.launches == []
    assert evidence.bead_mentions == []
    assert evidence.coverage["message_count"] == 1
