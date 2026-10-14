"""Declared Claude Code dispatcher and child-file identity coordinates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path


def claude_agent_transcript_stem(agent_id: str) -> str:
    """The provider claims the exact agent-<agentId> transcript stem."""
    return agent_id if agent_id.startswith("agent-") else f"agent-{agent_id}"


def claude_tool_result_identity(record: Mapping[str, object]) -> str | None:
    """Borrow one unambiguous result identity from the actual native record."""
    if record.get("type") != "user":
        return None
    message = record.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, Sequence) or isinstance(content, str | bytes | bytearray):
        return None
    identity: str | None = None
    for block in content:
        if not isinstance(block, Mapping) or block.get("type") != "tool_result":
            continue
        value = block.get("tool_use_id")
        if not isinstance(value, str) or not value:
            continue
        if identity is not None and value != identity:
            return None
        identity = value
    return identity


def claude_dispatch_operand(record: Mapping[str, object]) -> tuple[str, str] | None:
    """Borrow the unique result ID and the child named by its user record."""
    result = record.get("toolUseResult")
    agent_id = result.get("agentId") if isinstance(result, Mapping) else None
    if not isinstance(agent_id, str) or not agent_id:
        return None
    identity = claude_tool_result_identity(record)
    return None if identity is None else (identity, claude_agent_transcript_stem(agent_id))


def claude_agent_id_from_path(source_path: str) -> str | None:
    """Exact provider child filename claim shared by inference and admission."""
    name = Path(source_path).name
    if not name.startswith("agent-"):
        return None
    return name.removesuffix(".meta.json").removesuffix(".jsonl").removesuffix(".ndjson")


CLAUDE_AGENT_ID_FIELDS = ("agentId", "agent_id", "sessionId", "session_id")


def claude_agent_identity_operand(payload: Mapping[str, object]) -> tuple[str, str] | None:
    """The original metadata/journal alias precedence, without filename guessing."""
    for field in CLAUDE_AGENT_ID_FIELDS:
        value = payload.get(field)
        if isinstance(value, str) and value:
            return field, value
    return None
