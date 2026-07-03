"""Shared raw-materialization classification helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parsed_non_session_artifact_reason(
    *,
    archive_root: Path,
    origin: str,
    source_path: str,
    blob_hash: bytes | str | None,
) -> str | None:
    """Return why a parsed raw row should not materialize a session."""
    if _source_path_is_known_sidecar(source_path):
        return "source-path sidecar"
    leading_objects = raw_jsonl_leading_objects(_raw_blob_path(archive_root, blob_hash), limit=8)
    first_types = tuple(value for item in leading_objects if isinstance((value := item.get("type")), str) and value)
    if not leading_objects:
        return None
    if origin == "claude-code-session":
        if first_types and set(first_types) <= {"bridge-session", "last-prompt"}:
            return "Claude Code bridge/last-prompt sidecar"
        if first_types and set(first_types) <= {"file-history-snapshot", "progress"}:
            return "Claude Code file-history snapshot"
        if first_types and first_types[0] in {"custom-title", "started"}:
            return f"Claude Code {first_types[0]} sidecar"
        first_keys = set(leading_objects[0])
        if {"sessionId", "projectHash", "startTime", "lastUpdated", "kind"} <= first_keys:
            return "Claude Code metadata-only session descriptor"
    if origin == "claude-ai-export" and _claude_ai_empty_conversation(_raw_blob_path(archive_root, blob_hash)):
        return "Claude.ai empty conversation artifact"
    if origin == "codex-session" and set(first_types) == {"session_meta"}:
        return "Codex metadata-only session file"
    return None


def raw_jsonl_leading_objects(path: Path, *, limit: int) -> tuple[dict[str, Any], ...]:
    objects: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    return ()
                if not isinstance(payload, dict):
                    continue
                objects.append(payload)
                if len(objects) >= limit:
                    break
    except OSError:
        return ()
    return tuple(objects)


def _raw_blob_path(archive_root: Path, blob_hash: bytes | str | None) -> Path:
    blob_hash_hex = blob_hash.hex() if isinstance(blob_hash, bytes) else str(blob_hash or "")
    return archive_root / "blob" / blob_hash_hex[:2] / blob_hash_hex[2:]


def _claude_ai_empty_conversation(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    messages = payload.get("chat_messages")
    if not isinstance(messages, list):
        return False
    if not messages:
        return True
    for message in messages:
        if not isinstance(message, dict):
            return False
        text = message.get("text")
        content = message.get("content")
        attachments = message.get("attachments")
        files = message.get("files")
        if isinstance(text, str) and text.strip():
            return False
        if _claude_ai_content_has_content(content):
            return False
        if _claude_ai_attachments_have_content(attachments):
            return False
        if _claude_ai_attachments_have_content(files):
            return False
    return True


def _claude_ai_content_has_content(value: object) -> bool:
    if not value:
        return False
    if not isinstance(value, list):
        return True
    ignored_keys = {"citations", "flags", "start_timestamp", "stop_timestamp", "type"}
    for item in value:
        if not isinstance(item, dict):
            return True
        for key, item_value in item.items():
            if key in ignored_keys:
                continue
            if isinstance(item_value, str) and item_value.strip():
                return True
            if item_value and not isinstance(item_value, str):
                return True
    return False


def _claude_ai_attachments_have_content(value: object) -> bool:
    if not value:
        return False
    if not isinstance(value, list):
        return True
    ignored_keys = {"file_type", "file_uuid", "uuid", "id"}
    for item in value:
        if not isinstance(item, dict):
            return True
        for key, item_value in item.items():
            if key in ignored_keys:
                continue
            if isinstance(item_value, str) and item_value.strip():
                return True
            if isinstance(item_value, int | float) and item_value > 0:
                return True
            if item_value and not isinstance(item_value, str | int | float):
                return True
    return False


def _source_path_is_known_sidecar(source_path: str) -> bool:
    if not source_path:
        return False
    return any(
        marker in source_path
        for marker in (
            "/analysis/",
            "/subagents/workflows/",
            "/history.jsonl",
            "/sessions-index.json",
        )
    )


__all__ = ["parsed_non_session_artifact_reason", "raw_jsonl_leading_objects"]
