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
        if first_types and set(first_types) <= {"file-history-snapshot", "progress"}:
            return "Claude Code file-history snapshot"
        if first_types and first_types[0] in {"custom-title", "started"}:
            return f"Claude Code {first_types[0]} sidecar"
        first_keys = set(leading_objects[0])
        if {"sessionId", "projectHash", "startTime", "lastUpdated", "kind"} <= first_keys:
            return "Claude Code metadata-only session descriptor"
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
