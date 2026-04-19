"""Codex provider assembly — session_index.jsonl thread name sidecar."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from polylogue.logging import get_logger

from .assembly import SidecarData
from .parsers.base import ParsedConversation

logger = get_logger(__name__)


def _parse_codex_session_index(sessions_root: Path) -> dict[str, str]:
    """Parse ~/.codex/session_index.jsonl — append-only, newest entry wins per thread_id.

    Args:
        sessions_root: The ``sessions/`` directory. The index file lives at
            ``sessions_root.parent / "session_index.jsonl"``.

    Returns:
        Mapping of thread ID to thread name (latest entry wins).
    """
    index_path = sessions_root.parent / "session_index.jsonl"
    if not index_path.exists():
        return {}
    names: dict[str, str] = {}
    try:
        for line in index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    continue
                entry = cast(Mapping[str, object], parsed)
                tid = _coerce_codex_session_id(entry)
                name = _coerce_codex_thread_name(entry)
                if tid and name:
                    names[tid] = name  # Latest wins (append-only)
            except (json.JSONDecodeError, TypeError):
                continue
    except OSError as exc:
        logger.debug("Failed to read Codex session_index.jsonl: %s", exc)
    return names


class CodexAssemblySpec:
    """Codex provider assembly — session_index.jsonl thread name sidecar."""

    def discover_sidecars(self, source_paths: list[Path]) -> SidecarData:
        """Discover Codex thread names from session_index.jsonl.

        Returns ``{"thread_names": {thread_id: thread_name, ...}}``.
        """
        thread_names: dict[str, str] = {}
        seen_roots: set[Path] = set()
        for path in source_paths:
            # Walk up to find the sessions root
            for parent in path.parents:
                if parent.name == "sessions" and parent not in seen_roots:
                    seen_roots.add(parent)
                    thread_names.update(_parse_codex_session_index(parent))
                    break
        return {"thread_names": thread_names}

    def enrich_conversation(
        self,
        conv: ParsedConversation,
        sidecar_data: Mapping[str, Any],
    ) -> ParsedConversation:
        """Enrich a Codex conversation with thread name or first-user-message title."""
        thread_names = _coerce_thread_names(sidecar_data.get("thread_names"))
        cid = conv.provider_conversation_id

        # Try thread name from side index
        name = thread_names.get(cid)
        if name and name != conv.title:
            provider_meta = dict(conv.provider_meta) if conv.provider_meta else {}
            provider_meta["thread_name"] = name
            provider_meta["title_source"] = "session-index:thread-name"
            return ParsedConversation(
                provider_name=conv.provider_name,
                provider_conversation_id=conv.provider_conversation_id,
                title=name,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                messages=conv.messages,
                attachments=conv.attachments,
                provider_meta=provider_meta,
                parent_conversation_provider_id=conv.parent_conversation_provider_id,
                branch_type=conv.branch_type,
            )

        # Fallback: use first user message as title if current title is just the session_id
        if conv.title == cid and conv.messages:
            for msg in conv.messages:
                if msg.role == "user" and msg.text and msg.text.strip():
                    preview = msg.text.strip()[:80]
                    if len(msg.text.strip()) > 80:
                        preview += "..."
                    provider_meta = dict(conv.provider_meta) if conv.provider_meta else {}
                    provider_meta["title_source"] = "first-user-message"
                    return ParsedConversation(
                        provider_name=conv.provider_name,
                        provider_conversation_id=conv.provider_conversation_id,
                        title=preview,
                        created_at=conv.created_at,
                        updated_at=conv.updated_at,
                        messages=conv.messages,
                        attachments=conv.attachments,
                        provider_meta=provider_meta,
                        parent_conversation_provider_id=conv.parent_conversation_provider_id,
                        branch_type=conv.branch_type,
                    )

        return conv


def _coerce_thread_names(value: object | None) -> dict[str, str]:
    """Best-effort coercion to a thread-name sidecar map."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for thread_id, thread_name in value.items():
        if isinstance(thread_id, str) and isinstance(thread_name, str):
            result[thread_id] = thread_name
    return result


def _coerce_codex_session_id(entry: Mapping[str, object]) -> str | None:
    """Read a thread identifier from a parsed Codex session-index entry."""
    value = entry.get("id") or entry.get("thread_id")
    return value if isinstance(value, str) and value else None


def _coerce_codex_thread_name(entry: Mapping[str, object]) -> str | None:
    """Read a thread name from a parsed Codex session-index entry."""
    value = entry.get("thread_name") or entry.get("name")
    return value if isinstance(value, str) and value else None


__all__ = [
    "CodexAssemblySpec",
    "_parse_codex_session_index",
]
