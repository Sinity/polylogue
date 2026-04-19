"""Claude Code provider assembly — sessions-index.json sidecar discovery and enrichment."""

from __future__ import annotations

from pathlib import Path

from .assembly import SidecarData
from .parsers.base import ParsedConversation
from .parsers.claude_index import (
    SessionIndexEntry,
    enrich_conversation_from_index,
    parse_sessions_index,
)


class ClaudeCodeAssemblySpec:
    """Claude Code provider assembly — sessions-index.json sidecar."""

    def discover_sidecars(self, source_paths: list[Path]) -> SidecarData:
        """Discover Claude Code sessions-index.json sidecars.

        Returns ``{"session_index": {session_id: SessionIndexEntry, ...}}``.
        """
        indices: dict[Path, dict[str, SessionIndexEntry]] = {}
        for path in source_paths:
            parent = path.parent
            if parent not in indices:
                index_path = parent / "sessions-index.json"
                indices[parent] = parse_sessions_index(index_path)
        # Flatten to {session_id: SessionIndexEntry}
        flat: dict[str, SessionIndexEntry] = {}
        for entries in indices.values():
            flat.update(entries)
        return {"session_index": flat}

    def enrich_conversation(
        self,
        conv: ParsedConversation,
        sidecar_data: SidecarData,
    ) -> ParsedConversation:
        """Enrich a Claude Code conversation from the sessions-index sidecar."""
        data = sidecar_data.get("session_index")
        idx = _coerce_claude_code_session_index(data)
        if conv.provider_conversation_id in idx:
            return enrich_conversation_from_index(conv, idx[conv.provider_conversation_id])
        return conv


def _coerce_claude_code_session_index(
    value: object | None,
) -> dict[str, SessionIndexEntry]:
    """Best-effort coercion from dynamic sidecar data to session-index mapping."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, SessionIndexEntry] = {}
    for session_id, entry in value.items():
        if isinstance(session_id, str) and isinstance(entry, SessionIndexEntry):
            result[session_id] = entry
    return result


__all__ = [
    "ClaudeCodeAssemblySpec",
    "_coerce_claude_code_session_index",
]
