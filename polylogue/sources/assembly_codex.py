"""Codex provider assembly — session_index.jsonl thread name sidecar."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from polylogue.core.enums import TitleSource
from polylogue.core.json import json_document
from polylogue.logging import get_logger

from .assembly import CodexThreadNames, SidecarData
from .parsers.base import ParsedSession

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
                entry = json_document(parsed)
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

    def enrich_session(
        self,
        conv: ParsedSession,
        sidecar_data: SidecarData,
    ) -> ParsedSession:
        """Enrich a Codex session with thread name or first-user-message title."""
        thread_names: CodexThreadNames = sidecar_data.get("thread_names", {})
        cid = conv.provider_session_id

        # Try thread name from side index
        name = thread_names.get(cid)
        if name and name != conv.title:
            return conv.model_copy(
                update={
                    "title": name,
                    "title_source": TitleSource.ORIGIN,
                }
            )

        # Fallback: use first user message as title if current title is just the session_id
        if conv.title == cid and conv.messages:
            for msg in conv.messages:
                if msg.role == "user" and msg.text and msg.text.strip():
                    preview = msg.text.strip()[:80]
                    if len(msg.text.strip()) > 80:
                        preview += "..."
                    return conv.model_copy(
                        update={
                            "title": preview,
                            "title_source": TitleSource.HEURISTIC,
                        }
                    )

        return conv


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
