"""Claude Code sessions-index parsing and enrichment helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from polylogue.logging import get_logger

from ..base import ParsedConversation

logger = get_logger(__name__)

SessionIndexMapping: TypeAlias = Mapping[str, object]


@dataclass
class SessionIndexEntry:
    """Parsed entry from Claude Code sessions-index.json."""

    session_id: str
    full_path: str
    first_prompt: str | None
    summary: str | None
    message_count: int
    created: str | None
    modified: str | None
    git_branch: str | None
    project_path: str | None
    is_sidechain: bool
    file_mtime: int | None = None

    @classmethod
    def from_dict(cls, data: SessionIndexMapping) -> SessionIndexEntry:
        return cls(
            session_id=_session_index_text(data, "sessionId"),
            full_path=_session_index_text(data, "fullPath"),
            first_prompt=_session_index_optional_text(data, "firstPrompt"),
            summary=_session_index_optional_text(data, "summary"),
            message_count=_session_index_int(data, "messageCount"),
            created=_session_index_optional_text(data, "created"),
            modified=_session_index_optional_text(data, "modified"),
            git_branch=_session_index_optional_text(data, "gitBranch"),
            project_path=_session_index_optional_text(data, "projectPath"),
            is_sidechain=_session_index_bool(data, "isSidechain"),
            file_mtime=_session_index_optional_int(data, "fileMtime"),
        )


def parse_sessions_index(index_path: Path) -> dict[str, SessionIndexEntry]:
    if not index_path.exists():
        return {}
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        entries: dict[str, SessionIndexEntry] = {}
        for entry in _session_index_entries(data):
            session_id = _session_index_text(entry, "sessionId")
            if session_id:
                entries[session_id] = SessionIndexEntry.from_dict(entry)
        return entries
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.debug("Failed to parse sessions-index.json: %s", exc)
        return {}


def find_sessions_index(session_path: Path) -> Path | None:
    index_path = session_path.parent / "sessions-index.json"
    return index_path if index_path.exists() else None


def _session_index_entries(data: object) -> list[SessionIndexMapping]:
    if not isinstance(data, Mapping):
        return []
    entries = data.get("entries")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, Mapping) and "sessionId" in entry]


def _session_index_text(data: SessionIndexMapping, key: str) -> str:
    value = data.get(key)
    return value if isinstance(value, str) else ""


def _session_index_optional_text(data: SessionIndexMapping, key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) else None


def _session_index_int(data: SessionIndexMapping, key: str) -> int:
    value = data.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _session_index_optional_int(data: SessionIndexMapping, key: str) -> int | None:
    value = data.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _session_index_bool(data: SessionIndexMapping, key: str) -> bool:
    value = data.get(key)
    return value if isinstance(value, bool) else False


_GIT_BRANCH_PREFIXES = frozenset(
    {
        "feature/",
        "fix/",
        "bugfix/",
        "hotfix/",
        "release/",
        "chore/",
        "refactor/",
        "test/",
        "docs/",
        "ci/",
        "perf/",
    }
)

_GIT_BRANCH_EXACT = frozenset(
    {
        "main",
        "master",
        "develop",
        "dev",
        "staging",
        "production",
        "HEAD",
        "head",
    }
)


def _looks_like_git_branch(value: str) -> bool:
    """Return True if value looks like a git branch name rather than a title."""
    stripped = value.strip()
    if stripped in _GIT_BRANCH_EXACT:
        return True
    return any(stripped.startswith(prefix) for prefix in _GIT_BRANCH_PREFIXES)


def enrich_conversation_from_index(
    conv: ParsedConversation,
    index_entry: SessionIndexEntry,
) -> ParsedConversation:
    title = conv.title
    title_source = "original"
    if (
        index_entry.summary
        and index_entry.summary != "User Exits CLI Session"
        and not _looks_like_git_branch(index_entry.summary)
    ):
        title = index_entry.summary
        title_source = "session-index:summary"
    elif index_entry.first_prompt and index_entry.first_prompt != "No prompt":
        title = index_entry.first_prompt[:80]
        if len(index_entry.first_prompt) > 80:
            title += "..."
        title_source = "session-index:first-prompt"

    provider_meta = dict(conv.provider_meta) if conv.provider_meta else {}
    provider_meta.update(
        {
            "gitBranch": index_entry.git_branch,
            "projectPath": index_entry.project_path,
            "isSidechain": index_entry.is_sidechain,
            "summary": index_entry.summary,
            "firstPrompt": index_entry.first_prompt,
            "title_source": title_source,
        }
    )

    return ParsedConversation(
        provider_name=conv.provider_name,
        provider_conversation_id=conv.provider_conversation_id,
        title=title,
        created_at=index_entry.created or conv.created_at,
        updated_at=index_entry.modified or conv.updated_at,
        messages=conv.messages,
        attachments=conv.attachments,
        provider_meta=provider_meta,
    )


__all__ = [
    "SessionIndexEntry",
    "_looks_like_git_branch",
    "enrich_conversation_from_index",
    "find_sessions_index",
    "parse_sessions_index",
]
