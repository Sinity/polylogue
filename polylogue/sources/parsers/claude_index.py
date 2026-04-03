"""Claude Code sessions-index parsing and enrichment helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polylogue.logging import get_logger

from .base import ParsedConversation

logger = get_logger(__name__)


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
    def from_dict(cls, data: dict[str, Any]) -> SessionIndexEntry:
        return cls(
            session_id=data.get("sessionId", ""),
            full_path=data.get("fullPath", ""),
            first_prompt=data.get("firstPrompt"),
            summary=data.get("summary"),
            message_count=data.get("messageCount", 0),
            created=data.get("created"),
            modified=data.get("modified"),
            git_branch=data.get("gitBranch"),
            project_path=data.get("projectPath"),
            is_sidechain=data.get("isSidechain", False),
            file_mtime=data.get("fileMtime"),
        )


def parse_sessions_index(index_path: Path) -> dict[str, SessionIndexEntry]:
    if not index_path.exists():
        return {}
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        entries = data.get("entries", [])
        return {
            entry["sessionId"]: SessionIndexEntry.from_dict(entry)
            for entry in entries
            if isinstance(entry, dict) and "sessionId" in entry
        }
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.debug("Failed to parse sessions-index.json: %s", exc)
        return {}


def find_sessions_index(session_path: Path) -> Path | None:
    index_path = session_path.parent / "sessions-index.json"
    return index_path if index_path.exists() else None


_GIT_BRANCH_PREFIXES = frozenset({
    "feature/", "fix/", "bugfix/", "hotfix/", "release/",
    "chore/", "refactor/", "test/", "docs/", "ci/", "perf/",
})

_GIT_BRANCH_EXACT = frozenset({
    "main", "master", "develop", "dev", "staging", "production",
    "HEAD", "head",
})


def _looks_like_git_branch(value: str) -> bool:
    """Return True if value looks like a git branch name rather than a title."""
    stripped = value.strip()
    if stripped in _GIT_BRANCH_EXACT:
        return True
    for prefix in _GIT_BRANCH_PREFIXES:
        if stripped.startswith(prefix):
            return True
    return False


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
    provider_meta.update({
        "gitBranch": index_entry.git_branch,
        "projectPath": index_entry.project_path,
        "isSidechain": index_entry.is_sidechain,
        "summary": index_entry.summary,
        "firstPrompt": index_entry.first_prompt,
        "title_source": title_source,
    })

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
