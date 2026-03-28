"""Typed low-level query models for repository and SQLite read surfaces."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ConversationRecordQuery:
    """Canonical record-level conversation selection for storage reads."""

    source: str | None = None
    provider: str | None = None
    providers: tuple[str, ...] = ()
    parent_id: str | None = None
    since: str | None = None
    until: str | None = None
    title_contains: str | None = None
    limit: int | None = None
    offset: int = 0
    has_tool_use: bool = False
    has_thinking: bool = False
    min_messages: int | None = None
    max_messages: int | None = None
    min_words: int | None = None
    has_file_ops: bool = False
    has_git_ops: bool = False
    has_subagent: bool = False

    def with_limit(self, limit: int | None) -> ConversationRecordQuery:
        return replace(self, limit=limit)

    def with_offset(self, offset: int) -> ConversationRecordQuery:
        return replace(self, offset=offset)

    def for_count(self) -> ConversationRecordQuery:
        return replace(self, limit=None, offset=0)

    def for_search(self) -> tuple[str | None, list[str] | None]:
        if self.provider:
            return self.provider, None
        if self.providers:
            return None, list(self.providers)
        return None, None

    def to_list_kwargs(self) -> dict[str, object]:
        return {
            "source": self.source,
            "provider": self.provider,
            "providers": list(self.providers) or None,
            "parent_id": self.parent_id,
            "since": self.since,
            "until": self.until,
            "title_contains": self.title_contains,
            "limit": self.limit,
            "offset": self.offset,
            "has_tool_use": self.has_tool_use,
            "has_thinking": self.has_thinking,
            "min_messages": self.min_messages,
            "max_messages": self.max_messages,
            "min_words": self.min_words,
            "has_file_ops": self.has_file_ops,
            "has_git_ops": self.has_git_ops,
            "has_subagent": self.has_subagent,
        }

    def to_count_kwargs(self) -> dict[str, object]:
        data = self.to_list_kwargs()
        data.pop("limit")
        data.pop("offset")
        data.pop("parent_id")
        return data


__all__ = ["ConversationRecordQuery"]
