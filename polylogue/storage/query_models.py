"""Typed low-level query models for repository and SQLite read surfaces."""

from __future__ import annotations

from dataclasses import dataclass, replace

from typing_extensions import TypedDict


class ConversationListQueryKwargs(TypedDict):
    source: str | None
    provider: str | None
    providers: list[str] | None
    parent_id: str | None
    since: str | None
    until: str | None
    title_contains: str | None
    referenced_path: list[str] | None
    cwd_prefix: str | None
    action_terms: list[str] | None
    excluded_action_terms: list[str] | None
    tool_terms: list[str] | None
    excluded_tool_terms: list[str] | None
    repo_names: list[str] | None
    limit: int | None
    offset: int
    has_tool_use: bool
    has_thinking: bool
    has_paste: bool
    typed_only: bool
    min_messages: int | None
    max_messages: int | None
    min_words: int | None
    message_type: str | None


class ConversationCountQueryKwargs(TypedDict):
    source: str | None
    provider: str | None
    providers: list[str] | None
    since: str | None
    until: str | None
    title_contains: str | None
    referenced_path: list[str] | None
    cwd_prefix: str | None
    action_terms: list[str] | None
    excluded_action_terms: list[str] | None
    tool_terms: list[str] | None
    excluded_tool_terms: list[str] | None
    repo_names: list[str] | None
    has_tool_use: bool
    has_thinking: bool
    has_paste: bool
    typed_only: bool
    min_messages: int | None
    max_messages: int | None
    min_words: int | None
    message_type: str | None


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
    referenced_path: tuple[str, ...] = ()
    cwd_prefix: str | None = None
    action_terms: tuple[str, ...] = ()
    excluded_action_terms: tuple[str, ...] = ()
    tool_terms: tuple[str, ...] = ()
    excluded_tool_terms: tuple[str, ...] = ()
    repo_names: tuple[str, ...] = ()
    limit: int | None = None
    offset: int = 0
    has_tool_use: bool = False
    has_thinking: bool = False
    has_paste: bool = False
    typed_only: bool = False
    min_messages: int | None = None
    max_messages: int | None = None
    min_words: int | None = None
    since_session_id: str | None = None
    message_type: str | None = None

    def with_limit(self, limit: int | None) -> ConversationRecordQuery:
        return replace(self, limit=limit)

    def with_offset(self, offset: int) -> ConversationRecordQuery:
        return replace(self, offset=offset)

    def for_count(self) -> ConversationRecordQuery:
        return replace(self, limit=None, offset=0)

    def without_unstable_semantic_filters(self) -> ConversationRecordQuery:
        return replace(
            self,
            referenced_path=(),
            action_terms=(),
            excluded_action_terms=(),
        )

    def for_search(self) -> tuple[str | None, list[str] | None]:
        if self.provider:
            return self.provider, None
        if self.providers:
            return None, list(self.providers)
        return None, None

    def to_list_kwargs(self) -> ConversationListQueryKwargs:
        return {
            "source": self.source,
            "provider": self.provider,
            "providers": list(self.providers) or None,
            "parent_id": self.parent_id,
            "since": self.since,
            "until": self.until,
            "title_contains": self.title_contains,
            "referenced_path": list(self.referenced_path) or None,
            "cwd_prefix": self.cwd_prefix,
            "action_terms": list(self.action_terms) or None,
            "excluded_action_terms": list(self.excluded_action_terms) or None,
            "tool_terms": list(self.tool_terms) or None,
            "excluded_tool_terms": list(self.excluded_tool_terms) or None,
            "repo_names": list(self.repo_names) or None,
            "limit": self.limit,
            "offset": self.offset,
            "has_tool_use": self.has_tool_use,
            "has_thinking": self.has_thinking,
            "has_paste": self.has_paste,
            "typed_only": self.typed_only,
            "min_messages": self.min_messages,
            "max_messages": self.max_messages,
            "min_words": self.min_words,
            "message_type": self.message_type,
        }

    def to_count_kwargs(self) -> ConversationCountQueryKwargs:
        return {
            "source": self.source,
            "provider": self.provider,
            "providers": list(self.providers) or None,
            "since": self.since,
            "until": self.until,
            "title_contains": self.title_contains,
            "referenced_path": list(self.referenced_path) or None,
            "cwd_prefix": self.cwd_prefix,
            "action_terms": list(self.action_terms) or None,
            "excluded_action_terms": list(self.excluded_action_terms) or None,
            "tool_terms": list(self.tool_terms) or None,
            "excluded_tool_terms": list(self.excluded_tool_terms) or None,
            "repo_names": list(self.repo_names) or None,
            "has_tool_use": self.has_tool_use,
            "has_thinking": self.has_thinking,
            "has_paste": self.has_paste,
            "typed_only": self.typed_only,
            "min_messages": self.min_messages,
            "max_messages": self.max_messages,
            "min_words": self.min_words,
            "message_type": self.message_type,
        }


@dataclass(frozen=True, slots=True)
class SessionProfileListQuery:
    """Canonical session-profile read selection for storage/runtime boundaries."""

    provider: str | None = None
    since: str | None = None
    until: str | None = None
    first_message_since: str | None = None
    first_message_until: str | None = None
    session_date_since: str | None = None
    session_date_until: str | None = None
    min_wallclock_seconds: int | None = None
    max_wallclock_seconds: int | None = None
    sort: str = "source"
    tier: str = "merged"
    limit: int | None = 50
    offset: int = 0
    query: str | None = None


@dataclass(frozen=True, slots=True)
class SessionTimelineListQuery:
    """Canonical session timeline selection for work-event and phase reads."""

    conversation_id: str | None = None
    provider: str | None = None
    since: str | None = None
    until: str | None = None
    kind: str | None = None
    limit: int | None = 50
    offset: int = 0
    query: str | None = None


@dataclass(frozen=True, slots=True)
class WorkThreadListQuery:
    """Canonical work-thread read selection for repository/search runtime."""

    since: str | None = None
    until: str | None = None
    limit: int | None = 50
    offset: int = 0
    query: str | None = None


@dataclass(frozen=True, slots=True)
class SessionTagRollupListQuery:
    """Canonical tag-rollup selection for durable product aggregate reads."""

    provider: str | None = None
    since: str | None = None
    until: str | None = None
    query: str | None = None


@dataclass(frozen=True, slots=True)
class DaySessionSummaryListQuery:
    """Canonical day-summary selection for durable product aggregate reads."""

    provider: str | None = None
    since: str | None = None
    until: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactObservationListQuery:
    """Canonical durable-artifact observation selection for storage reads."""

    providers: tuple[str, ...] = ()
    support_statuses: tuple[str, ...] = ()
    artifact_kinds: tuple[str, ...] = ()
    limit: int | None = None
    offset: int = 0


__all__ = [
    "ArtifactObservationListQuery",
    "ConversationCountQueryKwargs",
    "ConversationListQueryKwargs",
    "ConversationRecordQuery",
    "DaySessionSummaryListQuery",
    "SessionProfileListQuery",
    "SessionTagRollupListQuery",
    "SessionTimelineListQuery",
    "WorkThreadListQuery",
]
