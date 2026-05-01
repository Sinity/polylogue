"""Shared MCP-side query normalization helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, TypeAlias

from pydantic import Field

from polylogue.lib.query.spec import ConversationQuerySpec
from polylogue.lib.semantic.content_projection import ContentProjectionSpec

MCPToolLimit: TypeAlias = Annotated[int, Field(ge=1)]
MCPToolOffset: TypeAlias = Annotated[int, Field(ge=0)]

_QUERY_PARAM_ALIASES = {
    "has_tool_use": "filter_has_tool_use",
    "has_thinking": "filter_has_thinking",
}


def normalize_query_params(params: Mapping[str, object]) -> dict[str, object]:
    """Normalize MCP query kwargs into substrate query-spec parameter names."""
    normalized = dict(params)
    for source_key, target_key in _QUERY_PARAM_ALIASES.items():
        if source_key in normalized:
            normalized[target_key] = normalized.pop(source_key)
    return normalized


def build_query_spec(**params: object) -> ConversationQuerySpec:
    """Build a ConversationQuerySpec from MCP-facing query kwargs."""
    return ConversationQuerySpec.from_params(normalize_query_params(params))


@dataclass(frozen=True, slots=True)
class MCPConversationQueryRequest:
    """Typed MCP query request shared across query/search tool surfaces."""

    query: str | None = None
    retrieval_lane: str | None = None
    provider: str | None = None
    since: str | None = None
    tag: str | None = None
    repo: str | None = None
    title: str | None = None
    referenced_path: str | None = None
    cwd_prefix: str | None = None
    action: str | None = None
    exclude_action: str | None = None
    action_sequence: str | None = None
    action_text: str | None = None
    tool: str | None = None
    exclude_tool: str | None = None
    sort: str | None = None
    has_tool_use: bool = False
    has_thinking: bool = False
    has_paste: bool = False
    typed_only: bool = False
    min_messages: int | None = None
    min_words: int | None = None
    since_session: str | None = None
    message_type: str | None = None
    offset: MCPToolOffset = 0
    limit: MCPToolLimit = 10

    def build_spec(self, clamp_limit: Callable[[int | object], int]) -> ConversationQuerySpec:
        """Build a ConversationQuerySpec from this request using the given clamp helper."""
        return build_query_spec(
            query=self.query,
            retrieval_lane=self.retrieval_lane or "auto",
            provider=self.provider,
            since=self.since,
            tag=self.tag,
            repo=self.repo,
            title=self.title,
            referenced_path=self.referenced_path,
            cwd_prefix=self.cwd_prefix,
            action=self.action,
            exclude_action=self.exclude_action,
            action_sequence=self.action_sequence,
            action_text=self.action_text,
            tool=self.tool,
            exclude_tool=self.exclude_tool,
            sort=self.sort,
            limit=clamp_limit(self.limit),
            has_tool_use=self.has_tool_use,
            has_thinking=self.has_thinking,
            has_paste=self.has_paste,
            typed_only=self.typed_only,
            min_messages=self.min_messages,
            min_words=self.min_words,
            since_session=self.since_session,
            message_type=self.message_type,
            offset=self.offset,
        )


@dataclass(frozen=True, slots=True)
class MCPContentProjectionRequest:
    """Typed MCP-side content projection request shared by read/export surfaces."""

    no_code_blocks: bool = False
    no_tool_calls: bool = False
    no_tool_outputs: bool = False
    no_file_reads: bool = False
    prose_only: bool = False

    def build_projection(self) -> ContentProjectionSpec:
        return ContentProjectionSpec.from_params(
            {
                "no_code_blocks": self.no_code_blocks,
                "no_tool_calls": self.no_tool_calls,
                "no_tool_outputs": self.no_tool_outputs,
                "no_file_reads": self.no_file_reads,
                "prose_only": self.prose_only,
            }
        )


__all__ = [
    "build_query_spec",
    "MCPToolLimit",
    "MCPToolOffset",
    "MCPContentProjectionRequest",
    "MCPConversationQueryRequest",
    "normalize_query_params",
]
