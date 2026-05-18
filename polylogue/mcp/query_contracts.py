"""Shared MCP-side query normalization helpers."""

from __future__ import annotations

import inspect
import sys
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Annotated, Any, TypeAlias

from pydantic import Field

from polylogue.archive.message.types import validate_message_type_filter
from polylogue.archive.query.spec import ConversationQuerySpec
from polylogue.archive.semantic.content_projection import ContentProjectionSpec

MCPToolLimit: TypeAlias = Annotated[int, Field(ge=1)]
MCPToolOffset: TypeAlias = Annotated[int, Field(ge=0)]

_QUERY_PARAM_ALIASES = {
    "has_tool_use": "filter_has_tool_use",
    "has_thinking": "filter_has_thinking",
    "has_paste": "filter_has_paste",
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
    normalized = normalize_query_params(params)
    if normalized.get("message_type") is not None:
        normalized["message_type"] = validate_message_type_filter(normalized["message_type"]).value
    return ConversationQuerySpec.from_params(normalized)


@dataclass(frozen=True, slots=True)
class MCPConversationQueryRequest:
    """Typed MCP query request shared across query/search tool surfaces."""

    query: str | None = None
    retrieval_lane: str | None = None
    provider: str | None = None
    since: str | None = None
    until: str | None = None
    tag: str | None = None
    repo: str | None = None
    title: str | None = None
    contains: str | None = None
    exclude_text: str | None = None
    exclude_provider: str | None = None
    exclude_tag: str | None = None
    has_type: str | None = None
    conv_id: str | None = None
    referenced_path: str | None = None
    cwd_prefix: str | None = None
    action: str | None = None
    exclude_action: str | None = None
    action_sequence: str | None = None
    action_text: str | None = None
    tool: str | None = None
    exclude_tool: str | None = None
    sort: str | None = None
    reverse: bool = False
    latest: str | None = None
    has_tool_use: bool = False
    has_thinking: bool = False
    has_paste: bool = False
    typed_only: bool = False
    min_messages: int | None = None
    max_messages: int | None = None
    min_words: int | None = None
    sample: int | None = None
    similar_text: str | None = None
    since_session: str | None = None
    since_session_id: str | None = None
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
            until=self.until,
            tag=self.tag,
            repo=self.repo,
            title=self.title,
            contains=self.contains,
            exclude_text=self.exclude_text,
            exclude_provider=self.exclude_provider,
            exclude_tag=self.exclude_tag,
            has_type=self.has_type,
            conv_id=self.conv_id,
            referenced_path=self.referenced_path,
            cwd_prefix=self.cwd_prefix,
            action=self.action,
            exclude_action=self.exclude_action,
            action_sequence=self.action_sequence,
            action_text=self.action_text,
            tool=self.tool,
            exclude_tool=self.exclude_tool,
            sort=self.sort,
            reverse=self.reverse,
            latest=self.latest,
            limit=clamp_limit(self.limit),
            has_tool_use=self.has_tool_use,
            has_thinking=self.has_thinking,
            has_paste=self.has_paste,
            typed_only=self.typed_only,
            min_messages=self.min_messages,
            max_messages=self.max_messages,
            min_words=self.min_words,
            sample=self.sample,
            similar_text=self.similar_text,
            since_session=self.since_session,
            since_session_id=self.since_session_id,
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


def conversation_query_request_signature(
    *,
    include_query: bool,
) -> inspect.Signature:
    """Build an ``inspect.Signature`` mirroring ``MCPConversationQueryRequest``.

    ``include_query`` controls whether the ``query`` parameter is surfaced —
    ``search`` exposes it as required, ``list_conversations`` omits it. The
    resulting signature drives MCP ``inputSchema`` derivation so the JSON
    schema and the typed request model cannot drift.
    """
    type_hints = typing.get_type_hints(
        MCPConversationQueryRequest,
        globalns=vars(sys.modules[MCPConversationQueryRequest.__module__]),
        include_extras=True,
    )
    parameters: list[inspect.Parameter] = []
    for field in fields(MCPConversationQueryRequest):
        if field.name == "query" and not include_query:
            continue
        annotation = type_hints.get(field.name, Any)
        # ``search`` exposes ``query`` as a required parameter to preserve the
        # historical MCP tool surface — even though the dataclass default is
        # ``None`` so other call sites can build empty requests.
        if field.name == "query" and include_query:
            parameters.append(
                inspect.Parameter(
                    field.name,
                    inspect.Parameter.KEYWORD_ONLY,
                    annotation=str,
                )
            )
            continue
        parameters.append(
            inspect.Parameter(
                field.name,
                inspect.Parameter.KEYWORD_ONLY,
                default=field.default,
                annotation=annotation,
            )
        )
    return inspect.Signature(parameters=parameters, return_annotation=str)


def build_conversation_query_request(**kwargs: Any) -> MCPConversationQueryRequest:
    """Build an ``MCPConversationQueryRequest`` from MCP tool kwargs.

    Only known dataclass fields are forwarded; unexpected MCP parameters
    raise ``TypeError`` via the dataclass constructor rather than silently
    mutating filter state.
    """
    valid = {field.name for field in fields(MCPConversationQueryRequest)}
    payload = {name: value for name, value in kwargs.items() if name in valid}
    return MCPConversationQueryRequest(**payload)


__all__ = [
    "build_conversation_query_request",
    "build_query_spec",
    "conversation_query_request_signature",
    "MCPToolLimit",
    "MCPToolOffset",
    "MCPContentProjectionRequest",
    "MCPConversationQueryRequest",
    "normalize_query_params",
]
