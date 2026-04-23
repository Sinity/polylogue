"""Shared content-kind projection for conversation/message output surfaces."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

from polylogue.lib.messages import MessageCollection
from polylogue.lib.roles import Role

if TYPE_CHECKING:
    from polylogue.lib.conversation_models import Conversation
    from polylogue.lib.message_models import Message


class ContentKind(str, Enum):
    """Canonical projected content kinds for rendered/query output."""

    PROSE = "prose"
    CODE = "code"
    TOOL_CALL = "tool_call"
    TOOL_OUTPUT = "tool_output"
    FILE_READ = "file_read"
    REASONING = "reasoning"
    SYSTEM_NOISE = "system_noise"
    ATTACHMENT = "attachment"


@dataclass(frozen=True, slots=True)
class ContentProjectionSpec:
    """Shared output projection controls for conversation content."""

    include_prose: bool = True
    include_code: bool = True
    include_tool_calls: bool = True
    include_tool_outputs: bool = True
    include_file_reads: bool = True
    include_reasoning: bool = True
    include_system_noise: bool = True
    include_attachments: bool = True

    @classmethod
    def prose_only(cls) -> ContentProjectionSpec:
        return cls(
            include_code=False,
            include_tool_calls=False,
            include_tool_outputs=False,
            include_file_reads=False,
            include_reasoning=False,
            include_system_noise=False,
            include_attachments=False,
        )

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> ContentProjectionSpec:
        spec = cls.prose_only() if bool(params.get("prose_only")) else cls()
        if params.get("no_code_blocks"):
            spec = replace(spec, include_code=False)
        if params.get("no_tool_calls"):
            spec = replace(spec, include_tool_calls=False)
        if params.get("no_tool_outputs"):
            spec = replace(spec, include_tool_outputs=False)
        if params.get("no_file_reads"):
            spec = replace(spec, include_file_reads=False)
        return spec

    def is_default(self) -> bool:
        return self == type(self)()

    def filters_content(self) -> bool:
        return not self.is_default()


@dataclass(frozen=True, slots=True)
class _Segment:
    kind: ContentKind
    text: str | None
    block: dict[str, object] | None = None


_CODE_FENCE_PATTERN = re.compile(r"```(?P<lang>[^\n`]*)\n?(?P<code>.*?)```", re.DOTALL)
_THINKING_PATTERN = re.compile(r"<(?:antml:)?thinking>(?P<thinking>.*?)</(?:antml:)?thinking>", re.DOTALL)


def coerce_content_projection_spec(
    value: ContentProjectionSpec | Mapping[str, object] | None,
) -> ContentProjectionSpec:
    if value is None:
        return ContentProjectionSpec()
    if isinstance(value, ContentProjectionSpec):
        return value
    return ContentProjectionSpec.from_params(value)


def project_conversation_content(
    conversation: Conversation,
    projection: ContentProjectionSpec | Mapping[str, object] | None,
) -> Conversation:
    """Return a conversation with content-kind projection applied."""
    spec = coerce_content_projection_spec(projection)
    if spec.is_default():
        return conversation

    tool_semantics = _tool_semantics_by_id(list(conversation.messages))
    projected_messages = [
        projected
        for message in conversation.messages
        if (projected := _project_message_content(message, spec, tool_semantics)) is not None
    ]
    return conversation.model_copy(update={"messages": MessageCollection(messages=projected_messages)})


def _tool_semantics_by_id(messages: Sequence[Message]) -> dict[str, str]:
    semantics: dict[str, str] = {}
    for message in messages:
        for block in message.content_blocks:
            if block.get("type") != "tool_use":
                continue
            tool_id = block.get("tool_id") or block.get("id")
            semantic_type = block.get("semantic_type")
            if isinstance(tool_id, str) and tool_id and isinstance(semantic_type, str) and semantic_type:
                semantics[tool_id] = semantic_type
    return semantics


def _project_message_content(
    message: Message,
    spec: ContentProjectionSpec,
    tool_semantics: Mapping[str, str],
) -> Message | None:
    segments = _segments_for_message(message, tool_semantics)
    kept_segments = [segment for segment in segments if _keep_segment(segment, spec)]
    kept_blocks = [dict(segment.block) for segment in kept_segments if segment.block is not None]
    projected_text = _render_segments_text(kept_segments)
    projected_attachments = list(message.attachments) if spec.include_attachments else []

    if not projected_text and not kept_blocks and not projected_attachments:
        return None

    return message.model_copy(
        update={
            "text": projected_text or None,
            "content_blocks": kept_blocks,
            "attachments": projected_attachments,
        }
    )


def _segments_for_message(
    message: Message,
    tool_semantics: Mapping[str, str],
) -> list[_Segment]:
    if message.content_blocks:
        return _segments_from_blocks(message.content_blocks, tool_semantics)
    return _segments_from_text(message)


def _segments_from_blocks(
    blocks: Sequence[Mapping[str, object]],
    tool_semantics: Mapping[str, str],
) -> list[_Segment]:
    segments: list[_Segment] = []
    for raw_block in blocks:
        block = dict(raw_block)
        block_type = str(block.get("type") or "text")
        if block_type == "text":
            segments.append(_Segment(ContentKind.PROSE, _block_text(block), block=block))
            continue
        if block_type == "code":
            segments.append(_Segment(ContentKind.CODE, _block_text(block), block=block))
            continue
        if block_type == "thinking":
            segments.append(_Segment(ContentKind.REASONING, _block_text(block), block=block))
            continue
        if block_type == "tool_use":
            segments.append(_Segment(ContentKind.TOOL_CALL, _tool_call_text(block), block=block))
            continue
        if block_type == "tool_result":
            tool_id = block.get("tool_id") or block.get("tool_use_id") or block.get("id")
            semantic_type = tool_semantics.get(str(tool_id), "")
            kind = ContentKind.FILE_READ if semantic_type == "file_read" else ContentKind.TOOL_OUTPUT
            segments.append(_Segment(kind, _tool_result_text(block), block=block))
            continue
        if block_type in {"image", "document", "file"}:
            segments.append(_Segment(ContentKind.ATTACHMENT, _attachment_text(block), block=block))
            continue
        segments.append(_Segment(ContentKind.PROSE, _block_text(block), block=block))
    return segments


def _segments_from_text(message: Message) -> list[_Segment]:
    text = (message.text or "").strip()
    if not text:
        return []
    if message.role == Role.TOOL or message.is_tool_use:
        return [_Segment(ContentKind.TOOL_OUTPUT, text)]
    if message.is_context_dump or message.is_system:
        return [_Segment(ContentKind.SYSTEM_NOISE, text)]

    segments: list[_Segment] = []
    cursor = 0
    for match in _CODE_FENCE_PATTERN.finditer(text):
        segments.extend(_segments_from_noncode_text(text[cursor : match.start()]))
        code = (match.group("code") or "").strip()
        if code:
            segments.append(_Segment(ContentKind.CODE, code))
        cursor = match.end()
    segments.extend(_segments_from_noncode_text(text[cursor:]))

    if segments:
        return segments

    if message.is_thinking:
        thinking = message.extract_thinking()
        if thinking:
            return [_Segment(ContentKind.REASONING, thinking)]
    return [_Segment(ContentKind.PROSE, text)]


def _segments_from_noncode_text(text: str) -> list[_Segment]:
    stripped = text.strip()
    if not stripped:
        return []

    segments: list[_Segment] = []
    cursor = 0
    for match in _THINKING_PATTERN.finditer(text):
        prose = text[cursor : match.start()].strip()
        if prose:
            segments.append(_Segment(ContentKind.PROSE, prose))
        thinking = (match.group("thinking") or "").strip()
        if thinking:
            segments.append(_Segment(ContentKind.REASONING, thinking))
        cursor = match.end()
    tail = text[cursor:].strip()
    if tail:
        segments.append(_Segment(ContentKind.PROSE, tail))
    return segments


def _block_text(block: Mapping[str, object]) -> str | None:
    for key in ("text", "code", "content", "thinking"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _tool_call_text(block: Mapping[str, object]) -> str:
    name = block.get("tool_name") or block.get("name") or "unknown"
    summary = _tool_input_summary(block.get("tool_input") or block.get("input"))
    return f"[Tool: {name}] {summary}".strip()


def _tool_result_text(block: Mapping[str, object]) -> str | None:
    return _block_text(block)


def _attachment_text(block: Mapping[str, object]) -> str:
    name = block.get("name") or block.get("title") or block.get("type") or "attachment"
    url = block.get("url")
    mime = block.get("media_type") or block.get("mime_type")
    parts = [str(name)]
    if isinstance(url, str) and url:
        parts.append(url)
    if isinstance(mime, str) and mime:
        parts.append(f"({mime})")
    return " ".join(parts)


def _tool_input_summary(value: object) -> str:
    if not isinstance(value, Mapping):
        return ""
    path = value.get("file_path") or value.get("path") or value.get("file")
    if path:
        return f"`{path}`"
    command = value.get("command")
    if isinstance(command, str) and command:
        return f"`{command[:77]}...`" if len(command) > 80 else f"`{command}`"
    pattern = value.get("pattern")
    if pattern:
        return f"`{pattern}`"
    query = value.get("query") or value.get("prompt")
    if isinstance(query, str) and query:
        return f'"{query[:57]}..."' if len(query) > 60 else f'"{query}"'
    for key, item in value.items():
        if isinstance(item, str) and item and len(item) < 60:
            return f"{key}={item}"
    return ""


def _keep_segment(segment: _Segment, spec: ContentProjectionSpec) -> bool:
    kind = segment.kind
    if kind is ContentKind.PROSE:
        return spec.include_prose
    if kind is ContentKind.CODE:
        return spec.include_code
    if kind is ContentKind.TOOL_CALL:
        return spec.include_tool_calls
    if kind is ContentKind.TOOL_OUTPUT:
        return spec.include_tool_outputs
    if kind is ContentKind.FILE_READ:
        return spec.include_tool_outputs and spec.include_file_reads
    if kind is ContentKind.REASONING:
        return spec.include_reasoning
    if kind is ContentKind.SYSTEM_NOISE:
        return spec.include_system_noise
    if kind is ContentKind.ATTACHMENT:
        return spec.include_attachments
    return True


def _render_segments_text(segments: Sequence[_Segment]) -> str:
    parts = [segment.text.strip() for segment in segments if isinstance(segment.text, str) and segment.text.strip()]
    return "\n\n".join(parts)


__all__ = [
    "ContentKind",
    "ContentProjectionSpec",
    "coerce_content_projection_spec",
    "project_conversation_content",
]
