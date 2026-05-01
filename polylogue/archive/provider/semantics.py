"""Shared semantic extraction helpers for provider adapters and harmonization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from polylogue.archive.viewport.viewports import (
    ContentBlock,
    ContentType,
    ReasoningTrace,
    ToolCall,
    classify_tool,
)
from polylogue.lib.json import JSONDocument, json_document
from polylogue.types import Provider


def content_text(value: object) -> str | None:
    """Extract text from string, dict, or list payloads used in content blocks."""
    if isinstance(value, str):
        return value
    record = json_document(value)
    if record:
        text = record.get("text")
        return text if isinstance(text, str) else None
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = content_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else None
    return None


def _string_field(block: Mapping[str, object], key: str) -> str | None:
    value = block.get(key)
    return value if isinstance(value, str) else None


def _content_block_record(value: object) -> JSONDocument | None:
    record = json_document(value)
    if record or isinstance(value, Mapping):
        return record
    return None


def extract_reasoning_traces(content: Sequence[object] | None, provider: Provider | str) -> list[ReasoningTrace]:
    """Extract reasoning traces from provider content blocks."""
    if not content:
        return []

    normalized_provider = provider if isinstance(provider, Provider) else Provider.from_string(provider)
    traces: list[ReasoningTrace] = []
    for raw_block in content:
        block = _content_block_record(raw_block)
        if block is None:
            continue
        block_type = block.get("type")
        text: str | None = None
        if block_type == "thinking":
            text = content_text(block.get("thinking")) or _string_field(block, "text")
        elif block.get("isThought"):
            text = _string_field(block, "text")

        if text:
            traces.append(ReasoningTrace(text=text, provider=normalized_provider, raw=block))

    return traces


def extract_tool_calls(content: Sequence[object] | None, provider: Provider | str) -> list[ToolCall]:
    """Extract tool calls from provider content blocks."""
    if not content:
        return []

    normalized_provider = provider if isinstance(provider, Provider) else Provider.from_string(provider)
    calls: list[ToolCall] = []
    for raw_block in content:
        block = _content_block_record(raw_block)
        if block is None:
            continue
        if block.get("type") != "tool_use":
            continue

        name = _string_field(block, "name") or ""
        normalized_input = json_document(block.get("input"))
        calls.append(
            ToolCall(
                name=name,
                id=_string_field(block, "id"),
                input=normalized_input,
                category=classify_tool(name, normalized_input),
                provider=normalized_provider,
                raw=block,
            )
        )

    return calls


def extract_content_blocks(content: Sequence[object] | None) -> list[ContentBlock]:
    """Extract harmonized content blocks with type classification."""
    if not content:
        return []

    blocks: list[ContentBlock] = []
    for raw_block in content:
        block = _content_block_record(raw_block)
        if block is None:
            continue
        block_type = block.get("type", "text")
        if block_type == "text":
            blocks.append(ContentBlock(type=ContentType.TEXT, text=_string_field(block, "text"), raw=block))
        elif block_type == "thinking":
            blocks.append(
                ContentBlock(
                    type=ContentType.THINKING,
                    text=content_text(block.get("thinking")) or _string_field(block, "text"),
                    raw=block,
                )
            )
        elif block_type == "tool_use":
            name = _string_field(block, "name") or ""
            normalized_input = json_document(block.get("input"))
            blocks.append(
                ContentBlock(
                    type=ContentType.TOOL_USE,
                    tool_call=ToolCall(
                        name=name,
                        id=_string_field(block, "id"),
                        input=normalized_input,
                        category=classify_tool(name, normalized_input),
                    ),
                    raw=block,
                )
            )
        elif block_type == "tool_result":
            blocks.append(
                ContentBlock(
                    type=ContentType.TOOL_RESULT,
                    text=content_text(block.get("content")) or _string_field(block, "text") or "",
                    raw=block,
                )
            )
        elif block_type == "code":
            blocks.append(
                ContentBlock(
                    type=ContentType.CODE,
                    text=_string_field(block, "text") or _string_field(block, "code"),
                    language=_string_field(block, "language"),
                    raw=block,
                )
            )

    return blocks


def extract_display_text_from_content_blocks(content: Sequence[object] | None) -> str:
    """Rebuild human-readable text from stored structured content blocks."""
    if not content:
        return ""

    parts: list[str] = []
    for raw_block in content:
        block = _content_block_record(raw_block)
        if block is None:
            continue
        if block.get("type") not in {"text", "code", "tool_result", "thinking"}:
            continue
        text = _string_field(block, "text")
        if text:
            parts.append(text)
    return "\n".join(parts)


def extract_claude_code_text(content: Sequence[object] | None) -> str:
    """Extract text from Claude Code content blocks, excluding non-text blocks."""
    if not content:
        return ""

    parts = []
    for raw_block in content:
        block = _content_block_record(raw_block)
        if block is None:
            continue
        if block.get("type") == "text":
            text = _string_field(block, "text")
            if text:
                parts.append(text)
    return "\n".join(filter(None, parts))


def extract_chatgpt_text(content: Mapping[str, object] | None) -> str:
    """Extract text from ChatGPT content structures."""
    if not content:
        return ""
    direct_text = _string_field(content, "text")
    if direct_text:
        return direct_text
    parts = content.get("parts", [])
    if not isinstance(parts, list):
        return str(parts) if parts else ""
    texts: list[str] = []
    for part in parts:
        if isinstance(part, str):
            texts.append(part)
            continue
        text = content_text(part)
        if text:
            texts.append(text)
    return "\n".join(texts)


def extract_codex_text(content: Sequence[object] | None) -> str:
    """Extract text from Codex content blocks."""
    if not content or not isinstance(content, list):
        return ""

    parts = []
    for raw_block in content:
        block = _content_block_record(raw_block)
        if block is None:
            continue
        text = _string_field(block, "text") or _string_field(block, "input_text") or _string_field(block, "output_text")
        if text:
            parts.append(text)
    return "\n".join(parts)
