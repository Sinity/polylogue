"""Shared semantic extraction helpers for provider adapters and harmonization."""

from __future__ import annotations

from typing import Any

from polylogue.lib.viewports import (
    ContentBlock,
    ContentType,
    ReasoningTrace,
    ToolCall,
    classify_tool,
)
from polylogue.types import Provider


def content_text(value: object) -> str | None:
    """Extract text from string, dict, or list payloads used in content blocks."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = value.get("text")
        return text if isinstance(text, str) else None
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = content_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else None
    return None


def extract_reasoning_traces(content: list[dict[str, Any]] | None, provider: Provider | str) -> list[ReasoningTrace]:
    """Extract reasoning traces from provider content blocks."""
    if not content:
        return []

    traces: list[ReasoningTrace] = []
    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        text = None
        if block_type == "thinking":
            text = block.get("thinking") or block.get("text")
        elif block.get("isThought"):
            text = block.get("text")

        if text:
            traces.append(ReasoningTrace(text=text, provider=provider, raw=block))

    return traces


def extract_tool_calls(content: list[dict[str, Any]] | None, provider: Provider | str) -> list[ToolCall]:
    """Extract tool calls from provider content blocks."""
    if not content:
        return []

    calls: list[ToolCall] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue

        name = block.get("name", "")
        input_data = block.get("input", {})
        normalized_input = input_data if isinstance(input_data, dict) else {}
        calls.append(
            ToolCall(
                name=name,
                id=block.get("id"),
                input=normalized_input,
                category=classify_tool(name, normalized_input),
                provider=provider,
                raw=block,
            )
        )

    return calls


def extract_content_blocks(content: list[dict[str, Any]] | None) -> list[ContentBlock]:
    """Extract harmonized content blocks with type classification."""
    if not content:
        return []

    blocks: list[ContentBlock] = []
    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type", "text")
        if block_type == "text":
            blocks.append(ContentBlock(type=ContentType.TEXT, text=block.get("text"), raw=block))
        elif block_type == "thinking":
            blocks.append(
                ContentBlock(
                    type=ContentType.THINKING,
                    text=block.get("thinking") or block.get("text"),
                    raw=block,
                )
            )
        elif block_type == "tool_use":
            name = block.get("name", "")
            input_data = block.get("input", {})
            normalized_input = input_data if isinstance(input_data, dict) else {}
            blocks.append(
                ContentBlock(
                    type=ContentType.TOOL_USE,
                    tool_call=ToolCall(
                        name=name,
                        id=block.get("id"),
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
                    text=content_text(block.get("content")) or content_text(block.get("text")) or "",
                    raw=block,
                )
            )
        elif block_type == "code":
            blocks.append(
                ContentBlock(
                    type=ContentType.CODE,
                    text=block.get("text") or block.get("code"),
                    language=block.get("language"),
                    raw=block,
                )
            )

    return blocks


def extract_claude_code_text(content: list[dict[str, Any]] | None) -> str:
    """Extract text from Claude Code content blocks, excluding non-text blocks."""
    if not content:
        return ""

    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(filter(None, parts))


def extract_chatgpt_text(content: dict[str, Any] | None) -> str:
    """Extract text from ChatGPT content structures."""
    if not content:
        return ""
    direct_text = content.get("text")
    if isinstance(direct_text, str) and direct_text:
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


def extract_codex_text(content: list[dict[str, Any]] | None) -> str:
    """Extract text from Codex content blocks."""
    if not content or not isinstance(content, list):
        return ""

    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get("text", "") or block.get("input_text", "") or block.get("output_text", "")
        if text:
            parts.append(text)
    return "\n".join(parts)
