"""Content-block-aware rendering for structure-preserving output.

Renders typed content blocks (thinking, tool_use, tool_result, code, etc.)
as visually distinct sections in markdown, HTML, and plaintext. This is
the core of structure-preserving rendering: instead of flattening everything
to ``message.text``, each block type gets its own presentation.

Usage::

    from polylogue.rendering.blocks import render_blocks_markdown

    markdown = render_blocks_markdown(message_content_blocks)
"""

from __future__ import annotations

import json
from typing import Any

from polylogue.lib.viewport_enums import ContentType


# -------------------------------------------------------------------
# Markdown rendering
# -------------------------------------------------------------------

def render_blocks_markdown(blocks: list[dict[str, Any]]) -> str:
    """Render a list of content block dicts to structure-preserving markdown.

    Each block is rendered according to its type:
    - ``text``: prose, passed through
    - ``thinking``: collapsible ``<details>`` section
    - ``tool_use``: tool header with name and input summary
    - ``tool_result``: code-fenced output
    - ``code``: language-tagged code fence
    - ``image``/``document``: reference with metadata
    """
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        rendered = _render_block_markdown(block)
        if rendered:
            parts.append(rendered)
    return "\n\n".join(parts) if parts else ""


def _render_block_markdown(block: dict[str, Any]) -> str:
    """Render a single content block to markdown."""
    block_type = block.get("type", "text")

    if block_type == "thinking":
        text = block.get("thinking") or block.get("text") or ""
        if not text.strip():
            return ""
        return (
            "<details><summary>Thinking</summary>\n\n"
            f"{text.strip()}\n\n"
            "</details>"
        )

    if block_type == "tool_use":
        name = block.get("name", "unknown")
        tool_input = block.get("input", {})
        summary = _tool_input_summary(name, tool_input)
        header = f"**Tool: {name}**"
        if summary:
            header += f" {summary}"
        return header

    if block_type == "tool_result":
        text = _extract_block_text(block)
        if not text.strip():
            return ""
        # Short results inline, long results fenced
        if len(text) < 200 and "\n" not in text:
            return f"→ {text.strip()}"
        return f"```\n{text.strip()}\n```"

    if block_type == "code":
        text = block.get("text") or block.get("code") or ""
        lang = block.get("language") or ""
        if not text.strip():
            return ""
        return f"```{lang}\n{text.strip()}\n```"

    if block_type in ("image", "document", "file"):
        url = block.get("url") or block.get("source", {}).get("url") or ""
        mime = block.get("media_type") or block.get("mime_type") or ""
        name = block.get("name") or block.get("title") or block_type
        parts = [f"[{name}]"]
        if url:
            parts[0] = f"[{name}]({url})"
        if mime:
            parts.append(f"({mime})")
        return " ".join(parts)

    # Default: text block
    text = block.get("text") or ""
    return text.strip() if text else ""


def _tool_input_summary(name: str, tool_input: Any) -> str:
    """Produce a short summary of tool input for display."""
    if not isinstance(tool_input, dict):
        return ""

    # Common tool patterns
    path = tool_input.get("file_path") or tool_input.get("path") or tool_input.get("file")
    if path:
        return f"`{path}`"

    command = tool_input.get("command")
    if command:
        # Truncate long commands
        if len(str(command)) > 80:
            return f"`{str(command)[:77]}...`"
        return f"`{command}`"

    pattern = tool_input.get("pattern")
    if pattern:
        return f"`{pattern}`"

    query = tool_input.get("query") or tool_input.get("prompt")
    if query:
        q = str(query)
        if len(q) > 60:
            return f'"{q[:57]}..."'
        return f'"{q}"'

    # Generic: show first key=value
    for key, value in tool_input.items():
        if isinstance(value, str) and len(value) < 60:
            return f"{key}={value}"
        break

    return ""


# -------------------------------------------------------------------
# HTML rendering
# -------------------------------------------------------------------

def render_blocks_html(blocks: list[dict[str, Any]]) -> str:
    """Render content blocks to semantic HTML."""
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        rendered = _render_block_html(block)
        if rendered:
            parts.append(rendered)
    return "\n".join(parts) if parts else ""


def _render_block_html(block: dict[str, Any]) -> str:
    """Render a single content block to HTML."""
    block_type = block.get("type", "text")

    if block_type == "thinking":
        text = block.get("thinking") or block.get("text") or ""
        if not text.strip():
            return ""
        from html import escape
        return (
            '<details class="thinking-block">\n'
            '  <summary class="thinking-label">Thinking</summary>\n'
            f'  <div class="thinking-content">{escape(text.strip())}</div>\n'
            '</details>'
        )

    if block_type == "tool_use":
        from html import escape
        name = escape(block.get("name", "unknown"))
        summary = _tool_input_summary(block.get("name", ""), block.get("input", {}))
        return (
            f'<div class="tool-use-block">'
            f'<span class="tool-name">{name}</span>'
            f'{" <code>" + escape(summary.strip("`\"")) + "</code>" if summary else ""}'
            f'</div>'
        )

    if block_type == "tool_result":
        from html import escape
        text = _extract_block_text(block)
        if not text.strip():
            return ""
        return f'<pre class="tool-result-block">{escape(text.strip())}</pre>'

    if block_type == "code":
        from html import escape
        text = block.get("text") or block.get("code") or ""
        lang = block.get("language") or ""
        if not text.strip():
            return ""
        lang_attr = f' data-language="{escape(lang)}"' if lang else ""
        return f'<pre class="code-block"{lang_attr}><code>{escape(text.strip())}</code></pre>'

    # Default: text
    text = block.get("text") or ""
    if not text.strip():
        return ""
    from html import escape
    # Simple paragraph wrapping for plain text
    paragraphs = text.strip().split("\n\n")
    return "\n".join(f"<p>{escape(p.strip())}</p>" for p in paragraphs if p.strip())


# -------------------------------------------------------------------
# Plaintext rendering
# -------------------------------------------------------------------

def render_blocks_plaintext(blocks: list[dict[str, Any]]) -> str:
    """Render content blocks to plain text (no formatting)."""
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "text")

        if block_type == "thinking":
            text = block.get("thinking") or block.get("text") or ""
            if text.strip():
                parts.append(f"[Thinking] {text.strip()}")

        elif block_type == "tool_use":
            name = block.get("name", "unknown")
            summary = _tool_input_summary(name, block.get("input", {}))
            parts.append(f"[Tool: {name}] {summary}".strip())

        elif block_type == "tool_result":
            text = _extract_block_text(block)
            if text.strip():
                parts.append(text.strip())

        else:
            text = block.get("text") or ""
            if text.strip():
                parts.append(text.strip())

    return "\n\n".join(parts) if parts else ""


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _extract_block_text(block: dict[str, Any]) -> str:
    """Extract displayable text from a content block."""
    text = block.get("text")
    if isinstance(text, str):
        return text
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts) if parts else ""
    return ""


def has_structured_blocks(blocks: list[dict[str, Any]] | None) -> bool:
    """Check if a block list contains non-text typed blocks worth rendering structurally."""
    if not blocks:
        return False
    return any(
        isinstance(b, dict) and b.get("type") in ("thinking", "tool_use", "tool_result", "code")
        for b in blocks
    )


__all__ = [
    "has_structured_blocks",
    "render_blocks_html",
    "render_blocks_markdown",
    "render_blocks_plaintext",
]
