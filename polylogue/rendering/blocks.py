"""Content-block-aware rendering for structure-preserving output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from polylogue.rendering.block_models import RenderableBlock

# -------------------------------------------------------------------
# Markdown rendering
# -------------------------------------------------------------------


def render_blocks_markdown(blocks: Sequence[RenderableBlock]) -> str:
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
        rendered = _render_block_markdown(block)
        if rendered:
            parts.append(rendered)
    return "\n\n".join(parts) if parts else ""


def _render_block_markdown(block: RenderableBlock) -> str:
    """Render a single content block to markdown."""
    block_type = block.type

    if block_type == "thinking":
        text = block.text or ""
        if not text.strip():
            return ""
        return f"<details><summary>Thinking</summary>\n\n{text.strip()}\n\n</details>"

    if block_type == "tool_use":
        name = block.tool_name or "unknown"
        tool_input = block.tool_input
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
        text = block.text or ""
        lang = block.language or ""
        if not text.strip():
            return ""
        return f"```{lang}\n{text.strip()}\n```"

    if block_type in ("image", "document", "file"):
        url = block.url or ""
        mime = block.mime_type or ""
        name = block.name or block_type
        parts = [f"[{name}]"]
        if url:
            parts[0] = f"[{name}]({url})"
        if mime:
            parts.append(f"({mime})")
        return " ".join(parts)

    # Default: text block
    text = block.text or ""
    return text.strip() if text else ""


def _tool_input_summary(name: str | None, tool_input: Mapping[str, object] | None) -> str:
    """Produce a short summary of tool input for display."""
    del name
    if tool_input is None:
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


def render_blocks_html(blocks: Sequence[RenderableBlock]) -> str:
    """Render content blocks to semantic HTML."""
    parts: list[str] = []
    for block in blocks:
        rendered = _render_block_html(block)
        if rendered:
            parts.append(rendered)
    return "\n".join(parts) if parts else ""


def _render_block_html(block: RenderableBlock) -> str:
    """Render a single content block to HTML."""
    block_type = block.type

    if block_type == "thinking":
        text = block.text or ""
        if not text.strip():
            return ""
        from html import escape

        return (
            '<details class="thinking-block">\n'
            '  <summary class="thinking-label">Thinking</summary>\n'
            f'  <div class="thinking-content">{escape(text.strip())}</div>\n'
            "</details>"
        )

    if block_type == "tool_use":
        from html import escape

        name = escape(block.tool_name or "unknown")
        summary = _tool_input_summary(block.tool_name, block.tool_input)
        summary_html = ""
        if summary:
            summary_html = " <code>" + escape(summary.strip('`"')) + "</code>"
        return f'<div class="tool-use-block"><span class="tool-name">{name}</span>{summary_html}</div>'

    if block_type == "tool_result":
        from html import escape

        text = _extract_block_text(block)
        if not text.strip():
            return ""
        return f'<pre class="tool-result-block">{escape(text.strip())}</pre>'

    if block_type == "code":
        from html import escape

        text = block.text or ""
        lang = block.language or ""
        if not text.strip():
            return ""
        lang_attr = f' data-language="{escape(lang)}"' if lang else ""
        return f'<pre class="code-block"{lang_attr}><code>{escape(text.strip())}</code></pre>'

    # Default: text
    text = block.text or ""
    if not text.strip():
        return ""
    from html import escape

    # Simple paragraph wrapping for plain text
    paragraphs = text.strip().split("\n\n")
    return "\n".join(f"<p>{escape(p.strip())}</p>" for p in paragraphs if p.strip())


# -------------------------------------------------------------------
# Plaintext rendering
# -------------------------------------------------------------------


def render_blocks_plaintext(blocks: Sequence[RenderableBlock]) -> str:
    """Render content blocks to plain text (no formatting)."""
    parts: list[str] = []
    for block in blocks:
        block_type = block.type

        if block_type == "thinking":
            text = block.text or ""
            if text.strip():
                parts.append(f"[Thinking] {text.strip()}")

        elif block_type == "tool_use":
            name = block.tool_name or "unknown"
            summary = _tool_input_summary(name, block.tool_input)
            parts.append(f"[Tool: {name}] {summary}".strip())

        elif block_type == "tool_result":
            text = _extract_block_text(block)
            if text.strip():
                parts.append(text.strip())

        else:
            text = block.text or ""
            if text.strip():
                parts.append(text.strip())

    return "\n\n".join(parts) if parts else ""


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _extract_block_text(block: RenderableBlock) -> str:
    """Extract displayable text from a content block."""
    return block.text or ""


def has_structured_blocks(blocks: Sequence[RenderableBlock] | None) -> bool:
    """Check if a block list contains non-text typed blocks worth rendering structurally."""
    if not blocks:
        return False
    return any(block.type in ("thinking", "tool_use", "tool_result", "code") for block in blocks)


__all__ = [
    "has_structured_blocks",
    "render_blocks_html",
    "render_blocks_markdown",
    "render_blocks_plaintext",
]
