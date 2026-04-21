"""Content-block-aware rendering for structure-preserving output."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from html import escape

from polylogue.rendering.block_models import RenderableBlock
from polylogue.types import ContentBlockType

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
    - ``image``/``document``/``file``: reference with metadata
    """
    parts: list[str] = []
    for block in blocks:
        rendered = _render_block_markdown(block)
        if rendered:
            parts.append(rendered)
    return "\n\n".join(parts) if parts else ""


def _render_block_markdown(block: RenderableBlock) -> str:
    """Render a single content block to markdown."""
    return _MARKDOWN_BLOCK_RENDERERS.get(block.type, _render_text_block_markdown)(block)


def _render_thinking_markdown(block: RenderableBlock) -> str:
    text = _strip_text(block.text)
    if not text:
        return ""
    return f"<details><summary>Thinking</summary>\n\n{text}\n\n</details>"


def _render_tool_use_markdown(block: RenderableBlock) -> str:
    name = block.tool_name or "unknown"
    summary = _tool_input_summary(name, block.tool_input)
    header = f"**Tool: {name}**"
    if summary:
        header += f" {summary}"
    return header


def _render_tool_result_markdown(block: RenderableBlock) -> str:
    text = _extract_block_text(block).strip()
    if not text:
        return ""
    if len(text) < 200 and "\n" not in text:
        return f"→ {text}"
    return f"```\n{text}\n```"


def _render_code_markdown(block: RenderableBlock) -> str:
    text = _strip_text(block.text)
    if not text:
        return ""
    return f"```{block.language or ''}\n{text}\n```"


def _render_media_markdown(block: RenderableBlock) -> str:
    url = block.url or ""
    mime = block.mime_type or ""
    name = block.name or block.type
    parts = [f"[{name}]"]
    if url:
        parts[0] = f"[{name}]({url})"
    if mime:
        parts.append(f"({mime})")
    return " ".join(parts)


def _render_text_block_markdown(block: RenderableBlock) -> str:
    return _strip_text(block.text)


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
    return _HTML_BLOCK_RENDERERS.get(block.type, _render_text_block_html)(block)


def _render_thinking_html(block: RenderableBlock) -> str:
    text = _strip_text(block.text)
    if not text:
        return ""
    return (
        '<details class="thinking-block">\n'
        '  <summary class="thinking-label">Thinking</summary>\n'
        f'  <div class="thinking-content">{escape(text)}</div>\n'
        "</details>"
    )


def _render_tool_use_html(block: RenderableBlock) -> str:
    name = escape(block.tool_name or "unknown")
    summary = _tool_input_summary(block.tool_name, block.tool_input)
    summary_html = ""
    if summary:
        summary_html = " <code>" + escape(summary.strip('`"')) + "</code>"
    return f'<div class="tool-use-block"><span class="tool-name">{name}</span>{summary_html}</div>'


def _render_tool_result_html(block: RenderableBlock) -> str:
    text = _strip_text(_extract_block_text(block))
    if not text:
        return ""
    return f'<pre class="tool-result-block">{escape(text)}</pre>'


def _render_code_html(block: RenderableBlock) -> str:
    text = _strip_text(block.text)
    if not text:
        return ""
    lang = block.language or ""
    lang_attr = f' data-language="{escape(lang)}"' if lang else ""
    return f'<pre class="code-block"{lang_attr}><code>{escape(text)}</code></pre>'


def _render_media_html(block: RenderableBlock) -> str:
    block_type = escape(block.type)
    name = escape(block.name or block.type)
    mime = escape(block.mime_type or "")
    url = block.url or ""
    parts = [f'<div class="media-block" data-type="{block_type}">']
    if url:
        parts.append(f'<a class="media-link" href="{escape(url)}">{name}</a>')
    else:
        parts.append(f'<span class="media-name">{name}</span>')
    if mime:
        parts.append(f'<span class="media-mime">{mime}</span>')
    parts.append("</div>")
    return "".join(parts)


def _render_text_block_html(block: RenderableBlock) -> str:
    # Simple paragraph wrapping for plain text
    if not block.text or not block.text.strip():
        return ""
    paragraphs = block.text.strip().split("\n\n")
    return "\n".join(f"<p>{escape(p.strip())}</p>" for p in paragraphs if p.strip())


# -------------------------------------------------------------------
# Plaintext rendering
# -------------------------------------------------------------------


def render_blocks_plaintext(blocks: Sequence[RenderableBlock]) -> str:
    """Render content blocks to plain text (no formatting)."""
    parts: list[str] = []
    for block in blocks:
        rendered = _render_block_plaintext(block)
        if rendered:
            parts.append(rendered)

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
    return any(block.type in _STRUCTURED_BLOCK_TYPES for block in blocks)


def _strip_text(value: str | None) -> str:
    return (value or "").strip()


def _render_block_plaintext(block: RenderableBlock) -> str:
    """Render a single content block to plaintext."""
    return _PLAIN_BLOCK_RENDERERS.get(block.type, _render_text_block_plaintext)(block)


_MEDIA_BLOCK_TYPES = frozenset(
    {
        ContentBlockType.IMAGE.value,
        ContentBlockType.DOCUMENT.value,
        "file",
    }
)


_MARKDOWN_BLOCK_RENDERERS: dict[str, Callable[[RenderableBlock], str]] = {
    ContentBlockType.THINKING.value: _render_thinking_markdown,
    ContentBlockType.TOOL_USE.value: _render_tool_use_markdown,
    ContentBlockType.TOOL_RESULT.value: _render_tool_result_markdown,
    ContentBlockType.CODE.value: _render_code_markdown,
    **dict.fromkeys(_MEDIA_BLOCK_TYPES, _render_media_markdown),
}


_HTML_BLOCK_RENDERERS: dict[str, Callable[[RenderableBlock], str]] = {
    ContentBlockType.THINKING.value: _render_thinking_html,
    ContentBlockType.TOOL_USE.value: _render_tool_use_html,
    ContentBlockType.TOOL_RESULT.value: _render_tool_result_html,
    ContentBlockType.CODE.value: _render_code_html,
    **dict.fromkeys(_MEDIA_BLOCK_TYPES, _render_media_html),
}


def _render_text_block_plaintext(block: RenderableBlock) -> str:
    return _strip_text(block.text)


def _render_tool_use_plaintext(block: RenderableBlock) -> str:
    name = block.tool_name or "unknown"
    summary = _tool_input_summary(name, block.tool_input)
    return f"[Tool: {name}] {summary}".strip()


def _render_thinking_plaintext(block: RenderableBlock) -> str:
    text = _strip_text(block.text)
    if not text:
        return ""
    return f"[Thinking] {text}"


def _render_tool_result_plaintext(block: RenderableBlock) -> str:
    return _strip_text(_extract_block_text(block))


def _render_media_plaintext(block: RenderableBlock) -> str:
    name = block.name or block.type
    parts = [name]
    if block.url:
        parts.append(block.url)
    if block.mime_type:
        parts.append(f"({block.mime_type})")
    return " ".join(parts)


_PLAIN_BLOCK_RENDERERS: dict[str, Callable[[RenderableBlock], str]] = {
    ContentBlockType.THINKING.value: _render_thinking_plaintext,
    ContentBlockType.TOOL_USE.value: _render_tool_use_plaintext,
    ContentBlockType.TOOL_RESULT.value: _render_tool_result_plaintext,
    **dict.fromkeys(_MEDIA_BLOCK_TYPES, _render_media_plaintext),
}


_STRUCTURED_BLOCK_TYPES = {
    ContentBlockType.THINKING.value,
    ContentBlockType.TOOL_USE.value,
    ContentBlockType.TOOL_RESULT.value,
    ContentBlockType.CODE.value,
} | set(_MEDIA_BLOCK_TYPES)


__all__ = [
    "has_structured_blocks",
    "render_blocks_html",
    "render_blocks_markdown",
    "render_blocks_plaintext",
]
