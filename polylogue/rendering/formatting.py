"""Session formatting functions for various output formats.

Provides format_session() for converting Session objects to:
- JSON, YAML
- HTML (with Pygments syntax highlighting)
- CSV (messages as rows)
- Markdown, plaintext
- Obsidian (YAML frontmatter + markdown)
- Org-mode

Used by both CLI query commands and the MCP server.
"""

from __future__ import annotations

import csv
import io
import json
from typing import TYPE_CHECKING

from polylogue.archive.semantic.content_projection import ContentProjectionSpec
from polylogue.surfaces.payloads import (
    JSONDocument,
    SessionDetailPayload,
    SessionListRowPayload,
    model_json_document,
)

if TYPE_CHECKING:
    from polylogue.archive.models import Session

SESSION_OUTPUT_FORMATS = (
    "markdown",
    "json",
    "html",
    "yaml",
    "plaintext",
    "csv",
    "obsidian",
    "org",
)


def normalize_session_output_format(output_format: str) -> str:
    """Return a supported session output format, falling back to markdown."""
    return output_format if output_format in SESSION_OUTPUT_FORMATS else "markdown"


def format_session(
    conv: Session,
    output_format: str,
    fields: str | None,
    content_projection: ContentProjectionSpec | None = None,
) -> str:
    """Format a single session for output.

    Args:
        conv: Session to format
        output_format: Output format (json, yaml, html, csv, obsidian, org, plaintext, markdown)
        fields: Optional comma-separated field selector for JSON/YAML output

    Returns:
        Formatted string
    """
    output_format = normalize_session_output_format(output_format)
    if content_projection is not None and content_projection.filters_content():
        conv = conv.with_content_projection(content_projection)

    if output_format == "json":
        return _conv_to_json(conv, fields)
    elif output_format == "yaml":
        return _conv_to_yaml(conv, fields)
    elif output_format == "html":
        return _conv_to_html(conv)
    elif output_format == "csv":
        return _conv_to_csv_messages(conv)
    elif output_format == "obsidian":
        return _conv_to_obsidian(conv)
    elif output_format == "org":
        return _conv_to_org(conv)
    elif output_format == "plaintext":
        return _conv_to_plaintext(conv)
    else:  # markdown
        return _conv_to_markdown(conv)


def _conv_to_dict(conv: Session, fields: str | None) -> JSONDocument:
    """Convert session to summary dict (message count, not content).

    Used for list-mode output where loading all message text is unnecessary.
    For full-content output, use _conv_to_json() instead.
    """
    selected = {field.strip() for field in fields.split(",")} if fields else None
    return SessionListRowPayload.from_session(conv).selected(selected)


def _conv_to_json(conv: Session, fields: str | None) -> str:
    """Convert a single session to full JSON with message content."""
    data = _conv_to_dict(conv, fields)
    if fields is None or "messages" in {field.strip() for field in fields.split(",")}:
        detail_payload = SessionDetailPayload.from_session(conv)
        data["messages"] = [
            model_json_document(message_payload, exclude_none=True) for message_payload in detail_payload.messages
        ]
    return json.dumps(data, indent=2)


def _conv_to_yaml(conv: Session, fields: str | None) -> str:
    """Convert session to YAML format.

    Args:
        conv: Session to format
        fields: Optional comma-separated field selector

    Returns:
        YAML-formatted string
    """
    import yaml

    data = _conv_to_dict(conv, fields)
    if fields is None or "messages" in {field.strip() for field in fields.split(",")}:
        detail_payload = SessionDetailPayload.from_session(conv)
        data["messages"] = [
            model_json_document(message_payload, exclude_none=True) for message_payload in detail_payload.messages
        ]

    return str(yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False))


def _conv_to_html(conv: Session) -> str:
    """Convert session to HTML with Pygments syntax highlighting.

    Delegates to the shared ``render_session_html`` function which
    uses the same Jinja2 template and Pygments highlighting as the
    rendering subsystem's ``HTMLRenderer``.
    """
    from polylogue.rendering.renderers.html import render_session_html

    return render_session_html(conv)


def _csv_safe(value: str) -> str:
    """Prevent CSV formula injection by prefixing dangerous leading characters."""
    if value and value[0] in ("=", "+", "-", "@"):
        return "'" + value
    return value


def _conv_to_csv_messages(conv: Session) -> str:
    """Convert a single session's messages to CSV rows."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["session_id", "message_id", "role", "timestamp", "text"])
    for msg in conv.messages:
        if not msg.text:
            continue
        writer.writerow(
            [
                str(conv.id),
                str(msg.id),
                _csv_safe(msg.role or ""),
                _csv_safe(msg.timestamp.isoformat() if msg.timestamp else ""),
                _csv_safe(msg.text),
            ]
        )
    return buf.getvalue().rstrip()


def _conv_to_markdown(conv: Session) -> str:
    """Convert session to markdown."""
    from polylogue.rendering.core_markdown import format_session_markdown

    return format_session_markdown(conv)


def _conv_to_plaintext(conv: Session) -> str:
    """Convert session to plain text (no markdown formatting).

    Strips all formatting, returning just the raw message content.
    Useful for piping to grep, wc, or other text processing tools.

    Args:
        conv: Session to format

    Returns:
        Plain text with messages separated by blank lines
    """
    lines = []

    for msg in conv.messages:
        if msg.text:
            # Just the raw text, no role labels or formatting
            lines.append(msg.text)
            lines.append("")

    return "\n".join(lines).strip()


def _yaml_safe(value: str) -> str:
    """Quote a YAML value if it contains special characters."""
    if any(c in value for c in ":#{}[]|>&*!?@`'\",\n\t"):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
        return f'"{escaped}"'
    return value


def _conv_to_obsidian(conv: Session) -> str:
    """Convert session to Obsidian-compatible markdown with YAML frontmatter."""
    tags_formatted = ", ".join(_yaml_safe(t) for t in conv.tags) if conv.tags else ""
    frontmatter = [
        "---",
        f"id: {_yaml_safe(str(conv.id))}",
        f"origin: {_yaml_safe(conv.origin.value)}",
        f"date: {conv.display_date.isoformat() if conv.display_date else 'unknown'}",
        f"tags: [{tags_formatted}]",
        "---",
        "",
    ]
    content = _conv_to_markdown(conv)
    return "\n".join(frontmatter) + content


def _conv_to_org(conv: Session) -> str:
    """Convert session to Org-mode format."""
    lines = [
        f"#+TITLE: {conv.display_title or conv.id}",
        f"#+DATE: {conv.display_date.strftime('%Y-%m-%d') if conv.display_date else 'unknown'}",
        f"#+PROPERTY: origin {conv.origin.value}",
        "",
    ]

    for msg in conv.messages:
        if not msg.text:
            continue
        role_label = (msg.role or "unknown").upper()
        lines.append(f"* {role_label}")
        lines.append(msg.text)
        lines.append("")

    return "\n".join(lines)
