"""Conversation formatting functions for various output formats.

Provides format_conversation() for converting Conversation objects to:
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

from polylogue.surface_payloads import (
    ConversationDetailPayload,
    ConversationListRowPayload,
    JSONDocument,
    model_json_document,
)

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation


def format_conversation(
    conv: Conversation,
    output_format: str,
    fields: str | None,
) -> str:
    """Format a single conversation for output.

    Args:
        conv: Conversation to format
        output_format: Output format (json, yaml, html, csv, obsidian, org, plaintext, markdown)
        fields: Optional comma-separated field selector for JSON/YAML output

    Returns:
        Formatted string
    """
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


def _conv_to_dict(conv: Conversation, fields: str | None) -> JSONDocument:
    """Convert conversation to summary dict (message count, not content).

    Used for list-mode output where loading all message text is unnecessary.
    For full-content output, use _conv_to_json() instead.
    """
    selected = {field.strip() for field in fields.split(",")} if fields else None
    return ConversationListRowPayload.from_conversation(conv).selected(selected)


def _conv_to_json(conv: Conversation, fields: str | None) -> str:
    """Convert a single conversation to full JSON with message content."""
    data = _conv_to_dict(conv, fields)
    if fields is None or "messages" in {field.strip() for field in fields.split(",")}:
        detail_payload = ConversationDetailPayload.from_conversation(conv)
        data["messages"] = [
            model_json_document(message_payload, exclude_none=True) for message_payload in detail_payload.messages
        ]
    return json.dumps(data, indent=2)


def _conv_to_yaml(conv: Conversation, fields: str | None) -> str:
    """Convert conversation to YAML format.

    Args:
        conv: Conversation to format
        fields: Optional comma-separated field selector

    Returns:
        YAML-formatted string
    """
    import yaml

    data = _conv_to_dict(conv, fields)
    if fields is None or "messages" in {field.strip() for field in fields.split(",")}:
        detail_payload = ConversationDetailPayload.from_conversation(conv)
        data["messages"] = [
            model_json_document(message_payload, exclude_none=True) for message_payload in detail_payload.messages
        ]

    return str(yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False))


def _conv_to_html(conv: Conversation) -> str:
    """Convert conversation to HTML with Pygments syntax highlighting.

    Delegates to the shared ``render_conversation_html`` function which
    uses the same Jinja2 template and Pygments highlighting as the
    rendering subsystem's ``HTMLRenderer``.
    """
    from polylogue.rendering.renderers.html import render_conversation_html

    return render_conversation_html(conv)


def _conv_to_csv_messages(conv: Conversation) -> str:
    """Convert a single conversation's messages to CSV rows."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["conversation_id", "message_id", "role", "timestamp", "text"])
    for msg in conv.messages:
        if not msg.text:
            continue
        writer.writerow(
            [
                str(conv.id),
                str(msg.id),
                msg.role or "",
                msg.timestamp.isoformat() if msg.timestamp else "",
                msg.text,
            ]
        )
    return buf.getvalue().rstrip()


def _conv_to_markdown(conv: Conversation) -> str:
    """Convert conversation to markdown."""
    from polylogue.rendering.core_markdown import format_conversation_markdown

    return format_conversation_markdown(conv)


def _conv_to_plaintext(conv: Conversation) -> str:
    """Convert conversation to plain text (no markdown formatting).

    Strips all formatting, returning just the raw message content.
    Useful for piping to grep, wc, or other text processing tools.

    Args:
        conv: Conversation to format

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


def _conv_to_obsidian(conv: Conversation) -> str:
    """Convert conversation to Obsidian-compatible markdown with YAML frontmatter."""
    tags_formatted = ", ".join(_yaml_safe(t) for t in conv.tags) if conv.tags else ""
    frontmatter = [
        "---",
        f"id: {_yaml_safe(str(conv.id))}",
        f"provider: {_yaml_safe(conv.provider)}",
        f"date: {conv.display_date.isoformat() if conv.display_date else 'unknown'}",
        f"tags: [{tags_formatted}]",
        "---",
        "",
    ]
    content = _conv_to_markdown(conv)
    return "\n".join(frontmatter) + content


def _conv_to_org(conv: Conversation) -> str:
    """Convert conversation to Org-mode format."""
    lines = [
        f"#+TITLE: {conv.display_title or conv.id}",
        f"#+DATE: {conv.display_date.strftime('%Y-%m-%d') if conv.display_date else 'unknown'}",
        f"#+PROPERTY: provider {conv.provider}",
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
