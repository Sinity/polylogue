from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import DictLoader, Environment, FileSystemLoader
from markdown_it import MarkdownIt

from .assets import asset_path
from .storage.db import open_connection
from .render_paths import render_root


@dataclass
class RenderResult:
    conversation_id: str
    markdown_path: Path
    html_path: Path


DEFAULT_HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{{ title }}</title>
  <style>
    body {
      font-family: system-ui, Segoe UI, Roboto, sans-serif;
      max-width: 960px;
      margin: 2rem auto;
      line-height: 1.6;
      padding: 0 1rem;
      background-color: #f9fafb;
      color: #111827;
    }
    pre {
      white-space: pre-wrap;
      background: #f3f4f6;
      padding: 1rem;
      border-radius: 0.5rem;
      border: 1px solid #e5e7eb;
    }
    code {
      font-family: ui-monospace, Menlo, monospace;
    }
    h1 { border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem; }
    h2 { margin-top: 2rem; color: #374151; }
    .metadata { color: #6b7280; font-size: 0.875rem; margin-bottom: 2rem; }
    .attachment { font-size: 0.875rem; color: #2563eb; }
  </style>
</head>
<body>
  {{ body|safe }}
</body>
</html>
"""


def _render_html(markdown_text: str, *, title: str, template_path: Path | None = None) -> str:
    md = MarkdownIt("commonmark", {"html": False, "linkify": True}).enable("table")
    body_html = md.render(markdown_text)

    loader: FileSystemLoader | DictLoader
    if template_path and template_path.exists():
        loader = FileSystemLoader(template_path.parent)
        template_name = template_path.name
    else:
        loader = DictLoader({"index.html": DEFAULT_HTML_TEMPLATE})
        template_name = "index.html"

    env = Environment(loader=loader, autoescape=True)
    template = env.get_template(template_name)
    return template.render(title=title, body=body_html)


def render_conversation(
    *,
    conversation_id: str,
    archive_root: Path,
    render_root_path: Path | None = None,
    template_path: Path | None = None,
) -> RenderResult:
    with open_connection(None) as conn:
        convo = conn.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if not convo:
            raise ValueError(f"Conversation not found: {conversation_id}")
        messages = conn.execute(
            """
            SELECT * FROM messages
            WHERE conversation_id = ?
            ORDER BY
                (timestamp IS NULL),
                CASE
                    WHEN timestamp IS NULL THEN NULL
                    WHEN timestamp GLOB '*[^0-9.]*' THEN CAST(strftime('%s', timestamp) AS INTEGER)
                    ELSE CAST(timestamp AS REAL)
                END,
                message_id
            """,
            (conversation_id,),
        ).fetchall()
        attachments = conn.execute(
            """
            SELECT
                attachment_refs.message_id,
                attachments.attachment_id,
                attachments.mime_type,
                attachments.size_bytes,
                attachments.path,
                attachments.provider_meta
            FROM attachment_refs
            JOIN attachments ON attachments.attachment_id = attachment_refs.attachment_id
            WHERE attachment_refs.conversation_id = ?
            """,
            (conversation_id,),
        ).fetchall()

    attachments_by_message: dict[str, list[Any]] = {}
    for att in attachments:
        attachments_by_message.setdefault(att["message_id"], []).append(att)

    def _append_attachment(att: dict[str, Any]) -> None:
        name = None
        meta = att["provider_meta"]
        if meta:
            try:
                meta_dict = json.loads(meta)
                name = meta_dict.get("name") or meta_dict.get("provider_id") or meta_dict.get("drive_id")
            except json.JSONDecodeError:
                name = None
        label = name or att["attachment_id"]
        path_value = att["path"] or str(asset_path(archive_root, att["attachment_id"]))
        lines.append(f"- Attachment: {label} ({path_value})")

    def _format_text(text: str) -> str:
        if not text:
            return ""
        # Handle JSON (tool use/result) by wrapping in code blocks
        if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
            try:
                parsed = json.loads(text)
                return f"```json\n{json.dumps(parsed, indent=2)}\n```"
            except json.JSONDecodeError:
                pass
        return text

    title = convo["title"] or conversation_id
    provider = convo["provider_name"]

    lines = [f"# {title}", "", f"Provider: {provider}", f"Conversation ID: {conversation_id}", ""]
    message_ids = set()
    for msg in messages:
        message_ids.add(msg["message_id"])
        role = msg["role"] or "message"
        text = msg["text"] or ""
        timestamp = msg["timestamp"]
        msg_atts = attachments_by_message.get(msg["message_id"], [])
        
        # Skip empty tool/system/message sections that have no content and no attachments
        if not text.strip() and not msg_atts:
            continue
            
        lines.append(f"## {role}")
        if timestamp:
            lines.append(f"_Timestamp: {timestamp}_")
        lines.append("")
        
        formatted_text = _format_text(text)
        if formatted_text:
            lines.append(formatted_text)
            lines.append("")
            
        for att in msg_atts:
            _append_attachment(att)
        lines.append("")

    orphan_keys = [key for key in attachments_by_message if key not in message_ids]
    if orphan_keys:
        lines.append("## attachments")
        lines.append("")
        for key in sorted(orphan_keys, key=lambda item: "" if item is None else str(item)):
            for att in attachments_by_message.get(key, []):
                _append_attachment(att)
        lines.append("")

    markdown_text = "\n".join(lines).strip() + "\n"

    output_root = render_root_path or (archive_root / "render")
    render_root_path = render_root(output_root, provider, conversation_id)
    render_root_path.mkdir(parents=True, exist_ok=True)
    md_path = render_root_path / "conversation.md"
    md_path.write_text(markdown_text, encoding="utf-8")

    html_path = render_root_path / "conversation.html"
    html_path.write_text(
        _render_html(markdown_text, title=title, template_path=template_path), encoding="utf-8"
    )

    return RenderResult(conversation_id=conversation_id, markdown_path=md_path, html_path=html_path)


__all__ = ["RenderResult", "render_conversation"]
