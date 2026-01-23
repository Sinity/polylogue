"""HTML renderer implementation using Jinja2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import DictLoader, Environment, FileSystemLoader
from markdown_it import MarkdownIt

from polylogue.assets import asset_path
from polylogue.render_paths import render_root
from polylogue.storage.db import open_connection


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


class HTMLRenderer:
    """Renders conversations to HTML format using Jinja2 templates."""

    def __init__(self, archive_root: Path, template_path: Path | None = None):
        """Initialize the HTML renderer.

        Args:
            archive_root: Root directory for archived conversations
            template_path: Optional path to custom Jinja2 HTML template
        """
        self.archive_root = archive_root
        self.template_path = template_path

    def supports_format(self) -> str:
        """Return the output format this renderer supports.

        Returns:
            'html'
        """
        return "html"

    def _render_html(self, markdown_text: str, title: str) -> str:
        """Convert markdown to HTML using Jinja2 template.

        Args:
            markdown_text: Markdown content to convert
            title: Page title

        Returns:
            Rendered HTML string
        """
        md = MarkdownIt("commonmark", {"html": False, "linkify": True}).enable("table")
        body_html = md.render(markdown_text)

        if self.template_path and self.template_path.exists():
            loader = FileSystemLoader(self.template_path.parent)
            template_name = self.template_path.name
        else:
            loader = DictLoader({"index.html": DEFAULT_HTML_TEMPLATE})
            template_name = "index.html"

        env = Environment(loader=loader, autoescape=True)
        template = env.get_template(template_name)
        return template.render(title=title, body=body_html)

    def render(self, conversation_id: str, output_path: Path) -> Path:
        """Render a conversation to HTML format.

        Args:
            conversation_id: ID of the conversation to render
            output_path: Directory where the HTML file should be written

        Returns:
            Path to the generated HTML file

        Raises:
            ValueError: If conversation not found
            IOError: If output path is invalid or write fails
        """
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

        def _append_attachment(att, lines: list[str]) -> None:
            name = None
            meta = att["provider_meta"]
            if meta:
                try:
                    meta_dict = json.loads(meta)
                    name = meta_dict.get("name") or meta_dict.get("provider_id") or meta_dict.get("drive_id")
                except json.JSONDecodeError:
                    name = None
            label = name or att["attachment_id"]
            path_value = att["path"] or str(asset_path(self.archive_root, att["attachment_id"]))
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
                _append_attachment(att, lines)
            lines.append("")

        orphan_keys = [key for key in attachments_by_message if key not in message_ids]
        if orphan_keys:
            lines.append("## attachments")
            lines.append("")
            for key in sorted(orphan_keys, key=lambda item: "" if item is None else str(item)):
                for att in attachments_by_message.get(key, []):
                    _append_attachment(att, lines)
            lines.append("")

        markdown_text = "\n".join(lines).strip() + "\n"

        # Determine output path
        render_root_path = render_root(output_path, provider, conversation_id)
        render_root_path.mkdir(parents=True, exist_ok=True)

        # Save markdown file (for backward compatibility)
        md_path = render_root_path / "conversation.md"
        md_path.write_text(markdown_text, encoding="utf-8")

        # Render HTML
        html_path = render_root_path / "conversation.html"
        html_content = self._render_html(markdown_text, title)
        html_path.write_text(html_content, encoding="utf-8")

        return html_path


__all__ = ["HTMLRenderer"]
