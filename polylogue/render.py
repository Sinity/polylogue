from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from markdown_it import MarkdownIt

from .assets import asset_path
from .db import open_connection


@dataclass
class RenderResult:
    conversation_id: str
    markdown_path: Path
    html_path: Path


def _render_html(markdown_text: str, *, title: str) -> str:
    md = MarkdownIt("commonmark", {"html": True, "linkify": True})
    body = md.render(markdown_text)
    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\" />\n"
        f"  <title>{title}</title>\n"
        "  <style>body{font-family:system-ui,Segoe UI,Roboto,sans-serif;max-width:960px;margin:2rem auto;line-height:1.6;padding:0 1rem;}pre{white-space:pre-wrap;}code{font-family:ui-monospace,Menlo,monospace;}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


def render_conversation(
    *,
    conversation_id: str,
    archive_root: Path,
) -> RenderResult:
    with open_connection(None) as conn:
        convo = conn.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if not convo:
            raise RuntimeError(f"Conversation not found: {conversation_id}")
        messages = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp, message_id",
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

    attachments_by_message = {}
    for att in attachments:
        attachments_by_message.setdefault(att["message_id"], []).append(att)

    title = convo["title"] or conversation_id
    provider = convo["provider_name"]

    lines = [f"# {title}", "", f"Provider: {provider}", f"Conversation ID: {conversation_id}", ""]
    for msg in messages:
        role = msg["role"] or "message"
        text = msg["text"] or ""
        timestamp = msg["timestamp"]
        lines.append(f"## {role}")
        if timestamp:
            lines.append(f"_Timestamp: {timestamp}_")
        lines.append("")
        lines.append(text)
        lines.append("")
        for att in attachments_by_message.get(msg["message_id"], []):
            name = None
            meta = att["provider_meta"]
            if meta:
                try:
                    meta_dict = json.loads(meta)
                    name = meta_dict.get("name") or meta_dict.get("provider_id") or meta_dict.get("drive_id")
                except Exception:
                    name = None
            label = name or att["attachment_id"]
            path_value = att["path"] or str(asset_path(archive_root, att["attachment_id"]))
            lines.append(f"- Attachment: {label} ({path_value})")
        lines.append("")

    markdown_text = "\n".join(lines).strip() + "\n"

    render_root = archive_root / "render" / provider / conversation_id
    render_root.mkdir(parents=True, exist_ok=True)
    md_path = render_root / "conversation.md"
    md_path.write_text(markdown_text, encoding="utf-8")

    html_path = render_root / "conversation.html"
    html_path.write_text(_render_html(markdown_text, title=title), encoding="utf-8")

    return RenderResult(conversation_id=conversation_id, markdown_path=md_path, html_path=html_path)


__all__ = ["RenderResult", "render_conversation"]
