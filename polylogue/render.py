from __future__ import annotations

from dataclasses import dataclass
import html
import json
from pathlib import Path

from markdown_it import MarkdownIt

from .assets import asset_path
from .db import open_connection
from .render_paths import render_root


@dataclass
class RenderResult:
    conversation_id: str
    markdown_path: Path
    html_path: Path


def _render_html(markdown_text: str, *, title: str) -> str:
    md = MarkdownIt("commonmark", {"html": False, "linkify": True})
    body = md.render(markdown_text)
    safe_title = html.escape(title, quote=True)
    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\" />\n"
        f"  <title>{safe_title}</title>\n"
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
    render_root_path: Path | None = None,
) -> RenderResult:
    with open_connection(None) as conn:
        convo = conn.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if not convo:
            raise RuntimeError(f"Conversation not found: {conversation_id}")
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

    attachments_by_message = {}
    for att in attachments:
        attachments_by_message.setdefault(att["message_id"], []).append(att)

    def _append_attachment(att) -> None:
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

    title = convo["title"] or conversation_id
    provider = convo["provider_name"]

    lines = [f"# {title}", "", f"Provider: {provider}", f"Conversation ID: {conversation_id}", ""]
    message_ids = set()
    for msg in messages:
        message_ids.add(msg["message_id"])
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
            _append_attachment(att)
        lines.append("")

    orphan_keys = [key for key in attachments_by_message.keys() if key not in message_ids]
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
    html_path.write_text(_render_html(markdown_text, title=title), encoding="utf-8")

    return RenderResult(conversation_id=conversation_id, markdown_path=md_path, html_path=html_path)


__all__ = ["RenderResult", "render_conversation"]
