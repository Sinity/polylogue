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
from collections.abc import Callable, Iterable, Iterator
from contextlib import closing
from typing import TYPE_CHECKING, TextIO

from polylogue.archive.semantic.content_projection import ContentProjectionSpec
from polylogue.core.localtime import format_local_datetime
from polylogue.surfaces.payloads import (
    JSONDocument,
    message_render_envelope_from_domain,
    model_json_document,
    session_list_envelope_from_domain,
)

if TYPE_CHECKING:
    from polylogue.archive.models import Message, Session

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
    output = io.StringIO()
    write_session(
        conv, lambda: iter(conv.messages), output, output_format, fields, content_projection=content_projection
    )
    return output.getvalue()


def write_session(
    header: Session,
    messages: Callable[[], Iterable[Message]],
    output: TextIO,
    output_format: str,
    fields: str | None,
    *,
    content_projection: ContentProjectionSpec | None = None,
    compact_json: bool = False,
    tags: Callable[[], Iterable[str]] | None = None,
) -> None:
    """Render the complete owned transcript without constructing a whole Session."""
    with closing(
        session_chunks(
            header,
            messages,
            output_format,
            fields,
            content_projection=content_projection,
            compact_json=compact_json,
            tags=tags,
        )
    ) as chunks:
        for chunk in chunks:
            output.write(chunk)


def session_chunks(
    header: Session,
    messages: Callable[[], Iterable[Message]],
    output_format: str,
    fields: str | None,
    *,
    content_projection: ContentProjectionSpec | None = None,
    compact_json: bool = False,
    tags: Callable[[], Iterable[str]] | None = None,
) -> Iterator[str]:
    from polylogue.archive.semantic.content_projection import iter_projected_content
    from polylogue.rendering.transcript import message_document_chunks, stripped_chunks

    def selected_messages() -> Iterable[Message]:
        return iter_projected_content(messages, content_projection)

    output_format = normalize_session_output_format(output_format)
    if output_format in {"json", "yaml"}:
        selected = {field.strip() for field in fields.split(",")} if fields else None
        metadata = session_list_envelope_from_domain(header, bound_title=False, messages=selected_messages).selected(
            selected
        )
        projected = (
            (
                model_json_document(
                    message_render_envelope_from_domain(message, session_id=header.id), exclude_none=True
                )
                for message in selected_messages()
            )
            if selected is None or "messages" in selected
            else None
        )
        arrays = {"tags": tags()} if tags is not None and "tags" in metadata else None
        if arrays is not None:
            metadata["tags"] = []
        yield from message_document_chunks(metadata, projected, output_format, compact=compact_json, arrays=arrays)
    elif output_format == "html":
        from polylogue.rendering.renderers.html import session_html_chunks

        yield from session_html_chunks(header, selected_messages())
    elif output_format == "csv":

        def csv_chunks() -> Iterator[str]:
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            writer.writerow(["session_id", "message_id", "role", "timestamp", "text"])
            yield buffer.getvalue()
            for message in selected_messages():
                if not message.text:
                    continue
                buffer.seek(0)
                buffer.truncate()
                writer.writerow(
                    [
                        str(header.id),
                        str(message.id),
                        _csv_safe(message.role or ""),
                        _csv_safe(message.timestamp.isoformat() if message.timestamp else ""),
                        _csv_safe(message.text),
                    ]
                )
                yield buffer.getvalue()

        yield from stripped_chunks(csv_chunks(), leading=False, characters="\r\n")
    elif output_format == "plaintext":
        yield from stripped_chunks(message.text + "\n\n" for message in selected_messages() if message.text)
    elif output_format == "org":
        yield f"#+TITLE: {header.display_title or header.id}\n#+DATE: {format_local_datetime(header.display_date, '%Y-%m-%d') or 'unknown'}\n#+PROPERTY: origin {header.origin.value}\n"
        for message in selected_messages():
            if message.text:
                yield f"\n* {(message.role or 'unknown').upper()}\n{message.text}\n"
    else:
        if output_format == "obsidian":
            yield f"---\nid: {_yaml_safe(str(header.id))}\norigin: {_yaml_safe(header.origin.value)}\ndate: {header.display_date.isoformat() if header.display_date else 'unknown'}\ntags: ["
            for position, tag in enumerate(tags() if tags is not None else header.tags):
                if position:
                    yield ", "
                yield _yaml_safe(tag)
            yield "]\n---\n"
        yield from stripped_chunks(_markdown_chunks(header, selected_messages()))
        yield "\n"


def _markdown_chunks(header: Session, messages: Iterable[Message]) -> Iterator[str]:
    from pathlib import Path

    from polylogue.rendering.block_models import coerce_renderable_blocks
    from polylogue.rendering.core_markdown import (
        MarkdownAttachment,
        _normalize_markdown_message,
        markdown_message_lines,
    )

    yield f"# {header.title or 'Untitled'}\n\nOrigin: {header.origin.value}\nSession ID: {header.id}\n\n"
    for message in messages:
        normalized = _normalize_markdown_message(
            message_id=str(message.id),
            role=message.role,
            text=message.text,
            timestamp=message.timestamp,
            default_role="message",
            blocks=coerce_renderable_blocks(message.blocks),
        )
        attachments = [
            MarkdownAttachment(str(att.id), att.path or att.source_url, att.name) for att in message.attachments
        ]
        lines = markdown_message_lines(normalized, attachments, Path("."))
        if lines:
            yield "\n".join(lines) + "\n"
    if header.attachments:
        from polylogue.rendering.core_markdown import append_attachment_markdown

        yield "## attachments\n\n"
        for att in header.attachments:
            lines = []
            append_attachment_markdown(
                MarkdownAttachment(str(att.id), att.path or att.source_url, att.name), lines, Path(".")
            )
            yield "\n".join(lines) + "\n"


def _conv_to_dict(conv: Session, fields: str | None, *, bound_title: bool = True) -> JSONDocument:
    """Convert session to summary dict (message count, not content).

    Used for list-mode output where loading all message text is unnecessary.
    For full-content output, use _conv_to_json() instead -- which calls this
    with ``bound_title=False`` since a full single-session read must stay
    lossless (polylogue-x7d PR #3420 follow-up).
    """
    selected = {field.strip() for field in fields.split(",")} if fields else None
    return session_list_envelope_from_domain(conv, bound_title=bound_title).selected(selected)


def _conv_to_html(conv: Session) -> str:
    from polylogue.rendering.renderers.html import render_session_html

    return render_session_html(conv)


def _csv_safe(value: str) -> str:
    """Prevent CSV formula injection by prefixing dangerous leading characters."""
    if value and value[0] in ("=", "+", "-", "@"):
        return "'" + value
    return value


def _yaml_safe(value: str) -> str:
    """Quote a YAML value if it contains special characters."""
    if any(c in value for c in ":#{}[]|>&*!?@`'\",\n\t"):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
        return f'"{escaped}"'
    return value
