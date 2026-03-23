"""Streaming output helpers for query execution."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv
    from polylogue.lib.models import Message
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.store import MessageRecord


def render_stream_message(message: Message | MessageRecord, output_format: str) -> str:
    """Render a single streamed message chunk."""
    if output_format == "plaintext":
        if not message.text:
            return ""
        role_label = (message.role or "unknown").upper().replace("[", "").replace("]", "")
        return f"[{role_label}]\n{message.text}\n\n"

    if output_format == "markdown":
        if not message.text:
            return ""
        role_label = (message.role or "unknown").capitalize()
        return f"## {role_label}\n\n{message.text}\n\n"

    if output_format == "json-lines":
        record = {
            "type": "message",
            "id": getattr(message, "id", getattr(message, "message_id", None)),
            "role": message.role,
            "timestamp": message.timestamp.isoformat() if getattr(message, "timestamp", None) else None,
            "text": message.text,
            "word_count": message.word_count,
        }
        return json.dumps(record, ensure_ascii=False) + "\n"

    return ""


def render_stream_header(
    *,
    conversation_id: str,
    title: str | None,
    provider: str | None,
    display_date: object | None,
    output_format: str,
    dialogue_only: bool,
    message_limit: int | None,
    stats: dict[str, Any] | None,
) -> str:
    """Render any stream prelude/header for the selected output format."""
    if hasattr(display_date, "strftime"):
        display_date_text = display_date.strftime("%Y-%m-%d %H:%M")
        display_date_value = display_date.isoformat() if hasattr(display_date, "isoformat") else str(display_date)
    elif display_date:
        display_date_text = str(display_date)
        display_date_value = str(display_date)
    else:
        display_date_text = None
        display_date_value = None

    if output_format == "markdown":
        lines = [f"# {title or conversation_id[:24]}", ""]
        if display_date_text is not None:
            lines.append(f"**Date**: {display_date_text}")
        if provider:
            lines.append(f"**Provider**: {provider}")
        if display_date_text is not None or provider:
            lines.append("")
        if dialogue_only and stats:
            line = f"_Showing {stats['dialogue_messages']} dialogue messages"
            if message_limit:
                line += f" (limit: {message_limit})"
            line += f" of {stats['total_messages']} total_"
            lines.extend([line, ""])
        return "\n".join(lines)

    if output_format == "json-lines":
        header = {
            "type": "header",
            "conversation_id": conversation_id,
            "title": title,
            "provider": provider,
            "date": display_date_value,
            "dialogue_only": dialogue_only,
            "message_limit": message_limit,
            "stats": stats,
        }
        return json.dumps(header) + "\n"

    return ""


def render_stream_footer(*, output_format: str, emitted_messages: int) -> str:
    """Render any stream closing/footer fragment."""
    if output_format == "markdown":
        return f"\n---\n_Streamed {emitted_messages} messages_\n"
    if output_format == "json-lines":
        return json.dumps({"type": "footer", "message_count": emitted_messages}) + "\n"
    return ""


def render_stream_transcript(
    *,
    conversation_id: str,
    title: str | None,
    provider: str | None,
    display_date: object | None,
    messages: list[Message],
    output_format: str,
    dialogue_only: bool = False,
    message_limit: int | None = None,
    stats: dict[str, Any] | None = None,
) -> tuple[str, int]:
    """Render the full stream transcript deterministically for proof/tests."""
    parts = [
        render_stream_header(
            conversation_id=conversation_id,
            title=title,
            provider=provider,
            display_date=display_date,
            output_format=output_format,
            dialogue_only=dialogue_only,
            message_limit=message_limit,
            stats=stats,
        )
    ]
    emitted = 0
    for message in messages[: message_limit if message_limit is not None else None]:
        chunk = render_stream_message(message, output_format)
        if chunk:
            parts.append(chunk)
            emitted += 1
    parts.append(render_stream_footer(output_format=output_format, emitted_messages=emitted))
    return "".join(parts), emitted


async def stream_conversation(
    env: AppEnv,
    repo: ConversationRepository,
    conversation_id: str,
    *,
    output_format: str = "plaintext",
    dialogue_only: bool = False,
    message_limit: int | None = None,
) -> int:
    """Stream conversation messages to stdout without buffering."""
    conv_record = await repo.queries.get_conversation(conversation_id)
    if not conv_record:
        click.echo(f"Conversation not found: {conversation_id}", err=True)
        raise SystemExit(1)

    stats = await repo.queries.get_conversation_stats(conversation_id)
    sys.stdout.write(
        render_stream_header(
            conversation_id=conversation_id,
            title=conv_record.title,
            provider=getattr(conv_record, "provider_name", None),
            display_date=(getattr(conv_record, "updated_at", None) or getattr(conv_record, "created_at", None)),
            output_format=output_format,
            dialogue_only=dialogue_only,
            message_limit=message_limit,
            stats=stats,
        )
    )
    sys.stdout.flush()

    count = 0
    async for message in repo.iter_messages(
        conversation_id,
        dialogue_only=dialogue_only,
        limit=message_limit,
    ):
        chunk = render_stream_message(message, output_format)
        if chunk:
            sys.stdout.write(chunk)
            count += 1
        sys.stdout.flush()

    sys.stdout.write(render_stream_footer(output_format=output_format, emitted_messages=count))
    sys.stdout.flush()

    return count


def write_message_streaming(message: Message | MessageRecord, output_format: str) -> None:
    """Write a single streamed message to stdout."""
    chunk = render_stream_message(message, output_format)
    if chunk:
        sys.stdout.write(chunk)
        sys.stdout.flush()


__all__ = [
    "render_stream_footer",
    "render_stream_header",
    "render_stream_message",
    "render_stream_transcript",
    "stream_conversation",
    "write_message_streaming",
]
