"""Conversation-page streaming helpers for static-site generation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, Template

from polylogue.logging import get_logger
from polylogue.rendering.block_models import coerce_renderable_blocks
from polylogue.rendering.core import build_rendered_message
from polylogue.rendering.core_messages import RenderedMessage
from polylogue.site.models import ConversationIndex

if TYPE_CHECKING:
    from polylogue.storage.repository import ConversationRepository

logger = get_logger(__name__)


async def write_template_stream(
    template: Template,
    output_path: Path,
    **context: object,
) -> None:
    """Render a template to disk without materializing the full output string."""
    stream = template.generate_async(**context)
    with output_path.open("w", encoding="utf-8") as handle:
        async for chunk in stream:
            handle.write(chunk)


async def iter_conversation_page_messages(
    repository: ConversationRepository,
    conversation_id: str,
    *,
    render_html: Callable[[str], str],
) -> AsyncIterator[RenderedMessage]:
    """Yield site message payloads lazily for a conversation page."""
    async for msg in repository.iter_messages(conversation_id):
        if not msg.text and not msg.content_blocks:
            continue
        yield build_rendered_message(
            message_id=msg.id,
            role=msg.role or "unknown",
            text=msg.text or "",
            timestamp=msg.timestamp,
            content_blocks=coerce_renderable_blocks(msg.content_blocks),
            parent_message_id=msg.parent_id,
            branch_index=msg.branch_index,
            render_html=render_html,
        )


async def generate_conversation_page(
    *,
    output_dir: Path,
    page_env: Environment,
    repository: ConversationRepository,
    conversation: ConversationIndex,
    render_html: Callable[[str], str],
    incremental: bool = True,
) -> str:
    """Generate one conversation page, or keep an up-to-date existing one."""
    template = page_env.get_template("conversation.html")
    page_path = output_dir / conversation.path
    page_path.parent.mkdir(parents=True, exist_ok=True)

    if incremental and page_path.exists():
        if conversation.updated_at:
            try:
                file_mtime = datetime.fromtimestamp(page_path.stat().st_mtime)
                conv_updated = datetime.fromisoformat(conversation.updated_at)
                if file_mtime > conv_updated:
                    return "reused"
            except (ValueError, OSError):
                pass
        else:
            return "reused"

    try:
        await write_template_stream(
            template,
            page_path,
            title=conversation.title,
            provider=conversation.provider,
            message_count=conversation.message_count,
            updated_at=conversation.updated_at,
            messages=iter_conversation_page_messages(
                repository,
                conversation.id,
                render_html=render_html,
            ),
        )
        return "rendered"
    except Exception as exc:
        page_path.unlink(missing_ok=True)
        logger.warning(
            "Skipping conversation page %s: %s",
            conversation.id,
            exc,
        )
        return "failed"
