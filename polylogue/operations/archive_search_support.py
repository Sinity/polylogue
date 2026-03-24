"""Archive-level conversation retrieval and search support."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.lib.query_spec import ConversationQuerySpec
from polylogue.paths import conversation_render_root
from polylogue.storage.search import SearchHit, SearchResult

if TYPE_CHECKING:
    from polylogue.lib.conversation_models import Conversation


def _build_search_snippet(text: str, query: str) -> str:
    """Create a deterministic snippet around the earliest query-term match."""
    if not text:
        return ""

    terms = [term.lower() for term in query.split() if term.strip()]
    lowered = text.lower()
    positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
    anchor = min(positions) if positions else 0
    start = max(0, anchor - 60)
    end = min(len(text), anchor + 140)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = f"...{snippet}"
    if end < len(text):
        snippet = f"{snippet}..."
    return snippet


def _conversation_search_hit(
    conversation: Conversation,
    *,
    query: str,
    render_root_path: Path,
) -> SearchHit:
    """Adapt a canonical conversation result into the SearchResult surface."""
    terms = [term.lower() for term in query.split() if term.strip()]
    matching_message = next(
        (
            msg
            for msg in conversation.messages
            if msg.text and any(term in msg.text.lower() for term in terms)
        ),
        next((msg for msg in conversation.messages if msg.text), None),
    )
    message_id = str(matching_message.id) if matching_message else ""
    timestamp = matching_message.timestamp.isoformat() if matching_message and matching_message.timestamp else None
    snippet = _build_search_snippet(matching_message.text or "", query) if matching_message else ""
    conversation_path = (
        conversation_render_root(render_root_path, conversation.provider, str(conversation.id)) / "conversation.md"
    )
    return SearchHit(
        conversation_id=str(conversation.id),
        provider_name=conversation.provider,
        source_name=None,
        message_id=message_id,
        title=conversation.display_title,
        timestamp=timestamp,
        snippet=snippet,
        conversation_path=conversation_path,
    )


class ArchiveSearchMixin:
    """Conversation retrieval and search methods for archive operations."""

    async def get_conversation(self, conversation_id: str):
        return await self.repository.view(conversation_id)

    async def get_conversations(self, conversation_ids: list[str]):
        return await self.repository.get_many(conversation_ids)

    async def list_conversations(
        self,
        *,
        provider: str | None = None,
        limit: int | None = None,
    ):
        return await self.repository.list(provider=provider, limit=limit)

    async def query_conversations(self, spec: ConversationQuerySpec):
        return await spec.list(self.repository)

    async def search(
        self,
        query: str,
        *,
        limit: int = 100,
        source: str | None = None,
        since: str | None = None,
    ) -> SearchResult:
        spec = ConversationQuerySpec(
            query_terms=(query,),
            providers=(source,) if source else (),
            since=since,
            limit=limit,
        )
        conversations = await self.query_conversations(spec)
        return SearchResult(
            hits=[
                _conversation_search_hit(
                    conversation,
                    query=query,
                    render_root_path=self.config.render_root,
                )
                for conversation in conversations
            ]
        )


__all__ = ["ArchiveSearchMixin"]
