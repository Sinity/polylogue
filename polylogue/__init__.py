"""Polylogue - AI Conversation Archive Library.

This library provides tools for parsing, storing, and querying AI conversation
exports from ChatGPT, Claude, Codex, Gemini, and other AI assistants.

Example::

    from polylogue import Polylogue

    async with Polylogue() as archive:
        # Statistics
        stats = await archive.stats()
        print(f"{stats.conversation_count} conversations")

        # Query conversations
        convs = await archive.filter().provider("claude").since("2024-01-01").list()
        for conv in convs:
            print(f"{conv.display_title}: {conv.message_count} messages")

        # Search
        results = await archive.search("python error handling")
        for hit in results.hits:
            print(hit.title)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.errors import PolylogueError
    from polylogue.facade import ArchiveStats, Polylogue
    from polylogue.lib.filters import ConversationFilter
    from polylogue.lib.models import Conversation, Message
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.search import SearchResult


def __getattr__(name: str) -> object:
    if name == "ArchiveStats":
        from polylogue.facade import ArchiveStats

        return ArchiveStats
    if name == "Conversation":
        from polylogue.lib.models import Conversation

        return Conversation
    if name == "ConversationFilter":
        from polylogue.lib.filters import ConversationFilter

        return ConversationFilter
    if name == "ConversationRepository":
        from polylogue.storage.repository import ConversationRepository

        return ConversationRepository
    if name == "Message":
        from polylogue.lib.models import Message

        return Message
    if name == "Polylogue":
        from polylogue.facade import Polylogue

        return Polylogue
    if name == "PolylogueError":
        from polylogue.errors import PolylogueError

        return PolylogueError
    if name == "SearchResult":
        from polylogue.storage.search import SearchResult

        return SearchResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ArchiveStats",
    "Conversation",
    "ConversationFilter",
    "ConversationRepository",
    "Message",
    "Polylogue",
    "PolylogueError",
    "SearchResult",
]
