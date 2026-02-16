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
        for conv in results:
            print(conv.display_title)
"""

# High-level API
from polylogue.errors import PolylogueError
from polylogue.facade import ArchiveStats, Polylogue
from polylogue.lib.filters import ConversationFilter
from polylogue.lib.models import Conversation, Message
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.search import SearchResult

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
