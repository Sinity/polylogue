"""Polylogue - AI Conversation Archive Library.

This library provides tools for parsing, storing, and querying AI conversation
exports from ChatGPT, Claude, Codex, Gemini, and other AI assistants.

Example:
    from polylogue import Polylogue

    # Initialize
    archive = Polylogue(archive_root="~/.polylogue")

    # Parse files
    result = archive.parse_file("chatgpt_export.json")
    print(f"Imported {result.counts['conversations']} conversations")

    # Query conversations
    conv = archive.get_conversation("claude:abc")
    if conv:
        for pair in conv.substantive_only().iter_pairs():
            print(f"Q: {pair.user.text[:50]}")
            print(f"A: {pair.assistant.text[:50]}")

    # Search
    results = archive.search("python error handling")
    for hit in results.hits:
        print(f"{hit.title}: {hit.snippet}")
"""

# High-level API
from polylogue.errors import PolylogueError
from polylogue.facade import ArchiveStats, Polylogue
from polylogue.lib.filters import ConversationFilter
from polylogue.lib.models import Conversation, Message
from polylogue.storage.search import SearchResult


def __getattr__(name: str) -> object:
    if name == "ConversationRepository":
        from polylogue.storage.repository import ConversationRepository

        return ConversationRepository
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ArchiveStats",
    "Conversation",
    "ConversationFilter",
    "Message",
    "Polylogue",
    "PolylogueError",
    "SearchResult",
]
