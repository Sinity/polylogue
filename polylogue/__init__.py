"""Polylogue - AI Conversation Archive Library.

This library provides tools for ingesting, storing, and querying AI conversation
exports from ChatGPT, Claude, Codex, Gemini, and other AI assistants.

Example:
    from polylogue import Polylogue

    # Initialize
    archive = Polylogue(archive_root="~/.polylogue")

    # Ingest files
    result = archive.ingest_file("chatgpt_export.json")
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
from polylogue.core.facade import Polylogue

# Core types for library users
from polylogue.config import Config, Source, ConfigError
from polylogue.lib.repository import ConversationRepository
from polylogue.lib.models import Conversation, Message
from polylogue.storage.search import SearchResult, SearchHit

__all__ = [
    # Main facade
    "Polylogue",
    # Configuration
    "Config",
    "Source",
    "ConfigError",
    # Data access
    "ConversationRepository",
    "Conversation",
    "Message",
    # Search
    "SearchResult",
    "SearchHit",
]

# Note: The following modules remain available as submodules:
# - polylogue.config (configuration)
# - polylogue.storage.db (database)
# - polylogue.pipeline.runner (pipeline runner)
# - polylogue.render (rendering)
# - polylogue.storage.search (search)
# - polylogue.types (type definitions)
