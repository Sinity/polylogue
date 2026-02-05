"""Polylogue library API.

This module provides the public API for consumers of polylogue as a library.
The key abstractions are:

- `Conversation`: A conversation with semantic projection methods
- `Message`: A single message with classification properties
- `DialoguePair`: A user message paired with its assistant response
- `Attachment`: File attachments associated with messages
- `ConversationRepository`: Query and retrieve conversations

Example usage:

    from polylogue.lib import ConversationRepository, Conversation
    from polylogue.storage.backends.sqlite import create_default_backend

    # Create repository with backend
    repo = ConversationRepository(backend=create_default_backend())

    # Get a conversation with projection support
    conv = repo.get("claude:abc123")
    if conv:
        # Use semantic projections
        for pair in conv.iter_pairs():
            print(f"User: {pair.user.text[:50]}...")
            print(f"Assistant: {pair.assistant.text[:50]}...")

        # Filter to substantive dialogue only
        clean = conv.substantive_only()
        print(clean.to_text())

        # Get statistics
        print(f"Messages: {conv.message_count}")
        print(f"Words: {conv.word_count}")
"""

from polylogue.lib.filters import ConversationFilter
from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Attachment, Conversation, DialoguePair, Message, Role
from polylogue.lib.projections import ConversationProjection


def __getattr__(name: str) -> object:
    if name == "ConversationRepository":
        from polylogue.storage.repository import ConversationRepository

        return ConversationRepository
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Attachment",
    "Conversation",
    "ConversationFilter",
    "DialoguePair",
    "Message",
    "MessageCollection",
    "Role",
]
