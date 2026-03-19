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
    from polylogue.storage.backends.connection import create_default_backend

    # Create repository with backend
    repo = ConversationRepository(backend=create_default_backend())

    # Get a conversation with projection support
    conv = await repo.get("claude-ai:abc123")
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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.lib.branch_type import BranchType
    from polylogue.lib.filters import ConversationFilter
    from polylogue.lib.messages import MessageCollection
    from polylogue.lib.models import Attachment, Conversation, DialoguePair, Message
    from polylogue.lib.projections import ConversationProjection
    from polylogue.lib.roles import Role
    from polylogue.lib.stats import ArchiveStats
    from polylogue.storage.repository import ConversationRepository


def __getattr__(name: str) -> object:
    lazy_exports = {
        "ArchiveStats": ("polylogue.lib.stats", "ArchiveStats"),
        "Attachment": ("polylogue.lib.models", "Attachment"),
        "BranchType": ("polylogue.lib.branch_type", "BranchType"),
        "Conversation": ("polylogue.lib.models", "Conversation"),
        "ConversationFilter": ("polylogue.lib.filters", "ConversationFilter"),
        "ConversationProjection": ("polylogue.lib.projections", "ConversationProjection"),
        "ConversationRepository": ("polylogue.storage.repository", "ConversationRepository"),
        "DialoguePair": ("polylogue.lib.models", "DialoguePair"),
        "Message": ("polylogue.lib.models", "Message"),
        "MessageCollection": ("polylogue.lib.messages", "MessageCollection"),
        "Role": ("polylogue.lib.roles", "Role"),
    }
    module_spec = lazy_exports.get(name)
    if module_spec is not None:
        module_name, attr_name = module_spec
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ArchiveStats",
    "Attachment",
    "BranchType",
    "Conversation",
    "ConversationFilter",
    "ConversationProjection",
    "ConversationRepository",
    "DialoguePair",
    "Message",
    "MessageCollection",
    "Role",
]
