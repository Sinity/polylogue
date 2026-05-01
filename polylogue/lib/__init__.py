"""Domain-model and query helper API for Polylogue.

``polylogue.lib`` intentionally exposes conversation/message domain types and
query/projection helpers. Archive runtime services such as the facade, sync
bridge, repository, and archive statistics live on more precise modules.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.archive.attachment.models import Attachment
    from polylogue.lib.conversation.branch_type import BranchType
    from polylogue.lib.conversation.models import Conversation
    from polylogue.lib.message.messages import MessageCollection
    from polylogue.lib.message.models import DialoguePair, Message
    from polylogue.lib.projection.projections import ConversationProjection
    from polylogue.lib.roles import Role
    from polylogue.lib.semantic.content_projection import ContentKind, ContentProjectionSpec


def __getattr__(name: str) -> object:
    lazy_exports = {
        "Attachment": ("polylogue.archive.attachment.models", "Attachment"),
        "BranchType": ("polylogue.lib.conversation.branch_type", "BranchType"),
        "ContentKind": ("polylogue.lib.semantic.content_projection", "ContentKind"),
        "ContentProjectionSpec": ("polylogue.lib.semantic.content_projection", "ContentProjectionSpec"),
        "Conversation": ("polylogue.lib.conversation.models", "Conversation"),
        "ConversationProjection": ("polylogue.lib.projection.projections", "ConversationProjection"),
        "DialoguePair": ("polylogue.lib.message.models", "DialoguePair"),
        "Message": ("polylogue.lib.message.models", "Message"),
        "MessageCollection": ("polylogue.lib.message.messages", "MessageCollection"),
        "Role": ("polylogue.lib.roles", "Role"),
    }
    module_spec = lazy_exports.get(name)
    if module_spec is not None:
        module_name, attr_name = module_spec
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Attachment",
    "BranchType",
    "ContentKind",
    "ContentProjectionSpec",
    "Conversation",
    "ConversationProjection",
    "DialoguePair",
    "Message",
    "MessageCollection",
    "Role",
]
