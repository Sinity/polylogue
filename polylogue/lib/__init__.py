"""Domain-model and query helper API for Polylogue.

``polylogue.lib`` intentionally exposes conversation/message domain types and
query/projection helpers. Archive runtime services such as the facade, sync
bridge, repository, and archive statistics live on more precise modules.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.lib.branch_type import BranchType
    from polylogue.lib.filters import ConversationFilter
    from polylogue.lib.messages import MessageCollection
    from polylogue.lib.models import Attachment, Conversation, DialoguePair, Message
    from polylogue.lib.projections import ConversationProjection
    from polylogue.lib.roles import Role


def __getattr__(name: str) -> object:
    lazy_exports = {
        "Attachment": ("polylogue.lib.models", "Attachment"),
        "BranchType": ("polylogue.lib.branch_type", "BranchType"),
        "Conversation": ("polylogue.lib.models", "Conversation"),
        "ConversationFilter": ("polylogue.lib.filters", "ConversationFilter"),
        "ConversationProjection": ("polylogue.lib.projections", "ConversationProjection"),
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
    "Attachment",
    "BranchType",
    "Conversation",
    "ConversationFilter",
    "ConversationProjection",
    "DialoguePair",
    "Message",
    "MessageCollection",
    "Role",
]
