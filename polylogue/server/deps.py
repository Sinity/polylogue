from polylogue.lib.repository import ConversationRepository
from polylogue.storage.backends.sqlite import create_default_backend


def get_repository() -> ConversationRepository:
    """Get a ConversationRepository with default backend."""
    return ConversationRepository(backend=create_default_backend())
