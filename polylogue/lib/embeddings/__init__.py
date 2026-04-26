"""Substrate-side embedding helpers (no CLI / click coupling)."""

from polylogue.lib.embeddings.runtime import (
    EmbedConversationOutcome,
    EmbedSingleStatus,
    PendingConversation,
    embed_conversation_sync,
    iter_pending_conversations,
)
from polylogue.lib.embeddings.stats import (
    EmbeddingStatusPayload,
    RetrievalBandPayload,
    embedding_status_payload,
)

__all__ = [
    "EmbedConversationOutcome",
    "EmbedSingleStatus",
    "EmbeddingStatusPayload",
    "PendingConversation",
    "RetrievalBandPayload",
    "embed_conversation_sync",
    "embedding_status_payload",
    "iter_pending_conversations",
]
