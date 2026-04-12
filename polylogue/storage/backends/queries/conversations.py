"""Small public root for conversation query families."""

from __future__ import annotations

from polylogue.storage.backends.queries.conversations_identity import (
    conversation_id_query,
    count_conversation_ids,
    get_last_sync_timestamp,
    get_metadata,
    iter_conversation_ids,
    list_tags,
    resolve_id,
    set_metadata,
    update_metadata_raw,
)
from polylogue.storage.backends.queries.conversations_reads import (
    count_conversations,
    get_conversation,
    get_conversations_batch,
    list_conversation_summaries,
    list_conversations,
    list_conversations_by_parent,
)
from polylogue.storage.backends.queries.conversations_search import (
    search_action_conversations,
    search_conversations,
)
from polylogue.storage.backends.queries.conversations_writes import (
    conversation_exists_by_hash,
    delete_conversation_sql,
    save_conversation_record,
)

__all__ = [
    "conversation_exists_by_hash",
    "conversation_id_query",
    "count_conversation_ids",
    "count_conversations",
    "delete_conversation_sql",
    "get_conversation",
    "get_conversations_batch",
    "get_last_sync_timestamp",
    "get_metadata",
    "iter_conversation_ids",
    "list_conversation_summaries",
    "list_conversations",
    "list_conversations_by_parent",
    "list_tags",
    "resolve_id",
    "save_conversation_record",
    "search_action_conversations",
    "search_conversations",
    "set_metadata",
    "update_metadata_raw",
]
