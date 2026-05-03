"""Raw conversation query public surface."""

from __future__ import annotations

from polylogue.storage.sqlite.queries.raw_reads import (
    get_known_source_mtimes,
    get_raw_blob_sizes,
    get_raw_conversation,
    get_raw_conversation_count,
    get_raw_conversation_states,
    get_raw_conversations_batch,
    get_raw_records_for_conversation,
    iter_raw_conversations,
    iter_raw_headers,
    iter_raw_ids,
    raw_header_query,
    raw_id_query,
)
from polylogue.storage.sqlite.queries.raw_state import (
    apply_raw_state_update,
    mark_raw_parsed,
    mark_raw_validated,
    reset_parse_status,
    reset_validation_status,
)
from polylogue.storage.sqlite.queries.raw_writes import save_raw_conversation

__all__ = [
    "apply_raw_state_update",
    "get_known_source_mtimes",
    "get_raw_blob_sizes",
    "get_raw_conversation",
    "get_raw_conversation_count",
    "get_raw_conversation_states",
    "get_raw_conversations_batch",
    "get_raw_records_for_conversation",
    "iter_raw_headers",
    "iter_raw_conversations",
    "iter_raw_ids",
    "mark_raw_parsed",
    "mark_raw_validated",
    "raw_header_query",
    "raw_id_query",
    "reset_parse_status",
    "reset_validation_status",
    "save_raw_conversation",
]
