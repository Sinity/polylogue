"""Small public root for session query families."""

from __future__ import annotations

from polylogue.storage.sqlite.queries.sessions_identity import (
    count_session_ids,
    get_last_sync_timestamp,
    get_metadata,
    iter_session_ids,
    list_tags,
    resolve_id,
    session_id_query,
    set_metadata,
    update_metadata_raw,
)
from polylogue.storage.sqlite.queries.sessions_reads import (
    count_sessions,
    get_session,
    get_sessions_batch,
    list_session_summaries,
    list_sessions,
    list_sessions_by_parent,
)
from polylogue.storage.sqlite.queries.sessions_search import (
    search_action_session_hits,
    search_action_sessions,
    search_session_evidence_hits,
    search_session_hits,
    search_sessions,
)
from polylogue.storage.sqlite.queries.sessions_writes import (
    delete_session_sql,
    session_exists_by_hash,
)

__all__ = [
    "count_session_ids",
    "count_sessions",
    "delete_session_sql",
    "get_session",
    "get_sessions_batch",
    "get_last_sync_timestamp",
    "get_metadata",
    "iter_session_ids",
    "list_session_summaries",
    "list_sessions",
    "list_sessions_by_parent",
    "list_tags",
    "resolve_id",
    "session_exists_by_hash",
    "session_id_query",
    "search_action_session_hits",
    "search_action_sessions",
    "search_session_evidence_hits",
    "search_session_hits",
    "search_sessions",
    "set_metadata",
    "update_metadata_raw",
]
