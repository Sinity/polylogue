"""Value-complete input bindings for session-scoped derived aggregates.

A derived aggregate is stale exactly when an input value its output depends on
has changed. Binding to identity instead — a sort key, an updated-at, a row
count, a partition key, or the session's own content hash — reports VALID after
a mutation that changes the output, because none of those move when a message's
role, model, or token counts do (measured on master 936f2ff: the profile stale
predicate compared sort key and source timestamp only, and an exact-master audit
confirmed a role-only mutation left every profile-derived table stale and
uninspectable).

The binding here is a digest over the exact message projection the profile
computation reads. Two properties make it load-bearing:

- **Complete.** Every column the computation consumes is in the digest. Adding
  a column to the computation without adding it here is the failure mode the
  projection constant exists to make reviewable in one place.
- **Recipe-bound.** The digest commits to :data:`SESSION_INPUT_RECIPE_VERSION`,
  so changing what the computation *means* invalidates every binding without
  touching a stored row.

The session's own ``content_hash`` is deliberately not the binding: it covers
the parser's semantic payload, while usage and model measurements are excluded
from that hash by design (``pipeline/ids.py``) and are read by the profile.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Mapping, Sequence

import aiosqlite

__all__ = [
    "SESSION_INPUT_EXCLUDED_COLUMNS",
    "SESSION_INPUT_PROJECTION_COLUMNS",
    "SESSION_PROVIDER_USAGE_EVENT_PROJECTION_COLUMNS",
    "SESSION_ROW_EXCLUDED_COLUMNS",
    "SESSION_ROW_PROJECTION_COLUMNS",
    "SessionInputDigest",
    "SESSION_INPUT_RECIPE_VERSION",
    "session_input_bindings",
    "session_input_bindings_async",
    "session_input_binding_sql",
    "session_row_binding_sql",
]

#: Bumped when the meaning of a session-scoped derivation changes without the
#: projection changing. Every stored binding compares unequal afterwards, which
#: is the whole invalidation mechanism: there is no separate freshness ledger.
SESSION_INPUT_RECIPE_VERSION = "3"

#: The exact session-row columns session-scoped aggregates read. The profile
#: caches several of these directly (``source_sort_key``, ``source_updated_at``,
#: ``canonical_session_date``, ``title``, ``source_name``, repository paths), so
#: a binding over messages alone reports valid after a session-row change that
#: moved the output -- which is the same defect one level up.
SESSION_ROW_PROJECTION_COLUMNS: tuple[str, ...] = (
    "origin",
    "title",
    "branch_type",
    "session_kind",
    "parent_session_id",
    "root_session_id",
    "git_branch",
    "git_repository_url",
    "provider_project_ref",
    "reported_duration_ms",
    "reported_cost_usd",
    "message_count",
    "word_count",
    "tool_use_count",
    "thinking_count",
    "paste_count",
    "user_message_count",
    "authored_user_message_count",
    "assistant_message_count",
    "system_message_count",
    "tool_message_count",
    "user_word_count",
    "authored_user_word_count",
    "assistant_word_count",
    "content_hash",
    "created_at_ms",
    "updated_at_ms",
    "sort_key_ms",
)

#: The exact message columns session-scoped aggregates read. ``content_hash``
#: carries the semantic payload (text, blocks, structure); the rest are values
#: it deliberately excludes but the aggregates consume.
SESSION_INPUT_PROJECTION_COLUMNS: tuple[str, ...] = (
    "position",
    "variant_index",
    "role",
    "message_type",
    "material_origin",
    "model_name",
    "word_count",
    "has_tool_use",
    "has_thinking",
    "has_paste",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "duration_ms",
    "occurred_at_ms",
    "stop_reason",
    "is_active_path",
    "content_hash",
)

# ``load_sync_batch`` hydrates these attachment values directly.  They live in
# two relations, so neither the message hash nor the session row can certify a
# prepared profile after one changes.
SESSION_ATTACHMENT_PROJECTION_COLUMNS: tuple[str, ...] = (
    "attachment_id",
    "display_name",
    "media_type",
    "byte_count",
    "source_url",
    "caption",
    "upload_origin",
    "message_id",
)

# ``sync_session_events_batch`` maps precisely these event values into the
# profile runtime, and the compaction count is a function of ``event_type``.
# ``summary`` is intentionally absent: the hydrator does not read it.
SESSION_EVENT_PROJECTION_COLUMNS: tuple[str, ...] = (
    "source_message_id",
    "source_message_provider_id",
    "position",
    "event_type",
    "payload_json",
    "occurred_at_ms",
    "boundary_start_position",
    "boundary_end_position",
    "boundary_message_id",
)

# ``_refresh_provider_usage_rollup`` derives ``session_model_usage`` from this
# persisted provider evidence immediately before the profile reads that rollup.
# These are its exact input columns: leaving them outside the binding lets a
# fixed-id usage correction change the profile's dominant model while
# inspection still reports the old partition VALID.
SESSION_PROVIDER_USAGE_EVENT_PROJECTION_COLUMNS: tuple[str, ...] = (
    "position",
    "provider_event_type",
    "model_name",
    "last_input_tokens",
    "last_output_tokens",
    "last_cached_input_tokens",
    "last_cache_write_tokens",
    "last_reasoning_output_tokens",
    "last_total_tokens",
    "total_input_tokens",
    "total_output_tokens",
    "total_cached_input_tokens",
    "total_cache_write_tokens",
    "total_reasoning_output_tokens",
    "total_tokens",
)

#: Every ``sessions`` column the projection deliberately leaves out, with the
#: reason it is not an input value. Completeness is the property this module
#: exists for, and an unclassified column is the one way to lose it silently:
#: a column added to ``sessions`` and read by the profile, but never added
#: here or above, makes the binding report VALID after the output moved. The
#: partition of the relation's columns is checked against the live schema by
#: ``tests/unit/storage/test_session_partition_convergence.py``, so adding a
#: column forces a decision rather than a default.
SESSION_ROW_EXCLUDED_COLUMNS: Mapping[str, str] = {
    "session_id": "the partition key: it selects the row rather than being a value in it",
    "native_id": "the partition key's other half (session_id = origin || ':' || native_id)",
    "raw_id": "acquisition provenance; a re-parse that changes what the profile reads moves content_hash",
    "parser_fingerprint": "which parser build produced the row, not what it says",
    "lowering_fingerprint": "which lowering produced the row, not what it says",
    "active_leaf_message_id": "hashed ParsedSession field, covered by the projected content_hash",
    "title_source": "hashed ParsedSession field, covered by the projected content_hash",
    "title_ref": "hashed ParsedSession field, covered by the projected content_hash",
    "display_name": "hashed ParsedSession field, covered by the projected content_hash",
    "pending_drafts_json": "hashed ParsedSession field, covered by the projected content_hash",
    "commit_hash": "hashed ParsedSession field, covered by the projected content_hash",
    "instructions_text": "hashed ParsedSession field, covered by the projected content_hash",
}

#: The same partition for ``messages``. The usage and timing measurements
#: ``pipeline/ids.py`` excludes from the message content hash are projected
#: above precisely because the hash cannot carry them; everything listed here
#: either is carried by that hash or is a coordinate the projection already
#: scopes and orders by.
SESSION_INPUT_EXCLUDED_COLUMNS: Mapping[str, str] = {
    "message_id": "generated from session_id, position, variant_index and native_id, all already bound",
    "session_id": "the partition key: the projection selects on it and groups by it",
    "native_id": "identity input to message_id, not a value the profile reads",
    "identity_source": "records which identity path fired, not what the message says",
    "parent_message_id": "lineage coordinate resolved from hashed parser fields",
    "is_active_leaf": "lineage marker derived from the same hashed payload as is_active_path",
    "content_address": "storage address of the content the projected content_hash already binds",
    "paste_boundary": "derived from the hashed paste_spans field",
    "model_effort": "hashed ParsedMessage field, covered by the projected content_hash",
    "sender_name": "hashed ParsedMessage field, covered by the projected content_hash",
    "recipient": "hashed ParsedMessage field, covered by the projected content_hash",
    "delivery_status": "hashed ParsedMessage field, covered by the projected content_hash",
    "end_turn": "hashed ParsedMessage field, covered by the projected content_hash",
    "user_context_text": "hashed ParsedMessage field, covered by the projected content_hash",
}

_HASHED_BLOB_COLUMNS = frozenset({"content_hash"})


def session_row_binding_sql(session_count: int) -> str:
    """Session-row projection SQL for ``session_count`` sessions."""
    if session_count < 1:
        raise ValueError("session_row_binding_sql requires at least one session")
    placeholders = ",".join("?" * session_count)
    projected = ",\n    ".join(
        f"lower(hex(s.{column}))" if column in _HASHED_BLOB_COLUMNS else f"s.{column}"
        for column in SESSION_ROW_PROJECTION_COLUMNS
    )
    return f"""
SELECT
    s.session_id,
    {projected}
FROM sessions s
WHERE s.session_id IN ({placeholders})
ORDER BY s.session_id
"""


def session_input_binding_sql(session_count: int) -> str:
    """Ordered projection SQL for ``session_count`` sessions.

    Ordering is in SQL rather than inside an aggregate: SQLite does not promise
    an order for rows fed to ``group_concat``, so a digest built that way would
    be unstable across query plans and report spurious staleness.
    """
    if session_count < 1:
        raise ValueError("session_input_binding_sql requires at least one session")
    placeholders = ",".join("?" * session_count)
    projected = ",\n    ".join(
        f"lower(hex(m.{column}))" if column in _HASHED_BLOB_COLUMNS else f"m.{column}"
        for column in SESSION_INPUT_PROJECTION_COLUMNS
    )
    return f"""
SELECT
    m.session_id,
    {projected}
FROM messages m
WHERE m.session_id IN ({placeholders})
ORDER BY m.session_id, m.position, m.variant_index, m.message_id
"""


def session_attachment_binding_sql(session_count: int) -> str:
    """Ordered attachment projection consumed by session hydration."""
    if session_count < 1:
        raise ValueError("session_attachment_binding_sql requires at least one session")
    placeholders = ",".join("?" * session_count)
    return f"""
SELECT
    r.session_id,
    a.attachment_id,
    a.display_name,
    a.media_type,
    a.byte_count,
    r.source_url,
    r.caption,
    r.upload_origin,
    r.message_id
FROM attachment_refs r
JOIN attachments a ON a.attachment_id = r.attachment_id
WHERE r.session_id IN ({placeholders})
ORDER BY r.session_id, r.message_id, r.position, a.attachment_id
"""


def session_event_binding_sql(session_count: int) -> str:
    """Ordered session-event projection consumed by profile construction."""
    if session_count < 1:
        raise ValueError("session_event_binding_sql requires at least one session")
    placeholders = ",".join("?" * session_count)
    projected = ",\n    ".join(f"se.{column}" for column in SESSION_EVENT_PROJECTION_COLUMNS)
    return f"""
SELECT
    se.session_id,
    {projected}
FROM session_events se
WHERE se.session_id IN ({placeholders})
ORDER BY se.session_id, se.position
"""


def session_provider_usage_event_binding_sql(session_count: int) -> str:
    """Ordered provider-usage projection consumed by the usage-rollup refresh."""
    if session_count < 1:
        raise ValueError("session_provider_usage_event_binding_sql requires at least one session")
    placeholders = ",".join("?" * session_count)
    projected = ",\n    ".join(f"pue.{column}" for column in SESSION_PROVIDER_USAGE_EVENT_PROJECTION_COLUMNS)
    return f"""
SELECT
    pue.session_id,
    {projected}
FROM session_provider_usage_events pue
WHERE pue.session_id IN ({placeholders})
ORDER BY pue.session_id, pue.position
"""


class SessionInputDigest:
    """Accumulates one binding per session from projection rows, in order.

    The sync and async writers hold different connection types and cannot share
    a query loop, so they share this instead: the digest definition — recipe,
    column list, row encoding — exists once, and each caller only feeds it rows.
    Two copies of the definition would drift and report false staleness on
    whichever route fell behind.
    """

    __slots__ = ("_digests",)

    def __init__(self, session_ids: Sequence[str]) -> None:
        self._digests = {session_id: hashlib.blake2b(digest_size=16) for session_id in session_ids}
        for digest in self._digests.values():
            digest.update(SESSION_INPUT_RECIPE_VERSION.encode("utf-8"))
            digest.update(b"\x00".join(column.encode("utf-8") for column in SESSION_ROW_PROJECTION_COLUMNS))
            digest.update(b"\x00".join(column.encode("utf-8") for column in SESSION_INPUT_PROJECTION_COLUMNS))

    def add_row(self, row: Sequence[object]) -> None:
        session_id = str(row[0])
        digest = self._digests.get(session_id)
        if digest is None:  # pragma: no cover - IN () cannot return an unasked id
            return
        digest.update(b"\x1e")
        digest.update(b"\x1f".join(b"" if value is None else str(value).encode("utf-8") for value in row[1:]))

    def add_related_row(self, relation: str, row: Sequence[object]) -> None:
        """Add one ordered value row from a non-message input relation."""
        session_id = str(row[0])
        digest = self._digests.get(session_id)
        if digest is None:  # pragma: no cover - IN () cannot return an unasked id
            return
        digest.update(b"\x1d")
        digest.update(relation.encode("utf-8"))
        digest.update(b"\x1e")
        digest.update(b"\x1f".join(b"" if value is None else str(value).encode("utf-8") for value in row[1:]))

    def result(self) -> dict[str, str]:
        return {session_id: digest.hexdigest() for session_id, digest in self._digests.items()}


def session_input_bindings(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
) -> Mapping[str, str]:
    """Digest the authoritative message projection for each session.

    A session with no messages gets a digest over the empty projection rather
    than being absent: valid-empty output must be distinguishable from work that
    was never performed, and the caller cannot tell those apart from a missing
    key.
    """
    unique = tuple(dict.fromkeys(session_ids))
    if not unique:
        return {}
    digest = SessionInputDigest(unique)
    for sql in (session_row_binding_sql(len(unique)), session_input_binding_sql(len(unique))):
        cursor = conn.execute(sql, unique)
        try:
            for row in cursor:
                digest.add_row(row)
        finally:
            cursor.close()
    for relation, sql in (
        ("attachments", session_attachment_binding_sql(len(unique))),
        ("session_events", session_event_binding_sql(len(unique))),
        ("provider_usage_events", session_provider_usage_event_binding_sql(len(unique))),
    ):
        cursor = conn.execute(sql, unique)
        try:
            for row in cursor:
                digest.add_related_row(relation, row)
        finally:
            cursor.close()
    return digest.result()


async def session_input_bindings_async(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
) -> dict[str, str]:
    """The same binding over an async connection.

    The two connection types cannot share a cursor loop, so they share the
    digest definition and the projection SQL instead. Nothing else differs, and
    a route that computed its own digest would report false staleness against
    whichever route it drifted from.
    """
    unique = tuple(dict.fromkeys(str(session_id) for session_id in session_ids))
    if not unique:
        return {}
    digest = SessionInputDigest(unique)
    for sql in (session_row_binding_sql(len(unique)), session_input_binding_sql(len(unique))):
        async with conn.execute(sql, unique) as cursor:
            async for row in cursor:
                digest.add_row(row)
    for relation, sql in (
        ("attachments", session_attachment_binding_sql(len(unique))),
        ("session_events", session_event_binding_sql(len(unique))),
        ("provider_usage_events", session_provider_usage_event_binding_sql(len(unique))),
    ):
        async with conn.execute(sql, unique) as cursor:
            async for row in cursor:
                digest.add_related_row(relation, row)
    return digest.result()
