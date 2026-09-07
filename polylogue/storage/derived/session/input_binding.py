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
    "SESSION_INPUT_PROJECTION_COLUMNS",
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
SESSION_INPUT_RECIPE_VERSION = "1"

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
    return digest.result()
