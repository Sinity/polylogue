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

__all__ = [
    "SESSION_INPUT_PROJECTION_COLUMNS",
    "SESSION_INPUT_RECIPE_VERSION",
    "session_input_bindings",
    "session_input_binding_sql",
]

#: Bumped when the meaning of a session-scoped derivation changes without the
#: projection changing. Every stored binding compares unequal afterwards, which
#: is the whole invalidation mechanism: there is no separate freshness ledger.
SESSION_INPUT_RECIPE_VERSION = "1"

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
    digests = {session_id: hashlib.blake2b(digest_size=16) for session_id in unique}
    for digest in digests.values():
        digest.update(SESSION_INPUT_RECIPE_VERSION.encode("utf-8"))
        digest.update(b"\x00".join(column.encode("utf-8") for column in SESSION_INPUT_PROJECTION_COLUMNS))

    cursor = conn.execute(session_input_binding_sql(len(unique)), unique)
    try:
        for row in cursor:
            session_id = str(row[0])
            if session_id not in digests:  # pragma: no cover - IN () cannot return an unasked id
                continue
            digest = digests[session_id]
            digest.update(b"\x1e")
            digest.update(b"\x1f".join(b"" if value is None else str(value).encode("utf-8") for value in row[1:]))
    finally:
        cursor.close()
    return {session_id: digest.hexdigest() for session_id, digest in digests.items()}
