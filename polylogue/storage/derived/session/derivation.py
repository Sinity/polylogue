"""Domain-owned convergence for the session-profile aggregate family.

``session_profiles`` is the authority for its own output. The only binding
stored beside it is the value-complete input digest in
``session_profiles.input_content_hash`` — identity the output rows cannot carry
themselves.

Inspection is authoritative, not advisory: it recomputes the input digest from
``messages`` instead of comparing a sort key, an updated-at, a row count, or the
session's own content hash. None of those move when a role, a model name, or a
token count does, and every one of those values feeds the profile.

Statuses are returned as the derivation kernel's string vocabulary rather than
its enum: storage may not import the daemon ring, and the vocabulary is the
contract either way.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from polylogue.storage.derived.session.input_binding import (
    SESSION_INPUT_RECIPE_VERSION,
    session_input_bindings,
)

__all__ = [
    "SESSION_PROFILE_DOMAIN",
    "SESSION_PROFILE_RECIPE_VERSION",
    "excess_session_profiles",
    "inspect_session_profiles",
    "publish_session_profile",
    "stored_session_profile_binding",
]

SESSION_PROFILE_DOMAIN = "session_profile"
SESSION_PROFILE_RECIPE_VERSION = SESSION_INPUT_RECIPE_VERSION

_VALID = "valid"
_MISSING = "missing"
_STALE = "stale"


@dataclass(frozen=True, slots=True)
class _StoredBinding:
    present: bool
    materializer_version: int | None
    input_binding: str | None


def stored_session_profile_binding(conn: sqlite3.Connection, session_id: str) -> str | None:
    """The binding a stored profile says it was computed from, if any."""
    row = conn.execute(
        "SELECT input_content_hash FROM session_profiles WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return str(row[0])


def _stored_bindings(conn: sqlite3.Connection, session_ids: Sequence[str]) -> Mapping[str, _StoredBinding]:
    unique = tuple(dict.fromkeys(session_ids))
    if not unique:
        return {}
    placeholders = ",".join("?" * len(unique))
    rows = conn.execute(
        f"""
        SELECT session_id, materializer_version, input_content_hash
        FROM session_profiles
        WHERE session_id IN ({placeholders})
        """,
        unique,
    ).fetchall()
    stored = {
        str(row[0]): _StoredBinding(
            present=True,
            materializer_version=None if row[1] is None else int(row[1]),
            input_binding=None if row[2] is None else str(row[2]),
        )
        for row in rows
    }
    for session_id in unique:
        stored.setdefault(session_id, _StoredBinding(present=False, materializer_version=None, input_binding=None))
    return stored


def inspect_session_profiles(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
    *,
    materializer_version: int,
) -> Mapping[str, str]:
    """Classify each session's profile from the output relation and its binding.

    A profile is valid only when it exists, was built by the current
    materializer, and its stored binding equals the digest recomputed now from
    the authoritative message projection. A profile with no stored binding is
    stale, never valid: a row that cannot say what it was computed from cannot
    certify itself.
    """
    unique = tuple(dict.fromkeys(session_ids))
    if not unique:
        return {}
    stored = _stored_bindings(conn, unique)
    current = session_input_bindings(conn, unique)
    statuses: dict[str, str] = {}
    for session_id in unique:
        record = stored[session_id]
        stale = (
            record.materializer_version != materializer_version
            or record.input_binding is None
            or record.input_binding != current.get(session_id)
        )
        if not record.present:
            statuses[session_id] = _MISSING
        else:
            statuses[session_id] = _STALE if stale else _VALID
    return statuses


def excess_session_profiles(conn: sqlite3.Connection, *, limit: int = 1000) -> tuple[str, ...]:
    """Profiles whose session is gone. The output relation names them itself."""
    rows = conn.execute(
        """
        SELECT sp.session_id
        FROM session_profiles AS sp
        LEFT JOIN sessions AS s ON s.session_id = sp.session_id
        WHERE s.session_id IS NULL
        ORDER BY sp.session_id
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return tuple(str(row[0]) for row in rows)


def publish_session_profile(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    input_binding: str,
    page_size: int = 200,
) -> bool:
    """Replace one session's profile partition, revalidating the binding.

    The binding is re-read inside ``BEGIN IMMEDIATE`` and compared to the one
    the replacement was computed against, and again after the rows are written.
    A mismatch means an ingest landed under the computation, so the transaction
    rolls back and the key stays pending: publishing anyway is how an output
    bound to inputs that no longer exist reaches an authoritative relation.

    Returns False for that refusal. An exception is a genuine failure and is
    left to the caller to attribute; the two are never collapsed.
    """
    from polylogue.storage.derived.session.rebuild import rebuild_session_insights_sync

    conn.execute("BEGIN IMMEDIATE")
    try:
        before = session_input_bindings(conn, (session_id,)).get(session_id, "")
        if before != input_binding:
            conn.rollback()
            return False
        rebuild_session_insights_sync(conn, session_ids=[session_id], page_size=page_size)
        conn.execute(
            "UPDATE session_profiles SET input_content_hash = ? WHERE session_id = ?",
            (input_binding, session_id),
        )
        after = session_input_bindings(conn, (session_id,)).get(session_id, "")
        if after != input_binding:
            conn.rollback()
            return False
    except BaseException:
        conn.rollback()
        raise
    conn.commit()
    return True
