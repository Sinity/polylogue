"""Canonical session-token resolution over an already owned Index snapshot."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.storage.io_phase_metrics import connection_cursor


def session_id_prefix_bounds(prefix: str) -> tuple[str, str | None]:
    """Return indexed lexicographic bounds for a session-id prefix."""

    if prefix == "":
        return "", None
    chars = list(prefix)
    while chars:
        last = ord(chars[-1])
        if last < 0x10FFFF:
            chars[-1] = chr(last + 1)
            return prefix, "".join(chars)
        chars.pop()
    return prefix, None


def _session_matches(
    conn: sqlite3.Connection,
    predicate: str,
    parameters: tuple[object, ...],
    *,
    limit: int,
    before_input: Callable[[str, tuple[str, ...], str, tuple[object, ...]], None] | None,
) -> list[sqlite3.Row]:
    # Select physical identities once. Prefix/suffix ambiguity has no stable
    # representative beyond its declared ORDER BY; payload hydration uses
    # these exact selected rows, never an independent competing LIMIT query.
    with connection_cursor(
        conn, f"SELECT rowid FROM sessions WHERE {predicate} ORDER BY session_id LIMIT ?", (*parameters, limit)
    ) as cursor:
        rowids = [row[0] for row in cursor]
    result: list[sqlite3.Row] = []
    for rowid in rowids:
        if before_input is not None:
            before_input("sessions", ("session_id",), "SELECT rowid FROM sessions WHERE rowid=?", (rowid,))
        with connection_cursor(conn, "SELECT session_id FROM sessions WHERE rowid=?", (rowid,)) as cursor:
            row = cursor.fetchone()
        if row is None:
            raise KeyError("selected session identity disappeared during its owned read")
        result.append(row)
    return result


def resolve_session_id_in_index(
    conn: sqlite3.Connection,
    token: str,
    *,
    before_input: Callable[[str, tuple[str, ...], str, tuple[object, ...]], None] | None = None,
) -> str:
    """Resolve a public session token using the archive's canonical lookup law."""
    if not conn.in_transaction:
        with connection_cursor(conn, "BEGIN DEFERRED"):
            pass
        try:
            return resolve_session_id_in_index(conn, token, before_input=before_input)
        finally:
            with connection_cursor(conn, "ROLLBACK"):
                pass
    exact = _session_matches(conn, "session_id=?", (token,), limit=1, before_input=before_input)
    if exact:
        return str(exact[0]["session_id"])
    if ":" in token:
        provider_token, native_id = token.split(":", 1)
        origin_id = f"{origin_from_provider(Provider.from_string(provider_token)).value}:{native_id}"
        exact = _session_matches(conn, "session_id=?", (origin_id,), limit=1, before_input=before_input)
        if exact:
            return str(exact[0]["session_id"])
    lower_bound, upper_bound = session_id_prefix_bounds(token)
    where = "session_id >= ?"
    params: tuple[object, ...] = (lower_bound,)
    if upper_bound is not None:
        where += " AND session_id < ?"
        params += (upper_bound,)
    rows = _session_matches(conn, where, params, limit=2, before_input=before_input)
    if not rows:
        if ":" not in token:
            like_token = token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            exact_suffix_rows = _session_matches(
                conn,
                "session_id LIKE '%:' || ? ESCAPE '\\'",
                (like_token,),
                limit=2,
                before_input=before_input,
            )
            if len(exact_suffix_rows) == 1:
                return str(exact_suffix_rows[0]["session_id"])
            if len(exact_suffix_rows) > 1:
                raise ValueError(f"session id suffix {token!r} is ambiguous")
            suffix_rows = _session_matches(
                conn,
                "session_id LIKE '%:' || ? || '%' ESCAPE '\\'",
                (like_token,),
                limit=2,
                before_input=before_input,
            )
            if len(suffix_rows) == 1:
                return str(suffix_rows[0]["session_id"])
            if len(suffix_rows) > 1:
                raise ValueError(f"session id prefix {token!r} is ambiguous")
        raise KeyError(token)
    if len(rows) > 1:
        raise ValueError(f"session id prefix {token!r} is ambiguous")
    return str(rows[0]["session_id"])
