"""Canonical session-token resolution over an already owned Index snapshot."""

from __future__ import annotations

import sqlite3

from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider


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


def resolve_session_id_in_index(conn: sqlite3.Connection, token: str) -> str:
    """Resolve a public session token using the archive's canonical lookup law."""
    exact = conn.execute("SELECT session_id FROM sessions WHERE session_id = ?", (token,)).fetchone()
    if exact is not None:
        return str(exact["session_id"])
    if ":" in token:
        provider_token, native_id = token.split(":", 1)
        origin_id = f"{origin_from_provider(Provider.from_string(provider_token)).value}:{native_id}"
        exact = conn.execute("SELECT session_id FROM sessions WHERE session_id = ?", (origin_id,)).fetchone()
        if exact is not None:
            return str(exact["session_id"])
    lower_bound, upper_bound = session_id_prefix_bounds(token)
    where = "session_id >= ?"
    params: list[str] = [lower_bound]
    if upper_bound is not None:
        where = f"{where} AND session_id < ?"
        params.append(upper_bound)
    rows = conn.execute(
        f"""
        SELECT session_id
        FROM sessions
        WHERE {where}
        ORDER BY session_id
        LIMIT 2
        """,
        tuple(params),
    ).fetchall()
    if not rows:
        # Suffix fallback: a bare native id, full or prefix, resolves only if
        # one stored origin-prefixed id matches it.
        if ":" not in token:
            like_token = token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            exact_suffix_rows = conn.execute(
                """
                SELECT session_id
                FROM sessions
                WHERE session_id LIKE '%:' || ? ESCAPE '\\'
                ORDER BY session_id
                LIMIT 2
                """,
                (like_token,),
            ).fetchall()
            if len(exact_suffix_rows) == 1:
                return str(exact_suffix_rows[0]["session_id"])
            if len(exact_suffix_rows) > 1:
                raise ValueError(f"session id suffix {token!r} is ambiguous")
            suffix_rows = conn.execute(
                """
                SELECT session_id
                FROM sessions
                WHERE session_id LIKE '%:' || ? || '%' ESCAPE '\\'
                ORDER BY session_id
                LIMIT 2
                """,
                (like_token,),
            ).fetchall()
            if len(suffix_rows) == 1:
                return str(suffix_rows[0]["session_id"])
            if len(suffix_rows) > 1:
                raise ValueError(f"session id prefix {token!r} is ambiguous")
        raise KeyError(token)
    if len(rows) > 1:
        raise ValueError(f"session id prefix {token!r} is ambiguous")
    return str(rows[0]["session_id"])
