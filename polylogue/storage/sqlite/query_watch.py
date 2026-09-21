"""Durable registration of a watched (standing) query definition.

The validation that decides whether a selection is watchable at all is
:mod:`polylogue.archive.query.watch_definition`, which is pure and importable
from a surface. This module is the write half and runs inside the caller's
transaction, because a name whose saved view committed but whose watch did not
would be silently unwatched.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping

from polylogue.archive.query.watch_definition import (
    WATCH_DEFINITION_GRAIN,
    WATCH_DEFINITION_LANE,
    WATCH_DEFINITION_RANK_POLICY,
    WatchDefinitionError,
    compile_watch_definition,
    validate_watch_definition,
)
from polylogue.storage.sqlite.query_objects import put_query, put_query_name


def register_query_watch(
    conn: sqlite3.Connection,
    *,
    name: str,
    query_params: Mapping[str, object],
    watch: bool,
    now_ms: int,
) -> str | None:
    """Bind one saved-view name to a durable watched definition, or clear it.

    Returns the registered ``query_hash``, or ``None`` when the name carries no
    watch.  ``watch=False`` clears an existing watch on that name rather than
    leaving a stale one behind: a saved view's definition is replaced wholesale
    on every save, so a watch that outlived the save would re-evaluate a
    definition the name no longer denotes.
    """
    if not watch:
        conn.execute(
            "UPDATE query_names SET watch = 0, updated_at_ms = ? WHERE name = ? AND watch = 1",
            (now_ms, name),
        )
        return None
    ast = validate_watch_definition(query_params)
    query = put_query(
        conn,
        ast,
        grain=WATCH_DEFINITION_GRAIN,
        lane=WATCH_DEFINITION_LANE,
        rank_policy=WATCH_DEFINITION_RANK_POLICY,
        created_at_ms=now_ms,
    )
    put_query_name(conn, name=name, query_hash=query.query_hash, watch=True, updated_at_ms=now_ms)
    return query.query_hash


__all__ = [
    "WATCH_DEFINITION_GRAIN",
    "WATCH_DEFINITION_LANE",
    "WATCH_DEFINITION_RANK_POLICY",
    "WatchDefinitionError",
    "compile_watch_definition",
    "register_query_watch",
    "validate_watch_definition",
]
