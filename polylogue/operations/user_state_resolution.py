"""Resolve a session alias for a durable user-state write.

The rebuildable index is the fast path; canonical mark and annotation owners in
``user.db`` are the durable fallback, so a user-state write still resolves after
an index rebuild has not yet replayed the session row. Both the Python facade
(``polylogue.api.archive``) and the daemon's mutation handlers need this, and a
daemon handler must not import a surface, so the resolution lives here.

Synchronous by design: the daemon's mark and annotation handlers run inside the
write authority's own thread, and every step here is a plain SQLite read.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from polylogue.core.enums import AssertionKind, Provider
from polylogue.core.sources import origin_from_provider
from polylogue.storage.sqlite.connection_profile import open_readonly_connection


def durable_session_alias_matches(token: str, canonical_id: str) -> bool:
    """Apply the same exact/provider/prefix/suffix alias shapes as the index."""
    if token == canonical_id or canonical_id.startswith(token):
        return True
    if ":" in token:
        provider_token, native_id = token.split(":", 1)
        provider_origin = origin_from_provider(Provider.from_string(provider_token)).value
        return canonical_id == f"{provider_origin}:{native_id}"
    _, separator, native_id = canonical_id.partition(":")
    return bool(separator and (native_id == token or native_id.startswith(token)))


def resolve_durable_user_state_session_id(
    archive_root: Path, token: str, *, connection: sqlite3.Connection | None = None, schema: str | None = None
) -> str | None:
    """Resolve a session alias from canonical mark/annotation owners.

    The index is rebuildable, while mark and annotation ownership is durable.
    When the index cannot resolve a token, match it against the canonical
    session ids already persisted in those user-state rows. Every accepted
    alias shape is checked against the complete durable owner set, and an
    ambiguous prefix fails closed instead of selecting an arbitrary owner.
    """
    if not token:
        return None
    if schema is not None and not schema.replace("_", "").isalnum():
        raise ValueError(f"invalid SQLite schema name: {schema!r}")
    table = f"{schema}.assertions" if schema else "assertions"

    def read(conn: sqlite3.Connection) -> str | None:
        matched: str | None = None
        with closing(
            conn.execute(
                f"SELECT target_ref, scope_ref FROM {table} WHERE kind IN (?, ?) "
                "AND COALESCE(status, 'active') != 'deleted'",
                (AssertionKind.MARK.value, AssertionKind.ANNOTATION.value),
            )
        ) as rows:
            for row in rows:
                for value in (row[0], row[1]):
                    if not isinstance(value, str) or not value.startswith("session:"):
                        continue
                    canonical_id = value[len("session:") :]
                    if canonical_id and durable_session_alias_matches(token, canonical_id):
                        if matched is not None and matched != canonical_id:
                            raise ValueError(f"session id alias {token!r} is ambiguous")
                        matched = canonical_id
        return matched

    if connection is not None:
        # The supplied reader retains its original pinned User snapshot.
        return read(connection)
    user_db = archive_root / "user.db"
    if not user_db.exists():
        return None
    try:
        with closing(open_readonly_connection(user_db)) as conn:
            return read(conn)
    except sqlite3.Error:
        return None


__all__ = ["durable_session_alias_matches", "resolve_durable_user_state_session_id"]
