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


def resolve_durable_user_state_session_id(archive_root: Path, token: str) -> str | None:
    """Resolve a session alias from canonical mark/annotation owners.

    The index is rebuildable, while mark and annotation ownership is durable.
    When the index cannot resolve a token, match it against the canonical
    session ids already persisted in those user-state rows. Every accepted
    alias shape is checked against the complete durable owner set, and an
    ambiguous prefix fails closed instead of selecting an arbitrary owner.
    """
    if not token:
        return None
    user_db = archive_root / "user.db"
    if not user_db.exists():
        return None
    try:
        with closing(open_readonly_connection(user_db)) as conn:
            rows = conn.execute(
                """
                SELECT target_ref, scope_ref
                FROM assertions
                WHERE kind IN (?, ?)
                  AND COALESCE(status, 'active') != 'deleted'
                """,
                (AssertionKind.MARK.value, AssertionKind.ANNOTATION.value),
            ).fetchall()
    except sqlite3.Error:
        return None

    canonical_ids: set[str] = set()
    for row in rows:
        for value in (row[0], row[1]):
            if not isinstance(value, str):
                continue
            if value.startswith("session:"):
                canonical_id = value[len("session:") :]
                if canonical_id:
                    canonical_ids.add(canonical_id)

    matches = {canonical_id for canonical_id in canonical_ids if durable_session_alias_matches(token, canonical_id)}
    if len(matches) > 1:
        raise ValueError(f"session id alias {token!r} is ambiguous")
    return next(iter(matches), None)


__all__ = ["durable_session_alias_matches", "resolve_durable_user_state_session_id"]
