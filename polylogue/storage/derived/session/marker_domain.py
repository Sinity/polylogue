"""Marker lowering as its own convergence domain.

polylogue-ylh7v.  Lowering user-tier markers used to be a tail on session-
profile publication: ``SessionProfileDerivation.publish`` committed the index
family and then opened a second, non-atomic ``user.db`` transaction, raising
``SessionProfileMarkerLoweringError`` when it failed *after* the index write had
already landed.  The other half of the same coupling ran in the opposite
direction -- ``inspect`` downgraded a perfectly valid index family to ``stale``
because a user assertion was absent, so a user-tier outage re-derived index
profiles that were never wrong.

The two tiers cannot share SQLite atomicity, and pretending otherwise is what
produced both defects.  Markers are therefore a domain: it declares its own
required space, inspects its own output relation (``assertions`` in the durable
user tier), computes its own replacement and publishes it under its own
transaction.  A user-tier failure now leaves *marker* work pending and leaves
the profile exactly as valid as the index says it is.

The candidates are read from ``blocks`` -- index-tier evidence -- so restart
inspection rediscovers them deterministically without an ingest hint or an
index-side success receipt.
"""

from __future__ import annotations

import bisect
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from polylogue.storage.derived.session.derivation import SESSION_PROFILE_DOMAIN

if TYPE_CHECKING:
    from polylogue.markers import MarkerCandidate

__all__ = [
    "SESSION_MARKER_DOMAIN",
    "SESSION_MARKER_RECIPE_VERSION",
    "SessionMarkerDerivation",
    "SessionMarkerReplacement",
    "marker_assertion_ids",
    "marker_assertions_present",
]

SESSION_MARKER_DOMAIN = "session_markers"
SESSION_MARKER_RECIPE_VERSION = "1"

_VALID = "valid"
_MISSING = "missing"


def marker_assertion_ids(conn: sqlite3.Connection, session_id: str) -> tuple[str, ...]:
    """Deterministic assertion ids this session's marker evidence lowers to."""
    from polylogue.markers.lowering import assertion_id_for_marker
    from polylogue.storage.derived.session.rebuild import marker_candidates_for_session_sync

    return tuple(
        assertion_id
        for candidate in marker_candidates_for_session_sync(conn, session_id)
        if (assertion_id := assertion_id_for_marker(candidate)) is not None
    )


def marker_assertions_present(conn: sqlite3.Connection, assertion_ids: Sequence[str]) -> bool:
    """Whether every requested marker assertion exists in the user tier."""
    # Marker identity intentionally coalesces identical markers in one block.
    # Compare unique requested IDs with SQL set membership, not occurrence count.
    unique_ids = tuple(dict.fromkeys(assertion_ids))
    if not unique_ids:
        return True
    placeholders = ",".join("?" * len(unique_ids))
    found = conn.execute(
        f"SELECT COUNT(*) FROM assertions WHERE assertion_id IN ({placeholders})",
        unique_ids,
    ).fetchone()
    return found is not None and int(found[0]) == len(unique_ids)


@dataclass(frozen=True, slots=True)
class SessionMarkerReplacement:
    """One session's marker candidates, prepared lease-free for publication."""

    key: str
    input_binding: str
    payload: tuple[MarkerCandidate, ...]
    generation_binding: str | None = None
    empty: bool = False


class SessionMarkerDerivation:
    """Lower one session's markers into the durable user tier.

    Structurally a daemon ``DerivationAdapter`` without importing the daemon
    ring, exactly like the FTS and session-profile adapters beside it.
    """

    domain = SESSION_MARKER_DOMAIN
    name = domain
    # Markers are lowered from the same session evidence the profile reads, so
    # they follow it in the declared order. The edge is deliberately one-way:
    # nothing about the profile's validity depends on this domain's output.
    prerequisites: tuple[str, ...] = (SESSION_PROFILE_DOMAIN,)
    recipe_version = SESSION_MARKER_RECIPE_VERSION

    def __init__(
        self,
        read_connection: Callable[[], sqlite3.Connection],
        marker_read_connection: Callable[[], sqlite3.Connection],
        marker_write_connection: Callable[[], sqlite3.Connection],
        *,
        session_scope: Callable[[object], Sequence[str] | None],
        page_size: int = 200,
        generation_binding: Callable[[], str] | None = None,
    ) -> None:
        self._read_connection = read_connection
        self._marker_read_connection = marker_read_connection
        self._marker_write_connection = marker_write_connection
        self._session_scope = session_scope
        self._page_size = page_size
        self._generation_binding = generation_binding

    def required_page(self, frame: object, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], str | None]:
        """Keyset-page the session space, or the frame's bounded scope."""
        from polylogue.storage.derived.session.derivation import _session_id_page

        scope = self._session_scope(frame)
        if scope is None:
            conn = self._read_connection()
            try:
                return _session_id_page(conn, cursor=cursor, limit=limit)
            finally:
                conn.close()
        keys = tuple(sorted(dict.fromkeys(str(key) for key in scope)))
        start = bisect.bisect(keys, cursor) if cursor is not None else 0
        page = keys[start : start + limit]
        return page, (page[-1] if start + len(page) < len(keys) and page else None)

    def excess_page(self, frame: object, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], str | None]:
        """No excess space: ``user.db`` is durable and irreplaceable.

        A marker assertion may carry a human judgment, so this domain never
        proposes retiring one. Deleting durable user rows because a derived
        relation no longer names them is the one thing the tier split exists to
        prevent.
        """
        del frame, cursor, limit
        return (), None

    def inspect(self, frame: object, keys: Sequence[str]) -> Mapping[str, str]:
        """Classify each session by whether its markers reached the user tier."""
        del frame
        conn = self._read_connection()
        try:
            conn.execute("BEGIN")
            ids_by_session = {key: marker_assertion_ids(conn, key) for key in keys}
        finally:
            conn.close()
        statuses: dict[str, str] = dict.fromkeys(keys, _VALID)
        pending = {key: ids for key, ids in ids_by_session.items() if ids}
        if not pending:
            return statuses
        marker_conn = self._marker_read_connection()
        try:
            for key, assertion_ids in pending.items():
                if not marker_assertions_present(marker_conn, assertion_ids):
                    statuses[key] = _MISSING
        finally:
            marker_conn.close()
        return statuses

    def prerequisite_keys(self, frame: object, key: str) -> tuple[tuple[str, str], ...]:
        del frame
        conn = self._read_connection()
        try:
            exists = conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (key,)).fetchone()
        finally:
            conn.close()
        return () if not exists else ((SESSION_PROFILE_DOMAIN, key),)

    def compute(self, frame: object, key: str) -> SessionMarkerReplacement:
        """Read one session's marker candidates from a stable index snapshot."""
        del frame
        from polylogue.storage.derived.session.rebuild import marker_candidates_for_session_sync

        generation = self._generation_binding() if self._generation_binding is not None else None
        conn = self._read_connection()
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN")
            candidates = tuple(marker_candidates_for_session_sync(conn, key))
            assertion_ids = marker_assertion_ids(conn, key)
        finally:
            conn.close()
        return SessionMarkerReplacement(
            key=key,
            input_binding=":".join(assertion_ids),
            payload=candidates,
            generation_binding=generation,
            empty=not candidates,
        )

    def publish(self, frame: object, replacement: object) -> bool:
        """Lower one session's markers in the user tier's own transaction.

        Typed ``object`` because the kernel's protocol admits any replacement;
        narrowing it would break the contract by contravariance.

        No index-tier write happens here and no index-tier write depends on
        this succeeding, so a user-tier failure propagates as an ordinary
        failed key for *this* domain. That is the whole point of the split:
        there is no longer a committed index family waiting on a second
        transaction that cannot share its atomicity.
        """
        del frame
        assert isinstance(replacement, SessionMarkerReplacement)
        from polylogue.markers import lower_markers

        if not replacement.payload:
            return True
        conn = self._marker_write_connection()
        try:
            lower_markers(conn, replacement.payload)
            conn.commit()
        finally:
            conn.close()
        return True

    def quiet(self, frame: object, key: str) -> bool:
        """Marker lowering has no hot-source policy of its own."""
        del frame, key
        return False
