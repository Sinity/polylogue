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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from polylogue.storage.derived.session.input_binding import (
    SESSION_INPUT_RECIPE_VERSION,
    session_input_bindings,
)
from polylogue.storage.sqlite.write_lease import write_lease

__all__ = [
    "SESSION_PROFILE_DOMAIN",
    "SessionProfileDerivation",
    "SessionProfileReplacement",
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

    The binding stamp is the publication boundary, not the row write. The
    existing session-insight writer commits internally, so a transaction wrapped
    around it would not be atomic and a rollback after it could not undo the
    rows; claiming otherwise would be the more dangerous shape, because a caller
    would trust an atomicity that is not there.

    What is atomic is the marker. Rows are rebuilt, then the binding is stamped
    under ``BEGIN IMMEDIATE`` only if the inputs are still the ones the
    computation read. A profile carrying no matching binding inspects stale
    (:func:`inspect_session_profiles`), so a race leaves the key pending and the
    next pass recomputes it — never a row certified against inputs that moved.

    Returns False for that refusal. An exception is a genuine failure and is
    left to the caller to attribute; the two are never collapsed.
    """
    from polylogue.storage.derived.session.rebuild import rebuild_session_insights_sync

    if session_input_bindings(conn, (session_id,)).get(session_id, "") != input_binding:
        return False

    rebuild_session_insights_sync(conn, session_ids=[session_id], page_size=page_size)

    conn.execute("BEGIN IMMEDIATE")
    try:
        if session_input_bindings(conn, (session_id,)).get(session_id, "") != input_binding:
            conn.execute(
                "UPDATE session_profiles SET input_content_hash = NULL WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
            return False
        conn.execute(
            "UPDATE session_profiles SET input_content_hash = ? WHERE session_id = ?",
            (input_binding, session_id),
        )
    except BaseException:
        conn.rollback()
        raise
    conn.commit()
    return True


#: One partition's publication is a bounded transaction over one session. A hold
#: longer than the storage busy timeout can starve a writer that is not on the
#: daemon's gate, so the budget names that boundary rather than a preference.
_PUBLISH_HOLD_BUDGET_S = 30.0


class SessionProfileDerivation:
    """Session profiles as a derivation the kernel can drive.

    Satisfies the kernel's adapter contract structurally rather than by
    inheritance: ``polylogue/storage`` may not import ``polylogue/daemon``, so
    the seam is the vocabulary (``required`` / ``inspect`` / ``compute`` /
    ``publish`` / ``prerequisites``) and the status strings above.

    ``required`` is the frame's session scope: the batch's sessions during
    incremental convergence, every session at an archive-wide boundary. Both
    reach the same inspection, so a restart that lost every scheduling hint
    reconstructs the identical pending set.
    """

    domain = SESSION_PROFILE_DOMAIN
    prerequisites: tuple[str, ...] = ()
    recipe_version = SESSION_PROFILE_RECIPE_VERSION

    def __init__(
        self,
        read_connection: Callable[[], sqlite3.Connection],
        write_connection: Callable[[], sqlite3.Connection],
        *,
        materializer_version: int,
        session_scope: Callable[[object], Sequence[str]],
        page_size: int = 200,
        quiet_keys: Callable[[object], frozenset[str]] | None = None,
    ) -> None:
        self._read_connection = read_connection
        self._write_connection = write_connection
        self._materializer_version = materializer_version
        self._session_scope = session_scope
        self._page_size = page_size
        self._quiet_keys = quiet_keys

    def required(self, frame: object) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self._session_scope(frame)))

    def inspect(self, frame: object, keys: Sequence[str]) -> Mapping[str, str]:
        conn = self._read_connection()
        try:
            return inspect_session_profiles(conn, keys, materializer_version=self._materializer_version)
        finally:
            conn.close()

    def excess_candidates(self, frame: object) -> tuple[str, ...]:
        conn = self._read_connection()
        try:
            return excess_session_profiles(conn)
        finally:
            conn.close()

    def quiet(self, frame: object, key: str) -> bool:
        return key in self._quiet_keys(frame) if self._quiet_keys is not None else False

    def compute(self, frame: object, key: str) -> SessionProfileReplacement:
        """Read the binding this replacement is computed against, lease-free.

        The profile row build still happens inside ``publish`` through the
        existing writer; moving row construction out of the lease is a separate
        change from making staleness value-complete. What matters here is that
        the binding read outside the lease is the one publication revalidates,
        so a computation that raced an ingest is refused rather than published.
        """
        conn = self._read_connection()
        try:
            binding = session_input_bindings(conn, (key,)).get(key, "")
        finally:
            conn.close()
        return SessionProfileReplacement(key=key, input_binding=binding, payload=key)

    def publish(self, frame: object, replacement: SessionProfileReplacement) -> bool:
        with write_lease(f"derivation.{self.domain}", max_hold_seconds=_PUBLISH_HOLD_BUDGET_S):
            conn = self._write_connection()
            try:
                return publish_session_profile(
                    conn,
                    replacement.key,
                    input_binding=replacement.input_binding,
                    page_size=self._page_size,
                )
            finally:
                conn.close()


@dataclass(frozen=True, slots=True)
class SessionProfileReplacement:
    """The kernel's Replacement shape, built without importing the daemon ring."""

    key: str
    input_binding: str
    payload: str
    empty: bool = False
