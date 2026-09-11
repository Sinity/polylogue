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

import bisect
import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from polylogue.storage.derived.session.input_binding import (
    SESSION_INPUT_RECIPE_VERSION,
    session_input_bindings,
    session_input_bindings_async,
)
from polylogue.storage.sqlite.write_lease import write_lease

__all__ = [
    "SESSION_PARTITION_INSPECT_CHUNK",
    "SESSION_PROFILE_DOMAIN",
    "SessionProfileDerivation",
    "SessionProfileReplacement",
    "SESSION_PROFILE_RECIPE_VERSION",
    "archive_session_partition_statuses",
    "archive_session_partition_statuses_async",
    "excess_session_profiles",
    "inspect_session_profiles",
    "inspect_session_profiles_async",
    "publish_session_profile",
    "publish_prepared_session_profile",
    "stored_session_profile_binding",
]

SESSION_PROFILE_DOMAIN = "session_profile"
SESSION_PROFILE_RECIPE_VERSION = SESSION_INPUT_RECIPE_VERSION

_VALID = "valid"
_MISSING = "missing"
_STALE = "stale"


def _marker_assertion_ids(conn: sqlite3.Connection, session_id: str) -> tuple[str, ...]:
    from polylogue.markers.lowering import assertion_id_for_marker
    from polylogue.storage.derived.session.rebuild import marker_candidates_for_session_sync

    return tuple(
        assertion_id
        for candidate in marker_candidates_for_session_sync(conn, session_id)
        if (assertion_id := assertion_id_for_marker(candidate)) is not None
    )


def _marker_assertions_present(conn: sqlite3.Connection, assertion_ids: Sequence[str]) -> bool:
    if not assertion_ids:
        return True
    placeholders = ",".join("?" * len(assertion_ids))
    found = conn.execute(
        f"SELECT COUNT(*) FROM assertions WHERE assertion_id IN ({placeholders})",
        tuple(assertion_ids),
    ).fetchone()
    return found is not None and int(found[0]) == len(assertion_ids)


def _lower_prepared_markers(
    marker_write_connection: Callable[[], sqlite3.Connection],
    prepared: object,
) -> None:
    """Publish marker candidates after the index family has committed.

    The user tier cannot share SQLite atomicity with the derived index tier.
    It is deliberately a separate, idempotent assertion transaction; restart
    inspection above re-discovers a missing lowering without relying on a
    volatile ingest hint.
    """
    from polylogue.markers import lower_markers
    from polylogue.storage.derived.session.rebuild import PreparedSessionInsightPartition

    if not isinstance(prepared, PreparedSessionInsightPartition) or prepared.bundle is None:
        return
    candidates = prepared.bundle.marker_candidates
    if not candidates:
        return
    conn = marker_write_connection()
    try:
        lower_markers(conn, candidates)
        conn.commit()
    finally:
        conn.close()


def _connection_generation(conn: sqlite3.Connection) -> str:
    """Return the physical main-database path a writer actually opened."""
    for _sequence, name, filename in conn.execute("PRAGMA database_list"):
        if name == "main" and filename:
            return str(Path(str(filename)).resolve())
    raise RuntimeError("session profile writer has no main database generation")


@dataclass(frozen=True, slots=True)
class _StoredPartition:
    """One session partition as the output relations themselves report it."""

    present: bool
    materializer_version: int | None
    input_binding: str | None
    declared_work_events: int = 0
    declared_phases: int = 0
    stored_work_events: int = 0
    stored_phases: int = 0
    latency_rows: int = 0


_ABSENT_PARTITION = _StoredPartition(present=False, materializer_version=None, input_binding=None)

#: The partition's sibling relations, read back per session. They are written
#: inside the same replacement as the profile row, so a partition whose siblings
#: disagree with the profile is not a fresh output with an accounting quirk --
#: it is a half-replaced partition, and inspecting only the profile row would
#: certify it.
_STORED_PARTITION_SQL = """
SELECT
    sp.session_id,
    sp.materializer_version,
    sp.input_content_hash,
    sp.work_event_count,
    sp.phase_count,
    (SELECT COUNT(*) FROM session_work_events e WHERE e.session_id = sp.session_id),
    (SELECT COUNT(*) FROM session_phases p WHERE p.session_id = sp.session_id),
    (SELECT COUNT(*) FROM session_latency_profiles l WHERE l.session_id = sp.session_id)
FROM session_profiles sp
WHERE sp.session_id IN ({placeholders})
"""


def stored_session_profile_binding(conn: sqlite3.Connection, session_id: str) -> str | None:
    """The binding a stored profile says it was computed from, if any."""
    row = conn.execute(
        "SELECT input_content_hash FROM session_profiles WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return str(row[0])


def _count(value: object) -> int:
    return int(value) if isinstance(value, int | float | str) else 0


def _partition_row(row: Sequence[object]) -> _StoredPartition:
    return _StoredPartition(
        present=True,
        materializer_version=None if row[1] is None else _count(row[1]),
        input_binding=None if row[2] is None else str(row[2]),
        declared_work_events=_count(row[3]),
        declared_phases=_count(row[4]),
        stored_work_events=_count(row[5]),
        stored_phases=_count(row[6]),
        latency_rows=_count(row[7]),
    )


def _classify_partition(
    stored: _StoredPartition,
    current_binding: str | None,
    *,
    materializer_version: int,
) -> str:
    """The one place a session partition's status is decided.

    Shared verbatim by every caller -- the batch route, the archive-wide route,
    and the async status route -- so no two of them can disagree about whether
    a partition is current.
    """
    if not stored.present:
        return _MISSING
    if stored.materializer_version != materializer_version:
        return _STALE
    if stored.input_binding is None or stored.input_binding != current_binding:
        return _STALE
    if stored.stored_work_events != stored.declared_work_events:
        return _STALE
    if stored.stored_phases != stored.declared_phases:
        return _STALE
    if stored.latency_rows != 1:
        return _STALE
    return _VALID


def _stored_partitions(conn: sqlite3.Connection, session_ids: Sequence[str]) -> Mapping[str, _StoredPartition]:
    unique = tuple(dict.fromkeys(session_ids))
    if not unique:
        return {}
    sql = _STORED_PARTITION_SQL.format(placeholders=",".join("?" * len(unique)))
    stored = {str(row[0]): _partition_row(row) for row in conn.execute(sql, unique).fetchall()}
    for session_id in unique:
        stored.setdefault(session_id, _ABSENT_PARTITION)
    return stored


def inspect_session_profiles(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
    *,
    materializer_version: int,
) -> Mapping[str, str]:
    """Classify each session partition from its output relations and binding.

    The partition is the family: the profile row plus the work events, phases
    and latency profile written in the same replacement. It is valid only when
    every one of those exists as the profile declares, the profile was built by
    the current materializer, and its stored binding equals the digest
    recomputed now from the authoritative message projection.

    A profile with no stored binding is stale, never valid: a row that cannot
    say what it was computed from cannot certify itself. A profile whose sibling
    relations disagree with its own declared counts is stale for the same
    reason -- the binding covers the whole partition, so half of it is none.
    """
    unique = tuple(dict.fromkeys(session_ids))
    if not unique:
        return {}
    stored = _stored_partitions(conn, unique)
    # A session with no profile row is MISSING whatever its inputs say, so its
    # projection is not read. Absence is the one status identity settles.
    built = tuple(session_id for session_id in unique if stored[session_id].present)
    current = session_input_bindings(conn, built) if built else {}
    return {
        session_id: _classify_partition(
            stored[session_id],
            current.get(session_id),
            materializer_version=materializer_version,
        )
        for session_id in unique
    }


async def inspect_session_profiles_async(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
    *,
    materializer_version: int,
) -> Mapping[str, str]:
    """:func:`inspect_session_profiles` over an async connection.

    Same SQL, same classification. Only the cursor loop differs, because the
    two connection types have no common one.
    """
    unique = tuple(dict.fromkeys(session_ids))
    if not unique:
        return {}
    sql = _STORED_PARTITION_SQL.format(placeholders=",".join("?" * len(unique)))
    stored: dict[str, _StoredPartition] = {}
    async with conn.execute(sql, unique) as cursor:
        async for row in cursor:
            stored[str(row[0])] = _partition_row(row)
    built = tuple(session_id for session_id in unique if session_id in stored)
    current = await session_input_bindings_async(conn, built) if built else {}
    return {
        session_id: _classify_partition(
            stored.get(session_id, _ABSENT_PARTITION),
            current.get(session_id),
            materializer_version=materializer_version,
        )
        for session_id in unique
    }


#: Sessions inspected per round trip. Bounds the placeholder list and the
#: projection working set; it is not a limit on how much of the archive is
#: inspected, which is always all of it.
SESSION_PARTITION_INSPECT_CHUNK = 500

#: Archive-wide enumeration. Every session is a required key, so the scope is
#: the ``sessions`` relation itself -- no cursor, no queue, no dirty list. A
#: restart that lost every scheduling hint reconstructs this set exactly.
_ARCHIVE_SESSION_IDS_SQL = "SELECT session_id FROM sessions ORDER BY session_id"


def _session_id_page(
    conn: sqlite3.Connection,
    *,
    cursor: str | None,
    limit: int,
) -> tuple[tuple[str, ...], str | None]:
    rows = conn.execute(
        "SELECT session_id FROM sessions WHERE session_id > COALESCE(?, '') ORDER BY session_id LIMIT ?",
        (cursor, limit + 1),
    ).fetchall()
    keys = tuple(str(row[0]) for row in rows[:limit])
    return keys, (keys[-1] if len(rows) > limit and keys else None)


def _excess_page(
    conn: sqlite3.Connection,
    *,
    cursor: str | None,
    limit: int,
) -> tuple[tuple[str, ...], str | None]:
    rows = conn.execute(
        """
        SELECT sp.session_id
        FROM session_profiles AS sp
        LEFT JOIN sessions AS s ON s.session_id = sp.session_id
        WHERE s.session_id IS NULL AND sp.session_id > COALESCE(?, '')
        ORDER BY sp.session_id
        LIMIT ?
        """,
        (cursor, limit + 1),
    ).fetchall()
    keys = tuple(str(row[0]) for row in rows[:limit])
    return keys, (keys[-1] if len(rows) > limit and keys else None)


def _chunked(values: Sequence[str], size: int) -> Iterator[tuple[str, ...]]:
    for start in range(0, len(values), size):
        yield tuple(values[start : start + size])


def archive_session_partition_statuses(
    conn: sqlite3.Connection,
    *,
    materializer_version: int,
    chunk_size: int = SESSION_PARTITION_INSPECT_CHUNK,
) -> dict[str, str]:
    """Every session partition's status, by the same inspection as one batch.

    Archive-wide inspection costs a pass over the message projection, which is
    what an authoritative answer costs. Narrowing the candidate set by an
    identity prefilter first would make it cheap and wrong: identity does not
    move when a role, a model name, or a token count does, so a prefiltered
    pass never reaches the sessions whose output actually changed.
    """
    session_ids = [str(row[0]) for row in conn.execute(_ARCHIVE_SESSION_IDS_SQL).fetchall()]
    statuses: dict[str, str] = {}
    for chunk in _chunked(session_ids, max(1, chunk_size)):
        statuses.update(inspect_session_profiles(conn, chunk, materializer_version=materializer_version))
    return statuses


async def archive_session_partition_statuses_async(
    conn: aiosqlite.Connection,
    *,
    materializer_version: int,
    chunk_size: int = SESSION_PARTITION_INSPECT_CHUNK,
) -> dict[str, str]:
    """:func:`archive_session_partition_statuses` over an async connection."""
    async with conn.execute(_ARCHIVE_SESSION_IDS_SQL) as cursor:
        session_ids = [str(row[0]) async for row in cursor]
    statuses: dict[str, str] = {}
    for chunk in _chunked(session_ids, max(1, chunk_size)):
        statuses.update(await inspect_session_profiles_async(conn, chunk, materializer_version=materializer_version))
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

    if conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone() is None:
        # An excess key: the session is gone, so the correct output is no rows.
        # Rebuilding would leave the orphan in place and inspection would report
        # it excess on every pass, which is a livelock rather than convergence.
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("DELETE FROM session_profiles WHERE session_id = ?", (session_id,))
        except BaseException:
            conn.rollback()
            raise
        conn.commit()
        return True

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


def publish_prepared_session_profile(
    conn: sqlite3.Connection,
    prepared: object,
    *,
    generation_is_current: Callable[[], bool] | None = None,
) -> bool:
    """Refresh usage then atomically publish a lease-free prepared partition.

    Usage refresh precedes the exact-value check.  This preserves the #4855
    order (message evidence, reconciliation, provider evidence, repricing)
    while refusing a bundle prepared from the previous rollup.  A later pass
    reads that canonical rollup outside the writer and can publish it whole.
    """
    from polylogue.storage.derived.session.rebuild import (
        PreparedSessionInsightPartition,
        _refresh_provider_usage_rollup,
        publish_prepared_session_insight_partition,
    )

    if not isinstance(prepared, PreparedSessionInsightPartition):
        raise TypeError(f"expected PreparedSessionInsightPartition, got {type(prepared).__name__}")
    if generation_is_current is not None and not generation_is_current():
        return False
    if prepared.bundle is not None:
        _refresh_provider_usage_rollup(conn, prepared.session_id)
        # The canonical usage rollup is independently derived from persisted
        # evidence.  Commit it before the prepared partition transaction: a
        # changed rollup must survive a binding refusal so the next lease-free
        # preparation reads the exact values that publication will verify.
        conn.commit()
    if generation_is_current is not None and not generation_is_current():
        return False
    return publish_prepared_session_insight_partition(conn, prepared)


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
        session_scope: Callable[[object], Sequence[str] | None],
        page_size: int = 200,
        quiet_keys: Callable[[object], frozenset[str]] | None = None,
        quiet_key: Callable[[object, str], bool] | None = None,
        marker_read_connection: Callable[[], sqlite3.Connection] | None = None,
        marker_write_connection: Callable[[], sqlite3.Connection] | None = None,
        generation_binding: Callable[[], str] | None = None,
    ) -> None:
        self._read_connection = read_connection
        self._write_connection = write_connection
        self._materializer_version = materializer_version
        self._session_scope = session_scope
        self._page_size = page_size
        self._quiet_keys = quiet_keys
        self._quiet_key = quiet_key
        self._marker_read_connection = marker_read_connection
        self._marker_write_connection = marker_write_connection
        self._generation_binding = generation_binding

    def required_page(self, frame: object, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], str | None]:
        """Keyset-page archive work; bounded incremental scopes stay bounded too."""
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

    def inspect(self, frame: object, keys: Sequence[str]) -> Mapping[str, str]:
        conn = self._read_connection()
        try:
            statuses = dict(inspect_session_profiles(conn, keys, materializer_version=self._materializer_version))
            if self._marker_read_connection is None:
                return statuses
            marker_ids_by_session = {
                key: _marker_assertion_ids(conn, key) for key, status in statuses.items() if status == _VALID
            }
        finally:
            conn.close()
        if not marker_ids_by_session:
            return statuses
        marker_conn = self._marker_read_connection()
        try:
            for key, assertion_ids in marker_ids_by_session.items():
                if not _marker_assertions_present(marker_conn, assertion_ids):
                    statuses[key] = _STALE
        finally:
            marker_conn.close()
        return statuses

    def excess_page(self, frame: object, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], str | None]:
        conn = self._read_connection()
        try:
            return _excess_page(conn, cursor=cursor, limit=limit)
        finally:
            conn.close()

    def quiet(self, frame: object, key: str) -> bool:
        if self._quiet_key is not None:
            return self._quiet_key(frame, key)
        return key in self._quiet_keys(frame) if self._quiet_keys is not None else False

    def prerequisite_keys(self, frame: object, key: str) -> tuple[()]:
        """Session profiles have no derivation-kernel prerequisite domain."""
        del frame, key
        return ()

    def compute(self, frame: object, key: str) -> SessionProfileReplacement:
        """Prepare the complete replacement from a lease-free read frame."""
        from polylogue.storage.derived.session.rebuild import prepare_session_insight_partition

        generation = self._generation_binding() if self._generation_binding is not None else None
        expected_generation = f"index-generation:{generation}" if generation is not None else None
        source_revision = getattr(frame, "source_revision", None)
        if (
            expected_generation is not None
            and isinstance(source_revision, str)
            and source_revision.startswith("index-generation:")
            and source_revision != expected_generation
        ):
            raise RuntimeError("session profile frame names a retired index generation")
        conn = self._read_connection()
        try:
            conn.row_factory = sqlite3.Row
            # One read transaction pins every session/message/attachment/event
            # query in this preparation to the same observed generation.
            conn.execute("BEGIN")
            prepared = prepare_session_insight_partition(conn, key)
        finally:
            conn.close()
        if generation is not None and self._generation_binding is not None and self._generation_binding() != generation:
            raise RuntimeError("active index generation changed while session profile was prepared")
        return SessionProfileReplacement(
            key=key,
            input_binding=prepared.input_binding,
            payload=prepared,
            generation_binding=generation,
        )

    def publish(self, frame: object, replacement: object) -> bool:
        """Typed ``object`` because the kernel's protocol admits any replacement.

        Narrowing it to this domain's own type would make the adapter fail the
        contract by contravariance -- a mismatch only a type check catches,
        since at runtime the kernel hands back exactly what ``compute`` made.
        """
        assert isinstance(replacement, SessionProfileReplacement)
        with write_lease(f"derivation.{self.domain}", max_hold_seconds=_PUBLISH_HOLD_BUDGET_S):
            if (
                replacement.generation_binding is not None
                and self._generation_binding is not None
                and self._generation_binding() != replacement.generation_binding
            ):
                return False
            conn = self._write_connection()
            try:
                if (
                    replacement.generation_binding is not None
                    and _connection_generation(conn) != replacement.generation_binding
                ):
                    return False
                if (
                    inspect_session_profiles(
                        conn,
                        (replacement.key,),
                        materializer_version=self._materializer_version,
                    )[replacement.key]
                    == _VALID
                ):
                    published = True
                else:
                    published = publish_prepared_session_profile(
                        conn,
                        replacement.payload,
                        generation_is_current=(
                            None
                            if replacement.generation_binding is None or self._generation_binding is None
                            else lambda: self._generation_binding() == replacement.generation_binding
                        ),
                    )
            finally:
                conn.close()
            if published and self._marker_write_connection is not None:
                _lower_prepared_markers(self._marker_write_connection, replacement.payload)
            return published


@dataclass(frozen=True, slots=True)
class SessionProfileReplacement:
    """The kernel's Replacement shape, built without importing the daemon ring."""

    key: str
    input_binding: str
    payload: object
    generation_binding: str | None = None
    empty: bool = False
