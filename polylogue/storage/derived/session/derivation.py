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
from polylogue.storage.derived.session.summary import SESSION_SUMMARY_DOMAIN
from polylogue.storage.derived.session.usage_rollup import (
    SESSION_USAGE_ROLLUP_DOMAIN,
    inspect_session_usage_rollups,
    session_usage_rollup_recipe_version,
)
from polylogue.storage.sqlite.write_lease import write_lease

__all__ = [
    "SESSION_PARTITION_INSPECT_CHUNK",
    "SESSION_PROFILE_DOMAIN",
    "SessionProfilePartFacts",
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
    latency_rows: int = 0


_ABSENT_PARTITION = _StoredPartition(present=False, materializer_version=None, input_binding=None)


@dataclass(frozen=True, slots=True)
class SessionProfilePartFacts:
    """One exact session-family observation for a sealed owner target.

    This stays storage-owned because it is a direct read of the index and user
    tiers. The daemon owner translates it into its transport-neutral receipt
    types without giving a maintenance surface a connection or a publisher.
    """

    session_present: bool
    status: str
    input_binding: str | None
    output_binding: str | None
    profiles: int


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


def _profile_demand_revision(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        "SELECT revision FROM session_profile_demand WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return 0 if row is None else _count(row[0])


def _count(value: object) -> int:
    return int(value) if isinstance(value, int | float | str) else 0


def _partition_row(row: Sequence[object]) -> _StoredPartition:
    return _StoredPartition(
        present=True,
        materializer_version=None if row[1] is None else _count(row[1]),
        input_binding=None if row[2] is None else str(row[2]),
        latency_rows=_count(row[3]),
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
    if stored.latency_rows != 1:
        return _STALE
    return _VALID


def _classify_partition_with_demand(
    stored: _StoredPartition,
    current_binding: str | None,
    *,
    materializer_version: int,
    demanded: bool,
) -> str:
    status = _classify_partition(
        stored,
        current_binding,
        materializer_version=materializer_version,
    )
    return _STALE if demanded and status == _VALID else status


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

    The partition is the family: the profile row plus the latency profile
    written in the same replacement. It is valid only when
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
    placeholders = ",".join("?" * len(unique))
    demanded = {
        str(row[0])
        for row in conn.execute(
            f"SELECT session_id FROM session_profile_demand WHERE session_id IN ({placeholders})",
            unique,
        ).fetchall()
    }
    # A session with no profile row is MISSING whatever its inputs say, so its
    # projection is not read. Absence is the one status identity settles.
    built = tuple(session_id for session_id in unique if stored[session_id].present)
    current = session_input_bindings(conn, built) if built else {}
    return {
        session_id: _classify_partition_with_demand(
            stored[session_id],
            current.get(session_id),
            materializer_version=materializer_version,
            demanded=session_id in demanded,
        )
        for session_id in unique
    }


def bound_session_profile_partitions(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
    *,
    materializer_version: int,
) -> Mapping[str, str]:
    """Classify partitions from the stored binding alone, reading no input row.

    :func:`inspect_session_profiles` establishes "the binding disagrees" by
    re-digesting the session's whole message, attachment, event and usage
    projection, which is correct and is why the kernel uses it on the page it
    is about to derive.  Doing that for every session in the archive is what
    made archive-wide readiness cost seconds on a real archive
    (polylogue-crwl6), and readiness has a cheaper source for the same fact:
    the index tier clears ``session_profiles.input_content_hash`` from a
    trigger on every write to a relation the binding digests
    (``session_profile_binding_*`` in ``archive_tiers/index.py``).  So for a
    row that still carries a binding, the digest recomputed now is the digest
    stored -- the database has already ruled out every way it could differ.

    Absence is unchanged and is the whole safety margin: a partition with no
    binding, a superseded materializer, or a missing sibling row is classified
    exactly as the recomputing route classifies it, through the same
    :func:`_classify_partition`.  This route can only ever agree with that one
    or refuse to certify; it has no way to invent a valid verdict.
    """
    unique = tuple(dict.fromkeys(session_ids))
    if not unique:
        return {}
    stored = _stored_partitions(conn, unique)
    placeholders = ",".join("?" * len(unique))
    demanded = {
        str(row[0])
        for row in conn.execute(
            f"SELECT session_id FROM session_profile_demand WHERE session_id IN ({placeholders})",
            unique,
        ).fetchall()
    }
    return {
        session_id: _classify_partition_with_demand(
            stored[session_id],
            stored[session_id].input_binding,
            materializer_version=materializer_version,
            demanded=session_id in demanded,
        )
        for session_id in unique
    }


async def bound_session_profile_partitions_async(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
    *,
    materializer_version: int,
) -> Mapping[str, str]:
    """:func:`bound_session_profile_partitions` over an async connection."""
    unique = tuple(dict.fromkeys(session_ids))
    if not unique:
        return {}
    sql = _STORED_PARTITION_SQL.format(placeholders=",".join("?" * len(unique)))
    stored: dict[str, _StoredPartition] = {}
    async with conn.execute(sql, unique) as cursor:
        async for row in cursor:
            stored[str(row[0])] = _partition_row(row)
    placeholders = ",".join("?" * len(unique))
    async with conn.execute(
        f"SELECT session_id FROM session_profile_demand WHERE session_id IN ({placeholders})",
        unique,
    ) as cursor:
        demanded = {str(row[0]) async for row in cursor}
    return {
        session_id: _classify_partition_with_demand(
            stored.get(session_id, _ABSENT_PARTITION),
            stored.get(session_id, _ABSENT_PARTITION).input_binding,
            materializer_version=materializer_version,
            demanded=session_id in demanded,
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
    placeholders = ",".join("?" * len(unique))
    async with conn.execute(
        f"SELECT session_id FROM session_profile_demand WHERE session_id IN ({placeholders})",
        unique,
    ) as cursor:
        demanded = {str(row[0]) async for row in cursor}
    return {
        session_id: _classify_partition_with_demand(
            stored.get(session_id, _ABSENT_PARTITION),
            current.get(session_id),
            materializer_version=materializer_version,
            demanded=session_id in demanded,
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
        """
        SELECT session_id FROM session_profile_demand
        WHERE session_id > COALESCE(?, '')
        ORDER BY session_id LIMIT ?
        """,
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
        SELECT d.session_id
        FROM session_profile_demand AS d
        LEFT JOIN sessions AS s ON s.session_id = d.session_id
        WHERE s.session_id IS NULL
          AND d.session_id > COALESCE(?, '')
        ORDER BY d.session_id
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
    """Prepare and atomically replace one profile partition.

    Usage reconciliation belongs to :data:`SESSION_USAGE_ROLLUP_DOMAIN`; this
    adapter refuses until that prerequisite is current. It then prepares the
    complete family and publishes it under one ``BEGIN IMMEDIATE`` transaction,
    rechecking both exact input values and the captured pending-demand revision.
    A refusal commits nothing.
    """
    from polylogue.storage.derived.session.rebuild import (
        prepare_session_insight_partition,
        publish_prepared_session_insight_partition,
    )

    demand_revision = _profile_demand_revision(conn, session_id)
    if conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone() is None:
        del page_size
        return publish_prepared_session_insight_partition(
            conn,
            prepare_session_insight_partition(conn, session_id),
            expected_demand_revision=demand_revision,
        )
    if session_input_bindings(conn, (session_id,)).get(session_id, "") != input_binding:
        return False

    if (
        inspect_session_usage_rollups(
            conn,
            (session_id,),
            recipe_version=session_usage_rollup_recipe_version(),
        )[session_id]
        != _VALID
    ):
        # The prerequisite owns this write. Refusing here costs one pass and
        # commits nothing; reconciling here would be the hidden usage write
        # this route exists without.
        return False

    del page_size  # the prepared replacement is already one bounded session
    prepared = prepare_session_insight_partition(conn, session_id)
    if prepared.input_binding != input_binding:
        return False
    return publish_prepared_session_insight_partition(
        conn,
        prepared,
        expected_demand_revision=demand_revision,
    )


def publish_prepared_session_profile(
    conn: sqlite3.Connection,
    prepared: object,
    *,
    expected_demand_revision: int = 0,
    generation_is_current: Callable[[], bool] | None = None,
) -> bool:
    """Atomically publish a lease-free prepared partition. Nothing else.

    This publisher owns exactly one transaction and commits exactly one thing:
    the prepared four-table family. It does **not** reconcile canonical usage.

    It used to. The usage reconciliation ran here and was committed
    before the prepared bundle's exact-value check -- so publication moved an
    input the prepared bundle had already read, and the check that followed
    necessarily failed for every session whose rollup had drifted. The first
    computation was doomed by construction and a second pass did the real
    work, while ``False`` was returned from a call that had already committed
    a usage change.

    :data:`SESSION_USAGE_ROLLUP_DOMAIN` now owns that reconciliation and the
    kernel converges it first (it is a declared prerequisite key below), so a
    bundle arrives here already prepared from the settled rollup.
    ``session_insight_compute_binding`` remains the exact-content backstop:
    if the rollup moved anyway, this refuses without side effects.
    """
    from polylogue.storage.derived.session.rebuild import (
        PreparedSessionInsightPartition,
        publish_prepared_session_insight_partition,
    )

    if not isinstance(prepared, PreparedSessionInsightPartition):
        raise TypeError(f"expected PreparedSessionInsightPartition, got {type(prepared).__name__}")
    if generation_is_current is not None and not generation_is_current():
        return False
    return publish_prepared_session_insight_partition(
        conn,
        prepared,
        expected_demand_revision=expected_demand_revision,
    )


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
    # ``prepare_session_insight_partition`` reads the materialized counters
    # from ``sessions`` for the bounded profile projection, so profile
    # publication must wait for this session's authoritative counter part.
    # It also reads the canonical ``session_model_usage`` rollup, which is
    # itself derived: reconciling it here would move an input this partition
    # had already read. The rollup is a separate domain converged first.
    prerequisites = (SESSION_SUMMARY_DOMAIN, SESSION_USAGE_ROLLUP_DOMAIN)
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
        generation_binding: Callable[[], str] | None = None,
    ) -> None:
        self._read_connection = read_connection
        self._write_connection = write_connection
        self._materializer_version = materializer_version
        self._session_scope = session_scope
        self._page_size = page_size
        self._quiet_keys = quiet_keys
        self._quiet_key = quiet_key
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
        """Classify each session from the index relations this domain owns.

        polylogue-ylh7v: an absent user-tier marker assertion no longer
        downgrades a valid index family. Marker delivery is its own domain
        (``storage/derived/session/marker_domain.py``) with its own required,
        inspect, compute and publish steps, so a user-tier outage leaves marker
        work pending instead of re-deriving profiles that were never wrong.
        """
        del frame
        conn = self._read_connection()
        try:
            # Output rows and input values must come from one commit; mixing
            # snapshots can certify a never-valid family.
            conn.execute("BEGIN")
            return dict(inspect_session_profiles(conn, keys, materializer_version=self._materializer_version))
        finally:
            conn.close()

    def selected_part_facts(self, frame: object, session_id: str) -> SessionProfilePartFacts:
        """Read the exact family facts a sealed owner target may certify.

        The ordinary adapter contract intentionally exposes statuses only.
        Maintenance needs neither discovery nor a connection escape hatch, but
        it must retain the current source binding, stored binding, and actual
        partition counts beside its one-key receipt. Keep that inspection in
        the storage adapter so it uses the same validity definition recurring
        convergence does -- which, since polylogue-ylh7v, is the index family
        alone.
        """
        conn = self._read_connection()
        try:
            # A read transaction freezes every field in this receipt to one
            # index snapshot.  Without it a concurrent replacement can mix a
            # profile from one commit with sibling counts from another.
            conn.execute("BEGIN")
            session_present = (
                conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone() is not None
            )
            stored = _stored_partitions(conn, (session_id,))[session_id]
            input_binding = session_input_bindings(conn, (session_id,)).get(session_id) if session_present else None
            # Do not infer sibling cardinalities from ``session_profiles``.
            # A sealed excess receipt must report the actual family even when
            # a partial historical/corrupt relation has no profile parent.
            # The publisher then gets a chance to retire that exact key rather
            # than falsely certifying the part absent.
            profiles = _count(
                conn.execute("SELECT COUNT(*) FROM session_profiles WHERE session_id = ?", (session_id,)).fetchone()[0]
            )
            status = _classify_partition(
                stored,
                input_binding,
                materializer_version=self._materializer_version,
            )
        finally:
            conn.close()
        return SessionProfilePartFacts(
            session_present=session_present,
            status=status,
            input_binding=input_binding,
            output_binding=stored.input_binding,
            profiles=profiles,
        )

    def selected_frame_is_current(self, frame: object) -> bool:
        """Whether a sealed frame still names this adapter's active generation."""
        if self._generation_binding is None:
            return True
        generation = self._generation_binding()
        return getattr(frame, "source_revision", None) == f"index-generation:{generation}"

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

    def prerequisite_keys(self, frame: object, key: str) -> tuple[tuple[str, str], ...]:
        """The exact upstream keys this session's profile reads.

        Its materialized summary counters, and its reconciled canonical usage
        rollup. Naming the rollup key here -- not just the domain -- is what
        makes the reconciliation happen *before* preparation for this session
        and only for this session.
        """
        del frame
        conn = self._read_connection()
        try:
            # Retiring an orphan consumes no session counters. Requiring the
            # deleted upstream row would prevent excess cleanup after restart.
            exists = conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (key,)).fetchone()
            if not exists:
                return ()
            return ((SESSION_SUMMARY_DOMAIN, key), (SESSION_USAGE_ROLLUP_DOMAIN, key))
        finally:
            conn.close()

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
            demand_revision = _profile_demand_revision(conn, key)
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
            demand_revision=demand_revision,
        )

    def publish(self, frame: object, replacement: object) -> bool:
        """Typed ``object`` because the kernel's protocol admits any replacement.

        Narrowing it to this domain's own type would make the adapter fail the
        contract by contravariance -- a mismatch only a type check catches,
        since at runtime the kernel hands back exactly what ``compute`` made.
        """
        assert isinstance(replacement, SessionProfileReplacement)
        generation_binding = self._generation_binding
        with write_lease(f"derivation.{self.domain}", max_hold_seconds=_PUBLISH_HOLD_BUDGET_S):
            if (
                replacement.generation_binding is not None
                and generation_binding is not None
                and generation_binding() != replacement.generation_binding
            ):
                return False
            conn = self._write_connection()
            try:
                if (
                    replacement.generation_binding is not None
                    and _connection_generation(conn) != replacement.generation_binding
                ):
                    return False
                if _profile_demand_revision(conn, replacement.key) != replacement.demand_revision:
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
                        expected_demand_revision=replacement.demand_revision,
                        generation_is_current=(
                            None
                            if replacement.generation_binding is None or generation_binding is None
                            else lambda: generation_binding() == replacement.generation_binding
                        ),
                    )
            finally:
                conn.close()
            # polylogue-ylh7v: no second, non-atomic user-tier transaction
            # runs behind this commit. Marker delivery is its own domain, so
            # this publication now has exactly one commit and exactly one
            # outcome.
            return published


@dataclass(frozen=True, slots=True)
class SessionProfileReplacement:
    """The kernel's Replacement shape, built without importing the daemon ring."""

    key: str
    input_binding: str
    payload: object
    generation_binding: str | None = None
    demand_revision: int = 0
    empty: bool = False
