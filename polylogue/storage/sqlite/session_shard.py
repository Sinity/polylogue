"""Session rows as a sealed SQLite file prepared off the writer (polylogue-bp12n.6).

Stage-A preparation inserts one session's message and block row tuples into a
private, index-free SQLite file and seals it. The writer never attaches or
copies the file: ``prepared_session_rows_from_shard`` exposes a sealed range
as ``PreparedSessionRows`` whose row sequences stream from the file, and the
writer validates and binds that carrier like any other prepared write. The
module writes only these private files, never an archive tier.

The shard is a transport, never authority. It holds exactly the rows
``prepare_session_rows`` would have produced, its tables carry no affinity so
every value round-trips with the type the builder bound, and the writer
re-derives nothing from it. A partial shard is not a shard: every row and the
seal are written in one transaction and the seal goes in last, so a builder
that dies mid-write leaves a file :func:`open_session_shard` refuses. There is
no shard state between "absent" and "complete".
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterator, Mapping, Sequence, Set
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload
from urllib.parse import quote

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.iterator_lifetime import settled_iterator
from polylogue.core.sql_settlement import current_native_sql_lifetimes
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers import archive_tiers_specs
from polylogue.storage.sqlite.archive_tiers.column_spec import ColumnSpec, TableColumnSpec

#: Bump when the shard's own layout changes shape; it is part of the seal.
SHARD_LAYOUT_VERSION = 3

#: The tables a shard transports, in the order the writer must copy them:
#: ``blocks.message_id`` references ``messages.message_id``.
SHARD_TABLES: tuple[str, ...] = ("messages", "blocks")

if TYPE_CHECKING:
    from polylogue.pipeline.ids import MessageOwnerResolution

_T = TypeVar("_T")

_OWNER_DDL = (
    "CREATE TABLE shard_owner_manifest (session_ordinal INTEGER PRIMARY KEY, owner_count INTEGER NOT NULL)",
    "CREATE TABLE shard_owner_key (session_ordinal INTEGER NOT NULL, ordinal INTEGER NOT NULL, owner_key TEXT NOT NULL, PRIMARY KEY(session_ordinal, ordinal)) WITHOUT ROWID",
    "CREATE TABLE shard_owner_lookup (session_ordinal INTEGER NOT NULL, kind TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(session_ordinal, kind, key)) WITHOUT ROWID",
    "CREATE TABLE shard_owner_ambiguity (session_ordinal INTEGER NOT NULL, kind TEXT NOT NULL, key TEXT NOT NULL, PRIMARY KEY(session_ordinal, kind, key)) WITHOUT ROWID",
)


class ShardRefusedError(Exception):
    """A shard file cannot be trusted and must not be attached."""


def _spec(table: str) -> TableColumnSpec:
    return archive_tiers_specs.TABLE_SPECS[table]


def _bound_columns(spec: TableColumnSpec) -> tuple[ColumnSpec, ...]:
    """The writable columns a row tuple actually carries a value for.

    ``TableColumnSpec.extract_tuple`` skips any column whose INSERT
    placeholder is a literal or expression rather than ``?``, so the shard
    stores exactly the same subset in exactly the same order.
    """
    return tuple(col for col in spec.writable_columns if col.extract_placeholder == "?")


def shard_table_ddl(table: str) -> str:
    """The shard's declaration of ``table``: no types, no constraints, no indexes.

    Omitting the type names is not laziness. A declared type gives the column
    an affinity, and affinity silently converts a value on the way in -- a
    ``native_id`` of ``"0042"`` stored in an INTEGER-affinity column comes
    back as ``42``. With no declared type the column has no affinity and
    every value round-trips as the builder bound it, which is what "the rows
    are the same rows" requires against a STRICT destination.
    """
    names = ", ".join(col.name for col in _bound_columns(_spec(table)))
    return f"CREATE TABLE {table} ({names})"


def shard_column_signature() -> str:
    """Digest of the column set every shard table carries.

    A shard built before a column was added or reordered projects the wrong
    values into the archive. The seal records this digest and the writer
    compares it, so a stale shard is refused rather than mis-copied.
    """
    payload = "\n".join((*[f"{table}:{shard_table_ddl(table)}" for table in SHARD_TABLES], *_OWNER_DDL))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_SEAL_DDL = """
CREATE TABLE shard_seal (
    layout_version INTEGER NOT NULL,
    column_signature TEXT NOT NULL,
    session_count INTEGER NOT NULL
)
"""

_SESSION_DDL = """
CREATE TABLE shard_session (
    session_id TEXT NOT NULL,
    content_hash BLOB NOT NULL,
    message_lo INTEGER NOT NULL,
    message_hi INTEGER NOT NULL,
    block_lo INTEGER NOT NULL,
    block_hi INTEGER NOT NULL
)
"""


@dataclass(frozen=True, slots=True)
class ShardSessionRows:
    """Where one session's rows live inside a shard.

    The ranges are rowid bounds, not a ``session_id`` predicate: the builder
    appends each session's rows contiguously into a table it alone writes and
    never deletes from, so rowids are dense and ascending. A range scan over
    the rowid btree costs the same as the index the shard deliberately does
    not carry.
    """

    session_id: str
    session_content_hash: bytes
    message_lo: int
    message_hi: int
    block_lo: int
    block_hi: int
    #: Content-derived fallback identities carried by the prepared rows.  The
    #: writer validates and reuses these; it never regenerates them.
    content_identities: Sequence[tuple[str, int]]
    owner_resolution: MessageOwnerResolution

    @property
    def message_row_count(self) -> int:
        return max(0, self.message_hi - self.message_lo + 1)

    @property
    def block_row_count(self) -> int:
        return max(0, self.block_hi - self.block_lo + 1)


@dataclass(frozen=True, slots=True)
class SessionShard:
    """A sealed shard file and the sessions it carries."""

    path: Path
    sessions: Sequence[ShardSessionRows]

    def by_session_id(self) -> Mapping[str, ShardSessionRows]:
        return ShardSessionMapping(self.path, len(self.sessions))


#: A read-only shard connection held open for one publication window, keyed
#: by the shard path. Context-local, so it never crosses threads or tasks.
_READER_WINDOW: ContextVar[tuple[Path, sqlite3.Connection] | None] = ContextVar(
    "polylogue_shard_reader_window", default=None
)


@contextmanager
def shard_owner_reader_window(resolution: object) -> Iterator[None]:
    """Reuse one read-only connection for a sealed owner resolution's lookups.

    A session write asks the shard-backed owner resolution once per message;
    opening a connection per question dominated publication of large
    sessions. Outside the window each lookup still opens its own reader.
    """
    keys = getattr(resolution, "keys", None)
    if not isinstance(keys, _ShardOwnerKeys) or _READER_WINDOW.get() is not None:
        yield
        return
    with _shard_connection(keys.path) as conn:
        token = _READER_WINDOW.set((keys.path, conn))
        try:
            yield
        finally:
            _READER_WINDOW.reset(token)


@contextmanager
def _shard_connection(path: Path, *, readonly: bool = True) -> Iterator[sqlite3.Connection]:
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner

    window = _READER_WINDOW.get() if readonly else None
    if window is not None and window[0] == path:
        yield window[1]
        return
    connection = connect_measured(_read_only_uri(path), uri=True) if readonly else connect_measured(path)
    owner = NativeSQLCustodyOwner(connection, lifetime_dependencies=current_native_sql_lifetimes())
    try:
        yield owner.require_connection()
    except BaseException as primary:
        from polylogue.storage.sqlite.connection_profile import _close_failed_native_construction

        _close_failed_native_construction(owner, primary)
        raise
    else:
        owner.close()


class SessionShardBuilder:
    """Writes one shard file. Not thread-safe: one builder per worker.

    Everything happens in a single transaction that commits in :meth:`seal`,
    so the file is either absent, empty, or complete -- see the module
    docstring.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn = connect_measured(path, isolation_level=None)
        self._sql_closed = False
        self._discard_on_close = True
        from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner

        NativeSQLCustodyOwner(self._conn, terminal_parent=self, lifetime_dependencies=current_native_sql_lifetimes())
        try:
            # A shard is scratch: it is read once, by one process, and deleted.
            # Its durability is the source file it was parsed from, so paying for
            # synchronous writes here would buy nothing the re-parse does not.
            self._conn.execute("PRAGMA synchronous = OFF")
            self._conn.execute("PRAGMA journal_mode = DELETE")
            for table in SHARD_TABLES:
                self._conn.execute(shard_table_ddl(table))
            self._conn.execute(_SEAL_DDL)
            self._conn.execute(_SESSION_DDL)
            for statement in _OWNER_DDL:
                self._conn.execute(statement)
            self._conn.execute("CREATE INDEX shard_session_id ON shard_session(session_id)")
            self._next_rowid = dict.fromkeys(SHARD_TABLES, 1)
            self._session_count = 0
            self._conn.execute("BEGIN IMMEDIATE")
        except BaseException as primary:
            try:
                self.abandon()
            except BaseException as cleanup:
                raise cleanup from primary
            raise

    def capture_owner_resolution(self, resolution: MessageOwnerResolution) -> None:
        """Seal the sole resolver's evidence beside this session's row tuples."""
        session = self._session_count + 1
        self._conn.execute("INSERT INTO shard_owner_manifest VALUES (?, ?)", (session, len(resolution.keys)))
        # Streamed batches: one statement per table, one row per value.
        self._conn.executemany(
            "INSERT INTO shard_owner_key VALUES (?, ?, ?)",
            ((session, ordinal, owner_key) for ordinal, owner_key in enumerate(resolution.keys)),
        )
        for kind, lookup in (
            ("physical", resolution.by_physical_coordinate),
            ("stable", resolution.by_stable_key),
            ("provider", resolution.unique_provider_keys),
        ):
            self._conn.executemany(
                "INSERT INTO shard_owner_lookup VALUES (?, ?, ?, ?)",
                (
                    (session, kind, json.dumps(lookup_key, separators=(",", ":")), owner_key)
                    for lookup_key, owner_key in lookup.items()
                ),
            )
        for kind, ambiguous in (
            ("physical", resolution.ambiguous_physical_coordinates),
            ("stable", resolution.ambiguous_stable_keys),
            ("key", resolution.ambiguous_keys),
            ("provider", resolution.ambiguous_provider_ids),
        ):
            self._conn.executemany(
                "INSERT INTO shard_owner_ambiguity VALUES (?, ?, ?)",
                ((session, kind, json.dumps(ambiguous_key, separators=(",", ":"))) for ambiguous_key in ambiguous),
            )

    def add(self, prepared: object) -> None:
        """Append one session's prepared rows.

        ``prepared`` is a ``PreparedSessionRows``; it is typed loosely here so
        this module stays importable from the writer without a cycle.
        """
        message_rows: Sequence[tuple[object, ...]] = prepared.message_rows  # type: ignore[attr-defined]
        block_rows: Sequence[tuple[object, ...]] = prepared.block_rows  # type: ignore[attr-defined]
        content_identities: Sequence[tuple[str, int]] = prepared.content_identities  # type: ignore[attr-defined]
        if len(content_identities) != len(message_rows):
            raise ShardRefusedError("prepared identity carrier does not cover every message row")
        self.capture_owner_resolution(prepared.owner_resolution)  # type: ignore[attr-defined]
        message_lo = self._append("messages", message_rows)
        block_lo = self._append("blocks", block_rows)
        self._append_manifest(
            prepared.session_id,  # type: ignore[attr-defined]
            prepared.session_content_hash,  # type: ignore[attr-defined]
            message_lo,
            message_lo + len(message_rows) - 1,
            block_lo,
            block_lo + len(block_rows) - 1,
        )

    def add_streamed(
        self,
        *,
        session_id: str,
        session_content_hash: bytes,
        message_rows: Iterator[tuple[object, ...]],
        block_rows: Iterator[tuple[object, ...]],
        owner_resolution: MessageOwnerResolution,
    ) -> None:
        """Copy bounded row windows for one disk-backed parsed session."""
        self.capture_owner_resolution(owner_resolution)
        message_lo = self._next_rowid["messages"]
        self._append_iter("messages", message_rows)
        block_lo = self._next_rowid["blocks"]
        self._append_iter("blocks", block_rows)
        self._append_manifest(
            session_id,
            session_content_hash,
            message_lo,
            self._next_rowid["messages"] - 1,
            block_lo,
            self._next_rowid["blocks"] - 1,
        )

    def _append_manifest(
        self,
        session_id: str,
        content_hash: bytes,
        message_lo: int,
        message_hi: int,
        block_lo: int,
        block_hi: int,
    ) -> None:
        self._conn.execute(
            "INSERT INTO shard_session VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, content_hash, message_lo, message_hi, block_lo, block_hi),
        )
        self._session_count += 1

    def _append_iter(self, table: str, rows: Iterator[tuple[object, ...]]) -> None:
        width = len(_bound_columns(_spec(table)))
        placeholders = ", ".join("?" * width)
        sql = f"INSERT INTO {table} VALUES ({placeholders})"
        window: list[tuple[object, ...]] = []

        def flush() -> None:
            if window:
                self._conn.executemany(sql, window).close()
                self._next_rowid[table] += len(window)
                window.clear()

        # The streamed rows settle with this append, including on failure.
        with settled_iterator(rows) as original_rows:
            for row in original_rows:
                check_compute_cancelled()
                window.append(row)
                if len(window) == 128:
                    flush()
            flush()

    def _append(self, table: str, rows: Sequence[tuple[object, ...]]) -> int:
        lo = self._next_rowid[table]
        if rows:
            width = len(_bound_columns(_spec(table)))
            placeholders = ", ".join("?" * width)
            self._conn.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
            self._next_rowid[table] = lo + len(rows)
        return lo

    def seal(self) -> SessionShard:
        """Commit the shard and return it. The seal row is the last write."""
        count = int(self._conn.execute("SELECT COUNT(*) FROM shard_owner_manifest").fetchone()[0])
        if count != self._session_count:
            raise ShardRefusedError("each prepared session requires its captured owner resolution")
        for session, expected in self._conn.execute("SELECT session_ordinal, owner_count FROM shard_owner_manifest"):
            count, lo, hi = self._conn.execute(
                "SELECT COUNT(*), MIN(ordinal), MAX(ordinal) FROM shard_owner_key WHERE session_ordinal=?", (session,)
            ).fetchone()
            if count != expected or (expected and (lo != 0 or hi != expected - 1)):
                raise ShardRefusedError("prepared owner resolution is incomplete")
        for table in SHARD_TABLES:
            row = self._conn.execute(f"SELECT COALESCE(MAX(rowid), 0) FROM {table}").fetchone()
            if int(row[0]) != self._next_rowid[table] - 1:
                # Dense ascending rowids are what the range scans assume; a
                # gap means this file cannot address its own sessions.
                self._conn.execute("ROLLBACK")
                self.close()
                raise ShardRefusedError(f"shard {self.path}: {table} rowids are not dense")
        self._conn.execute(
            "INSERT INTO shard_seal VALUES (?, ?, ?)",
            (SHARD_LAYOUT_VERSION, shard_column_signature(), self._session_count),
        )
        self._conn.execute("COMMIT")
        self._discard_on_close = False
        self.close()
        return SessionShard(path=self.path, sessions=ShardSessionSequence(self.path, self._session_count))

    def close(self) -> None:
        from polylogue.storage.sqlite.connection_profile import close_parent_native_connection, retire_native_sql_parent

        if not self._sql_closed:
            close_parent_native_connection(self, self._conn)
            self._sql_closed = True
        if self._discard_on_close:
            discard_session_shard(self.path)
        retire_native_sql_parent(self)

    def abandon(self) -> None:
        """Discard only after the actual builder connection closes."""
        self._discard_on_close = True
        self.close()


def build_session_shard(directory: Path, prepared_sessions: Sequence[object]) -> SessionShard:
    """Build and seal one shard holding ``prepared_sessions``' rows."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"shard-{uuid.uuid4().hex}.db"
    builder = SessionShardBuilder(path)
    try:
        for prepared in prepared_sessions:
            builder.add(prepared)
        return builder.seal()
    except BaseException as primary:
        try:
            builder.abandon()
        except BaseException as cleanup:
            raise cleanup from primary
        raise


#: Identity rows one random-access read fetches. Writers index the sequence
#: in message order, so one read serves the next page of lookups; a lookup
#: outside the held page replaces it, keeping at most one page resident.
_IDENTITY_PAGE_ROWS = 4096


class ShardIdentitySequence(Sequence[tuple[str, int]]):
    """Address identity rows on disk without rebuilding a whole-session tuple."""

    def __init__(self, path: Path, lo: int, hi: int) -> None:
        self.path = path
        self.lo = lo
        self.hi = hi
        # (first index, rows): replaced as one value, so a concurrent reader
        # never pairs one page's start with another page's rows.
        self._window: tuple[int, tuple[tuple[str, int], ...]] = (0, ())

    def __len__(self) -> int:
        return max(0, self.hi - self.lo + 1)

    @overload
    def __getitem__(self, index: int) -> tuple[str, int]: ...

    @overload
    def __getitem__(self, index: slice) -> list[tuple[str, int]]: ...

    def __getitem__(self, index: int | slice) -> tuple[str, int] | list[tuple[str, int]]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        page_start, page = self._window
        offset = index - page_start
        if not 0 <= offset < len(page):
            # One connection per page of lookups, not one per message: the
            # writer resolves every message of a session through here.
            start = index - index % _IDENTITY_PAGE_ROWS
            end = min(start + _IDENTITY_PAGE_ROWS, len(self)) - 1
            with _shard_connection(self.path) as conn:
                page = tuple(
                    (str(identity), int(occurrence))
                    for identity, occurrence in conn.execute(
                        "SELECT content_identity, content_occurrence FROM messages "
                        "WHERE rowid BETWEEN ? AND ? ORDER BY rowid",
                        (self.lo + start, self.lo + end),
                    )
                )
            if len(page) != end - start + 1:
                raise ShardRefusedError("prepared message identity row disappeared")
            self._window = (start, page)
            offset = index - start
        return page[offset]

    def __iter__(self) -> Iterator[tuple[str, int]]:
        for index in range(len(self)):
            yield self[index]


def _read_message_identities(
    conn: sqlite3.Connection,
    *,
    path: Path,
    message_lo: int,
    message_hi: int,
    session_id: str,
) -> Sequence[tuple[str, int]]:
    """Read the identity carrier from the sealed message rows, without hashing."""
    if message_hi < message_lo:
        return ()
    count, invalid = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(CASE WHEN session_id != ? "
        "OR typeof(content_identity) != 'text' OR content_identity = '' "
        "OR typeof(content_occurrence) != 'integer' OR content_occurrence < 0 "
        "THEN 1 ELSE 0 END), 0) "
        "FROM messages WHERE rowid BETWEEN ? AND ?",
        (session_id, message_lo, message_hi),
    ).fetchone()
    if count != message_hi - message_lo + 1:
        raise ShardRefusedError(f"shard {conn}: message identity range is incomplete for {session_id}")
    if invalid:
        raise ShardRefusedError(f"shard {conn}: invalid message identity for {session_id}")
    return ShardIdentitySequence(path, message_lo, message_hi)


def _session_entry(conn: sqlite3.Connection, path: Path, row: tuple[Any, ...]) -> ShardSessionRows:
    session_id = str(row[0])
    message_lo, message_hi = int(row[2]), int(row[3])
    return ShardSessionRows(
        session_id=session_id,
        session_content_hash=bytes(row[1]),
        message_lo=message_lo,
        message_hi=message_hi,
        block_lo=int(row[4]),
        block_hi=int(row[5]),
        owner_resolution=shard_owner_resolution(path, int(row[6])),
        content_identities=_read_message_identities(
            conn, path=path, message_lo=message_lo, message_hi=message_hi, session_id=session_id
        ),
    )


class ShardSessionSequence(Sequence[ShardSessionRows]):
    """Address a sealed session manifest without retaining its Python rows."""

    def __init__(self, path: Path, count: int) -> None:
        self.path = path
        self._count = count

    def __len__(self) -> int:
        return self._count

    @overload
    def __getitem__(self, index: int) -> ShardSessionRows: ...

    @overload
    def __getitem__(self, index: slice) -> list[ShardSessionRows]: ...

    def __getitem__(self, index: int | slice) -> ShardSessionRows | list[ShardSessionRows]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._count))]
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError(index)
        with _shard_connection(self.path) as conn:
            row = conn.execute(
                "SELECT session_id, content_hash, message_lo, message_hi, block_lo, block_hi, rowid "
                "FROM shard_session WHERE rowid = ?",
                (index + 1,),
            ).fetchone()
            if row is None:
                raise ShardRefusedError("sealed shard session row disappeared")
            return _session_entry(conn, self.path, row)

    def __iter__(self) -> Iterator[ShardSessionRows]:
        for start in range(0, self._count, 512):
            with _shard_connection(self.path) as connection:
                page = tuple(
                    _session_entry(connection, self.path, row)
                    for row in connection.execute(
                        "SELECT session_id, content_hash, message_lo, message_hi, block_lo, block_hi, rowid "
                        "FROM shard_session WHERE rowid BETWEEN ? AND ? ORDER BY rowid",
                        (start + 1, min(start + 512, self._count)),
                    )
                )
            yield from page


class ShardSessionMapping(Mapping[str, ShardSessionRows]):
    """Resolve exact session identities using the sealed manifest index."""

    def __init__(self, path: Path, count: int) -> None:
        self.path = path
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[str]:
        after = 0
        while True:
            with _shard_connection(self.path) as connection:
                rows = connection.execute(
                    "SELECT rowid, session_id FROM shard_session WHERE rowid > ? ORDER BY rowid LIMIT 512", (after,)
                ).fetchall()
            if not rows:
                return
            after = int(rows[-1][0])
            for _rowid, session_id in rows:
                yield str(session_id)

    def __getitem__(self, session_id: str) -> ShardSessionRows:
        with _shard_connection(self.path) as conn:
            rows = conn.execute(
                "SELECT session_id, content_hash, message_lo, message_hi, block_lo, block_hi, rowid "
                "FROM shard_session INDEXED BY shard_session_id WHERE session_id = ? LIMIT 2",
                (session_id,),
            ).fetchall()
            if not rows:
                raise KeyError(session_id)
            if len(rows) != 1:
                raise ShardRefusedError(f"shard {self.path}: a session id appears twice in the manifest")
            return _session_entry(conn, self.path, rows[0])


def open_session_shard(path: Path) -> SessionShard:
    """Read a shard's manifest, refusing anything a builder did not seal.

    The open is read-write on purpose: a builder killed mid-transaction
    leaves a hot rollback journal, and only a writable connection may replay
    it. A read-only open would fail with an error indistinguishable from a
    missing file. After the rollback the file holds no seal, which is the
    refusal this function exists to make.
    """
    if not path.exists():
        raise ShardRefusedError(f"shard {path}: file is absent")
    checked_count = 0
    try:
        with _shard_connection(path, readonly=False) as conn:
            seal = conn.execute("SELECT layout_version, column_signature, session_count FROM shard_seal").fetchall()
            if len(seal) != 1:
                raise ShardRefusedError(f"shard {path}: unsealed ({len(seal)} seal rows)")
            layout_version, column_signature, session_count = seal[0]
            if int(layout_version) != SHARD_LAYOUT_VERSION:
                raise ShardRefusedError(f"shard {path}: layout version {layout_version}")
            if column_signature != shard_column_signature():
                raise ShardRefusedError(f"shard {path}: column signature does not match this build")
            actual_count, max_rowid = conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(rowid), 0) FROM shard_session"
            ).fetchone()
            checked_count = int(actual_count)
            if checked_count != int(session_count) or int(max_rowid) != checked_count:
                raise ShardRefusedError(
                    f"shard {path}: seal claims {session_count} sessions, manifest has {checked_count}"
                )
            # The sealed manifest preserves original parser output ordinals.
            # A repeated identity must remain inspectable before the canonical
            # key selector can refuse it. Identity-addressed writer bindings
            # independently require exactly one range in ShardSessionMapping.
            owner_count = int(conn.execute("SELECT COUNT(*) FROM shard_owner_manifest").fetchone()[0])
            if owner_count != session_count:
                raise ShardRefusedError("sealed shard lacks captured session owner evidence")
            for ordinal, expected in conn.execute("SELECT session_ordinal, owner_count FROM shard_owner_manifest"):
                count, lo, hi = conn.execute(
                    "SELECT COUNT(*), MIN(ordinal), MAX(ordinal) FROM shard_owner_key WHERE session_ordinal=?",
                    (ordinal,),
                ).fetchone()
                if (
                    ordinal < 1
                    or ordinal > session_count
                    or expected < 0
                    or count != expected
                    or (expected and (lo != 0 or hi != expected - 1))
                ):
                    raise ShardRefusedError("sealed shard owner evidence is incomplete")
            maxima = {
                table: int(conn.execute(f"SELECT COALESCE(MAX(rowid), 0) FROM {table}").fetchone()[0])
                for table in SHARD_TABLES
            }
            for session_id, _hash, message_lo, message_hi, block_lo, block_hi in conn.execute(
                "SELECT session_id, content_hash, message_lo, message_hi, block_lo, block_hi "
                "FROM shard_session ORDER BY rowid"
            ):
                # Empty sessions use the builder's canonical (1, 0) range;
                # every non-empty range must stay within the rows physically
                # present in the sealed file.  Without this check a damaged
                # manifest could be accepted and silently publish a session
                # missing some of its messages or blocks.
                ranges = (("messages", int(message_lo), int(message_hi)), ("blocks", int(block_lo), int(block_hi)))
                for table, lo, hi in ranges:
                    if hi < lo:
                        if lo != hi + 1 or lo < 1 or lo > maxima[table] + 1:
                            raise ShardRefusedError(f"shard {path}: invalid empty {table} range for {session_id}")
                    elif lo < 1 or hi > maxima[table]:
                        raise ShardRefusedError(f"shard {path}: {table} range is outside sealed rows for {session_id}")
                _read_message_identities(
                    conn,
                    path=path,
                    message_lo=int(message_lo),
                    message_hi=int(message_hi),
                    session_id=str(session_id),
                )
    except sqlite3.DatabaseError as exc:
        raise ShardRefusedError(f"shard {path}: unreadable ({exc})") from exc
    return SessionShard(path=path, sessions=ShardSessionSequence(path, checked_count))


def discard_session_shard(path: Path) -> None:
    """Remove a shard and any journal it left behind."""
    for candidate in (
        path,
        path.with_name(path.name + "-journal"),
        path.with_name(path.name + "-wal"),
        path.with_name(path.name + "-shm"),
    ):
        candidate.unlink(missing_ok=True)


def _read_only_uri(path: Path) -> str:
    """A ``file:`` URI naming ``path`` read-only, safe for any path spelling.

    The path is percent-encoded: an unescaped ``?`` or ``#`` in a directory
    name would otherwise start the URI's query or fragment and attach some
    other file entirely.
    """
    return f"file:{quote(str(path))}?mode=ro"


__all__ = [
    "SHARD_LAYOUT_VERSION",
    "SHARD_TABLES",
    "SessionShard",
    "SessionShardBuilder",
    "ShardRefusedError",
    "ShardSessionRows",
    "build_session_shard",
    "discard_session_shard",
    "open_session_shard",
    "shard_column_signature",
    "shard_table_ddl",
]


class _ShardOwnerKeys(Sequence[str]):
    def __init__(self, path: Path, session: int, count: int) -> None:
        self.path, self.session, self._count = path, session, count

    def __len__(self) -> int:
        return self._count

    @overload
    def __getitem__(self, index: int) -> str: ...
    @overload
    def __getitem__(self, index: slice) -> list[str]: ...
    def __getitem__(self, index: int | slice) -> str | list[str]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._count))]
        ordinal = index + self._count if index < 0 else index
        if not 0 <= ordinal < self._count:
            raise IndexError(index)
        with _shard_connection(self.path) as conn:
            row = conn.execute(
                "SELECT owner_key FROM shard_owner_key WHERE session_ordinal=? AND ordinal=?", (self.session, ordinal)
            ).fetchone()
        if row is None:
            raise ShardRefusedError("sealed message-owner row disappeared")
        return str(row[0])

    def __iter__(self) -> Iterator[str]:
        start = 0
        while start < self._count:
            with _shard_connection(self.path) as conn:
                rows = conn.execute(
                    "SELECT ordinal, owner_key FROM shard_owner_key WHERE session_ordinal=? AND ordinal>=? ORDER BY ordinal LIMIT 512",
                    (self.session, start),
                ).fetchall()
            if not rows or any(int(row[0]) != start + i for i, row in enumerate(rows)):
                raise ShardRefusedError("sealed message-owner range is incomplete")
            if start + len(rows) > self._count:
                raise ShardRefusedError("sealed message-owner range exceeds its manifest")
            start += len(rows)
            yield from (str(row[1]) for row in rows)


class _ShardOwnerLookup(Mapping[_T, str]):
    def __init__(self, path: Path, session: int, kind: str) -> None:
        self.path, self.session, self.kind = path, session, kind

    def __getitem__(self, key: _T) -> str:
        with _shard_connection(self.path) as conn:
            row = conn.execute(
                "SELECT value FROM shard_owner_lookup WHERE session_ordinal=? AND kind=? AND key=?",
                (self.session, self.kind, json.dumps(key, separators=(",", ":"))),
            ).fetchone()
        if row is None:
            raise KeyError(key)
        return str(row[0])

    def __iter__(self) -> Iterator[_T]:
        after = ""
        while True:
            with _shard_connection(self.path) as conn:
                rows = conn.execute(
                    "SELECT key FROM shard_owner_lookup WHERE session_ordinal=? AND kind=? AND key>? ORDER BY key LIMIT 512",
                    (self.session, self.kind, after),
                ).fetchall()
            if not rows:
                return
            after = str(rows[-1][0])
            for row in rows:
                value = json.loads(row[0])
                yield cast(_T, tuple(value) if self.kind == "physical" else value)

    def __len__(self) -> int:
        with _shard_connection(self.path) as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM shard_owner_lookup WHERE session_ordinal=? AND kind=?",
                    (self.session, self.kind),
                ).fetchone()[0]
            )


class _ShardOwnerAmbiguities(Set[_T]):
    def __init__(self, path: Path, session: int, kind: str) -> None:
        self.path, self.session, self.kind = path, session, kind

    def __contains__(self, key: object) -> bool:
        with _shard_connection(self.path) as conn:
            return (
                conn.execute(
                    "SELECT 1 FROM shard_owner_ambiguity WHERE session_ordinal=? AND kind=? AND key=?",
                    (self.session, self.kind, json.dumps(key, separators=(",", ":"))),
                ).fetchone()
                is not None
            )

    def __iter__(self) -> Iterator[_T]:
        after = ""
        while True:
            with _shard_connection(self.path) as conn:
                rows = conn.execute(
                    "SELECT key FROM shard_owner_ambiguity WHERE session_ordinal=? AND kind=? AND key>? ORDER BY key LIMIT 512",
                    (self.session, self.kind, after),
                ).fetchall()
            if not rows:
                return
            after = str(rows[-1][0])
            for row in rows:
                value = json.loads(row[0])
                yield cast(_T, tuple(value) if self.kind == "physical" else value)

    def __len__(self) -> int:
        with _shard_connection(self.path) as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM shard_owner_ambiguity WHERE session_ordinal=? AND kind=?",
                    (self.session, self.kind),
                ).fetchone()[0]
            )


def shard_owner_resolution(path: Path, session_ordinal: int) -> MessageOwnerResolution:
    """Borrow sealed owner evidence; every native reader closes before transfer."""
    from polylogue.pipeline.ids import MessageOwnerResolution

    with _shard_connection(path) as conn:
        row = conn.execute(
            "SELECT owner_count FROM shard_owner_manifest WHERE session_ordinal=?", (session_ordinal,)
        ).fetchone()
    if row is None:
        raise ShardRefusedError("sealed shard lacks required message-owner preparation")
    return MessageOwnerResolution(
        keys=_ShardOwnerKeys(path, session_ordinal, int(row[0])),
        by_physical_coordinate=_ShardOwnerLookup[tuple[int, int]](path, session_ordinal, "physical"),
        ambiguous_physical_coordinates=_ShardOwnerAmbiguities[tuple[int, int]](path, session_ordinal, "physical"),
        by_stable_key=_ShardOwnerLookup[str](path, session_ordinal, "stable"),
        ambiguous_stable_keys=_ShardOwnerAmbiguities[str](path, session_ordinal, "stable"),
        ambiguous_keys=_ShardOwnerAmbiguities[str](path, session_ordinal, "key"),
        unique_provider_keys=_ShardOwnerLookup[str](path, session_ordinal, "provider"),
        ambiguous_provider_ids=_ShardOwnerAmbiguities[str](path, session_ordinal, "provider"),
    )
