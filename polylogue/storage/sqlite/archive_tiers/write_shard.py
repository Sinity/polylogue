"""Session rows as a SQLite file the writer bulk-copies (polylogue-bp12n.6).

Writer module: index.

``PreparedSessionRows`` moves row *construction* off the writer thread but
still hands back Python tuples, so the writer pays one parameter-binding
round trip per row inside its hold. A shard moves the binding off too: the
stage-A worker inserts its tuples into a private, index-free SQLite file and
the writer runs ``INSERT INTO main.t SELECT ... FROM shard.t`` -- a C-level
copy whose only Python cost is the statement itself.

The shard is a transport, never authority. It holds exactly the rows
``prepare_session_rows`` would have produced, its tables carry no affinity so
every value round-trips with the type the builder bound, and the writer
re-derives nothing from it. Two consequences follow and both are load-bearing:

* **Acceptance is unchanged.** A shard is admitted under exactly the gates
  ``PreparedSessionRows`` is admitted under -- matching session content hash,
  full replace, no lineage slicing -- plus one more: the copy replaces the
  row-*building* loops only, so it needs a session with no prior rows to
  reconcile (``session_row_existed=False``). Any other case falls back to
  building rows inline, which is what ``prepared=None`` already does.
* **A partial shard is not a shard.** Every row and the seal are written in
  one transaction and the seal goes in last. A builder that dies mid-write
  leaves a file whose commit never happened; the rollback restores it to the
  state before the transaction, where no seal exists, and
  :func:`open_session_shard` refuses it. There is no shard state between
  "absent" and "complete".
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from polylogue.storage.sqlite.archive_tiers import archive_tiers_specs
from polylogue.storage.sqlite.archive_tiers.column_spec import ColumnSpec, TableColumnSpec

#: Bump when the shard's own layout changes shape; it is part of the seal.
SHARD_LAYOUT_VERSION = 1

#: The tables a shard transports, in the order the writer must copy them:
#: ``blocks.message_id`` references ``messages.message_id``.
SHARD_TABLES: tuple[str, ...] = ("messages", "blocks")

_ATTACH_SCHEMA_PREFIX = "polylogue_shard"


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
    payload = "\n".join(f"{table}:{shard_table_ddl(table)}" for table in SHARD_TABLES)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _copy_projection(spec: TableColumnSpec, alias: str) -> str:
    """The SELECT list that feeds ``spec``'s INSERT column list from a shard.

    Positional correspondence with ``insert_column_names`` is the whole
    contract: a column the shard carries is read from it, and a column whose
    placeholder is a literal keeps that literal, exactly as the row-tuple
    path does.
    """
    parts: list[str] = []
    for col in spec.writable_columns:
        parts.append(f"{alias}.{col.name}" if col.extract_placeholder == "?" else col.extract_placeholder)
    return ", ".join(parts)


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
    sessions: tuple[ShardSessionRows, ...]

    def by_session_id(self) -> dict[str, ShardSessionRows]:
        return {entry.session_id: entry for entry in self.sessions}


class SessionShardBuilder:
    """Writes one shard file. Not thread-safe: one builder per worker.

    Everything happens in a single transaction that commits in :meth:`seal`,
    so the file is either absent, empty, or complete -- see the module
    docstring.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn = sqlite3.connect(path, isolation_level=None)
        # A shard is scratch: it is read once, by one process, and deleted.
        # Its durability is the source file it was parsed from, so paying for
        # synchronous writes here would buy nothing the re-parse does not.
        self._conn.execute("PRAGMA synchronous = OFF")
        self._conn.execute("PRAGMA journal_mode = DELETE")
        for table in SHARD_TABLES:
            self._conn.execute(shard_table_ddl(table))
        self._conn.execute(_SEAL_DDL)
        self._conn.execute(_SESSION_DDL)
        self._next_rowid = dict.fromkeys(SHARD_TABLES, 1)
        self._sessions: list[ShardSessionRows] = []
        self._conn.execute("BEGIN IMMEDIATE")

    def add(self, prepared: object) -> None:
        """Append one session's prepared rows.

        ``prepared`` is a ``PreparedSessionRows``; it is typed loosely here so
        this module stays importable from the writer without a cycle.
        """
        message_rows: Sequence[tuple[object, ...]] = prepared.message_rows  # type: ignore[attr-defined]
        block_rows: Sequence[tuple[object, ...]] = prepared.block_rows  # type: ignore[attr-defined]
        message_lo = self._append("messages", message_rows)
        block_lo = self._append("blocks", block_rows)
        self._sessions.append(
            ShardSessionRows(
                session_id=prepared.session_id,  # type: ignore[attr-defined]
                session_content_hash=prepared.session_content_hash,  # type: ignore[attr-defined]
                message_lo=message_lo,
                message_hi=message_lo + len(message_rows) - 1,
                block_lo=block_lo,
                block_hi=block_lo + len(block_rows) - 1,
            )
        )

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
        for table in SHARD_TABLES:
            row = self._conn.execute(f"SELECT COALESCE(MAX(rowid), 0) FROM {table}").fetchone()
            if int(row[0]) != self._next_rowid[table] - 1:
                # Dense ascending rowids are what the range scans assume; a
                # gap means this file cannot address its own sessions.
                self._conn.execute("ROLLBACK")
                self._conn.close()
                raise ShardRefusedError(f"shard {self.path}: {table} rowids are not dense")
        self._conn.executemany(
            "INSERT INTO shard_session VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    entry.session_id,
                    entry.session_content_hash,
                    entry.message_lo,
                    entry.message_hi,
                    entry.block_lo,
                    entry.block_hi,
                )
                for entry in self._sessions
            ],
        )
        self._conn.execute(
            "INSERT INTO shard_seal VALUES (?, ?, ?)",
            (SHARD_LAYOUT_VERSION, shard_column_signature(), len(self._sessions)),
        )
        self._conn.execute("COMMIT")
        self._conn.close()
        return SessionShard(path=self.path, sessions=tuple(self._sessions))

    def abandon(self) -> None:
        """Discard an unsealed shard and its file."""
        try:
            self._conn.close()
        finally:
            discard_session_shard(self.path)


def build_session_shard(directory: Path, prepared_sessions: Sequence[object]) -> SessionShard:
    """Build and seal one shard holding ``prepared_sessions``' rows."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"shard-{uuid.uuid4().hex}.db"
    builder = SessionShardBuilder(path)
    try:
        for prepared in prepared_sessions:
            builder.add(prepared)
    except BaseException:
        builder.abandon()
        raise
    return builder.seal()


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
    try:
        with closing(sqlite3.connect(path)) as conn:
            seal = conn.execute("SELECT layout_version, column_signature, session_count FROM shard_seal").fetchall()
            if len(seal) != 1:
                raise ShardRefusedError(f"shard {path}: unsealed ({len(seal)} seal rows)")
            layout_version, column_signature, session_count = seal[0]
            if int(layout_version) != SHARD_LAYOUT_VERSION:
                raise ShardRefusedError(f"shard {path}: layout version {layout_version}")
            if column_signature != shard_column_signature():
                raise ShardRefusedError(f"shard {path}: column signature does not match this build")
            rows = conn.execute(
                "SELECT session_id, content_hash, message_lo, message_hi, block_lo, block_hi FROM shard_session"
            ).fetchall()
            if len(rows) != int(session_count):
                raise ShardRefusedError(f"shard {path}: seal claims {session_count} sessions, manifest has {len(rows)}")
            # A session is addressed by its id, so two entries under one id
            # would let the writer copy the wrong rowid range for one of
            # them. Refuse the file rather than pick.
            if len({str(row[0]) for row in rows}) != len(rows):
                raise ShardRefusedError(f"shard {path}: a session id appears twice in the manifest")
            maxima = {
                table: int(conn.execute(f"SELECT COALESCE(MAX(rowid), 0) FROM {table}").fetchone()[0])
                for table in SHARD_TABLES
            }
            for session_id, _hash, message_lo, message_hi, block_lo, block_hi in rows:
                # Empty sessions use the builder's canonical (1, 0) range;
                # every non-empty range must stay within the rows physically
                # present in the sealed file.  Without this check a damaged
                # manifest could be accepted and silently publish a session
                # missing some of its messages or blocks.
                ranges = (("messages", int(message_lo), int(message_hi)), ("blocks", int(block_lo), int(block_hi)))
                for table, lo, hi in ranges:
                    if hi < lo:
                        if (lo, hi) != (1, 0):
                            raise ShardRefusedError(f"shard {path}: invalid empty {table} range for {session_id}")
                    elif lo < 1 or hi > maxima[table]:
                        raise ShardRefusedError(f"shard {path}: {table} range is outside sealed rows for {session_id}")
    except sqlite3.DatabaseError as exc:
        raise ShardRefusedError(f"shard {path}: unreadable ({exc})") from exc
    return SessionShard(
        path=path,
        sessions=tuple(
            ShardSessionRows(
                session_id=str(row[0]),
                session_content_hash=bytes(row[1]),
                message_lo=int(row[2]),
                message_hi=int(row[3]),
                block_lo=int(row[4]),
                block_hi=int(row[5]),
            )
            for row in rows
        ),
    )


def discard_session_shard(path: Path) -> None:
    """Remove a shard and any journal it left behind."""
    for candidate in (path, path.with_name(path.name + "-journal"), path.with_name(path.name + "-wal")):
        with suppress(OSError):
            os.unlink(candidate)


def _read_only_uri(path: Path) -> str:
    """A ``file:`` URI naming ``path`` read-only, safe for any path spelling.

    The path is percent-encoded: an unescaped ``?`` or ``#`` in a directory
    name would otherwise start the URI's query or fragment and attach some
    other file entirely.
    """
    return f"file:{quote(str(path))}?mode=ro"


@contextmanager
def attached_session_shard(conn: sqlite3.Connection, shard: SessionShard) -> Iterator[str]:
    """ATTACH ``shard`` read-only for the body and DETACH afterwards.

    Read-only is enforced by SQLite, not by discipline: the shard is a
    transport, and a writer that could mutate it could make it disagree with
    the source it was parsed from.

    The attachment brackets whole transactions and never sits inside one:
    SQLite refuses to detach a database a live transaction has read from, so
    a caller that opens the attachment mid-transaction could not close it.
    Entering inside one is therefore refused outright rather than left to
    fail at the exit.
    """
    if conn.in_transaction:
        raise ShardRefusedError("a shard attaches around a transaction, never inside one")
    schema = f"{_ATTACH_SCHEMA_PREFIX}_{uuid.uuid4().hex[:8]}"
    conn.execute(f"ATTACH DATABASE ? AS {schema}", (_read_only_uri(shard.path),))
    try:
        yield schema
    finally:
        conn.execute(f"DETACH DATABASE {schema}")


def copy_shard_session_rows(
    conn: sqlite3.Connection,
    schema: str,
    entry: ShardSessionRows,
) -> None:
    """Bulk-copy one session's message and block rows out of an attached shard.

    Messages first: ``blocks.message_id`` references them.
    """
    messages_spec = _spec("messages")
    blocks_spec = _spec("blocks")
    conn.execute(
        f"INSERT INTO messages ({messages_spec.insert_column_names}) "
        f"SELECT {_copy_projection(messages_spec, 's')} FROM {schema}.messages AS s "
        "WHERE s.rowid BETWEEN ? AND ?",
        (entry.message_lo, entry.message_hi),
    )
    conn.execute(
        f"INSERT OR REPLACE INTO blocks ({blocks_spec.insert_column_names}) "
        f"SELECT {_copy_projection(blocks_spec, 's')} FROM {schema}.blocks AS s "
        "WHERE s.rowid BETWEEN ? AND ?",
        (entry.block_lo, entry.block_hi),
    )


__all__ = [
    "SHARD_LAYOUT_VERSION",
    "SHARD_TABLES",
    "SessionShard",
    "SessionShardBuilder",
    "ShardRefusedError",
    "ShardSessionRows",
    "attached_session_shard",
    "build_session_shard",
    "copy_shard_session_rows",
    "discard_session_shard",
    "open_session_shard",
    "shard_column_signature",
    "shard_table_ddl",
]
