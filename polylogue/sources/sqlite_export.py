"""Canonical logical export of a mutable SQLite source.

An export is the retained material for a live database member: the declared
logical tables, their schema objects and their typed rows, serialized in a
byte-reproducible framing. A page image is not the material -- it changes
after every commit, checkpoint and vacuum, cannot be proven against the live
database, and re-snapshots content the archive already holds.

Framing is one JSON document per line so a multi-gigabyte member streams in
and out in bounded memory:

    {"polylogue_sqlite_export":1,...,"tables":["threads",...]}
    {"table":"threads","columns":["id","title"],"sql":"CREATE TABLE ..."}
    [["t","019f..."],["t","a title"]]
    ...

Values carry their SQLite storage class so text, integers, reals, blobs and
NULL stay distinct: ``["i",5]``, ``["f",1.5]``, ``["t","text"]``, ``["tx",
"<hex>"]`` for TEXT whose bytes are not UTF-8, ``["b","<hex>"]`` for a blob,
and a bare ``null``.

``rowid`` is exported as a column whenever the table has an implicit one, so
a reconstruction preserves insertion order and every ``ORDER BY rowid`` a
parser issues answers exactly as it did against the live database.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, Any

from polylogue.core.binary_signatures import SQLITE_MAGIC_HEADER

EXPORT_VERSION = 1
EXPORT_MAGIC = b'{"polylogue_sqlite_export":1'
#: Enough bytes to decide framing from a blob prefix without reading a row.
EXPORT_PROBE_BYTES = len(EXPORT_MAGIC)


class LogicalExportError(ValueError):
    """A retained export is not readable as one."""


@dataclass(frozen=True, slots=True)
class MemberExportScope:
    """The export scope and header one declared database member is acquired under."""

    tables: tuple[str, ...] | None = None
    member: str | None = None
    origin: str | None = None
    kind: str | None = None


@dataclass(frozen=True, slots=True)
class LogicalExportHeader:
    """The first line of an export: what was acquired and from which member."""

    version: int
    member: str | None
    origin: str | None
    kind: str | None
    tables: tuple[str, ...]
    #: Declared tables the source did not have. A member whose product is
    #: wholly absent exports nothing, and this is the only record of why.
    missing: tuple[str, ...]
    columns: dict[str, tuple[str, ...]]
    schema: tuple[tuple[str | None, ...], ...]


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _schema_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _encode_value(storage_class: str, value: Any) -> Any:
    if storage_class == "null":
        return None
    if storage_class == "integer":
        return ["i", int(value)]
    if storage_class == "real":
        return ["f", float(value)]
    if storage_class == "blob":
        assert isinstance(value, bytes)
        return ["b", value.hex()]
    assert isinstance(value, bytes)
    try:
        return ["t", value.decode("utf-8")]
    except UnicodeDecodeError:
        return ["tx", value.hex()]


def _decode_value(encoded: Any) -> Any:
    if encoded is None:
        return None
    if not isinstance(encoded, list) or len(encoded) != 2:
        raise LogicalExportError(f"malformed export value: {encoded!r}")
    tag, payload = encoded
    if tag == "i":
        return int(payload)
    if tag == "f":
        return float(payload)
    if tag == "t":
        return str(payload)
    if tag == "b":
        return bytes.fromhex(str(payload))
    if tag == "tx":
        return bytes.fromhex(str(payload))
    raise LogicalExportError(f"unknown export value tag: {tag!r}")


def _connect_source(path: Path, *, immutable: bool) -> sqlite3.Connection:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    if immutable:
        uri += "&immutable=1"
    return sqlite3.connect(uri, uri=True)


def _table_plan(conn: sqlite3.Connection, table: str, table_sql: str) -> tuple[list[str], str, list[str]]:
    """Return the exported columns, the deterministic row order, and the declared columns."""
    quoted = '"' + table.replace('"', '""') + '"'
    columns = conn.execute(f"PRAGMA table_info({quoted})").fetchall()
    column_names = [_schema_text(row[1]) for row in columns]
    has_integer_primary_key = any(_schema_text(row[2]).upper() == "INTEGER" and int(row[5]) == 1 for row in columns)
    is_without_rowid = "WITHOUT ROWID" in table_sql.upper()
    selected = (["rowid"] if not is_without_rowid and not has_integer_primary_key else []) + column_names
    if not is_without_rowid:
        # ``rowid`` is unique, never NULL, and the physical storage order, so
        # it is a total order that costs no sorter. Every rowid table's rowid
        # is part of the exported content -- either as the INTEGER PRIMARY KEY
        # or as the synthetic column above -- so ordering by it adds no
        # dependency the digest did not already have. A declared PRIMARY KEY
        # is not a substitute: SQLite lets a rowid table's PK columns be NULL,
        # so it is not reliably unique.
        return selected, "rowid", column_names
    ordered = [
        name for _primary_key_position, name in sorted((int(row[5]), _schema_text(row[1])) for row in columns if row[5])
    ]
    order_terms: list[str] = []
    for name in ordered or column_names:
        quoted_column = '"' + name.replace('"', '""') + '"'
        order_terms.extend((f"typeof({quoted_column}) COLLATE BINARY", f"{quoted_column} COLLATE BINARY"))
    return selected, ", ".join(order_terms), column_names


def write_logical_export(
    source: Path,
    handle: IO[bytes],
    *,
    scope: MemberExportScope | None = None,
    tables: Sequence[str] | None = None,
    immutable: bool = False,
) -> None:
    """Stream the canonical export of *source* into *handle*.

    ``tables`` restricts the export to a declared member's logical product;
    ``None`` exports every ordinary table, which is what a whole-database
    logical revision digests.

    Schema and rows are read inside one explicit transaction: without it a
    concurrent WAL commit can make the export a combination of two source
    states, and no later comparison could detect that.
    """
    scope = scope or MemberExportScope()
    if tables is not None:
        scope = replace(scope, tables=tuple(tables))
    member, origin, kind = scope.member, scope.origin, scope.kind
    tables = scope.tables
    with closing(_connect_source(source, immutable=immutable)) as conn:
        # SQLite permits arbitrary bytes in a TEXT value. Preserve those bytes
        # so a table outside the parser's scope can neither prevent the export
        # nor collapse distinct logical values.
        conn.text_factory = bytes
        conn.execute("BEGIN")
        schema_objects = conn.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
        declared = None if tables is None else set(tables)
        selected_schema = [
            tuple(_schema_text(value) if value is not None else None for value in row)
            for row in schema_objects
            if declared is None or _schema_text(row[2]) in declared
        ]
        table_sql = {
            _schema_text(row[1]): _schema_text(row[3])
            for row in schema_objects
            if _schema_text(row[0]) == "table" and (declared is None or _schema_text(row[1]) in declared)
        }
        exported_tables = [
            name for name, sql in table_sql.items() if not sql.lstrip().upper().startswith("CREATE VIRTUAL TABLE")
        ]
        # Plan every table before the header so a reader can answer a shape
        # question -- "does this export carry these tables with these columns?"
        # -- from the first line, without materializing a single row.
        plans = {table: _table_plan(conn, table, table_sql[table]) for table in exported_tables}
        missing = () if tables is None else tuple(sorted(set(tables) - set(exported_tables)))
        sequence_present = bool(
            conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_sequence'").fetchone()
        )
        sequence_rows: list[list[Any]] | None = None
        if sequence_present:
            # sqlite_sequence is SQLite-owned and excluded from the schema
            # enumeration above with the rest of the sqlite_% names, but its
            # contents are logical state: an insert-then-delete on an
            # AUTOINCREMENT table leaves every user row identical while
            # advancing the stored high-water mark.
            sequence_rows = [
                [_schema_text(name), _schema_text(seq) if seq is not None else None]
                for name, seq in conn.execute("SELECT name, seq FROM sqlite_sequence ORDER BY name")
                if declared is None or _schema_text(name) in declared
            ]
        header = (
            '{"polylogue_sqlite_export":'
            + str(EXPORT_VERSION)
            + ',"kind":'
            + _dumps(kind)
            + ',"member":'
            + _dumps(member)
            + ',"missing":'
            + _dumps(list(missing))
            + ',"origin":'
            + _dumps(origin)
            + ',"columns":'
            + _dumps({table: plans[table][2] for table in exported_tables})
            + ',"schema":'
            + _dumps(selected_schema)
            + ',"sqlite_sequence":'
            + _dumps(sequence_rows)
            + ',"tables":'
            + _dumps(exported_tables)
            + "}\n"
        )
        handle.write(header.encode("utf-8"))
        for table in exported_tables:
            selected, order, _declared = plans[table]
            handle.write(
                (
                    '{"table":'
                    + _dumps(table)
                    + ',"columns":'
                    + _dumps(selected)
                    + ',"sql":'
                    + _dumps(table_sql[table])
                    + "}\n"
                ).encode("utf-8")
            )
            quoted = '"' + table.replace('"', '""') + '"'
            projection = ", ".join(
                f"typeof({quoted_name}), {quoted_name}"
                for name in selected
                for quoted_name in ('"' + name.replace('"', '""') + '"',)
            )
            statement = f"SELECT {projection} FROM {quoted}" + (f" ORDER BY {order}" if order else "")
            for row in conn.execute(statement):
                encoded = [
                    _encode_value(_schema_text(storage_class), value)
                    for storage_class, value in zip(row[::2], row[1::2], strict=True)
                ]
                handle.write((_dumps(encoded) + "\n").encode("utf-8"))


def logical_export_bytes(source: Path, **kwargs: Any) -> bytes:
    """Return the canonical export of *source* as bytes."""
    from io import BytesIO

    buffer = BytesIO()
    write_logical_export(source, buffer, **kwargs)
    return buffer.getvalue()


class _HashingSink:
    """A write-only sink that keeps the digest and discards the bytes."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()

    def write(self, payload: bytes) -> int:
        self._digest.update(payload)
        return len(payload)

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def logical_export_digest(source: Path, **kwargs: Any) -> str:
    """Digest *source*'s canonical export without materializing it."""
    sink = _HashingSink()
    write_logical_export(source, sink, **kwargs)  # type: ignore[arg-type]
    return sink.hexdigest()


def looks_like_logical_export_bytes(payload: bytes) -> bool:
    """Return whether *payload* begins an export document."""
    return payload.startswith(EXPORT_MAGIC)


def looks_like_logical_export_path(path: Path) -> bool:
    """Return whether *path* holds an export rather than a SQLite database."""
    try:
        with path.open("rb") as handle:
            return looks_like_logical_export_bytes(handle.read(EXPORT_PROBE_BYTES))
    except OSError:
        return False


def read_export_header(path: Path) -> LogicalExportHeader:
    """Read an export's first line."""
    with path.open("rb") as handle:
        line = handle.readline()
    return _parse_header(line)


def _parse_header(line: bytes) -> LogicalExportHeader:
    try:
        payload = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LogicalExportError("export header is not a JSON document") from exc
    if not isinstance(payload, dict) or payload.get("polylogue_sqlite_export") != EXPORT_VERSION:
        raise LogicalExportError("not a polylogue SQLite export")
    return LogicalExportHeader(
        version=EXPORT_VERSION,
        member=payload.get("member"),
        origin=payload.get("origin"),
        kind=payload.get("kind"),
        tables=tuple(str(name) for name in payload.get("tables", ())),
        missing=tuple(str(name) for name in payload.get("missing", ())),
        columns={
            str(table): tuple(str(name) for name in names) for table, names in dict(payload.get("columns", {})).items()
        },
        schema=tuple(tuple(row) for row in payload.get("schema", ())),
    )


def logical_source_shape(path: Path, *, immutable: bool = False) -> dict[str, tuple[str, ...]]:
    """Return ``{table: columns}`` for an export or a live SQLite database.

    Detection asks a shape question of every acquired file, so it must not
    cost a reconstruction: an export answers it from its header line.
    """
    if looks_like_logical_export_path(path):
        return dict(read_export_header(path).columns)
    uri = f"{path.resolve().as_uri()}?mode=ro"
    if immutable:
        uri += "&immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        tables = [str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()]
        shape: dict[str, tuple[str, ...]] = {}
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            shape[table] = tuple(str(row[1]) for row in conn.execute(f"PRAGMA table_info({quoted})").fetchall())
    return shape


def _iter_export(path: Path) -> Iterator[tuple[LogicalExportHeader | dict[str, Any] | list[Any], str]]:
    with path.open("rb") as handle:
        first = handle.readline()
        yield _parse_header(first), "header"
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            yield payload, "table" if isinstance(payload, dict) else "row"


def _create_statement(table: str, columns: Sequence[str]) -> str:
    """Recreate the table untyped so every stored value round-trips exactly.

    The original DDL is retained in the export as evidence, but replaying it
    would refuse rows a generated column or a CHECK constraint owns. An
    untyped table applies no affinity conversion, so an INTEGER stays an
    INTEGER and a TEXT stays a TEXT.
    """
    quoted_table = '"' + table.replace('"', '""') + '"'
    declared = ", ".join('"' + name.replace('"', '""') + '"' for name in columns if name != "rowid")
    return f"CREATE TABLE {quoted_table} ({declared})"


def materialize_export(path: Path, destination: Path) -> None:
    """Rebuild an export's declared tables into a standalone SQLite file."""
    with closing(sqlite3.connect(destination)) as conn:
        conn.execute("PRAGMA journal_mode=OFF")
        table: str | None = None
        columns: list[str] = []
        insert = ""
        for payload, kind in _iter_export(path):
            if kind == "header":
                continue
            if kind == "table":
                assert isinstance(payload, dict)
                table = str(payload["table"])
                columns = [str(name) for name in payload["columns"]]
                conn.execute(_create_statement(table, columns))
                quoted_table = '"' + table.replace('"', '""') + '"'
                targets = ", ".join('"' + name.replace('"', '""') + '"' for name in columns)
                placeholders = ", ".join("?" for _ in columns)
                insert = f"INSERT INTO {quoted_table} ({targets}) VALUES ({placeholders})"
                continue
            assert isinstance(payload, list)
            values = [_decode_value(item) for item in payload]
            text_bytes = [
                position for position, item in enumerate(payload) if isinstance(item, list) and item and item[0] == "tx"
            ]
            if text_bytes:
                # A TEXT value whose bytes are not UTF-8 only survives the
                # round trip as a bytes parameter cast back to TEXT.
                placeholders = ", ".join(
                    "CAST(? AS TEXT)" if position in set(text_bytes) else "?" for position in range(len(values))
                )
                assert table is not None
                quoted_table = '"' + table.replace('"', '""') + '"'
                targets = ", ".join('"' + name.replace('"', '""') + '"' for name in columns)
                conn.execute(f"INSERT INTO {quoted_table} ({targets}) VALUES ({placeholders})", values)
            else:
                conn.execute(insert, values)
        conn.commit()


def open_logical_source(path: Path, *, immutable: bool = False, timeout: float = 5.0) -> sqlite3.Connection:
    """Open *path* for reading, whether it is an export or a live database.

    Every parser of a mutable SQLite member reads through here: the retained
    material is an export, while detection and title enrichment still read the
    operator's live file directly. A reconstruction is unlinked as soon as it
    is open, so the connection owns it and closing the connection releases it.
    """
    if not looks_like_logical_source_path(path):
        raise sqlite3.DatabaseError(f"not a SQLite database or logical export: {path}")
    if not looks_like_logical_export_path(path):
        uri = f"{path.resolve().as_uri()}?mode=ro"
        if immutable:
            uri += "&immutable=1"
        return sqlite3.connect(uri, uri=True, timeout=timeout)
    handle, name = tempfile.mkstemp(prefix=".polylogue-export.", suffix=".sqlite")
    os.close(handle)
    reconstruction = Path(name)
    reconstruction.unlink()
    try:
        materialize_export(path, reconstruction)
        conn = sqlite3.connect(f"{reconstruction.as_uri()}?mode=ro", uri=True, timeout=timeout)
    except BaseException:
        reconstruction.unlink(missing_ok=True)
        raise
    reconstruction.unlink(missing_ok=True)
    return conn


def looks_like_logical_source_path(path: Path) -> bool:
    """Return whether *path* holds a SQLite database or a logical export."""
    try:
        with path.open("rb") as handle:
            prefix = handle.read(max(EXPORT_PROBE_BYTES, len(SQLITE_MAGIC_HEADER)))
    except OSError:
        return False
    return prefix.startswith(SQLITE_MAGIC_HEADER) or looks_like_logical_export_bytes(prefix)


def looks_like_logical_source_bytes(payload: bytes) -> bool:
    """Return whether *payload* begins a SQLite database or a logical export."""
    return payload.startswith(SQLITE_MAGIC_HEADER) or looks_like_logical_export_bytes(payload)


__all__ = [
    "EXPORT_MAGIC",
    "EXPORT_PROBE_BYTES",
    "EXPORT_VERSION",
    "LogicalExportError",
    "LogicalExportHeader",
    "MemberExportScope",
    "logical_export_bytes",
    "logical_export_digest",
    "logical_source_shape",
    "looks_like_logical_export_bytes",
    "looks_like_logical_export_path",
    "looks_like_logical_source_bytes",
    "looks_like_logical_source_path",
    "materialize_export",
    "open_logical_source",
    "read_export_header",
    "write_logical_export",
]
