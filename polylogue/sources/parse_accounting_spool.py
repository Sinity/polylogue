"""Scratch-backed complete parser admission outcomes.

The parser's exceptional outcomes are part of the acceptance proof.  They may
be numerous, so the retained parser writes them to the preparation database
and replays them for conservation checks rather than retaining one model per
input record.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from polylogue.core.sql_settlement import current_native_sql_lifetimes
from polylogue.sources.value_bounds import (
    MAX_STORABLE_VALUE_BYTES,
    ValueBoundRefusedError,
    require_storable_string,
)
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner

if TYPE_CHECKING:
    from polylogue.sources.parsers.base_models import AdmissionOutcome

_TABLE = "prepared_parse_accounting_outcome"


def _database_path(connection: sqlite3.Connection) -> Path:
    row = connection.execute("PRAGMA database_list").fetchone()
    if row is None or not row[2]:
        raise RuntimeError("spilled parse accounting requires a file-backed preparation database")
    return Path(str(row[2]))


class SpilledParseAccountingOutcomes:
    """Replayable, ordered outcomes held in the prepared artifact database."""

    __slots__ = ("_connection", "path", "accounting_id", "count")

    def __init__(self, connection: sqlite3.Connection | None, path: Path, accounting_id: str, count: int) -> None:
        self._connection = connection
        self.path = path
        self.accounting_id = accounting_id
        self.count = count

    def __len__(self) -> int:
        return self.count

    def iter_unit(self, unit: str) -> Iterator[dict[str, object]]:
        if self._connection is not None:
            yield from _iter_unit(self._connection, self.accounting_id, unit)
            return
        with _reader(self.path) as connection:
            yield from _iter_unit(connection, self.accounting_id, unit)

    def count_by_unit(self) -> dict[str, int]:
        if self._connection is not None:
            rows = self._connection.execute(
                f"SELECT unit, COUNT(*) FROM {_TABLE} WHERE accounting_id = ? GROUP BY unit",
                (self.accounting_id,),
            )
            return {str(unit): int(count) for unit, count in rows}
        with _reader(self.path) as connection:
            rows = connection.execute(
                f"SELECT unit, COUNT(*) FROM {_TABLE} WHERE accounting_id = ? GROUP BY unit",
                (self.accounting_id,),
            )
            return {str(unit): int(count) for unit, count in rows}

    def overlaps(self, unit: str, start: int, end: int) -> bool:
        query = f"SELECT 1 FROM {_TABLE} WHERE accounting_id = ? AND unit = ? AND ordinal >= ? AND ordinal < ? LIMIT 1"
        if self._connection is not None:
            return self._connection.execute(query, (self.accounting_id, unit, start, end)).fetchone() is not None
        with _reader(self.path) as connection:
            return connection.execute(query, (self.accounting_id, unit, start, end)).fetchone() is not None

    def __iter__(self) -> Iterator[AdmissionOutcome]:
        from polylogue.sources.parsers.base_models import AdmissionOutcome

        units = self.count_by_unit()
        for unit in sorted(units):
            for item in self.iter_unit(unit):
                yield AdmissionOutcome.model_validate(item)

    @classmethod
    def from_prepared_reference(cls, path: Path, accounting_id: str, count: int) -> SpilledParseAccountingOutcomes:
        with _reader(path) as connection:
            row = connection.execute(
                f"SELECT COUNT(*) FROM {_TABLE} WHERE accounting_id = ?", (accounting_id,)
            ).fetchone()
        if row is None or int(row[0]) != count:
            raise ValueError("prepared parse accounting outcomes changed")
        return cls(None, path, accounting_id, count)


class SqliteParseAccountingWriter:
    """Write unique typed outcomes into a caller-owned preparation database."""

    __slots__ = ("_connection", "_path", "_id", "_expected", "_count", "_finished")

    def __init__(self, connection: sqlite3.Connection, expected: dict[object, int]) -> None:
        self._connection = connection
        self._path = _database_path(connection)
        self._id = uuid.uuid4().hex
        self._expected = {str(getattr(unit, "value", unit)): int(count) for unit, count in expected.items()}
        self._count = 0
        self._finished = False
        connection.execute(
            f"CREATE TABLE IF NOT EXISTS {_TABLE} ("
            "accounting_id TEXT NOT NULL, unit TEXT NOT NULL, ordinal INTEGER NOT NULL, outcome_json TEXT NOT NULL, "
            "PRIMARY KEY (accounting_id, unit, ordinal)) WITHOUT ROWID"
        )

    def append(self, outcome: object) -> None:
        if self._finished:
            raise RuntimeError("parse accounting is already finished")
        from polylogue.sources.parsers.base_models import AdmissionOutcome

        parsed = outcome if isinstance(outcome, AdmissionOutcome) else AdmissionOutcome.model_validate(outcome)
        unit = parsed.unit.value
        if not 0 <= parsed.ordinal < self._expected.get(unit, -1):
            raise ValueError(f"admission ordinal outside denominator: {unit}[{parsed.ordinal}]")
        encoded = require_storable_string(parsed.model_dump_json(), kind="parse accounting outcome")
        try:
            self._connection.execute(
                f"INSERT INTO {_TABLE} (accounting_id, unit, ordinal, outcome_json) VALUES (?, ?, ?, ?)",
                (self._id, unit, parsed.ordinal, encoded),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"duplicate admission outcome for {unit}[{parsed.ordinal}]") from exc
        except sqlite3.DataError as exc:
            if "too big" not in str(exc):
                raise
            observed = sum(len(value.encode("utf-8", "surrogatepass")) for value in (self._id, unit, encoded))
            raise ValueBoundRefusedError("parse accounting outcome row", observed, MAX_STORABLE_VALUE_BYTES) from exc
        self._count += 1

    def finish(self) -> SpilledParseAccountingOutcomes:
        if self._finished:
            raise RuntimeError("parse accounting is already finished")
        self._finished = True
        outcomes = SpilledParseAccountingOutcomes(self._connection, self._path, self._id, self._count)
        actual = outcomes.count_by_unit()
        if actual != {unit: count for unit, count in self._expected.items() if count}:
            raise ValueError(f"admission denominator mismatch: expected={self._expected}, actual={actual}")
        return outcomes


def _iter_unit(connection: sqlite3.Connection, accounting_id: str, unit: str) -> Iterator[dict[str, object]]:
    cursor = connection.execute(
        f"SELECT outcome_json FROM {_TABLE} WHERE accounting_id = ? AND unit = ? ORDER BY ordinal",
        (accounting_id, unit),
    )
    try:
        for (encoded,) in cursor:
            item = json.loads(encoded)
            if not isinstance(item, dict):
                raise ValueError("prepared parse outcome is not an object")
            yield item
    finally:
        cursor.close()


def _reader(path: Path) -> AbstractContextManager[sqlite3.Connection]:
    class Reader:
        owner: NativeSQLCustodyOwner

        def __enter__(self) -> sqlite3.Connection:
            connection = connect_measured(f"file:{quote(str(path))}?mode=ro", uri=True)
            self.owner = NativeSQLCustodyOwner(connection, lifetime_dependencies=current_native_sql_lifetimes())
            return self.owner.require_connection()

        def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> None:
            if exc is not None:
                from polylogue.storage.sqlite.connection_profile import _close_failed_native_construction

                _close_failed_native_construction(self.owner, exc)
            else:
                self.owner.close()

    return Reader()


__all__ = ["SpilledParseAccountingOutcomes", "SqliteParseAccountingWriter"]
