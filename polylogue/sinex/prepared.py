"""Prepare the existing Source outbox through its original captured writer plan."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from typing import Any, cast

from polylogue.sinex.models import PublicationMode, PublicationOriginal, PublicationPayload
from polylogue.sinex.obligations import stage_payload
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.revision_governance import _PreparedSourceProducer
from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

_TABLES = (
    "sinex_publication_obligations",
    "sinex_publication_payloads",
    "sinex_publication_segments",
    "sinex_publication_originals",
)


class _Result:
    def __init__(self, rows: list[sqlite3.Row], rowcount: int) -> None:
        self.rows = rows
        self.rowcount = rowcount

    def __iter__(self) -> Iterator[sqlite3.Row]:
        return iter(self.rows)

    def fetchone(self) -> sqlite3.Row | None:
        return self.rows.pop(0) if self.rows else None

    def fetchall(self) -> list[sqlite3.Row]:
        rows, self.rows = self.rows, []
        return rows


class PreparedPublicationStore:
    """Adapt only this producer's SQL to the existing typed Source schedule."""

    in_transaction = True
    row_factory: Any = sqlite3.Row

    def __init__(self, seal: PreparedIndexMutation, payload: PublicationPayload) -> None:
        self.seal = seal
        self.producer = _PreparedSourceProducer(seal)
        self.payload = payload
        for table in _TABLES:
            self.producer._load_artifact_inputs(
                table,
                f"SELECT rowid FROM {table} WHERE object_id=? AND protocol_version=?",
                (payload.object_id, payload.protocol_version),
            )

    def execute(self, sql: str, parameters: tuple[object, ...] = ()) -> _Result:
        operation = sql.lstrip().split(None, 1)[0].upper()
        if operation == "SELECT":
            with self.seal.source_rows(sql, parameters) as cursor:
                cursor.row_factory = sqlite3.Row
                return _Result(cursor.fetchall(), cursor.rowcount)
        match = re.match(r"\s*INSERT(?:\s+OR\s+IGNORE)?\s+INTO\s+(\w+)", sql, re.IGNORECASE)
        if match is None or match[1] not in _TABLES:
            raise ValueError("publication_source_statement_not_declared")
        table = match[1]
        count = 5 if table in {"sinex_publication_segments", "sinex_publication_originals"} else 4
        keys = tuple(
            self.seal.retain_literal_scalar(cast(None | int | float | str | bytes, value))
            for value in parameters[:count]
        )
        self.seal.source_allocation_dependencies(table)
        pieces = sql.split("?")
        if len(pieces) != len(parameters) + 1:
            raise ValueError("publication_statement_parameter_count")
        lowered: list[str] = []
        bindings: list[object] = []
        for piece, value in zip(pieces, parameters, strict=False):
            expression, literal_parameters = self.producer.binding_literal(value)
            lowered.append(piece + expression)
            bindings.extend(literal_parameters)
        sql = "".join(lowered) + pieces[-1]
        sql = sql.replace(f"INTO {table} (", f"INTO {table} (rowid, ", 1).replace("VALUES (", "VALUES (?, ", 1)
        with self.seal.source_statement(
            sql, (None, *bindings), table=table, writable_targets=((table, keys),), allocation_parameter=0
        ) as cursor:
            return _Result([], cursor.rowcount)


def prepare_publication_payload(
    seal: PreparedIndexMutation, payload: PublicationPayload, *, mode: PublicationMode, now_ms: int
) -> None:
    # Staging is part of the same Source capture that acknowledges accepted
    # parsing. The replay owner alone publishes it under ordinary admission.
    store = PreparedPublicationStore(seal, payload)
    stage_payload(store, payload=payload, mode=mode, now_ms=now_ms)  # type: ignore[arg-type]


class _OriginalRead:
    def __init__(self, seal: PreparedIndexMutation) -> None:
        self.seal = seal

    def execute(self, sql: str, parameters: tuple[object, ...] = ()) -> _Result:
        with self.seal.source_rows(sql, parameters) as cursor:
            cursor.row_factory = sqlite3.Row
            return _Result(cursor.fetchall(), cursor.rowcount)


def prepared_originals_for_raw_ids(
    seal: PreparedIndexMutation, raw_ids: tuple[str, ...]
) -> tuple[PublicationOriginal, ...]:
    from polylogue.sinex.originals import originals_for_raw_ids

    producer = _PreparedSourceProducer(seal)
    for raw_id in raw_ids:
        producer._load_artifact_inputs("raw_sessions", "SELECT rowid FROM raw_sessions WHERE raw_id=?", (raw_id,))
        producer._load_artifact_inputs("blob_refs", "SELECT rowid FROM blob_refs WHERE ref_id=?", (raw_id,))
        producer._load_artifact_inputs(
            "excised_content",
            "SELECT rowid FROM excised_content WHERE removed_hash IN "
            "(SELECT blob_hash FROM raw_sessions WHERE raw_id=? UNION SELECT blob_hash FROM blob_refs WHERE ref_id=?)",
            (raw_id, raw_id),
        )
    return originals_for_raw_ids(_OriginalRead(seal), raw_ids, blob_store=BlobStore(seal.archive_root / "blob"))  # type: ignore[arg-type]
