"""Request-owned annotation validation and immutable batch evidence on disk."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Generator
from contextlib import closing

from polylogue.annotations.batch import AnnotationBatch
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.digest import RECEIPT, REFERENCE, canonical_bytes
from polylogue.core.json import JSONDocument


class AnnotationImportSpill:
    """One scratch relation; its caller retains the native SQL lifetime.

    Only one row is decoded at a time. Accepted row keys are unique in SQLite;
    failure order and assertion order stay their physical JSONL line order.
    Sealing forbids mutation before authorization and durable publication.
    """

    def __init__(self, connection: sqlite3.Connection, header: AnnotationBatch) -> None:
        self.connection = connection
        self.header = header.provenance_document()
        self.sealed = False
        self.connection.executescript(
            """CREATE TABLE annotation_import_rows (
                 line INTEGER PRIMARY KEY, row_key TEXT UNIQUE NOT NULL,
                 row_json TEXT NOT NULL, assertion_ref TEXT NOT NULL,
                 confidence REAL, abstained INTEGER NOT NULL);
               CREATE TABLE annotation_import_failures (
                 line INTEGER PRIMARY KEY, failure_json BLOB NOT NULL);
               CREATE TABLE annotation_import_refs (
                 ref TEXT PRIMARY KEY, resolved INTEGER NOT NULL);"""
        )

    def has_row_key(self, key: str) -> bool:
        return (
            self.connection.execute("SELECT 1 FROM annotation_import_rows WHERE row_key=?", (key,)).fetchone()
            is not None
        )

    def ref_resolution(self, ref: str) -> bool | None:
        row = self.connection.execute("SELECT resolved FROM annotation_import_refs WHERE ref=?", (ref,)).fetchone()
        return bool(row[0]) if row is not None else None

    def record_ref_resolution(self, ref: str, resolved: bool) -> None:
        if self.sealed:
            raise RuntimeError("annotation import was already sealed")
        self.connection.execute("INSERT INTO annotation_import_refs VALUES (?, ?)", (ref, int(resolved)))

    def append_row(
        self,
        line: int,
        row_key: str,
        row_json: str,
        assertion_ref: str,
        confidence: float | None,
        abstained: bool,
    ) -> None:
        if self.sealed:
            raise RuntimeError("annotation import was already sealed")
        self.connection.execute(
            "INSERT INTO annotation_import_rows VALUES (?, ?, ?, ?, ?, ?)",
            (line, row_key, row_json, assertion_ref, confidence, int(abstained)),
        )

    def append_failure(self, line: int, document: JSONDocument) -> None:
        if self.sealed:
            raise RuntimeError("annotation import was already sealed")
        self.connection.execute(
            "INSERT INTO annotation_import_failures VALUES (?, ?)",
            (line, canonical_bytes(document, RECEIPT)),
        )

    def seal(self) -> None:
        valid, abstained = self.connection.execute(
            "SELECT count(*), coalesce(sum(abstained),0) FROM annotation_import_rows"
        ).fetchone()
        invalid = self.connection.execute("SELECT count(*) FROM annotation_import_failures").fetchone()[0]
        self.header.update(
            total_count=valid + invalid, valid_count=valid, invalid_count=invalid, abstained_count=abstained
        )
        self.connection.commit()
        self.sealed = True

    @property
    def batch_ref(self) -> str:
        return f"annotation-batch:{self.header['batch_id']}"

    def rows(self) -> Generator[tuple[str, str, float | None], None, None]:
        if not self.sealed:
            raise RuntimeError("annotation import is not sealed")
        with closing(
            self.connection.execute(
                "SELECT row_json, assertion_ref, confidence FROM annotation_import_rows ORDER BY line"
            )
        ) as cursor:
            for row in cursor:
                check_compute_cancelled()
                yield row

    def column_chunks(self, column: str) -> Generator[bytes, None, None]:
        if not self.sealed:
            raise RuntimeError("annotation import is not sealed")
        if column == "metadata_json":
            yield canonical_bytes(self.header["metadata"], REFERENCE)
            return
        if column == "assertion_refs_json":
            query = "SELECT assertion_ref FROM annotation_import_rows ORDER BY line"
        elif column == "validation_failures_json":
            query = "SELECT failure_json FROM annotation_import_failures ORDER BY line"
        else:
            raise ValueError("unknown annotation batch collection")
        yield b"["
        with closing(self.connection.execute(query)) as cursor:
            for index, (item,) in enumerate(cursor):
                check_compute_cancelled()
                if index:
                    yield b","
                yield canonical_bytes(item, REFERENCE) if column == "assertion_refs_json" else bytes(item)
        yield b"]"

    def provenance_chunks(self) -> Generator[bytes, None, None]:
        yield b"{"
        for index, key in enumerate(sorted(self.header)):
            check_compute_cancelled()
            if index:
                yield b","
            yield canonical_bytes(key, REFERENCE) + b":"
            if key in {"assertion_refs", "validation_failures", "metadata"}:
                yield from self.column_chunks(key + "_json")
            else:
                yield canonical_bytes(self.header[key], REFERENCE)
        yield b"}"

    def provenance_digest(self) -> str:
        digest = hashlib.sha256()
        with closing(self.provenance_chunks()) as chunks:
            for part in chunks:
                digest.update(part)
        return digest.hexdigest()

    def rows_digest(self) -> str:
        """Bind validated row values as well as the durable provenance roster."""
        digest = hashlib.sha256()
        with closing(
            self.connection.execute(
                "SELECT line, row_json, assertion_ref, confidence, abstained FROM annotation_import_rows ORDER BY line"
            )
        ) as rows:
            for row in rows:
                check_compute_cancelled()
                digest.update(canonical_bytes(tuple(row), REFERENCE))
                digest.update(b"\n")
        return digest.hexdigest()
