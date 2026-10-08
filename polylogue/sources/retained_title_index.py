"""Operation-owned retained title lookup over original selected evidence."""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Generator, Iterable, Iterator, Mapping
from contextlib import closing
from pathlib import Path

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.storage.sqlite.connection_profile import (
    NativeSQLCustodyOwner,
    open_scratch_connection,
    readonly_connection_context,
    retained_native_sql_owners_for_lifetime,
)


def title_evidence_digest(rows: Iterable[tuple[str, str]]) -> str:
    """Hash exact ordered title identities and values without a serialized copy."""
    digest = hashlib.sha256(b"polylogue:retained-title-evidence:v1\0")
    for identity, title in rows:
        check_compute_cancelled()
        for value in (identity, title):
            encoded = value.encode("utf-8", "surrogatepass")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


class RetainedTitleIndex(Mapping[str, str]):
    """Retain the canonical first selected title per identity until bundle close."""

    def __init__(self, rows: Iterable[tuple[str, str]]) -> None:
        self._directory = tempfile.TemporaryDirectory(prefix="polylogue-retained-titles-")
        self._path = Path(self._directory.name) / "titles.db"
        self._writer: NativeSQLCustodyOwner | None = None
        self._sealed = False
        self._closed = False
        self._count = 0
        try:
            self._writer = open_scratch_connection(self._path, lifetime_dependencies=(self,))
            conn = self._writer.require_connection()
            conn.execute("BEGIN")
            conn.execute("CREATE TABLE titles (identity BLOB PRIMARY KEY, title BLOB NOT NULL) WITHOUT ROWID")
            for identity, title in rows:
                check_compute_cancelled()
                inserted = conn.execute(
                    "INSERT OR IGNORE INTO titles VALUES (?, ?)",
                    (identity.encode("utf-8", "surrogatepass"), title.encode("utf-8", "surrogatepass")),
                ).rowcount
                self._count += inserted
            conn.commit()
            self._writer.close()
            self._sealed = True
        except BaseException as primary:
            if self._writer is None and retained_native_sql_owners_for_lifetime(self):
                raise
            try:
                self.close()
            except BaseException as settlement:
                raise settlement from primary
            raise

    def _require_readable(self) -> None:
        if not self._sealed or self._closed:
            raise RuntimeError("retained title index is not sealed and readable")

    def __len__(self) -> int:
        self._require_readable()
        return self._count

    def __getitem__(self, identity: str) -> str:
        self._require_readable()
        with readonly_connection_context(self._path, validate_schema=False, lifetime_dependencies=(self,)) as conn:
            cursor = conn.execute(
                "SELECT title FROM titles WHERE identity=?", (identity.encode("utf-8", "surrogatepass"),)
            )
            try:
                row = cursor.fetchone()
            finally:
                cursor.close()
        if row is None:
            raise KeyError(identity)
        return bytes(row[0]).decode("utf-8", "surrogatepass")

    def rows(self) -> Generator[tuple[str, str], None, None]:
        self._require_readable()
        with readonly_connection_context(self._path, validate_schema=False, lifetime_dependencies=(self,)) as conn:
            cursor = conn.execute("SELECT identity,title FROM titles ORDER BY identity")
            try:
                for identity, title in cursor:
                    check_compute_cancelled()
                    yield (
                        bytes(identity).decode("utf-8", "surrogatepass"),
                        bytes(title).decode("utf-8", "surrogatepass"),
                    )
            finally:
                cursor.close()

    def __iter__(self) -> Iterator[str]:
        with closing(self.rows()) as rows:
            for identity, _title in rows:
                yield identity

    def evidence_digest(self) -> str:
        with closing(self.rows()) as rows:
            return title_evidence_digest(rows)

    def close(self) -> None:
        if self._closed:
            return
        if not self._sealed and self._writer is not None:
            self._writer.close()
        if retained_native_sql_owners_for_lifetime(self):
            raise RuntimeError("retained title index still has unsettled SQLite readers")
        self._directory.cleanup()
        self._closed = True
