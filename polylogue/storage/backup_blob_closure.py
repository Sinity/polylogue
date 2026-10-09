"""The blob set a closed backup package must hold by itself.

A backup package restores ``source.db`` rows together with the blob bytes they
reference. Backup verification (``operations/archive_backup.py``) and the migration backup
gate (``storage/sqlite/migration_runner.py``) both decide completeness here,
from the package alone: its own source tier, its index tier when the package
carries one, its pending publication reservations, and its authenticated
declared-absent assertion. A live acquisition file is never consulted -- it
can change or vanish after the receipt is written, and a restore has only the
package.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import stat
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from polylogue.storage.blob_integrity import project_source_blob_liveness
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

SOURCE_DECLARED_ABSENT_FILE = "source-declared-absent.json"
SOURCE_DECLARED_ABSENT_FORMAT = "polylogue-source-declared-absent-v1"
SOURCE_DECLARED_ABSENT_AUTHORITY = "polylogue-2x6xu"


def _open_source_tier(source_db: Path, *, immutable: bool) -> sqlite3.Connection:
    return open_readonly_connection(
        source_db,
        immutable=immutable,
        validate_schema=False,
        timeout_class="offline-bulk" if immutable else "background-read",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_generation_tables_exist(conn: sqlite3.Connection) -> bool:
    """Return whether the source tier has crossed the generation migration."""

    tables = conn.execute(
        "SELECT name FROM sqlite_schema WHERE type='table' AND name IN ('source_generations', 'source_items')"
    ).fetchall()
    return len(tables) == 2


def source_blob_reservations(source_db: Path, *, immutable: bool = True) -> set[str]:
    """Read pending publication receipts independently of committed liveness."""

    with closing(_open_source_tier(source_db, immutable=immutable)) as source_conn:
        has_reservations = source_conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'blob_publication_reservations'"
        ).fetchone()
        if has_reservations is None:
            return set()
        reservations: set[str] = set()
        for (blob_hash,) in source_conn.execute("SELECT DISTINCT blob_hash FROM blob_publication_reservations"):
            if not isinstance(blob_hash, bytes) or len(blob_hash) != 32:
                raise RuntimeError("source.blob_publication_reservations has invalid blob_hash evidence")
            reservations.add(blob_hash.hex())
        return reservations


def load_source_declared_absent(source_db: Path, assertion_path: Path) -> set[str]:
    """Load and authenticate the operator declaration against ``source.db``."""
    _, hashes = read_source_declared_absent_assertion(assertion_path, source_db_sha256=_sha256_file(source_db))
    return hashes


def read_source_declared_absent_assertion(
    assertion_path: Path, *, source_db_sha256: str
) -> tuple[dict[str, object], set[str]]:
    """Authenticate one declaration against the owner-observed physical image."""

    try:
        metadata = assertion_path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"source declared-absent assertion is missing: {assertion_path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"source declared-absent assertion is not a real regular file: {assertion_path}")
    if metadata.st_nlink != 1:
        raise RuntimeError(f"source declared-absent assertion has multiple hard links: {assertion_path}")
    try:
        assertion = json.loads(assertion_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("source declared-absent assertion is not valid JSON") from exc
    if not isinstance(assertion, dict) or assertion.get("format") != SOURCE_DECLARED_ABSENT_FORMAT:
        raise RuntimeError("source declared-absent assertion has an unknown format")
    if assertion.get("freeze_authority") != SOURCE_DECLARED_ABSENT_AUTHORITY:
        raise RuntimeError("source declared-absent assertion lacks polylogue-2x6xu freeze authority")
    if assertion.get("source_db_sha256") != source_db_sha256:
        raise RuntimeError("source declared-absent assertion is bound to different source.db bytes")
    raw_hashes = assertion.get("declared_absent_blob_hashes")
    if not isinstance(raw_hashes, list) or not raw_hashes:
        raise RuntimeError("source declared-absent assertion has an empty declared set")
    if any(
        not isinstance(blob_hash, str)
        or len(blob_hash) != 64
        or any(char not in "0123456789abcdef" for char in blob_hash)
        for blob_hash in raw_hashes
    ):
        raise RuntimeError("source declared-absent assertion has invalid blob hashes")
    hashes = {str(blob_hash) for blob_hash in raw_hashes}
    if len(hashes) != len(raw_hashes):
        raise RuntimeError("source declared-absent assertion contains duplicate blob hashes")
    return assertion, hashes


@dataclass(frozen=True, slots=True)
class PackageBlobClosure:
    """Blob hashes a package's own tiers reference, split by owner."""

    source_hashes: frozenset[str]
    index_hashes: frozenset[str]
    reservations: frozenset[str]
    declared_absent: frozenset[str]
    declared_absent_asserted: bool

    @property
    def effective_source_hashes(self) -> frozenset[str]:
        return self.source_hashes - self.declared_absent

    @property
    def required(self) -> frozenset[str]:
        """Every blob whose bytes the package itself must carry."""
        return self.effective_source_hashes | self.index_hashes | self.reservations


def package_blob_closure(backup_root: Path) -> PackageBlobClosure:
    """Project the required blob set of a package from its own tiers.

    Tiers are opened ``immutable`` so a closed package is read without
    materializing SQLite sidecars beside it. A package without ``source.db``
    references no source blobs.
    """
    source_db = backup_root / "source.db"
    if not source_db.exists() and not source_db.is_symlink():
        return PackageBlobClosure(frozenset(), frozenset(), frozenset(), frozenset(), False)
    assertion_path = backup_root / SOURCE_DECLARED_ABSENT_FILE
    asserted = assertion_path.exists() or assertion_path.is_symlink()
    declared_absent: set[str] = set()
    if asserted:
        with closing(_open_source_tier(source_db, immutable=True)) as source_conn:
            generations = source_generation_tables_exist(source_conn)
        if generations:
            raise RuntimeError("source declared-absent assertion is only valid before source generations exist")
        declared_absent = load_source_declared_absent(source_db, assertion_path)
    index_db = backup_root / "index.db"
    projection = project_source_blob_liveness(
        source_db,
        index_db=index_db if index_db.exists() or index_db.is_symlink() else None,
        immutable=True,
    )
    source_hashes: set[str] = set()
    index_hashes: set[str] = set()
    for owner, hashes in projection.owner_hashes:
        (source_hashes if owner.startswith("source.db.") else index_hashes).update(hashes)
    return PackageBlobClosure(
        source_hashes=frozenset(source_hashes),
        index_hashes=frozenset(index_hashes),
        reservations=frozenset(source_blob_reservations(source_db)),
        declared_absent=frozenset(declared_absent),
        declared_absent_asserted=asserted,
    )


__all__ = [
    "SOURCE_DECLARED_ABSENT_AUTHORITY",
    "SOURCE_DECLARED_ABSENT_FILE",
    "SOURCE_DECLARED_ABSENT_FORMAT",
    "PackageBlobClosure",
    "load_source_declared_absent",
    "package_blob_closure",
    "source_blob_reservations",
    "source_generation_tables_exist",
]
