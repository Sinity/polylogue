"""Private, bounded-memory proof of one configured Drive folder listing."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.security.excision_policy import ExcisionPolicyError
from polylogue.sources.drive.source_protocol import DriveSourceAPI
from polylogue.sources.drive.types import DriveAccessDeniedError, DriveFile, DriveNotFoundError


def drive_source_prefix(source_name: str) -> str:
    return f"drive:{quote(source_name, safe='')}:"


def drive_source_coordinate(source_name: str, folder_id: str, file_id: str) -> str:
    return f"{drive_source_prefix(source_name)}/{quote(folder_id, safe='')}/{quote(file_id, safe='')}.json"


def drive_cache_directory(root: Path, folder_id: str) -> Path:
    return root / hashlib.sha256(folder_id.encode()).hexdigest()


class DriveListingWitness:
    """One full listing and exact file-to-retained-Raw relation, owned by intake.

    Scratch is private and disappears on close. No remote readiness survives a
    daemon restart: a new listing reconstructs bindings from durable Source.
    SQLite here is a spill file, never an archive tier or durable authority.
    """

    selection_rule = "aistudio_prompt_mime_v1"

    def __init__(self, source_name: str, folder_ref: str) -> None:
        self.source_name = source_name
        self.folder_ref = folder_ref
        self.folder_id: str | None = None
        self.started_at: str | None = None
        self.listed_at: str | None = None
        self.reobserved_at: str | None = None
        self.enumeration_error: str | None = None
        self.listing_complete = False
        self.postlisting_complete = False
        self.changed = False
        self.generation = 0
        self._lock = threading.RLock()
        self._scratch = tempfile.TemporaryDirectory(prefix="polylogue-drive-listing-")
        self._conn = sqlite3.connect(Path(self._scratch.name) / "listing.sqlite", check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=OFF")
        self._conn.execute(
            "CREATE TABLE listing (id TEXT PRIMARY KEY, name TEXT, mime TEXT, revision TEXT, size INTEGER, coordinate TEXT UNIQUE, raw_id TEXT, acquired_revision TEXT, failure_phase TEXT, failure_type TEXT, permanent INTEGER)"
        )
        self._conn.execute(
            "CREATE TABLE current (id TEXT PRIMARY KEY, name TEXT, mime TEXT, revision TEXT, size INTEGER)"
        )

    def enumerate(self, client: DriveSourceAPI, folder_id: str) -> None:
        self.generation += 1
        self.folder_id = folder_id
        self.started_at = datetime.now(UTC).isoformat()
        with self._lock:
            for file in client.iter_json_files(folder_id):
                check_compute_cancelled()
                self._conn.execute(
                    "INSERT INTO listing (id,name,mime,revision,size,coordinate) VALUES (?,?,?,?,?,?)",
                    (
                        file.file_id,
                        file.name,
                        file.mime_type,
                        file.modified_time,
                        file.size_bytes,
                        drive_source_coordinate(self.source_name, folder_id, file.file_id),
                    ),
                )
            self._conn.commit()
            self.listing_complete = True
            self.listed_at = datetime.now(UTC).isoformat()
            self.generation += 1

    def reobserve(self, client: DriveSourceAPI) -> None:
        if self.folder_id is None:
            return
        with self._lock:
            self.generation += 1
            self.postlisting_complete = False
            resolved = client.resolve_folder_id(self.folder_ref)
            self.changed = self.changed or resolved != self.folder_id
            self._conn.execute("DELETE FROM current")
            for file in client.iter_json_files(resolved):
                check_compute_cancelled()
                self._conn.execute(
                    "INSERT INTO current VALUES (?,?,?,?,?)",
                    (file.file_id, file.name, file.mime_type, file.modified_time, file.size_bytes),
                )
            self._conn.commit()
            # A rename changes presentation, never file identity or bytes.
            self.changed = self.changed or (
                self._conn.execute(
                    "SELECT EXISTS(SELECT id,mime,revision,size FROM listing EXCEPT SELECT id,mime,revision,size FROM current) OR EXISTS(SELECT id,mime,revision,size FROM current EXCEPT SELECT id,mime,revision,size FROM listing)"
                ).fetchone()[0]
                != 0
            )
            self.postlisting_complete = True
            self.reobserved_at = datetime.now(UTC).isoformat()
            self.generation += 1

    def files(self) -> Iterator[DriveFile]:
        after = ""
        while True:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT id,name,mime,revision,size FROM listing WHERE id>? ORDER BY id LIMIT 128", (after,)
                ).fetchall()
            if not rows:
                return
            for row in rows:
                yield DriveFile(*row)
            after = str(rows[-1][0])

    def members(self) -> Iterator[tuple[str, str | None, str | None, str | None, str | None, bool]]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT coordinate,revision,raw_id,acquired_revision,failure_type,COALESCE(permanent,0) FROM listing ORDER BY id"
            )
            try:
                yield from (
                    (str(path), revision, raw, acquired_revision, failure, bool(permanent))
                    for path, revision, raw, acquired_revision, failure, permanent in cursor
                )
            finally:
                cursor.close()

    def record_acquired_revision(self, coordinate: str, revision: str | None) -> None:
        with self._lock:
            self._conn.execute("UPDATE listing SET acquired_revision=? WHERE coordinate=?", (revision, coordinate))
            self.generation += 1

    @property
    def acquisition_failure_count(self) -> int:
        with self._lock:
            return int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM listing WHERE failure_type IS NOT NULL AND failure_phase!='persist'"
                ).fetchone()[0]
            )

    def bind_raw(self, coordinate: str, raw_id: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE listing SET raw_id=? WHERE coordinate=?", (raw_id, coordinate))
            self.generation += 1

    def record_failure(self, coordinate: str, phase: str, error: Exception) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE listing SET failure_phase=?,failure_type=?,permanent=? WHERE coordinate=?",
                (
                    phase,
                    type(error).__name__,
                    isinstance(error, (DriveNotFoundError, DriveAccessDeniedError, ExcisionPolicyError)),
                    coordinate,
                ),
            )
            self.generation += 1

    @property
    def failure_count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM listing WHERE failure_type IS NOT NULL").fetchone()[0])

    def summary(self) -> dict[str, object]:
        with self._lock:
            digest = hashlib.sha256()
            count = 0
            for row in self._conn.execute("SELECT id,name,mime,revision,size FROM listing ORDER BY id"):
                digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode())
                digest.update(b"\n")
                count += 1
            post_digest = hashlib.sha256()
            post_count = 0
            for row in self._conn.execute("SELECT id,name,mime,revision,size FROM current ORDER BY id"):
                post_digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode())
                post_digest.update(b"\n")
                post_count += 1
            return {
                "configured_source": self.source_name,
                "resolved_folder": self.folder_id,
                "selection_rule": self.selection_rule,
                "listing_digest": digest.hexdigest() if self.listing_complete else None,
                "listed_count": count if self.listing_complete else None,
                "postlisting_digest": post_digest.hexdigest() if self.postlisting_complete else None,
                "postlisted_count": post_count if self.postlisting_complete else None,
                "started_at": self.started_at,
                "listed_at": self.listed_at,
                "reobserved_at": self.reobserved_at,
                "listing_complete": self.listing_complete,
                "postlisting_complete": self.postlisting_complete,
                "changed": self.changed,
                "enumeration_error": self.enumeration_error,
            }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
            self._scratch.cleanup()
