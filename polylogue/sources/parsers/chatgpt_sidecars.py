"""ChatGPT export sidecar resolvers: asset naming and sandbox-file joins.

The 2026-07-29 GDPR/Takeout export ships attachment BYTES for the first time,
as ``.dat`` ZIP members whose basenames are opaque file ids. Two sibling JSON
files supply names/metadata for those ids:

- ``conversation_asset_file_names.json`` — a flat ``{dat_basename: filename}``
  map, 1,656 entries, conversation-asset ids (``file-<b64ish>``).
- ``library_files.json`` — 2,367 richer records (mime, size, sha256, upload/
  processed times, and an EXACT ``origination_message_id``/
  ``origination_thread_id`` join back to the producing assistant message),
  keyed by ``file_id`` (``file_<32hex>`` for Library files, but also seen on
  conversation attachments referencing an existing Library upload).

Together the two sources name all 3,228 known ``.dat`` blobs (bd polylogue-0hwv).
``library_files.json`` also carries the identity join this module uses to
resolve ``sandbox:/mnt/data/<name>`` links the model emits in assistant text
for Code-Interpreter-produced files, which carry no file id of their own
(bd polylogue-dt5s). Resolution is a strict tiered join, not fuzzy matching:

  tier 1  exact ``origination_message_id`` match, name also matches
  tier 2  exact ``origination_message_id`` match, name differs (still an
          identity-grade join — the id is authoritative, the name is a label)
  tier 3  ``origination_thread_id`` + name match (message id unknown/absent)
  tier 4  a globally unique library file_name match (no id evidence at all)
  tier 5  a globally AMBIGUOUS name match (>1 candidate) — evidence, not
          identity; never resolved to a specific file
  tier 6  unresolved — no candidate anywhere

Every resolution records which tier produced it, so an audit can always tell
an id join from a name guess (see ``SandboxResolution.tier``/``method``).
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import IO, TYPE_CHECKING

from polylogue.core.compute_cancel import check_compute_cancelled

if TYPE_CHECKING:
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner

_DAT_SUFFIX = ".dat"
_ASSET_POINTER_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*://")


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


@dataclass(frozen=True, slots=True)
class LibraryFileRecord:
    """One ``library_files.json`` entry (the ChatGPT Library file catalog)."""

    file_id: str
    file_name: str | None
    file_extension: str | None
    mime_type: str | None
    file_size_bytes: int | None
    sha256_digest: str | None
    origination_message_id: str | None
    origination_thread_id: str | None
    library_artifact_type: str | None
    directory_id: str | None
    created_at: str | None
    file_upload_time: str | None
    file_processed_time: str | None


@dataclass(frozen=True, slots=True)
class ResolvedAsset:
    """A ``.dat`` blob's real identity, resolved from one of the two sidecars."""

    file_id: str
    name: str | None
    mime_type: str | None
    size_bytes: int | None
    sha256_digest: str | None
    #: "library_files" (richer — mime/size/sha) or "conversation_asset_file_names"
    #: (name only). ``library_files`` is preferred whenever both hit.
    source: str


@dataclass(frozen=True, slots=True)
class SandboxResolution:
    """Result of joining one ``sandbox:/mnt/data/<name>`` link to a real file."""

    #: 1-6, see module docstring. 6 means unresolved (metadata-only reference).
    tier: int
    method: str
    file: LibraryFileRecord | None
    #: The library file_name that was actually matched (may differ from the
    #: sandbox link's own filename at tier 2 — the id is authoritative there).
    matched_name: str | None


def parse_library_files(payload: object) -> dict[str, LibraryFileRecord]:
    """Parse ``library_files.json`` into records keyed by ``file_id``."""
    if not isinstance(payload, list):
        return {}
    records: dict[str, LibraryFileRecord] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        file_id = entry.get("file_id")
        if not isinstance(file_id, str) or not file_id:
            continue
        records[file_id] = LibraryFileRecord(
            file_id=file_id,
            file_name=_optional_str(entry.get("file_name")),
            file_extension=_optional_str(entry.get("file_extension")),
            mime_type=_optional_str(entry.get("mime_type")),
            file_size_bytes=_optional_int(entry.get("file_size_bytes")),
            sha256_digest=_optional_str(entry.get("sha256_digest")),
            origination_message_id=_optional_str(entry.get("origination_message_id")),
            origination_thread_id=_optional_str(entry.get("origination_thread_id")),
            library_artifact_type=_optional_str(entry.get("library_artifact_type")),
            directory_id=_optional_str(entry.get("directory_id")),
            created_at=_optional_str(entry.get("created_at")),
            file_upload_time=_optional_str(entry.get("file_upload_time")),
            file_processed_time=_optional_str(entry.get("file_processed_time")),
        )
    return records


def parse_asset_file_names(payload: object) -> dict[str, str]:
    """Parse ``conversation_asset_file_names.json`` (``{dat_basename: name}``).

    Keys are normalized by stripping a trailing ``.dat`` so they line up
    directly with the bare file ids referenced by ``asset_pointer`` and
    ``attachments[].id`` in conversation payloads (neither carries ``.dat``).
    """
    if not isinstance(payload, dict):
        return {}
    names: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value:
            continue
        normalized_key = key[: -len(_DAT_SUFFIX)] if key.endswith(_DAT_SUFFIX) else key
        names[normalized_key] = value
    return names


class ChatGPTAssetIndex:
    """Own a paged sidecar index until its assembly consumers finish."""

    def __init__(self) -> None:
        from polylogue.storage.sqlite.connection_profile import open_scratch_connection

        self._directory = tempfile.TemporaryDirectory(prefix="polylogue-chatgpt-assets-")
        self._path = Path(self._directory.name) / "assets.db"
        self._writer: NativeSQLCustodyOwner | None = None
        self._sealed = False
        self._closed = False
        self._ordinal = 0
        self._asset_group = 0
        try:
            self._writer = open_scratch_connection(self._path, lifetime_dependencies=(self,))
            conn = self._connection()
            conn.execute("BEGIN")
            conn.execute(
                "CREATE TABLE library (id BLOB PRIMARY KEY, ordinal INTEGER NOT NULL, record TEXT NOT NULL, name BLOB, message BLOB, thread BLOB) WITHOUT ROWID"
            )
            conn.execute("CREATE INDEX library_name ON library(name, ordinal)")
            conn.execute("CREATE INDEX library_message ON library(message, ordinal)")
            conn.execute("CREATE INDEX library_thread_name ON library(thread, name, ordinal)")
            conn.execute("CREATE TABLE names (id BLOB PRIMARY KEY, name TEXT NOT NULL) WITHOUT ROWID")
            conn.execute(
                "CREATE TABLE asset_inputs (group_id INTEGER, asset BLOB, member BLOB, hash TEXT, size INTEGER, PRIMARY KEY(group_id, asset, member)) WITHOUT ROWID"
            )
            conn.execute(
                "CREATE TABLE assets (key BLOB PRIMARY KEY, asset BLOB NOT NULL, rendition INTEGER NOT NULL, hash TEXT NOT NULL, size INTEGER NOT NULL) WITHOUT ROWID"
            )
            conn.execute("CREATE INDEX assets_renditions ON assets(asset, rendition, key)")

        except BaseException as primary:
            from polylogue.storage.sqlite.connection_profile import retained_native_sql_owners_for_lifetime

            if self._writer is None and retained_native_sql_owners_for_lifetime(self):
                # The factory already attempted close and owns the typed
                # settlement failure plus this exact artifact dependency.
                raise
            try:
                self.close()
            except BaseException as settlement:
                raise settlement from primary
            raise

    def _connection(self) -> sqlite3.Connection:
        if self._writer is None:
            raise RuntimeError("ChatGPT asset writer was not acquired")
        return self._writer.require_connection()

    @staticmethod
    def _key(value: str | None) -> bytes | None:
        return None if value is None else value.encode("utf-8", "surrogatepass")

    def _insert_library(self, record: LibraryFileRecord) -> None:
        conn = self._connection()
        conn.execute(
            "INSERT INTO library VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
            "record=excluded.record, name=excluded.name, message=excluded.message, thread=excluded.thread",
            (
                self._key(record.file_id),
                self._ordinal,
                json.dumps(asdict(record)),
                self._key(record.file_name),
                self._key(record.origination_message_id),
                self._key(record.origination_thread_id),
            ),
        )
        self._ordinal += 1

    def _insert_name(self, file_id: str, name: str) -> None:
        self._connection().execute(
            "INSERT INTO names VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET name=excluded.name",
            (self._key(file_id), json.dumps(name)),
        )

    def load_stream(self, source: IO[bytes], *, library: bool) -> bool:
        """Project complete sidecar fields into the same indexed authority.

        A savepoint prevents malformed trailing syntax or a CRC failure from
        leaving a partially accepted sidecar. Name-key order preserves Python
        dict duplicate-key and normalized-key collision semantics exactly.
        """
        import io

        from polylogue.schemas.observation_spill import SpilledKey, _ExactJSONText, owned_scalar_events
        from polylogue.sources.detection_projection import DetectorProjection, _project

        encoding = json.detect_encoding(source.read(4))
        source.seek(0)
        text = io.BufferedReader(_ExactJSONText(source, encoding, strip_bom=False))

        conn = self._connection()
        conn.execute("SAVEPOINT sidecar_input")
        try:
            with ExitStack() as stack:
                events = stack.enter_context(owned_scalar_events(text))
                first = next(events, None)
                if first is None:
                    raise ValueError("empty ChatGPT sidecar input")
                event, value = first
                claimed = event not in ("null",)
                with ExitStack() as stack:
                    if library and event == "start_array":
                        rule = DetectorProjection(
                            fields={field.name: DetectorProjection() for field in fields(LibraryFileRecord)}
                        )
                        for event, value in events:
                            check_compute_cancelled()
                            if event == "end_array":
                                break
                            projected = _project(events, event, value, rule, stack)
                            for record in parse_library_files([projected]).values():
                                self._insert_library(record)
                    elif not library and event == "start_map":
                        conn.execute(
                            "CREATE TEMP TABLE raw_names (key BLOB PRIMARY KEY, ordinal INTEGER NOT NULL, value TEXT) WITHOUT ROWID"
                        )
                        for ordinal, (event, value) in enumerate(events):
                            check_compute_cancelled()
                            if event == "end_map":
                                break
                            if isinstance(value, SpilledKey):
                                value = value.read()
                            if event != "map_key" or not isinstance(value, str):
                                raise ValueError("invalid sidecar name map")
                            key = value
                            event, value = next(events)
                            projected = _project(events, event, value, DetectorProjection(), stack)
                            accepted = json.dumps(projected) if isinstance(projected, str) and projected else None
                            conn.execute(
                                "INSERT INTO raw_names VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                                (self._key(key), ordinal, accepted),
                            )
                        cursor = conn.execute(
                            "SELECT key, value FROM raw_names WHERE value IS NOT NULL ORDER BY ordinal"
                        )
                        try:
                            while page := cursor.fetchmany(256):
                                for key, value in page:
                                    check_compute_cancelled()
                                    file_id = bytes(key).decode("utf-8", "surrogatepass")
                                    if file_id.endswith(_DAT_SUFFIX):
                                        file_id = file_id[: -len(_DAT_SUFFIX)]
                                    self._insert_name(file_id, json.loads(value))
                        finally:
                            cursor.close()
                        conn.execute("DROP TABLE raw_names")
                    else:
                        _project(events, event, value, None, stack)
                    if next(events, None) is not None:
                        raise ValueError("sidecar has trailing JSON data")
            # Settles a member's actual CRC even if the tokenizer buffered its
            # last structural token before the underlying stream reached EOF.
            while text.read(16384):
                check_compute_cancelled()
        except BaseException:
            conn.execute("ROLLBACK TO sidecar_input")
            conn.execute("RELEASE sidecar_input")
            raise
        else:
            conn.execute("RELEASE sidecar_input")
        finally:
            text.close()
        return claimed

    def begin_asset_group(self) -> int:
        self._asset_group += 1
        return self._asset_group

    def has_asset_member(self, group: int, asset: str, member: str) -> bool:
        return (
            self._connection()
            .execute(
                "SELECT 1 FROM asset_inputs WHERE group_id=? AND asset=? AND member=?",
                (group, self._key(asset), self._key(member)),
            )
            .fetchone()
            is not None
        )

    def record_asset(self, group: int, asset: str, member: str, blob: tuple[str, int]) -> None:
        check_compute_cancelled()
        self._connection().execute(
            "INSERT OR IGNORE INTO asset_inputs VALUES (?, ?, ?, ?, ?)",
            (group, self._key(asset), self._key(member), *blob),
        )

    def finish_asset_group(self, group: int) -> None:
        conn = self._connection()
        cursor = conn.execute(
            "SELECT asset, member, hash, size, COUNT(*) OVER (PARTITION BY asset) "
            "FROM asset_inputs WHERE group_id=? ORDER BY asset, member",
            (group,),
        )
        try:
            for asset, member, digest, size, count in cursor:
                check_compute_cancelled()
                key = asset if count == 1 else asset + b"#" + member
                conn.execute(
                    "INSERT INTO assets VALUES (?, ?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET "
                    "asset=excluded.asset, rendition=excluded.rendition, hash=excluded.hash, size=excluded.size",
                    (key, asset, int(count > 1), digest, size),
                )
        finally:
            cursor.close()
        conn.execute("DELETE FROM asset_inputs WHERE group_id=?", (group,))

    def _paged_rows(self, sql: str, parameters: tuple[object, ...] = ()) -> Iterator[tuple[object, ...]]:
        from polylogue.storage.sqlite.connection_profile import readonly_connection_context

        if not self._sealed or self._closed:
            raise RuntimeError("ChatGPT asset index is not a sealed readable artifact")
        with readonly_connection_context(self._path, validate_schema=False, lifetime_dependencies=(self,)) as conn:
            cursor = conn.execute(sql, parameters)
            try:
                while page := cursor.fetchmany(256):
                    check_compute_cancelled()
                    yield from (tuple(row) for row in page)
            finally:
                cursor.close()

    @property
    def asset_blobs(self) -> Mapping[str, tuple[str, int]]:
        return _AssetBlobs(self)

    def rendition_keys(self, asset: str) -> Iterator[str]:
        for row in self._paged_rows(
            "SELECT key FROM assets WHERE asset=? AND rendition=1 ORDER BY key",
            (self._key(asset),),
        ):
            key = row[0]
            assert isinstance(key, bytes)
            yield key.decode("utf-8", "surrogatepass")

    def seal(self) -> None:
        if not self._sealed:
            self._connection().commit()
            assert self._writer is not None
            self._writer.close()
            self._sealed = True

    def close(self) -> None:
        """Delete only after every actual SQL creator has settled."""
        from polylogue.storage.sqlite.connection_profile import retained_native_sql_owners_for_lifetime

        if self._closed:
            return
        if not self._sealed and self._writer is not None:
            self._writer.close()
        if retained_native_sql_owners_for_lifetime(self):
            raise RuntimeError("ChatGPT asset index still has unsettled SQLite readers")
        self._directory.cleanup()
        self._closed = True

    def _rows(self, sql: str, parameters: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
        from polylogue.storage.sqlite.connection_profile import readonly_connection_context

        if not self._sealed or self._closed:
            raise RuntimeError("ChatGPT asset index is not a sealed readable artifact")
        with readonly_connection_context(
            self._path,
            validate_schema=False,
            lifetime_dependencies=(self,),
        ) as conn:
            return [tuple(row) for row in conn.execute(sql, parameters).fetchall()]

    def evidence_digest(self) -> str:
        """Hash logical lookup evidence, excluding disposable path and handles."""
        from polylogue.storage.sqlite.connection_profile import readonly_connection_context

        if not self._sealed or self._closed:
            raise RuntimeError("ChatGPT asset evidence is not sealed")
        digest = hashlib.sha256(b"polylogue:chatgpt-asset-evidence:v1\0")
        with readonly_connection_context(
            self._path,
            validate_schema=False,
            lifetime_dependencies=(self,),
        ) as conn:
            for sql in (
                "SELECT id, record FROM library ORDER BY ordinal",
                "SELECT id, name FROM names ORDER BY id",
                "SELECT key, hash, size FROM assets ORDER BY key",
            ):
                digest.update(sql.encode())
                cursor = conn.execute(sql)
                try:
                    while page := cursor.fetchmany(256):
                        check_compute_cancelled()
                        for row in page:
                            for value in row:
                                encoded = (
                                    value if isinstance(value, bytes) else str(value).encode("utf-8", "surrogatepass")
                                )
                                digest.update(len(encoded).to_bytes(8, "big"))
                                digest.update(encoded)
                finally:
                    cursor.close()
        return digest.hexdigest()

    @classmethod
    def build(
        cls,
        *,
        library_files_payload: object = None,
        asset_file_names_payload: object = None,
    ) -> ChatGPTAssetIndex:
        """Index already-decoded caller values through the same paged owner."""
        index = cls()
        try:
            if isinstance(library_files_payload, list):
                for entry in library_files_payload:
                    check_compute_cancelled()
                    for record in parse_library_files([entry]).values():
                        index._insert_library(record)
            if isinstance(asset_file_names_payload, dict):
                for key, value in asset_file_names_payload.items():
                    check_compute_cancelled()
                    for file_id, name in parse_asset_file_names({key: value}).items():
                        index._insert_name(file_id, name)
            index.seal()
            return index
        except BaseException:
            index.close()
            raise

    @classmethod
    def empty(cls) -> ChatGPTAssetIndex:
        return cls.build()

    @property
    def is_empty(self) -> bool:
        return not bool(self._rows("SELECT EXISTS(SELECT 1 FROM library) OR EXISTS(SELECT 1 FROM names)")[0][0])

    @staticmethod
    def _record(row: tuple[object, ...] | None) -> LibraryFileRecord | None:
        if row is None:
            return None
        return LibraryFileRecord(**json.loads(str(row[0])))

    def library_record(self, file_id: str) -> LibraryFileRecord | None:
        rows = self._rows("SELECT record FROM library WHERE id=?", (self._key(_normalize_file_id(file_id)),))
        return self._record(rows[0] if rows else None)

    def resolve_dat(self, file_id: str) -> ResolvedAsset | None:
        clean = _normalize_file_id(file_id)
        record = self.library_record(clean)
        if record is not None:
            return ResolvedAsset(
                clean, record.file_name, record.mime_type, record.file_size_bytes, record.sha256_digest, "library_files"
            )
        rows = self._rows("SELECT name FROM names WHERE id=?", (self._key(clean),))
        if not rows:
            return None
        return ResolvedAsset(clean, json.loads(str(rows[0][0])), None, None, None, "conversation_asset_file_names")

    def _first(self, where: str, parameters: tuple[object, ...]) -> LibraryFileRecord | None:
        rows = self._rows("SELECT record FROM library WHERE " + where + " ORDER BY ordinal LIMIT 1", parameters)
        return self._record(rows[0] if rows else None)

    def resolve_sandbox(
        self,
        *,
        message_id: str | None,
        thread_id: str | None,
        file_name: str,
    ) -> SandboxResolution:
        if message_id:
            exact = self._first("message=? AND name=?", (self._key(message_id), self._key(file_name)))
            if exact is not None:
                return SandboxResolution(1, "message_id+name", exact, exact.file_name)
            by_message = self._first("message=?", (self._key(message_id),))
            if by_message is not None:
                return SandboxResolution(2, "message_id", by_message, by_message.file_name)
        if thread_id:
            by_thread = self._first("thread=? AND name=?", (self._key(thread_id), self._key(file_name)))
            if by_thread is not None:
                return SandboxResolution(3, "thread_id+name", by_thread, file_name)
        count = self._rows("SELECT COUNT(*) FROM library WHERE name=?", (self._key(file_name),))[0][0]
        if count == 1:
            record = self._first("name=?", (self._key(file_name),))
            return SandboxResolution(4, "global_name_unique", record, file_name)
        if count:
            return SandboxResolution(5, "global_name_ambiguous", None, file_name)
        return SandboxResolution(6, "unresolved", None, None)


def strip_asset_pointer_scheme(pointer: str) -> str:
    """Return the bare file id an asset-pointer URI names.

    ChatGPT writes an asset's identity as a URI whose scheme records which
    storage backend served it — ``file-service://file-<id>`` for conversation
    uploads, ``sediment://file_<32hex>`` for the computer-use screenshots and
    later assets. Only the id joins: it is what the export's ``.dat``
    basenames and ``library_files.json`` keys are.
    """
    return _ASSET_POINTER_SCHEME_RE.sub("", pointer, count=1)


def _normalize_file_id(file_id: str) -> str:
    bare = strip_asset_pointer_scheme(file_id)
    if bare.endswith(_DAT_SUFFIX):
        return bare[: -len(_DAT_SUFFIX)]
    return bare


__all__ = [
    "ChatGPTAssetIndex",
    "LibraryFileRecord",
    "ResolvedAsset",
    "SandboxResolution",
    "parse_asset_file_names",
    "parse_library_files",
    "strip_asset_pointer_scheme",
]


class _AssetBlobs(Mapping[str, tuple[str, int]]):
    """Expose exact acquired-member keys through the existing lookup contract."""

    def __init__(self, index: ChatGPTAssetIndex) -> None:
        self.index = index

    def rendition_keys(self, asset: str) -> Iterator[str]:
        return self.index.rendition_keys(asset)

    def __getitem__(self, key: str) -> tuple[str, int]:
        rows = self.index._rows("SELECT hash, size FROM assets WHERE key=?", (self.index._key(key),))
        if not rows:
            raise KeyError(key)
        digest, size = rows[0]
        assert isinstance(size, int)
        return str(digest), size

    def __iter__(self) -> Iterator[str]:
        for row in self.index._paged_rows("SELECT key FROM assets ORDER BY key"):
            key = row[0]
            assert isinstance(key, bytes)
            yield key.decode("utf-8", "surrogatepass")

    def __len__(self) -> int:
        count = self.index._rows("SELECT COUNT(*) FROM assets")[0][0]
        assert isinstance(count, int)
        return count
