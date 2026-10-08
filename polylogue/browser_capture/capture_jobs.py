"""Receiver-authoritative browser capture jobs.

The extension may cache an opaque job id, but this SQLite registry is the
durable authority for profile-loss recovery.  It deliberately stores only a
keyed account scope, never an account identifier or provider credential.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import tempfile
import threading
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Protocol, cast
from uuid import UUID, uuid4

import ijson

from polylogue.browser_capture.capture_job_events import (
    project_capture_job_timelines,
    read_capture_job_events,
    read_capture_job_retention,
)
from polylogue.browser_capture.capture_stream import (
    CaptureSummary,
    SpoolStorageExhaustedError,
    StagedCapture,
    stage_capture_chunks,
    stage_retained_capture,
    summarize_capture_file,
)
from polylogue.browser_capture.receiver import backfill_checkpoint_root
from polylogue.core.digest import CAPTURE, CanonicalizationError, KeyCollisionError, canonical_bytes
from polylogue.paths import browser_capture_spool_root

# All registry readers retain the intent cell separately under their snapshot.
_JOB_COLUMNS = (
    "rowid AS job_rowid, job_id, provider, scope_key, scope_kind, invocation_json, "
    "intent_key, revision, checkpoint_artifact_ref, checkpoint_size, checkpoint_sequence, "
    "checkpoint_digest, receipt_json, retry_json, lease_json, created_at, updated_at, "
    "retention_json, retention_declared"
)

_RETRY_STATES = frozenset({"ready", "retry_wait", "held", "completed", "abandoned"})


class CaptureJobError(Exception):
    def __init__(self, status: int, code: str, details: dict[str, object] | None = None) -> None:
        super().__init__(code)
        self.status = status
        self.code = code
        self.details = details or {}


def canonical_json(value: object) -> str:
    """Return the capture protocol's canonical JSON text for *value*.

    The extension recomputes this text in JavaScript, so the profile admits
    only numbers a double represents exactly.
    """
    try:
        return canonical_bytes(value, CAPTURE).decode("utf-8")
    except KeyCollisionError as exc:
        raise CaptureJobError(400, "non_canonical_key_collision") from exc
    except (CanonicalizationError, TypeError, UnicodeEncodeError) as exc:
        raise CaptureJobError(400, "non_canonical_json") from exc


def canonical_digest(value: object) -> str:
    hasher = hashlib.sha256()
    for part in capture_canonical_chunks(value):
        hasher.update(part)
    return CAPTURE.digest_prefix + hasher.hexdigest()


def _native_json_digest(value: object) -> str:
    from polylogue.browser_capture.native_preparation import json_chunks

    hasher = hashlib.sha256()
    for piece in json_chunks(value, sort_keys=True):
        hasher.update(piece)
    return hasher.hexdigest()


def _canonical_checkpoint_parts(events: Iterable[tuple[str, object]]) -> Iterator[bytes]:
    """Encode vetted JSON tokens under CAPTURE without retaining the tree.

    Objects must already be in normalized key order. The tokenizer and scalar
    encoder allocate a complete token; streaming does not solve that residual.
    """
    frames: list[dict[str, object]] = []
    started = False

    def start_value() -> bytes:
        nonlocal started
        if not frames:
            if started:
                raise CaptureJobError(400, "invalid_checkpoint")
            started = True
            return b""
        frame = frames[-1]
        if frame["kind"] == "array":
            if frame["populated"]:
                return b","
            frame["populated"] = True
        elif frame["pending"]:
            frame["pending"] = False
        else:
            raise CaptureJobError(400, "invalid_checkpoint")
        return b""

    try:
        for event, value in events:
            if event == "map_key":
                if not frames or frames[-1]["kind"] != "map" or frames[-1]["pending"]:
                    raise CaptureJobError(400, "invalid_checkpoint")
                frame = frames[-1]
                encoded = canonical_bytes(value, CAPTURE)
                key = json.loads(encoded)
                previous = frame["previous"]
                if previous is not None and key <= previous:
                    raise CaptureJobError(400, "checkpoint_noncanonical_key_order")
                if frame["populated"]:
                    yield b","
                frame.update(previous=key, key=key, populated=True, pending=True)
                yield encoded
                yield b":"
            elif event in {"start_map", "start_array"}:
                yield start_value()
                yield b"{" if event == "start_map" else b"["
                frames.append(
                    {
                        "kind": "map" if event == "start_map" else "array",
                        "populated": False,
                        "pending": False,
                        "previous": None,
                        "key": None,
                    }
                )
            elif event in {"end_map", "end_array"}:
                if not frames:
                    raise CaptureJobError(400, "invalid_checkpoint")
                frame = frames.pop()
                if frame["pending"] or frame["kind"] != ("map" if event == "end_map" else "array"):
                    raise CaptureJobError(400, "invalid_checkpoint")
                yield b"}" if event == "end_map" else b"]"
            else:
                if event not in {"null", "boolean", "number", "string"}:
                    raise CaptureJobError(400, "invalid_checkpoint")
                yield start_value()
                yield canonical_bytes(value, CAPTURE)
    except (ijson.JSONError, UnicodeError, CanonicalizationError, TypeError) as exc:
        raise CaptureJobError(400, "non_canonical_json") from exc
    if not started or frames:
        raise CaptureJobError(400, "invalid_checkpoint")


def _canonical_checkpoint_digest(stream: BinaryIO) -> tuple[str, str | None]:
    """Hash canonical tokens, to compare with both exact wire and declared SHA."""
    hasher = hashlib.sha256()
    conversation_ref = None

    def observed_events() -> Iterator[tuple[str, object]]:
        nonlocal conversation_ref
        for prefix, event, value in ijson.parse(stream):
            if prefix == "conversation_ref" and event == "string":
                conversation_ref = json.loads(canonical_bytes(value, CAPTURE))
            yield event, value

    for part in _canonical_checkpoint_parts(observed_events()):
        hasher.update(part)
    return "sha256:" + hasher.hexdigest(), conversation_ref


def capture_canonical_chunks(value: object) -> Generator[bytes, None, None]:
    """CAPTURE bytes from lazy collections; scalar encoding keeps its profile."""
    import unicodedata

    from polylogue.schemas.observation_spill import SpilledObject

    if isinstance(value, Mapping):
        yield b"{"
        items: Iterator[tuple[str, object]]
        if isinstance(value, SpilledObject):
            items = value.normalized_sorted_items(lambda key: unicodedata.normalize("NFC", key))
        else:
            normalized: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise CaptureJobError(400, "non_canonical_json")
                name = unicodedata.normalize("NFC", key)
                if name in normalized:
                    raise CaptureJobError(400, "non_canonical_key_collision")
                normalized[name] = item
            items = ((name, normalized[name]) for name in sorted(normalized))
        try:
            for index, (key, item) in enumerate(items):
                if index:
                    yield b","
                yield canonical_bytes(key, CAPTURE)
                yield b":"
                yield from capture_canonical_chunks(item)
        except ValueError as error:
            if str(error) == "normalized_json_key_collision":
                raise CaptureJobError(400, "non_canonical_key_collision") from error
            raise
        finally:
            close = getattr(items, "close", None)
            if close is not None:
                close()
        yield b"}"
    elif isinstance(value, (list, tuple)):
        yield b"["
        for index, item in enumerate(value):
            if index:
                yield b","
            yield from capture_canonical_chunks(item)
        yield b"]"
    else:
        try:
            yield canonical_bytes(value, CAPTURE)
        except (CanonicalizationError, TypeError, UnicodeEncodeError) as error:
            raise CaptureJobError(400, "non_canonical_json") from error


def capture_job_database_path(spool_path: Path | None = None) -> Path:
    return capture_job_store_root(spool_path) / "registry.sqlite3"


def capture_job_store_root(spool_path: Path | None = None) -> Path:
    """Return the isolated filesystem namespace for protocol-2 receiver state.

    The earlier registry schema is durable recovery evidence. A schema change
    must not open it with ``CREATE TABLE IF NOT EXISTS`` and then mutate it.
    Keep its registry and artifacts in place while the current protocol owns a
    separate store.
    """
    return (spool_path or browser_capture_spool_root()) / "capture-jobs" / "v2"


def capture_job_scope_namespace(spool_path: Path | None = None) -> str:
    """Return a stable pseudonym namespace independent of bearer rotation."""
    root = (spool_path or browser_capture_spool_root()).expanduser().resolve()
    digest = hashlib.sha256(f"polylogue:capture-job-scope:v1\0{root}".encode()).hexdigest()
    return f"cjs1:{digest}"


def _now() -> datetime:
    return datetime.now(UTC)


def _stamp(value: datetime | None = None) -> str:
    return (value or _now()).isoformat().replace("+00:00", "Z")


_SCHEMA_LOCK = threading.Lock()
_SCHEMA_READY: set[tuple[str, int, int]] = set()
_ARTIFACT_SWEEP_LOCK = threading.Lock()


class _ArtifactSweep(Protocol):
    def __next__(self) -> os.DirEntry[str]: ...
    def close(self) -> None: ...


@dataclass(slots=True)
class _ArtifactFrontier:
    physical: tuple[int, int, int, int]
    entries: _ArtifactSweep
    pending: deque[str] = field(default_factory=deque)
    exhausted: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def close(self) -> None:
        self.entries.close()
        self.pending.clear()


_ARTIFACT_SWEEPS: dict[str, _ArtifactFrontier] = {}
_GC_PREDICATE = (
    "json_extract(retention_json, '$.state')='eligible' "
    "AND json_extract(retention_json, '$.timeline_authoritative')=0 "
    "AND json_extract(retry_json, '$.state') IN ('completed','abandoned') "
    "AND checkpoint_sequence IS NOT NULL AND receipt_json IS NOT NULL AND receipt_json!=''"
)


def _database_identity(path: Path) -> tuple[str, int, int] | None:
    """The file a completed schema upgrade applies to: its path and inode."""
    try:
        status = path.stat()
    except FileNotFoundError:
        return None
    return (str(path), status.st_dev, status.st_ino)


@dataclass(slots=True)
class CaptureJobRegistry:
    spool_path: Path | None
    receiver_id: str

    _result_owner: ExitStack | None = field(default=None, init=False, repr=False)

    @contextmanager
    def result_scope(self) -> Iterator[None]:
        """Keep lazy capture values alive through their exact response consumer."""
        if self._result_owner is not None:
            raise RuntimeError("capture result scope already owned")
        with ExitStack() as owner:
            self._result_owner = owner
            try:
                yield
            finally:
                self._result_owner = None

    protocol_min: int = 2
    protocol_max: int = 2

    def capabilities(self) -> dict[str, object]:
        return {
            "schema": "polylogue.capture-jobs.capabilities.v1",
            "protocol_min": self.protocol_min,
            "protocol_max": self.protocol_max,
            "scope_namespace": capture_job_scope_namespace(self.spool_path),
            "checkpoint_transport": "canonical-artifact-v1",
        }

    def _connect(self) -> sqlite3.Connection:
        path = capture_job_database_path(self.spool_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        from polylogue.storage.io_phase_metrics import connect_measured

        connection = connect_measured(path, isolation_level=None)
        try:
            connection.row_factory = sqlite3.Row
            identity = _database_identity(path)
            if identity is not None and identity in _SCHEMA_READY:
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("PRAGMA foreign_keys=ON")
                return connection
            with _SCHEMA_LOCK:
                # WAL configuration also mutates the database. Serialize the
                # whole cold setup and recheck after another opener finishes.
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("PRAGMA foreign_keys=ON")
                identity = _database_identity(path)
                if identity is None or identity not in _SCHEMA_READY:
                    connection.execute("PRAGMA journal_mode=WAL")
                    connection.execute("BEGIN IMMEDIATE")
                    self._ensure_schema(connection)
                    connection.commit()
                    identity = _database_identity(path)
                    if identity is not None:
                        _SCHEMA_READY.add(identity)
            return connection
        except BaseException:
            # Closing the original creator connection rolls back incomplete
            # setup, including failures before schema admission.
            connection.close()
            raise

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS capture_jobs (
                job_id TEXT PRIMARY KEY, provider TEXT NOT NULL, scope_key TEXT NOT NULL,
                scope_kind TEXT NOT NULL, invocation_json TEXT,
                intent_key TEXT NOT NULL, intent_json TEXT NOT NULL, revision INTEGER NOT NULL,
                checkpoint_artifact_ref TEXT, checkpoint_size INTEGER,
                checkpoint_sequence INTEGER, checkpoint_digest TEXT,
                receipt_json TEXT, retry_json TEXT NOT NULL, lease_json TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                retention_json TEXT NOT NULL DEFAULT '{"state":"active","hold_reason":null,"timeline_authoritative":true}',
                retention_declared INTEGER NOT NULL DEFAULT 0,
                UNIQUE(provider, scope_key, intent_key)
            ) STRICT"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS capture_job_receipts (
                job_id TEXT NOT NULL, request_id TEXT NOT NULL, checkpoint_sequence INTEGER NOT NULL,
                checkpoint_digest TEXT NOT NULL, receipt_json TEXT NOT NULL,
                PRIMARY KEY(job_id, request_id)
            ) STRICT"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS capture_job_orphans (
                source_digest TEXT PRIMARY KEY, orphan_kind TEXT NOT NULL, diagnostic TEXT NOT NULL,
                created_at TEXT NOT NULL
            ) STRICT"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS capture_job_native_acquisitions (
                job_id TEXT NOT NULL, acquisition_id TEXT NOT NULL,
                binding_json TEXT NOT NULL, member_names_json TEXT NOT NULL, state TEXT NOT NULL,
                plan_digest TEXT, header_json TEXT, final_receipt_json TEXT,
                PRIMARY KEY(job_id, acquisition_id),
                FOREIGN KEY(job_id) REFERENCES capture_jobs(job_id) ON DELETE CASCADE
            ) STRICT"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS capture_job_native_members (
                job_id TEXT NOT NULL, acquisition_id TEXT NOT NULL, member_name TEXT NOT NULL,
                sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL, metadata_json TEXT NOT NULL,
                PRIMARY KEY(job_id, acquisition_id, member_name),
                FOREIGN KEY(job_id, acquisition_id)
                    REFERENCES capture_job_native_acquisitions(job_id, acquisition_id) ON DELETE CASCADE
            ) STRICT"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS capture_job_native_plan (
                job_id TEXT NOT NULL, acquisition_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
                descriptor_json TEXT NOT NULL, descriptor_digest TEXT NOT NULL,
                PRIMARY KEY(job_id, acquisition_id, ordinal),
                FOREIGN KEY(job_id, acquisition_id)
                    REFERENCES capture_job_native_acquisitions(job_id, acquisition_id) ON DELETE CASCADE
            ) STRICT"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS capture_job_native_artifacts (
                job_id TEXT NOT NULL, acquisition_id TEXT NOT NULL, purpose TEXT NOT NULL,
                sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                PRIMARY KEY(job_id, acquisition_id, purpose),
                FOREIGN KEY(job_id, acquisition_id)
                    REFERENCES capture_job_native_acquisitions(job_id, acquisition_id) ON DELETE CASCADE
            ) STRICT"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS capture_job_native_assets (
                job_id TEXT NOT NULL, acquisition_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
                outcome_json TEXT NOT NULL, sha256 TEXT, size_bytes INTEGER,
                PRIMARY KEY(job_id, acquisition_id, ordinal),
                FOREIGN KEY(job_id, acquisition_id, ordinal)
                    REFERENCES capture_job_native_plan(job_id, acquisition_id, ordinal) ON DELETE CASCADE
            ) STRICT"""
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS capture_native_member_artifact ON capture_job_native_members(sha256)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS capture_native_asset_artifact ON capture_job_native_assets(sha256)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS capture_native_prepared_artifact ON capture_job_native_artifacts(sha256)"
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS capture_job_update_receipts (
                job_id TEXT NOT NULL, request_id TEXT NOT NULL, request_digest TEXT NOT NULL,
                receipt_json TEXT NOT NULL, PRIMARY KEY(job_id, request_id)
            ) STRICT"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS capture_job_events (
                event_id TEXT PRIMARY KEY, job_id TEXT NOT NULL,
                event_revision INTEGER NOT NULL, job_revision INTEGER NOT NULL, kind TEXT NOT NULL,
                refs_json TEXT NOT NULL, payload_json TEXT NOT NULL,
                request_id TEXT NOT NULL, occurred_at TEXT NOT NULL,
                UNIQUE(job_id, request_id), UNIQUE(job_id, event_revision)
            ) STRICT"""
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS capture_job_discovery ON capture_jobs(provider, scope_key, created_at DESC, job_id DESC)"
        )

        # Receiver registry metadata only; no archive tier identity changes.
        connection.execute(
            "CREATE INDEX IF NOT EXISTS capture_job_gc_unleased ON capture_jobs(updated_at) WHERE "
            + _GC_PREDICATE
            + " AND lease_json IS NULL"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS capture_job_gc_leased ON capture_jobs("
            "lease_json, updated_at) WHERE " + _GC_PREDICATE + " AND lease_json IS NOT NULL"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS capture_native_unfinished ON capture_job_native_acquisitions(job_id) "
            "WHERE final_receipt_json IS NULL AND state!='cancelled'"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS capture_job_checkpoint_root ON capture_jobs(checkpoint_artifact_ref)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS capture_receipt_checkpoint_root ON capture_job_receipts(checkpoint_digest)"
        )

    def _spool_root(self) -> Path:
        return self.spool_path or browser_capture_spool_root()

    def _checkpoint_artifact_path(self, digest: object) -> Path:
        if not isinstance(digest, str) or len(digest) != 71 or not digest.startswith("sha256:"):
            raise CaptureJobError(400, "invalid_checkpoint_digest")
        try:
            int(digest[7:], 16)
        except ValueError as exc:
            raise CaptureJobError(400, "invalid_checkpoint_digest") from exc
        if digest[7:] != digest[7:].lower():
            raise CaptureJobError(400, "invalid_checkpoint_digest")
        return capture_job_store_root(self._spool_root()) / "artifacts" / (digest[7:] + ".checkpoint")

    def _publish_checkpoint_artifact(self, staged: StagedCapture, digest: str) -> str:
        if "sha256:" + staged.sha256 != digest:
            raise CaptureJobError(400, "checkpoint_digest_mismatch")
        target = self._checkpoint_artifact_path(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Link publishes without replacing an immutable artifact another
            # checkpoint or reader already owns.
            os.link(staged.path, target)
        except FileExistsError:
            with target.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != staged.sha256:
                    raise CaptureJobError(500, "checkpoint_artifact_corrupt") from None
        for directory in (target.parent, target.parent.parent, self._spool_root()):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return digest

    @contextmanager
    def checkpoint_artifact(self, job_id: str, digest: str, body: dict[str, object]) -> Iterator[tuple[BinaryIO, int]]:
        """Open exact scoped custody while the current lease is fenced.

        Checkpoint replacement retains receipt roots. The shared inode lock
        must be respected by terminal-job artifact collection, so cancellation
        or replacement cannot remove the bytes of an admitted read.
        """
        handle = None
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = self._require_scoped(
                    connection, job_id, body.get("provider"), body.get("scope"), body.get("client_protocol")
                )
                self._require_live_lease(job_id, row, body)
                rooted = (
                    row["checkpoint_artifact_ref"] == digest
                    or connection.execute(
                        "SELECT 1 FROM capture_job_receipts WHERE job_id=? AND checkpoint_digest=? LIMIT 1",
                        (job_id, digest),
                    ).fetchone()
                )
                if not rooted:
                    raise CaptureJobError(404, "checkpoint_artifact_not_owned")
                try:
                    handle = self._checkpoint_artifact_path(digest).open("rb")
                except FileNotFoundError as exc:
                    raise CaptureJobError(500, "checkpoint_artifact_missing") from exc
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                size = os.fstat(handle.fileno()).st_size
            # The immutable inode lock carries custody after scope admission.
            # Hashing an arbitrarily large artifact must not retain the writer
            # transaction and block unrelated job controls or cancellation.
            if "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                raise CaptureJobError(500, "checkpoint_artifact_corrupt")
            handle.seek(0)
            yield handle, size
        finally:
            if handle is not None:
                handle.close()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner

        connection = self._connect()
        owner = NativeSQLCustodyOwner(connection)
        try:
            with connection:
                yield connection
        finally:
            owner.close()

    def _validate_scope(self, provider: object, scope: object, protocol: object) -> tuple[str, dict[str, object]]:
        if not isinstance(provider, str) or not provider or provider != provider.lower():
            raise CaptureJobError(400, "invalid_provider")
        self._validate_protocol(protocol)
        if not isinstance(scope, dict):
            raise CaptureJobError(400, "invalid_capture_scope")
        if scope.get("kind") == "account":
            key = scope.get("key")
            if set(scope) != {"kind", "key"} or not isinstance(key, str) or not key.startswith("h1:") or len(key) != 46:
                raise CaptureJobError(400, "invalid_account_scope")
        elif scope.get("kind") == "invocation":
            capability = scope.get("resume_capability")
            if set(scope) != {"kind", "resume_capability"} or not isinstance(capability, str) or not capability:
                raise CaptureJobError(400, "invalid_invocation_scope")
        else:
            raise CaptureJobError(400, "invalid_capture_scope")
        return provider, scope

    @staticmethod
    def _scope_summary(row: sqlite3.Row) -> dict[str, object]:
        # Invocation capabilities are returned only at their explicit creation
        # boundary. Listing a job never grants the authority to resume it.
        return (
            {"kind": "account", "key": row["scope_key"]} if row["scope_kind"] == "account" else {"kind": "invocation"}
        )

    def _validate_protocol(self, protocol: object) -> None:
        if not isinstance(protocol, int) or not self.protocol_min <= protocol <= self.protocol_max:
            raise CaptureJobError(
                426, "incompatible_client", {"receiver_min": self.protocol_min, "receiver_max": self.protocol_max}
            )

    @staticmethod
    def _intent(intent: object) -> dict[str, object]:
        if (
            not isinstance(intent, dict)
            or intent.get("schema_version") != 1
            or not isinstance(intent.get("version"), int)
            or intent["version"] < 1
        ):
            raise CaptureJobError(400, "invalid_intent")
        if not isinstance(intent.get("intent_key"), str) or not intent["intent_key"].startswith("i1:"):
            raise CaptureJobError(400, "invalid_intent")
        if intent.get("digest") != canonical_digest(intent.get("payload")):
            raise CaptureJobError(409, "intent_digest_mismatch")
        return intent

    def _summary(self, connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, object]:
        intent = cast(
            dict[str, object],
            self._json_cell(
                connection,
                row["job_rowid"],
                "intent_json",
                self._require_result_owner(),
                table="capture_jobs",
            ),
        )
        lease = json.loads(row["lease_json"]) if row["lease_json"] else None
        retry = json.loads(row["retry_json"])
        retention = json.loads(row["retention_json"])
        latest_receipt = json.loads(row["receipt_json"]) if row["receipt_json"] else None
        return {
            "job_id": row["job_id"],
            "provider": row["provider"],
            "scope": self._scope_summary(row),
            "intent_key": row["intent_key"],
            "intent_version": intent["version"],
            "intent_digest": intent["digest"],
            "intent": intent,
            "revision": row["revision"],
            "checkpoint_sequence": row["checkpoint_sequence"],
            "checkpoint_digest": row["checkpoint_digest"],
            "retry": retry,
            "retention": retention,
            "checkpoint": (
                {
                    "sequence": row["checkpoint_sequence"],
                    "digest": row["checkpoint_digest"],
                    "artifact_ref": row["checkpoint_artifact_ref"],
                    "size_bytes": row["checkpoint_size"],
                }
                if row["checkpoint_artifact_ref"]
                else None
            ),
            "latest_receipt": latest_receipt,
            "checkpoint_updated_at": latest_receipt["acknowledged_at"] if latest_receipt else None,
            "lease_generation": lease["generation"] if lease else 0,
            "lease_expires_at": lease["expires_at"] if lease else None,
            "lease": (
                {
                    "generation": lease["generation"],
                    "session_id": lease["session_id"],
                    "expires_at": lease["expires_at"],
                }
                if lease
                else None
            ),
            "min_client_protocol": self.protocol_min,
            "max_client_protocol": self.protocol_max,
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _retry(value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            raise CaptureJobError(400, "invalid_retry_state")
        state = value.get("state")
        attempt = value.get("attempt")
        reason = value.get("reason")
        next_eligible_at = value.get("next_eligible_at")
        if state not in _RETRY_STATES or not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
            raise CaptureJobError(400, "invalid_retry_state")
        if reason is not None and (not isinstance(reason, str) or len(reason) > 256):
            raise CaptureJobError(400, "invalid_retry_state")
        if next_eligible_at is not None:
            if not isinstance(next_eligible_at, str):
                raise CaptureJobError(400, "invalid_retry_state")
            try:
                datetime.fromisoformat(next_eligible_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise CaptureJobError(400, "invalid_retry_state") from exc
        if state == "retry_wait" and next_eligible_at is None:
            raise CaptureJobError(400, "invalid_retry_state")
        return {
            "state": state,
            "attempt": attempt,
            "reason": reason,
            "next_eligible_at": next_eligible_at,
        }

    @staticmethod
    def _lease(row: sqlite3.Row) -> dict[str, object] | None:
        value = json.loads(row["lease_json"]) if row["lease_json"] else None
        return value if isinstance(value, dict) else None

    def _require_lease_identity(self, job_id: str, row: sqlite3.Row, body: dict[str, object]) -> dict[str, object]:
        lease = self._lease(row)
        supplied_proof = body.get("proof")
        expected_proof = self._proof(job_id, lease) if lease else ""
        if (
            not lease
            or body.get("lease_id") != lease.get("lease_id")
            or body.get("generation") != lease.get("generation")
            or not isinstance(supplied_proof, str)
            or not hmac.compare_digest(supplied_proof, expected_proof)
        ):
            raise CaptureJobError(409, "lease_replaced")
        return lease

    def _require_live_lease(self, job_id: str, row: sqlite3.Row, body: dict[str, object]) -> dict[str, object]:
        lease = self._require_lease_identity(job_id, row, body)
        expires_at = lease.get("expires_at")
        if not isinstance(expires_at, str) or datetime.fromisoformat(expires_at.replace("Z", "+00:00")) <= _now():
            raise CaptureJobError(409, "lease_expired")
        return lease

    @contextmanager
    def artifact_progress(
        self, job_id: str, body: dict[str, object], *, native: bool = False
    ) -> Iterator[Callable[[], None]]:
        """Keep one admitted physical operation fenced as actual work resumes.

        A network pause alone does not revoke its unchanged authority. Only
        this process-local admission can renew that lease after expiry; every
        new request still requires a live lease. Replacement or explicit job
        authority pause is checked before extending it, never overwritten.
        """
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = self._require_scoped(
                connection, job_id, body.get("provider"), body.get("scope"), body.get("client_protocol")
            )
            self._require_live_lease(job_id, job, body)
            if native and json.loads(job["retry_json"])["state"] in {"held", "abandoned"}:
                raise CaptureJobError(409, "capture_authority_paused")
            if native:
                _job, acquisition = self._native_row(connection, job_id, body)
                if acquisition["state"] == "cancelled":
                    raise CaptureJobError(409, "native_acquisition_cancelled")
        active = True

        def progress() -> None:
            if not active:
                raise RuntimeError("capture artifact operation has settled")
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                job = self._require_scoped(
                    connection, job_id, body.get("provider"), body.get("scope"), body.get("client_protocol")
                )
                lease = self._require_lease_identity(job_id, job, body)
                if native and json.loads(job["retry_json"])["state"] in {"held", "abandoned"}:
                    raise CaptureJobError(409, "capture_authority_paused")
                if native:
                    acquisition = connection.execute(
                        "SELECT state FROM capture_job_native_acquisitions WHERE job_id=? AND acquisition_id=?",
                        (job_id, body.get("acquisition_id")),
                    ).fetchone()
                    if acquisition is None or acquisition["state"] == "cancelled":
                        raise CaptureJobError(409, "native_acquisition_cancelled")
                now = _now()
                expires_at = datetime.fromisoformat(str(lease["expires_at"]).replace("Z", "+00:00"))
                if expires_at <= now + timedelta(seconds=60):
                    renewed = {**lease, "expires_at": _stamp(now + timedelta(seconds=120))}
                    connection.execute(
                        "UPDATE capture_jobs SET lease_json=? WHERE job_id=?", (canonical_json(renewed), job_id)
                    )

        try:
            yield progress
        finally:
            active = False

    def _census_legacy_orphans(self, connection: sqlite3.Connection) -> None:
        root = backfill_checkpoint_root(self.spool_path)
        if root.is_dir():
            for path in root.glob("*.json"):
                try:
                    with path.open("rb") as stream:
                        digest = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
                        stream.seek(0)
                        valid = False
                        try:
                            for prefix, event, _value in ijson.parse(stream):
                                if prefix == "checkpoint" and event not in {"map_key", "end_map", "end_array"}:
                                    valid = event == "start_map"
                        except (ijson.JSONError, UnicodeError):
                            valid = False
                except OSError as exc:
                    connection.execute(
                        """INSERT INTO capture_job_orphans VALUES (?, ?, ?, ?)
                        ON CONFLICT(source_digest) DO UPDATE SET orphan_kind=excluded.orphan_kind, diagnostic=excluded.diagnostic""",
                        (
                            "path-sha256:"
                            + hashlib.sha256(str(path).encode("utf-8", errors="surrogatepass")).hexdigest(),
                            "unreadable_legacy_checkpoint",
                            json.dumps(
                                {"message": "checkpoint bytes could not be read", "errno_class": type(exc).__name__},
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            _stamp(),
                        ),
                    )
                    continue
                kind = "legacy_backfill_checkpoint" if valid else "malformed_legacy_checkpoint"
                diagnostic = json.dumps(
                    {
                        "message": "account scope unresolved; captured custody retained pending ownership proof",
                        "errno_class": None,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                connection.execute(
                    """INSERT INTO capture_job_orphans VALUES (?, ?, ?, ?)
                    ON CONFLICT(source_digest) DO UPDATE SET
                        orphan_kind=excluded.orphan_kind,
                        diagnostic=excluded.diagnostic""",
                    (digest, kind, diagnostic, _stamp()),
                )

    def _require_scoped(
        self, connection: sqlite3.Connection, job_id: str, provider: object, scope: object, protocol: object
    ) -> sqlite3.Row:
        normalized_provider, normalized_scope = self._validate_scope(provider, scope, protocol)
        row = connection.execute("SELECT " + _JOB_COLUMNS + " FROM capture_jobs WHERE job_id=?", (job_id,)).fetchone()
        matches = row is not None and hmac.compare_digest(row["provider"], normalized_provider)
        if matches and row is not None and row["scope_kind"] == normalized_scope["kind"]:
            if row["scope_kind"] == "account":
                matches = hmac.compare_digest(row["scope_key"], str(normalized_scope["key"]))
            else:
                invocation = json.loads(row["invocation_json"])
                matches = hmac.compare_digest(
                    invocation["resume_capability"], str(normalized_scope["resume_capability"])
                )
        else:
            matches = False
        if row is not None and json.loads(row["retention_json"]).get("retiring") is True:
            matches = False
        if not matches:
            raise CaptureJobError(404, "capture_job_not_found")
        return cast(sqlite3.Row, row)

    def _creation_scope(
        self, provider: object, scope: object, protocol: object
    ) -> tuple[str, str, str, dict[str, object] | None]:
        if isinstance(scope, dict) and scope.get("kind") == "invocation":
            if set(scope) != {"kind", "creation_token", "binding"}:
                raise CaptureJobError(400, "invalid_invocation_scope")
            token, binding = scope["creation_token"], scope["binding"]
            if not isinstance(token, str) or not token or not isinstance(binding, dict):
                raise CaptureJobError(400, "invalid_invocation_scope")
            try:
                UUID(token)
            except ValueError as exc:
                raise CaptureJobError(400, "invalid_invocation_scope") from exc
            required = {
                "preparation_instance_id",
                "extension_instance_id",
                "acquisition_sequence",
                "raw_revision",
                "native_id",
                "source_url",
                "document_id",
                "invocation_id",
            }
            original_witness = {"extension_instance_id", "acquisition_sequence", "invocation_id"}
            if set(binding) != required or any(
                not isinstance(binding[key], str) or not binding[key] for key in required - original_witness
            ):
                raise CaptureJobError(400, "invalid_invocation_binding")
            sequence = binding["acquisition_sequence"]
            if sequence is not None and (type(sequence) is not int or not 1 <= sequence <= (1 << 53) - 1):
                raise CaptureJobError(400, "invalid_invocation_binding")
            invocation_id = binding["invocation_id"]
            if invocation_id is not None and (not isinstance(invocation_id, str) or not invocation_id):
                raise CaptureJobError(400, "invalid_invocation_binding")
            observed_instance = binding["extension_instance_id"]
            if observed_instance is not None and (not isinstance(observed_instance, str) or not observed_instance):
                raise CaptureJobError(400, "invalid_invocation_binding")
            # The creation token authorizes this preparation operation. It does
            # not manufacture observation order for a passive or historical raw
            # revision whose original invocation witness was never retained.
            validated_provider, _ = self._validate_scope(
                provider, {"kind": "invocation", "resume_capability": token}, protocol
            )
            key = "v1:" + hashlib.sha256(token.encode()).hexdigest()
            return (
                validated_provider,
                "invocation",
                key,
                {"binding": binding, "resume_capability": secrets.token_urlsafe(32)},
            )
        validated_provider, validated_scope = self._validate_scope(provider, scope, protocol)
        return validated_provider, "account", str(validated_scope["key"]), None

    def _creation_response(self, connection: sqlite3.Connection, row: sqlite3.Row, created: bool) -> dict[str, object]:
        result: dict[str, object] = {"created": created, "job": self._summary(connection, row)}
        if row["scope_kind"] == "invocation":
            invocation = json.loads(row["invocation_json"])
            result["scope"] = {"kind": "invocation", "resume_capability": invocation["resume_capability"]}
        else:
            result["scope"] = self._scope_summary(row)
        return result

    def create(self, body: dict[str, object]) -> tuple[int, dict[str, object]]:
        provider, scope_kind, scope_key, invocation = self._creation_scope(
            body.get("provider"), body.get("scope"), body.get("client_protocol")
        )
        intent = self._intent(body.get("intent"))
        now = _stamp()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            found = connection.execute(
                "SELECT " + _JOB_COLUMNS + " FROM capture_jobs WHERE provider=? AND scope_key=? AND intent_key=?",
                (provider, scope_key, intent["intent_key"]),
            ).fetchone()
            if found is not None:
                if self._retirement_eligible(connection, found, _now()):
                    raise CaptureJobError(503, "capture_job_retirement_pending")
                with ExitStack() as intent_owner:
                    stored_intent = cast(
                        dict[str, object],
                        self._json_cell(
                            connection, found["job_rowid"], "intent_json", intent_owner, table="capture_jobs"
                        ),
                    )
                    if stored_intent["digest"] != intent["digest"]:
                        raise CaptureJobError(409, "intent_key_conflict")
                if found["scope_kind"] != scope_kind or (
                    invocation is not None and json.loads(found["invocation_json"])["binding"] != invocation["binding"]
                ):
                    raise CaptureJobError(409, "invocation_binding_conflict")
                return 200, self._creation_response(connection, found, False)
            job_id = str(uuid4())
            connection.execute(
                # Named columns, not positional: a receiver database created by an
                # earlier build carries extra columns this build never writes.
                """INSERT INTO capture_jobs (
                    job_id, provider, scope_key, scope_kind, invocation_json, intent_key, intent_json, revision,
                    checkpoint_artifact_ref, checkpoint_sequence, checkpoint_digest, receipt_json,
                    retry_json, lease_json, created_at, updated_at, retention_json
                 ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL, NULL, NULL, ?, NULL, ?, ?, ?)""",
                (
                    job_id,
                    provider,
                    scope_key,
                    scope_kind,
                    json.dumps(
                        {**invocation, "binding": dict(cast(dict[str, object], invocation["binding"]).items())},
                        separators=(",", ":"),
                    )
                    if invocation is not None
                    else None,
                    intent["intent_key"],
                    "",
                    canonical_json({"state": "ready", "attempt": 0}),
                    now,
                    now,
                    canonical_json({"state": "active", "hold_reason": None, "timeline_authoritative": True}),
                ),
            )
            row = connection.execute(
                "SELECT " + _JOB_COLUMNS + " FROM capture_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            self._write_json_cell(connection, row["job_rowid"], "intent_json", intent, table="capture_jobs")
            self._append_event(
                connection,
                job_id,
                "created",
                "create:" + job_id,
                row["revision"],
                {},
                {"provider": provider, "intent_key": intent["intent_key"]},
                advance_revision=False,
            )
            return 201, self._creation_response(connection, row, True)

    def discover(self, body: dict[str, object]) -> dict[str, object]:
        provider, scope = self._validate_scope(body.get("provider"), body.get("scope"), body.get("client_protocol"))
        if scope["kind"] != "account":
            raise CaptureJobError(403, "invocation_discovery_forbidden")
        intent_key = body.get("intent_key")
        if intent_key is not None and (not isinstance(intent_key, str) or not intent_key.startswith("i1:")):
            raise CaptureJobError(400, "invalid_intent")
        cursor = body.get("cursor")
        if cursor is not None and (
            not isinstance(cursor, dict)
            or set(cursor) != {"created_at", "job_id"}
            or any(not isinstance(value, str) for value in cursor.values())
        ):
            raise CaptureJobError(400, "invalid_capture_job_cursor")
        with self._connection() as connection:
            connection.execute("BEGIN")
            parameters: list[object] = [provider, scope["key"]]
            fractional_upper, whole_prefix, whole_upper = self._lease_expiry_bounds(_now())
            parameters.extend((fractional_upper, whole_prefix, whole_upper))
            predicate = (
                "provider=? AND scope_kind='account' AND scope_key=? "
                "AND COALESCE(json_extract(retention_json, '$.retiring'),0)=0 "
                "AND NOT ("
                + _GC_PREDICATE
                + " AND (lease_json IS NULL OR lease_json<=? OR (lease_json>=? AND lease_json<=?)) "
                "AND NOT EXISTS (SELECT 1 FROM capture_job_native_acquisitions n "
                "WHERE n.job_id=capture_jobs.job_id AND n.final_receipt_json IS NULL AND n.state!='cancelled'))"
            )
            if intent_key:
                predicate += " AND intent_key=?"
                parameters.append(intent_key)
            total = connection.execute("SELECT COUNT(*) FROM capture_jobs WHERE " + predicate, parameters).fetchone()[0]
            if cursor is not None:
                predicate += " AND (created_at, job_id) < (?, ?)"
                parameters.extend([cursor["created_at"], cursor["job_id"]])
            rows = connection.execute(
                "SELECT "
                + _JOB_COLUMNS
                + " FROM capture_jobs WHERE "
                + predicate
                + " ORDER BY created_at DESC, job_id DESC LIMIT 26",
                parameters,
            ).fetchall()
            jobs = [self._summary(connection, row) for row in rows[:25]]
            after = {"created_at": rows[24]["created_at"], "job_id": rows[24]["job_id"]} if len(rows) > 25 else None
            return {"jobs": jobs, "total": total, "cursor": after, "has_more": after is not None}

    def list_orphans(self, protocol: object, cursor: str | None = None) -> dict[str, object]:
        """Reconcile retained custody and read one page from its existing owner."""
        self._validate_protocol(protocol)
        if cursor is not None and not isinstance(cursor, str):
            raise CaptureJobError(400, "invalid_orphan_cursor")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if cursor is None:
                self._census_legacy_orphans(connection)
            total = connection.execute("SELECT COUNT(*) FROM capture_job_orphans").fetchone()[0]
            rows = connection.execute(
                "SELECT * FROM capture_job_orphans WHERE source_digest > ? ORDER BY source_digest LIMIT 26",
                (cursor or "",),
            ).fetchall()
            after = rows[24]["source_digest"] if len(rows) > 25 else None
            return {
                "orphans": [
                    {**dict(row), "diagnostic": diagnostic["message"], "errno_class": diagnostic["errno_class"]}
                    for row in rows[:25]
                    for diagnostic in [json.loads(row["diagnostic"])]
                ],
                "total": total,
                "cursor": after,
                "has_more": after is not None,
            }

    @contextmanager
    def inspect_orphan(self, source_digest: str, protocol: object) -> Iterator[tuple[BinaryIO, int]]:
        """Open exact retained evidence without asserting an account or consuming it.

        The legacy producer is retired. Hash and send the same open inode so
        inspection cannot select another file by a filename supplied by a client.
        """
        self._validate_protocol(protocol)
        if not source_digest.startswith("sha256:") or len(source_digest) != 71:
            raise CaptureJobError(400, "invalid_orphan_digest")
        try:
            bytes.fromhex(source_digest[7:])
        except ValueError as exc:
            raise CaptureJobError(400, "invalid_orphan_digest") from exc
        root = backfill_checkpoint_root(self.spool_path)
        for path in root.glob("*.json"):
            with path.open("rb") as stream:
                observed = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
                if not hmac.compare_digest(source_digest, observed):
                    continue
                size = stream.seek(0, 2)
                stream.seek(0)
                yield stream, size
                return
        raise CaptureJobError(404, "orphan_payload_not_found")

    def get(self, job_id: str, body: dict[str, object]) -> dict[str, object]:
        with self._connection() as connection:
            connection.execute("BEGIN")
            row = self._require_scoped(
                connection,
                job_id,
                body.get("provider"),
                body.get("scope"),
                body.get("client_protocol"),
            )
            receipts = [
                json.loads(receipt["receipt_json"])
                for receipt in connection.execute(
                    "SELECT receipt_json FROM capture_job_receipts WHERE job_id=? ORDER BY checkpoint_sequence",
                    (job_id,),
                ).fetchall()
            ]
            updates = [
                json.loads(receipt["receipt_json"])
                for receipt in connection.execute(
                    "SELECT receipt_json FROM capture_job_update_receipts WHERE job_id=? ORDER BY rowid",
                    (job_id,),
                ).fetchall()
            ]
            events, _cursor = read_capture_job_events(connection, job_id, 500, decode_page=self._event_page)
            lifecycle = read_capture_job_retention(connection, job_id)
            return {
                "job": self._summary(connection, row),
                "lifecycle": lifecycle,
                "receipts": [*receipts, *updates],
                "events": events,
                "timelines": project_capture_job_timelines(events),
            }

    def _stage_json_chunks(self, chunks: Iterator[bytes], *, durable: bool) -> StagedCapture:
        try:
            return stage_capture_chunks(chunks, spool_root=self._spool_root(), durable=durable)
        except SpoolStorageExhaustedError as error:
            raise CaptureJobError(507, "spool_storage_exhausted") from error

    def _require_result_owner(self) -> ExitStack:
        if self._result_owner is None:
            raise RuntimeError("capture read requires its result scope")
        return self._result_owner

    def _json_cell(
        self,
        connection: sqlite3.Connection,
        rowid: int,
        column: str,
        owner: ExitStack,
        *,
        table: str = "capture_job_events",
    ) -> object:
        """Detach one exact pinned capture cell to request-owned lazy JSON scratch."""
        from polylogue.core.compute_cancel import check_compute_cancelled
        from polylogue.schemas.observation_spill import StreamedJSONDocument
        from polylogue.storage.sqlite.connection_profile import native_sql_owner_for_connection
        from polylogue.storage.sqlite.literal_cells import stream_literal_blob

        if (table, column) not in {
            ("capture_jobs", "intent_json"),
            ("capture_job_events", "refs_json"),
            ("capture_job_events", "payload_json"),
            ("capture_job_native_acquisitions", "header_json"),
            ("capture_job_native_assets", "outcome_json"),
            ("capture_job_native_members", "metadata_json"),
        }:
            raise RuntimeError("undeclared capture JSON cell")
        native = native_sql_owner_for_connection(connection)
        if native is None:
            raise RuntimeError("capture literal read lost original native owner")

        def chunks() -> Generator[bytes, None, None]:
            with native.readonly_blob(table, column, rowid) as blob:
                yield from stream_literal_blob(blob, len(blob), check_compute_cancelled)

        staged = self._stage_json_chunks(chunks(), durable=False)
        owner.callback(staged.discard)
        return owner.enter_context(StreamedJSONDocument(staged.path))

    def _write_json_cell(
        self,
        connection: sqlite3.Connection,
        rowid: int,
        column: str,
        value: object,
        *,
        table: str = "capture_job_events",
        native_json: bool = False,
    ) -> None:
        from polylogue.browser_capture.native_preparation import json_chunks
        from polylogue.storage.sqlite.literal_cells import SQLiteLiteralWriteError, write_literal_text

        parts = json_chunks(value, sort_keys=True) if native_json else capture_canonical_chunks(value)
        staged = self._stage_json_chunks(parts, durable=False)
        try:

            def chunks() -> Generator[bytes, None, None]:
                with staged.path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        yield chunk

            write_literal_text(connection, table, column, rowid, byte_length=staged.size_bytes, chunks=chunks)
        except SQLiteLiteralWriteError as error:
            raise CaptureJobError(
                413 if error.physical_limit else 500,
                "capture_literal_physical_limit" if error.physical_limit else "capture_literal_publication_failed",
            ) from error
        finally:
            staged.discard()

    def _append_event(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        kind: object,
        request_id: str,
        expected_revision: int,
        refs: dict[str, object],
        payload: dict[str, object],
        *,
        advance_revision: bool = True,
    ) -> dict[str, object]:
        if not isinstance(kind, str) or kind not in {
            "created",
            "first-seen",
            "detected-new",
            "capture-attempted",
            "acknowledged",
            "held-with-reason",
            "explicit-no-op",
            "adopted",
            "resumed",
            "completed",
            "abandoned",
        }:
            raise CaptureJobError(400, "invalid_capture_job_event")
        if not isinstance(request_id, str):
            raise CaptureJobError(400, "invalid_event_request_id")
        if request_id == "":
            raise CaptureJobError(400, "invalid_event_request_id")
        if not isinstance(expected_revision, int):
            raise CaptureJobError(400, "invalid_event_revision")
        if isinstance(expected_revision, bool):
            raise CaptureJobError(400, "invalid_event_revision")
        if not isinstance(refs, dict):
            raise CaptureJobError(400, "invalid_capture_job_event")
        if not isinstance(payload, dict):
            raise CaptureJobError(400, "invalid_capture_job_event")
        existing = connection.execute(
            "SELECT rowid AS event_rowid, event_id, job_id, event_revision, job_revision, kind, request_id, occurred_at FROM capture_job_events WHERE job_id=? AND request_id=?",
            (job_id, request_id),
        ).fetchone()
        hasher = hashlib.sha256()
        for part in capture_canonical_chunks({"kind": kind, "refs": refs, "payload": payload}):
            hasher.update(part)
        digest = "sha256:" + hasher.hexdigest()
        if existing is not None:
            with ExitStack() as check_owner:
                stored = self._json_cell(connection, existing["event_rowid"], "payload_json", check_owner)
                if not isinstance(stored, dict) or stored.get("digest") != digest:
                    raise CaptureJobError(409, "event_request_conflict")
            return self._event_dict(connection, existing)
        row = connection.execute("SELECT revision FROM capture_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise CaptureJobError(404, "capture_job_not_found")
        if expected_revision != row["revision"]:
            raise CaptureJobError(409, "cas_mismatch", {"revision": row["revision"]})
        event_revision = connection.execute(
            "SELECT COALESCE(MAX(event_revision), -1) + 1 FROM capture_job_events WHERE job_id=?", (job_id,)
        ).fetchone()[0]
        job_revision = expected_revision + 1 if advance_revision else expected_revision
        now = _stamp()
        event_id = str(uuid4())
        stored_payload = {"digest": digest, "value": payload}
        cursor = connection.execute(
            "INSERT INTO capture_job_events "
            "(event_id, job_id, event_revision, job_revision, kind, refs_json, payload_json, request_id, occurred_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                job_id,
                event_revision,
                job_revision,
                kind,
                "",
                "",
                request_id,
                now,
            ),
        )
        event_rowid = cast(int, cursor.lastrowid)
        cursor.close()
        self._write_json_cell(connection, event_rowid, "refs_json", refs)
        self._write_json_cell(connection, event_rowid, "payload_json", stored_payload)
        if advance_revision:
            updated = connection.execute(
                "UPDATE capture_jobs SET revision=?, updated_at=? WHERE job_id=? AND revision=?",
                (job_revision, now, job_id, expected_revision),
            )
            if updated.rowcount != 1:
                raise CaptureJobError(409, "cas_mismatch")
        return {
            "event_id": event_id,
            "job_id": job_id,
            "event_revision": event_revision,
            "job_revision": job_revision,
            "kind": kind,
            "refs": refs,
            "payload": payload,
            "request_id": request_id,
            "occurred_at": now,
        }

    def _event_page(self, connection: sqlite3.Connection, rows: list[sqlite3.Row]) -> list[dict[str, object]]:
        """Detach one page while releasing each event's two spill owners in turn."""
        from polylogue.browser_capture.native_preparation import json_chunks
        from polylogue.schemas.observation_spill import StreamedJSONDocument

        def chunks() -> Iterator[bytes]:
            yield b"["
            for index, row in enumerate(rows):
                if index:
                    yield b","
                with ExitStack() as event_owner:
                    yield from json_chunks(self._event_dict(connection, row, event_owner))
            yield b"]"

        owner = self._require_result_owner()
        staged = self._stage_json_chunks(chunks(), durable=False)
        owner.callback(staged.discard)
        return cast(list[dict[str, object]], owner.enter_context(StreamedJSONDocument(staged.path)))

    def _event_dict(
        self, connection: sqlite3.Connection, row: sqlite3.Row, owner: ExitStack | None = None
    ) -> dict[str, object]:
        owner = owner if owner is not None else self._require_result_owner()
        payload = self._json_cell(connection, row["event_rowid"], "payload_json", owner)
        refs = self._json_cell(connection, row["event_rowid"], "refs_json", owner)
        if not isinstance(payload, dict):
            raise CaptureJobError(500, "invalid_stored_event_payload")
        return {
            "event_id": row["event_id"],
            "job_id": row["job_id"],
            "event_revision": row["event_revision"],
            "job_revision": row["job_revision"],
            "kind": row["kind"],
            "refs": refs,
            "payload": payload.get("value", payload),
            "request_id": row["request_id"],
            "occurred_at": row["occurred_at"],
        }

    def event(self, job_id: str, body: dict[str, object]) -> dict[str, object]:
        request_id = body.get("request_id")
        expected_revision = body.get("expected_revision")
        kind, refs, payload = body.get("kind"), body.get("refs", {}), body.get("payload", {})
        if not isinstance(kind, str) or not isinstance(request_id, str) or not isinstance(expected_revision, int):
            raise CaptureJobError(400, "invalid_capture_job_event")
        if isinstance(expected_revision, bool) or not isinstance(refs, dict) or not isinstance(payload, dict):
            raise CaptureJobError(400, "invalid_capture_job_event")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._require_scoped(
                connection, job_id, body.get("provider"), body.get("scope"), body.get("client_protocol")
            )
            self._require_live_lease(job_id, row, body)
            existing = connection.execute(
                "SELECT 1 FROM capture_job_events WHERE job_id=? AND request_id=?", (job_id, request_id)
            ).fetchone()
            event = self._append_event(connection, job_id, kind, request_id, expected_revision, refs, payload)
            next_row = connection.execute(
                "SELECT " + _JOB_COLUMNS + " FROM capture_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            return {"event": event, "job": self._summary(connection, next_row), "duplicate": existing is not None}

    def events(self, job_id: str, body: dict[str, object]) -> dict[str, object]:
        limit = body.get("limit", 100)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 500:
            raise CaptureJobError(400, "invalid_event_limit")
        before_revision = body.get("before_revision")
        if before_revision is not None and (
            not isinstance(before_revision, int) or isinstance(before_revision, bool) or before_revision < 0
        ):
            raise CaptureJobError(400, "invalid_event_cursor")
        with self._connection() as connection:
            connection.execute("BEGIN")
            self._require_scoped(
                connection, job_id, body.get("provider"), body.get("scope"), body.get("client_protocol")
            )
            events, next_cursor = read_capture_job_events(
                connection, job_id, limit, before_revision, decode_page=self._event_page
            )
            return {
                "events": events,
                "timelines": project_capture_job_timelines(events),
                "limit": limit,
                "has_more": next_cursor is not None,
                "next_before_revision": next_cursor,
            }

    def _proof(self, job_id: str, lease: dict[str, object]) -> str:
        message = "\0".join(
            (
                "polylogue:capture-lease:v1",
                job_id,
                str(lease["lease_id"]),
                str(lease["generation"]),
                str(lease["request_id"]),
                str(lease["session_id"]),
            )
        )
        return (
            base64.urlsafe_b64encode(hmac.new(self.receiver_id.encode(), message.encode(), hashlib.sha256).digest())
            .rstrip(b"=")
            .decode()
        )

    def adopt(self, job_id: str, body: dict[str, object]) -> dict[str, object]:
        ttl = body.get("lease_ttl_seconds", 120)
        if not isinstance(ttl, int) or not 1 <= ttl <= 300:
            raise CaptureJobError(400, "invalid_lease_ttl")
        request_id, session_id = body.get("request_id"), body.get("session_id")
        if not isinstance(request_id, str) or not isinstance(session_id, str):
            raise CaptureJobError(400, "invalid_lease_request")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._require_scoped(
                connection, job_id, body.get("provider"), body.get("scope"), body.get("client_protocol")
            )
            if self._retirement_eligible(connection, row, _now()):
                raise CaptureJobError(503, "capture_job_retirement_pending")
            lease = json.loads(row["lease_json"]) if row["lease_json"] else None
            generation = lease["generation"] if lease else 0
            if not isinstance(generation, int):
                raise CaptureJobError(409, "invalid_stored_lease")
            if lease and lease["request_id"] == request_id and lease["session_id"] == session_id:
                expires_at = datetime.fromisoformat(str(lease["expires_at"]).replace("Z", "+00:00"))
                if expires_at > _now():
                    return {
                        "job": self._summary(connection, row),
                        "lease": {**lease, "proof": self._proof(job_id, lease)},
                    }
            if body.get("expected_revision") != row["revision"] or body.get("expected_lease_generation") != generation:
                raise CaptureJobError(
                    409, "cas_mismatch", {"revision": row["revision"], "lease_generation": generation}
                )
            if lease and datetime.fromisoformat(lease["expires_at"].replace("Z", "+00:00")) > _now():
                raise CaptureJobError(409, "lease_held")
            now = _now()
            next_lease = {
                "lease_id": str(uuid4()),
                "generation": generation + 1,
                "request_id": request_id,
                "session_id": session_id,
                "expires_at": _stamp(now + timedelta(seconds=ttl)),
            }
            revision = row["revision"] + 1
            connection.execute(
                "UPDATE capture_jobs SET revision=?, lease_json=?, updated_at=? WHERE job_id=?",
                (revision, canonical_json(next_lease), _stamp(now), job_id),
            )
            next_row = connection.execute(
                "SELECT " + _JOB_COLUMNS + " FROM capture_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            return {
                "job": self._summary(connection, next_row),
                "lease": {**next_lease, "proof": self._proof(job_id, next_lease)},
            }

    def _retention_after_retry(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        current: dict[str, object],
        next_retry: dict[str, object],
    ) -> dict[str, object]:
        """Retire a job that just reached a terminal retry state.

        Clients drive retry to ``completed``/``abandoned`` and never send a
        retention object, so without this the receiver's own creation default
        is the only retention any job ever holds and ``gc()`` collects
        nothing. A job that a client has already spoken for keeps what it
        declared. Authoritativeness is read from the evidence that defines it:
        a job still holding conversation-bearing timeline events is the record
        of those conversations and outlives its retry state.
        """
        if next_retry.get("state") not in {"completed", "abandoned"}:
            return current
        declared = connection.execute(
            "SELECT retention_declared FROM capture_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if (
            declared is None
            or declared[0]
            or current != {"state": "active", "hold_reason": None, "timeline_authoritative": True}
        ):
            return current
        return {
            "state": "eligible",
            "hold_reason": None,
            "timeline_authoritative": self._holds_conversation_timeline(connection, job_id),
        }

    def _holds_conversation_timeline(self, connection: sqlite3.Connection, job_id: str) -> bool:
        """Inspect only the lazy scalar timeline ref under one pinned snapshot."""
        cursor = connection.execute("SELECT rowid FROM capture_job_events WHERE job_id=?", (job_id,))
        try:
            for (rowid,) in cursor:
                with ExitStack() as owner:
                    refs = self._json_cell(connection, rowid, "refs_json", owner)
                    ref = refs.get("conversation_ref") if isinstance(refs, dict) else None
                    if isinstance(ref, str) and ref:
                        return True
            return False
        finally:
            cursor.close()

    def _retention_after_checkpoint(
        self, connection: sqlite3.Connection, job_id: str, current: dict[str, object]
    ) -> None:
        """Re-read authoritativeness once the checkpoint's timeline event exists.

        The production extension calls ``update()`` and then ``checkpoint()``
        (``browser-extension/src/background/runtime.js``). For a job that is
        already at a terminal retry state, ``_retention_after_retry`` therefore
        runs while the job holds NO timeline event and records
        ``eligible``/``timeline_authoritative=false`` -- and checkpointing is
        the only route that ever creates one, so the event it appends a moment
        later would never revise that verdict. ``gc()`` then deleted the fresh
        checkpoint, its receipts and that very timeline once the lease expired.

        Authoritativeness is a fact about retained evidence, not a client
        policy choice, so it is recomputed here whatever wrote the retention
        row. Nothing else about the retention is touched: a ``held`` job keeps
        its hold and an ``active`` job keeps its state.
        """
        if current.get("timeline_authoritative") is True:
            return
        if not self._holds_conversation_timeline(connection, job_id):
            return
        connection.execute(
            "UPDATE capture_jobs SET retention_json=? WHERE job_id=?",
            (canonical_json({**current, "timeline_authoritative": True}), job_id),
        )

    def update(self, job_id: str, body: dict[str, object]) -> dict[str, object]:
        request_id = body.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise CaptureJobError(400, "invalid_request_id")
        retry = self._retry(body.get("retry")) if "retry" in body else None
        retention = body.get("retention") if "retention" in body else None
        if retention is not None:
            if not isinstance(retention, dict) or retention.get("state") not in {"active", "held", "eligible"}:
                raise CaptureJobError(400, "invalid_retention_state")
            hold_reason = retention.get("hold_reason")
            timeline_authoritative = retention.get("timeline_authoritative", True)
            if (
                not isinstance(timeline_authoritative, bool)
                or (retention["state"] == "held" and (not isinstance(hold_reason, str) or not hold_reason))
                or (retention["state"] != "held" and hold_reason is not None)
            ):
                raise CaptureJobError(400, "invalid_retention_state")
            retention = {
                "state": retention["state"],
                "hold_reason": hold_reason,
                "timeline_authoritative": timeline_authoritative,
            }
        ttl = body.get("lease_ttl_seconds")
        if ttl is not None and (not isinstance(ttl, int) or isinstance(ttl, bool) or not 1 <= ttl <= 300):
            raise CaptureJobError(400, "invalid_lease_ttl")
        if retry is None and ttl is None and retention is None:
            raise CaptureJobError(400, "empty_capture_job_update")
        request_digest = canonical_digest({"retry": retry, "lease_ttl_seconds": ttl, "retention": retention})
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._require_scoped(
                connection, job_id, body.get("provider"), body.get("scope"), body.get("client_protocol")
            )
            lease = self._require_live_lease(job_id, row, body)
            existing = connection.execute(
                "SELECT request_digest, receipt_json FROM capture_job_update_receipts WHERE job_id=? AND request_id=?",
                (job_id, request_id),
            ).fetchone()
            if existing:
                if not hmac.compare_digest(existing["request_digest"], request_digest):
                    raise CaptureJobError(409, "request_id_conflict")
                return {
                    "job": self._summary(connection, row),
                    "receipt": json.loads(existing["receipt_json"]),
                    "duplicate": True,
                }
            if body.get("expected_revision") != row["revision"]:
                raise CaptureJobError(409, "cas_mismatch", {"revision": row["revision"]})
            current_retry = json.loads(row["retry_json"])
            current_retention = json.loads(row["retention_json"])
            next_retry = retry or current_retry
            next_retention = retention or self._retention_after_retry(connection, job_id, current_retention, next_retry)
            next_lease = dict(lease)
            now = _now()
            if ttl is not None:
                next_lease["expires_at"] = _stamp(now + timedelta(seconds=ttl))
            retention_declaration_changed = retention is not None and not bool(row["retention_declared"])
            if (
                next_retry == current_retry
                and next_retention == current_retention
                and next_lease == lease
                and not retention_declaration_changed
            ):
                receipt = {
                    "receipt_id": str(uuid4()),
                    "request_id": request_id,
                    "job_id": job_id,
                    "kind": "capture_job_update",
                    "revision": row["revision"],
                    "retry": current_retry,
                    "retention": current_retention,
                    "lease_expires_at": lease["expires_at"],
                    "acknowledged_at": _stamp(now),
                    "no_op": True,
                }
                connection.execute(
                    "INSERT INTO capture_job_update_receipts VALUES (?, ?, ?, ?)",
                    (job_id, request_id, request_digest, canonical_json(receipt)),
                )
                return {"job": self._summary(connection, row), "receipt": receipt, "duplicate": True}
            revision = row["revision"] + 1
            receipt = {
                "receipt_id": str(uuid4()),
                "request_id": request_id,
                "job_id": job_id,
                "kind": "capture_job_update",
                "revision": revision,
                "retry": next_retry,
                "retention": next_retention,
                "lease_expires_at": next_lease["expires_at"],
                "acknowledged_at": _stamp(now),
            }
            connection.execute(
                "UPDATE capture_jobs SET revision=?, retry_json=?, retention_json=?, lease_json=?, updated_at=?, retention_declared=MAX(retention_declared, ?) WHERE job_id=?",
                (
                    revision,
                    canonical_json(next_retry),
                    canonical_json(next_retention),
                    canonical_json(next_lease),
                    _stamp(now),
                    int(retention is not None),
                    job_id,
                ),
            )
            connection.execute(
                "INSERT INTO capture_job_update_receipts VALUES (?, ?, ?, ?)",
                (job_id, request_id, request_digest, canonical_json(receipt)),
            )
            next_row = connection.execute(
                "SELECT " + _JOB_COLUMNS + " FROM capture_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            return {"job": self._summary(connection, next_row), "receipt": receipt, "duplicate": False}

    def gc(
        self, *, now: datetime | None = None, limit: int = 100, incremental_artifacts: bool = False
    ) -> dict[str, object]:
        """Explicitly drain eligible jobs; incremental calls perform one page."""
        if not 1 <= limit <= 1000:
            raise CaptureJobError(400, "invalid_gc_limit")
        current = now or _now()
        deleted: list[str] = []
        while len(deleted) < limit:
            result = self._retire_page(current)
            deleted.extend(cast(list[str], result["deleted"]))
            if incremental_artifacts or not result["progress"]:
                break
        self._collect_checkpoint_artifacts((), incremental=incremental_artifacts, quantum=64)
        return {"deleted": deleted, "count": len(deleted)}

    def maintenance_step(self) -> None:
        """One lifecycle turn; physical rows and files retain restart custody."""
        from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError

        try:
            self.gc(incremental_artifacts=True)
        except NativeConnectionSettlementError as exc:
            raise CaptureJobError(503, "capture_job_registry_unsettled") from exc

    def close_maintenance(self) -> None:
        """Release only this receiver store's disposable directory frontier."""
        with _ARTIFACT_SWEEP_LOCK:
            frontier = _ARTIFACT_SWEEPS.pop(str(capture_job_database_path(self.spool_path)), None)
        if frontier is not None:
            with frontier.lock:
                frontier.close()

    def _retirement_eligible(self, connection: sqlite3.Connection, row: sqlite3.Row, current: datetime) -> bool:
        retention = json.loads(row["retention_json"])
        if retention.get("retiring") is True:
            return True
        if retention.get("state") != "eligible" or retention.get("timeline_authoritative", True):
            return False
        if json.loads(row["retry_json"]).get("state") not in {"completed", "abandoned"}:
            return False
        if row["checkpoint_sequence"] is None or not row["receipt_json"]:
            return False
        if connection.execute(
            "SELECT 1 FROM capture_job_native_acquisitions WHERE job_id=? "
            "AND final_receipt_json IS NULL AND state!='cancelled' LIMIT 1",
            (row["job_id"],),
        ).fetchone():
            return False
        lease = self._lease(row)
        if lease is None:
            return True
        try:
            expiry = datetime.fromisoformat(str(lease.get("expires_at")).replace("Z", "+00:00"))
        except ValueError:
            return False
        return expiry.tzinfo is not None and expiry <= current

    @staticmethod
    def _lease_expiry_bounds(current: datetime) -> tuple[str, str, str]:
        current_utc = current.astimezone(UTC)
        prefix = '{"expires_at":"'
        fractional_upper = prefix + current_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ") + '"' + chr(0x10FFFF)
        whole_prefix = prefix + current_utc.strftime("%Y-%m-%dT%H:%M:%SZ") + '"'
        return fractional_upper, whole_prefix, whole_prefix + chr(0x10FFFF)

    def _gc_candidates(
        self, connection: sqlite3.Connection, current: datetime, remaining: Callable[[], int]
    ) -> Iterator[sqlite3.Row]:
        """Page each eligibility index in its own order, past rejected guards."""
        # All producers use canonical expires_at-first UTC JSON; separate
        # fractional and whole-second intervals preserve exact expiry semantics.
        fractional_upper, whole_prefix, whole_upper = self._lease_expiry_bounds(current)
        branches: tuple[tuple[str, tuple[object, ...], str, str], ...] = (
            ("lease_json IS NULL", (), "updated_at, rowid", "updated_at, rowid"),
            (
                "lease_json IS NOT NULL AND lease_json<=?",
                (fractional_upper,),
                "lease_json, updated_at, rowid",
                "lease_json, updated_at, rowid",
            ),
            (
                "lease_json IS NOT NULL AND lease_json>=? AND lease_json<=?",
                (whole_prefix, whole_upper),
                "lease_json, updated_at, rowid",
                "lease_json, updated_at, rowid",
            ),
        )
        for predicate, parameters, order, key_columns in branches:
            after: tuple[object, ...] | None = None
            while remaining() > 0:
                key_filter = (
                    "" if after is None else " AND (" + key_columns + ") > (" + ",".join("?" for _ in after) + ")"
                )
                cursor = connection.execute(
                    "SELECT "
                    + _JOB_COLUMNS
                    + " FROM capture_jobs WHERE "
                    + _GC_PREDICATE
                    + " AND "
                    + predicate
                    + key_filter
                    + " AND NOT EXISTS (SELECT 1 FROM capture_job_native_acquisitions n "
                    "WHERE n.job_id=capture_jobs.job_id AND n.final_receipt_json IS NULL AND n.state!='cancelled')"
                    + " ORDER BY "
                    + order
                    + " LIMIT ?",
                    (*parameters, *(after or ()), remaining()),
                )
                try:
                    rows = cursor.fetchall()
                finally:
                    cursor.close()
                if not rows:
                    break
                for row in rows:
                    after = ((row["lease_json"],) if row["lease_json"] is not None else ()) + (
                        row["updated_at"],
                        row["job_rowid"],
                    )
                    yield row
                    if remaining() <= 0:
                        return

    def _retire_page(self, current: datetime) -> dict[str, object]:
        from polylogue.core.compute_cancel import check_compute_cancelled

        check_compute_cancelled()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for row in self._gc_candidates(connection, current, lambda: 1):
                if not self._retirement_eligible(connection, row, current):
                    continue
                job_id = row["job_id"]
                retention = json.loads(row["retention_json"])
                if retention.get("retiring") is not True:
                    # Existing eligible JSON/index authority is the restart
                    # marker. No remaining lease may revive a retired job.
                    connection.execute(
                        "UPDATE capture_jobs SET retention_json=?, lease_json=NULL WHERE job_id=?",
                        (canonical_json({**retention, "retiring": True}), job_id),
                    )
                # Drain leaves before their parents. No paged parent DELETE may
                # cascade an unbounded native plan or acquisition membership.
                for table in (
                    "capture_job_native_assets",
                    "capture_job_native_members",
                    "capture_job_native_artifacts",
                    "capture_job_native_plan",
                    "capture_job_native_acquisitions",
                    "capture_job_events",
                    "capture_job_receipts",
                    "capture_job_update_receipts",
                ):
                    cursor = connection.execute(
                        "DELETE FROM "
                        + table
                        + " WHERE rowid IN (SELECT rowid FROM "
                        + table
                        + " WHERE job_id=? LIMIT 64)",
                        (job_id,),
                    )
                    if cursor.rowcount:
                        return {"deleted": [], "progress": True}
                connection.execute("DELETE FROM capture_jobs WHERE job_id=?", (job_id,))
                return {"deleted": [job_id], "progress": True}
            return {"deleted": [], "progress": False}

    def _collect_checkpoint_artifacts(self, retired: Iterable[str], *, incremental: bool, quantum: int = 1) -> None:
        directory = capture_job_store_root(self._spool_root()) / "artifacts"
        if not directory.is_dir():
            return
        if not incremental:
            from itertools import chain

            with self._connection() as connection, os.scandir(directory) as entries:
                connection.execute("BEGIN IMMEDIATE")
                for name in chain(retired, (entry.name for entry in entries)):
                    self._collect_checkpoint_artifact(connection, directory, name)
            return
        identity = _database_identity(capture_job_database_path(self.spool_path))
        if identity is None:
            return
        status = directory.stat()
        physical = (identity[1], identity[2], status.st_dev, status.st_ino)
        replaced: _ArtifactFrontier | None = None
        with _ARTIFACT_SWEEP_LOCK:
            frontier = _ARTIFACT_SWEEPS.get(identity[0])
            if frontier is not None and frontier.physical != physical:
                replaced = _ARTIFACT_SWEEPS.pop(identity[0])
                frontier = None
            if frontier is None:
                frontier = _ArtifactFrontier(physical, os.scandir(directory))
                _ARTIFACT_SWEEPS[identity[0]] = frontier
        if replaced is not None:
            with replaced.lock:
                replaced.close()
        with frontier.lock:
            # A directory cursor is not a receipt. Keep its bounded pending
            # names across BEGIN/query/IO failure and settle successful checks
            # individually. Restart can safely rescan the physical namespace.
            if not frontier.exhausted:
                try:
                    while len(frontier.pending) < quantum:
                        frontier.pending.append(next(frontier.entries).name)
                except StopIteration:
                    frontier.exhausted = True
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                for name in tuple(frontier.pending):
                    if self._collect_checkpoint_artifact(connection, directory, name):
                        frontier.pending.remove(name)
            if frontier.exhausted and not frontier.pending:
                with _ARTIFACT_SWEEP_LOCK:
                    if _ARTIFACT_SWEEPS.get(identity[0]) is frontier:
                        del _ARTIFACT_SWEEPS[identity[0]]
                frontier.close()

    def _collect_checkpoint_artifact(self, connection: sqlite3.Connection, directory: Path, name: str) -> bool:
        """Return true only after this entry's root/physical check settles."""
        if name.endswith(".native"):
            digest = name.removesuffix(".native")
            rooted = connection.execute(
                "SELECT 1 FROM capture_job_native_members WHERE sha256=? "
                "UNION ALL SELECT 1 FROM capture_job_native_assets WHERE sha256=? "
                "UNION ALL SELECT 1 FROM capture_job_native_artifacts WHERE sha256=? LIMIT 1",
                (digest, digest, digest),
            ).fetchone()
        elif name.endswith(".checkpoint"):
            digest = "sha256:" + name.removesuffix(".checkpoint")
            rooted = connection.execute(
                "SELECT 1 FROM capture_jobs WHERE checkpoint_artifact_ref=? "
                "UNION ALL SELECT 1 FROM capture_job_receipts WHERE checkpoint_digest=? LIMIT 1",
                (digest, digest),
            ).fetchone()
        else:
            return True
        if rooted:
            return True
        try:
            with open(directory / name, "rb") as handle:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return False
                if os.stat(directory / name).st_ino == os.fstat(handle.fileno()).st_ino:
                    os.unlink(directory / name)
        except FileNotFoundError:
            return True
        return True

    def _native_artifact_path(self, sha256: object) -> Path:
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise CaptureJobError(400, "invalid_native_digest")
        try:
            if len(bytes.fromhex(sha256)) != 32 or sha256 != sha256.lower():
                raise ValueError
        except ValueError as exc:
            raise CaptureJobError(400, "invalid_native_digest") from exc
        return capture_job_store_root(self._spool_root()) / "artifacts" / (sha256 + ".native")

    def _publish_native_artifact(self, staged: StagedCapture) -> None:
        target = self._native_artifact_path(staged.sha256)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(staged.path, target)
        except FileExistsError:
            with target.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != staged.sha256:
                    raise CaptureJobError(500, "native_artifact_corrupt") from None
        with target.open("rb") as handle:
            os.fsync(handle.fileno())
        for parent in (target.parent, target.parent.parent, self._spool_root()):
            fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def _native_row(
        self, connection: sqlite3.Connection, job_id: str, body: dict[str, object]
    ) -> tuple[sqlite3.Row, sqlite3.Row]:
        job = self._require_scoped(
            connection, job_id, body.get("provider"), body.get("scope"), body.get("client_protocol")
        )
        self._require_live_lease(job_id, job, body)
        row = connection.execute(
            "SELECT rowid AS acquisition_rowid, job_id, acquisition_id, binding_json, member_names_json, state, plan_digest, final_receipt_json FROM capture_job_native_acquisitions WHERE job_id=? AND acquisition_id=?",
            (job_id, body.get("acquisition_id")),
        ).fetchone()
        if row is None:
            raise CaptureJobError(404, "native_acquisition_not_found")
        if row["state"] == "cancelled":
            raise CaptureJobError(409, "native_acquisition_cancelled")
        return job, row

    @contextmanager
    def native_member_artifact(
        self,
        job_id: str,
        member_name: str,
        body: dict[str, object],
        *,
        progress: Callable[[], None] | None = None,
    ) -> Iterator[tuple[BinaryIO, sqlite3.Row]]:
        """Borrow exactly rooted raw bytes without holding the writer during IO."""
        handle = None
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                _, acquisition = self._native_row(connection, job_id, body)
                member = connection.execute(
                    "SELECT rowid AS member_rowid, job_id, acquisition_id, member_name, sha256, size_bytes FROM capture_job_native_members WHERE job_id=? AND acquisition_id=? AND member_name=?",
                    (job_id, acquisition["acquisition_id"], member_name),
                ).fetchone()
                if member is None:
                    raise CaptureJobError(404, "native_member_not_owned")
                try:
                    handle = self._native_artifact_path(member["sha256"]).open("rb")
                except FileNotFoundError as exc:
                    raise CaptureJobError(500, "native_artifact_missing") from exc
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            hasher = hashlib.sha256()
            while chunk := handle.read(64 * 1024):
                hasher.update(chunk)
                if progress is not None:
                    progress()
            if os.fstat(handle.fileno()).st_size != member["size_bytes"] or hasher.hexdigest() != member["sha256"]:
                raise CaptureJobError(500, "native_artifact_corrupt")
            handle.seek(0)
            yield handle, member
        finally:
            if handle is not None:
                handle.close()

    def native_begin(self, job_id: str, body: dict[str, object]) -> dict[str, object]:
        acquisition_id, binding = body.get("acquisition_id"), body.get("binding")
        member_names = body.get("member_names")
        if not isinstance(member_names, list) or any(not isinstance(name, str) for name in member_names):
            raise CaptureJobError(400, "invalid_native_members")
        if len(set(member_names)) != len(member_names):
            raise CaptureJobError(400, "invalid_native_members")
        members_json = json.dumps(sorted(member_names), separators=(",", ":"))
        if not isinstance(acquisition_id, str) or not isinstance(binding, dict):
            raise CaptureJobError(400, "invalid_native_acquisition")
        try:
            UUID(acquisition_id)
        except ValueError as exc:
            raise CaptureJobError(400, "invalid_native_acquisition") from exc
        self._creation_scope(
            body.get("provider"),
            {"kind": "invocation", "creation_token": acquisition_id, "binding": binding},
            body.get("client_protocol"),
        )
        binding_json = json.dumps({key: binding[key] for key in binding}, separators=(",", ":"), sort_keys=True)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = self._require_scoped(
                connection, job_id, body.get("provider"), body.get("scope"), body.get("client_protocol")
            )
            self._require_live_lease(job_id, job, body)
            if json.loads(job["retry_json"])["state"] in {"held", "abandoned"}:
                raise CaptureJobError(409, "capture_authority_paused")
            accepted_names = {"conversation"}
            if job["provider"] == "grok":
                accepted_names.add("responses")
                if "response_nodes" in member_names:
                    accepted_names.add("response_nodes")
            if set(member_names) != accepted_names:
                raise CaptureJobError(400, "invalid_native_members")
            if job["scope_kind"] == "invocation" and json.loads(job["invocation_json"])["binding"] != binding:
                raise CaptureJobError(409, "invocation_binding_conflict")
            existing = connection.execute(
                "SELECT rowid AS acquisition_rowid, job_id, acquisition_id, binding_json, member_names_json, state, plan_digest, final_receipt_json FROM capture_job_native_acquisitions WHERE job_id=? AND acquisition_id=?",
                (job_id, acquisition_id),
            ).fetchone()
            if existing is not None:
                if existing["binding_json"] != binding_json or existing["member_names_json"] != members_json:
                    raise CaptureJobError(409, "native_acquisition_conflict")
                return {
                    "job": self._summary(connection, job),
                    "acquisition_id": acquisition_id,
                    "state": existing["state"],
                    "duplicate": True,
                }
            if body.get("expected_revision") != job["revision"]:
                raise CaptureJobError(409, "cas_mismatch", {"revision": job["revision"]})
            connection.execute(
                "INSERT INTO capture_job_native_acquisitions(job_id, acquisition_id, binding_json, member_names_json, state) VALUES (?, ?, ?, ?, 'acquiring')",
                (job_id, acquisition_id, binding_json, members_json),
            )
            return {
                "job": self._summary(connection, job),
                "acquisition_id": acquisition_id,
                "state": "acquiring",
                "duplicate": False,
            }

    def native_member(self, job_id: str, body: dict[str, object], staged: StagedCapture) -> dict[str, object]:
        member, metadata = body.get("member_name"), body.get("metadata")
        if member not in {"conversation", "responses", "response_nodes"} or not isinstance(metadata, dict):
            raise CaptureJobError(400, "invalid_native_member")
        if body.get("sha256") != staged.sha256 or body.get("size_bytes") != staged.size_bytes:
            raise CaptureJobError(400, "native_member_integrity_mismatch")
        metadata_digest = _native_json_digest(metadata)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job, acquisition = self._native_row(connection, job_id, body)
            if json.loads(job["retry_json"])["state"] in {"held", "abandoned"}:
                raise CaptureJobError(409, "capture_authority_paused")
            if member not in json.loads(acquisition["member_names_json"]):
                raise CaptureJobError(400, "invalid_native_member")
            existing = connection.execute(
                "SELECT rowid AS member_rowid, job_id, acquisition_id, member_name, sha256, size_bytes FROM capture_job_native_members WHERE job_id=? AND acquisition_id=? AND member_name=?",
                (job_id, acquisition["acquisition_id"], member),
            ).fetchone()
            if existing is not None:
                with ExitStack() as metadata_owner:
                    stored = self._json_cell(
                        connection,
                        existing["member_rowid"],
                        "metadata_json",
                        metadata_owner,
                        table="capture_job_native_members",
                    )
                    if (existing["sha256"], existing["size_bytes"]) != (
                        staged.sha256,
                        staged.size_bytes,
                    ) or _native_json_digest(stored) != metadata_digest:
                        raise CaptureJobError(409, "native_member_conflict")
                self._publish_native_artifact(staged)
                return {
                    "job": self._summary(connection, job),
                    "acquisition_id": acquisition["acquisition_id"],
                    "member_name": member,
                    "sha256": staged.sha256,
                    "size_bytes": staged.size_bytes,
                    "duplicate": True,
                }
            if acquisition["state"] != "acquiring":
                raise CaptureJobError(409, "native_acquisition_sealed")
            if body.get("expected_revision") != job["revision"]:
                raise CaptureJobError(409, "cas_mismatch", {"revision": job["revision"]})
            self._publish_native_artifact(staged)
            cursor = connection.execute(
                "INSERT INTO capture_job_native_members VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, acquisition["acquisition_id"], member, staged.sha256, staged.size_bytes, ""),
            )
            member_rowid = cast(int, cursor.lastrowid)
            cursor.close()
            self._write_json_cell(
                connection,
                member_rowid,
                "metadata_json",
                metadata,
                table="capture_job_native_members",
                native_json=True,
            )
            return {
                "job": self._summary(connection, job),
                "acquisition_id": acquisition["acquisition_id"],
                "member_name": member,
                "sha256": staged.sha256,
                "size_bytes": staged.size_bytes,
                "duplicate": False,
            }

    def native_cancel(self, job_id: str, body: dict[str, object]) -> dict[str, object]:
        """Fence one acquisition while retaining every committed artifact."""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = self._require_scoped(
                connection, job_id, body.get("provider"), body.get("scope"), body.get("client_protocol")
            )
            self._require_live_lease(job_id, job, body)
            acquisition = connection.execute(
                "SELECT rowid AS acquisition_rowid, job_id, acquisition_id, binding_json, member_names_json, state, plan_digest, final_receipt_json FROM capture_job_native_acquisitions WHERE job_id=? AND acquisition_id=?",
                (job_id, body.get("acquisition_id")),
            ).fetchone()
            if acquisition is None:
                raise CaptureJobError(404, "native_acquisition_not_found")
            if acquisition["final_receipt_json"]:
                raise CaptureJobError(409, "native_acquisition_published")
            duplicate = acquisition["state"] == "cancelled"
            connection.execute(
                "UPDATE capture_job_native_acquisitions SET state='cancelled' WHERE job_id=? AND acquisition_id=?",
                (job_id, acquisition["acquisition_id"]),
            )
            return {"acquisition_id": acquisition["acquisition_id"], "state": "cancelled", "duplicate": duplicate}

    def native_prepare(self, job_id: str, body: dict[str, object]) -> dict[str, object]:
        """Seal canonical turns and an exact asset plan, retaining literal raw."""
        from polylogue.browser_capture.models import BrowserCaptureProvenance
        from polylogue.browser_capture.native_preparation import envelope_prefix
        from polylogue.core.enums import Provider
        from polylogue.core.sql_settlement import retain_native_sql_lifetimes
        from polylogue.sources.parsers.browser_capture import parse_native_member_streams
        from polylogue.sources.prepared_message_sink import ScratchSessionSpill, SqliteMessageStore
        from polylogue.storage.sqlite.connection_profile import (
            retained_native_sql_owners_for_lifetime,
        )

        with self.artifact_progress(job_id, body, native=True) as progress:
            with self._connection() as connection:
                job, acquisition = self._native_row(connection, job_id, body)
                binding = json.loads(acquisition["binding_json"])
                names = json.loads(acquisition["member_names_json"])
                prepared = connection.execute(
                    "SELECT * FROM capture_job_native_artifacts WHERE job_id=? AND acquisition_id=? AND purpose='prefix'",
                    (job_id, acquisition["acquisition_id"]),
                ).fetchone()
                members_bound = {
                    row["member_name"]: {"sha256": row["sha256"], "size_bytes": row["size_bytes"]}
                    for row in connection.execute(
                        "SELECT member_name, sha256, size_bytes FROM capture_job_native_members WHERE job_id=? AND acquisition_id=?",
                        (job_id, acquisition["acquisition_id"]),
                    )
                }
                revision = canonical_digest(
                    {"provider": job["provider"], "native_id": binding["native_id"], "members": members_bound}
                )
                if set(members_bound) != set(names) or binding["raw_revision"] != revision:
                    raise CaptureJobError(409, "native_raw_revision_conflict")
                provenance = BrowserCaptureProvenance.model_validate(body.get("provenance"))
                if any(
                    getattr(provenance, field) != binding[field]
                    for field in ("source_url", "extension_instance_id", "acquisition_sequence")
                ):
                    raise CaptureJobError(409, "native_provenance_conflict")
                metadata = body.get("provider_meta", {})
                if not isinstance(metadata, dict):
                    raise CaptureJobError(400, "invalid_native_metadata")
                header = {"provenance": provenance.model_dump(mode="json"), "provider_meta": metadata}

                if prepared is not None:
                    with ExitStack() as header_owner:
                        stored_header = cast(
                            dict[str, object],
                            self._json_cell(
                                connection,
                                acquisition["acquisition_rowid"],
                                "header_json",
                                header_owner,
                                table="capture_job_native_acquisitions",
                            ),
                        )
                        if _native_json_digest(stored_header["request"]) != _native_json_digest(header):
                            raise CaptureJobError(409, "native_preparation_conflict")
                        summary = cast(dict[str, object], stored_header["summary"])
                        return {
                            "job": self._summary(connection, job),
                            "acquisition_id": acquisition["acquisition_id"],
                            "state": acquisition["state"],
                            "plan_digest": acquisition["plan_digest"],
                            "summary": {key: summary[key] for key in summary},
                            "duplicate": True,
                        }
            scratch_root = capture_job_store_root(self._spool_root()) / "preparation"
            scratch_root.mkdir(parents=True, exist_ok=True)
            directory = tempfile.TemporaryDirectory(prefix="native-", dir=scratch_root)
            store = None
            staged = None
            try:
                with retain_native_sql_lifetimes(directory):
                    store = SqliteMessageStore(Path(directory.name) / "messages.sqlite")
                    store.conn.execute(
                        "CREATE TABLE capture_preparation_plan (ordinal INTEGER PRIMARY KEY, descriptor_json TEXT NOT NULL)"
                    )
                    spill = ScratchSessionSpill(store)
                    with ExitStack() as stack:
                        members = {
                            name: stack.enter_context(
                                self.native_member_artifact(job_id, name, body, progress=progress)
                            )[0]
                            for name in names
                        }
                        parsed = parse_native_member_streams(
                            Provider.from_string(job["provider"]),
                            members,
                            binding["native_id"],
                            spill,
                            progress=progress,
                        )
                        # Full provider prose stays in the on-disk envelope; the
                        # browser needs only acquisition/control summary facts.
                        info = {
                            "title": None,
                            "turn_count": len(parsed.messages),
                            "attachment_count": len(parsed.attachments),
                            "session_kind": "temporary" if parsed.session_kind.value == "temporary" else "standard",
                            "needs_follow_up": parsed.source_name is Provider.CHATGPT,
                        }
                        if parsed.source_name is Provider.CHATGPT:
                            for message in parsed.messages:
                                progress()
                                if message.is_active_leaf and not message.active_leaf_fallback:
                                    info["needs_follow_up"] = (
                                        message.role.value != "assistant"
                                        or message.delivery_status
                                        not in {"finished_successfully", "finished", "complete", "completed"}
                                    )
                        sealed_header = {"request": header, "summary": info}
                        staged = self._stage_json_chunks(
                            envelope_prefix(parsed, spill, members, provenance, metadata, progress), durable=True
                        )
                    plan_hash = hashlib.sha256()
                    for ordinal, descriptor in store.conn.execute(
                        "SELECT ordinal, descriptor_json FROM capture_preparation_plan ORDER BY ordinal"
                    ):
                        progress()
                        plan_hash.update(str(ordinal).encode("ascii") + b"\0" + descriptor.encode("ascii") + b"\n")
                    plan_digest = "sha256:" + plan_hash.hexdigest()
                    progress()
                    with self._connection() as connection:
                        connection.execute("BEGIN IMMEDIATE")
                        job, acquisition = self._native_row(connection, job_id, body)
                        if json.loads(job["retry_json"])["state"] in {"held", "abandoned"}:
                            raise CaptureJobError(409, "capture_authority_paused")
                        if acquisition["state"] != "acquiring":
                            with ExitStack() as header_owner:
                                stored_header = cast(
                                    dict[str, object],
                                    self._json_cell(
                                        connection,
                                        acquisition["acquisition_rowid"],
                                        "header_json",
                                        header_owner,
                                        table="capture_job_native_acquisitions",
                                    ),
                                )
                                if acquisition["plan_digest"] != plan_digest or _native_json_digest(
                                    stored_header
                                ) != _native_json_digest(sealed_header):
                                    raise CaptureJobError(409, "native_preparation_conflict")
                            return {
                                "job": self._summary(connection, job),
                                "acquisition_id": acquisition["acquisition_id"],
                                "state": acquisition["state"],
                                "plan_digest": plan_digest,
                                "summary": info,
                                "duplicate": True,
                            }
                        self._publish_native_artifact(staged)
                        connection.execute(
                            "INSERT INTO capture_job_native_artifacts VALUES (?, ?, 'prefix', ?, ?)",
                            (job_id, acquisition["acquisition_id"], staged.sha256, staged.size_bytes),
                        )
                        for ordinal, descriptor in store.conn.execute(
                            "SELECT ordinal, descriptor_json FROM capture_preparation_plan ORDER BY ordinal"
                        ):
                            connection.execute(
                                "INSERT INTO capture_job_native_plan VALUES (?, ?, ?, ?, ?)",
                                (
                                    job_id,
                                    acquisition["acquisition_id"],
                                    ordinal,
                                    descriptor,
                                    "sha256:" + hashlib.sha256(descriptor.encode("ascii")).hexdigest(),
                                ),
                            )
                        connection.execute(
                            "UPDATE capture_job_native_acquisitions SET state='prepared', plan_digest=? WHERE job_id=? AND acquisition_id=?",
                            (plan_digest, job_id, acquisition["acquisition_id"]),
                        )
                        self._write_json_cell(
                            connection,
                            acquisition["acquisition_rowid"],
                            "header_json",
                            sealed_header,
                            table="capture_job_native_acquisitions",
                            native_json=True,
                        )
                        return {
                            "job": self._summary(connection, job),
                            "acquisition_id": acquisition["acquisition_id"],
                            "state": "prepared",
                            "plan_digest": plan_digest,
                            "summary": info,
                            "duplicate": False,
                        }
            finally:
                try:
                    if staged is not None:
                        staged.discard()
                finally:
                    try:
                        if store is not None:
                            store.close()
                    finally:
                        # Original native owners retain the exact directory on
                        # failed close; no caller cleanup can retire their bytes.
                        if not retained_native_sql_owners_for_lifetime(directory):
                            directory.cleanup()

    def native_plan(self, job_id: str, body: dict[str, object]) -> dict[str, object]:
        after = body.get("after", -1)
        if type(after) is not int or after < -1:
            raise CaptureJobError(400, "invalid_native_plan_cursor")
        with self._connection() as connection:
            connection.execute("BEGIN")
            _job, acquisition = self._native_row(connection, job_id, body)
            if acquisition["plan_digest"] is None:
                raise CaptureJobError(409, "native_preparation_pending")
            rows = connection.execute(
                "SELECT p.ordinal, p.descriptor_json, p.descriptor_digest, a.rowid AS asset_rowid FROM capture_job_native_plan p LEFT JOIN capture_job_native_assets a USING(job_id, acquisition_id, ordinal) WHERE p.job_id=? AND p.acquisition_id=? AND p.ordinal>? ORDER BY p.ordinal LIMIT 65",
                (job_id, acquisition["acquisition_id"], after),
            ).fetchall()
            more = len(rows) > 64
            rows = rows[:64]
            return {
                "acquisition_id": acquisition["acquisition_id"],
                "plan_digest": acquisition["plan_digest"],
                "assets": [
                    {
                        "ordinal": row["ordinal"],
                        "descriptor": json.loads(row["descriptor_json"]),
                        "descriptor_digest": row["descriptor_digest"],
                        "receipt": self._json_cell(
                            connection,
                            row["asset_rowid"],
                            "outcome_json",
                            self._require_result_owner(),
                            table="capture_job_native_assets",
                        )
                        if row["asset_rowid"] is not None
                        else None,
                    }
                    for row in rows
                ],
                "after": rows[-1]["ordinal"] if more else None,
            }

    def native_asset(
        self, job_id: str, body: dict[str, object], staged: StagedCapture | None = None
    ) -> dict[str, object]:
        ordinal, outcome = body.get("ordinal"), body.get("outcome")
        if type(ordinal) is not int or ordinal < 0 or not isinstance(outcome, dict):
            raise CaptureJobError(400, "invalid_native_asset_receipt")
        status = outcome.get("status")
        if not isinstance(status, str) or not status or (status == "acquired") != (staged is not None):
            raise CaptureJobError(400, "invalid_native_asset_receipt")
        if staged is not None and (body.get("sha256"), body.get("size_bytes")) != (staged.sha256, staged.size_bytes):
            raise CaptureJobError(400, "native_asset_integrity_mismatch")
        outcome_digest = _native_json_digest(outcome)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job, acquisition = self._native_row(connection, job_id, body)
            if json.loads(job["retry_json"])["state"] in {"held", "abandoned"}:
                raise CaptureJobError(409, "capture_authority_paused")
            if body.get("plan_digest") != acquisition["plan_digest"] or acquisition["plan_digest"] is None:
                raise CaptureJobError(409, "native_plan_conflict")
            plan = connection.execute(
                "SELECT descriptor_digest, descriptor_json FROM capture_job_native_plan WHERE job_id=? AND acquisition_id=? AND ordinal=?",
                (job_id, acquisition["acquisition_id"], ordinal),
            ).fetchone()
            if plan is None or body.get("descriptor_digest") != plan["descriptor_digest"]:
                raise CaptureJobError(409, "native_asset_occurrence_conflict")
            if status == "retained_native_bytes":
                metadata = json.loads(plan["descriptor_json"])["provider_meta"]
                if (outcome.get("sha256"), outcome.get("size_bytes")) != (
                    metadata.get("native_inline_sha256"),
                    metadata.get("native_inline_size_bytes"),
                ) or metadata.get("native_inline_sha256") is None:
                    raise CaptureJobError(409, "native_inline_receipt_conflict")
            existing = connection.execute(
                "SELECT rowid AS asset_rowid, job_id, acquisition_id, ordinal, sha256, size_bytes FROM capture_job_native_assets WHERE job_id=? AND acquisition_id=? AND ordinal=?",
                (job_id, acquisition["acquisition_id"], ordinal),
            ).fetchone()
            sha, size = (staged.sha256, staged.size_bytes) if staged is not None else (None, None)
            if existing is not None:
                with ExitStack() as outcome_owner:
                    stored = self._json_cell(
                        connection,
                        existing["asset_rowid"],
                        "outcome_json",
                        outcome_owner,
                        table="capture_job_native_assets",
                    )
                    if (existing["sha256"], existing["size_bytes"]) != (sha, size) or _native_json_digest(
                        stored
                    ) != outcome_digest:
                        raise CaptureJobError(409, "native_asset_receipt_conflict")
                if staged is not None:
                    self._publish_native_artifact(staged)
                return {"ordinal": ordinal, "plan_digest": acquisition["plan_digest"], "duplicate": True}
            if acquisition["state"] != "prepared":
                raise CaptureJobError(409, "native_acquisition_sealed")
            if staged is not None:
                self._publish_native_artifact(staged)
            cursor = connection.execute(
                "INSERT INTO capture_job_native_assets VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, acquisition["acquisition_id"], ordinal, "", sha, size),
            )
            asset_rowid = cast(int, cursor.lastrowid)
            cursor.close()
            self._write_json_cell(
                connection, asset_rowid, "outcome_json", outcome, table="capture_job_native_assets", native_json=True
            )
            return {"ordinal": ordinal, "plan_digest": acquisition["plan_digest"], "duplicate": False}

    @contextmanager
    def native_artifact(
        self,
        job_id: str,
        body: dict[str, object],
        *,
        purpose: str | None = None,
        ordinal: int | None = None,
        progress: Callable[[], None] | None = None,
    ) -> Iterator[tuple[BinaryIO, sqlite3.Row]]:
        handle = None
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                _job, acquisition = self._native_row(connection, job_id, body)
                if purpose is not None:
                    artifact = connection.execute(
                        "SELECT * FROM capture_job_native_artifacts WHERE job_id=? AND acquisition_id=? AND purpose=?",
                        (job_id, acquisition["acquisition_id"], purpose),
                    ).fetchone()
                else:
                    artifact = connection.execute(
                        "SELECT * FROM capture_job_native_assets WHERE job_id=? AND acquisition_id=? AND ordinal=? AND sha256 IS NOT NULL",
                        (job_id, acquisition["acquisition_id"], ordinal),
                    ).fetchone()
                if artifact is None:
                    raise CaptureJobError(404, "native_artifact_not_owned")
                try:
                    handle = self._native_artifact_path(artifact["sha256"]).open("rb")
                except FileNotFoundError as exc:
                    raise CaptureJobError(500, "native_artifact_missing") from exc
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            hasher = hashlib.sha256()
            while chunk := handle.read(64 * 1024):
                hasher.update(chunk)
                if progress is not None:
                    progress()
            if os.fstat(handle.fileno()).st_size != artifact["size_bytes"] or hasher.hexdigest() != artifact["sha256"]:
                raise CaptureJobError(500, "native_artifact_corrupt")
            handle.seek(0)
            yield handle, artifact
        finally:
            if handle is not None:
                handle.close()

    def native_finalize(self, job_id: str, body: dict[str, object]) -> dict[str, object]:
        """Seal only the exact prepared plan's terminal, occurrence-bound assets."""
        from polylogue.browser_capture.native_preparation import json_chunks, raw_chunks

        with self.artifact_progress(job_id, body, native=True) as progress:
            with self._connection() as connection:
                job, acquisition = self._native_row(connection, job_id, body)
                if acquisition["plan_digest"] is None or body.get("plan_digest") != acquisition["plan_digest"]:
                    raise CaptureJobError(409, "native_plan_conflict")
                final = connection.execute(
                    "SELECT * FROM capture_job_native_artifacts WHERE job_id=? AND acquisition_id=? AND purpose='final'",
                    (job_id, acquisition["acquisition_id"]),
                ).fetchone()
                if final is not None:
                    return {
                        "acquisition_id": acquisition["acquisition_id"],
                        "sha256": final["sha256"],
                        "size_bytes": final["size_bytes"],
                        "duplicate": True,
                    }
                missing = connection.execute(
                    "SELECT 1 FROM capture_job_native_plan p LEFT JOIN capture_job_native_assets a USING(job_id, acquisition_id, ordinal) WHERE p.job_id=? AND p.acquisition_id=? AND a.ordinal IS NULL LIMIT 1",
                    (job_id, acquisition["acquisition_id"]),
                ).fetchone()
                if missing is not None:
                    raise CaptureJobError(409, "native_asset_receipts_pending")

            def chunks() -> Iterator[bytes]:
                with self.native_artifact(job_id, body, purpose="prefix", progress=progress) as (prefix, _row):
                    yield from raw_chunks(prefix, progress)
                after = -1
                first = True
                while True:
                    progress()
                    with ExitStack() as page_owner:
                        with self._connection() as connection:
                            connection.execute("BEGIN")
                            rows = connection.execute(
                                "SELECT p.ordinal, p.descriptor_json, a.rowid AS asset_rowid, a.sha256, a.size_bytes FROM capture_job_native_plan p JOIN capture_job_native_assets a USING(job_id, acquisition_id, ordinal) WHERE p.job_id=? AND p.acquisition_id=? AND p.ordinal>? ORDER BY p.ordinal LIMIT 64",
                                (job_id, acquisition["acquisition_id"], after),
                            ).fetchall()
                            outcomes = [
                                self._json_cell(
                                    connection,
                                    row["asset_rowid"],
                                    "outcome_json",
                                    page_owner,
                                    table="capture_job_native_assets",
                                )
                                for row in rows
                            ]
                        if not rows:
                            break
                        for row, outcome in zip(rows, outcomes, strict=True):
                            progress()
                            after = row["ordinal"]
                            descriptor = json.loads(row["descriptor_json"])
                            descriptor.pop("original_record_ordinal", None)
                            descriptor.pop("original_record_key", None)
                            descriptor["provider_meta"]["asset_acquisition"] = outcome
                            if not first:
                                yield b","
                            first = False
                            if row["sha256"] is None:
                                yield from json_chunks(descriptor)
                            else:
                                descriptor["size_bytes"] = row["size_bytes"]
                                descriptor["provider_meta"]["content_sha256"] = row["sha256"]
                                parts = iter(json_chunks(descriptor))
                                pending = next(parts)
                                for piece in parts:
                                    yield pending
                                    pending = piece
                                if pending != b"}":
                                    raise RuntimeError("native attachment serializer lost object boundary")
                                yield b',"content_base64":"'
                                with self.native_artifact(job_id, body, ordinal=after, progress=progress) as (
                                    asset,
                                    _asset_row,
                                ):
                                    while block := asset.read(65535):
                                        progress()
                                        yield base64.b64encode(block)
                                yield b'"}'
                yield b"]}}"

            staged = self._stage_json_chunks(chunks(), durable=True)
            try:
                progress()
                with self._connection() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    job, acquisition = self._native_row(connection, job_id, body)
                    if json.loads(job["retry_json"])["state"] in {"held", "abandoned"}:
                        raise CaptureJobError(409, "capture_authority_paused")
                    final = connection.execute(
                        "SELECT sha256, size_bytes FROM capture_job_native_artifacts WHERE job_id=? AND acquisition_id=? AND purpose='final'",
                        (job_id, acquisition["acquisition_id"]),
                    ).fetchone()
                    if final is not None:
                        if (final["sha256"], final["size_bytes"]) != (staged.sha256, staged.size_bytes):
                            raise CaptureJobError(409, "native_final_artifact_conflict")
                    else:
                        self._publish_native_artifact(staged)
                        connection.execute(
                            "INSERT INTO capture_job_native_artifacts VALUES (?, ?, 'final', ?, ?)",
                            (job_id, acquisition["acquisition_id"], staged.sha256, staged.size_bytes),
                        )
                        connection.execute(
                            "UPDATE capture_job_native_acquisitions SET state='sealed' WHERE job_id=? AND acquisition_id=?",
                            (job_id, acquisition["acquisition_id"]),
                        )
                return {
                    "acquisition_id": acquisition["acquisition_id"],
                    "sha256": staged.sha256,
                    "size_bytes": staged.size_bytes,
                    "duplicate": final is not None,
                }
            finally:
                staged.discard()

    def native_publish(
        self, job_id: str, body: dict[str, object], admit: Callable[[StagedCapture, CaptureSummary], dict[str, object]]
    ) -> dict[str, object]:
        """Record exact final admission under the same acquisition lease fence."""
        with (
            self.artifact_progress(job_id, body, native=True) as progress,
            self.native_artifact(job_id, body, purpose="final", progress=progress) as (handle, artifact),
        ):
            staged = stage_retained_capture(
                handle,
                self._native_artifact_path(artifact["sha256"]),
                size_bytes=artifact["size_bytes"],
                sha256=artifact["sha256"],
                spool_root=self._spool_root(),
            )
            try:
                summary = summarize_capture_file(staged.path)
                progress()
                with self._connection() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    job, acquisition = self._native_row(connection, job_id, body)
                    if json.loads(job["retry_json"])["state"] in {"held", "abandoned"}:
                        raise CaptureJobError(409, "capture_authority_paused")
                    if (
                        body.get("plan_digest") != acquisition["plan_digest"]
                        or body.get("sha256") != artifact["sha256"]
                    ):
                        raise CaptureJobError(409, "native_final_artifact_conflict")
                    if acquisition["final_receipt_json"]:
                        return cast(dict[str, object], json.loads(acquisition["final_receipt_json"]))
                    payload = admit(staged, summary)
                    connection.execute(
                        "UPDATE capture_job_native_acquisitions SET state='published', final_receipt_json=? WHERE job_id=? AND acquisition_id=?",
                        (json.dumps(payload, separators=(",", ":")), job_id, acquisition["acquisition_id"]),
                    )
                    return payload
            finally:
                staged.discard()

    def checkpoint(self, job_id: str, body: dict[str, object], staged: StagedCapture) -> dict[str, object]:
        checkpoint = body.get("checkpoint")
        if (
            not isinstance(checkpoint, dict)
            or type(checkpoint.get("sequence")) is not int
            or checkpoint["sequence"] < 0
            or checkpoint["sequence"] > (1 << 53) - 1
            or "payload" in checkpoint
        ):
            raise CaptureJobError(400, "invalid_checkpoint")
        with staged.path.open("rb") as handle:
            semantic_digest, conversation_ref = _canonical_checkpoint_digest(handle)
        if semantic_digest != "sha256:" + staged.sha256 or checkpoint.get("digest") != semantic_digest:
            raise CaptureJobError(400, "checkpoint_digest_mismatch")
        request_id = body.get("request_id")
        if not isinstance(request_id, str):
            raise CaptureJobError(400, "invalid_request_id")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._require_scoped(
                connection, job_id, body.get("provider"), body.get("scope"), body.get("client_protocol")
            )
            self._require_live_lease(job_id, row, body)
            # Custody is durable before either a new ref or a duplicate receipt
            # is acknowledged. Retry after directory-fsync failure resyncs the
            # existing immutable file instead of returning an unsafe ACK.
            artifact_ref = self._publish_checkpoint_artifact(staged, semantic_digest)
            receipt_row = connection.execute(
                "SELECT receipt_json, checkpoint_sequence, checkpoint_digest FROM capture_job_receipts WHERE job_id=? AND request_id=?",
                (job_id, request_id),
            ).fetchone()
            if receipt_row:
                if (receipt_row["checkpoint_sequence"], receipt_row["checkpoint_digest"]) != (
                    checkpoint["sequence"],
                    checkpoint["digest"],
                ):
                    raise CaptureJobError(409, "request_id_conflict")
                return {
                    "job": self._summary(connection, row),
                    "receipt": json.loads(receipt_row["receipt_json"]),
                    "duplicate": True,
                }
            if body.get("expected_revision") != row["revision"]:
                raise CaptureJobError(409, "cas_mismatch", {"revision": row["revision"]})
            if row["checkpoint_sequence"] is not None and checkpoint["sequence"] < row["checkpoint_sequence"]:
                raise CaptureJobError(409, "older_checkpoint")
            if row["checkpoint_sequence"] == checkpoint["sequence"]:
                if checkpoint["digest"] != row["checkpoint_digest"]:
                    raise CaptureJobError(409, "checkpoint_conflict")
                receipt = {
                    **json.loads(row["receipt_json"]),
                    "receipt_id": str(uuid4()),
                    "request_id": request_id,
                    "revision": row["revision"],
                    "acknowledged_at": _stamp(),
                    "no_op": True,
                }
                connection.execute(
                    "INSERT INTO capture_job_receipts VALUES (?, ?, ?, ?, ?)",
                    (job_id, request_id, checkpoint["sequence"], checkpoint["digest"], canonical_json(receipt)),
                )
                return {"job": self._summary(connection, row), "receipt": receipt, "duplicate": True}
            revision, now = row["revision"] + 1, _stamp()
            receipt = {
                "receipt_id": str(uuid4()),
                "request_id": request_id,
                "job_id": job_id,
                "revision": revision,
                "checkpoint_sequence": checkpoint["sequence"],
                "checkpoint_digest": checkpoint["digest"],
                "acknowledged_at": now,
            }
            connection.execute(
                "UPDATE capture_jobs SET revision=?, checkpoint_artifact_ref=?, checkpoint_size=?, checkpoint_sequence=?, checkpoint_digest=?, receipt_json=?, updated_at=? WHERE job_id=?",
                (
                    revision,
                    artifact_ref,
                    staged.size_bytes,
                    checkpoint["sequence"],
                    checkpoint["digest"],
                    canonical_json(receipt),
                    now,
                    job_id,
                ),
            )
            connection.execute(
                "INSERT INTO capture_job_receipts VALUES (?, ?, ?, ?, ?)",
                (job_id, request_id, checkpoint["sequence"], checkpoint["digest"], canonical_json(receipt)),
            )
            # Checkpointing is the only progress the client reports, so it is
            # the only place a timeline can come from: no client posts to the
            # event route, and the "created" event carries no refs, which the
            # projection excludes. The event does not advance the job revision
            # -- the checkpoint already did, and the client's next CAS is
            # against that.
            self._append_event(
                connection,
                job_id,
                "capture-attempted",
                f"checkpoint:{job_id}:{checkpoint['sequence']}:{checkpoint['digest']}",
                revision,
                {"conversation_ref": conversation_ref or f"intent:{row['intent_key']}"},
                {"checkpoint_sequence": checkpoint["sequence"], "checkpoint_digest": checkpoint["digest"]},
                advance_revision=False,
            )
            self._retention_after_checkpoint(connection, job_id, json.loads(row["retention_json"]))
            next_row = connection.execute(
                "SELECT " + _JOB_COLUMNS + " FROM capture_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            return {"job": self._summary(connection, next_row), "receipt": receipt, "duplicate": False}


def registry_for_receiver(spool_path: Path | None, receiver_id: str) -> CaptureJobRegistry:
    """Build the registry without moving the receiver bearer downstream.

    HTTP authentication remains the bearer token's only job. Lease proofs are
    opaque fencing values derived from the stable, non-secret receiver identity;
    they reject stale clients but grant no route access on their own.
    """
    return CaptureJobRegistry(spool_path, receiver_id)
