"""Receiver-authoritative browser capture jobs.

The extension may cache an opaque job id, but this SQLite registry is the
durable authority for profile-loss recovery.  It deliberately stores only a
keyed account scope, never an account identifier or provider credential.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import sqlite3
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import uuid4

from polylogue.browser_capture.capture_job_events import (
    project_capture_job_timelines,
    read_capture_job_events,
    read_capture_job_retention,
)
from polylogue.browser_capture.receiver import backfill_checkpoint_root
from polylogue.paths import browser_capture_spool_root

_RETRY_STATES = frozenset({"ready", "retry_wait", "held", "completed", "abandoned"})


class CaptureJobError(Exception):
    def __init__(self, status: int, code: str, details: dict[str, object] | None = None) -> None:
        super().__init__(code)
        self.status = status
        self.code = code
        self.details = details or {}


def canonical_json(value: object) -> str:
    if value is None or isinstance(value, bool):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, str):
        return json.dumps(unicodedata.normalize("NFC", value), ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) <= 9_007_199_254_740_991:
        return str(value)
    if isinstance(value, list):
        return "[" + ",".join(canonical_json(item) for item in value) + "]"
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        entries = sorted(
            ((unicodedata.normalize("NFC", key), item) for key, item in value.items()),
            key=lambda entry: entry[0],
        )
        if any(entries[index - 1][0] == entries[index][0] for index in range(1, len(entries))):
            raise CaptureJobError(400, "non_canonical_key_collision")
        return (
            "{"
            + ",".join(json.dumps(key, ensure_ascii=False) + ":" + canonical_json(item) for key, item in entries)
            + "}"
        )
    raise CaptureJobError(400, "non_canonical_json")


def canonical_digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


def capture_job_database_path(spool_path: Path | None = None) -> Path:
    return (spool_path or browser_capture_spool_root()) / "capture-jobs" / "registry.sqlite3"


def capture_job_scope_namespace(spool_path: Path | None = None) -> str:
    """Return a stable pseudonym namespace independent of bearer rotation."""
    root = (spool_path or browser_capture_spool_root()).expanduser().resolve()
    digest = hashlib.sha256(f"polylogue:capture-job-scope:v1\0{root}".encode()).hexdigest()
    return f"cjs1:{digest}"


def _now() -> datetime:
    return datetime.now(UTC)


def _stamp(value: datetime | None = None) -> str:
    return (value or _now()).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class CaptureJobRegistry:
    spool_path: Path | None
    receiver_id: str

    protocol_min: int = 1
    protocol_max: int = 1

    def capabilities(self) -> dict[str, object]:
        return {
            "schema": "polylogue.capture-jobs.capabilities.v1",
            "protocol_min": self.protocol_min,
            "protocol_max": self.protocol_max,
            "scope_namespace": capture_job_scope_namespace(self.spool_path),
        }

    def _connect(self) -> sqlite3.Connection:
        path = capture_job_database_path(self.spool_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS capture_jobs (
                job_id TEXT PRIMARY KEY, provider TEXT NOT NULL, account_scope TEXT NOT NULL,
                intent_key TEXT NOT NULL, intent_json TEXT NOT NULL, revision INTEGER NOT NULL,
                checkpoint_json TEXT, checkpoint_sequence INTEGER, checkpoint_digest TEXT,
                receipt_json TEXT, retry_json TEXT NOT NULL, lease_json TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                request_budget_json TEXT NOT NULL DEFAULT '{"max_requests":1000,"used":0}',
                retention_json TEXT NOT NULL DEFAULT '{"state":"active","hold_reason":null,"timeline_authoritative":true}',
                UNIQUE(provider, account_scope, intent_key)
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
        columns = {row[1] for row in connection.execute("PRAGMA table_info(capture_job_events)")}
        if "job_revision" not in columns:
            connection.execute("ALTER TABLE capture_job_events ADD COLUMN job_revision INTEGER NOT NULL DEFAULT 0")
        job_columns = {row[1] for row in connection.execute("PRAGMA table_info(capture_jobs)")}
        if "request_budget_json" not in job_columns:
            connection.execute(
                'ALTER TABLE capture_jobs ADD COLUMN request_budget_json TEXT NOT NULL DEFAULT \'{"max_requests":1000,"used":0}\''
            )
        if "retention_json" not in job_columns:
            connection.execute(
                'ALTER TABLE capture_jobs ADD COLUMN retention_json TEXT NOT NULL DEFAULT \'{"state":"active","hold_reason":null,"timeline_authoritative":true}\''
            )
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _validate_scope(self, provider: object, account_scope: object, protocol: object) -> tuple[str, str]:
        if not isinstance(provider, str) or not provider or provider != provider.lower():
            raise CaptureJobError(400, "invalid_provider")
        if not isinstance(account_scope, str) or not account_scope.startswith("h1:") or len(account_scope) != 46:
            raise CaptureJobError(400, "invalid_account_scope")
        if not isinstance(protocol, int) or not self.protocol_min <= protocol <= self.protocol_max:
            raise CaptureJobError(
                426, "incompatible_client", {"receiver_min": self.protocol_min, "receiver_max": self.protocol_max}
            )
        return provider, account_scope

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

    def _summary(self, row: sqlite3.Row) -> dict[str, object]:
        intent = json.loads(row["intent_json"])
        lease = json.loads(row["lease_json"]) if row["lease_json"] else None
        retry = json.loads(row["retry_json"])
        budget = json.loads(row["request_budget_json"])
        retention = json.loads(row["retention_json"])
        latest_receipt = json.loads(row["receipt_json"]) if row["receipt_json"] else None
        return {
            "job_id": row["job_id"],
            "provider": row["provider"],
            "account_scope": row["account_scope"],
            "intent_key": row["intent_key"],
            "intent_version": intent["version"],
            "intent_digest": intent["digest"],
            "intent": intent,
            "revision": row["revision"],
            "checkpoint_sequence": row["checkpoint_sequence"],
            "checkpoint_digest": row["checkpoint_digest"],
            "retry": retry,
            "request_budget": budget,
            "retention": retention,
            "checkpoint": json.loads(row["checkpoint_json"]) if row["checkpoint_json"] else None,
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

    def _require_live_lease(self, job_id: str, row: sqlite3.Row, body: dict[str, object]) -> dict[str, object]:
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
        expires_at = lease.get("expires_at")
        if not isinstance(expires_at, str) or datetime.fromisoformat(expires_at.replace("Z", "+00:00")) <= _now():
            raise CaptureJobError(409, "lease_expired")
        return lease

    def _census_legacy_orphans(self, connection: sqlite3.Connection) -> list[dict[str, object]]:
        root = backfill_checkpoint_root(self.spool_path)
        unreadable: list[dict[str, object]] = []
        if root.is_dir():
            for path in sorted(root.glob("*.json")):
                try:
                    raw = path.read_bytes()
                except OSError as exc:
                    unreadable.append(
                        {
                            "orphan_kind": "unreadable_legacy_checkpoint",
                            "path": str(path),
                            "errno_class": type(exc).__name__,
                        }
                    )
                    continue
                digest = "sha256:" + hashlib.sha256(raw).hexdigest()
                try:
                    payload = json.loads(raw)
                    valid = isinstance(payload, dict) and isinstance(payload.get("checkpoint"), dict)
                except json.JSONDecodeError:
                    valid = False
                kind = "legacy_backfill_checkpoint" if valid else "malformed_legacy_checkpoint"
                diagnostic = "account scope unavailable; explicit migration or abandonment required"
                connection.execute(
                    """INSERT INTO capture_job_orphans VALUES (?, ?, ?, ?)
                    ON CONFLICT(source_digest) DO UPDATE SET
                        orphan_kind=excluded.orphan_kind,
                        diagnostic=excluded.diagnostic""",
                    (digest, kind, diagnostic, _stamp()),
                )
        rows = connection.execute(
            "SELECT source_digest, orphan_kind, diagnostic, created_at FROM capture_job_orphans ORDER BY created_at"
        ).fetchall()
        return [*map(dict, rows), *unreadable]

    def _require_scoped(
        self, connection: sqlite3.Connection, job_id: str, provider: object, account_scope: object, protocol: object
    ) -> sqlite3.Row:
        normalized_provider, normalized_scope = self._validate_scope(provider, account_scope, protocol)
        row = connection.execute("SELECT * FROM capture_jobs WHERE job_id=?", (job_id,)).fetchone()
        if (
            row is None
            or not hmac.compare_digest(row["provider"], normalized_provider)
            or not hmac.compare_digest(row["account_scope"], normalized_scope)
        ):
            raise CaptureJobError(404, "capture_job_not_found")
        return cast(sqlite3.Row, row)

    def create(self, body: dict[str, object]) -> tuple[int, dict[str, object]]:
        provider, scope = self._validate_scope(
            body.get("provider"), body.get("account_scope"), body.get("client_protocol")
        )
        intent = self._intent(body.get("intent"))
        now = _stamp()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            found = connection.execute(
                "SELECT * FROM capture_jobs WHERE provider=? AND account_scope=? AND intent_key=?",
                (provider, scope, intent["intent_key"]),
            ).fetchone()
            if found is not None:
                if json.loads(found["intent_json"])["digest"] != intent["digest"]:
                    raise CaptureJobError(409, "intent_key_conflict")
                return 200, {"created": False, "job": self._summary(found)}
            job_id = str(uuid4())
            connection.execute(
                "INSERT INTO capture_jobs VALUES (?, ?, ?, ?, ?, 0, NULL, NULL, NULL, NULL, ?, NULL, ?, ?, ?, ?)",
                (
                    job_id,
                    provider,
                    scope,
                    intent["intent_key"],
                    canonical_json(intent),
                    canonical_json({"state": "ready", "attempt": 0}),
                    now,
                    now,
                    canonical_json({"max_requests": 1000, "used": 0}),
                    canonical_json({"state": "active", "hold_reason": None, "timeline_authoritative": True}),
                ),
            )
            row = connection.execute("SELECT * FROM capture_jobs WHERE job_id=?", (job_id,)).fetchone()
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
            return 201, {"created": True, "job": self._summary(row)}

    def discover(self, body: dict[str, object]) -> dict[str, object]:
        provider, scope = self._validate_scope(
            body.get("provider"), body.get("account_scope"), body.get("client_protocol")
        )
        intent_key = body.get("intent_key")
        if intent_key is not None and (not isinstance(intent_key, str) or not intent_key.startswith("i1:")):
            raise CaptureJobError(400, "invalid_intent")
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM capture_jobs WHERE provider=? AND account_scope=?"
                + (" AND intent_key=?" if intent_key else "")
                + " ORDER BY updated_at DESC",
                (provider, scope, intent_key) if intent_key else (provider, scope),
            ).fetchall()
            return {"jobs": [self._summary(row) for row in rows]}

    def list_orphans(self, protocol: object) -> dict[str, object]:
        """Return the global legacy census only on its explicit operator route."""
        self._validate_protocol(protocol)
        with self._connection() as connection:
            return {"orphans": self._census_legacy_orphans(connection)}

    def get(self, job_id: str, body: dict[str, object]) -> dict[str, object]:
        with self._connection() as connection:
            connection.execute("BEGIN")
            row = self._require_scoped(
                connection,
                job_id,
                body.get("provider"),
                body.get("account_scope"),
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
            events = read_capture_job_events(connection, job_id, 500)
            lifecycle = read_capture_job_retention(connection, job_id)
            return {
                "job": self._summary(row),
                "lifecycle": lifecycle,
                "receipts": [*receipts, *updates],
                "events": events,
                "timelines": project_capture_job_timelines(events),
            }

    @staticmethod
    def _append_event(
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
            "SELECT * FROM capture_job_events WHERE job_id=? AND request_id=?", (job_id, request_id)
        ).fetchone()
        digest = canonical_digest({"kind": kind, "refs": refs, "payload": payload})
        if existing is not None:
            stored = json.loads(existing["payload_json"])
            if not isinstance(stored, dict) or stored.get("digest") != digest:
                raise CaptureJobError(409, "event_request_conflict")
            return CaptureJobRegistry._event_dict(existing)
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
        connection.execute(
            "INSERT INTO capture_job_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                job_id,
                event_revision,
                job_revision,
                kind,
                canonical_json(refs),
                canonical_json(stored_payload),
                request_id,
                now,
            ),
        )
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

    @staticmethod
    def _event_dict(row: sqlite3.Row) -> dict[str, object]:
        payload = json.loads(row["payload_json"])
        return {
            "event_id": row["event_id"],
            "job_id": row["job_id"],
            "event_revision": row["event_revision"],
            "job_revision": row["job_revision"],
            "kind": row["kind"],
            "refs": json.loads(row["refs_json"]),
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
                connection, job_id, body.get("provider"), body.get("account_scope"), body.get("client_protocol")
            )
            self._require_live_lease(job_id, row, body)
            existing = connection.execute(
                "SELECT 1 FROM capture_job_events WHERE job_id=? AND request_id=?", (job_id, request_id)
            ).fetchone()
            event = self._append_event(connection, job_id, kind, request_id, expected_revision, refs, payload)
            next_row = connection.execute("SELECT * FROM capture_jobs WHERE job_id=?", (job_id,)).fetchone()
            return {"event": event, "job": self._summary(next_row), "duplicate": existing is not None}

    def events(self, job_id: str, body: dict[str, object]) -> dict[str, object]:
        limit = body.get("limit", 100)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 500:
            raise CaptureJobError(400, "invalid_event_limit")
        with self._connection() as connection:
            connection.execute("BEGIN")
            self._require_scoped(
                connection, job_id, body.get("provider"), body.get("account_scope"), body.get("client_protocol")
            )
            events = read_capture_job_events(connection, job_id, limit)
            total = connection.execute("SELECT COUNT(*) FROM capture_job_events WHERE job_id=?", (job_id,)).fetchone()[
                0
            ]
            return {
                "events": events,
                "timelines": project_capture_job_timelines(events),
                "limit": limit,
                "has_more": total > limit,
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
                connection, job_id, body.get("provider"), body.get("account_scope"), body.get("client_protocol")
            )
            lease = json.loads(row["lease_json"]) if row["lease_json"] else None
            generation = lease["generation"] if lease else 0
            if not isinstance(generation, int):
                raise CaptureJobError(409, "invalid_stored_lease")
            if lease and lease["request_id"] == request_id and lease["session_id"] == session_id:
                expires_at = datetime.fromisoformat(str(lease["expires_at"]).replace("Z", "+00:00"))
                if expires_at > _now():
                    return {"job": self._summary(row), "lease": {**lease, "proof": self._proof(job_id, lease)}}
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
            next_row = connection.execute("SELECT * FROM capture_jobs WHERE job_id=?", (job_id,)).fetchone()
            return {"job": self._summary(next_row), "lease": {**next_lease, "proof": self._proof(job_id, next_lease)}}

    def update(self, job_id: str, body: dict[str, object]) -> dict[str, object]:
        request_id = body.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise CaptureJobError(400, "invalid_request_id")
        retry = self._retry(body.get("retry")) if "retry" in body else None
        retention = body.get("retention") if "retention" in body else None
        if retention is not None:
            if not isinstance(retention, dict) or retention.get("state") not in {"active", "held", "eligible"}:
                raise CaptureJobError(400, "invalid_retention_state")
            retention = {
                "state": retention["state"],
                "hold_reason": retention.get("hold_reason"),
                "timeline_authoritative": retention.get("timeline_authoritative", True),
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
                connection, job_id, body.get("provider"), body.get("account_scope"), body.get("client_protocol")
            )
            lease = self._require_live_lease(job_id, row, body)
            existing = connection.execute(
                "SELECT request_digest, receipt_json FROM capture_job_update_receipts WHERE job_id=? AND request_id=?",
                (job_id, request_id),
            ).fetchone()
            if existing:
                if not hmac.compare_digest(existing["request_digest"], request_digest):
                    raise CaptureJobError(409, "request_id_conflict")
                return {"job": self._summary(row), "receipt": json.loads(existing["receipt_json"]), "duplicate": True}
            if body.get("expected_revision") != row["revision"]:
                raise CaptureJobError(409, "cas_mismatch", {"revision": row["revision"]})
            current_retry = json.loads(row["retry_json"])
            current_retention = json.loads(row["retention_json"])
            next_retry = retry or current_retry
            next_retention = retention or current_retention
            next_lease = dict(lease)
            now = _now()
            if ttl is not None:
                next_lease["expires_at"] = _stamp(now + timedelta(seconds=ttl))
            if next_retry == current_retry and next_retention == current_retention and next_lease == lease:
                receipt = {
                    "receipt_id": str(uuid4()),
                    "request_id": request_id,
                    "job_id": job_id,
                    "kind": "capture_job_update",
                    "revision": row["revision"],
                    "retry": current_retry,
                    "lease_expires_at": lease["expires_at"],
                    "acknowledged_at": _stamp(now),
                    "no_op": True,
                }
                connection.execute(
                    "INSERT INTO capture_job_update_receipts VALUES (?, ?, ?, ?)",
                    (job_id, request_id, request_digest, canonical_json(receipt)),
                )
                return {"job": self._summary(row), "receipt": receipt, "duplicate": True}
            revision = row["revision"] + 1
            receipt = {
                "receipt_id": str(uuid4()),
                "request_id": request_id,
                "job_id": job_id,
                "kind": "capture_job_update",
                "revision": revision,
                "retry": next_retry,
                "lease_expires_at": next_lease["expires_at"],
                "acknowledged_at": _stamp(now),
            }
            connection.execute(
                "UPDATE capture_jobs SET revision=?, retry_json=?, retention_json=?, lease_json=?, updated_at=? WHERE job_id=?",
                (
                    revision,
                    canonical_json(next_retry),
                    canonical_json(next_retention),
                    canonical_json(next_lease),
                    _stamp(now),
                    job_id,
                ),
            )
            connection.execute(
                "INSERT INTO capture_job_update_receipts VALUES (?, ?, ?, ?)",
                (job_id, request_id, request_digest, canonical_json(receipt)),
            )
            next_row = connection.execute("SELECT * FROM capture_jobs WHERE job_id=?", (job_id,)).fetchone()
            return {"job": self._summary(next_row), "receipt": receipt, "duplicate": False}

    def gc(self, *, now: datetime | None = None, limit: int = 100) -> dict[str, object]:
        """Delete only explicitly eligible, acknowledged, non-authoritative jobs."""
        if not 1 <= limit <= 1000:
            raise CaptureJobError(400, "invalid_gc_limit")
        current = now or _now()
        deleted: list[str] = []
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute("SELECT * FROM capture_jobs ORDER BY updated_at").fetchall()
            for row in rows:
                if len(deleted) >= limit:
                    break
                retention = json.loads(row["retention_json"])
                lease = self._lease(row)
                if retention.get("state") != "eligible" or retention.get("timeline_authoritative", True):
                    continue
                expires_at = lease.get("expires_at") if lease else None
                if isinstance(expires_at, str) and datetime.fromisoformat(expires_at.replace("Z", "+00:00")) > current:
                    continue
                if row["checkpoint_sequence"] is None or not row["receipt_json"]:
                    continue
                connection.execute("DELETE FROM capture_job_events WHERE job_id=?", (row["job_id"],))
                connection.execute("DELETE FROM capture_job_receipts WHERE job_id=?", (row["job_id"],))
                connection.execute("DELETE FROM capture_job_update_receipts WHERE job_id=?", (row["job_id"],))
                connection.execute("DELETE FROM capture_jobs WHERE job_id=?", (row["job_id"],))
                deleted.append(row["job_id"])
        return {"deleted": deleted, "count": len(deleted)}

    def checkpoint(self, job_id: str, body: dict[str, object]) -> dict[str, object]:
        checkpoint = body.get("checkpoint")
        if (
            not isinstance(checkpoint, dict)
            or not isinstance(checkpoint.get("sequence"), int)
            or checkpoint["sequence"] < 0
            or checkpoint.get("digest") != canonical_digest(checkpoint.get("payload"))
        ):
            raise CaptureJobError(400, "invalid_checkpoint")
        request_id = body.get("request_id")
        if not isinstance(request_id, str):
            raise CaptureJobError(400, "invalid_request_id")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._require_scoped(
                connection, job_id, body.get("provider"), body.get("account_scope"), body.get("client_protocol")
            )
            self._require_live_lease(job_id, row, body)
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
                    "job": self._summary(row),
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
                return {"job": self._summary(row), "receipt": receipt, "duplicate": True}
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
                "UPDATE capture_jobs SET revision=?, checkpoint_json=?, checkpoint_sequence=?, checkpoint_digest=?, receipt_json=?, updated_at=? WHERE job_id=?",
                (
                    revision,
                    canonical_json(checkpoint),
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
            next_row = connection.execute("SELECT * FROM capture_jobs WHERE job_id=?", (job_id,)).fetchone()
            return {"job": self._summary(next_row), "receipt": receipt, "duplicate": False}


def registry_for_receiver(spool_path: Path | None, receiver_id: str) -> CaptureJobRegistry:
    """Build the registry without moving the receiver bearer downstream.

    HTTP authentication remains the bearer token's only job. Lease proofs are
    opaque fencing values derived from the stable, non-secret receiver identity;
    they reject stale clients but grant no route access on their own.
    """
    return CaptureJobRegistry(spool_path, receiver_id)
