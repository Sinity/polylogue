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
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import uuid4

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
    if value is None or isinstance(value, (str, bool)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) <= 9_007_199_254_740_991:
        return str(value)
    if isinstance(value, list):
        return "[" + ",".join(canonical_json(item) for item in value) + "]"
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return (
            "{"
            + ",".join(json.dumps(key, ensure_ascii=False) + ":" + canonical_json(value[key]) for key in sorted(value))
            + "}"
        )
    raise CaptureJobError(400, "non_canonical_json")


def canonical_digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


def capture_job_database_path(spool_path: Path | None = None) -> Path:
    return (spool_path or browser_capture_spool_root()) / "capture-jobs" / "registry.sqlite3"


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

    def _connect(self) -> sqlite3.Connection:
        path = capture_job_database_path(self.spool_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS capture_jobs (
                job_id TEXT PRIMARY KEY, provider TEXT NOT NULL, account_scope TEXT NOT NULL,
                intent_key TEXT NOT NULL, intent_json TEXT NOT NULL, revision INTEGER NOT NULL,
                checkpoint_json TEXT, checkpoint_sequence INTEGER, checkpoint_digest TEXT,
                receipt_json TEXT, retry_json TEXT NOT NULL, lease_json TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
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
        return connection

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
            "checkpoint": json.loads(row["checkpoint_json"]) if row["checkpoint_json"] else None,
            "latest_receipt": json.loads(row["receipt_json"]) if row["receipt_json"] else None,
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
        if root.is_dir():
            for path in sorted(root.glob("*.json")):
                try:
                    raw = path.read_bytes()
                except OSError:
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
                    "INSERT OR IGNORE INTO capture_job_orphans VALUES (?, ?, ?, ?)",
                    (digest, kind, diagnostic, _stamp()),
                )
        rows = connection.execute(
            "SELECT source_digest, orphan_kind, diagnostic, created_at FROM capture_job_orphans ORDER BY created_at"
        ).fetchall()
        return [dict(row) for row in rows]

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
        with self._connect() as connection:
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
                "INSERT INTO capture_jobs VALUES (?, ?, ?, ?, ?, 0, NULL, NULL, NULL, NULL, ?, NULL, ?, ?)",
                (
                    job_id,
                    provider,
                    scope,
                    intent["intent_key"],
                    canonical_json(intent),
                    canonical_json({"state": "ready", "attempt": 0}),
                    now,
                    now,
                ),
            )
            row = connection.execute("SELECT * FROM capture_jobs WHERE job_id=?", (job_id,)).fetchone()
            return 201, {"created": True, "job": self._summary(row)}

    def discover(self, body: dict[str, object]) -> dict[str, object]:
        provider, scope = self._validate_scope(
            body.get("provider"), body.get("account_scope"), body.get("client_protocol")
        )
        intent_key = body.get("intent_key")
        if intent_key is not None and (not isinstance(intent_key, str) or not intent_key.startswith("i1:")):
            raise CaptureJobError(400, "invalid_intent")
        with self._connect() as connection:
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
        with self._connect() as connection:
            return {"orphans": self._census_legacy_orphans(connection)}

    def get(self, job_id: str, body: dict[str, object]) -> dict[str, object]:
        with self._connect() as connection:
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
            return {"job": self._summary(row), "receipts": [*receipts, *updates]}

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
        with self._connect() as connection:
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
        ttl = body.get("lease_ttl_seconds")
        if ttl is not None and (not isinstance(ttl, int) or isinstance(ttl, bool) or not 1 <= ttl <= 300):
            raise CaptureJobError(400, "invalid_lease_ttl")
        if retry is None and ttl is None:
            raise CaptureJobError(400, "empty_capture_job_update")
        request_digest = canonical_digest({"retry": retry, "lease_ttl_seconds": ttl})
        with self._connect() as connection:
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
            next_retry = retry or current_retry
            next_lease = dict(lease)
            now = _now()
            if ttl is not None:
                next_lease["expires_at"] = _stamp(now + timedelta(seconds=ttl))
            if next_retry == current_retry and next_lease == lease:
                return {"job": self._summary(row), "receipt": None, "duplicate": True}
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
                "UPDATE capture_jobs SET revision=?, retry_json=?, lease_json=?, updated_at=? WHERE job_id=?",
                (revision, canonical_json(next_retry), canonical_json(next_lease), _stamp(now), job_id),
            )
            connection.execute(
                "INSERT INTO capture_job_update_receipts VALUES (?, ?, ?, ?)",
                (job_id, request_id, request_digest, canonical_json(receipt)),
            )
            next_row = connection.execute("SELECT * FROM capture_jobs WHERE job_id=?", (job_id,)).fetchone()
            return {"job": self._summary(next_row), "receipt": receipt, "duplicate": False}

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
        with self._connect() as connection:
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
                return {"job": self._summary(row), "receipt": json.loads(row["receipt_json"]), "duplicate": True}
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
