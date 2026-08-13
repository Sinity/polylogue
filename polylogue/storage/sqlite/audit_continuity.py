"""Replayable cross-tier write-ahead control for durable ``audit.db`` writes.

SQLite cannot atomically commit transactions spanning source.db and audit.db.
The source control row is therefore the authoritative write-ahead command:
prepare it in source.db, commit the audit mutation plus its head, then promote
the source head.  Startup can complete the first two crash windows because the
pending row contains the exact typed command, and it rejects an audit image
whose head regressed after source promotion.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import stat
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

from polylogue.storage.sqlite.audit_leaf import (
    AuditLeafError,
    VerifiedAuditLeaf,
    open_verified_audit_connection,
    open_verified_audit_read_connection,
    open_verified_sqlite_read_connection,
    open_verified_sqlite_write_connection,
)

_FORMAT = "polylogue.audit-continuity-command.v1"
AUDIT_CONTINUITY_GENESIS_HEAD_SHA256 = "3230fdd585a4fd2d71b7d720bcfe5d697ff120fdb32aecde394e89d407c7198f"
_SOURCE_CONTINUITY_SCHEMA_VERSION = 32
_AUDIT_CONTINUITY_SCHEMA_VERSION = 2
_T = TypeVar("_T")


class AuditContinuityError(RuntimeError):
    """Audit and source durable control state cannot prove one continuity head."""


@dataclass(frozen=True, slots=True)
class AuditMutation:
    """One typed audit command with generated identity and replay inputs."""

    kind: str
    mutation_id: str
    created_at_ms: int
    payload: Mapping[str, object]

    def command(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "mutation_id": self.mutation_id,
            "created_at_ms": self.created_at_ms,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_command(cls, raw: object) -> AuditMutation:
        if not isinstance(raw, dict):
            raise AuditContinuityError("pending audit continuity command is not an object")
        kind = raw.get("kind")
        mutation_id = raw.get("mutation_id")
        created_at_ms = raw.get("created_at_ms")
        payload = raw.get("payload")
        if (
            not isinstance(kind, str)
            or not kind
            or not isinstance(mutation_id, str)
            or not mutation_id
            or not isinstance(created_at_ms, int)
            or created_at_ms < 0
            or not isinstance(payload, dict)
        ):
            raise AuditContinuityError("pending audit continuity command is malformed")
        return cls(kind=kind, mutation_id=mutation_id, created_at_ms=created_at_ms, payload=payload)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def prepared_audit_continuity_command(
    mutation: AuditMutation, *, prior_generation: int, prior_head_sha256: str
) -> dict[str, object]:
    """Derive the sole source-WAL command and target for one mutation."""

    command = mutation.command()
    command_sha256 = _sha256(command)
    return {
        "format": _FORMAT,
        "prior_generation": prior_generation,
        "prior_head_sha256": prior_head_sha256,
        "next_generation": prior_generation + 1,
        "command": command,
        "command_sha256": command_sha256,
        "next_head_sha256": _sha256({"previous_head_sha256": prior_head_sha256, "command_sha256": command_sha256}),
    }


def audit_semantic_sha256(path: Path) -> str:
    """Hash audit content while excluding the self-mutating continuity head."""

    try:
        with open_verified_audit_read_connection(path) as connection:
            return _audit_semantic_sha256_connection(connection)
    except (AuditLeafError, sqlite3.DatabaseError) as exc:
        raise AuditContinuityError("cannot hash audit content for continuity validation") from exc


@contextmanager
def _open_source_read_connection(path: Path) -> Iterator[sqlite3.Connection]:
    try:
        with open_verified_sqlite_read_connection(path) as connection:
            yield connection
    except AuditLeafError as exc:
        raise AuditContinuityError(f"cannot safely read source continuity tier: {path}: {exc}") from exc


@contextmanager
def _open_source_write_connection(path: Path) -> Iterator[sqlite3.Connection]:
    try:
        with open_verified_sqlite_write_connection(path) as connection:
            yield connection
    except AuditLeafError as exc:
        raise AuditContinuityError(f"cannot safely write source continuity tier: {path}: {exc}") from exc


def _entry_is_absent(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return True
    except OSError as exc:
        raise AuditContinuityError(f"cannot inspect audit continuity tier entry: {path}") from exc
    return False


def _audit_semantic_sha256_connection(connection: sqlite3.Connection) -> str:
    """Return the continuity-independent semantic digest for one open audit DB."""

    lines = (line for line in connection.iterdump() if "audit_continuity_head" not in line)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


class AuditContinuityCoordinator:
    """Coordinate typed audit commands through source.db's durable WAL row."""

    def __init__(
        self,
        archive_root: Path,
        *,
        phase_hook: Callable[[str, AuditMutation], None] | None = None,
    ) -> None:
        self.archive_root = archive_root.resolve()
        self.source_path = self.archive_root / "source.db"
        self.audit_path = self.archive_root / "audit.db"
        self._phase_hook = phase_hook

    def execute(self, mutation: AuditMutation, apply: Callable[[sqlite3.Connection, AuditMutation], _T]) -> _T:
        """Prepare one command, commit audit bytes, then promote source control."""

        self._phase("before_source_prepare", mutation)
        prepared = self._prepare(mutation)
        self._phase("after_source_prepare", mutation)
        try:
            result = self._apply_prepared(prepared, apply, allow_rebind=mutation.kind == "rebind")
        except Exception:
            # _apply_prepared has exited its audit transaction before this
            # handler runs. Clear this exact source WAL entry only when the
            # audit head still proves no commit happened, so validation rejects
            # cannot wedge every later audit mutation.
            self._abort_prepared(prepared)
            raise
        self._phase("after_audit_commit", mutation)
        self._promote(prepared)
        self._phase("after_source_promotion", mutation)
        return result

    def reconcile(self, apply: Callable[[sqlite3.Connection, AuditMutation], object]) -> None:
        """Deterministically complete a pending command or reject a stale audit image."""

        if not self.is_available():
            return
        prepared = self._pending()
        if prepared is None:
            self._assert_committed_head_matches_audit()
            return
        mutation = AuditMutation.from_command(prepared["command"])
        self._apply_prepared(prepared, apply, allow_rebind=mutation.kind == "rebind")
        self._promote(prepared)

    def reconcile_pending_rebind(self, mutation_id: str) -> bool:
        """Complete only the named operation-owned rebind command, if pending."""

        prepared = self._pending()
        if prepared is None:
            return self.has_committed_mutation(mutation_id)
        mutation = AuditMutation.from_command(prepared["command"])
        if mutation.kind != "rebind" or mutation.mutation_id != mutation_id:
            raise AuditContinuityError("pending audit continuity command does not belong to this restore rebind")
        if not self.is_available():
            raise AuditContinuityError("pending restore rebind lacks a readable audit continuity head")
        self._apply_prepared(prepared, lambda _conn, _mutation: None, allow_rebind=True)
        self._promote(prepared)
        return True

    def has_pending_rebind(self, mutation_id: str) -> bool:
        """Return whether source.db has the named restore-owned rebind prepared."""

        prepared = self._pending()
        if prepared is None:
            return False
        mutation = AuditMutation.from_command(prepared["command"])
        if mutation.kind != "rebind" or mutation.mutation_id != mutation_id:
            raise AuditContinuityError("pending audit continuity command does not belong to this restore rebind")
        return True

    def is_available(self) -> bool:
        """Return whether both schema halves needed for coordinated writes exist."""

        if _entry_is_absent(self.source_path) or _entry_is_absent(self.audit_path):
            return False
        try:
            with (
                _open_source_read_connection(self.source_path) as source,
                open_verified_audit_read_connection(self.audit_path) as audit,
            ):
                source_version = int(source.execute("PRAGMA user_version").fetchone()[0] or 0)
                audit_version = int(audit.execute("PRAGMA user_version").fetchone()[0] or 0)
                source_has_control = self._has_table(source, "audit_continuity_control")
                audit_has_head = self._has_table(audit, "audit_continuity_head")
                if not source_has_control and source_version >= _SOURCE_CONTINUITY_SCHEMA_VERSION:
                    raise AuditContinuityError("current source schema is missing audit continuity control")
                if not audit_has_head and audit_version >= _AUDIT_CONTINUITY_SCHEMA_VERSION:
                    raise AuditContinuityError("current audit schema is missing audit continuity head")
                if not source_has_control or not audit_has_head:
                    return False
                source.execute("SELECT 1 FROM audit_continuity_control WHERE singleton = 1").fetchone()
                audit.execute("SELECT 1 FROM audit_continuity_head WHERE singleton = 1").fetchone()
                if self._is_unbound_populated_precontinuity_audit(source, audit):
                    raise AuditContinuityError(
                        "populated pre-continuity audit journal requires authenticated post-migration binding"
                    )
        except AuditLeafError as exc:
            raise AuditContinuityError(str(exc)) from exc
        except sqlite3.OperationalError as exc:
            raise AuditContinuityError("cannot inspect audit continuity compatibility state") from exc
        except sqlite3.DatabaseError as exc:
            raise AuditContinuityError("cannot inspect audit continuity compatibility state") from exc
        return True

    def needs_precontinuity_binding(self) -> bool:
        """Return whether a migrated populated audit journal still has only genesis heads."""

        self._require_paths()
        try:
            with (
                _open_source_read_connection(self.source_path) as source,
                open_verified_audit_read_connection(self.audit_path) as audit,
            ):
                source_version = int(source.execute("PRAGMA user_version").fetchone()[0] or 0)
                audit_version = int(audit.execute("PRAGMA user_version").fetchone()[0] or 0)
                source_has_control = self._has_table(source, "audit_continuity_control")
                audit_has_head = self._has_table(audit, "audit_continuity_head")
                if not source_has_control and source_version >= _SOURCE_CONTINUITY_SCHEMA_VERSION:
                    raise AuditContinuityError("current source schema is missing audit continuity control")
                if not audit_has_head and audit_version >= _AUDIT_CONTINUITY_SCHEMA_VERSION:
                    raise AuditContinuityError("current audit schema is missing audit continuity head")
                return (
                    source_has_control
                    and audit_has_head
                    and self._is_unbound_populated_precontinuity_audit(source, audit)
                )
        except AuditLeafError as exc:
            raise AuditContinuityError("cannot inspect pre-continuity audit binding state") from exc
        except sqlite3.DatabaseError as exc:
            raise AuditContinuityError("cannot inspect pre-continuity audit binding state") from exc

    def bind_precontinuity_audit(self, *, mutation_id: str, now_ms: int, audit_semantic_sha256: str) -> None:
        """Bind a populated v1 audit journal through its first source-backed head.

        Published v2/v32 migrations seeded matching genesis rows for both fresh
        and upgraded archives.  A populated upgraded journal needs this explicit
        command, whose head commits the authenticated pre-migration semantic
        digest, before ordinary coordinated mutations are allowed.
        """

        if len(audit_semantic_sha256) != 64:
            raise AuditContinuityError("pre-continuity binding requires an audit semantic sha256")
        if self.has_committed_mutation(mutation_id):
            return
        prepared = self._pending()
        if prepared is not None:
            pending = AuditMutation.from_command(prepared["command"])
            if pending.kind != "bind_precontinuity_audit" or pending.mutation_id != mutation_id:
                raise AuditContinuityError(
                    "pending audit continuity command does not belong to this pre-continuity binding"
                )
            self._apply_prepared(prepared, lambda _conn, _mutation: None)
            self._promote(prepared)
            return
        if not self.needs_precontinuity_binding():
            raise AuditContinuityError("pre-continuity audit binding no longer has matching unbound genesis heads")
        self.execute(
            AuditMutation(
                "bind_precontinuity_audit",
                mutation_id,
                now_ms,
                {"audit_semantic_sha256": audit_semantic_sha256},
            ),
            lambda _conn, _mutation: None,
        )

    def runtime_probe(self) -> str:
        """Exercise the coordinator's released-schema or compatibility state."""

        if not self.is_available():
            return "standby until source.db and audit.db both install continuity control"
        if self._pending() is not None:
            raise AuditContinuityError("runtime probe found an unreconciled audit continuity command")
        self._assert_committed_head_matches_audit()
        return "reconciled matching source/audit continuity heads"

    def seed_or_rebind(self, *, mutation_id: str, now_ms: int, evidence: Mapping[str, object]) -> None:
        """Advance continuity after an authenticated adoption or verified restore.

        This is intentionally a typed WAL command too.  The caller has already
        authenticated the external publication; this method only binds that
        exact evidence to the new audit image without trusting inode identity.
        """

        expected_image_sha256 = evidence.get("audit_image_sha256")
        if not isinstance(expected_image_sha256, str) or len(expected_image_sha256) != 64:
            raise AuditContinuityError("rebind requires an exact audit image sha256")
        if self.has_committed_mutation(mutation_id):
            return
        # Adoption and restore publish their immutable evidence only after
        # this machine head advances. Resume this exact source-WAL command on
        # retry instead of treating it as an unrelated competing mutation.
        if self.has_pending_rebind(mutation_id):
            self.reconcile_pending_rebind(mutation_id)
            return
        mutation = AuditMutation("rebind", mutation_id, now_ms, dict(evidence))

        # A verified restored image can contain an older audit head. Its
        # authenticated image hash is the authority to rebind it.
        self.execute(mutation, lambda _conn, _mutation: None)

    def has_committed_mutation(self, mutation_id: str) -> bool:
        """Return whether both tiers already committed this exact mutation id."""
        self._require_paths()
        try:
            with (
                _open_source_read_connection(self.source_path) as source,
                open_verified_audit_read_connection(self.audit_path) as audit,
            ):
                source_row = source.execute(
                    "SELECT committed_generation, committed_head_sha256 FROM audit_continuity_control WHERE singleton = 1"
                ).fetchone()
                audit_row = audit.execute(
                    "SELECT generation, head_sha256, mutation_id FROM audit_continuity_head WHERE singleton = 1"
                ).fetchone()
        except sqlite3.DatabaseError as exc:
            if "no such table" in str(exc).lower():
                return False
            raise AuditContinuityError("cannot read audit continuity commit state") from exc
        if source_row is None or audit_row is None:
            raise AuditContinuityError("audit continuity control row is missing")
        if (int(source_row[0]), str(source_row[1])) != (int(audit_row[0]), str(audit_row[1])):
            return False
        return isinstance(audit_row[2], str) and audit_row[2] == mutation_id

    def reconcile_restore_rebind(
        self,
        mutation: AuditMutation,
        *,
        prior_generation: int,
        prior_head_sha256: str,
    ) -> bool:
        """Resume one exact restore rebind without minting a second source head."""

        if mutation.kind != "rebind":
            raise AuditContinuityError("restore continuity reconciliation requires a rebind mutation")
        expected = prepared_audit_continuity_command(
            mutation, prior_generation=prior_generation, prior_head_sha256=prior_head_sha256
        )
        pending = self._pending()
        if pending is not None:
            if pending != expected:
                raise AuditContinuityError("pending restore rebind does not match its immutable prepared evidence")
            self._apply_prepared(pending, lambda _conn, _mutation: None, allow_rebind=True)
            self._promote(pending)
            return True
        with _open_source_read_connection(self.source_path) as source:
            row = source.execute(
                "SELECT committed_generation, committed_head_sha256 FROM audit_continuity_control WHERE singleton = 1"
            ).fetchone()
        if row is None:
            raise AuditContinuityError("source audit continuity control is missing")
        prior = (prior_generation, prior_head_sha256)
        target_generation = expected["next_generation"]
        target_head = expected["next_head_sha256"]
        if not isinstance(target_generation, int) or not isinstance(target_head, str):
            raise AuditContinuityError("restore rebind target is malformed")
        target = (target_generation, target_head)
        current = (int(row[0]), str(row[1]))
        if current == prior:
            return False
        if current != target:
            raise AuditContinuityError("promoted restore rebind does not match its immutable prepared evidence")
        self._repair_promoted_rebind(expected)
        return True

    def _phase(self, name: str, mutation: AuditMutation) -> None:
        if self._phase_hook is not None:
            self._phase_hook(name, mutation)

    def _prepare(self, mutation: AuditMutation) -> dict[str, object]:
        self._require_paths()
        with _open_source_write_connection(self.source_path) as conn, conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT committed_generation, committed_head_sha256, pending_mutation_id FROM audit_continuity_control WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise AuditContinuityError("source audit continuity control is missing")
            if row[2] is not None:
                raise AuditContinuityError("another audit continuity mutation is already pending")
            prepared = prepared_audit_continuity_command(
                mutation, prior_generation=int(row[0]), prior_head_sha256=str(row[1])
            )
            payload_json = _canonical_json(prepared)
            conn.execute(
                """
                UPDATE audit_continuity_control
                SET pending_mutation_id = ?, pending_payload_json = ?, pending_payload_sha256 = ?, prepared_at_ms = ?
                WHERE singleton = 1 AND pending_mutation_id IS NULL
                """,
                (mutation.mutation_id, payload_json, _sha256(prepared), mutation.created_at_ms),
            )
            conn.commit()
        return prepared

    def _pending(self) -> dict[str, object] | None:
        self._require_paths()
        with _open_source_read_connection(self.source_path) as conn:
            row = conn.execute(
                "SELECT committed_generation, committed_head_sha256, pending_payload_json, pending_payload_sha256 FROM audit_continuity_control WHERE singleton = 1"
            ).fetchone()
        if row is None:
            raise AuditContinuityError("source audit continuity control is missing")
        pending_json = row[2]
        if pending_json is None:
            return None
        if not isinstance(pending_json, str):
            raise AuditContinuityError("source audit continuity pending command is malformed")
        try:
            prepared = json.loads(pending_json)
        except json.JSONDecodeError as exc:
            raise AuditContinuityError("source audit continuity pending command is invalid JSON") from exc
        if not isinstance(prepared, dict) or _sha256(prepared) != row[3]:
            raise AuditContinuityError("source audit continuity pending command checksum mismatch")
        if (
            prepared.get("format") != _FORMAT
            or prepared.get("prior_generation") != row[0]
            or prepared.get("prior_head_sha256") != row[1]
        ):
            raise AuditContinuityError("source audit continuity pending command does not bind its committed head")
        self._validate_prepared(prepared)
        return prepared

    def _apply_prepared(
        self,
        prepared: dict[str, object],
        apply: Callable[[sqlite3.Connection, AuditMutation], _T],
        *,
        allow_rebind: bool = False,
    ) -> _T:
        self._validate_prepared(prepared)
        mutation = AuditMutation.from_command(prepared["command"])
        if mutation.kind == "rebind" and not self._audit_has_prepared_target(prepared, mutation):
            # Writer setup persists WAL mode in the main header. Authenticate a
            # restored image before opening that mutating connection, but do
            # not re-hash an audit side that already committed this target.
            self._assert_rebind_image(mutation)
        with open_verified_audit_connection(self.audit_path) as conn, conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT generation, head_sha256, mutation_id FROM audit_continuity_head WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise AuditContinuityError("audit continuity head is missing")
            current = (int(row[0]), str(row[1]), row[2])
            prior = (cast(int, prepared["prior_generation"]), str(prepared["prior_head_sha256"]))
            target = (cast(int, prepared["next_generation"]), str(prepared["next_head_sha256"]))
            if current[:2] == target and current[2] == mutation.mutation_id:
                conn.commit()
                return cast(_T, None)
            if mutation.kind == "bind_precontinuity_audit":
                self._assert_precontinuity_audit_semantics(conn, mutation)
            if current[:2] != prior:
                if allow_rebind and mutation.kind == "rebind":
                    pass
                else:
                    raise AuditContinuityError("audit continuity head does not match the prepared source command")
            result = (
                cast(_T, None) if mutation.kind in {"rebind", "bind_precontinuity_audit"} else apply(conn, mutation)
            )
            conn.execute(
                "UPDATE audit_continuity_head SET generation = ?, head_sha256 = ?, mutation_id = ?, advanced_at_ms = ? WHERE singleton = 1",
                (*target, mutation.mutation_id, mutation.created_at_ms),
            )
            conn.commit()
            return result

    def _promote(self, prepared: Mapping[str, object]) -> None:
        mutation = AuditMutation.from_command(prepared["command"])
        with _open_source_write_connection(self.source_path) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE audit_continuity_control
                SET committed_generation = ?, committed_head_sha256 = ?,
                    pending_mutation_id = NULL, pending_payload_json = NULL,
                    pending_payload_sha256 = NULL, prepared_at_ms = NULL
                WHERE singleton = 1 AND pending_mutation_id = ? AND pending_payload_sha256 = ?
                """,
                (
                    prepared["next_generation"],
                    prepared["next_head_sha256"],
                    mutation.mutation_id,
                    _sha256(dict(prepared)),
                ),
            )
            if cursor.rowcount != 1:
                raise AuditContinuityError("source audit continuity promotion lost its prepared command")
            conn.commit()

    def _abort_prepared(self, prepared: Mapping[str, object]) -> None:
        """Discard a rejected WAL command after proving its audit transaction rolled back."""

        mutation = AuditMutation.from_command(prepared["command"])
        prior = (cast(int, prepared["prior_generation"]), str(prepared["prior_head_sha256"]))
        target = (cast(int, prepared["next_generation"]), str(prepared["next_head_sha256"]))
        with open_verified_audit_connection(self.audit_path) as audit:
            audit.execute("BEGIN IMMEDIATE")
            row = audit.execute(
                "SELECT generation, head_sha256, mutation_id FROM audit_continuity_head WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise AuditContinuityError("audit continuity head is missing while aborting a prepared command")
            current = (int(row[0]), str(row[1]), row[2])
            if current[:2] == target and current[2] == mutation.mutation_id:
                # The audit commit did land. Keep the WAL command for normal
                # promotion instead of mistaking an ambiguous failure for rollback.
                return
            if current[:2] != prior:
                raise AuditContinuityError("cannot abort prepared command after an unrelated audit head change")
            with _open_source_write_connection(self.source_path) as source, source:
                source.execute("BEGIN IMMEDIATE")
                cursor = source.execute(
                    """
                UPDATE audit_continuity_control
                SET pending_mutation_id = NULL, pending_payload_json = NULL,
                    pending_payload_sha256 = NULL, prepared_at_ms = NULL
                WHERE singleton = 1 AND committed_generation = ? AND committed_head_sha256 = ?
                  AND pending_mutation_id = ? AND pending_payload_sha256 = ?
                """,
                    (prior[0], prior[1], mutation.mutation_id, _sha256(dict(prepared))),
                )
                if cursor.rowcount != 1:
                    raise AuditContinuityError("source audit continuity abort lost its prepared command")
                source.commit()

    def _repair_promoted_rebind(self, prepared: Mapping[str, object]) -> None:
        """Advance a restored audit head to an already-promoted exact target."""

        mutation = AuditMutation.from_command(prepared["command"])
        prior = (cast(int, prepared["prior_generation"]), str(prepared["prior_head_sha256"]))
        target = (cast(int, prepared["next_generation"]), str(prepared["next_head_sha256"]))
        self._assert_rebind_image(mutation)
        with open_verified_audit_connection(self.audit_path) as audit, audit:
            audit.execute("BEGIN IMMEDIATE")
            row = audit.execute(
                "SELECT generation, head_sha256, mutation_id FROM audit_continuity_head WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise AuditContinuityError("audit continuity head is missing while repairing promoted rebind")
            current = (int(row[0]), str(row[1]), row[2])
            if current[:2] == target and current[2] == mutation.mutation_id:
                audit.commit()
                return
            if current[:2] != prior:
                raise AuditContinuityError("restored audit head does not match the exact promoted rebind prior")
            audit.execute(
                "UPDATE audit_continuity_head SET generation = ?, head_sha256 = ?, mutation_id = ?, advanced_at_ms = ? "
                "WHERE singleton = 1",
                (*target, mutation.mutation_id, mutation.created_at_ms),
            )
            audit.commit()

    def _audit_has_prepared_target(self, prepared: Mapping[str, object], mutation: AuditMutation) -> bool:
        target = (cast(int, prepared["next_generation"]), str(prepared["next_head_sha256"]))
        try:
            with open_verified_audit_read_connection(self.audit_path) as audit:
                row = audit.execute(
                    "SELECT generation, head_sha256, mutation_id FROM audit_continuity_head WHERE singleton = 1"
                ).fetchone()
        except (AuditLeafError, sqlite3.DatabaseError) as exc:
            raise AuditContinuityError("cannot inspect audit continuity head before rebind") from exc
        return row is not None and (int(row[0]), str(row[1])) == target and row[2] == mutation.mutation_id

    def _assert_committed_head_matches_audit(self) -> None:
        with (
            _open_source_read_connection(self.source_path) as source,
            open_verified_audit_read_connection(self.audit_path) as audit,
        ):
            source_row = source.execute(
                "SELECT committed_generation, committed_head_sha256 FROM audit_continuity_control WHERE singleton = 1"
            ).fetchone()
            audit_row = audit.execute(
                "SELECT generation, head_sha256 FROM audit_continuity_head WHERE singleton = 1"
            ).fetchone()
        if source_row is None or audit_row is None:
            raise AuditContinuityError("audit continuity control row is missing")
        if (int(source_row[0]), str(source_row[1])) != (int(audit_row[0]), str(audit_row[1])):
            raise AuditContinuityError("audit continuity head regressed or was replaced after source promotion")

    def _validate_prepared(self, prepared: Mapping[str, object]) -> None:
        command = prepared.get("command")
        if not isinstance(command, dict) or prepared.get("format") != _FORMAT:
            raise AuditContinuityError("audit continuity command format mismatch")
        prior_generation = prepared.get("prior_generation")
        next_generation = prepared.get("next_generation")
        if not isinstance(prior_generation, int) or not isinstance(next_generation, int):
            raise AuditContinuityError("audit continuity command generations are malformed")
        if next_generation != prior_generation + 1:
            raise AuditContinuityError("audit continuity command generation is non-monotonic")
        command_sha256 = _sha256(command)
        if prepared.get("command_sha256") != command_sha256:
            raise AuditContinuityError("audit continuity command payload checksum mismatch")
        expected_head = _sha256(
            {"previous_head_sha256": prepared.get("prior_head_sha256"), "command_sha256": command_sha256}
        )
        if prepared.get("next_head_sha256") != expected_head:
            raise AuditContinuityError("audit continuity command head checksum mismatch")

    def _assert_rebind_image(self, mutation: AuditMutation) -> None:
        expected = mutation.payload.get("audit_image_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise AuditContinuityError("rebind command lacks an audit image sha256")
        digest = hashlib.sha256()
        try:
            with VerifiedAuditLeaf(self.audit_path.parent, filename=self.audit_path.name) as leaf:
                with leaf.anchored_path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                leaf.assert_unchanged()
        except (AuditLeafError, OSError) as exc:
            raise AuditContinuityError("cannot read audit image for rebind") from exc
        if digest.hexdigest() != expected:
            raise AuditContinuityError("audit image changed before continuity rebind")

    @staticmethod
    def _has_table(connection: sqlite3.Connection, name: str) -> bool:
        return (
            connection.execute("SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = ?", (name,)).fetchone()
            is not None
        )

    def _is_unbound_populated_precontinuity_audit(self, source: sqlite3.Connection, audit: sqlite3.Connection) -> bool:
        source_head = source.execute(
            "SELECT committed_generation, committed_head_sha256 FROM audit_continuity_control WHERE singleton = 1"
        ).fetchone()
        audit_head = audit.execute(
            "SELECT generation, head_sha256 FROM audit_continuity_head WHERE singleton = 1"
        ).fetchone()
        if source_head != (0, AUDIT_CONTINUITY_GENESIS_HEAD_SHA256) or audit_head != (
            0,
            AUDIT_CONTINUITY_GENESIS_HEAD_SHA256,
        ):
            return False
        tables = tuple(
            str(row[0])
            for row in audit.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "AND name != 'audit_continuity_head' ORDER BY name"
            )
        )
        for name in tables:
            quoted_name = name.replace('"', '""')
            if audit.execute(f'SELECT 1 FROM "{quoted_name}" LIMIT 1').fetchone() is not None:
                return True
        return False

    def _assert_precontinuity_audit_semantics(self, connection: sqlite3.Connection, mutation: AuditMutation) -> None:
        expected = mutation.payload.get("audit_semantic_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise AuditContinuityError("pre-continuity binding lacks an audit semantic sha256")
        if _audit_semantic_sha256_connection(connection) != expected:
            raise AuditContinuityError("pre-continuity audit journal differs from its authenticated migration evidence")

    def _require_paths(self) -> None:
        for path in (self.source_path, self.audit_path):
            try:
                metadata = path.lstat()
            except FileNotFoundError as exc:
                raise AuditContinuityError("audit continuity requires initialized source.db and audit.db") from exc
            except OSError as exc:
                raise AuditContinuityError(f"cannot inspect audit continuity tier entry: {path}") from exc
            if not stat.S_ISREG(metadata.st_mode):
                raise AuditContinuityError(f"audit continuity tier entry is not an owned regular file: {path}")


__all__ = [
    "AuditContinuityCoordinator",
    "AuditContinuityError",
    "AuditMutation",
    "audit_semantic_sha256",
    "prepared_audit_continuity_command",
]
