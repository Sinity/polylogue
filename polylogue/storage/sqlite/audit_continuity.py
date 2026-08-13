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
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

from polylogue.storage.sqlite.audit_leaf import AuditLeafError, VerifiedAuditLeaf, open_verified_audit_connection

_FORMAT = "polylogue.audit-continuity-command.v1"
AUDIT_CONTINUITY_GENESIS_HEAD_SHA256 = "3230fdd585a4fd2d71b7d720bcfe5d697ff120fdb32aecde394e89d407c7198f"
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


def audit_semantic_sha256(path: Path) -> str:
    """Hash audit content while excluding the self-mutating continuity head."""

    try:
        with open_verified_audit_connection(path) as connection:
            lines = (line for line in connection.iterdump() if "audit_continuity_head" not in line)
            return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    except (AuditLeafError, sqlite3.DatabaseError) as exc:
        raise AuditContinuityError("cannot hash audit content for continuity validation") from exc


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

        if not self.source_path.is_file() or not self.audit_path.is_file():
            return False
        try:
            with (
                closing(sqlite3.connect(self.source_path)) as source,
                open_verified_audit_connection(self.audit_path) as audit,
            ):
                source.execute("SELECT 1 FROM audit_continuity_control WHERE singleton = 1").fetchone()
                audit.execute("SELECT 1 FROM audit_continuity_head WHERE singleton = 1").fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return False
            raise AuditContinuityError("cannot inspect audit continuity compatibility state") from exc
        except sqlite3.DatabaseError as exc:
            raise AuditContinuityError("cannot inspect audit continuity compatibility state") from exc
        return True

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
                closing(sqlite3.connect(self.source_path)) as source,
                open_verified_audit_connection(self.audit_path) as audit,
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

    def _phase(self, name: str, mutation: AuditMutation) -> None:
        if self._phase_hook is not None:
            self._phase_hook(name, mutation)

    def _prepare(self, mutation: AuditMutation) -> dict[str, object]:
        self._require_paths()
        with closing(sqlite3.connect(self.source_path)) as conn, conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT committed_generation, committed_head_sha256, pending_mutation_id FROM audit_continuity_control WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise AuditContinuityError("source audit continuity control is missing")
            if row[2] is not None:
                raise AuditContinuityError("another audit continuity mutation is already pending")
            generation = int(row[0])
            previous_head = str(row[1])
            command = mutation.command()
            command_sha256 = _sha256(command)
            prepared = {
                "format": _FORMAT,
                "prior_generation": generation,
                "prior_head_sha256": previous_head,
                "next_generation": generation + 1,
                "command": command,
                "command_sha256": command_sha256,
                "next_head_sha256": _sha256({"previous_head_sha256": previous_head, "command_sha256": command_sha256}),
            }
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
        with closing(sqlite3.connect(self.source_path)) as conn:
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
            if mutation.kind == "rebind":
                # A retry after the audit-side commit sees the target head and
                # returns above. Any other state must still prove that the
                # exact authenticated image is present before it can rebind.
                self._assert_rebind_image(mutation)
            if current[:2] != prior:
                if allow_rebind and mutation.kind == "rebind":
                    pass
                else:
                    raise AuditContinuityError("audit continuity head does not match the prepared source command")
            result = cast(_T, None) if mutation.kind == "rebind" else apply(conn, mutation)
            conn.execute(
                "UPDATE audit_continuity_head SET generation = ?, head_sha256 = ?, mutation_id = ?, advanced_at_ms = ? WHERE singleton = 1",
                (*target, mutation.mutation_id, mutation.created_at_ms),
            )
            conn.commit()
            return result

    def _promote(self, prepared: Mapping[str, object]) -> None:
        mutation = AuditMutation.from_command(prepared["command"])
        with closing(sqlite3.connect(self.source_path)) as conn, conn:
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
            with closing(sqlite3.connect(self.source_path)) as source, source:
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

    def _assert_committed_head_matches_audit(self) -> None:
        with (
            closing(sqlite3.connect(self.source_path)) as source,
            open_verified_audit_connection(self.audit_path) as audit,
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

    def _require_paths(self) -> None:
        if not self.source_path.is_file() or not self.audit_path.is_file():
            raise AuditContinuityError("audit continuity requires initialized source.db and audit.db")


__all__ = ["AuditContinuityCoordinator", "AuditContinuityError", "AuditMutation", "audit_semantic_sha256"]
