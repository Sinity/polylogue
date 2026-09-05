"""Immutable admission policy derived from durable excision intent."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from polylogue.core.enums import AssertionKind
from polylogue.storage.sqlite.archive_tiers.bootstrap import archive_tier_spec, initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


class ExcisionPolicyError(RuntimeError):
    """A candidate or write does not satisfy the durable excision policy."""


def _current_schema_identity() -> str:
    return ";".join(
        f"{tier.value}:{archive_tier_spec(tier).version}"
        for tier in (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT)
    )


@dataclass(frozen=True, slots=True)
class ExcisionPolicySnapshot:
    """Content identities forbidden by one exact user/audit state.

    Only hashes and assertion references are retained.  Reasons, actors, and
    other user-entered values never cross into source-generation state.
    """

    removed_hashes: tuple[bytes, ...]
    assertion_refs: tuple[str, ...]
    user_generation: int
    audit_generation: int
    audit_head: str
    source_generation_id: str | None
    code_identity: str = "polylogue.excision-policy.v1"
    schema_identity: str = field(default_factory=_current_schema_identity)

    def __post_init__(self) -> None:
        if any(len(value) != 32 for value in self.removed_hashes):
            raise ValueError("excision policy hashes must be SHA-256 digests")
        if tuple(sorted(set(self.removed_hashes))) != self.removed_hashes:
            raise ValueError("excision policy hashes must be sorted and unique")

    @property
    def digest(self) -> str:
        payload = {
            "code": self.code_identity,
            "schema": self.schema_identity,
            "user_generation": self.user_generation,
            "audit_generation": self.audit_generation,
            "audit_head": self.audit_head,
            "source_generation_id": self.source_generation_id,
            "assertion_refs": self.assertion_refs,
            "removed_hashes": [value.hex() for value in self.removed_hashes],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def allows(self, blob_hash: bytes) -> bool:
        return blob_hash not in self.removed_hashes

    def assert_admissible(self, blob_hash: bytes, *, source_path: str) -> None:
        if not self.allows(blob_hash):
            raise ExcisionPolicyError(f"content at {source_path!r} is excluded by excision policy {self.digest}")


def _generation(conn: sqlite3.Connection, table: str, column: str, default: int = 0) -> int:
    row = conn.execute(f"SELECT {column} FROM {table} WHERE singleton=1").fetchone()
    return int(row[0]) if row and row[0] is not None else default


def build_excision_policy_snapshot(
    archive_root: Path,
    *,
    source_generation_id: str | None = None,
) -> ExcisionPolicySnapshot:
    """Read canonical user intent and audit continuity into an immutable value."""
    user_db = archive_root / archive_tier_spec(ArchiveTier.USER).filename
    audit_db = archive_root / archive_tier_spec(ArchiveTier.AUDIT).filename
    initialize_archive_database(user_db, ArchiveTier.USER)
    initialize_archive_database(audit_db, ArchiveTier.AUDIT)
    user = sqlite3.connect(user_db)
    audit = sqlite3.connect(audit_db)
    try:
        hashes: set[bytes] = set()
        refs: list[str] = []
        rows = user.execute(
            "SELECT assertion_id, value_json, status FROM assertions WHERE lower(kind)=?",
            (AssertionKind.EXCISION_RECORD.value.lower(),),
        )
        for assertion_id, value_json, status in rows:
            if status in {"deleted", "rejected"}:
                continue
            refs.append(str(assertion_id))
            try:
                value = json.loads(str(value_json))
            except (TypeError, json.JSONDecodeError):
                raise ExcisionPolicyError(f"invalid excision assertion {assertion_id}") from None
            if not isinstance(value, dict):
                continue
            for raw_hash in value.get("removed_blob_hashes", ()):
                if isinstance(raw_hash, str):
                    try:
                        decoded = bytes.fromhex(raw_hash)
                    except ValueError:
                        raise ExcisionPolicyError(f"invalid excision hash in assertion {assertion_id}") from None
                    if len(decoded) != 32:
                        raise ExcisionPolicyError(f"invalid excision hash in assertion {assertion_id}")
                    hashes.add(decoded)
        head = audit.execute("SELECT generation, head_sha256 FROM audit_continuity_head WHERE singleton=1").fetchone()
        audit_generation = int(head[0]) if head else 0
        audit_head = str(head[1]) if head else ""
        return ExcisionPolicySnapshot(
            tuple(sorted(hashes)),
            tuple(sorted(set(refs))),
            _generation(user, "query_unit_frame_state", "epoch"),
            audit_generation,
            audit_head,
            source_generation_id,
        )
    finally:
        user.close()
        audit.close()


def read_excision_policy_projection(conn: sqlite3.Connection, source_generation_id: str) -> dict[str, object] | None:
    """Read one replaceable generation-local policy binding."""
    if (
        conn.execute("SELECT 1 FROM sqlite_schema WHERE type='table' AND name='excision_policy_projections'").fetchone()
        is None
    ):
        return None
    row = conn.execute(
        """SELECT policy_digest, user_generation, audit_generation, audit_head,
                  assertion_refs_json, generated_at_ms
           FROM excision_policy_projections WHERE source_generation_id=?""",
        (source_generation_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "policy_digest": str(row[0]),
        "user_generation": int(row[1]),
        "audit_generation": int(row[2]),
        "audit_head": str(row[3]),
        "assertion_refs_json": str(row[4]),
        "generated_at_ms": int(row[5]),
    }


__all__ = [
    "ExcisionPolicyError",
    "ExcisionPolicySnapshot",
    "build_excision_policy_snapshot",
    "read_excision_policy_projection",
]
