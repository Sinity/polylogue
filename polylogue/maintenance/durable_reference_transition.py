"""The rebuild's durable-reference rebind step.

:mod:`polylogue.maintenance.assertion_transition` owns the exact classification
and the atomic two-tier write. This module is what a rebuild calls: it derives
every input of a :class:`TransitionBinding` from the archive's own files,
produces the message-identity map the ``result_set_members`` successor
transition requires, and publishes one receipt naming every disposition.

Identity ownership is the axis. ``sessions.session_id``, ``messages.message_id``
and ``blocks.block_id`` are generated columns, so their text is a function of
the index schema and moves when that schema does. Every other public ref kind is
content-addressed, externally owned, or owned by a durable tier, so an index
rebuild cannot move it.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from polylogue.core.enums import Origin
from polylogue.core.refs import EvidenceRef, EvidenceRefKind, ObjectRef, parse_public_ref
from polylogue.maintenance.assertion_transition import (
    AssertionTransitionPlan,
    DurableReference,
    IdentityMigrationMap,
    ObjectRefDisposition,
    ObjectRefReconciliationError,
    SourceIdentityClaims,
    TransitionBinding,
    apply_assertion_transition,
    enumerate_durable_reference_inventory,
    reconcile_object_refs,
)
from polylogue.storage.sqlite.archive_tiers.schema_identity import DerivedTier, read_schema_identity

#: Receipt schema for one planned or applied durable-reference transition.
TRANSITION_RECEIPT_SCHEMA = "polylogue.durable-reference-transition.v1"
#: ``.maintenance-state`` child holding the receipts.
TRANSITION_RECEIPT_DIRECTORY = "durable-reference-transitions"

#: Object-ref kinds whose text is a function of index.db rows. ``action`` reads
#: through the ``actions`` view over ``blocks``, and a ``delegation`` id is
#: either a tool-use block id or a pair of session ids, so both move with the
#: index even though neither has an exact probe here yet.
INDEX_GOVERNED_KINDS = frozenset({"session", "message", "block", "action", "delegation"})
#: The subset of :data:`INDEX_GOVERNED_KINDS` this route can resolve exactly.
RESOLVABLE_KINDS = frozenset({"session", "message", "block"})
#: The closed source-origin vocabulary a real session id starts with.
_ORIGIN_TOKENS = frozenset(origin.value for origin in Origin)

_GRAIN_RELATIONS: dict[EvidenceRefKind, tuple[str, str]] = {
    "session": ("sessions", "session_id"),
    "message": ("messages", "message_id"),
    "block": ("blocks", "block_id"),
}


@dataclass(frozen=True, slots=True)
class DurableReferenceTransition:
    """One planned transition and the evidence that binds it to two archives."""

    binding: TransitionBinding
    plan: AssertionTransitionPlan
    inventory: tuple[DurableReference, ...]
    candidate_index_path: str
    predecessor_index_path: str

    @property
    def dispositions(self) -> dict[str, int]:
        counts = {disposition.value: 0 for disposition in ObjectRefDisposition}
        for row in self.plan.rows:
            counts[row.disposition.value] += 1
        return counts

    @property
    def is_blocked(self) -> bool:
        return any(
            row.disposition in {ObjectRefDisposition.BLOCKING_MISSING, ObjectRefDisposition.AMBIGUOUS}
            for row in self.plan.rows
        )

    def receipt(self, *, applied: bool) -> dict[str, Any]:
        """The record of what this transition classified, in full."""
        return {
            "schema": TRANSITION_RECEIPT_SCHEMA,
            "applied": applied,
            "binding": self.binding.as_dict(),
            "binding_digest": self.binding.digest(),
            "plan_digest": self.plan.digest(),
            "candidate_index_path": self.candidate_index_path,
            "predecessor_index_path": self.predecessor_index_path,
            "durable_reference_cells": len(self.inventory),
            "classified_references": len(self.plan.rows),
            "dispositions": self.dispositions,
            "forward": [list(pair) for pair in self.plan.forward],
            "references": [row.as_dict() for row in self.plan.rows],
        }

    def receipt_filename(self) -> str:
        return f"{self.plan.digest()}.json"


def source_session_claims(source_conn: sqlite3.Connection) -> SourceIdentityClaims:
    """Session identities the durable source tier still holds acquired bytes for.

    A claim makes a candidate's failure to produce that session
    ``blocking-missing`` instead of ``orphaned``: the evidence to rebuild it is
    present, so its absence is an adjudication, not a loss. Source identity is
    session-grained -- ``raw_sessions`` records ``(origin, native_id)`` and
    nothing finer -- so message and block refs are never claimed here.
    """
    if not _table_exists(source_conn, "raw_sessions"):
        raise ObjectRefReconciliationError("source tier has no raw_sessions relation to claim identities from")
    rows = source_conn.execute("SELECT origin, native_id FROM raw_sessions WHERE native_id IS NOT NULL")
    return SourceIdentityClaims.from_refs(f"session:{origin}:{native_id}" for origin, native_id in rows)


def index_identity(conn: sqlite3.Connection) -> str:
    """The identity a plan binds an index generation to.

    The stamped derived-schema identity when the generation carries one. An
    index written before that ledger existed carries no stamp, and its own DDL
    is then the only identity it has -- which is exactly the thing whose change
    moves a generated id.
    """
    stamped = read_schema_identity(conn, DerivedTier.INDEX)
    if stamped is not None:
        return stamped
    rows = conn.execute(
        "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
    ).fetchall()
    return _digest([[str(column) if column is not None else None for column in row] for row in rows])


def resolve_candidate_references(conn: sqlite3.Connection, refs: Iterable[str]) -> frozenset[str]:
    """The subset of ``refs`` an index generation carries.

    A ref whose kind the index does not govern is always a member: a rebuild
    cannot move an identity it does not compute.
    """
    resolved: set[str] = set()
    for ref in refs:
        target = governed_target(ref)
        if target is None:
            resolved.add(ref)
            continue
        relation, column, identity = target
        if conn.execute(f"SELECT 1 FROM {relation} WHERE {column} = ?", (identity,)).fetchone() is not None:
            resolved.add(ref)
    return frozenset(resolved)


def governed_target(ref: str) -> tuple[str, str, str] | None:
    """``(relation, column, exact id)`` for an index-governed ref, else ``None``.

    Raises for a kind the index governs but this route cannot probe, so an
    unprobeable identity can never be silently reported as surviving.
    """
    parsed = parse_public_ref(ref)
    if isinstance(parsed, EvidenceRef):
        # An evidence ref is only an archive pointer when its session component
        # names a real Origin. Opaque-grammar columns carry values such as a
        # runtime build id that parse as this shape and address nothing in the
        # index; reporting one as a lost session is a false alarm.
        if not _names_an_origin(parsed.session_id):
            return None
        grain: EvidenceRefKind = parsed.ref_kind
        identity = {
            "session": parsed.session_id,
            "message": str(parsed.message_id),
            "block": f"{parsed.message_id}:{parsed.block_index}",
        }[grain]
    else:
        if parsed.kind not in INDEX_GOVERNED_KINDS:
            return None
        if parsed.kind not in RESOLVABLE_KINDS:
            raise ObjectRefReconciliationError(f"index-governed ref kind has no exact candidate probe: {parsed.kind}")
        grain = cast("EvidenceRefKind", parsed.kind)
        identity = ":".join((parsed.object_id, *parsed.qualifiers))
    relation, column = _GRAIN_RELATIONS[grain]
    return relation, column, identity


def message_identity_migration_map(
    conn: sqlite3.Connection,
    refs: Sequence[str],
    *,
    producer: str,
) -> IdentityMigrationMap | None:
    """Map old message-grained refs onto the ids the candidate actually holds.

    The old form is ``<session_id>:<native_id or position.variant>``; the new
    form disambiguates the two with an ``n:``/``p:`` infix. Which branch fired
    is not recoverable from the old text, so each old id is probed as both
    successors against the candidate. Exactly one hit is the map entry, two are
    a genuine collision and refuse, none leaves the ref unmapped for the
    orphan/blocking classification to decide.
    """
    entries: list[tuple[str, str]] = []
    for ref in dict.fromkeys(refs):
        successor = _successor_ref(conn, ref)
        if successor is not None and successor != ref:
            entries.append((ref, successor))
    if not entries:
        return None
    return IdentityMigrationMap(producer=producer, entries=tuple(entries))


def plan_durable_reference_transition(
    *,
    user_conn: sqlite3.Connection,
    audit_conn: sqlite3.Connection,
    candidate_index_conn: sqlite3.Connection,
    predecessor_index_conn: sqlite3.Connection,
    source_claims: SourceIdentityClaims,
    package_version: str,
    producer: str,
    candidate_index_path: str,
    predecessor_index_path: str,
) -> DurableReferenceTransition:
    """Classify every durable public reference against a candidate index."""
    inventory = enumerate_durable_reference_inventory(user_conn, audit_conn)
    refs = tuple(dict.fromkeys(item.value for item in inventory))
    predecessor = resolve_candidate_references(predecessor_index_conn, refs)
    present = resolve_candidate_references(candidate_index_conn, refs)
    migration = message_identity_migration_map(
        candidate_index_conn, tuple(ref for ref in refs if ref not in present), producer=producer
    )
    # The candidate set has to carry the map's endpoints too: a migrated ref is
    # classified by whether the id it moves *to* is really there.
    targets = () if migration is None else tuple(new for _old, new in migration.entries)
    candidate = resolve_candidate_references(candidate_index_conn, (*refs, *targets))
    binding = TransitionBinding(
        predecessor_digest=index_identity(predecessor_index_conn),
        candidate_digest=index_identity(candidate_index_conn),
        source_seal=_digest(sorted(source_claims.refs)),
        package_version=package_version,
        schema_versions=(("user", _user_version(user_conn)), ("audit", _user_version(audit_conn))),
        migration_map_digest=None if migration is None else migration.digest(),
    )
    plan = reconcile_object_refs(
        refs,
        candidate_refs=candidate,
        source_claims=source_claims,
        predecessor_refs=predecessor,
        migration_map=migration,
        binding=binding,
    )
    return DurableReferenceTransition(
        binding=binding,
        plan=plan,
        inventory=inventory,
        candidate_index_path=candidate_index_path,
        predecessor_index_path=predecessor_index_path,
    )


def apply_durable_reference_transition(
    *,
    user_conn: sqlite3.Connection,
    audit_conn: sqlite3.Connection,
    transition: DurableReferenceTransition,
    verified_backup: bool,
) -> None:
    """Apply one planned transition to both durable tiers, or leave them untouched."""
    apply_assertion_transition(
        user_conn,
        transition.plan,
        binding=transition.binding,
        verified_backup=verified_backup,
        audit_conn=audit_conn,
    )
    user_conn.commit()
    audit_conn.commit()


def publish_transition_receipt(archive_root: Path, transition: DurableReferenceTransition, *, applied: bool) -> str:
    """Write the transition's receipt into the archive's maintenance state."""
    from polylogue.maintenance.receipt_fs import atomic_replace_receipt, maintenance_receipt_directory

    payload = json.dumps(transition.receipt(applied=applied), indent=2, sort_keys=True).encode("utf-8")
    filename = transition.receipt_filename()
    with maintenance_receipt_directory(archive_root, TRANSITION_RECEIPT_DIRECTORY) as directory_fd:
        atomic_replace_receipt(directory_fd, filename, payload)
    return filename


def _names_an_origin(session_id: str) -> bool:
    origin, separator, native_id = session_id.partition(":")
    return bool(separator and native_id) and origin in _ORIGIN_TOKENS


def _successor_ref(conn: sqlite3.Connection, ref: str) -> str | None:
    """The candidate's id for one old message- or block-grained ref."""
    parsed = parse_public_ref(ref)
    if isinstance(parsed, EvidenceRef):
        if parsed.message_id is None:
            return None
        successor = _successor_message_id(conn, parsed.message_id)
        if successor is None:
            return None
        evidence = EvidenceRef(parsed.session_id, successor, parsed.block_index).format()
        return evidence if _resolves(conn, evidence) else None
    if parsed.kind == "message":
        successor = _successor_message_id(conn, parsed.object_id)
        return None if successor is None else ObjectRef("message", successor).format()
    if parsed.kind == "block":
        successor = _successor_message_id(conn, parsed.object_id)
        if successor is None:
            return None
        block = ObjectRef("block", successor, parsed.qualifiers).format()
        return block if _resolves(conn, block) else None
    return None


def _successor_message_id(conn: sqlite3.Connection, old: str) -> str | None:
    sessions = _candidate_session_prefixes(conn, old)
    if len(sessions) > 1:
        raise ObjectRefReconciliationError(f"old message id matches more than one candidate session: {old}")
    if not sessions:
        return None
    session_id, tail = sessions[0]
    successors = [
        candidate
        for candidate in (f"{session_id}:n:{tail}", f"{session_id}:p:{tail}")
        if conn.execute("SELECT 1 FROM messages WHERE message_id = ?", (candidate,)).fetchone() is not None
    ]
    if len(successors) > 1:
        raise ObjectRefReconciliationError(f"old message id has two candidate successors: {old}")
    return successors[0] if successors else None


def _candidate_session_prefixes(conn: sqlite3.Connection, old: str) -> list[tuple[str, str]]:
    """Every ``(session_id, tail)`` split of ``old`` the candidate knows a session for."""
    found: list[tuple[str, str]] = []
    for index, character in enumerate(old):
        if character != ":":
            continue
        session_id, tail = old[:index], old[index + 1 :]
        if not tail:
            continue
        if conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone() is not None:
            found.append((session_id, tail))
    return found


def _resolves(conn: sqlite3.Connection, ref: str) -> bool:
    target = governed_target(ref)
    if target is None:
        return True
    relation, column, identity = target
    return conn.execute(f"SELECT 1 FROM {relation} WHERE {column} = ?", (identity,)).fetchone() is not None


def _user_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0] or 0) if row is not None else 0


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute("SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?", (name,)).fetchone()
        is not None
    )


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "INDEX_GOVERNED_KINDS",
    "RESOLVABLE_KINDS",
    "TRANSITION_RECEIPT_DIRECTORY",
    "TRANSITION_RECEIPT_SCHEMA",
    "DurableReferenceTransition",
    "apply_durable_reference_transition",
    "governed_target",
    "index_identity",
    "message_identity_migration_map",
    "plan_durable_reference_transition",
    "publish_transition_receipt",
    "resolve_candidate_references",
    "source_session_claims",
]
