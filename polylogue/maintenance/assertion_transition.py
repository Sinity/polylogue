"""Exact, candidate-bound reconciliation of durable assertion references.

This module deliberately knows nothing about similarity or historical index
rows.  A reference is preserved when its exact identity is in the candidate,
or is classified from a sealed source claim.  Identity changes are accepted
only as a complete, producer-supplied map.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from polylogue.core.hashing import hash_payload
from polylogue.core.refs import EvidenceRef, ObjectRef, parse_public_ref
from polylogue.storage.sqlite.archive_tiers.durable_references import (
    DurableReferenceField,
    DurableTier,
    durable_reference_relations,
)
from polylogue.storage.sqlite.query_objects import membership_merkle_root


class ObjectRefDisposition(StrEnum):
    PRESERVED = "preserved"
    EXPECTED_RESTORED = "expected-restored"
    EXPLICITLY_MIGRATED = "explicitly-migrated"
    ORPHANED = "orphaned"
    BLOCKING_MISSING = "blocking-missing"
    AMBIGUOUS = "ambiguous"


class ObjectRefReconciliationError(ValueError):
    """Raised when exact identity evidence is incomplete or contradictory."""


@dataclass(frozen=True, slots=True)
class SourceIdentityClaims:
    """The sealed source identity set used by one reconciliation."""

    refs: frozenset[str]

    @classmethod
    def from_refs(cls, refs: Iterable[str | ObjectRef | EvidenceRef]) -> SourceIdentityClaims:
        return cls(frozenset(_public_ref_text(ref) for ref in refs))


@dataclass(frozen=True, slots=True)
class IdentityMigrationMap:
    """A producer-issued total old-to-new identity map.

    A sequence is used instead of a mapping so duplicate and conflicting
    endpoints cannot be silently collapsed by the caller.
    """

    producer: str
    entries: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.producer.strip():
            raise ObjectRefReconciliationError("identity migration producer is required")
        old_to_new: dict[str, str] = {}
        new_to_old: dict[str, str] = {}
        for old, new in self.entries:
            old_text, new_text = _public_ref_text(old), _public_ref_text(new)
            prior = old_to_new.get(old_text)
            if prior is not None and prior != new_text:
                raise ObjectRefReconciliationError("identity migration map has conflicting old endpoints")
            prior_old = new_to_old.get(new_text)
            if prior_old is not None and prior_old != old_text:
                raise ObjectRefReconciliationError("identity migration map has conflicting new endpoints")
            old_to_new[old_text] = new_text
            new_to_old[new_text] = old_text
        object.__setattr__(self, "entries", tuple(old_to_new.items()))

    def target_for(self, ref: str | ObjectRef | EvidenceRef) -> str | None:
        try:
            ref_text = _public_ref_text(ref)
        except ObjectRefReconciliationError:
            return None
        return dict(self.entries).get(ref_text)

    def digest(self) -> str:
        return _digest({"producer": self.producer, "entries": self.entries})

    @property
    def inverse(self) -> tuple[tuple[str, str], ...]:
        return tuple((new, old) for old, new in self.entries)


@dataclass(frozen=True, slots=True)
class ReconciledObjectRef:
    source: str
    disposition: ObjectRefDisposition
    target: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {"source": self.source, "disposition": self.disposition.value, "target": self.target}


@dataclass(frozen=True, slots=True)
class DurableReference:
    """One reference-bearing durable relation cell.

    ``grammar`` is explicit because evidence fields may contain provider
    evidence or external locators which are intentionally opaque to an
    ObjectRef transition.
    """

    relation: str
    column: str
    identity: str
    grammar: str
    value: str


@dataclass(frozen=True, slots=True)
class TransitionBinding:
    """Immutable identities which make a plan applicable to one candidate."""

    predecessor_digest: str
    candidate_digest: str
    source_seal: str
    package_version: str
    schema_versions: tuple[tuple[str, int], ...]
    migration_map_digest: str | None = None

    def digest(self) -> str:
        return _digest(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "predecessor_digest": self.predecessor_digest,
            "candidate_digest": self.candidate_digest,
            "source_seal": self.source_seal,
            "package_version": self.package_version,
            "schema_versions": list(self.schema_versions),
            "migration_map_digest": self.migration_map_digest,
        }


@dataclass(frozen=True, slots=True)
class AssertionTransitionPlan:
    binding: TransitionBinding
    rows: tuple[ReconciledObjectRef, ...]
    forward: tuple[tuple[str, str], ...]
    inverse: tuple[tuple[str, str], ...]

    @property
    def is_empty(self) -> bool:
        return not self.forward

    def digest(self) -> str:
        return _digest(
            {
                "binding": self.binding.as_dict(),
                "rows": [row.as_dict() for row in self.rows],
                "forward": self.forward,
                "inverse": self.inverse,
            }
        )


def reconcile_object_refs(
    refs: Iterable[str | ObjectRef | EvidenceRef],
    *,
    candidate_refs: Iterable[str | ObjectRef | EvidenceRef],
    source_claims: SourceIdentityClaims,
    predecessor_refs: Iterable[str | ObjectRef | EvidenceRef] = (),
    migration_map: IdentityMigrationMap | None = None,
    binding: TransitionBinding,
) -> AssertionTransitionPlan:
    """Classify every durable reference against exact candidate evidence."""

    requested = tuple(dict.fromkeys(_public_ref_text(ref) for ref in refs))
    candidate = frozenset(_public_ref_text(ref) for ref in candidate_refs)
    predecessor = frozenset(_public_ref_text(ref) for ref in predecessor_refs)
    if migration_map is not None:
        mapped_old = {old for old, _ in migration_map.entries}
        missing = mapped_old - set(requested)
        if missing:
            raise ObjectRefReconciliationError("identity migration map is not total for durable references")
        if binding.migration_map_digest is not None and binding.migration_map_digest != migration_map.digest():
            raise ObjectRefReconciliationError("identity migration map is not bound to this transition")
    elif binding.migration_map_digest is not None:
        raise ObjectRefReconciliationError("transition binding requires an identity migration map")

    rows: list[ReconciledObjectRef] = []
    forward: list[tuple[str, str]] = []
    for source in requested:
        target = migration_map.target_for(source) if migration_map else None
        if target is not None:
            if target not in candidate:
                disposition = ObjectRefDisposition.BLOCKING_MISSING
                target = None
            else:
                disposition = ObjectRefDisposition.EXPLICITLY_MIGRATED
                forward.append((source, target))
        elif source in candidate:
            disposition = ObjectRefDisposition.PRESERVED
        elif source in source_claims.refs:
            disposition = ObjectRefDisposition.BLOCKING_MISSING
        else:
            disposition = ObjectRefDisposition.ORPHANED
        if disposition is ObjectRefDisposition.PRESERVED and source not in predecessor and source in source_claims.refs:
            disposition = ObjectRefDisposition.EXPECTED_RESTORED
        rows.append(ReconciledObjectRef(source, disposition, target))

    inverse = tuple((new, old) for old, new in forward)
    return AssertionTransitionPlan(binding, tuple(rows), tuple(forward), inverse)


def enumerate_assertion_object_refs(conn: sqlite3.Connection) -> tuple[str, ...]:
    """Enumerate assertion and result-set ObjectRefs without a tier census."""

    refs: list[str] = []
    rows = conn.execute(
        "SELECT scope_ref, target_ref, evidence_refs_json, supersedes_json FROM assertions ORDER BY assertion_id"
    )
    for row in rows:
        values: list[object] = [row[0], row[1]]
        for index in (2, 3):
            try:
                decoded = json.loads(str(row[index]))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ObjectRefReconciliationError("assertion reference JSON is invalid") from exc
            if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
                raise ObjectRefReconciliationError("assertion reference JSON must be a string list")
            values.extend(decoded)
        for value in values:
            if value is None:
                continue
            try:
                refs.append(ObjectRef.parse(str(value)).format())
            except (TypeError, ValueError):
                continue
    if _table_exists(conn, "result_set_members"):
        for (value,) in conn.execute("SELECT member_ref FROM result_set_members ORDER BY result_set_id, rank"):
            try:
                refs.append(ObjectRef.parse(str(value)).format())
            except (TypeError, ValueError):
                continue
    return tuple(dict.fromkeys(refs))


def enumerate_durable_reference_inventory(
    user_conn: sqlite3.Connection,
    audit_conn: sqlite3.Connection | None = None,
) -> tuple[DurableReference, ...]:
    """Return every typed public reference in the declared durable relations."""

    inventory: list[DurableReference] = []
    inventory.extend(_inventory_for_connection(user_conn, "user"))
    if audit_conn is not None:
        inventory.extend(_inventory_for_connection(audit_conn, "audit"))
    return tuple(inventory)


def _inventory_for_connection(
    conn: sqlite3.Connection,
    tier: DurableTier,
) -> tuple[DurableReference, ...]:
    relations = durable_reference_relations(tier)
    columns_by_table = _schema_columns(conn)
    tables = frozenset(columns_by_table)
    declared_columns = {(relation.table, field.column) for relation in relations for field in relation.fields}
    for table, columns in columns_by_table.items():
        for column in columns:
            if _looks_like_reference_column(column) and (table, column) not in declared_columns:
                raise ObjectRefReconciliationError(
                    f"durable reference column lacks a descriptor: {tier}.{table}.{column}"
                )

    inventory: list[DurableReference] = []
    for relation in relations:
        if relation.table not in tables:
            continue
        present = set(columns_by_table[relation.table])
        fields = tuple(field for field in relation.fields if field.column in present)
        if not fields:
            continue
        identity_columns = tuple(column for column in relation.identity_columns if column in present)
        if len(identity_columns) != len(relation.identity_columns):
            raise ObjectRefReconciliationError(f"durable relation identity is incomplete: {tier}.{relation.table}")
        selected = tuple(dict.fromkeys((*identity_columns, *(field.column for field in fields))))
        quoted = ", ".join(_quote_identifier(column) for column in selected)
        for row in conn.execute(f"SELECT {quoted} FROM {_quote_identifier(relation.table)}"):
            values = dict(zip(selected, row, strict=True))
            identity = ":".join(str(values[column]) for column in identity_columns)
            for field in fields:
                raw = values[field.column]
                for value in _field_values(field, raw):
                    try:
                        parsed = _parse_durable_public_ref(value)
                    except (TypeError, ValueError) as exc:
                        if field.grammar == "public":
                            raise ObjectRefReconciliationError(
                                f"unsupported public reference in {tier}.{relation.table}.{field.column}"
                            ) from exc
                        continue
                    inventory.append(
                        DurableReference(
                            f"{relation.table}:{identity}", field.column, identity, field.grammar, parsed.format()
                        )
                    )
    return tuple(inventory)


def apply_assertion_transition(
    conn: sqlite3.Connection,
    plan: AssertionTransitionPlan,
    *,
    binding: TransitionBinding,
    verified_backup: bool,
    audit_conn: sqlite3.Connection | None = None,
    append_audit: Callable[[AssertionTransitionPlan], None] | None = None,
) -> None:
    """Apply a declared map atomically to all declared durable references.

    The caller must supply the same binding and a verified durable backup. User
    and audit connections are coordinated with savepoints; the caller owns the
    outer transactions and commits them only after both succeed.
    """

    if binding != plan.binding:
        raise ObjectRefReconciliationError("transition binding is stale")
    if not verified_backup:
        raise ObjectRefReconciliationError("explicit identity transition requires a verified backup")
    if any(row.disposition is ObjectRefDisposition.BLOCKING_MISSING for row in plan.rows):
        raise ObjectRefReconciliationError("cannot apply a plan with blocking missing references")
    if any(row.disposition is ObjectRefDisposition.AMBIGUOUS for row in plan.rows):
        raise ObjectRefReconciliationError("cannot apply an ambiguous plan")
    if not plan.forward:
        return
    _validate_reference_catalog(conn, "user")
    if audit_conn is not None:
        _validate_reference_catalog(audit_conn, "audit")
    connections: tuple[sqlite3.Connection, ...] = (
        (conn,) if audit_conn is None or audit_conn is conn else (conn, audit_conn)
    )
    connections_by_tier: tuple[tuple[sqlite3.Connection, DurableTier], ...] = ((conn, "user"),) + (
        ((audit_conn, "audit"),) if audit_conn is not None and audit_conn is not conn else ()
    )
    _validate_plan_inventory(connections_by_tier, plan)
    _reject_sealed_reference_rewrites(connections_by_tier, plan)
    savepoints = tuple(f"assertion_transition_{index}" for index in range(len(connections)))
    # Releasing the outermost savepoint on a connection in autocommit would
    # commit that tier by itself, so a crash between the two releases could
    # leave the user transition durable with no audit record. Opening an
    # explicit transaction first makes every release a nested one: nothing is
    # durable on either tier until the caller commits both.
    for connection in connections:
        if not connection.in_transaction:
            connection.execute("BEGIN")
    for connection, savepoint in zip(connections, savepoints, strict=True):
        connection.execute(f"SAVEPOINT {savepoint}")
    try:
        for connection, tier in connections_by_tier:
            _apply_declared_references(connection, tier, plan)
            if tier == "user":
                successor_refs = _migrate_result_set_manifests(connection, plan)
                _apply_declared_references(connection, tier, plan, forward=successor_refs)
        if append_audit is not None:
            append_audit(plan)
        for connection, savepoint in zip(connections, savepoints, strict=True):
            connection.execute(f"RELEASE SAVEPOINT {savepoint}")
    except Exception:
        for connection, savepoint in reversed(tuple(zip(connections, savepoints, strict=True))):
            connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        raise


def _public_ref_text(value: str | ObjectRef | EvidenceRef) -> str:
    text = value.format() if isinstance(value, (ObjectRef, EvidenceRef)) else value
    try:
        return _parse_durable_public_ref(text).format()
    except (TypeError, ValueError) as exc:
        raise ObjectRefReconciliationError(f"invalid public reference: {text!r}") from exc


def _parse_durable_public_ref(value: str) -> ObjectRef | EvidenceRef:
    """Parse archive refs without mistaking paths and receipt ids for evidence."""

    lowered = value.lower()
    if (
        value.startswith(("/", "\\"))
        or "://" in value
        or lowered.startswith(("file:/", "path:", "receipt:", "codex-receipt:"))
    ):
        raise ValueError("external locator is not an archive evidence ref")
    return parse_public_ref(value)


def _tables(conn: sqlite3.Connection) -> tuple[str, ...]:
    return tuple(
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    )


def _schema_columns(conn: sqlite3.Connection) -> dict[str, tuple[str, ...]]:
    tables = _tables(conn)
    return {table: _columns(conn, table) for table in tables}


def _columns(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(str(row[1]) for row in conn.execute(f"PRAGMA table_info({_quote_identifier(table)})"))


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _looks_like_reference_column(column: str) -> bool:
    return column.endswith("_ref") or column.endswith("_refs_json")


def _validate_reference_catalog(conn: sqlite3.Connection, tier: DurableTier) -> None:
    relations = durable_reference_relations(tier)
    declared = {(relation.table, field.column) for relation in relations for field in relation.fields}
    columns_by_table = _schema_columns(conn)
    for table, columns in columns_by_table.items():
        for column in columns:
            if _looks_like_reference_column(column) and (table, column) not in declared:
                raise ObjectRefReconciliationError(
                    f"durable reference column lacks a descriptor: {tier}.{table}.{column}"
                )
    tables = frozenset(columns_by_table)
    version = _tier_schema_version(conn)
    if version <= 0 and not _has_complete_durable_schema_marker(conn, tier):
        # No migration cursor and no complete-schema marker: nothing here
        # claims to be a whole durable catalog.
        return
    for relation in relations:
        if version and relation.since_version > version:
            # The file predates the migration that introduced this relation.
            # A valid older archive is not an incomplete catalog.
            continue
        if relation.table not in tables:
            raise ObjectRefReconciliationError(f"durable relation is missing: {tier}.{relation.table}")
        present = set(columns_by_table[relation.table])
        missing = set(relation.identity_columns) - present
        missing.update(field.column for field in relation.fields if field.column not in present)
        if missing:
            raise ObjectRefReconciliationError(
                f"durable relation is incomplete: {tier}.{relation.table}.{','.join(sorted(missing))}"
            )


def _has_complete_durable_schema_marker(conn: sqlite3.Connection, tier: DurableTier) -> bool:
    """Whether an unversioned file still presents a whole durable catalog."""
    marker = "query_unit_frame_state" if tier == "user" else "archive_authority"
    return marker in _tables(conn)


def _tier_schema_version(conn: sqlite3.Connection) -> int:
    """Return the durable file's migration cursor (``PRAGMA user_version``)."""
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0] or 0) if row is not None else 0


def _field_values(
    field: DurableReferenceField,
    raw: object,
) -> tuple[str, ...]:
    if field.cardinality == "scalar":
        if raw is None:
            return ()
        if not isinstance(raw, str):
            raise ObjectRefReconciliationError("durable scalar reference must be text")
        return (raw,)
    if raw is None:
        return ()
    try:
        values = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ObjectRefReconciliationError("durable reference JSON is invalid") from exc
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ObjectRefReconciliationError("durable reference JSON must be a string list")
    return tuple(values)


def _mapped_reference(value: str, forward: dict[str, str]) -> str | None:
    replacement = forward.get(value)
    if replacement is not None:
        return replacement
    try:
        canonical = _public_ref_text(value)
    except ObjectRefReconciliationError:
        return None
    return forward.get(canonical)


def _apply_declared_references(
    conn: sqlite3.Connection,
    tier: DurableTier,
    plan: AssertionTransitionPlan,
    *,
    forward: dict[str, str] | None = None,
) -> None:
    columns_by_table = _schema_columns(conn)
    tables = frozenset(columns_by_table)
    forward = dict(plan.forward) if forward is None else forward
    for relation in durable_reference_relations(tier):
        if relation.transition != "rewrite" or relation.table not in tables:
            continue
        present = set(columns_by_table[relation.table])
        identity_columns = tuple(column for column in relation.identity_columns if column in present)
        fields = tuple(field for field in relation.fields if field.column in present)
        if len(identity_columns) != len(relation.identity_columns):
            raise ObjectRefReconciliationError(f"durable relation identity is incomplete: {tier}.{relation.table}")
        selected = tuple(dict.fromkeys((*identity_columns, *(field.column for field in fields))))
        quoted = ", ".join(_quote_identifier(column) for column in selected)
        for row in conn.execute(f"SELECT {quoted} FROM {_quote_identifier(relation.table)}"):
            values = dict(zip(selected, row, strict=True))
            identity_where = " AND ".join(f"{_quote_identifier(column)} = ?" for column in identity_columns)
            identity_values = tuple(values[column] for column in identity_columns)
            changes: dict[str, str] = {}
            for field in fields:
                raw = values[field.column]
                if field.cardinality == "scalar":
                    if not isinstance(raw, str):
                        continue
                    replacement = _mapped_reference(raw, forward)
                    if replacement is not None:
                        changes[field.column] = replacement
                    continue
                raw_values = _field_values(field, raw)
                changed = tuple(_mapped_reference(value, forward) or value for value in raw_values)
                if changed != raw_values:
                    changes[field.column] = json.dumps(changed)
            if changes:
                assignments = ", ".join(f"{_quote_identifier(column)} = ?" for column in changes)
                conn.execute(
                    f"UPDATE {_quote_identifier(relation.table)} SET {assignments} WHERE {identity_where}",
                    (*changes.values(), *identity_values),
                )


def _migrate_result_set_manifests(conn: sqlite3.Connection, plan: AssertionTransitionPlan) -> dict[str, str]:
    """Clone changed immutable result manifests and repoint their owners."""
    required = {"result_sets", "result_set_members"}
    present = required & set(_tables(conn))
    if present and present != required:
        raise ObjectRefReconciliationError("result-set manifest schema is incomplete")
    if not present:
        return {}
    forward = dict(plan.forward)
    successors: dict[str, str] = {}
    rows = tuple(
        conn.execute(
            """
            SELECT result_set_id, query_hash, grain, corpus_epoch, member_count,
                   membership_merkle_root, ordered_rank_hash, exactness,
                   persistence_class, created_at_ms, privacy_class,
                   retention_policy_json, excision_link, promoted_at_ms
            FROM result_sets ORDER BY result_set_id
            """
        )
    )
    for row in rows:
        result_set_id = str(row[0])
        member_refs = tuple(
            str(member_row[0])
            for member_row in conn.execute(
                "SELECT member_ref FROM result_set_members WHERE result_set_id = ? ORDER BY rank", (result_set_id,)
            )
        )
        _validate_result_set_manifest(row, member_refs)
        transitioned = tuple(_mapped_reference(member_ref, forward) or member_ref for member_ref in member_refs)
        if transitioned == member_refs:
            continue
        if len(transitioned) != len(set(transitioned)):
            raise ObjectRefReconciliationError("identity transition collapses result-set members")
        successor_id = _result_set_successor_id(result_set_id, member_refs, transitioned)
        _insert_result_set_successor(conn, successor_id, row, transitioned)
        _repoint_result_set_owners(conn, result_set_id, successor_id)
        successors[f"result-set:{result_set_id}"] = f"result-set:{successor_id}"
    return successors


def _validate_result_set_manifest(row: tuple[object, ...], member_refs: tuple[str, ...]) -> None:
    member_count = row[4]
    if isinstance(member_count, bool) or not isinstance(member_count, int):
        raise ObjectRefReconciliationError("result-set manifest member count is invalid")
    if member_count != len(member_refs):
        raise ObjectRefReconciliationError("result-set member count does not match its manifest")
    if str(row[5]) != membership_merkle_root(member_refs):
        raise ObjectRefReconciliationError("result-set membership root does not match its members")
    if str(row[6]) != hash_payload(list(member_refs)):
        raise ObjectRefReconciliationError("result-set rank hash does not match its members")


def _result_set_successor_id(
    result_set_id: str,
    member_refs: tuple[str, ...],
    transitioned: tuple[str, ...],
) -> str:
    return f"{result_set_id}:transition:{hash_payload((result_set_id, member_refs, transitioned))}"


def _insert_result_set_successor(
    conn: sqlite3.Connection,
    successor_id: str,
    row: tuple[object, ...],
    member_refs: tuple[str, ...],
) -> None:
    # The successor id is a hash of the manifest it derives from, so an id
    # that already exists already carries this exact content: a retried
    # application converges instead of raising.
    conn.execute(
        """
        INSERT OR IGNORE INTO result_sets (
            result_set_id, query_hash, grain, corpus_epoch, member_count,
            membership_merkle_root, ordered_rank_hash, exactness,
            persistence_class, created_at_ms, privacy_class,
            retention_policy_json, excision_link, promoted_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            successor_id,
            *row[1:4],
            len(member_refs),
            membership_merkle_root(member_refs),
            hash_payload(list(member_refs)),
            *row[7:],
        ),
    )
    conn.executemany(
        "INSERT OR IGNORE INTO result_set_members (result_set_id, rank, member_ref) VALUES (?, ?, ?)",
        ((successor_id, rank, member_ref) for rank, member_ref in enumerate(member_refs)),
    )


def _repoint_result_set_owners(conn: sqlite3.Connection, old: str, new: str) -> None:
    if (
        _table_exists(conn, "query_excision_ledger")
        and conn.execute("SELECT 1 FROM query_excision_ledger WHERE result_set_id = ?", (old,)).fetchone()
    ):
        raise ObjectRefReconciliationError("cannot replace an excised result-set manifest")
    _refuse_holdout_result_set(conn, old)
    # ``retained_query_runs`` and ``query_evaluation_receipts`` record what was
    # executed against which manifest. That manifest still exists, and their
    # ``result_set_id`` is the exact-retry identity ``put_retained_query_run``
    # compares against, so rewriting it would make a replayed execution look
    # like a conflicting one. Only the forward-looking baseline moves.
    owner_columns = (("watched_query_baselines", "result_set_id"),)
    tables = frozenset(_tables(conn))
    for table, column in owner_columns:
        if table in tables and column in _columns(conn, table):
            conn.execute(
                f"UPDATE {_quote_identifier(table)} SET {_quote_identifier(column)} = ? "
                f"WHERE {_quote_identifier(column)} = ?",
                (new, old),
            )


def _refuse_holdout_result_set(conn: sqlite3.Connection, old: str) -> None:
    """Refuse to give a holdout manifest a successor id.

    Holdout contamination is recorded per ``result_set_id``
    (``holdout_cohorts.has_holdout_contamination``) and is permanent once
    true. Two ids for one cohort would let an undeclared access recorded
    against either id read as clean under the other, so the cohort keeps
    exactly one manifest identity.
    """
    if not _table_exists(conn, "result_set_holdout_policies"):
        return
    policy = conn.execute(
        "SELECT 1 FROM result_set_holdout_policies WHERE result_set_id = ?",
        (old,),
    ).fetchone()
    if policy is not None:
        raise ObjectRefReconciliationError("cannot replace a result set under a holdout policy")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute("SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?", (name,)).fetchone()
        is not None
    )


def _validate_plan_inventory(
    connections: tuple[tuple[sqlite3.Connection, DurableTier], ...],
    plan: AssertionTransitionPlan,
) -> None:
    """Require a plan to classify every durable public reference before mutation."""
    classified = {row.source for row in plan.rows}
    for connection, tier in connections:
        inventory = _inventory_for_connection(connection, tier)
        present = {item.value for item in inventory}
        unclassified = present - classified
        if unclassified:
            raise ObjectRefReconciliationError(
                f"transition plan does not classify durable {tier} references: {sorted(unclassified)!r}"
            )


def _reject_sealed_reference_rewrites(
    connections: tuple[tuple[sqlite3.Connection, DurableTier], ...],
    plan: AssertionTransitionPlan,
) -> None:
    """Keep authenticated receipts and audit history byte-for-byte stable."""
    mapped = {old for old, _new in plan.forward}
    for connection, tier in connections:
        sealed_relations = {
            relation.table for relation in durable_reference_relations(tier) if relation.transition == "sealed"
        }
        for item in _inventory_for_connection(connection, tier):
            relation = item.relation.split(":", 1)[0]
            if relation in sealed_relations and item.value in mapped:
                raise ObjectRefReconciliationError(
                    f"identity migration cannot rewrite sealed durable relation: {tier}.{relation}"
                )


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AssertionTransitionPlan",
    "DurableReference",
    "IdentityMigrationMap",
    "ObjectRefDisposition",
    "ObjectRefReconciliationError",
    "ReconciledObjectRef",
    "SourceIdentityClaims",
    "TransitionBinding",
    "apply_assertion_transition",
    "enumerate_assertion_object_refs",
    "enumerate_durable_reference_inventory",
    "reconcile_object_refs",
]
