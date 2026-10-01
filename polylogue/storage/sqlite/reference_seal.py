"""Off-writer proof that an index mutation preserves resolved durable refs.

The seal is prepared before an index write transaction and is bound to the
actual archive root selected by the writer.  It reads only declared typed
reference fields in ``user.db`` and ``audit.db``; free-form assertion values,
settings, and receipt payloads are deliberately outside this contract.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
import threading
from builtins import BaseExceptionGroup
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from polylogue.storage.index_generation import IndexGeneration

from polylogue.core.compute_cancel import compute_cancel_requested
from polylogue.core.refs import (
    EvidenceRef,
    ObjectRef,
    parse_delegation_ancestry_object_id,
    parse_delegation_edge_object_id,
    parse_delegation_subtree_object_id,
    parse_public_ref,
)
from polylogue.storage.block_anchor import (
    BlockAnchor,
    InvalidBlockAnchorError,
    parse_block_anchor,
    resolve_block_anchor,
)
from polylogue.storage.sqlite.audit_leaf import VerifiedAuditLeaf
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    NativeSQLCustodyOwner,
    _open_readonly_owner,
    open_readonly_connection,
    open_scratch_connection,
)
from polylogue.storage.sqlite.write_lease import current_sql_custody

_LIVE_SEALS_LOCK = threading.RLock()
_LIVE_SEALS: dict[int, PreparedIndexMutation] = {}
_FORK_ABANDONED_SEALS: list[PreparedIndexMutation] = []


def _before_seal_fork() -> None:
    _LIVE_SEALS_LOCK.acquire()


def _after_seal_fork_parent() -> None:
    _LIVE_SEALS_LOCK.release()


def _after_seal_fork_child() -> None:
    global _LIVE_SEALS, _LIVE_SEALS_LOCK
    # Keep the copied handles unreachable for use without running SQLite
    # finalizers during fork. Parent settlement retains its original owners.
    _FORK_ABANDONED_SEALS.extend(_LIVE_SEALS.values())
    _LIVE_SEALS = {}
    _LIVE_SEALS_LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_before_seal_fork,
        after_in_parent=_after_seal_fork_parent,
        after_in_child=_after_seal_fork_child,
    )


def retained_reference_seals_on_current_thread() -> tuple[PreparedIndexMutation, ...]:
    """Keep failed observer cleanup reachable by its original preparation worker."""
    with _LIVE_SEALS_LOCK:
        return tuple(
            seal
            for seal in _LIVE_SEALS.values()
            if seal.index_pid == os.getpid() and seal.index_thread is threading.current_thread()
        )


def _check_reference_cancellation() -> None:
    if compute_cancel_requested():
        raise asyncio.CancelledError("durable-reference proof cancelled by its owner")


class ReferenceSealError(RuntimeError):
    """A durable reference cannot be proven safe across an index mutation."""


class ReferenceSealStaleError(ReferenceSealError):
    """A tier changed after the off-writer reference census was prepared."""


@dataclass(frozen=True, slots=True)
class KnownSourceMutationReceipt:
    """Proof handle for one exact Source mutation committed under this seal."""

    _seal: PreparedIndexMutation
    _source_identity: tuple[int, int, int, int]
    _prior_data_version: int
    _table: str
    _columns: tuple[str, ...]
    _rows: tuple[tuple[object, ...], ...]
    _seal_nonce: object


@dataclass(frozen=True, slots=True)
class KnownSourceMutationPermit:
    """Exact rows an admitted Source writer is authorized to update."""

    _seal: PreparedIndexMutation
    _source_identity: tuple[int, int, int, int]
    _prior_data_version: int
    _table: str
    _columns: tuple[str, ...]
    _rows: tuple[tuple[object, ...], ...]
    _seal_nonce: object

    def require_rows(
        self,
        table: str,
        columns: tuple[str, ...],
        rows: tuple[tuple[object, ...], ...],
    ) -> None:
        if (table, columns, rows) != (self._table, self._columns, self._rows):
            raise ReferenceSealError("Source writer does not match the exact prepared mutation")

    def committed(self) -> KnownSourceMutationReceipt:
        """Mint a receipt only after the exact Source connection context committed."""
        return self._seal._record_known_source_commit(self)


_ACTIVE_MUTATION_SCOPE: ContextVar[IndexMutationScope | None] = ContextVar(
    "polylogue_active_index_mutation_scope", default=None
)


def _current_task() -> object | None:
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


@dataclass(frozen=True, slots=True)
class _ResolvedReference:
    kind: str
    owner_session_id: str
    object_id: str
    qualifier: str | None = None
    scope_session_id: str | None = None
    target_message_id: str | None = None
    wire_ref: str = ""
    has_session_alias: bool = False


def _tier_identity(path: Path) -> tuple[int, int, int, int]:
    stat_result = path.stat()
    return stat_result.st_dev, stat_result.st_ino, stat_result.st_size, stat_result.st_mtime_ns


def _same_incarnation(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    """Compare the file object while allowing its expected contents to change."""
    return left[:2] == right[:2]


def index_path_for_connection(conn: sqlite3.Connection) -> Path:
    """Return the explicit main-index path for an already-open writer."""
    row = next((item for item in conn.execute("PRAGMA database_list") if str(item[1]) == "main"), None)
    if row is None or not str(row[2]):
        raise ReferenceSealError("the index writer has no named main database")
    return Path(str(row[2])).resolve(strict=True)


def _json_strings(conn: sqlite3.Connection, value: object, *, field: str) -> Iterator[str]:
    """Stream one declared JSON string-array field without retaining its rows."""
    try:
        kind = conn.execute("SELECT json_type(?)", (value,)).fetchone()[0]
        if kind != "array":
            raise ReferenceSealError(f"durable {field} must be a JSON string array")
        events = conn.execute("SELECT type, value FROM json_each(?)", (value,))
        for event_type, item in events:
            _check_reference_cancellation()
            if event_type != "text" or not isinstance(item, str):
                raise ReferenceSealError(f"durable {field} must contain only strings")
            yield item
    except sqlite3.Error as exc:
        raise ReferenceSealError(f"durable {field} is not valid JSON") from exc


def _relevant_ref(value: str) -> ObjectRef | EvidenceRef | BlockAnchor | None:
    try:
        return parse_block_anchor(value)
    except InvalidBlockAnchorError:
        pass
    try:
        parsed = parse_public_ref(value)
    except ValueError as exc:
        raise ReferenceSealError(f"durable reference is malformed: {value!r}") from exc
    if isinstance(parsed, EvidenceRef):
        return parsed
    if parsed.kind in {
        "session",
        "message",
        "block",
        "action",
        "delegation",
        "run",
        "observed-event",
        "context-snapshot",
    }:
        return parsed
    return None


def _references_from_user(conn: sqlite3.Connection) -> Iterable[str]:
    for row in conn.execute("SELECT scope_ref, target_ref, author_ref, evidence_refs_json FROM assertions"):
        for column in ("scope_ref", "target_ref", "author_ref"):
            value = row[column]
            if value is not None:
                yield str(value)
        yield from _json_strings(conn, row["evidence_refs_json"], field="assertions.evidence_refs_json")

    for row in conn.execute(
        "SELECT target_ref, source_result_ref, actor_ref, model_ref, prompt_ref, assertion_refs_json "
        "FROM annotation_batches"
    ):
        for column in ("target_ref", "source_result_ref", "actor_ref", "model_ref", "prompt_ref"):
            yield str(row[column])
        yield from _json_strings(conn, row["assertion_refs_json"], field="annotation_batches.assertion_refs_json")

    for (model_refs,) in conn.execute("SELECT model_refs_json FROM query_evaluation_receipts"):
        yield from _json_strings(conn, model_refs, field="query_evaluation_receipts.model_refs_json")

    for (session_id,) in conn.execute("SELECT session_id FROM session_marker_delivery"):
        yield ObjectRef("session", str(session_id)).format()

    for (value,) in conn.execute("SELECT member_ref FROM result_set_members"):
        yield str(value)

    for row in conn.execute(
        "SELECT snapshot_ref, recipient_ref, run_ref, segment_refs_json, evidence_refs_json, assertion_refs_json, "
        "delivered_by_ref FROM context_deliveries"
    ):
        for column in ("recipient_ref", "run_ref", "delivered_by_ref"):
            value = row[column]
            if value is not None:
                yield str(value)
        # segment_refs_json holds local image segment identifiers, not public
        # archive references. Their typed references live in the image model.
        for column in ("evidence_refs_json", "assertion_refs_json"):
            yield from _json_strings(conn, row[column], field=f"context_deliveries.{column}")
        try:
            from polylogue.storage.sqlite.archive_tiers.context_delivery_write import read_context_delivery

            delivery = read_context_delivery(conn, str(row["snapshot_ref"]))
            if delivery is None:
                raise ReferenceSealError("context delivery disappeared during its read snapshot")
        except Exception as exc:
            if isinstance(exc, ReferenceSealError):
                raise
            raise ReferenceSealError("durable context image cannot be parsed by its declared model") from exc
        yield from (ref.format() for ref in delivery.context_image.object_refs)
        yield from (ref.format() for ref in delivery.context_image.evidence_refs)
        yield from delivery.context_image.assertion_refs
        for segment in delivery.context_image.segments:
            yield from (ref.format() for ref in segment.object_refs)
            yield from (ref.format() for ref in segment.evidence_refs)
            yield from segment.assertion_refs


def _references_from_audit(conn: sqlite3.Connection) -> Iterable[str]:
    for table in ("operation_preview_targets", "operation_targets"):
        for (value,) in conn.execute(f"SELECT target_ref FROM {table}"):
            _check_reference_cancellation()
            yield str(value)


def _resolve_target(conn: sqlite3.Connection, ref: ObjectRef | EvidenceRef | BlockAnchor) -> _ResolvedReference | None:
    if isinstance(ref, BlockAnchor):
        resolution = resolve_block_anchor(conn, ref)
        if resolution.state not in {"ok", "drifted_position", "drifted_message", "relocated_lineage"}:
            return None
        return _ResolvedReference(
            "block-anchor",
            ref.session_id,
            ref.content_hash_hex,
            scope_session_id=ref.session_id,
            target_message_id=resolution.resolved_message_id,
        )
    if isinstance(ref, EvidenceRef):
        from polylogue.storage.sqlite.session_identity import resolve_session_id_in_index

        try:
            scope_session_id = resolve_session_id_in_index(conn, ref.session_id)
        except (KeyError, ValueError):
            return None
        if ref.message_id is None:
            return _ResolvedReference("session", scope_session_id, scope_session_id)
        if ref.block_index is None:
            row = conn.execute("SELECT session_id FROM messages WHERE message_id = ?", (ref.message_id,)).fetchone()
            if row is None or _locate_composed_message(conn, scope_session_id, ref.message_id) is None:
                return None
            return _ResolvedReference(
                "message",
                str(row[0]),
                ref.message_id,
                scope_session_id=scope_session_id,
                target_message_id=ref.message_id,
            )
        row = conn.execute(
            "SELECT m.session_id, b.message_id, b.position FROM messages AS m "
            "JOIN blocks AS b ON b.message_id = m.message_id "
            "WHERE m.message_id = ? AND b.position = ?",
            (ref.message_id, ref.block_index),
        ).fetchone()
        if row is None or _locate_composed_message(conn, scope_session_id, ref.message_id) is None:
            return None
        return _ResolvedReference(
            "block-position", str(row[0]), str(row[1]), str(row[2]), scope_session_id, str(row[1])
        )

    if ref.kind == "delegation":
        root = parse_delegation_ancestry_object_id(ref.object_id) or parse_delegation_subtree_object_id(ref.object_id)
        if root is not None:
            row = conn.execute("SELECT session_id FROM sessions WHERE session_id = ?", (root,)).fetchone()
            return None if row is None else _ResolvedReference("delegation-root", root, ref.object_id)
        edge = parse_delegation_edge_object_id(ref.object_id)
        if edge is None:
            row = conn.execute(
                "SELECT parent_session_id, child_session_id, instruction_message_id FROM delegation_facts "
                "WHERE instruction_tool_use_block_id = ? LIMIT 1",
                (ref.object_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT parent_session_id, child_session_id, instruction_message_id FROM delegation_facts "
                "WHERE parent_session_id = ? AND child_session_id = ? "
                "AND mapping_state IN ('edge_only', 'quarantined', 'authority-contradicted') LIMIT 1",
                edge,
            ).fetchone()
        if row is None:
            return None
        return _ResolvedReference(
            "delegation",
            str(row[0]),
            ref.object_id,
            scope_session_id=str(row[1]) if row[1] is not None else None,
            target_message_id=str(row[2]) if row[2] is not None else None,
        )
    if ref.kind in {"run", "observed-event", "context-snapshot"}:
        from polylogue.storage.sqlite.run_projection_relations import (
            context_snapshot_relation_sql,
            observed_event_relation_sql,
            run_relation_sql,
        )

        if ref.kind == "run":
            relation, table, column = run_relation_sql(), "runs", "run_ref"
        elif ref.kind == "observed-event":
            relation, table, column = observed_event_relation_sql(source_where="1=1"), "observed_events", "event_ref"
        else:
            relation, table, column = context_snapshot_relation_sql(), "context_snapshots", "snapshot_ref"
        row = conn.execute(f"{relation} SELECT session_id FROM {table} WHERE {column} = ?", (ref.format(),)).fetchone()
        return None if row is None else _ResolvedReference(ref.kind, str(row[0]), ref.object_id)
    if ref.kind == "session":
        from polylogue.storage.sqlite.session_identity import resolve_session_id_in_index

        try:
            session_id = resolve_session_id_in_index(conn, ref.object_id)
        except (KeyError, ValueError):
            return None
        return _ResolvedReference("session", session_id, session_id)
    if ref.kind == "message":
        row = conn.execute("SELECT session_id FROM messages WHERE message_id = ?", (ref.object_id,)).fetchone()
        return (
            None
            if row is None
            else _ResolvedReference("message", str(row[0]), ref.object_id, target_message_id=ref.object_id)
        )
    if ref.kind in {"block", "action"}:
        if ref.qualifiers:
            row = conn.execute(
                "SELECT m.session_id, b.message_id, b.position, b.block_id FROM messages AS m "
                "JOIN blocks AS b ON b.message_id = m.message_id "
                "WHERE b.block_id = ? OR (m.message_id = ? AND b.position = ?) LIMIT 1",
                (ref.object_id, ref.object_id, ref.qualifiers[-1]),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT m.session_id, b.message_id, b.position, b.block_id FROM messages AS m "
                "JOIN blocks AS b ON b.message_id = m.message_id "
                "WHERE b.block_id = ?",
                (ref.object_id,),
            ).fetchone()
        if row is None:
            return None
        if str(row[3]) == ref.object_id:
            return _ResolvedReference("block-id", str(row[0]), str(row[3]), target_message_id=str(row[1]))
        return _ResolvedReference(
            "block-position", str(row[0]), str(row[1]), str(row[2]), target_message_id=str(row[1])
        )
    return None


def _resolve(conn: sqlite3.Connection, ref: ObjectRef | EvidenceRef | BlockAnchor) -> _ResolvedReference | None:
    target = _resolve_target(conn, ref)
    if target is None:
        return None
    lookup_session = (
        ref.session_id
        if isinstance(ref, EvidenceRef)
        else ref.object_id
        if isinstance(ref, ObjectRef) and ref.kind == "session"
        else None
    )
    canonical_session = target.scope_session_id or target.owner_session_id
    wire = ref.to_text() if isinstance(ref, BlockAnchor) else ref.format()
    return replace(
        target, wire_ref=wire, has_session_alias=lookup_session is not None and lookup_session != canonical_session
    )


def _still_resolves(conn: sqlite3.Connection, ref: _ResolvedReference) -> bool:
    # Re-run the original typed lookup. Surviving canonical rows do not prove
    # that an unqualified alias still identifies that same row.
    parsed = _relevant_ref(ref.wire_ref)
    if parsed is None:
        return False
    actual = _resolve_target(conn, parsed)
    if ref.kind == "block-anchor" and actual is not None:
        # A content anchor explicitly permits unique relocation. Its declared
        # target is the hash within the original composed scope; message IDs
        # only select affected witnesses, not the anchor's identity.
        actual = replace(actual, target_message_id=None)
        ref = replace(ref, target_message_id=None)
    return actual is not None and actual == replace(ref, wire_ref="", has_session_alias=False)


def _locate_composed_message(conn: sqlite3.Connection, session_id: str, message_id: str) -> int | None:
    # The canonical composition locator lives with the session write/read
    # projection.  Resolve lazily to keep the seal independent during module
    # initialization; calls happen only after the archive substrate is loaded.
    from polylogue.storage.sqlite.archive_tiers.write import locate_composed_message

    return locate_composed_message(conn, session_id, message_id)


class PreparedIndexMutation:
    """Live observer set and typed reachability captured before writer admission."""

    def __init__(self, index_path: Path, *, archive_root: Path) -> None:
        self.archive_root = archive_root.resolve(strict=True)
        self.index_path = index_path.resolve(strict=True)
        from polylogue.storage.archive_identity import resolve_active_index_path

        if resolve_active_index_path(self.archive_root).resolve(strict=True) != self.index_path:
            raise ReferenceSealError("active reference seal requires the archive's actual active Index")
        self.index_thread = threading.current_thread()
        self.index_pid = os.getpid()
        self.index_task = _current_task()
        self.index_identity = _tier_identity(self.index_path)
        self._paths = {
            "index": self.index_path,
            "source": self.archive_root / "source.db",
            "user": self.archive_root / "user.db",
            "audit": self.archive_root / "audit.db",
        }
        self._identities = {name: _tier_identity(path) for name, path in self._paths.items()}
        self._observers: dict[str, sqlite3.Connection] = {}
        self._observer_leaves: dict[str, VerifiedAuditLeaf] = {}
        self._versions: dict[str, int] = {}
        self._candidate_path: Path | None = None
        self._candidate_identity: tuple[int, int, int, int] | None = None
        self._candidate_version: int | None = None
        self._candidate_schema: tuple[int, str | None] | None = None
        self.candidate_missing_session_count = 0
        self.candidate_first_missing_session_id: str | None = None
        self._source_mutation_nonce = object()
        self._pending_source_permit: KnownSourceMutationPermit | None = None
        self._pending_source_receipt: KnownSourceMutationReceipt | None = None
        self._closed = False
        self._session_namespace_noted = False
        self._scratch_directory: tempfile.TemporaryDirectory[str] | None = None
        self._owned_scratch_connection: sqlite3.Connection | None = None
        with _LIVE_SEALS_LOCK:
            _LIVE_SEALS[id(self)] = self
        try:
            self._scratch_directory = tempfile.TemporaryDirectory(prefix="polylogue-reference-seal-")
            _check_reference_cancellation()
            scratch_owner = open_scratch_connection(
                Path(self._scratch_directory.name) / "refs.db", terminal_parent=self
            )
            assert scratch_owner.connection is not None
            self._owned_scratch_connection = scratch_owner.connection
            self._scratch.set_progress_handler(lambda: int(compute_cancel_requested()), 2000)
            self._scratch.executescript(
                "CREATE TEMP TABLE resolved_refs ("
                "kind TEXT NOT NULL, owner_session_id TEXT NOT NULL, object_id TEXT NOT NULL, "
                "qualifier TEXT NOT NULL, scope_session_id TEXT NOT NULL, target_message_id TEXT NOT NULL, "
                "wire_ref TEXT PRIMARY KEY, has_session_alias INTEGER NOT NULL) "
                "WITHOUT ROWID;"
                "CREATE INDEX temp.resolved_refs_by_session ON resolved_refs(owner_session_id, kind, object_id, qualifier);"
                "CREATE INDEX temp.resolved_refs_by_scope_session ON resolved_refs(scope_session_id, kind, object_id, qualifier);"
                "CREATE INDEX temp.resolved_refs_by_target_message ON resolved_refs(target_message_id);"
                "CREATE INDEX temp.resolved_refs_aliases ON resolved_refs(has_session_alias) WHERE has_session_alias = 1;"
                "CREATE TEMP TABLE destructive_message_ids(message_id TEXT PRIMARY KEY) WITHOUT ROWID;"
                "CREATE TEMP TABLE candidate_refs ("
                "kind TEXT NOT NULL, owner_session_id TEXT NOT NULL, object_id TEXT NOT NULL, "
                "qualifier TEXT NOT NULL, scope_session_id TEXT NOT NULL, target_message_id TEXT NOT NULL, "
                "wire_ref TEXT PRIMARY KEY, has_session_alias INTEGER NOT NULL) "
                "WITHOUT ROWID;"
            )
            for name, path in self._paths.items():
                self._observers[name] = self._open_observer(name, path)
                if self._observer_identity(name) != self._identities[name]:
                    raise ReferenceSealStaleError(f"the {name}.db file changed while opening its observer")
            self._read_resolved_references()
        except BaseException as exc:
            with suppress(BaseException):
                self.close()
            if compute_cancel_requested():
                raise asyncio.CancelledError("durable-reference preparation cancelled") from exc
            raise

    @property
    def _scratch(self) -> sqlite3.Connection:
        connection = self._owned_scratch_connection
        if connection is None:
            raise ReferenceSealError("reference proof has no live scratch connection")
        return connection

    def _open_observer(self, name: str, path: Path) -> sqlite3.Connection:
        leaf = VerifiedAuditLeaf(path.parent, filename=path.name, identity_access="lock-preserving")
        leaf.__enter__()
        self._observer_leaves[name] = leaf
        conn = open_readonly_connection(leaf.anchored_path, validate_schema=False)
        self._observers[name] = conn
        from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner

        NativeSQLCustodyOwner(conn, terminal_parent=self)
        try:
            leaf.assert_unchanged()
            conn.row_factory = sqlite3.Row
            conn.set_progress_handler(lambda: int(compute_cancel_requested()), 2000)
            return conn
        except BaseException:
            try:
                self._close_native_connection(conn)
            except BaseException:
                # Retain both actual handles for owner-thread recovery.
                raise
            else:
                self._observers.pop(name, None)
            raise

    def _close_native_connection(self, connection: sqlite3.Connection) -> None:
        from polylogue.storage.sqlite.connection_profile import close_parent_native_connection

        close_parent_native_connection(self, connection)

    def _observer_identity(self, name: str) -> tuple[int, int, int, int]:
        metadata = self._observer_leaves[name].identity_metadata()
        return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns

    @staticmethod
    def _writer_identity(conn: sqlite3.Connection) -> tuple[int, int, int, int]:
        row = conn.execute("PRAGMA database_list").fetchall()
        path = next((Path(str(item[2])) for item in row if str(item[1]) == "main"), None)
        if path is None:
            raise ReferenceSealError("the index writer has no main database")
        return _tier_identity(path)

    def _read_resolved_references(self) -> None:
        index_observer = self._observers["index"]
        before_index = int(index_observer.execute("PRAGMA data_version").fetchone()[0])
        index_observer.execute("BEGIN")
        try:
            for name in ("source", "user", "audit"):
                observer = self._observers[name]
                before = int(observer.execute("PRAGMA data_version").fetchone()[0])
                observer.execute("BEGIN")
                try:
                    if name == "user":
                        refs = _references_from_user(observer)
                    elif name == "audit":
                        refs = _references_from_audit(observer)
                    else:
                        observer.execute("SELECT 1 FROM sqlite_schema LIMIT 1").fetchone()
                        refs = ()
                    for raw in refs:
                        _check_reference_cancellation()
                        parsed = _relevant_ref(raw)
                        if parsed is not None:
                            target = _resolve(index_observer, parsed)
                            if target is not None:
                                self._scratch.execute(
                                    "INSERT OR IGNORE INTO resolved_refs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                    (
                                        target.kind,
                                        target.owner_session_id,
                                        target.object_id,
                                        target.qualifier or "",
                                        target.scope_session_id or "",
                                        target.target_message_id or "",
                                        target.wire_ref,
                                        int(target.has_session_alias),
                                    ),
                                )
                except BaseException:
                    observer.rollback()
                    raise
                else:
                    observer.commit()
                after = int(observer.execute("PRAGMA data_version").fetchone()[0])
                if before != after:
                    raise ReferenceSealStaleError(f"{name}.db changed during reference preparation")
                self._versions[name] = after
        except BaseException:
            index_observer.rollback()
            raise
        else:
            index_observer.commit()
        after_index = int(index_observer.execute("PRAGMA data_version").fetchone()[0])
        if before_index != after_index:
            raise ReferenceSealStaleError("index.db changed during reference preparation")
        self._versions["index"] = after_index
        self._scratch.commit()

    def note_session_namespace_change(self) -> None:
        self._require_new_work()
        if not self._session_namespace_noted:
            self._scratch.execute(
                "INSERT OR IGNORE INTO candidate_refs SELECT * FROM resolved_refs WHERE has_session_alias = 1"
            )
            self._session_namespace_noted = True

    def note_deleted_session(self, session_id: str) -> None:
        self._require_new_work()
        self.note_session_namespace_change()
        self._scratch.execute(
            "INSERT OR IGNORE INTO candidate_refs "
            "SELECT kind, owner_session_id, object_id, qualifier, scope_session_id, target_message_id, wire_ref, has_session_alias FROM resolved_refs "
            "WHERE owner_session_id = ? OR scope_session_id = ?",
            (session_id, session_id),
        )

    def note_lineage_change(self, conn: sqlite3.Connection, session_id: str) -> None:
        """Track refs scoped to every composed transcript below a changed node."""
        self._require_new_work()
        self.note_session_namespace_change()
        from polylogue.archive.topology.edge import topology_status_composes_sql

        status_predicate = topology_status_composes_sql("l.status")
        descendants = conn.execute(
            f"""
            WITH RECURSIVE affected(session_id) AS (
                SELECT ?
                UNION
                SELECT l.src_session_id
                FROM session_links AS l
                JOIN affected AS a ON l.resolved_dst_session_id = a.session_id
                WHERE l.inheritance = 'prefix-sharing' AND {status_predicate}
            )
            SELECT session_id FROM affected
            """,
            (session_id,),
        )
        for (affected_session_id,) in descendants:
            _check_reference_cancellation()
            self._scratch.execute(
                "INSERT OR IGNORE INTO candidate_refs "
                "SELECT kind, owner_session_id, object_id, qualifier, scope_session_id, target_message_id, wire_ref, has_session_alias "
                "FROM resolved_refs WHERE scope_session_id = ? OR (owner_session_id = ? AND "
                "kind IN ('delegation', 'delegation-root', 'run', 'observed-event', 'context-snapshot'))",
                (str(affected_session_id), str(affected_session_id)),
            )
        self._scratch.commit()

    def note_deleted_message_ids(self, message_ids: Iterable[str]) -> None:
        self._require_new_work()
        self._scratch.executemany(
            "INSERT OR IGNORE INTO destructive_message_ids VALUES (?)", ((message_id,) for message_id in message_ids)
        )
        self._scratch.execute(
            "INSERT OR IGNORE INTO candidate_refs "
            "SELECT r.kind, r.owner_session_id, r.object_id, r.qualifier, r.scope_session_id, r.target_message_id, r.wire_ref, r.has_session_alias "
            "FROM resolved_refs AS r JOIN destructive_message_ids AS d "
            "ON d.message_id = r.target_message_id"
        )
        self._scratch.execute("DELETE FROM destructive_message_ids")
        self._scratch.commit()

    @contextmanager
    def mutation_scope(self, conn: sqlite3.Connection) -> Iterator[IndexMutationScope]:
        """Own the outer index transaction while retaining this exact seal."""
        if conn.in_transaction:
            raise ReferenceSealError("an index mutation scope must start before BEGIN")
        from polylogue.storage.sqlite.write_lease import require_write_lease

        require_write_lease("prepared index mutation", archive_root=self.archive_root)
        self.validate_for_writer(conn)
        with _owned_index_transaction(IndexMutationScope(self, conn)) as scope:
            yield scope

    def validate_for_writer(self, conn: sqlite3.Connection) -> None:
        self._require_new_work()
        if conn.in_transaction:
            raise ReferenceSealStaleError("reference seal validation must precede the writer transaction")
        if self._writer_identity(conn) != self.index_identity:
            raise ReferenceSealStaleError("the index file incarnation changed after reference preparation")
        for name, _path in self._paths.items():
            if self._observer_identity(name) != self._identities[name]:
                raise ReferenceSealStaleError(f"the {name}.db file incarnation changed after reference preparation")
            observer = self._observers[name]
            current = int(observer.execute("PRAGMA data_version").fetchone()[0])
            if current != self._versions[name]:
                raise ReferenceSealStaleError(f"{name}.db changed after reference preparation")

    def observer(self, tier: str) -> sqlite3.Connection:
        """Return one retained readonly observer to its seal-owning worker."""
        self._require_new_work()
        try:
            return self._observers[tier]
        except KeyError as exc:
            raise ValueError(f"unknown reference-seal tier {tier!r}") from exc

    def observer_version(self, tier: str) -> int:
        self._require_new_work()
        try:
            return self._versions[tier]
        except KeyError as exc:
            raise ValueError(f"unknown reference-seal tier {tier!r}") from exc

    def validate_observers_current(self) -> None:
        """Point-check every retained observer immediately after gate admission."""
        self._require_new_work()
        for name, _path in self._paths.items():
            if self._observer_identity(name) != self._identities[name]:
                raise ReferenceSealStaleError(f"the {name}.db file incarnation changed after preparation")
            current = int(self._observers[name].execute("PRAGMA data_version").fetchone()[0])
            if current != self._versions[name]:
                raise ReferenceSealStaleError(f"{name}.db changed after preparation")

    @staticmethod
    def _candidate_schema_identity(observer: sqlite3.Connection) -> tuple[int, str | None]:
        user_version = int(observer.execute("PRAGMA user_version").fetchone()[0])
        has_identity = observer.execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'schema_identity'"
        ).fetchone()
        identity = None
        if has_identity is not None:
            row = observer.execute("SELECT identity FROM schema_identity WHERE tier = 'index'").fetchone()
            identity = str(row[0]) if row is not None else None
        return user_version, identity

    def prepare_candidate_reachability(self, candidate_index_path: Path) -> tuple[int, int, int, int]:
        """Prove promotion reachability off-writer and retain its observer.

        The same candidate connection remains open through writer admission.
        The full reference and active-coverage scans happen here; callers must
        only perform point currency checks after acquiring archive custody.
        """
        self._require_new_work()
        if self._candidate_path is not None:
            raise ReferenceSealError("this reference seal already has a promotion candidate")
        candidate = Path(candidate_index_path).resolve(strict=True)
        identity_before = _tier_identity(candidate)
        observer = self._open_observer("candidate", candidate)
        first: _ResolvedReference | None = None
        lost_count = 0
        version_before = version_after = -1
        primary: BaseException | None = None
        schema: tuple[int, str | None] = (0, None)
        missing_count = 0
        first_missing: str | None = None
        try:
            version_before = int(observer.execute("PRAGMA data_version").fetchone()[0])
            schema = self._candidate_schema_identity(observer)
            observer.execute("BEGIN")
            reference_rows = self._scratch.execute(
                "SELECT kind, owner_session_id, object_id, qualifier, scope_session_id, target_message_id, wire_ref, has_session_alias "
                "FROM resolved_refs ORDER BY kind, object_id, qualifier"
            )
            for row in reference_rows:
                _check_reference_cancellation()
                ref = _ResolvedReference(
                    str(row[0]),
                    str(row[1]),
                    str(row[2]),
                    str(row[3]) or None,
                    str(row[4]) or None,
                    str(row[5]) or None,
                    str(row[6]),
                    bool(row[7]),
                )
                if not _still_resolves(observer, ref):
                    lost_count += 1
                    if first is None:
                        first = ref

            # Promotion also must not drop a session that the active index
            # still serves from retained raw evidence. Keep this complete scan
            # beside the typed-reference proof so neither is repeated under
            # the lifecycle lock or physical writer lease.
            active = self._observers["index"]
            source = self._observers["source"]
            active_before = int(active.execute("PRAGMA data_version").fetchone()[0])
            source_before = int(source.execute("PRAGMA data_version").fetchone()[0])
            if active_before != self._versions["index"] or source_before != self._versions["source"]:
                raise ReferenceSealStaleError("archive changed before promotion coverage validation")
            active.execute("BEGIN")
            source.execute("BEGIN")
            missing_count = 0
            first_missing = None
            after = ""
            page_size = min(
                512,
                int(source.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)),
                int(observer.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)),
            )
            if page_size < 1:
                raise ReferenceSealError("SQLite variable limit cannot compare promotion coverage")
            while True:
                _check_reference_cancellation()
                rows = active.execute(
                    "SELECT session_id, raw_id FROM sessions "
                    "WHERE session_id > ? AND raw_id IS NOT NULL ORDER BY session_id LIMIT ?",
                    (after, page_size),
                ).fetchall()
                if not rows:
                    break
                after = str(rows[-1][0])
                raw_ids = tuple(dict.fromkeys(str(row[1]) for row in rows))
                retained = {
                    str(row[0])
                    for row in source.execute(
                        f"SELECT raw_id FROM raw_sessions WHERE raw_id IN ({','.join('?' for _ in raw_ids)})",
                        raw_ids,
                    )
                }
                owed = tuple(str(row[0]) for row in rows if str(row[1]) in retained)
                if not owed:
                    continue
                present = {
                    str(row[0])
                    for row in observer.execute(
                        f"SELECT session_id FROM sessions WHERE session_id IN ({','.join('?' for _ in owed)})",
                        owed,
                    )
                }
                for session_id in owed:
                    _check_reference_cancellation()
                    if session_id not in present:
                        missing_count += 1
                        if first_missing is None:
                            first_missing = session_id
            source.rollback()
            active.rollback()
            active_after = int(active.execute("PRAGMA data_version").fetchone()[0])
            source_after = int(source.execute("PRAGMA data_version").fetchone()[0])
            if active_after != active_before or source_after != source_before:
                raise ReferenceSealStaleError("archive changed during promotion coverage validation")
            observer.rollback()
            version_after = int(observer.execute("PRAGMA data_version").fetchone()[0])
        except BaseException as exc:
            primary = exc
            with suppress(BaseException):
                if observer.in_transaction:
                    observer.rollback()
            with suppress(BaseException):
                if self._observers.get("source") is not None and self._observers["source"].in_transaction:
                    self._observers["source"].rollback()
            with suppress(BaseException):
                if self._observers.get("index") is not None and self._observers["index"].in_transaction:
                    self._observers["index"].rollback()
        if primary is not None:
            with suppress(BaseException):
                observer.close()
            raise primary
        identity_after = _tier_identity(candidate)
        if identity_after != identity_before or version_after != version_before:
            observer.close()
            raise ReferenceSealStaleError("promotion candidate changed during durable-reference validation")
        if lost_count and first is not None:
            observer.close()
            raise ReferenceSealError(
                f"index promotion would orphan {lost_count} resolved durable reference(s); "
                f"first lost {first.kind} reference in session {first.owner_session_id!r}"
            )
        self._candidate_path = candidate
        self._candidate_identity = identity_after
        self._candidate_version = version_after
        self._candidate_schema = schema
        self.candidate_missing_session_count = missing_count
        self.candidate_first_missing_session_id = first_missing
        self._observers["candidate"] = observer
        return identity_after

    def validate_candidate_current(self, candidate_index_path: Path) -> tuple[int, int, int, int]:
        """Point-check the retained candidate proof after writer admission."""
        self._require_new_work()
        candidate = Path(candidate_index_path).resolve(strict=True)
        if candidate != self._candidate_path or self._candidate_identity is None:
            raise ReferenceSealStaleError("promotion candidate differs from its prepared proof")
        observer = self._observers["candidate"]
        identity_before = _tier_identity(candidate)
        version_before = int(observer.execute("PRAGMA data_version").fetchone()[0])
        schema = self._candidate_schema_identity(observer)
        version_after = int(observer.execute("PRAGMA data_version").fetchone()[0])
        identity_after = _tier_identity(candidate)
        if identity_before != identity_after or identity_after != self._candidate_identity:
            raise ReferenceSealStaleError("promotion candidate incarnation changed after reference preparation")
        if version_before != version_after or version_after != self._candidate_version:
            raise ReferenceSealStaleError("promotion candidate changed after reference preparation")
        if schema != self._candidate_schema:
            raise ReferenceSealStaleError("promotion candidate schema changed after reference preparation")
        return self._candidate_identity

    def prepare_known_source_mutation(
        self,
        table: str,
        columns: tuple[str, ...],
        rows: tuple[tuple[object, ...], ...],
    ) -> KnownSourceMutationPermit:
        """Bind one exact prepared Source write to this observer baseline."""
        self._require_new_work()
        if not table.isidentifier() or not columns or any(not column.isidentifier() for column in columns):
            raise ReferenceSealError("known Source mutation must name declared SQL identifiers")
        if any(len(row) != len(columns) + 1 for row in rows):
            raise ReferenceSealError("known Source mutation rows must carry declared values and raw_id")
        source_identity = _tier_identity(self._paths["source"])
        if source_identity != self._identities["source"]:
            raise ReferenceSealStaleError("source.db incarnation changed before prepared source publication")
        if self._pending_source_permit is not None:
            raise ReferenceSealError("this seal already has a pending known Source mutation")
        permit = KnownSourceMutationPermit(
            self,
            source_identity,
            self._versions["source"],
            table,
            columns,
            rows,
            self._source_mutation_nonce,
        )
        self._pending_source_permit = permit
        return permit

    def _record_known_source_commit(self, permit: KnownSourceMutationPermit) -> KnownSourceMutationReceipt:
        self._require_live_owner()
        if permit is not self._pending_source_permit or permit._seal_nonce is not self._source_mutation_nonce:
            raise ReferenceSealError("Source writer used a permit outside its prepared seal")
        if self._pending_source_receipt is not None:
            raise ReferenceSealError("known Source mutation permit was already committed")
        receipt = KnownSourceMutationReceipt(
            self,
            permit._source_identity,
            permit._prior_data_version,
            permit._table,
            permit._columns,
            permit._rows,
            permit._seal_nonce,
        )
        self._pending_source_receipt = receipt
        return receipt

    def accept_known_source_commit(self, receipt: KnownSourceMutationReceipt) -> None:
        """Settle an already committed Source receipt even after cancellation."""
        self._require_live_owner()
        observer = self._observers["source"]
        observer.set_progress_handler(None, 0)
        try:
            self._accept_known_source_commit(receipt)
        finally:
            observer.set_progress_handler(lambda: int(compute_cancel_requested()), 2000)

    def _accept_known_source_commit(self, receipt: KnownSourceMutationReceipt) -> None:
        if (
            receipt._seal is not self
            or receipt is not self._pending_source_receipt
            or receipt._seal_nonce is not self._source_mutation_nonce
            or receipt._source_identity != self._identities["source"]
            or receipt._prior_data_version != self._versions["source"]
        ):
            raise ReferenceSealError("Source commit receipt does not belong to this prepared seal")
        identity_before = _tier_identity(self._paths["source"])
        if not _same_incarnation(identity_before, receipt._source_identity):
            raise ReferenceSealStaleError("source.db incarnation changed during the prepared source publication")
        observer = self._observers["source"]
        if observer.in_transaction:
            raise ReferenceSealError("cannot advance Source authority while its observer has a read transaction")
        if not receipt._rows:
            raise ReferenceSealError("an empty Source mutation cannot advance the observer baseline")
        version_before = int(observer.execute("PRAGMA data_version").fetchone()[0])
        if version_before == receipt._prior_data_version:
            raise ReferenceSealStaleError("the committed Source mutation did not advance the retained observer")
        columns_sql = ", ".join(receipt._columns)
        for expected in receipt._rows:
            expected_values, raw_id = expected[:-1], expected[-1]
            actual = observer.execute(
                f"SELECT {columns_sql} FROM {receipt._table} WHERE raw_id = ?", (raw_id,)
            ).fetchone()
            if actual is None or tuple(actual) != expected_values:
                raise ReferenceSealStaleError("committed Source rows differ from the exact prepared mutation")
        identity_after = _tier_identity(self._paths["source"])
        version_after = int(observer.execute("PRAGMA data_version").fetchone()[0])
        if not _same_incarnation(identity_before, identity_after) or version_after != version_before:
            raise ReferenceSealStaleError("source.db changed while the exact committed rows were being verified")
        # This is the sole baseline advance: a one-shot receipt from the
        # declared update path, with exact rows reread through this observer
        # between stable incarnation and data_version checks.
        self._identities["source"] = identity_after
        self._versions["source"] = version_after
        self._pending_source_permit = None
        self._pending_source_receipt = None

    def validate_reachability(self, conn: sqlite3.Connection) -> None:
        self._require_new_work()
        if not _same_incarnation(self._writer_identity(conn), self.index_identity):
            raise ReferenceSealStaleError("reference seal settled on a different index incarnation")
        rows = self._scratch.execute(
            "SELECT kind, owner_session_id, object_id, qualifier, scope_session_id, target_message_id, wire_ref, has_session_alias "
            "FROM candidate_refs "
            "ORDER BY kind, object_id, qualifier"
        )
        lost_count = 0
        first: _ResolvedReference | None = None
        for row in rows:
            _check_reference_cancellation()
            ref = _ResolvedReference(
                str(row[0]),
                str(row[1]),
                str(row[2]),
                str(row[3]) or None,
                str(row[4]) or None,
                str(row[5]) or None,
                str(row[6]),
                bool(row[7]),
            )
            if not _still_resolves(conn, ref):
                lost_count += 1
                if first is None:
                    first = ref
        if lost_count and first is not None:
            raise ReferenceSealError(
                f"index mutation would orphan {lost_count} resolved durable reference(s); "
                f"first lost {first.kind} reference in session {first.owner_session_id!r}"
            )

    def _require_new_work(self) -> None:
        self._require_live_owner()
        _check_reference_cancellation()

    def _require_live_owner(self) -> None:
        if self._closed:
            raise ReferenceSealError("reference seal is closed")
        if (
            os.getpid() != self.index_pid
            or threading.current_thread() is not self.index_thread
            or _current_task() is not self.index_task
        ):
            raise ReferenceSealError("reference seal must be used by its observing index owner")

    def close(self) -> None:
        if self._closed:
            return
        if (
            self.index_pid != os.getpid()
            or self.index_thread is not threading.current_thread()
            or self.index_task is not _current_task()
        ):
            raise ReferenceSealError("reference-seal cleanup must run in its preparing execution unit")
        first_error: BaseException | None = None

        def settle(action: Callable[[], object]) -> bool:
            nonlocal first_error
            try:
                action()
                return True
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                return False

        for name, observer in tuple(self._observers.items()):
            if settle(partial(self._close_native_connection, observer)):
                self._observers.pop(name, None)
        for name, leaf in tuple(self._observer_leaves.items()):
            if name not in self._observers and settle(leaf.close):
                self._observer_leaves.pop(name, None)
        if self._owned_scratch_connection is not None:
            scratch = self._owned_scratch_connection
            if settle(lambda: self._close_native_connection(scratch)):
                self._owned_scratch_connection = None
        if self._owned_scratch_connection is None and self._scratch_directory is not None:
            directory = self._scratch_directory
            if settle(directory.cleanup):
                self._scratch_directory = None
        from polylogue.storage.sqlite.connection_profile import native_sql_children, retire_native_sql_parent

        for owner in native_sql_children(self):
            if not owner._settled:
                settle(owner.close)
        self._closed = (
            not self._observers
            and not self._observer_leaves
            and self._owned_scratch_connection is None
            and self._scratch_directory is None
            and all(owner._settled for owner in native_sql_children(self))
        )
        if self._closed:
            retire_native_sql_parent(self)
            with _LIVE_SEALS_LOCK:
                _LIVE_SEALS.pop(id(self), None)
        if first_error is not None:
            raise first_error

    def __enter__(self) -> PreparedIndexMutation:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> None:
        try:
            self.close()
        except BaseException as close_error:
            if exc is None:
                raise
            exc.add_note(f"reference-seal cleanup also failed: {close_error}")


@dataclass(frozen=True, slots=True)
class IndexMutationDestination:
    """Explicit derived-only destination, never permission from an absent root."""

    index_path: Path | None
    kind: Literal["owned_inactive", "standalone", "standalone_memory"]
    generation: IndexGeneration | None = None
    memory_connection: sqlite3.Connection | None = None

    @classmethod
    def owned_inactive(cls, generation: IndexGeneration) -> IndexMutationDestination:
        destination = cls(Path(generation.index_path).resolve(strict=True), "owned_inactive", generation)
        destination.validate()
        return destination

    @classmethod
    def standalone(cls, index_path: Path) -> IndexMutationDestination:
        destination = cls(index_path.resolve(strict=True), "standalone")
        destination.validate()
        return destination

    @classmethod
    def standalone_memory(cls, conn: sqlite3.Connection) -> IndexMutationDestination:
        destination = cls(None, "standalone_memory", memory_connection=conn)
        destination.validate()
        return destination

    def validate(self) -> None:
        if self.kind == "standalone_memory":
            conn = self.memory_connection
            if conn is None or self.index_path is not None or self.generation is not None:
                raise ReferenceSealError("standalone memory Index lacks its exact declared connection")
            databases = conn.execute("PRAGMA database_list").fetchall()
            if any(str(row[1]) not in {"main", "temp"} or str(row[2]) for row in databases):
                raise ReferenceSealError("standalone memory Index cannot contain an archive database")
            return
        if self.index_path is None or self.memory_connection is not None:
            raise ReferenceSealError("named Index destination lacks its actual path")
        if self.kind == "owned_inactive":
            from polylogue.storage.index_generation import IndexGenerationStore

            generation = self.generation
            if generation is None or generation.state != "inactive":
                raise ReferenceSealError("Index destination lacks an inactive generation owner")
            current = IndexGenerationStore.for_archive_root(Path(generation.archive_root), repair_anchor=False).load(
                generation.generation_id
            )
            if current != generation or Path(current.index_path).resolve(strict=True) != self.index_path:
                raise ReferenceSealStaleError("inactive Index generation ownership changed")
            return
        if self.generation is not None:
            raise ReferenceSealError("standalone Index destination carries archive generation metadata")
        parent = self.index_path.parent
        if ".index-generations" in self.index_path.parts or any(
            (parent / filename).exists() or (parent / filename).is_symlink()
            for filename in (
                "source.db",
                "user.db",
                "audit.db",
                "ops.db",
                "embeddings.db",
                "generation.json",
                ".polylogue-format.json",
                ".index-active-pointer",
                ".index-generations",
                ".index-rebuild-transactions",
                ".bootstrap",
            )
        ):
            raise ReferenceSealError("an archive or generation cannot be declared a standalone Index")

    @contextmanager
    def mutation_scope(self, conn: sqlite3.Connection) -> Iterator[IndexMutationScope]:
        self.validate()
        if conn.in_transaction:
            raise ReferenceSealError("an Index mutation scope must start before BEGIN")
        if self.kind == "standalone_memory":
            matches = conn is self.memory_connection
        else:
            matches = index_path_for_connection(conn).resolve(strict=True) == self.index_path
        if not matches:
            raise ReferenceSealError("Index destination does not match its connection")
        with _owned_index_transaction(IndexMutationScope(None, conn, destination=self)) as scope:
            yield scope


@contextmanager
def _owned_index_transaction(scope: IndexMutationScope) -> Iterator[IndexMutationScope]:
    token = _ACTIVE_MUTATION_SCOPE.set(scope)
    try:
        _check_reference_cancellation()
        scope.conn.execute("BEGIN IMMEDIATE")
        yield scope
        if scope._active and not scope._committed:
            scope.commit()
    except BaseException as primary:
        if not scope._cleanup_started:
            try:
                scope.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup("Index mutation and scope cleanup failed", [primary, cleanup]) from primary
        raise
    finally:
        try:
            # commit/rollback already made their single cleanup attempt. An
            # unsuccessful attempt stays retained for an explicit owner retry.
            if not scope._cleanup_started:
                scope.close()
        finally:
            _ACTIVE_MUTATION_SCOPE.reset(token)


@dataclass(slots=True)
class IndexMutationScope:
    """Exact writer transaction that borrows a prepared durable-reference seal."""

    seal: PreparedIndexMutation | None
    conn: sqlite3.Connection
    destination: IndexMutationDestination | None = None
    owner_thread: threading.Thread = field(default_factory=threading.current_thread)
    owner_pid: int = field(default_factory=os.getpid)
    owner_task: object | None = field(default_factory=_current_task)
    _active: bool = True
    _committed: bool = False
    _cleanup_started: bool = False
    _rollback_required: bool = True
    _user_owner: NativeSQLCustodyOwner | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if (self.seal is None) == (self.destination is None):
            raise ReferenceSealError("Index transaction requires exactly one declared destination authority")

    def suppression_reader(self) -> sqlite3.Connection | None:
        """Borrow the declared User observer for this exact commit window."""
        self.require_new_work(self.conn)
        if self.seal is not None:
            return self.seal.observer("user")
        destination = self.destination
        if destination is None:
            raise ReferenceSealError("Index scope has no declared destination")
        destination.validate()
        if destination.kind != "owned_inactive":
            return None
        generation = destination.generation
        if generation is None:
            raise ReferenceSealError("inactive Index lacks its archive owner")
        path = Path(generation.archive_root) / "user.db"
        if not path.is_file():
            raise ReferenceSealError("declared archive is missing its required durable User tier")
        if self._user_owner is None:
            # The same scope owns one reader, with actual creator custody and
            # scope lifetime retained before factory setup SQL can fail.
            try:
                self._user_owner = _open_readonly_owner(path, validate_schema=False, lifetime_dependencies=(self,))
            except NativeConnectionSettlementError as failure:
                # Construction already attempted close. Retain its exact owner
                # for explicit retry, without re-closing it during unwinding.
                self._user_owner = failure.owner
                self._active = False
                self._cleanup_started = True
                try:
                    self._rollback_index()
                except BaseException as rollback:
                    raise BaseExceptionGroup(
                        "User construction cleanup and Index rollback failed", [failure, rollback]
                    ) from failure
                raise
        owner = self._user_owner
        custody = current_sql_custody()
        if custody is not owner.custody:
            raise ReferenceSealError("suppression reader belongs to another admitted writer")
        if custody is not None:
            custody.assert_namespace()
        return owner.require_connection()

    def _close_suppression_reader(self) -> None:
        owner = self._user_owner
        if owner is not None:
            owner.close()
            self._user_owner = None

    def note_session_namespace_change(self) -> None:
        self.require_connection(self.conn)
        _check_reference_cancellation()
        if self.seal is not None:
            self.seal.note_session_namespace_change()

    def note_deleted_session(self, session_id: str) -> None:
        self.require_connection(self.conn)
        _check_reference_cancellation()
        if self.seal is not None:
            self.seal.note_deleted_session(session_id)

    def note_deleted_message_ids(self, message_ids: Iterable[str]) -> None:
        self.require_connection(self.conn)
        _check_reference_cancellation()
        if self.seal is not None:
            self.seal.note_deleted_message_ids(message_ids)

    def note_lineage_change(self, session_id: str) -> None:
        self.require_connection(self.conn)
        _check_reference_cancellation()
        if self.seal is not None:
            self.seal.note_lineage_change(self.conn, session_id)

    def require_connection(self, conn: sqlite3.Connection) -> None:
        if not self._active or conn is not self.conn or not self._same_owner():
            raise ReferenceSealError("operation requires the matching live index mutation scope")

    def require_new_work(self, conn: sqlite3.Connection) -> None:
        self.require_connection(conn)
        _check_reference_cancellation()

    def _same_owner(self) -> bool:
        return (
            os.getpid() == self.owner_pid
            and threading.current_thread() is self.owner_thread
            and _current_task() is self.owner_task
        )

    def validate_reachability(self, conn: sqlite3.Connection) -> None:
        self.require_connection(conn)
        _check_reference_cancellation()
        if self.seal is not None:
            self.seal.validate_reachability(conn)
        elif self.destination is not None:
            self.destination.validate()
            matches = (
                conn is self.destination.memory_connection
                if self.destination.kind == "standalone_memory"
                else index_path_for_connection(conn).resolve(strict=True) == self.destination.index_path
            )
            if not matches:
                raise ReferenceSealStaleError("Index transaction destination changed")

    def commit(self) -> None:
        self.require_connection(self.conn)
        if not self.conn.in_transaction:
            raise ReferenceSealError("index mutation scope cannot commit without its transaction")
        self.validate_reachability(self.conn)
        self.conn.commit()
        self._committed = True
        self._active = False
        self._rollback_required = False
        self._cleanup_started = True
        self._close_suppression_reader()

    @property
    def settled(self) -> bool:
        return not self._rollback_required and self._user_owner is None

    def rollback(self) -> None:
        self.close()

    def _rollback_index(self) -> None:
        failures: list[BaseException] = []
        try:
            self.conn.set_progress_handler(None, 0)
        except BaseException as failure:
            failures.append(failure)
        try:
            self.conn.rollback()
        except BaseException as failure:
            failures.append(failure)
        else:
            self._rollback_required = False
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("Index rollback failed", failures)

    def close(self) -> None:
        if os.getpid() != self.owner_pid or threading.current_thread() is not self.owner_thread:
            raise ReferenceSealError("Index scope cleanup belongs to another process or thread")
        if _current_task() is not self.owner_task and (
            not isinstance(self.owner_task, asyncio.Task) or not self.owner_task.done()
        ):
            raise ReferenceSealError("Index scope cleanup belongs to another task")
        self._active = False
        self._cleanup_started = True
        failures: list[BaseException] = []
        if self._rollback_required:
            try:
                self._rollback_index()
            except BaseException as failure:
                failures.append(failure)
        try:
            self._close_suppression_reader()
        except BaseException as failure:
            failures.append(failure)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("Index and User scope cleanup failed", failures)


def current_index_mutation_scope() -> IndexMutationScope | None:
    """Return the exact scope only to owners that pass it explicitly onward."""
    scope = _ACTIVE_MUTATION_SCOPE.get()
    return scope if scope is not None and scope._active else None


def _current_index_mutation_scope(conn: sqlite3.Connection) -> IndexMutationScope:
    scope = current_index_mutation_scope()
    if scope is None:
        raise ReferenceSealError("a destructive lineage rewrite requires its outer index mutation scope")
    scope.require_connection(conn)
    return scope


def note_current_session_namespace_change(conn: sqlite3.Connection) -> None:
    _current_index_mutation_scope(conn).note_session_namespace_change()


def note_current_deleted_session(conn: sqlite3.Connection, session_id: str) -> None:
    _current_index_mutation_scope(conn).note_deleted_session(session_id)


def note_current_deleted_message_ids(conn: sqlite3.Connection, message_ids: Iterable[str]) -> None:
    _current_index_mutation_scope(conn).note_deleted_message_ids(message_ids)


def note_current_lineage_change(conn: sqlite3.Connection, session_id: str) -> None:
    _current_index_mutation_scope(conn).note_lineage_change(session_id)


__all__ = [
    "IndexMutationDestination",
    "IndexMutationScope",
    "PreparedIndexMutation",
    "ReferenceSealError",
    "ReferenceSealStaleError",
    "current_index_mutation_scope",
    "note_current_session_namespace_change",
    "note_current_deleted_message_ids",
    "note_current_lineage_change",
    "note_current_deleted_session",
]
