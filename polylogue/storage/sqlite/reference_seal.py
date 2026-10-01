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
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

from polylogue.core.compute_cancel import compute_cancel_requested
from polylogue.core.refs import EvidenceRef, ObjectRef, parse_public_ref
from polylogue.core.sqlite_scratch import connect_scratch_database
from polylogue.storage.sqlite.audit_leaf import VerifiedAuditLeaf
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

_LIVE_SEALS_LOCK = threading.RLock()
_LIVE_SEALS: dict[int, PreparedIndexMutation] = {}


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


def _relevant_ref(value: str) -> ObjectRef | EvidenceRef | None:
    try:
        parsed = parse_public_ref(value)
    except ValueError as exc:
        raise ReferenceSealError(f"durable reference is malformed: {value!r}") from exc
    if isinstance(parsed, EvidenceRef):
        return parsed
    if parsed.kind in {"session", "message", "block", "action"}:
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


def _resolve(conn: sqlite3.Connection, ref: ObjectRef | EvidenceRef) -> _ResolvedReference | None:
    if isinstance(ref, EvidenceRef):
        from polylogue.storage.sqlite.archive_tiers.archive import resolve_session_id_in_index

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

    if ref.kind == "session":
        from polylogue.storage.sqlite.archive_tiers.archive import resolve_session_id_in_index

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


def _still_resolves(conn: sqlite3.Connection, ref: _ResolvedReference) -> bool:
    if ref.kind == "session":
        query = "SELECT 1 FROM sessions WHERE session_id = ?"
        params: tuple[object, ...] = (ref.object_id,)
    elif ref.kind == "message":
        query = "SELECT 1 FROM messages WHERE message_id = ?"
        params = (ref.object_id,)
    elif ref.kind == "block-id":
        query = "SELECT 1 FROM blocks WHERE block_id = ?"
        params = (ref.object_id,)
    else:
        query = "SELECT 1 FROM blocks WHERE message_id = ? AND position = ?"
        params = (ref.object_id, int(ref.qualifier or "-1"))
    if conn.execute(query, params).fetchone() is None:
        return False
    if ref.scope_session_id is not None and ref.kind in {"message", "block-id", "block-position"}:
        return (
            ref.target_message_id is not None
            and _locate_composed_message(conn, ref.scope_session_id, ref.target_message_id) is not None
        )
    return True


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
        self._scratch_directory: tempfile.TemporaryDirectory | None = None
        self._scratch: sqlite3.Connection | None = None
        with _LIVE_SEALS_LOCK:
            _LIVE_SEALS[id(self)] = self
        try:
            self._scratch_directory = tempfile.TemporaryDirectory(prefix="polylogue-reference-seal-")
            _check_reference_cancellation()
            self._scratch = connect_scratch_database(Path(self._scratch_directory.name) / "refs.db")
            self._scratch.set_progress_handler(lambda: int(compute_cancel_requested()), 2000)
            self._scratch.executescript(
                "CREATE TABLE resolved_refs ("
                "kind TEXT NOT NULL, owner_session_id TEXT NOT NULL, object_id TEXT NOT NULL, "
                "qualifier TEXT NOT NULL, scope_session_id TEXT NOT NULL, target_message_id TEXT NOT NULL, "
                "PRIMARY KEY(kind, owner_session_id, object_id, qualifier, scope_session_id, target_message_id)) "
                "WITHOUT ROWID;"
                "CREATE INDEX resolved_refs_by_session ON resolved_refs(owner_session_id, kind, object_id, qualifier);"
                "CREATE INDEX resolved_refs_by_scope_session ON resolved_refs(scope_session_id, kind, object_id, qualifier);"
                "CREATE INDEX resolved_refs_by_target_message ON resolved_refs(target_message_id);"
                "CREATE TABLE destructive_message_ids(message_id TEXT PRIMARY KEY) WITHOUT ROWID;"
                "CREATE TABLE candidate_refs ("
                "kind TEXT NOT NULL, owner_session_id TEXT NOT NULL, object_id TEXT NOT NULL, "
                "qualifier TEXT NOT NULL, scope_session_id TEXT NOT NULL, target_message_id TEXT NOT NULL, "
                "PRIMARY KEY(kind, owner_session_id, object_id, qualifier, scope_session_id, target_message_id)) "
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

    def _open_observer(self, name: str, path: Path) -> sqlite3.Connection:
        leaf = VerifiedAuditLeaf(path.parent, filename=path.name, identity_access="lock-preserving")
        leaf.__enter__()
        self._observer_leaves[name] = leaf
        conn = open_readonly_connection(leaf.anchored_path, validate_schema=False)
        self._observers[name] = conn
        try:
            leaf.assert_unchanged()
            conn.row_factory = sqlite3.Row
            conn.set_progress_handler(lambda: int(compute_cancel_requested()), 2000)
            return conn
        except BaseException:
            try:
                conn.close()
            except BaseException:
                # Retain both actual handles for owner-thread recovery.
                raise
            else:
                self._observers.pop(name, None)
            raise

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
                                    "INSERT OR IGNORE INTO resolved_refs VALUES (?, ?, ?, ?, ?, ?)",
                                    (
                                        target.kind,
                                        target.owner_session_id,
                                        target.object_id,
                                        target.qualifier or "",
                                        target.scope_session_id or "",
                                        target.target_message_id or "",
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

    def note_deleted_session(self, session_id: str) -> None:
        self._require_live_owner()
        self._scratch.execute(
            "INSERT OR IGNORE INTO candidate_refs "
            "SELECT kind, owner_session_id, object_id, qualifier, scope_session_id, target_message_id FROM resolved_refs "
            "WHERE owner_session_id = ? OR scope_session_id = ?",
            (session_id, session_id),
        )

    def note_lineage_change(self, conn: sqlite3.Connection, session_id: str) -> None:
        """Track refs scoped to every composed transcript below a changed node."""
        self._require_live_owner()
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
                "SELECT kind, owner_session_id, object_id, qualifier, scope_session_id, target_message_id "
                "FROM resolved_refs WHERE scope_session_id = ?",
                (str(affected_session_id),),
            )
        self._scratch.commit()

    def note_deleted_message_ids(self, message_ids: Iterable[str]) -> None:
        self._require_live_owner()
        self._scratch.executemany(
            "INSERT OR IGNORE INTO destructive_message_ids VALUES (?)", ((message_id,) for message_id in message_ids)
        )
        self._scratch.execute(
            "INSERT OR IGNORE INTO candidate_refs "
            "SELECT r.kind, r.owner_session_id, r.object_id, r.qualifier, r.scope_session_id, r.target_message_id "
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
        scope = IndexMutationScope(self, conn)
        token = _ACTIVE_MUTATION_SCOPE.set(scope)
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield scope
            if not scope._committed:
                scope.commit()
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            scope.close()
            _ACTIVE_MUTATION_SCOPE.reset(token)

    def validate_for_writer(self, conn: sqlite3.Connection) -> None:
        self._require_live_owner()
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
        self._require_live_owner()
        try:
            return self._observers[tier]
        except KeyError as exc:
            raise ValueError(f"unknown reference-seal tier {tier!r}") from exc

    def observer_version(self, tier: str) -> int:
        self._require_live_owner()
        try:
            return self._versions[tier]
        except KeyError as exc:
            raise ValueError(f"unknown reference-seal tier {tier!r}") from exc

    def validate_observers_current(self) -> None:
        """Point-check every retained observer immediately after gate admission."""
        self._require_live_owner()
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
        self._require_live_owner()
        if self._candidate_path is not None:
            raise ReferenceSealError("this reference seal already has a promotion candidate")
        candidate = Path(candidate_index_path).resolve(strict=True)
        identity_before = _tier_identity(candidate)
        observer = self._open_observer("candidate", candidate)
        first: _ResolvedReference | None = None
        lost_count = 0
        version_before = version_after = -1
        primary: BaseException | None = None
        try:
            version_before = int(observer.execute("PRAGMA data_version").fetchone()[0])
            schema = self._candidate_schema_identity(observer)
            observer.execute("BEGIN")
            rows = self._scratch.execute(
                "SELECT kind, owner_session_id, object_id, qualifier, scope_session_id, target_message_id "
                "FROM resolved_refs ORDER BY kind, object_id, qualifier"
            )
            for row in rows:
                _check_reference_cancellation()
                ref = _ResolvedReference(
                    str(row[0]),
                    str(row[1]),
                    str(row[2]),
                    str(row[3]) or None,
                    str(row[4]) or None,
                    str(row[5]) or None,
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
            first_missing: str | None = None
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
        self._require_live_owner()
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
        self._require_live_owner()
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
        """Advance the same observer only after verifying this exact committed write."""
        self._require_live_owner()
        if (
            not isinstance(receipt, KnownSourceMutationReceipt)
            or receipt._seal is not self
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
        self._require_live_owner()
        if not _same_incarnation(self._writer_identity(conn), self.index_identity):
            raise ReferenceSealStaleError("reference seal settled on a different index incarnation")
        rows = self._scratch.execute(
            "SELECT kind, owner_session_id, object_id, qualifier, scope_session_id, target_message_id "
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

    def _require_live_owner(self) -> None:
        _check_reference_cancellation()
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
            if settle(observer.close):
                self._observers.pop(name, None)
        for name, leaf in tuple(self._observer_leaves.items()):
            if name not in self._observers and settle(leaf.close):
                self._observer_leaves.pop(name, None)
        if self._scratch is not None:
            scratch = self._scratch
            if settle(scratch.close):
                self._scratch = None
        if self._scratch is None and self._scratch_directory is not None:
            directory = self._scratch_directory
            if settle(directory.cleanup):
                self._scratch_directory = None
        self._closed = (
            not self._observers
            and not self._observer_leaves
            and self._scratch is None
            and self._scratch_directory is None
        )
        if self._closed:
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


@dataclass(slots=True)
class IndexMutationScope:
    """Exact writer transaction that borrows a prepared durable-reference seal."""

    seal: PreparedIndexMutation
    conn: sqlite3.Connection
    owner_thread: threading.Thread = field(default_factory=threading.current_thread)
    owner_pid: int = field(default_factory=os.getpid)
    owner_task: object | None = field(default_factory=_current_task)
    _active: bool = True
    _committed: bool = False

    def note_deleted_session(self, session_id: str) -> None:
        self.require_connection(self.conn)
        self.seal.note_deleted_session(session_id)

    def note_deleted_message_ids(self, message_ids: Iterable[str]) -> None:
        self.require_connection(self.conn)
        self.seal.note_deleted_message_ids(message_ids)

    def note_lineage_change(self, session_id: str) -> None:
        self.require_connection(self.conn)
        self.seal.note_lineage_change(self.conn, session_id)

    def require_connection(self, conn: sqlite3.Connection) -> None:
        if not self._active or conn is not self.conn or not self._same_owner():
            raise ReferenceSealError("operation requires the matching live index mutation scope")

    def _same_owner(self) -> bool:
        return (
            os.getpid() == self.owner_pid
            and threading.current_thread() is self.owner_thread
            and _current_task() is self.owner_task
        )

    def validate_reachability(self, conn: sqlite3.Connection) -> None:
        self.require_connection(conn)
        self.seal.validate_reachability(conn)

    def commit(self) -> None:
        self.require_connection(self.conn)
        if not self.conn.in_transaction:
            raise ReferenceSealError("index mutation scope cannot commit without its transaction")
        self.seal.validate_reachability(self.conn)
        self.conn.commit()
        self._committed = True
        self._active = False

    def rollback(self) -> None:
        self.require_connection(self.conn)
        self.conn.rollback()
        self._active = False

    def close(self) -> None:
        self._active = False


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


def note_current_deleted_session(conn: sqlite3.Connection, session_id: str) -> None:
    _current_index_mutation_scope(conn).note_deleted_session(session_id)


def note_current_deleted_message_ids(conn: sqlite3.Connection, message_ids: Iterable[str]) -> None:
    _current_index_mutation_scope(conn).note_deleted_message_ids(message_ids)


def note_current_lineage_change(conn: sqlite3.Connection, session_id: str) -> None:
    _current_index_mutation_scope(conn).note_lineage_change(session_id)


__all__ = [
    "IndexMutationScope",
    "PreparedIndexMutation",
    "ReferenceSealError",
    "ReferenceSealStaleError",
    "current_index_mutation_scope",
    "note_current_deleted_message_ids",
    "note_current_lineage_change",
    "note_current_deleted_session",
]
