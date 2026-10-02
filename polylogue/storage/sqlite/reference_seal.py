"""Off-writer proof that an index mutation preserves resolved durable refs.

The seal is prepared before an index write transaction and is bound to the
actual archive root selected by the writer.  It reads only declared typed
reference fields in ``user.db`` and ``audit.db``; free-form assertion values,
settings, and receipt payloads are deliberately outside this contract.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import pickle
import re
import sqlite3
import struct
import tempfile
import threading
from builtins import BaseExceptionGroup
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import closing, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import partial
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from polylogue.storage.blob_publication import PreparedBlobPublicationClaim
    from polylogue.storage.index_generation import ActiveWriterLease, IndexGeneration
    from polylogue.storage.sqlite.write_lease import ArchiveWriteCustody

from polylogue.core.compute_cancel import compute_cancel_requested
from polylogue.core.enums import Origin
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
from polylogue.storage.io_phase_metrics import (
    bind_readonly_incremental_blob_custody,
    close_connection_cursor,
    connection_cursor,
    live_connection_cursors,
)
from polylogue.storage.sqlite.audit_leaf import VerifiedAuditLeaf
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    NativeSQLCustodyOwner,
    _close_failed_native_construction,
    _open_readonly_owner,
    native_sql_children,
    native_sql_parent_for_connection,
    open_isolated_write_connection,
    open_readonly_connection,
    open_scratch_connection,
    open_source_tier_write_connection,
)
from polylogue.storage.sqlite.literal_cells import (
    LITERAL_CHUNK_BYTES,
    SQLiteLiteralCell,
    cell_projection,
    literal_metadata,
    quote_identifier,
    stream_literal_blob,
)
from polylogue.storage.sqlite.write_lease import ArchiveWriteCustody, current_sql_custody, require_write_lease

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


class _SourceAllocationCollisionError(ReferenceSealError):
    """A real scratch allocation selected an untouched original coordinate."""

    def __init__(self, table: str, rowid: int) -> None:
        super().__init__("selected Source allocation collides with an original physical row")
        self.table = table
        self.rowid = rowid


class ReferenceSealStaleError(ReferenceSealError):
    """A tier changed after the off-writer reference census was prepared."""


@dataclass(frozen=True, slots=True)
class KnownTierCell:
    """Immutable literal locator in this exact original disk witness."""

    _seal: PreparedIndexMutation
    _cell_id: int


@dataclass(frozen=True, slots=True)
class KnownTierRowImage:
    """Complete columns and physical rowid, with no Python variable cells."""

    _seal: PreparedIndexMutation
    table: str
    columns: tuple[str, ...]
    rowid: int | None
    cells: tuple[KnownTierCell, ...]


@dataclass(frozen=True, slots=True)
class KnownTierRowEffect:
    """One complete, declared SQLite row transition of an existing producer.

    Columns use the canonical table order. Absence is distinct from a row
    containing NULL values. Producers include trigger and foreign-key effects
    in the same stream; a missing transition cannot be absorbed at commit.
    """

    table: str
    columns: tuple[str, ...]
    old: KnownTierRowImage | None
    new: KnownTierRowImage | None
    _trigger_parent_ordinal: int | None = None
    _canonical_trigger: str | None = None


@dataclass(frozen=True, slots=True)
class KnownTierMutationReceipt:
    """Proof handle for one exact durable-tier mutation committed under this seal."""

    _seal: PreparedIndexMutation
    _tier_identity: tuple[int, int, int, int]
    _prior_data_version: int
    _table: str
    _columns: tuple[str, ...]
    _rows: tuple[tuple[object, ...], ...]
    _seal_nonce: object
    _key_column: str
    _effect_count: int
    _tier: Literal["source", "user"]


@dataclass(frozen=True, slots=True)
class KnownTierMutationPermit:
    """Complete row effects of one admitted dedicated durable-tier transaction."""

    _seal: PreparedIndexMutation
    _tier_identity: tuple[int, int, int, int]
    _prior_data_version: int
    _table: str
    _columns: tuple[str, ...]
    _rows: tuple[tuple[object, ...], ...]
    _seal_nonce: object

    _key_column: str
    _tier: Literal["source", "user"]
    _connection: sqlite3.Connection | None = field(default=None, init=False, compare=False, repr=False)
    _custody: ArchiveWriteCustody | None = field(default=None, init=False, compare=False, repr=False)
    _literal_attach_uri: str | None = field(default=None, init=False, compare=False, repr=False)
    _setup_temp_object: tuple[str, str | None] | None = field(default=None, init=False, compare=False, repr=False)
    _setup_connection: sqlite3.Connection | None = field(default=None, init=False, compare=False, repr=False)
    _setup_pragma: tuple[str, str, str | None] | None = field(default=None, init=False, compare=False, repr=False)
    _setup_attempted: bool = field(default=False, init=False, compare=False, repr=False)
    _effects: int = field(default=0, init=False, compare=False, repr=False)
    _commit_allowed: bool = field(default=False, init=False, compare=False, repr=False)
    _commit_attempted: bool = field(default=False, init=False, compare=False, repr=False)
    _acceptance_reservation: bool = field(default=False, init=False, compare=False, repr=False)
    _writer_data_version: int | None = field(default=None, init=False, compare=False, repr=False)
    _failure: BaseException | None = field(default=None, init=False, compare=False, repr=False)
    _source_schedule_started: bool = field(default=False, init=False, compare=False, repr=False)
    _receipt_accepted: bool = field(default=False, init=False, compare=False, repr=False)

    @property
    def tier(self) -> Literal["source", "user"]:
        return self._tier

    @property
    def terminal_parent(self) -> PreparedIndexMutation:
        return self._seal

    @contextmanager
    def hold_authority(self) -> Iterator[KnownTierMutationPermit]:
        self._seal.validate_observers_current()
        require_write_lease("known Source mutation", archive_root=self._seal.archive_root)
        custody = current_sql_custody()
        if custody is None:
            raise ReferenceSealError("known Source mutation requires actual physical archive custody")
        self._seal._retain_mutation_custody(custody)
        object.__setattr__(self, "_custody", custody)
        try:
            with custody.known_tier_mutation(self):
                yield self
        finally:
            object.__setattr__(self, "_custody", None)

    @contextmanager
    def mutation_connection(self) -> Iterator[sqlite3.Connection]:
        factory = (
            open_source_tier_write_connection
            if self._tier == "source"
            else partial(open_isolated_write_connection, purpose="prepared User mutation")
        )
        connection = factory(self._seal._paths[self._tier], archive_root=self._seal.archive_root, mutation_permit=self)
        owner = next(child for child in native_sql_children(self._seal) if child.connection is connection)
        try:
            yield owner.require_connection()
        except BaseException as primary:
            if self._failure is not None and self._failure is not primary:
                # SQLite wraps exceptions from the exact-row callback. Keep
                # the actual authority/cancellation fault at the product seam.
                _close_failed_native_construction(owner, self._failure)
                raise self._failure from primary
            _close_failed_native_construction(owner, primary)
            raise
        else:
            owner.close()

    def configure_mutation_connection(self, connection: sqlite3.Connection, statements: tuple[str, ...]) -> None:
        self._seal._require_live_owner()
        if self._custody is None or current_sql_custody() is not self._custody:
            raise ReferenceSealError("known tier profile setup requires its admitted physical custody")
        if self._setup_attempted or self._connection is not None or connection.in_transaction:
            raise ReferenceSealError("known tier profile setup requires one fresh dedicated connection")
        if not any(child.connection is connection for child in native_sql_children(self._seal)):
            raise ReferenceSealError("known tier profile setup requires its registered original native child")
        with self._seal._owned_cursor(connection, "PRAGMA database_list") as cursor:
            path = next(str(row[2]) for row in cursor if row[1] == "main")
        if Path(path).resolve() != self._seal._paths[self._tier].resolve():
            raise ReferenceSealError("known tier profile setup selected another physical tier")
        object.__setattr__(self, "_setup_attempted", True)
        object.__setattr__(self, "_setup_connection", connection)
        try:
            with self._seal._owned_cursor(connection, "PRAGMA data_version") as cursor:
                object.__setattr__(self, "_writer_data_version", int(cursor.fetchone()[0]))
            self._seal.validate_observers_current()
            for statement in statements:
                match = re.fullmatch(r"PRAGMA (?:(main)\.)?([a-z_]+) = ([A-Za-z_0-9-]+)", statement)
                if match is None:
                    raise ReferenceSealError("known tier profile setup requires one exact declared PRAGMA")
                schema, pragma, value = match.groups()
                object.__setattr__(self, "_setup_pragma", (pragma.casefold(), value.casefold(), schema))
                try:
                    with self._seal._owned_cursor(connection, statement):
                        pass
                finally:
                    object.__setattr__(self, "_setup_pragma", None)
        finally:
            object.__setattr__(self, "_setup_pragma", None)
            object.__setattr__(self, "_setup_connection", None)
        self._bind_mutation_connection(connection)

    def _bind_mutation_connection(self, connection: sqlite3.Connection) -> None:
        self._seal._require_live_owner()
        if self._custody is None or current_sql_custody() is not self._custody:
            raise ReferenceSealError("known Source connection has no admitted physical custody")
        if self._connection is not None or connection.in_transaction:
            raise ReferenceSealError("known Source mutation requires one fresh dedicated connection")
        with self._seal._owned_cursor(connection, "PRAGMA database_list") as cursor:
            path = next(str(row[2]) for row in cursor if row[1] == "main")
        if Path(path).resolve() != self._seal._paths[self._tier].resolve():
            raise ReferenceSealError("known Source mutation selected another tier")
        object.__setattr__(self, "_connection", connection)
        native_owner = next(child for child in native_sql_children(self._seal) if child.connection is connection)
        bind_readonly_incremental_blob_custody(
            connection, native_owner.retain_incremental_blob, native_owner._require_incremental_read
        )
        self._bind_literal_input(native_owner)
        if self._tier == "source":
            self._seal._verify_source_trigger_definitions(connection)
        if self._writer_data_version is None:
            raise ReferenceSealError("known tier guard binding has no original setup version")
        with self._seal._owned_cursor(
            self._seal._scratch,
            "SELECT table_name, columns_blob, keys_blob FROM temp.known_tier_effect_tables WHERE tier=? ORDER BY table_name",
            (self._tier,),
        ) as metadata:
            for ordinal, (table, columns_blob, _keys_blob) in enumerate(metadata):
                if table == "sqlite_sequence":
                    # SQLite updates AUTOINCREMENT state internally without
                    # authorizer callbacks and forbids triggers on this table.
                    # Its selected complete effect is verified at commit.
                    continue
                columns = pickle.loads(columns_blob)
                rowid_alias = self._seal._physical_rowid_alias(connection, table, columns)

                def check_effect(
                    phase: str,
                    operation: str,
                    old_rowid: int | None,
                    new_rowid: int | None,
                    table: str = table,
                ) -> int:
                    try:
                        self._seal._require_new_work()
                        return self._consume_native_effect(connection, table, phase, operation, old_rowid, new_rowid)
                    except BaseException as failure:
                        object.__setattr__(self, "_failure", failure)
                        raise

                function = f"polylogue_known_tier_effect_{ordinal}"
                connection.create_function(function, 4, check_effect)
                for operation in ("INSERT", "UPDATE", "DELETE"):
                    old = "NULL" if operation == "INSERT" else f"OLD.{quote_identifier(rowid_alias)}"
                    new = "NULL" if operation == "DELETE" else f"NEW.{quote_identifier(rowid_alias)}"
                    for phase in ("BEFORE", "AFTER"):
                        trigger_name = f"polylogue_known_tier_{ordinal}_{operation.lower()}_{phase.lower()}"
                        self._create_native_temp_object(
                            connection,
                            trigger_name,
                            table,
                            f"CREATE TEMP TRIGGER {trigger_name} "
                            f"{phase} {operation} ON main.{quote_identifier(table)} "
                            f"BEGIN SELECT {function}('{phase}', '{operation}', {old}, {new}); END",
                        )

    def _consume_native_effect(
        self,
        connection: sqlite3.Connection,
        table: str,
        phase: str,
        operation: str,
        old_rowid: int | None,
        new_rowid: int | None,
    ) -> int:
        # BEFORE INSERT is a precursor of either INSERT or UPSERT UPDATE.
        # Omitted rowids and defaults are not final here. No row transition
        # consumes authority until SQLite exposes its actual physical image.
        if phase == "BEFORE" and operation == "INSERT":
            return 1
        if operation not in {"INSERT", "UPDATE", "DELETE"} or phase not in {"BEFORE", "AFTER"}:
            raise ReferenceSealError("native effect has no declared SQLite phase")
        if any(value is not None and type(value) is not int for value in (old_rowid, new_rowid)):
            raise ReferenceSealError("native effect requires SQLite physical row identifiers")
        expected_state = 2 if phase == "AFTER" and operation != "INSERT" else 0
        with self._seal._owned_cursor(
            self._seal._scratch,
            "SELECT effect_id,old_image,new_image,parent_effect_id,canonical_trigger FROM temp.known_tier_effects "
            "WHERE tier=? AND table_name=? AND old_rowid IS ? AND new_rowid IS ? "
            "AND consumed=? ORDER BY effect_id LIMIT 1",
            (self._tier, table, old_rowid, new_rowid, expected_state),
        ) as cursor:
            expected = cursor.fetchone()
        if expected is None:
            raise ReferenceSealError("tier transaction attempted an undeclared physical row transition")
        if expected[3] is not None:
            self._require_exact_trigger_parent(expected[3], expected[4], table, expected[2])
        image_id = expected[1] if phase == "BEFORE" else expected[2]
        if image_id is not None and not self._seal._matches_retained_row(
            connection, self._seal._retained_row_image(image_id)
        ):
            raise ReferenceSealError("tier transaction differs from its exact retained literal image")
        # DELETE AFTER has no new row. Its exact OLD was checked before the
        # deletion; later authorized trigger effects may reuse that rowid.
        # Terminal absence/presence remains mandatory in the commit preflight.
        with self._seal._owned_cursor(
            self._seal._scratch,
            "UPDATE temp.known_tier_effects SET consumed=? WHERE effect_id=?",
            (2 if phase == "BEFORE" else 1, expected[0]),
        ):
            pass
        if phase == "AFTER":
            object.__setattr__(self, "_effects", self._effects + 1)
        return 1

    def _require_exact_trigger_parent(
        self,
        parent_effect_id: int,
        trigger: str,
        table: str,
        child_image_id: int | None,
    ) -> None:
        """Use exact parent BEFORE authority across canonical AFTER ordering."""
        with self._seal._owned_cursor(
            self._seal._scratch,
            "SELECT table_name,old_image,new_image,consumed FROM temp.known_tier_effects "
            "WHERE tier='source' AND effect_id=?",
            (parent_effect_id,),
        ) as cursor:
            parent = cursor.fetchone()
        # BEFORE has state2. AFTER can already have state1 when SQLite runs
        # a guard before its canonical main AFTER trigger. Both retain the
        # same successfully checked OLD image; no assumed trigger order.
        if parent is None or parent[1] is None or parent[3] not in (1, 2):
            raise ReferenceSealError("canonical child lacks its exact checked parent BEFORE effect")
        old = self._seal._retained_row_image(parent[1])
        new = None if parent[2] is None else self._seal._retained_row_image(parent[2])
        child = None if child_image_id is None else self._seal._retained_row_image(child_image_id)
        self._seal._validate_source_trigger_relation(trigger, old, new, table, child)

    def _bind_literal_input(self, owner: NativeSQLCustodyOwner) -> None:
        connection = owner.require_connection()
        witness_path = self._seal._original_witness_path()
        identity = _tier_identity(witness_path)[:2]
        uri = witness_path.as_uri() + "?mode=ro"
        self._seal._retain_live_literal_reader(owner, self)
        object.__setattr__(self, "_literal_attach_uri", uri)
        try:
            # SQLite's authorizer cannot see a bound ATTACH operand. Quote the
            # original owner's exact URI so authorization compares its actual
            # value; do not authorize an unknown/null attachment target.
            quoted_uri = "'" + uri.replace("'", "''") + "'"
            with self._seal._owned_cursor(connection, f"ATTACH DATABASE {quoted_uri} AS polylogue_literal_witness"):
                pass
        finally:
            object.__setattr__(self, "_literal_attach_uri", None)
        with self._seal._owned_cursor(connection, "PRAGMA database_list") as cursor:
            selected = next((Path(str(row[2])) for row in cursor if row[1] == "polylogue_literal_witness"), None)
        if (
            selected is None
            or selected.resolve(strict=True) != witness_path
            or _tier_identity(selected)[:2] != identity
        ):
            raise ReferenceSealStaleError("literal attachment differs from the original witness file")
        self._create_native_temp_object(
            connection,
            "polylogue_source_literals",
            None,
            "CREATE TEMP VIEW polylogue_source_literals AS "
            "SELECT rowid AS cell_id,literal FROM polylogue_literal_witness.known_tier_literals",
        )

    def _create_native_temp_object(
        self, connection: sqlite3.Connection, name: str, table: str | None, statement: str
    ) -> None:
        if connection is not self._connection or self._setup_temp_object is not None:
            raise ReferenceSealError("native guard setup requires its original dedicated connection")
        object.__setattr__(self, "_setup_temp_object", (name, table))
        try:
            with self._seal._owned_cursor(connection, statement):
                pass
        finally:
            object.__setattr__(self, "_setup_temp_object", None)

    def authorize_tier_sql(
        self,
        connection: sqlite3.Connection,
        action: int,
        first: str | None,
        second: str | None,
        schema: str | None,
        trigger: str | None,
    ) -> bool:
        if action == sqlite3.SQLITE_TRANSACTION and first == "ROLLBACK":
            if connection is self._connection:
                if (
                    os.getpid() != self._seal.index_pid
                    or threading.current_thread() is not self._seal.index_thread
                    or _current_task() is not self._seal.index_task
                ):
                    return False
                # Physical cleanup remains available after seal retirement or
                # cancellation. A foreign connection's rollback cannot alter
                # this dedicated transaction's receipt state.
                if not self._acceptance_reservation:
                    object.__setattr__(self, "_commit_attempted", False)
            return True
        self._seal._require_live_owner()
        if self._custody is None or current_sql_custody() is not self._custody:
            return False
        reads = {sqlite3.SQLITE_READ, sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
        if action == sqlite3.SQLITE_FUNCTION and (second or "").startswith("polylogue_known_tier_effect_"):
            # Row guard callbacks run only from their original TEMP guards.
            # Direct SELECT cannot manufacture BEFORE/AFTER consumption.
            suffix = (second or "").removeprefix("polylogue_known_tier_effect_")
            return suffix.isdecimal() and trigger in {
                f"polylogue_known_tier_{suffix}_{operation}_{phase}"
                for operation in ("insert", "update", "delete")
                for phase in ("before", "after")
            }
        if action in reads:
            return True
        if action == sqlite3.SQLITE_PRAGMA:
            pragma = (first or "").lower()
            # A missing argument does not imply a read: optimize, checkpoint,
            # and incremental_vacuum can change storage without an argument.
            # Parameterized schema inspection is a read, not a setting.
            if pragma in {
                "table_info",
                "table_xinfo",
                "index_info",
                "index_xinfo",
                "index_list",
                "foreign_key_list",
                "foreign_key_check",
            }:
                return True
            if second is None:
                return pragma in {
                    "application_id",
                    "busy_timeout",
                    "cache_size",
                    "collation_list",
                    "compile_options",
                    "database_list",
                    "data_version",
                    "encoding",
                    "foreign_keys",
                    "freelist_count",
                    "function_list",
                    "journal_mode",
                    "journal_size_limit",
                    "locking_mode",
                    "mmap_size",
                    "page_count",
                    "page_size",
                    "query_only",
                    "recursive_triggers",
                    "schema_version",
                    "synchronous",
                    "table_list",
                    "temp_store",
                    "user_version",
                    "wal_autocheckpoint",
                }
            # Only the exact factory statement currently executing on this
            # registered connection has setup permission. No setting remains
            # available after setup success, failure, or guard binding.
            return connection is self._setup_connection and self._setup_pragma == (pragma, second.casefold(), schema)
        if connection is not self._connection:
            return False
        if action == sqlite3.SQLITE_ATTACH:
            return self._literal_attach_uri is not None and first == self._literal_attach_uri
        if (
            self._setup_temp_object is not None
            and self._setup_temp_object[1] is not None
            and action == sqlite3.SQLITE_INSERT
            and first == "sqlite_master"
            and schema == "main"
        ):
            # SQLite reports this synthetic schema authorization for a TEMP
            # trigger on main. Ordinary main DDL still has no CREATE permission.
            return True
        if schema == "temp" and self._setup_temp_object is not None:
            if action in {sqlite3.SQLITE_CREATE_TEMP_TRIGGER, sqlite3.SQLITE_CREATE_TEMP_VIEW}:
                return self._setup_temp_object == (first, second)
            if first == "sqlite_temp_master" and action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE}:
                return True
        if action == sqlite3.SQLITE_TRANSACTION:
            if first == "BEGIN" and self._acceptance_reservation and self._commit_attempted:
                return True
            if first == "COMMIT" and self._commit_allowed and not self._commit_attempted:
                object.__setattr__(self, "_commit_attempted", True)
                return True
            return first == "BEGIN" and not self._commit_attempted
        if self._commit_attempted:
            return False
        if action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE}:
            if (
                self._tier == "source"
                and first == "raw_existence_changes"
                and action == sqlite3.SQLITE_INSERT
                and trigger not in {"raw_existence_delete", "raw_existence_key_change"}
            ):
                return False
            if (
                self._tier == "source"
                and first == "raw_existence_journal_control"
                and (action != sqlite3.SQLITE_UPDATE or trigger != "raw_existence_journal_prune")
            ):
                return False
            with self._seal._owned_cursor(
                self._seal._scratch,
                "SELECT 1 FROM temp.known_tier_effect_tables WHERE tier=? AND table_name = ?",
                (self._tier, first),
            ) as cursor:
                return schema == "main" and first != "sqlite_sequence" and cursor.fetchone() is not None
        return action == sqlite3.SQLITE_SAVEPOINT and self._connection.in_transaction

    def apply_source_statements(self, connection: sqlite3.Connection) -> None:
        """Publish the same witness's canonical schedule on its dedicated writer."""
        self._seal._require_live_owner()
        if (
            self._tier != "source"
            or connection is not self._connection
            or not connection.in_transaction
            or self._commit_attempted
            or self._source_schedule_started
            or self._seal._pending_tier_permits.get("source") is not self
            or self._custody is None
            or current_sql_custody() is not self._custody
        ):
            raise ReferenceSealError("Source statement application requires its exact admitted original permit")
        if self._failure is not None:
            raise self._failure
        with self._seal._owned_cursor(
            self._seal._scratch, "SELECT 1 FROM temp.known_tier_statements WHERE consumed!=0 LIMIT 1"
        ) as cursor:
            if cursor.fetchone() is not None:
                raise ReferenceSealError("original Source statement schedule cannot be reapplied")
        # BEGIN IMMEDIATE is already owned. A foreign Source commit between
        # guard setup and that reservation must refuse before even a no-op
        # scheduled statement is consumed or any durable effect is executed.
        self._seal.validate_observers_current()
        self._seal._settle_witness_metadata()
        self.require_writer_currency()
        object.__setattr__(self, "_source_schedule_started", True)
        previous = 0
        while True:
            _check_reference_cancellation()
            with self._seal._owned_cursor(
                self._seal._scratch,
                "SELECT statement_id,sql FROM temp.known_tier_statements "
                "WHERE statement_id>? ORDER BY statement_id LIMIT 1",
                (previous,),
            ) as cursor:
                row = cursor.fetchone()
            if row is None:
                return
            statement_id, sql = row
            bindings = self._seal._source_statement_bindings(statement_id)
            with self._seal._owned_cursor(
                self._seal._scratch,
                "UPDATE temp.known_tier_statements SET consumed=-1 WHERE statement_id=? AND consumed=0",
                (statement_id,),
            ) as cursor:
                if cursor.rowcount != 1:
                    raise ReferenceSealError("Source statement lost its original unconsumed identity")
            try:
                with self._seal._owned_cursor(connection, sql, bindings):
                    pass
            except BaseException as failure:
                self._remember_application_failure(failure)
                raise
            with self._seal._owned_cursor(
                self._seal._scratch,
                "UPDATE temp.known_tier_statements SET consumed=1 WHERE statement_id=? AND consumed=-1",
                (statement_id,),
            ):
                pass
            previous = statement_id

    def _remember_application_failure(self, failure: BaseException) -> None:
        # SQLite may have just wrapped the original callback fault. Keep the
        # first actual failure instead of replacing it with that wrapper.
        if self._failure is None:
            object.__setattr__(self, "_failure", failure)

    def allow_commit(self, connection: sqlite3.Connection) -> None:
        self._seal._require_live_owner()
        if connection is not self._connection or not connection.in_transaction:
            raise ReferenceSealError("known Source commit does not own its dedicated transaction")
        if self._failure is not None:
            raise self._failure
        if self._tier == "source":
            with self._seal._owned_cursor(
                self._seal._scratch, "SELECT 1 FROM temp.known_tier_statements WHERE consumed!=1 LIMIT 1"
            ) as cursor:
                if cursor.fetchone() is not None:
                    raise ReferenceSealError("Source commit is missing a complete canonical statement schedule")
        # BEGIN IMMEDIATE has now reserved this physical tier. Validate the
        # original observers after reservation so a foreign commit between
        # preparation and BEGIN cannot be absorbed by this mutation.
        self._seal.validate_observers_current()
        self._seal._settle_witness_metadata()
        self.require_writer_currency()
        # Internal sequence writes are admitted only as the declared terminal
        # image of guarded AUTOINCREMENT inserts. Explicit SQL to that table
        # is denied, and the same writer version excludes foreign changes.
        self._seal._verify_known_tier_postimage(connection, self._tier)
        with self._seal._owned_cursor(
            self._seal._scratch,
            "UPDATE temp.known_tier_effects SET consumed=1 WHERE tier=? AND table_name='sqlite_sequence' AND consumed=0",
            (self._tier,),
        ) as cursor:
            implicit = cursor.rowcount
        object.__setattr__(self, "_effects", self._effects + implicit)
        with self._seal._owned_cursor(
            self._seal._scratch,
            "SELECT 1 FROM temp.known_tier_effects WHERE tier=? AND consumed != 1 LIMIT 1",
            (self._tier,),
        ) as cursor:
            if cursor.fetchone():
                raise ReferenceSealError("tier commit is missing declared row effects")
        self._seal._settle_witness_metadata()
        object.__setattr__(self, "_commit_allowed", True)

    def require_writer_currency(self) -> None:
        self._seal._assert_witness_currency()
        connection = self._connection
        if connection is None or self._writer_data_version is None:
            raise ReferenceSealError("tier mutation has no original dedicated writer version")
        with self._seal._owned_cursor(connection, "PRAGMA data_version") as cursor:
            if int(cursor.fetchone()[0]) != self._writer_data_version:
                raise ReferenceSealStaleError("another connection committed during the known tier mutation")

    @contextmanager
    def acceptance_reservation(self) -> Iterator[None]:
        """Reserve the same tier through advancement of the original observer.

        SQLite leaves this connection's data_version unchanged for its own
        commit. A foreign commit in the commit-to-reservation gap changes it.
        Read transactions cache that value, so compare again after acquiring
        BEGIN IMMEDIATE, which excludes foreign commits until rollback.
        """
        connection = self._connection
        if connection is None or connection.in_transaction or not self._commit_attempted:
            raise ReferenceSealError("tier acceptance requires its completed dedicated writer")
        self.require_writer_currency()
        object.__setattr__(self, "_acceptance_reservation", True)
        try:
            try:
                with self._seal._owned_cursor(connection, "BEGIN IMMEDIATE"):
                    pass
                self.require_writer_currency()
                yield
            except BaseException as primary:
                try:
                    connection.rollback()
                except BaseException as cleanup:
                    raise BaseExceptionGroup(
                        "Tier acceptance and reservation cleanup failed", [primary, cleanup]
                    ) from primary
                raise
            else:
                connection.rollback()
        finally:
            object.__setattr__(self, "_acceptance_reservation", False)

    def require_rows(
        self,
        table: str,
        columns: tuple[str, ...],
        rows: tuple[tuple[object, ...], ...],
    ) -> None:
        if (table, columns, rows) != (self._table, self._columns, self._rows):
            raise ReferenceSealError("Source writer does not match the exact prepared mutation")

    def committed(self) -> KnownTierMutationReceipt:
        """Mint a receipt only after the exact Source connection context committed."""
        return self._seal._record_known_tier_commit(self)


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
    with closing(conn.execute("PRAGMA database_list")) as cursor:
        row = next((item for item in cursor if str(item[1]) == "main"), None)
    if row is None or not str(row[2]):
        raise ReferenceSealError("the index writer has no named main database")
    return Path(str(row[2])).resolve(strict=True)


def _user_reference_owner(conn: sqlite3.Connection) -> PreparedIndexMutation:
    owner = native_sql_parent_for_connection(conn)
    if not isinstance(owner, PreparedIndexMutation) or owner.observer("user") is not conn:
        raise ReferenceSealError("durable reference extraction requires this seal's original User observer")
    return owner


def _reference_rows(
    conn: sqlite3.Connection, tier: str, table: str, columns: tuple[str, ...]
) -> Generator[sqlite3.Row, None, None]:
    """Hydrate finite declared fields only after original input amendment."""
    owner = native_sql_parent_for_connection(conn)
    if not isinstance(owner, PreparedIndexMutation) or owner.observer(tier) is not conn:
        raise ReferenceSealError("reference fields require this seal's original tier observer")
    relation = quote_identifier(table)
    projection = ",".join(quote_identifier(column) for column in columns)
    with owner._owned_cursor(conn, f"SELECT rowid FROM {relation}") as metadata:
        for (rowid,) in metadata:
            _check_reference_cancellation()
            owner._amend_original_input_fields(tier, table, rowid, columns)
            with owner._owned_cursor(
                conn, f"SELECT rowid AS physical_rowid,{projection} FROM {relation} WHERE rowid=?", (rowid,)
            ) as cursor:
                row = cursor.fetchone()
            if row is None:
                raise ReferenceSealStaleError("reference row disappeared inside its original pinned snapshot")
            yield row


def _json_strings(conn: sqlite3.Connection, table: str, column: str, rowid: int) -> Iterator[str]:
    """Read a declared native JSON array without transferring its scalar cell."""
    owner = _user_reference_owner(conn)
    field = f"{table}.{column}"
    owner._amend_original_input_fields("user", table, rowid, (column,))
    relation, cell = quote_identifier(table), quote_identifier(column)
    try:
        with owner._owned_cursor(conn, f"SELECT json_type({cell}) FROM {relation} WHERE rowid=?", (rowid,)) as cursor:
            row = cursor.fetchone()
        if row is None or row[0] != "array":
            raise ReferenceSealError(f"durable {field} must be a JSON string array")
        with owner._owned_cursor(
            conn,
            f"SELECT item.type,item.value FROM {relation} AS owner,json_each(owner.{cell}) AS item WHERE owner.rowid=?",
            (rowid,),
        ) as events:
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


@dataclass(frozen=True, slots=True)
class _ReferenceAnchor:
    wire: str
    permits_absence: bool
    assertion_id: str = ""
    field: str = ""
    position: int = -1
    assertion_target: str = ""


def _references_from_user(conn: sqlite3.Connection) -> Generator[_ReferenceAnchor, None, None]:
    with closing(
        _reference_rows(conn, "user", "assertions", ("assertion_id", "kind", "scope_ref", "target_ref", "author_ref"))
    ) as cursor:
        for row in cursor:
            for column in ("scope_ref", "target_ref", "author_ref"):
                value = row[column]
                if value is not None:
                    lifecycle = column == "target_ref" and str(row["kind"]) in {
                        "suppression",
                        "excision_record",
                        "excision_request",
                    }
                    yield _ReferenceAnchor(
                        str(value),
                        lifecycle and str(value).startswith("session:"),
                        str(row["assertion_id"]),
                        column,
                        -1,
                        str(row["target_ref"]),
                    )
            for position, value in enumerate(
                _json_strings(conn, "assertions", "evidence_refs_json", row["physical_rowid"])
            ):
                yield _ReferenceAnchor(
                    value, False, str(row["assertion_id"]), "evidence_refs_json", position, str(row["target_ref"])
                )
    for value in _other_user_references(conn):
        yield _ReferenceAnchor(value, False)


def _other_user_references(conn: sqlite3.Connection) -> Generator[str, None, None]:
    owner = _user_reference_owner(conn)
    with closing(
        _reference_rows(
            conn,
            "user",
            "annotation_batches",
            ("target_ref", "source_result_ref", "actor_ref", "model_ref", "prompt_ref"),
        )
    ) as cursor:
        for row in cursor:
            for column in ("target_ref", "source_result_ref", "actor_ref", "model_ref", "prompt_ref"):
                yield str(row[column])
            yield from _json_strings(conn, "annotation_batches", "assertion_refs_json", row["physical_rowid"])
    with owner._owned_cursor(conn, "SELECT rowid FROM query_evaluation_receipts") as cursor:
        for (rowid,) in cursor:
            yield from _json_strings(conn, "query_evaluation_receipts", "model_refs_json", rowid)
    with closing(_reference_rows(conn, "user", "session_marker_delivery", ("session_id",))) as cursor:
        for row in cursor:
            yield ObjectRef("session", str(row["session_id"])).format()
    with closing(_reference_rows(conn, "user", "result_set_members", ("member_ref",))) as cursor:
        for row in cursor:
            yield str(row["member_ref"])
    with closing(
        _reference_rows(
            conn, "user", "context_deliveries", ("snapshot_ref", "recipient_ref", "run_ref", "delivered_by_ref")
        )
    ) as cursor:
        for row in cursor:
            for column in ("recipient_ref", "run_ref", "delivered_by_ref"):
                value = row[column]
                if value is not None:
                    yield str(value)
            for column in ("evidence_refs_json", "assertion_refs_json"):
                yield from _json_strings(conn, "context_deliveries", column, row["physical_rowid"])
            try:
                from polylogue.storage.sqlite.archive_tiers.context_delivery_write import read_context_delivery

                # The canonical model reader consumes every delivery field.
                # Account these actual original bytes before model hydration;
                # its whole-model native/Pydantic allocation remains explicit.
                columns, _keys = owner._known_tier_table_shape("user", "context_deliveries")
                owner._amend_original_input_fields("user", "context_deliveries", row["physical_rowid"], columns)
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


def _references_from_audit(conn: sqlite3.Connection) -> Generator[_ReferenceAnchor, None, None]:
    for table in ("operation_preview_targets", "operation_targets"):
        with closing(_reference_rows(conn, "audit", table, ("target_ref",))) as cursor:
            for row in cursor:
                _check_reference_cancellation()
                yield _ReferenceAnchor(str(row["target_ref"]), True)


def _index_input_hook(
    conn: sqlite3.Connection,
) -> Callable[[str, tuple[str, ...], str, tuple[object, ...]], None] | None:
    owner = native_sql_parent_for_connection(conn)
    if not isinstance(owner, PreparedIndexMutation) or owner._original_input_demand is None:
        return None
    if owner.observer("index") is not conn:
        raise ReferenceSealError("Index input amendment requires its original pinned observer")
    return owner.before_index_input


def _index_reference_rows(
    conn: sqlite3.Connection, table: str, columns: tuple[str, ...], rowid_sql: str, parameters: tuple[object, ...]
) -> Generator[sqlite3.Row, None, None]:
    before_input = _index_input_hook(conn)
    relation = quote_identifier(table)
    projection = ",".join(quote_identifier(column) for column in columns)
    with connection_cursor(conn, rowid_sql, parameters) as identities:
        for (rowid,) in identities:
            if before_input is not None:
                before_input(table, columns, f"SELECT rowid FROM {relation} WHERE rowid=?", (rowid,))
            with connection_cursor(conn, f"SELECT {projection} FROM {relation} WHERE rowid=?", (rowid,)) as cursor:
                row = cursor.fetchone()
            if row is None:
                raise ReferenceSealStaleError("selected reference input disappeared inside its owned read")
            yield cast(sqlite3.Row, row)


def _block_reference_row(
    conn: sqlite3.Connection, predicate: str, parameters: tuple[object, ...], *, first_only: bool = False
) -> sqlite3.Row | None:
    limit = " LIMIT 1" if first_only else ""
    with connection_cursor(
        conn,
        "SELECT m.rowid,b.rowid FROM messages AS m JOIN blocks AS b ON b.message_id=m.message_id "
        f"WHERE {predicate}{limit}",
        parameters,
    ) as cursor:
        selected = cursor.fetchone()
    if selected is None:
        return None
    before_input = _index_input_hook(conn)
    if before_input is not None:
        before_input("messages", ("session_id",), "SELECT rowid FROM messages WHERE rowid=?", (selected[0],))
        before_input(
            "blocks", ("message_id", "position", "block_id"), "SELECT rowid FROM blocks WHERE rowid=?", (selected[1],)
        )
    with connection_cursor(
        conn,
        "SELECT m.session_id,b.message_id,b.position,b.block_id "
        "FROM messages AS m JOIN blocks AS b ON b.message_id=m.message_id WHERE m.rowid=? AND b.rowid=?",
        (selected[0], selected[1]),
    ) as cursor:
        row = cursor.fetchone()
    if row is None:
        raise ReferenceSealStaleError("selected block reference inputs disappeared inside their owned read")
    return cast(sqlite3.Row, row)


def _resolve_target(conn: sqlite3.Connection, ref: ObjectRef | EvidenceRef | BlockAnchor) -> _ResolvedReference | None:
    if isinstance(ref, BlockAnchor):
        resolution = resolve_block_anchor(conn, ref, before_input=_index_input_hook(conn))
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
            scope_session_id = resolve_session_id_in_index(conn, ref.session_id, before_input=_index_input_hook(conn))
        except (KeyError, ValueError):
            return None
        if ref.message_id is None:
            return _ResolvedReference("session", scope_session_id, scope_session_id)
        if ref.block_index is None:
            with closing(
                _index_reference_rows(
                    conn,
                    "messages",
                    ("session_id",),
                    "SELECT rowid FROM messages WHERE message_id=?",
                    (ref.message_id,),
                )
            ) as rows:
                row = next(rows, None)
            if row is None or _locate_composed_message(conn, scope_session_id, ref.message_id) is None:
                return None
            return _ResolvedReference(
                "message",
                str(row[0]),
                ref.message_id,
                scope_session_id=scope_session_id,
                target_message_id=ref.message_id,
            )
        row = _block_reference_row(conn, "m.message_id=? AND b.position=?", (ref.message_id, ref.block_index))

        if row is None or _locate_composed_message(conn, scope_session_id, ref.message_id) is None:
            return None
        return _ResolvedReference(
            "block-position", str(row[0]), str(row[1]), str(row[2]), scope_session_id, str(row[1])
        )

    if ref.kind == "delegation":
        root = parse_delegation_ancestry_object_id(ref.object_id) or parse_delegation_subtree_object_id(ref.object_id)
        if root is not None:
            with closing(
                _index_reference_rows(
                    conn, "sessions", ("session_id",), "SELECT rowid FROM sessions WHERE session_id=?", (root,)
                )
            ) as rows:
                row = next(rows, None)
            return None if row is None else _ResolvedReference("delegation-root", root, ref.object_id)
        edge = parse_delegation_edge_object_id(ref.object_id)
        operands: tuple[object, ...]
        if edge is None:
            predicate = "instruction_tool_use_block_id=?"
            operands = (ref.object_id,)
        else:
            predicate = (
                "parent_session_id=? AND child_session_id=? "
                "AND mapping_state IN ('edge_only', 'quarantined', 'authority-contradicted')"
            )
            operands = edge
        with closing(
            _index_reference_rows(
                conn,
                "delegation_facts",
                ("parent_session_id", "child_session_id", "instruction_message_id"),
                f"SELECT rowid FROM delegation_facts WHERE {predicate} LIMIT 1",
                operands,
            )
        ) as rows:
            row = next(rows, None)

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
        with closing(
            _index_reference_rows(
                conn,
                "sessions",
                ("session_id",),
                f"{relation} SELECT s.rowid FROM {table} AS selected JOIN sessions AS s ON s.session_id=selected.session_id "
                f"WHERE selected.{column}=?",
                (ref.format(),),
            )
        ) as rows:
            row = next(rows, None)

        return None if row is None else _ResolvedReference(ref.kind, str(row[0]), ref.object_id)
    if ref.kind == "session":
        from polylogue.storage.sqlite.session_identity import resolve_session_id_in_index

        try:
            session_id = resolve_session_id_in_index(conn, ref.object_id, before_input=_index_input_hook(conn))
        except (KeyError, ValueError):
            return None
        return _ResolvedReference("session", session_id, session_id)
    if ref.kind == "message":
        with closing(
            _index_reference_rows(
                conn, "messages", ("session_id",), "SELECT rowid FROM messages WHERE message_id=?", (ref.object_id,)
            )
        ) as rows:
            row = next(rows, None)
        return (
            None
            if row is None
            else _ResolvedReference("message", str(row[0]), ref.object_id, target_message_id=ref.object_id)
        )
    if ref.kind in {"block", "action"}:
        if ref.qualifiers:
            row = _block_reference_row(
                conn,
                "b.block_id=? OR (m.message_id=? AND b.position=?)",
                (ref.object_id, ref.object_id, ref.qualifiers[-1]),
                first_only=True,
            )
        else:
            row = _block_reference_row(conn, "b.block_id=?", (ref.object_id,))

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

    return locate_composed_message(conn, session_id, message_id, before_input=_index_input_hook(conn))


class PreparedIndexMutation:
    """Live observer set and typed reachability captured before writer admission."""

    def __init__(
        self,
        index_path: Path | None,
        *,
        archive_root: Path,
        destination: IndexMutationDestination | None = None,
        input_demand: Callable[[int], None] | None = None,
        _source_only: bool = False,
    ) -> None:
        self._capabilities = frozenset({"source"} if _source_only else {"index", "source", "user", "audit"})
        if _source_only and (index_path is not None or destination is not None):
            raise ReferenceSealError("Source-only preparation cannot inherit an Index destination")
        if not _source_only and index_path is None:
            raise ReferenceSealError("Index preparation requires its actual Index path")
        self._configured_root = archive_root.absolute()
        self.archive_root = archive_root.resolve(strict=True)
        self._index_path = None if index_path is None else index_path.resolve(strict=True)
        self._active_index_path: Path | None = None
        if "index" in self._capabilities:
            from polylogue.storage.archive_identity import resolve_active_index_path

            self._active_index_path = resolve_active_index_path(self.archive_root).resolve(strict=True)
            if destination is not None:
                destination.validate()
                generation = destination.generation
                if (
                    destination.kind != "owned_inactive"
                    or destination.index_path != self.index_path
                    or generation is None
                    or Path(generation.archive_root).resolve(strict=True) != self.archive_root
                ):
                    raise ReferenceSealError("prepared Index destination does not belong to this archive")
            elif self._active_index_path != self.index_path:
                raise ReferenceSealError("active reference seal requires the archive's actual active Index")
        self.destination = destination
        self.index_thread = threading.current_thread()
        self.index_pid = os.getpid()
        self.index_task = _current_task()
        self._index_identity = None if self._index_path is None else _tier_identity(self._index_path)
        # Source-only preparation captures only its actual durable namespace.
        # Missing or unreadable Index/User files confer no additional capability.
        self._configured_paths = {
            name: self._configured_root / f"{name}.db"
            for name in ("source", "user", "audit")
            if name in self._capabilities
        }
        self._namespace_paths = (
            self._configured_root,
            *(
                (self._configured_root / "index.db", self._configured_root / ".index-active-pointer")
                if "index" in self._capabilities
                else ()
            ),
            *self._configured_paths.values(),
        )
        self._namespace = {path: self._namespace_identity(path) for path in self._namespace_paths}
        self._paths: dict[str, Path] = {
            name: path.resolve(strict=True) for name, path in self._configured_paths.items()
        }
        if "index" in self._capabilities:
            self._paths["index"] = self.index_path
        self._assert_configured_namespace()
        self._identities = {name: _tier_identity(path) for name, path in self._paths.items()}
        self._observers: dict[str, sqlite3.Connection] = {}
        self._observer_leaves: dict[str, VerifiedAuditLeaf] = {}
        self._versions: dict[str, int] = {}
        self._original_input_epochs = dict.fromkeys(self._paths, 0)
        self._original_input_demand: Callable[[int], None] | None = input_demand
        self._candidate_path: Path | None = None
        self._candidate_identity: tuple[int, int, int, int] | None = None
        self._candidate_version: int | None = None
        self._candidate_schema: tuple[int, str | None] | None = None
        self.candidate_missing_session_count = 0
        self.candidate_first_missing_session_id: str | None = None
        self._tier_mutation_nonce = object()
        self._original_reads_active = False
        self._live_literal_reader: NativeSQLCustodyOwner | None = None
        self._source_stage_ready = False
        self._source_stage_receipt_accepted = False
        self._source_producer_active = False
        self._source_rows_active = False
        self._source_baseline_active = False
        self._source_statement_active = False
        self._source_capture_internal = False
        self._source_capture_failure: BaseException | None = None
        self._source_capture_tables: dict[str, tuple[tuple[str, ...], int]] = {}
        self._source_statement_table: str | None = None
        self._source_statement_allocation_parameter: int | None = None
        self._source_statement_root_seen = False
        self._source_statement_prepared_cells: dict[str, KnownTierCell] | None = None
        self._source_sequence_seeded: set[str] = set()
        self._pending_tier_permits: dict[str, KnownTierMutationPermit] = {}
        self._pending_tier_receipts: dict[str, KnownTierMutationReceipt] = {}
        self._mutation_custody: ArchiveWriteCustody | None = None
        self._publication_exclusion: ActiveWriterLease | None = None
        self._publication_payload_cleanup: Callable[[], None] | None = None
        self._publication_lifetime_bound = False
        self._closed = False
        self._cleanup_requested = False
        self._session_namespace_noted = False
        self._scratch_directory: tempfile.TemporaryDirectory[str] | None = None
        self._owned_scratch_connection: sqlite3.Connection | None = None
        self._witness_file_identity: tuple[int, int] | None = None
        self._witness_data_version: int | None = None
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
            bind_readonly_incremental_blob_custody(
                self._scratch, scratch_owner.retain_incremental_blob, scratch_owner._require_incremental_read
            )
            self._scratch.set_progress_handler(lambda: int(compute_cancel_requested()), 2000)
            with closing(self._scratch.execute("PRAGMA temp_store=FILE")):
                pass
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
                "CREATE TEMP TABLE reference_anchors("
                "wire_ref TEXT NOT NULL, tier TEXT NOT NULL, assertion_id TEXT NOT NULL, field TEXT NOT NULL, "
                "position INTEGER NOT NULL, assertion_target TEXT NOT NULL, permits_absence INTEGER NOT NULL, "
                "retired INTEGER NOT NULL DEFAULT 0, projected_retired INTEGER NOT NULL DEFAULT 0, "
                "PRIMARY KEY(wire_ref, tier, assertion_id, field, position)) WITHOUT ROWID;"
                "CREATE TEMP TABLE authorized_removals(session_id TEXT PRIMARY KEY) WITHOUT ROWID;"
                "CREATE TEMP TABLE candidate_refs ("
                "kind TEXT NOT NULL, owner_session_id TEXT NOT NULL, object_id TEXT NOT NULL, "
                "qualifier TEXT NOT NULL, scope_session_id TEXT NOT NULL, target_message_id TEXT NOT NULL, "
                "wire_ref TEXT PRIMARY KEY, has_session_alias INTEGER NOT NULL) "
                "WITHOUT ROWID;"
                "CREATE TEMP TABLE original_blob_inputs("
                "blob_hash BLOB NOT NULL,byte_length INTEGER NOT NULL,charged INTEGER NOT NULL DEFAULT 0,prepared_claim_json TEXT,"
                "PRIMARY KEY(blob_hash,byte_length)) WITHOUT ROWID;"
                "CREATE TEMP TABLE original_input_rows("
                "tier TEXT NOT NULL,epoch INTEGER NOT NULL,table_name TEXT NOT NULL,physical_rowid INTEGER NOT NULL,"
                "image_id INTEGER,"
                "PRIMARY KEY(tier,epoch,table_name,physical_rowid)) WITHOUT ROWID;"
                "CREATE TEMP TABLE original_input_fields("
                "tier TEXT NOT NULL,epoch INTEGER NOT NULL,table_name TEXT NOT NULL,physical_rowid INTEGER NOT NULL,"
                "column_name TEXT NOT NULL,byte_length INTEGER NOT NULL,"
                "PRIMARY KEY(tier,epoch,table_name,physical_rowid,column_name)) WITHOUT ROWID;"
                "CREATE TABLE known_tier_row_images("
                "image_id INTEGER PRIMARY KEY, table_name TEXT NOT NULL, columns_blob BLOB NOT NULL, "
                "physical_rowid INTEGER, cell_ids_blob BLOB NOT NULL);"
                "CREATE TABLE known_tier_literal_cells("
                "cell_id INTEGER PRIMARY KEY, storage_class TEXT NOT NULL, byte_length INTEGER NOT NULL, fixed_blob BLOB);"
                "CREATE TABLE known_tier_literals("
                "literal BLOB NOT NULL);"
                "CREATE TEMP VIEW polylogue_source_literals AS SELECT rowid AS cell_id,literal FROM main.known_tier_literals;"
            )
            self._witness_file_identity = _tier_identity(Path(self._scratch_directory.name) / "refs.db")[:2]
            with closing(self._scratch.execute("PRAGMA data_version")) as cursor:
                self._witness_data_version = int(cursor.fetchone()[0])
            for name, path in self._paths.items():
                self._observers[name] = self._open_observer(name, path)
                if self._observer_identity(name) != self._identities[name]:
                    raise ReferenceSealStaleError(f"the {name}.db file changed while opening its observer")
            self._scratch.set_authorizer(self._authorize_witness_main)
            if "index" in self._capabilities:
                self._read_resolved_references()
            else:
                observer = self._observers["source"]
                with closing(observer.execute("PRAGMA data_version")) as cursor:
                    before = int(cursor.fetchone()[0])
                with closing(observer.execute("BEGIN")):
                    pass
                with closing(observer.execute("SELECT 1 FROM sqlite_schema LIMIT 1")) as cursor:
                    cursor.fetchone()
                with closing(observer.execute("COMMIT")):
                    pass
                with closing(observer.execute("PRAGMA data_version")) as cursor:
                    after = int(cursor.fetchone()[0])
                if before != after or self._observer_identity("source") != self._identities["source"]:
                    raise ReferenceSealStaleError("Source changed while preparing its original capability")
                self._versions["source"] = after
        except BaseException as exc:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup("Reference preparation and cleanup failed", [exc, cleanup_error]) from exc
            if compute_cancel_requested():
                raise asyncio.CancelledError("durable-reference preparation cancelled") from exc
            raise

    @classmethod
    def source_only(
        cls, *, archive_root: Path, input_demand: Callable[[int], None] | None = None
    ) -> PreparedIndexMutation:
        """Prepare the existing durable Source capability without opening Index."""
        return cls(None, archive_root=archive_root, input_demand=input_demand, _source_only=True)

    def _require_capability(self, tier: str) -> None:
        if tier not in self._capabilities:
            raise ReferenceSealError(f"original mutation witness has no {tier} capability")

    @property
    def index_path(self) -> Path:
        self._require_capability("index")
        assert self._index_path is not None
        return self._index_path

    @property
    def index_identity(self) -> tuple[int, int, int, int]:
        self._require_capability("index")
        assert self._index_identity is not None
        return self._index_identity

    @property
    def _scratch(self) -> sqlite3.Connection:
        connection = self._owned_scratch_connection
        if connection is None:
            raise ReferenceSealError("reference proof has no live scratch connection")
        return connection

    def _open_observer(self, name: str, path: Path) -> sqlite3.Connection:
        self._assert_configured_namespace()
        leaf = VerifiedAuditLeaf(path.parent, filename=path.name, identity_access="lock-preserving")
        leaf.__enter__()
        self._observer_leaves[name] = leaf
        conn = open_readonly_connection(leaf.anchored_path, validate_schema=False)
        self._observers[name] = conn
        from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner

        NativeSQLCustodyOwner(conn, terminal_parent=self)
        # Constructor and promotion callers settle this parent once on every
        # failure. Closing here would retry a failed child during that unwind.
        leaf.assert_unchanged()
        self._assert_configured_namespace()
        conn.row_factory = sqlite3.Row
        conn.set_progress_handler(lambda: int(compute_cancel_requested()), 2000)
        return conn

    def _close_native_connection(self, connection: sqlite3.Connection) -> None:
        from polylogue.storage.sqlite.connection_profile import close_parent_native_connection

        close_parent_native_connection(self, connection)

    @staticmethod
    def _namespace_identity(path: Path) -> tuple[int, int, int, str | None] | None:
        try:
            before = path.lstat()
        except FileNotFoundError:
            return None
        link = os.readlink(path) if path.is_symlink() else None
        after = path.lstat()
        identity = (before.st_dev, before.st_ino, before.st_mode, link)
        if (after.st_dev, after.st_ino, after.st_mode) != identity[:3]:
            raise ReferenceSealStaleError("configured archive namespace changed during capture")
        return identity

    def _assert_configured_namespace(self) -> None:
        from polylogue.storage.archive_identity import resolve_active_index_path

        for path, identity in self._namespace.items():
            if self._namespace_identity(path) != identity:
                raise ReferenceSealStaleError("configured archive namespace changed after reference preparation")
        for name, path in self._configured_paths.items():
            if path.resolve(strict=True) != self._paths[name]:
                raise ReferenceSealStaleError(f"configured {name}.db target changed after reference preparation")
        if (
            "index" in self._capabilities
            and resolve_active_index_path(self._configured_root).resolve(strict=True) != self._active_index_path
        ):
            raise ReferenceSealStaleError("configured active Index changed after reference preparation")

    def _observer_identity(self, name: str) -> tuple[int, int, int, int]:
        self._require_capability(name)
        self._assert_configured_namespace()
        metadata = self._observer_leaves[name].identity_metadata()
        return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns

    @staticmethod
    def _writer_identity(conn: sqlite3.Connection) -> tuple[int, int, int, int]:
        with closing(conn.execute("PRAGMA database_list")) as cursor:
            row = cursor.fetchall()
        path = next((Path(str(item[2])) for item in row if str(item[1]) == "main"), None)
        if path is None:
            raise ReferenceSealError("the index writer has no main database")
        return _tier_identity(path)

    def _read_resolved_references(self) -> None:
        self._require_capability("index")
        index_observer = self._observers["index"]
        with self._owned_cursor(index_observer, "PRAGMA data_version") as cursor:
            before_index = int(cursor.fetchone()[0])
        with self._owned_cursor(index_observer, "BEGIN"):
            pass
        # Failed preparation retains its snapshots for the parent's one
        # terminal cleanup attempt; successful readonly snapshots commit.
        for name in ("source", "user", "audit"):
            observer = self._observers[name]
            with self._owned_cursor(observer, "PRAGMA data_version") as cursor:
                before = int(cursor.fetchone()[0])
            with self._owned_cursor(observer, "BEGIN"):
                pass
            refs: Iterable[_ReferenceAnchor]
            if name == "user":
                refs = _references_from_user(observer)
            elif name == "audit":
                refs = _references_from_audit(observer)
            else:
                with self._owned_cursor(observer, "SELECT 1 FROM sqlite_schema LIMIT 1") as cursor:
                    cursor.fetchone()
                refs = ()
            primary: BaseException | None = None
            try:
                for anchor in refs:
                    _check_reference_cancellation()
                    parsed = _relevant_ref(anchor.wire)
                    if parsed is not None:
                        target = _resolve(index_observer, parsed)
                        if target is not None:
                            with self._owned_cursor(
                                self._scratch,
                                "INSERT OR IGNORE INTO reference_anchors "
                                "(wire_ref,tier,assertion_id,field,position,assertion_target,permits_absence) "
                                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (
                                    target.wire_ref,
                                    name,
                                    anchor.assertion_id,
                                    anchor.field,
                                    anchor.position,
                                    anchor.assertion_target,
                                    int(anchor.permits_absence),
                                ),
                            ):
                                pass
                            with self._owned_cursor(
                                self._scratch,
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
                            ):
                                pass
            except BaseException as failure:
                primary = failure
                raise
            finally:
                if isinstance(refs, Generator):
                    try:
                        refs.close()
                    except BaseException as cleanup:
                        if primary is not None:
                            raise BaseExceptionGroup(
                                "Reference scan and cursor settlement failed", [primary, cleanup]
                            ) from primary
                        raise
            observer.commit()
            with self._owned_cursor(observer, "PRAGMA data_version") as cursor:
                after = int(cursor.fetchone()[0])
            if before != after:
                raise ReferenceSealStaleError(f"{name}.db changed during reference preparation")
            self._versions[name] = after
        index_observer.commit()
        with self._owned_cursor(index_observer, "PRAGMA data_version") as cursor:
            after_index = int(cursor.fetchone()[0])
        if before_index != after_index:
            raise ReferenceSealStaleError("index.db changed during reference preparation")
        self._versions["index"] = after_index
        self._scratch.commit()

    def note_session_namespace_change(self) -> None:
        self._require_capability("index")
        self._require_new_work()
        if not self._session_namespace_noted:
            with closing(
                self._scratch.execute(
                    "INSERT OR IGNORE INTO candidate_refs SELECT * FROM resolved_refs WHERE has_session_alias = 1"
                )
            ):
                pass
            self._session_namespace_noted = True

    def note_deleted_session(self, session_id: str) -> None:
        self._require_capability("index")
        self._require_new_work()
        self.note_session_namespace_change()
        with closing(
            self._scratch.execute(
                "INSERT OR IGNORE INTO candidate_refs "
                "SELECT kind, owner_session_id, object_id, qualifier, scope_session_id, target_message_id, wire_ref, has_session_alias FROM resolved_refs "
                "WHERE owner_session_id = ? OR scope_session_id = ?",
                (session_id, session_id),
            )
        ):
            pass

    def authorize_session_removal(self, session_ids: tuple[str, ...]) -> None:
        """Retain only exact begun removal targets from this physical apply."""
        self._require_capability("index")
        self._require_new_work()
        from polylogue.storage.sqlite.write_lease import permitted_session_removals

        permitted = permitted_session_removals(archive_root=self.archive_root)
        if not set(session_ids).issubset(permitted):
            raise ReferenceSealError("session disappearance is outside the validated removal plan")
        with closing(
            self._scratch.executemany(
                "INSERT OR IGNORE INTO authorized_removals VALUES (?)", ((sid,) for sid in session_ids)
            )
        ):
            pass

    def _intentional_absence(
        self, conn: sqlite3.Connection, ref: _ResolvedReference, *, projected_user_effects: bool = False
    ) -> bool:
        from polylogue.storage.sqlite.write_lease import permitted_session_removals

        permitted = permitted_session_removals(archive_root=self.archive_root)
        with closing(
            self._scratch.execute(
                "SELECT 1 FROM reference_anchors WHERE wire_ref = ? AND retired = 0 AND permits_absence = 0 "
                "AND (?=0 OR projected_retired=0)",
                (ref.wire_ref, int(projected_user_effects)),
            )
        ) as cursor:
            if cursor.fetchone():
                return False
        with closing(
            self._scratch.execute(
                "SELECT 1 FROM reference_anchors WHERE wire_ref = ? AND retired = 0 AND permits_absence = 1 "
                "AND (?=0 OR projected_retired=0)",
                (ref.wire_ref, int(projected_user_effects)),
            )
        ) as cursor:
            if not cursor.fetchone():
                return False
        for session_id in (ref.owner_session_id, ref.scope_session_id):
            if session_id is None:
                continue
            if session_id not in permitted:
                return False
            with closing(
                self._scratch.execute("SELECT 1 FROM authorized_removals WHERE session_id = ?", (session_id,))
            ) as cursor:
                if not cursor.fetchone():
                    return False
            with closing(conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))) as cursor:
                if cursor.fetchone():
                    return False
        parsed = _relevant_ref(ref.wire_ref)
        return parsed is not None and _resolve_target(conn, parsed) is None

    def note_lineage_change(self, conn: sqlite3.Connection, session_id: str) -> None:
        """Track refs scoped to every composed transcript below a changed node."""
        self._require_capability("index")
        self._require_new_work()
        self.note_session_namespace_change()
        from polylogue.archive.topology.edge import topology_status_composes_sql

        status_predicate = topology_status_composes_sql("l.status")
        with closing(
            conn.execute(
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
        ) as descendants:
            for (affected_session_id,) in descendants:
                _check_reference_cancellation()
                with closing(
                    self._scratch.execute(
                        "INSERT OR IGNORE INTO candidate_refs "
                        "SELECT kind, owner_session_id, object_id, qualifier, scope_session_id, target_message_id, wire_ref, has_session_alias "
                        "FROM resolved_refs WHERE scope_session_id = ? OR (owner_session_id = ? AND "
                        "kind IN ('delegation', 'delegation-root', 'run', 'observed-event', 'context-snapshot'))",
                        (str(affected_session_id), str(affected_session_id)),
                    )
                ):
                    pass
        self._scratch.commit()

    def note_deleted_message_ids(self, message_ids: Iterable[str]) -> None:
        self._require_capability("index")
        self._require_new_work()
        with closing(
            self._scratch.executemany(
                "INSERT OR IGNORE INTO destructive_message_ids VALUES (?)",
                ((message_id,) for message_id in message_ids),
            )
        ):
            pass
        with closing(
            self._scratch.execute(
                "INSERT OR IGNORE INTO candidate_refs "
                "SELECT r.kind, r.owner_session_id, r.object_id, r.qualifier, r.scope_session_id, r.target_message_id, r.wire_ref, r.has_session_alias "
                "FROM resolved_refs AS r JOIN destructive_message_ids AS d "
                "ON d.message_id = r.target_message_id"
            )
        ):
            pass
        with closing(self._scratch.execute("DELETE FROM destructive_message_ids")):
            pass
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
        self._require_capability("index")
        self._require_new_work()
        if self.destination is not None:
            self.destination.validate()
        if conn.in_transaction:
            raise ReferenceSealStaleError("reference seal validation must precede the writer transaction")
        if self._writer_identity(conn) != self.index_identity:
            raise ReferenceSealStaleError("the index file incarnation changed after reference preparation")
        self.validate_observers_current()

    def _require_unpinned_observer(self, name: str) -> sqlite3.Connection:
        self._require_capability(name)
        observer = self._observers[name]
        # in_transaction misses autocommit SELECT snapshots. Do not close a
        # producer's active cursor here: its own read must finish explicitly.
        retained_blobs = any(
            child.connection is observer and child._incremental_blobs for child in native_sql_children(self)
        )
        if observer.in_transaction or live_connection_cursors(observer) or retained_blobs:
            raise ReferenceSealError(f"{name} observer currency requires all original reads to finish")
        return observer

    def observer(self, tier: str) -> sqlite3.Connection:
        """Return one retained readonly observer to its seal-owning worker."""
        self._require_capability(tier)
        self._require_new_work()
        try:
            return self._observers[tier]
        except KeyError as exc:
            raise ValueError(f"unknown reference-seal tier {tier!r}") from exc

    def observer_version(self, tier: str) -> int:
        self._require_capability(tier)
        self._require_new_work()
        try:
            return self._versions[tier]
        except KeyError as exc:
            raise ValueError(f"unknown reference-seal tier {tier!r}") from exc

    @contextmanager
    def original_read_snapshot(
        self,
        *,
        input_demand: Callable[[int], None] | None = None,
        prepaid_blob_inputs: Iterable[tuple[str, bytes, int]] = (),
    ) -> Iterator[None]:
        """Pin preparation reads on these same observers, then prove currency.

        This is a preparation window, never an accepted authority advance.
        Every read cursor and incremental cell handle must settle before the
        snapshots end and fresh same-observer versions can be sampled. An
        exclusive creator may register input_demand before any native input
        hydration; each original coordinate is charged once per accepted epoch.
        """
        self.validate_observers_current()
        if self._original_reads_active:
            raise ReferenceSealError("original preparation snapshots cannot nest")
        opened: list[sqlite3.Connection] = []
        primary: BaseException | None = None
        self._original_reads_active = True
        previous_input_demand = self._original_input_demand
        if input_demand is not None:
            self._original_input_demand = input_demand
        try:
            for name in self._paths:
                observer = self._observers[name]
                with self._owned_cursor(observer, "BEGIN"):
                    pass
                opened.append(observer)
                with self._owned_cursor(observer, "SELECT 1 FROM sqlite_schema LIMIT 1") as cursor:
                    cursor.fetchone()
            # Only the actual append owner has already declared accepted plan
            # payload bytes. Its exact original raw/hash/size operands must
            # match here before those CAS inputs can be marked prepaid.
            for raw_id, expected_hash, expected_size in prepaid_blob_inputs:
                _check_reference_cancellation()
                if self._original_input_demand is None:
                    raise ReferenceSealError("prepaid original inputs require the actual creator demand hook")
                if type(expected_hash) is not bytes or type(expected_size) is not int:
                    raise ReferenceSealError(
                        "prepaid acquisition operands require canonical hash and integer byte size"
                    )
                actual_hash, actual_size = self._original_blob_input(raw_id)
                if actual_hash != expected_hash or actual_size != expected_size:
                    raise ReferenceSealStaleError(
                        "accepted append input differs from its original acquisition descriptor"
                    )
                with self._owned_cursor(
                    self._scratch,
                    "INSERT INTO temp.original_blob_inputs(blob_hash,byte_length,charged) VALUES (?,?,1) "
                    "ON CONFLICT(blob_hash,byte_length) DO UPDATE SET charged=1",
                    (actual_hash, actual_size),
                ):
                    pass
            yield
            for name in self._paths:
                observer = self._observers[name]
                if live_connection_cursors(observer) or any(
                    child.connection is observer and child._incremental_blobs for child in native_sql_children(self)
                ):
                    raise ReferenceSealError(f"{name} preparation must settle its original read children")
        except BaseException as error:
            primary = error
            raise
        finally:
            failures: list[BaseException] = []
            for observer in reversed(opened):
                owner = next(child for child in native_sql_children(self) if child.connection is observer)
                if live_connection_cursors(observer) or owner._incremental_blobs:
                    owner.close_required = True
                    failures.append(
                        NativeConnectionSettlementError(
                            owner, ReferenceSealError("Original read children remain unsettled")
                        )
                    )
                    continue
                observer.set_progress_handler(None, 0)
                try:
                    observer.rollback()
                except BaseException as error:
                    failures.append(error)
                finally:
                    observer.set_progress_handler(lambda: int(compute_cancel_requested()), 2000)
            self._original_reads_active = False
            self._original_input_demand = previous_input_demand
            if failures:
                if primary is not None:
                    failures.insert(0, primary)
                raise BaseExceptionGroup("Original preparation snapshots failed to settle", failures)
        self.validate_observers_current()

    @contextmanager
    def original_rows(self, tier: str, sql: str, parameters: tuple[object, ...] = ()) -> Iterator[sqlite3.Cursor]:
        """Borrow one exact native cursor only within this original read window."""
        self._require_new_work()
        if not self._original_reads_active:
            raise ReferenceSealError("original rows require this seal's pinned preparation window")
        if tier not in self._paths:
            raise ValueError("original preparation rows require a declared archive tier")
        observer = self.observer(tier)
        owner = next(child for child in native_sql_children(self) if child.connection is observer)
        cursor = owner.require_connection().cursor()
        primary: BaseException | None = None
        try:
            cursor.execute(sql, parameters)
            yield cursor
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                close_connection_cursor(observer, cursor)
            except BaseException as cleanup:
                owner.close_required = True
                settlement_failure = (
                    cleanup
                    if primary is None
                    else BaseExceptionGroup("Original read and cursor close failed", [primary, cleanup])
                )
                raise NativeConnectionSettlementError(owner, settlement_failure) from cleanup

    def _retain_fixed_cell(self, metadata: SQLiteLiteralCell) -> KnownTierCell:
        self._require_witness_main_mutable()
        with self._owned_cursor(
            self._scratch,
            "INSERT INTO known_tier_literal_cells(storage_class,byte_length,fixed_blob) VALUES (?,?,?)",
            (metadata.storage_class, metadata.byte_length, metadata.fixed_bytes()),
        ) as cursor:
            cell_id = cursor.lastrowid
        assert cell_id is not None
        return KnownTierCell(self, cell_id)

    @contextmanager
    def _writable_literal_blob(self, cell_id: int) -> Iterator[sqlite3.Blob]:
        """Only preparation writes this original witness's exact literal slot."""
        self._require_witness_main_mutable()
        owner = next(child for child in native_sql_children(self) if child.connection is self._scratch)
        connection = owner.require_connection()
        # The measured public connection permits readonly handles only. This
        # exact scratch-only slot is the sole native writable entry point.
        blob = sqlite3.Connection.blobopen(connection, "known_tier_literals", "literal", cell_id, readonly=False)
        owner.retain_incremental_blob(blob)
        primary: BaseException | None = None
        try:
            yield blob
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                owner.close_incremental_blob(blob)
            except BaseException as cleanup:
                owner.close_required = True
                try:
                    self._scratch.set_authorizer(self._authorize_witness_main)
                except BaseException as authorization_failure:
                    cleanup = BaseExceptionGroup(
                        "Literal Blob close and mutation exclusion failed", [cleanup, authorization_failure]
                    )
                settlement_failure = (
                    cleanup
                    if primary is None
                    else BaseExceptionGroup("Literal preparation and native Blob close failed", [primary, cleanup])
                )
                raise NativeConnectionSettlementError(owner, settlement_failure) from cleanup

    def _retain_variable_cell(self, metadata: SQLiteLiteralCell, chunks: Iterable[bytes]) -> KnownTierCell:
        self._require_witness_main_mutable()
        with self._owned_cursor(
            self._scratch,
            "INSERT INTO known_tier_literal_cells(storage_class,byte_length) VALUES (?,?)",
            (metadata.storage_class, metadata.byte_length),
        ) as cursor:
            cell_id = cursor.lastrowid
        assert cell_id is not None
        with self._owned_cursor(
            self._scratch,
            "INSERT INTO known_tier_literals(rowid,literal) VALUES (?,zeroblob(?))",
            (cell_id, metadata.byte_length),
        ):
            pass
        offset = 0
        with self._writable_literal_blob(cell_id) as blob:
            for chunk in chunks:
                _check_reference_cancellation()
                if not isinstance(chunk, bytes):
                    raise TypeError("native literal preparation requires exact byte chunks")
                for start in range(0, len(chunk), LITERAL_CHUNK_BYTES):
                    _check_reference_cancellation()
                    part = chunk[start : start + LITERAL_CHUNK_BYTES]
                    if offset + len(part) > metadata.byte_length:
                        raise ReferenceSealError("retained literal exceeds its declared byte length")
                    blob.write(part)
                    offset += len(part)
            if offset != metadata.byte_length:
                raise ReferenceSealError("retained literal does not match its exact declared byte length")
        return KnownTierCell(self, cell_id)

    def retain_literal_stream(
        self, storage_class: Literal["text", "blob"], byte_length: int, chunks: Iterable[bytes]
    ) -> KnownTierCell:
        """Retain exact prepared bytes on this original witness, without a cell copy in Python."""
        self._require_new_work()
        if byte_length < 0:
            raise ValueError("literal byte length must be nonnegative")
        metadata = literal_metadata(storage_class, None, byte_length)

        def bounded_chunks() -> Iterator[bytes]:
            for chunk in chunks:
                _check_reference_cancellation()
                if not isinstance(chunk, bytes):
                    raise TypeError("literal preparation requires exact byte chunks")
                for offset in range(0, len(chunk), LITERAL_CHUNK_BYTES):
                    yield chunk[offset : offset + LITERAL_CHUNK_BYTES]

        return self._retain_variable_cell(metadata, bounded_chunks())

    def retain_literal_scalar(self, value: None | int | float | str | bytes) -> KnownTierCell:
        """Retain an already-owned scalar; native large-cell producers use their literal stream."""
        self._require_new_work()
        if value is None:
            return self._retain_fixed_cell(literal_metadata("null", None, None))
        if isinstance(value, int):
            return self._retain_fixed_cell(literal_metadata("integer", value, None))
        if isinstance(value, float):
            return self._retain_fixed_cell(literal_metadata("real", value, None))
        if isinstance(value, bytes):
            return self.retain_literal_stream("blob", len(value), (value,))
        if isinstance(value, str):
            # Encoding each slice bounds temporary bytes. The caller already
            # owns this scalar; this is not a reader for a native large cell.
            chunks = (
                value[offset : offset + LITERAL_CHUNK_BYTES].encode("utf-8")
                for offset in range(0, len(value), LITERAL_CHUNK_BYTES)
            )
            metadata = literal_metadata("text", None, 0)
            # UTF-8 length is counted in a first bounded pass, preserving the
            # exact encoding without retaining a second complete byte string.
            length = 0
            for offset in range(0, len(value), LITERAL_CHUNK_BYTES):
                _check_reference_cancellation()
                length += len(value[offset : offset + LITERAL_CHUNK_BYTES].encode("utf-8"))
            metadata = replace(metadata, byte_length=length)
            return self._retain_variable_cell(metadata, chunks)
        raise TypeError("prepared literal must have a SQLite storage class")

    def overlay_tier_row(
        self, image: KnownTierRowImage, changes: dict[str, KnownTierCell], *, rowid: int | None
    ) -> KnownTierRowImage:
        """Reuse immutable original cells and replace only the exact prepared fields."""
        self._require_new_work()
        if image._seal is not self or len(image.cells) != len(image.columns):
            raise ReferenceSealError("row overlay requires this exact original complete image")
        if any(column not in image.columns for column in changes):
            raise ReferenceSealError("row overlay changes a column outside its canonical table")
        for cell in changes.values():
            self._literal_cell_metadata(cell)
        cells = tuple(changes.get(column, cell) for column, cell in zip(image.columns, image.cells, strict=True))
        return KnownTierRowImage(self, image.table, image.columns, rowid, cells)

    @contextmanager
    def source_producer(self, *, require_empty_schedule: bool = False) -> Iterator[None]:
        """Borrow finite canonical Source state in this original read window."""
        self._require_new_work()
        self._require_witness_main_mutable()
        if not self._original_reads_active or self._source_producer_active:
            raise ReferenceSealError("Source producer requires one pinned original preparation owner")
        if "source" in self._pending_tier_permits:
            raise ReferenceSealError("Source producer cannot alter an already prepared exact permit")
        self._provision_source_stage()
        if self._source_stage_receipt_accepted:
            self._rebase_accepted_source_stage()
        if require_empty_schedule:
            with self._owned_cursor(self._scratch, "SELECT 1 FROM temp.known_tier_statements LIMIT 1") as cursor:
                if cursor.fetchone() is not None:
                    raise ReferenceSealError("reservation preparation requires the first original Source unit")
        self._source_producer_active = True
        try:
            yield
            self._load_source_only_normalization_inputs()
            owner = next(child for child in native_sql_children(self) if child.connection is self._scratch)
            if self._source_statement_active or live_connection_cursors(self._scratch) or owner._incremental_blobs:
                owner.close_required = True
                raise NativeConnectionSettlementError(
                    owner, ReferenceSealError("Source producer left actual native child handles unsettled")
                )
            self._settle_witness_metadata()
        except BaseException:
            # A failed preparation never emits a permit or acknowledges a
            # source cursor. Original literal/native custody remains retained.
            self._cleanup_requested = True
            raise
        finally:
            self._source_producer_active = False

    def _rebase_accepted_source_stage(self) -> None:
        """Reuse selected state only after original authoritative advancement."""
        self._require_witness_main_mutable()
        if not self._source_stage_receipt_accepted or "source" in self._pending_tier_permits:
            raise ReferenceSealError("Source stage cannot discard a pending or uncertain original tape")
        for table in ("known_tier_effects", "known_tier_statements"):
            predicate = "tier='source' AND " if table == "known_tier_effects" else ""
            with self._owned_cursor(
                self._scratch, f"SELECT 1 FROM temp.{table} WHERE {predicate}consumed!=1 LIMIT 1"
            ) as cursor:
                if cursor.fetchone() is not None:
                    raise ReferenceSealError("Source stage contains unconsumed original publication evidence")
        # Successful physical settlement already released the original MAIN
        # attachment gate. Accepted complete postimages become the same
        # selected stage's inputs; literal descriptors retain their owner.
        with self._owned_cursor(
            self._scratch, "DELETE FROM temp.polylogue_source_stage_rows WHERE current_image IS NULL"
        ):
            pass
        with self._owned_cursor(
            self._scratch, "UPDATE temp.polylogue_source_stage_rows SET input_image=current_image,touched=0"
        ):
            pass
        for statement in (
            "DELETE FROM temp.known_tier_statement_bindings",
            "DELETE FROM temp.known_tier_statements",
            "DELETE FROM temp.known_tier_effects WHERE tier='source'",
            "DELETE FROM temp.known_tier_effect_tables WHERE tier='source'",
        ):
            with self._owned_cursor(self._scratch, statement):
                pass
        self._source_stage_receipt_accepted = False

    def load_source_row(self, image: KnownTierRowImage) -> bool:
        if not self._source_producer_active:
            raise ReferenceSealError("Source row loading requires its original producer context")
        return self._load_source_row(image)

    def retain_source_row(self, table: str, rowid: int) -> KnownTierRowImage | None:
        """Retain a selected current stage row after its actual read closes."""
        self._require_new_work()
        if not self._source_producer_active or self._source_rows_active or self._source_statement_active:
            raise ReferenceSealError("selected Source cells require its idle original producer owner")
        columns, _keys = self._known_tier_table_shape("source", table)
        with self._owned_cursor(
            self._scratch,
            "SELECT coalesce(current_image,input_image),touched FROM temp.polylogue_source_stage_rows WHERE table_name=? AND physical_rowid=?",
            (table, rowid),
        ) as cursor:
            input_row = cursor.fetchone()
        reuse = None
        if input_row is not None and input_row[0] is not None:
            reuse = self._retained_row_image(input_row[0])
        return self._retain_native_row(self._scratch, table, columns, rowid, reuse=reuse)

    def source_allocation_dependencies(self, table: str) -> None:
        if not self._source_producer_active:
            raise ReferenceSealError("Source allocation loading requires its original producer context")
        self._source_allocation_dependencies(table)

    def source_row_is_touched(self, table: str, rowid: int) -> bool:
        self._require_new_work()
        if not self._source_producer_active:
            raise ReferenceSealError("Source predicate merging requires its original producer context")
        with self._owned_cursor(
            self._scratch,
            "SELECT touched FROM temp.polylogue_source_stage_rows WHERE table_name=? AND physical_rowid=?",
            (table, rowid),
        ) as cursor:
            row = cursor.fetchone()
        return row is not None and bool(row[0])

    def _source_sequence_images(self) -> tuple[KnownTierRowImage, ...]:
        """The two canonical allocator entries are schema-sized native inputs."""
        with self._owned_cursor(
            self._scratch,
            "SELECT 1 FROM sqlite_sequence WHERE typeof(name)!='text' OR typeof(seq)!='integer' OR name NOT IN "
            "(SELECT name FROM sqlite_schema WHERE type='table' AND instr(upper(sql),'AUTOINCREMENT')>0) LIMIT 1",
        ) as cursor:
            if cursor.fetchone() is not None:
                raise ReferenceSealError("Source allocator metadata is outside canonical managed table identities")
        with self._owned_cursor(
            self._scratch, "SELECT 1 FROM sqlite_sequence GROUP BY name HAVING count(*)!=1 LIMIT 1"
        ) as cursor:
            if cursor.fetchone() is not None:
                raise ReferenceSealError("Source allocator entries lack canonical table uniqueness")
        with self._owned_cursor(self._scratch, "SELECT rowid FROM sqlite_sequence ORDER BY rowid") as cursor:
            rowids = tuple(row[0] for row in cursor)
        columns, _keys = self._known_tier_table_shape("source", "sqlite_sequence")
        images = []
        for rowid in rowids:
            _check_reference_cancellation()
            image = self._retain_native_row(self._scratch, "sqlite_sequence", columns, rowid)
            if image is None:
                raise ReferenceSealError("Source allocator input disappeared during original preparation")
            images.append(image)
        return tuple(images)

    def _capture_source_sequence_changes(self, before: tuple[KnownTierRowImage, ...]) -> None:
        after = self._source_sequence_images()
        # These dictionaries contain only the canonical schema's two internal
        # allocator entries, independent of the producer's selected cohort.
        old_rows: dict[int, KnownTierRowImage] = {}
        new_rows: dict[int, KnownTierRowImage] = {}
        for images, destination in ((before, old_rows), (after, new_rows)):
            for image in images:
                if image.rowid is None:
                    raise ReferenceSealError("canonical Source allocator metadata lacks its physical rowid")
                destination[image.rowid] = image
        columns, keys = self._known_tier_table_shape("source", "sqlite_sequence")
        for rowid in sorted(old_rows.keys() | new_rows.keys()):
            old, new = old_rows.get(rowid), new_rows.get(rowid)
            if self._row_images_equal(old, new):
                continue
            if new is None:
                raise ReferenceSealError("canonical Source producer removed internal allocator metadata")
            with self._owned_cursor(
                self._scratch,
                "INSERT OR IGNORE INTO temp.known_tier_effect_tables VALUES(?,?,?,?)",
                ("source", "sqlite_sequence", pickle.dumps(columns, protocol=5), pickle.dumps(keys, protocol=5)),
            ):
                pass
            old_id = None if old is None else self._retain_row_image(old)
            new_id = self._retain_row_image(new)
            with self._owned_cursor(
                self._scratch, "SELECT coalesce(max(ordinal),0)+1 FROM temp.known_tier_effects WHERE tier='source'"
            ) as cursor:
                ordinal = cursor.fetchone()[0]
            with self._owned_cursor(
                self._scratch,
                "INSERT INTO temp.known_tier_effects("
                "tier,ordinal,table_name,old_image,new_image,old_rowid,new_rowid,consumed) "
                "VALUES('source',?,'sqlite_sequence',?,?,?,?, -1)",
                (ordinal, old_id, new_id, None if old is None else old.rowid, new.rowid),
            ):
                pass
            with self._owned_cursor(
                self._scratch,
                "INSERT INTO temp.polylogue_source_stage_rows("
                "table_name,physical_rowid,current_image,touched) VALUES('sqlite_sequence',?,?,1) "
                "ON CONFLICT(table_name,physical_rowid) DO UPDATE SET current_image=excluded.current_image,touched=1",
                (new.rowid, new_id),
            ):
                pass

    @contextmanager
    def source_statement(
        self,
        sql: str,
        parameters: tuple[object, ...] = (),
        *,
        table: str,
        writable_targets: Iterable[tuple[str, tuple[KnownTierCell, ...]]],
        prepared_cells: dict[str, KnownTierCell] | None = None,
        allocation_parameter: int | None = None,
    ) -> Iterator[sqlite3.Cursor]:
        """Execute the actual finite producer and retain ordered physical effects."""
        self._require_new_work()
        self._require_witness_main_mutable()
        if not self._source_producer_active or self._source_rows_active or self._source_statement_active:
            raise ReferenceSealError("Source statement requires its idle original producer context")
        columns, keys = self._known_tier_table_shape("source", table)
        # Retention is validated before executing any producer effects. The
        # canonical builder spells literal-slot expressions explicitly.
        if any(value is not None and type(value) not in (bool, int, float) for value in parameters):
            raise ReferenceSealError(
                "Source statement bindings require fixed native values or original literal-slot IDs"
            )
        if allocation_parameter is not None:
            if (
                type(allocation_parameter) is not int
                or not 0 <= allocation_parameter < len(parameters)
                or parameters[allocation_parameter] is not None
            ):
                raise ReferenceSealError("Source allocation requires its declared NULL binding operand")
            self._physical_rowid_alias(self._scratch, table, columns)
        binding_cells = tuple(self.retain_literal_scalar(cast(None | int | float, value)) for value in parameters)
        if prepared_cells is not None:
            for column, cell in prepared_cells.items():
                if column not in columns:
                    raise ReferenceSealError("prepared Source literal names a noncanonical column")
                self._literal_cell_metadata(cell)
        # Target generators may read and close selected pages before yielding
        # each immutable primary key. No caller cursor remains during DML.
        self._declare_source_writable_keys(writable_targets)
        self._provision_source_capture()
        self._source_allocation_dependencies(table)
        if table == "raw_sessions":
            self._source_allocation_dependencies("raw_existence_changes")
        owner = next(child for child in native_sql_children(self) if child.connection is self._scratch)
        connection = owner.require_connection()
        while True:
            _check_reference_cancellation()
            # This original native savepoint covers MAIN literal cells and
            # TEMP capture/schedule metadata together. Hydrated collision
            # inputs are retained only after the complete attempt rolls back.
            with self._owned_cursor(connection, "SAVEPOINT polylogue_source_allocation_attempt"):
                pass
            try:
                with self._source_statement_attempt(
                    sql,
                    parameters,
                    table=table,
                    columns=columns,
                    keys=keys,
                    prepared_cells=prepared_cells,
                    allocation_parameter=allocation_parameter,
                    binding_cells=binding_cells,
                ) as cursor:
                    yield cursor
            except BaseException as failure:
                if live_connection_cursors(connection) or owner._incremental_blobs or owner.close_required:
                    self._cleanup_requested = True
                    raise
                try:
                    with self._owned_cursor(connection, "ROLLBACK TO polylogue_source_allocation_attempt"):
                        pass
                    with self._owned_cursor(connection, "RELEASE polylogue_source_allocation_attempt"):
                        pass
                except BaseException as cleanup:
                    owner.close_required = True
                    self._cleanup_requested = True
                    raise NativeConnectionSettlementError(
                        owner,
                        BaseExceptionGroup(
                            "Source statement and original allocation rollback failed", [failure, cleanup]
                        ),
                    ) from cleanup
                if not isinstance(failure, _SourceAllocationCollisionError):
                    self._cleanup_requested = True
                    raise
                # No statement has been delivered to the caller on this
                # private exception. Reserve the actual original occupied
                # row, including its exact FK inputs, without fake effects.
                _check_reference_cancellation()
                image = self.retain_tier_row("source", failure.table, failure.rowid)
                if image is None or not self.load_source_row(image):
                    self._cleanup_requested = True
                    raise ReferenceSealStaleError("original allocation collision input cannot be retained") from failure
                continue
            else:
                try:
                    with self._owned_cursor(connection, "RELEASE polylogue_source_allocation_attempt"):
                        pass
                except BaseException as cleanup:
                    owner.close_required = True
                    self._cleanup_requested = True
                    raise NativeConnectionSettlementError(owner, cleanup) from cleanup
                return

    @contextmanager
    def _source_statement_attempt(
        self,
        sql: str,
        parameters: tuple[object, ...],
        *,
        table: str,
        columns: tuple[str, ...],
        keys: tuple[int, ...],
        prepared_cells: dict[str, KnownTierCell] | None,
        allocation_parameter: int | None,
        binding_cells: tuple[KnownTierCell, ...],
    ) -> Iterator[sqlite3.Cursor]:
        before_sequence = self._source_sequence_images()
        with self._owned_cursor(
            self._scratch, "SELECT coalesce(max(ordinal),0) FROM temp.known_tier_effects WHERE tier='source'"
        ) as ordinal_cursor:
            first_ordinal = ordinal_cursor.fetchone()[0] + 1
        with self._owned_cursor(
            self._scratch,
            "INSERT OR IGNORE INTO temp.known_tier_effect_tables VALUES(?,?,?,?)",
            ("source", table, pickle.dumps(columns, protocol=5), pickle.dumps(keys, protocol=5)),
        ):
            pass
        owner = next(child for child in native_sql_children(self) if child.connection is self._scratch)
        connection = owner.require_connection()
        if live_connection_cursors(connection) or owner._incremental_blobs:
            raise ReferenceSealError("Source statement requires settled original selected readers")
        if not connection.getconfig(sqlite3.SQLITE_DBCONFIG_ENABLE_TRIGGER):
            raise ReferenceSealError("Source producer requires restored canonical main triggers")
        for pragma in ("foreign_keys", "recursive_triggers"):
            with self._owned_cursor(connection, f"PRAGMA {pragma}") as enforcement_cursor:
                if enforcement_cursor.fetchone()[0] != 1:
                    raise ReferenceSealError("Source producer requires immutable canonical enforcement settings")
        self._source_capture_failure = None
        self._source_statement_root_seen = False
        self._source_statement_table = table
        self._source_statement_allocation_parameter = allocation_parameter
        self._source_statement_prepared_cells = None if prepared_cells is None else dict(prepared_cells)
        self._source_statement_active = True
        cursor: sqlite3.Cursor | None = None
        primary: BaseException | None = None
        try:
            connection.set_authorizer(self._authorize_witness_main)
            # Retain the actual cursor before executing. A callback exception
            # can keep its execute traceback alive; an unassigned connection
            # convenience cursor would prevent the collision savepoint from
            # settling and retrying on the same original owner.
            cursor = connection.cursor()
            cursor.row_factory = sqlite3.Row
            cursor.execute(sql, parameters)
            if not self._source_statement_root_seen:
                raise ReferenceSealError("Source producer statement omitted its declared canonical root mutation")
            yield cursor
        except BaseException as failure:
            primary = self._source_capture_primary(failure)
            if not isinstance(primary, _SourceAllocationCollisionError):
                self._cleanup_requested = True
            if primary is not failure:
                raise primary from failure
            raise
        finally:
            cleanup_failures: list[BaseException] = []
            try:
                if cursor is not None:
                    close_connection_cursor(connection, cursor)
            except BaseException as cleanup:
                cleanup_failures.append(cleanup)
            finally:
                self._source_statement_active = False
                self._source_statement_table = None
                self._source_statement_allocation_parameter = None
                self._source_statement_prepared_cells = None
                # Expire the precisely scoped statement authorizations, even
                # after cancellation. Original physical children remain owned.
                try:
                    connection.set_authorizer(self._authorize_witness_main)
                except BaseException as cleanup:
                    cleanup_failures.append(cleanup)
            if cleanup_failures:
                owner.close_required = True
                failures = cleanup_failures if primary is None else [primary, *cleanup_failures]
                settlement_failure = (
                    failures[0]
                    if len(failures) == 1
                    else BaseExceptionGroup("Source producer and physical statement settlement failed", failures)
                )
                raise NativeConnectionSettlementError(owner, settlement_failure) from cleanup_failures[0]
        try:
            with self._owned_cursor(
                self._scratch, "SELECT 1 FROM temp.known_tier_effects WHERE tier='source' AND consumed=-2 LIMIT 1"
            ) as cursor:
                if cursor.fetchone() is not None:
                    raise ReferenceSealError("Source statement left incomplete actual physical effects")
            self._capture_source_sequence_changes(before_sequence)
            self._retain_source_statement(sql, table, binding_cells, allocation_parameter, first_ordinal)
        except BaseException:
            self._cleanup_requested = True
            raise

    def _source_capture_primary(self, failure: BaseException) -> BaseException:
        # Capture can set this field during the native execute call; its exact
        # original exception remains the authority over SQLite's wrapper.
        return failure if self._source_capture_failure is None else self._source_capture_failure

    def _retain_source_statement(
        self,
        sql: str,
        table: str,
        bindings: tuple[KnownTierCell, ...],
        allocation_parameter: int | None,
        first_ordinal: int,
    ) -> None:
        """Retain the canonical producer's exact SQL and ordered fixed operands."""
        self._require_witness_main_mutable()
        with self._owned_cursor(
            self._scratch, "SELECT coalesce(max(ordinal),0) FROM temp.known_tier_effects WHERE tier='source'"
        ) as cursor:
            last_ordinal = cursor.fetchone()[0]
        if allocation_parameter is not None:
            with self._owned_cursor(
                self._scratch,
                "SELECT new_rowid FROM temp.known_tier_effects WHERE tier='source' "
                "AND ordinal BETWEEN ? AND ? AND table_name=? AND old_image IS NULL "
                "AND new_image IS NOT NULL AND parent_effect_id IS NULL AND canonical_trigger IS NULL "
                "ORDER BY ordinal LIMIT 2",
                (first_ordinal, last_ordinal, table),
            ) as cursor:
                allocated = cursor.fetchone()
                if cursor.fetchone() is not None:
                    raise ReferenceSealError("one declared allocation operand cannot describe multiple root inserts")
            if allocated is not None:
                if allocated[0] is None:
                    raise ReferenceSealError("declared rowid allocation has no captured physical identity")
                changed = list(bindings)
                changed[allocation_parameter] = self.retain_literal_scalar(allocated[0])
                bindings = tuple(changed)
        with self._owned_cursor(
            self._scratch,
            "INSERT INTO temp.known_tier_statements(sql,root_table,allocation_parameter,first_ordinal,last_ordinal) "
            "VALUES(?,?,?,?,?)",
            (sql, table, allocation_parameter, first_ordinal, last_ordinal),
        ) as cursor:
            statement_id = cursor.lastrowid
        for position, cell in enumerate(bindings):
            _check_reference_cancellation()
            self._literal_cell_metadata(cell)
            with self._owned_cursor(
                self._scratch,
                "INSERT INTO temp.known_tier_statement_bindings VALUES(?,?,?)",
                (statement_id, position, cell._cell_id),
            ):
                pass

    def _source_statement_bindings(self, statement_id: int) -> tuple[object, ...]:
        """Read only the finite canonical statement's fixed-width operands."""
        values: list[object] = []
        with self._owned_cursor(
            self._scratch,
            "SELECT position,cell_id FROM temp.known_tier_statement_bindings WHERE statement_id=? ORDER BY position",
            (statement_id,),
        ) as cursor:
            for position, cell_id in cursor:
                _check_reference_cancellation()
                if position != len(values):
                    raise ReferenceSealError("retained Source statement bindings lost their exact order")
                kind, _size, fixed = self._literal_cell_metadata(KnownTierCell(self, cell_id))
                if kind == "null":
                    values.append(None)
                elif kind == "integer" and fixed is not None:
                    values.append(int.from_bytes(fixed, "big", signed=True))
                elif kind == "real" and fixed is not None:
                    values.append(struct.unpack(">d", fixed)[0])
                else:
                    raise ReferenceSealError("retained statement operand is not its declared fixed native binding")
        return tuple(values)

    def _declare_source_writable_keys(self, targets: Iterable[tuple[str, tuple[KnownTierCell, ...]]]) -> None:
        """Retain exact per-statement roles separately from readable hydration."""
        if not self._source_producer_active or self._source_statement_active or self._source_rows_active:
            raise ReferenceSealError("Source target declaration requires its idle original producer")
        with self._owned_cursor(self._scratch, "DELETE FROM temp.polylogue_source_writable_keys"):
            pass
        for table, cells in targets:
            _check_reference_cancellation()
            if table in {"sqlite_sequence", "raw_existence_journal_control"}:
                raise ReferenceSealError("canonical implicit Source effects cannot be direct writable targets")
            _columns, keys = self._known_tier_table_shape("source", table)
            if len(cells) != len(keys):
                raise ReferenceSealError("Source writable target omits its canonical primary key")
            digest = self._literal_cells_digest(cells)
            # Only schema-sized immutable locator tuples enter this record;
            # complete key values remain in the original literal owner.
            with self._owned_cursor(
                self._scratch,
                "INSERT INTO temp.polylogue_source_writable_keys(table_name,key_digest,key_cells) VALUES(?,?,?)",
                (table, digest, pickle.dumps(tuple(cell._cell_id for cell in cells), protocol=5)),
            ):
                pass

    def _source_image_has_writable_role(self, image: KnownTierRowImage) -> bool:
        _columns, keys = self._known_tier_table_shape("source", image.table)
        digest = self._literal_key_digest(image, keys)
        with self._owned_cursor(
            self._scratch,
            "SELECT key_cells FROM temp.polylogue_source_writable_keys WHERE table_name=? AND key_digest=?",
            (image.table, digest),
        ) as cursor:
            for row in cursor:
                _check_reference_cancellation()
                cells = tuple(KnownTierCell(self, cell_id) for cell_id in pickle.loads(row[0]))
                if len(cells) != len(keys):
                    raise ReferenceSealError("retained writable key has no canonical column shape")
                if all(
                    self._literal_cells_equal(image.cells[position], cell)
                    for position, cell in zip(keys, cells, strict=True)
                ):
                    return True
        return False

    @contextmanager
    def _owned_cursor(
        self,
        connection: sqlite3.Connection,
        sql: str,
        parameters: tuple[object, ...] = (),
    ) -> Iterator[sqlite3.Cursor]:
        """Retain each finite statement before execution on its original owner."""
        cursor = connection.cursor()
        primary: BaseException | None = None
        try:
            cursor.execute(sql, parameters)
            yield cursor
        except BaseException as failure:
            primary = failure
            raise
        finally:
            try:
                close_connection_cursor(connection, cursor)
            except BaseException as cleanup:
                owner = next(child for child in native_sql_children(self) if child.connection is connection)
                owner.close_required = True
                self._cleanup_requested = True
                settlement_failure = (
                    cleanup
                    if primary is None
                    else BaseExceptionGroup(
                        "Original statement and physical cursor settlement failed", [primary, cleanup]
                    )
                )
                raise NativeConnectionSettlementError(owner, settlement_failure) from cleanup

    @contextmanager
    def source_rows(self, sql: str, parameters: tuple[object, ...] = ()) -> Iterator[sqlite3.Cursor]:
        """Borrow selected native reads with original scratch-owner settlement."""
        self._require_new_work()
        if (
            not self._source_producer_active
            or self._source_statement_active
            or self._source_baseline_active
            or self._source_rows_active
        ):
            raise ReferenceSealError("selected Source read requires its idle original producer context")
        owner = next(child for child in native_sql_children(self) if child.connection is self._scratch)
        connection = owner.require_connection()
        self._source_rows_active = True
        cursor: sqlite3.Cursor | None = None
        primary: BaseException | None = None
        try:
            # Expire cached statements before this precise readonly phase.
            connection.set_authorizer(self._authorize_witness_main)
            cursor = connection.cursor()
            cursor.row_factory = sqlite3.Row
            cursor.execute(sql, parameters)
            yield cursor
        except BaseException as failure:
            primary = failure
            raise
        finally:
            try:
                if cursor is not None:
                    close_connection_cursor(connection, cursor)
            except BaseException as cleanup:
                owner.close_required = True
                settlement_failure = (
                    cleanup
                    if primary is None
                    else BaseExceptionGroup(
                        "Selected Source read and native cursor settlement failed", [primary, cleanup]
                    )
                )
                raise NativeConnectionSettlementError(owner, settlement_failure) from cleanup
            finally:
                self._source_rows_active = False

    def _provision_source_stage(self) -> None:
        """Provision canonical Source DDL on this same private native witness."""
        from polylogue.storage.sqlite.archive_tiers.bootstrap import archive_tier_spec
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
        from polylogue.storage.sqlite.migration_runner import _iter_migration_statements

        self._require_new_work()
        self._require_witness_main_mutable()
        if self._source_stage_ready:
            return
        if not self._original_reads_active:
            raise ReferenceSealError("Source provisioning requires its pinned original preparation window")
        # Earlier literal/descriptor preparation may have opened this owner's
        # metadata transaction. Settle it explicitly before profile settings;
        # caller-owned native reads are refused by the settlement check.
        self._settle_witness_metadata()
        ddl = archive_tier_spec(ArchiveTier.SOURCE).ddl
        # This fixed production DDL uses ordinary ASCII object identifiers.
        # Check even IF NOT EXISTS names before SQLite could silently borrow
        # one of the original witness's unrelated metadata relations.
        names = {
            match.group(1).casefold()
            for match in re.finditer(
                r"CREATE\s+(?:UNIQUE\s+)?(?:TABLE|INDEX|TRIGGER|VIEW)\s+"
                r"(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][A-Za-z_0-9]*)",
                ddl,
                re.IGNORECASE,
            )
        }
        with self._owned_cursor(self._scratch, "SELECT name FROM main.sqlite_schema") as cursor:
            existing = {str(row[0]).casefold() for row in cursor}
        if names & existing:
            raise ReferenceSealError("canonical Source objects collide with the original witness schema")
        try:
            for statement in ("PRAGMA foreign_keys=ON", "PRAGMA recursive_triggers=ON"):
                with self._owned_cursor(self._scratch, statement):
                    pass
            with self._owned_cursor(self._scratch, "PRAGMA foreign_keys") as cursor:
                foreign_keys = cursor.fetchone()[0]
            if foreign_keys != 1:
                raise ReferenceSealError("selected Source state requires actual foreign-key enforcement")
            for statement in _iter_migration_statements(ddl):
                _check_reference_cancellation()
                with self._owned_cursor(self._scratch, statement):
                    pass
            with self._owned_cursor(
                self._scratch,
                "CREATE TEMP TABLE polylogue_source_stage_rows("
                "table_name TEXT NOT NULL,physical_rowid INTEGER NOT NULL,input_image INTEGER,"
                "touched INTEGER NOT NULL DEFAULT 0,load_state INTEGER NOT NULL DEFAULT 2,current_image INTEGER,"
                "load_order INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(table_name,physical_rowid)) WITHOUT ROWID",
            ):
                pass
            with self._owned_cursor(
                self._scratch,
                "CREATE INDEX temp.polylogue_source_input_work ON polylogue_source_stage_rows(load_order DESC) "
                "WHERE load_state!=2",
            ):
                pass
            with self._owned_cursor(
                self._scratch,
                "CREATE TEMP TABLE polylogue_source_sequence_baseline(physical_rowid INTEGER PRIMARY KEY,name,seq)",
            ):
                pass
            self._provision_effect_metadata()
            with self._owned_cursor(
                self._scratch,
                "CREATE TEMP TABLE polylogue_source_writable_keys("
                "target_id INTEGER PRIMARY KEY,table_name TEXT NOT NULL,key_digest BLOB NOT NULL,key_cells BLOB NOT NULL)",
            ):
                pass
            with self._owned_cursor(
                self._scratch,
                "CREATE INDEX temp.polylogue_source_writable_key_lookup "
                "ON polylogue_source_writable_keys(table_name,key_digest)",
            ):
                pass
            self._settle_witness_metadata()
            self._source_stage_ready = True
            self._seed_source_controls()
        except BaseException:
            # Partial schema setup is preparation failure, never permission
            # to borrow existing relations on another attempt. The original
            # creator must retire this entire witness and its actual handles.
            self._cleanup_requested = True
            raise

    @contextmanager
    def _source_hydration(self) -> Iterator[None]:
        """Only exact native baseline loading suppresses canonical main triggers."""
        self._require_new_work()
        self._require_witness_main_mutable()
        if not self._source_stage_ready or self._source_baseline_active or self._source_statement_active:
            raise ReferenceSealError("Source hydration requires its idle original selected state")
        owner = next(child for child in native_sql_children(self) if child.connection is self._scratch)
        connection = owner.require_connection()
        if live_connection_cursors(connection) or owner._incremental_blobs:
            raise ReferenceSealError("Source hydration must not interrupt an existing native read")
        if not connection.getconfig(sqlite3.SQLITE_DBCONFIG_ENABLE_TRIGGER):
            raise ReferenceSealError("Source producer cannot inherit disabled canonical triggers")
        # TEMP triggers still run with ENABLE_TRIGGER false. Their native
        # capture path must recognize this exact baseline phase explicitly.
        # Keep the actual selected state's current sequence rows, including
        # earlier producer effects. Explicit baseline rowids can themselves
        # advance AUTOINCREMENT; those hydration changes are never effects.
        with self._owned_cursor(connection, "DELETE FROM temp.polylogue_source_sequence_baseline"):
            pass
        with self._owned_cursor(
            connection,
            "INSERT INTO temp.polylogue_source_sequence_baseline SELECT rowid,name,seq FROM main.sqlite_sequence",
        ):
            pass
        self._source_baseline_active = True
        primary: BaseException | None = None
        try:
            connection.setconfig(sqlite3.SQLITE_DBCONFIG_ENABLE_TRIGGER, False)
            yield
        except BaseException as failure:
            primary = failure
            raise
        finally:
            try:
                with self._owned_cursor(connection, "DELETE FROM main.sqlite_sequence"):
                    pass
                with self._owned_cursor(
                    connection,
                    "INSERT INTO main.sqlite_sequence(rowid,name,seq) "
                    "SELECT physical_rowid,name,seq FROM temp.polylogue_source_sequence_baseline "
                    "ORDER BY physical_rowid",
                ):
                    pass
                connection.setconfig(sqlite3.SQLITE_DBCONFIG_ENABLE_TRIGGER, True)
                if not connection.getconfig(sqlite3.SQLITE_DBCONFIG_ENABLE_TRIGGER):
                    raise ReferenceSealError("Source hydration failed to restore canonical triggers")
                if live_connection_cursors(connection) or owner._incremental_blobs:
                    raise ReferenceSealError("Source hydration left an actual native statement or Blob unsettled")
            except BaseException as cleanup:
                owner.close_required = True
                settlement_failure = (
                    cleanup
                    if primary is None
                    else BaseExceptionGroup("Source hydration and trigger restoration failed", [primary, cleanup])
                )
                raise NativeConnectionSettlementError(owner, settlement_failure) from cleanup
            finally:
                self._source_baseline_active = False
                if primary is not None:
                    self._cleanup_requested = True

    def _source_image_insert(self, image: KnownTierRowImage) -> None:
        """Insert exact original cells with no Python variable-value adapter."""
        if not self._source_baseline_active or image._seal is not self or type(image.rowid) is not int:
            raise ReferenceSealError("Source baseline insertion requires its exact native phase and row image")
        alias = self._physical_rowid_alias(self._scratch, image.table, image.columns)
        fields = tuple(zip(image.columns, image.cells, strict=True))
        if alias in image.columns:
            alias_cell = image.cells[image.columns.index(alias)]
            if not self._literal_scalar_equal(alias_cell, image.rowid):
                raise ReferenceSealError("Source baseline rowid differs from its INTEGER PRIMARY KEY alias")
            fields = tuple((column, cell) for column, cell in fields if column != alias)
        expressions = [self.source_literal_expression(cell) for _column, cell in fields]
        columns = ",".join((quote_identifier(alias), *(quote_identifier(column) for column, _cell in fields)))
        values = ",".join(("?", *(expression for expression, _parameters in expressions)))
        parameters = (image.rowid, *(value for _expression, fields in expressions for value in fields))
        with self._owned_cursor(
            self._scratch, f"INSERT INTO {quote_identifier(image.table)}({columns}) VALUES ({values})", parameters
        ):
            pass

    def _seed_source_controls(self) -> None:
        """Replace only canonical DDL seeds with their exact original inputs."""
        for table in ("raw_existence_journal_control", "audit_continuity_control"):
            image = self.retain_tier_row("source", table, 1)
            if image is None:
                raise ReferenceSealError("original Source omits its canonical seeded control")
            image_id = self._retain_row_image(image)
            with self._source_hydration():
                # These two canonical DDL seeds have no inbound FKs. Their
                # original images must precede the first captured statement.
                with self._owned_cursor(self._scratch, f"DELETE FROM {quote_identifier(table)} WHERE rowid=1"):
                    pass
                self._source_image_insert(image)
                with self._owned_cursor(
                    self._scratch,
                    "INSERT INTO temp.polylogue_source_stage_rows(table_name,physical_rowid,input_image) VALUES (?,?,?)",
                    (table, 1, image_id),
                ):
                    pass

    def _queue_source_input(self, image: KnownTierRowImage) -> bool:
        if image._seal is not self or type(image.rowid) is not int or image.table == "sqlite_sequence":
            raise ReferenceSealError("Source inputs require this original canonical row image")
        with self._owned_cursor(
            self._scratch,
            "SELECT 1 FROM temp.polylogue_source_stage_rows WHERE table_name=? AND physical_rowid=?",
            (image.table, image.rowid),
        ) as cursor:
            if cursor.fetchone() is not None:
                # Includes staged deletion/update and a dependency already in
                # this native work queue. Original rows never resurrect either.
                return False
        columns, _keys = self._known_tier_table_shape("source", image.table)
        if image.columns != columns or not self._matches_retained_row(self._observers["source"], image):
            raise ReferenceSealError("Source input is not its exact original complete row")
        image_id = self._retain_row_image(image)
        with self._owned_cursor(
            self._scratch,
            "INSERT INTO temp.polylogue_source_stage_rows("
            "table_name,physical_rowid,input_image,load_state,load_order) "
            "VALUES (?,?,?,0,?)",
            (image.table, image.rowid, image_id, image_id),
        ):
            pass
        return True

    def _load_source_row(self, image: KnownTierRowImage) -> bool:
        """Hydrate exact original FK inputs with a native iterative queue.

        A long supersession chain consumes disk work rows, not Python stack
        frames. Read inputs remain separate from actual writable targets.
        """
        self._require_new_work()
        self._require_witness_main_mutable()
        if not self._original_reads_active or not self._source_stage_ready:
            raise ReferenceSealError("Source inputs require the pinned original preparation window")
        if not self._queue_source_input(image):
            return False
        while True:
            _check_reference_cancellation()
            with self._owned_cursor(
                self._scratch,
                "SELECT table_name,physical_rowid,input_image,load_state "
                "FROM temp.polylogue_source_stage_rows WHERE load_state!=2 ORDER BY load_order DESC LIMIT 1",
            ) as cursor:
                pending = cursor.fetchone()
            if pending is None:
                return True
            current = self._retained_row_image(pending[2])
            if pending[3] == 0:
                with self._owned_cursor(
                    self._scratch,
                    "UPDATE temp.polylogue_source_stage_rows SET load_state=1 WHERE table_name=? AND physical_rowid=?",
                    (current.table, current.rowid),
                ):
                    pass
                for parent_image in self._source_fk_input_images(current):
                    self._queue_source_input(parent_image)
                continue
            self._seed_source_sequence(current.table)
            with self._source_hydration():
                alias = self._physical_rowid_alias(self._scratch, current.table, current.columns)
                with self._owned_cursor(
                    self._scratch,
                    f"SELECT 1 FROM {quote_identifier(current.table)} WHERE {quote_identifier(alias)}=?",
                    (current.rowid,),
                ) as cursor:
                    exists = cursor.fetchone() is not None
                if exists:
                    if not self._matches_retained_row(self._scratch, current):
                        raise ReferenceSealError("Source baseline would overwrite a staged physical coordinate")
                else:
                    self._source_image_insert(current)
                with self._owned_cursor(
                    self._scratch,
                    "UPDATE temp.polylogue_source_stage_rows SET load_state=2,current_image=input_image "
                    "WHERE table_name=? AND physical_rowid=?",
                    (current.table, current.rowid),
                ):
                    pass

    def _source_fk_input_images(self, image: KnownTierRowImage) -> Iterator[KnownTierRowImage]:
        """Find actual FK parents through indexed joins on original rows."""
        observer = self._observers["source"]
        child_alias = self._physical_rowid_alias(observer, image.table, image.columns)
        with self._owned_cursor(observer, f"PRAGMA foreign_key_list({quote_identifier(image.table)})") as cursor:
            foreign_keys = tuple(cursor)
        # This is schema-sized metadata, never a set of cohort row identities.
        groups: dict[int, list[tuple[int, int, str, str, str | None, str, str, str]]] = {}
        for entry in foreign_keys:
            groups.setdefault(entry[0], []).append(entry)
        for entries in groups.values():
            parent = str(entries[0][2])
            parent_columns, parent_keys = self._known_tier_table_shape("source", parent)
            parent_alias = self._physical_rowid_alias(observer, parent, parent_columns)
            entries.sort(key=lambda entry: entry[1])
            predicates = []
            for position, entry in enumerate(entries):
                target = parent_columns[parent_keys[position]] if entry[4] is None else str(entry[4])
                predicates.append(f"p.{quote_identifier(target)}=c.{quote_identifier(str(entry[3]))}")
            with self._owned_cursor(
                observer,
                f"SELECT p.{quote_identifier(parent_alias)} FROM {quote_identifier(parent)} AS p "
                f"JOIN {quote_identifier(image.table)} AS c ON {' AND '.join(predicates)} "
                f"WHERE c.{quote_identifier(child_alias)}=?",
                (image.rowid,),
            ) as cursor:
                # A valid declared FK points to one unique parent. Keep its
                # native read settled before retaining any parent cell stream.
                row = cursor.fetchone()
                extra = cursor.fetchone()
            if extra is not None:
                raise ReferenceSealError("canonical Source FK does not identify one original parent")
            if row is not None:
                with self._owned_cursor(
                    self._scratch,
                    "SELECT 1 FROM temp.polylogue_source_stage_rows WHERE table_name=? AND physical_rowid=?",
                    (parent, row[0]),
                ) as cursor:
                    if cursor.fetchone() is not None:
                        # Canonical material DDL declares its self-FK twice.
                        # Reuse the same selected input and literal custody.
                        continue
                parent_image = self.retain_tier_row("source", parent, row[0])
                if parent_image is None:
                    raise ReferenceSealError("original Source FK parent disappeared inside its snapshot")
                yield parent_image

    def _seed_source_sequence(self, table: str) -> None:
        """Bind both named AUTOINCREMENT state and global entry allocation."""
        if table in self._source_sequence_seeded:
            return
        with self._owned_cursor(
            self._scratch, "SELECT sql FROM main.sqlite_schema WHERE type='table' AND name=?", (table,)
        ) as cursor:
            ddl = cursor.fetchone()
        if ddl is None or "AUTOINCREMENT" not in str(ddl[0]).upper():
            return
        observer = self._observers["source"]
        # Managed Source has exactly the two canonical AUTOINCREMENT tables.
        # Direct sqlite_sequence housekeeping is unsupported archive metadata
        # mutation; ordinary SQLite permits it. High seq values are preserved.
        # The named sequence and the physical maximum are separate inputs.
        # Preserve the real maximum entry even if it belongs to the sibling
        # accepted-marker allocator, rather than inventing a sentinel row.
        with self._owned_cursor(
            observer,
            "SELECT rowid FROM sqlite_sequence WHERE name=? "
            "UNION SELECT rowid FROM (SELECT rowid FROM sqlite_sequence ORDER BY rowid DESC LIMIT 1)",
            (table,),
        ) as cursor:
            rowids = tuple(row[0] for row in cursor)
        for rowid in rowids:
            image = self.retain_tier_row("source", "sqlite_sequence", rowid)
            if image is None:
                raise ReferenceSealError("original sequence allocation input disappeared")
            with self._owned_cursor(
                self._scratch,
                "SELECT 1 FROM temp.polylogue_source_stage_rows WHERE table_name='sqlite_sequence' AND physical_rowid=?",
                (rowid,),
            ) as cursor:
                loaded = cursor.fetchone() is not None
            if loaded:
                continue
            image_id = self._retain_row_image(image)
            with self._source_hydration():
                expressions = tuple(self.source_literal_expression(cell) for cell in image.cells)
                with self._owned_cursor(
                    self._scratch,
                    "INSERT INTO temp.polylogue_source_sequence_baseline(physical_rowid,name,seq) "
                    f"VALUES (?,{expressions[0][0]},{expressions[1][0]})",
                    (rowid, *(value for _expression, values in expressions for value in values)),
                ):
                    pass
                with self._owned_cursor(
                    self._scratch,
                    "INSERT INTO temp.polylogue_source_stage_rows(table_name,physical_rowid,input_image) "
                    "VALUES ('sqlite_sequence',?,?)",
                    (rowid, image_id),
                ):
                    pass
        self._source_sequence_seeded.add(table)

    def _source_allocation_dependencies(self, table: str) -> None:
        """Load the real surviving maximum before a canonical INSERT.

        Original rows touched by earlier staged effects are suppressed. New
        stage rows remain visible to SQLite's own allocation. Each original
        lookup uses the physical-rowid index, including the maximum edge.
        """
        self._require_new_work()
        self._require_witness_main_mutable()
        if not self._original_reads_active or not self._source_stage_ready or self._source_statement_active:
            raise ReferenceSealError("Source allocation inputs require its idle original producer window")
        columns, _keys = self._known_tier_table_shape("source", table)
        observer = self._observers["source"]
        alias = self._physical_rowid_alias(observer, table, columns)
        before: int | None = None
        while True:
            _check_reference_cancellation()
            predicate = "" if before is None else f" WHERE {quote_identifier(alias)}<?"
            with self._owned_cursor(
                observer,
                f"SELECT {quote_identifier(alias)} FROM {quote_identifier(table)}{predicate} "
                f"ORDER BY {quote_identifier(alias)} DESC LIMIT 1",
                () if before is None else (before,),
            ) as cursor:
                candidate = cursor.fetchone()
            if candidate is None:
                original_maximum = None
                break
            original_maximum = candidate[0]
            with self._owned_cursor(
                self._scratch,
                "SELECT touched FROM temp.polylogue_source_stage_rows WHERE table_name=? AND physical_rowid=?",
                (table, original_maximum),
            ) as cursor:
                state = cursor.fetchone()
            if state is None or state[0] == 0:
                break
            before = original_maximum
        stage_alias = self._physical_rowid_alias(self._scratch, table, columns)
        with self._owned_cursor(
            self._scratch,
            f"SELECT {quote_identifier(stage_alias)} FROM {quote_identifier(table)} "
            f"ORDER BY {quote_identifier(stage_alias)} DESC LIMIT 1",
        ) as cursor:
            staged = cursor.fetchone()
        if original_maximum is not None and (staged is None or original_maximum > staged[0]):
            image = self.retain_tier_row("source", table, original_maximum)
            if image is None:
                raise ReferenceSealError("original allocation maximum disappeared inside its pinned snapshot")
            self._load_source_row(image)
        self._seed_source_sequence(table)

    def _physical_rowid_alias(self, connection: sqlite3.Connection, table: str, columns: tuple[str, ...]) -> str:
        # The direct read pragma avoids the virtual table's schema setup on
        # the original sealed reader. Retain the actual cursor before execute
        # so an authorizer/read fault cannot leave an unassigned statement.
        with self._owned_cursor(connection, "PRAGMA main.table_list") as cursor:
            shape = next((row for row in cursor if row[0] == "main" and row[1] == table), None)
        if shape is None or shape[4]:
            raise ReferenceSealError("rowid image cannot address a WITHOUT ROWID table")
        alias = next(
            (name for name in ("rowid", "_rowid_", "oid") if name not in {column.lower() for column in columns}), None
        )
        if alias is not None:
            return alias
        # An actual INTEGER PRIMARY KEY remains the physical alias even when
        # all three hidden-rowid spellings are shadowed. The inline DESC
        # exception has a separate PK index and must never be guessed as one.
        with self._owned_cursor(connection, f"PRAGMA table_info({quote_identifier(table)})") as cursor:
            primary = tuple(row for row in cursor if row[5])
        with self._owned_cursor(connection, f"PRAGMA index_list({quote_identifier(table)})") as cursor:
            has_primary_index = any(row[3] == "pk" for row in cursor)
        if len(primary) == 1 and str(primary[0][2]).upper() == "INTEGER" and not has_primary_index:
            return str(primary[0][1])
        raise ReferenceSealError("selected row image requires the table's actual accessible rowid")

    def _literal_cells_equal(self, left: KnownTierCell, right: KnownTierCell) -> bool:
        if self._literal_cell_metadata(left) != self._literal_cell_metadata(right):
            return False
        # Retained chunks may come from different producer slicing. Compare
        # the literal stream with fixed bounded reads, independent of slices.
        return self._literal_streams_equal(self._literal_cell_chunks(left), self._literal_cell_chunks(right))

    def _literal_scalar_equal(self, cell: KnownTierCell, value: None | int | float | str | bytes) -> bool:
        """Compare a declared scalar without allocating another retained slot."""
        kind, size, fixed = self._literal_cell_metadata(cell)
        if value is None:
            return kind == "null"
        if type(value) is int:
            return kind == "integer" and fixed == value.to_bytes(8, "big", signed=True)
        if type(value) is float:
            return kind == "real" and fixed == struct.pack(">d", value)
        if isinstance(value, bytes):
            expected_kind = "blob"
            length = len(value)
            chunks = (value[offset : offset + LITERAL_CHUNK_BYTES] for offset in range(0, length, LITERAL_CHUNK_BYTES))
        elif isinstance(value, str):
            expected_kind = "text"
            length = 0
            for offset in range(0, len(value), LITERAL_CHUNK_BYTES):
                _check_reference_cancellation()
                length += len(value[offset : offset + LITERAL_CHUNK_BYTES].encode("utf-8"))
            chunks = (
                value[offset : offset + LITERAL_CHUNK_BYTES].encode("utf-8")
                for offset in range(0, len(value), LITERAL_CHUNK_BYTES)
            )
        else:
            raise TypeError("exact scalar comparison requires a SQLite storage class")
        return (kind, size) == (expected_kind, length) and self._literal_streams_equal(
            self._literal_cell_chunks(cell), chunks
        )

    @staticmethod
    def _literal_streams_equal(left: Generator[bytes, None, None], right: Generator[bytes, None, None]) -> bool:
        def normalized(chunks: Iterable[bytes]) -> Iterator[bytes]:
            pending = bytearray()
            for chunk in chunks:
                view = memoryview(chunk)
                offset = 0
                while offset < len(view):
                    count = min(LITERAL_CHUNK_BYTES - len(pending), len(view) - offset)
                    pending.extend(view[offset : offset + count])
                    offset += count
                    if len(pending) == LITERAL_CHUNK_BYTES:
                        yield bytes(pending)
                        pending.clear()
            if pending:
                yield bytes(pending)

        try:
            return all(a == b for a, b in zip_longest(normalized(left), normalized(right)))
        finally:
            # A mismatch must settle both original streamed readers too.
            try:
                left.close()
            finally:
                right.close()

    def _retain_row_image(self, image: KnownTierRowImage) -> int:
        """Persist schema-width locators, never a pickled variable row payload."""
        self._require_witness_main_mutable()
        if image._seal is not self or len(image.columns) != len(image.cells):
            raise ReferenceSealError("retained row image must belong to this exact original witness")
        for cell in image.cells:
            self._literal_cell_metadata(cell)
        with self._owned_cursor(
            self._scratch,
            "INSERT INTO known_tier_row_images(table_name,columns_blob,physical_rowid,cell_ids_blob) VALUES (?,?,?,?)",
            (
                image.table,
                pickle.dumps(image.columns, protocol=5),
                image.rowid,
                pickle.dumps(tuple(cell._cell_id for cell in image.cells), protocol=5),
            ),
        ) as cursor:
            image_id = cursor.lastrowid
        assert image_id is not None
        return image_id

    def _retained_row_image(self, image_id: int) -> KnownTierRowImage:
        with self._owned_cursor(
            self._scratch,
            "SELECT table_name,columns_blob,physical_rowid,cell_ids_blob FROM known_tier_row_images WHERE image_id=?",
            (image_id,),
        ) as cursor:
            row = cursor.fetchone()
        if row is None:
            raise ReferenceSealError("row image locator has no retained original descriptor")
        columns = pickle.loads(row[1])
        cells = tuple(KnownTierCell(self, cell_id) for cell_id in pickle.loads(row[3]))
        if len(columns) != len(cells):
            raise ReferenceSealError("retained row image omits its canonical cells")
        return KnownTierRowImage(self, str(row[0]), columns, row[2], cells)

    def _row_images_equal(self, left: KnownTierRowImage | None, right: KnownTierRowImage | None) -> bool:
        if left is None or right is None:
            return left is right
        if left._seal is not self or right._seal is not self:
            raise ReferenceSealError("row comparison requires the same original witness")
        if (left.table, left.columns, left.rowid) != (right.table, right.columns, right.rowid):
            return False
        if len(left.cells) != len(left.columns) or len(right.cells) != len(right.columns):
            raise ReferenceSealError("row comparison omits canonical cells")
        return all(self._literal_cells_equal(a, b) for a, b in zip(left.cells, right.cells, strict=True))

    def _matches_retained_row(
        self, connection: sqlite3.Connection, image: KnownTierRowImage, *, settlement: bool = False
    ) -> bool:
        """Compare exact physical row identity and literal bytes, never a hash verdict."""
        if image._seal is not self or image.rowid is None:
            raise ReferenceSealError("physical row comparison requires this witness's exact rowid image")
        if len(image.cells) != len(image.columns):
            raise ReferenceSealError("physical row image omits canonical cells")
        alias = self._physical_rowid_alias(connection, image.table, image.columns)
        projection = ",".join(cell_projection(quote_identifier(column)) for column in image.columns)
        source_sql = f"FROM {quote_identifier(image.table)} WHERE {quote_identifier(alias)}=?"
        with self._owned_cursor(
            connection, f"SELECT {quote_identifier(alias)},{projection} {source_sql}", (image.rowid,)
        ) as cursor:
            row = cursor.fetchone()
        if row is None or row[0] != image.rowid:
            return False
        owner = next(child for child in native_sql_children(self) if child.connection is connection)
        check: Callable[[], None] = (lambda: None) if settlement else _check_reference_cancellation
        for position, (column, expected) in enumerate(zip(image.columns, image.cells, strict=True)):
            check()
            actual = literal_metadata(*row[1 + 3 * position : 1 + 3 * position + 3])
            kind, size, fixed = self._literal_cell_metadata(expected)
            if (actual.storage_class, actual.byte_length) != (kind, size):
                return False
            if kind not in {"text", "blob"}:
                if actual.fixed_bytes() != fixed:
                    return False
                continue
            with owner.readonly_blob(image.table, column, image.rowid, settlement=settlement) as blob:
                literal = stream_literal_blob(blob, actual.byte_length, check)
                if not self._literal_streams_equal(self._literal_cell_chunks(expected, settlement=settlement), literal):
                    return False
        return True

    def before_index_input(
        self, table: str, columns: tuple[str, ...], rowid_sql: str, parameters: tuple[object, ...]
    ) -> None:
        """Charge explicit canonical Index input coordinates before hydration.

        Canonical read owners provide their exact selected physical rowids,
        never SQL rewriting or a borrowed-current-generation connection.
        """
        self._require_capability("index")
        self._require_new_work()
        if self._original_input_demand is None:
            return
        observer = self.observer("index")
        if not observer.in_transaction:
            raise ReferenceSealError("Index input demand requires its same original pinned observer")
        with self._owned_cursor(observer, rowid_sql, parameters) as cursor:
            for (rowid,) in cursor:
                _check_reference_cancellation()
                self._amend_original_input_fields("index", table, rowid, columns)

    def _original_blob_input(self, raw_id: str) -> tuple[bytes, int]:
        if not self._original_reads_active:
            raise ReferenceSealError("original CAS input demand requires its pinned acquisition read window")
        observer = self.observer("source")
        with self._owned_cursor(observer, "SELECT rowid FROM raw_sessions WHERE raw_id=?", (raw_id,)) as cursor:
            selected = cursor.fetchone()
        if selected is None:
            raise ReferenceSealStaleError("original CAS input acquisition descriptor is absent")
        self._amend_original_input_fields("source", "raw_sessions", selected[0], ("blob_hash", "blob_size"))
        with self._owned_cursor(
            observer, "SELECT blob_hash,blob_size FROM raw_sessions WHERE rowid=?", (selected[0],)
        ) as cursor:
            row = cursor.fetchone()
        if row is None:
            raise ReferenceSealStaleError("original CAS input acquisition descriptor disappeared")
        if not isinstance(row[0], bytes) or len(row[0]) != 32 or type(row[1]) is not int or row[1] < 0:
            raise ReferenceSealError("original CAS input has no canonical hash and exact byte size")
        return row[0], row[1]

    def retain_original_blob_input(self, raw_id: str) -> tuple[bytes, int]:
        """Amend demand before reading one exact originally acquired CAS input.

        This bookkeeping confers no Blob publication or reference authority.
        Exclusive discovery starts with zero accounted inputs. Every actual
        original acquisition is charged before its first payload read.
        """
        self._require_new_work()
        if self._source_statement_active:
            raise ReferenceSealError("original CAS demand cannot enter a staged statement savepoint")
        blob_hash, byte_length = self._original_blob_input(raw_id)
        self._amend_cas_input(blob_hash, byte_length)
        return blob_hash, byte_length

    def retain_prepared_blob_input(self, claim: PreparedBlobPublicationClaim) -> tuple[bytes, int]:
        """Enroll this actual prepared input before payload use, not publication.

        The same current creator charges immutable CAS identity once. File
        and accepted-reservation proof establish input provenance only; this
        bookkeeping cannot grant a Source write or release an old reservation.
        """
        from polylogue.storage.blob_publication import PreparedBlobPublicationClaim, _prepared_claim_record

        self._require_new_work()
        if not self._original_reads_active or not isinstance(claim, PreparedBlobPublicationClaim):
            raise ReferenceSealError("prepared CAS input requires its exact claim in the original read window")
        self._require_capability("source")
        source_path = claim.publisher.source_db_path
        if source_path.resolve() != self._paths["source"].resolve():
            raise ReferenceSealError("prepared CAS input belongs to another Source target")
        if not _same_incarnation(_tier_identity(source_path), self._identities["source"]):
            raise ReferenceSealStaleError("prepared CAS input Source incarnation changed")
        # This input is read under the original window, whose entry/exit
        # bracket owns currency. Publication's unpinned validator cannot run
        # inside this caller-owned snapshot.
        receipt = claim.receipt
        if (
            receipt.publisher_id != claim.publisher.publisher_id
            or receipt.blob_hash != claim.seal.sha256
            or type(receipt.size_bytes) is not int
            or receipt.size_bytes != claim.seal.size
            or receipt.size_bytes < 0
        ):
            raise ReferenceSealError("prepared CAS input differs from its actual publisher/file claim")
        try:
            blob_hash = bytes.fromhex(receipt.blob_hash)
        except ValueError as failure:
            raise ReferenceSealError("prepared CAS input lacks canonical hash bytes") from failure
        if len(blob_hash) != 32 or blob_hash.hex() != receipt.blob_hash:
            raise ReferenceSealError("prepared CAS input lacks canonical hash bytes")
        if claim.prepared_path.exists():
            claim.publisher._validate_claim_path(claim.prepared_path)
            claim.seal.verify(claim.prepared_path, full=False)
        else:
            # The actual receipt, never a guessed staged raw identity, proves
            # this same input already entered the final CAS namespace.
            observer = self.observer("source")
            with self._owned_cursor(
                observer,
                "SELECT rowid FROM blob_publication_reservations WHERE publication_id=?",
                (receipt.publication_id,),
            ) as cursor:
                reservation = cursor.fetchone()
            if reservation is not None:
                self._amend_original_input_fields(
                    "source",
                    "blob_publication_reservations",
                    reservation[0],
                    ("blob_hash", "size_bytes", "publisher_id"),
                )
            claim.publisher.validate_published_claim(observer, claim, source_path="")
        self._amend_cas_input(blob_hash, receipt.size_bytes, prepared_claim=_prepared_claim_record(claim))
        return blob_hash, receipt.size_bytes

    def _amend_cas_input(self, blob_hash: bytes, byte_length: int, *, prepared_claim: str | None = None) -> None:
        with self._owned_cursor(
            self._scratch,
            "SELECT charged FROM temp.original_blob_inputs WHERE blob_hash=? AND byte_length=?",
            (blob_hash, byte_length),
        ) as cursor:
            row = cursor.fetchone()
        charged = row is not None and bool(row[0])
        try:
            if self._original_input_demand is not None and not charged:
                self._original_input_demand(byte_length)
                charged = True
            with self._owned_cursor(
                self._scratch,
                "INSERT INTO temp.original_blob_inputs(blob_hash,byte_length,charged,prepared_claim_json) VALUES (?,?,?,?) "
                "ON CONFLICT(blob_hash,byte_length) DO UPDATE SET charged=excluded.charged,"
                "prepared_claim_json=coalesce(original_blob_inputs.prepared_claim_json,excluded.prepared_claim_json)",
                (blob_hash, byte_length, int(charged), prepared_claim),
            ):
                pass
        except BaseException:
            # A successful amendment followed by uncertain bookkeeping may
            # not be retried as if the current creator had not paid it.
            self._cleanup_requested = True
            raise

    def retain_tier_row(self, tier: str, table: str, rowid: int) -> KnownTierRowImage | None:
        """Retain one selected original row without fetching any TEXT/BLOB cell.

        The original read window proves currency when its native snapshots
        end. These immutable selected cells remain in the same witness through
        live effect comparison and every failed physical settlement.
        """
        self._require_new_work()
        if not self._original_reads_active:
            raise ReferenceSealError("selected original cells require the pinned original read window")
        if self._source_statement_active:
            raise ReferenceSealError("original input retention cannot enter a staged statement savepoint")
        columns, _keys = self._known_tier_table_shape(tier, table)
        observer = self.observer(tier)
        self._amend_original_input_fields(tier, table, rowid, columns)
        coordinate = (tier, self._original_input_epochs[tier], table, rowid)
        with self._owned_cursor(
            self._scratch,
            "SELECT image_id FROM temp.original_input_rows "
            "WHERE tier=? AND epoch=? AND table_name=? AND physical_rowid=?",
            coordinate,
        ) as cursor:
            retained = cursor.fetchone()
        if retained is not None:
            return self._retained_row_image(retained[0])
        # The successful demand amendment is outside this copy savepoint.
        # A settled read/copy failure rolls back every partial literal/image,
        # so retry reuses the charge without accumulating abandoned slots.
        with self._owned_cursor(self._scratch, "SAVEPOINT polylogue_original_input_copy"):
            pass
        try:
            image = self._retain_native_row(observer, table, columns, rowid)
            if image is not None:
                image_id = self._retain_row_image(image)
                with self._owned_cursor(
                    self._scratch,
                    "INSERT INTO temp.original_input_rows(tier,epoch,table_name,physical_rowid,image_id) VALUES (?,?,?,?,?)",
                    (*coordinate, image_id),
                ):
                    pass
        except BaseException as failure:
            children = tuple(
                child for child in native_sql_children(self) if child.connection in (observer, self._scratch)
            )
            if any(
                child.close_required
                or child._parent_cleanup_requested
                or (child.connection is self._scratch and child._incremental_blobs)
                for child in children
            ):
                self._cleanup_requested = True
                raise
            try:
                with self._owned_cursor(self._scratch, "ROLLBACK TO polylogue_original_input_copy"):
                    pass
                with self._owned_cursor(self._scratch, "RELEASE polylogue_original_input_copy"):
                    pass
            except BaseException as cleanup:
                self._cleanup_requested = True
                owner = next(child for child in children if child.connection is self._scratch)
                owner.close_required = True
                raise NativeConnectionSettlementError(
                    owner, BaseExceptionGroup("Original input copy and rollback failed", [failure, cleanup])
                ) from cleanup
            raise
        else:
            try:
                with self._owned_cursor(self._scratch, "RELEASE polylogue_original_input_copy"):
                    pass
            except BaseException as cleanup:
                self._cleanup_requested = True
                owner = next(child for child in native_sql_children(self) if child.connection is self._scratch)
                owner.close_required = True
                raise NativeConnectionSettlementError(owner, cleanup) from cleanup
            return image

    def _amend_original_input_fields(self, tier: str, table: str, rowid: int, columns: tuple[str, ...]) -> None:
        """Charge exact declared native inputs before any Python hydration.

        Constructor reference reads use the same coordinate and field ledger
        as complete row retention. Numeric literals have eight bytes; NULL
        has none. TEXT uses native UTF-8 byte length, not character count.
        Successful charges survive later literal-copy failure. These calls
        cannot enter a Source mutation savepoint.
        """
        if self._original_input_demand is None:
            return
        if self._source_statement_active:
            raise ReferenceSealError("original input demand cannot enter a staged statement savepoint")
        observer = self.observer(tier)
        if not observer.in_transaction:
            raise ReferenceSealError("original input demand requires the actual pinned native observer")
        if not table.isidentifier():
            raise ReferenceSealError("input demand requires an actual declared native relation")
        # Read inputs include generated fields, such as sessions.session_id.
        # Effect images deliberately retain only writable table_info fields;
        # input accounting must also cover generated values actually hydrated.
        with self._owned_cursor(observer, f"PRAGMA table_xinfo({quote_identifier(table)})") as cursor:
            canonical = tuple(row[1] for row in cursor)
        if not columns or any(column not in canonical for column in columns):
            raise ReferenceSealError("input demand must name actual declared native fields")
        coordinate = (tier, self._original_input_epochs[tier], table, rowid)
        with self._owned_cursor(
            self._scratch,
            "SELECT column_name FROM temp.original_input_fields "
            "WHERE tier=? AND epoch=? AND table_name=? AND physical_rowid=?",
            coordinate,
        ) as cursor:
            charged = {row[0] for row in cursor}
        pending = tuple(column for column in columns if column not in charged)
        if not pending:
            return
        alias = self._physical_rowid_alias(observer, table, canonical)
        projection = ",".join(
            f"CASE typeof({quote_identifier(column)}) "
            f"WHEN 'null' THEN 0 WHEN 'integer' THEN 8 WHEN 'real' THEN 8 "
            f"ELSE length(CAST({quote_identifier(column)} AS BLOB)) END"
            for column in pending
        )
        with self._owned_cursor(
            observer,
            f"SELECT {projection} FROM {quote_identifier(table)} WHERE {quote_identifier(alias)}=?",
            (rowid,),
        ) as cursor:
            lengths = cursor.fetchone()
        if lengths is None:
            return
        try:
            self._original_input_demand(sum(lengths))
            for column, length in zip(pending, lengths, strict=True):
                with self._owned_cursor(
                    self._scratch,
                    "INSERT INTO temp.original_input_fields "
                    "(tier,epoch,table_name,physical_rowid,column_name,byte_length) VALUES (?,?,?,?,?,?)",
                    (*coordinate, column, length),
                ):
                    pass
        except BaseException:
            # An uncertain amendment or failed ledger update cannot become
            # uncharged work again. The original owner must physically settle.
            self._cleanup_requested = True
            raise

    def _retain_native_row(
        self,
        connection: sqlite3.Connection,
        table: str,
        columns: tuple[str, ...],
        rowid: int,
        *,
        reuse: KnownTierRowImage | None = None,
        prepared_cells: dict[str, KnownTierCell] | None = None,
    ) -> KnownTierRowImage | None:
        """Read selected OLD/NEW cells on the same actual native child owner."""
        self._require_witness_main_mutable()
        if reuse is not None and (reuse._seal is not self or reuse.table != table or reuse.columns != columns):
            raise ReferenceSealError("native row reuse requires this exact complete table image")
        candidates = {} if prepared_cells is None else {column: (cell,) for column, cell in prepared_cells.items()}
        if any(column not in columns for column in candidates):
            raise ReferenceSealError("prepared literal names a field outside the actual captured table")
        if reuse is not None:
            if len(reuse.cells) != len(columns):
                raise ReferenceSealError("native row reuse omits canonical cells")
            for column, cell in zip(columns, reuse.cells, strict=True):
                current = candidates.get(column, ())
                if all(candidate._cell_id != cell._cell_id for candidate in current):
                    candidates[column] = (*current, cell)
        for cells in candidates.values():
            for cell in cells:
                self._literal_cell_metadata(cell)
        owner = next((child for child in native_sql_children(self) if child.connection is connection), None)
        if owner is None:
            raise ReferenceSealError("native row capture requires this witness's original SQL child")
        owner.require_connection()
        alias = self._physical_rowid_alias(connection, table, columns)
        projection = ",".join(
            f"typeof({quote_identifier(column)}), CASE WHEN typeof({quote_identifier(column)}) IN ('integer','real') THEN {quote_identifier(column)} END"
            for column in columns
        )
        with self._owned_cursor(
            connection,
            f"SELECT {projection} FROM {quote_identifier(table)} WHERE {quote_identifier(alias)}=?",
            (rowid,),
        ) as cursor:
            row = cursor.fetchone()
        if row is None:
            return None
        retained: list[KnownTierCell] = []
        for position, column in enumerate(columns):
            _check_reference_cancellation()
            kind, number = row[2 * position : 2 * position + 2]
            field_candidates = candidates.get(column, ())
            if kind in {"text", "blob"}:
                with owner.readonly_blob(table, column, rowid) as blob:
                    metadata = literal_metadata(kind, None, len(blob))
                    matching: KnownTierCell | None = None
                    for candidate in field_candidates:
                        candidate_kind, candidate_size, _fixed = self._literal_cell_metadata(candidate)
                        if (candidate_kind, candidate_size) != (metadata.storage_class, metadata.byte_length):
                            continue
                        if self._literal_streams_equal(
                            self._literal_cell_chunks(candidate),
                            stream_literal_blob(blob, metadata.byte_length, _check_reference_cancellation),
                        ):
                            matching = candidate
                            break
                        blob.seek(0)
                    if matching is not None:
                        retained.append(matching)
                    else:
                        retained.append(
                            self._retain_variable_cell(
                                metadata, stream_literal_blob(blob, metadata.byte_length, _check_reference_cancellation)
                            )
                        )
            else:
                metadata = literal_metadata(kind, number, None)
                matching = next(
                    (
                        candidate
                        for candidate in field_candidates
                        if self._literal_cell_metadata(candidate)
                        == (metadata.storage_class, metadata.byte_length, metadata.fixed_bytes())
                    ),
                    None,
                )
                retained.append(matching if matching is not None else self._retain_fixed_cell(metadata))
        return KnownTierRowImage(self, table, columns, rowid, tuple(retained))

    def _literal_cell_metadata(self, cell: KnownTierCell) -> tuple[str, int, bytes | None]:
        if cell._seal is not self:
            raise ReferenceSealError("literal cell belongs to another original witness")
        with self._owned_cursor(
            self._scratch,
            "SELECT storage_class,byte_length,fixed_blob FROM known_tier_literal_cells WHERE cell_id=?",
            (cell._cell_id,),
        ) as cursor:
            row = cursor.fetchone()
        if row is None:
            raise ReferenceSealError("literal cell locator has no retained original image")
        return str(row[0]), int(row[1]), row[2]

    def _literal_cell_chunks(self, cell: KnownTierCell, *, settlement: bool = False) -> Generator[bytes, None, None]:
        kind, size, fixed = self._literal_cell_metadata(cell)
        if kind not in {"text", "blob"}:
            assert fixed is not None
            yield fixed
            return
        owner = next(child for child in native_sql_children(self) if child.connection is self._scratch)
        check: Callable[[], None] = (lambda: None) if settlement else _check_reference_cancellation
        with owner.readonly_blob("known_tier_literals", "literal", cell._cell_id, settlement=settlement) as blob:
            yield from stream_literal_blob(blob, size, check)

    def source_literal_expression(self, cell: KnownTierCell) -> tuple[str, tuple[object, ...]]:
        """Read the same native slot in prepared and live canonical SQL."""
        self._require_new_work()
        kind, _size, fixed = self._literal_cell_metadata(cell)
        if kind == "null":
            return "?", (None,)
        if kind == "integer":
            assert fixed is not None
            return "?", (int.from_bytes(fixed, "big", signed=True),)
        if kind == "real":
            assert fixed is not None
            return "?", (struct.unpack(">d", fixed)[0],)
        expression = "(SELECT literal FROM temp.polylogue_source_literals WHERE cell_id=?)"
        if kind == "text":
            expression = "CAST(" + expression + " AS TEXT)"
        elif kind != "blob":
            raise ReferenceSealError("native literal has no declared SQLite storage class")
        return expression, (cell._cell_id,)

    def _literal_key_digest(self, image: KnownTierRowImage, keys: tuple[int, ...]) -> bytes:
        """Index accelerator only; exact rows and key bytes still must compare."""
        if image._seal is not self:
            raise ReferenceSealError("literal row belongs to another original witness")
        return self._literal_cells_digest(tuple(image.cells[position] for position in keys))

    def _literal_cells_digest(self, cells: tuple[KnownTierCell, ...]) -> bytes:
        digest = hashlib.sha256()
        for cell in cells:
            kind, size, _fixed = self._literal_cell_metadata(cell)
            encoded = kind.encode("ascii")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            digest.update(size.to_bytes(8, "big"))
            for chunk in self._literal_cell_chunks(cell):
                digest.update(chunk)
        return digest.digest()

    def validate_observers_current(self) -> None:
        """Point-check every retained observer immediately after gate admission."""
        self._require_new_work()
        for name, _path in self._paths.items():
            observer = self._require_unpinned_observer(name)
            if self._observer_identity(name) != self._identities[name]:
                raise ReferenceSealStaleError(f"the {name}.db file incarnation changed after preparation")
            with self._owned_cursor(observer, "PRAGMA data_version") as cursor:
                current = int(cursor.fetchone()[0])
            if current != self._versions[name]:
                raise ReferenceSealStaleError(f"{name}.db changed after preparation")

    @staticmethod
    def _candidate_schema_identity(observer: sqlite3.Connection) -> tuple[int, str | None]:
        with closing(observer.execute("PRAGMA user_version")) as cursor:
            user_version = int(cursor.fetchone()[0])
        with closing(
            observer.execute("SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'schema_identity'")
        ) as cursor:
            has_identity = cursor.fetchone()
        identity = None
        if has_identity is not None:
            with closing(observer.execute("SELECT identity FROM schema_identity WHERE tier = 'index'")) as cursor:
                row = cursor.fetchone()
            identity = str(row[0]) if row is not None else None
        return user_version, identity

    def prepare_candidate_reachability(self, candidate_index_path: Path) -> tuple[int, int, int, int]:
        """Prove promotion reachability off-writer and retain its observer.

        The same candidate connection remains open through writer admission.
        The full reference and active-coverage scans happen here; callers must
        only perform point currency checks after acquiring archive custody.
        """
        self._require_capability("index")
        self._require_new_work()
        if self._candidate_path is not None:
            raise ReferenceSealError("this reference seal already has a promotion candidate")
        candidate = Path(candidate_index_path).resolve(strict=True)
        identity_before = _tier_identity(candidate)
        observer = self._open_observer("candidate", candidate)
        first: _ResolvedReference | None = None
        lost_count = 0
        version_before = version_after = -1
        schema: tuple[int, str | None] = (0, None)
        missing_count = 0
        first_missing: str | None = None
        # The promotion preparer owns this seal on success and failure.
        # Leave failed snapshots/handles registered for that terminal pass.
        with closing(observer.execute("PRAGMA data_version")) as cursor:
            version_before = int(cursor.fetchone()[0])
        schema = self._candidate_schema_identity(observer)
        with closing(observer.execute("BEGIN")):
            pass
        with closing(
            self._scratch.execute(
                "SELECT kind, owner_session_id, object_id, qualifier, scope_session_id, target_message_id, wire_ref, has_session_alias "
                "FROM resolved_refs ORDER BY kind, object_id, qualifier"
            )
        ) as reference_rows:
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
        active = self._require_unpinned_observer("index")
        source = self._require_unpinned_observer("source")
        with closing(active.execute("PRAGMA data_version")) as cursor:
            active_before = int(cursor.fetchone()[0])
        with closing(source.execute("PRAGMA data_version")) as cursor:
            source_before = int(cursor.fetchone()[0])
        if active_before != self._versions["index"] or source_before != self._versions["source"]:
            raise ReferenceSealStaleError("archive changed before promotion coverage validation")
        with closing(active.execute("BEGIN")):
            pass
        with closing(source.execute("BEGIN")):
            pass
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
            with closing(
                active.execute(
                    "SELECT session_id, raw_id FROM sessions "
                    "WHERE session_id > ? AND raw_id IS NOT NULL ORDER BY session_id LIMIT ?",
                    (after, page_size),
                )
            ) as cursor:
                rows = cursor.fetchall()
            if not rows:
                break
            after = str(rows[-1][0])
            raw_ids = tuple(dict.fromkeys(str(row[1]) for row in rows))
            with closing(
                source.execute(
                    f"SELECT raw_id FROM raw_sessions WHERE raw_id IN ({','.join('?' for _ in raw_ids)})",
                    raw_ids,
                )
            ) as cursor:
                retained = {str(row[0]) for row in cursor}
            owed = tuple(str(row[0]) for row in rows if str(row[1]) in retained)
            if not owed:
                continue
            with closing(
                observer.execute(
                    f"SELECT session_id FROM sessions WHERE session_id IN ({','.join('?' for _ in owed)})",
                    owed,
                )
            ) as cursor:
                present = {str(row[0]) for row in cursor}
            for session_id in owed:
                _check_reference_cancellation()
                if session_id not in present:
                    missing_count += 1
                    if first_missing is None:
                        first_missing = session_id
        source.commit()
        active.commit()
        with closing(active.execute("PRAGMA data_version")) as cursor:
            active_after = int(cursor.fetchone()[0])
        with closing(source.execute("PRAGMA data_version")) as cursor:
            source_after = int(cursor.fetchone()[0])
        if active_after != active_before or source_after != source_before:
            raise ReferenceSealStaleError("archive changed during promotion coverage validation")
        observer.commit()
        with closing(observer.execute("PRAGMA data_version")) as cursor:
            version_after = int(cursor.fetchone()[0])
        identity_after = _tier_identity(candidate)
        self._assert_configured_namespace()
        if identity_after != identity_before or version_after != version_before:
            raise ReferenceSealStaleError("promotion candidate changed during durable-reference validation")
        if lost_count and first is not None:
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
        self._require_capability("index")
        self._require_new_work()
        candidate = Path(candidate_index_path).resolve(strict=True)
        if candidate != self._candidate_path or self._candidate_identity is None:
            raise ReferenceSealStaleError("promotion candidate differs from its prepared proof")
        observer = self._require_unpinned_observer("candidate")
        identity_before = _tier_identity(candidate)
        with closing(observer.execute("PRAGMA data_version")) as cursor:
            version_before = int(cursor.fetchone()[0])
        schema = self._candidate_schema_identity(observer)
        with closing(observer.execute("PRAGMA data_version")) as cursor:
            version_after = int(cursor.fetchone()[0])
        identity_after = _tier_identity(candidate)
        if identity_before != identity_after or identity_after != self._candidate_identity:
            raise ReferenceSealStaleError("promotion candidate incarnation changed after reference preparation")
        if version_before != version_after or version_after != self._candidate_version:
            raise ReferenceSealStaleError("promotion candidate changed after reference preparation")
        if schema != self._candidate_schema:
            raise ReferenceSealStaleError("promotion candidate schema changed after reference preparation")
        return self._candidate_identity

    def require_source_target(self, path: Path) -> None:
        """Bind a publisher to this retained Source observer's exact target."""
        self._require_new_work()
        if path.resolve() != self._paths["source"].resolve():
            raise ReferenceSealError("publisher Source path does not belong to this prepared seal")
        if not _same_incarnation(_tier_identity(path), self._identities["source"]):
            raise ReferenceSealStaleError("publisher Source incarnation changed after preparation")
        self.validate_observers_current()

    def _known_tier_table_shape(self, tier: str, table: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
        self._require_capability(tier)
        if not table.isidentifier():
            raise ReferenceSealError("known tier effects require declared table identifiers")
        with self._owned_cursor(self._observers[tier], f'PRAGMA table_info("{table}")') as cursor:
            info = cursor.fetchall()
        columns = tuple(str(row[1]) for row in info)
        if table == "sqlite_sequence" and columns == ("name", "seq"):
            return columns, (0,)
        keys = tuple(
            position
            for _order, position in sorted((int(row[5]), position) for position, row in enumerate(info) if row[5])
        )
        if not columns or not keys or any(not column.isidentifier() for column in columns):
            raise ReferenceSealError("known tier effects require a canonical keyed table")
        return columns, keys

    def _retain_declared_update_target(
        self,
        tier: str,
        table: str,
        complete: tuple[str, ...],
        columns: tuple[str, ...],
        values: tuple[object, ...],
        key_column: str,
    ) -> KnownTierRowImage:
        _check_reference_cancellation()
        if len(values) != len(columns) + 1 or any(
            value is not None and not isinstance(value, (int, float, str, bytes)) for value in values
        ):
            raise ReferenceSealError("prepared update must carry its exact SQLite scalar values and target key")
        observer = self._observers[tier]
        alias = self._physical_rowid_alias(observer, table, complete)
        with self._owned_cursor(
            observer,
            f"SELECT {quote_identifier(alias)} FROM {quote_identifier(table)} "
            f"WHERE {quote_identifier(key_column)} IS ?",
            (values[-1],),
        ) as cursor:
            found = cursor.fetchone()
        if found is None:
            raise ReferenceSealError("declared partial update lacks its original physical target")
        old = self.retain_tier_row(tier, table, found[0])
        if old is None:
            raise ReferenceSealStaleError("declared update target disappeared from its original snapshot")
        return old

    def _updated_row_effects(
        self,
        tier: str,
        table: str,
        columns: tuple[str, ...],
        rows: Iterable[tuple[object, ...]],
        key_column: str,
    ) -> Iterator[KnownTierRowEffect]:
        complete, keys = self._known_tier_table_shape(tier, table)
        if tuple(complete[position] for position in keys) != (key_column,) or any(
            column not in complete for column in columns
        ):
            raise ReferenceSealError("prepared values do not match their declared table key and columns")
        if key_column in columns:
            raise ReferenceSealError("declared partial updates cannot rewrite their target key")
        for values in rows:
            old = self._retain_declared_update_target(tier, table, complete, columns, values, key_column)
            cells = dict(zip(complete, old.cells, strict=True))
            for column, value in zip(columns, values[:-1], strict=True):
                cells[column] = self.retain_literal_scalar(cast(None | int | float | str | bytes, value))
            new = KnownTierRowImage(self, table, complete, old.rowid, tuple(cells[column] for column in complete))
            yield KnownTierRowEffect(table, complete, old, new)

    def _validate_user_row_effect(self, effect: KnownTierRowEffect) -> None:
        """Accept only the existing bound removal's exact assertion provenance."""
        self._require_capability("user")
        from polylogue.storage.sqlite.write_lease import permitted_session_removals

        if effect.table == "query_unit_frame_state":
            if effect.columns != ("singleton", "epoch") or effect.old is None or effect.new is None:
                raise ReferenceSealError("User frame effects must retain their canonical singleton")
            kind, _size, fixed = self._literal_cell_metadata(effect.old.cells[1])
            if kind != "integer" or fixed is None:
                raise ReferenceSealError("User frame epoch must retain its INTEGER storage class")
            epoch = int.from_bytes(fixed, "big", signed=True)
            if epoch == (1 << 63) - 1:
                raise ReferenceSealError("User frame epoch exceeds SQLite's INTEGER range")
            if (
                not self._literal_scalar_equal(effect.old.cells[0], 1)
                or not self._literal_scalar_equal(effect.new.cells[0], 1)
                or not self._literal_scalar_equal(effect.new.cells[1], epoch + 1)
            ):
                raise ReferenceSealError("each assertion effect must advance the User frame exactly once")
            return
        if effect.table != "assertions":
            raise ReferenceSealError("known User effects are confined to the bound assertion producer")
        old = None if effect.old is None else dict(zip(effect.columns, effect.old.cells, strict=True))
        new = None if effect.new is None else dict(zip(effect.columns, effect.new.cells, strict=True))
        image = old if old is not None else new
        assert image is not None
        lifecycle = any(
            self._literal_scalar_equal(image["kind"], kind)
            for kind in ("suppression", "excision_record", "excision_request")
        )
        permitted = permitted_session_removals(archive_root=self.archive_root, assertion_content=not lifecycle)
        if not permitted:
            raise ReferenceSealError("User effects require the exact validated session removal")
        # Only exact permission inputs belong here. This TEMP relation does
        # not alter the original live reference census or Index permissions.
        with self._owned_cursor(
            self._scratch,
            "CREATE TEMP TABLE IF NOT EXISTS user_effect_removals(session_id TEXT PRIMARY KEY) WITHOUT ROWID",
        ):
            pass
        with self._owned_cursor(self._scratch, "DELETE FROM temp.user_effect_removals"):
            pass
        with closing(
            self._scratch.executemany(
                "INSERT INTO temp.user_effect_removals(session_id) VALUES (?)", ((value,) for value in permitted)
            )
        ):
            pass
        if lifecycle:
            if new is None:
                raise ReferenceSealError("lifecycle effects must retain the validated removal request/history")
            target, parameters = self.source_literal_expression(new["target_ref"])
            with self._owned_cursor(
                self._scratch,
                f"SELECT typeof({target})='text' AND substr({target},1,8)='session:' AND EXISTS(SELECT 1 FROM temp.user_effect_removals WHERE session_id=substr({target},9))",
                parameters * 3,
            ) as cursor:
                target_allowed = bool(cursor.fetchone()[0])
            if not target_allowed:
                raise ReferenceSealError("lifecycle effect has no exact permitted session target")
            if old is not None:
                immutable = set(effect.columns) - {"value_json", "status", "updated_at_ms"}
                if any(not self._literal_cells_equal(old[column], new[column]) for column in immutable):
                    raise ReferenceSealError("lifecycle update changed undeclared assertion provenance")
            elif (
                not self._literal_scalar_equal(new["scope_ref"], None)
                or not self._literal_scalar_equal(new["author_ref"], "user:local")
                or not any(self._literal_scalar_equal(new["evidence_refs_json"], value) for value in ("[]", "null"))
            ):
                raise ReferenceSealError("new removal receipt introduces unrelated reference provenance")
            return
        if old is None:
            raise ReferenceSealError("content assertion effects require the bound excision owner")
        assertion_id, id_parameters = self.source_literal_expression(old["assertion_id"])
        target, target_parameters = self.source_literal_expression(old["target_ref"])
        with self._owned_cursor(
            self._scratch,
            f"SELECT 1 FROM temp.reference_anchors a JOIN temp.resolved_refs r ON r.wire_ref=a.wire_ref WHERE a.tier='user' AND a.assertion_id IS {assertion_id} AND a.field='target_ref' AND a.assertion_target IS {target} AND EXISTS(SELECT 1 FROM temp.user_effect_removals p WHERE p.session_id=r.owner_session_id) AND (r.scope_session_id='' OR EXISTS(SELECT 1 FROM temp.user_effect_removals p WHERE p.session_id=r.scope_session_id)) LIMIT 1",
            (*id_parameters, *target_parameters),
        ) as cursor:
            target_allowed = cursor.fetchone() is not None
        if not target_allowed:
            raise ReferenceSealError("assertion target belongs outside the exact excision closure")
        if new is None:
            return
        changed = {column for column in effect.columns if not self._literal_cells_equal(old[column], new[column])}
        with self._owned_cursor(
            self._scratch,
            f"SELECT typeof({assertion_id})='text' AND substr({assertion_id},1,7)='marker-'",
            id_parameters * 2,
        ) as cursor:
            marker = bool(cursor.fetchone()[0])
        if not marker or not changed.issubset(
            {"target_ref", "value_json", "body_text", "evidence_refs_json", "status", "updated_at_ms"}
        ):
            raise ReferenceSealError("excision assertion update is not its exact marker tombstone")
        new_target, new_parameters = self.source_literal_expression(new["target_ref"])
        with self._owned_cursor(
            self._scratch,
            f"SELECT typeof({new_target})='text' AND {new_target} IS ('assertion:' || {assertion_id})",
            (*new_parameters, *new_parameters, *id_parameters),
        ) as cursor:
            target_matches = bool(cursor.fetchone()[0])
        if (
            not target_matches
            or not self._literal_scalar_equal(new["value_json"], "{}")
            or not self._literal_scalar_equal(new["body_text"], None)
            or not self._literal_scalar_equal(new["evidence_refs_json"], "[]")
            or not self._literal_scalar_equal(new["status"], "deleted")
        ):
            raise ReferenceSealError("marker tombstone retains undeclared content or target")

    def _verify_user_effect_conservation(self) -> None:
        self._require_capability("user")
        with self._owned_cursor(
            self._scratch, "SELECT count(*) FROM temp.known_tier_effects WHERE tier='user' AND table_name='assertions'"
        ) as cursor:
            assertion_count = int(cursor.fetchone()[0])
        with self._owned_cursor(
            self._scratch,
            "SELECT count(*) FROM temp.known_tier_effects WHERE tier='user' AND table_name='query_unit_frame_state'",
        ) as cursor:
            frame_count = int(cursor.fetchone()[0])
        if assertion_count != frame_count:
            raise ReferenceSealError("User effects omit an assertion frame trigger transition")

    def _retire_user_fields(self, *, projected: bool) -> None:
        """Retire only provenance physically removed by the accepted exact effects."""
        self._require_capability("user")
        if not projected:
            # Preparation proved each field against the original assertion
            # provenance. Only its accepted receipt promotes that projection.
            with self._owned_cursor(
                self._scratch, "UPDATE temp.reference_anchors SET retired=1 WHERE tier='user' AND projected_retired=1"
            ):
                pass
            return
        field_column = "projected_retired"
        with self._owned_cursor(
            self._scratch,
            "SELECT old_image,new_image FROM temp.known_tier_effects WHERE tier='user' AND table_name='assertions'",
        ) as rows:
            for old_image_id, new_image_id in rows:
                _check_reference_cancellation()
                if old_image_id is None:
                    continue
                old_image = self._retained_row_image(old_image_id)
                old = dict(zip(old_image.columns, old_image.cells, strict=True))
                assertion_id, parameters = self.source_literal_expression(old["assertion_id"])
                if new_image_id is None:
                    with self._owned_cursor(
                        self._scratch,
                        f"UPDATE temp.reference_anchors SET {field_column}=1 WHERE tier='user' AND assertion_id IS {assertion_id}",
                        parameters,
                    ):
                        pass
                    continue
                new_image = self._retained_row_image(new_image_id)
                new = dict(zip(new_image.columns, new_image.cells, strict=True))
                for field in ("target_ref", "scope_ref", "author_ref", "evidence_refs_json"):
                    if not self._literal_cells_equal(old[field], new[field]):
                        with self._owned_cursor(
                            self._scratch,
                            f"UPDATE temp.reference_anchors SET {field_column}=1 WHERE tier='user' AND assertion_id IS {assertion_id} AND field=?",
                            (*parameters, field),
                        ):
                            pass

    def _provision_source_capture(self) -> None:
        """Install physical capture on this original owner's canonical Source tables."""
        if self._source_capture_tables:
            return
        if not self._source_producer_active or self._source_statement_active:
            raise ReferenceSealError("Source capture setup requires its idle original producer")
        with self._owned_cursor(
            self._observers["source"],
            "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
        ) as cursor:
            tables = tuple(str(row[0]) for row in cursor)
        try:
            for number, table in enumerate(tables):
                _check_reference_cancellation()
                columns, _keys = self._known_tier_table_shape("source", table)
                alias = self._physical_rowid_alias(self._scratch, table, columns)
                function = f"polylogue_source_capture_{number}"

                def capture(
                    phase: str, operation: str, old_rowid: int | None, new_rowid: int | None, *, table: str = table
                ) -> int:
                    if self._source_baseline_active or not self._source_statement_active:
                        return 1
                    if self._source_capture_internal:
                        raise ReferenceSealError("Source capture re-entered a canonical producer mutation")
                    self._source_capture_internal = True
                    try:
                        return self._capture_source_transition(table, phase, operation, old_rowid, new_rowid)
                    except BaseException as failure:
                        self._source_capture_failure = failure
                        raise
                    finally:
                        self._source_capture_internal = False

                self._scratch.create_function(function, 4, capture)
                for operation in ("INSERT", "UPDATE", "DELETE"):
                    old = "NULL" if operation == "INSERT" else f"OLD.{quote_identifier(alias)}"
                    new = "NULL" if operation == "DELETE" else f"NEW.{quote_identifier(alias)}"
                    for phase in ("BEFORE", "AFTER"):
                        name = f"polylogue_source_stage_{number}_{operation.lower()}_{phase.lower()}"
                        with self._owned_cursor(
                            self._scratch,
                            f"CREATE TEMP TRIGGER {name} {phase} {operation} ON main.{quote_identifier(table)} "
                            f"BEGIN SELECT {function}('{phase}','{operation}',{old},{new}); END",
                        ):
                            pass
                self._source_capture_tables[table] = (columns, number)
            with self._owned_cursor(
                self._scratch,
                "CREATE INDEX IF NOT EXISTS temp.known_tier_capture_pending "
                "ON known_tier_effects(tier,table_name,consumed,effect_id)",
            ):
                pass
        except BaseException:
            self._cleanup_requested = True
            raise

    def _source_capture_parent(
        self, table: str, operation: str, child: KnownTierRowImage | None
    ) -> tuple[int | None, str | None]:
        if table == "raw_existence_changes" and operation == "INSERT":
            parent_table = "raw_sessions"
        elif table == "raw_existence_journal_control" and operation == "UPDATE":
            parent_table = "raw_existence_changes"
        else:
            return None, None
        with self._owned_cursor(
            self._scratch,
            "SELECT effect_id,old_image,new_image,new_rowid FROM temp.known_tier_effects "
            "WHERE tier='source' AND table_name=? AND consumed IN (-2,-1) AND old_image IS NOT NULL "
            "ORDER BY effect_id DESC LIMIT 1",
            (parent_table,),
        ) as cursor:
            parent = cursor.fetchone()
        if parent is None:
            raise ReferenceSealError("canonical Source child lacks its actual captured BEFORE parent")
        old = self._retained_row_image(parent[1])
        new = None if parent[2] is None else self._retained_row_image(parent[2])
        if table == "raw_existence_journal_control":
            if parent[3] is not None:
                raise ReferenceSealError("journal pruning child has a surviving parent transition")
            trigger = "raw_existence_journal_prune"
        elif parent[3] is None:
            trigger = "raw_existence_delete"
        else:
            trigger = "raw_existence_key_change"
            # Canonical main AFTER may execute before TEMP parent AFTER.
            # Its physical NEW already exists. Retain it under the same OLD
            # BEFORE authority without assuming an ordering between triggers.
            if new is None:
                new = self._retain_native_row(self._scratch, old.table, old.columns, parent[3], reuse=old)
                if new is None:
                    raise ReferenceSealError("canonical key-change parent has no physical NEW image")
                with self._owned_cursor(
                    self._scratch,
                    "UPDATE temp.known_tier_effects SET new_image=? WHERE effect_id=?",
                    (self._retain_row_image(new), parent[0]),
                ):
                    pass
        if child is not None:
            self._validate_source_trigger_relation(trigger, old, new, table, child)
        return parent[0], trigger

    def _capture_source_transition(
        self, table: str, phase: str, operation: str, old_rowid: int | None, new_rowid: int | None
    ) -> int:
        self._require_new_work()
        if phase not in {"BEFORE", "AFTER"} or operation not in {"INSERT", "UPDATE", "DELETE"}:
            raise ReferenceSealError("Source capture has no canonical physical phase")
        if any(value is not None and type(value) is not int for value in (old_rowid, new_rowid)):
            raise ReferenceSealError("Source capture omitted actual SQLite physical rowids")
        if phase == "BEFORE" and operation == "INSERT":
            return 1
        columns, _number = self._source_capture_tables[table]
        if phase == "BEFORE":
            assert old_rowid is not None
            with self._owned_cursor(
                self._scratch,
                "SELECT coalesce(current_image,input_image) FROM temp.polylogue_source_stage_rows "
                "WHERE table_name=? AND physical_rowid=?",
                (table, old_rowid),
            ) as cursor:
                prior = cursor.fetchone()
            reuse = None if prior is None or prior[0] is None else self._retained_row_image(prior[0])
            old = self._retain_native_row(self._scratch, table, columns, old_rowid, reuse=reuse)
            if old is None:
                raise ReferenceSealError("Source BEFORE capture lost its actual physical OLD")
            parent, trigger = self._source_capture_parent(table, operation, None)
            self._record_source_stage_before(
                table, operation, old, new_rowid, parent_effect_id=parent, canonical_trigger=trigger
            )
            return 1
        if (
            operation == "INSERT"
            and table == self._source_statement_table
            and self._source_statement_allocation_parameter is not None
            and new_rowid is not None
            and not self.source_row_is_touched(table, new_rowid)
        ):
            # Only this actual allocated root coordinate can trigger a
            # preparation retry. Trigger allocations and physical failures
            # keep their original refusal; live publication never retries.
            alias = self._physical_rowid_alias(self._observers["source"], table, columns)
            with self._owned_cursor(
                self._observers["source"],
                f"SELECT 1 FROM {quote_identifier(table)} WHERE {quote_identifier(alias)}=?",
                (new_rowid,),
            ) as original:
                occupied = original.fetchone() is not None
            if occupied:
                raise _SourceAllocationCollisionError(table, new_rowid)
        reuse = None
        if operation != "INSERT":
            with self._owned_cursor(
                self._scratch,
                "SELECT old_image FROM temp.known_tier_effects WHERE tier='source' AND table_name=? "
                "AND old_rowid IS ? AND new_rowid IS ? AND consumed=-2 ORDER BY effect_id DESC LIMIT 1",
                (table, old_rowid, new_rowid),
            ) as cursor:
                pending = cursor.fetchone()
            if pending is None:
                raise ReferenceSealError("Source AFTER capture lacks its actual preceding OLD")
            reuse = self._retained_row_image(pending[0])
        prepared = self._source_statement_prepared_cells if table == self._source_statement_table else None
        new = (
            None
            if new_rowid is None
            else self._retain_native_row(self._scratch, table, columns, new_rowid, reuse=reuse, prepared_cells=prepared)
        )
        if operation != "DELETE" and new is None:
            raise ReferenceSealError("Source AFTER capture lost its actual physical NEW")
        parent, trigger = self._source_capture_parent(table, operation, new)
        self._record_source_stage_after(
            table, operation, old_rowid, new, parent_effect_id=parent, canonical_trigger=trigger
        )
        return 1

    def _source_only_original_exists(self, image: KnownTierRowImage) -> bool:
        """Use the actual pinned Source coordinate, never missing foreign tiers."""
        alias = self._physical_rowid_alias(self._observers["source"], image.table, image.columns)
        with self._owned_cursor(
            self._observers["source"],
            f"SELECT 1 FROM {quote_identifier(image.table)} WHERE {quote_identifier(alias)}=?",
            (image.rowid,),
        ) as cursor:
            return cursor.fetchone() is not None

    def _validate_source_only_transition(
        self,
        table: str,
        old: KnownTierRowImage | None,
        new: KnownTierRowImage | None,
    ) -> None:
        if "index" in self._capabilities:
            return
        # Membership and census replacement are Source metadata operations.
        # Original durable carrier identities require proofs this capability
        # does not own. Newly staged inputs still undergo the complete tape,
        # FK, allocation and terminal postimage proofs before publication.
        carrier_fields = {
            "raw_sessions": (
                "raw_id",
                "native_id",
                "source_path",
                "source_index",
                "blob_hash",
                "blob_size",
                "canonical_source_path",
            ),
            "raw_artifacts": ("artifact_id", "origin", "source_path", "source_index"),
            "raw_hook_events": (
                "hook_event_id",
                "origin",
                "native_id",
                "session_native_id",
                "source_path",
                "payload_json",
                "blob_hash",
            ),
            "hook_event_carriers": (
                "source_id",
                "relative_path",
                "hook_event_id",
                "blob_hash",
                "payload_digest",
                "carrier_role",
            ),
            "history_sidecars": ("sidecar_id", "origin", "source_path", "payload_json", "content_hash"),
            "source_attachments": (
                "source_generation_id",
                "reference_id",
                "payload_identity",
                "blob_hash",
                "byte_count",
            ),
            "material_observations": ("material_id", "blob_hash", "supersedes_material_id"),
            "material_evidence_links": ("material_id", "evidence_ref", "relation"),
        }
        if table == "raw_sessions" and new is not None:
            values = dict(zip(new.columns, new.cells, strict=True))
            if not any(self._literal_scalar_equal(values["origin"], origin.value) for origin in Origin):
                raise ReferenceSealError("Source-only raw transition lacks a canonical origin")
        if table == "raw_profile_identity_receipts" and new is not None:
            values = dict(zip(new.columns, new.cells, strict=True))
            expression, parameters = self.source_literal_expression(values["profile_key"])
            with self._owned_cursor(
                self._scratch,
                f"SELECT typeof({expression})='text' AND length({expression})=12 "
                f"AND {expression} NOT GLOB '*[^0123456789abcdef]*'",
                parameters * 3,
            ) as cursor:
                if not cursor.fetchone()[0]:
                    raise ReferenceSealError("Source profile receipt lacks its canonical qualifier")
        if old is None or not self._source_only_original_exists(old):
            return
        if table == "blob_refs":
            # INSERT OR REPLACE records its actual transient DELETE. Its exact
            # original key must survive the complete preparation, checked at
            # adoption rather than pretending that DELETE did not occur.
            return
        if table == "raw_profile_identity_receipts":
            if not self._row_images_equal(old, new):
                raise ReferenceSealError("Source-only preparation cannot retire or replace an original profile receipt")
            return
        if table not in carrier_fields:
            return
        if new is None:
            raise ReferenceSealError(
                "Source-only preparation lacks foreign reference proof for original carrier retirement"
            )
        if new.rowid != old.rowid:
            raise ReferenceSealError("Source-only preparation cannot replace an original carrier physical identity")
        before = dict(zip(old.columns, old.cells, strict=True))
        after = dict(zip(new.columns, new.cells, strict=True))
        for column in carrier_fields[table]:
            if not self._literal_cells_equal(before[column], after[column]):
                raise ReferenceSealError("Source-only preparation changes original carrier acquisition identity")
        if table == "raw_sessions":
            envelope = self._source_membership_envelope()
            if any(not self._literal_cells_equal(before[column], after[column]) for column in envelope) and (
                not self._literal_scalar_equal(before["revision_kind"], "full")
                or not all(self._literal_scalar_equal(after[column], value) for column, value in envelope.items())
            ):
                raise ReferenceSealError(
                    "Source-only preparation rewrites original revision governance outside canonical membership normalization"
                )
            if not self._literal_cells_equal(before["origin"], after["origin"]) and not self._literal_scalar_equal(
                before["origin"], "unknown-export"
            ):
                raise ReferenceSealError("Source-only preparation changes a confident original origin")
            if not self._literal_scalar_equal(before["file_mtime_ms"], None) and not self._literal_cells_equal(
                before["file_mtime_ms"], after["file_mtime_ms"]
            ):
                raise ReferenceSealError("Source-only preparation changes an established acquisition mtime")
            for column in ("addressing_mode", "content_identity"):
                if not self._literal_scalar_equal(before[column], None) and not self._literal_cells_equal(
                    before[column], after[column]
                ):
                    raise ReferenceSealError("Source-only preparation changes established member acquisition evidence")

    @staticmethod
    def _source_membership_envelope() -> dict[str, None | str]:
        return {
            "logical_source_key": None,
            "revision_kind": "unknown",
            "source_revision": None,
            "predecessor_source_revision": None,
            "predecessor_raw_id": None,
            "baseline_raw_id": None,
            "append_start_offset": None,
            "append_end_offset": None,
            "acquisition_generation": None,
            "revision_authority": "quarantined",
        }

    def _source_only_normalizations(self) -> Iterator[tuple[KnownTierRowImage, KnownTierRowImage]]:
        if "index" in self._capabilities:
            return
        after = 0
        while True:
            with self._owned_cursor(
                self._scratch,
                "SELECT effect_id,old_image,new_image FROM temp.known_tier_effects "
                "WHERE tier='source' AND table_name='raw_sessions' AND old_image IS NOT NULL "
                "AND new_image IS NOT NULL AND effect_id>? ORDER BY effect_id LIMIT 1",
                (after,),
            ) as cursor:
                row = cursor.fetchone()
            if row is None:
                return
            after = row[0]
            old, new = self._retained_row_image(row[1]), self._retained_row_image(row[2])
            before = dict(zip(old.columns, old.cells, strict=True))
            current = dict(zip(new.columns, new.cells, strict=True))
            if self._literal_scalar_equal(before["revision_kind"], "full") and not self._literal_cells_equal(
                before["revision_kind"], current["revision_kind"]
            ):
                yield old, new

    def _load_source_only_normalization_inputs(self) -> None:
        # Run at the end of the original producer window. Complete indexed
        # matching inputs can expose omitted replacements; hydration never
        # resurrects a touched deletion or overwrites a staged update.
        for old, _new in self._source_only_normalizations():
            for table in ("raw_session_memberships", "raw_membership_census", "raw_authority_parser_census"):
                with self.original_rows(
                    "source",
                    f"SELECT rowid FROM {quote_identifier(table)} "
                    "WHERE raw_id=(SELECT raw_id FROM raw_sessions WHERE rowid=?)",
                    (old.rowid,),
                ) as rows:
                    for (rowid,) in rows:
                        _check_reference_cancellation()
                        image = self.retain_tier_row("source", table, rowid)
                        if image is not None:
                            self.load_source_row(image)

    def _verify_source_only_normalizations(self) -> None:
        from polylogue.archive.revision_authority import raw_authority_parser_fingerprint

        fingerprint: str | None = None
        for old, new in self._source_only_normalizations():
            _check_reference_cancellation()
            values = dict(zip(new.columns, new.cells, strict=True))
            if not all(
                self._literal_scalar_equal(values[field], value)
                for field, value in self._source_membership_envelope().items()
            ):
                raise ReferenceSealError("Source-only normalization lacks its exact canonical governance postimage")
            raw, parameters = self.source_literal_expression(values["raw_id"])
            with self._owned_cursor(
                self._scratch,
                f"SELECT 1 FROM main.raw_sessions WHERE raw_id!={raw} "
                f"AND (predecessor_raw_id={raw} OR baseline_raw_id={raw}) LIMIT 1",
                parameters * 3,
            ) as selected:
                if selected.fetchone() is not None:
                    raise ReferenceSealError("Source-only normalization has a selected byte revision dependent")
            # Original matching coordinates are indexed. Suppress only exact
            # touched physical originals, and include all staged new matches
            # above; no whole-tier set or selected-only negative proof.
            with self._owned_cursor(
                self._observers["source"],
                "SELECT child.rowid FROM raw_sessions AS parent JOIN raw_sessions AS child "
                "ON child.raw_id!=parent.raw_id "
                "AND (child.predecessor_raw_id=parent.raw_id OR child.baseline_raw_id=parent.raw_id) "
                "WHERE parent.rowid=?",
                (old.rowid,),
            ) as originals:
                for (rowid,) in originals:
                    _check_reference_cancellation()
                    with self._owned_cursor(
                        self._scratch,
                        "SELECT touched FROM temp.polylogue_source_stage_rows "
                        "WHERE table_name='raw_sessions' AND physical_rowid=?",
                        (rowid,),
                    ) as overlay:
                        touched = overlay.fetchone()
                    if touched is None or not touched[0]:
                        raise ReferenceSealError("Source-only normalization has an original byte revision dependent")
            if fingerprint is None:
                fingerprint = raw_authority_parser_fingerprint()
            with self._owned_cursor(
                self._scratch,
                "SELECT 1 FROM main.raw_membership_census AS m JOIN main.raw_authority_parser_census AS p "
                "ON p.raw_id=m.raw_id "
                f"WHERE m.raw_id={raw} AND m.parser_fingerprint=? AND p.parser_fingerprint=? "
                "AND p.status='complete' "
                "AND EXISTS(SELECT 1 FROM temp.known_tier_effects AS captured WHERE captured.tier='source' "
                "AND captured.table_name='raw_membership_census' AND captured.new_rowid=m.rowid AND captured.new_image IS NOT NULL) "
                "AND EXISTS(SELECT 1 FROM temp.known_tier_effects AS captured WHERE captured.tier='source' "
                "AND captured.table_name='raw_authority_parser_census' AND captured.new_rowid=p.rowid AND captured.new_image IS NOT NULL) "
                "AND m.member_count=(SELECT count(*) FROM main.raw_session_memberships AS actual WHERE actual.raw_id=m.raw_id) "
                "AND ((m.member_count=0 AND m.status='non_session') OR "
                "(m.member_count>0 AND m.status='complete' AND m.revision_authority='quarantined')) "
                "AND json_type(p.logical_keys_json)='array' "
                "AND json_array_length(p.logical_keys_json)=m.member_count "
                "AND NOT EXISTS(SELECT 1 FROM json_each(p.logical_keys_json) AS observed "
                "WHERE observed.type!='text' OR NOT EXISTS(SELECT 1 FROM main.raw_session_memberships AS actual "
                "WHERE actual.raw_id=m.raw_id AND actual.logical_source_key=observed.value)) "
                "AND (SELECT count(DISTINCT observed.value) FROM json_each(p.logical_keys_json) AS observed)=m.member_count",
                (*parameters, fingerprint, fingerprint),
            ) as census:
                if census.fetchone() is None:
                    raise ReferenceSealError(
                        "Source-only normalization lacks its complete recognized matching membership/parser census"
                    )

    def _verify_source_only_blob_reference_restoration(self) -> None:
        if "index" in self._capabilities:
            return
        with self._owned_cursor(
            self._scratch,
            "SELECT old_image FROM temp.known_tier_effects "
            "WHERE tier='source' AND table_name='blob_refs' AND old_image IS NOT NULL ORDER BY ordinal",
        ) as effects:
            for (image_id,) in effects:
                _check_reference_cancellation()
                old = self._retained_row_image(image_id)
                if not self._source_only_original_exists(old):
                    continue
                values = dict(zip(old.columns, old.cells, strict=True))
                predicates = []
                parameters: tuple[object, ...] = ()
                for field in ("blob_hash", "ref_type", "ref_id"):
                    expression, operands = self.source_literal_expression(values[field])
                    predicates.append(f"{quote_identifier(field)} IS {expression}")
                    parameters += operands
                with self._owned_cursor(
                    self._scratch, f"SELECT 1 FROM main.blob_refs WHERE {' AND '.join(predicates)}", parameters
                ) as surviving:
                    if surviving.fetchone() is None:
                        raise ReferenceSealError("Source-only preparation retires an original blob reference key")

    def _record_source_stage_before(
        self,
        table: str,
        operation: str,
        old: KnownTierRowImage,
        new_rowid: int | None,
        *,
        parent_effect_id: int | None = None,
        canonical_trigger: str | None = None,
    ) -> int:
        """Retain actual OLD before SQLite changes or cascades its selected row."""
        columns, keys = self._known_tier_table_shape("source", table)
        if old._seal is not self or old.table != table or old.columns != columns:
            raise ReferenceSealError("Source capture omitted its original complete OLD image")
        if operation not in {"UPDATE", "DELETE"} or type(old.rowid) is not int:
            raise ReferenceSealError("Source BEFORE capture has no exact existing physical row")
        if operation == "DELETE" and new_rowid is not None:
            raise ReferenceSealError("Source deletion invents a NEW physical row")
        if operation == "UPDATE" and type(new_rowid) is not int:
            raise ReferenceSealError("Source update omits its actual NEW physical rowid")
        if operation == "DELETE":
            self._validate_source_only_transition(table, old, None)
        if parent_effect_id is None and not self._source_image_has_writable_role(old):
            raise ReferenceSealError("readable Source dependency is outside declared writable keys")
        with self._owned_cursor(
            self._scratch,
            "INSERT OR IGNORE INTO temp.known_tier_effect_tables VALUES(?,?,?,?)",
            ("source", table, pickle.dumps(columns, protocol=5), pickle.dumps(keys, protocol=5)),
        ):
            pass
        old_image = self._retain_row_image(old)
        with self._owned_cursor(
            self._scratch, "SELECT coalesce(max(ordinal),0)+1 FROM temp.known_tier_effects WHERE tier='source'"
        ) as cursor:
            ordinal = cursor.fetchone()[0]
        with self._owned_cursor(
            self._scratch,
            "INSERT INTO temp.known_tier_effects("
            "tier,ordinal,table_name,old_image,old_rowid,new_rowid,consumed,parent_effect_id,canonical_trigger) "
            "VALUES('source',?,?,?,?,?,-2,?,?)",
            (ordinal, table, old_image, old.rowid, new_rowid, parent_effect_id, canonical_trigger),
        ) as cursor:
            effect_id = cursor.lastrowid
        if type(effect_id) is not int:
            raise ReferenceSealError("Source capture has no original native effect identity")
        return effect_id

    def _record_source_stage_after(
        self,
        table: str,
        operation: str,
        old_rowid: int | None,
        new: KnownTierRowImage | None,
        *,
        parent_effect_id: int | None = None,
        canonical_trigger: str | None = None,
    ) -> None:
        """Finish the same ordered tape with SQLite's complete actual NEW."""
        columns, keys = self._known_tier_table_shape("source", table)
        if operation not in {"INSERT", "UPDATE", "DELETE"}:
            raise ReferenceSealError("Source AFTER capture has no declared physical operation")
        if new is not None and (
            new._seal is not self or new.table != table or new.columns != columns or type(new.rowid) is not int
        ):
            raise ReferenceSealError("Source capture omitted its original complete NEW image")
        if operation == "DELETE" and new is not None or operation != "DELETE" and new is None:
            raise ReferenceSealError("Source AFTER capture disagrees with its physical operation")
        if operation == "INSERT":
            self._validate_source_only_transition(table, None, new)
        elif operation == "UPDATE":
            with self._owned_cursor(
                self._scratch,
                "SELECT old_image FROM temp.known_tier_effects WHERE tier='source' AND table_name=? "
                "AND old_rowid IS ? AND new_rowid IS ? AND consumed=-2 ORDER BY effect_id DESC LIMIT 1",
                (table, old_rowid, None if new is None else new.rowid),
            ) as pending_old:
                retained_old = pending_old.fetchone()
            if retained_old is None:
                raise ReferenceSealError("Source-only transition omitted its complete captured OLD")
            self._validate_source_only_transition(table, self._retained_row_image(retained_old[0]), new)
        new_image = None if new is None else self._retain_row_image(new)
        new_rowid = None if new is None else new.rowid
        if new is not None and parent_effect_id is None and not self._source_image_has_writable_role(new):
            raise ReferenceSealError("Source NEW key is outside the exact declared writable keys")
        if operation == "INSERT":
            if old_rowid is not None:
                raise ReferenceSealError("Source insertion invents an OLD physical row")
            with self._owned_cursor(
                self._scratch,
                "INSERT OR IGNORE INTO temp.known_tier_effect_tables VALUES(?,?,?,?)",
                ("source", table, pickle.dumps(columns, protocol=5), pickle.dumps(keys, protocol=5)),
            ):
                pass
            with self._owned_cursor(
                self._scratch, "SELECT coalesce(max(ordinal),0)+1 FROM temp.known_tier_effects WHERE tier='source'"
            ) as cursor:
                ordinal = cursor.fetchone()[0]
            with self._owned_cursor(
                self._scratch,
                "INSERT INTO temp.known_tier_effects("
                "tier,ordinal,table_name,new_image,new_rowid,consumed,parent_effect_id,canonical_trigger) "
                "VALUES('source',?,?,?,?,-1,?,?)",
                (ordinal, table, new_image, new_rowid, parent_effect_id, canonical_trigger),
            ):
                pass
        else:
            with self._owned_cursor(
                self._scratch,
                "SELECT effect_id,parent_effect_id,canonical_trigger FROM temp.known_tier_effects "
                "WHERE tier='source' AND table_name=? AND old_rowid IS ? AND new_rowid IS ? "
                "AND consumed=-2 ORDER BY effect_id DESC LIMIT 1",
                (table, old_rowid, new_rowid),
            ) as cursor:
                pending = cursor.fetchone()
            if pending is None or pending[1:] != (parent_effect_id, canonical_trigger):
                raise ReferenceSealError("Source AFTER capture lacks the same exact BEFORE provenance")
            with self._owned_cursor(
                self._scratch,
                "UPDATE temp.known_tier_effects SET new_image=?,consumed=-1 WHERE effect_id=?",
                (new_image, pending[0]),
            ):
                pass
        for rowid, image_id in ((old_rowid, None), (new_rowid, new_image)):
            if rowid is None:
                continue
            with self._owned_cursor(
                self._scratch,
                "INSERT INTO temp.polylogue_source_stage_rows("
                "table_name,physical_rowid,touched,load_state,current_image) VALUES(?,?,1,2,?) "
                "ON CONFLICT(table_name,physical_rowid) DO UPDATE SET touched=1,current_image=excluded.current_image",
                (table, rowid, image_id),
            ):
                pass

    def prepare_source_mutation(self) -> KnownTierMutationPermit:
        """Adopt the captured original Source tape without copying or replanning."""
        self._require_new_work()
        self._require_witness_main_mutable()
        if not self._source_stage_ready or self._source_producer_active or self._original_reads_active:
            raise ReferenceSealError("Source adoption requires its completed original preparation window")
        if "source" in self._pending_tier_permits:
            raise ReferenceSealError("this original seal already has a pending Source mutation")
        identity = self._tier_identity_for_known_mutation("source")
        try:
            with self._owned_cursor(
                self._scratch,
                "SELECT effect_id,ordinal,table_name,old_image,new_image,old_rowid,new_rowid,"
                "consumed,parent_effect_id,canonical_trigger FROM temp.known_tier_effects "
                "WHERE tier='source' ORDER BY ordinal",
            ) as effects:
                for row in effects:
                    _check_reference_cancellation()
                    if row[7] != -1:
                        raise ReferenceSealError("Source adoption contains an incomplete or already consumed effect")
                    old = None if row[3] is None else self._retained_row_image(row[3])
                    new = None if row[4] is None else self._retained_row_image(row[4])
                    columns, _keys = self._known_tier_table_shape("source", row[2])
                    if (
                        old is None
                        and new is None
                        or any(
                            image is not None and (image.columns != columns or type(image.rowid) is not int)
                            for image in (old, new)
                        )
                    ):
                        raise ReferenceSealError("captured Source effect omits its complete physical image")
                    target = row[6] if old is None else row[5]
                    with self._owned_cursor(
                        self._scratch,
                        "SELECT new_rowid,new_image FROM temp.known_tier_effects "
                        "WHERE tier='source' AND table_name=? AND effect_id<? "
                        "AND (old_rowid=? OR new_rowid=?) ORDER BY effect_id DESC LIMIT 1",
                        (row[2], row[0], target, target),
                    ) as cursor:
                        prior = cursor.fetchone()
                    if prior is not None:
                        predecessor = (
                            None if prior[1] is None or prior[0] != target else self._retained_row_image(prior[1])
                        )
                        matches = self._row_images_equal(predecessor, old)
                    elif old is not None:
                        matches = self._matches_retained_row(self._observers["source"], old)
                    else:
                        alias = self._physical_rowid_alias(self._observers["source"], row[2], columns)
                        with self._owned_cursor(
                            self._observers["source"],
                            f"SELECT 1 FROM {quote_identifier(row[2])} WHERE {quote_identifier(alias)}=?",
                            (target,),
                        ) as cursor:
                            matches = cursor.fetchone() is None
                    if not matches:
                        raise ReferenceSealStaleError(
                            "captured Source OLD differs from its original or preceding effect"
                        )
                    parent_ordinal = None
                    if row[8] is not None:
                        with self._owned_cursor(
                            self._scratch,
                            "SELECT ordinal FROM temp.known_tier_effects WHERE effect_id=? AND tier='source'",
                            (row[8],),
                        ) as cursor:
                            parent = cursor.fetchone()
                        if parent is None:
                            raise ReferenceSealError("captured Source effect lost its exact parent identity")
                        parent_ordinal = parent[0]
                    effect = KnownTierRowEffect(row[2], columns, old, new, parent_ordinal, row[9])
                    if self._prepare_source_trigger_relation("source", row[1], effect) != row[8]:
                        raise ReferenceSealError("captured Source child changed its canonical parent provenance")
            self._verify_source_only_normalizations()
            self._verify_source_only_blob_reference_restoration()
            self._verify_implicit_sequence_effects("source")
            self._verify_known_tier_postimage(self._scratch, "source")
            self.validate_observers_current()
            with self._owned_cursor(self._scratch, "UPDATE temp.known_tier_effects SET consumed=0 WHERE tier='source'"):
                pass
            self._settle_witness_metadata()
        except BaseException:
            self._cleanup_requested = True
            raise
        permit = KnownTierMutationPermit(
            self, identity, self._versions["source"], "", (), (), self._tier_mutation_nonce, "", "source"
        )
        self._pending_tier_permits["source"] = permit
        return permit

    def _provision_effect_metadata(self) -> None:
        """Share the original native effect tape without implicit script commits."""
        self._require_new_work()
        statements = (
            "CREATE TEMP TABLE IF NOT EXISTS temp.known_tier_effect_tables("
            "tier TEXT NOT NULL, table_name TEXT NOT NULL, columns_blob BLOB NOT NULL, keys_blob BLOB NOT NULL, "
            "PRIMARY KEY(tier,table_name)) WITHOUT ROWID",
            "CREATE TEMP TABLE IF NOT EXISTS temp.known_tier_effects("
            "effect_id INTEGER PRIMARY KEY, tier TEXT NOT NULL, ordinal INTEGER NOT NULL,table_name TEXT NOT NULL, "
            "old_image INTEGER, new_image INTEGER, old_rowid INTEGER, new_rowid INTEGER, "
            "consumed INTEGER NOT NULL DEFAULT 0,parent_effect_id INTEGER,canonical_trigger TEXT)",
            "CREATE TEMP TABLE IF NOT EXISTS known_tier_statements("
            "statement_id INTEGER PRIMARY KEY,sql TEXT NOT NULL,root_table TEXT NOT NULL,"
            "allocation_parameter INTEGER,first_ordinal INTEGER NOT NULL,last_ordinal INTEGER NOT NULL,"
            "consumed INTEGER NOT NULL DEFAULT 0)",
            "CREATE TEMP TABLE IF NOT EXISTS known_tier_statement_bindings("
            "statement_id INTEGER NOT NULL,position INTEGER NOT NULL,cell_id INTEGER NOT NULL,"
            "PRIMARY KEY(statement_id,position)) WITHOUT ROWID",
            "CREATE UNIQUE INDEX IF NOT EXISTS temp.known_tier_effect_ordinal ON known_tier_effects(tier,ordinal)",
            "CREATE INDEX IF NOT EXISTS temp.known_tier_effect_old ON known_tier_effects(tier,table_name,old_rowid,effect_id)",
            "CREATE INDEX IF NOT EXISTS temp.known_tier_effect_new ON known_tier_effects(tier,table_name,new_rowid,effect_id)",
        )
        for statement in statements:
            _check_reference_cancellation()
            with self._owned_cursor(self._scratch, statement):
                pass

    def _rollback_known_tier_preparation(self, primary: BaseException) -> None:
        # Original field demand predates this savepoint. Only pending effects,
        # NEW literals and projected User flags belong to its rollback.
        failures: list[BaseException] = []
        try:
            self._scratch.set_progress_handler(None, 0)
            with self._owned_cursor(self._scratch, "ROLLBACK TO prepare_known_tier_effects"):
                pass
            with self._owned_cursor(self._scratch, "RELEASE prepare_known_tier_effects"):
                pass
        except BaseException as cleanup:
            failures.append(cleanup)
        finally:
            try:
                self._scratch.set_progress_handler(lambda: int(compute_cancel_requested()), 2000)
            except BaseException as cleanup:
                failures.append(cleanup)
        if failures:
            self._cleanup_requested = True
            raise BaseExceptionGroup("Tier preparation and witness rollback failed", [primary, *failures]) from primary

    def prepare_known_tier_mutation(
        self,
        table: str | None = None,
        columns: tuple[str, ...] = (),
        rows: tuple[tuple[object, ...], ...] = (),
        *,
        key_column: str | None = None,
        tier: Literal["source", "user"],
        effects: Iterable[KnownTierRowEffect] | None = None,
    ) -> KnownTierMutationPermit:
        """Retain complete OLD-to-NEW effects against this original observer."""
        self._require_new_work()
        if tier not in {"source", "user"}:
            raise ReferenceSealError("known mutation requires a declared Source or User tier")
        self._require_capability(tier)
        if tier == "source" and "index" not in self._capabilities:
            raise ReferenceSealError("Source-only effects require the actual canonical Source producer tape")
        if tier == "user" and effects is None:
            raise ReferenceSealError("User effects require the exact removal effect witness")
        if effects is None:
            if table is None or key_column is None or not columns:
                raise ReferenceSealError("prepared values require a declared table, columns and key")
            effects = self._updated_row_effects(tier, table, columns, rows, key_column)
        elif table is not None or columns or rows or key_column is not None:
            raise ReferenceSealError("complete row effects cannot also declare partial prepared values")
        identity = self._tier_identity_for_known_mutation(tier)
        if tier in self._pending_tier_permits:
            raise ReferenceSealError("this original seal already has a pending known tier mutation")
        self._provision_effect_metadata()
        savepoint_active = False
        released_uncommitted = False
        try:
            with self.original_read_snapshot():
                if table is not None:
                    assert key_column is not None
                    complete, keys = self._known_tier_table_shape(tier, table)
                    if (
                        tuple(complete[position] for position in keys) != (key_column,)
                        or any(column not in complete for column in columns)
                        or key_column in columns
                    ):
                        raise ReferenceSealError("prepared values do not match their declared table key and columns")
                    # Paid original inputs live outside the effect rollback savepoint.
                    # Collection below reuses these exact same-epoch descriptors.
                    for values in rows:
                        self._retain_declared_update_target(tier, table, complete, columns, values, key_column)
                with self._owned_cursor(self._scratch, "SAVEPOINT prepare_known_tier_effects"):
                    pass
                savepoint_active = True
                try:
                    if tier == "user":
                        with self._owned_cursor(
                            self._scratch, "UPDATE reference_anchors SET projected_retired=0 WHERE tier='user'"
                        ):
                            pass
                    with self._owned_cursor(self._scratch, "DELETE FROM temp.known_tier_effects WHERE tier=?", (tier,)):
                        pass
                    with self._owned_cursor(
                        self._scratch, "DELETE FROM temp.known_tier_effect_tables WHERE tier=?", (tier,)
                    ):
                        pass
                    for ordinal, effect in enumerate(effects, 1):
                        _check_reference_cancellation()
                        canonical, keys = self._known_tier_table_shape(tier, effect.table)
                        if effect.columns != canonical or (effect.old is None and effect.new is None):
                            raise ReferenceSealError("row effect does not carry the canonical complete table image")
                        for image in (effect.old, effect.new):
                            if image is not None and (
                                image._seal is not self
                                or image.table != effect.table
                                or image.columns != canonical
                                or len(image.cells) != len(canonical)
                                or type(image.rowid) is not int
                            ):
                                raise ReferenceSealError(
                                    "row effect omits canonical literal cells or physical identity"
                                )
                        if tier == "user":
                            self._validate_user_row_effect(effect)
                        old_rowid = None if effect.old is None else effect.old.rowid
                        new_rowid = None if effect.new is None else effect.new.rowid
                        target_rowid = new_rowid if old_rowid is None else old_rowid
                        with self._owned_cursor(
                            self._scratch,
                            "SELECT old_rowid,new_rowid,new_image FROM temp.known_tier_effects WHERE tier=? AND table_name=? AND (old_rowid=? OR new_rowid=?) ORDER BY effect_id DESC LIMIT 1",
                            (tier, effect.table, target_rowid, target_rowid),
                        ) as cursor:
                            prior = cursor.fetchone()
                        if prior is not None:
                            original = (
                                self._retained_row_image(prior[2])
                                if prior[2] is not None and prior[1] == target_rowid
                                else None
                            )
                            matches = self._row_images_equal(original, effect.old)
                        elif effect.old is None:
                            alias = self._physical_rowid_alias(self._observers[tier], effect.table, canonical)
                            with self._owned_cursor(
                                self._observers[tier],
                                f"SELECT 1 FROM {quote_identifier(effect.table)} WHERE {quote_identifier(alias)}=?",
                                (target_rowid,),
                            ) as cursor:
                                matches = cursor.fetchone() is None
                        else:
                            matches = self._matches_retained_row(self._observers[tier], effect.old)
                        if not matches:
                            raise ReferenceSealStaleError(
                                "declared OLD row differs from the original observer or prior exact effect"
                            )
                        with self._owned_cursor(
                            self._scratch,
                            "INSERT OR IGNORE INTO temp.known_tier_effect_tables VALUES (?,?,?,?)",
                            (tier, effect.table, pickle.dumps(canonical, protocol=5), pickle.dumps(keys, protocol=5)),
                        ):
                            pass
                        parent_effect_id = self._prepare_source_trigger_relation(tier, ordinal, effect)
                        old_image = None if effect.old is None else self._retain_row_image(effect.old)
                        new_image = None if effect.new is None else self._retain_row_image(effect.new)
                        with self._owned_cursor(
                            self._scratch,
                            "INSERT INTO temp.known_tier_effects(tier,ordinal,table_name,old_image,new_image,old_rowid,new_rowid,parent_effect_id,canonical_trigger) VALUES (?,?,?,?,?,?,?,?,?)",
                            (
                                tier,
                                ordinal,
                                effect.table,
                                old_image,
                                new_image,
                                old_rowid,
                                new_rowid,
                                parent_effect_id,
                                effect._canonical_trigger,
                            ),
                        ):
                            pass
                    self._verify_implicit_sequence_effects(tier)
                    if tier == "user":
                        self._verify_user_effect_conservation()
                        self._retire_user_fields(projected=True)
                except BaseException as primary:
                    savepoint_active = False
                    self._rollback_known_tier_preparation(primary)
                    raise
            # Original readers have physically settled and their fresh
            # versions still match. Only now can this projection become pending.
            with self._owned_cursor(self._scratch, "RELEASE prepare_known_tier_effects"):
                pass
            savepoint_active = False
            released_uncommitted = True
            self._scratch.commit()
            released_uncommitted = False
        except BaseException as primary:
            if savepoint_active:
                self._rollback_known_tier_preparation(primary)
            elif released_uncommitted:
                self._cleanup_requested = True
            raise
        permit = KnownTierMutationPermit(
            self,
            identity,
            self._versions[tier],
            table or "",
            columns,
            rows,
            self._tier_mutation_nonce,
            key_column or "",
            tier,
        )
        self._pending_tier_permits[tier] = permit
        return permit

    def _verify_source_trigger_definitions(self, connection: sqlite3.Connection) -> None:
        with self._owned_cursor(
            self._scratch,
            "SELECT 1 FROM temp.known_tier_effect_tables WHERE tier='source' "
            "AND table_name IN ('raw_sessions','raw_existence_changes','raw_existence_journal_control') LIMIT 1",
        ) as cursor:
            needs_canonical = cursor.fetchone() is not None
        if not needs_canonical:
            return
        if not self._source_stage_ready:
            raise ReferenceSealError("Source implicit effects require the canonical prepared producer state")
        for trigger in ("raw_existence_delete", "raw_existence_key_change", "raw_existence_journal_prune"):
            with self._owned_cursor(
                self._scratch, "SELECT sql FROM main.sqlite_schema WHERE type='trigger' AND name=?", (trigger,)
            ) as cursor:
                canonical = cursor.fetchone()
            with self._owned_cursor(
                connection, "SELECT sql FROM main.sqlite_schema WHERE type='trigger' AND name=?", (trigger,)
            ) as cursor:
                actual = cursor.fetchone()
            if canonical is None or actual != canonical:
                raise ReferenceSealStaleError("Source implicit effect trigger differs from the canonical prepared DDL")

    def _prepare_source_trigger_relation(
        self,
        tier: str,
        ordinal: int,
        effect: KnownTierRowEffect,
    ) -> int | None:
        implicit_child = tier == "source" and (
            (effect.table == "raw_existence_changes" and effect.old is None)
            or effect.table == "raw_existence_journal_control"
        )
        if not implicit_child:
            if effect._trigger_parent_ordinal is not None or effect._canonical_trigger is not None:
                raise ReferenceSealError("row effect invents a canonical Source trigger relation")
            return None
        parent_ordinal = effect._trigger_parent_ordinal
        if type(parent_ordinal) is not int or not 1 <= parent_ordinal < ordinal:
            raise ReferenceSealError("canonical Source child requires its preceding exact parent effect")
        with self._owned_cursor(
            self._scratch,
            "SELECT effect_id,old_image,new_image FROM temp.known_tier_effects WHERE tier='source' AND ordinal=?",
            (parent_ordinal,),
        ) as cursor:
            parent = cursor.fetchone()
        if parent is None or parent[1] is None or effect._canonical_trigger is None:
            raise ReferenceSealError("canonical Source child omits exact parent provenance")
        self._validate_source_trigger_relation(
            effect._canonical_trigger,
            self._retained_row_image(parent[1]),
            None if parent[2] is None else self._retained_row_image(parent[2]),
            effect.table,
            effect.new,
        )
        parent_id = parent[0]
        if not isinstance(parent_id, int):
            raise ReferenceSealError("canonical Source parent lacks its original effect identity")
        return parent_id

    def _validate_source_trigger_relation(
        self,
        trigger: str,
        old: KnownTierRowImage,
        new: KnownTierRowImage | None,
        table: str,
        child: KnownTierRowImage | None,
    ) -> None:
        """Complete canonical relation; no raw-id subset grants child writes."""
        if child is None:
            raise ReferenceSealError("canonical Source child omits its complete NEW image")
        parent_cells = dict(zip(old.columns, old.cells, strict=True))
        child_cells = dict(zip(child.columns, child.cells, strict=True))
        if trigger in {"raw_existence_delete", "raw_existence_key_change"}:
            if old.table != "raw_sessions" or table != "raw_existence_changes":
                raise ReferenceSealError("journal child has a different canonical parent table")
            if trigger == "raw_existence_delete" and new is not None:
                raise ReferenceSealError("journal delete child requires the exact parent deletion")
            if trigger == "raw_existence_key_change" and (
                new is None or self._literal_cells_equal(parent_cells["raw_id"], new.cells[new.columns.index("raw_id")])
            ):
                raise ReferenceSealError("journal key-change child requires a changed exact parent key")
            if not self._literal_cells_equal(parent_cells["raw_id"], child_cells["raw_id"]):
                raise ReferenceSealError("journal child differs from its exact parent's OLD raw_id")
            if not self._literal_scalar_equal(child_cells["sequence"], child.rowid):
                raise ReferenceSealError("journal sequence differs from its actual INTEGER PRIMARY KEY rowid")
        elif trigger == "raw_existence_journal_prune":
            if old.table != "raw_existence_changes" or new is not None or table != "raw_existence_journal_control":
                raise ReferenceSealError("journal floor change lacks its exact canonical prune parent")
            if not self._literal_scalar_equal(child_cells["singleton"], 1) or child.rowid != 1:
                raise ReferenceSealError("canonical journal floor change targets a different control row")
        else:
            raise ReferenceSealError("Source effect names an unsupported canonical trigger")

    def _tier_identity_for_known_mutation(self, tier: str) -> tuple[int, int, int, int]:
        self._require_capability(tier)
        self.validate_observers_current()
        identity = self._observer_identity(tier)
        if identity != self._identities[tier]:
            raise ReferenceSealStaleError(f"{tier}.db changed before exact tier preparation")
        return identity

    def _verify_implicit_sequence_effects(self, tier: str) -> None:
        """Every actual AUTOINCREMENT advance carries its exact native image."""
        with self._owned_cursor(
            self._scratch,
            "SELECT table_name FROM temp.known_tier_effect_tables WHERE tier=? AND table_name!='sqlite_sequence'",
            (tier,),
        ) as tables:
            observer = self._observers[tier]
            for (table,) in tables:
                _check_reference_cancellation()
                with self._owned_cursor(
                    observer, "SELECT sql FROM sqlite_schema WHERE type='table' AND name=?", (table,)
                ) as cursor:
                    schema = cursor.fetchone()
                if schema is None or "AUTOINCREMENT" not in str(schema[0]).upper():
                    continue
                with self._owned_cursor(observer, "SELECT seq FROM sqlite_sequence WHERE name=?", (table,)) as cursor:
                    original = cursor.fetchone()
                if original is not None and type(original[0]) is not int:
                    raise ReferenceSealError("AUTOINCREMENT state has no exact INTEGER image")
                sequence = 0 if original is None else original[0]
                inserted = False
                with self._owned_cursor(
                    self._scratch,
                    "SELECT new_rowid FROM temp.known_tier_effects WHERE tier=? AND table_name=? AND old_image IS NULL AND new_image IS NOT NULL",
                    (tier, table),
                ) as inserts:
                    for (rowid,) in inserts:
                        _check_reference_cancellation()
                        inserted = True
                        sequence = max(sequence, rowid)
                if not inserted or (original is not None and sequence == original[0]):
                    continue
                declared = False
                with self._owned_cursor(
                    self._scratch,
                    "SELECT new_image FROM temp.known_tier_effects WHERE tier=? AND table_name='sqlite_sequence' AND new_image IS NOT NULL ORDER BY effect_id DESC",
                    (tier,),
                ) as sequences:
                    for (image_id,) in sequences:
                        image = self._retained_row_image(image_id)
                        if self._literal_scalar_equal(image.cells[0], table):
                            declared = self._literal_scalar_equal(image.cells[1], sequence)
                            break
                if not declared:
                    raise ReferenceSealError("AUTOINCREMENT insertion omits its exact internal sequence effect")

    def _verify_known_tier_postimage(self, connection: sqlite3.Connection, tier: str) -> None:
        # Every touched physical coordinate has one terminal presence/image.
        # A rowid change checks both the retired coordinate and its new one.
        with closing(
            self._scratch.execute(
                "SELECT table_name,old_rowid FROM temp.known_tier_effects WHERE tier=? AND old_rowid IS NOT NULL "
                "UNION SELECT table_name,new_rowid FROM temp.known_tier_effects WHERE tier=? AND new_rowid IS NOT NULL",
                (tier, tier),
            )
        ) as coordinates:
            for table, rowid in coordinates:
                _check_reference_cancellation()
                with closing(
                    self._scratch.execute(
                        "SELECT new_rowid,new_image FROM temp.known_tier_effects "
                        "WHERE tier=? AND table_name=? AND (old_rowid=? OR new_rowid=?) "
                        "ORDER BY effect_id DESC LIMIT 1",
                        (tier, table, rowid, rowid),
                    )
                ) as cursor:
                    expected = cursor.fetchone()
                assert expected is not None
                if expected[0] == rowid and expected[1] is not None:
                    matches = self._matches_retained_row(connection, self._retained_row_image(expected[1]))
                else:
                    columns, _keys = self._known_tier_table_shape(tier, table)
                    alias = self._physical_rowid_alias(connection, table, columns)
                    with closing(
                        connection.execute(
                            f"SELECT 1 FROM {quote_identifier(table)} WHERE {quote_identifier(alias)}=?", (rowid,)
                        )
                    ) as cursor:
                        matches = cursor.fetchone() is None
                if not matches:
                    raise ReferenceSealStaleError("tier post-image differs from its complete declared literal effects")

    def _record_known_tier_commit(self, permit: KnownTierMutationPermit) -> KnownTierMutationReceipt:
        self._require_live_owner()
        if (
            permit is not self._pending_tier_permits.get(permit._tier)
            or permit._seal_nonce is not self._tier_mutation_nonce
        ):
            raise ReferenceSealError("Tier writer used a permit outside its prepared seal")
        if permit._tier in self._pending_tier_receipts:
            raise ReferenceSealError("known tier mutation permit was already committed")
        if (
            permit._custody is None
            or current_sql_custody() is not permit._custody
            or permit._custody.known_tier_authority is not permit
            or permit._connection is None
            or permit._connection.in_transaction
            or not permit._commit_allowed
            or not permit._commit_attempted
            or permit._failure is not None
        ):
            raise ReferenceSealError("Tier receipt requires its actual completed dedicated transaction")
        receipt = KnownTierMutationReceipt(
            self,
            permit._tier_identity,
            permit._prior_data_version,
            permit._table,
            permit._columns,
            permit._rows,
            permit._seal_nonce,
            permit._key_column,
            permit._effects,
            permit._tier,
        )
        self._pending_tier_receipts[permit._tier] = receipt
        return receipt

    def accept_known_tier_commit(self, receipt: KnownTierMutationReceipt) -> None:
        """Settle an already committed tier receipt even after cancellation."""
        self._require_live_owner()
        permit = self._pending_tier_permits.get(receipt._tier)
        if permit is None or receipt is not self._pending_tier_receipts.get(receipt._tier):
            raise ReferenceSealError("tier acceptance requires its original pending receipt")
        # Settlement advances the original observer under the same dedicated
        # writer reservation. Cancellation cannot interrupt that advancement
        # after the durable producer has already committed.
        settlement_connections = (*self._observers.values(), self._scratch)
        for connection in settlement_connections:
            connection.set_progress_handler(None, 0)
        try:
            with permit.acceptance_reservation():
                self._accept_known_tier_commit(receipt)
        finally:
            for connection in settlement_connections:
                connection.set_progress_handler(lambda: int(compute_cancel_requested()), 2000)

    def _accept_known_tier_commit(self, receipt: KnownTierMutationReceipt) -> None:
        if (
            receipt._seal is not self
            or receipt is not self._pending_tier_receipts.get(receipt._tier)
            or receipt._seal_nonce is not self._tier_mutation_nonce
            or receipt._tier_identity != self._identities[receipt._tier]
            or receipt._prior_data_version != self._versions[receipt._tier]
        ):
            raise ReferenceSealError("Tier commit receipt does not belong to this prepared seal")
        self._assert_configured_namespace()
        identity_before = _tier_identity(self._paths[receipt._tier])
        if not _same_incarnation(identity_before, receipt._tier_identity):
            raise ReferenceSealStaleError("durable tier incarnation changed during the prepared publication")
        observer = self._require_unpinned_observer(receipt._tier)
        permit = self._pending_tier_permits.get(receipt._tier)
        if (
            permit is None
            or permit._custody is None
            or current_sql_custody() is not permit._custody
            or permit._custody.known_tier_authority is not permit
        ):
            raise ReferenceSealError("Tier acceptance lost its exact physical mutation authority")
        for tier in self._observers:
            if tier == receipt._tier:
                continue
            sibling = self._require_unpinned_observer(tier)
            if self._observer_identity(tier) != self._identities[tier]:
                raise ReferenceSealStaleError(f"{tier}.db changed during known tier publication")
            with closing(sibling.execute("PRAGMA data_version")) as cursor:
                version = int(cursor.fetchone()[0])
            if version != self._versions[tier]:
                raise ReferenceSealStaleError(f"{tier}.db changed during known tier publication")
        with closing(observer.execute("PRAGMA data_version")) as cursor:
            version_before = int(cursor.fetchone()[0])
        if receipt._effect_count == 0 and version_before != receipt._prior_data_version:
            raise ReferenceSealStaleError("a no-op tier transaction cannot absorb another mutation")
        if receipt._effect_count and version_before == receipt._prior_data_version:
            raise ReferenceSealStaleError("Tier effects did not reach an actual committed transaction")
        self._assert_configured_namespace()
        identity_after = _tier_identity(self._paths[receipt._tier])
        with closing(observer.execute("PRAGMA data_version")) as cursor:
            version_after = int(cursor.fetchone()[0])
        if not _same_incarnation(identity_before, identity_after) or version_after != version_before:
            raise ReferenceSealStaleError("durable tier changed while its original observer was advancing")
        # Exact consumed effects and terminal postimages were mandatory before
        # commit. The same writer's unchanged data_version and retained write
        # reservation exclude a foreign commit through this baseline advance.
        if receipt._tier == "user":
            self._retire_user_fields(projected=False)
            self._settle_witness_metadata()
        self._identities[receipt._tier] = identity_after
        self._versions[receipt._tier] = version_after
        self._original_input_epochs[receipt._tier] += 1
        self._pending_tier_permits.pop(receipt._tier)
        self._pending_tier_receipts.pop(receipt._tier)
        if receipt._tier == "source":
            self._source_stage_receipt_accepted = True
        object.__setattr__(permit, "_receipt_accepted", True)

    def validate_reachability(self, conn: sqlite3.Connection) -> None:
        self._validate_reachability(conn, projected_user_effects=False)

    def preflight_reachability(self, conn: sqlite3.Connection) -> None:
        """Compare staged Index against only this bound User effect projection."""
        if "user" not in self._pending_tier_permits:
            raise ReferenceSealError("projected User preflight requires its original exact pending effects")
        self._validate_reachability(conn, projected_user_effects=True)

    def _validate_reachability(self, conn: sqlite3.Connection, *, projected_user_effects: bool) -> None:
        self._require_capability("index")
        self._require_new_work()
        if not _same_incarnation(self._writer_identity(conn), self.index_identity):
            raise ReferenceSealStaleError("reference seal settled on a different index incarnation")
        with closing(
            self._scratch.execute(
                "SELECT kind, owner_session_id, object_id, qualifier, scope_session_id, target_message_id, wire_ref, has_session_alias "
                "FROM candidate_refs "
                "ORDER BY kind, object_id, qualifier"
            )
        ) as rows:
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
                with closing(
                    self._scratch.execute(
                        "SELECT 1 FROM reference_anchors WHERE wire_ref=? AND retired=0 "
                        "AND (?=0 OR projected_retired=0) LIMIT 1",
                        (ref.wire_ref, int(projected_user_effects)),
                    )
                ) as cursor:
                    if not cursor.fetchone():
                        continue
                if not _still_resolves(conn, ref) and not self._intentional_absence(
                    conn, ref, projected_user_effects=projected_user_effects
                ):
                    lost_count += 1
                    if first is None:
                        first = ref
        if lost_count and first is not None:
            raise ReferenceSealError(
                f"index mutation would orphan {lost_count} resolved durable reference(s); "
                f"first lost {first.kind} reference in session {first.owner_session_id!r}"
            )

    def _settle_witness_metadata(self) -> None:
        """End this owner's TEMP bookkeeping transaction before currency reads."""
        owner = next(child for child in native_sql_children(self) if child.connection is self._scratch)
        if live_connection_cursors(self._scratch) or owner._incremental_blobs:
            raise ReferenceSealError("witness currency requires its actual prepared readers to settle")
        self._scratch.commit()
        self._assert_witness_currency()

    def _assert_witness_currency(self) -> None:
        self._require_live_owner()
        if self._scratch.in_transaction or live_connection_cursors(self._scratch):
            raise ReferenceSealError("witness currency cannot absorb a caller-owned active snapshot")
        with self._owned_cursor(self._scratch, "PRAGMA data_version") as cursor:
            if int(cursor.fetchone()[0]) != self._witness_data_version:
                raise ReferenceSealStaleError("another connection changed the original literal witness")

    def _original_witness_path(self) -> Path:
        self._require_new_work()
        with self._owned_cursor(self._scratch, "PRAGMA database_list") as cursor:
            path = next(Path(str(row[2])) for row in cursor if row[1] == "main")
        if self._scratch_directory is None or path != Path(self._scratch_directory.name) / "refs.db":
            raise ReferenceSealError("literal attachment lost its original witness directory")
        if _tier_identity(path)[:2] != self._witness_file_identity:
            raise ReferenceSealStaleError("literal witness file incarnation changed")
        self._assert_witness_currency()
        return path.resolve(strict=True)

    def _require_witness_main_mutable(self) -> None:
        self._require_new_work()
        if self._live_literal_reader is not None:
            raise ReferenceSealError("original witness MAIN is frozen through dedicated native settlement")
        owner = next(child for child in native_sql_children(self) if child.connection is self._scratch)
        if owner.close_required or owner._parent_cleanup_requested:
            raise NativeConnectionSettlementError(
                owner, ReferenceSealError("original witness requires its native creator's terminal cleanup")
            )
        owner.require_connection()

    def _authorize_witness_main(
        self, action: int, first: str | None, second: str | None, schema: str | None, trigger: str | None
    ) -> int:
        if self._source_statement_active:
            if self._source_capture_internal and action == sqlite3.SQLITE_PRAGMA:
                # Only the active Native capture reads its physical schema;
                # producer SQL and every enforcement setter remain denied.
                return (
                    sqlite3.SQLITE_OK
                    if schema == "main"
                    and (
                        first == "table_list"
                        and second is None
                        or first in {"table_info", "index_list"}
                        and second in self._source_capture_tables
                    )
                    else sqlite3.SQLITE_DENY
                )
            if action == sqlite3.SQLITE_FUNCTION and (second or "").startswith("polylogue_"):
                for _table, (_columns, number) in self._source_capture_tables.items():
                    if second != f"polylogue_source_capture_{number}":
                        continue
                    guards = {
                        f"polylogue_source_stage_{number}_{operation}_{phase}"
                        for operation in ("insert", "update", "delete")
                        for phase in ("before", "after")
                    }
                    return sqlite3.SQLITE_OK if trigger in guards else sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_DENY
            if action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE}:
                if self._source_capture_internal:
                    # The callback's existing literal/row owner writes only
                    # these exact MAIN metadata relations and TEMP effect
                    # records. Reentrant canonical producer DML is refused.
                    return (
                        sqlite3.SQLITE_OK
                        if schema == "temp"
                        or schema == "main"
                        and first in {"known_tier_literals", "known_tier_literal_cells", "known_tier_row_images"}
                        else sqlite3.SQLITE_DENY
                    )
                if schema != "main" or first not in self._source_capture_tables:
                    return sqlite3.SQLITE_DENY
                if not self._source_statement_root_seen:
                    if trigger is not None or first != self._source_statement_table:
                        return sqlite3.SQLITE_DENY
                    if self._source_statement_allocation_parameter is not None and action != sqlite3.SQLITE_INSERT:
                        return sqlite3.SQLITE_DENY
                    self._source_statement_root_seen = True
                if first == "raw_existence_changes" and action != sqlite3.SQLITE_DELETE:
                    return (
                        sqlite3.SQLITE_OK
                        if action == sqlite3.SQLITE_INSERT
                        and trigger in {"raw_existence_delete", "raw_existence_key_change"}
                        else sqlite3.SQLITE_DENY
                    )
                if first == "raw_existence_journal_control":
                    return (
                        sqlite3.SQLITE_OK
                        if action == sqlite3.SQLITE_UPDATE and trigger == "raw_existence_journal_prune"
                        else sqlite3.SQLITE_DENY
                    )
                return sqlite3.SQLITE_OK
            if action in {
                sqlite3.SQLITE_READ,
                sqlite3.SQLITE_SELECT,
                sqlite3.SQLITE_FUNCTION,
                sqlite3.SQLITE_RECURSIVE,
            }:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        if self._source_rows_active:
            if action == sqlite3.SQLITE_FUNCTION and (second or "").startswith("polylogue_"):
                return sqlite3.SQLITE_DENY
            if action in {
                sqlite3.SQLITE_READ,
                sqlite3.SQLITE_SELECT,
                sqlite3.SQLITE_FUNCTION,
                sqlite3.SQLITE_RECURSIVE,
            }:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_PRAGMA and (
                (second is None and first in {"database_list", "data_version", "foreign_keys", "recursive_triggers"})
                or first
                in {
                    "table_info",
                    "table_xinfo",
                    "table_list",
                    "foreign_key_list",
                    "index_list",
                    "index_info",
                    "index_xinfo",
                }
            ):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        scratch_pending = self._cleanup_requested or any(
            child.connection is self._owned_scratch_connection
            and (child.close_required or child._parent_cleanup_requested)
            for child in native_sql_children(self)
        )
        if (self._live_literal_reader is None and not scratch_pending) or schema == "temp":
            return sqlite3.SQLITE_OK
        if action in {
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_FUNCTION,
            sqlite3.SQLITE_RECURSIVE,
            sqlite3.SQLITE_TRANSACTION,
            sqlite3.SQLITE_SAVEPOINT,
        }:
            return sqlite3.SQLITE_OK
        if (
            action == sqlite3.SQLITE_PRAGMA
            and second is None
            and first
            in {
                "database_list",
                "data_version",
                "table_info",
                "table_xinfo",
                "table_list",
                "foreign_key_list",
                "index_info",
                "index_xinfo",
                "index_list",
                "encoding",
            }
        ):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    def _retain_live_literal_reader(self, owner: NativeSQLCustodyOwner, permit: KnownTierMutationPermit) -> None:
        self._require_witness_main_mutable()
        if owner not in native_sql_children(self) or owner.connection is self._scratch:
            raise ReferenceSealError("literal attachment requires this seal's actual dedicated native child")
        scratch_owner = next(child for child in native_sql_children(self) if child.connection is self._scratch)
        scratch_owner.require_connection()
        if self._scratch.in_transaction or scratch_owner._incremental_blobs or live_connection_cursors(self._scratch):
            raise ReferenceSealError("literal attachment requires settled original preparation handles")
        owner.require_connection()
        owner.retain_settlement_callback(partial(self._retire_live_literal_reader, owner, permit))
        self._live_literal_reader = owner
        # Reinstalling the authorizer expires cached statements. A statement
        # compiled during preparation cannot bypass the now-frozen MAIN law.
        self._scratch.set_authorizer(self._authorize_witness_main)

    def _retire_live_literal_reader(self, owner: NativeSQLCustodyOwner, permit: KnownTierMutationPermit) -> None:
        self._require_live_owner()
        if self._live_literal_reader is not owner or owner.connection is not None:
            raise ReferenceSealError("literal phase retires only after its original native child physically settles")
        if permit._tier == "source" and permit._receipt_accepted and self._mutation_custody is not None:
            # The actual child has already settled its handles and its own
            # custody claim. Release only this completed unit's seal claim;
            # parent/Index/publication owners retain their independent claims.
            custody = self._mutation_custody
            custody.release_sql_owner(self)
            self._mutation_custody = None
        # Expire cached statements before unlocking. If native installation
        # fails, the exact phase and callback remain retained for creator retry.
        self._scratch.set_authorizer(self._authorize_witness_main)
        self._live_literal_reader = None

    def _require_new_work(self) -> None:
        self._require_live_owner()
        if self._cleanup_requested:
            raise ReferenceSealError("reference seal requires original-owner terminal cleanup")
        _check_reference_cancellation()
        self._assert_configured_namespace()

    def _require_live_owner(self) -> None:
        if self._closed:
            raise ReferenceSealError("reference seal is closed")
        if (
            os.getpid() != self.index_pid
            or threading.current_thread() is not self.index_thread
            or _current_task() is not self.index_task
        ):
            raise ReferenceSealError("reference seal must be used by its observing index owner")

    def _retain_mutation_custody(self, custody: ArchiveWriteCustody) -> None:
        self._require_new_work()
        if self._mutation_custody is not None and self._mutation_custody is not custody:
            raise ReferenceSealError("original tier mutation witness cannot borrow another physical custody")
        if self._mutation_custody is None:
            custody.retain_sql_owner(self)
            self._mutation_custody = custody

    def retain_publication_lifetime(self, exclusion: ActiveWriterLease, close_payload: Callable[[], None]) -> None:
        """Keep this publication's rebuild exclusion through physical cleanup."""
        from polylogue.storage.index_generation import ActiveWriterLease

        self._require_new_work()
        if not isinstance(exclusion, ActiveWriterLease):
            raise ReferenceSealError("publication requires its actual active-writer exclusion")
        exclusion.require_owner(self.archive_root)
        if self._publication_exclusion is not None or self._publication_payload_cleanup is not None:
            raise ReferenceSealError("reference seal already owns a publication lifetime")
        self._publication_exclusion = exclusion
        self._publication_payload_cleanup = close_payload
        self._publication_lifetime_bound = True

    @property
    def publication_lifetime_bound(self) -> bool:
        """Whether this seal ever accepted its publication's terminal lifetime."""
        return self._publication_lifetime_bound

    def close(self) -> None:
        if self._closed:
            return
        if (
            self.index_pid != os.getpid()
            or self.index_thread is not threading.current_thread()
            or self.index_task is not _current_task()
        ):
            raise ReferenceSealError("reference-seal cleanup must run in its preparing execution unit")
        from polylogue.storage.sqlite.connection_profile import request_native_sql_parent_cleanup

        self._cleanup_requested = True
        request_native_sql_parent_cleanup(self)
        failures: list[BaseException] = []
        attempted_connections: set[int] = set()

        def settle(action: Callable[[], object]) -> bool:
            try:
                action()
                return True
            except BaseException as exc:
                failures.append(exc)
                return False

        if self._publication_payload_cleanup is not None and settle(self._publication_payload_cleanup):
            self._publication_payload_cleanup = None
        for name, observer in tuple(self._observers.items()):
            attempted_connections.add(id(observer))
            if settle(partial(self._close_native_connection, observer)):
                self._observers.pop(name, None)
        for name, leaf in tuple(self._observer_leaves.items()):
            if name not in self._observers and settle(leaf.close):
                self._observer_leaves.pop(name, None)
        for owner in native_sql_children(self):
            if (
                not owner._settled
                and owner._connection_identity not in attempted_connections
                and owner.connection is not self._owned_scratch_connection
            ):
                attempted_connections.add(owner._connection_identity)
                settle(owner.close)
        dependents_settled = all(
            owner._settled
            or owner.connection is self._owned_scratch_connection
            or any(owner.connection is observer for observer in self._observers.values())
            for owner in native_sql_children(self)
        )
        if dependents_settled and self._owned_scratch_connection is not None:
            scratch = self._owned_scratch_connection
            attempted_connections.add(id(scratch))
            if settle(lambda: self._close_native_connection(scratch)):
                self._owned_scratch_connection = None
        if self._owned_scratch_connection is None and self._scratch_directory is not None:
            directory = self._scratch_directory
            if settle(directory.cleanup):
                self._scratch_directory = None
        from polylogue.storage.sqlite.connection_profile import retire_native_sql_parent

        sql_settled = (
            not self._observers
            and not self._observer_leaves
            and self._owned_scratch_connection is None
            and self._scratch_directory is None
            and all(owner._settled for owner in native_sql_children(self))
        )
        if sql_settled and self._publication_payload_cleanup is None and self._mutation_custody is not None:
            custody = self._mutation_custody
            if settle(lambda: custody.release_sql_owner(self)):
                self._mutation_custody = None
        if (
            sql_settled
            and self._mutation_custody is None
            and self._publication_payload_cleanup is None
            and self._publication_exclusion is not None
        ):
            exclusion = self._publication_exclusion
            if settle(exclusion.close) or not exclusion.held:
                self._publication_exclusion = None
        self._closed = (
            sql_settled
            and self._mutation_custody is None
            and self._publication_payload_cleanup is None
            and self._publication_exclusion is None
        )
        if self._closed:
            self._original_input_demand = None
            retire_native_sql_parent(self)
            with _LIVE_SEALS_LOCK:
                _LIVE_SEALS.pop(id(self), None)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("Reference-seal cleanup failed", failures)

    def __enter__(self) -> PreparedIndexMutation:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> None:
        try:
            self.close()
        except BaseException as close_error:
            if exc is None:
                raise
            raise BaseExceptionGroup("Reference mutation and seal cleanup failed", [exc, close_error]) from exc


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
            with closing(conn.execute("PRAGMA database_list")) as cursor:
                databases = cursor.fetchall()
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
        with closing(scope.conn.execute("BEGIN IMMEDIATE")):
            pass
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
    _archive_cleanup_started: bool = False
    _archive_cleanup_failed: bool = False
    _rollback_required: bool = True
    _user_owner: NativeSQLCustodyOwner | None = field(default=None, init=False, repr=False)
    _user_admission_custody: ArchiveWriteCustody | None = field(default=None, init=False, repr=False)

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
            parent = native_sql_parent_for_connection(self.conn)
            custody = current_sql_custody()
            if parent is not None:
                from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

                if not isinstance(parent, ArchiveStore) or parent._owned_index_connection is not self.conn:
                    raise ReferenceSealError("suppression reader requires its exact Index Store owner")
                if custody is not None:
                    # Preserve the original grant once on the Store. Its
                    # terminal census owns both Index and User, including
                    # failures before this reader's initializer returns.
                    parent._retain_sql_custody(custody)
            self._user_admission_custody = custody
            # The same scope owns one reader, with actual creator custody and
            # scope lifetime retained before factory setup SQL can fail.
            try:
                self._user_owner = _open_readonly_owner(
                    path, validate_schema=False, lifetime_dependencies=(self,), terminal_parent=parent
                )
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
        if custody is not self._user_admission_custody:
            raise ReferenceSealError("suppression reader belongs to another admitted writer")
        if custody is not None:
            custody.assert_namespace()
        return owner.require_connection()

    def _close_suppression_reader(self) -> None:
        owner = self._user_owner
        if owner is not None:
            owner.close()
            if owner._terminal_parent is not None:
                owner.retire_terminal_parent(owner._terminal_parent)
            self._user_owner = None
            self._user_admission_custody = None

    def note_session_namespace_change(self) -> None:
        self.require_connection(self.conn)
        _check_reference_cancellation()
        if self.seal is not None:
            self.seal.note_session_namespace_change()

    def authorize_session_removal(self, session_ids: tuple[str, ...]) -> None:
        self.require_connection(self.conn)
        if self.seal is not None:
            self.seal.authorize_session_removal(session_ids)

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

    def preflight_reachability(self) -> None:
        self.require_new_work(self.conn)
        if self.seal is None:
            raise ReferenceSealError("durable removal preflight requires the original reference witness")
        self.seal.preflight_reachability(self.conn)

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
        return not self._rollback_required and self._user_owner is None and not self._archive_cleanup_failed

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

    def require_cleanup_connection(self, conn: sqlite3.Connection) -> None:
        if conn is not self.conn:
            raise ReferenceSealError("Index scope cleanup requires its exact connection")
        if os.getpid() != self.owner_pid or threading.current_thread() is not self.owner_thread:
            raise ReferenceSealError("Index scope cleanup belongs to another process or thread")
        if _current_task() is not self.owner_task and (
            not isinstance(self.owner_task, asyncio.Task) or not self.owner_task.done()
        ):
            raise ReferenceSealError("Index scope cleanup belongs to another task")

    def close(self) -> None:
        self.require_cleanup_connection(self.conn)
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
