"""Staged retained-input ingestion on the existing daemon compute and writer."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterable
from contextlib import closing
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path
from time import monotonic, time
from typing import TypeVar
from uuid import uuid4

from polylogue.core.raw_failure_evidence import (
    CohortMembershipRefusalError,
    RetainedRawDecodeRefusalError,
    RetainedRawDependencyRefusalError,
)
from polylogue.logging import WARNING, emit
from polylogue.operations.audit import AuditRepository, MachineRequestBinding
from polylogue.operations.bindings import runtime_operation_binding
from polylogue.operations.daemon_execution import (
    OperationRuntime,
    _validate_identity,
    operation_envelope,
    validate_execution_request,
)
from polylogue.operations.daemon_protocol import DaemonOperationEnvelope, DaemonOperationRequest
from polylogue.operations.ingest_acceptance import IngestActuator, ingest_plan
from polylogue.operations.ingest_inputs import (
    PreparedSourceMemberDisposition,
    PreparedSourceRecord,
    discover_ingest_input_spool,
    enumerate_ingest_input,
    retain_input_page,
    spool_connection,
    unlink_spool,
)
from polylogue.operations.insight_acceptance import SessionInsightPartReceipt
from polylogue.operations.machine_lifecycle import machine_request_state
from polylogue.operations.machine_receipts import (
    MAX_INLINE_INGEST_SESSION_IDS,
    MAX_INLINE_RAW_IDS_PER_INPUT,
    MAX_MACHINE_RECEIPT_PAGES,
    MAX_PAGE_ITEMS,
    IngestHistoricalReceiptV2,
    IngestInputHistoricalReceipt,
    IngestInputPageHistoricalReceipt,
    IngestInputRawMemberHistorical,
    IngestInputRawPageHistoricalReceipt,
    IngestInputRawPagesDigest,
    IngestInsightPageHistoricalReceipt,
    IngestRefusalPageHistoricalReceipt,
    IngestRefusalPagesDigest,
    IngestRefusedMembershipHistorical,
    IngestTerminalSummaryHistorical,
    InsightTargetHistoricalReceipt,
    ingest_insight_pages_digest,
    ingest_terminal_outcome,
    ingest_unconverged_error,
)
from polylogue.operations.mutation_transaction import (
    MutationPreview,
    MutationReceipt,
    OperationExecutor,
    StartedBoundMutation,
)
from polylogue.operations.operation_context import PinnedOperationRead, open_operation_read
from polylogue.operations.operation_context_types import OperationContext
from polylogue.operations.source_item_settlement import settle_materialized_source_items
from polylogue.sources.origin_specs import retained_enumeration_fingerprint
from polylogue.sources.pickle_spool import PickleSpool
from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation
from polylogue.storage.blob_publication import (
    ArchiveBlobPublisher,
    consume_blob_publication_receipt,
    publication_refused,
)
from polylogue.storage.source_generation_receipts import iter_source_item_raw_receipts, source_generation_receipt_page
from polylogue.storage.sqlite.archive_tiers.raw_admission import RawAdmissionArm, execute_source_item_admission
from polylogue.storage.sqlite.archive_tiers.source_items import (
    FrozenSourceInput,
    RetainedSourceGeneration,
    RetainedSourceInput,
    abort_prepared_source_manifest,
    append_prepared_source_inputs,
    begin_prepared_source_manifest,
    complete_source_item_enumeration,
    page_retained_source_inputs,
    retained_source_generation_header,
    seal_prepared_source_manifest,
)
from polylogue.storage.sqlite.archive_tiers.source_write import ContentExcisedError
from polylogue.storage.sqlite.connection_profile import open_isolated_write_connection
from polylogue.storage.sqlite.population_admission import assert_population_admitted
from polylogue.storage.sqlite.reference_seal import (
    IndexMutationDestination,
    ReferenceSealStaleError,
    index_path_for_connection,
)

_T = TypeVar("_T")


class ArchiveIdentityStaleError(ValueError):
    """The archive identity moved (a generation was promoted) after this execution pinned it.

    Transient: clearing the pin and driving again reads the promoted generation.
    """


@dataclass(slots=True)
class _ExcisedRecords:
    """Records of one source item the source tier refused as durably excised.

    An excised record never becomes a raw member, so it leaves the item's
    record denominator. A ZIP member whose every record was excised keeps its
    central-directory ordinal accounted as a refused disposition.
    """

    members: PickleSpool[tuple[int, str]] = field(default_factory=PickleSpool)

    def add(self, prepared: PreparedSourceRecord) -> None:
        if prepared.member.entry_ordinal is not None:
            self.members.append((prepared.member.entry_ordinal, str(prepared.record.source_path)))

    def record_member_dispositions(
        self, connection: sqlite3.Connection, *, source_generation_id: str, source_item_id: str, observed_at_ms: int
    ) -> None:
        from polylogue.storage.sqlite.archive_tiers.source_items import (
            SourceItemMemberDisposition,
            record_source_item_member_disposition,
        )

        for ordinal, member_name in self.members:
            admitted = connection.execute(
                "SELECT 1 FROM source_item_raw_members m JOIN raw_container_coordinates c ON c.raw_id=m.raw_id "
                "WHERE m.source_generation_id=? AND m.source_item_id=? AND c.entry_ordinal=?",
                (source_generation_id, source_item_id, ordinal),
            ).fetchone()
            if admitted is None:
                record_source_item_member_disposition(
                    connection,
                    source_generation_id=source_generation_id,
                    source_item_id=source_item_id,
                    entry_ordinal=ordinal,
                    member_name=member_name,
                    disposition=SourceItemMemberDisposition.REFUSED,
                    diagnostic="content excised",
                    observed_at_ms=observed_at_ms,
                )

    def close(self) -> None:
        self.members.close()


class IngestStoppedError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class IngestProjectionUnrecoverableError(RuntimeError):
    """A re-drive cannot tell which sessions its interrupted attempt changed."""


class IngestReprepareRequiredError(RuntimeError):
    """The pinned generation moved under this attempt; nothing here is permanent.

    Raised by :meth:`IngestExecution.require_publication_identity` when a concurrent index
    promotion or writer already advanced the archive past the snapshot this
    attempt pinned. The work this attempt already did is still valid (the
    manifest and prepared source are content-addressed), only the active
    generation changed underneath it -- the caller must reprepare or retry
    against the new generation, never terminalize as failed.
    """


@dataclass(frozen=True, slots=True)
class SourceReceiptSpool:
    """One pinned Source projection reduced on disk by raw and logical ID."""

    path: Path
    source_generation_id: str
    item_count: int
    enumeration_complete: bool
    complete: bool
    confirmed_raw_count: int
    unresolved_raw_count: int
    index_destination: IndexMutationDestination | None = None

    def close(self) -> None:
        unlink_spool(self.path)

    def raw_page(self, after: str | None = None) -> tuple[str, ...]:
        with spool_connection(self.path, read_only=True) as conn:
            return tuple(
                str(row[0])
                for row in conn.execute(
                    "SELECT raw_id FROM raws WHERE raw_id>? ORDER BY raw_id LIMIT 256",
                    (after or "",),
                ).fetchall()
            )

    def complete_raw_sessions(self, raw_ids: tuple[str, ...]) -> tuple[str, ...]:
        """Sessions the archive holds, at this projection, for exactly these raws."""
        if not raw_ids:
            return ()
        with spool_connection(self.path, read_only=True) as conn:
            return tuple(
                str(row[0])
                for row in conn.execute(
                    "SELECT DISTINCT l.expected_session_id FROM raw_logicals r "
                    "JOIN logicals l ON l.logical_key=r.logical_key "
                    f"WHERE r.raw_id IN ({','.join('?' * len(raw_ids))}) AND l.complete=1 "
                    "ORDER BY l.expected_session_id",
                    raw_ids,
                ).fetchall()
            )

    def session_page(self, after: str | None = None) -> tuple[str, ...]:
        with spool_connection(self.path, read_only=True) as conn:
            return tuple(
                str(row[0])
                for row in conn.execute(
                    "SELECT DISTINCT expected_session_id FROM logicals WHERE expected_session_id>? "
                    "ORDER BY expected_session_id LIMIT 256",
                    (after or "",),
                ).fetchall()
            )


def _spool_source_receipt(
    source_conn: sqlite3.Connection,
    index_conn: sqlite3.Connection,
    generation_id: str,
    path: Path,
    *,
    check_stop: Callable[[], None] | None = None,
    index_destination: IndexMutationDestination | None = None,
) -> SourceReceiptSpool:
    """Project all witnesses under one pinned pair without an unbounded result."""
    generation = source_conn.execute(
        "SELECT item_count FROM source_generations WHERE source_generation_id=?", (generation_id,)
    ).fetchone()
    if generation is None or type(generation[0]) is not int or generation[0] < 1:
        raise ValueError("accepted source generation is absent or empty")
    item_count = int(generation[0])
    observed_count = 0
    enumeration_complete = True
    complete = True
    cursor: tuple[str, str] | None = None
    with spool_connection(path) as spool:
        spool.executescript(
            "CREATE TABLE items(ordinal INTEGER PRIMARY KEY, source_item_id TEXT NOT NULL, coordinate TEXT NOT NULL, "
            "raw_count INTEGER NOT NULL, unresolved_count INTEGER NOT NULL, retired_count INTEGER NOT NULL, "
            "source_complete INTEGER NOT NULL);"
            "CREATE TABLE item_raws(source_item_id TEXT NOT NULL, raw_id TEXT NOT NULL, raw_blob_hash BLOB NOT NULL, "
            "complete INTEGER NOT NULL, "
            "PRIMARY KEY(source_item_id, raw_id)) WITHOUT ROWID;"
            "CREATE UNIQUE INDEX items_source_id ON items(source_item_id);"
            "CREATE TABLE raws(raw_id TEXT PRIMARY KEY, complete INTEGER NOT NULL, parser_complete INTEGER NOT NULL) WITHOUT ROWID;"
            "CREATE TABLE logicals(logical_key TEXT PRIMARY KEY, expected_session_id TEXT NOT NULL, complete INTEGER NOT NULL) WITHOUT ROWID;"
            "CREATE INDEX logicals_session ON logicals(expected_session_id);"
            "CREATE TABLE raw_logicals(raw_id TEXT NOT NULL, logical_key TEXT NOT NULL, "
            "PRIMARY KEY(raw_id, logical_key)) WITHOUT ROWID;"
            "CREATE TABLE raw_metadata(source_item_id TEXT PRIMARY KEY, page_ref TEXT NOT NULL, page_count INTEGER NOT NULL, "
            "raw_count INTEGER NOT NULL, unresolved_count INTEGER NOT NULL, digest TEXT NOT NULL) WITHOUT ROWID;"
        )
        while True:
            page = source_generation_receipt_page(
                source_conn, source_generation_id=generation_id, after=cursor, check_stop=check_stop
            )
            if not page.items:
                break
            for item in page.items:
                if check_stop is not None:
                    check_stop()
                enumeration_complete &= item.enumeration_complete
                complete &= item.source_complete
                raw_count = unresolved_count = 0
                with closing(
                    iter_source_item_raw_receipts(
                        source_conn,
                        index_conn,
                        source_generation_id=generation_id,
                        source_item_id=item.source_item_id,
                        check_stop=check_stop,
                    )
                ) as raw_receipts:
                    for raw in raw_receipts:
                        raw_complete = raw.parser_complete
                        for logical in raw.logicals:
                            raw_complete &= logical.complete
                            if check_stop is not None:
                                check_stop()
                            spool.execute(
                                "INSERT INTO logicals VALUES (?, ?, ?) ON CONFLICT(logical_key) DO UPDATE SET "
                                "complete=MIN(complete, excluded.complete)",
                                (logical.logical_source_key, logical.expected_session_id, int(logical.complete)),
                            )
                            spool.execute(
                                "INSERT INTO raw_logicals VALUES (?, ?) ON CONFLICT DO NOTHING",
                                (raw.raw_id, logical.logical_source_key),
                            )
                        member_hash = source_conn.execute(
                            "SELECT raw_blob_hash FROM main.source_item_raw_members "
                            "WHERE source_generation_id=? AND source_item_id=? AND raw_id=? "
                            "ORDER BY record_coordinate LIMIT 1",
                            (generation_id, item.source_item_id, raw.raw_id),
                        ).fetchone()
                        if member_hash is None or not isinstance(member_hash[0], bytes):
                            raw_complete = False
                        raw_count += 1
                        unresolved_count += not raw_complete
                        complete &= raw_complete
                        spool.execute(
                            "INSERT INTO item_raws VALUES (?, ?, ?, ?)",
                            (
                                item.source_item_id,
                                raw.raw_id,
                                member_hash[0] if member_hash is not None else b"",
                                int(raw_complete),
                            ),
                        )
                        spool.execute(
                            "INSERT INTO raws VALUES (?, ?, ?) ON CONFLICT(raw_id) DO UPDATE SET "
                            "complete=MIN(complete, excluded.complete), "
                            "parser_complete=MIN(parser_complete, excluded.parser_complete)",
                            (raw.raw_id, int(raw_complete), int(raw.parser_complete)),
                        )
                spool.execute(
                    "INSERT INTO items VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        observed_count,
                        item.source_item_id,
                        item.logical_coordinate,
                        raw_count,
                        unresolved_count,
                        item.retired_count,
                        int(item.source_complete and unresolved_count == 0),
                    ),
                )
                observed_count += 1
            cursor = page.next_cursor
        enumeration_complete &= observed_count == item_count
        complete &= enumeration_complete
        confirmed = int(spool.execute("SELECT COUNT(*) FROM raws WHERE complete=1").fetchone()[0])
        unresolved = int(spool.execute("SELECT COUNT(*) FROM raws WHERE complete=0").fetchone()[0])
    return SourceReceiptSpool(
        path, generation_id, observed_count, enumeration_complete, complete, confirmed, unresolved, index_destination
    )


def _record_int(value: object, *, field: str) -> int:
    """Reject malformed durable operation fields before using them as timestamps."""

    if type(value) is not int:
        raise ValueError(f"accepted ingest {field} is not an integer")
    return value


class IngestExecution:
    """One accepted denominator, with no worker held across phase awaits."""

    def __init__(self, request: DaemonOperationRequest, context: OperationContext) -> None:
        if context.runtime is None:
            raise PermissionError("daemon_required")
        self.request = request
        self.context = context
        self._setup(context.runtime, context.archive_root, context.runtime.audit_for_request(request, context))

    def _setup(self, runtime: OperationRuntime, archive_root: Path, audit: AuditRepository) -> None:
        self.runtime = runtime
        self.archive_root = archive_root
        self.audit = audit
        self.executor = OperationExecutor(audit=self.audit, archive_root=archive_root)
        self.snapshot: PinnedOperationRead | None = None
        self.binding: MachineRequestBinding | None = None
        # polylogue-cois9: the live archive identity digest observed when this
        # execution first pinned a read.  It folds in the rebuildable index
        # tier's inode, so it is only good for detecting a generation change
        # *within* this run -- never as the durable audit key.
        self.observed_identity: str | None = None
        self.record: dict[str, object] | None = None
        self.started_mutation: StartedBoundMutation | None = None
        self.resumed = False
        self.terminalized = False
        #: Set once this execution may have durably accepted its request (the
        #: acceptance write began, or a resent request found its record).
        #: Before that there is no accepted attempt to fence or settle.
        self.acceptance_attempted = False
        fd, name = tempfile.mkstemp(prefix="polylogue-ingest-state-", suffix=".sqlite", dir=os.environ.get("TMPDIR"))
        os.close(fd)
        self.state_path = Path(name)
        with spool_connection(self.state_path) as state:
            state.executescript(
                "CREATE TABLE refusals(ordinal INTEGER PRIMARY KEY, logical_key TEXT NOT NULL, raw_id TEXT NOT NULL, "
                "reason TEXT NOT NULL, UNIQUE(logical_key, raw_id, reason));"
                "CREATE TABLE changed_sessions(session_id TEXT PRIMARY KEY, message_count INTEGER NOT NULL) WITHOUT ROWID;"
                # Raws this execution's admission introduced, not duplicates
                # of retained raws.
                "CREATE TABLE admitted_raws(raw_id TEXT PRIMARY KEY) WITHOUT ROWID;"
            )
        self.session_id_pages_ref: str | None = None
        self.session_id_page_count = 0
        self.session_ids_digest: str | None = None
        self.inline_session_ids: list[str] = []
        self.changed_session_count = 0
        self.changed_message_count = 0
        #: Cohorts that published a session this attempt did not change: the
        #: content was already there, from before this request or from an
        #: interrupted attempt of it.
        self.unchanged_publications = 0
        self.refused_count = 0
        self.source_items_sealable = False
        self._materialization_index_destination: IndexMutationDestination | None = None
        self._materialization_destination_bound = False
        self.inline_refusals: list[IngestRefusedMembershipHistorical] = []
        self.refusal_pages_ref: str | None = None
        self.refusal_page_count = 0
        self.refusal_pages_digest: str | None = None
        self.insight_pages_ref: str | None = None
        self.insight_page_count = 0
        self.insight_pages_digest: str | None = None
        self.profile_convergence_complete = False
        #: The profile receipts of this execution's first convergence: a
        #: retry after its insight pages were persisted reuses them, since a
        #: second convergence reports those profiles ``already_satisfied``.
        self.retained_profile_parts: tuple[SessionInsightPartReceipt, ...] | None = None
        self.publisher = ArchiveBlobPublisher(archive_root / "source.db", archive_root / "blob")

    def record_refusal(self, logical_key: str, refusal: RetainedRawDecodeRefusalError) -> None:
        with spool_connection(self.state_path) as state:
            state.execute(
                # A transient retry drives the generation again from the start:
                # one refused membership is one refusal, however many drives saw it.
                "INSERT OR IGNORE INTO refusals(logical_key, raw_id, reason) VALUES (?, ?, ?)",
                (logical_key, refusal.raw_id, refusal.kind.value),
            )

    @staticmethod
    def _validate_index_destination(destination: IndexMutationDestination) -> None:
        try:
            destination.validate()
        except (FileNotFoundError, ReferenceSealStaleError) as exc:
            raise IngestReprepareRequiredError("retained replay Index destination changed; retry required") from exc

    @staticmethod
    def _require_materialization_destination(
        receipt: SourceReceiptSpool,
        expected: IndexMutationDestination | None,
    ) -> None:
        if receipt.index_destination != expected:
            raise IngestReprepareRequiredError("materialization receipt Index destination changed")

    def record_membership_refusal(self, logical_key: str, raw_id: str, reason: str) -> None:
        with spool_connection(self.state_path) as state:
            state.execute(
                "INSERT OR IGNORE INTO refusals(logical_key, raw_id, reason) VALUES (?, ?, ?)",
                (logical_key, raw_id, reason),
            )

    def record_changed_session(self, session_id: str, message_count: int) -> None:
        with spool_connection(self.state_path) as state:
            state.execute(
                "INSERT INTO changed_sessions VALUES (?, ?) ON CONFLICT(session_id) DO UPDATE SET "
                "message_count=excluded.message_count",
                (session_id, message_count),
            )

    def record_admitted_raws(self, raw_ids: Iterable[str]) -> None:
        with spool_connection(self.state_path) as state:
            state.executemany("INSERT OR IGNORE INTO admitted_raws VALUES (?)", ((raw_id,) for raw_id in raw_ids))

    def changed_session_recorded(self, session_id: str) -> bool:
        """Whether this execution already recorded ``session_id`` as changed (a retry re-publishing its own work)."""
        assert_population_admitted(self.state_path)
        with spool_connection(self.state_path, read_only=True) as state:
            return (
                state.execute("SELECT 1 FROM changed_sessions WHERE session_id = ?", (session_id,)).fetchone()
                is not None
            )

    def stop_reason(self) -> str | None:
        return self.durable_stop_reason(self.runtime.stop_reason(self.request))

    def durable_stop_reason(self, reason: str | None) -> str | None:
        """Fold the accepted request's durable stop, deadline and authority expiry into ``reason``."""
        if self.record is not None:
            deadline = self.record.get("accepted_deadline_unix_ms")
            if self.record.get("stop_reason"):
                reason = str(self.record["stop_reason"])
            elif deadline is not None and int(time() * 1000) >= _record_int(deadline, field="deadline"):
                reason = "deadline"
        if self.started_mutation is not None:
            expires_at_ms = self.started_mutation.authorization.expires_at_ms
            if expires_at_ms is None:
                raise ValueError("accepted ingest authority has no expiry")
            if int(time() * 1000) >= expires_at_ms:
                reason = reason or "deadline"
        return reason

    def check_stop(self) -> None:
        reason = self.stop_reason()
        if reason is not None:
            raise IngestStoppedError(reason)

    def recover_machine_request(self) -> dict[str, object] | None:
        """Recover this exchange's durable row and adopt the identity it is keyed by.

        polylogue-cois9: ``machine_requests`` rows (and their
        ``machine_request_parts`` children) are keyed by the archive identity
        digest that was live when they were written.  That digest folds in the
        rebuildable index tier's inode, so an index-generation promotion across
        a daemon restart re-derives a different one.  The audit lookup is now
        keyed on ``request_id`` within this audit database, and this adopts the
        stored identity so every later part read and write addresses the same
        durable rows rather than starting a second, unlinked exchange.
        """

        assert self.binding is not None
        record = self.audit.machine_request(self.binding)
        if record is not None:
            stored = str(record["archive_identity"])
            if stored != self.binding.archive_identity:
                self.binding = replace(self.binding, archive_identity=stored)
        return record

    def first_snapshot(self, snapshot: PinnedOperationRead) -> None:
        """Validate the request's archive preconditions and bind its durable key."""
        _validate_identity(self.request, self.context, snapshot)
        self.binding = MachineRequestBinding(
            snapshot.identity.authority_identity_digest,
            str(self.request.request_id),
            self.context.principal.actor_ref,
            self.request.fingerprint,
            self.request.operation,
        )

    def observe_snapshot(self, snapshot: PinnedOperationRead) -> None:
        self.runtime.observe_snapshot(self.request, snapshot)

    async def read(self, work: Callable[[PinnedOperationRead], _T]) -> _T:
        def observed() -> _T:
            self.check_stop()
            with open_operation_read(self.archive_root, publication_guard=self.runtime.publication_guard) as snapshot:
                identity = snapshot.identity.authority_identity_digest
                if self.observed_identity is None:
                    self.first_snapshot(snapshot)
                    self.observed_identity = identity
                elif identity != self.observed_identity:
                    raise ArchiveIdentityStaleError("archive_identity_stale")
                self.snapshot = snapshot
                self.observe_snapshot(snapshot)
                return work(snapshot)

        return await self.runtime.compute_phase(observed)

    async def source_write(self, work: Callable[[sqlite3.Connection], _T]) -> _T:
        def publish() -> _T:
            self.check_stop()
            # Reservations settle before the source reference transaction.
            # They remain protected if reference publication later fails.
            self.publisher.flush()
            with (
                closing(
                    open_isolated_write_connection(
                        self.archive_root / "source.db",
                        purpose="accepted ingest source publication",
                        archive_root=self.archive_root,
                    )
                ) as connection,
                connection,
            ):
                connection.execute("BEGIN IMMEDIATE")
                return work(connection)

        return await self.runtime.write_phase("ingest.source", publish)

    async def settle_source_items(self, generation: RetainedSourceGeneration, receipt: SourceReceiptSpool) -> int:
        """Publish only item outcomes proved by this generation's materialization."""
        if receipt.source_generation_id != generation.source_generation_id:
            raise ValueError("materialization receipt names a different source generation")
        if self.snapshot is None:
            raise ValueError("materialization receipt has no pinned archive identity")
        identity = self.snapshot.identity

        def settle(connection: sqlite3.Connection) -> int:
            # The active generation may move while the callback waits for the
            # single writer. Recheck immediately before Source mutations.
            self.require_publication_identity(identity)
            self._require_materialization_destination(receipt, self._materialization_index_destination)
            if receipt.index_destination is not None:
                self._validate_index_destination(receipt.index_destination)
            changed = settle_materialized_source_items(
                connection,
                source_generation_id=generation.source_generation_id,
                receipt_path=receipt.path,
                observed_at_ms=int(time() * 1000),
                check_stop=self.check_stop,
            )
            if receipt.index_destination is not None:
                self._validate_index_destination(receipt.index_destination)
            census = connection.execute(
                "SELECT sealable FROM source_item_reconciliation WHERE source_generation_id=?",
                (generation.source_generation_id,),
            ).fetchone()
            self.source_items_sealable = census is not None and bool(census[0])
            return changed

        return await self.source_write(settle)

    async def abort_prepared(self, generation_id: str) -> None:
        """Settle a failed pre-accept attempt even after stop was requested."""
        await self.runtime.compute_phase(self.publisher.discard_pending)

        def discard() -> None:
            with (
                closing(
                    open_isolated_write_connection(
                        self.archive_root / "source.db",
                        purpose="abandoned ingest source preparation",
                        archive_root=self.archive_root,
                    )
                ) as connection,
                connection,
            ):
                connection.execute("BEGIN IMMEDIATE")
                abort_prepared_source_manifest(
                    connection, source_generation_id=generation_id, publisher_id=self.publisher.publisher_id
                )

        await self.runtime.write_phase("ingest.source", discard)

    def require_publication_identity(self, expected: ArchiveIdentity) -> None:
        self.check_stop()
        location = ArchiveLocation.resolve(self.archive_root)
        current = ArchiveIdentity.resolve_location(location)
        if (current.authority_identity_digest, current.active_generation) != (
            expected.authority_identity_digest,
            expected.active_generation,
        ):
            raise IngestReprepareRequiredError("ingest publication generation changed; reprepare required")

    async def accept(self) -> RetainedSourceGeneration | None:
        # Required derivation capability must exist before accepting retained work.
        self.runtime.require_session_maintenance()

        def recover(_snapshot: PinnedOperationRead) -> dict[str, object] | None:
            with self.audit.settled_machine_read():
                return self.recover_machine_request()

        self.record = await self.read(recover)
        if self.record is None:
            source_path = self.request.payload.get("source_path")
            if source_path is not None and not isinstance(source_path, str):
                raise ValueError("ingest source path is not a string")
            source_name = self.request.payload.get("source_name")
            if source_name is not None and not isinstance(source_name, str):
                raise ValueError("ingest source name is not a string")
            generation_id = str(uuid4())
            fingerprint = await self.runtime.compute_phase(retained_enumeration_fingerprint)
            spool = await self.runtime.compute_phase(
                lambda: discover_ingest_input_spool(
                    Path(str(self.request.payload["path"])), source_path=source_path, check_stop=self.check_stop
                )
            )
            try:
                await self.source_write(
                    lambda conn: begin_prepared_source_manifest(
                        conn,
                        source_generation_id=generation_id,
                        publisher_id=self.publisher.publisher_id,
                        enumeration_fingerprint=fingerprint,
                        source_name=source_name,
                    )
                )
                ordinal = 0
                # Refusals are reported page by page; only the first is kept,
                # so a source of many excised files stays page-bounded.
                first_refused: FrozenSourceInput | None = None
                after_coordinate: str | None = None
                while True:

                    def compute_page(after: str | None = after_coordinate) -> tuple[FrozenSourceInput, ...]:
                        return retain_input_page(
                            spool,
                            after_coordinate=after,
                            publisher=self.publisher,
                            check_stop=self.check_stop,
                        )

                    page = await self.runtime.compute_phase(compute_page)
                    if not page:
                        break

                    page_refused: list[FrozenSourceInput] = []

                    def stage_page(
                        conn: sqlite3.Connection,
                        start: int = ordinal,
                        batch: tuple[FrozenSourceInput, ...] = page,
                        refused: list[FrozenSourceInput] = page_refused,
                    ) -> int:
                        # ``source_write`` flushed the page's publications first.
                        # An input whose bytes are excised -- refused by that
                        # flush, or excised after it succeeded, as the ledger
                        # read in this source transaction shows -- has no
                        # reservation: it is a permanent skip, not a manifest
                        # member.
                        from polylogue.storage.sqlite.archive_tiers.source_write import is_blob_hash_excised

                        admitted = tuple(
                            item
                            for item in batch
                            if not publication_refused(self.publisher, item.blob_hash)
                            and not is_blob_hash_excised(conn, bytes.fromhex(item.blob_hash))
                        )
                        refused[:] = [item for item in batch if item not in admitted]
                        if admitted:
                            append_prepared_source_inputs(conn, generation_id, start, admitted)
                        return len(admitted)

                    ordinal += await self.source_write(stage_page)
                    after_coordinate = page[-1].coordinate
                    for refused_input in page_refused:
                        emit(
                            "ingest.accepted_input.content_excised",
                            outcome="skipped",
                            reason="content_excised",
                            blob_hash=refused_input.blob_hash,
                        )
                    if first_refused is None and page_refused:
                        first_refused = page_refused[0]
                    # This page's refusals are reconciled; the publisher need
                    # not keep them for the rest of the walk.
                    self.publisher.forget_refusals()
                if ordinal == 0 and first_refused is not None:
                    raise ContentExcisedError(
                        blob_hash=bytes.fromhex(first_refused.blob_hash),
                        source_path=first_refused.source_path,
                    )
                manifest = await self.source_write(
                    lambda conn: seal_prepared_source_manifest(conn, generation_id, sealed_at_ms=int(time() * 1000))
                )
            except BaseException:
                await self.abort_prepared(generation_id)
                raise
            finally:
                await self.runtime.compute_phase(lambda: unlink_spool(spool))

            def accept_prepared() -> dict[str, object]:
                self.check_stop()
                assert self.binding is not None
                now_ms = int(time() * 1000)
                deadline = self.runtime.request_deadline_unix_ms(self.request)
                assert deadline is not None  # ingest retains its declared execution deadline
                expires_at_ms = deadline
                instance = self.audit.ensure_archive_authority(now_ms=now_ms)
                actuator = IngestActuator(manifest, instance, self.binding.archive_identity, now_ms, expires_at_ms)
                plan = ingest_plan(
                    manifest,
                    archive_instance_id=instance,
                    archive_identity_digest=self.binding.archive_identity,
                    now_ms=now_ms,
                    expires_at_ms=expires_at_ms,
                )
                authorization = OperationExecutor(now_ms=lambda: now_ms).authorize_bound(
                    runtime_operation_binding(actuator),
                    MutationPreview(preview_ref=f"preview:{plan.plan_hash}", plan=plan),
                    self.context.principal,
                    confirmation_strength="role_only",
                )
                with self.audit.bind_machine_request(
                    self.binding,
                    transition="accept_ingest",
                    deadline_unix_ms=self.runtime.request_deadline_unix_ms(self.request),
                ):
                    self.audit.accept_ingest(manifest, self.context.principal, plan=plan, authorization=authorization)
                record = self.recover_machine_request()
                assert record is not None
                return record

            self.acceptance_attempted = True
            try:
                self.record = await self.runtime.write_phase("ingest.accept", accept_prepared)
            except BaseException:

                def accepted_after_settlement() -> bool:
                    with self.audit.settled_machine_read():
                        return self.recover_machine_request() is not None

                if not await self.runtime.write_phase("ingest.accept-settle", accepted_after_settlement):
                    await self.abort_prepared(generation_id)
                raise
        else:
            self.resumed = True
            self.acceptance_attempted = True
        if self.resumed:
            # A resent request reads its durable state only. The daemon's
            # ingest owner re-drives an interrupted accepted generation
            # (``redrive_accepted_ingests``); a second driver here would race it.
            return None

        def load_started() -> StartedBoundMutation:
            assert self.binding is not None
            # This execution just accepted the request. Another audit
            # continuity transition (a concurrent request's write) holding the
            # lock at this instant is contention, not absence: wait for it on
            # this compute phase, which stays cancellable, rather than refuse
            # an accepted ingest as indeterminate.
            with self.audit.settled_machine_read(wait_for_lock=True):
                parts = self.audit.machine_parts(self.binding)
                if len(parts) != 1 or not parts[0].get("operation_id") or not parts[0].get("authorization_ref"):
                    raise ValueError("legacy accepted source intent lacks an audited ingest execution")
                preview, authorization = self.audit.authorization_for_principal(
                    str(parts[0]["authorization_ref"]), self.context.principal
                )
                return StartedBoundMutation(
                    plan=preview.plan, authorization=authorization, operation_id=str(parts[0]["operation_id"])
                )

        self.started_mutation = await self.runtime.compute_phase(load_started)
        return await self.accepted_generation()

    async def accepted_generation(self) -> RetainedSourceGeneration:
        """Open the accepted retained generation this execution drives."""
        assert self.record is not None
        if self.record["artifact_kind"] != "source-generation":
            raise ValueError("accepted ingest does not reference a source generation")
        generation_id = str(self.record["artifact_ref"])
        generation = await self.read(
            lambda pinned: retained_source_generation_header(pinned.archive.source_connection, generation_id)
        )
        fingerprint = await self.runtime.compute_phase(retained_enumeration_fingerprint)
        if generation.enumeration_fingerprint != fingerprint:
            raise ValueError("accepted source enumeration decoder is no longer available")
        return generation

    async def input_page(
        self, generation: RetainedSourceGeneration, after: tuple[str, str] | None
    ) -> tuple[RetainedSourceInput, ...]:
        page = await self.read(
            lambda pinned: page_retained_source_inputs(
                pinned.archive.source_connection, generation.source_generation_id, after=after
            )
        )
        if any(fingerprint != generation.enumeration_fingerprint for _, fingerprint in page):
            raise ValueError("accepted source generation has mixed decoder identities")
        return tuple(item for item, _ in page)

    async def state(self) -> dict[str, object]:
        def read() -> dict[str, object]:
            assert self.binding is not None
            with self.audit.settled_machine_read():
                record = self.recover_machine_request()
                if record is None:
                    raise ValueError("ingest request has no durable binding")
                self.record = record
                return machine_request_state(self.audit, record)

        return await self.runtime.compute_phase(read)

    def accepted_source_name(self) -> str | None:
        """The source name bound at acceptance, which the request must still carry."""
        source_name = self.request.payload.get("source_name")
        if source_name is not None and not isinstance(source_name, str):
            raise ValueError("accepted ingest source name is not a string")
        assert self.started_mutation is not None
        if source_name != self.started_mutation.plan.context.get("source_name"):
            raise ValueError("accepted ingest source name changed after manifest binding")
        return source_name

    async def enumerate_item(
        self, generation: RetainedSourceGeneration, item: RetainedSourceInput
    ) -> tuple[range, int] | None:
        if item.enumeration_complete:
            return None
        assert self.record is not None
        acquired_at_ms = _record_int(self.record["accepted_at_ms"], field="accepted timestamp")
        source_name = self.accepted_source_name()
        iterator = enumerate_ingest_input(
            item,
            enumeration_fingerprint=generation.enumeration_fingerprint,
            source_generation_id=generation.source_generation_id,
            publisher=self.publisher,
            acquired_at_ms=acquired_at_ms,
            check_stop=self.check_stop,
            source_name=source_name,
        )
        coordinates: PickleSpool[str] = PickleSpool()
        member_count: int | None = None
        published_terminal = False
        excised: _ExcisedRecords | None = None
        try:
            excised = _ExcisedRecords()
            current = await self.runtime.compute_phase(lambda: next(iterator, None))
            while current is not None:
                self.check_stop()
                following = await self.runtime.compute_phase(lambda: next(iterator, None))
                member_count = current.member_count
                await self._publish_record(
                    generation,
                    item,
                    current,
                    coordinates if following is None else None,
                    acquired_at_ms,
                    range(member_count) if following is None and member_count is not None else None,
                    member_count if following is None else None,
                    excised=excised,
                    admitted_coordinates=coordinates,
                )
                published_terminal = following is None
                current = following
            if not coordinates and not published_terminal:
                # The caller completes empty inputs in one writer transaction
                # for the whole accepted input page. This keeps a large set of
                # no-record files from paying a source connection per file.
                count = member_count if member_count is not None else 0
                return range(count), count
        finally:
            try:
                await self.runtime.compute_phase(iterator.close)
            finally:
                if excised is not None:
                    excised.close()
                coordinates.close()
        return None

    async def _publish_record(
        self,
        generation: RetainedSourceGeneration,
        item: RetainedSourceInput,
        prepared: PreparedSourceRecord | PreparedSourceMemberDisposition | None,
        completed_coordinates: Iterable[str] | None,
        observed_at_ms: int,
        completed_member_ordinals: Iterable[int] | None = None,
        member_count: int | None = None,
        *,
        excised: _ExcisedRecords,
        admitted_coordinates: PickleSpool[str],
    ) -> None:
        admitted_raw_ids: list[str] = []

        def publish(connection: sqlite3.Connection) -> None:
            admitted_raw_ids.clear()
            if isinstance(prepared, PreparedSourceMemberDisposition):
                from polylogue.storage.sqlite.archive_tiers.source_items import (
                    SourceItemMemberDisposition,
                    record_source_item_member_disposition,
                )

                record_source_item_member_disposition(
                    connection,
                    source_generation_id=prepared.source_generation_id,
                    source_item_id=prepared.source_item_id,
                    entry_ordinal=prepared.entry_ordinal,
                    member_name=prepared.member_name,
                    disposition=SourceItemMemberDisposition(prepared.disposition),
                    diagnostic=prepared.diagnostic,
                    observed_at_ms=observed_at_ms,
                )
            elif prepared is not None:
                try:
                    admission = execute_source_item_admission(connection, prepared.admission, prepared.member)
                    admitted_coordinates.append(prepared.member.record_coordinate)
                    if admission.arm is not RawAdmissionArm.SKIP_DUPLICATE:
                        admitted_raw_ids.append(admission.raw_id)
                except ContentExcisedError:
                    # The archive forgets on purpose: durably excised bytes are
                    # a skip, not a failed ingest. The admission savepoint left
                    # no raw row; release the publication reservation so blob
                    # GC reclaims the staged bytes, as the live batch does.
                    request = prepared.admission.request
                    consume_blob_publication_receipt(connection, request.blob_publication_receipt_id, request.blob_hash)
                    excised.add(prepared)
                    emit("ingest.excised_record.skipped", raw_id=prepared.admission.raw_id, outcome="skipped")
            if completed_coordinates is not None:
                excised.record_member_dispositions(
                    connection,
                    source_generation_id=generation.source_generation_id,
                    source_item_id=item.source_item_id,
                    observed_at_ms=observed_at_ms,
                )
                complete_source_item_enumeration(
                    connection,
                    source_generation_id=generation.source_generation_id,
                    source_item_id=item.source_item_id,
                    enumeration_fingerprint=generation.enumeration_fingerprint,
                    record_coordinates=completed_coordinates,
                    enumerated_at_ms=observed_at_ms,
                    member_ordinals=completed_member_ordinals,
                    member_count=member_count,
                    check_stop=self.check_stop,
                )

        await self.source_write(publish)
        if admitted_raw_ids:
            self.record_admitted_raws(admitted_raw_ids)

    async def receipt(
        self,
        generation_id: str,
        *,
        index_destination: IndexMutationDestination | None = None,
    ) -> SourceReceiptSpool:
        fd, name = tempfile.mkstemp(prefix="polylogue-source-receipt-", suffix=".sqlite", dir=os.environ.get("TMPDIR"))
        os.close(fd)
        path = Path(name)

        def read_receipt(pinned: PinnedOperationRead) -> SourceReceiptSpool:
            if index_destination is None:
                index_connection = pinned.archive.index_connection
                if index_connection is None:
                    raise RuntimeError("accepted ingest requires the pinned active index tier")
                return _spool_source_receipt(
                    pinned.archive.source_connection,
                    index_connection,
                    generation_id,
                    path,
                    check_stop=self.check_stop,
                )

            self._validate_index_destination(index_destination)
            if index_destination.kind != "owned_inactive" or index_destination.index_path is None:
                raise ValueError("receipt Index destination is not the owned inactive generation")
            from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

            assert index_destination.generation is not None
            try:
                candidate = ArchiveStore.open_owned_inactive_read(index_destination.generation)
            except FileNotFoundError as exc:
                raise IngestReprepareRequiredError("owned cold Index disappeared before receipt read") from exc
            try:
                index_connection = candidate.index_connection
                if index_connection is None:
                    raise RuntimeError("owned cold Index reader has no Index connection")
                if index_path_for_connection(index_connection).resolve(strict=True) != index_destination.index_path:
                    raise IngestReprepareRequiredError("receipt Index connection differs from its owned destination")
                index_connection.execute("BEGIN")
                index_connection.execute("SELECT rootpage FROM main.sqlite_schema LIMIT 1").fetchone()
                self._validate_index_destination(index_destination)
                receipt = _spool_source_receipt(
                    pinned.archive.source_connection,
                    index_connection,
                    generation_id,
                    path,
                    check_stop=self.check_stop,
                    index_destination=index_destination,
                )
                self._validate_index_destination(index_destination)
                return receipt
            finally:
                candidate.close()

        try:
            return await self.read(read_receipt)
        except BaseException:
            unlink_spool(path)
            raise

    async def materialize(self, generation_id: str) -> SourceReceiptSpool:
        """Publish this original generation through the resident Raw owner."""
        initial = await self.receipt(generation_id)
        assert self.snapshot is not None
        expected = self.snapshot.identity
        try:
            with spool_connection(initial.path, read_only=True) as conn:
                retired = conn.execute("SELECT 1 FROM items WHERE retired_count>0 LIMIT 1").fetchone()
            if retired is not None:
                raise ValueError("accepted raw member was retired; it cannot be readmitted")

            def refused(keys: tuple[str, ...], refusal: RetainedRawDecodeRefusalError) -> None:
                for key in keys:
                    self.record_refusal(key, refusal)
                emit(
                    "ingest.membership.refused",
                    logical_source_keys=keys,
                    raw_id=refusal.raw_id,
                    reason=refusal.kind.value,
                    outcome="refused",
                )

            def dependency_refused(refusal: RetainedRawDependencyRefusalError) -> None:
                for key in refusal.logical_source_keys:
                    self.record_membership_refusal(key, refusal.subject_raw_id, "required_raw_dependency_refused")
                emit(
                    "ingest.membership.refused",
                    logical_source_keys=refusal.logical_source_keys,
                    raw_id=refusal.subject_raw_id,
                    reason="required_raw_dependency_refused",
                    dependency_raw_id=refusal.dependency.raw_id,
                    dependency_reason=refusal.dependency.kind.value,
                    outcome="refused",
                )

            def membership_refused(refusal: CohortMembershipRefusalError) -> None:
                self.record_membership_refusal(refusal.logical_source_key, refusal.raw_id, refusal.reason)
                emit(
                    "ingest.membership.refused",
                    logical_source_keys=(refusal.logical_source_key,),
                    raw_id=refusal.raw_id,
                    reason=refusal.reason,
                    outcome="refused",
                )

            cursor: str | None = None
            while raw_page := initial.raw_page(cursor):
                self.check_stop()
                materialization = await self.runtime.materialize_retained_raw_ids(
                    raw_page,
                    on_terminal_refusal=refused,
                    on_dependency_refusal=dependency_refused,
                    on_membership_refusal=membership_refused,
                    before_publication=partial(self.require_publication_identity, expected),
                )
                if self._materialization_destination_bound:
                    if self._materialization_index_destination != materialization.index_destination:
                        raise ReferenceSealStaleError("retained raws used different Index destinations")
                else:
                    self._materialization_index_destination = materialization.index_destination
                    self._materialization_destination_bound = True
                replay = materialization.outcome
                for publication in replay.receipts:
                    for logical_key, raw_id, decision in publication.membership_refusals:
                        self.record_membership_refusal(logical_key, raw_id, decision.value)
                    for session_id, before_hash, after_hash, message_count in publication.session_outputs:
                        if before_hash != after_hash:
                            self.record_changed_session(session_id, message_count)
                        elif not self.changed_session_recorded(session_id):
                            self.unchanged_publications += 1
                # The page's published siblings are accounted above; a raw
                # that failed retryably fails this operation, which retries.
                replay.require_complete()
                cursor = raw_page[-1]
        finally:
            initial.close()
        settled = await self.receipt(
            generation_id,
            index_destination=(
                self._materialization_index_destination if self._materialization_destination_bound else None
            ),
        )
        try:
            await self._attribute_converged_sessions(settled)
        except BaseException:
            settled.close()
            raise
        return settled

    async def _attribute_converged_sessions(self, settled: SourceReceiptSpool) -> None:
        """Count the sessions the archive serves from raws this ingest introduced.

        Under single-pass convergence the daemon's raw owner may publish an
        accepted raw before this materialization reaches it, which then finds
        the content already written. The ingest still reports what the archive
        holds for the inputs it introduced: every session of the settled
        projection whose row is served from one of those raws. A raw that
        duplicated a retained one introduced nothing, so a repeated ingest
        still reports no change.
        """
        cursor = ""
        while True:
            self.check_stop()
            with spool_connection(self.state_path, read_only=True) as state:
                raw_ids = tuple(
                    str(row[0])
                    for row in state.execute(
                        "SELECT raw_id FROM admitted_raws WHERE raw_id>? ORDER BY raw_id LIMIT 256", (cursor,)
                    )
                )
            if not raw_ids:
                return
            cursor = raw_ids[-1]
            complete = set(settled.complete_raw_sessions(raw_ids))
            if not complete:
                continue

            def served_sessions(
                pinned: PinnedOperationRead, raws: tuple[str, ...] = raw_ids
            ) -> tuple[tuple[str, int], ...]:
                index_connection = pinned.archive.index_connection
                if index_connection is None:
                    raise RuntimeError("accepted ingest requires the pinned index tier")
                return tuple(
                    (str(row[0]), int(row[1]))
                    for row in index_connection.execute(
                        f"SELECT session_id, message_count FROM sessions WHERE raw_id IN ({','.join('?' * len(raws))})",
                        raws,
                    )
                )

            for session_id, message_count in await self.read(served_sessions):
                if session_id in complete and not self.changed_session_recorded(session_id):
                    self.record_changed_session(session_id, message_count)

    async def converge_profiles(self, receipt: SourceReceiptSpool) -> tuple[SessionInsightPartReceipt, ...]:
        """Derive only the exact sessions proved by this source denominator.

        Stops at the first page with an unattempted or unsuccessful target;
        ``profile_convergence_complete`` records whether every page was
        carried through, so a stopped convergence cannot read as done.
        """
        self.profile_convergence_complete = False
        if not receipt.complete:
            return ()
        parts: list[SessionInsightPartReceipt] = []
        assert self.started_mutation is not None
        expected_recipe = str(self.started_mutation.plan.context["recipe_version"])
        cursor: str | None = None
        while session_ids := receipt.session_page(cursor):
            self.check_stop()
            part = await self.runtime.converge_ingest_sessions(
                session_ids,
                expected_recipe=expected_recipe,
                stop_requested=self.stop_reason,
            )
            parts.append(part)
            if part.remaining_unattempted_target_refs or any(
                target.disposition not in {"already_satisfied", "published"} for target in part.targets
            ):
                return tuple(parts)
            cursor = session_ids[-1]
        self.profile_convergence_complete = True
        return tuple(parts)

    async def historical_receipt(
        self,
        generation: RetainedSourceGeneration,
        receipt: SourceReceiptSpool,
        profile_parts: tuple[SessionInsightPartReceipt, ...],
        insight_pages: list[IngestInsightPageHistoricalReceipt] | None = None,
    ) -> IngestHistoricalReceiptV2:
        """Persist exact input pages before sealing their bounded audit root."""
        assert self.started_mutation is not None and self.started_mutation.operation_id is not None
        operation_id = self.started_mutation.operation_id
        digest = hashlib.sha256()
        cursor: tuple[str, str] | None = None
        input_count = page_count = 0
        with spool_connection(receipt.path, read_only=True) as observed:
            while source_page := await self.input_page(generation, cursor):
                inputs: list[IngestInputHistoricalReceipt] = []
                for item in source_page:
                    row = observed.execute(
                        "SELECT coordinate, raw_count, unresolved_count, retired_count FROM items WHERE source_item_id=?",
                        (item.source_item_id,),
                    ).fetchone()
                    if row is None:
                        inputs.append(
                            IngestInputHistoricalReceipt(
                                source_item_id=item.source_item_id,
                                logical_coordinate=item.coordinate,
                                denominator=0,
                                raw_ids=None,
                                unknown_attribution="source item missing from terminal projection",
                            )
                        )
                        continue
                    if str(row[0]) != item.coordinate:
                        raise ValueError("terminal source item coordinate changed")
                    raw_count, unresolved_count, retired_count = map(int, row[1:])
                    page_metadata = observed.execute(
                        "SELECT page_ref, page_count, raw_count, unresolved_count, digest "
                        "FROM raw_metadata WHERE source_item_id=?",
                        (item.source_item_id,),
                    ).fetchone()
                    raw_ids: list[str] | None = None
                    unresolved: list[str] = []
                    if page_metadata is None:
                        if raw_count > MAX_INLINE_RAW_IDS_PER_INPUT:
                            raise ValueError("terminal raw attribution lacks its persisted pages")
                        with closing(
                            observed.execute(
                                "SELECT raw_id, complete FROM item_raws WHERE source_item_id=? ORDER BY raw_id LIMIT ?",
                                (item.source_item_id, MAX_INLINE_RAW_IDS_PER_INPUT),
                            )
                        ) as raw_cursor:
                            raw_status = raw_cursor.fetchall()
                        raw_ids = [str(raw_id) for raw_id, _complete in raw_status]
                        unresolved = [str(raw_id) for raw_id, complete in raw_status if not complete]
                        if len(raw_ids) != raw_count or len(unresolved) != unresolved_count:
                            raise ValueError("terminal raw attribution changed")
                    inputs.append(
                        IngestInputHistoricalReceipt(
                            source_item_id=item.source_item_id,
                            logical_coordinate=item.coordinate,
                            denominator=raw_count + retired_count,
                            raw_ids=None if page_metadata is not None else raw_ids,
                            unresolved_raw_ids=[] if page_metadata is not None else unresolved,
                            raw_id_pages_ref=None if page_metadata is None else page_metadata[0],
                            raw_id_page_count=0 if page_metadata is None else page_metadata[1],
                            raw_id_count=0 if page_metadata is None else page_metadata[2],
                            unresolved_raw_count=0 if page_metadata is None else page_metadata[3],
                            raw_ids_digest=None if page_metadata is None else page_metadata[4],
                            unknown_attribution=(
                                "retired source record has no raw identity" if retired_count else None
                            ),
                        )
                    )
                input_page = IngestInputPageHistoricalReceipt.from_items(page_count, inputs)

                def persist_input_page(page: IngestInputPageHistoricalReceipt = input_page) -> None:
                    self.audit.append_ingest_input_page(operation_id, page)

                await self.runtime.write_phase("ingest.input_page", persist_input_page)
                digest.update(f"{input_page.ordinal}:{input_page.digest}\n".encode("ascii"))
                input_count += len(inputs)
                page_count += 1
                cursor = (source_page[-1].coordinate, source_page[-1].source_item_id)
        if input_count != generation.item_count or input_count != receipt.item_count:
            raise ValueError("terminal source input projection differs from its accepted denominator")
        if insight_pages is None:
            insight_pages = self._insight_pages(profile_parts)
        root = IngestHistoricalReceiptV2(
            source_generation_id=generation.source_generation_id,
            final_sequence=1,
            input_count=input_count,
            input_pages_ref=operation_id,
            input_page_count=page_count,
            input_pages_digest=digest.hexdigest(),
            insight_pages=[] if self.insight_pages_ref is not None else insight_pages,
            insight_pages_ref=self.insight_pages_ref,
            insight_page_count=self.insight_page_count,
            insight_pages_digest=self.insight_pages_digest,
            summary=IngestTerminalSummaryHistorical(
                enumeration_complete=receipt.enumeration_complete,
                source_complete=receipt.complete and self.source_items_sealable and self.refused_count == 0,
                confirmed_raw_count=receipt.confirmed_raw_count,
                unresolved_raw_count=receipt.unresolved_raw_count,
                profile_targets_observed=sum(len(part.targets) for part in profile_parts),
                refused_membership_count=self.refused_count,
                refused_memberships=self.inline_refusals,
                refused_membership_pages_ref=self.refusal_pages_ref,
                refused_membership_page_count=self.refusal_page_count,
                refused_memberships_digest=self.refusal_pages_digest,
                parse_projection_known=True,
                processed_session_ids=self.inline_session_ids,
                processed_session_id_pages_ref=self.session_id_pages_ref,
                processed_session_id_page_count=self.session_id_page_count,
                processed_session_ids_digest=self.session_ids_digest,
                processed_message_count=self.changed_message_count,
                changed_session_count=self.changed_session_count,
                changed_message_count=self.changed_message_count,
                profile_convergence_complete=self.profile_convergence_complete,
            ),
        )
        return root

    @staticmethod
    def _insight_pages(
        profile_parts: tuple[SessionInsightPartReceipt, ...],
    ) -> list[IngestInsightPageHistoricalReceipt]:
        return [
            IngestInsightPageHistoricalReceipt(
                ordinal=ordinal,
                targets=[InsightTargetHistoricalReceipt.model_validate(asdict(target)) for target in part.targets],
                unattempted_target_refs=list(part.remaining_unattempted_target_refs),
            )
            for ordinal, part in enumerate(profile_parts)
        ]

    async def finalize(
        self,
        generation: RetainedSourceGeneration,
        receipt: SourceReceiptSpool,
        profile_parts: tuple[SessionInsightPartReceipt, ...],
    ) -> IngestHistoricalReceiptV2:
        assert self.started_mutation is not None
        await self.settle_source_items(generation, receipt)
        started = self.started_mutation
        operation_id = started.operation_id
        assert operation_id is not None
        with spool_connection(receipt.path) as observed:
            item_cursor: tuple[int] | None = None
            while True:
                self.check_stop()
                with closing(
                    observed.execute(
                        "SELECT ordinal, source_item_id, raw_count, unresolved_count FROM items "
                        "WHERE ordinal>? ORDER BY ordinal LIMIT 1",
                        (-1 if item_cursor is None else item_cursor[0],),
                    )
                ) as cursor:
                    item = cursor.fetchone()
                if item is None:
                    break
                ordinal, item_id, raw_count, unresolved_count = item
                item_cursor = (int(ordinal),)
                if raw_count <= MAX_INLINE_RAW_IDS_PER_INPUT:
                    continue
                raw_digest = IngestInputRawPagesDigest()
                raw_page_count = 0
                after_raw = ""
                observed_raw_count = observed_unresolved = 0
                while True:
                    self.check_stop()
                    with closing(
                        observed.execute(
                            "SELECT raw_id, complete FROM item_raws WHERE source_item_id=? AND raw_id>? "
                            "ORDER BY raw_id LIMIT ?",
                            (item_id, after_raw, MAX_PAGE_ITEMS),
                        )
                    ) as cursor:
                        rows = cursor.fetchall()
                    if not rows:
                        break
                    raw_page = IngestInputRawPageHistoricalReceipt(
                        source_item_id=str(item_id),
                        ordinal=raw_page_count,
                        raws=[
                            IngestInputRawMemberHistorical(raw_id=str(raw_id), unresolved=not complete)
                            for raw_id, complete in rows
                        ],
                    )
                    raw_digest.update(raw_page)
                    observed_raw_count += len(rows)
                    observed_unresolved += sum(not complete for _raw_id, complete in rows)
                    after_raw = str(rows[-1][0])
                    raw_page_count += 1

                    def persist_raw_page(page: IngestInputRawPageHistoricalReceipt = raw_page) -> None:
                        self.audit.append_ingest_input_raw_page(operation_id, page)

                    await self.runtime.write_phase("ingest.input_raw_page", persist_raw_page)
                if (observed_raw_count, observed_unresolved) != (raw_count, unresolved_count):
                    raise ValueError("terminal raw attribution changed")
                observed.execute(
                    "INSERT INTO raw_metadata VALUES (?, ?, ?, ?, ?, ?)",
                    (item_id, operation_id, raw_page_count, raw_count, unresolved_count, raw_digest.hexdigest()),
                )
        with spool_connection(self.state_path, read_only=True) as state:
            counts = state.execute("SELECT COUNT(*), COALESCE(SUM(message_count), 0) FROM changed_sessions").fetchone()
            self.changed_session_count, self.changed_message_count = int(counts[0]), int(counts[1])
            self.refused_count = int(state.execute("SELECT COUNT(*) FROM refusals").fetchone()[0])
            refusal_cursor = state.execute("SELECT logical_key, raw_id, reason FROM refusals ORDER BY ordinal")
            if self.refused_count <= MAX_PAGE_ITEMS:
                self.inline_refusals = [
                    IngestRefusedMembershipHistorical(
                        logical_source_key=str(key), raw_id=str(raw_id), reason=str(reason)
                    )
                    for key, raw_id, reason in refusal_cursor
                ]
            else:
                # Each page is persisted as it is built and folded into the
                # digest, so memory holds one page, not every refusal.
                refusal_digest = IngestRefusalPagesDigest()
                refusal_page_count = 0
                while refusal_rows := refusal_cursor.fetchmany(MAX_PAGE_ITEMS):
                    self.check_stop()
                    refusal_page = IngestRefusalPageHistoricalReceipt(
                        ordinal=refusal_page_count,
                        refusals=[
                            IngestRefusedMembershipHistorical(
                                logical_source_key=str(key), raw_id=str(raw_id), reason=str(reason)
                            )
                            for key, raw_id, reason in refusal_rows
                        ],
                    )

                    def persist_refusal_page(page: IngestRefusalPageHistoricalReceipt = refusal_page) -> None:
                        self.audit.append_ingest_refusal_page(operation_id, page)

                    await self.runtime.write_phase("ingest.refusal_page", persist_refusal_page)
                    refusal_digest.update(refusal_page)
                    refusal_page_count += 1
                self.refusal_pages_ref = operation_id
                self.refusal_page_count = refusal_page_count
                self.refusal_pages_digest = refusal_digest.hexdigest()
            if self.changed_session_count <= MAX_INLINE_INGEST_SESSION_IDS:
                self.inline_session_ids = [
                    str(row[0]) for row in state.execute("SELECT session_id FROM changed_sessions ORDER BY session_id")
                ]
            else:
                self.session_id_pages_ref = operation_id
                self.session_id_page_count = (self.changed_session_count + MAX_PAGE_ITEMS - 1) // MAX_PAGE_ITEMS
                digest = hashlib.sha256()
                digest.update(b"[")
                emitted = 0
                cursor = state.execute("SELECT session_id FROM changed_sessions ORDER BY session_id")
                ordinal = 0
                while rows := cursor.fetchmany(MAX_PAGE_ITEMS):
                    id_page = tuple(str(row[0]) for row in rows)
                    for session_id in id_page:
                        digest.update(("," if emitted else "").encode("ascii"))
                        digest.update(json.dumps(session_id, ensure_ascii=False, separators=(",", ":")).encode())
                        emitted += 1

                    def persist_id_page(page: tuple[str, ...] = id_page, page_ordinal: int = ordinal) -> None:
                        self.audit.append_ingest_session_id_page(operation_id, page_ordinal, page)

                    await self.runtime.write_phase("ingest.session_ids", persist_id_page)
                    ordinal += 1
                digest.update(b"]")
                if emitted != self.changed_session_count:
                    raise ValueError("changed session projection changed while paging")
                self.session_ids_digest = digest.hexdigest()
        insight_pages = self._insight_pages(profile_parts)
        if len(insight_pages) > MAX_MACHINE_RECEIPT_PAGES:
            self.insight_pages_ref = operation_id
            self.insight_page_count = len(insight_pages)
            self.insight_pages_digest = ingest_insight_pages_digest(insight_pages)
            for insight_page in insight_pages:

                def persist_insight_page(page: IngestInsightPageHistoricalReceipt = insight_page) -> None:
                    self.audit.append_ingest_insight_page(operation_id, page)

                await self.runtime.write_phase(
                    "ingest.insight_page",
                    persist_insight_page,
                )
        history = await self.historical_receipt(generation, receipt, profile_parts, insight_pages)
        final = MutationReceipt(
            operation=started.plan.operation,
            plan_hash=started.plan.plan_hash,
            status="applied",
            target_refs=started.plan.target_refs,
            affected_count=1,
            detail=None,
            receipt_ref=None,
            applied_at=started.plan.prepared_at,
            historical_receipt=history,
        )
        binding = self.binding

        def finalize_unless_stopped() -> None:
            # Under the writer: a cancel or deadline fence committed after the
            # last ``check_stop`` wins over this terminal write.
            record = self.audit.machine_request(binding) if binding is not None else None
            if record is not None and record.get("stop_reason"):
                raise IngestStoppedError(str(record["stop_reason"]))
            # The accepted deadline and authorization expiry are never written
            # into ``stop_reason``: they are evaluated here, now, as well.
            expired = self.durable_stop_reason(None)
            if expired is not None:
                raise IngestStoppedError(expired)
            self.executor.finalize_bound(started, receipt=final)

        await self.runtime.write_phase("ingest.finalize", finalize_unless_stopped)
        self.terminalized = True
        return history

    async def mark_unknown(self, reason: str) -> None:
        if self.terminalized:
            return
        if self.started_mutation is None:
            # Accepted, but the authority load failed (a transient admission
            # refusal right after ``accept_ingest`` committed): the durable
            # attempt is settled by its operation id, never left running
            # under this live process.
            operation_id = await self._accepted_operation_id()
            if operation_id is None:
                return
            await self.runtime.write_phase(
                "ingest.unknown",
                lambda: self.audit.finalize_attempt(operation_id, status="unknown", unknown_reason=reason[:512]),
            )
            self.terminalized = True
            return
        started = self.started_mutation
        await self.runtime.write_phase(
            "ingest.unknown",
            lambda: self.executor.finalize_bound(started, unknown_reason=reason[:512]),
        )
        self.terminalized = True

    async def _accepted_operation_id(self) -> str | None:
        binding = self.binding
        if binding is None:
            return None

        def read() -> str | None:
            with self.audit.settled_machine_read():
                parts = self.audit.machine_parts(binding)
            operation_id = parts[0].get("operation_id") if len(parts) == 1 else None
            return str(operation_id) if operation_id else None

        return await asyncio.to_thread(read)

    async def fence(self, reason: str) -> None:
        if self.binding is not None:
            binding = self.binding

            def stop() -> None:
                record = self.audit.machine_request(binding)
                if record is not None:
                    self.audit.stop_machine_batch(binding, reason)

            await self.runtime.write_phase("ingest.stop", stop)


class IngestRedrive(IngestExecution):
    """The ingest owner's attempt on an accepted generation left without a terminal checkpoint.

    It resumes the original operation run under a new attempt and drives the
    retained manifest through the same phases as a fresh request. It reads no
    request payload: the accepted plan binds the source name, and the input
    path is not read again, so the original input need not exist. It stops
    when its owner stops, the request is cancelled, or the request's durable
    accepted deadline or authorization expiry passes, exactly as the
    original attempt would have.
    """

    def __init__(
        self,
        runtime: OperationRuntime,
        archive_root: Path,
        audit: AuditRepository,
        *,
        operation_id: str,
        record: dict[str, object],
        stop_requested: Callable[[], str | None],
    ) -> None:
        self._setup(runtime, archive_root, audit)
        self.operation_id = operation_id
        self.record = record
        self.binding = MachineRequestBinding(
            str(record["archive_identity"]),
            str(record["request_id"]),
            str(record["principal_ref"]),
            str(record["fingerprint"]),
            str(record["operation_name"]),
        )
        self._stop_requested = stop_requested
        #: Whether generation content was materialized before this re-drive's
        #: first drive -- decided once: a transient retry sees this attempt's
        #: own work, whose projection it still holds.
        self.prior_materialization: bool | None = None

    def stop_reason(self) -> str | None:
        return self._stop_requested() or self.durable_stop_reason(None)

    async def finalize(
        self,
        generation: RetainedSourceGeneration,
        receipt: SourceReceiptSpool,
        profile_parts: tuple[SessionInsightPartReceipt, ...],
    ) -> IngestHistoricalReceiptV2:
        if self.prior_materialization or self.unchanged_publications:
            # The interrupted attempt may have published some of these, and
            # its changed-session projection died with it; an applied receipt
            # would under-report what this request wrote.
            # Fenced as well as settled: the outcome is decided (indeterminate),
            # so no later start re-drives the whole generation only to reach
            # this same verdict.
            await self.fence("refused")
            await self.mark_unknown(
                "the interrupted attempt's changed-session projection is not recoverable: "
                "generation content was already materialized when the re-drive began"
            )
            raise IngestProjectionUnrecoverableError("re-drive finalized as indeterminate")
        return await super().finalize(generation, receipt, profile_parts)

    def first_snapshot(self, snapshot: PinnedOperationRead) -> None:
        return None

    def observe_snapshot(self, snapshot: PinnedOperationRead) -> None:
        return None

    def accepted_source_name(self) -> str | None:
        assert self.started_mutation is not None
        source_name = self.started_mutation.plan.context.get("source_name")
        if source_name is not None and not isinstance(source_name, str):
            raise ValueError("accepted ingest source name is not a string")
        return source_name

    async def resume(self) -> RetainedSourceGeneration:
        def load_started() -> StartedBoundMutation:
            with self.audit.settled_machine_read():
                plan, authorization = self.audit.ingest_operation_authority(self.operation_id)
            return StartedBoundMutation(plan=plan, authorization=authorization, operation_id=self.operation_id)

        self.started_mutation = await self.runtime.compute_phase(load_started)
        generation = await self.accepted_generation()

        def already_materialized() -> bool:
            from polylogue.operations.ingest_acceptance import generation_materialized

            return generation_materialized(self.archive_root, generation.source_generation_id)

        # A generation raw already materialized may be the interrupted
        # attempt's work, whose changed-session projection died with it.
        if self.prior_materialization is None:
            self.prior_materialization = await self.runtime.compute_phase(already_materialized)
        return generation

    def repin(self) -> None:
        """Forget the pinned archive identity before a retry after the generation moved."""
        self.observed_identity = None

    async def release(self, reason: str) -> None:
        """Hand a claimed run back as interrupted, so any later owner -- even one in this process -- reclaims it."""
        if self.terminalized:
            return
        if self.started_mutation is not None:
            await self.mark_unknown(reason)
            return
        await release_redrive_claim(self.runtime, self.audit, self.operation_id, reason)
        self.terminalized = True

    async def settle_failed(self, reason: str) -> None:
        """Terminalize this attempt as failed; a retry would meet the same refusal."""
        if self.terminalized:
            return
        await self.runtime.write_phase(
            "ingest.redrive-failed",
            lambda: self.audit.finalize_attempt(self.operation_id, status="failed", error_summary=reason[:512]),
        )
        self.terminalized = True


def transient_storage_fault(exc: BaseException) -> bool:
    """A storage read that may succeed on retry: a busy or locked database, a
    file momentarily unopenable or unreadable. A full disk, corruption or a
    read-only archive are not transient."""
    if isinstance(exc, sqlite3.OperationalError):
        message = str(exc).lower()
        return any(token in message for token in ("database is locked", "database is busy", "unable to open"))
    if isinstance(exc, OSError):
        return exc.errno in {errno.EACCES, errno.EAGAIN, errno.EBUSY, errno.EINTR}
    return False


async def claim_interrupted_ingest(runtime: OperationRuntime, audit: AuditRepository, operation_id: str) -> bool:
    """Take an interrupted run under a new attempt owned by this process, if still unowned."""
    return await runtime.write_phase("ingest.redrive", lambda: audit.resume_interrupted_ingest(operation_id))


async def release_redrive_claim(
    runtime: OperationRuntime, audit: AuditRepository, operation_id: str, reason: str
) -> None:
    """Hand a claimed, not yet driven run back as interrupted with no running attempt."""
    await runtime.write_phase(
        "ingest.redrive-release",
        lambda: audit.finalize_attempt(operation_id, status="unknown", unknown_reason=reason[:512]),
    )


async def redrive_accepted_ingests(
    runtime: OperationRuntime,
    archive_root: Path,
    *,
    stop_requested: Callable[[str], str | None],
    on_commit: Callable[[], None] | None = None,
    on_claimed: Callable[[], None] | None = None,
) -> None:
    """Drive every accepted, unstopped ingest without a terminal checkpoint to its receipt.

    This is the daemon ingest owner's recovery of work a dead process
    accepted. Every eligible run is claimed under a new attempt first, and
    ``on_claimed`` fires once all are claimed, so a caller can hold its
    listeners until a resent request reads ``running`` rather than
    ``interrupted``. Runs are then driven one at a time. A transient refusal
    (the index generation moved, control admission is full) retries the same
    claimed run; a stop settles like a fresh request's stop (fenced and
    indeterminate, since effects may be partial); a generation that cannot
    be driven at all terminalizes as failed. An owner shutdown leaves the run
    for the next owner. ``stop_requested`` receives the request id.
    """
    from polylogue.core.compute import DaemonBackpressureError
    from polylogue.operations.audit import AuditRepository

    # Claims are durable rows; the execution (its scratch state file and blob
    # publisher) is built only when its run is driven, one at a time.
    claimed: list[tuple[str, dict[str, object]]] = []
    released = "the ingest owner stopped before the terminal checkpoint"
    try:
        if not (archive_root / "audit.db").is_file():
            return
        audit = AuditRepository(
            archive_root / "audit.db",
            attempt_owner_id=AuditRepository.current_process_attempt_owner(),
            on_commit=on_commit,
        )

        def discover() -> tuple[tuple[str, dict[str, object]], ...]:
            with audit.settled_machine_read():
                return audit.interrupted_ingest_requests()

        try:
            for operation_id, record in await runtime.compute_phase(discover):
                if stop_requested(str(record["request_id"])) == "shutdown":
                    # Claims already taken go back too: their attempts name
                    # this live process, which a recreated server could
                    # never reclaim.
                    for claimed_id, _record in claimed:
                        with contextlib.suppress(Exception):
                            await release_redrive_claim(runtime, audit, claimed_id, released)
                    return
                if await claim_interrupted_ingest(runtime, audit, operation_id):
                    claimed.append((operation_id, record))
        except BaseException:
            # A claim phase that fails midway hands back what it took, for
            # the same reason.
            for claimed_id, _record in claimed:
                with contextlib.suppress(Exception):
                    await release_redrive_claim(runtime, audit, claimed_id, "the claim phase failed")
            raise
    finally:
        if on_claimed is not None:
            on_claimed()

    for position, (operation_id, record) in enumerate(claimed):
        request_id = str(record["request_id"])
        try:
            execution = IngestRedrive(
                runtime,
                archive_root,
                audit,
                operation_id=operation_id,
                record=record,
                stop_requested=partial(stop_requested, request_id),
            )
        except BaseException as exc:
            # No execution exists to drive or settle this run (its scratch
            # state could not be created): hand it and every later claim
            # back, since their attempts name this live process and nothing
            # in it would ever reclaim them.
            reason = f"the re-drive could not start: {type(exc).__name__}"
            for remaining_id, _record in claimed[position:]:
                with contextlib.suppress(Exception):
                    await release_redrive_claim(runtime, audit, remaining_id, reason)
            raise
        try:
            emit("ingest.redrive.started", operation_id=operation_id, request_id=request_id, outcome="running")
            backoff_s = 0.5
            while True:
                try:
                    generation = await execution.resume()
                    history = await drive_accepted_generation(execution, generation)
                except Exception as exc:
                    if not (
                        isinstance(
                            exc, (IngestReprepareRequiredError, ArchiveIdentityStaleError, DaemonBackpressureError)
                        )
                        or transient_storage_fault(exc)
                    ):
                        raise
                    # Transient: the claim stays with this owner and the same
                    # run is driven again; every phase settles by content hash.
                    emit(
                        "ingest.redrive.retry",
                        operation_id=operation_id,
                        request_id=request_id,
                        outcome="running",
                        error_type=type(exc).__name__,
                        error_detail=str(exc)[:512],
                    )
                    # Cleanup takes no admission: a full control class would
                    # refuse it too and escape this retry.
                    await asyncio.to_thread(execution.publisher.discard_pending)
                    execution.repin()
                    execution.check_stop()
                    await asyncio.sleep(backoff_s)
                    backoff_s = min(backoff_s * 2, 30.0)
                    continue
                emit(
                    "ingest.redrive.finished",
                    operation_id=operation_id,
                    request_id=request_id,
                    outcome=ingest_terminal_outcome(history),
                )
                break
        except IngestStoppedError as exc:
            if exc.reason == "shutdown":
                # Every claimed run goes back as interrupted: the attempts
                # name this live process, so a server recreated in it could
                # otherwise never reclaim them.
                try:
                    await execution.release(released)
                finally:
                    # One failed release must not strand the later claims.
                    for remaining_id, _record in claimed[position + 1 :]:
                        with contextlib.suppress(Exception):
                            await release_redrive_claim(runtime, audit, remaining_id, released)
                return
            # As a fresh request's stop: fenced first, then indeterminate,
            # because sessions published before the stop remain.
            await execution.fence(exc.reason)
            await execution.mark_unknown(exc.reason)
        except IngestProjectionUnrecoverableError:
            emit(
                "ingest.redrive.finished",
                operation_id=operation_id,
                request_id=request_id,
                outcome="indeterminate",
            )
        except Exception as exc:
            emit(
                "ingest.redrive.failed",
                level=WARNING,
                operation_id=operation_id,
                request_id=request_id,
                outcome="failed",
                error_type=type(exc).__name__,
                error_detail=str(exc)[:512],
            )
            await execution.settle_failed(f"{type(exc).__name__}: {exc}")
        finally:
            await asyncio.to_thread(execution.publisher.discard_pending)
            unlink_spool(execution.state_path)


async def drive_accepted_generation(
    execution: IngestExecution, generation: RetainedSourceGeneration
) -> IngestHistoricalReceiptV2:
    """Enumerate, materialize, converge and finalize one accepted generation.

    Every phase reads the retained manifest and settles by content hash, so a
    fresh request and a re-drive of an interrupted one take the same route
    from whatever state an earlier attempt left.
    """
    cursor: tuple[str, str] | None = None
    seen = 0
    while page := await execution.input_page(generation, cursor):
        empty: list[tuple[RetainedSourceInput, range, int]] = []
        for item in page:
            execution.check_stop()
            completion = await execution.enumerate_item(generation, item)
            if completion is not None:
                empty.append((item, *completion))
        if empty:
            assert execution.record is not None
            accepted_at_ms = _record_int(execution.record["accepted_at_ms"], field="accepted timestamp")

            def complete_empty_page(
                conn: sqlite3.Connection,
                *,
                batch: tuple[tuple[RetainedSourceInput, range, int], ...] = tuple(empty),
                observed_at_ms: int = accepted_at_ms,
            ) -> None:
                for item, ordinals, member_count in batch:
                    complete_source_item_enumeration(
                        conn,
                        source_generation_id=generation.source_generation_id,
                        source_item_id=item.source_item_id,
                        enumeration_fingerprint=generation.enumeration_fingerprint,
                        record_coordinates=(),
                        enumerated_at_ms=observed_at_ms,
                        member_ordinals=ordinals,
                        member_count=member_count,
                        check_stop=execution.check_stop,
                    )

            await execution.source_write(complete_empty_page)
        seen += len(page)
        if seen > generation.item_count:
            raise ValueError("accepted source generation exceeds its bound count")
        cursor = (page[-1].coordinate, page[-1].source_item_id)
    if seen != generation.item_count:
        raise ValueError("accepted source generation has an incomplete manifest")
    receipt = await execution.materialize(generation.source_generation_id)
    try:
        if execution.retained_profile_parts is None:
            execution.retained_profile_parts = await execution.converge_profiles(receipt)
        profile_parts = execution.retained_profile_parts
        execution.check_stop()
        return await execution.finalize(generation, receipt, profile_parts)
    finally:
        receipt.close()


async def execute_ingest_operation(
    request: DaemonOperationRequest, context: OperationContext
) -> DaemonOperationEnvelope:
    """Accept immutable input before any raw admission, then settle each phase."""
    from polylogue.core.compute import DaemonBackpressureError

    started = monotonic()
    request = validate_execution_request(request, context)
    execution = IngestExecution(request, context)
    try:
        generation = await execution.accept()
        if execution.resumed:
            state = await execution.state()
            restored = state.get("result")
            if state["outcome"] in {"completed", "degraded"} and isinstance(restored, dict):
                from polylogue.operations.machine_receipts import decode_machine_receipt

                # The durable state wraps the receipt; decode the receipt itself.
                history = decode_machine_receipt(restored.get("historical_receipt"))
                if not isinstance(history, IngestHistoricalReceiptV2):
                    raise ValueError("completed ingest has another historical receipt kind")
                outcome = ingest_terminal_outcome(history)
                return operation_envelope(
                    request,
                    context,
                    snapshot=execution.snapshot,
                    started_at=started,
                    outcome=outcome,
                    reference=execution.record,
                    error=None if outcome == "completed" else ingest_unconverged_error(history),
                    result={
                        "source_generation_id": history.source_generation_id,
                        "outcome": outcome,
                        "sequence": history.final_sequence,
                        "historical_receipt": history.model_dump(mode="json"),
                    },
                )
            # A prior attempt passed durable acceptance without a terminal
            # checkpoint. Its ingest owner re-drives it from the retained
            # manifest; this request reports the durable state it observes.
            return operation_envelope(
                request,
                context,
                snapshot=execution.snapshot,
                started_at=started,
                outcome=str(state["outcome"]),
                reference=execution.record,
                result=state,
            )
        assert generation is not None
        history = await drive_accepted_generation(execution, generation)
        outcome = ingest_terminal_outcome(history)
        return operation_envelope(
            request,
            context,
            snapshot=execution.snapshot,
            started_at=started,
            outcome=outcome,
            reference=execution.record,
            error=None if outcome == "completed" else ingest_unconverged_error(history),
            result={
                "source_generation_id": generation.source_generation_id,
                "outcome": outcome,
                "sequence": history.final_sequence,
                "historical_receipt": history.model_dump(mode="json"),
            },
        )
    except IngestStoppedError as exc:
        # Fence first: once the request carries its stop reason, a process
        # death before the attempt settles cannot hand it to the re-driver.
        if execution.acceptance_attempted:
            await execution.fence(exc.reason)
            await execution.mark_unknown(exc.reason)
        return operation_envelope(
            request,
            context,
            snapshot=execution.snapshot,
            started_at=started,
            outcome="cancelled" if exc.reason == "cancelled" else "timed-out",
            reference=execution.record,
        )
    except (IngestReprepareRequiredError, ArchiveIdentityStaleError, DaemonBackpressureError):
        # Transient after acceptance: left unstopped, the accepted generation
        # stays eligible for its ingest owner's re-drive.
        if execution.acceptance_attempted:
            await execution.mark_unknown("accepted ingest met a transient refusal before its terminal checkpoint")
        raise
    except Exception:
        # A failure before acceptance (such as every input refused as
        # excised) has no accepted attempt to fence or settle. Settling anyway
        # takes a settled audit read, which a concurrent reader (the caller
        # polling this request) makes fail, and that error would replace the
        # typed refusal.
        if execution.acceptance_attempted:
            await execution.fence("refused")
            await execution.mark_unknown("accepted ingest lacks a terminal checkpoint")
        raise
    finally:
        await asyncio.to_thread(execution.publisher.discard_pending)
        unlink_spool(execution.state_path)
