"""Staged retained-input ingestion on the existing daemon compute and writer."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import monotonic, time
from typing import TypeVar
from uuid import uuid4

from polylogue.logging import emit
from polylogue.operations.audit import MachineRequestBinding
from polylogue.operations.bindings import runtime_operation_binding
from polylogue.operations.daemon_execution import _validate_identity, operation_envelope, validate_execution_request
from polylogue.operations.daemon_protocol import DaemonOperationEnvelope, DaemonOperationRequest
from polylogue.operations.ingest_acceptance import IngestActuator, ingest_plan
from polylogue.operations.ingest_inputs import (
    PreparedSourceMemberDisposition,
    PreparedSourceRecord,
    discover_ingest_input_spool,
    enumerate_ingest_input,
    retain_input_page,
    spool_connection,
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
    IngestInsightPageHistoricalReceipt,
    IngestRefusalPageHistoricalReceipt,
    IngestRefusalPagesDigest,
    IngestRefusedMembershipHistorical,
    IngestTerminalSummaryHistorical,
    InsightTargetHistoricalReceipt,
    ingest_input_raw_pages_digest,
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
from polylogue.operations.operation_context import OperationContext, PinnedOperationRead, open_operation_read
from polylogue.sources.origin_specs import retained_enumeration_fingerprint
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.revision_backfill import enrich_sessions_from_archive, parse_retained_raw_sessions
from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation
from polylogue.storage.blob_publication import ArchiveBlobPublisher, consume_blob_publication_receipt
from polylogue.storage.ingest_governance import (
    CensusPublication,
    CohortMembershipRefusalError,
    CohortPublication,
    PreparedIngestCohort,
    PreparedRawCensus,
    discard_prepared_ingest_cohort,
    prepare_ingest_cohort,
    prepare_raw_census,
    publish_ingest_cohort,
    publish_raw_census,
)
from polylogue.storage.raw_authority import RAW_AUTHORITY_PARSER_FINGERPRINT
from polylogue.storage.source_generation_receipts import source_generation_receipt_page
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.raw_admission import execute_source_item_admission
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

_T = TypeVar("_T")


@dataclass(slots=True)
class _ExcisedRecords:
    """Records of one source item the source tier refused as durably excised.

    An excised record never becomes a raw member, so it leaves the item's
    record denominator. A ZIP member whose every record was excised keeps its
    central-directory ordinal accounted as a refused disposition.
    """

    coordinates: set[str] = field(default_factory=set)
    members: dict[int, str] = field(default_factory=dict)

    def add(self, prepared: PreparedSourceRecord) -> None:
        self.coordinates.add(prepared.member.record_coordinate)
        if prepared.member.entry_ordinal is not None:
            self.members[prepared.member.entry_ordinal] = str(prepared.record.source_path)

    def record_member_dispositions(
        self, connection: sqlite3.Connection, *, source_generation_id: str, source_item_id: str, observed_at_ms: int
    ) -> None:
        from polylogue.storage.sqlite.archive_tiers.source_items import (
            SourceItemMemberDisposition,
            record_source_item_member_disposition,
        )

        for ordinal, member_name in sorted(self.members.items()):
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


def _parse_assembled_retained_raw(archive: ArchiveStore, raw_id: str) -> list[ParsedSession]:
    """Parse one retained raw and apply its provider's session assembly.

    The same composition the live writer uses
    (``LiveBatchProcessor._parse_retained_raw_sessions``), so a session
    admitted through this operation carries the same assembled title and
    enrichment as one admitted by the watcher or a from-empty build.
    """
    sessions = parse_retained_raw_sessions(archive, raw_id)
    provider, _blob_hash, source_path, _kind, _size = archive.raw_revision_descriptor(raw_id)
    return enrich_sessions_from_archive(archive, provider, source_path, sessions)


class IngestStoppedError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class SourceReceiptSpool:
    """One pinned source/index projection reduced on disk by raw and logical ID."""

    path: Path
    source_generation_id: str
    item_count: int
    enumeration_complete: bool
    complete: bool
    confirmed_raw_count: int
    unresolved_raw_count: int

    def close(self) -> None:
        self.path.unlink(missing_ok=True)

    def pending_raw_page(self, after: str | None = None) -> tuple[str, ...]:
        with spool_connection(self.path, read_only=True) as conn:
            return tuple(
                str(row[0])
                for row in conn.execute(
                    "SELECT raw_id FROM raws WHERE parser_complete=0 AND raw_id>? ORDER BY raw_id LIMIT 256",
                    (after or "",),
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
            "raws_json TEXT NOT NULL, retired_count INTEGER NOT NULL);"
            "CREATE UNIQUE INDEX items_source_id ON items(source_item_id);"
            "CREATE TABLE raws(raw_id TEXT PRIMARY KEY, complete INTEGER NOT NULL, parser_complete INTEGER NOT NULL) WITHOUT ROWID;"
            "CREATE TABLE logicals(logical_key TEXT PRIMARY KEY, expected_session_id TEXT NOT NULL, complete INTEGER NOT NULL) WITHOUT ROWID;"
            "CREATE INDEX logicals_session ON logicals(expected_session_id);"
            "CREATE TABLE raw_metadata(source_item_id TEXT PRIMARY KEY, page_ref TEXT NOT NULL, page_count INTEGER NOT NULL, "
            "raw_count INTEGER NOT NULL, unresolved_count INTEGER NOT NULL, digest TEXT NOT NULL) WITHOUT ROWID;"
        )
        while True:
            page = source_generation_receipt_page(
                source_conn, index_conn, source_generation_id=generation_id, after=cursor
            )
            if not page.items:
                break
            retired_by_item: dict[str, int] = {}
            for retired in page.retired_coordinates:
                retired_by_item[retired.source_item_id] = retired_by_item.get(retired.source_item_id, 0) + 1
            for item in page.items:
                raw_status = [(raw.raw_id, raw.complete) for raw in item.raws]
                spool.execute(
                    "INSERT INTO items VALUES (?, ?, ?, ?, ?)",
                    (
                        observed_count,
                        item.source_item_id,
                        item.logical_coordinate,
                        json.dumps(raw_status, separators=(",", ":")),
                        retired_by_item.get(item.source_item_id, 0),
                    ),
                )
                observed_count += 1
                enumeration_complete &= item.enumeration_complete
                complete &= item.complete
                for raw in item.raws:
                    spool.execute(
                        "INSERT INTO raws VALUES (?, ?, ?) ON CONFLICT(raw_id) DO UPDATE SET "
                        "complete=MIN(complete, excluded.complete), "
                        "parser_complete=MIN(parser_complete, excluded.parser_complete)",
                        (raw.raw_id, int(raw.complete), int(raw.parser_complete)),
                    )
                    for logical in raw.logicals:
                        spool.execute(
                            "INSERT INTO logicals VALUES (?, ?, ?) ON CONFLICT(logical_key) DO UPDATE SET "
                            "complete=MIN(complete, excluded.complete)",
                            (logical.logical_source_key, logical.expected_session_id, int(logical.complete)),
                        )
            cursor = page.next_cursor
        enumeration_complete &= observed_count == item_count
        complete &= enumeration_complete
        confirmed = int(spool.execute("SELECT COUNT(*) FROM raws WHERE complete=1").fetchone()[0])
        unresolved = int(spool.execute("SELECT COUNT(*) FROM raws WHERE complete=0").fetchone()[0])
    return SourceReceiptSpool(
        path, generation_id, observed_count, enumeration_complete, complete, confirmed, unresolved
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
        self.runtime = context.runtime
        self.audit = self.runtime.audit_for_request(request, context)
        self.executor = OperationExecutor(audit=self.audit, archive_root=context.archive_root)
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
        fd, name = tempfile.mkstemp(prefix="polylogue-ingest-state-", suffix=".sqlite", dir=os.environ.get("TMPDIR"))
        os.close(fd)
        self.state_path = Path(name)
        with spool_connection(self.state_path) as state:
            state.executescript(
                "CREATE TABLE refusals(ordinal INTEGER PRIMARY KEY, logical_key TEXT NOT NULL, raw_id TEXT NOT NULL, "
                "reason TEXT NOT NULL);"
                "CREATE TABLE changed_sessions(session_id TEXT PRIMARY KEY, message_count INTEGER NOT NULL) WITHOUT ROWID;"
            )
        self.session_id_pages_ref: str | None = None
        self.session_id_page_count = 0
        self.session_ids_digest: str | None = None
        self.inline_session_ids: list[str] = []
        self.changed_session_count = 0
        self.changed_message_count = 0
        self.refused_count = 0
        self.inline_refusals: list[IngestRefusedMembershipHistorical] = []
        self.refusal_pages_ref: str | None = None
        self.refusal_page_count = 0
        self.refusal_pages_digest: str | None = None
        self.insight_pages_ref: str | None = None
        self.insight_page_count = 0
        self.insight_pages_digest: str | None = None
        self.profile_convergence_complete = False
        self.publisher = ArchiveBlobPublisher(context.archive_root / "source.db", context.archive_root / "blob")

    def record_refusal(self, refusal: CohortMembershipRefusalError) -> None:
        with spool_connection(self.state_path) as state:
            state.execute(
                "INSERT INTO refusals(logical_key, raw_id, reason) VALUES (?, ?, ?)",
                (refusal.logical_source_key, refusal.raw_id, refusal.reason[:512]),
            )

    def record_changed_session(self, session_id: str, message_count: int) -> None:
        with spool_connection(self.state_path) as state:
            state.execute(
                "INSERT INTO changed_sessions VALUES (?, ?) ON CONFLICT(session_id) DO UPDATE SET "
                "message_count=excluded.message_count",
                (session_id, message_count),
            )

    def stop_reason(self) -> str | None:
        reason = self.runtime.stop_reason(self.request)
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

    async def read(self, work: Callable[[PinnedOperationRead], _T]) -> _T:
        def observed() -> _T:
            self.check_stop()
            with open_operation_read(
                self.context.archive_root, publication_guard=self.runtime.publication_guard
            ) as snapshot:
                identity = snapshot.identity.authority_identity_digest
                if self.binding is None:
                    _validate_identity(self.request, self.context, snapshot)
                    self.observed_identity = identity
                    self.binding = MachineRequestBinding(
                        identity,
                        str(self.request.request_id),
                        self.context.principal.actor_ref,
                        self.request.fingerprint,
                        self.request.operation,
                    )
                elif identity != self.observed_identity:
                    raise ValueError("archive_identity_stale")
                self.snapshot = snapshot
                self.runtime.observe_snapshot(self.request, snapshot)
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
                        self.context.archive_root / "source.db",
                        purpose="accepted ingest source publication",
                        archive_root=self.context.archive_root,
                    )
                ) as connection,
                connection,
            ):
                connection.execute("BEGIN IMMEDIATE")
                return work(connection)

        return await self.runtime.write_phase("ingest.source", publish)

    async def abort_prepared(self, generation_id: str) -> None:
        """Settle a failed pre-accept attempt even after stop was requested."""
        await self.runtime.compute_phase(self.publisher.discard_pending)

        def discard() -> None:
            with (
                closing(
                    open_isolated_write_connection(
                        self.context.archive_root / "source.db",
                        purpose="abandoned ingest source preparation",
                        archive_root=self.context.archive_root,
                    )
                ) as connection,
                connection,
            ):
                connection.execute("BEGIN IMMEDIATE")
                abort_prepared_source_manifest(
                    connection, source_generation_id=generation_id, publisher_id=self.publisher.publisher_id
                )

        await self.runtime.write_phase("ingest.source", discard)

    async def archive_write(self, work: Callable[[ArchiveStore], _T]) -> _T:
        assert self.snapshot is not None
        expected = self.snapshot.identity

        def publish() -> _T:
            self.check_stop()
            location = ArchiveLocation.resolve(self.context.archive_root)
            current = ArchiveIdentity.resolve_location(location)
            if (current.authority_identity_digest, current.active_generation) != (
                expected.authority_identity_digest,
                expected.active_generation,
            ):
                raise ValueError("ingest publication generation changed; reprepare required")
            with ArchiveStore.open_existing(self.context.archive_root, read_only=False) as archive:
                if archive.index_db_path.resolve() != location.active_index_path.resolve():
                    raise ValueError("ingest writer opened another index generation")
                result = work(archive)
                archive.commit()
                return result

        return await self.runtime.write_phase("ingest.publish", publish)

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
                after_coordinate: str | None = None
                while True:

                    def compute_page(after: str | None = after_coordinate) -> tuple[FrozenSourceInput, ...]:
                        return retain_input_page(
                            spool,
                            after_coordinate=after,
                            source_path=source_path,
                            publisher=self.publisher,
                            check_stop=self.check_stop,
                        )

                    page = await self.runtime.compute_phase(compute_page)
                    if not page:
                        break

                    def stage_page(
                        conn: sqlite3.Connection,
                        start: int = ordinal,
                        batch: tuple[FrozenSourceInput, ...] = page,
                    ) -> None:
                        append_prepared_source_inputs(conn, generation_id, start, batch)

                    await self.source_write(stage_page)
                    ordinal += len(page)
                    after_coordinate = page[-1].coordinate
                manifest = await self.source_write(
                    lambda conn: seal_prepared_source_manifest(conn, generation_id, sealed_at_ms=int(time() * 1000))
                )
            except BaseException:
                await self.abort_prepared(generation_id)
                raise
            finally:
                await self.runtime.compute_phase(lambda: spool.unlink(missing_ok=True))

            def accept_prepared() -> dict[str, object]:
                self.check_stop()
                assert self.binding is not None
                now_ms = int(time() * 1000)
                deadline = self.runtime.request_deadline_unix_ms(self.request)
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
        if self.resumed:
            # Historical recovery starts from audit state only.  Do not even
            # open retained source membership before deciding whether a prior
            # attempt reached its terminal audit checkpoint.
            return None
        if self.record["artifact_kind"] != "source-generation":
            raise ValueError("accepted ingest does not reference a source generation")

        def load_started() -> StartedBoundMutation:
            assert self.binding is not None
            with self.audit.settled_machine_read():
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

    async def enumerate_item(
        self, generation: RetainedSourceGeneration, item: RetainedSourceInput
    ) -> tuple[tuple[int, ...], int] | None:
        if item.enumeration_complete:
            return None
        assert self.record is not None
        acquired_at_ms = _record_int(self.record["accepted_at_ms"], field="accepted timestamp")
        source_name = self.request.payload.get("source_name")
        if source_name is not None and not isinstance(source_name, str):
            raise ValueError("accepted ingest source name is not a string")
        assert self.started_mutation is not None
        if source_name != self.started_mutation.plan.context.get("source_name"):
            raise ValueError("accepted ingest source name changed after manifest binding")
        iterator = enumerate_ingest_input(
            item,
            source_generation_id=generation.source_generation_id,
            publisher=self.publisher,
            acquired_at_ms=acquired_at_ms,
            check_stop=self.check_stop,
            source_name=source_name,
        )
        coordinates: list[str] = []
        member_ordinals: set[int] = set()
        member_count: int | None = None
        published_terminal = False
        excised = _ExcisedRecords()
        try:
            current = await self.runtime.compute_phase(lambda: next(iterator, None))
            while current is not None:
                self.check_stop()
                following = await self.runtime.compute_phase(lambda: next(iterator, None))
                if isinstance(current, PreparedSourceMemberDisposition):
                    member_ordinals.add(current.entry_ordinal)
                    member_count = current.member_count
                else:
                    coordinates.append(current.member.record_coordinate)
                    if current.member.entry_ordinal is not None:
                        member_ordinals.add(current.member.entry_ordinal)
                    member_count = current.member_count
                await self._publish_record(
                    generation,
                    item,
                    current,
                    tuple(coordinates) if following is None else None,
                    acquired_at_ms,
                    tuple(sorted(member_ordinals)) if following is None else None,
                    member_count if following is None else None,
                    excised=excised,
                )
                published_terminal = following is None
                current = following
            if not coordinates and not published_terminal:
                # The caller completes empty inputs in one writer transaction
                # for the whole accepted input page. This keeps a large set of
                # no-record files from paying a source connection per file.
                return tuple(sorted(member_ordinals)), member_count if member_count is not None else 0
        finally:
            await self.runtime.compute_phase(iterator.close)
        return None

    async def _publish_record(
        self,
        generation: RetainedSourceGeneration,
        item: RetainedSourceInput,
        prepared: PreparedSourceRecord | PreparedSourceMemberDisposition | None,
        completed_coordinates: tuple[str, ...] | None,
        observed_at_ms: int,
        completed_member_ordinals: tuple[int, ...] | None = None,
        member_count: int | None = None,
        *,
        excised: _ExcisedRecords,
    ) -> None:
        def publish(connection: sqlite3.Connection) -> None:
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
                    execute_source_item_admission(connection, prepared.admission, prepared.member)
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
                    record_coordinates=tuple(c for c in completed_coordinates if c not in excised.coordinates),
                    enumerated_at_ms=observed_at_ms,
                    member_ordinals=completed_member_ordinals,
                    member_count=member_count,
                )

        await self.source_write(publish)

    async def receipt(self, generation_id: str) -> SourceReceiptSpool:
        fd, name = tempfile.mkstemp(prefix="polylogue-source-receipt-", suffix=".sqlite", dir=os.environ.get("TMPDIR"))
        os.close(fd)
        path = Path(name)

        def read_receipt(pinned: PinnedOperationRead) -> SourceReceiptSpool:
            index_connection = pinned.archive.index_connection
            if index_connection is None:
                raise RuntimeError("accepted ingest requires the pinned index tier")
            return _spool_source_receipt(
                pinned.archive.source_connection,
                index_connection,
                generation_id,
                path,
            )

        try:
            return await self.read(read_receipt)
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    async def materialize(self, generation_id: str) -> SourceReceiptSpool:
        """Reuse canonical census and cohort publication, reconciling first."""
        initial = await self.receipt(generation_id)
        try:
            with spool_connection(initial.path, read_only=True) as conn:
                retired = conn.execute("SELECT 1 FROM items WHERE retired_count>0 LIMIT 1").fetchone()
            if retired is not None:
                raise ValueError("accepted raw member was retired; it cannot be readmitted")
            cursor: str | None = None
            while raw_page := initial.pending_raw_page(cursor):
                for raw_id in raw_page:
                    for _attempt in range(3):
                        observed_at_ms = int(time() * 1000)

                        def prepare_census(
                            pinned: PinnedOperationRead,
                            *,
                            census_raw_id: str = raw_id,
                            census_observed_at_ms: int = observed_at_ms,
                        ) -> PreparedRawCensus:
                            return prepare_raw_census(
                                pinned.archive,
                                census_raw_id,
                                parser_fingerprint=RAW_AUTHORITY_PARSER_FINGERPRINT,
                                parse_retained_raw=parse_retained_raw_sessions,
                                censused_at_ms=census_observed_at_ms,
                            )

                        prepared: PreparedRawCensus = await self.read(prepare_census)

                        def publish_census(
                            archive: ArchiveStore, *, census: PreparedRawCensus = prepared
                        ) -> CensusPublication:
                            return publish_raw_census(archive, census)

                        result: CensusPublication = await self.archive_write(publish_census)
                        if result.published:
                            break
                    else:
                        raise ValueError("accepted raw census kept changing during preparation")
                cursor = raw_page[-1]
        finally:
            initial.close()

        observed = await self.receipt(generation_id)
        try:
            with spool_connection(observed.path) as pending:
                pending.execute("CREATE TABLE pending(logical_key TEXT PRIMARY KEY) WITHOUT ROWID")
                pending.execute(
                    "CREATE TABLE attempts(logical_key TEXT PRIMARY KEY, count INTEGER NOT NULL) WITHOUT ROWID"
                )
                pending.execute("INSERT INTO pending SELECT logical_key FROM logicals WHERE complete=0")
                while row := pending.execute("SELECT logical_key FROM pending ORDER BY logical_key LIMIT 1").fetchone():
                    self.check_stop()
                    key = str(row[0])
                    pending.execute("DELETE FROM pending WHERE logical_key=?", (key,))
                    pending.execute(
                        "INSERT INTO attempts VALUES (?, 1) ON CONFLICT(logical_key) DO UPDATE SET count=count+1",
                        (key,),
                    )
                    attempts = int(
                        pending.execute("SELECT count FROM attempts WHERE logical_key=?", (key,)).fetchone()[0]
                    )
                    if attempts > 3:
                        raise ValueError("accepted membership cohort kept changing during preparation")
                    observed_at_ms = int(time() * 1000)

                    def prepare_cohort(
                        pinned: PinnedOperationRead,
                        *,
                        cohort_key: str = key,
                        cohort_observed_at_ms: int = observed_at_ms,
                    ) -> PreparedIngestCohort:
                        return prepare_ingest_cohort(
                            pinned.archive,
                            logical_source_key=cohort_key,
                            source_generation_id=generation_id,
                            parser_fingerprint=RAW_AUTHORITY_PARSER_FINGERPRINT,
                            parse_retained_raw=_parse_assembled_retained_raw,
                            acquired_at_ms=cohort_observed_at_ms,
                        )

                    prepared_cohort: PreparedIngestCohort | None = None
                    try:
                        prepared_cohort = await self.read(prepare_cohort)
                    except CohortMembershipRefusalError as refusal:
                        # Refuse this key while the rest of the generation continues.
                        emit(
                            "ingest.membership.refused",
                            logical_source_key=refusal.logical_source_key,
                            raw_id=refusal.raw_id,
                            reason=refusal.reason,
                            outcome="refused",
                        )
                        self.record_refusal(refusal)
                        continue
                    assert prepared_cohort is not None
                    try:
                        self.check_stop()

                        def publish_cohort(
                            archive: ArchiveStore, *, cohort: PreparedIngestCohort = prepared_cohort
                        ) -> tuple[CohortPublication, int | None]:
                            before = {
                                session_id: archive._conn.execute(
                                    "SELECT content_hash FROM sessions WHERE session_id = ?", (session_id,)
                                ).fetchone()
                                for session_id in cohort.affected_session_ids
                            }
                            published = publish_ingest_cohort(archive, cohort)
                            if not published.published or published.session_id is None:
                                return published, None
                            after = archive._conn.execute(
                                "SELECT content_hash, message_count FROM sessions WHERE session_id = ?",
                                (published.session_id,),
                            ).fetchone()
                            if after is None:
                                raise RuntimeError("published ingest cohort has no session row")
                            prior = before.get(published.session_id)
                            changed = prior is None or prior[0] != after[0]
                            return published, int(after[1]) if changed else None

                        publication, changed_message_count = await self.archive_write(publish_cohort)
                    finally:

                        def discard_cohort(*, cohort: PreparedIngestCohort = prepared_cohort) -> None:
                            discard_prepared_ingest_cohort(cohort)

                        await self.runtime.compute_phase(discard_cohort)
                    if publication.reprepare_required:
                        for redo_key in (key, *publication.reprepare_logical_source_keys):
                            pending.execute("INSERT OR IGNORE INTO pending VALUES (?)", (redo_key,))
                    if changed_message_count is not None and publication.session_id is not None:
                        self.record_changed_session(publication.session_id, changed_message_count)
        finally:
            observed.close()
        # A published classification may still be ambiguous or incomplete.
        # Only exact source/application/head witnesses can certify it.
        return await self.receipt(generation_id)

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
                self.request,
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
                        "SELECT coordinate, raws_json, retired_count FROM items WHERE source_item_id=?",
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
                    raw_status = json.loads(str(row[1]))
                    raw_ids = sorted({str(raw_id) for raw_id, _complete in raw_status})
                    unresolved = sorted({str(raw_id) for raw_id, complete in raw_status if not complete})
                    retired_count = int(row[2])
                    page_metadata = observed.execute(
                        "SELECT page_ref, page_count, raw_count, unresolved_count, digest "
                        "FROM raw_metadata WHERE source_item_id=?",
                        (item.source_item_id,),
                    ).fetchone()
                    inputs.append(
                        IngestInputHistoricalReceipt(
                            source_item_id=item.source_item_id,
                            logical_coordinate=item.coordinate,
                            denominator=len(raw_ids) + retired_count,
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
                source_complete=receipt.complete and self.refused_count == 0,
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
        started = self.started_mutation
        operation_id = started.operation_id
        assert operation_id is not None
        with spool_connection(receipt.path) as observed:
            for item_id, raw_json in observed.execute("SELECT source_item_id, raws_json FROM items ORDER BY ordinal"):
                raw_status = sorted(json.loads(str(raw_json)), key=lambda raw: raw[0])
                if len(raw_status) <= MAX_INLINE_RAW_IDS_PER_INPUT:
                    continue
                raw_pages = [
                    IngestInputRawPageHistoricalReceipt(
                        source_item_id=str(item_id),
                        ordinal=ordinal,
                        raws=[
                            IngestInputRawMemberHistorical(raw_id=str(raw_id), unresolved=not complete)
                            for raw_id, complete in raw_status[offset : offset + MAX_PAGE_ITEMS]
                        ],
                    )
                    for ordinal, offset in enumerate(range(0, len(raw_status), MAX_PAGE_ITEMS))
                ]
                observed.execute(
                    "INSERT INTO raw_metadata VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        item_id,
                        operation_id,
                        len(raw_pages),
                        len(raw_status),
                        sum(not complete for _raw_id, complete in raw_status),
                        ingest_input_raw_pages_digest(raw_pages),
                    ),
                )
                for raw_page in raw_pages:

                    def persist_raw_page(page: IngestInputRawPageHistoricalReceipt = raw_page) -> None:
                        self.audit.append_ingest_input_raw_page(operation_id, page)

                    await self.runtime.write_phase("ingest.input_raw_page", persist_raw_page)
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
        await self.runtime.write_phase("ingest.finalize", lambda: self.executor.finalize_bound(started, receipt=final))
        self.terminalized = True
        return history

    async def mark_unknown(self, reason: str) -> None:
        if self.started_mutation is None or self.terminalized:
            return
        started = self.started_mutation
        await self.runtime.write_phase(
            "ingest.unknown",
            lambda: self.executor.finalize_bound(started, unknown_reason=reason[:512]),
        )
        self.terminalized = True

    async def fence(self, reason: str) -> None:
        if self.binding is not None:
            binding = self.binding

            def stop() -> None:
                record = self.audit.machine_request(binding)
                if record is not None:
                    self.audit.stop_machine_batch(binding, reason)

            await self.runtime.write_phase("ingest.stop", stop)


async def execute_ingest_operation(
    request: DaemonOperationRequest, context: OperationContext
) -> DaemonOperationEnvelope:
    """Accept immutable input before any raw admission, then settle each phase."""
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
            # A prior process passed durable acceptance without a terminal
            # checkpoint.  It is indeterminate, never a reason to replay from
            # current source or index evidence.
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
        cursor: tuple[str, str] | None = None
        seen = 0
        while page := await execution.input_page(generation, cursor):
            empty: list[tuple[RetainedSourceInput, tuple[int, ...], int]] = []
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
                    batch: tuple[tuple[RetainedSourceInput, tuple[int, ...], int], ...] = tuple(empty),
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
            profile_parts = await execution.converge_profiles(receipt)
            execution.check_stop()
            history = await execution.finalize(generation, receipt, profile_parts)
        finally:
            receipt.close()
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
        await execution.mark_unknown(exc.reason)
        await execution.fence(exc.reason)
        return operation_envelope(
            request,
            context,
            snapshot=execution.snapshot,
            started_at=started,
            outcome="cancelled" if exc.reason == "cancelled" else "timed-out",
            reference=execution.record,
        )
    except Exception:
        await execution.mark_unknown("accepted ingest lacks a terminal checkpoint")
        await execution.fence("refused")
        raise
    finally:
        await execution.runtime.compute_phase(execution.publisher.discard_pending)
        execution.state_path.unlink(missing_ok=True)
