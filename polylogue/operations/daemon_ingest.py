"""Staged retained-input ingestion on the existing daemon compute and writer."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict, replace
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
    enumerate_ingest_input,
    prepare_ingest_inputs,
)
from polylogue.operations.insight_acceptance import SessionInsightPartReceipt
from polylogue.operations.machine_lifecycle import machine_request_state
from polylogue.operations.machine_receipts import (
    MAX_INLINE_INGEST_SESSION_IDS,
    MAX_INLINE_RAW_IDS_PER_INPUT,
    MAX_MACHINE_RECEIPT_PAGES,
    MAX_PAGE_ITEMS,
    IngestHistoricalReceipt,
    IngestInputHistoricalReceipt,
    IngestInputPageHistoricalReceipt,
    IngestInputRawMemberHistorical,
    IngestInputRawPageHistoricalReceipt,
    IngestInsightPageHistoricalReceipt,
    IngestRefusedMembershipHistorical,
    IngestTerminalSummaryHistorical,
    InsightTargetHistoricalReceipt,
    ingest_input_raw_pages_digest,
    ingest_insight_pages_digest,
    ingest_session_ids_digest,
)
from polylogue.operations.mutation_transaction import (
    MutationPreview,
    MutationReceipt,
    OperationExecutor,
    StartedBoundMutation,
)
from polylogue.operations.operation_context import OperationContext, PinnedOperationRead, open_operation_read
from polylogue.sources.origin_specs import retained_enumeration_fingerprint
from polylogue.sources.revision_backfill import parse_retained_raw_sessions
from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation
from polylogue.storage.blob_publication import ArchiveBlobPublisher
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
from polylogue.storage.source_generation_receipts import SourceGenerationReceipt, source_generation_receipt
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.raw_admission import execute_source_item_admission
from polylogue.storage.sqlite.archive_tiers.source_items import (
    RetainedSourceGeneration,
    RetainedSourceInput,
    complete_source_item_enumeration,
    retained_source_generation,
)
from polylogue.storage.sqlite.connection_profile import open_isolated_write_connection

_T = TypeVar("_T")


class IngestStoppedError(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


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
        # polylogue-163ku: per-key cohort refusals collected while the rest of
        # the generation continues; surfaced counted in the terminal receipt.
        self.refused_memberships: list[CohortMembershipRefusalError] = []
        self.changed_session_messages: dict[str, int] = {}
        self.session_id_pages_ref: str | None = None
        self.session_id_page_count = 0
        self.session_ids_digest: str | None = None
        self.insight_pages_ref: str | None = None
        self.insight_page_count = 0
        self.insight_pages_digest: str | None = None
        self.input_raw_page_metadata: dict[str, tuple[str, int, int, int, str]] = {}
        self.publisher = ArchiveBlobPublisher(context.archive_root / "source.db", context.archive_root / "blob")

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
            manifest = await self.runtime.compute_phase(
                lambda: prepare_ingest_inputs(
                    Path(str(self.request.payload["path"])),
                    source_path=source_path,
                    source_name=source_name,
                    source_generation_id=str(uuid4()),
                    publisher=self.publisher,
                    check_stop=self.check_stop,
                )
            )

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
                self.publisher.flush()
                with self.audit.bind_machine_request(
                    self.binding,
                    transition="accept_ingest",
                    deadline_unix_ms=self.runtime.request_deadline_unix_ms(self.request),
                ):
                    self.audit.accept_ingest(manifest, self.context.principal, plan=plan, authorization=authorization)
                record = self.recover_machine_request()
                assert record is not None
                return record

            self.record = await self.runtime.write_phase("ingest.accept", accept_prepared)
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
            lambda pinned: retained_source_generation(pinned.archive.source_connection, generation_id)
        )
        fingerprint = await self.runtime.compute_phase(retained_enumeration_fingerprint)
        if generation.enumeration_fingerprint != fingerprint:
            raise ValueError("accepted source enumeration decoder is no longer available")
        return generation

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

    async def enumerate_item(self, generation: RetainedSourceGeneration, item: RetainedSourceInput) -> None:
        if item.enumeration_complete:
            return
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
                )
                current = following
            if not coordinates:
                await self._publish_record(
                    generation,
                    item,
                    None,
                    (),
                    acquired_at_ms,
                    tuple(sorted(member_ordinals)),
                    member_count if member_count is not None else 0,
                )
        finally:
            await self.runtime.compute_phase(iterator.close)

    async def _publish_record(
        self,
        generation: RetainedSourceGeneration,
        item: RetainedSourceInput,
        prepared: PreparedSourceRecord | PreparedSourceMemberDisposition | None,
        completed_coordinates: tuple[str, ...] | None,
        observed_at_ms: int,
        completed_member_ordinals: tuple[int, ...] | None = None,
        member_count: int | None = None,
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
                execute_source_item_admission(connection, prepared.admission, prepared.member)
            if completed_coordinates is not None:
                complete_source_item_enumeration(
                    connection,
                    source_generation_id=generation.source_generation_id,
                    source_item_id=item.source_item_id,
                    enumeration_fingerprint=generation.enumeration_fingerprint,
                    record_coordinates=completed_coordinates,
                    enumerated_at_ms=observed_at_ms,
                    member_ordinals=completed_member_ordinals,
                    member_count=member_count,
                )

        await self.source_write(publish)

    async def receipt(self, generation_id: str) -> SourceGenerationReceipt:
        def read_receipt(pinned: PinnedOperationRead) -> SourceGenerationReceipt:
            index_connection = pinned.archive.index_connection
            if index_connection is None:
                raise RuntimeError("accepted ingest requires the pinned index tier")
            return source_generation_receipt(
                pinned.archive.source_connection,
                index_connection,
                source_generation_id=generation_id,
                active_generation=pinned.identity.active_generation,
            )

        return await self.read(read_receipt)

    async def materialize(self, generation_id: str) -> SourceGenerationReceipt:
        """Reuse canonical census and cohort publication, reconciling first."""
        initial = await self.receipt(generation_id)
        if initial.retired_coordinates:
            raise ValueError("accepted raw member was retired; it cannot be readmitted")
        raw_ids = tuple(sorted(set(initial.confirmed_raw_ids + initial.unresolved_raw_ids)))
        # Built once for the whole request: every per-key cohort preparation
        # and every publication revalidation reuses this exact ownership set.
        owned_raw_ids = frozenset(raw_ids)
        uncensused = {raw.raw_id for item in initial.items for raw in item.raws if not raw.parser_complete}
        for raw_id in sorted(uncensused):
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

                def publish_census(archive: ArchiveStore, *, census: PreparedRawCensus = prepared) -> CensusPublication:
                    return publish_raw_census(archive, census)

                result: CensusPublication = await self.archive_write(publish_census)
                if result.published:
                    break
            else:
                raise ValueError("accepted raw census kept changing during preparation")

        observed = await self.receipt(generation_id)
        pending_keys = {
            logical.logical_source_key
            for item in observed.items
            for raw in item.raws
            for logical in raw.logicals
            if not logical.complete
        }
        attempts: dict[str, int] = {}
        while pending_keys:
            self.check_stop()
            key = min(pending_keys)
            pending_keys.remove(key)
            attempts[key] = attempts.get(key, 0) + 1
            if attempts[key] > 3:
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
                    accepted_raw_ids=owned_raw_ids,
                    parser_fingerprint=RAW_AUTHORITY_PARSER_FINGERPRINT,
                    parse_retained_raw=parse_retained_raw_sessions,
                    acquired_at_ms=cohort_observed_at_ms,
                )

            try:
                prepared_cohort: PreparedIngestCohort = await self.read(prepare_cohort)
            except CohortMembershipRefusalError as refusal:
                # polylogue-163ku: refuse this key, not the generation. The
                # remaining keys still publish; the receipt reports this one as
                # incomplete and the terminal summary counts and names it.
                emit(
                    "ingest.membership.refused",
                    logical_source_key=refusal.logical_source_key,
                    raw_id=refusal.raw_id,
                    reason=refusal.reason,
                    outcome="refused",
                )
                self.refused_memberships.append(refusal)
                continue
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
                pending_keys.add(key)
                pending_keys.update(publication.reprepare_logical_source_keys)
            if changed_message_count is not None and publication.session_id is not None:
                self.changed_session_messages[publication.session_id] = changed_message_count
        # A published classification may still be ambiguous or incomplete.
        # Only exact source/application/head witnesses can certify it.
        return await self.receipt(generation_id)

    async def converge_profiles(self, receipt: SourceGenerationReceipt) -> tuple[SessionInsightPartReceipt, ...]:
        """Derive only the exact sessions proved by this source denominator."""
        if not receipt.complete:
            return ()
        session_ids = tuple(
            sorted(
                {logical.expected_session_id for item in receipt.items for raw in item.raws for logical in raw.logicals}
            )
        )
        parts: list[SessionInsightPartReceipt] = []
        assert self.started_mutation is not None
        expected_recipe = str(self.started_mutation.plan.context["recipe_version"])
        for offset in range(0, len(session_ids), 256):
            self.check_stop()
            part = await self.runtime.converge_ingest_sessions(
                self.request,
                session_ids[offset : offset + 256],
                expected_recipe=expected_recipe,
                stop_requested=self.stop_reason,
            )
            parts.append(part)
            if part.remaining_unattempted_target_refs or any(
                target.disposition not in {"already_satisfied", "published"} for target in part.targets
            ):
                break
        return tuple(parts)

    def historical_receipt(
        self,
        generation: RetainedSourceGeneration,
        receipt: SourceGenerationReceipt,
        profile_parts: tuple[SessionInsightPartReceipt, ...],
        insight_pages: list[IngestInsightPageHistoricalReceipt] | None = None,
    ) -> IngestHistoricalReceipt:
        """Freeze the final observation while its source/index snapshots exist.

        Larger ID sets use immutable audit page references. Later reads never
        call ``source_generation_receipt`` to reconstruct changed identities.
        """

        observed = {item.source_item_id: item for item in receipt.items}
        retired_by_item: dict[str, int] = {}
        for retired in receipt.retired_coordinates:
            retired_by_item[retired.source_item_id] = retired_by_item.get(retired.source_item_id, 0) + 1
        inputs: list[IngestInputHistoricalReceipt] = []
        for item in generation.inputs:
            item_receipt = observed.get(item.source_item_id)
            if item_receipt is None:
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
            raw_ids = sorted({raw.raw_id for raw in item_receipt.raws})
            unresolved = sorted({raw.raw_id for raw in item_receipt.raws if not raw.complete})
            retired_count = retired_by_item.get(item.source_item_id, 0)
            page_metadata = self.input_raw_page_metadata.get(item.source_item_id)
            inputs.append(
                IngestInputHistoricalReceipt(
                    source_item_id=item.source_item_id,
                    logical_coordinate=item_receipt.logical_coordinate,
                    denominator=len(raw_ids) + retired_count,
                    raw_ids=None if page_metadata is not None else raw_ids,
                    unresolved_raw_ids=[] if page_metadata is not None else unresolved,
                    raw_id_pages_ref=None if page_metadata is None else page_metadata[0],
                    raw_id_page_count=0 if page_metadata is None else page_metadata[1],
                    raw_id_count=0 if page_metadata is None else page_metadata[2],
                    unresolved_raw_count=0 if page_metadata is None else page_metadata[3],
                    raw_ids_digest=None if page_metadata is None else page_metadata[4],
                    unknown_attribution=("retired source record has no raw identity" if retired_count else None),
                )
            )
        pages = [
            IngestInputPageHistoricalReceipt.from_items(ordinal, inputs[offset : offset + 256])
            for ordinal, offset in enumerate(range(0, len(inputs), 256))
        ]
        if insight_pages is None:
            insight_pages = self._insight_pages(profile_parts)
        return IngestHistoricalReceipt(
            source_generation_id=generation.source_generation_id,
            final_sequence=1,
            input_count=len(inputs),
            input_pages=pages,
            insight_pages=[] if self.insight_pages_ref is not None else insight_pages,
            insight_pages_ref=self.insight_pages_ref,
            insight_page_count=self.insight_page_count,
            insight_pages_digest=self.insight_pages_digest,
            summary=IngestTerminalSummaryHistorical(
                enumeration_complete=receipt.enumeration_complete,
                source_complete=receipt.complete,
                confirmed_raw_count=len(receipt.confirmed_raw_ids),
                unresolved_raw_count=len(receipt.unresolved_raw_ids),
                profile_targets_observed=sum(len(part.targets) for part in profile_parts),
                refused_membership_count=len(self.refused_memberships),
                refused_memberships=[
                    IngestRefusedMembershipHistorical(
                        logical_source_key=refusal.logical_source_key,
                        raw_id=refusal.raw_id,
                        reason=refusal.reason[:512],
                    )
                    for refusal in self.refused_memberships[:256]
                ],
                parse_projection_known=True,
                processed_session_ids=(
                    sorted(self.changed_session_messages) if self.session_id_pages_ref is None else []
                ),
                processed_session_id_pages_ref=self.session_id_pages_ref,
                processed_session_id_page_count=self.session_id_page_count,
                processed_session_ids_digest=self.session_ids_digest,
                processed_message_count=sum(self.changed_session_messages.values()),
                changed_session_count=len(self.changed_session_messages),
                changed_message_count=sum(self.changed_session_messages.values()),
            ),
        )

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
        receipt: SourceGenerationReceipt,
        profile_parts: tuple[SessionInsightPartReceipt, ...],
    ) -> IngestHistoricalReceipt:
        assert self.started_mutation is not None
        started = self.started_mutation
        operation_id = started.operation_id
        assert operation_id is not None
        observed_items = {item.source_item_id: item for item in receipt.items}
        for item in generation.inputs:
            observed_item = observed_items.get(item.source_item_id)
            if observed_item is None:
                continue
            raws = sorted(observed_item.raws, key=lambda raw: raw.raw_id)
            if len(raws) <= MAX_INLINE_RAW_IDS_PER_INPUT:
                continue
            raw_pages = [
                IngestInputRawPageHistoricalReceipt(
                    source_item_id=item.source_item_id,
                    ordinal=ordinal,
                    raws=[
                        IngestInputRawMemberHistorical(raw_id=raw.raw_id, unresolved=not raw.complete)
                        for raw in raws[offset : offset + MAX_PAGE_ITEMS]
                    ],
                )
                for ordinal, offset in enumerate(range(0, len(raws), MAX_PAGE_ITEMS))
            ]
            self.input_raw_page_metadata[item.source_item_id] = (
                operation_id,
                len(raw_pages),
                len(raws),
                sum(not raw.complete for raw in raws),
                ingest_input_raw_pages_digest(raw_pages),
            )
            for raw_page in raw_pages:

                def persist_raw_page(page: IngestInputRawPageHistoricalReceipt = raw_page) -> None:
                    self.audit.append_ingest_input_raw_page(operation_id, page)

                await self.runtime.write_phase("ingest.input_raw_page", persist_raw_page)
        session_ids = sorted(self.changed_session_messages)
        if len(session_ids) > MAX_INLINE_INGEST_SESSION_IDS:
            self.session_id_pages_ref = operation_id
            self.session_id_page_count = (len(session_ids) + MAX_PAGE_ITEMS - 1) // MAX_PAGE_ITEMS
            self.session_ids_digest = ingest_session_ids_digest(session_ids)
            for ordinal, offset in enumerate(range(0, len(session_ids), MAX_PAGE_ITEMS)):
                id_page = tuple(session_ids[offset : offset + MAX_PAGE_ITEMS])

                def persist_id_page(page: tuple[str, ...] = id_page, page_ordinal: int = ordinal) -> None:
                    self.audit.append_ingest_session_id_page(operation_id, page_ordinal, page)

                await self.runtime.write_phase(
                    "ingest.session_ids",
                    persist_id_page,
                )
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
        history = self.historical_receipt(generation, receipt, profile_parts, insight_pages)
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
            if state["outcome"] == "completed" and isinstance(restored, dict):
                history = IngestHistoricalReceipt.model_validate(restored)
                return operation_envelope(
                    request,
                    context,
                    snapshot=execution.snapshot,
                    started_at=started,
                    outcome="completed",
                    reference=execution.record,
                    result={
                        "source_generation_id": history.source_generation_id,
                        "outcome": "completed",
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
        for item in generation.inputs:
            execution.check_stop()
            await execution.enumerate_item(generation, item)
        receipt = await execution.materialize(generation.source_generation_id)
        profile_parts = await execution.converge_profiles(receipt)
        execution.check_stop()
        history = await execution.finalize(generation, receipt, profile_parts)
        return operation_envelope(
            request,
            context,
            snapshot=execution.snapshot,
            started_at=started,
            outcome="completed",
            reference=execution.record,
            result={
                "source_generation_id": generation.source_generation_id,
                "outcome": "completed",
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
