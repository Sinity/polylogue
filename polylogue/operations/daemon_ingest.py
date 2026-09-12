"""Staged retained-input ingestion on the existing daemon compute and writer."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from time import monotonic, time
from typing import TypeVar
from uuid import uuid4

from polylogue.operations.audit import MachineRequestBinding
from polylogue.operations.bindings import runtime_operation_binding
from polylogue.operations.daemon_execution import _validate_identity, operation_envelope, validate_execution_request
from polylogue.operations.daemon_protocol import DaemonOperationEnvelope, DaemonOperationRequest
from polylogue.operations.ingest_acceptance import IngestActuator, ingest_plan
from polylogue.operations.ingest_inputs import PreparedSourceRecord, enumerate_ingest_input, prepare_ingest_inputs
from polylogue.operations.insight_acceptance import SessionInsightPartReceipt
from polylogue.operations.machine_lifecycle import machine_request_state
from polylogue.operations.machine_receipts import (
    IngestHistoricalReceipt,
    IngestInputHistoricalReceipt,
    IngestInputPageHistoricalReceipt,
    IngestInsightPageHistoricalReceipt,
    IngestTerminalSummaryHistorical,
    InsightTargetHistoricalReceipt,
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
        self.record: dict[str, object] | None = None
        self.started_mutation: StartedBoundMutation | None = None
        self.resumed = False
        self.terminalized = False
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

    async def read(self, work: Callable[[PinnedOperationRead], _T]) -> _T:
        def observed() -> _T:
            self.check_stop()
            with open_operation_read(
                self.context.archive_root, publication_guard=self.runtime.publication_guard
            ) as snapshot:
                if self.binding is None:
                    _validate_identity(self.request, self.context, snapshot)
                    self.binding = MachineRequestBinding(
                        snapshot.identity.authority_identity_digest,
                        str(self.request.request_id),
                        self.context.principal.actor_ref,
                        self.request.fingerprint,
                        self.request.operation,
                    )
                elif snapshot.identity.authority_identity_digest != self.binding.archive_identity:
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
            assert self.binding is not None
            with self.audit.settled_machine_read():
                return self.audit.machine_request(self.binding)

        self.record = await self.read(recover)
        if self.record is None:
            source_path = self.request.payload.get("source_path")
            if source_path is not None and not isinstance(source_path, str):
                raise ValueError("ingest source path is not a string")
            manifest = await self.runtime.compute_phase(
                lambda: prepare_ingest_inputs(
                    Path(str(self.request.payload["path"])),
                    source_path=source_path,
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
                record = self.audit.machine_request(self.binding)
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
                record = self.audit.machine_request(self.binding)
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
        iterator = enumerate_ingest_input(
            item,
            source_generation_id=generation.source_generation_id,
            publisher=self.publisher,
            acquired_at_ms=acquired_at_ms,
            check_stop=self.check_stop,
        )
        coordinates: list[str] = []
        try:
            current = await self.runtime.compute_phase(lambda: next(iterator, None))
            while current is not None:
                self.check_stop()
                following = await self.runtime.compute_phase(lambda: next(iterator, None))
                coordinates.append(current.member.record_coordinate)
                await self._publish_record(
                    generation, item, current, tuple(coordinates) if following is None else None, acquired_at_ms
                )
                current = following
            if not coordinates:
                await self._publish_record(generation, item, None, (), acquired_at_ms)
        finally:
            await self.runtime.compute_phase(iterator.close)

    async def _publish_record(
        self,
        generation: RetainedSourceGeneration,
        item: RetainedSourceInput,
        prepared: PreparedSourceRecord | None,
        completed_coordinates: tuple[str, ...] | None,
        observed_at_ms: int,
    ) -> None:
        def publish(connection: sqlite3.Connection) -> None:
            if prepared is not None:
                execute_source_item_admission(connection, prepared.admission, prepared.member)
            if completed_coordinates is not None:
                complete_source_item_enumeration(
                    connection,
                    source_generation_id=generation.source_generation_id,
                    source_item_id=item.source_item_id,
                    enumeration_fingerprint=generation.enumeration_fingerprint,
                    record_coordinates=completed_coordinates,
                    enumerated_at_ms=observed_at_ms,
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
                    accepted_raw_ids=raw_ids,
                    parser_fingerprint=RAW_AUTHORITY_PARSER_FINGERPRINT,
                    parse_retained_raw=parse_retained_raw_sessions,
                    acquired_at_ms=cohort_observed_at_ms,
                )

            prepared_cohort: PreparedIngestCohort = await self.read(prepare_cohort)
            try:
                self.check_stop()

                def publish_cohort(
                    archive: ArchiveStore, *, cohort: PreparedIngestCohort = prepared_cohort
                ) -> CohortPublication:
                    return publish_ingest_cohort(archive, cohort)

                publication: CohortPublication = await self.archive_write(publish_cohort)
            finally:

                def discard_cohort(*, cohort: PreparedIngestCohort = prepared_cohort) -> None:
                    discard_prepared_ingest_cohort(cohort)

                await self.runtime.compute_phase(discard_cohort)
            if publication.reprepare_required:
                pending_keys.add(key)
                pending_keys.update(publication.reprepare_logical_source_keys)
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
    ) -> IngestHistoricalReceipt:
        """Freeze the final observation while its source/index snapshots exist.

        The returned receipt is intentionally self-contained.  Later reads use
        the audit event and never call ``source_generation_receipt`` again.
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
            inputs.append(
                IngestInputHistoricalReceipt(
                    source_item_id=item.source_item_id,
                    logical_coordinate=item_receipt.logical_coordinate,
                    denominator=len(raw_ids) + retired_count,
                    raw_ids=raw_ids,
                    unresolved_raw_ids=unresolved,
                    unknown_attribution=("retired source record has no raw identity" if retired_count else None),
                )
            )
        pages = [
            IngestInputPageHistoricalReceipt.from_items(ordinal, inputs[offset : offset + 256])
            for ordinal, offset in enumerate(range(0, len(inputs), 256))
        ]
        insight_pages = [
            IngestInsightPageHistoricalReceipt(
                ordinal=ordinal,
                targets=[InsightTargetHistoricalReceipt.model_validate(asdict(target)) for target in part.targets],
                unattempted_target_refs=list(part.remaining_unattempted_target_refs),
            )
            for ordinal, part in enumerate(profile_parts)
        ]
        return IngestHistoricalReceipt(
            source_generation_id=generation.source_generation_id,
            final_sequence=1,
            input_count=len(inputs),
            input_pages=pages,
            insight_pages=insight_pages,
            summary=IngestTerminalSummaryHistorical(
                enumeration_complete=receipt.enumeration_complete,
                source_complete=receipt.complete,
                confirmed_raw_count=len(receipt.confirmed_raw_ids),
                unresolved_raw_count=len(receipt.unresolved_raw_ids),
                profile_targets_observed=sum(len(part.targets) for part in profile_parts),
            ),
        )

    async def finalize(
        self,
        generation: RetainedSourceGeneration,
        receipt: SourceGenerationReceipt,
        profile_parts: tuple[SessionInsightPartReceipt, ...],
    ) -> IngestHistoricalReceipt:
        assert self.started_mutation is not None
        started = self.started_mutation
        history = self.historical_receipt(generation, receipt, profile_parts)
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
