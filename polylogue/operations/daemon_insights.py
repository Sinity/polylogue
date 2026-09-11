"""Staged exact-manifest insight maintenance using the resident shared owner."""

from __future__ import annotations

from dataclasses import asdict
from time import monotonic, time
from typing import cast

from polylogue.core.errors import ArchiveTierUnavailableError
from polylogue.operations.audit import MachineRequestBinding
from polylogue.operations.bindings import OperationBinding, runtime_operation_binding
from polylogue.operations.daemon_execution import _validate_identity, operation_envelope, validate_execution_request
from polylogue.operations.daemon_protocol import DaemonOperationEnvelope, DaemonOperationRequest
from polylogue.operations.insight_acceptance import AcceptedInsightPart, InsightAcceptance, SessionInsightPartReceipt
from polylogue.operations.insight_planning import (
    AcceptedInsightActuator,
    InsightManifest,
    insight_page_plan,
    insight_terminal_view_counts,
    prepare_insight_manifest,
)
from polylogue.operations.machine_lifecycle import machine_request_state
from polylogue.operations.machine_receipts import InsightPartHistoricalReceipt
from polylogue.operations.mutation_transaction import MutationReceipt, OperationExecutor, StartedBoundMutation
from polylogue.operations.operation_context import OperationContext, PinnedOperationRead, open_operation_read


class InsightStoppedError(RuntimeError):
    pass


class InsightExecution:
    def __init__(self, request: DaemonOperationRequest, context: OperationContext) -> None:
        if context.runtime is None:
            raise PermissionError("daemon_required")
        self.request = request
        self.context = context
        self.runtime = context.runtime
        self.runtime.require_session_maintenance()
        self.audit = self.runtime.audit_for_request(request, context)
        self.executor = OperationExecutor(audit=self.audit, archive_root=context.archive_root)
        self.operation: OperationBinding[object, object] = runtime_operation_binding(AcceptedInsightActuator())
        self.binding: MachineRequestBinding | None = None
        self.snapshot: PinnedOperationRead | None = None
        self.record: dict[str, object] | None = None
        self.parts: tuple[AcceptedInsightPart, ...] = ()
        self.expiry_ms = self.runtime.request_deadline_unix_ms(request)

    def stop_reason(self) -> str | None:
        reason = self.runtime.stop_reason(self.request)
        if self.record is not None and self.record.get("stop_reason"):
            reason = str(self.record["stop_reason"])
        if int(time() * 1000) >= self.expiry_ms:
            reason = reason or "deadline"
        return reason

    def check_stop(self) -> None:
        reason = self.stop_reason()
        if reason is not None:
            raise InsightStoppedError(reason)

    async def accept(self) -> None:
        def prepare() -> InsightManifest | None:
            self.check_stop()
            with open_operation_read(
                self.context.archive_root, publication_guard=self.runtime.publication_guard
            ) as pinned:
                _validate_identity(self.request, self.context, pinned)
                self.snapshot = pinned
                self.runtime.observe_snapshot(self.request, pinned)
                self.binding = MachineRequestBinding(
                    pinned.identity.authority_identity_digest,
                    str(self.request.request_id),
                    self.context.principal.actor_ref,
                    self.request.fingerprint,
                    self.request.operation,
                )
                with self.audit.settled_machine_read():
                    self.record = self.audit.machine_request(self.binding)
                    if self.record is not None and self.record["artifact_kind"] == "execution-batch":
                        self.parts = self.audit.sealed_insight_parts(self.binding, self.context.principal)
                        deadline = self.record.get("accepted_deadline_unix_ms")
                        if deadline is not None:
                            if type(deadline) is not int:
                                raise ValueError("accepted insight deadline is not an integer")
                            self.expiry_ms = deadline
                        return None
                generation, recipe = self.runtime.session_profile_plan_binding(
                    opened_index_path=pinned.archive.index_db_path
                )
                return prepare_insight_manifest(
                    pinned.archive,
                    cast(list[str] | None, self.request.payload.get("session_ids")),
                    index_generation=generation,
                    recipe_version=recipe,
                    check_stop=self.check_stop,
                )

        manifest = await self.runtime.compute_phase(prepare)
        if manifest is None:
            return
        assert self.binding is not None
        binding = self.binding
        acceptance = InsightAcceptance(self.audit, binding, self.context.principal)
        previous: str | None = None
        for ordinal in range(len(manifest.pages)):

            def stage(ordinal: int = ordinal, previous: str | None = previous) -> str:
                self.check_stop()
                now_ms = int(time() * 1000)
                instance = self.audit.ensure_archive_authority(now_ms=now_ms)
                staged = self.audit.machine_parts(binding)
                if staged:
                    first = self.audit.preview_for_principal(str(staged[0]["preview_ref"]), self.context.principal)
                    now_ms = first.plan.prepared_at_ms
                    self.expiry_ms = first.plan.expires_at_ms
                plan = insight_page_plan(
                    manifest,
                    ordinal,
                    previous_preview_ref=previous,
                    archive_instance_id=instance,
                    archive_identity_digest=binding.archive_identity,
                    now_ms=now_ms,
                    expires_at_ms=self.expiry_ms,
                )
                preview = acceptance.stage_preview(plan)
                acceptance.ensure_staged_authorization(self.executor, self.operation, preview)
                return preview.preview_ref

            previous = await self.runtime.write_phase("insights.stage", stage)
        assert previous is not None

        def seal() -> tuple[AcceptedInsightPart, ...]:
            self.check_stop()
            parts = acceptance.seal(
                head_preview_ref=previous,
                page_count=len(manifest.pages),
                manifest_digest=manifest.digest,
                deadline_unix_ms=self.expiry_ms,
            )
            self.record = self.audit.machine_request(binding)
            return parts

        self.parts = await self.runtime.write_phase("insights.accept", seal)

    async def begin(self, part: AcceptedInsightPart) -> StartedBoundMutation | None:
        def start() -> StartedBoundMutation | None:
            self.check_stop()
            assert self.binding is not None
            durable_part = self.audit.machine_parts(self.binding)[part.ordinal]
            if durable_part["operation_id"] is not None:
                # A consumed page is reconciled from its receipt, never rerun.
                return None
            with self.audit.bind_machine_request(
                self.binding, transition="consume_authorization_and_start", part=part.ordinal
            ):
                return self.executor.begin_accepted_insight_part(
                    self.operation,
                    part,
                    principal=self.context.principal,
                    machine_part=part.ordinal,
                    args=None,
                )

        return await self.runtime.write_phase("insights.begin", start)

    async def state(self) -> dict[str, object]:
        def read() -> dict[str, object]:
            assert self.binding is not None
            with self.audit.settled_machine_read():
                record = self.audit.machine_request(self.binding)
                if record is None:
                    raise ValueError("insight request has no durable binding")
                self.record = record
                return machine_request_state(self.audit, record)

        return await self.runtime.compute_phase(read)

    async def stop(self, reason: str) -> None:
        if self.binding is not None and self.record is not None:
            binding = self.binding
            await self.runtime.write_phase("insights.stop", lambda: self.audit.stop_machine_batch(binding, reason))

    async def finalize(
        self,
        part: AcceptedInsightPart,
        started: StartedBoundMutation,
        observed: SessionInsightPartReceipt,
        *,
        terminal_summary: dict[str, int] | None,
    ) -> None:
        complete = not observed.remaining_unattempted_target_refs and all(
            target.disposition in {"already_satisfied", "published"} for target in observed.targets
        )
        history = (
            InsightPartHistoricalReceipt.model_validate(
                {
                    "kind": "insight-part/v1",
                    "ordinal": part.ordinal,
                    "page_count": part.page_count,
                    "manifest_digest": part.manifest_digest,
                    "index_generation": part.index_generation,
                    "recipe_version": part.recipe_version,
                    "targets": tuple(asdict(target) for target in observed.targets),
                    "unattempted_target_refs": observed.remaining_unattempted_target_refs,
                    "terminal_summary": terminal_summary,
                }
            )
            if complete
            else None
        )
        writes = sum(target.publication_known_committed for target in observed.targets)
        receipt = MutationReceipt(
            operation=started.plan.operation,
            plan_hash=started.plan.plan_hash,
            status=("applied" if writes else "already_satisfied") if complete else "unknown",
            target_refs=started.plan.target_refs,
            affected_count=writes,
            detail=None,
            receipt_ref=None,
            applied_at=started.plan.prepared_at,
            historical_receipt=history,
        )
        await self.runtime.write_phase(
            "insights.finalize", lambda: self.executor.finalize_bound(started, receipt=receipt)
        )


async def execute_insights_rebuild_operation(
    request: DaemonOperationRequest, context: OperationContext
) -> DaemonOperationEnvelope:
    started_at = monotonic()
    request = validate_execution_request(request, context)
    execution = InsightExecution(request, context)
    active: StartedBoundMutation | None = None
    active_part: AcceptedInsightPart | None = None
    active_observed: SessionInsightPartReceipt | None = None
    try:
        await execution.accept()
        totals = {"profiles": 0, "work_events": 0, "phases": 0}
        for part in execution.parts:
            active_part = part
            active_observed = None
            active = await execution.begin(part)
            if active is None:

                def prior(part: AcceptedInsightPart = part) -> InsightPartHistoricalReceipt | None:
                    assert execution.binding is not None
                    with execution.audit.settled_machine_read():
                        raw = execution.audit.machine_parts(execution.binding)[part.ordinal]
                        run = execution.audit.get_operation(str(raw["operation_id"]))
                        if run is None or run["status"] != "completed":
                            return None
                        history = execution.audit.historical_machine_receipt(str(raw["operation_id"]))
                        if not isinstance(history, InsightPartHistoricalReceipt):
                            return None
                        if (
                            history.ordinal != part.ordinal
                            or history.page_count != part.page_count
                            or history.manifest_digest != part.manifest_digest
                            or history.index_generation != part.index_generation
                            or history.recipe_version != part.recipe_version
                            or history.unattempted_target_refs
                            or tuple(target.target_ref for target in history.targets)
                            != tuple(target.target_ref for target in part.targets)
                        ):
                            raise ValueError("historical insight receipt does not match its accepted page")
                        return history

                history = await execution.runtime.compute_phase(prior)
                if history is None:
                    break
                for historical_target in history.targets:
                    for family in totals:
                        totals[family] += getattr(historical_target.certified_counts, family)
                continue
            observed = await execution.runtime.converge_insight_part(
                request, part, stop_requested=execution.stop_reason
            )
            active_observed = observed
            complete = not observed.remaining_unattempted_target_refs and all(
                target.disposition in {"already_satisfied", "published"} for target in observed.targets
            )
            for observed_target in observed.targets:
                for family in totals:
                    totals[family] += getattr(observed_target.certified_counts, family)
            summary = None
            if complete and part.ordinal == part.page_count - 1:

                def views(part: AcceptedInsightPart = part) -> tuple[int, int]:
                    with open_operation_read(
                        context.archive_root, publication_guard=execution.runtime.publication_guard
                    ) as pinned:
                        generation, recipe = execution.runtime.session_profile_plan_binding(
                            opened_index_path=pinned.archive.index_db_path
                        )
                        if (generation, recipe) != (part.index_generation, part.recipe_version):
                            raise ValueError("accepted insight generation or recipe changed before final view read")
                        execution.snapshot = pinned
                        execution.runtime.observe_snapshot(request, pinned)
                        return insight_terminal_view_counts(
                            pinned.archive,
                            tuple(target for accepted in execution.parts for target in accepted.targets),
                            check_stop=execution.check_stop,
                        )

                threads, tags = await execution.runtime.compute_phase(views)
                summary = {**totals, "threads": threads, "tag_rollups": tags}
            await execution.finalize(part, active, observed, terminal_summary=summary)
            active = None
            active_observed = None
            if not complete:
                await execution.stop(execution.runtime.stop_reason(request) or "part_incomplete")
                break
    except InsightStoppedError as exc:
        if active is not None:
            if active_observed is not None and active_part is not None:
                await execution.finalize(active_part, active, active_observed, terminal_summary=None)
            else:
                await execution.runtime.write_phase(
                    "insights.unknown",
                    lambda: execution.executor.finalize_bound(active, unknown_reason="cancelled during accepted part"),
                )
        await execution.stop(str(exc))
    except ArchiveTierUnavailableError as exc:
        # This can only be a pre-acceptance refusal: insight planning needs
        # derived state, while source-tier acquisition intentionally keeps it
        # closed. Do not create an audit acceptance that later looks like a
        # partially executed insight operation.
        if active is None and execution.record is None:
            return operation_envelope(
                request,
                context,
                snapshot=execution.snapshot,
                started_at=started_at,
                outcome="rejected",
                error={"code": exc.code, "detail": str(exc), "retryable": True},
            )
        if active is not None:
            if active_observed is not None and active_part is not None:
                await execution.finalize(active_part, active, active_observed, terminal_summary=None)
            else:
                await execution.runtime.write_phase(
                    "insights.unknown",
                    lambda: execution.executor.finalize_bound(
                        active, unknown_reason="accepted part lacks a settled receipt"
                    ),
                )
        await execution.stop("refused")
        raise
    except Exception:
        if active is not None:
            if active_observed is not None and active_part is not None:
                await execution.finalize(active_part, active, active_observed, terminal_summary=None)
            else:
                await execution.runtime.write_phase(
                    "insights.unknown",
                    lambda: execution.executor.finalize_bound(
                        active, unknown_reason="accepted part lacks a settled receipt"
                    ),
                )
        await execution.stop("refused")
        raise
    state = await execution.state()
    return operation_envelope(
        request,
        context,
        snapshot=execution.snapshot,
        started_at=started_at,
        outcome=str(state["outcome"]),
        reference=execution.record,
        result=state.get("result", state),
    )
