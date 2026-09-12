"""Canonical machine mutation adapters over the shared audited executor."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from time import time
from typing import Any, cast

from polylogue.operations.audit import AuditRepository, MachineRequestBinding
from polylogue.operations.bindings import OperationBinding, runtime_operation_binding
from polylogue.operations.daemon_protocol import DaemonOperationRequest
from polylogue.operations.delete_authorization import _canonical_session_ids
from polylogue.operations.machine_lifecycle import machine_request_state
from polylogue.operations.mutation_actuators import (
    BulkMetadataSetActuator,
    BulkMetadataSetArgs,
    BulkTagActuator,
    BulkTagArgs,
    SessionDeleteActuator,
    SessionDeleteArgs,
)
from polylogue.operations.mutation_transaction import (
    MAX_MUTATION_PLAN_TARGETS,
    MutationPreview,
    OperationExecutor,
    compute_parameter_digest,
)
from polylogue.operations.operation_context import OperationContext, PinnedOperationRead
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _execute_named_mutation(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
    actuator: Any,
    args: Any,
) -> dict[str, object]:
    """Run one legacy domain actuator under the daemon's write authority."""
    assert context.runtime is not None
    binding = runtime_operation_binding(actuator)
    executor = OperationExecutor(audit=audit, archive_root=context.archive_root)
    preview = executor.prepare_bound_for_archive(binding, args, context.principal, archive_root=context.archive_root)
    authorization = executor.authorize_bound(binding, preview, context.principal, confirmation_strength="bound_token")
    receipt = executor.execute_bound(binding, preview, authorization, args)
    if receipt.status in {"blocked", "unknown"}:
        raise ValueError(receipt.detail or f"{actuator.operation} did not apply")
    return {
        "operation": request.operation,
        "outcome": "completed",
        "sequence": 1,
        "effect": "committed" if receipt.affected_count else "no-effect",
        "affected_count": receipt.affected_count,
        "result": dict(receipt.domain_receipt),
    }


def mutation_session_excision(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    from polylogue.operations.mutation_actuators import SessionExcisionActuator, SessionExcisionArgs

    payload = request.payload
    args = SessionExcisionArgs(
        archive_root=context.archive_root,
        session_id=str(payload["session_id"]),
        reason=str(payload["reason"]),
        actor=str(payload["actor"]),
        cascade_lineage=bool(payload.get("cascade_lineage", False)),
    )
    return _execute_named_mutation(request, context, audit, snapshot, SessionExcisionActuator(), args)


def mutation_session_lifecycle_request(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    from polylogue.operations.mutation_actuators import SessionLifecycleRequestActuator, SessionLifecycleRequestArgs
    from polylogue.security.lifecycle import LifecycleMode

    payload = request.payload
    args = SessionLifecycleRequestArgs(
        archive_root=context.archive_root,
        session_id=str(payload["session_id"]),
        mode=cast(LifecycleMode, str(payload["mode"])),
        reason=str(payload["reason"]),
        actor=str(payload["actor"]),
        now_ms=int(time() * 1000),
    )
    return _execute_named_mutation(request, context, audit, snapshot, SessionLifecycleRequestActuator(), args)


def mutation_identity_reset(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    from polylogue.operations.mutation_actuators import IdentityResetActuator, IdentityResetArgs

    payload = request.payload
    args = IdentityResetArgs(
        archive_root=context.archive_root,
        session_ids=tuple(cast(list[str], payload["session_ids"])),
        reason=str(payload["reason"]),
    )
    return _execute_named_mutation(request, context, audit, snapshot, IdentityResetActuator(), args)


def mutation_raw_authority_blocker_resolve(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    from polylogue.operations.mutation_actuators import BlockerResolveActuator, BlockerResolveArgs

    payload = request.payload
    args = BlockerResolveArgs(
        archive_root=context.archive_root,
        blocker_id=str(payload["blocker_id"]),
        resolution=str(payload["resolution"]),
        assertion_id=cast(str | None, payload.get("assertion_id")),
        judgment_disposition=cast(str | None, payload.get("judgment_disposition")),
    )
    return _execute_named_mutation(request, context, audit, snapshot, BlockerResolveActuator(), args)


def _reset_targets(root: Path, payload: dict[str, object]) -> list[tuple[str, Path]]:
    """Resolve reset targets in the daemon before any deletion is attempted."""
    from polylogue.paths import blob_store_root, cache_home, data_home, drive_cache_path, drive_token_path, state_home

    flags = {key: bool(payload.get(key, False)) for key in ("index", "database", "blob", "assets", "cache", "auth")}
    if bool(payload.get("reset_all", False)):
        flags.update(dict.fromkeys(flags, True))
    if not any(flags.values()):
        raise ValueError("reset requires at least one target")
    targets: list[tuple[str, Path]] = []
    if flags["index"] or flags["database"]:
        if (root / ".index-active-pointer").exists():
            raise ValueError("reset is unsafe for a managed active generation")
        names = [("index database", "index.db")] if flags["index"] else []
        if flags["database"]:
            names = [
                ("source database", "source.db"),
                ("index database", "index.db"),
                ("embeddings database", "embeddings.db"),
                ("ops database", "ops.db"),
            ]
            if not bool(payload.get("include_source_db", False)):
                names = [item for item in names if item[1] != "source.db"]
            if not bool(payload.get("include_user_db", False)):
                pass
        for name, filename in names:
            path = root / filename
            if path.exists():
                targets.append((name, path))
            for suffix in ("-wal", "-shm"):
                sidecar = path.with_name(f"{path.name}{suffix}")
                if sidecar.exists():
                    targets.append((f"{name} {suffix}", sidecar))
    if flags["database"] and bool(payload.get("include_user_db", False)):
        path = root / "user.db"
        if path.exists():
            targets.append(("user database", path))
    if flags["blob"]:
        path = blob_store_root()
        if path.exists():
            targets.append(("blob store", path))
    if flags["assets"]:
        path = data_home() / "assets"
        if path.exists():
            targets.append(("assets", path))
    if flags["cache"]:
        for name, path in (
            ("cache/indexes", cache_home()),
            ("inferred schemas", data_home() / "schemas"),
            ("drive cache", drive_cache_path()),
        ):
            if path.exists():
                targets.append((name, path))
    if flags["auth"]:
        path = drive_token_path()
        if path.exists():
            targets.append(("OAuth token", path))
    if bool(payload.get("reset_all", False)):
        path = state_home() / "last-source.json"
        if path.exists():
            targets.append(("last-source state", path))
    return targets


def maintenance_reset(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    targets = _reset_targets(context.archive_root, request.payload)
    deleted = 0
    for _name, path in targets:
        if path.is_file():
            path.unlink()
            deleted += 1
        elif path.is_dir():
            shutil.rmtree(path)
            deleted += 1
    return {
        "operation": request.operation,
        "outcome": "completed",
        "sequence": 1,
        "effect": "committed" if deleted else "no-effect",
        "affected_count": deleted,
        "result": {"deleted": deleted, "targets": [name for name, _path in targets]},
    }


def maintenance_blob_gc_recover(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    from polylogue.operations.mutation_actuators import (
        PendingBlobGCGenerationAbandonActuator,
        PendingBlobGCGenerationAbandonArgs,
    )

    args = PendingBlobGCGenerationAbandonArgs(
        archive_root=context.archive_root,
        generation_id=str(request.payload["generation_id"]),
    )
    return _execute_named_mutation(
        request,
        context,
        audit,
        snapshot,
        PendingBlobGCGenerationAbandonActuator(),
        args,
    )


def _audit_int(value: object, *, field: str) -> int:
    """Reject malformed durable counters before scheduling a mutation batch."""

    if type(value) is not int:
        raise ValueError(f"machine mutation {field} is not an integer")
    return value


def _binding(
    request: DaemonOperationRequest, context: OperationContext, snapshot: PinnedOperationRead
) -> MachineRequestBinding:
    return MachineRequestBinding(
        snapshot.identity.authority_identity_digest,
        str(request.request_id),
        context.principal.actor_ref,
        request.fingerprint,
        request.operation,
    )


def _refs(payload: dict[str, object], singular: str) -> tuple[str, ...]:
    many = payload.get(f"{singular}s")
    return tuple(cast(list[str], many)) if many is not None else (str(payload[singular]),)


def _previews(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
    operation: OperationBinding[Any, object],
    args: tuple[object, ...],
) -> tuple[MutationPreview, ...]:
    executor = OperationExecutor()
    instance = audit.ensure_archive_authority(now_ms=int(time() * 1000))
    previews = []
    for item in args:
        raw_plan = operation.actuator.prepare(item)
        previews.append(
            executor.prepare_bound(
                operation,
                item,
                context.principal,
                archive_instance_id=instance,
                archive_identity_digest=snapshot.identity.authority_identity_digest,
                parameter_digest=compute_parameter_digest(raw_plan),
                raw_plan=raw_plan,
            )
        )
    return tuple(previews)


def mutation_session_delete_preview(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    binding = _binding(request, context, snapshot)
    prior = audit.machine_request(binding)
    if prior is not None:
        refs = tuple(str(part["artifact_ref"]) for part in audit.machine_parts(binding))
        previews = tuple(audit.preview_for_principal(ref, context.principal) for ref in refs)
        ids = tuple(target.ref.removeprefix("session:") for preview in previews for target in preview.plan.targets)
    else:
        ids = _canonical_session_ids(snapshot.archive, tuple(cast(list[str], request.payload["session_ids"])))
        operation = runtime_operation_binding(SessionDeleteActuator())
        args = tuple(
            SessionDeleteArgs(snapshot.archive, ids[offset : offset + MAX_MUTATION_PLAN_TARGETS])
            for offset in range(0, len(ids), MAX_MUTATION_PLAN_TARGETS)
        )
        previews = _previews(request, context, audit, snapshot, operation, args)
        with audit.bind_machine_request(binding, transition="create_preview_batch"):
            refs = tuple(audit.create_preview_batch(tuple(preview.plan for preview in previews), context.principal))
    return {
        "status": "prepared",
        "operation": "delete",
        "preview_ref": refs[0],
        "preview_refs": list(refs),
        "session_ids": list(ids),
        "session_count": len(ids),
        "expires_at_ms": min(preview.plan.expires_at_ms for preview in previews),
    }


def mutation_session_delete_authorize(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    binding = _binding(request, context, snapshot)
    prior = audit.machine_request(binding)
    if prior is not None:
        refs = [str(part["artifact_ref"]) for part in audit.machine_parts(binding)]
    else:
        previews = tuple(
            audit.preview_for_principal(ref, context.principal) for ref in _refs(request.payload, "preview_ref")
        )
        operation = runtime_operation_binding(SessionDeleteActuator())
        executor = OperationExecutor()
        authorizations = tuple(
            executor.authorize_bound(operation, preview, context.principal, confirmation_strength="bound_token")
            for preview in previews
        )
        with audit.bind_machine_request(binding, transition="issue_authorization_batch"):
            refs = audit.issue_authorization_batch(previews, context.principal, authorizations)
    return {"status": "authorized", "authorization_ref": refs[0], "authorization_refs": refs}


def mutation_session_delete_cancel(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    binding = _binding(request, context, snapshot)
    if audit.machine_request(binding) is None:
        previews = tuple(
            audit.preview_for_principal(ref, context.principal) for ref in _refs(request.payload, "preview_ref")
        )
        with audit.bind_machine_request(binding, transition="cancel_preview_batch"):
            audit.cancel_preview_batch(previews, context.principal)
    refs = _refs(request.payload, "preview_ref")
    return {"status": "cancelled", "preview_ref": refs[0], "preview_refs": list(refs)}


def _part_args(
    archive: ArchiveStore,
    preview: MutationPreview,
    *,
    requested_session_ids: tuple[str, ...] | None = None,
) -> tuple[OperationBinding[Any, object], object]:
    # Inline bulk requests retain their original selection for fresh planning,
    # including unresolved IDs and duplicates counted by the existing actuator.
    # The immutable request fingerprint verifies this input on every resumption.
    ids = (
        requested_session_ids
        if requested_session_ids is not None
        else tuple(target.ref.removeprefix("session:") for target in preview.plan.targets)
    )
    if preview.plan.operation == "mutate-delete-session":
        return runtime_operation_binding(SessionDeleteActuator()), SessionDeleteArgs(archive, ids)
    if preview.plan.operation == "mutate-bulk-tag-sessions":
        return runtime_operation_binding(BulkTagActuator()), BulkTagArgs(
            archive, ids, tuple(cast(list[str], preview.plan.context["tags"]))
        )
    if preview.plan.operation == "mutate-bulk-set-metadata":
        pairs = tuple((str(pair[0]), pair[1]) for pair in cast(list[list[object]], preview.plan.context["pairs"]))
        return runtime_operation_binding(BulkMetadataSetActuator()), BulkMetadataSetArgs(archive, ids, pairs)
    raise ValueError("unsupported durable machine mutation family")


def _execute_batch(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
    refs: tuple[str, ...],
) -> dict[str, object]:
    assert context.runtime is not None
    binding = _binding(request, context, snapshot)
    record = audit.machine_request(binding)
    if record is None:
        with audit.bind_machine_request(
            binding,
            transition="accept_execution_batch",
            deadline_unix_ms=context.runtime.request_deadline_unix_ms(request),
        ):
            audit.accept_execution_batch(refs, context.principal)
        record = audit.machine_request(binding)
    assert record is not None
    if record["artifact_kind"] != "execution-batch":
        raise ValueError("machine request is not an execution batch")
    with ArchiveStore.open_existing(context.archive_root, read_only=False) as archive:
        executor = OperationExecutor(audit=audit, archive_root=context.archive_root)
        for part in audit.machine_parts(binding):
            if record.get("stop_reason"):
                break
            if part["operation_id"] is not None:
                run = audit.get_operation(str(part["operation_id"]))
                if (
                    run is not None
                    and run["status"] == "completed"
                    and not _audit_int(run["unknown_count"], field="unknown count")
                ):
                    continue
                # Startup's shared recovery classifier owns interrupted domain
                # receipts. A consumed part is never replayed or reauthorized.
                break
            stop = context.runtime.stop_reason(request)
            deadline = record.get("accepted_deadline_unix_ms")
            if stop is None and deadline is not None and int(time() * 1000) >= _audit_int(deadline, field="deadline"):
                stop = "deadline"
            if stop is not None:
                audit.stop_machine_batch(binding, stop)
                break
            try:
                preview, authorization = audit.authorization_for_principal(
                    str(part["authorization_ref"]), context.principal
                )
                requested_ids = None
                if request.operation in {"mutation.session.tag", "mutation.session.metadata"}:
                    offset = _audit_int(part["ordinal"], field="part ordinal") * MAX_MUTATION_PLAN_TARGETS
                    requested_ids = tuple(
                        cast(list[str], request.payload["session_ids"])[offset : offset + MAX_MUTATION_PLAN_TARGETS]
                    )
                operation, args = _part_args(archive, preview, requested_session_ids=requested_ids)
                with audit.bind_machine_request(
                    binding,
                    transition="consume_authorization_and_start",
                    part=_audit_int(part["ordinal"], field="part ordinal"),
                ):
                    executor.execute_bound(operation, preview, authorization, args)
            except Exception:
                # The existing executor has already finalized known receipts or
                # recorded unknown effects. Stop only the untouched suffix.
                audit.stop_machine_batch(binding, "refused")
                break
    current = audit.machine_request(binding)
    assert current is not None
    return machine_request_state(audit, current)


def mutation_session_delete_execute(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    return _execute_batch(request, context, audit, snapshot, _refs(request.payload, "authorization_ref"))


def _inline_mutation(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
    *,
    metadata: bool,
) -> dict[str, object]:
    binding = _binding(request, context, snapshot)
    existing = audit.machine_request(binding)
    if existing is not None:
        refs = tuple(str(part["authorization_ref"]) for part in audit.machine_parts(binding))
    else:
        ids = tuple(cast(list[str], request.payload["session_ids"]))
        operation = runtime_operation_binding(BulkMetadataSetActuator() if metadata else BulkTagActuator())
        args: list[object] = []
        for offset in range(0, len(ids), MAX_MUTATION_PLAN_TARGETS):
            chunk = ids[offset : offset + MAX_MUTATION_PLAN_TARGETS]
            if metadata:
                pairs = tuple((str(pair[0]), pair[1]) for pair in cast(list[list[object]], request.payload["pairs"]))
                args.append(BulkMetadataSetArgs(snapshot.archive, chunk, pairs))
            else:
                args.append(BulkTagArgs(snapshot.archive, chunk, tuple(cast(list[str], request.payload["tags"]))))
        previews = _previews(request, context, audit, snapshot, operation, tuple(args))
        preview_refs = audit.create_preview_batch(tuple(preview.plan for preview in previews), context.principal)
        previews = tuple(replace(preview, preview_ref=ref) for preview, ref in zip(previews, preview_refs, strict=True))
        executor = OperationExecutor()
        authorizations = tuple(executor.authorize_bound(operation, preview, context.principal) for preview in previews)
        refs = tuple(audit.issue_authorization_batch(previews, context.principal, authorizations))
    return _execute_batch(request, context, audit, snapshot, refs)


def mutation_session_tag(
    request: DaemonOperationRequest, context: OperationContext, audit: AuditRepository, snapshot: PinnedOperationRead
) -> dict[str, object]:
    return _inline_mutation(request, context, audit, snapshot, metadata=False)


def mutation_session_metadata(
    request: DaemonOperationRequest, context: OperationContext, audit: AuditRepository, snapshot: PinnedOperationRead
) -> dict[str, object]:
    return _inline_mutation(request, context, audit, snapshot, metadata=True)
