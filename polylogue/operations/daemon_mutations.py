"""Canonical machine mutation adapters over the shared audited executor."""

from __future__ import annotations

from collections.abc import Sequence
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
        # ``execute_bound`` stamps ``mutation-operation:<operation_id>`` onto
        # the finalized receipt. Dropping it here left every surface of every
        # actuator routed through this helper without a handle onto the
        # operation_runs/operation_attempts rows the mutation just wrote.
        "receipt_ref": receipt.receipt_ref,
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
            # ``embeddings.db`` is absent deliberately: bootstrap classifies it
            # ``expensive_rebuild`` because nothing replays its vectors from
            # source.db -- they are re-purchased from the embedding provider.
            # Deleting it is a repurchase, not a reset, so ``--database`` keeps
            # it and the CLI names the embedding-preservation route instead.
            names = [
                ("source database", "source.db"),
                ("index database", "index.db"),
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
    """Delete the resolved reset targets under the audited executor.

    polylogue-4fbgw: this used to unlink files and ``rmtree`` trees inline
    with ``audit``/``snapshot`` unused, so the product's largest destructive
    surface wrote no operation_previews / operation_authorizations /
    operation_runs / operation_attempts rows at all. Targets are resolved once
    here and handed to :class:`FilesystemResetActuator`, so PREPARE and APPLY
    see the identical set and the audit rows precede the first deletion.
    """
    from polylogue.operations.mutation_actuators import FilesystemResetActuator, FilesystemResetArgs

    targets = _reset_targets(context.archive_root, request.payload)
    args = FilesystemResetArgs(archive_root=context.archive_root, targets=tuple(targets))
    return _execute_named_mutation(request, context, audit, snapshot, FilesystemResetActuator(), args)


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


def maintenance_blob_publications_abandon(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    """Discharge named publication-reservation debt under daemon write authority."""
    from polylogue.operations.mutation_actuators import (
        BlobPublicationAbandonActuator,
        BlobPublicationAbandonArgs,
    )

    args = BlobPublicationAbandonArgs(
        archive_root=context.archive_root,
        publication_ids=tuple(str(value) for value in cast(Sequence[object], request.payload["publication_ids"])),
    )
    return _execute_named_mutation(
        request,
        context,
        audit,
        snapshot,
        BlobPublicationAbandonActuator(),
        args,
    )


def maintenance_blob_refs_replace_from_source(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    """Repoint raw-backed missing blob refs under daemon write authority."""
    from polylogue.operations.maintenance_actuators import (
        BlobReferenceSourceReplaceActuator,
        BlobReferenceSourceReplaceArgs,
    )

    payload = request.payload
    max_count = payload.get("max_count")
    args = BlobReferenceSourceReplaceArgs(
        archive_root=context.archive_root,
        manifest_path=Path(str(payload["manifest_path"])),
        max_count=None if max_count is None else int(cast(int, max_count)),
        sample_size=int(cast(int, payload.get("sample_size", 30))),
    )
    return _execute_named_mutation(
        request,
        context,
        audit,
        snapshot,
        BlobReferenceSourceReplaceActuator(),
        args,
    )


def maintenance_blob_refs_prune_orphans(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    """Quarantine and prune orphan blob refs under daemon write authority."""
    from polylogue.operations.maintenance_actuators import (
        BlobReferenceOrphanPruneActuator,
        BlobReferenceOrphanPruneArgs,
    )

    payload = request.payload
    max_count = payload.get("max_count")
    quarantine_path = payload.get("quarantine_path")
    args = BlobReferenceOrphanPruneArgs(
        archive_root=context.archive_root,
        quarantine_path=None if quarantine_path is None else Path(str(quarantine_path)),
        max_count=None if max_count is None else int(cast(int, max_count)),
        sample_size=int(cast(int, payload.get("sample_size", 30))),
    )
    return _execute_named_mutation(
        request,
        context,
        audit,
        snapshot,
        BlobReferenceOrphanPruneActuator(),
        args,
    )


def maintenance_demo_augment(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    """Apply the deterministic demo-only post-ingest writes under daemon authority.

    Ingest alone produces the parsed session/message tree; the demo world's
    provider usage, insight materialization, canonical repo name and synthetic
    embeddings are layered on afterwards so a daemon-ingested demo archive
    matches ``polylogue demo seed``'s semantic contract. Idempotent: a repeated
    request re-applies the same deterministic content.
    """
    del audit, snapshot

    from polylogue.demo import apply_demo_post_ingest_augmentation

    with_overlays = bool(request.payload.get("with_overlays", False))
    apply_demo_post_ingest_augmentation(context.archive_root)
    if with_overlays:
        from polylogue.scenarios import seed_demo_user_overlays

        seed_demo_user_overlays(context.archive_root)
    return {
        "operation": request.operation,
        "outcome": "completed",
        "sequence": 1,
        "effect": "committed",
        "result": {"augmented": True, "overlays": with_overlays},
    }


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
    remove_tags = tuple(cast(list[str], request.payload.get("remove_tags") or []))
    if remove_tags:
        from polylogue.operations.mutation_actuators import TagRemoveActuator, TagRemoveArgs

        tokens = tuple(cast(list[str], request.payload["session_ids"]))

        def build(archive: ArchiveStore) -> Any:
            for token in tokens:
                session_id = _resolve_session_target(archive, context.archive_root, token)
                for tag in remove_tags:
                    yield (TagRemoveActuator(), TagRemoveArgs(archive=archive, session_id=session_id, tag=tag))

        return _execute_user_state_mutations(request, context, audit, build)
    return _inline_mutation(request, context, audit, snapshot, metadata=False)


def mutation_session_metadata(
    request: DaemonOperationRequest, context: OperationContext, audit: AuditRepository, snapshot: PinnedOperationRead
) -> dict[str, object]:
    return _inline_mutation(request, context, audit, snapshot, metadata=True)


def _resolve_session_target(archive: ArchiveStore, archive_root: Path, token: str) -> str:
    """Resolve one session token under the daemon's write authority.

    The index is the fast path and the durable user-tier owners are the
    fallback, matching the Python facade's ``_resolve_user_state_session_id``
    exactly -- a mark written through the daemon and the same mark written
    through the facade must name the same canonical session.
    """
    from polylogue.operations.user_state_resolution import resolve_durable_user_state_session_id

    try:
        resolved = archive.resolve_session_id(token)
    except (KeyError, ValueError):
        resolved = None
    if resolved:
        return str(resolved)
    durable = resolve_durable_user_state_session_id(archive_root, token)
    if durable:
        return durable
    raise ValueError(f"session {token!r} not found")


def _execute_user_state_mutations(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    build: Any,
) -> dict[str, object]:
    """Run a batch of user-tier actuator cycles against one writable handle.

    ``build(archive)`` yields ``(actuator, args)`` pairs. The mark and
    annotation actuators take an open :class:`ArchiveStore` in their args (the
    primitive they drive is a ``user.db`` method), so unlike the archive-root
    actuators in :func:`_execute_named_mutation` they need the writable handle
    opened here -- the daemon's writer, not a second one in an adapter process.
    """
    assert context.runtime is not None
    executor = OperationExecutor(audit=audit, archive_root=context.archive_root)
    affected = 0
    receipts: list[dict[str, object]] = []
    with ArchiveStore.open_existing(context.archive_root, read_only=False) as archive:
        for actuator, args in build(archive):
            binding = runtime_operation_binding(actuator)
            preview = executor.prepare_bound_for_archive(
                binding, args, context.principal, archive_root=context.archive_root
            )
            authorization = executor.authorize_bound(
                binding, preview, context.principal, confirmation_strength="bound_token"
            )
            receipt = executor.execute_bound(binding, preview, authorization, args)
            if receipt.status in {"blocked", "unknown"}:
                raise ValueError(receipt.detail or f"{actuator.operation} did not apply")
            affected += receipt.affected_count
            receipts.append(dict(receipt.domain_receipt))
    return {
        "operation": request.operation,
        "outcome": "completed",
        "sequence": 1,
        "effect": "committed" if affected else "no-effect",
        "affected_count": affected,
        "result": {"receipts": receipts},
    }


def mutation_session_mark(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    """Add or remove whole-session star/pin/archive marks."""
    from polylogue.core.user_state_targets import TARGET_SESSION, validate_mark_type
    from polylogue.operations.mutation_actuators import MarkAddActuator, MarkArgs, MarkRemoveActuator

    payload = request.payload
    tokens = tuple(cast(list[str], payload["session_ids"]))
    adds = tuple(validate_mark_type(str(value)) for value in cast(list[str], payload.get("add_marks") or []))
    removes = tuple(validate_mark_type(str(value)) for value in cast(list[str], payload.get("remove_marks") or []))

    def build(archive: ArchiveStore) -> Any:
        for token in tokens:
            session_id = _resolve_session_target(archive, context.archive_root, token)
            for mark_type in adds:
                yield (
                    MarkAddActuator(),
                    MarkArgs(
                        archive=archive,
                        target_type=TARGET_SESSION,
                        target_id=session_id,
                        mark_type=mark_type,
                        owner_session_id=session_id,
                    ),
                )
            for mark_type in removes:
                yield (
                    MarkRemoveActuator(),
                    MarkArgs(
                        archive=archive,
                        target_type=TARGET_SESSION,
                        target_id=session_id,
                        mark_type=mark_type,
                        owner_session_id=session_id,
                    ),
                )

    return _execute_user_state_mutations(request, context, audit, build)


def mutation_annotation_save(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    """Create or update one session-scoped annotation body."""
    from polylogue.core.user_state_targets import TARGET_SESSION
    from polylogue.operations.mutation_actuators import AnnotationSaveActuator, AnnotationSaveArgs

    payload = request.payload
    annotation_id = str(payload["annotation_id"])
    note_text = str(payload["note_text"])
    token = str(payload["session_id"])

    def build(archive: ArchiveStore) -> Any:
        session_id = _resolve_session_target(archive, context.archive_root, token)
        yield (
            AnnotationSaveActuator(),
            AnnotationSaveArgs(
                archive=archive,
                annotation_id=annotation_id,
                target_type=TARGET_SESSION,
                target_id=session_id,
                note_text=note_text,
                owner_session_id=session_id,
            ),
        )

    return _execute_user_state_mutations(request, context, audit, build)


def mutation_assertion_candidate_capture(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    """Capture one terminal assertion candidate under the daemon's writer.

    ``polylogue note`` reached ``user.db`` through
    ``Polylogue.capture_assertion_candidate``, which opened a writable
    ``ArchiveStore`` in the CLI process (polylogue-gjwto / polylogue-r29bv
    criterion 2). The actuator cycle is unchanged; the assertion id is minted
    here because the operation, not the adapter, owns the durable identity.
    """
    import hashlib
    import uuid

    from polylogue.core.enums import AssertionKind
    from polylogue.core.refs import normalize_object_ref_text
    from polylogue.operations.mutation_actuators import (
        CaptureAssertionCandidateActuator,
        CaptureAssertionCandidateArgs,
    )
    from polylogue.storage.sqlite.archive_tiers.user_write import ArchiveAssertionEnvelope
    from polylogue.surfaces.payloads import AssertionClaimPayload

    assert context.runtime is not None
    payload = request.payload
    author_ref = str(payload.get("author_ref") or "user:local")
    raw_idempotency_key = payload.get("idempotency_key")
    idempotency_key = None if raw_idempotency_key is None else str(raw_idempotency_key)
    if idempotency_key is None:
        assertion_id = f"assertion-terminal-note:{uuid.uuid4()}"
    else:
        identity = hashlib.sha256(
            f"{normalize_object_ref_text(author_ref)}\0{idempotency_key.strip()}".encode(errors="surrogatepass")
        ).hexdigest()
        assertion_id = f"assertion-terminal-note:{identity}"

    raw_cwd = payload.get("cwd")
    ttl_seconds = payload.get("ttl_seconds")
    executor = OperationExecutor(audit=audit, archive_root=context.archive_root)
    actuator = CaptureAssertionCandidateActuator()
    binding = runtime_operation_binding(actuator)
    with ArchiveStore.open_existing(context.archive_root, read_only=False) as archive:
        args = CaptureAssertionCandidateArgs(
            archive=archive,
            body_text=str(payload["body_text"]),
            kind=AssertionKind.from_string(str(payload["kind"])),
            refs=tuple(str(ref) for ref in cast(list[str], payload.get("refs") or [])),
            scope_refs=tuple(str(ref) for ref in cast(list[str], payload.get("scope_refs") or [])),
            cwd=None if raw_cwd is None else Path(str(raw_cwd)),
            author_ref=author_ref,
            author_kind=str(payload.get("author_kind") or "user"),
            idempotency_key=idempotency_key,
            assertion_id=assertion_id,
            ttl_seconds=None if ttl_seconds is None else int(cast(int, ttl_seconds)),
        )
        preview = executor.prepare_bound_for_archive(
            binding, args, context.principal, archive_root=context.archive_root
        )
        authorization = executor.authorize_bound(
            binding, preview, context.principal, confirmation_strength="bound_token"
        )
        receipt = executor.execute_bound(binding, preview, authorization, args)
    if receipt.status in {"blocked", "unknown"}:
        raise ValueError(receipt.detail or f"{actuator.operation} did not apply")
    envelope = receipt.domain_receipt["envelope"]
    assert isinstance(envelope, ArchiveAssertionEnvelope)
    return {
        "operation": request.operation,
        "outcome": "completed",
        "sequence": 1,
        "effect": "committed" if receipt.affected_count else "no-effect",
        "affected_count": receipt.affected_count,
        "result": AssertionClaimPayload.from_envelope(envelope).model_dump(mode="json"),
    }


def mutation_user_setting_set(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    """Insert-or-update one durable ``user_settings`` row under the daemon's writer.

    ``polylogue setting set`` used to reach ``user.db`` through
    ``Polylogue.set_setting``, which opened a writable ``ArchiveStore`` in the
    CLI process (polylogue-gjwto / polylogue-r29bv). The actuator cycle is
    unchanged; what moves is which process holds the write authority, and the
    ``user_settings`` envelope is lowered to JSON here because the wire result
    cannot carry a dataclass.
    """
    from polylogue.operations.mutation_actuators import SetUserSettingActuator, SetUserSettingArgs
    from polylogue.storage.sqlite.archive_tiers.user_settings_write import ArchiveUserSettingEnvelope

    assert context.runtime is not None
    payload = request.payload
    executor = OperationExecutor(audit=audit, archive_root=context.archive_root)
    actuator = SetUserSettingActuator()
    binding = runtime_operation_binding(actuator)
    with ArchiveStore.open_existing(context.archive_root, read_only=False) as archive:
        args = SetUserSettingArgs(
            archive=archive,
            setting_key=str(payload["setting_key"]),
            value=payload.get("value"),
            author_ref=str(payload.get("author_ref") or "user:local"),
        )
        preview = executor.prepare_bound_for_archive(
            binding, args, context.principal, archive_root=context.archive_root
        )
        authorization = executor.authorize_bound(
            binding, preview, context.principal, confirmation_strength="bound_token"
        )
        receipt = executor.execute_bound(binding, preview, authorization, args)
    if receipt.status in {"blocked", "unknown"}:
        raise ValueError(receipt.detail or f"{actuator.operation} did not apply")
    envelope = receipt.domain_receipt["envelope"]
    assert isinstance(envelope, ArchiveUserSettingEnvelope)
    return {
        "operation": request.operation,
        "outcome": "completed",
        "sequence": 1,
        "effect": "committed" if receipt.affected_count else "no-effect",
        "affected_count": receipt.affected_count,
        "result": {
            "setting_key": envelope.setting_key,
            "value": envelope.value,
            "updated_at_ms": envelope.updated_at_ms,
            "author_ref": envelope.author_ref,
        },
    }


def mutation_annotation_import_batch(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    """Import one bounded JSONL annotation batch under the daemon's writer.

    ``polylogue annotations import`` used to build the product-layer request
    and drive ``OperationExecutor`` against ``user.db`` from the CLI process
    itself (``Polylogue.import_annotation_batch`` ->
    ``polylogue.annotations.importer.import_annotation_batch``), invisible to
    the mutation-authority layering rule because
    ``polylogue.annotations.importer`` names neither
    ``_execute_facade_mutation`` nor one of the four executor modules the CLI
    rule matched (polylogue-gjwto / polylogue-r29bv AC3). The actuator cycle
    inside ``import_annotation_batch`` is unchanged; what moves is which
    process holds the write authority and the ref resolver it uses --
    ``resolve_ref_against_archive`` runs the *same* plan
    ``Polylogue.resolve_ref`` runs, against the pinned reader this operation
    already holds, so a durable ``user.db`` admission decision cannot diverge
    between the facade and this handler (polylogue-j5u2b).

    ``import_annotation_batch`` is ``async`` for API parity with the other
    ``Polylogue`` methods it shares a signature shape with, not because it
    awaits anything genuinely concurrent, but running it means driving a
    coroutine from this synchronous handler. A bare ``asyncio.run(...)`` here
    would mint a *new* asyncio task, and the write lease is bound to the task
    (or thread, absent a task) that acquired it -- so the actuator's write
    would run as an unauthorized child task and ``open_verified_sqlite_write_
    connection`` would raise ``UnleasedWriteError`` (polylogue-5vps8 /
    polylogue-1oa7o). ``delegate_write_lease`` / ``adopt_write_lease`` is the
    declared mechanism for exactly this: mint the delegation on this thread,
    which already holds the lease, and adopt it inside the new task.
    """
    import asyncio

    from polylogue.annotations.importer import (
        AnnotationBatchImportRequest,
        AnnotationBatchImportResult,
        import_annotation_batch,
    )
    from polylogue.core.write_lease import adopt_write_lease, delegate_write_lease
    from polylogue.operations.ref_resolution import resolve_ref_against_archive

    assert context.runtime is not None
    payload = request.payload
    product_request = AnnotationBatchImportRequest(
        jsonl=str(payload["jsonl"]),
        batch_id=str(payload["batch_id"]),
        schema_id=str(payload["schema_id"]),
        schema_version=int(cast(int, payload["schema_version"])),
        target_ref=str(payload["target_ref"]),
        source_result_ref=str(payload["source_result_ref"]),
        actor_ref=str(payload["actor_ref"]),
        model_ref=str(payload["model_ref"]),
        prompt_ref=str(payload["prompt_ref"]),
        metadata=cast(dict[str, object], payload.get("metadata") or {}),
    )

    class _DaemonImportArchiveHandle:
        """Supplies exactly what ``import_annotation_batch`` reads off ``poly``."""

        archive_root = context.archive_root

        async def resolve_ref(self, ref: str) -> Any:
            return resolve_ref_against_archive(snapshot.archive, ref, archive_root=context.archive_root)

    delegation = delegate_write_lease()

    async def _run() -> AnnotationBatchImportResult:
        with adopt_write_lease(delegation):
            return await import_annotation_batch(cast(Any, _DaemonImportArchiveHandle()), product_request)

    result = asyncio.run(_run())
    return {
        "operation": request.operation,
        "outcome": "completed",
        "sequence": 1,
        "effect": "committed" if result.valid_count else "no-effect",
        "affected_count": result.valid_count,
        "result": result.model_dump(mode="json"),
    }


def mutation_judgment_record(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    """Write one comparative judgment, or one assertion-review batch.

    Both families write ``user.db`` assertion rows through the storage layer's
    own transactional chokepoints, which the Python facade previously called
    from whatever process happened to hold the surface. Running them here puts
    them behind the daemon's single writer; the write itself is unchanged.
    """
    import sqlite3

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.archive_tiers.user_write import (
        ArchiveAssertionBulkJudgmentItemEnvelope,
        judge_assertion_candidates,
        upsert_comparative_judgment_assertion,
    )
    from polylogue.storage.sqlite.connection_profile import open_connection

    payload = request.payload
    user_db = context.archive_root / "user.db"
    kind = str(payload["judgment_kind"])
    if kind == "comparative":
        initialize_archive_database(user_db, ArchiveTier.USER)
    elif not user_db.exists():
        raise ValueError("assertion user tier is not initialized")

    conn = open_connection(user_db)
    conn.row_factory = sqlite3.Row
    try:
        if kind == "comparative":
            from polylogue.operations.judgment_wire import comparative_judgment_from_wire_form

            body = payload["comparative"]
            assert isinstance(body, dict)
            envelope = upsert_comparative_judgment_assertion(
                conn,
                comparative_judgment_from_wire_form(body),
                author_kind=str(payload.get("author_kind") or "user"),
            )
            result: dict[str, object] = {
                "assertion_id": envelope.assertion_id,
                "status": envelope.status.value,
            }
            affected = 1
        else:
            from polylogue.surfaces.payloads import AssertionBulkJudgmentPayload

            reviews = cast(list[dict[str, object]], payload["reviews"])
            batch = judge_assertion_candidates(
                conn,
                tuple(
                    ArchiveAssertionBulkJudgmentItemEnvelope(
                        candidate_ref=str(item["candidate_ref"]),
                        decision=str(item["decision"]),
                        reason=None if item.get("reason") is None else str(item["reason"]),
                        actor_ref=str(item["actor_ref"]),
                        inject=bool(item.get("inject", False)),
                        replacement_body_text=(
                            None if item.get("replacement_body_text") is None else str(item["replacement_body_text"])
                        ),
                        replacement_kind=(
                            None if item.get("replacement_kind") is None else str(item["replacement_kind"])
                        ),
                        expected_evidence_digest=(
                            None
                            if item.get("expected_evidence_digest") is None
                            else str(item["expected_evidence_digest"])
                        ),
                    )
                    for item in reviews
                ),
            )
            result = AssertionBulkJudgmentPayload.from_envelope(batch).model_dump(mode="json")
            affected = batch.applied_count
        conn.commit()
    finally:
        conn.close()
    return {
        "operation": request.operation,
        "outcome": "completed",
        "sequence": 1,
        "effect": "committed" if affected else "no-effect",
        "affected_count": affected,
        "result": result,
    }
