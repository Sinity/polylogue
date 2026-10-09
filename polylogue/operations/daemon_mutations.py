"""Canonical machine mutation adapters over the shared audited executor."""

from __future__ import annotations

import functools
import json
import tempfile
from builtins import BaseExceptionGroup
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic, time
from typing import Any, BinaryIO, cast

from polylogue.operations.audit import (
    MACHINE_PAGE_KINDS,
    MACHINE_PAGE_PARTS,
    AuditRepository,
    MachineRequestBinding,
    machine_pages_kind,
)
from polylogue.operations.bindings import OperationBinding, runtime_operation_binding
from polylogue.operations.daemon_protocol import DaemonAuthorization, DaemonOperationRequest, daemon_operation_spec
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
    MUTATION_PLAN_PAGE_SIZE,
    ConfirmationRequiredError,
    MutationPreview,
    OperationExecutor,
    StartedBoundMutation,
    compute_parameter_digest,
)
from polylogue.operations.operation_context import OperationControlRead, PinnedOperationRead
from polylogue.operations.operation_context_types import OperationContext
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.surfaces.query_rows import query_session_row


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
    spec = daemon_operation_spec(request.operation)
    if spec is None:
        raise ValueError(f"{request.operation} is not a declared operation")
    if (
        spec.authorization is DaemonAuthorization.CONFIRMATION
        and binding.actuator.required_confirmation != "role_only"
        and request.payload.get("confirm") is not True
    ):
        raise ConfirmationRequiredError(f"{request.operation} requires explicit confirmation")
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


def identity_reset_targets(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead | OperationControlRead,
) -> dict[str, object]:
    from polylogue.surfaces.outcome import decide_outcome

    offset = int(cast(int, request.payload.get("offset", 0)))
    ids, total = audit.identity_reset_preview_target_page(
        str(request.payload["preview_request_id"]),
        context.principal,
        archive_identity=snapshot.identity.authority_identity_digest,
        offset=offset,
        page_size=int(cast(int, request.payload.get("page_size", 256))),
    )
    next_offset = offset + len(ids)
    return {
        "session_ids": list(ids),
        "total": total,
        "offset": offset,
        "next_offset": next_offset if next_offset < total else None,
        "outcome": decide_outcome(matched=total).to_dict(),
    }


def mutation_identity_reset_authorize(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    from polylogue.operations.daemon_protocol import AcceptedOperationReference
    from polylogue.operations.mutation_actuators import IdentityResetActuator

    if request.payload.get("confirm") is not True:
        raise ConfirmationRequiredError("identity reset requires explicit confirmation")
    binding = _binding(request, context, snapshot)
    operation = runtime_operation_binding(IdentityResetActuator())
    with _fenced_on_failure(audit, binding):
        for offset, refs, final in _mutation_reference_pages(
            request,
            context,
            audit,
            snapshot,
            authorize=True,
            kind="authorization-batch",
        ):
            previews = tuple(audit.preview_for_principal(ref, context.principal) for ref in refs)
            with audit.bind_identity_reset_request(
                binding,
                request,
                transition="issue_authorization_batch",
                page=(offset, final),
            ):
                executor = OperationExecutor()
                authorizations = tuple(
                    executor.authorize_bound(
                        operation,
                        preview,
                        context.principal,
                        confirmation_strength="bound_token",
                        identity_reset_custody=audit.require_accepted_identity_reset_custody(
                            preview,
                            context.principal,
                            ordinal=offset + index,
                            issued_at_ms=int(time() * 1000),
                        ),
                    )
                    for index, preview in enumerate(previews)
                )
                audit.issue_authorization_batch(previews, context.principal, authorizations)
    record = audit.machine_request(binding)
    assert record is not None
    return {"status": "authorized", "reference": AcceptedOperationReference.from_record(record).to_dict()}


def mutation_identity_reset(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    binding = _binding(request, context, snapshot)
    with _fenced_on_failure(audit, binding):
        for offset, refs, final in _mutation_reference_pages(
            request,
            context,
            audit,
            snapshot,
            authorize=False,
            kind="execution-batch",
        ):
            with audit.bind_identity_reset_request(
                binding,
                request,
                transition="accept_execution_batch",
                page=(offset, final),
            ):
                audit.accept_execution_batch(refs, context.principal)
    state = _execute_batch(request, context, audit, snapshot, ())
    if state["outcome"] == "completed":
        result = state.get("result")
        if not isinstance(result, dict):
            raise ValueError("identity reset historical result is unavailable")
        return result
    return state


_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm")


def reset_confirmation_paths(targets: list[tuple[str, Path]]) -> tuple[str, ...]:
    """The resolved targets an operator's reset confirmation names.

    A SQLite ``-wal``/``-shm`` sidecar exists exactly while some connection
    has its database open -- the daemon's own pinned read creates the index
    sidecars between the CLI preview and this resolution. A sidecar is
    therefore deleted with its confirmed database but is never part of the
    confirmed identity; comparing it would refuse every reset of an open tier.
    """

    resolved = {path.resolve() for _name, path in targets}
    primaries = {
        path
        for path in resolved
        if not (
            path.name.endswith(_SQLITE_SIDECAR_SUFFIXES)
            and path.with_name(path.name.removesuffix("-wal").removesuffix("-shm")) in resolved
        )
    }
    return tuple(sorted(str(path) for path in primaries))


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
        # Only the tiers bootstrap recreates from nothing. source.db, user.db
        # and audit.db are durable: an established archive whose format
        # marker names a missing one refuses to open, so no reset names them.
        # ``embeddings.db`` holds purchased vectors that nothing replays.
        names = [("index database", "index.db")]
        if flags["database"]:
            names.append(("ops database", "ops.db"))
        for name, filename in names:
            path = root / filename
            if path.exists():
                targets.append((name, path))
            for suffix in ("-wal", "-shm"):
                sidecar = path.with_name(f"{path.name}{suffix}")
                if sidecar.exists():
                    targets.append((f"{name} {suffix}", sidecar))
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
    expected = payload.get("expected_targets")
    if expected is not None:
        if not isinstance(expected, list) or any(not isinstance(path, str) for path in expected):
            raise ValueError("reset expected_targets must be a list of absolute paths")
        resolved_expected = tuple(sorted(str(Path(path).resolve()) for path in expected))
        if resolved_expected != reset_confirmation_paths(targets):
            raise ValueError("reset targets changed since confirmation; preview the targets again")
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

    This handler runs inside the daemon, which holds ``index.db`` and
    ``ops.db`` open. A request naming them is staged instead: the executor
    records the authorized plan and its running attempt, nothing is deleted,
    and the next daemon start applies the whole plan before any tier opens
    (``apply_staged_archive_resets``). A target that is or holds a tier no
    reset deletes is refused before any audit row.
    """
    from polylogue.operations.mutation_actuators import FilesystemResetActuator, FilesystemResetArgs
    from polylogue.operations.reset_safety import UnresettableArchiveTierError, classify_reset_targets

    targets = _reset_targets(context.archive_root, request.payload)
    classes = classify_reset_targets(context.archive_root, targets, served_index_path=snapshot.archive.index_db_path)
    if classes.unresettable:
        raise UnresettableArchiveTierError(classes.unresettable_names)
    args = FilesystemResetArgs(archive_root=context.archive_root, targets=tuple(targets))
    if classes.derived_tier_files:
        return _stage_reset_for_daemon_start(request, context, audit, FilesystemResetActuator(), args)
    return _execute_named_mutation(request, context, audit, snapshot, FilesystemResetActuator(), args)


def _stage_reset_for_daemon_start(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    actuator: Any,
    args: Any,
) -> dict[str, object]:
    """Authorize a tier reset and record its running attempt without applying it.

    The attempt belongs to this daemon process. Once the process exits the
    attempt's owner is dead, and the next start's quiescent seam resolves the
    plan through ``FilesystemResetActuator.recover``, which deletes exactly
    the authorized objects. Until then the run stays ``running``, so an
    overlapping reset is refused as concurrent work.
    """
    binding = runtime_operation_binding(actuator)
    if request.payload.get("confirm") is not True:
        raise ConfirmationRequiredError(f"{request.operation} requires explicit confirmation")
    executor = OperationExecutor(audit=audit, archive_root=context.archive_root)
    preview = executor.prepare_bound_for_archive(binding, args, context.principal, archive_root=context.archive_root)
    authorization = executor.authorize_bound(binding, preview, context.principal, confirmation_strength="bound_token")
    started = executor.begin_bound(binding, preview, authorization, args)
    return {
        "operation": request.operation,
        "outcome": "completed",
        "sequence": 1,
        # The durable effect of this request is the staged, authorized plan.
        "effect": "committed",
        "affected_count": 0,
        "receipt_ref": f"mutation-operation:{started.operation_id}",
        "result": {
            "state": "staged",
            "applies_at": "daemon_start",
            "deleted": 0,
            "targets": [name for name, _path in args.targets],
        },
    }


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


def maintenance_secret_scan(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    """Run the secret scanner under the resident writer's admission."""
    del audit, snapshot
    from polylogue.security.secret_scan import (
        DEFAULT_SECRET_SCAN_PAGE_SIZE,
        SECRET_SCAN_VERSION,
        count_pending_secret_scan_sessions,
        scan_archive_for_secret_candidates,
        scan_session_for_secret_candidates,
    )

    payload = request.payload
    detail: dict[str, Any]
    origin = cast(str | None, payload.get("origin"))
    if payload.get("status_only"):
        detail = {
            "scanner_version": SECRET_SCAN_VERSION,
            "remaining_pending": count_pending_secret_scan_sessions(
                context.archive_root / "index.db", context.archive_root / "ops.db", origin=origin
            ),
        }
        effect = "no-effect"
    elif payload.get("scan_all"):
        limit = cast(int | None, payload.get("max_sessions"))
        page_size = limit or DEFAULT_SECRET_SCAN_PAGE_SIZE
        detail = {
            "pages": 0,
            "sessions_scanned": 0,
            "blocks_scanned": 0,
            "candidates_found": 0,
            "errors": 0,
            "remaining_pending": 0,
        }
        while True:
            assert context.runtime is not None
            stop = context.runtime.stop_reason(request)
            if stop is not None:
                from polylogue.archive.query.execution_control import QueryCancelledError

                raise QueryCancelledError(
                    f"secret scan stopped after {detail['sessions_scanned']} sessions: {stop}; "
                    "rerun to continue from durable coverage"
                )
            page = scan_archive_for_secret_candidates(context.archive_root, max_sessions=page_size, origin=origin)
            detail["pages"] += 1
            for key in ("sessions_scanned", "blocks_scanned", "candidates_found", "errors"):
                detail[key] += getattr(page, key)
            detail["remaining_pending"] = page.remaining_pending
            if limit is not None or not page.more_pending or page.sessions_scanned == 0:
                break
        detail["more_pending"] = detail["remaining_pending"] > 0
        effect = "committed" if detail["sessions_scanned"] else "no-effect"
    else:
        session_id = str(payload["session_id"])
        scanned = scan_session_for_secret_candidates(context.archive_root, session_id)
        detail = scanned.as_dict()
        effect = "committed" if scanned.found else "no-effect"
    return {
        "operation": request.operation,
        "outcome": "completed",
        "sequence": 1,
        "effect": effect,
        "result": detail,
    }


def maintenance_schema_quarantine(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    """Persist schema-verification quarantine verdicts under the resident writer."""
    del audit, snapshot
    from polylogue.schemas.validation.corpus import quarantine_raw_sessions

    verdicts = cast(Sequence[Mapping[str, object]], request.payload["verdicts"])
    marked = quarantine_raw_sessions(
        context.archive_root, [(str(verdict["raw_id"]), str(verdict["reason"])) for verdict in verdicts]
    )
    return {
        "operation": request.operation,
        "outcome": "completed",
        "sequence": 1,
        "effect": "committed" if marked else "no-effect",
        "result": {"affected_count": marked},
    }


def mutation_work_evidence_graph_replace(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    """Replace one stored work-evidence graph under the resident writer.

    The repository route is ``async``; see
    :func:`mutation_annotation_import_batch` for why the coroutine adopts a
    delegated write lease instead of running as an unauthorized child task.
    """
    import asyncio

    from polylogue.analysis.work_evidence import WorkEvidenceGraph
    from polylogue.core.write_lease import adopt_write_lease, delegate_write_lease
    from polylogue.operations.work_evidence_writes import (
        WorkEvidenceGraphReplacement,
        replace_work_evidence_graph_checked,
    )
    from polylogue.storage.archive_identity import resolve_active_index_path
    from polylogue.storage.repository import SessionRepository

    del audit, snapshot
    payload = request.payload
    graph = WorkEvidenceGraph.model_validate(payload["graph"])
    expected_base_digest = cast(str | None, payload.get("expected_base_digest"))
    delegation = delegate_write_lease()

    async def _run() -> WorkEvidenceGraphReplacement:
        with adopt_write_lease(delegation):
            async with SessionRepository(
                db_path=resolve_active_index_path(context.archive_root), archive_root=context.archive_root
            ) as repository:
                return await replace_work_evidence_graph_checked(
                    repository, graph, expected_base_digest=expected_base_digest
                )

    replacement = asyncio.run(_run())
    return {
        "operation": request.operation,
        "outcome": "completed",
        "sequence": 1,
        "effect": "committed" if replacement.changed else "no-effect",
        "affected_count": 1 if replacement.changed else 0,
        "result": replacement.to_dict(),
    }


def maintenance_backup(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    """Refuse generic dispatch, which pins a reader across backup checkpoints."""
    del request, context, audit, snapshot
    raise RuntimeError("maintenance.backup requires snapshotless staged execution")


def maintenance_restore_verified_backup(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead,
) -> dict[str, object]:
    """Refuse generic dispatch: population owns a separate destination lease."""
    del request, context, audit, snapshot
    raise RuntimeError("maintenance.restore_verified_backup requires snapshotless staged execution")


def maintenance_embedding_failure_resolve(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: OperationControlRead,
) -> dict[str, object]:
    """Resolve an active embedding failure under the daemon writer."""
    del audit, snapshot
    from polylogue.storage.archive_identity import ArchiveLocation
    from polylogue.storage.embeddings.materialization import resolve_embedding_failure_with_lifecycle

    embeddings_db = ArchiveLocation.resolve(context.archive_root).active_tier("embeddings").configured_path
    if not embeddings_db.is_file():
        raise FileNotFoundError("embeddings.db not found")
    payload = request.payload
    failure = resolve_embedding_failure_with_lifecycle(
        embeddings_db,
        failure_id=str(payload["failure_id"]),
        action=cast(Any, payload["resolution"]),
        note=cast(str | None, payload.get("note")),
        superseded_by=cast(str | None, payload.get("superseded_by")),
    )
    return {
        "operation": request.operation,
        "outcome": "completed",
        "sequence": 1,
        "effect": "committed",
        "affected_count": 1,
        "result": {
            "failure_id": failure.failure_id,
            "session_id": failure.session_id,
            "lifecycle_state": failure.lifecycle_state,
            "resolution_action": failure.resolution_action,
            "resolution_note": failure.resolution_note,
            "superseded_by": failure.superseded_by,
        },
    }


def _audit_int(value: object, *, field: str) -> int:
    """Reject malformed durable counters before scheduling a mutation batch."""

    if type(value) is not int:
        raise ValueError(f"machine mutation {field} is not an integer")
    return value


def _binding(
    request: DaemonOperationRequest, context: OperationContext, snapshot: PinnedOperationRead | OperationControlRead
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
    *,
    expires_at_ms: int | None = None,
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
                expires_at_ms=expires_at_ms,
            )
        )
    return tuple(previews)


def _accepted_pages(audit: AuditRepository, binding: MachineRequestBinding, kind: str) -> int | None:
    """Parts a paged request already accepted: ``None`` when it is complete."""
    prior = audit.machine_request(binding)
    if prior is None:
        return 0
    if prior["artifact_kind"] == machine_pages_kind(kind):
        if prior.get("stop_reason"):
            # Startup fenced a request its dead daemon left half-accepted.
            raise ValueError(f"{binding.operation_name} was interrupted before it was fully accepted; submit it again")
        return _audit_int(prior["part_count"], field="part count")
    return None


@contextmanager
def _fenced_on_failure(audit: AuditRepository, binding: MachineRequestBinding) -> Iterator[None]:
    """Stop a staged request whose later page fails, so it reads terminal."""
    try:
        yield
    except Exception:
        record = audit.machine_request(binding)
        if record is not None and record["artifact_kind"] in MACHINE_PAGE_KINDS and not record.get("stop_reason"):
            audit.stop_machine_batch(binding, "refused")
        raise


def _page_bounds(
    total: int,
    start: int,
    *,
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    binding: MachineRequestBinding,
) -> Iterator[tuple[int, int, bool]]:
    """``(offset, end, final)`` for each page of ``total`` parts from ``start``.

    Between pages, a cancellation fences the staged request, so it reads
    cancelled instead of every remaining page being accepted regardless. The
    request deadline does not: a durably accepted batch that is still
    progressing is finished, not failed and restarted, and its authority is
    judged by its progress (``AuditRepository.handshake_as_of_ms``).
    """
    for offset in range(start, total, MACHINE_PAGE_PARTS):
        if offset > 0 and context.runtime is not None:
            stop = context.runtime.stop_reason(request)
            if stop == "cancelled":
                audit.stop_machine_batch(binding, stop)
                from polylogue.archive.query.execution_control import QueryCancelledError

                raise QueryCancelledError(f"{binding.operation_name} stopped after {offset} parts: {stop}")
        end = min(offset + MACHINE_PAGE_PARTS, total)
        yield offset, end, end == total


def mutation_session_delete_authorize(
    request: DaemonOperationRequest, context: OperationContext, audit: AuditRepository, snapshot: PinnedOperationRead
) -> dict[str, object]:
    from polylogue.operations.daemon_protocol import AcceptedOperationReference

    binding = _binding(request, context, snapshot)
    operation = runtime_operation_binding(SessionDeleteActuator())
    with _fenced_on_failure(audit, binding):
        for offset, refs, final in _mutation_reference_pages(
            request, context, audit, snapshot, authorize=True, kind="authorization-batch"
        ):
            previews = tuple(audit.preview_for_principal(ref, context.principal) for ref in refs)
            as_of_ms = audit.handshake_as_of_ms(binding, preview_refs=refs)
            executor = OperationExecutor(now_ms=functools.partial(int, as_of_ms))
            authorizations = tuple(
                executor.authorize_bound(operation, preview, context.principal, confirmation_strength="bound_token")
                for preview in previews
            )
            with audit.bind_machine_request(binding, transition="issue_authorization_batch", page=(offset, final)):
                audit.issue_authorization_batch(previews, context.principal, authorizations)
    record = audit.machine_request(binding)
    assert record is not None
    result: dict[str, object] = {
        "status": "authorized",
        "reference": AcceptedOperationReference.from_record(record).to_dict(),
        "source_request_id": audit.machine_preview_origin(binding),
    }
    if record["part_count"] == 1:
        result["authorization_ref"] = next(audit.iter_machine_parts(binding))["artifact_ref"]
    return result


def mutation_session_delete_cancel(
    request: DaemonOperationRequest, context: OperationContext, audit: AuditRepository, snapshot: PinnedOperationRead
) -> dict[str, object]:
    from polylogue.operations.daemon_protocol import AcceptedOperationReference

    binding = _binding(request, context, snapshot)
    with _fenced_on_failure(audit, binding):
        for offset, refs, final in _mutation_reference_pages(
            request, context, audit, snapshot, authorize=True, kind="cancelled-preview-batch"
        ):
            previews = tuple(audit.preview_for_principal(ref, context.principal) for ref in refs)
            with audit.bind_machine_request(binding, transition="cancel_preview_batch", page=(offset, final)):
                audit.cancel_preview_batch(previews, context.principal)
    record = audit.machine_request(binding)
    assert record is not None
    return {
        "status": "cancelled",
        "reference": AcceptedOperationReference.from_record(record).to_dict(),
        "source_request_id": audit.machine_preview_origin(binding),
    }


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
    if preview.plan.operation == "mutate-identity-reset":
        from polylogue.operations.mutation_actuators import IdentityResetActuator, IdentityResetArgs

        return runtime_operation_binding(IdentityResetActuator()), IdentityResetArgs(
            archive.user_db_path.parent,
            tuple(cast(list[str], preview.plan.context["session_ids"])),
            str(preview.plan.context["reason"]),
        )
    if preview.plan.operation == "mutate-delete-session":
        return runtime_operation_binding(SessionDeleteActuator()), SessionDeleteArgs(archive, ids)
    if requested_session_ids is None and preview.plan.operation in {
        "mutate-bulk-tag-sessions",
        "mutate-bulk-set-metadata",
    }:
        ids = tuple(cast(list[str], preview.plan.context["requested_session_ids"]))
    if preview.plan.operation == "mutate-bulk-tag-sessions":
        return runtime_operation_binding(BulkTagActuator()), BulkTagArgs(
            archive,
            ids,
            tuple(cast(list[str], preview.plan.context["tags"])),
            author_ref=cast("str | None", preview.plan.context["author_ref"]),
            author_kind=cast("str | None", preview.plan.context["author_kind"]),
        )
    if preview.plan.operation == "mutate-bulk-set-metadata":
        pairs = tuple((str(pair[0]), pair[1]) for pair in cast(list[list[object]], preview.plan.context["pairs"]))
        return runtime_operation_binding(BulkMetadataSetActuator()), BulkMetadataSetArgs(archive, ids, pairs)
    from polylogue.operations.mutation_actuators import (
        AnnotationSaveActuator,
        AnnotationSaveArgs,
        MarkAddActuator,
        MarkArgs,
        MarkRemoveActuator,
        TagRemoveActuator,
        TagRemoveArgs,
    )

    meaning = preview.plan.context
    if preview.plan.operation in {"mutate-add-mark", "mutate-remove-mark"}:
        actuator = MarkAddActuator() if preview.plan.operation == "mutate-add-mark" else MarkRemoveActuator()
        return runtime_operation_binding(actuator), MarkArgs(
            archive=archive,
            target_type=str(meaning["target_type"]),
            target_id=str(meaning["target_id"]),
            mark_type=str(meaning["mark_type"]),
            owner_session_id=cast(str | None, meaning["owner_session_id"]),
        )
    if preview.plan.operation == "mutate-save-annotation":
        return runtime_operation_binding(AnnotationSaveActuator()), AnnotationSaveArgs(
            archive=archive,
            annotation_id=str(meaning["annotation_id"]),
            target_type=str(meaning["target_type"]),
            target_id=str(meaning["target_id"]),
            note_text=str(meaning["note_text"]),
            owner_session_id=cast(str | None, meaning["owner_session_id"]),
        )
    if preview.plan.operation == "mutate-remove-tag":
        return runtime_operation_binding(TagRemoveActuator()), TagRemoveArgs(
            archive=archive,
            session_id=str(meaning["session_id"]),
            tag=str(meaning["tag"]),
        )
    raise ValueError("unsupported durable machine mutation family")


def _execute_batch(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead | OperationControlRead,
    refs: tuple[str, ...],
) -> dict[str, object]:
    assert context.runtime is not None
    binding = _binding(request, context, snapshot)
    accepted = _accepted_pages(audit, binding, "execution-batch")
    if accepted is not None:
        with _fenced_on_failure(audit, binding):
            for offset, end, final in _page_bounds(
                len(refs), accepted, request=request, context=context, audit=audit, binding=binding
            ):
                with audit.bind_machine_request(binding, transition="accept_execution_batch", page=(offset, final)):
                    audit.accept_execution_batch(refs[offset:end], context.principal)
    record = audit.machine_request(binding)
    assert record is not None
    if record["artifact_kind"] != "execution-batch":
        raise ValueError("machine request is not an execution batch")
    with ArchiveStore.open_existing(context.archive_root, read_only=False) as archive:
        executor = OperationExecutor(audit=audit, archive_root=context.archive_root)
        for part in audit.iter_machine_parts(binding):
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
            # Only an explicit cancellation stops a progressing execution: a
            # request deadline would fence the untouched suffix of a deletion
            # that is still advancing, leaving it partial for no reason.
            if context.runtime.stop_reason(request) == "cancelled":
                audit.stop_machine_batch(binding, "cancelled")
                break
            try:
                preview, authorization = audit.authorization_for_principal(
                    str(part["authorization_ref"]), context.principal
                )
                requested_ids = None
                if request.operation in {"mutation.session.tag", "mutation.session.metadata"}:
                    offset = _audit_int(part["ordinal"], field="part ordinal") * MUTATION_PLAN_PAGE_SIZE
                    requested_ids = tuple(
                        cast(list[str], request.payload["session_ids"])[offset : offset + MUTATION_PLAN_PAGE_SIZE]
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
    request: DaemonOperationRequest, context: OperationContext, audit: AuditRepository, snapshot: PinnedOperationRead
) -> dict[str, object]:
    binding = _binding(request, context, snapshot)
    with _fenced_on_failure(audit, binding):
        for offset, refs, final in _mutation_reference_pages(
            request, context, audit, snapshot, authorize=False, kind="execution-batch"
        ):
            with audit.bind_machine_request(binding, transition="accept_execution_batch", page=(offset, final)):
                audit.accept_execution_batch(refs, context.principal)
    return _execute_batch(request, context, audit, snapshot, ())


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
        for offset in range(0, len(ids), MUTATION_PLAN_PAGE_SIZE):
            chunk = ids[offset : offset + MUTATION_PLAN_PAGE_SIZE]
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
    archive.require_user_tier()
    durable = resolve_durable_user_state_session_id(archive_root, token, connection=archive._conn, schema="user_tier")
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
            evidence_refs=tuple(str(ref) for ref in cast(list[str], payload.get("evidence_refs") or [])),
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
        created_at_ms=(int(cast(int, payload["created_at_ms"])) if payload.get("created_at_ms") is not None else None),
    )

    registry = None
    schema_definition_json = payload.get("schema_definition_json")
    if schema_definition_json is not None:
        from polylogue.annotations.schema import AnnotationSchema, AnnotationSchemaRegistry

        if not isinstance(schema_definition_json, str):
            raise ValueError("schema_definition_json must be a string")
        schema = AnnotationSchema.from_canonical_definition_json(schema_definition_json)
        if schema.schema_id != product_request.schema_id or schema.version != product_request.schema_version:
            raise ValueError("schema_definition_json does not match the import request schema identity")
        registry = AnnotationSchemaRegistry()
        registry.register(schema)

    class _DaemonImportArchiveHandle:
        """Supplies exactly what ``import_annotation_batch`` reads off ``poly``."""

        archive_root = context.archive_root

        async def resolve_ref(self, ref: str) -> Any:
            return resolve_ref_against_archive(snapshot.archive, ref, archive_root=context.archive_root)

    delegation = delegate_write_lease()
    runtime = context.runtime

    def accept() -> None:
        # Validation can outlive the exchange's deadline. Cross the acceptance
        # boundary only if the exchange is still live, so the runtime never
        # reports timed-out or disconnected-before-acceptance for a write that
        # then commits.
        runtime.begin_unbound_write(request, snapshot=snapshot)

    async def _run() -> AnnotationBatchImportResult:
        with adopt_write_lease(delegation):
            handle = cast(Any, _DaemonImportArchiveHandle())
            if registry is None:
                return await import_annotation_batch(handle, product_request, before_durable_execution=accept)
            return await import_annotation_batch(
                handle, product_request, registry=registry, before_durable_execution=accept
            )

    result = asyncio.run(_run())
    return {
        "operation": request.operation,
        "outcome": "completed",
        "sequence": 1,
        "effect": "committed",
        "affected_count": result.valid_count + 1,
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

    conn = open_connection(user_db, archive_root=context.archive_root)
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
                        replacement_value=item.get("replacement_value"),
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


def _observe_frontier_authority(request: DaemonOperationRequest, context: OperationContext) -> OperationControlRead:
    """Control provenance plus the Index condition the frontier owner reads.

    Control authority carries Source/Audit versions only. Frontier inspection
    also reads the active Index, so a caller's Index precondition is observed
    on that tier instead of being compared against an absent version.
    """
    from polylogue.archive.query.execution_control import QueryExecutionContext
    from polylogue.operations.daemon_execution import _observe_explicit_index_condition
    from polylogue.operations.operation_context import observe_control_authority

    read_control = context.read_control or QueryExecutionContext(
        call_id=str(request.request_id),
        query_ref=request.fingerprint,
        deadline_monotonic=None,
        owner_ref=context.principal.actor_ref,
    )
    authority = _observe_explicit_index_condition(
        request,
        observe_control_authority(context.archive_root),
        archive_root=context.archive_root,
        read_control=read_control,
    )
    read_control.mark_cleanup_complete()
    return authority


async def execute_raw_authority_blocker_resolve_operation(
    request: DaemonOperationRequest,
    context: OperationContext,
) -> Any:
    """Acknowledge one frontier obligation on its original admitted preparation owner."""
    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.operations.daemon_execution import _validate_identity, operation_envelope, validate_execution_request
    from polylogue.operations.daemon_protocol import validate_operation_result
    from polylogue.operations.mutation_actuators import BlockerResolveActuator, BlockerResolveArgs
    from polylogue.operations.operation_context import observe_control_authority
    from polylogue.storage.frontier_inspection import prepared_frontier_blocker_acknowledgement

    request = validate_execution_request(request, context)
    if request.payload.get("confirm") is not True:
        raise ConfirmationRequiredError(f"{request.operation} requires explicit confirmation")
    runtime = context.runtime
    assert runtime is not None
    await runtime.recover_interrupted_operations(resolver_actor_ref=context.principal.actor_ref)
    started = monotonic()
    audit = runtime.audit_for_request(request, context)

    def execute() -> Any:
        authority = _observe_frontier_authority(request, context)
        _validate_identity(request, context, authority)
        runtime.observe_snapshot(request, authority)
        blocker_id = str(request.payload["blocker_id"])
        resolution = str(request.payload["resolution"]).strip()
        actuator = BlockerResolveActuator()
        binding = runtime_operation_binding(actuator)
        executor = OperationExecutor(audit=audit, archive_root=context.archive_root)
        begun = None
        try:
            with prepared_frontier_blocker_acknowledgement(
                context.archive_root,
                blocker_id,
                resolution=resolution,
                input_demand=runtime.prepared_compute_adapter().amend_current_input_demand,
            ) as prepared:
                args = BlockerResolveArgs(context.archive_root, blocker_id, resolution, prepared)

                def begin() -> StartedBoundMutation:
                    # Re-observe the same authority the admission read did:
                    # control provenance alone carries no Index version, so a
                    # caller's Index precondition compared against it always
                    # refused as ``schema_version_mismatch``.
                    current = _observe_frontier_authority(request, context)
                    _validate_identity(request, context, current)
                    if current.identity != authority.identity:
                        raise ValueError("archive changed before blocker acknowledgement authorization")
                    runtime.begin_unbound_write(request, snapshot=authority)
                    preview = executor.prepare_bound_for_archive(
                        binding, args, context.principal, archive_root=context.archive_root
                    )
                    authorization = executor.authorize_bound(
                        binding, preview, context.principal, confirmation_strength="bound_token"
                    )
                    return executor.begin_bound(binding, preview, authorization, args)

                begun = admit_stage_write("operation.frontier.blocker.begin", begin)
            # Starting the audited attempt changes Source continuity. Capture the
            # publication operands only afterwards; the original seal is settled,
            # never refreshed or reused after that durable intent.
            with prepared_frontier_blocker_acknowledgement(
                context.archive_root,
                blocker_id,
                resolution=resolution,
                input_demand=runtime.prepared_compute_adapter().amend_current_input_demand,
            ) as prepared:
                args = BlockerResolveArgs(context.archive_root, blocker_id, resolution, prepared)
                receipt = admit_stage_write(
                    "operation.frontier.blocker.publish", lambda: actuator.apply(begun.plan, args)
                )
        except BaseException as exc:
            if begun is None:
                raise
            error_summary = str(exc)[:512]
            try:
                admit_stage_write(
                    "operation.frontier.blocker.indeterminate",
                    lambda: executor.finalize_bound(
                        begun,
                        error_summary=error_summary,
                        unknown_reason="acknowledgement failed after durable intent",
                    ),
                )
            except BaseException as cleanup:
                raise BaseExceptionGroup("Acknowledgement and audit finalization failed", [exc, cleanup]) from exc
            raise
        finalized = admit_stage_write(
            "operation.frontier.blocker.finalize", lambda: executor.finalize_bound(begun, receipt=receipt)
        )
        assert finalized is not None
        receipt = finalized
        result = {
            "operation": request.operation,
            "outcome": "completed",
            "sequence": 1,
            "effect": "committed" if receipt.affected_count else "no-effect",
            "affected_count": receipt.affected_count,
            "receipt_ref": receipt.receipt_ref,
            "result": dict(receipt.domain_receipt),
        }
        validate_operation_result(request.operation, result)
        settled = observe_control_authority(context.archive_root)
        return operation_envelope(
            request, context, snapshot=settled, admitted_snapshot=authority, started_at=started, result=result
        )

    return await runtime.prepared_phase("frontier.blocker.resolve", execute, estimated_bytes=0, exclusive_bytes=True)


async def execute_raw_authority_frontier_operation(request: DaemonOperationRequest, context: OperationContext) -> Any:
    """Measure original frontier inputs through this operation's supplied owner."""
    from dataclasses import asdict

    from polylogue.operations.daemon_execution import _validate_identity, operation_envelope, validate_execution_request
    from polylogue.operations.daemon_protocol import validate_operation_result
    from polylogue.operations.operation_context import observe_control_authority
    from polylogue.storage.frontier_inspection import FrontierInspectionOutcome, inspect_prepared_raw_authority_frontier

    request = validate_execution_request(request, context)
    runtime = context.runtime
    if runtime is None:
        raise PermissionError("daemon_required")
    started = monotonic()

    def accept() -> None:
        authority = _observe_frontier_authority(request, context)
        _validate_identity(request, context, authority)
        runtime.observe_snapshot(request, authority)
        runtime.begin_unbound_write(request, snapshot=authority)

    await runtime.write_phase("frontier.accept", accept)

    def inspect() -> FrontierInspectionOutcome:
        adapter = runtime.prepared_compute_adapter()
        return inspect_prepared_raw_authority_frontier(
            context.archive_root,
            input_demand=adapter.amend_current_input_demand,
            check_physical_dependencies=True,
        )

    measured = await runtime.prepared_phase("frontier.inspect", inspect, estimated_bytes=0, exclusive_bytes=True)
    result = {
        "operation": request.operation,
        "outcome": "completed",
        "sequence": 1,
        "effect": "no-effect" if measured.mode == "current" else "committed",
        "result": asdict(measured),
    }
    validate_operation_result(request.operation, result)
    authority = observe_control_authority(context.archive_root)
    return operation_envelope(request, context, snapshot=authority, started_at=started, result=result)


def _source_mutation_parts(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead | OperationControlRead,
    *,
    authorize: bool,
) -> tuple[Iterator[dict[str, object]], int]:
    """Resolve authenticated completed phase authority on the resident owner."""
    key = "preview_request_id" if authorize else "authorization_request_id"
    target = request.payload.get(key)
    if target is None:
        if request.operation.startswith("mutation.identity-reset"):
            raise ValueError("reset phase requires its sealed source request")
        ref = str(request.payload["preview_ref" if authorize else "authorization_ref"])
        return iter(({"ordinal": 0, "artifact_ref": ref},)), 1
    record = audit.machine_request_for_principal(
        snapshot.identity.authority_identity_digest, str(target), context.principal.actor_ref
    )
    family = (
        "mutation.identity-reset"
        if request.operation.startswith("mutation.identity-reset")
        else "mutation.session.delete"
    )
    expected = f"{family}.preview" if authorize else f"{family}.authorize"
    kind = "preview-batch" if authorize else "authorization-batch"
    if record is None or record["operation_name"] != expected or record["artifact_kind"] != kind:
        raise ValueError("delete operation reference does not name a sealed phase")
    source = MachineRequestBinding(
        *(
            str(record[field])
            for field in ("archive_identity", "request_id", "principal_ref", "fingerprint", "operation_name")
        )
    )
    return audit.iter_machine_parts(source), _audit_int(record["part_count"], field="part count")


def _mutation_reference_pages(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    snapshot: PinnedOperationRead | OperationControlRead,
    *,
    authorize: bool,
    kind: str,
) -> Iterator[tuple[int, tuple[str, ...], bool]]:
    binding = _binding(request, context, snapshot)
    accepted = _accepted_pages(audit, binding, kind)
    if accepted is None:
        return
    parts, total = _source_mutation_parts(request, context, audit, snapshot, authorize=authorize)
    refs: list[str] = []
    offset = accepted
    for part in parts:
        if _audit_int(part["ordinal"], field="part ordinal") < accepted:
            continue
        if context.runtime is not None and context.runtime.stop_reason(request) == "cancelled":
            from polylogue.archive.query.execution_control import QueryCancelledError

            raise QueryCancelledError("Delete phase was cancelled before all authority pages were sealed.")
        refs.append(str(part["artifact_ref"]))
        if len(refs) == MACHINE_PAGE_PARTS:
            yield offset, tuple(refs), offset + len(refs) == total
            offset += len(refs)
            refs.clear()
    if refs:
        yield offset, tuple(refs), offset + len(refs) == total
        offset += len(refs)
    if offset != total:
        raise ValueError("delete operation authority part count is incomplete")


class MutationSelectionError(ValueError):
    """The resident selected relation cannot authorize this mutation."""

    def __init__(self, code: str, detail: str, data: dict[str, object] | None = None) -> None:
        self.code = code
        self.detail = detail
        self.data = data or {}
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class _PreparedMutationSelection:
    authority: OperationControlRead
    frame: str
    session_count: int
    sample: tuple[str, ...]


_MUTATION_SELECTION_PAGE_SIZE = 500


def _prepare_mutation_selection(
    request: DaemonOperationRequest,
    context: OperationContext,
    document: BinaryIO,
) -> _PreparedMutationSelection:
    """Seal canonical selected identities on disk inside one admitted read frame."""
    from polylogue.archive.query.transaction import archive_snapshot_epoch
    from polylogue.operations.daemon_execution import _validate_identity
    from polylogue.operations.daemon_reads import (
        DaemonReadDependencies,
        execute_read_operation,
        requires_vector_snapshot,
    )
    from polylogue.operations.operation_context import abort_checkpoint, open_operation_read
    from polylogue.storage.io_phase_metrics import connect_measured
    from polylogue.surfaces.outcome import OutcomeEnvelope

    runtime = context.runtime
    assert runtime is not None
    checkpoint = abort_checkpoint(context.read_control) if context.read_control is not None else lambda: None
    selected = request.payload.get("selection")
    params = dict(cast(dict[str, object], selected["params"])) if isinstance(selected, dict) else {}
    mode = str(selected["mode"]) if isinstance(selected, dict) else "explicit"
    dependencies = context.read_dependencies or DaemonReadDependencies()
    vector_recipe = (
        dependencies.vector_binding.recipe
        if dependencies.vector_binding is not None
        and requires_vector_snapshot(
            "cli.query", {"params": params}, acquisition_enabled=bool(dependencies.vector_binding.voyage_key)
        )
        else None
    )
    with open_operation_read(
        context.archive_root,
        publication_guard=runtime.publication_guard,
        vector_recipe=vector_recipe,
        execution_context=context.read_control,
    ) as pinned:
        _validate_identity(request, context, pinned)
        runtime.observe_snapshot(request, pinned)
        dependencies = replace(
            dependencies,
            vector_connection=pinned.archive.operation_vector_connection,
            vector_failure=pinned.vector_failure or dependencies.vector_failure,
            raise_if_aborted=checkpoint,
        )
        frame = f"{pinned.archive.index_db_path.resolve()}:{archive_snapshot_epoch(pinned.archive)}"
        sample: list[str] = []
        count = 0
        with (
            tempfile.TemporaryDirectory(prefix="polylogue-selection-") as directory,
            closing(connect_measured(Path(directory) / "keys.db")) as keys,
        ):
            keys.execute("CREATE TABLE selected_ids (id BLOB PRIMARY KEY) WITHOUT ROWID")

            def seen(session_id: str) -> bool:
                return (
                    keys.execute(
                        "SELECT 1 FROM selected_ids WHERE id = ?", (session_id.encode("utf-8", "surrogatepass"),)
                    ).fetchone()
                    is not None
                )

            def retain(session_id: str) -> None:
                nonlocal count
                checkpoint()
                if seen(session_id):
                    return
                keys.execute("INSERT INTO selected_ids VALUES (?)", (session_id.encode("utf-8", "surrogatepass"),))
                # Each identity is an existing exact JSON scalar; the complete
                # selected population is never represented as a Python list.
                document.write(json.dumps(session_id, ensure_ascii=True).encode("ascii") + b"\n")
                count += 1
                if len(sample) < 5:
                    sample.append(session_id)

            if mode == "explicit":
                tokens = cast(list[str], request.payload["session_ids"])
                if request.operation == "mutation.session.delete.preview":
                    for offset in range(0, len(tokens), 256):
                        chunk = tuple(tokens[offset : offset + 256])
                        exact = pinned.archive.resolve_exact_session_ids(chunk, page_size=256)
                        for token in chunk:
                            sid = exact.get(token)
                            if sid is None:
                                raise MutationSelectionError(
                                    "selection_is_stale", "An exact deletion target no longer exists."
                                )
                            if seen(sid):
                                raise MutationSelectionError(
                                    "selection_is_not_canonical",
                                    "Deletion targets must be distinct canonical sessions.",
                                )
                            retain(sid)
                else:
                    for token in tokens:
                        retain(_resolve_session_target(pinned.archive, context.archive_root, token))
            else:
                requested_limit = params.get("limit")
                if mode == "page" and requested_limit is not None and type(requested_limit) is not int:
                    raise ValueError("query display limit must be an integer")
                page_size = (
                    (requested_limit if requested_limit is not None else 10)
                    if mode == "page"
                    else 1
                    if mode == "first"
                    else 2
                    if mode == "single"
                    else _MUTATION_SELECTION_PAGE_SIZE
                )
                params = {**params, "limit": page_size, "offset": params.get("offset", 0) if mode == "page" else 0}
                expected_total: int | None = None
                total_is_known = False
                while True:
                    checkpoint()
                    page = execute_read_operation(
                        "cli.query",
                        {"params": params},
                        archive=pinned.archive,
                        serving_identity=context.serving_identity,
                        dependencies=dependencies,
                    )
                    outcome = OutcomeEnvelope.model_validate(page["outcome"])
                    if not outcome.rows_are_authoritative:
                        raise MutationSelectionError(
                            "selection_not_authoritative",
                            "Mutation selection is not authoritative.",
                            {"outcome": outcome.model_dump(mode="json")},
                        )
                    raw_rows = page.get("hits") if isinstance(page.get("hits"), list) else page.get("items")
                    rows = raw_rows if isinstance(raw_rows, list) else []
                    block_grain = isinstance(page.get("hits"), list)
                    raw_total = page.get("total")
                    if raw_total is not None:
                        if type(raw_total) is not int or raw_total < 0:
                            raise MutationSelectionError(
                                "query_selection_incomplete", "Canonical selection returned an invalid total."
                            )
                        if total_is_known and raw_total != expected_total:
                            raise MutationSelectionError(
                                "query_selection_incomplete", "Canonical selection changed its total."
                            )
                        expected_total, total_is_known = raw_total, True
                    elif total_is_known:
                        raise MutationSelectionError(
                            "query_selection_incomplete", "Canonical selection stopped reporting its total."
                        )
                    if mode != "page" and "next_offset" not in page:
                        raise MutationSelectionError(
                            "query_selection_incomplete", "Canonical selection omitted continuation metadata."
                        )
                    for row in rows:
                        if not isinstance(row, dict):
                            raise ValueError("canonical selection returned a non-object row")
                        session = query_session_row(row, ranked=block_grain)
                        selected_id = session.get("id") or session.get("session_id")
                        if not isinstance(selected_id, str) or not selected_id:
                            raise ValueError("canonical selection returned no session identity")
                        if seen(selected_id) and not block_grain:
                            raise MutationSelectionError(
                                "query_selection_incomplete", "Canonical selection repeated a session row."
                            )
                        retain(selected_id)
                        if mode == "first" or mode == "single" and count >= 2:
                            break
                    if mode == "page" or mode == "first" and count or mode == "single" and count >= 2:
                        break
                    cursor, next_offset = page.get("next_cursor"), page.get("next_offset")
                    if cursor is None and next_offset is None:
                        if total_is_known and not block_grain and count != expected_total:
                            raise MutationSelectionError(
                                "query_selection_incomplete", "Canonical selection ended before its reported total."
                            )
                        break
                    if cursor is None:
                        current_offset = params.get("offset", 0)
                        if (
                            type(next_offset) is not int
                            or type(current_offset) is not int
                            or next_offset <= current_offset
                        ):
                            raise MutationSelectionError(
                                "query_pagination_stalled", "Canonical selection continuation did not advance."
                            )
                        if next_offset != current_offset + len(rows):
                            raise MutationSelectionError(
                                "query_selection_incomplete",
                                "Canonical selection continuation skipped or overlapped rows.",
                            )
                        if (
                            total_is_known
                            and not block_grain
                            and expected_total is not None
                            and next_offset > expected_total
                        ):
                            raise MutationSelectionError(
                                "query_selection_incomplete",
                                "Canonical selection continuation exceeded its reported total.",
                            )
                    if (
                        not rows
                        or cursor is not None
                        and cursor == params.get("cursor")
                        or cursor is None
                        and next_offset == params.get("offset")
                    ):
                        raise MutationSelectionError("query_pagination_stalled", "Canonical selection did not advance.")
                    params = {**params, "cursor": cursor, "offset": 0 if cursor is not None else next_offset}
        if request.operation == "mutation.session.mark" and mode not in {"explicit", "page"} and count == 0:
            raise MutationSelectionError("selection_empty", "No sessions matched the mutation selection.")
        if mode == "single" and count > 1:
            raise MutationSelectionError(
                "selection_ambiguous",
                "Mutation selection matched more than one session.",
                {"session_ids_sample": sample},
            )
        document.flush()
        document.seek(0)
        return _PreparedMutationSelection(
            OperationControlRead(pinned.identity, dict(pinned.schema_versions), pinned.degraded_components),
            frame,
            count,
            tuple(sample),
        )


def _prepare_identity_reset_selection(
    request: DaemonOperationRequest,
    context: OperationContext,
    document: BinaryIO,
) -> _PreparedMutationSelection:
    """Freeze source identities on disk before publishing any reset authority."""
    from polylogue.archive.query.transaction import archive_snapshot_epoch
    from polylogue.operations.cli_aux_reads import _iter_sessions_from_source_path, _resolve_session_prefixes
    from polylogue.operations.daemon_execution import _validate_identity
    from polylogue.operations.operation_context import abort_checkpoint, open_operation_read
    from polylogue.storage.sqlite.connection_profile import readonly_temp_staging

    runtime = context.runtime
    assert runtime is not None
    checkpoint = abort_checkpoint(context.read_control) if context.read_control is not None else lambda: None
    with open_operation_read(context.archive_root, publication_guard=runtime.publication_guard) as pinned:
        _validate_identity(request, context, pinned)
        runtime.observe_snapshot(request, pinned)
        authority = OperationControlRead(pinned.identity, dict(pinned.schema_versions), pinned.degraded_components)
        frame = f"{pinned.archive.index_db_path.resolve()}:{archive_snapshot_epoch(pinned.archive)}"
        session = request.payload.get("session")
        selected = (
            iter(_resolve_session_prefixes(pinned.archive, [session]))
            if isinstance(session, str)
            else _iter_sessions_from_source_path(pinned.archive, Path(str(request.payload["source_path"])))
        )
        count = 0
        sample: list[str] = []
        try:
            with readonly_temp_staging(pinned.archive._conn, temp_store="FILE"):
                for session_id in selected:
                    checkpoint()
                    document.write(json.dumps(session_id).encode("utf-8") + b"\n")
                    count += 1
                    if len(sample) < 5:
                        sample.append(session_id)
        finally:
            if not isinstance(session, str):
                cast(Any, selected).close()
        checkpoint()
        document.flush()
        return _PreparedMutationSelection(authority, frame, count, tuple(sample))


async def execute_selected_preview_operation(request: DaemonOperationRequest, context: OperationContext) -> Any:
    """Prepare any canonical selection on the resident owner without client IDs."""
    from polylogue.archive.query.transaction import archive_snapshot_epoch
    from polylogue.operations.daemon_execution import _validate_identity, operation_envelope
    from polylogue.operations.operation_context import open_operation_read

    runtime = context.runtime
    if runtime is None:
        raise PermissionError("daemon_required")
    started = monotonic()
    audit = runtime.audit_for_request(request, context)

    def prior() -> tuple[OperationControlRead, dict[str, object] | None]:
        with open_operation_read(context.archive_root, publication_guard=runtime.publication_guard) as pinned:
            _validate_identity(request, context, pinned)
            runtime.observe_snapshot(request, pinned)
            authority = OperationControlRead(pinned.identity, dict(pinned.schema_versions), pinned.degraded_components)
            # A lookup on a compute worker holds no writer lease; it reads the
            # settled audit tier, never audit's writer leaf.
            with audit.settled_machine_read():
                return authority, audit.machine_request(_binding(request, context, authority))

    authority, existing = await runtime.compute_phase(prior)
    with tempfile.TemporaryFile(mode="w+b") as document:
        selection = None
        if existing is None:
            selection = await runtime.compute_phase(
                lambda: (
                    _prepare_identity_reset_selection
                    if request.operation == "mutation.identity-reset.preview"
                    else _prepare_mutation_selection
                )(request, context, document)
            )
            authority = selection.authority
        elif existing["artifact_kind"] != "preview-batch":
            raise MutationSelectionError(
                "mutation_acceptance_interrupted", "The original deletion preview was not sealed."
            )
        binding = _binding(request, context, authority)

        def prepare() -> dict[str, object]:
            if selection is not None:
                # This callback already holds the daemon writer gate. Pin the
                # existing read scope without submitting a nested gate owner.
                with open_operation_read(context.archive_root) as pinned:
                    _validate_identity(request, context, pinned)
                    current = f"{pinned.archive.index_db_path.resolve()}:{archive_snapshot_epoch(pinned.archive)}"
                    if current != selection.frame:
                        raise MutationSelectionError(
                            "selection_frame_changed", "The deletion selection changed before acceptance."
                        )
                if selection.session_count == 0 and request.operation != "mutation.identity-reset.preview":
                    return {
                        "status": "prepared",
                        "operation": "delete",
                        "session_count": 0,
                        "session_ids_sample": [],
                        "affected_count": 0,
                    }
                total = max(1, (selection.session_count + MUTATION_PLAN_PAGE_SIZE - 1) // MUTATION_PLAN_PAGE_SIZE)
                offset = 0
                chunk: list[str] = []
                pending: list[MutationPreview] = []
                from polylogue.operations.mutation_actuators import IdentityResetActuator, IdentityResetArgs

                reset = request.operation == "mutation.identity-reset.preview"
                operation: OperationBinding[Any, object] = runtime_operation_binding(
                    IdentityResetActuator() if reset else SessionDeleteActuator()
                )
                with (
                    _fenced_on_failure(audit, binding),
                    ArchiveStore.open_existing(context.archive_root, read_only=False) as archive,
                ):
                    instance = audit.ensure_archive_authority(now_ms=int(time() * 1000))
                    executor = OperationExecutor()

                    def retain() -> None:
                        args = (
                            IdentityResetArgs(context.archive_root, tuple(chunk), str(request.payload["reason"]))
                            if reset
                            else SessionDeleteArgs(archive, tuple(chunk))
                        )
                        raw = operation.actuator.prepare(args)
                        pending.append(
                            executor.prepare_bound(
                                operation,
                                args,
                                context.principal,
                                archive_instance_id=instance,
                                archive_identity_digest=authority.identity.authority_identity_digest,
                                parameter_digest=compute_parameter_digest(raw),
                                raw_plan=raw,
                            )
                        )
                        chunk.clear()

                    def accept() -> None:
                        nonlocal offset
                        if runtime.stop_reason(request) == "cancelled":
                            from polylogue.archive.query.execution_control import QueryCancelledError

                            raise QueryCancelledError("Delete preview cancelled before selection was sealed.")
                        with audit.bind_machine_request(
                            binding, transition="create_preview_batch", page=(offset, offset + len(pending) == total)
                        ):
                            audit.create_preview_batch(tuple(preview.plan for preview in pending), context.principal)
                        offset += len(pending)
                        pending.clear()

                    document.seek(0)
                    for line in document:
                        sid = json.loads(line)
                        if not isinstance(sid, str) or not sid:
                            raise ValueError("sealed deletion identity is invalid")
                        chunk.append(sid)
                        if len(chunk) == MUTATION_PLAN_PAGE_SIZE:
                            retain()
                            if len(pending) == MACHINE_PAGE_PARTS:
                                accept()
                    if chunk or selection.session_count == 0:
                        retain()
                    if pending:
                        accept()
                    if offset != total:
                        raise ValueError("sealed deletion part count differs from its complete selection")
            return audit.machine_preview_summary(binding)

        result = await runtime.write_phase(request.operation, prepare)
    return operation_envelope(request, context, snapshot=authority, started_at=started, result=result)


def _combined_user_intents(
    archive: ArchiveStore, document: BinaryIO, payload: dict[str, object]
) -> Iterator[tuple[Any, object]]:
    """Lower a sealed identity stream to the existing bounded actuators."""
    import hashlib

    from polylogue.core.user_state_targets import TARGET_SESSION
    from polylogue.operations.mutation_actuators import (
        AnnotationSaveActuator,
        AnnotationSaveArgs,
        MarkAddActuator,
        MarkArgs,
        MarkRemoveActuator,
        TagRemoveActuator,
        TagRemoveArgs,
    )

    def intents(ids: tuple[str, ...]) -> Iterator[tuple[Any, object]]:
        tags = tuple(cast(list[str], payload.get("tags") or []))
        pairs = tuple((pair[0], pair[1]) for pair in cast(list[list[str]], payload.get("pairs") or []))
        if tags:
            yield BulkTagActuator(), BulkTagArgs(archive, ids, tags)
        if pairs:
            yield BulkMetadataSetActuator(), BulkMetadataSetArgs(archive, ids, pairs)
        for sid in ids:
            for tag in cast(list[str], payload.get("remove_tags") or []):
                yield TagRemoveActuator(), TagRemoveArgs(archive, sid, tag)
            for field, actuator in (("add_marks", MarkAddActuator()), ("remove_marks", MarkRemoveActuator())):
                for mark in cast(list[str], payload.get(field) or []):
                    yield actuator, MarkArgs(archive, TARGET_SESSION, sid, mark, owner_session_id=sid)
            note = payload.get("note_text")
            if isinstance(note, str):
                digest = hashlib.sha256(sid.encode("utf-8", errors="surrogatepass")).hexdigest()
                yield (
                    AnnotationSaveActuator(),
                    AnnotationSaveArgs(archive, f"note-{digest}", TARGET_SESSION, sid, note, owner_session_id=sid),
                )

    chunk: list[str] = []
    emitted = False
    document.seek(0)
    for line in document:
        sid = json.loads(line)
        if not isinstance(sid, str) or not sid:
            raise ValueError("sealed mutation identity is invalid")
        chunk.append(sid)
        if len(chunk) == MUTATION_PLAN_PAGE_SIZE:
            yield from intents(tuple(chunk))
            chunk.clear()
            emitted = True
    if chunk or not emitted:
        yield from intents(tuple(chunk))


async def execute_session_mark_operation(request: DaemonOperationRequest, context: OperationContext) -> Any:
    """Seal selection and every intent before applying one durable User batch."""
    from polylogue.archive.query.transaction import archive_snapshot_epoch
    from polylogue.operations.daemon_execution import _validate_identity, operation_envelope
    from polylogue.operations.operation_context import open_operation_read

    runtime = context.runtime
    if runtime is None:
        raise PermissionError("daemon_required")
    started = monotonic()
    audit = runtime.audit_for_request(request, context)

    def prior() -> tuple[OperationControlRead, dict[str, object] | None]:
        with open_operation_read(context.archive_root, publication_guard=runtime.publication_guard) as pinned:
            _validate_identity(request, context, pinned)
            runtime.observe_snapshot(request, pinned)
            authority = OperationControlRead(pinned.identity, dict(pinned.schema_versions), pinned.degraded_components)
            # A lookup on a compute worker holds no writer lease; it reads the
            # settled audit tier, never audit's writer leaf.
            with audit.settled_machine_read():
                return authority, audit.machine_request(_binding(request, context, authority))

    authority, existing = await runtime.compute_phase(prior)
    with tempfile.TemporaryFile(mode="w+b") as document:
        selection = None
        if existing is None:
            selection = await runtime.compute_phase(lambda: _prepare_mutation_selection(request, context, document))
            authority = selection.authority
        elif existing["artifact_kind"] != "execution-batch":
            raise MutationSelectionError(
                "mutation_acceptance_interrupted", "The original mutation selection was not sealed."
            )
        binding = _binding(request, context, authority)

        def apply() -> dict[str, object]:
            if selection is not None:
                # The writer owns admission through acceptance and execution.
                # Compare the actual pinned Index and User eligibility frame,
                # including tag changes that preserve the selected row count.
                # write_phase already excludes publication; a second bridge
                # hold would inherit the active lease into another task.
                with open_operation_read(context.archive_root) as pinned:
                    _validate_identity(request, context, pinned)
                    current = f"{pinned.archive.index_db_path.resolve()}:{archive_snapshot_epoch(pinned.archive)}"
                    if current != selection.frame:
                        raise MutationSelectionError(
                            "selection_frame_changed", "The mutation selection changed before acceptance."
                        )
                per_session = sum(
                    len(cast(list[str], request.payload.get(field) or []))
                    for field in ("add_marks", "remove_marks", "remove_tags")
                ) + (request.payload.get("note_text") is not None)
                chunks = max(1, (selection.session_count + MUTATION_PLAN_PAGE_SIZE - 1) // MUTATION_PLAN_PAGE_SIZE)
                total = selection.session_count * per_session + chunks * sum(
                    bool(request.payload.get(field)) for field in ("tags", "pairs")
                )
                if selection.session_count == 0 or total == 0:
                    return {
                        "outcome": "completed",
                        "sequence": 1,
                        "effect": "no-effect",
                        "session_count": 0,
                        "session_ids_sample": [],
                        "affected_count": 0,
                        "tag_count": 0,
                        "applied_count": 0,
                    }
                offset = 0
                pending: list[tuple[OperationBinding[Any, object], MutationPreview]] = []
                with (
                    _fenced_on_failure(audit, binding),
                    ArchiveStore.open_existing(context.archive_root, read_only=False) as archive,
                ):
                    instance = audit.ensure_archive_authority(now_ms=int(time() * 1000))
                    executor = OperationExecutor()

                    def accept() -> None:
                        nonlocal offset
                        if runtime.stop_reason(request) == "cancelled":
                            from polylogue.archive.query.execution_control import QueryCancelledError

                            raise QueryCancelledError("Mutation cancelled before all intents were sealed.")
                        previews = tuple(preview for _, preview in pending)
                        refs = audit.create_preview_batch(
                            tuple(preview.plan for preview in previews), context.principal
                        )
                        previews = tuple(
                            replace(preview, preview_ref=ref) for preview, ref in zip(previews, refs, strict=True)
                        )
                        authorizations = tuple(
                            executor.authorize_bound(operation, preview, context.principal)
                            for (operation, _), preview in zip(pending, previews, strict=True)
                        )
                        authorized = tuple(audit.issue_authorization_batch(previews, context.principal, authorizations))
                        with audit.bind_machine_request(
                            binding, transition="accept_execution_batch", page=(offset, offset + len(pending) == total)
                        ):
                            audit.accept_execution_batch(authorized, context.principal)
                        offset += len(pending)
                        pending.clear()

                    for actuator, args in _combined_user_intents(archive, document, request.payload):
                        operation = runtime_operation_binding(actuator)
                        raw = actuator.prepare(args)
                        preview = executor.prepare_bound(
                            operation,
                            args,
                            context.principal,
                            archive_instance_id=instance,
                            archive_identity_digest=authority.identity.authority_identity_digest,
                            parameter_digest=compute_parameter_digest(raw),
                            raw_plan=raw,
                        )
                        pending.append((operation, preview))
                        if len(pending) == MACHINE_PAGE_PARTS:
                            accept()
                    if pending:
                        accept()
                    if offset != total:
                        raise ValueError("sealed mutation part count differs from its complete intent")
            return _execute_batch(request, context, audit, authority, ())

        state = await runtime.write_phase("session.mark", apply)
    return operation_envelope(
        request, context, snapshot=authority, started_at=started, outcome=str(state["outcome"]), result=state
    )
