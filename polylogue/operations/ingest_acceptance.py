"""Closed authority for one daemon-owned retained source generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from polylogue.operations.mutation_transaction import (
    ConfirmationStrength,
    DestructiveClass,
    MutationPlan,
    MutationReceipt,
    MutationTarget,
    RecoveryRedrivenByOwnerError,
    RecoveryResolution,
    RecoverySettledIndeterminateError,
    ReplayHandles,
    build_typed_plan,
    register_recovery_route,
)
from polylogue.storage.derived.session.derivation import SESSION_PROFILE_RECIPE_VERSION
from polylogue.storage.sqlite.archive_tiers.source_items import FrozenSourceManifest, SealedSourceManifestRef

INGEST_OPERATION = "ingest-archive-runtime"


def ingest_context(manifest: FrozenSourceManifest | SealedSourceManifestRef) -> dict[str, object]:
    context: dict[str, object] = {
        "source_generation_id": manifest.source_generation_id,
        "manifest_digest": manifest.manifest_digest,
        "enumeration_fingerprint": manifest.enumeration_fingerprint,
        "input_count": manifest.input_count if isinstance(manifest, SealedSourceManifestRef) else len(manifest.inputs),
        "recipe_version": SESSION_PROFILE_RECIPE_VERSION,
    }
    if isinstance(manifest, SealedSourceManifestRef):
        context["custody_digest"] = manifest.custody_digest
    if manifest.source_name is not None:
        context["source_name"] = manifest.source_name
    return context


def ingest_plan(
    manifest: SealedSourceManifestRef,
    *,
    archive_instance_id: str,
    archive_identity_digest: str,
    now_ms: int,
    expires_at_ms: int,
) -> MutationPlan:
    if not isinstance(manifest, SealedSourceManifestRef):
        raise TypeError("new ingest plans require a staged source manifest")
    context = ingest_context(manifest)
    digest = hashlib.sha256(manifest.manifest_digest.encode()).hexdigest()
    target = MutationTarget(
        kind="source",
        ref=f"source:{manifest.source_generation_id}",
        policy_key="ingest-generation",
        identity_digest=digest,
        effect_identity=f"{INGEST_OPERATION}:{manifest.source_generation_id}:{manifest.manifest_digest}",
        durability="durable",
        recovery="reconcile_required",
    )
    return build_typed_plan(
        operation=INGEST_OPERATION,
        operation_version=1,
        archive_instance_id=archive_instance_id,
        archive_identity_digest=archive_identity_digest,
        targets=(target,),
        affected_tiers=("source", "index", "user"),
        parameter_digest=digest,
        required_capabilities=("archive.ingest",),
        destructive_class="additive",
        required_confirmation="role_only",
        prepared_at_ms=now_ms,
        expires_at_ms=expires_at_ms,
        context=context,
    )


def generation_materialized(archive_root: Path, source_generation_id: str) -> bool:
    """Whether any raw of this accepted generation was already materialized (parsed)."""
    from contextlib import closing

    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    with closing(open_readonly_connection(archive_root / "source.db", validate_schema=False)) as conn:
        return (
            conn.execute(
                "SELECT 1 FROM source_item_raw_members AS m JOIN raw_sessions AS r ON r.raw_id = m.raw_id "
                "WHERE m.source_generation_id = ? AND r.parsed_at_ms IS NOT NULL LIMIT 1",
                (source_generation_id,),
            ).fetchone()
            is not None
        )


@dataclass(frozen=True, slots=True)
class IngestRecovery:
    """Recovery route for an interrupted ingest.

    An accepted generation whose request was never stopped belongs to the
    daemon's ingest owner, which re-drives it from the retained manifest to
    the original request's terminal receipt
    (``polylogue.operations.daemon_ingest.redrive_accepted_ingests``). Generic
    recovery leaves that run to the owner. A request that was stopped
    (cancelled, past its deadline, or refused) has its outcome already
    decided: with no generation content materialized its run is
    terminalized as not replayable; with content already published its effect
    is partial, so the run keeps its indeterminate state instead of being
    rewritten as failed with no effect.
    """

    operation: str = INGEST_OPERATION

    def recover(self, handles: ReplayHandles, plan: MutationPlan) -> RecoveryResolution:
        from polylogue.operations.audit import AuditRepository

        generation_id = plan.context.get("source_generation_id")
        if not isinstance(generation_id, str):
            return RecoveryResolution("not-replayable", "the ingest plan names no accepted source generation")
        audit = AuditRepository.for_archive_root(handles.archive_root)
        with audit.settled_machine_read():
            accepted, stop_reason = audit.accepted_ingest_stop_reason(generation_id)
        if not accepted:
            return RecoveryResolution("not-replayable", "no accepted ingest request binds this source generation")
        if stop_reason is None:
            raise RecoveryRedrivenByOwnerError(f"source generation {generation_id} awaits its ingest owner")
        if generation_materialized(handles.archive_root, generation_id):
            raise RecoverySettledIndeterminateError(
                f"source generation {generation_id} was stopped ({stop_reason}) after publishing content"
            )
        return RecoveryResolution(
            "not-replayable", f"the ingest request was stopped ({stop_reason}) before its terminal checkpoint"
        )


@dataclass(frozen=True, slots=True)
class IngestActuator:
    """Plan/inspection adapter; only the daemon's phased owner publishes."""

    manifest: SealedSourceManifestRef
    archive_instance_id: str
    archive_identity_digest: str
    now_ms: int
    expires_at_ms: int
    operation: str = INGEST_OPERATION
    destructive_class: DestructiveClass = "additive"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, _args: object) -> MutationPlan:
        return ingest_plan(
            self.manifest,
            archive_instance_id=self.archive_instance_id,
            archive_identity_digest=self.archive_identity_digest,
            now_ms=self.now_ms,
            expires_at_ms=self.expires_at_ms,
        )

    def apply(self, _plan: MutationPlan, _args: object) -> MutationReceipt:
        raise RuntimeError("ingest runtime publication is owned by the daemon phased driver")

    def recover(self, handles: ReplayHandles, plan: MutationPlan) -> RecoveryResolution:
        return IngestRecovery().recover(handles, plan)


register_recovery_route(IngestRecovery())
