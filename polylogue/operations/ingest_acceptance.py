"""Closed authority for one daemon-owned retained source generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from polylogue.operations.mutation_transaction import (
    ConfirmationStrength,
    DestructiveClass,
    MutationPlan,
    MutationReceipt,
    MutationTarget,
    RecoveryResolution,
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
    manifest: FrozenSourceManifest | SealedSourceManifestRef,
    *,
    archive_instance_id: str,
    archive_identity_digest: str,
    now_ms: int,
    expires_at_ms: int,
) -> MutationPlan:
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


@dataclass(frozen=True, slots=True)
class IngestRecovery:
    """Recovery route for an interrupted ingest, which is not re-driven here.

    Startup recovery cannot run the daemon's phased ingest driver, and a
    resumed request for an accepted generation without a terminal checkpoint
    reports its indeterminate state rather than replaying it. The run is
    terminalized so it is not a barrier; the accepted generation itself stays
    in ``source.db`` for an owner that re-drives it (polylogue-x7u3x).
    """

    operation: str = INGEST_OPERATION

    def recover(self, _handles: ReplayHandles, _plan: MutationPlan) -> RecoveryResolution:
        return RecoveryResolution(
            "not-replayable", "the accepted source generation is retained but not re-driven by startup recovery"
        )


@dataclass(frozen=True, slots=True)
class IngestActuator:
    """Plan/inspection adapter; only the daemon's phased owner publishes."""

    manifest: FrozenSourceManifest | SealedSourceManifestRef
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
