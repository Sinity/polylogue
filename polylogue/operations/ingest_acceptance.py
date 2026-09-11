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
    RecoveryDisposition,
    build_typed_plan,
)
from polylogue.storage.derived.session.derivation import SESSION_PROFILE_RECIPE_VERSION
from polylogue.storage.sqlite.archive_tiers.source_items import FrozenSourceManifest

INGEST_OPERATION = "ingest-archive-runtime"


def ingest_context(manifest: FrozenSourceManifest) -> dict[str, object]:
    return {
        "source_generation_id": manifest.source_generation_id,
        "manifest_digest": manifest.manifest_digest,
        "enumeration_fingerprint": manifest.enumeration_fingerprint,
        "input_count": len(manifest.inputs),
        "recipe_version": SESSION_PROFILE_RECIPE_VERSION,
    }


def ingest_plan(
    manifest: FrozenSourceManifest,
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
class IngestActuator:
    """Plan/inspection adapter; only the daemon's phased owner publishes."""

    manifest: FrozenSourceManifest
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

    def inspect_recovery(self, _operation: object, _args: object) -> RecoveryDisposition:
        return RecoveryDisposition("unknown", "operator-blocking", "ingest publication requires exact domain evidence")
