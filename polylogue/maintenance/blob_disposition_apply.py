"""Consume one accepted blob disposition plan under explicit authorization.

Two effects, in one order that cannot be reversed:

1. **Restore** every ``restore_required`` member into its ordinary spool
   through the production receiver that admission already reads. Restoration
   never touches the physical blob: the historical carrier survives this
   module unconditionally, so a crash at any boundary leaves at least one
   verified copy.
2. **Delete** ``source_present`` and ``superseded_prefix`` members through
   the canonical blob-GC seam, which owns publisher exclusion, the final
   locked liveness recheck, and crash-consistent generation intent.

The plan is a capability, not a worklist. This module makes no classification
judgment: every member's proof is revalidated immediately before its effect,
and any drift — a changed source, a changed object, a new referent, a
different digest, a different denominator — invalidates the whole plan and
returns control to compilation.

Deletion is bounded to unreferenced members by construction. A member whose
content is proven at its source but which a durable row still references
stays on disk; removing it is the reference owner's decision, and the GC seam
refuses it anyway.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from polylogue.maintenance.blob_disposition import (
    BlobDisposition,
    BlobDispositionContext,
    BlobDispositionMember,
    BlobDispositionPlan,
    RestorationDestination,
)

TOOL_VERSION = "blob-disposition-apply-v1"


class DispositionApplyError(RuntimeError):
    """Raised when an apply cannot prove its exact authorized effect set."""


class MemberOutcome(StrEnum):
    """One terminal outcome per plan member. There is no unknown outcome."""

    RESTORED = "restored"
    RESTORATION_ALREADY_PRESENT = "restoration_already_present"
    DELETED = "deleted"
    RETAINED_REFERENCED = "retained_referenced"
    RETAINED_ABSENT = "retained_absent"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class MemberResult:
    blob_hash: str
    outcome: MemberOutcome
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"blob_hash": self.blob_hash, "outcome": self.outcome.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class DispositionApplyReceipt:
    """Complete before/after evidence, derived only from member outcomes."""

    tool_version: str
    plan_digest: str
    archive_root: str
    blob_root: str
    dry_run: bool
    results: tuple[MemberResult, ...]
    reclaimed_bytes: int = 0
    blockers: tuple[str, ...] = ()

    @property
    def counts(self) -> dict[str, int]:
        counts = {outcome.value: 0 for outcome in MemberOutcome}
        for result in self.results:
            counts[result.outcome.value] += 1
        return counts

    @property
    def ok(self) -> bool:
        return not self.blockers and self.counts[MemberOutcome.BLOCKED.value] == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "tool_version": self.tool_version,
            "plan_digest": self.plan_digest,
            "archive_root": self.archive_root,
            "blob_root": self.blob_root,
            "dry_run": self.dry_run,
            "ok": self.ok,
            "counts": self.counts,
            "reclaimed_bytes": self.reclaimed_bytes,
            "blockers": list(self.blockers),
            "results": [result.to_dict() for result in self.results],
        }


def _revalidate(member: BlobDispositionMember, *, context: BlobDispositionContext) -> str | None:
    """Re-derive the member's own proof at the moment of effect."""
    path = context.blob_store.blob_path(member.blob_hash)
    if not path.is_file():
        return "physical object vanished between planning and apply"
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        return f"physical object became unreadable: {exc}"
    if size_bytes != member.size_bytes:
        return f"physical object changed size {member.size_bytes} -> {size_bytes}"
    referenced_now = member.blob_hash in context.referenced_hashes
    if referenced_now and not member.referenced:
        return "a new durable reference appeared after planning"
    if member.disposition is BlobDisposition.RESTORE_REQUIRED:
        for prover in context.provers:
            if prover.prove(member.blob_hash, path, size_bytes) is not None:
                return "a source proof appeared after planning; restoration is no longer justified"
        return None
    expected = member.proof
    if expected is None:
        return "member carries no proof to revalidate"
    for prover in context.provers:
        if prover.name != expected.prover:
            continue
        proof = prover.prove(member.blob_hash, path, size_bytes)
        if proof is None:
            return f"{expected.prover} no longer proves this object at its source"
        if proof.source_path != expected.source_path or proof.mode is not expected.mode:
            return f"{expected.prover} now proves a different source or mode"
        return None
    return f"prover {expected.prover} is not available at apply time"


def _resident_hook_event(spool_root: Path, event_id: str) -> Path | None:
    """Locate an event anywhere in the spool, not only in today's shard.

    ``enqueue_hook_event`` shards by the current day and only refuses a
    collision inside that shard, so a same-identity event spooled on another
    day would be delivered twice.
    """
    for candidate in sorted(spool_root.rglob(f"{event_id}.json")):
        if candidate.is_file():
            return candidate
    return None


def _restore_hook_event(member: BlobDispositionMember, *, path: Path, spool_root: Path) -> MemberResult:
    from polylogue.sources.hooks import (
        HookSpoolRecordError,
        enqueue_hook_event,
        read_hook_spool_record,
    )

    try:
        envelope = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        return MemberResult(member.blob_hash, MemberOutcome.BLOCKED, f"carrier is not a readable envelope: {exc}")
    if not isinstance(envelope, dict):
        return MemberResult(member.blob_hash, MemberOutcome.BLOCKED, "carrier envelope is not an object")
    event_id = envelope.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        return MemberResult(member.blob_hash, MemberOutcome.BLOCKED, "carrier envelope has no event identity")
    resident = _resident_hook_event(spool_root, event_id)
    if resident is not None:
        try:
            existing = read_hook_spool_record(resident)
        except HookSpoolRecordError as exc:
            return MemberResult(member.blob_hash, MemberOutcome.BLOCKED, f"destination is unreadable: {exc}")
        if existing != envelope:
            return MemberResult(
                member.blob_hash,
                MemberOutcome.BLOCKED,
                "destination holds a different event under the same identity",
            )
        return MemberResult(member.blob_hash, MemberOutcome.RESTORATION_ALREADY_PRESENT, str(resident))
    try:
        published = enqueue_hook_event(
            event_type=str(envelope["event_type"]),
            session_id=str(envelope["session_id"]),
            provider=str(envelope["provider"]),
            timestamp=str(envelope["timestamp"]),
            payload=dict(envelope["payload"]),
            root=spool_root,
            event_id=str(envelope["event_id"]),
        )
    except (KeyError, TypeError, HookSpoolRecordError, OSError) as exc:
        return MemberResult(member.blob_hash, MemberOutcome.BLOCKED, f"ordinary spool admission refused: {exc}")
    try:
        restored = read_hook_spool_record(published)
    except HookSpoolRecordError as exc:
        return MemberResult(member.blob_hash, MemberOutcome.BLOCKED, f"restored file does not read back: {exc}")
    if restored != envelope:
        return MemberResult(
            member.blob_hash,
            MemberOutcome.BLOCKED,
            "destination holds a different event under the same identity",
        )
    _fsync_directory(published.parent)
    return MemberResult(member.blob_hash, MemberOutcome.RESTORED, str(published))


def _restore_browser_capture(member: BlobDispositionMember, *, path: Path, spool_root: Path) -> MemberResult:
    from pydantic import ValidationError

    from polylogue.browser_capture.models import BrowserCaptureEnvelope
    from polylogue.browser_capture.receiver import (
        BrowserCaptureSpoolConflictError,
        SpoolQuotaExceededError,
        capture_artifact_path,
        capture_dedup_content_hash,
        write_capture_envelope_bytes,
    )

    try:
        raw = path.read_bytes()
        envelope = BrowserCaptureEnvelope.model_validate_json(raw)
    except (OSError, ValidationError, ValueError) as exc:
        return MemberResult(member.blob_hash, MemberOutcome.BLOCKED, f"carrier is not a valid capture: {exc}")
    destination = capture_artifact_path(envelope, spool_root)
    if destination.is_file():
        try:
            existing = BrowserCaptureEnvelope.model_validate_json(destination.read_bytes())
        except (OSError, ValidationError, ValueError) as exc:
            return MemberResult(member.blob_hash, MemberOutcome.BLOCKED, f"destination is unreadable: {exc}")
        if capture_dedup_content_hash(existing) != capture_dedup_content_hash(envelope):
            return MemberResult(
                member.blob_hash,
                MemberOutcome.BLOCKED,
                "destination holds a different capture under the same identity",
            )
        return MemberResult(member.blob_hash, MemberOutcome.RESTORATION_ALREADY_PRESENT, str(destination))
    try:
        write_capture_envelope_bytes(raw, spool_path=spool_root)
    except (BrowserCaptureSpoolConflictError, SpoolQuotaExceededError, OSError, ValueError) as exc:
        return MemberResult(member.blob_hash, MemberOutcome.BLOCKED, f"ordinary spool admission refused: {exc}")
    if not destination.is_file():
        return MemberResult(member.blob_hash, MemberOutcome.BLOCKED, "capture receiver published no artifact")
    try:
        restored = BrowserCaptureEnvelope.model_validate_json(destination.read_bytes())
    except (OSError, ValidationError, ValueError) as exc:
        return MemberResult(member.blob_hash, MemberOutcome.BLOCKED, f"restored file does not read back: {exc}")
    if capture_dedup_content_hash(restored) != capture_dedup_content_hash(envelope):
        return MemberResult(member.blob_hash, MemberOutcome.BLOCKED, "restored capture is not content-equivalent")
    _fsync_directory(destination.parent)
    return MemberResult(member.blob_hash, MemberOutcome.RESTORED, str(destination))


def _fsync_directory(path: Path) -> None:
    """Persist the atomic rename's directory entry before claiming success."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def restore_plan_members(
    plan: BlobDispositionPlan,
    *,
    context: BlobDispositionContext,
    hook_spool_root: Path,
    browser_capture_spool: Path,
    dry_run: bool = True,
) -> tuple[MemberResult, ...]:
    """Restore every sole-copy carrier into its ordinary spool.

    This never deletes or modifies the historical carrier, so an interruption
    at any point leaves the blob intact and the operation resumable.
    """
    results: list[MemberResult] = []
    for member in plan.members_for(BlobDisposition.RESTORE_REQUIRED):
        drift = _revalidate(member, context=context)
        if drift is not None:
            results.append(MemberResult(member.blob_hash, MemberOutcome.BLOCKED, drift))
            continue
        if member.restoration is None:
            results.append(
                MemberResult(member.blob_hash, MemberOutcome.BLOCKED, "restore_required member names no destination")
            )
            continue
        if dry_run:
            results.append(
                MemberResult(
                    member.blob_hash,
                    MemberOutcome.RESTORED,
                    f"would restore to {member.restoration.destination.value}",
                )
            )
            continue
        path = context.blob_store.blob_path(member.blob_hash)
        if member.restoration.destination is RestorationDestination.HOOK_EVENT_SPOOL:
            results.append(_restore_hook_event(member, path=path, spool_root=hook_spool_root))
        else:
            results.append(_restore_browser_capture(member, path=path, spool_root=browser_capture_spool))
    return tuple(results)


def _authorization_blockers(
    plan: BlobDispositionPlan,
    *,
    authorized_digest: str,
    context: BlobDispositionContext,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if not plan.accepted:
        blockers.append(f"plan is not acceptable: {plan.unresolved_count} unresolved members")
    actual = plan.digest()
    if actual != authorized_digest:
        blockers.append(f"authorized digest {authorized_digest[:16]} does not match plan digest {actual[:16]}")
    if str(context.blob_store.root) != plan.blob_root:
        blockers.append(f"plan blob namespace {plan.blob_root} is not the namespace being applied")
    referenced_present = len(context.referenced_hashes & {member.blob_hash for member in plan.members})
    if referenced_present != plan.denominator.referenced_present_count:
        blockers.append(
            "referenced-and-present denominator drifted "
            f"{plan.denominator.referenced_present_count} -> {referenced_present}"
        )
    return tuple(blockers)


def apply_disposition_plan(
    plan: BlobDispositionPlan,
    *,
    context: BlobDispositionContext,
    authorized_digest: str,
    source_db: Path,
    index_db: Path,
    hook_spool_root: Path,
    browser_capture_spool: Path,
    writer_block_reason: str | None = None,
    dry_run: bool = True,
) -> DispositionApplyReceipt:
    """Restore, then delete, exactly what the authorized plan names."""
    blockers = list(_authorization_blockers(plan, authorized_digest=authorized_digest, context=context))
    if writer_block_reason is not None and not dry_run:
        blockers.append(f"an archive writer is active: {writer_block_reason}")
    if blockers:
        return DispositionApplyReceipt(
            tool_version=TOOL_VERSION,
            plan_digest=plan.digest(),
            archive_root=plan.archive_root,
            blob_root=plan.blob_root,
            dry_run=dry_run,
            results=(),
            blockers=tuple(blockers),
        )

    results: list[MemberResult] = list(
        restore_plan_members(
            plan,
            context=context,
            hook_spool_root=hook_spool_root,
            browser_capture_spool=browser_capture_spool,
            dry_run=dry_run,
        )
    )
    if any(result.outcome is MemberOutcome.BLOCKED for result in results):
        return DispositionApplyReceipt(
            tool_version=TOOL_VERSION,
            plan_digest=plan.digest(),
            archive_root=plan.archive_root,
            blob_root=plan.blob_root,
            dry_run=dry_run,
            results=tuple(results),
            blockers=("restoration did not complete; no deletion was attempted",),
        )

    removable: list[BlobDispositionMember] = []
    for disposition in (BlobDisposition.SOURCE_PRESENT, BlobDisposition.SUPERSEDED_PREFIX):
        for member in plan.members_for(disposition):
            drift = _revalidate(member, context=context)
            if drift is not None:
                results.append(MemberResult(member.blob_hash, MemberOutcome.BLOCKED, drift))
                continue
            if member.referenced or member.blob_hash in context.referenced_hashes:
                results.append(
                    MemberResult(
                        member.blob_hash,
                        MemberOutcome.RETAINED_REFERENCED,
                        "content is proven at its source but a durable row still references the object",
                    )
                )
                continue
            removable.append(member)

    if any(result.outcome is MemberOutcome.BLOCKED for result in results):
        return DispositionApplyReceipt(
            tool_version=TOOL_VERSION,
            plan_digest=plan.digest(),
            archive_root=plan.archive_root,
            blob_root=plan.blob_root,
            dry_run=dry_run,
            results=tuple(results),
            blockers=("member revalidation failed; no deletion was attempted",),
        )

    # An empty removable set has no effect to serialize: entering the GC seam
    # would only report its own unmet preconditions as this plan's blockers.
    if dry_run or not removable:
        results.extend(
            MemberResult(member.blob_hash, MemberOutcome.DELETED, "would unlink through the blob-GC seam")
            for member in removable
        )
        return DispositionApplyReceipt(
            tool_version=TOOL_VERSION,
            plan_digest=plan.digest(),
            archive_root=plan.archive_root,
            blob_root=plan.blob_root,
            dry_run=dry_run,
            results=tuple(results),
            reclaimed_bytes=sum(member.size_bytes for member in removable) if dry_run else 0,
        )

    from polylogue.storage.blob_gc import unlink_unreferenced_blob_hashes_under_exclusion

    deleted, reclaimed, errors = unlink_unreferenced_blob_hashes_under_exclusion(
        source_db,
        index_db,
        context.blob_store.root,
        {member.blob_hash for member in removable},
    )
    for member in removable:
        if context.blob_store.blob_path(member.blob_hash).exists():
            results.append(
                MemberResult(member.blob_hash, MemberOutcome.RETAINED_ABSENT, "the GC seam declined this member")
            )
        else:
            results.append(MemberResult(member.blob_hash, MemberOutcome.DELETED, ""))
    return DispositionApplyReceipt(
        tool_version=TOOL_VERSION,
        plan_digest=plan.digest(),
        archive_root=plan.archive_root,
        blob_root=plan.blob_root,
        dry_run=False,
        results=tuple(results),
        reclaimed_bytes=reclaimed,
        blockers=tuple(errors),
    )


def write_receipt(path: Path, receipt: DispositionApplyReceipt) -> None:
    """Publish an append-only receipt, durably, before returning success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    payload = json.dumps(receipt.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


__all__ = [
    "TOOL_VERSION",
    "DispositionApplyError",
    "DispositionApplyReceipt",
    "MemberOutcome",
    "MemberResult",
    "apply_disposition_plan",
    "restore_plan_members",
    "write_receipt",
]
