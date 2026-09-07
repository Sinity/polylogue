"""Consume one accepted blob disposition plan under explicit authorization.

Two effects, in one order that cannot lose material:

1. **Restore** every ``restore_required`` member into its ordinary spool
   through the production receiver that admission already reads, and read the
   published bytes back before calling the material resident. Restoration
   never touches the physical blob, so a crash at any boundary leaves at least
   one verified copy.
2. **Delete** every member no durable row references, through the canonical
   blob-GC seam, which owns publisher exclusion, the final locked liveness
   recheck, and crash-consistent generation intent.

The disposition selects nothing here: being unreferenced is the whole
criterion, and it is exactly the criterion recurring GC applies. A proven
object and an unexplained orphan are the same deletion once no row names
either. What a disposition still decides is restoration — a ``restore_required``
member is the only verified carrier of wanted material, so its material must
be resident in an ordinary spool before it is eligible at all.

A member a durable row references is never deleted, whatever its disposition,
and the plan's ``unresolved`` count gates nothing: an object still unexplained
that something references is what the namespace is being whittled down to.

The plan is a capability, not a worklist. Its members are the population; the
reference set and the GC seam's own locked recheck decide the effect.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from polylogue.maintenance.blob_disposition import (
    BLOB_REFERENCE_RELATIONS,
    BlobDisposition,
    BlobDispositionContext,
    BlobDispositionMember,
    BlobDispositionPlan,
    RestorationDestination,
)
from polylogue.storage.blob_store import BlobNamespaceEntryKind, BlobNamespaceIssue, BlobStore

if TYPE_CHECKING:
    from polylogue.browser_capture.models import BrowserCaptureEnvelope
    from polylogue.browser_capture.receiver import CaptureConvergence, SpoolUsage

TOOL_VERSION = "blob-disposition-apply-v3"

# The cohort label for a namespace entry that is not a blob at all: a SQLite
# sibling stranded beside a content-addressed object, a crash-left staging
# file, anything the store's own walk refuses to convert into a hash. It has
# no plan member, no hash and no reference.
INVALID_ENTRY_COHORT = "invalid_namespace_entry"

_ISSUE_LABELS = frozenset({issue.value for issue in BlobNamespaceIssue} | {"unclassified"})


class DispositionApplyError(RuntimeError):
    """Raised when an apply cannot prove its exact authorized effect set."""


class RestorationOutcome(StrEnum):
    """What became of one sole-copy carrier's material."""

    RESTORED = "restored"
    RESTORATION_ALREADY_PRESENT = "restoration_already_present"
    RESTORATION_SUPERSEDED = "restoration_superseded"
    BLOCKED = "blocked"


class MemberOutcome(StrEnum):
    """One terminal outcome per member. There is no unknown outcome."""

    DELETED = "deleted"
    RETAINED_REFERENCED = "retained_referenced"
    RETAINED_ABSENT = "retained_absent"
    BLOCKED = "blocked"


# The material is in an ordinary spool under these outcomes and only these.
# A superseded revision is among them: one provider session keeps exactly one
# spool artifact, and the revision the spool converged on is the material the
# archive wants. A carrier the spool declines as older is accounted for by the
# newer artifact holding that identity, not stranded.
_COMPLETED_RESTORATIONS = frozenset(
    {
        RestorationOutcome.RESTORED,
        RestorationOutcome.RESTORATION_ALREADY_PRESENT,
        RestorationOutcome.RESTORATION_SUPERSEDED,
    }
)
# Outcomes that leave the object physically in the namespace.
_RETAINED_IN_NAMESPACE = frozenset({MemberOutcome.RETAINED_REFERENCED, MemberOutcome.BLOCKED})


@dataclass(frozen=True, slots=True)
class RestorationResult:
    """One carrier's material, and where an ordinary spool now holds it."""

    blob_hash: str
    outcome: RestorationOutcome
    detail: str = ""
    destination: str = ""
    spool_path: str = ""

    @property
    def completed(self) -> bool:
        return self.outcome in _COMPLETED_RESTORATIONS

    def to_dict(self) -> dict[str, str]:
        return {
            "blob_hash": self.blob_hash,
            "outcome": self.outcome.value,
            "detail": self.detail,
            "destination": self.destination,
            "spool_path": self.spool_path,
        }


@dataclass(frozen=True, slots=True)
class MemberResult:
    """One member's terminal outcome and the evidence that produced it."""

    blob_hash: str
    outcome: MemberOutcome
    detail: str = ""
    cohort: str = ""
    referenced: bool = False
    size_bytes: int = 0
    from_path: str = ""
    restoration_outcome: str = ""
    restored_to: str = ""

    @property
    def is_blob(self) -> bool:
        return self.cohort != INVALID_ENTRY_COHORT

    def to_dict(self) -> dict[str, object]:
        return {
            "blob_hash": self.blob_hash,
            "outcome": self.outcome.value,
            "detail": self.detail,
            "cohort": self.cohort,
            "referenced": self.referenced,
            "size_bytes": self.size_bytes,
            "from_path": self.from_path,
            "restoration_outcome": self.restoration_outcome,
            "restored_to": self.restored_to,
        }


@dataclass(frozen=True, slots=True)
class NamespaceTotals:
    """What the physical namespace held when it was measured."""

    blob_count: int = 0
    blob_bytes: int = 0
    invalid_entry_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "blob_count": self.blob_count,
            "blob_bytes": self.blob_bytes,
            "invalid_entry_count": self.invalid_entry_count,
        }


def measure_namespace(store: BlobStore) -> NamespaceTotals:
    """Count and size the physical namespace through the store's own walk."""
    blob_count = 0
    blob_bytes = 0
    invalid_entry_count = 0
    for entry in store.iter_namespace():
        if entry.kind is not BlobNamespaceEntryKind.BLOB:
            invalid_entry_count += 1
            continue
        blob_count += 1
        try:
            blob_bytes += entry.path.stat().st_size
        except OSError:
            continue
    return NamespaceTotals(blob_count=blob_count, blob_bytes=blob_bytes, invalid_entry_count=invalid_entry_count)


@dataclass(frozen=True, slots=True)
class DispositionApplyReceipt:
    """Complete before/after evidence, derived only from member outcomes."""

    tool_version: str
    plan_digest: str
    archive_root: str
    blob_root: str
    dry_run: bool
    results: tuple[MemberResult, ...]
    restorations: tuple[RestorationResult, ...] = ()
    namespace_before: NamespaceTotals = field(default_factory=NamespaceTotals)
    blockers: tuple[str, ...] = ()

    @property
    def counts(self) -> dict[str, int]:
        counts = {outcome.value: 0 for outcome in MemberOutcome}
        for result in self.results:
            counts[result.outcome.value] += 1
        return counts

    @property
    def restoration_counts(self) -> dict[str, int]:
        counts = {outcome.value: 0 for outcome in RestorationOutcome}
        for restoration in self.restorations:
            counts[restoration.outcome.value] += 1
        return counts

    @property
    def cohorts(self) -> dict[str, dict[str, int]]:
        """Everything deleted and everything left, per cohort.

        Cohorts are the plan's own dispositions plus the invalid namespace
        entries, so a reader sees which population each outcome came from —
        including how much of the unresolved residue is still on disk —
        without re-deriving it from the member list.
        """
        cohorts: dict[str, dict[str, int]] = {}
        for result in self.results:
            cohort = cohorts.setdefault(
                result.cohort or "unknown",
                {"members": 0, "bytes": 0, "deleted_bytes": 0, "retained_bytes": 0}
                | {outcome.value: 0 for outcome in MemberOutcome},
            )
            cohort["members"] += 1
            cohort["bytes"] += result.size_bytes
            cohort[result.outcome.value] += 1
            if result.outcome is MemberOutcome.DELETED:
                cohort["deleted_bytes"] += result.size_bytes
            elif result.outcome in _RETAINED_IN_NAMESPACE:
                cohort["retained_bytes"] += result.size_bytes
        return cohorts

    @property
    def deleted_count(self) -> int:
        return sum(1 for result in self.results if result.is_blob and result.outcome is MemberOutcome.DELETED)

    @property
    def deleted_bytes(self) -> int:
        return sum(
            result.size_bytes for result in self.results if result.is_blob and result.outcome is MemberOutcome.DELETED
        )

    @property
    def invalid_entries_deleted(self) -> int:
        return sum(1 for result in self.results if not result.is_blob and result.outcome is MemberOutcome.DELETED)

    @property
    def invalid_entry_bytes_deleted(self) -> int:
        return sum(
            result.size_bytes
            for result in self.results
            if not result.is_blob and result.outcome is MemberOutcome.DELETED
        )

    @property
    def namespace_after(self) -> NamespaceTotals:
        """What the namespace holds once this run's deletions are accounted for.

        Derived from the member outcomes rather than measured a second time,
        so a dry rehearsal reports exactly the totals its active twin would.
        """
        return NamespaceTotals(
            blob_count=self.namespace_before.blob_count - self.deleted_count,
            blob_bytes=self.namespace_before.blob_bytes - self.deleted_bytes,
            invalid_entry_count=self.namespace_before.invalid_entry_count - self.invalid_entries_deleted,
        )

    @property
    def ok(self) -> bool:
        return (
            not self.blockers
            and self.counts[MemberOutcome.BLOCKED.value] == 0
            and self.restoration_counts[RestorationOutcome.BLOCKED.value] == 0
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "tool_version": self.tool_version,
            "plan_digest": self.plan_digest,
            "archive_root": self.archive_root,
            "blob_root": self.blob_root,
            "dry_run": self.dry_run,
            "ok": self.ok,
            "counts": self.counts,
            "cohorts": self.cohorts,
            "totals": {
                "before": self.namespace_before.to_dict(),
                "after": self.namespace_after.to_dict(),
                "deleted_count": self.deleted_count,
                "deleted_bytes": self.deleted_bytes,
                "invalid_entries_deleted": self.invalid_entries_deleted,
                "invalid_entry_bytes_deleted": self.invalid_entry_bytes_deleted,
            },
            # What ``referenced: false`` on a deleted member was decided
            # against, named so the receipt explains its own deletions.
            "reference_relations": [f"{table}.{column}" for table, column in BLOB_REFERENCE_RELATIONS],
            "restorations": {
                "counts": self.restoration_counts,
                "members": [restoration.to_dict() for restoration in self.restorations],
            },
            "blockers": list(self.blockers),
            "results": [result.to_dict() for result in self.results],
        }


def _carrier_drift(member: BlobDispositionMember, *, context: BlobDispositionContext) -> str | None:
    """Confirm the physical carrier is still the object the plan described."""
    path = context.blob_store.blob_path(member.blob_hash)
    if not path.is_file():
        return "physical object vanished between planning and apply"
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        return f"physical object became unreadable: {exc}"
    if size_bytes != member.size_bytes:
        return f"physical object changed size {member.size_bytes} -> {size_bytes}"
    return None


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


@dataclass(slots=True)
class _RehearsedSpools:
    """What the spools would hold once a rehearsal's own restorations landed.

    A rehearsal writes nothing, so material an earlier member of the same run
    would have published is invisible on disk. Carrying it here is what makes
    a rehearsal's counts the counts its apply produces: 584 session identities
    across 880 carriers converge on 584 artifacts either way.
    """

    hook_events: dict[str, tuple[dict[str, object], Path | None]] = field(default_factory=dict)
    captures: dict[Path, BrowserCaptureEnvelope] = field(default_factory=dict)
    _usage: SpoolUsage | None = None
    _added_files: int = 0
    _added_bytes: int = 0

    def hook_event(self, spool_root: Path, event_id: str) -> tuple[dict[str, object] | None, Path | None, str | None]:
        """Return the event this identity finds resident, where, and any read failure."""
        from polylogue.sources.hooks import HookSpoolRecordError, read_hook_spool_record

        rehearsed = self.hook_events.get(event_id)
        if rehearsed is not None:
            return rehearsed[0], rehearsed[1], None
        resident = _resident_hook_event(spool_root, event_id)
        if resident is None:
            return None, None, None
        try:
            return read_hook_spool_record(resident), resident, None
        except HookSpoolRecordError as exc:
            return None, resident, f"destination is unreadable: {exc}"

    def record_hook_event(self, event_id: str, envelope: dict[str, object], *, path: Path | None = None) -> None:
        self.hook_events[event_id] = (envelope, path)

    def capture(self, target: Path) -> tuple[BrowserCaptureEnvelope | None, str | None]:
        """Return the capture this artifact name would find resident, and any read failure."""
        from pydantic import ValidationError

        from polylogue.browser_capture.models import BrowserCaptureEnvelope

        rehearsed = self.captures.get(target)
        if rehearsed is not None:
            return rehearsed, None
        if not target.is_file():
            return None, None
        try:
            return BrowserCaptureEnvelope.model_validate_json(target.read_bytes()), None
        except (OSError, ValidationError, ValueError):
            return None, f"existing capture artifact is unreadable or malformed: {target.name}"

    def record_capture(self, target: Path, envelope: BrowserCaptureEnvelope, *, size_bytes: int, new: bool) -> None:
        self.captures[target] = envelope
        if new:
            self._added_files += 1
            self._added_bytes += size_bytes

    def capture_quota_refusal(self, spool_root: Path) -> str | None:
        """Report the quota the receiver would refuse a new artifact against.

        The receiver checks the quota only for an artifact it is about to
        create, and every creation this run rehearses counts toward the next
        one's check.
        """
        from polylogue.browser_capture.receiver import SPOOL_MAX_BYTES, SPOOL_MAX_FILES, spool_usage

        if self._usage is None:
            self._usage = spool_usage(spool_root)
        file_count = self._usage.file_count + self._added_files
        total_bytes = self._usage.total_bytes + self._added_bytes
        if file_count >= SPOOL_MAX_FILES or total_bytes >= SPOOL_MAX_BYTES:
            return (
                f"capture spool quota exceeded: {file_count} files, {total_bytes} bytes "
                f"(limits: {SPOOL_MAX_FILES} files, {SPOOL_MAX_BYTES} bytes)"
            )
        return None


def _hook_enqueue_arguments(envelope: dict[str, object]) -> tuple[dict[str, object] | None, str | None]:
    """Return the admission call this envelope makes, or why it cannot make one."""
    arguments: dict[str, object] = {}
    for name in ("event_type", "session_id", "provider", "timestamp", "event_id"):
        value = envelope.get(name)
        if not isinstance(value, str) or not value:
            return None, f"carrier envelope has no {name}"
        arguments[name] = value
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        return None, "carrier envelope has no payload object"
    arguments["payload"] = dict(payload)
    return arguments, None


def _restore_hook_event(
    member: BlobDispositionMember,
    *,
    path: Path,
    spool_root: Path,
    rehearsed: _RehearsedSpools,
    dry_run: bool,
) -> RestorationResult:
    from polylogue.sources.hooks import (
        HookSpoolRecordError,
        enqueue_hook_event,
        read_hook_spool_record,
    )

    destination = RestorationDestination.HOOK_EVENT_SPOOL.value
    try:
        envelope = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        return RestorationResult(
            member.blob_hash, RestorationOutcome.BLOCKED, f"carrier is not a readable envelope: {exc}", destination
        )
    if not isinstance(envelope, dict):
        return RestorationResult(
            member.blob_hash, RestorationOutcome.BLOCKED, "carrier envelope is not an object", destination
        )
    event_id = envelope.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        return RestorationResult(
            member.blob_hash, RestorationOutcome.BLOCKED, "carrier envelope has no event identity", destination
        )
    resident, resident_path, unreadable = rehearsed.hook_event(spool_root, event_id)
    if unreadable is not None:
        return RestorationResult(member.blob_hash, RestorationOutcome.BLOCKED, unreadable, destination)
    if resident is not None:
        if resident != envelope:
            return RestorationResult(
                member.blob_hash,
                RestorationOutcome.BLOCKED,
                "destination holds a different event under the same identity",
                destination,
            )
        return RestorationResult(
            member.blob_hash,
            RestorationOutcome.RESTORATION_ALREADY_PRESENT,
            "the ordinary spool already holds this event",
            destination,
            "" if resident_path is None else str(resident_path),
        )
    arguments, refusal = _hook_enqueue_arguments(envelope)
    if arguments is None:
        return RestorationResult(
            member.blob_hash, RestorationOutcome.BLOCKED, f"ordinary spool admission refused: {refusal}", destination
        )
    if dry_run:
        rehearsed.record_hook_event(event_id, envelope)
        return RestorationResult(
            member.blob_hash, RestorationOutcome.RESTORED, f"would restore to {destination}", destination
        )
    try:
        published = enqueue_hook_event(root=spool_root, **arguments)  # type: ignore[arg-type]
    except (KeyError, TypeError, HookSpoolRecordError, OSError) as exc:
        return RestorationResult(
            member.blob_hash, RestorationOutcome.BLOCKED, f"ordinary spool admission refused: {exc}", destination
        )
    # Acquisition derives fields the spool file does not carry and both sides
    # serialize independently, so the published carrier is verified by the
    # production read route rather than by its bytes.
    try:
        restored = read_hook_spool_record(published)
    except HookSpoolRecordError as exc:
        return RestorationResult(
            member.blob_hash, RestorationOutcome.BLOCKED, f"restored file does not read back: {exc}", destination
        )
    if restored != envelope:
        return RestorationResult(
            member.blob_hash,
            RestorationOutcome.BLOCKED,
            "destination holds a different event under the same identity",
            destination,
        )
    _fsync_directory(published.parent)
    return RestorationResult(member.blob_hash, RestorationOutcome.RESTORED, "", destination, str(published))


def _converged_capture(
    member: BlobDispositionMember,
    *,
    convergence: CaptureConvergence,
    target: Path,
    destination_kind: str,
) -> RestorationResult | None:
    """Translate the spool's verdict on a resident identity, or None to publish."""
    from polylogue.browser_capture.receiver import CaptureConvergence

    if convergence is CaptureConvergence.DUPLICATE:
        return RestorationResult(
            member.blob_hash,
            RestorationOutcome.RESTORATION_ALREADY_PRESENT,
            "the ordinary spool already holds this capture",
            destination_kind,
            str(target),
        )
    if convergence is CaptureConvergence.SUPERSEDED:
        return RestorationResult(
            member.blob_hash,
            RestorationOutcome.RESTORATION_SUPERSEDED,
            "the ordinary spool holds a newer or richer capture of this session",
            destination_kind,
            str(target),
        )
    if convergence is CaptureConvergence.NAME_COLLISION:
        return RestorationResult(
            member.blob_hash,
            RestorationOutcome.BLOCKED,
            f"ordinary spool admission refused: capture artifact name collision for {target.name}",
            destination_kind,
        )
    return None


def _restore_browser_capture(
    member: BlobDispositionMember,
    *,
    path: Path,
    spool_root: Path,
    rehearsed: _RehearsedSpools,
    dry_run: bool,
) -> RestorationResult:
    """Hand one carrier to the capture spool and record the verdict it gives.

    One provider session keeps one artifact, so a carrier arriving at an
    occupied name is an ordinary revision the spool converges: it publishes
    the newer or richer one and declines the rest. Only a malformed envelope,
    a genuinely different session claiming the name, or the spool quota stops
    a carrier here.
    """
    from pydantic import ValidationError

    from polylogue.browser_capture.models import BrowserCaptureEnvelope
    from polylogue.browser_capture.receiver import (
        BrowserCaptureSpoolConflictError,
        SpoolQuotaExceededError,
        capture_artifact_path,
        capture_convergence,
        write_capture_envelope_bytes,
    )

    destination_kind = RestorationDestination.BROWSER_CAPTURE_SPOOL.value
    try:
        raw = path.read_bytes()
        envelope = BrowserCaptureEnvelope.model_validate_json(raw)
    except (OSError, ValidationError, ValueError) as exc:
        return RestorationResult(
            member.blob_hash, RestorationOutcome.BLOCKED, f"carrier is not a valid capture: {exc}", destination_kind
        )
    target = capture_artifact_path(envelope, spool_root)

    if dry_run:
        resident, unreadable = rehearsed.capture(target)
        if unreadable is not None:
            return RestorationResult(
                member.blob_hash,
                RestorationOutcome.BLOCKED,
                f"ordinary spool admission refused: {unreadable}",
                destination_kind,
            )
        if resident is not None:
            converged = _converged_capture(
                member,
                convergence=capture_convergence(envelope, resident),
                target=target,
                destination_kind=destination_kind,
            )
            if converged is not None:
                return converged
        else:
            refusal = rehearsed.capture_quota_refusal(spool_root)
            if refusal is not None:
                return RestorationResult(
                    member.blob_hash,
                    RestorationOutcome.BLOCKED,
                    f"ordinary spool admission refused: {refusal}",
                    destination_kind,
                )
        rehearsed.record_capture(target, envelope, size_bytes=len(raw), new=resident is None)
        return RestorationResult(
            member.blob_hash,
            RestorationOutcome.RESTORED,
            f"would restore to {destination_kind}",
            destination_kind,
            str(target),
        )

    try:
        written = write_capture_envelope_bytes(raw, spool_path=spool_root)
    except (BrowserCaptureSpoolConflictError, SpoolQuotaExceededError, OSError, ValueError) as exc:
        return RestorationResult(
            member.blob_hash,
            RestorationOutcome.BLOCKED,
            f"ordinary spool admission refused: {exc}",
            destination_kind,
        )
    converged = _converged_capture(
        member, convergence=written.convergence, target=written.path, destination_kind=destination_kind
    )
    if converged is not None:
        return converged
    # The capture receiver publishes the acquired bytes verbatim, so the only
    # honest residency check is the carrier's own bytes read back out.
    try:
        published = written.path.read_bytes()
    except OSError as exc:
        return RestorationResult(
            member.blob_hash, RestorationOutcome.BLOCKED, f"restored file does not read back: {exc}", destination_kind
        )
    if published != raw:
        return RestorationResult(
            member.blob_hash,
            RestorationOutcome.BLOCKED,
            f"restored capture is {len(published)} bytes, not the carrier's {len(raw)}",
            destination_kind,
        )
    _fsync_directory(written.path.parent)
    return RestorationResult(member.blob_hash, RestorationOutcome.RESTORED, "", destination_kind, str(written.path))


def _fsync_directory(path: Path) -> None:
    """Persist the removed directory entry before claiming the effect."""
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
) -> tuple[RestorationResult, ...]:
    """Make every sole-copy carrier's material resident in an ordinary spool.

    This never deletes or modifies the historical carrier, so an interruption
    at any point leaves the blob intact and the operation resumable. Material
    a configured source already holds is not published a second time: a
    carrier restored by an earlier pass is proven at the spool it was restored
    into, which is exactly the residency this step exists to establish.

    A rehearsal resolves every destination and evaluates the same admission
    rule the apply hands its carriers to, so the two report the same counts.
    """
    results: list[RestorationResult] = []
    rehearsed = _RehearsedSpools()
    for member in plan.members_for(BlobDisposition.RESTORE_REQUIRED):
        drift = _carrier_drift(member, context=context)
        if drift is not None:
            results.append(RestorationResult(member.blob_hash, RestorationOutcome.BLOCKED, drift))
            continue
        if member.restoration is None:
            results.append(
                RestorationResult(
                    member.blob_hash, RestorationOutcome.BLOCKED, "restore_required member names no destination"
                )
            )
            continue
        destination = member.restoration.destination
        path = context.blob_store.blob_path(member.blob_hash)
        size_bytes = path.stat().st_size
        resident = next(
            (proof for prover in context.provers if (proof := prover.prove(member.blob_hash, path, size_bytes))), None
        )
        if resident is not None:
            results.append(
                RestorationResult(
                    member.blob_hash,
                    RestorationOutcome.RESTORATION_ALREADY_PRESENT,
                    f"{resident.prover} proves this material at a configured source",
                    destination.value,
                    resident.source_path,
                )
            )
            continue
        if destination is RestorationDestination.HOOK_EVENT_SPOOL:
            results.append(
                _restore_hook_event(member, path=path, spool_root=hook_spool_root, rehearsed=rehearsed, dry_run=dry_run)
            )
        else:
            results.append(
                _restore_browser_capture(
                    member, path=path, spool_root=browser_capture_spool, rehearsed=rehearsed, dry_run=dry_run
                )
            )
    return tuple(results)


def _namespace_relative_path(entry: str) -> tuple[str, ...] | None:
    """Recover the namespace-relative path an invalid-entry record names.

    The record is ``<relative path>: <issue>``. Only a strictly relative,
    ``..``-free path is accepted: a path that could climb out of the namespace
    is not a namespace entry at all.
    """
    relative, separator, issue = entry.rpartition(": ")
    if not separator or issue not in _ISSUE_LABELS or not relative:
        return None
    pure = PurePosixPath(relative)
    if pure.is_absolute():
        return None
    parts = pure.parts
    if not parts or any(part in ("", "..", ".") for part in parts):
        return None
    return parts


def _member_result(
    member: BlobDispositionMember,
    *,
    outcome: MemberOutcome,
    detail: str,
    referenced: bool,
    from_path: Path,
    restoration: RestorationResult | None,
) -> MemberResult:
    return MemberResult(
        blob_hash=member.blob_hash,
        outcome=outcome,
        detail=detail,
        cohort=member.disposition.value,
        referenced=referenced,
        size_bytes=member.size_bytes,
        from_path=str(from_path),
        restoration_outcome="" if restoration is None else restoration.outcome.value,
        restored_to="" if restoration is None else restoration.spool_path,
    )


def _classify_member(
    member: BlobDispositionMember,
    *,
    context: BlobDispositionContext,
    restorations: dict[str, RestorationResult],
) -> tuple[MemberResult, bool]:
    """Decide one member's outcome, and whether it is a deletion candidate."""
    source = context.blob_store.blob_path(member.blob_hash)
    referenced = member.referenced or member.blob_hash in context.referenced_hashes
    restoration = restorations.get(member.blob_hash)
    if referenced:
        return (
            _member_result(
                member,
                outcome=MemberOutcome.RETAINED_REFERENCED,
                detail="a durable row still names this object",
                referenced=True,
                from_path=source,
                restoration=restoration,
            ),
            False,
        )
    if not source.exists():
        return (
            _member_result(
                member,
                outcome=MemberOutcome.RETAINED_ABSENT,
                detail="the object is no longer in the namespace",
                referenced=False,
                from_path=source,
                restoration=restoration,
            ),
            False,
        )
    if member.disposition is BlobDisposition.RESTORE_REQUIRED:
        if restoration is None:
            return (
                _member_result(
                    member,
                    outcome=MemberOutcome.BLOCKED,
                    detail="restoration produced no result for this member",
                    referenced=False,
                    from_path=source,
                    restoration=None,
                ),
                False,
            )
        if not restoration.completed:
            return (
                _member_result(
                    member,
                    outcome=MemberOutcome.BLOCKED,
                    detail=f"the only verified carrier is not resident in a spool: {restoration.detail}",
                    referenced=False,
                    from_path=source,
                    restoration=restoration,
                ),
                False,
            )
    return (
        _member_result(
            member,
            outcome=MemberOutcome.DELETED,
            detail="",
            referenced=False,
            from_path=source,
            restoration=restoration,
        ),
        True,
    )


def _delete_invalid_entries(
    plan: BlobDispositionPlan,
    *,
    blob_root: Path,
    dry_run: bool,
    synced: set[Path],
) -> list[MemberResult]:
    """Remove the namespace's non-blob entries.

    A SQLite ``-wal`` or ``-shm`` stranded beside a content-addressed object
    is a byproduct of something having opened that object as a database. The
    object's own bytes are its identity, so the sidecar carries nothing the
    namespace owns.
    """
    results: list[MemberResult] = []
    for entry in plan.denominator.invalid_namespace_entries:
        parts = _namespace_relative_path(entry)
        if parts is None:
            results.append(
                MemberResult(
                    blob_hash="",
                    outcome=MemberOutcome.BLOCKED,
                    detail=f"unreadable namespace-entry record: {entry}",
                    cohort=INVALID_ENTRY_COHORT,
                    from_path=entry,
                )
            )
            continue
        source = blob_root.joinpath(*parts)
        try:
            source_stat = source.lstat()
        except FileNotFoundError:
            results.append(
                MemberResult(
                    blob_hash="",
                    outcome=MemberOutcome.RETAINED_ABSENT,
                    detail="the entry is no longer in the namespace",
                    cohort=INVALID_ENTRY_COHORT,
                    from_path=str(source),
                )
            )
            continue
        except OSError as exc:
            results.append(
                MemberResult(
                    blob_hash="",
                    outcome=MemberOutcome.BLOCKED,
                    detail=f"namespace entry is unreadable: {exc}",
                    cohort=INVALID_ENTRY_COHORT,
                    from_path=str(source),
                )
            )
            continue
        if not stat.S_ISREG(source_stat.st_mode):
            # A directory or a symlink is not one stray object: unlinking it
            # would either fail or follow the entry somewhere the namespace
            # does not own.
            results.append(
                MemberResult(
                    blob_hash="",
                    outcome=MemberOutcome.BLOCKED,
                    detail="namespace entry is not a regular file",
                    cohort=INVALID_ENTRY_COHORT,
                    from_path=str(source),
                )
            )
            continue
        if not dry_run:
            try:
                source.unlink()
            except OSError as exc:
                results.append(
                    MemberResult(
                        blob_hash="",
                        outcome=MemberOutcome.BLOCKED,
                        detail=f"unlink failed: {exc}",
                        cohort=INVALID_ENTRY_COHORT,
                        size_bytes=source_stat.st_size,
                        from_path=str(source),
                    )
                )
                continue
            synced.add(source.parent)
        results.append(
            MemberResult(
                blob_hash="",
                outcome=MemberOutcome.DELETED,
                detail="",
                cohort=INVALID_ENTRY_COHORT,
                size_bytes=source_stat.st_size,
                from_path=str(source),
            )
        )
    return results


def _authorization_blockers(
    plan: BlobDispositionPlan,
    *,
    authorized_digest: str,
    context: BlobDispositionContext,
) -> tuple[str, ...]:
    blockers: list[str] = []
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
    """Restore sole copies, then delete every member no durable row names."""
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

    before = measure_namespace(context.blob_store)
    restorations = restore_plan_members(
        plan,
        context=context,
        hook_spool_root=hook_spool_root,
        browser_capture_spool=browser_capture_spool,
        dry_run=dry_run,
    )
    by_hash = {restoration.blob_hash: restoration for restoration in restorations}
    classified = [_classify_member(member, context=context, restorations=by_hash) for member in plan.members]
    candidates = [result for result, deletable in classified if deletable]
    results = [result for result, _ in classified]

    seam_blockers: tuple[str, ...] = ()
    if not dry_run and candidates:
        results, seam_blockers = _delete_candidates(
            results,
            candidates=candidates,
            context=context,
            source_db=source_db,
            index_db=index_db,
        )

    synced: set[Path] = set()
    results.extend(_delete_invalid_entries(plan, blob_root=context.blob_store.root, dry_run=dry_run, synced=synced))
    for directory in sorted(synced):
        _fsync_directory(directory)
    return DispositionApplyReceipt(
        tool_version=TOOL_VERSION,
        plan_digest=plan.digest(),
        archive_root=plan.archive_root,
        blob_root=plan.blob_root,
        dry_run=dry_run,
        results=tuple(results),
        restorations=restorations,
        namespace_before=before,
        blockers=seam_blockers,
    )


_GC_SCHEMA_ABSENT_BLOCKERS = frozenset(
    {
        "blob GC durable member-intent schema is unavailable",
        "blob GC durable namespace-identity schema is unavailable",
    }
)
DIRECT_UNLINK_DETAIL = "deleted by direct unlink: the source tier predates the blob-GC member-intent schema"


def _delete_candidates_directly(
    results: list[MemberResult],
    *,
    candidates: list[MemberResult],
    context: BlobDispositionContext,
    source_db: Path,
) -> tuple[list[MemberResult], tuple[str, ...]]:
    """Unlink unreferenced candidates without the GC generation ledger.

    Used only when the source tier has no ``gc_generation_members`` table.
    Publishers are excluded for the whole pass; a hash that is referenced now
    stays on disk and is reported retained.
    """
    from polylogue.storage.blob_publication import exclude_archive_blob_publishers

    errors: list[str] = []
    deleted: set[str] = set()
    retained: set[str] = set()
    touched: set[Path] = set()
    with exclude_archive_blob_publishers(source_db):
        for result in candidates:
            if result.blob_hash in context.referenced_hashes:
                retained.add(result.blob_hash)
                continue
            path = context.blob_store.blob_path(result.blob_hash)
            try:
                path.unlink()
            except FileNotFoundError:
                deleted.add(result.blob_hash)
                continue
            except OSError as exc:
                errors.append(f"{result.blob_hash[:16]}: {exc}")
                continue
            deleted.add(result.blob_hash)
            touched.add(path.parent)
    for directory in sorted(touched):
        _fsync_directory(directory)
    updated = [
        replace(result, detail=DIRECT_UNLINK_DETAIL)
        if result.blob_hash in deleted and result.outcome is MemberOutcome.DELETED
        else replace(
            result, outcome=MemberOutcome.RETAINED_REFERENCED, detail="referenced by a durable row at unlink time"
        )
        if result.blob_hash in retained and result.outcome is MemberOutcome.DELETED
        else result
        for result in results
    ]
    return updated, tuple(errors)


def _delete_candidates(
    results: list[MemberResult],
    *,
    candidates: list[MemberResult],
    context: BlobDispositionContext,
    source_db: Path,
    index_db: Path,
) -> tuple[list[MemberResult], tuple[str, ...]]:
    """Unlink the candidate set through the canonical blob-GC seam.

    The seam repeats the liveness decision under its own write locks, so a
    reference that appeared since this pass read the reference set leaves the
    object on disk. A candidate still present afterwards is never reported
    deleted: the seam either declined it, or failed the whole generation and
    said why.
    """
    from polylogue.storage.blob_gc import unlink_unreferenced_blob_hashes_under_exclusion

    _, _, errors = unlink_unreferenced_blob_hashes_under_exclusion(
        source_db,
        index_db,
        context.blob_store.root,
        {result.blob_hash for result in candidates},
    )
    if errors and all(error in _GC_SCHEMA_ABSENT_BLOCKERS for error in errors):
        # A source tier older than the GC member-intent schema cannot record a
        # generation, but the disposition plan already carries the liveness
        # decision and this pass re-read the reference set; unlink directly
        # and say so on every member.
        results, errors = _delete_candidates_directly(
            results, candidates=candidates, context=context, source_db=source_db
        )
    declined = {result.blob_hash for result in candidates if context.blob_store.blob_path(result.blob_hash).exists()}
    if not declined:
        return results, tuple(errors)
    outcome, detail = (
        (MemberOutcome.BLOCKED, "the blob-GC seam did not complete this generation")
        if errors
        else (MemberOutcome.RETAINED_REFERENCED, "the blob-GC seam's locked recheck found this object still protected")
    )
    return [
        replace(result, outcome=outcome, detail=detail)
        if result.blob_hash in declined and result.outcome is MemberOutcome.DELETED
        else result
        for result in results
    ], tuple(errors)


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
    "INVALID_ENTRY_COHORT",
    "TOOL_VERSION",
    "DispositionApplyError",
    "DispositionApplyReceipt",
    "MemberOutcome",
    "MemberResult",
    "NamespaceTotals",
    "RestorationOutcome",
    "RestorationResult",
    "apply_disposition_plan",
    "measure_namespace",
    "restore_plan_members",
    "write_receipt",
]
