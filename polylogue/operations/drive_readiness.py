"""Configured Drive completeness, independent of local cold-build settlement."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.archive.session_revision_membership import MembershipDecision
from polylogue.core.evidence import Measured
from polylogue.core.raw_failure_evidence import RAW_FAILURE_TERMINAL_EVIDENCE_SUPPORT_STATUS_PAIRS
from polylogue.sources.drive.source_support import _parse_modified_time
from polylogue.sources.drive.witness import DriveListingWitness
from polylogue.storage.tier_access import capture_sqlite_read

if TYPE_CHECKING:
    from polylogue.config import Config, DriveConfig, PolylogueConfig, Source


class DriveCatchupState(str, Enum):
    COMPLETE = "complete"
    PENDING = "pending"
    RETRYABLE = "retryable"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DriveFileFailure:
    coordinate: str
    code: str
    permanent: bool = False


@dataclass(frozen=True, slots=True)
class DriveCatchupReport:
    state: DriveCatchupState
    changed_count: int = 0
    witness: tuple[Mapping[str, object], ...] = ()
    enumerated_count: int | None = None
    acquired_count: int | None = None
    materialization_pending: int | None = None
    failures: tuple[DriveFileFailure, ...] = ()
    gaps: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "changed_count": self.changed_count,
            "witness": [dict(witness) for witness in self.witness],
            "enumerated_count": self.enumerated_count,
            "acquired_count": self.acquired_count,
            "materialization_pending": self.materialization_pending,
            "failures": [
                {"coordinate": failure.coordinate, "code": failure.code, "permanent": failure.permanent}
                for failure in self.failures
            ],
            "gaps": list(self.gaps),
        }


@dataclass(slots=True)
class DriveReadinessObservation:
    """The daemon's current listing custody. Losing it withholds readiness."""

    archive_root: Path
    witnesses: dict[str, DriveListingWitness] = field(default_factory=dict)
    resume_materialization: bool = False

    def close(self) -> None:
        for witness in self.witnesses.values():
            witness.close()
        self.witnesses.clear()


_OBSERVATION_LOCK = threading.RLock()
_OBSERVATIONS: dict[Path, DriveReadinessObservation] = {}


def drive_readiness_observation(root: Path) -> DriveReadinessObservation:
    with _OBSERVATION_LOCK:
        return _OBSERVATIONS.setdefault(root.absolute(), DriveReadinessObservation(root.absolute()))


def reset_drive_readiness_observation(root: Path) -> None:
    with _OBSERVATION_LOCK:
        prior = _OBSERVATIONS.pop(root.absolute(), None)
        if prior is not None:
            prior.close()


def reobserve_drive_readiness(observation: DriveReadinessObservation, config: DriveConfig | None) -> None:
    from polylogue.sources.drive import _resolved_drive_client

    client = _resolved_drive_client(ui=None, client=None, drive_config=config)
    for witness in observation.witnesses.values():
        witness.reobserve(client)


def inspect_drive_readiness(
    sources: Sequence[Source],
    source: sqlite3.Connection | None,
    index: sqlite3.Connection | None,
    witnesses: Mapping[str, DriveListingWitness],
    *,
    changed_count: int = 0,
    raw_owner_available: bool = True,
) -> DriveCatchupReport:
    """Bind every listed revision to retained Source and the supplied Index view.

    Raw presence alone is insufficient. Each membership needs its current
    applied cohort head in this exact Index, including superseded revisions.
    """
    configured = [item for item in sources if item.is_drive]
    if not configured:
        return DriveCatchupReport(
            DriveCatchupState.COMPLETE, changed_count, enumerated_count=0, acquired_count=0, materialization_pending=0
        )
    if source is None or index is None:
        return DriveCatchupReport(
            DriveCatchupState.UNKNOWN, changed_count, gaps=("drive_archive_authority_unavailable",)
        )
    summaries: list[Mapping[str, object]] = []
    failures: list[DriveFileFailure] = []
    gaps: set[str] = set()
    enumerated = acquired = pending = 0
    unknown = changed = blocked = False
    for item in configured:
        witness = witnesses.get(item.name)
        if witness is None or witness.folder_ref != item.folder:
            unknown = True
            gaps.add("drive_listing_witness_missing")
            continue
        summary = witness.summary()
        summaries.append(summary)
        if not witness.listing_complete or not witness.postlisting_complete:
            unknown = True
            gaps.add("drive_listing_witness_unfinished")
        if witness.enumeration_error:
            gaps.add("drive_listing_failed")
        changed = changed or witness.changed
        for coordinate, revision, raw_id, acquired_revision, failure, permanent in witness.members():
            enumerated += 1
            if failure is not None:
                failures.append(DriveFileFailure(coordinate, failure, permanent))
                continue
            if revision is None:
                unknown = True
                gaps.add("drive_revision_unmeasured")
            if raw_id is not None and acquired_revision != revision:
                unknown = True
                gaps.add("drive_acquired_revision_unproved")
                continue
            if raw_id is None:
                timestamp = _parse_modified_time(revision)
                if timestamp is not None:
                    row = source.execute(
                        "SELECT raw_id FROM raw_sessions WHERE source_path=? AND file_mtime_ms=? ORDER BY acquired_at_ms DESC,raw_id LIMIT 1",
                        (coordinate, round(timestamp * 1000)),
                    ).fetchone()
                    if row is not None:
                        raw_id = str(row[0])
            row = (
                None
                if raw_id is None
                else source.execute(
                    "SELECT raw_id,parse_error FROM raw_sessions WHERE raw_id=? AND source_path=?", (raw_id, coordinate)
                ).fetchone()
            )
            if row is None:
                pending += 1
                gaps.add("drive_raw_not_retained")
                continue
            acquired += 1
            terminal = any(
                (kind, support) in RAW_FAILURE_TERMINAL_EVIDENCE_SUPPORT_STATUS_PAIRS
                for kind, support in source.execute(
                    "SELECT artifact_kind,support_status FROM raw_artifacts WHERE raw_id=?", (raw_id,)
                )
            )
            if row[1] is not None or terminal:
                failures.append(DriveFileFailure(coordinate, "drive_raw_parse_refused", terminal))
                continue
            memberships = source.execute(
                "SELECT logical_source_key,decision FROM raw_session_memberships WHERE raw_id=?", (raw_id,)
            )
            member_count = 0
            census = source.execute("SELECT status FROM raw_membership_census WHERE raw_id=?", (raw_id,)).fetchone()
            materialized = census is not None and census[0] == "complete"
            for key, decision in memberships:
                member_count += 1
                if decision == MembershipDecision.AMBIGUOUS:
                    blocked = True
                    failures.append(DriveFileFailure(coordinate, "drive_membership_ambiguous"))
                elif decision == MembershipDecision.DEFERRED:
                    gaps.add("drive_membership_deferred")
                if decision not in {
                    MembershipDecision.APPLIED,
                    MembershipDecision.SUPERSEDED_PREFIX,
                    MembershipDecision.SUPERSEDED_EQUIVALENT,
                    MembershipDecision.SUPERSEDED_BY_WINNER,
                }:
                    materialized = False
                    continue
                head = source.execute(
                    "SELECT raw_id FROM raw_session_memberships WHERE logical_source_key=? AND decision='applied'",
                    (key,),
                ).fetchone()
                if (
                    head is None
                    or index.execute(
                        "SELECT 1 FROM sessions WHERE session_id=? AND raw_id=?", (key, head[0])
                    ).fetchone()
                    is None
                ):
                    materialized = False
            if not member_count or not materialized:
                pending += 1
                gaps.add("drive_materialization_pending")
    if not raw_owner_available:
        unknown = True
        gaps.add("drive_raw_owner_unavailable")
    if blocked:
        state = DriveCatchupState.BLOCKED
    elif failures:
        state = (
            DriveCatchupState.BLOCKED if any(failure.permanent for failure in failures) else DriveCatchupState.RETRYABLE
        )
    elif unknown:
        state = DriveCatchupState.UNKNOWN
    elif "drive_listing_failed" in gaps:
        state = DriveCatchupState.RETRYABLE
    elif changed or pending:
        state = DriveCatchupState.PENDING
        if changed:
            gaps.add("drive_listing_changed")
    else:
        state = DriveCatchupState.COMPLETE
    return DriveCatchupReport(
        state,
        changed_count,
        tuple(summaries),
        None if unknown else enumerated,
        None if unknown else acquired,
        None if unknown else pending,
        tuple(failures),
        tuple(sorted(gaps)),
    )


def configured_source_component(report: DriveCatchupReport) -> dict[str, object]:
    return {
        "component": "configured_sources",
        "scope": "configured_remote_source_completeness",
        "state": "ready"
        if report.state is DriveCatchupState.COMPLETE
        else "unknown"
        if report.state is DriveCatchupState.UNKNOWN
        else "degraded",
        "summary": "configured remote sources complete"
        if report.state is DriveCatchupState.COMPLETE
        else "Drive source completeness: " + report.state.value,
        "counts": {
            "enumerated": report.enumerated_count,
            "acquired": report.acquired_count,
            "materialization_pending": report.materialization_pending,
        },
        "caveats": list(report.gaps),
        "evidence_refs": [],
        "last_success": None,
        "last_attempt": None,
        "repair_hint": None,
        "drive": report.to_dict(),
    }


def configured_status_sources(config: object | None) -> tuple[Source, ...] | None:
    """Use the executing configuration, including resolved paths and credentials."""
    if config is None:
        return None
    sources = getattr(config, "sources", None)
    if sources is not None:
        return tuple(sources)
    from polylogue.config import PolylogueConfig, resolve_runtime_config

    if isinstance(config, PolylogueConfig):
        return resolve_runtime_config(cli_overrides=config.raw).sources
    return None


def configured_source_observation_fingerprint(config: Config | PolylogueConfig) -> str:
    """Bind a cached status reading to its executing scope and listing custody."""
    root = Path(config.archive_root)
    sources = configured_status_sources(config)
    with _OBSERVATION_LOCK:
        observation = _OBSERVATIONS.get(root.absolute())
        witnesses = {} if observation is None else observation.witnesses.copy()
    identity = (
        str(root.absolute()),
        None if sources is None else tuple((source.name, source.folder) for source in sources if source.is_drive),
        DriveListingWitness.selection_rule,
        tuple(sorted((name, id(witness), witness.generation) for name, witness in witnesses.items())),
    )
    return hashlib.sha256(repr(identity).encode()).hexdigest()


def configured_source_readiness_from_archive(archive: object, config: object | None) -> dict[str, object]:
    """The single composer shared by resident and executing status snapshots."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    if not isinstance(archive, ArchiveStore):
        raise TypeError("configured source readiness requires the executing archive")
    sources = configured_status_sources(config)
    if sources is None:
        report = DriveCatchupReport(DriveCatchupState.UNKNOWN, gaps=("drive_configuration_unobserved",))
    else:
        measured = capture_sqlite_read(
            lambda: inspect_drive_readiness(
                sources,
                None if "source" in archive.operation_degraded_components else archive.source_connection,
                archive.index_connection,
                drive_readiness_observation(Path(archive.archive_root)).witnesses,
            )
        )
        report = (
            measured.value
            if isinstance(measured, Measured)
            else DriveCatchupReport(DriveCatchupState.UNKNOWN, gaps=("drive_archive_authority_unavailable",))
        )
    return configured_source_component(report)


def inspect_current_drive_readiness(
    root: Path,
    sources: Sequence[Source],
    witnesses: Mapping[str, DriveListingWitness],
    *,
    changed_count: int = 0,
    raw_owner_available: bool = True,
) -> DriveCatchupReport:
    """Inspect the pinned current generation without opening an independent frame."""
    from polylogue.core.errors import ArchiveTierUnavailableError, SchemaRefusalError
    from polylogue.operations.operation_context import open_operation_read

    def read() -> DriveCatchupReport:
        with open_operation_read(root) as pinned:
            return inspect_drive_readiness(
                sources,
                None if "source" in pinned.degraded_components else pinned.archive.source_connection,
                pinned.archive.index_connection,
                witnesses,
                changed_count=changed_count,
                raw_owner_available=raw_owner_available,
            )

    try:
        measured = capture_sqlite_read(read)
    except (ArchiveTierUnavailableError, SchemaRefusalError):
        return DriveCatchupReport(
            DriveCatchupState.UNKNOWN, changed_count, gaps=("drive_archive_authority_unavailable",)
        )
    return (
        measured.value
        if isinstance(measured, Measured)
        else DriveCatchupReport(DriveCatchupState.UNKNOWN, changed_count, gaps=("drive_archive_authority_unavailable",))
    )
