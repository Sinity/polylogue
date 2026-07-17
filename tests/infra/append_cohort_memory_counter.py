"""Phase-level evidence collection for watcher append/cohort investigations.

The live incident was in ``_ingest_append_plans_archive`` while
``classify_raw_revision_cohort`` reread historical full snapshots.  This
helper distinguishes the durable metadata replay-plan hot path from that
conservative classifier fallback.  It deliberately does not serialize parsed
object graphs: the observer records kernel-visible process/cgroup/I/O state and
deterministic route, payload, and batch counts instead.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

from polylogue.scenarios import (
    MeasurementScope,
    WorkloadPhaseObservation,
    WorkloadReceipt,
    WorkloadRunStatus,
    watcher_append_cohort_canary_spec,
)


@dataclass(frozen=True, slots=True)
class MemoryPhase:
    """A point-in-time kernel and process memory observation."""

    name: str
    rss_bytes: int | None
    pss_bytes: int | None
    anonymous_pss_bytes: int | None
    cgroup_anon_bytes: int | None
    cgroup_file_bytes: int | None
    io_read_bytes: int | None
    io_write_bytes: int | None
    batch_count: int
    plan_count: int


@dataclass
class AppendCohortMemoryCounter:
    """Measurements from the production watcher-append authority route."""

    calls_by_site: Counter[str] = field(default_factory=Counter)
    bytes_by_site: Counter[str] = field(default_factory=Counter)
    batch_count: int = 0
    plan_count: int = 0
    phases: list[MemoryPhase] = field(default_factory=list)
    _io_baseline: dict[str, int] | None = field(default_factory=lambda: _proc_io())

    def record(self, site: str, byte_count: int = 0) -> None:
        self.calls_by_site[site] += 1
        self.bytes_by_site[site] += int(byte_count)

    def snapshot(self, name: str) -> None:
        memory_stat = _cgroup_memory_stat()
        io_now = _proc_io()
        self.phases.append(
            MemoryPhase(
                name=name,
                rss_bytes=_rss_bytes(),
                pss_bytes=_smaps_rollup_bytes("Pss:"),
                anonymous_pss_bytes=_smaps_rollup_bytes("Pss_Anon:"),
                cgroup_anon_bytes=memory_stat.get("anon"),
                cgroup_file_bytes=memory_stat.get("file"),
                io_read_bytes=_io_delta(io_now, self._io_baseline, "read_bytes"),
                io_write_bytes=_io_delta(io_now, self._io_baseline, "write_bytes"),
                batch_count=self.batch_count,
                plan_count=self.plan_count,
            )
        )

    def summary(self) -> str:
        lines = [
            "AppendCohortMemoryCounter:",
            f"  batches={self.batch_count}, plans={self.plan_count}",
        ]
        for site in sorted(self.calls_by_site):
            lines.append(f"  {site}: {self.calls_by_site[site]} call(s), {self.bytes_by_site[site]:,} bytes")
        for phase in self.phases:
            lines.append(
                "  "
                f"{phase.name}: rss={_format_bytes(phase.rss_bytes)}, pss={_format_bytes(phase.pss_bytes)}, "
                f"anon_pss={_format_bytes(phase.anonymous_pss_bytes)}, "
                f"cgroup_anon={_format_bytes(phase.cgroup_anon_bytes)}, "
                f"cgroup_file={_format_bytes(phase.cgroup_file_bytes)}, "
                f"io_read={_format_bytes(phase.io_read_bytes)}, io_write={_format_bytes(phase.io_write_bytes)}, "
                f"batches={phase.batch_count}, plans={phase.plan_count}"
            )
        return "\n".join(lines)

    def workload_receipt(
        self,
        *,
        profile_id: str,
        archive_id: str,
        build_id: str,
        runtime_id: str,
        generation_id: str,
    ) -> WorkloadReceipt:
        """Project kernel observations into the shared scoped receipt contract."""
        spec = watcher_append_cohort_canary_spec(profile_id=profile_id, archive_id=archive_id)
        peak_rss = max((phase.rss_bytes for phase in self.phases if phase.rss_bytes is not None), default=None)
        observations: list[WorkloadPhaseObservation] = []
        for phase in self.phases:
            observations.extend(
                (
                    WorkloadPhaseObservation(
                        name=f"{phase.name}:{MeasurementScope.PROCESS_TREE.value}",
                        measurement_scope=MeasurementScope.PROCESS_TREE,
                        current_rss_bytes=phase.rss_bytes,
                        peak_rss_bytes=peak_rss if phase.name == "quiescent" else None,
                        current_pss_bytes=phase.pss_bytes,
                        anon_bytes=phase.anonymous_pss_bytes,
                        read_io_bytes=phase.io_read_bytes,
                        write_io_bytes=phase.io_write_bytes,
                        progress_completed=phase.plan_count,
                        progress_total=self.plan_count,
                        quiescent=phase.name == "quiescent",
                    ),
                    WorkloadPhaseObservation(
                        name=f"{phase.name}:{MeasurementScope.CGROUP.value}",
                        measurement_scope=MeasurementScope.CGROUP,
                        anon_bytes=phase.cgroup_anon_bytes,
                        file_cache_bytes=phase.cgroup_file_bytes,
                        progress_completed=phase.plan_count,
                        progress_total=self.plan_count,
                        quiescent=phase.name == "quiescent",
                    ),
                )
            )
        return WorkloadReceipt.from_observations(
            spec=spec,
            status=WorkloadRunStatus.SUCCEEDED,
            build_id=build_id,
            runtime_id=runtime_id,
            archive_id=archive_id,
            generation_id=generation_id,
            frame_id=None,
            phases=tuple(observations),
            cleanup_complete=True,
            notes=(
                "The synchronous append route owns no child workers; its current process is the complete process tree. "
                "Pytest archive teardown is outside the measured workload boundary.",
                "peak_rss_bytes is the maximum sampled RSS across declared route boundaries.",
            ),
        )


def _rss_bytes() -> int | None:
    try:
        resident_pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (IndexError, OSError, ValueError):
        return None


def _smaps_rollup_bytes(field: str) -> int | None:
    try:
        for line in Path("/proc/self/smaps_rollup").read_text(encoding="ascii").splitlines():
            if line.startswith(field):
                return int(line.split()[1]) * 1024
    except (IndexError, OSError, ValueError):
        return None
    return None


def _cgroup_memory_stat() -> dict[str, int]:
    try:
        cgroup_path = next(
            line.split(":", 2)[2] or "/"
            for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
            if line.startswith("0::")
        )
        memory_stat = Path("/sys/fs/cgroup") / cgroup_path.lstrip("/") / "memory.stat"
        return {
            name: int(value)
            for line in memory_stat.read_text(encoding="utf-8").splitlines()
            if len(parts := line.split()) == 2
            and parts[0] in {"anon", "file"}
            and (name := parts[0])
            and (value := parts[1]).isdigit()
        }
    except (OSError, StopIteration):
        return {}


def _proc_io() -> dict[str, int] | None:
    try:
        return {
            key.strip(): int(value.strip())
            for line in Path("/proc/self/io").read_text(encoding="utf-8").splitlines()
            if ":" in line and (key := line.partition(":")[0]) and (value := line.partition(":")[2]).strip().isdigit()
        }
    except OSError:
        return None


def _io_delta(now: dict[str, int] | None, baseline: dict[str, int] | None, key: str) -> int | None:
    if now is None or baseline is None or key not in now or key not in baseline:
        return None
    return now[key] - baseline[key]


def _format_bytes(value: int | None) -> str:
    return "unavailable" if value is None else f"{value:,}"


@contextmanager
def append_cohort_memory_counter() -> Iterator[AppendCohortMemoryCounter]:
    """Instrument append ingestion, replay metadata, and classifier fallback."""
    from polylogue.sources.live import append_ingest
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    counter = AppendCohortMemoryCounter()
    real_append_ingest = append_ingest._ingest_append_plans_archive
    real_classify = ArchiveStore.classify_raw_revision_cohort
    real_replay_plan = ArchiveStore.raw_revision_replay_plan
    real_read_all = ArchiveBlobPublisher.read_all
    in_cohort_classification = False

    def counted_append_ingest(owner: Any, plans: list[Any], archive_root: Path) -> Any:
        counter.batch_count += 1
        counter.plan_count += len(plans)
        counter.record("watcher_append_payload", sum(len(plan.payload) for plan in plans))
        counter.snapshot("watcher_append:before")
        result = real_append_ingest(owner, plans, archive_root)
        counter.snapshot("watcher_append:after")
        return result

    def counted_classify(archive: ArchiveStore, logical_source_key: str) -> Any:
        nonlocal in_cohort_classification
        counter.record("classify_raw_revision_cohort")
        counter.snapshot("classify_raw_revision_cohort:before")
        in_cohort_classification = True
        try:
            result = real_classify(archive, logical_source_key)
        finally:
            in_cohort_classification = False
        counter.record("accepted_raw_ids", len(result.accepted_raw_ids))
        counter.snapshot("classify_raw_revision_cohort:after")
        return result

    def counted_replay_plan(archive: ArchiveStore, logical_source_key: str) -> Any:
        counter.record("raw_revision_replay_plan")
        counter.snapshot("raw_revision_replay_plan:before")
        result = real_replay_plan(archive, logical_source_key)
        counter.record("accepted_raw_ids", len(result.accepted_raw_ids))
        counter.snapshot("raw_revision_replay_plan:after")
        return result

    def counted_read_all(publisher: ArchiveBlobPublisher, hash_hex: str) -> bytes:
        payload = real_read_all(publisher, hash_hex)
        counter.record(
            "historical_full_blob.read_all" if in_cohort_classification else "replay_raw_blob.read_all", len(payload)
        )
        return payload

    with (
        patch.object(append_ingest, "_ingest_append_plans_archive", counted_append_ingest),
        patch.object(ArchiveStore, "classify_raw_revision_cohort", counted_classify),
        patch.object(ArchiveStore, "raw_revision_replay_plan", counted_replay_plan),
        patch.object(ArchiveBlobPublisher, "read_all", counted_read_all),
    ):
        yield counter


__all__ = ["AppendCohortMemoryCounter", "MemoryPhase", "append_cohort_memory_counter"]
