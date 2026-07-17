"""Run a receipt-backed bounded raw-authority fixed-point scenario.

The default is deliberately small enough for local verification.  The caller
may supply the July-15 component/raw cardinalities for a long, contained run;
the JSON receipt records the exact requested shape rather than presenting a
small projection as production evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import resource
import shutil
import sqlite3
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO, cast

from polylogue.config import Config, get_config
from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.scenarios.workload import (
    WorkloadPhaseObservation,
    WorkloadReceipt,
    WorkloadRunStatus,
    raw_authority_fixed_point_spec,
)
from polylogue.schemas.workload_tiers import WorkloadScaleTier
from polylogue.storage import repair
from polylogue.storage.raw_authority import RawReplayPlanStatus
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session_blob_ref


@dataclass(frozen=True, slots=True)
class RawAuthorityScalePass:
    number: int
    mode: str
    candidate_count: int
    executable_candidate_count: int
    repaired_count: int
    executable_component_count: int
    fixed_point: bool
    plan_status_counts: dict[str, int]
    wall_ms: int
    peak_rss_bytes: int
    peak_pss_bytes: int | None
    peak_swap_bytes: int | None
    cpu_ms: int | None
    read_io_bytes: int | None
    write_io_bytes: int | None


@dataclass(frozen=True, slots=True)
class ProcessSample:
    """Boundary sample for the runner process, using kernel-owned counters."""

    rss_bytes: int
    pss_bytes: int | None
    swap_bytes: int | None
    cpu_ms: int | None
    read_io_bytes: int | None
    write_io_bytes: int | None
    io_full_avg10: float | None
    memory_full_avg10: float | None


@dataclass(frozen=True, slots=True)
class RawAuthorityScaleScenario:
    """Private-free synthetic corpus shape derived from frontier aggregates."""

    components: int
    direct_candidates: int
    expanded_candidates: int
    total_payload_bytes: int
    component_cohorts: tuple[tuple[int, int, int], ...] | None = None
    component_byte_cohorts: tuple[tuple[int, int, int, int], ...] | None = None
    terminal_sibling_outcome: str = "terminal"

    def __post_init__(self) -> None:
        if self.components < 1:
            raise ValueError("scenario requires at least one authority component")
        if self.direct_candidates < self.components:
            raise ValueError("scenario direct candidates must cover every component")
        if self.expanded_candidates < self.direct_candidates:
            raise ValueError("scenario expanded candidates cannot be smaller than direct candidates")
        if self.total_payload_bytes < self.expanded_candidates * 256:
            raise ValueError("scenario payload budget is too small for valid JSONL evidence")
        if self.terminal_sibling_outcome not in {"terminal", "deferred"}:
            raise ValueError("scenario terminal sibling outcome must be terminal or deferred")
        if self.component_cohorts is not None:
            if not self.component_cohorts:
                raise ValueError("scenario component cohorts cannot be empty")
            if any(
                raw_count < 1 or direct_candidate_count < 1 or direct_candidate_count > raw_count or component_count < 1
                for raw_count, direct_candidate_count, component_count in self.component_cohorts
            ):
                raise ValueError("scenario component cohorts must contain positive valid counts")
            if (
                sum(component_count for _raw_count, _direct_count, component_count in self.component_cohorts)
                != self.components
            ):
                raise ValueError("scenario component cohorts disagree with component count")
            if (
                sum(raw_count * component_count for raw_count, _direct_count, component_count in self.component_cohorts)
                != self.expanded_candidates
            ):
                raise ValueError("scenario component cohorts disagree with expanded candidate count")
            if (
                sum(
                    direct_candidate_count * component_count
                    for _raw_count, direct_candidate_count, component_count in self.component_cohorts
                )
                != self.direct_candidates
            ):
                raise ValueError("scenario component cohorts disagree with direct candidate count")
        if self.component_byte_cohorts is not None:
            if not self.component_byte_cohorts:
                raise ValueError("scenario component byte cohorts cannot be empty")
            if any(
                raw_count < 1
                or direct_candidate_count < 1
                or direct_candidate_count > raw_count
                or upper_bound_blob_bytes < 1
                or component_count < 1
                for raw_count, direct_candidate_count, upper_bound_blob_bytes, component_count in self.component_byte_cohorts
            ):
                raise ValueError("scenario component byte cohorts must contain positive valid counts and byte bounds")
            if (
                sum(
                    component_count
                    for _raw_count, _direct_count, _upper_bound, component_count in self.component_byte_cohorts
                )
                != self.components
            ):
                raise ValueError("scenario component byte cohorts disagree with component count")
            if (
                sum(
                    raw_count * component_count
                    for raw_count, _direct_count, _upper_bound, component_count in self.component_byte_cohorts
                )
                != self.expanded_candidates
            ):
                raise ValueError("scenario component byte cohorts disagree with expanded candidate count")
            if (
                sum(
                    direct_candidate_count * component_count
                    for _raw_count, direct_candidate_count, _upper_bound, component_count in self.component_byte_cohorts
                )
                != self.direct_candidates
            ):
                raise ValueError("scenario component byte cohorts disagree with direct candidate count")
            if self.component_cohorts is not None:
                component_marginals: dict[tuple[int, int], int] = {}
                for raw_count, direct_candidate_count, component_count in self.component_cohorts:
                    component_marginals[(raw_count, direct_candidate_count)] = component_count
                byte_marginals: dict[tuple[int, int], int] = {}
                for raw_count, direct_candidate_count, _upper_bound, component_count in self.component_byte_cohorts:
                    key = (raw_count, direct_candidate_count)
                    byte_marginals[key] = byte_marginals.get(key, 0) + component_count
                if byte_marginals != component_marginals:
                    raise ValueError("scenario component and byte cohort marginals disagree")

    @classmethod
    def from_profile(cls, profile: object) -> RawAuthorityScaleScenario:
        if not isinstance(profile, dict) or profile.get("format") != "raw-authority-scale-profile-v1":
            raise ValueError("scenario profile must be a raw-authority-scale-profile-v1 document")

        def count(field: str) -> int:
            value = profile.get(field)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"scenario profile has no integral {field}")
            return value

        cohort_payload = profile.get("component_cohort_distribution")
        cohorts: tuple[tuple[int, int, int], ...] | None = None
        if cohort_payload is not None:
            if not isinstance(cohort_payload, list):
                raise ValueError("scenario profile component cohorts must be a list")
            parsed_cohorts: list[tuple[int, int, int]] = []
            for item in cohort_payload:
                if not isinstance(item, dict):
                    raise ValueError("scenario profile component cohort must be an object")
                raw_count = item.get("component_raw_count")
                direct_candidate_count = item.get("direct_candidate_count")
                component_count = item.get("component_count")
                if (
                    not isinstance(raw_count, int)
                    or isinstance(raw_count, bool)
                    or not isinstance(direct_candidate_count, int)
                    or isinstance(direct_candidate_count, bool)
                    or not isinstance(component_count, int)
                    or isinstance(component_count, bool)
                ):
                    raise ValueError("scenario profile component cohort values must be integral")
                parsed_cohorts.append((raw_count, direct_candidate_count, component_count))
            cohorts = tuple(parsed_cohorts)

        byte_cohort_payload = profile.get("component_byte_cohort_distribution")
        byte_cohorts: tuple[tuple[int, int, int, int], ...] | None = None
        if byte_cohort_payload is not None:
            if not isinstance(byte_cohort_payload, list):
                raise ValueError("scenario profile component byte cohorts must be a list")
            parsed_byte_cohorts: list[tuple[int, int, int, int]] = []
            for item in byte_cohort_payload:
                if not isinstance(item, dict):
                    raise ValueError("scenario profile component byte cohort must be an object")
                values = (
                    item.get("component_raw_count"),
                    item.get("direct_candidate_count"),
                    item.get("upper_bound_blob_bytes"),
                    item.get("component_count"),
                )
                if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
                    raise ValueError("scenario profile component byte cohort values must be integral")
                parsed_byte_cohorts.append(cast(tuple[int, int, int, int], values))
            byte_cohorts = tuple(parsed_byte_cohorts)

        return cls(
            components=count("authority_component_count"),
            direct_candidates=count("candidate_count"),
            expanded_candidates=count("expanded_candidate_count"),
            total_payload_bytes=count("expanded_total_blob_bytes"),
            component_cohorts=cohorts,
            component_byte_cohorts=byte_cohorts,
            terminal_sibling_outcome=str(profile.get("terminal_sibling_outcome", "terminal")),
        )


def _write_payload(path: Path, *, native_id: str, revision: int, target_size: int, previous: Path | None) -> None:
    """Stream a prefix-related Codex JSONL raw without retaining its body in memory."""
    header = (
        f'{{"type":"session_meta","payload":{{"id":"{native_id}","timestamp":"2026-07-15T00:00:00Z"}}}}\n'
        if previous is None
        else f'{{"type":"response_item","payload":{{"type":"message","id":"{native_id}-{revision}","role":"user","content":[{{"type":"input_text","text":"revision-{revision}"}}]}}}}\n'
    ).encode()
    previous_size = previous.stat().st_size if previous is not None else 0
    if target_size < previous_size + len(header):
        raise ValueError("scenario payload allocation cannot preserve valid JSONL evidence")
    with path.open("wb") as destination:
        if previous is not None:
            with previous.open("rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
        destination.write(header)
        remaining = target_size - previous_size - len(header)
        chunk = b" " * min(1024 * 1024, remaining)
        while remaining:
            amount = min(len(chunk), remaining)
            destination.write(chunk[:amount])
            remaining -= amount


def _independent_payload(*, native_id: str, target_size: int) -> bytes:
    """Build one bounded standalone JSONL raw without a disk staging file."""
    header = f'{{"type":"session_meta","payload":{{"id":"{native_id}","timestamp":"2026-07-15T00:00:00Z"}}}}\n'.encode()
    if target_size < len(header):
        raise ValueError("scenario payload allocation cannot preserve valid JSONL evidence")
    return header + (b" " * (target_size - len(header)))


def _component_counts(total: int, *, components: int) -> list[int]:
    counts = [0] * components
    for index in range(total):
        counts[index % components] += 1
    return counts


def _component_authority_shape(scenario: RawAuthorityScaleScenario) -> list[tuple[int, int]]:
    """Return direct-candidate and pre-materialized sibling counts per component."""
    if scenario.component_byte_cohorts is not None:
        return [
            (direct_candidate_count, raw_count - direct_candidate_count)
            for raw_count, direct_candidate_count, _upper_bound, component_count in scenario.component_byte_cohorts
            for _ in range(component_count)
        ]
    if scenario.component_cohorts is not None:
        return [
            (direct_candidate_count, raw_count - direct_candidate_count)
            for raw_count, direct_candidate_count, component_count in scenario.component_cohorts
            for _ in range(component_count)
        ]
    direct = _component_counts(scenario.direct_candidates, components=scenario.components)
    siblings = _component_counts(
        scenario.expanded_candidates - scenario.direct_candidates,
        components=scenario.components,
    )
    return list(zip(direct, siblings, strict=True))


def _uses_independent_component_members(scenario: RawAuthorityScaleScenario) -> bool:
    """Whether explicit cohort outcomes require standalone raw identities."""
    return scenario.component_cohorts is not None or scenario.component_byte_cohorts is not None


def _component_byte_upper_bounds(scenario: RawAuthorityScaleScenario) -> list[int] | None:
    """Return one sanitized byte-bucket cap for each generated component."""
    if scenario.component_byte_cohorts is None:
        return None
    return [
        upper_bound_blob_bytes
        for _raw_count, _direct_count, upper_bound_blob_bytes, component_count in scenario.component_byte_cohorts
        for _ in range(component_count)
    ]


def _row_sizes(
    scenario: RawAuthorityScaleScenario,
    component_counts: list[int],
    *,
    component_byte_upper_bounds: list[int] | None,
) -> list[list[int]]:
    """Give each component a valid evidence budget while preserving byte buckets."""
    minimum_rows = [
        [256 * (revision + 1) for revision in range(count)]
        if not _uses_independent_component_members(scenario)
        else [256] * count
        for count in component_counts
    ]
    minimum_component_bytes = [sum(rows) for rows in minimum_rows]
    if component_byte_upper_bounds is None:
        target_component_bytes = list(minimum_component_bytes)
        remaining = scenario.total_payload_bytes - sum(target_component_bytes)
        if remaining < 0:
            raise ValueError("scenario payload budget cannot preserve the requested revision topology")
        terminal_extra, remainder = divmod(remaining, scenario.components)
        for component, target in enumerate(target_component_bytes):
            target_component_bytes[component] = target + terminal_extra + (1 if component < remainder else 0)
    else:
        if len(component_byte_upper_bounds) != scenario.components:
            raise ValueError("scenario component byte bounds disagree with component count")
        target_component_bytes = []
        capacities: list[int] = []
        for minimum, upper_bound in zip(minimum_component_bytes, component_byte_upper_bounds, strict=True):
            lower_bound = (upper_bound // 2) + 1
            target = max(minimum, lower_bound)
            if target > upper_bound:
                raise ValueError("scenario byte bucket cannot fit the requested component topology")
            target_component_bytes.append(target)
            capacities.append(upper_bound - target)
        remaining = scenario.total_payload_bytes - sum(target_component_bytes)
        if remaining < 0 or remaining > sum(capacities):
            raise ValueError("scenario payload budget cannot preserve the requested byte-bucket distribution")
        for component, capacity in enumerate(capacities):
            addition = min(remaining, capacity)
            target_component_bytes[component] += addition
            remaining -= addition
        if remaining:
            raise RuntimeError("component byte allocation did not consume its exact payload budget")
    if sum(target_component_bytes) != scenario.total_payload_bytes:
        raise ValueError("scenario payload budget cannot preserve the requested revision topology")
    sizes: list[list[int]] = []
    for minimums, target in zip(minimum_rows, target_component_bytes, strict=True):
        rows = list(minimums)
        extra_bytes = target - sum(rows)
        if _uses_independent_component_members(scenario):
            extra_per_member, remainder = divmod(extra_bytes, len(rows))
            for member in range(len(rows)):
                rows[member] += extra_per_member + (1 if member < remainder else 0)
        else:
            rows[-1] += extra_bytes
        sizes.append(rows)
    return sizes


def _rss_bytes() -> int:
    # Linux ru_maxrss is KiB; this repository's supported development host is Linux.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _proc_kb(path: Path, keys: set[str]) -> dict[str, int]:
    values: dict[str, int] = {}
    with contextlib.suppress(OSError):
        for line in path.read_text().splitlines():
            key, separator, raw = line.partition(":")
            if separator and key in keys:
                with contextlib.suppress(ValueError, IndexError):
                    values[key] = int(raw.split()[0])
    return values


def _pressure_full_avg10(kind: str) -> float | None:
    with contextlib.suppress(OSError, ValueError):
        for line in Path(f"/proc/pressure/{kind}").read_text().splitlines():
            fields = line.split()
            if not fields or fields[0] != "full":
                continue
            for field in fields[1:]:
                key, _, raw = field.partition("=")
                if key == "avg10":
                    return float(raw)
    return None


def _process_sample() -> ProcessSample:
    pid = os.getpid()
    status = _proc_kb(Path(f"/proc/{pid}/status"), {"VmRSS", "VmSwap"})
    smaps = _proc_kb(Path(f"/proc/{pid}/smaps_rollup"), {"Pss", "SwapPss"})
    process_io = _proc_kb(Path(f"/proc/{pid}/io"), {"read_bytes", "write_bytes"})
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return ProcessSample(
        rss_bytes=int(status.get("VmRSS", 0)) * 1024,
        pss_bytes=smaps.get("Pss", 0) * 1024 if "Pss" in smaps else None,
        swap_bytes=smaps.get("SwapPss", 0) * 1024 if "SwapPss" in smaps else None,
        cpu_ms=round((usage.ru_utime + usage.ru_stime) * 1000),
        read_io_bytes=process_io.get("read_bytes"),
        write_io_bytes=process_io.get("write_bytes"),
        io_full_avg10=_pressure_full_avg10("io"),
        memory_full_avg10=_pressure_full_avg10("memory"),
    )


def _delta(after: int | None, before: int | None) -> int | None:
    if after is None or before is None:
        return None
    return max(0, after - before)


def _assert_admission(
    sample: ProcessSample, *, max_io_full_avg10: float | None, max_memory_full_avg10: float | None
) -> None:
    if max_io_full_avg10 is not None and (value := sample.io_full_avg10) is not None and value > max_io_full_avg10:
        raise RuntimeError(
            f"I/O pressure gate refused raw-authority scale proof: full avg10={value:.2f} > {max_io_full_avg10:.2f}"
        )
    if (
        max_memory_full_avg10 is not None
        and (value := sample.memory_full_avg10) is not None
        and value > max_memory_full_avg10
    ):
        raise RuntimeError(
            "memory pressure gate refused raw-authority scale proof: "
            f"full avg10={value:.2f} > {max_memory_full_avg10:.2f}"
        )


def _generated_archive_id(rows: list[tuple[str, int, str]]) -> str:
    """Bind the receipt to corpus content rather than its temporary directory."""
    manifest = {
        "format": "raw-authority-scale-proof-v1",
        "rows": [
            {"blob_hash": blob_hash, "native_id": native_id, "revision": revision}
            for native_id, revision, blob_hash in rows
        ],
    }
    encoded = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return f"raw-authority-scale:{hashlib.sha256(encoded).hexdigest()}"


def _requested_shape(scenario: RawAuthorityScaleScenario, *, pass_limit: int) -> dict[str, object]:
    """Serialize the scenario without exposing an implementation-only null field."""
    payload: dict[str, object] = {
        "components": scenario.components,
        "direct_candidates": scenario.direct_candidates,
        "expanded_candidates": scenario.expanded_candidates,
        "total_payload_bytes": scenario.total_payload_bytes,
        "pass_limit": pass_limit,
    }
    if scenario.component_cohorts is not None:
        payload["component_cohort_distribution"] = [
            {
                "component_raw_count": raw_count,
                "direct_candidate_count": direct_candidate_count,
                "component_count": component_count,
            }
            for raw_count, direct_candidate_count, component_count in scenario.component_cohorts
        ]
        payload["terminal_sibling_outcome"] = scenario.terminal_sibling_outcome
    if scenario.component_byte_cohorts is not None:
        payload["component_byte_cohort_distribution"] = [
            {
                "component_raw_count": raw_count,
                "direct_candidate_count": direct_candidate_count,
                "upper_bound_blob_bytes": upper_bound_blob_bytes,
                "component_count": component_count,
            }
            for raw_count, direct_candidate_count, upper_bound_blob_bytes, component_count in scenario.component_byte_cohorts
        ]
        payload["terminal_sibling_outcome"] = scenario.terminal_sibling_outcome
    return payload


def _record_repair_pass(
    *,
    number: int,
    mode: str,
    config: Config,
    pass_limit: int,
    max_payload_bytes: int,
) -> tuple[RawAuthorityScalePass, str]:
    """Run one real repair/census pass and reject incomplete evidence."""
    before = _process_sample()
    started = time.perf_counter()
    result = repair.repair_raw_materialization(
        config,
        raw_artifact_limit=pass_limit,
        max_payload_bytes=max_payload_bytes,
        dry_run=mode == "dry_run",
    )
    wall_ms = int((time.perf_counter() - started) * 1000)
    after = _process_sample()
    metrics = result.metrics
    candidate_value = metrics.get("raw_materialization_candidate_count")
    executable_candidate_value = metrics.get("raw_materialization_executable_candidate_count", candidate_value)
    if (
        not isinstance(candidate_value, int | float)
        or isinstance(candidate_value, bool)
        or not float(candidate_value).is_integer()
        or candidate_value < 0
    ):
        raise RuntimeError(
            "raw-authority scale proof requires a non-negative integral raw-materialization candidate count"
        )
    if (
        not isinstance(executable_candidate_value, int | float)
        or isinstance(executable_candidate_value, bool)
        or not float(executable_candidate_value).is_integer()
        or executable_candidate_value < 0
        or executable_candidate_value > candidate_value
    ):
        raise RuntimeError("raw-authority scale proof requires a bounded integral executable-candidate count")
    if result.census_receipt is None:
        raise RuntimeError("raw-authority scale proof requires a census receipt for every replay pass")
    receipt = result.census_receipt
    expected_modes = {"apply", "census"} if mode == "apply" else {"dry_run"}
    if receipt.mode not in expected_modes:
        raise RuntimeError(f"raw-authority scale proof expected {mode} census evidence, received {receipt.mode}")
    return (
        RawAuthorityScalePass(
            number=number,
            mode=receipt.mode,
            candidate_count=int(candidate_value),
            executable_candidate_count=int(executable_candidate_value),
            repaired_count=result.repaired_count,
            executable_component_count=int(metrics.get("raw_materialization_selected_executable_component_count", 0)),
            fixed_point=receipt.fixed_point,
            plan_status_counts={
                status.value: sum(outcome.status is status for outcome in result.plan_outcomes)
                for status in RawReplayPlanStatus
            },
            wall_ms=wall_ms,
            peak_rss_bytes=max(_rss_bytes(), before.rss_bytes, after.rss_bytes),
            peak_pss_bytes=max(value for value in (before.pss_bytes, after.pss_bytes) if value is not None)
            if before.pss_bytes is not None or after.pss_bytes is not None
            else None,
            peak_swap_bytes=max(value for value in (before.swap_bytes, after.swap_bytes) if value is not None)
            if before.swap_bytes is not None or after.swap_bytes is not None
            else None,
            cpu_ms=_delta(after.cpu_ms, before.cpu_ms),
            read_io_bytes=_delta(after.read_io_bytes, before.read_io_bytes),
            write_io_bytes=_delta(after.write_io_bytes, before.write_io_bytes),
        ),
        receipt.residual_digest,
    )


def run_raw_authority_scale_proof(
    workdir: Path,
    *,
    components: int = 16,
    raws: int = 24,
    scenario: RawAuthorityScaleScenario | None = None,
    pass_limit: int = 4,
    keep: bool = False,
    prepare_only: bool = False,
    max_payload_bytes: int = repair.RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES,
    max_io_full_avg10: float | None = 2.0,
    max_memory_full_avg10: float | None = 2.0,
) -> dict[str, object]:
    """Exercise real authority census/replay and emit a stable receipt payload."""
    if pass_limit < 1:
        raise ValueError("pass_limit must be positive")
    if max_payload_bytes < 1:
        raise ValueError("max_payload_bytes must be positive")
    if scenario is None:
        if components < 1 or raws < components:
            raise ValueError("require components >= 1 and raws >= components")
        scenario = RawAuthorityScaleScenario(
            components=components,
            direct_candidates=raws,
            expanded_candidates=raws,
            total_payload_bytes=raws * 1024,
        )
    if components != 16 and components != scenario.components:
        raise ValueError("components and scenario.components disagree")
    if raws != 24 and raws != scenario.direct_candidates:
        raise ValueError("raws and scenario.direct_candidates disagree")
    if scenario.components < 1:
        raise ValueError("require components >= 1, raws >= components, and pass_limit >= 1")
    admission_sample = _process_sample()
    _assert_admission(
        admission_sample,
        max_io_full_avg10=max_io_full_avg10,
        max_memory_full_avg10=max_memory_full_avg10,
    )
    generation_samples = [admission_sample]

    def check_generation_pressure() -> None:
        sample = _process_sample()
        _assert_admission(
            sample,
            max_io_full_avg10=max_io_full_avg10,
            max_memory_full_avg10=max_memory_full_avg10,
        )
        generation_samples.append(sample)

    root = workdir.expanduser().resolve() / "raw-authority-scale-proof"
    if root.exists():
        shutil.rmtree(root)
    initialize_active_archive_root(root)
    component_shape = _component_authority_shape(scenario)
    component_sizes = _row_sizes(
        scenario,
        [direct_count + sibling_count for direct_count, sibling_count in component_shape],
        component_byte_upper_bounds=_component_byte_upper_bounds(scenario),
    )
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        publisher = archive._blob_publisher
        if publisher is None:
            raise RuntimeError("raw-authority scale proof requires a writable blob publisher")
        source_conn = archive._ensure_source_conn()
        generated_rows: list[tuple[str, int, str]] = []
        pending_rows: list[tuple[str, str, str, int, bool, int]] = []
        acquired_at_ms = 0
        staging = root / "payload-staging"
        staging.mkdir()
        with tempfile.TemporaryDirectory(dir=staging) as temporary_directory:
            temporary_root = Path(temporary_directory)
            for component, ((direct_count, sibling_count), row_sizes) in enumerate(
                zip(component_shape, component_sizes, strict=True)
            ):
                session_native_id = f"scale-authority-component-{component:05d}"
                component_source_path = f"/synthetic/raw-authority/authority-component-{component:05d}.jsonl"
                row_kinds = [False] * direct_count + [True] * sibling_count
                previous: Path | None = None
                for member, (row_size, terminalized) in enumerate(zip(row_sizes, row_kinds, strict=True)):
                    source_label = f"scale-{component:05d}-member-{member:05d}"
                    row_native_id = (
                        session_native_id
                        if not _uses_independent_component_members(scenario)
                        else f"{session_native_id}-member-{member:05d}"
                    )
                    if _uses_independent_component_members(scenario):
                        blob_hash, blob_size = publisher.write_from_bytes(
                            _independent_payload(native_id=row_native_id, target_size=row_size)
                        )
                    else:
                        payload_path = temporary_root / f"{source_label}.jsonl"
                        _write_payload(
                            payload_path,
                            native_id=row_native_id,
                            revision=member,
                            target_size=row_size,
                            previous=previous,
                        )
                        blob_hash, blob_size = publisher.write_from_path(payload_path)
                        payload_path.unlink()
                    previous = publisher.blob_path(blob_hash)
                    source_path = component_source_path
                    pending_rows.append((row_native_id, source_path, blob_hash, blob_size, terminalized, component))
                    generated_rows.append((source_label, member, blob_hash))
                    if len(pending_rows) < 128:
                        continue
                    publisher.flush()
                    check_generation_pressure()
                    with source_conn:
                        for (
                            row_native_id,
                            source_path,
                            row_hash,
                            row_size_bytes,
                            terminalized,
                            row_component,
                        ) in pending_rows:
                            raw_id = write_source_raw_session_blob_ref(
                                source_conn,
                                origin=origin_from_provider(Provider.CODEX),
                                capture_mode=Provider.CODEX,
                                source_path=source_path,
                                source_index=row_component,
                                blob_hash=bytes.fromhex(row_hash),
                                blob_size=row_size_bytes,
                                acquired_at_ms=acquired_at_ms,
                                native_id=None if terminalized else row_native_id,
                                blob_publication_receipt_id=publisher.receipt_id(row_hash),
                                manage_transaction=False,
                            )
                            if terminalized:
                                with sqlite3.connect(root / "index.db") as index_conn:
                                    index_conn.execute(
                                        """
                                        INSERT INTO raw_revision_applications (
                                            decision_id, raw_id, session_id, logical_source_key,
                                            source_revision, acquisition_generation, decision, detail, decided_at_ms
                                        ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, 0)
                                        """,
                                        (
                                            f"synthetic-terminal:{raw_id}",
                                            raw_id,
                                            f"codex-session:{row_native_id}",
                                            f"synthetic:authority-component:{row_component:05d}",
                                            row_hash,
                                            "ambiguous"
                                            if scenario.terminal_sibling_outcome == "terminal"
                                            else "deferred",
                                            f"synthetic {scenario.terminal_sibling_outcome} sibling",
                                        ),
                                    )
                            acquired_at_ms += 1
                    pending_rows.clear()
                    check_generation_pressure()
                    # Publication atomically moves prepared blobs.  Continue
                    # the prefix chain from the now-stable final blob rather
                    # than the moved temporary path, so large components can
                    # flush without retaining thousands of prepared blobs.
                    previous = publisher.blob_path(blob_hash)
            if pending_rows:
                publisher.flush()
                check_generation_pressure()
                with source_conn:
                    for (
                        row_native_id,
                        source_path,
                        row_hash,
                        row_size_bytes,
                        terminalized,
                        row_component,
                    ) in pending_rows:
                        raw_id = write_source_raw_session_blob_ref(
                            source_conn,
                            origin=origin_from_provider(Provider.CODEX),
                            capture_mode=Provider.CODEX,
                            source_path=source_path,
                            source_index=row_component,
                            blob_hash=bytes.fromhex(row_hash),
                            blob_size=row_size_bytes,
                            acquired_at_ms=acquired_at_ms,
                            native_id=None if terminalized else row_native_id,
                            blob_publication_receipt_id=publisher.receipt_id(row_hash),
                            manage_transaction=False,
                        )
                        if terminalized:
                            with sqlite3.connect(root / "index.db") as index_conn:
                                index_conn.execute(
                                    """
                                    INSERT INTO raw_revision_applications (
                                        decision_id, raw_id, session_id, logical_source_key,
                                        source_revision, acquisition_generation, decision, detail, decided_at_ms
                                    ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, 0)
                                    """,
                                    (
                                        f"synthetic-terminal:{raw_id}",
                                        raw_id,
                                        f"codex-session:{row_native_id}",
                                        f"synthetic:authority-component:{row_component:05d}",
                                        row_hash,
                                        "ambiguous" if scenario.terminal_sibling_outcome == "terminal" else "deferred",
                                        f"synthetic {scenario.terminal_sibling_outcome} sibling",
                                    ),
                                )
                        acquired_at_ms += 1
                check_generation_pressure()
        staging.rmdir()
    archive_id = _generated_archive_id(generated_rows)
    config = Config(archive_root=root, render_root=root, sources=[], db_path=root / "index.db")
    achieved_shape = repair.raw_materialization_scale_profile(config)
    if prepare_only:
        prepared_report: dict[str, object] = {
            "archive_root": str(root),
            "requested_shape": _requested_shape(scenario, pass_limit=pass_limit),
            "achieved_shape": achieved_shape,
            "admission_sample": asdict(admission_sample),
            "generation_samples": [asdict(sample) for sample in generation_samples],
            "prepared_only": True,
        }
        (root / "raw-authority-scale-preflight.json").write_text(
            json.dumps(prepared_report, indent=2, sort_keys=True) + "\n"
        )
        if not keep:
            shutil.rmtree(root)
        return prepared_report
    if scenario.expanded_candidates != scenario.direct_candidates and not _uses_independent_component_members(scenario):
        raise ValueError(
            "an expanded-authority scenario requires explicit terminal/deferred cohort outcomes; "
            "use prepare_only to inspect its exact topology until those outcome variants are implemented"
        )
    pass_receipts: list[RawAuthorityScalePass] = []
    for number in range(1, (scenario.components * 3) + 4):
        pass_receipt, _digest = _record_repair_pass(
            number=number,
            mode="apply",
            config=config,
            pass_limit=pass_limit,
            max_payload_bytes=max_payload_bytes,
        )
        pass_receipts.append(pass_receipt)
        if pass_receipt.executable_candidate_count == 0:
            break
    else:
        raise RuntimeError("raw-authority scale proof did not drain bounded apply passes")
    fixed_point_digests: list[str] = []
    for _ in range(2):
        pass_receipt, digest = _record_repair_pass(
            number=len(pass_receipts) + 1,
            mode="dry_run",
            config=config,
            pass_limit=pass_limit,
            max_payload_bytes=max_payload_bytes,
        )
        if pass_receipt.executable_candidate_count != 0:
            raise RuntimeError("raw-authority scale proof lost quiescence during fixed-point confirmation")
        pass_receipts.append(pass_receipt)
        fixed_point_digests.append(digest)
    if fixed_point_digests[0] != fixed_point_digests[1] or not pass_receipts[-1].fixed_point:
        raise RuntimeError("raw-authority scale proof did not reach two matching quiescent fixed-point censuses")
    profile_id = (
        f"workload-profile:synthetic-raw-authority:{scenario.components}:"
        f"{scenario.direct_candidates}:{scenario.expanded_candidates}"
    )
    spec = raw_authority_fixed_point_spec(
        profile_id=profile_id,
        archive_id=archive_id,
        scale_tier=WorkloadScaleTier.CI_ACTIVATION,
    )
    total_wall_ms = sum(item.wall_ms for item in pass_receipts)
    peak_rss = max(item.peak_rss_bytes for item in pass_receipts)
    peak_pss = max((item.peak_pss_bytes for item in pass_receipts if item.peak_pss_bytes is not None), default=None)
    peak_swap = max((item.peak_swap_bytes for item in pass_receipts if item.peak_swap_bytes is not None), default=None)
    total_cpu_ms = sum(item.cpu_ms or 0 for item in pass_receipts)
    total_read_io = sum(item.read_io_bytes or 0 for item in pass_receipts)
    total_write_io = sum(item.write_io_bytes or 0 for item in pass_receipts)
    receipt = WorkloadReceipt.from_observations(
        spec=spec,
        status=WorkloadRunStatus.SUCCEEDED,
        build_id="git:local",
        runtime_id=f"python:{platform.python_version()}",
        archive_id=archive_id,
        generation_id=None,
        frame_id=None,
        phases=(
            WorkloadPhaseObservation(name="generate"),
            WorkloadPhaseObservation(name="acquire"),
            WorkloadPhaseObservation(name="census"),
            WorkloadPhaseObservation(
                name="replay",
                wall_ms=total_wall_ms,
                cpu_ms=total_cpu_ms,
                peak_rss_bytes=peak_rss,
                peak_pss_bytes=peak_pss,
                swap_bytes=peak_swap,
                read_io_bytes=total_read_io,
                write_io_bytes=total_write_io,
            ),
            WorkloadPhaseObservation(name="postflight"),
            WorkloadPhaseObservation(name="quiescent", cleanup_complete=not keep, quiescent=True),
        ),
        cleanup_complete=not keep,
        notes=(
            f"requested_components={scenario.components}",
            f"requested_direct_candidates={scenario.direct_candidates}",
            f"requested_expanded_candidates={scenario.expanded_candidates}",
            f"requested_payload_bytes={scenario.total_payload_bytes}",
            f"resource_envelope_bytes={max_payload_bytes}",
            "repair_raw_materialization combines census and replay; its measured resource totals are recorded once on replay.",
            "This receipt is a generated projection; only an explicit July-15-sized invocation is production-shaped evidence.",
        ),
    )
    report: dict[str, object] = {
        "archive_root": str(root),
        "requested_shape": _requested_shape(scenario, pass_limit=pass_limit),
        "achieved_shape": achieved_shape,
        "admission_sample": asdict(admission_sample),
        "generation_samples": [asdict(sample) for sample in generation_samples],
        "passes": [asdict(item) for item in pass_receipts],
        "fixed_point_digests": fixed_point_digests,
        "receipt": receipt.to_payload(),
    }
    (root / "raw-authority-scale-receipt.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not keep:
        shutil.rmtree(root)
    return report


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path(".cache") / "raw-authority-scale-proof")
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--raws", type=int, default=24)
    parser.add_argument(
        "--scenario-profile",
        type=Path,
        default=None,
        help="Read an aggregate scale profile produced by --capture-profile and generate that synthetic shape.",
    )
    parser.add_argument("--pass-limit", type=int, default=4)
    parser.add_argument(
        "--max-payload-bytes",
        type=int,
        default=repair.RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES,
        help="Bound each authority component replay; deferred components remain typed residual debt.",
    )
    parser.add_argument("--keep", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build and inspect a private-free synthetic frontier without asserting replay convergence.",
    )
    parser.add_argument("--max-io-full-avg10", type=float, default=2.0)
    parser.add_argument("--max-memory-full-avg10", type=float, default=2.0)
    parser.add_argument("--allow-contended-host", action="store_true")
    parser.add_argument(
        "--capture-profile",
        type=Path,
        default=None,
        help="Write a read-only aggregate frontier profile for a private-free synthetic scenario, then exit.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.capture_profile is not None:
        profile = repair.raw_materialization_scale_profile(get_config())
        args.capture_profile.parent.mkdir(parents=True, exist_ok=True)
        args.capture_profile.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
        print(json.dumps(profile, indent=2, sort_keys=True) if args.json else args.capture_profile, file=stdout)
        return 0
    scenario = None
    if args.scenario_profile is not None:
        scenario = RawAuthorityScaleScenario.from_profile(json.loads(args.scenario_profile.read_text()))
    pressure_limit = None if args.allow_contended_host else args.max_io_full_avg10
    memory_limit = None if args.allow_contended_host else args.max_memory_full_avg10
    payload = run_raw_authority_scale_proof(
        args.workdir,
        components=args.components,
        raws=args.raws,
        scenario=scenario,
        pass_limit=args.pass_limit,
        keep=args.keep,
        prepare_only=args.prepare_only,
        max_payload_bytes=args.max_payload_bytes,
        max_io_full_avg10=pressure_limit,
        max_memory_full_avg10=memory_limit,
    )
    out = stdout
    if out is None:
        import sys

        out = sys.stdout
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True), file=out)
    elif args.prepare_only:
        print(payload["archive_root"], file=out)
    else:
        receipt = cast(dict[str, object], payload["receipt"])
        print(receipt["receipt_id"], file=out)
    return 0


__all__ = ["main", "run_raw_authority_scale_proof"]


if __name__ == "__main__":
    raise SystemExit(main())
