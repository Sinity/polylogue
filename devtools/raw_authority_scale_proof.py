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
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session


@dataclass(frozen=True, slots=True)
class RawAuthorityScalePass:
    number: int
    mode: str
    candidate_count: int
    repaired_count: int
    executable_component_count: int
    fixed_point: bool
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


def _payload(native_id: str, revision: int) -> bytes:
    records = [f'{{"type":"session_meta","payload":{{"id":"{native_id}","timestamp":"2026-07-15T00:00:00Z"}}}}\n']
    records.extend(
        f'{{"type":"response_item","payload":{{"type":"message","id":"{native_id}-{index}","role":"user","content":[{{"type":"input_text","text":"revision-{index}"}}]}}}}\n'
        for index in range(revision + 1)
    )
    return "".join(records).encode()


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


def _record_repair_pass(
    *,
    number: int,
    mode: str,
    config: Config,
    pass_limit: int,
) -> tuple[RawAuthorityScalePass, str]:
    """Run one real repair/census pass and reject incomplete evidence."""
    before = _process_sample()
    started = time.perf_counter()
    result = repair.repair_raw_materialization(
        config,
        raw_artifact_limit=pass_limit,
        dry_run=mode == "dry_run",
    )
    wall_ms = int((time.perf_counter() - started) * 1000)
    after = _process_sample()
    metrics = result.metrics
    candidate_value = metrics.get("raw_materialization_candidate_count")
    if (
        not isinstance(candidate_value, int | float)
        or isinstance(candidate_value, bool)
        or not float(candidate_value).is_integer()
        or candidate_value < 0
    ):
        raise RuntimeError(
            "raw-authority scale proof requires a non-negative integral raw-materialization candidate count"
        )
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
            repaired_count=result.repaired_count,
            executable_component_count=int(metrics.get("raw_materialization_selected_executable_component_count", 0)),
            fixed_point=receipt.fixed_point,
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
    pass_limit: int = 4,
    keep: bool = False,
    max_io_full_avg10: float | None = 2.0,
    max_memory_full_avg10: float | None = 2.0,
) -> dict[str, object]:
    """Exercise real authority census/replay and emit a stable receipt payload."""
    if components < 1 or raws < components or pass_limit < 1:
        raise ValueError("require components >= 1, raws >= components, and pass_limit >= 1")
    admission_sample = _process_sample()
    _assert_admission(
        admission_sample,
        max_io_full_avg10=max_io_full_avg10,
        max_memory_full_avg10=max_memory_full_avg10,
    )
    root = workdir.expanduser().resolve() / "raw-authority-scale-proof"
    if root.exists():
        shutil.rmtree(root)
    initialize_active_archive_root(root)
    revision_counts = [1] * components
    for index in range(raws - components):
        revision_counts[index % components] += 1
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        publisher = archive._blob_publisher
        if publisher is None:
            raise RuntimeError("raw-authority scale proof requires a writable blob publisher")
        source_conn = archive._ensure_source_conn()
        generated_rows: list[tuple[str, int, str]] = []
        for component, revisions in enumerate(revision_counts):
            native_id = f"scale-{component:05d}"
            for revision in range(revisions):
                blob_hash, _blob_size = publisher.write_from_bytes(_payload(native_id, revision))
                generated_rows.append((native_id, revision, blob_hash))
        publisher.flush()
        acquired_at_ms = 0
        with source_conn:
            for native_id, revision, blob_hash in generated_rows:
                write_source_raw_session(
                    source_conn,
                    origin=origin_from_provider(Provider.CODEX),
                    capture_mode=Provider.CODEX,
                    source_path=f"/synthetic/raw-authority/{native_id}.jsonl",
                    source_index=revision,
                    payload=_payload(native_id, revision),
                    acquired_at_ms=acquired_at_ms,
                    blob_publication_receipt_id=publisher.receipt_id(blob_hash),
                    manage_transaction=False,
                )
                acquired_at_ms += 1
    archive_id = _generated_archive_id(generated_rows)
    config = Config(archive_root=root, render_root=root, sources=[], db_path=root / "index.db")
    pass_receipts: list[RawAuthorityScalePass] = []
    for number in range(1, (components * 3) + 4):
        pass_receipt, _digest = _record_repair_pass(
            number=number,
            mode="apply",
            config=config,
            pass_limit=pass_limit,
        )
        pass_receipts.append(pass_receipt)
        if pass_receipt.candidate_count == 0:
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
        )
        if pass_receipt.candidate_count != 0:
            raise RuntimeError("raw-authority scale proof lost quiescence during fixed-point confirmation")
        pass_receipts.append(pass_receipt)
        fixed_point_digests.append(digest)
    if fixed_point_digests[0] != fixed_point_digests[1] or not pass_receipts[-1].fixed_point:
        raise RuntimeError("raw-authority scale proof did not reach two matching quiescent fixed-point censuses")
    profile_id = f"workload-profile:synthetic-raw-authority:{components}:{raws}"
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
            f"requested_components={components}",
            f"requested_raws={raws}",
            "repair_raw_materialization combines census and replay; its measured resource totals are recorded once on replay.",
            "This receipt is a generated projection; only an explicit July-15-sized invocation is production-shaped evidence.",
        ),
    )
    report: dict[str, object] = {
        "archive_root": str(root),
        "requested_shape": {"components": components, "raws": raws, "pass_limit": pass_limit},
        "admission_sample": asdict(admission_sample),
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
    parser.add_argument("--pass-limit", type=int, default=4)
    parser.add_argument("--keep", action="store_true")
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
    pressure_limit = None if args.allow_contended_host else args.max_io_full_avg10
    memory_limit = None if args.allow_contended_host else args.max_memory_full_avg10
    payload = run_raw_authority_scale_proof(
        args.workdir,
        components=args.components,
        raws=args.raws,
        pass_limit=args.pass_limit,
        keep=args.keep,
        max_io_full_avg10=pressure_limit,
        max_memory_full_avg10=memory_limit,
    )
    out = stdout
    if out is None:
        import sys

        out = sys.stdout
    receipt = cast(dict[str, object], payload["receipt"])
    print(json.dumps(payload, indent=2, sort_keys=True) if args.json else receipt["receipt_id"], file=out)
    return 0


__all__ = ["main", "run_raw_authority_scale_proof"]


if __name__ == "__main__":
    raise SystemExit(main())
