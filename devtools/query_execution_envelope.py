"""Measure repeated incident-scale query resource envelopes.

The command is an opt-in lab check. It opens the supplied archive through the
public query route and never writes to the archive.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import disk_usage
from typing import Any

from polylogue import Polylogue
from polylogue.core.errors import SchemaVersionMismatchError

DEFAULT_EXPRESSION = "actions where tool:shell | group by tool | count"
DEFAULT_ROUNDS = 20
DEFAULT_WARMUP = 3
DEFAULT_BASELINE = 5
DEFAULT_TOLERANCE = 0.25
DEFAULT_MAX_RSS_MB = 1536
DEFAULT_MAX_PSS_MB = 1536
DEFAULT_MAX_SWAP_GROWTH_MB = 64
DEFAULT_MAX_TEMP_GROWTH_MB = 64
MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ResourceSample:
    """One process and temporary-filesystem observation."""

    rss_bytes: int
    pss_bytes: int
    swap_bytes: int
    temp_delta_bytes: int


PROC_MEMORY_FIELDS = ("VmRSS", "Pss", "VmSwap")


class ResourceProbeUnavailableError(RuntimeError):
    """procfs did not report a field the declared envelope is measured against."""


def _parse_proc_memory(text: str) -> tuple[int, int, int]:
    """Return RSS, PSS, and swap in bytes from concatenated procfs reports.

    A missing field is refused rather than defaulted to zero: a zero sample
    satisfies every declared limit, so the envelope would pass without
    measuring anything.
    """
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key not in PROC_MEMORY_FIELDS:
            continue
        fields = value.split()
        if not fields:
            continue
        values[key] = int(fields[0]) * 1024
    missing = [field for field in PROC_MEMORY_FIELDS if field not in values]
    if missing:
        raise ResourceProbeUnavailableError(f"procfs did not report {', '.join(missing)}")
    return values["VmRSS"], values["Pss"], values["VmSwap"]


def _proc_memory() -> tuple[int, int, int]:
    """Return current RSS, PSS, and swap from procfs for this process."""
    try:
        status = Path("/proc/self/status").read_text(encoding="ascii")
        smaps_rollup = Path("/proc/self/smaps_rollup").read_text(encoding="ascii")
    except (OSError, ValueError) as exc:
        raise ResourceProbeUnavailableError(f"procfs memory reports are unreadable: {exc}") from exc
    return _parse_proc_memory(status + smaps_rollup)


def _temp_used_bytes(temp_root: Path) -> int:
    """Return used bytes on the filesystem containing the temp root."""
    try:
        usage = disk_usage(temp_root)
    except OSError:
        return 0
    return usage.total - usage.free


async def _query_once(archive: Polylogue, expression: str) -> dict[str, Any]:
    envelope = await archive.query_units(expression, limit=100)
    return envelope.model_dump(mode="json")


async def measure_query_envelope(
    archive_root: Path,
    *,
    expression: str = DEFAULT_EXPRESSION,
    rounds: int = DEFAULT_ROUNDS,
    warmup: int = DEFAULT_WARMUP,
    baseline_rounds: int = DEFAULT_BASELINE,
    tolerance: float = DEFAULT_TOLERANCE,
    sample_interval_s: float = 0.05,
    max_rss_bytes: int = DEFAULT_MAX_RSS_MB * MIB,
    max_pss_bytes: int = DEFAULT_MAX_PSS_MB * MIB,
    max_swap_growth_bytes: int = DEFAULT_MAX_SWAP_GROWTH_MB * MIB,
    max_temp_growth_bytes: int = DEFAULT_MAX_TEMP_GROWTH_MB * MIB,
) -> dict[str, Any]:
    """Run repeated aggregate reads and return a resource receipt."""
    if rounds < 20:
        raise ValueError("rounds must be at least 20")
    if warmup < 0 or baseline_rounds < 1:
        raise ValueError("warmup must be non-negative and baseline_rounds must be positive")
    if tolerance < 0 or sample_interval_s < 0:
        raise ValueError("tolerance and sample interval must be non-negative")
    if min(max_rss_bytes, max_pss_bytes, max_swap_growth_bytes, max_temp_growth_bytes) < 0:
        raise ValueError("resource envelope limits must be non-negative")

    archive_root = archive_root.resolve()
    db_path = archive_root / "index.db"
    if not db_path.is_file():
        raise FileNotFoundError(db_path)

    temp_root = Path(os.environ.get("TMPDIR", "/tmp"))
    temp_before = _temp_used_bytes(temp_root)
    initial_rss, initial_pss, initial_swap = _proc_memory()
    initial = ResourceSample(initial_rss, initial_pss, initial_swap, 0)
    peak = initial
    stop = threading.Event()

    def observe() -> ResourceSample:
        nonlocal peak
        rss, pss, swap = _proc_memory()
        candidate = ResourceSample(rss, pss, swap, max(0, _temp_used_bytes(temp_root) - temp_before))
        peak = ResourceSample(
            max(peak.rss_bytes, candidate.rss_bytes),
            max(peak.pss_bytes, candidate.pss_bytes),
            max(peak.swap_bytes, candidate.swap_bytes),
            max(peak.temp_delta_bytes, candidate.temp_delta_bytes),
        )
        return candidate

    def sample() -> None:
        # The measured loop observes every round on this thread too, so a
        # persistent probe failure still reaches the caller.
        while not stop.is_set():
            try:
                observe()
            except ResourceProbeUnavailableError:
                return
            time.sleep(sample_interval_s)

    sampler = threading.Thread(target=sample, name="query-envelope-sampler", daemon=True)
    sampler.start()
    started = time.perf_counter()
    result_counts: list[int] = []
    samples: list[dict[str, Any]] = []
    try:
        async with Polylogue(archive_root=archive_root, db_path=db_path) as archive:
            for phase, count in (("warmup", warmup), ("baseline", baseline_rounds), ("measured", rounds)):
                for round_number in range(count):
                    result_count = len((await _query_once(archive, expression)).get("items", []))
                    result_counts.append(result_count)
                    sample_now = observe()
                    samples.append(
                        {
                            "phase": phase,
                            "round": round_number + 1,
                            "result_item_count": result_count,
                            **asdict(sample_now),
                        }
                    )
        quiescent = observe()
    finally:
        stop.set()
        sampler.join(timeout=2)

    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    baseline = [
        ResourceSample(item["rss_bytes"], item["pss_bytes"], item["swap_bytes"], item["temp_delta_bytes"])
        for item in samples
        if item["phase"] == "baseline"
    ]
    measured = [
        ResourceSample(item["rss_bytes"], item["pss_bytes"], item["swap_bytes"], item["temp_delta_bytes"])
        for item in samples
        if item["phase"] == "measured"
    ]
    baseline_rss = max(sample.rss_bytes for sample in baseline)
    baseline_pss = max(sample.pss_bytes for sample in baseline)
    baseline_swap = max(sample.swap_bytes for sample in baseline)
    baseline_temp = max(sample.temp_delta_bytes for sample in baseline)
    final = measured[-3:]
    return_checks = {
        "rss": all(current.rss_bytes <= max(1, baseline_rss) * (1 + tolerance) for current in final),
        "pss": all(current.pss_bytes <= max(1, baseline_pss) * (1 + tolerance) for current in final),
        "swap": all(current.swap_bytes <= baseline_swap + max_swap_growth_bytes for current in final),
        "temp": all(current.temp_delta_bytes <= baseline_temp + max_temp_growth_bytes for current in final),
    }
    absolute_checks = {
        "rss": peak.rss_bytes <= max_rss_bytes,
        "pss": peak.pss_bytes <= max_pss_bytes,
        "swap": peak.swap_bytes <= initial_swap + max_swap_growth_bytes,
        "temp": peak.temp_delta_bytes <= max_temp_growth_bytes,
    }
    returned = all(return_checks.values()) and all(absolute_checks.values())
    return {
        "status": "succeeded" if returned else "failed",
        "archive_root": str(archive_root),
        "archive_generation": db_path.resolve().parent.name,
        "archive_index_bytes": db_path.stat().st_size,
        "expression": expression,
        "rounds": rounds,
        "warmup_rounds": warmup,
        "baseline_rounds": baseline_rounds,
        "tolerance": tolerance,
        "declared_envelope": {
            "max_rss_bytes": max_rss_bytes,
            "max_pss_bytes": max_pss_bytes,
            "max_swap_growth_bytes": max_swap_growth_bytes,
            "max_temp_growth_bytes": max_temp_growth_bytes,
            "return_tolerance": tolerance,
        },
        "steady_state_baseline": {
            "rss_bytes": baseline_rss,
            "pss_bytes": baseline_pss,
            "swap_bytes": baseline_swap,
            "temp_delta_bytes": baseline_temp,
        },
        "peak": asdict(peak),
        "initial_sample": asdict(initial),
        "quiescent_sample": asdict(quiescent),
        "final_samples": [asdict(sample) for sample in final],
        "samples": samples,
        "return_checks": return_checks,
        "absolute_checks": absolute_checks,
        "result_item_counts": result_counts,
        "elapsed_ms": elapsed_ms,
        "returned_to_envelope": returned,
        "regression_path": (
            "Rerun this command against the same promoted generation. A failed status identifies RSS/PSS, "
            "swap, temp, or return-to-baseline drift; compare the named check and samples."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--baseline-rounds", type=int, default=DEFAULT_BASELINE)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--max-rss-mb", type=int, default=DEFAULT_MAX_RSS_MB)
    parser.add_argument("--max-pss-mb", type=int, default=DEFAULT_MAX_PSS_MB)
    parser.add_argument("--max-swap-growth-mb", type=int, default=DEFAULT_MAX_SWAP_GROWTH_MB)
    parser.add_argument("--max-temp-growth-mb", type=int, default=DEFAULT_MAX_TEMP_GROWTH_MB)
    parser.add_argument("--expression", default=DEFAULT_EXPRESSION)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = asyncio.run(
            measure_query_envelope(
                args.archive_root,
                expression=args.expression,
                rounds=args.rounds,
                warmup=args.warmup,
                baseline_rounds=args.baseline_rounds,
                tolerance=args.tolerance,
                sample_interval_s=args.sample_interval,
                max_rss_bytes=args.max_rss_mb * MIB,
                max_pss_bytes=args.max_pss_mb * MIB,
                max_swap_growth_bytes=args.max_swap_growth_mb * MIB,
                max_temp_growth_bytes=args.max_temp_growth_mb * MIB,
            )
        )
    except (SchemaVersionMismatchError, ResourceProbeUnavailableError) as exc:
        regression_path = (
            "Re-run against the exact promoted generation after its schema lifecycle action completes; "
            "do not bypass the archive compatibility check."
            if isinstance(exc, SchemaVersionMismatchError)
            else "Re-run on a host whose procfs reports VmRSS, Pss, and VmSwap; the envelope is unmeasured here."
        )
        receipt = {
            "status": "blocked-env",
            "archive_root": str(args.archive_root.resolve()),
            "blocking_error": str(exc),
            "regression_path": regression_path,
        }
        text = json.dumps(receipt, indent=2, sort_keys=True)
        print(text)
        if args.receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(text + "\n", encoding="utf-8")
        return 2
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    text = json.dumps(receipt, indent=2, sort_keys=True)
    print(text)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(text + "\n", encoding="utf-8")
    return 0 if receipt["returned_to_envelope"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
