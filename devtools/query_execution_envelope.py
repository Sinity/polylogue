"""Measure repeated incident-scale query resource envelopes.

The command is intentionally an opt-in lab check. It opens the supplied
archive through the public read route and never writes to the archive.
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

DEFAULT_EXPRESSION = "actions where tool:shell | group by tool, session.origin | count"
DEFAULT_ROUNDS = 20
DEFAULT_WARMUP = 3
DEFAULT_BASELINE = 5
DEFAULT_TOLERANCE = 0.25


@dataclass(frozen=True, slots=True)
class ResourceSample:
    rss_bytes: int
    pss_bytes: int
    swap_bytes: int
    temp_delta_bytes: int


def _proc_memory() -> tuple[int, int, int]:
    """Return current RSS, PSS, and swap from procfs for this process."""
    rss = pss = swap = 0
    try:
        for line in (Path("/proc/self/status").read_text() + Path("/proc/self/smaps_rollup").read_text()).splitlines():
            key, _, value = line.partition(":")
            fields = value.split()
            if not fields:
                continue
            if key == "VmRSS":
                rss = int(fields[0]) * 1024
            elif key == "Pss":
                pss = int(fields[0]) * 1024
            elif key == "VmSwap":
                swap = int(fields[0]) * 1024
    except OSError:
        pass
    return rss, pss, swap


def _temp_used_bytes(temp_root: Path) -> int:
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
) -> dict[str, Any]:
    """Run repeated aggregate reads and return a bounded resource receipt."""
    if rounds < 20:
        raise ValueError("rounds must be at least 20")
    if warmup < 0 or baseline_rounds < 1:
        raise ValueError("warmup must be non-negative and baseline_rounds must be positive")
    archive_root = archive_root.resolve()
    db_path = archive_root / "index.db"
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    temp_root = Path(os.environ.get("TMPDIR", "/tmp"))
    temp_before = _temp_used_bytes(temp_root)
    peak = ResourceSample(0, 0, 0, 0)
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not stop.is_set():
            rss, pss, swap = _proc_memory()
            candidate = ResourceSample(rss, pss, swap, max(0, _temp_used_bytes(temp_root) - temp_before))
            peak = ResourceSample(
                max(peak.rss_bytes, candidate.rss_bytes),
                max(peak.pss_bytes, candidate.pss_bytes),
                max(peak.swap_bytes, candidate.swap_bytes),
                max(peak.temp_delta_bytes, candidate.temp_delta_bytes),
            )
            time.sleep(sample_interval_s)

    sampler = threading.Thread(target=sample, name="query-envelope-sampler", daemon=True)
    sampler.start()
    started = time.perf_counter()
    result_counts: list[int] = []
    per_round: list[ResourceSample] = []
    try:
        async with Polylogue(archive_root=archive_root, db_path=db_path) as archive:
            for _ in range(warmup):
                result_counts.append(len((await _query_once(archive, expression)).get("items", [])))
            for _ in range(baseline_rounds + rounds):
                result_counts.append(len((await _query_once(archive, expression)).get("items", [])))
                rss, pss, swap = _proc_memory()
                per_round.append(ResourceSample(rss, pss, swap, max(0, _temp_used_bytes(temp_root) - temp_before)))
    finally:
        stop.set()
        sampler.join(timeout=2)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    baseline = per_round[:baseline_rounds]
    measured = per_round[baseline_rounds:]
    baseline_rss = max(sample.rss_bytes for sample in baseline)
    baseline_pss = max(sample.pss_bytes for sample in baseline)
    baseline_swap = max(sample.swap_bytes for sample in baseline)
    baseline_temp = max(sample.temp_delta_bytes for sample in baseline)
    final = measured[-3:]
    returned = (
        all(
            current <= max(1, baseline_value) * (1 + tolerance)
            for current, baseline_value in ((sample.rss_bytes, baseline_rss) for sample in final)
        )
        and all(
            current <= max(1, baseline_value) * (1 + tolerance)
            for current, baseline_value in ((sample.pss_bytes, baseline_pss) for sample in final)
        )
        and all(sample.swap_bytes <= baseline_swap + 64 * 1024 * 1024 for sample in final)
    )
    receipt = {
        "status": "succeeded" if returned else "failed",
        "archive_root": str(archive_root),
        "archive_index_bytes": db_path.stat().st_size,
        "expression": expression,
        "rounds": rounds,
        "warmup_rounds": warmup,
        "baseline_rounds": baseline_rounds,
        "tolerance": tolerance,
        "steady_state_envelope": {
            "rss_bytes": baseline_rss,
            "pss_bytes": baseline_pss,
            "swap_bytes": baseline_swap,
            "temp_delta_bytes": baseline_temp,
            "return_tolerance": tolerance,
        },
        "peak": asdict(peak),
        "final_samples": [asdict(sample) for sample in final],
        "result_item_counts": result_counts,
        "elapsed_ms": elapsed_ms,
        "returned_to_envelope": returned,
        "regression_path": "rerun this command against the same promoted generation; status=failed identifies envelope drift",
    }
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--baseline-rounds", type=int, default=DEFAULT_BASELINE)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--sample-interval", type=float, default=0.05)
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
            )
        )
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
