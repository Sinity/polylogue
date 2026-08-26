"""Single authority for the CLI interaction benchmark workload contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class InteractionWorkload:
    name: str
    route: str
    background: bool = False


INTERACTION_WORKLOADS: tuple[InteractionWorkload, ...] = (
    InteractionWorkload("cold-status", "installed-cli.status"),
    InteractionWorkload("warm-status", "installed-cli.status"),
    InteractionWorkload("find-read", "uds.cli.query"),
    InteractionWorkload("static-completion", "uds.completion"),
    InteractionWorkload("live-completion", "uds.completion", background=True),
    InteractionWorkload("fuzzy-launch", "installed-cli.select"),
    InteractionWorkload("pagination", "uds.cli.query.pagination"),
    InteractionWorkload("cancellation", "uds.cli.query.cancel"),
    InteractionWorkload("concurrent-reads", "uds.cli.query.concurrent", background=True),
    InteractionWorkload("incremental-ingest", "daemon.ingest", background=True),
    InteractionWorkload("derivation-catch-up", "daemon.derivation", background=True),
    InteractionWorkload("inactive-candidate", "daemon.candidate", background=True),
)

PROFILE_METRICS: tuple[str, ...] = (
    "cold_start_ms",
    "warm_roundtrip_ms",
    "static_completion_ms",
    "live_completion_ms",
    "first_byte_ms",
    "full_render_ms",
    "fuzzy_launch_ms",
    "pagination_ms",
    "cancellation_ms",
    "peak_rss_kib",
    "imported_modules",
    "rows",
    "bytes",
    "queue_delay_ms",
    "writer_hold_ms",
    "background_operations",
    "background_throughput",
    "concurrent_interference_p95_ms",
)

TERMINAL_COLUMNS: tuple[int, ...] = (40, 80, 120, 200)
RESPONSE_SIZES: tuple[int, ...] = (1, 100, 10_000)


def profile_manifest() -> dict[str, object]:
    """Return the reproducible workload/denominator declaration."""
    return {
        "workloads": [asdict(workload) for workload in INTERACTION_WORKLOADS],
        "metrics": list(PROFILE_METRICS),
        "terminal_columns": list(TERMINAL_COLUMNS),
        "response_rows": list(RESPONSE_SIZES),
        "denominators": {
            "cold_trials": "one fresh installed process per sample",
            "warm_trials": "one direct typed UDS request per sample",
            "concurrent_reads": "four simultaneous read requests",
            "background_throughput": "completed background operations / mixed-load second",
        },
    }


def record_metrics(benchmark: Any, **metrics: int | float | str) -> None:
    """Attach decomposed production-route metrics to pytest-benchmark output."""
    unknown = set(metrics) - set(PROFILE_METRICS)
    if unknown:
        raise ValueError(f"unknown CLI interaction benchmark metrics: {sorted(unknown)}")
    extra_info = getattr(benchmark, "extra_info", None)
    if extra_info is not None:
        extra_info.update(metrics)


__all__ = [
    "INTERACTION_WORKLOADS",
    "InteractionWorkload",
    "PROFILE_METRICS",
    "RESPONSE_SIZES",
    "TERMINAL_COLUMNS",
    "profile_manifest",
    "record_metrics",
]
