"""Shared helpers for the check command."""

from __future__ import annotations

import sys
import time

from polylogue.cli.shared.check_models import VacuumResult
from polylogue.cli.shared.helpers import fail
from polylogue.cli.shared.types import AppEnv
from polylogue.protocols import ProgressCallback


def format_count_mapping(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value:,}" for key, value in sorted(counts.items()))


def format_semantic_metric_summary(metric_summary: dict[str, dict[str, int]]) -> str:
    return ", ".join(
        (
            f"{metric}(preserved={counts.get('preserved', 0):,}, "
            f"declared_loss={counts.get('declared_loss', 0):,}, "
            f"critical_loss={counts.get('critical_loss', 0):,})"
        )
        for metric, counts in sorted(metric_summary.items())
    )


def parse_schema_samples(raw: str) -> int | None:
    value = raw.strip().lower()
    if value == "all":
        return None
    try:
        parsed = int(value)
    except ValueError:
        fail("doctor", "--schema-samples must be a positive integer or 'all'")
    if parsed <= 0:
        fail("doctor", "--schema-samples must be a positive integer or 'all'")
    return parsed


def make_count_progress_callback(*, label: str, unit: str) -> ProgressCallback:
    """Return a stderr progress reporter for monotonically increasing counters."""
    start = time.monotonic()
    count = 0

    def _cb(amount: int, desc: str | None = None) -> None:
        nonlocal count
        count += amount
        elapsed = time.monotonic() - start
        rate = count / elapsed if elapsed > 0 else 0
        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        print(
            f"\r{label}: {desc or f'{count:,} {unit}'} ({rate:.1f}/s, {elapsed_str} elapsed)...",
            end="",
            flush=True,
            file=sys.stderr,
        )

    return _cb


def make_schema_progress_callback() -> ProgressCallback:
    """Return a stderr progress reporter for schema verification."""
    return make_count_progress_callback(label="Verifying schemas", unit="raw records")


def make_session_product_progress_callback() -> ProgressCallback:
    """Return a stderr progress reporter for session-product repairs."""
    return make_count_progress_callback(label="Repairing session insights", unit="conversations")


def vacuum_database(env: AppEnv) -> VacuumResult:
    """Run VACUUM and return a machine-readable result."""
    from polylogue.storage.backends.connection import open_connection

    try:
        with open_connection(env.config.db_path) as conn:
            conn.execute("VACUUM")
        return VacuumResult(ok=True, detail="Running VACUUM to reclaim space...\n  VACUUM complete.")
    except Exception as exc:
        return VacuumResult(ok=False, detail=f"Running VACUUM to reclaim space...\n  VACUUM failed: {exc}")


def run_vacuum(env: AppEnv) -> None:
    """Run VACUUM to reclaim unused space."""
    result = vacuum_database(env)
    env.ui.console.print("")
    env.ui.console.print(result.detail)
