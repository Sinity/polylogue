"""Shared helpers for the check command."""

from __future__ import annotations

import sys
import time

from polylogue.cli.shared.helpers import fail
from polylogue.core.protocols import ProgressCallback


def format_count_mapping(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value:,}" for key, value in sorted(counts.items()))


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
