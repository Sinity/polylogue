"""Shared contract for the bounded direct-intake convergence benchmarks."""

from __future__ import annotations

from collections.abc import Mapping


def intake_measurement(
    *,
    expected_files: int,
    expected_sessions: int,
    expected_messages: int,
    succeeded_files: int,
    failed_files: int,
    skipped_files: int,
    excluded_files: int,
    deferred_files: int,
    refused_bytes: int,
    stored_sessions: int,
    stored_messages: int,
    stage_summary: Mapping[str, int],
) -> dict[str, int]:
    """Return truthful intake and stage counts, refusing incomplete fixtures.

    ``DaemonConverger.summary()`` covers only retained file states: successful
    states are evicted after their batch completes. A fully converged batch
    therefore reports zero for all four stage counters. Retained in-progress
    states include pending and deliberately unrun stages and must block a
    complete intake measurement.
    """
    counts = {
        "intake_expected_files": expected_files,
        "intake_succeeded_files": succeeded_files,
        "intake_failed_files": failed_files,
        "intake_skipped_files": skipped_files,
        "intake_excluded_files": excluded_files,
        "intake_deferred_files": deferred_files,
        "intake_refused_bytes": refused_bytes,
        "stored_sessions": stored_sessions,
        "stored_messages": stored_messages,
        "stage_total_files": int(stage_summary["total"]),
        "stage_converged_files": int(stage_summary["converged"]),
        "stage_failed_files": int(stage_summary["failed"]),
        "stage_in_progress_files": int(stage_summary["in_progress"]),
    }
    incomplete = (
        succeeded_files != expected_files
        or failed_files != 0
        or skipped_files != 0
        or excluded_files != 0
        or deferred_files != 0
        or refused_bytes != 0
        or stored_sessions != expected_sessions
        or stored_messages != expected_messages
        or int(stage_summary["failed"]) != 0
        or int(stage_summary["in_progress"]) != 0
        or int(stage_summary["total"])
        != int(stage_summary["converged"]) + int(stage_summary["failed"]) + int(stage_summary["in_progress"])
    )
    if incomplete:
        raise ValueError(f"direct-intake fixture incomplete: {counts}")
    return counts
