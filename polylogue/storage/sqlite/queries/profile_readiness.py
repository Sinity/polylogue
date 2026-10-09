"""Inspect the selected profile relation in bounded batches on its original reader."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass

from polylogue.storage.derived.session.derivation import (
    SESSION_PARTITION_INSPECT_CHUNK,
    bound_session_profile_partitions,
)
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION


@dataclass(frozen=True, slots=True)
class ProfileReadinessCounts:
    rows: int
    expected: int
    missing: int
    stale: int


def read_profile_readiness(
    connection: sqlite3.Connection,
    relation: str,
    parameters: Sequence[object],
    *,
    checkpoint: Callable[[], None],
) -> ProfileReadinessCounts:
    """Count only the caller's canonical session scope and inspect every selected partition."""
    rows = expected = missing = stale = 0
    with closing(connection.execute(f"SELECT s.session_id, sp.session_id {relation}", parameters)) as selected:
        while batch := selected.fetchmany(SESSION_PARTITION_INSPECT_CHUNK):
            checkpoint()
            statuses = bound_session_profile_partitions(
                connection,
                tuple(str(row[0]) for row in batch),
                materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
            )
            expected += len(batch)
            rows += sum(row[1] is not None for row in batch)
            missing += sum(status == "missing" for status in statuses.values())
            stale += sum(status == "stale" for status in statuses.values())
    return ProfileReadinessCounts(rows, expected, missing, stale)
