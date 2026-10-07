"""Small collector harness for Antigravity parser contract tests.

Production callers provide prepared-store sinks. Unit tests use this explicit
collector only for tiny synthetic fixtures, keeping test assertions concise.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

from polylogue.sources.parsers.antigravity import parse_trajectory_db as _parse_trajectory_db
from polylogue.sources.parsers.base import AdmissionOutcome, AdmissionUnit, ParseAccounting, ParsedSession


class _TestAccounting:
    def __init__(self, expected: dict[AdmissionUnit, int]) -> None:
        self.expected = expected
        self.outcomes: list[AdmissionOutcome] = []

    def append(self, outcome: AdmissionOutcome) -> None:
        self.outcomes.append(outcome)

    def finish(self) -> ParseAccounting:
        return ParseAccounting(expected=self.expected, outcomes=self.outcomes)


def parse_trajectory_db(
    path: Path,
    fallback_id: str | None = None,
    *,
    immutable: bool = False,
) -> Iterator[ParsedSession]:
    """Collect the explicitly synthetic test source into ordinary lists."""
    grouping_path = path.with_name(f".{path.name}.test-spill.sqlite")
    grouping = sqlite3.connect(grouping_path)
    sessions = list(
        _parse_trajectory_db(
            path,
            fallback_id,
            immutable=immutable,
            grouping=grouping,
            message_sink_factory=list,
            event_sink_factory=list,
            accounting_factory=_TestAccounting,
        )
    )
    grouping.commit()
    yield from sessions
