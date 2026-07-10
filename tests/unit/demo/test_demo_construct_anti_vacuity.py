"""Anti-vacuity witnesses for the Incident 14:32 proof-world constructs (#polylogue-212.11).

Per docs/design/incident-1432-proof-world.md's anti-circularity rule: for every
declared construct there must be a test that withholds or deletes the evidence
and asserts the dependent construct check goes red. A verifier that stays green
when its proof is removed validates shape, not reality. Each test below seeds
the real demo archive through the real provider parsers, confirms the new
construct is green, mutates the archive to remove exactly the evidence the
construct depends on, and re-measures to confirm the construct goes red while
the seeding path itself is untouched.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.demo import evaluate_demo_constructs, seed_demo_archive


def _coverage_for(archive_root: Path, construct_id: str) -> bool:
    coverage = evaluate_demo_constructs(archive_root)
    (row,) = (row for row in coverage if row.construct_id == construct_id)
    return row.ok


@pytest.mark.asyncio
async def test_source_outage_interval_construct_goes_red_when_event_withheld(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    await seed_demo_archive(archive_root, force=True, with_overlays=False)

    assert _coverage_for(archive_root, "source_outage_interval_events") is True

    with sqlite3.connect(archive_root / "index.db") as conn:
        deleted = conn.execute("DELETE FROM session_events WHERE event_type = 'source_outage'").rowcount
        conn.commit()
    assert deleted >= 1

    assert _coverage_for(archive_root, "source_outage_interval_events") is False
