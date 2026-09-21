"""The daemon's startup lineage component reports; it does not repair.

polylogue-ga6ib / polylogue-6kur AC4. ``session_links.branch_point_message_id``
for a prefix-sharing edge has exactly one producer: the scoped, in-transaction
refinement inside ``_resolve_session_graph``. Until this change the daemon also
ran that same refinement UNSCOPED across the whole index on every start, so a
producer regression was silently corrected on the next restart and never
surfaced. These tests pin the two halves of the replacement: the producer is
the only corrector, and a dangling edge the producer left behind is *reported*
and still there afterwards.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.daemon.cli import _lineage_startup_lifecycle_phase
from polylogue.daemon.lineage_startup import LineageStartupCensus, census_lineage_startup_sync
from polylogue.logging import capture
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers import write as write_module
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.write import (
    count_dangling_prefix_branch_points,
    read_archive_session_envelope,
    write_parsed_session_to_archive,
)
from polylogue.storage.sqlite.connection import open_connection


def _msg(provider_message_id: str, role: Role, text: str, position: int) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=provider_message_id,
        role=role,
        position=position,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text, position=0)],
    )


def _three_generation_sessions() -> tuple[ParsedSession, ParsedSession, ParsedSession]:
    """Grandparent, parent (grandparent's prefix + a tail turn), child (the
    parent's first two turns + its own tail).

    Writing the grandparent last re-extracts the parent to tail-only storage,
    which deletes exactly the rows the child's branch point names.
    """
    grandparent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="gp",
        title="grandparent",
        messages=[
            _msg("m0", Role.USER, "m0", 0),
            _msg("m1", Role.ASSISTANT, "m1", 1),
            _msg("m2", Role.USER, "m2", 2),
            _msg("m3", Role.ASSISTANT, "m3", 3),
        ],
    )
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="parent",
        title="parent",
        parent_session_provider_id="gp",
        branch_type=BranchType.FORK,
        messages=[
            _msg("m0", Role.USER, "m0", 0),
            _msg("m1", Role.ASSISTANT, "m1", 1),
            _msg("m2", Role.USER, "m2", 2),
            _msg("m3", Role.ASSISTANT, "m3", 3),
            _msg("m4", Role.USER, "m4", 4),
        ],
    )
    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="child",
        title="child",
        parent_session_provider_id="parent",
        branch_type=BranchType.FORK,
        messages=[
            _msg("m0", Role.USER, "m0", 0),
            _msg("m1", Role.ASSISTANT, "m1", 1),
            _msg("x2", Role.USER, "x2", 2),
        ],
    )
    return grandparent, parent, child


def _archive_with_three_generations(archive_root: Path) -> str:
    """Write parent, child, then grandparent through the production writer."""
    initialize_active_archive_root(archive_root)
    grandparent, parent, child = _three_generation_sessions()
    with open_connection(archive_root / "index.db") as conn:
        write_parsed_session_to_archive(conn, parent)
        child_id = write_parsed_session_to_archive(conn, child)
        write_parsed_session_to_archive(conn, grandparent)
        conn.commit()
    return child_id


def _composed_texts(archive_root: Path, session_id: str) -> list[str]:
    with open_connection(archive_root / "index.db") as conn:
        conn.row_factory = sqlite3.Row
        return [message.blocks[0].text for message in read_archive_session_envelope(conn, session_id).messages]


def _census_counts(archive_root: Path) -> tuple[int, int]:
    with open_connection(archive_root / "index.db") as conn:
        conn.row_factory = sqlite3.Row
        return count_dangling_prefix_branch_points(conn)


def test_the_writer_is_the_only_corrector_of_its_own_branch_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the producer intact there is nothing for a sweep to find.

    This is one half of the two-sided anti-vacuity condition: it fixes the
    denominator, so the sibling test below cannot pass by reporting a dangling
    edge that was always there. Anti-vacuity: neutralize
    ``_repair_stale_prefix_branch_points_db`` (exactly what the sibling does)
    and every assertion here goes red.
    """
    archive_root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    child_id = _archive_with_three_generations(archive_root)

    assert _census_counts(archive_root) == (0, 0)
    assert _composed_texts(archive_root, child_id) == ["m0", "m1", "x2"]

    with capture() as records:
        census = census_lineage_startup_sync()

    assert census == LineageStartupCensus(dangling_edges=0, dangling_sessions=0)
    assert census.converged is True
    reported = [record for record in records if record.get("event") == "daemon.lineage.startup_census"]
    assert [record["outcome"] for record in reported] == ["ok"]


def test_a_dangling_edge_the_producer_left_behind_is_reported_and_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neutralizing the producer's own correction must stay visible.

    The audit's anti-vacuity condition, executed: with
    ``_repair_stale_prefix_branch_points_db`` made a no-op the stale
    ``branch_point_message_id`` must still be observable after
    ``write_parsed_session_to_archive`` commits -- if some other route silently
    corrected it, this archive would compose fully and the assertions below
    would be red, which would mean a second mask nobody has found yet.

    Anti-vacuity for the behaviour this change introduces: restore an
    archive-wide corrective sweep in the startup component and the final two
    assertions (the edge is STILL dangling, the child STILL composes short)
    go red. That is precisely the masking AC4 forbids.
    """
    archive_root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    # Scoped to the WRITE only. The real function is back in place for the
    # census below, so a startup component that corrected anything would be
    # able to -- which is what makes the closing assertions a real mutant
    # detector rather than a tautology.
    with patch.object(write_module, "_repair_stale_prefix_branch_points_db", lambda *args, **kwargs: 0):
        child_id = _archive_with_three_generations(archive_root)

    # The bad row is observable between the write and the census.
    assert _census_counts(archive_root) == (1, 1)
    assert _composed_texts(archive_root, child_id) == ["x2"]
    with open_connection(archive_root / "index.db") as conn:
        branch_point, exists = conn.execute(
            """
            SELECT l.branch_point_message_id,
                   EXISTS (SELECT 1 FROM messages m WHERE m.message_id = l.branch_point_message_id)
            FROM session_links AS l
            WHERE l.src_session_id = ?
            """,
            (child_id,),
        ).fetchone()
    assert branch_point is not None
    assert exists == 0

    with capture() as records:
        census = census_lineage_startup_sync()

    assert census == LineageStartupCensus(dangling_edges=1, dangling_sessions=1)
    assert census.converged is False
    reported = [record for record in records if record.get("event") == "daemon.lineage.startup_census"]
    assert len(reported) == 1
    assert reported[0]["outcome"] == "degraded"
    assert reported[0]["reason"] == "dangling_prefix_branch_points"
    assert reported[0]["rows"] == 1
    assert reported[0]["sessions"] == 1

    # The census reported. It did not repair.
    assert _census_counts(archive_root) == (1, 1)
    assert _composed_texts(archive_root, child_id) == ["x2"]


def test_startup_census_on_an_absent_index_is_converged_and_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An archive that does not exist yet has no dangling edges to report.

    Anti-vacuity: returning ``measured=False`` (or emitting a degraded event)
    for a missing index would make every first daemon start report a condition
    that is not there.
    """
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "nothing-here"))
    with capture() as records:
        census = census_lineage_startup_sync()
    assert census == LineageStartupCensus(dangling_edges=0, dangling_sessions=0)
    assert census.converged is True
    assert [record for record in records if str(record.get("event", "")).startswith("daemon.lineage.")] == []


@pytest.mark.parametrize(
    ("census", "expected_phase"),
    [
        (LineageStartupCensus(dangling_edges=0, dangling_sessions=0), "component_ready"),
        (LineageStartupCensus(dangling_edges=3, dangling_sessions=2), "component_degraded"),
        (LineageStartupCensus(dangling_edges=0, dangling_sessions=0, measured=False), "component_degraded"),
    ],
)
def test_startup_lifecycle_phase_follows_the_census(census: LineageStartupCensus, expected_phase: str) -> None:
    """A non-zero count -- or a census that could not run -- reaches an operator.

    Anti-vacuity: return ``component_ready`` unconditionally and the last two
    rows go red, which is the state the deleted sweep used to guarantee.
    """
    assert _lineage_startup_lifecycle_phase(census) == expected_phase
