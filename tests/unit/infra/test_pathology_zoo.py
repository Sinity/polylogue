"""Focused contract tests for the production-ingested pathology zoo."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from shutil import copytree

import pytest

from polylogue.maintenance.pathology_zoo import PATHOLOGY_ZOO_MANIFEST, PathologyZooCanaryEligibility
from polylogue.maintenance.reindex_canary import CanarySelectionError, select_canary_sessions
from tests.infra.pathology_zoo import (
    CLAUDE_VINTAGE_LIVE_PROOF_SESSION_ID,
    PathologyZoo,
    build_pathology_zoo,
)


@pytest.fixture(scope="module")
def pathology_zoo(tmp_path_factory: pytest.TempPathFactory) -> PathologyZoo:
    return build_pathology_zoo(tmp_path_factory.mktemp("pathology-zoo"))


def test_pathology_zoo_manifest_covers_every_v0_dimension(pathology_zoo: PathologyZoo) -> None:
    expected = {
        "whale-component",
        "append-revision-chain",
        "fork-resume-prefix-tail-lineage",
        "lineage-cycle-candidate",
        "multi-session-raw",
        "quarantined-head-open-blocker",
        "genuinely-empty-session",
        "hook-event-raw",
        "claude-design-session-origin",
        "export-vintage-reorder",
        "claude-vintage-live-proof",
        "lifecycle-anchor-drift",
        "non-stream-safe-origin",
        "attachment-with-acquired-bytes",
        "attachment-without-acquired-bytes",
        "session-events-and-sidecars",
    }
    assert expected <= {member.pathology for member in pathology_zoo.manifest}
    assert all(
        member.motivating_beads and member.raw_paths and member.durable_paths for member in pathology_zoo.manifest
    )


def test_pathology_zoo_members_are_queryable_after_real_ingest(pathology_zoo: PathologyZoo) -> None:
    with sqlite3.connect(pathology_zoo.archive_root / "index.db") as conn:
        session_ids = {str(row[0]) for row in conn.execute("SELECT session_id FROM sessions")}
    assert {session_id for member in pathology_zoo.manifest for session_id in member.session_ids} <= session_ids
    assert pathology_zoo.members_for("claude-vintage-live-proof")[0].motivating_beads == (
        "polylogue-claude-vintage-live-proof",
        "polylogue-0qfy",
    )


def test_pathology_zoo_manifest_binds_every_wire_path_to_durable_evidence(pathology_zoo: PathologyZoo) -> None:
    """Every named wire input has a source-tier receipt, including the hook spool."""
    with sqlite3.connect(pathology_zoo.archive_root / "source.db") as conn:
        for member in pathology_zoo.manifest:
            table = "raw_hook_events" if member.member_id == "hook-event" else "raw_sessions"
            for durable_path in member.durable_paths:
                assert conn.execute(
                    f"SELECT COUNT(*) > 0 FROM {table} WHERE source_path = ?", (durable_path,)
                ).fetchone() == (1,)


def test_pathology_zoo_canary_eligibility_is_explicit(pathology_zoo: PathologyZoo) -> None:
    session_backed = [
        member
        for member in pathology_zoo.manifest
        if member.canary_eligibility is PathologyZooCanaryEligibility.SESSION_BACKED
    ]
    non_session_only = [
        member
        for member in pathology_zoo.manifest
        if member.canary_eligibility is PathologyZooCanaryEligibility.NON_SESSION_ONLY
    ]

    assert all(member.session_ids for member in session_backed)
    assert non_session_only == [pathology_zoo.members_for("hook-event-raw")[0]]
    assert all(not member.session_ids for member in non_session_only)


def test_zoo_preserves_the_named_vintage_and_lifecycle_red_cases(pathology_zoo: PathologyZoo) -> None:
    """The named K-class entries remain production-ingested, not parser-only samples."""
    with sqlite3.connect(pathology_zoo.archive_root / "index.db") as conn:
        content_blocks = conn.execute(
            "SELECT COUNT(*) FROM blocks WHERE session_id = ?",
            (f"claude-ai-export:{CLAUDE_VINTAGE_LIVE_PROOF_SESSION_ID}",),
        ).fetchone()
        lifecycle = conn.execute(
            "SELECT source_message_provider_id FROM session_events WHERE session_id = ? AND event_type = 'generation_lifecycle'",
            ("chatgpt-export:zoo-lifecycle-anchor-drift",),
        ).fetchall()
    with sqlite3.connect(pathology_zoo.archive_root / "source.db") as conn:
        vintage_raws = conn.execute(
            "SELECT COUNT(*) FROM raw_sessions WHERE origin = ? AND native_id = ?",
            ("chatgpt-export", "zoo-vintage-reorder"),
        ).fetchone()
    assert content_blocks == (1,)
    assert lifecycle == [("lifecycle-b",)]
    assert vintage_raws == (2,)


def test_pathology_zoo_manifest_is_consumed_by_real_canary_selection(
    pathology_zoo: PathologyZoo, tmp_path: Path
) -> None:
    """0x7nh's selector receives every replayable manifest member as a raw-id input."""
    assert pathology_zoo.canary_session_ids == tuple(
        sorted(
            {
                session_id
                for member in PATHOLOGY_ZOO_MANIFEST
                if member.canary_eligibility is PathologyZooCanaryEligibility.SESSION_BACKED
                for session_id in member.session_ids
            }
        )
    )
    selection = select_canary_sessions(
        pathology_zoo.archive_root / "index.db",
        sessions_per_origin=1,
        pathology_session_ids=pathology_zoo.canary_session_ids,
    )

    assert selection.pathology_session_ids == pathology_zoo.canary_session_ids
    assert set(pathology_zoo.canary_session_ids) <= set(selection.selected_session_ids)
    assert len(selection.selected_raw_ids) == len(set(selection.selected_raw_ids))

    mutated_root = tmp_path / "pathology-zoo-canary-mutation"
    copytree(pathology_zoo.archive_root, mutated_root)
    with sqlite3.connect(mutated_root / "index.db") as conn:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", ("codex-session:zoo-append-self",))
        conn.commit()
    with pytest.raises(CanarySelectionError, match="not indexed"):
        select_canary_sessions(
            mutated_root / "index.db",
            sessions_per_origin=1,
            pathology_session_ids=pathology_zoo.canary_session_ids,
        )
