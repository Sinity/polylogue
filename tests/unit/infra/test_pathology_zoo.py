"""Focused contract tests for the production-ingested pathology zoo."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.infra.pathology_zoo import (
    PathologyZoo,
    build_pathology_zoo,
    pathology_zoo_integration_gaps,
    pathology_zoo_manifest,
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
        "content-blocks-vintage",
        "lifecycle-anchor-drift",
        "non-stream-safe-origin",
        "attachment-with-acquired-bytes",
        "attachment-without-acquired-bytes",
        "session-events-and-sidecars",
    }
    assert expected <= {member.pathology for member in pathology_zoo.manifest}
    assert all(member.motivating_beads and member.session_ids and member.raw_paths for member in pathology_zoo.manifest)


def test_pathology_zoo_members_are_queryable_after_real_ingest(pathology_zoo: PathologyZoo) -> None:
    with sqlite3.connect(pathology_zoo.archive_root / "index.db") as conn:
        session_ids = {str(row[0]) for row in conn.execute("SELECT session_id FROM sessions")}
    assert {session_id for member in pathology_zoo.manifest for session_id in member.session_ids} <= session_ids
    assert pathology_zoo.members_for("content-blocks-vintage")[0].motivating_beads == (
        "polylogue-yazae",
        "polylogue-0qfy",
    )


def test_zoo_preserves_the_named_vintage_and_lifecycle_red_cases(pathology_zoo: PathologyZoo) -> None:
    """The three K-class entries remain production-ingested, not parser-only samples."""
    with sqlite3.connect(pathology_zoo.archive_root / "index.db") as conn:
        content_blocks = conn.execute(
            "SELECT COUNT(*) FROM blocks WHERE session_id = ?", ("claude-ai-export:zoo-content-blocks-vintage",)
        ).fetchone()
        lifecycle = conn.execute(
            "SELECT source_message_provider_id FROM session_events WHERE session_id = ? AND event_type = 'generation_lifecycle'",
            ("chatgpt-export:zoo-lifecycle-anchor-drift",),
        ).fetchall()
    with sqlite3.connect(pathology_zoo.archive_root / "source.db") as conn:
        vintage_raws = conn.execute(
            "SELECT COUNT(*) FROM raw_sessions WHERE source_path LIKE ?", ("%vintage-%.json",)
        ).fetchone()
    assert content_blocks == (1,)
    assert lifecycle == [("lifecycle-b",)]
    assert vintage_raws == (2,)


def test_pathology_zoo_fails_when_the_production_ingest_route_is_bypassed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: replacing the real parse harness leaves no archive to validate."""
    import tests.infra.pathology_zoo as zoo_module

    async def bypassed_ingest(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(zoo_module, "parse_sources_archive", bypassed_ingest)
    with pytest.raises(RuntimeError, match="after production ingest"):
        build_pathology_zoo(tmp_path / "bypassed")


def test_absent_registry_and_canary_hooks_are_explicitly_not_faked() -> None:
    """t0m73 and 0x7nh have no extension point in this checkout; retain the gap as evidence."""
    manifest_ids = {member.member_id for member in pathology_zoo_manifest()}
    assert {"vintage-reorder", "content-blocks-vintage", "lifecycle-anchor-drift"} <= manifest_ids
    gaps = {gap.consumer: gap.follow_up for gap in pathology_zoo_integration_gaps()}
    assert set(gaps) == {"polylogue-t0m73", "polylogue-0x7nh"}
    assert all("extension point" in follow_up for follow_up in gaps.values())
