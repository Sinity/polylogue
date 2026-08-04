"""Focused contract tests for the production-ingested pathology zoo."""

from __future__ import annotations

import sqlite3

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
    assert all(
        member.motivating_beads and member.raw_paths and member.durable_paths for member in pathology_zoo.manifest
    )


def test_pathology_zoo_members_are_queryable_after_real_ingest(pathology_zoo: PathologyZoo) -> None:
    with sqlite3.connect(pathology_zoo.archive_root / "index.db") as conn:
        session_ids = {str(row[0]) for row in conn.execute("SELECT session_id FROM sessions")}
    assert {session_id for member in pathology_zoo.manifest for session_id in member.session_ids} <= session_ids
    assert pathology_zoo.members_for("content-blocks-vintage")[0].motivating_beads == (
        "polylogue-yazae",
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


def test_pathology_zoo_pathologies_have_production_structural_assertions(pathology_zoo: PathologyZoo) -> None:
    """Every manifest member names a durable, queryable archive condition."""
    with (
        sqlite3.connect(pathology_zoo.archive_root / "source.db") as source,
        sqlite3.connect(pathology_zoo.archive_root / "index.db") as index,
    ):
        assertions = {
            "whale-component": lambda: (
                index.execute(
                    "SELECT message_count >= 48 FROM sessions WHERE session_id = ?", ("codex-session:zoo-whale",)
                ).fetchone()
                == (1,)
            ),
            "append-self-describing": lambda: (
                source.execute(
                    "SELECT COUNT(*) FROM raw_sessions WHERE source_path IN (?, ?)",
                    pathology_zoo.members_for("append-revision-chain")[0].durable_paths,
                ).fetchone()
                == (2,)
            ),
            "append-opaque": lambda: (
                source.execute(
                    "SELECT COUNT(*) FROM raw_sessions WHERE source_path IN (?, ?)",
                    pathology_zoo.members_for("append-revision-chain")[1].durable_paths,
                ).fetchone()
                == (2,)
            ),
            "fork-prefix-tail": lambda: (
                index.execute(
                    "SELECT resolved_dst_session_id FROM session_links WHERE src_session_id = ?",
                    ("codex-session:zoo-lineage-child",),
                ).fetchone()
                == ("codex-session:zoo-lineage-parent",)
            ),
            "lineage-cycle": lambda: (
                index.execute(
                    "SELECT status, resolved_dst_session_id FROM session_links WHERE src_session_id = ?",
                    ("codex-session:zoo-cycle-a",),
                ).fetchone()
                == ("quarantined", None)
            ),
            "grouped-jsonl": lambda: (
                index.execute(
                    "SELECT COUNT(*) FROM sessions WHERE session_id IN (?, ?)",
                    ("claude-code-session:zoo-group-a", "claude-code-session:zoo-group-b"),
                ).fetchone()
                == (2,)
            ),
            "quarantined-head": lambda: (
                index.execute(
                    "SELECT status, resolved_dst_session_id FROM session_links WHERE src_session_id = ?",
                    ("codex-session:zoo-orphan-head",),
                ).fetchone()
                == (None, None)
            ),
            "empty-session": lambda: (
                index.execute(
                    "SELECT message_count FROM sessions WHERE session_id = ?", ("claude-code-session:zoo-empty",)
                ).fetchone()
                == (0,)
            ),
            "hook-event": lambda: (
                source.execute(
                    "SELECT session_native_id, event_type FROM raw_hook_events WHERE hook_event_id = 'hook:zoo-hook-event'"
                ).fetchone()
                == ("zoo-hook-parent", "PostToolUse")
            ),
            "claude-design": lambda: (
                index.execute(
                    "SELECT origin FROM sessions WHERE session_id = ?", ("claude-design-session:zoo-design-session",)
                ).fetchone()
                == ("claude-design-session",)
            ),
            "vintage-reorder": lambda: (
                source.execute("SELECT COUNT(*) FROM raw_sessions WHERE source_path LIKE '%vintage-%.json'").fetchone()
                == (2,)
            ),
            "content-blocks-vintage": lambda: (
                index.execute(
                    "SELECT COUNT(*) FROM blocks WHERE session_id = ?", ("claude-ai-export:zoo-content-blocks-vintage",)
                ).fetchone()
                == (1,)
            ),
            "lifecycle-anchor-drift": lambda: (
                index.execute(
                    "SELECT source_message_provider_id FROM session_events WHERE session_id = ? AND event_type = 'generation_lifecycle'",
                    ("chatgpt-export:zoo-lifecycle-anchor-drift",),
                ).fetchone()
                == ("lifecycle-b",)
            ),
            "non-stream-safe": lambda: (
                index.execute(
                    "SELECT COUNT(*) FROM sessions WHERE session_id = ?",
                    ("antigravity-session:zoo-06-00:zoo-06-00.md",),
                ).fetchone()
                == (1,)
            ),
            "attachment-with-bytes": lambda: (
                index.execute(
                    "SELECT COUNT(*) > 0 FROM attachment_refs WHERE session_id = ?",
                    ("aistudio-drive:zoo-attachment-bytes",),
                ).fetchone()
                == (1,)
            ),
            "attachment-without-bytes": lambda: (
                index.execute(
                    "SELECT COUNT(*) > 0 FROM attachment_refs WHERE session_id = ?",
                    ("claude-ai-export:zoo-attachment-metadata",),
                ).fetchone()
                == (1,)
            ),
            "events-sidecars": lambda: (
                index.execute(
                    "SELECT COUNT(*) > 0 FROM session_events WHERE session_id = ?",
                    ("chatgpt-export:zoo-lifecycle-anchor-drift",),
                ).fetchone()
                == (1,)
            ),
        }
        assert {member.member_id for member in pathology_zoo.manifest} == set(assertions)
        assert all(assertions[member.member_id]() for member in pathology_zoo.manifest)


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


def test_only_the_absent_canary_hook_is_deferred() -> None:
    """The archive-verification consumer exists; only the differ canary remains absent."""
    manifest_ids = {member.member_id for member in pathology_zoo_manifest()}
    assert {"vintage-reorder", "content-blocks-vintage", "lifecycle-anchor-drift"} <= manifest_ids
    assert [gap.consumer for gap in pathology_zoo_integration_gaps()] == ["polylogue-0x7nh"]
