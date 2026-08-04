"""Focused real-generation tests for the reindex canary differ.

The production dependency is ``compare_reindex_generations`` reading two
canonical ``index.db`` files.  The anti-vacuity mutation for the core test is
changing the candidate block text: a synthetic summary comparator would stay
green, while the real blocks read model must report the changed row.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.maintenance.reindex_canary import (
    CanaryDifferenceReview,
    CanarySelectionError,
    DifferenceClassification,
    DifferenceOperation,
    ExpectedDifference,
    UnclassifiedCanaryDiffError,
    compare_reindex_generations,
    load_canary_report,
    select_canary_sessions,
    write_canary_report,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _seed_index(
    path: Path,
    *,
    sessions: tuple[str, ...] = ("alpha",),
    block_text: str = "stable transcript",
    profile_materialized_at: str = "first-run",
    profile_message_count: int = 1,
    origins: tuple[str, ...] | None = None,
) -> None:
    initialize_archive_database(path, ArchiveTier.INDEX)
    with sqlite3.connect(path) as connection:
        session_origins = origins or ("codex-session",) * len(sessions)
        assert len(session_origins) == len(sessions)
        for native_id, origin in zip(sessions, session_origins, strict=True):
            session_id = f"{origin}:{native_id}"
            connection.execute(
                """
                INSERT INTO sessions(native_id, origin, raw_id, content_hash, message_count)
                VALUES (?, ?, ?, ?, 1)
                """,
                (native_id, origin, f"raw-{native_id}", native_id.encode().ljust(32, b"-")),
            )
            connection.execute(
                """
                INSERT INTO messages(session_id, position, role, material_origin, content_hash)
                VALUES (?, 0, 'user', 'human_authored', ?)
                """,
                (session_id, native_id.encode().ljust(32, b"m")),
            )
            connection.execute(
                """
                INSERT INTO blocks(message_id, session_id, position, block_type, text)
                VALUES (?, ?, 0, 'text', ?)
                """,
                (f"{session_id}:0.0", session_id, block_text),
            )
            connection.execute(
                """
                INSERT INTO session_profiles(session_id, materialized_at, message_count, tags_json)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, profile_materialized_at, profile_message_count, '{"b":2,"a":1}'),
            )
        connection.commit()


def test_equal_real_generations_ignore_only_materialization_metadata(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current, profile_materialized_at="current-build")
    _seed_index(candidate, profile_materialized_at="candidate-build")

    report = compare_reindex_generations(current, candidate)

    assert report.differences == ()
    assert report.unclassified_count == 0
    assert {"sessions", "messages", "blocks", "session_profiles"}.issubset(report.compared_tables)


def test_differ_reports_real_core_and_derived_row_changes(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current, sessions=("alpha", "removed"))
    _seed_index(
        candidate,
        sessions=("alpha", "added"),
        block_text="changed transcript",
        profile_message_count=2,
    )

    report = compare_reindex_generations(current, candidate)

    assert report.unexpected_count > 0
    assert report.unclassified_count == 0
    operations = {(item.table, item.operation) for item in report.differences}
    assert ("blocks", DifferenceOperation.CHANGED) in operations
    assert ("blocks", DifferenceOperation.ADDED) in operations
    assert ("blocks", DifferenceOperation.REMOVED) in operations
    assert any(
        item.table == "session_profiles"
        and item.operation is DifferenceOperation.CHANGED
        and "message_count" in item.changed_columns
        for item in report.differences
    )
    assert all(item.classification is DifferenceClassification.UNEXPECTED for item in report.differences)


def test_expected_difference_is_structurally_accounted_for(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current)
    _seed_index(candidate, profile_message_count=2)

    report = compare_reindex_generations(
        current,
        candidate,
        expected=(
            ExpectedDifference(
                table="session_profiles",
                columns=("message_count",),
                bead_ref="polylogue-example",
                rationale="the reviewed materializer change updates this aggregate",
            ),
        ),
    )

    profile_changes = [item for item in report.differences if item.table == "session_profiles"]
    assert profile_changes
    assert all(item.classification is DifferenceClassification.EXPECTED for item in profile_changes)
    assert all("polylogue-example" in item.rationale for item in profile_changes)
    assert report.expected_count == len(profile_changes)


def test_selected_sessions_bound_the_canary_to_a_real_subset(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current, sessions=("kept", "outside"))
    _seed_index(candidate, sessions=("kept", "outside"))
    with sqlite3.connect(candidate) as connection:
        connection.execute(
            "UPDATE blocks SET text = 'outside changed' WHERE session_id = ?",
            ("codex-session:outside",),
        )
        connection.commit()

    report = compare_reindex_generations(current, candidate, session_ids=("codex-session:kept",))

    assert report.session_ids == ("codex-session:kept",)
    assert report.differences == ()


def test_canary_comparison_is_read_only(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    _seed_index(current)
    _seed_index(candidate)
    before = current.stat().st_ino, current.stat().st_size, candidate.stat().st_ino, candidate.stat().st_size

    compare_reindex_generations(current, candidate)

    after = current.stat().st_ino, current.stat().st_size, candidate.stat().st_ino, candidate.stat().st_size
    assert after == before


def test_selector_samples_each_origin_and_keeps_explicit_inputs(tmp_path: Path) -> None:
    index = tmp_path / "index.db"
    _seed_index(
        index,
        sessions=("codex-a", "codex-pathology", "chat-a", "chat-sample", "claude-a", "claude-pathology"),
        origins=(
            "codex-session",
            "codex-session",
            "chatgpt-export",
            "chatgpt-export",
            "claude-ai-export",
            "claude-ai-export",
        ),
    )

    selection = select_canary_sessions(
        index,
        sessions_per_origin=1,
        pathology_session_ids=("codex-session:codex-pathology", "claude-ai-export:claude-pathology"),
        sample_session_ids=("chatgpt-export:chat-sample",),
    )

    assert selection.origin_counts == (
        ("chatgpt-export", 2),
        ("claude-ai-export", 2),
        ("codex-session", 2),
    )
    assert selection.selected_session_ids == (
        "chatgpt-export:chat-a",
        "chatgpt-export:chat-sample",
        "claude-ai-export:claude-a",
        "claude-ai-export:claude-pathology",
        "codex-session:codex-a",
        "codex-session:codex-pathology",
    )
    assert selection.selected_raw_ids == (
        "raw-chat-a",
        "raw-chat-sample",
        "raw-claude-a",
        "raw-claude-pathology",
        "raw-codex-a",
        "raw-codex-pathology",
    )


def test_selector_refuses_unknown_or_non_replayable_explicit_sessions(tmp_path: Path) -> None:
    index = tmp_path / "index.db"
    _seed_index(index)
    with pytest.raises(CanarySelectionError, match="not indexed"):
        select_canary_sessions(index, pathology_session_ids=("codex-session:missing",))
    with sqlite3.connect(index) as connection:
        connection.execute("UPDATE sessions SET raw_id = NULL")
        connection.commit()
    with pytest.raises(CanarySelectionError, match="no raw_id"):
        select_canary_sessions(index, pathology_session_ids=("codex-session:alpha",))


def test_durable_report_refuses_unclassified_diffs(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    report_path = tmp_path / "reports" / "canary.json"
    _seed_index(current)
    _seed_index(candidate, block_text="changed transcript")
    comparison = compare_reindex_generations(current, candidate)
    selection = select_canary_sessions(current, sessions_per_origin=1)

    with pytest.raises(UnclassifiedCanaryDiffError, match="classification is incomplete"):
        write_canary_report(report_path, selection=selection, comparison=comparison, reviews=())
    assert not report_path.exists()


def test_durable_report_persists_explicit_review_for_every_diff(tmp_path: Path) -> None:
    current = tmp_path / "current.db"
    candidate = tmp_path / "candidate.db"
    report_path = tmp_path / "reports" / "canary.json"
    _seed_index(current)
    _seed_index(candidate, block_text="changed transcript")
    comparison = compare_reindex_generations(current, candidate)
    selection = select_canary_sessions(current, sessions_per_origin=1)
    reviews = tuple(
        CanaryDifferenceReview.for_difference(
            difference,
            classification=DifferenceClassification.UNEXPECTED,
            reference="polylogue-follow-up",
            rationale="the canary has no reviewed expected delta for this row",
        )
        for difference in comparison.differences
    )

    durable = write_canary_report(report_path, selection=selection, comparison=comparison, reviews=reviews)
    loaded = load_canary_report(report_path)

    assert durable.unclassified_count == 0
    assert report_path.exists()
    assert loaded["schema_version"] == 1
    comparison_payload = loaded["comparison"]
    assert isinstance(comparison_payload, dict)
    summary = comparison_payload["summary"]
    assert isinstance(summary, dict)
    assert summary["unclassified_count"] == 0
    assert summary["unexpected_count"] == len(reviews)
