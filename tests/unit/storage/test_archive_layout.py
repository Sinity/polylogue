"""Contract tests for the shared archive storage-layout owner (#2177).

These pin the semantics that ``polylogue config paths`` and the daemon
``/metrics`` exposition both delegate to, so the two surfaces cannot drift
again.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.storage import archive_layout


def test_tier_vocabulary_is_derived_from_specs() -> None:
    assert archive_layout.ARCHIVE_TIER_ORDER == ("source", "index", "embeddings", "user", "ops")
    assert archive_layout.BACKUP_REQUIRED_TIERS == ("source", "embeddings", "user")
    assert archive_layout.ARCHIVE_ACTIVE_TIER_ROLES == (
        "source",
        "index",
        "embeddings",
        "user",
        "ops",
        "unknown",
    )
    assert archive_layout.ARCHIVE_STORAGE_LAYOUTS == (
        "archive_missing",
        "archive_partial",
        "archive_complete",
    )
    assert archive_layout.ARCHIVE_LAYOUT_BLOCKER_LABELS == (
        "no_archive_tiers_present",
        "missing_archive_tiers",
        "schema_mismatch:source",
        "schema_mismatch:index",
        "schema_mismatch:embeddings",
        "schema_mismatch:user",
        "schema_mismatch:ops",
        "missing_backup_required_tier:source",
        "missing_backup_required_tier:embeddings",
        "missing_backup_required_tier:user",
    )


def test_classify_storage_layout() -> None:
    assert archive_layout.classify_storage_layout(present_count=0, final_shape_ready=False) == "archive_missing"
    assert archive_layout.classify_storage_layout(present_count=3, final_shape_ready=False) == "archive_partial"
    assert archive_layout.classify_storage_layout(present_count=5, final_shape_ready=True) == "archive_complete"


def test_blockers_for_empty_archive_match_cli_contract() -> None:
    # Mirrors tests/unit/cli/test_paths.py: no schema versions inspected.
    blockers = archive_layout.archive_layout_blockers(
        present_count=0,
        final_shape_ready=False,
        missing_backup_required=("source", "embeddings", "user"),
    )
    assert blockers == [
        "no_archive_tiers_present",
        "missing_archive_tiers",
        "missing_backup_required_tier:source",
        "missing_backup_required_tier:embeddings",
        "missing_backup_required_tier:user",
    ]


def test_blockers_include_schema_mismatch_in_tier_order() -> None:
    blockers = archive_layout.archive_layout_blockers(
        present_count=1,
        final_shape_ready=False,
        schema_mismatches=("ops", "source"),
        missing_backup_required=("user", "source"),
    )
    assert blockers == [
        "missing_archive_tiers",
        "schema_mismatch:source",
        "schema_mismatch:ops",
        "missing_backup_required_tier:source",
        "missing_backup_required_tier:user",
    ]


def test_blockers_empty_for_complete_archive() -> None:
    assert archive_layout.archive_layout_blockers(present_count=5, final_shape_ready=True) == []


def test_active_tier_role_resolves_known_and_unknown(tmp_path: Path) -> None:
    tier_paths = {
        "source": tmp_path / "source.db",
        "index": tmp_path / "index.db",
        "ops": tmp_path / "ops.db",
    }
    assert archive_layout.active_tier_role(tmp_path / "index.db", tier_paths) == "index"
    assert archive_layout.active_tier_role(tmp_path / "custom.sqlite", tier_paths) == "unknown"
    # Sequence-of-pairs form is also accepted.
    assert archive_layout.active_tier_role(tmp_path / "source.db", list(tier_paths.items())) == "source"
