"""Anti-vacuity tests for the six-tier schema census."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS, initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.schema_inventory import (
    SchemaCensusError,
    assert_complete_census,
    canonical_schema_objects,
    capture_schema_census,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _fresh_archive(root: Path) -> None:
    for spec in ARCHIVE_TIER_SPECS.values():
        initialize_archive_database(root / spec.filename, spec.tier)


def test_canonical_inventory_contains_all_object_kinds_and_generated_fields() -> None:
    objects = {tier: canonical_schema_objects(tier) for tier in ArchiveTier}

    assert set(objects) == set(ArchiveTier)
    for tier, tier_objects in objects.items():
        refs = [obj.object_ref for obj in tier_objects]
        assert refs
        assert len(refs) == len(set(refs)), tier.value
        assert {obj.object_type for obj in tier_objects} >= {"table", "column"}

    index_objects = objects[ArchiveTier.INDEX]
    query_time_derived = {"action_pairs", "threads", "thread_sessions", "delegation_facts", "session_tag_rollups"}
    index_relations = {(obj.object_type, obj.name) for obj in index_objects if obj.object_type in {"table", "view"}}
    assert {("view", name) for name in query_time_derived} <= index_relations
    assert not {(object_type, name) for object_type, name in index_relations if object_type == "table"} & {
        ("table", name) for name in query_time_derived
    }
    assert any(obj.object_type == "view" and obj.name == "actions" for obj in index_objects)
    assert any(obj.object_type == "table" and obj.name == "messages_fts" and obj.virtual for obj in index_objects)
    assert any(
        obj.object_type == "column" and obj.table_name == "messages" and obj.name == "message_id" and obj.generated_kind
        for obj in index_objects
    )
    assert any(
        obj.object_type == "column"
        and obj.table_name == "blocks"
        and obj.name == "tool_command"
        and obj.generated_kind == "virtual"
        for obj in index_objects
    )


def test_fresh_six_tier_census_is_complete_and_row_counts_are_explicit(tmp_path: Path) -> None:
    _fresh_archive(tmp_path)
    before = {spec.filename: (tmp_path / spec.filename).stat().st_mtime_ns for spec in ARCHIVE_TIER_SPECS.values()}

    census = capture_schema_census(tmp_path, observed_at_ns=123, hash_files=False)

    assert_complete_census(census)
    assert census.complete is True
    assert len(census.tiers) == len(ArchiveTier)
    assert census.archive_identity_digest
    assert {tier.tier for tier in census.tiers} == set(ArchiveTier)
    for tier in census.tiers:
        assert tier.actual_version == tier.expected_version
        assert tier.row_counts
        assert all(count >= 0 for _name, count in tier.row_counts)
    after = {spec.filename: (tmp_path / spec.filename).stat().st_mtime_ns for spec in ARCHIVE_TIER_SPECS.values()}
    assert after == before


def test_unexpected_declaration_invalidates_completeness(tmp_path: Path) -> None:
    _fresh_archive(tmp_path)
    connection = sqlite3.connect(tmp_path / "user.db")
    try:
        connection.execute("CREATE TABLE undeclared_census_mutation (value TEXT)")
        connection.commit()
    finally:
        connection.close()

    census = capture_schema_census(tmp_path, observed_at_ns=123, count_rows=False, hash_files=False)

    assert census.complete is False
    user = next(tier for tier in census.tiers if tier.tier is ArchiveTier.USER)
    assert any("unexpected objects" in error for error in user.errors)
    with pytest.raises(SchemaCensusError, match="unexpected objects"):
        assert_complete_census(census)


def test_missing_tier_is_error_and_never_an_empty_pass(tmp_path: Path) -> None:
    _fresh_archive(tmp_path)
    (tmp_path / "audit.db").unlink()

    census = capture_schema_census(tmp_path, observed_at_ns=123, count_rows=False, hash_files=False)

    assert census.complete is False
    audit = next(tier for tier in census.tiers if tier.tier is ArchiveTier.AUDIT)
    assert audit.actual_version is None
    assert audit.row_counts == ()
    assert any("tier file is missing" in error for error in audit.errors)


def test_changed_definition_is_detected_even_when_object_name_survives(tmp_path: Path) -> None:
    _fresh_archive(tmp_path)
    connection = sqlite3.connect(tmp_path / "ops.db")
    try:
        connection.execute("DROP INDEX idx_slo_samples_label_time")
        connection.execute("CREATE INDEX idx_slo_samples_label_time ON slo_samples(observed_at_ms)")
        connection.commit()
    finally:
        connection.close()

    census = capture_schema_census(tmp_path, observed_at_ns=123, count_rows=False, hash_files=False)

    ops = next(tier for tier in census.tiers if tier.tier is ArchiveTier.OPS)
    assert any("changed declared objects" in error for error in ops.errors)
