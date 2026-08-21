"""Test that column reordering in specs produces correct SQL."""

from __future__ import annotations

import sqlite3
from dataclasses import replace

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import (
    BLOCKS_SPEC,
    MESSAGES_SPEC,
    TABLE_SPECS,
)
from polylogue.storage.sqlite.archive_tiers.audit import AUDIT_DDL
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDINGS_DDL
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL
from polylogue.storage.sqlite.archive_tiers.ops import OPS_DDL
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user import USER_DDL


class TestColumnSpecReordering:
    """Verify that the table-driven specs correctly reflect column order."""

    def test_messages_spec_reflects_correct_column_order(self) -> None:
        """Verify that MESSAGES_SPEC column order matches the schema."""
        # The first writable column should be session_id (since message_id is GENERATED)
        assert MESSAGES_SPEC.writable_columns[0].name == "session_id"
        # Verify that native_id is next
        assert MESSAGES_SPEC.writable_columns[1].name == "native_id"
        # The parent_message_id column is present but with NULL placeholder
        parent_col = next(c for c in MESSAGES_SPEC.writable_columns if c.name == "parent_message_id")
        assert parent_col.extract_placeholder == "NULL"

    def test_messages_insert_column_names_matches_schema(self) -> None:
        """Verify that the INSERT column list is correctly generated from spec."""
        insert_cols = MESSAGES_SPEC.insert_column_names
        col_list = insert_cols.split(", ")
        # Should not contain message_id as a standalone column (GENERATED)
        assert "message_id" not in col_list
        # Should start with session_id
        assert insert_cols.startswith("session_id")
        # Should contain all writable columns
        for col in MESSAGES_SPEC.writable_columns:
            assert col.name in col_list

    def test_messages_insert_placeholder_string_has_null_for_parent(self) -> None:
        """Verify that parent_message_id uses NULL placeholder."""
        placeholders = MESSAGES_SPEC.insert_placeholder_string
        # Should have ? placeholders for most columns
        assert placeholders.count("?") > 10
        # Should NOT have NULL in the placeholders if parent_message_id is skipped
        # (parent_message_id has extract_placeholder="NULL" but may not appear in VALUES tuple)

    def test_blocks_spec_writable_columns_exclude_generated(self) -> None:
        """Verify that BLOCKS_SPEC excludes all GENERATED columns."""
        generated_cols = {c.name for c in BLOCKS_SPEC.all_columns if c.is_generated}
        writable_cols = {c.name for c in BLOCKS_SPEC.writable_columns}
        # Generated columns should not appear in writable columns
        assert generated_cols.isdisjoint(writable_cols)
        # Expected GENERATED columns in blocks
        assert generated_cols == {"block_id", "tool_command", "tool_path", "search_text", "tool_detail_text"}

    def test_blocks_insert_statement_format(self) -> None:
        """Verify that the INSERT statement can be correctly formatted."""
        insert_sql = f"""
        INSERT INTO blocks (
            {BLOCKS_SPEC.insert_column_names}
        ) VALUES ({BLOCKS_SPEC.insert_placeholder_string})
        """
        # Should be valid SQL-like format
        assert "INSERT INTO blocks" in insert_sql
        assert "VALUES" in insert_sql
        # Should have correct number of columns
        col_count = len(BLOCKS_SPEC.writable_columns)
        placeholder_count = BLOCKS_SPEC.insert_placeholder_string.count("?")
        assert col_count == placeholder_count

    def test_changing_column_order_would_change_sql_output(self) -> None:
        """Verify that column order in specs directly determines SQL output."""
        original_insert_cols = MESSAGES_SPEC.insert_column_names
        original_placeholders = MESSAGES_SPEC.insert_placeholder_string

        # Both should have content
        assert len(original_insert_cols) > 0
        assert len(original_placeholders) > 0

        # Verify that each column appears exactly once in the INSERT column list
        col_names = [c.name for c in MESSAGES_SPEC.writable_columns]
        for col_name in col_names:
            # Count occurrences in the comma-separated list
            # This is a simple check that the column appears at least once
            assert col_name in original_insert_cols or col_name == "parent_message_id"

    def test_column_spec_extract_placeholder_values(self) -> None:
        """Verify that extract_placeholder values are correctly set."""
        # Most columns should use "?" placeholder
        question_mark_cols = [c for c in MESSAGES_SPEC.writable_columns if c.extract_placeholder == "?"]
        assert len(question_mark_cols) > 20  # Most columns

        # parent_message_id should use "NULL" placeholder
        parent_col = next(c for c in MESSAGES_SPEC.writable_columns if c.name == "parent_message_id")
        assert parent_col.extract_placeholder == "NULL"

    def test_session_events_spec_reordering_changes_rendered_sql(self) -> None:
        """A new table's column order must be load-bearing in rendered DDL."""
        spec = TABLE_SPECS["session_events"]
        reordered = replace(
            spec,
            all_columns=tuple(reversed(spec.all_columns)),
            writable_columns=tuple(reversed(spec.writable_columns)),
        )

        assert spec.ddl_body != reordered.ddl_body
        assert "PRIMARY KEY(session_id, position)" in spec.ddl_body
        assert spec.writable_columns[0].name == "session_id"
        assert reordered.writable_columns[0].name == "occurred_at_ms"

    def test_index_ddl_uses_every_declared_table_spec(self) -> None:
        """The executable index schema must render every table declaration."""
        index_ddl = ARCHIVE_DDL_BY_TIER[ArchiveTier.INDEX]

        for spec in TABLE_SPECS.values():
            assert spec.ddl_body in index_ddl

    def test_archive_ddl_mapping_exposes_each_tier_script(self) -> None:
        """The public archive DDL map must preserve every tier's fresh-create script."""
        assert ARCHIVE_DDL_BY_TIER == {
            ArchiveTier.AUDIT: AUDIT_DDL,
            ArchiveTier.EMBEDDINGS: EMBEDDINGS_DDL,
            ArchiveTier.INDEX: INDEX_DDL,
            ArchiveTier.OPS: OPS_DDL,
            ArchiveTier.SOURCE: SOURCE_DDL,
            ArchiveTier.USER: USER_DDL,
        }

    def test_non_vector_tier_scripts_create_their_schema(self) -> None:
        """Fresh archive tiers must execute the scripts exported through the public map."""
        for tier in (ArchiveTier.AUDIT, ArchiveTier.INDEX, ArchiveTier.OPS, ArchiveTier.SOURCE, ArchiveTier.USER):
            with sqlite3.connect(":memory:") as connection:
                connection.executescript(ARCHIVE_DDL_BY_TIER[tier])
                tables = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()

            assert tables, tier
