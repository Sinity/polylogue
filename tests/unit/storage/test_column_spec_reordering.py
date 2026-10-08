"""Test that column reordering in specs produces correct SQL."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.column_spec import TableColumnSpec


class TestColumnSpecReordering:
    """Verify that the table-driven specs correctly reflect column order."""

    @staticmethod
    def _specs() -> tuple[TableColumnSpec, TableColumnSpec, dict[str, TableColumnSpec]]:
        from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import (
            BLOCKS_SPEC,
            MESSAGES_SPEC,
            TABLE_SPECS,
        )

        return BLOCKS_SPEC, MESSAGES_SPEC, TABLE_SPECS

    def test_messages_spec_reflects_correct_column_order(self) -> None:
        """Verify that messages_spec column order matches the schema."""
        _, messages_spec, _ = self._specs()
        # The first writable column should be session_id (since message_id is GENERATED)
        assert messages_spec.writable_columns[0].name == "session_id"
        # Verify that native_id is next
        assert messages_spec.writable_columns[1].name == "native_id"
        # The parent_message_id column is present but with NULL placeholder
        parent_col = next(c for c in messages_spec.writable_columns if c.name == "parent_message_id")
        assert parent_col.extract_placeholder == "NULL"

    def test_messages_insert_column_names_matches_schema(self) -> None:
        """Verify that the INSERT column list is correctly generated from spec."""
        _, messages_spec, _ = self._specs()
        insert_cols = messages_spec.insert_column_names
        col_list = insert_cols.split(", ")
        # Should not contain message_id as a standalone column (GENERATED)
        assert "message_id" not in col_list
        # Should start with session_id
        assert insert_cols.startswith("session_id")
        # Should contain all writable columns
        for col in messages_spec.writable_columns:
            assert col.name in col_list

    def test_blocks_spec_writable_columns_exclude_generated(self) -> None:
        """Verify that BLOCKS_SPEC excludes all GENERATED columns."""
        blocks_spec, _, _ = self._specs()
        generated_cols = {c.name for c in blocks_spec.all_columns if c.is_generated}
        writable_cols = {c.name for c in blocks_spec.writable_columns}
        # Generated columns should not appear in writable columns
        assert generated_cols.isdisjoint(writable_cols)
        # Expected GENERATED columns in blocks
        assert generated_cols == {"block_id", "tool_command", "tool_path", "search_text"}

    def test_block_query_projection_uses_canonical_table_columns(self) -> None:
        """The compact block SELECT must name only declared storage columns.

        The query read model used to subtract ``tool_result_outcome_unknown_
        reason`` from this projection, so a bounded message page reported an
        unknown tool outcome with no reason; it now selects the whole declared
        block row (polylogue-blpir).
        """
        from polylogue.storage.sqlite.archive_tiers.write import ARCHIVE_BLOCK_ROW_COLUMNS

        blocks_spec, _, _ = self._specs()
        declared = {column.name for column in blocks_spec.all_columns}

        assert ARCHIVE_BLOCK_ROW_COLUMNS
        assert set(ARCHIVE_BLOCK_ROW_COLUMNS) <= declared
        assert len(ARCHIVE_BLOCK_ROW_COLUMNS) == len(set(ARCHIVE_BLOCK_ROW_COLUMNS))
        assert "tool_result_outcome_unknown_reason" in ARCHIVE_BLOCK_ROW_COLUMNS

    def test_blocks_insert_statement_format(self) -> None:
        """Verify that the INSERT statement can be correctly formatted."""
        blocks_spec, _, _ = self._specs()
        insert_sql = f"""
        INSERT INTO blocks (
            {blocks_spec.insert_column_names}
        ) VALUES ({blocks_spec.insert_placeholder_string})
        """
        # Should be valid SQL-like format
        assert "INSERT INTO blocks" in insert_sql
        assert "VALUES" in insert_sql
        # Should have correct number of columns
        col_count = len(blocks_spec.writable_columns)
        placeholder_count = blocks_spec.insert_placeholder_string.count("?")
        assert col_count == placeholder_count

    def test_session_events_spec_reordering_changes_rendered_sql(self) -> None:
        """A new table's column order must be load-bearing in rendered DDL."""
        _, _, table_specs = self._specs()
        spec = table_specs["session_events"]
        reordered = replace(
            spec,
            all_columns=tuple(reversed(spec.all_columns)),
            writable_columns=tuple(reversed(spec.writable_columns)),
        )

        assert spec.ddl_body != reordered.ddl_body
        assert "PRIMARY KEY(session_id, position)" in spec.ddl_body
        assert spec.writable_columns[0].name == "session_id"
        assert reordered.writable_columns[0].name == spec.writable_columns[-1].name

    def test_each_tier_ddl_uses_every_table_spec_it_declares(self) -> None:
        """Every table declaration renders into the executable schema of its own tier.

        ``TABLE_SPECS`` spans the index and embeddings tiers, so each spec is
        checked against the tier that owns it, and together the per-tier maps
        must cover the whole declaration set.
        """
        from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
        from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import (
            EMBEDDINGS_TABLE_SPECS,
            INDEX_TABLE_SPECS,
        )
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        blocks_spec, messages_spec, table_specs = self._specs()
        specs_by_tier = {
            ArchiveTier.INDEX: {"messages": messages_spec, "blocks": blocks_spec, **INDEX_TABLE_SPECS},
            ArchiveTier.EMBEDDINGS: EMBEDDINGS_TABLE_SPECS,
        }

        assert {name for specs in specs_by_tier.values() for name in specs} == set(table_specs)
        for tier, specs in specs_by_tier.items():
            tier_ddl = ARCHIVE_DDL_BY_TIER[tier]
            for name, spec in specs.items():
                assert spec.ddl_body in tier_ddl, f"{name} does not render into the {tier.value} tier"

    def test_archive_ddl_mapping_exposes_each_tier_script(self) -> None:
        """The public archive DDL map must preserve every tier's fresh-create script."""
        from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
        from polylogue.storage.sqlite.archive_tiers.audit import AUDIT_DDL
        from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDINGS_DDL
        from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL
        from polylogue.storage.sqlite.archive_tiers.ops import OPS_DDL
        from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
        from polylogue.storage.sqlite.archive_tiers.user import USER_DDL

        assert {
            ArchiveTier.AUDIT: AUDIT_DDL,
            ArchiveTier.EMBEDDINGS: EMBEDDINGS_DDL,
            ArchiveTier.INDEX: INDEX_DDL,
            ArchiveTier.OPS: OPS_DDL,
            ArchiveTier.SOURCE: SOURCE_DDL,
            ArchiveTier.USER: USER_DDL,
        } == ARCHIVE_DDL_BY_TIER

    def test_non_vector_tier_scripts_create_their_schema(self) -> None:
        """Fresh archive tiers must execute the scripts exported through the public map."""
        from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        for tier in (ArchiveTier.AUDIT, ArchiveTier.INDEX, ArchiveTier.OPS, ArchiveTier.SOURCE, ArchiveTier.USER):
            with sqlite3.connect(":memory:") as connection:
                connection.executescript(ARCHIVE_DDL_BY_TIER[tier])
                tables = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()

            assert tables, tier

    def test_tier_ddl_modules_build_fresh_create_scripts(self) -> None:
        """Tier DDL modules must build the scripts their public API exports."""
        from importlib import import_module, reload

        modules = {
            "polylogue.storage.sqlite.archive_tiers.audit": "AUDIT_DDL",
            "polylogue.storage.sqlite.archive_tiers.embeddings": "EMBEDDINGS_DDL",
            "polylogue.storage.sqlite.archive_tiers.index": "INDEX_DDL",
            "polylogue.storage.sqlite.archive_tiers.ops": "OPS_DDL",
            "polylogue.storage.sqlite.archive_tiers.source": "SOURCE_DDL",
            "polylogue.storage.sqlite.archive_tiers.user": "USER_DDL",
        }

        for module_name, ddl_name in modules.items():
            module = reload(import_module(module_name))
            ddl = getattr(module, ddl_name)

            assert isinstance(ddl, str)
            assert "CREATE" in ddl
