"""Tests for pure functions in polylogue.cli.commands.status."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.cli.commands.status import (
    _ARCHIVE_CLI_ROUTES,
    _ARCHIVE_FACADE_ROUTES,
    _ARCHIVE_TIER_ENUM,
    _BUILTIN_DAEMON_URL,
    _archive_cli_route_status,
    _archive_facade_route_status,
    _archive_one_tier_status,
    _archive_primary_tier_count,
    _archive_readiness_counts,
    _archive_route_count_summary,
    _archive_runtime_path_status,
    _archive_status_surfaces,
    _archive_table_counts,
    _archive_tier_files,
    _archive_tier_status,
    _column_exists,
    _default_daemon_url,
    _direct_archive_counts,
    _fast_count,
    _fmt_bytes,
    _table_exists,
)
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


class TestDefaultDaemonUrl:
    """Tests for _default_daemon_url()."""

    def test_returns_builtin_when_env_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without POLYLOGUE_DAEMON_URL env var, returns the built-in URL."""
        monkeypatch.delenv("POLYLOGUE_DAEMON_URL", raising=False)
        assert _default_daemon_url() == _BUILTIN_DAEMON_URL

    def test_returns_override_when_env_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With POLYLOGUE_DAEMON_URL env var, returns the override."""
        override_url = "http://custom.host:9999"
        monkeypatch.setenv("POLYLOGUE_DAEMON_URL", override_url)
        assert _default_daemon_url() == override_url

    def test_empty_env_var_returns_builtin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty string env var is falsy, returns built-in."""
        monkeypatch.setenv("POLYLOGUE_DAEMON_URL", "")
        assert _default_daemon_url() == _BUILTIN_DAEMON_URL


class TestFmtBytes:
    """Tests for _fmt_bytes(n)."""

    def test_small_bytes_formats_as_kb(self) -> None:
        """Values <1MB format as KB."""
        assert _fmt_bytes(500_000) == "500 KB"
        assert _fmt_bytes(100_000) == "100 KB"
        assert _fmt_bytes(1_000) == "1 KB"

    def test_megabytes_format_with_decimal(self) -> None:
        """Values >=1MB and <1GB format with one decimal place."""
        assert _fmt_bytes(5_000_000) == "5.0 MB"
        assert _fmt_bytes(1_500_000) == "1.5 MB"
        assert _fmt_bytes(1_000_000) == "1.0 MB"

    def test_gigabytes_format_with_decimal(self) -> None:
        """Values >=1GB format with one decimal place."""
        assert _fmt_bytes(5_000_000_000) == "5.0 GB"
        assert _fmt_bytes(1_500_000_000) == "1.5 GB"
        assert _fmt_bytes(1_000_000_000) == "1.0 GB"

    def test_boundary_values(self) -> None:
        """Test exact boundary values."""
        assert _fmt_bytes(999_000) == "999 KB"
        assert _fmt_bytes(1_000_000) == "1.0 MB"
        assert _fmt_bytes(999_900_000) == "999.9 MB"
        assert _fmt_bytes(1_000_000_000) == "1.0 GB"

    def test_zero_and_small_values(self) -> None:
        """Test zero and very small values."""
        assert _fmt_bytes(0) == "0 KB"
        assert _fmt_bytes(1) == "0 KB"  # rounds down


class TestArchiveFacadeRouteStatus:
    """Tests for _archive_facade_route_status()."""

    def test_returns_checked_true(self) -> None:
        """Always returns checked=True."""
        result = _archive_facade_route_status()
        assert result["checked"] is True

    def test_total_method_count_matches_routes(self) -> None:
        """total_method_count equals length of routes dict."""
        result = _archive_facade_route_status()
        assert result["total_method_count"] == len(_ARCHIVE_FACADE_ROUTES)
        assert result["total_method_count"] == len(result["routes"])

    def test_route_counts_sum_to_total(self) -> None:
        """Sum of route_counts values equals total_method_count."""
        result = _archive_facade_route_status()
        total = sum(result["route_counts"].values())
        assert total == result["total_method_count"]

    def test_tier_counts_include_all_tiers(self) -> None:
        """tier_counts includes all distinct tier values."""
        result = _archive_facade_route_status()
        expected_tiers = {"source", "index", "embeddings", "user", "none"}
        assert set(result["tier_counts"].keys()).issubset(expected_tiers)

    def test_unsupported_methods_empty_for_current_catalog(self) -> None:
        """Current catalog has no unsupported methods."""
        result = _archive_facade_route_status()
        assert result["unsupported_methods"] == []
        assert result["unsupported_method_count"] == 0

    def test_archive_ready_method_count_counts_routed_and_direct(self) -> None:
        """archive_ready_method_count sums archive_routed + archive_direct."""
        result = _archive_facade_route_status()
        routed = result["route_counts"].get("archive_routed", 0)
        direct = result["route_counts"].get("archive_direct", 0)
        assert result["archive_ready_method_count"] == routed + direct

    def test_each_route_has_required_keys(self) -> None:
        """Each route entry has route, tier, detail keys."""
        result = _archive_facade_route_status()
        for info in result["routes"].values():
            assert "route" in info
            assert "tier" in info
            assert "detail" in info
            assert isinstance(info["route"], str)
            assert isinstance(info["tier"], str)
            assert isinstance(info["detail"], str)

    def test_route_values_are_valid(self) -> None:
        """Each route value is one of the valid types."""
        result = _archive_facade_route_status()
        valid_routes = {"archive_routed", "archive_direct", "not_archive_runtime", "unsupported"}
        for info in result["routes"].values():
            assert info["route"] in valid_routes

    def test_tier_values_are_valid(self) -> None:
        """Each tier value is one of the valid tiers."""
        result = _archive_facade_route_status()
        valid_tiers = {"source", "index", "embeddings", "user", "none"}
        for info in result["routes"].values():
            assert info["tier"] in valid_tiers


class TestArchiveCliRouteStatus:
    """Tests for _archive_cli_route_status()."""

    def test_returns_checked_true(self) -> None:
        """Always returns checked=True."""
        result = _archive_cli_route_status()
        assert result["checked"] is True

    def test_total_command_count_matches_routes(self) -> None:
        """total_command_count equals length of _ARCHIVE_CLI_ROUTES."""
        result = _archive_cli_route_status()
        assert result["total_command_count"] == len(_ARCHIVE_CLI_ROUTES)
        assert result["total_command_count"] == len(result["routes"])

    def test_route_counts_sum_to_total(self) -> None:
        """Sum of route_counts equals total_command_count."""
        result = _archive_cli_route_status()
        total = sum(result["route_counts"].values())
        assert total == result["total_command_count"]

    def test_unsupported_commands_empty_for_current_catalog(self) -> None:
        """Current catalog has no unsupported commands."""
        result = _archive_cli_route_status()
        assert result["unsupported_commands"] == []
        assert result["unsupported_command_count"] == 0

    def test_archive_ready_command_count(self) -> None:
        """archive_ready_command_count sums archive_routed + archive_direct."""
        result = _archive_cli_route_status()
        routed = result["route_counts"].get("archive_routed", 0)
        direct = result["route_counts"].get("archive_direct", 0)
        assert result["archive_ready_command_count"] == routed + direct


class TestArchiveRuntimePathStatus:
    """Tests for _archive_runtime_path_status()."""

    def test_returns_checked_true(self) -> None:
        """Always returns checked=True."""
        result = _archive_runtime_path_status()
        assert result["checked"] is True

    def test_archive_routing_ready_when_no_blockers(self) -> None:
        """archive_routing_ready is True when no unsupported primary methods."""
        result = _archive_runtime_path_status()
        assert result["archive_routing_ready"] is True
        # Kept as a compatibility alias for older status consumers.
        assert result["archive_runtime_ready"] is True
        assert result["final_shape_blockers"] == []

    def test_primary_ingest_store_is_archive_file_set(self) -> None:
        """primary_ingest_store reports archive_file_set for current catalog."""
        result = _archive_runtime_path_status()
        assert result["primary_ingest_store"] == "archive_file_set"

    def test_ingest_write_mode_is_archive(self) -> None:
        """ingest_write_mode is 'archive' for current catalog."""
        result = _archive_runtime_path_status()
        assert result["ingest_write_mode"] == "archive"

    def test_archive_ingest_write_targets(self) -> None:
        """archive_ingest_write_targets includes source.db and index.db."""
        result = _archive_runtime_path_status()
        assert set(result["archive_ingest_write_targets"]) == {"source.db", "index.db"}

    def test_archive_tier_targets_complete(self) -> None:
        """archive_tier_targets includes all five tiers."""
        result = _archive_runtime_path_status()
        assert set(result["archive_tier_targets"]) == {"source.db", "index.db", "embeddings.db", "user.db", "ops.db"}

    def test_facade_tier_route_counts_populated(self) -> None:
        """facade_tier_route_counts is populated from facade status."""
        result = _archive_runtime_path_status()
        assert isinstance(result["facade_tier_route_counts"], dict)
        assert len(result["facade_tier_route_counts"]) > 0

    def test_cli_tier_route_counts_populated(self) -> None:
        """cli_tier_route_counts is populated from CLI status."""
        result = _archive_runtime_path_status()
        assert isinstance(result["cli_tier_route_counts"], dict)
        assert len(result["cli_tier_route_counts"]) > 0


class TestArchiveRouteCountSummary:
    """Tests for _archive_route_count_summary()."""

    def test_none_returns_none(self) -> None:
        """None input returns 'none'."""
        assert _archive_route_count_summary(None) == "none"

    def test_empty_dict_returns_none(self) -> None:
        """Empty dict returns 'none'."""
        assert _archive_route_count_summary({}) == "none"

    def test_non_dict_returns_none(self) -> None:
        """Non-dict input returns 'none'."""
        assert _archive_route_count_summary([]) == "none"
        assert _archive_route_count_summary("invalid") == "none"

    def test_single_tier_count(self) -> None:
        """Single tier formats correctly."""
        result = _archive_route_count_summary({"index": 3})
        assert result == "index:3"

    def test_multiple_tiers_sorted(self) -> None:
        """Multiple tiers are sorted alphabetically by key."""
        result = _archive_route_count_summary({"index": 3, "user": 2, "source": 1})
        assert result == "index:3,source:1,user:2"

    def test_preserves_counts(self) -> None:
        """Counts are accurately preserved."""
        counts = {"a": 10, "z": 5, "m": 20}
        result = _archive_route_count_summary(counts)
        assert "a:10" in result
        assert "z:5" in result
        assert "m:20" in result


class TestArchivePrimaryTierCount:
    """Tests for _archive_primary_tier_count()."""

    def test_index_tier_with_sessions(self) -> None:
        """index tier with sessions in counts returns ('sessions', count)."""
        result = _archive_primary_tier_count("index", {"sessions": 42})
        assert result == ("sessions", 42)

    def test_source_tier_with_raw_sessions(self) -> None:
        """source tier with raw_sessions in counts returns ('raw_sessions', count)."""
        result = _archive_primary_tier_count("source", {"raw_sessions": 100})
        assert result == ("raw_sessions", 100)

    def test_user_tier_with_annotations(self) -> None:
        """user tier with annotations in counts returns ('annotations', count)."""
        result = _archive_primary_tier_count("user", {"annotations": 5})
        assert result == ("annotations", 5)

    def test_embeddings_tier_with_embedding_status(self) -> None:
        """embeddings tier with embedding_status returns ('embedding_status', count)."""
        result = _archive_primary_tier_count("embeddings", {"embedding_status": 1})
        assert result == ("embedding_status", 1)

    def test_ops_tier_with_ingest_attempts(self) -> None:
        """ops tier with ingest_attempts returns ('ingest_attempts', count)."""
        result = _archive_primary_tier_count("ops", {"ingest_attempts": 10})
        assert result == ("ingest_attempts", 10)

    def test_returns_none_when_primary_table_missing(self) -> None:
        """Returns None when primary table not in counts."""
        result = _archive_primary_tier_count("index", {})
        assert result is None

    def test_returns_none_for_unknown_tier(self) -> None:
        """Returns None for unknown tier."""
        result = _archive_primary_tier_count("unknown", {"anything": 5})
        assert result is None


class TestArchiveTierFiles:
    """Tests for _archive_tier_files()."""

    def test_returns_all_five_tiers(self, tmp_path: Path) -> None:
        """Returns dict with all five tier names."""
        result = _archive_tier_files(tmp_path)
        expected_tiers = {"source", "index", "embeddings", "user", "ops"}
        assert set(result.keys()) == expected_tiers

    def test_each_tier_maps_to_correct_path(self, tmp_path: Path) -> None:
        """Each tier maps to the correct .db file path."""
        result = _archive_tier_files(tmp_path)
        assert result["source"] == tmp_path / "source.db"
        assert result["index"] == tmp_path / "index.db"
        assert result["embeddings"] == tmp_path / "embeddings.db"
        assert result["user"] == tmp_path / "user.db"
        assert result["ops"] == tmp_path / "ops.db"

    def test_paths_are_path_objects(self, tmp_path: Path) -> None:
        """Returned values are Path objects."""
        result = _archive_tier_files(tmp_path)
        for path in result.values():
            assert isinstance(path, Path)


class TestFastCount:
    """Tests for _fast_count()."""

    def test_counts_rows_in_table(self, tmp_path: Path) -> None:
        """Counts rows correctly from a simple table."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO test (id) VALUES (1), (2), (3)")
            conn.commit()
            result = _fast_count(conn, "SELECT COUNT(*) FROM test")
            assert result == 3
        finally:
            conn.close()

    def test_returns_zero_for_empty_table(self, tmp_path: Path) -> None:
        """Returns 0 for empty table."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
            conn.commit()
            result = _fast_count(conn, "SELECT COUNT(*) FROM test")
            assert result == 0
        finally:
            conn.close()

    def test_uses_parameters(self, tmp_path: Path) -> None:
        """Correctly uses parameterized queries."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'a'), (2, 'b'), (3, 'a')")
            conn.commit()
            result = _fast_count(conn, "SELECT COUNT(*) FROM test WHERE value = ?", ("a",))
            assert result == 2
        finally:
            conn.close()

    def test_handles_null_result(self, tmp_path: Path) -> None:
        """Handles NULL return value gracefully."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
            result = _fast_count(conn, "SELECT NULL")
            assert result == 0
        finally:
            conn.close()


class TestTableExists:
    """Tests for _table_exists()."""

    def test_returns_true_for_existing_table(self, tmp_path: Path) -> None:
        """Returns True when table exists."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE mytable (id INTEGER PRIMARY KEY)")
            assert _table_exists(conn, "mytable") is True
        finally:
            conn.close()

    def test_returns_false_for_nonexistent_table(self, tmp_path: Path) -> None:
        """Returns False when table does not exist."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            assert _table_exists(conn, "nonexistent") is False
        finally:
            conn.close()

    def test_works_with_sqlite_master(self, tmp_path: Path) -> None:
        """Correctly queries sqlite_master."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE first (id INTEGER)")
            conn.execute("CREATE TABLE second (id INTEGER)")
            assert _table_exists(conn, "first") is True
            assert _table_exists(conn, "second") is True
            assert _table_exists(conn, "third") is False
        finally:
            conn.close()


class TestColumnExists:
    """Tests for _column_exists()."""

    def test_returns_true_for_existing_column(self, tmp_path: Path) -> None:
        """Returns True when column exists."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE mytable (id INTEGER, name TEXT)")
            assert _column_exists(conn, "mytable", "id") is True
            assert _column_exists(conn, "mytable", "name") is True
        finally:
            conn.close()

    def test_returns_false_for_nonexistent_column(self, tmp_path: Path) -> None:
        """Returns False when column does not exist."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE mytable (id INTEGER)")
            assert _column_exists(conn, "mytable", "missing") is False
        finally:
            conn.close()

    def test_works_with_multiple_columns(self, tmp_path: Path) -> None:
        """Works correctly with multiple columns."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE mytable (id INTEGER, name TEXT, value REAL, active INTEGER)")
            assert _column_exists(conn, "mytable", "id") is True
            assert _column_exists(conn, "mytable", "name") is True
            assert _column_exists(conn, "mytable", "value") is True
            assert _column_exists(conn, "mytable", "active") is True
            assert _column_exists(conn, "mytable", "missing") is False
        finally:
            conn.close()


class TestArchiveTableCounts:
    """Tests for _archive_table_counts()."""

    def test_counts_existing_tables(self, tmp_path: Path) -> None:
        """Counts all existing tables in the list."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY)")
            conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO sessions VALUES (1), (2)")
            conn.execute("INSERT INTO messages VALUES (1), (2), (3)")
            conn.commit()
            result, precision = _archive_table_counts(conn, ["sessions", "messages"], db_size_bytes=0)
            assert result["sessions"] == 2
            assert result["messages"] == 3
            assert precision == {"sessions": "exact", "messages": "exact"}
        finally:
            conn.close()

    def test_omits_nonexistent_tables(self, tmp_path: Path) -> None:
        """Only returns counts for tables that exist."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO sessions VALUES (1)")
            conn.commit()
            result, precision = _archive_table_counts(
                conn,
                ["sessions", "nonexistent", "missing"],
                db_size_bytes=0,
            )
            assert "sessions" in result
            assert "nonexistent" not in result
            assert "missing" not in result
            assert precision == {"sessions": "exact"}
        finally:
            conn.close()

    def test_empty_list_returns_empty_dict(self, tmp_path: Path) -> None:
        """Empty table list returns empty dict."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            result, precision = _archive_table_counts(conn, [], db_size_bytes=0)
            assert result == {}
            assert precision == {}
        finally:
            conn.close()


class TestArchiveOneTierStatus:
    """Tests for _archive_one_tier_status()."""

    def test_missing_file_returns_missing_status(self, tmp_path: Path) -> None:
        """Missing file returns version_status='missing'."""
        nonexistent = tmp_path / "nonexistent.db"
        result = _archive_one_tier_status("index", nonexistent)
        assert result["exists"] is False
        assert result["version_status"] == "missing"
        assert result["size_bytes"] is None
        assert result["user_version"] is None

    def test_existing_empty_file_with_correct_version(self, tmp_path: Path) -> None:
        """Existing file with correct version returns 'ok'."""
        db_path = tmp_path / "index.db"
        conn = sqlite3.connect(db_path)
        expected_version = ARCHIVE_VERSION_BY_TIER[_ARCHIVE_TIER_ENUM["index"]]
        conn.execute(f"PRAGMA user_version = {expected_version}")
        conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        result = _archive_one_tier_status("index", db_path)
        assert result["exists"] is True
        assert result["user_version"] == expected_version
        assert result["expected_user_version"] == expected_version
        assert result["version_status"] == "ok"
        assert result["size_bytes"] is not None
        assert result["table_counts"]["sessions"] == 0

    def test_version_mismatch(self, tmp_path: Path) -> None:
        """File with mismatched version returns 'mismatch'."""
        db_path = tmp_path / "index.db"
        conn = sqlite3.connect(db_path)
        expected_version = ARCHIVE_VERSION_BY_TIER[_ARCHIVE_TIER_ENUM["index"]]
        wrong_version = expected_version + 999
        conn.execute(f"PRAGMA user_version = {wrong_version}")
        conn.commit()
        conn.close()

        result = _archive_one_tier_status("index", db_path)
        assert result["user_version"] == wrong_version
        assert result["expected_user_version"] == expected_version
        assert result["version_status"] == "mismatch"

    def test_table_counts_include_existing_tables(self, tmp_path: Path) -> None:
        """table_counts includes rows from existing tables."""
        db_path = tmp_path / "index.db"
        conn = sqlite3.connect(db_path)
        expected_version = ARCHIVE_VERSION_BY_TIER[_ARCHIVE_TIER_ENUM["index"]]
        conn.execute(f"PRAGMA user_version = {expected_version}")
        conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO sessions VALUES (1), (2)")
        conn.execute("INSERT INTO messages VALUES (1)")
        conn.commit()
        conn.close()

        result = _archive_one_tier_status("index", db_path)
        assert result["table_counts"]["sessions"] == 2
        assert result["table_counts"]["messages"] == 1


class TestArchiveTierStatus:
    """Tests for _archive_tier_status()."""

    def test_returns_status_for_all_tiers(self, tmp_path: Path) -> None:
        """Returns status dict for all five tiers."""
        result = _archive_tier_status(tmp_path)
        expected_tiers = {"source", "index", "embeddings", "user", "ops"}
        assert set(result.keys()) == expected_tiers

    def test_all_missing_tiers(self, tmp_path: Path) -> None:
        """When no tier files exist, all show as missing."""
        result = _archive_tier_status(tmp_path)
        for status in result.values():
            assert status["exists"] is False
            assert status["version_status"] == "missing"

    def test_with_existing_index_tier(self, tmp_path: Path) -> None:
        """Correctly detects existing index.db."""
        db_path = tmp_path / "index.db"
        conn = sqlite3.connect(db_path)
        expected_version = ARCHIVE_VERSION_BY_TIER[_ARCHIVE_TIER_ENUM["index"]]
        conn.execute(f"PRAGMA user_version = {expected_version}")
        conn.commit()
        conn.close()

        result = _archive_tier_status(tmp_path)
        assert result["index"]["exists"] is True
        assert result["index"]["version_status"] == "ok"
        assert result["source"]["exists"] is False
        assert result["user"]["exists"] is False


class TestDirectArchiveCounts:
    """Tests for _direct_archive_counts()."""

    def test_empty_archive_returns_zeros(self, tmp_path: Path) -> None:
        """Archive with no sessions returns zeros."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            result = _direct_archive_counts(conn)
            assert result["sessions"] == 0
            assert result["messages"] == 0
            assert result["raw_records"] == 0
        finally:
            conn.close()

    def test_with_sessions_and_message_count_column(self, tmp_path: Path) -> None:
        """Uses message_count column when available."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, message_count INTEGER)")
            conn.execute("INSERT INTO sessions VALUES (1, 5), (2, 3)")
            conn.commit()
            result = _direct_archive_counts(conn)
            assert result["sessions"] == 2
            assert result["messages"] == 8  # sum of message_count
        finally:
            conn.close()

    def test_with_sessions_and_messages_table(self, tmp_path: Path) -> None:
        """Falls back to counting messages table when message_count column absent."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY)")
            conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id INTEGER)")
            conn.execute("INSERT INTO sessions VALUES (1), (2)")
            conn.execute("INSERT INTO messages VALUES (1, 1), (2, 1), (3, 2)")
            conn.commit()
            result = _direct_archive_counts(conn)
            assert result["sessions"] == 2
            assert result["messages"] == 3
        finally:
            conn.close()


class TestArchiveReadinessCounts:
    """Tests for _archive_readiness_counts()."""

    def test_basic_counts_with_empty_archive(self, tmp_path: Path) -> None:
        """Counts all zeros for empty archive."""
        db_path = tmp_path / "test.db"
        initialize_archive_database(db_path, ArchiveTier.INDEX)
        conn = sqlite3.connect(db_path)
        try:
            result = _archive_readiness_counts(conn, source_conn=None, source_check_available=False)
            assert result["session_count"] == 0
            assert result["message_count"] == 0
            assert result["raw_link_count"] == 0
        finally:
            conn.close()

    def test_counts_with_data(self, tmp_path: Path) -> None:
        """Counts correctly with data present."""
        db_path = tmp_path / "test.db"
        initialize_archive_database(db_path, ArchiveTier.INDEX)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                """
                INSERT INTO sessions (native_id, origin, raw_id, content_hash)
                VALUES (?, 'codex-session', ?, ?), (?, 'codex-session', ?, ?)
                """,
                ("native-1", "raw1", b"1" * 32, "native-2", "raw2", b"2" * 32),
            )
            conn.executemany(
                """
                INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash)
                VALUES (?, ?, ?, 'user', 'message', ?)
                """,
                [
                    ("codex-session:native-1", "m1", 0, b"a" * 32),
                    ("codex-session:native-1", "m2", 1, b"b" * 32),
                    ("codex-session:native-2", "m3", 0, b"c" * 32),
                ],
            )
            conn.commit()
            result = _archive_readiness_counts(conn, source_conn=None, source_check_available=False)
            assert result["session_count"] == 2
            assert result["message_count"] == 3
            assert result["raw_link_count"] == 2
        finally:
            conn.close()

    def test_missing_raw_session_detection(self, tmp_path: Path) -> None:
        """Detects sessions with raw_id that don't exist in source."""
        db_path = tmp_path / "test.db"
        source_path = tmp_path / "source.db"
        initialize_archive_database(db_path, ArchiveTier.INDEX)
        initialize_archive_database(source_path, ArchiveTier.SOURCE)
        conn = sqlite3.connect(db_path)
        source_conn = sqlite3.connect(source_path)
        try:
            conn.execute(
                """
                INSERT INTO sessions (native_id, origin, raw_id, content_hash)
                VALUES
                    ('native-1', 'codex-session', 'raw1', ?),
                    ('native-2', 'codex-session', 'raw2', ?),
                    ('native-3', 'codex-session', 'raw3', ?)
                """,
                (b"1" * 32, b"2" * 32, b"3" * 32),
            )
            source_conn.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
                ) VALUES
                    ('raw1', 'codex-session', 'native-1', 'one.jsonl', 0, ?, 32, 1),
                    ('raw2', 'codex-session', 'native-2', 'two.jsonl', 0, ?, 32, 2)
                """,
                (bytes.fromhex("1" * 64), bytes.fromhex("2" * 64)),
            )
            conn.commit()
            source_conn.commit()
            result = _archive_readiness_counts(conn, source_conn=source_conn, source_check_available=True)
            assert result["missing_raw_session_count"] == 1  # raw3 is missing
        finally:
            conn.close()
            source_conn.close()


class TestArchiveStatusSurfaces:
    """Tests for _archive_status_surfaces()."""

    def test_archive_sessions_always_ready(self) -> None:
        """archive_sessions surface is always ready=True."""
        counts: dict[str, int] = {
            "session_count": 0,
            "raw_link_count": 0,
            "missing_raw_session_count": 0,
            "message_count": 0,
            "text_block_count": 0,
            "messages_fts_count": 0,
            "profile_row_count": 0,
            "missing_profile_row_count": 0,
            "work_event_row_count": 0,
            "missing_work_events_materialization": 0,
            "phase_row_count": 0,
            "missing_phases_materialization": 0,
            "thread_count": 0,
            "missing_thread_materialization": 0,
            "action_count": 0,
            "missing_session_profile_materialization": 0,
            "missing_latency_materialization": 0,
        }
        result = _archive_status_surfaces(counts, source_check_available=True)
        assert result["archive_sessions"]["ready"] is True
        assert result["archive_sessions"]["blockers"] == []

    def test_tool_usage_always_ready(self) -> None:
        """tool_usage surface is always ready=True."""
        counts: dict[str, int] = {
            "session_count": 0,
            "raw_link_count": 0,
            "missing_raw_session_count": 0,
            "message_count": 0,
            "text_block_count": 0,
            "messages_fts_count": 0,
            "profile_row_count": 0,
            "missing_profile_row_count": 0,
            "work_event_row_count": 0,
            "missing_work_events_materialization": 0,
            "phase_row_count": 0,
            "missing_phases_materialization": 0,
            "thread_count": 0,
            "missing_thread_materialization": 0,
            "action_count": 0,
            "missing_session_profile_materialization": 0,
            "missing_latency_materialization": 0,
        }
        result = _archive_status_surfaces(counts, source_check_available=True)
        assert result["tool_usage"]["ready"] is True

    def test_raw_artifacts_unavailable_when_source_check_unavailable(self) -> None:
        """raw_artifacts shows ready=None when source_check_available is False."""
        counts: dict[str, int] = {
            "session_count": 1,
            "raw_link_count": 1,
            "missing_raw_session_count": 0,
            "message_count": 1,
            "text_block_count": 1,
            "messages_fts_count": 1,
            "profile_row_count": 0,
            "missing_profile_row_count": 0,
            "work_event_row_count": 0,
            "missing_work_events_materialization": 0,
            "phase_row_count": 0,
            "missing_phases_materialization": 0,
            "thread_count": 0,
            "missing_thread_materialization": 0,
            "action_count": 0,
            "missing_session_profile_materialization": 0,
            "missing_latency_materialization": 0,
        }
        result = _archive_status_surfaces(counts, source_check_available=False)
        assert result["raw_artifacts"]["ready"] is None
        assert "source_tier_unavailable" in result["raw_artifacts"]["blockers"]

    def test_raw_artifacts_failed_when_missing(self) -> None:
        """raw_artifacts shows ready=False when missing_raw_session_count > 0."""
        counts: dict[str, int] = {
            "session_count": 1,
            "raw_link_count": 1,
            "missing_raw_session_count": 1,
            "message_count": 1,
            "text_block_count": 1,
            "messages_fts_count": 1,
            "profile_row_count": 0,
            "missing_profile_row_count": 0,
            "work_event_row_count": 0,
            "missing_work_events_materialization": 0,
            "phase_row_count": 0,
            "missing_phases_materialization": 0,
            "thread_count": 0,
            "missing_thread_materialization": 0,
            "action_count": 0,
            "missing_session_profile_materialization": 0,
            "missing_latency_materialization": 0,
        }
        result = _archive_status_surfaces(counts, source_check_available=True)
        assert result["raw_artifacts"]["ready"] is False
        assert "missing_source_raw_sessions" in result["raw_artifacts"]["blockers"]

    def test_search_fts_mismatch_blocker(self) -> None:
        """search surface blocked when text_block_count != messages_fts_count."""
        counts: dict[str, int] = {
            "session_count": 1,
            "raw_link_count": 0,
            "missing_raw_session_count": 0,
            "message_count": 10,
            "text_block_count": 10,
            "messages_fts_count": 8,  # mismatch
            "profile_row_count": 0,
            "missing_profile_row_count": 0,
            "work_event_row_count": 0,
            "missing_work_events_materialization": 0,
            "phase_row_count": 0,
            "missing_phases_materialization": 0,
            "thread_count": 0,
            "missing_thread_materialization": 0,
            "action_count": 0,
            "missing_session_profile_materialization": 0,
            "missing_latency_materialization": 0,
        }
        result = _archive_status_surfaces(counts, source_check_available=True)
        assert result["search"]["ready"] is False
        assert "messages_fts_row_mismatch" in result["search"]["blockers"]

    def test_materialization_blockers(self) -> None:
        """Surfaces blocked when materialization missing."""
        counts: dict[str, int] = {
            "session_count": 2,
            "raw_link_count": 0,
            "missing_raw_session_count": 0,
            "message_count": 1,
            "text_block_count": 1,
            "messages_fts_count": 1,
            "profile_row_count": 0,
            "missing_profile_row_count": 0,
            "work_event_row_count": 0,
            "missing_work_events_materialization": 1,  # missing for 1 session
            "phase_row_count": 0,
            "missing_phases_materialization": 2,  # missing for all 2 sessions
            "thread_count": 0,
            "missing_thread_materialization": 0,
            "action_count": 0,
            "missing_session_profile_materialization": 0,
            "missing_latency_materialization": 0,
        }
        result = _archive_status_surfaces(counts, source_check_available=True)
        assert result["timeline_work_events"]["ready"] is False
        assert "missing_work_events_materialization" in result["timeline_work_events"]["blockers"]
        assert result["timeline_phases"]["ready"] is False
        assert "missing_phases_materialization" in result["timeline_phases"]["blockers"]

    def test_timeline_surfaces_block_on_stale_or_mismatched_rows(self) -> None:
        """Timeline surfaces use canonical readiness shape, not materialization presence alone."""
        counts: dict[str, int] = {
            "session_count": 2,
            "raw_link_count": 0,
            "missing_raw_session_count": 0,
            "message_count": 1,
            "text_block_count": 1,
            "messages_fts_count": 1,
            "profile_row_count": 2,
            "missing_profile_row_count": 0,
            "missing_session_profile_materialization": 0,
            "work_event_row_count": 1,
            "expected_work_event_row_count": 3,
            "stale_work_event_row_count": 1,
            "orphan_work_event_row_count": 0,
            "missing_work_events_materialization": 0,
            "phase_row_count": 4,
            "expected_phase_row_count": 2,
            "stale_phase_row_count": 0,
            "orphan_phase_row_count": 1,
            "missing_phases_materialization": 0,
            "thread_count": 1,
            "root_thread_count": 2,
            "stale_thread_count": 1,
            "orphan_thread_count": 0,
            "missing_thread_materialization": 0,
            "action_count": 0,
            "missing_latency_materialization": 0,
        }

        result = _archive_status_surfaces(counts, source_check_available=True)

        assert result["timeline_work_events"]["ready"] is False
        assert result["timeline_work_events"]["blockers"] == [
            "stale_work_event_row_count",
            "work_event_row_mismatch",
        ]
        assert result["timeline_work_events"]["evidence"]["expected_work_event_row_count"] == 3
        assert result["timeline_phases"]["ready"] is False
        assert result["timeline_phases"]["blockers"] == [
            "orphan_phase_row_count",
            "phase_row_mismatch",
        ]
        assert result["threads"]["ready"] is False
        assert result["threads"]["blockers"] == [
            "stale_thread_count",
            "thread_root_mismatch",
        ]

    def test_all_surfaces_present(self) -> None:
        """All expected surfaces are present in result."""
        counts: dict[str, int] = {
            "session_count": 0,
            "raw_link_count": 0,
            "missing_raw_session_count": 0,
            "message_count": 0,
            "text_block_count": 0,
            "messages_fts_count": 0,
            "profile_row_count": 0,
            "missing_profile_row_count": 0,
            "work_event_row_count": 0,
            "missing_work_events_materialization": 0,
            "phase_row_count": 0,
            "missing_phases_materialization": 0,
            "thread_count": 0,
            "missing_thread_materialization": 0,
            "action_count": 0,
            "missing_session_profile_materialization": 0,
            "missing_latency_materialization": 0,
        }
        result = _archive_status_surfaces(counts, source_check_available=True)
        expected_surfaces = {
            "archive_sessions",
            "raw_artifacts",
            "search",
            "session_profiles",
            "timeline_work_events",
            "timeline_phases",
            "threads",
            "tool_usage",
            "latency_profiles",
        }
        assert set(result.keys()) == expected_surfaces
