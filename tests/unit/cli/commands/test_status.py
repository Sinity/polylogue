"""Tests for CLI status adapters and their canonical read producers."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

import pytest

from polylogue.cli.commands.status import (
    _BUILTIN_DAEMON_URL,
    _archive_primary_tier_count,
    _default_daemon_url,
    _fmt_bytes,
)
from polylogue.operations.daemon_status import _archive_tiers
from polylogue.storage.archive_identity import archive_file_set_root

# polylogue-ogn1: archive-readiness lives in the substrate so substrate callers
# reach it without importing a surface. status.py delegates here rather than
# owning a second copy.
from polylogue.storage.archive_readiness import (
    _action_readiness_counts,
    _archive_readiness_counts,
    _archive_status_surfaces,
    _fast_count,
    _view_exists,
    probe_archive_tier,
)
from polylogue.storage.introspection import column_exists as _column_exists
from polylogue.storage.introspection import table_exists as _table_exists
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.archive_templates import bootstrap_archive_root


class _ArchiveTierResult(TypedDict):
    table_counts: dict[str, int]
    table_count_precision: dict[str, str]


@dataclass(frozen=True)
class _ArchiveFixture:
    archive_root: Path
    operation_schema_versions: dict[str, int]


class _ArchiveStats(TypedDict):
    total_sessions: int
    total_messages: int


class _DirectStatusPayload(TypedDict):
    archive_stats: _ArchiveStats
    archive_tiers: dict[str, _ArchiveTierResult]


_DaemonTierName = Literal["source", "index", "embeddings", "user", "ops"]


def _archive_fixture(root: Path, version: int) -> ArchiveStore:
    return cast(
        ArchiveStore,
        _ArchiveFixture(archive_root=root, operation_schema_versions={"index": version}),
    )


def _daemon_tier_name(value: str) -> _DaemonTierName:
    """Adapt the all-tier fixture vocabulary to the daemon helper contract."""

    return cast(_DaemonTierName, value)


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

    def test_user_tier_with_assertions(self) -> None:
        """user tier with assertions in counts returns ('assertions', count)."""
        result = _archive_primary_tier_count("user", {"assertions": 5})
        assert result == ("assertions", 5)

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


class TestArchiveFileSetRoot:
    """The canonical operation root follows configured index topology."""

    def test_default_index_path_anchors_file_set(self, tmp_path: Path) -> None:
        assert archive_file_set_root(archive_root=tmp_path, db_path=tmp_path / "index.db") == tmp_path

    def test_explicit_index_path_anchors_split_file_set(self, tmp_path: Path) -> None:
        generation = tmp_path / "generation-1"
        assert archive_file_set_root(archive_root=tmp_path, db_path=generation / "index.db") == generation

    def test_bootstrap_materializes_each_canonical_tier_path(self, tmp_path: Path) -> None:
        bootstrap_archive_root(tmp_path)
        expected_tiers = {"source", "index", "embeddings", "user", "audit", "ops"}
        assert {path.stem for path in tmp_path.glob("*.db")} == expected_tiers


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

    def test_view_is_not_a_table_but_is_a_schema_view(self, tmp_path: Path) -> None:
        conn = sqlite3.connect(tmp_path / "test.db")
        try:
            conn.execute("CREATE TABLE base (id INTEGER)")
            conn.execute("CREATE VIEW derived AS SELECT id FROM base")
            assert _table_exists(conn, "derived") is False
            assert _view_exists(conn, "derived") is True
        finally:
            conn.close()


class TestActionReadinessCounts:
    def test_real_index_schema_exposes_queryable_empty_actions_view(self, tmp_path: Path) -> None:
        db_path = tmp_path / "index.db"
        initialize_archive_database(db_path, ArchiveTier.INDEX)
        with sqlite3.connect(db_path) as conn:
            assert _action_readiness_counts(conn) == {
                "action_count": 0,
                "tool_use_block_count": 0,
                "actions_view_present": True,
                "actions_view_error": None,
            }

    def test_removed_actions_view_is_reported_absent(self, tmp_path: Path) -> None:
        db_path = tmp_path / "index.db"
        initialize_archive_database(db_path, ArchiveTier.INDEX)
        with sqlite3.connect(db_path) as conn:
            conn.execute("DROP VIEW actions")
            counts = _action_readiness_counts(conn)
        assert counts["actions_view_present"] is False

    def test_broken_actions_view_is_reported_unreadable(self, tmp_path: Path) -> None:
        db_path = tmp_path / "index.db"
        initialize_archive_database(db_path, ArchiveTier.INDEX)
        with sqlite3.connect(db_path) as conn:
            conn.execute("DROP VIEW actions")
            conn.execute("CREATE VIEW actions AS SELECT * FROM missing_action_source")
            counts = _action_readiness_counts(conn)
        assert counts["actions_view_present"] is True
        assert counts["actions_view_error"] is not None

    def test_partial_actions_view_exposes_exact_cardinality_mismatch(self, tmp_path: Path) -> None:
        with sqlite3.connect(tmp_path / "partial.db") as conn:
            conn.execute("CREATE TABLE blocks (block_type TEXT NOT NULL)")
            conn.executemany("INSERT INTO blocks VALUES (?)", [("tool_use",), ("tool_use",)])
            conn.execute("CREATE INDEX blocks_type_idx ON blocks(block_type)")
            conn.execute("CREATE VIEW actions AS SELECT rowid FROM blocks WHERE block_type = 'tool_use' AND rowid = 1")
            counts = _action_readiness_counts(conn)
        result = _archive_status_surfaces({**counts, "missing_latency_materialization": 0}, source_check_available=True)
        assert counts["tool_use_block_count"] == 2
        assert counts["action_count"] == 1
        assert result["tool_usage"]["ready"] is False
        assert result["tool_usage"]["blockers"] == ["actions_tool_use_count_mismatch"]


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
    """Pinned operation tier counts retain exact workload evidence."""

    def test_counts_existing_tables(self, tmp_path: Path) -> None:
        """Counts all present tables in the declared index-tier workload."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY)")
            conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO sessions VALUES (1), (2)")
            conn.execute("INSERT INTO messages VALUES (1), (2), (3)")
            conn.commit()
            result = cast(
                _ArchiveTierResult,
                _archive_tiers(_archive_fixture(tmp_path, ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]), conn)["index"],
            )
            assert result["table_counts"]["sessions"] == 2
            assert result["table_counts"]["messages"] == 3
            assert result["table_count_precision"] == {"sessions": "exact", "messages": "exact"}
        finally:
            conn.close()

    def test_omits_nonexistent_tables(self, tmp_path: Path) -> None:
        """Only returns counts for declared tables that exist."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO sessions VALUES (1)")
            conn.commit()
            result = cast(
                _ArchiveTierResult,
                _archive_tiers(_archive_fixture(tmp_path, ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]), conn)["index"],
            )
            assert "sessions" in result["table_counts"]
            assert "nonexistent" not in result["table_counts"]
            assert "missing" not in result["table_counts"]
            assert result["table_count_precision"] == {"sessions": "exact"}
        finally:
            conn.close()


class TestArchiveOneTierStatus:
    """Tests for the canonical archive tier probe."""

    def test_missing_file_returns_missing_status(self, tmp_path: Path) -> None:
        """Missing file returns version_status='missing'."""
        nonexistent = tmp_path / "nonexistent.db"
        result = probe_archive_tier(ArchiveTier.INDEX, nonexistent)
        assert result.exists is False
        assert result.version_status == "missing"
        assert result.size_bytes == 0
        assert result.user_version is None

    def test_existing_empty_file_with_correct_version(self, tmp_path: Path) -> None:
        """Existing file with correct version returns 'ok'."""
        db_path = tmp_path / "index.db"
        conn = sqlite3.connect(db_path)
        expected_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]
        conn.execute(f"PRAGMA user_version = {expected_version}")
        conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        result = probe_archive_tier(ArchiveTier.INDEX, db_path)
        assert result.exists is True
        assert result.user_version == expected_version
        assert result.expected_user_version == expected_version
        assert result.version_status == "ok"
        assert result.size_bytes > 0

    def test_version_mismatch(self, tmp_path: Path) -> None:
        """File with mismatched version returns 'mismatch'."""
        db_path = tmp_path / "index.db"
        conn = sqlite3.connect(db_path)
        expected_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]
        wrong_version = expected_version + 999
        conn.execute(f"PRAGMA user_version = {wrong_version}")
        conn.commit()
        conn.close()

        result = probe_archive_tier(ArchiveTier.INDEX, db_path)
        assert result.user_version == wrong_version
        assert result.expected_user_version == expected_version
        assert result.version_status == "mismatch"

    def test_table_counts_include_existing_tables(self, tmp_path: Path) -> None:
        """table_counts includes rows from existing tables."""
        db_path = tmp_path / "index.db"
        conn = sqlite3.connect(db_path)
        expected_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]
        conn.execute(f"PRAGMA user_version = {expected_version}")
        conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO sessions VALUES (1), (2)")
        conn.execute("INSERT INTO messages VALUES (1)")
        conn.commit()

        result = _archive_tiers(_archive_fixture(tmp_path, expected_version), conn)
        index_result = cast(_ArchiveTierResult, result["index"])
        assert index_result["table_counts"]["sessions"] == 2
        assert index_result["table_counts"]["messages"] == 1

    def test_table_counts_include_view_backed_index_relations(self, tmp_path: Path) -> None:
        """Declared compatibility views are counted alongside physical tables."""
        db_path = tmp_path / "index.db"
        conn = sqlite3.connect(db_path)
        expected_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]
        conn.execute(f"PRAGMA user_version = {expected_version}")
        conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO sessions VALUES (1), (2)")
        for view in ("threads", "thread_sessions", "actions"):
            conn.execute(f"CREATE VIEW {view} AS SELECT id FROM sessions")
        conn.commit()
        try:
            result = _archive_tiers(_archive_fixture(tmp_path, expected_version), conn)
            index_result = cast(_ArchiveTierResult, result["index"])
            assert index_result["table_counts"]["threads"] == 2
            assert index_result["table_counts"]["thread_sessions"] == 2
            assert index_result["table_counts"]["actions"] == 2
        finally:
            conn.close()

    def test_table_counts_preserve_unreadable_view(self, tmp_path: Path) -> None:
        """A broken declared view is unavailable, not an observed zero."""
        db_path = tmp_path / "index.db"
        conn = sqlite3.connect(db_path)
        expected_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]
        conn.execute(f"PRAGMA user_version = {expected_version}")
        conn.execute("CREATE VIEW actions AS SELECT * FROM missing_action_source")
        conn.commit()
        try:
            result = _archive_tiers(_archive_fixture(tmp_path, expected_version), conn)
            index_result = cast(_ArchiveTierResult, result["index"])
            assert "actions" not in index_result["table_counts"]
            assert index_result["table_count_precision"]["actions"] == "unavailable"
        finally:
            conn.close()


class TestArchiveOneTierStatusDaemonParity:
    """polylogue-703 regression guard: CLI and daemon must agree by construction.

    Both surfaces now build their per-tier exists/size/user_version/
    version_status facts from the single shared
    ``polylogue.storage.archive_readiness.probe_archive_tier`` (used here and
    ``polylogue.daemon.status._archive_tier_status``). This test would fail
    if either surface went back to computing those facts independently (e.g.
    a version-mismatch tier reported "ok" by one surface and "mismatch" by
    the other) -- the exact bug class behind the 2026-07-03 production
    disagreement (bare CLI status: FTS 100%/844.6 MB vs daemon-backed web
    header: degraded/28.5 GB during an unacknowledged rebuild).
    """

    def test_version_mismatch_agrees_with_daemon(self, tmp_path: Path) -> None:
        from polylogue.daemon.status import _archive_tier_status as _daemon_archive_tier_status

        db_path = tmp_path / "index.db"
        conn = sqlite3.connect(db_path)
        expected_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]
        conn.execute(f"PRAGMA user_version = {expected_version + 999}")
        conn.commit()
        conn.close()

        cli_result = probe_archive_tier(ArchiveTier.INDEX, db_path)
        daemon_result = _daemon_archive_tier_status("index", db_path)

        assert cli_result.exists is daemon_result.exists
        assert cli_result.user_version == daemon_result.user_version
        assert cli_result.expected_user_version == daemon_result.expected_user_version
        assert cli_result.version_status == daemon_result.version_status == "mismatch"
        assert cli_result.size_bytes == daemon_result.size_bytes

    def test_missing_tier_agrees_with_daemon(self, tmp_path: Path) -> None:
        from polylogue.daemon.status import _archive_tier_status as _daemon_archive_tier_status

        nonexistent = tmp_path / "nonexistent.db"
        cli_result = probe_archive_tier(ArchiveTier.INDEX, nonexistent)
        daemon_result = _daemon_archive_tier_status("index", nonexistent)

        assert cli_result.exists is daemon_result.exists is False
        assert cli_result.version_status == daemon_result.version_status == "missing"

    def test_invalid_tier_agrees_with_daemon_and_stays_explicit(self, tmp_path: Path) -> None:
        """A corrupt present tier is invalid, never silently reported healthy."""
        from polylogue.daemon.status import _archive_tier_status as _daemon_archive_tier_status

        db_path = tmp_path / "index.db"
        db_path.write_bytes(b"not a sqlite database")

        cli_result = probe_archive_tier(ArchiveTier.INDEX, db_path)
        daemon_result = _daemon_archive_tier_status("index", db_path)

        assert cli_result.exists is daemon_result.exists is True
        assert cli_result.user_version is daemon_result.user_version is None
        assert cli_result.version_status == daemon_result.version_status == "invalid"
        assert cli_result.size_bytes == daemon_result.size_bytes == db_path.stat().st_size


class TestArchiveTierStatus:
    """Tests for the canonical daemon tier-status producer."""

    @staticmethod
    def _statuses(root: Path) -> dict[str, dict[str, object]]:
        from polylogue.daemon.status import _archive_tier_status

        return {
            tier.value: _archive_tier_status(
                _daemon_tier_name(tier.value),
                root / f"{tier.value}.db",
            ).model_dump()
            for tier in ArchiveTier
        }

    def test_returns_status_for_all_tiers(self, tmp_path: Path) -> None:
        """Returns status for all durable and derived tier files."""
        result = self._statuses(tmp_path)
        expected_tiers = {"source", "index", "embeddings", "user", "audit", "ops"}
        assert set(result.keys()) == expected_tiers

    def test_all_missing_tiers(self, tmp_path: Path) -> None:
        """When no tier files exist, all show as missing."""
        result = self._statuses(tmp_path)
        for status in result.values():
            assert status["exists"] is False
            assert status["version_status"] == "missing"

    def test_with_existing_index_tier(self, tmp_path: Path) -> None:
        """Correctly detects existing index.db."""
        db_path = tmp_path / "index.db"
        conn = sqlite3.connect(db_path)
        expected_version = ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]
        conn.execute(f"PRAGMA user_version = {expected_version}")
        conn.commit()
        conn.close()

        result = self._statuses(tmp_path)
        assert result["index"]["exists"] is True
        assert result["index"]["version_status"] == "ok"
        assert result["source"]["exists"] is False
        assert result["user"]["exists"] is False


class TestDirectArchiveCounts:
    """Canonical direct OperationResult retains archive workload counts."""

    @staticmethod
    def _status_result(root: Path) -> _DirectStatusPayload:
        from polylogue.cli.operation_kernel import configured_read_operation
        from polylogue.config import Config

        config = Config(archive_root=root, render_root=root / "render", sources=[], db_path=root / "index.db")
        result = configured_read_operation(config, "status", {}, daemon_disabled=True)
        assert result.operation == "status"
        return cast(_DirectStatusPayload, result.value)

    def test_empty_archive_returns_zeros(self, tmp_path: Path) -> None:
        """Archive with no sessions returns zeros."""
        bootstrap_archive_root(tmp_path)
        result = self._status_result(tmp_path)
        stats = result["archive_stats"]
        assert stats["total_sessions"] == 0
        assert stats["total_messages"] == 0

    def test_session_message_counts_drive_canonical_total(self, tmp_path: Path) -> None:
        """Canonical stats retain the session message-count workload law."""
        bootstrap_archive_root(tmp_path)
        with sqlite3.connect(tmp_path / "index.db") as conn:
            conn.execute(
                """
                INSERT INTO sessions (native_id, origin, raw_id, content_hash, message_count)
                VALUES (?, 'codex-session', ?, ?, 5), (?, 'codex-session', ?, ?, 3)
                """,
                ("native-1", "raw1", b"1" * 32, "native-2", "raw2", b"2" * 32),
            )
            conn.commit()
        result = self._status_result(tmp_path)
        stats = result["archive_stats"]
        assert stats["total_sessions"] == 2
        assert stats["total_messages"] == 8

    def test_with_sessions_and_messages_table(self, tmp_path: Path) -> None:
        """Canonical direct status counts sessions and messages from the snapshot."""
        bootstrap_archive_root(tmp_path)
        db_path = tmp_path / "index.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO sessions (native_id, origin, raw_id, content_hash, message_count)
                VALUES (?, 'codex-session', ?, ?, 2), (?, 'codex-session', ?, ?, 1)
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
        result = self._status_result(tmp_path)
        stats = result["archive_stats"]
        assert stats["total_sessions"] == 2
        assert stats["total_messages"] == 3

    def test_counts_unidentified_artifacts_from_source_tier(self, tmp_path: Path) -> None:
        """The canonical status snapshot retains exact source artifact workload."""
        index_path = tmp_path / "index.db"
        source_path = tmp_path / "source.db"
        bootstrap_archive_root(tmp_path)
        source_conn = sqlite3.connect(source_path)
        try:
            from polylogue.storage.sqlite.archive_tiers.source_write import list_raw_artifacts

            source_conn.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
                ) VALUES ('raw1', 'claude-code-session', 'native-1', 'one.jsonl', 0, ?, 32, 1)
                """,
                (bytes.fromhex("1" * 64),),
            )
            source_conn.execute(
                """
                INSERT INTO raw_artifacts (
                    artifact_id, raw_id, origin, source_path, source_index, artifact_kind,
                    support_status, classification_reason, parse_as_session, schema_eligible,
                    first_observed_at_ms, last_observed_at_ms
                ) VALUES
                    ('art1', 'raw1', 'claude-code-session', 'one.jsonl', 0, 'unknown',
                     'unknown', 'unrecognized document payload', 0, 0, 1, 1),
                    ('art2', 'raw1', 'claude-code-session', 'two.jsonl', 0, 'agent_sidecar_meta',
                     'recognized_unparsed', 'agent sidecar metadata path', 0, 0, 1, 1)
                """
            )
            source_conn.commit()
            artifacts = list_raw_artifacts(source_conn, raw_id="raw1")
            assert sum(artifact.artifact_kind == "unknown" for artifact in artifacts) == 1
        finally:
            source_conn.close()

        conn = sqlite3.connect(index_path)
        try:
            result = self._status_result(tmp_path)
            source_counts = result["archive_tiers"]["source"]["table_counts"]
            assert source_counts["raw_artifacts"] == 2
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

    def test_tool_usage_ready_for_genuine_zero_tool_archive(self) -> None:
        counts: dict[str, object] = {
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
            "tool_use_block_count": 0,
            "actions_view_present": True,
            "actions_view_error": None,
            "missing_session_profile_materialization": 0,
            "missing_latency_materialization": 0,
        }
        result = _archive_status_surfaces(counts, source_check_available=True)
        assert result["tool_usage"]["ready"] is True

    @pytest.mark.parametrize(
        ("overrides", "blocker"),
        [
            ({"actions_view_present": False}, "actions_view_missing"),
            ({"actions_view_present": True, "actions_view_error": "broken"}, "actions_view_unreadable"),
            (
                {"actions_view_present": True, "actions_view_error": None, "tool_use_block_count": 1},
                "actions_tool_use_count_mismatch",
            ),
        ],
    )
    def test_tool_usage_blocks_on_non_vacuous_action_failures(self, overrides: dict[str, object], blocker: str) -> None:
        counts: dict[str, object] = {
            "action_count": 0,
            "tool_use_block_count": 0,
            "actions_view_present": True,
            "actions_view_error": None,
            "missing_latency_materialization": 0,
        }
        counts.update(overrides)
        result = _archive_status_surfaces(counts, source_check_available=True)
        assert result["tool_usage"]["ready"] is False
        assert result["tool_usage"]["blockers"] == [blocker]

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
        counts: dict[str, object] = {
            "session_count": 1,
            "raw_link_count": 1,
            "missing_raw_session_count": 1,
            "raw_authority_parser_census": {"available": True},
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

    def test_derived_timeline_surfaces_do_not_use_marker_counts(self) -> None:
        """Timeline surfaces remain ordinary derived projections."""
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
            "phase_row_count": 0,
            "thread_count": 0,
            "action_count": 0,
        }
        result = _archive_status_surfaces(counts, source_check_available=True)
        assert result["timeline_work_events"]["ready"] is True
        assert result["timeline_work_events"]["blockers"] == []
        assert result["timeline_phases"]["ready"] is True
        assert result["timeline_phases"]["blockers"] == []

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
