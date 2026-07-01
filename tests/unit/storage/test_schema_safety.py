"""Tests for schema bootstrap safety, schema DDL parity, and SQL edge cases.

Covers:
- 447d765: executescript() issues implicit COMMIT, breaking transactional rollback
- f33ef29: Async SQLite schema diverged from sync (DDL was duplicated)
- 7ebfd71: OFFSET without LIMIT → SQLite error
- 177195c: LIKE wildcards (%, _) not escaped in title search
- f6896a9: fetch_limit=1000 silently truncated results

These tests verify structural invariants about the storage layer that
are invisible in unit tests with small datasets.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite.schema import SCHEMA_DDL, SCHEMA_VERSION

# =============================================================================
# Schema DDL parity: sync and async must use the same DDL (f33ef29)
# =============================================================================


class TestSchemaDDLParity:
    """Sync and async backends must share SCHEMA_DDL, not duplicate it.

    The original bug (f33ef29) was that the async backend had its own
    copy of the DDL that diverged from the sync version. The fix was to
    share SCHEMA_DDL as a single source of truth.
    """

    def test_async_backend_imports_shared_schema_ddl(self) -> None:
        """Async backend must import SCHEMA_DDL from schema.py, not define its own."""
        import inspect

        from polylogue.storage.sqlite import async_sqlite

        source = inspect.getsource(async_sqlite)
        # Must import SCHEMA_DDL, not define it
        assert "from polylogue.storage.sqlite.schema import" in source
        assert "SCHEMA_DDL" in source

    def test_schema_ddl_has_all_required_tables(self) -> None:
        """SCHEMA_DDL must create all required tables."""
        required_tables = [
            "sessions",
            "messages",
            "blocks",
            "attachments",
        ]
        ddl_lower = SCHEMA_DDL.lower()
        for table in required_tables:
            assert f"create table if not exists {table}" in ddl_lower, f"SCHEMA_DDL missing table: {table}"

    def test_schema_ddl_has_all_required_indexes(self) -> None:
        """SCHEMA_DDL must create required indexes."""
        required_indexes = [
            "idx_sessions_origin_sort",
            "idx_messages_session_position",
        ]
        ddl_lower = SCHEMA_DDL.lower()
        for idx in required_indexes:
            assert idx.lower() in ddl_lower, f"SCHEMA_DDL missing index: {idx}"

    def test_schema_version_is_positive_int(self) -> None:
        assert isinstance(SCHEMA_VERSION, int)
        assert SCHEMA_VERSION > 0

    def test_schema_ddl_applied_to_fresh_database(self, tmp_path: Path) -> None:
        """SCHEMA_DDL must apply cleanly to a fresh database."""
        db_path = tmp_path / "fresh.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(SCHEMA_DDL)
            # Verify tables exist
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = {row[0] for row in cursor.fetchall()}
            assert "sessions" in tables
            assert "messages" in tables
        finally:
            conn.close()

    def test_schema_ddl_is_idempotent(self, tmp_path: Path) -> None:
        """Applying SCHEMA_DDL twice must not error."""
        db_path = tmp_path / "idempotent.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(SCHEMA_DDL)
            conn.executescript(SCHEMA_DDL)  # second application
            # Verify still works
            cursor = conn.execute("SELECT COUNT(*) FROM sessions")
            assert cursor.fetchone()[0] == 0
        finally:
            conn.close()


# =============================================================================
# Transaction rollback safety (447d765)
# =============================================================================


class TestTransactionRollbackSafety:
    """Schema edits that run inside a transaction must leave DB unchanged on failure.

    The original bug (447d765) was that `executescript()` issues an implicit
    COMMIT before executing, which breaks transaction isolation. If a
    script fails mid-way after executescript, the partial changes are
    committed and can't be rolled back.

    The fix was to use `execute()` calls within a BEGIN/COMMIT block
    instead of `executescript()`.
    """

    def test_executescript_implicit_commit_behavior(self, tmp_path: Path) -> None:
        """Document that executescript() commits before executing.

        This test documents the SQLite behavior that caused the bug.
        executescript() issues an implicit COMMIT of any pending transaction.
        """
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        try:
            # Create a table and insert data in a transaction
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'original')")
            conn.commit()

            # Start a new transaction
            conn.execute("BEGIN")
            conn.execute("INSERT INTO test VALUES (2, 'in_transaction')")

            # executescript() will implicitly COMMIT the pending INSERT
            try:
                conn.executescript("CREATE TABLE dummy (x INTEGER); INVALID SQL;")
            except sqlite3.OperationalError:
                pass

            # Check: row 2 was committed despite the script failing!
            # This is the documented behavior that caused the bug.
            cursor = conn.execute("SELECT COUNT(*) FROM test")
            count = cursor.fetchone()[0]
            # The partial transaction was implicitly committed
            assert count == 2, "executescript() should have committed the pending transaction"
        finally:
            conn.close()

    def test_execute_preserves_transaction(self, tmp_path: Path) -> None:
        """execute() within BEGIN/ROLLBACK correctly rolls back."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'original')")
            conn.commit()

            # Start a transaction and make changes
            conn.execute("BEGIN")
            conn.execute("INSERT INTO test VALUES (2, 'will_rollback')")

            # Rollback should undo the insert
            conn.execute("ROLLBACK")

            cursor = conn.execute("SELECT COUNT(*) FROM test")
            count = cursor.fetchone()[0]
            assert count == 1, "ROLLBACK should have undone the INSERT"
        finally:
            conn.close()

    def test_open_connection_schema_matches_index_tier_ddl(self, tmp_path: Path) -> None:
        """The normal index bootstrap path and index-tier DDL create the same schema.

        This catches drift between the current archive-root bootstrap path and
        the index-tier DDL.
        """
        from polylogue.storage.sqlite.connection import open_connection

        bootstrapped_path = tmp_path / "archive" / "index.db"
        with open_connection(bootstrapped_path) as conn:
            cursor = conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            bootstrapped_tables = {row[0]: row[1] for row in cursor.fetchall()}

        fresh_path = tmp_path / "fresh.db"
        fresh_conn = sqlite3.connect(str(fresh_path))
        try:
            fresh_conn.executescript(SCHEMA_DDL)
            cursor = fresh_conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            fresh_tables = {row[0]: row[1] for row in cursor.fetchall()}
        finally:
            fresh_conn.close()

        skip_prefixes = ("messages_fts",)
        bootstrapped_names = {t for t in bootstrapped_tables if not any(t.startswith(p) for p in skip_prefixes)}
        fresh_names = {t for t in fresh_tables if not any(t.startswith(p) for p in skip_prefixes)}

        assert bootstrapped_names == fresh_names, (
            f"Table mismatch. Only in bootstrapped: {bootstrapped_names - fresh_names}. "
            f"Only in raw DDL: {fresh_names - bootstrapped_names}"
        )


# =============================================================================
# SQL edge cases: OFFSET without LIMIT (7ebfd71)
# =============================================================================


class TestSQLEdgeCases:
    """SQL edge cases that caused production bugs."""

    def test_offset_without_limit_is_error(self, tmp_path: Path) -> None:
        """SQLite requires LIMIT when using OFFSET. Verify our code handles this."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
            for i in range(10):
                conn.execute("INSERT INTO test VALUES (?)", (i,))
            conn.commit()

            # OFFSET without LIMIT is a syntax error in SQLite
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("SELECT id FROM test OFFSET 5")

            # LIMIT -1 with OFFSET is the correct pattern
            cursor = conn.execute("SELECT id FROM test LIMIT -1 OFFSET 5")
            rows = cursor.fetchall()
            assert len(rows) == 5
        finally:
            conn.close()

    def test_like_wildcards_in_search(self, tmp_path: Path) -> None:
        """LIKE patterns with % and _ must be escaped properly.

        Regression: commit 177195c — unescaped LIKE wildcards in title search
        meant searching for "100% done" would match everything.
        """
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("CREATE TABLE sessions (id TEXT, title TEXT)")
            conn.execute("INSERT INTO sessions VALUES ('1', '100% done')")
            conn.execute("INSERT INTO sessions VALUES ('2', 'normal title')")
            conn.execute("INSERT INTO sessions VALUES ('3', '100 percent done')")
            conn.commit()

            # Unescaped: "100%" would match "100 percent done" too (% is wildcard)
            cursor = conn.execute(
                "SELECT id FROM sessions WHERE title LIKE ?",
                ("%100%%",),  # unescaped % acts as wildcard
            )
            unescaped_results = cursor.fetchall()
            # This matches both '100% done' AND '100 percent done'
            assert len(unescaped_results) >= 2

            # Escaped: only match literal "100%"
            cursor = conn.execute(
                "SELECT id FROM sessions WHERE title LIKE ? ESCAPE '\\'",
                ("%100\\%%",),  # escaped % matches literal %
            )
            escaped_results = cursor.fetchall()
            assert len(escaped_results) == 1
            assert escaped_results[0]["id"] == "1"
        finally:
            conn.close()

    def test_like_underscore_wildcard(self, tmp_path: Path) -> None:
        """Underscore in LIKE is a single-char wildcard, must be escaped."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("CREATE TABLE sessions (id TEXT, title TEXT)")
            conn.execute("INSERT INTO sessions VALUES ('1', 'test_case')")
            conn.execute("INSERT INTO sessions VALUES ('2', 'testXcase')")
            conn.commit()

            # Unescaped: _ matches any single character
            cursor = conn.execute(
                "SELECT id FROM sessions WHERE title LIKE ?",
                ("%test_case%",),
            )
            assert len(cursor.fetchall()) == 2  # matches both

            # Escaped: only match literal underscore
            cursor = conn.execute(
                "SELECT id FROM sessions WHERE title LIKE ? ESCAPE '\\'",
                ("%test\\_case%",),
            )
            assert len(cursor.fetchall()) == 1  # only the real underscore
        finally:
            conn.close()


# =============================================================================
# SQL: FTS5 COUNT(*) performance guard (4a77379)
# =============================================================================


class TestFTS5CountGuard:
    """COUNT(*) on FTS5 virtual tables is O(N), not O(1).

    Regression: commit 4a77379 — COUNT(*) on the FTS5 table took 227 seconds
    because FTS5 virtual tables don't maintain row counts. The fix was to use
    COUNT(*) on the regular messages table instead.
    """

    def test_count_on_regular_table_not_fts(self, test_db: Path) -> None:
        """Message count must come from the messages table, not messages_fts."""
        from polylogue.storage.sqlite.connection import open_connection

        with open_connection(test_db) as conn:
            conn.execute("""
                INSERT INTO sessions (native_id, origin, content_hash)
                VALUES ('pc1', 'unknown-export', zeroblob(32))
            """)
            session_id = "unknown-export:pc1"
            conn.execute(
                """
                INSERT INTO messages (session_id, native_id, position, role, content_hash)
                VALUES (?, 'm1', 0, 'user', zeroblob(32))
                """,
                (session_id,),
            )
            message_id = f"{session_id}:m1"
            conn.execute(
                """
                INSERT INTO blocks (message_id, session_id, position, block_type, text)
                VALUES (?, ?, 0, 'text', 'hello world')
                """,
                (message_id, session_id),
            )
            conn.commit()

            # COUNT on regular table (fast, O(1) with SQLite's page count optimization)
            cursor = conn.execute("SELECT COUNT(*) FROM messages")
            count = cursor.fetchone()[0]
            assert count == 1

            # Verify FTS table exists and is queryable
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'hello'")
                fts_count = cursor.fetchone()[0]
                assert fts_count == 1
            except sqlite3.OperationalError:
                pytest.skip("FTS5 not available in this SQLite build")


# =============================================================================
# SQL: Session filters with edge case inputs
# =============================================================================


class TestSessionFilterSQL:
    """Test SQL filter generation with adversarial inputs."""

    def test_build_filters_with_special_characters(self) -> None:
        """Filter builder must handle SQL-special characters in input.

        _build_session_filters returns (where_clause_str, params_list).
        It uses parameterized queries, so special chars in input are safe.
        """
        from polylogue.storage.sqlite.queries.filter_builder import _build_session_filters

        where_clause, params = _build_session_filters(title_contains="test'provider")
        assert isinstance(where_clause, str)
        assert isinstance(params, list)
        # The quote should be in params (handled by parameterization), not in SQL
        assert "%test'provider%" in params
        # SQL clause must use ? placeholder, not string interpolation
        assert "?" in where_clause

    def test_build_filters_with_empty_provider(self) -> None:
        """Empty provider should still produce valid SQL."""
        from polylogue.storage.sqlite.queries.filter_builder import _build_session_filters

        where_clause, params = _build_session_filters(provider="")
        assert isinstance(where_clause, str)
        assert isinstance(params, list)

    def test_build_filters_with_no_args(self) -> None:
        """No filters should produce empty/trivial WHERE clause."""
        from polylogue.storage.sqlite.queries.filter_builder import _build_session_filters

        where_clause, params = _build_session_filters()
        assert isinstance(where_clause, str)
        assert isinstance(params, list)
        assert len(params) == 0

    def test_build_filters_with_title_contains_special(self) -> None:
        """Title search with SQL LIKE special characters must be safe."""
        from polylogue.storage.sqlite.queries.filter_builder import _build_session_filters

        where_clause, params = _build_session_filters(title_contains="100% done")
        assert isinstance(where_clause, str)
        # The special chars should be in params, not embedded in SQL
        assert len(params) > 0

    def test_build_filters_rejects_invalid_since_filter(self) -> None:
        """Invalid date filters should not silently broaden to epoch zero."""
        from polylogue.storage.sqlite.queries.filter_builder import _build_session_filters

        with pytest.raises(ValueError, match="Invalid date filter value"):
            _build_session_filters(since="not-a-date")

    def test_build_filters_rejects_invalid_until_filter(self) -> None:
        """Invalid date filters should not silently broaden to epoch zero."""
        from polylogue.storage.sqlite.queries.filter_builder import _build_session_filters

        with pytest.raises(ValueError, match="Invalid date filter value"):
            _build_session_filters(until="not-a-date")


# =============================================================================
# SQL: fetch_limit pagination correctness (f6896a9)
# =============================================================================


class TestFetchLimitPagination:
    """fetch_limit must not silently truncate results.

    Regression: commit f6896a9 — fetch_limit=1000 was applied per-chunk but
    the global limit was ignored, causing results to be silently capped at
    the chunk size when post-filters (tags, excludes) were active.

    This tests the low-level pagination SQL pattern used in async_sqlite.py.
    """

    def test_limit_offset_returns_exact_count(self, tmp_path: Path) -> None:
        """LIMIT N OFFSET M returns exactly N rows when enough data exists."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE msgs (id INTEGER PRIMARY KEY, text TEXT)")
            for i in range(100):
                conn.execute("INSERT INTO msgs VALUES (?, ?)", (i, f"msg-{i}"))
            conn.commit()

            # Simulate chunked pagination: chunk_size=20, global limit=50
            collected: list[tuple[int]] = []
            offset = 0
            chunk_size = 20
            global_limit = 50

            while len(collected) < global_limit:
                remaining = global_limit - len(collected)
                fetch_limit = min(chunk_size, remaining)
                cursor = conn.execute(
                    "SELECT id FROM msgs ORDER BY id LIMIT ? OFFSET ?",
                    (fetch_limit, offset),
                )
                rows = cursor.fetchall()
                if not rows:
                    break
                collected.extend(rows)
                offset += len(rows)
                if len(rows) < fetch_limit:
                    break

            assert len(collected) == 50
            # Verify correct IDs (0..49)
            assert [r[0] for r in collected] == list(range(50))
        finally:
            conn.close()

    def test_limit_with_fewer_rows_than_limit(self, tmp_path: Path) -> None:
        """When total rows < limit, return all rows without error."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE msgs (id INTEGER PRIMARY KEY)")
            for i in range(5):
                conn.execute("INSERT INTO msgs VALUES (?)", (i,))
            conn.commit()

            cursor = conn.execute(
                "SELECT id FROM msgs ORDER BY id LIMIT ? OFFSET ?",
                (1000, 0),
            )
            rows = cursor.fetchall()
            assert len(rows) == 5
        finally:
            conn.close()

    def test_offset_beyond_data_returns_empty(self, tmp_path: Path) -> None:
        """OFFSET past all data returns empty, not error."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE msgs (id INTEGER PRIMARY KEY)")
            for i in range(10):
                conn.execute("INSERT INTO msgs VALUES (?)", (i,))
            conn.commit()

            cursor = conn.execute(
                "SELECT id FROM msgs ORDER BY id LIMIT ? OFFSET ?",
                (10, 999),
            )
            rows = cursor.fetchall()
            assert rows == []
        finally:
            conn.close()


# =============================================================================
# SQL: Analytics queries use indexes (avoids full table scans)
# =============================================================================


class TestAnalyticsQueryPlan:
    """Analytics queries must use indexes, not full table scans.

    The provider breakdown query (GROUP BY source_name) should use
    idx_sessions_provider. Without this index, large archives
    would require a full table scan.
    """

    def test_provider_group_by_uses_index(self, tmp_path: Path) -> None:
        """GROUP BY source_name should use idx_sessions_provider."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(SCHEMA_DDL)
            # EXPLAIN QUERY PLAN shows the access method
            cursor = conn.execute("EXPLAIN QUERY PLAN SELECT origin, COUNT(*) FROM sessions GROUP BY origin")
            plan = " ".join(row[3] if len(row) > 3 else str(row) for row in cursor.fetchall())
            # Should scan the covering index, not the table
            assert "idx_sessions_origin_sort" in plan or "COVERING" in plan or "INDEX" in plan.upper(), (
                f"Expected index usage for GROUP BY origin, got: {plan}"
            )
        finally:
            conn.close()

    def test_messages_by_session_uses_index(self, tmp_path: Path) -> None:
        """WHERE session_id = ? should use idx_messages_session."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(SCHEMA_DDL)
            cursor = conn.execute(
                "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM messages WHERE session_id = ?",
                ("test-id",),
            )
            plan = " ".join(row[3] if len(row) > 3 else str(row) for row in cursor.fetchall())
            assert "idx_messages_session_position" in plan or "INDEX" in plan.upper(), (
                f"Expected index usage for WHERE session_id, got: {plan}"
            )
        finally:
            conn.close()

    def test_observed_event_tool_outcome_grouping_uses_expression_index(self, tmp_path: Path) -> None:
        """Tool outcome GROUP BY should use the matching observed-event expression index."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(SCHEMA_DDL)
            cursor = conn.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT COALESCE(NULLIF(json_extract(e.payload_json, '$.handler_kind'), ''), 'unknown') AS group_key,
                       COUNT(*) AS count
                FROM session_observed_events e
                JOIN sessions s ON s.session_id = e.session_id
                WHERE e.kind = 'tool_finished'
                GROUP BY group_key
                ORDER BY count DESC, group_key DESC
                LIMIT 20 OFFSET 0
                """
            )
            plan = " | ".join(row[3] if len(row) > 3 else str(row) for row in cursor.fetchall())
            assert "idx_session_observed_events_kind_handler" in plan, (
                f"Expected observed-event handler expression index, got: {plan}"
            )
            assert "TEMP B-TREE FOR GROUP BY" not in plan.upper(), f"Unexpected temp GROUP BY plan: {plan}"
        finally:
            conn.close()

    def test_session_count_is_cheap(self, tmp_path: Path) -> None:
        """COUNT(*) on sessions table should not require FTS scan."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(SCHEMA_DDL)
            cursor = conn.execute("EXPLAIN QUERY PLAN SELECT COUNT(*) FROM sessions")
            plan = " ".join(row[3] if len(row) > 3 else str(row) for row in cursor.fetchall())
            # Must scan the sessions table, NOT messages_fts
            assert "messages_fts" not in plan.lower(), f"COUNT(*) on sessions should not touch FTS: {plan}"
        finally:
            conn.close()
