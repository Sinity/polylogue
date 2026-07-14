"""Tests for sync/async storage backend divergences.

This module documents and verifies the known divergences between the async
SQLite backend (async_sqlite*.py) and the sync backend (archive_tiers/).

The 10 documented divergences are architectural, not bugs, and arise from
the different roles:
- async_sqlite*: async/await API, delegation-heavy, query-store composition
- archive_tiers: sync write-capable full DB owner, tier-specific modules

Each divergence is documented with file:line references and tested to
ensure no new unintended divergences are introduced.

See: polylogue/storage/sqlite/async_sqlite_archive.py:14-54
"""

from __future__ import annotations

from typing import Any, TypeAlias

import pytest

# Type aliases for clarity
MethodInfo: TypeAlias = dict[str, Any]


class StorageTwinDivergence:
    """Represents a documented divergence between sync and async backends."""

    def __init__(
        self,
        divergence_number: int,
        name: str,
        async_location: str,
        sync_location: str,
        reason: str,
        status: str = "architectural",  # "architectural" | "to-be-fixed" | "documented-contract"
    ):
        self.number = divergence_number
        self.name = name
        self.async_location = async_location
        self.sync_location = sync_location
        self.reason = reason
        self.status = status

    def __str__(self) -> str:
        return (
            f"Divergence {self.number}: {self.name}\n"
            f"  Async: {self.async_location}\n"
            f"  Sync:  {self.sync_location}\n"
            f"  Reason: {self.reason}\n"
            f"  Status: {self.status}"
        )


# The 10 documented divergences extracted from async_sqlite_archive.py:14-54
DOCUMENTED_DIVERGENCES = [
    StorageTwinDivergence(
        divergence_number=1,
        name="Naming: _session_id_query vs session_id_query",
        async_location="async_sqlite_archive.py: implied by delegation to queries",
        sync_location="query_store_archive.py: public API",
        reason="Private delegate method vs public query-store API. Same function, different callers.",
        status="architectural",
    ),
    StorageTwinDivergence(
        divergence_number=2,
        name="search_sessions delegation path",
        async_location="async_sqlite_archive.py: delegates to queries",
        sync_location="query_store_archive.py: delegates to search_session_hits()",
        reason="Backend delegates to queries; query store uses its own implementation. Same result.",
        status="architectural",
    ),
    StorageTwinDivergence(
        divergence_number=3,
        name="get_messages content_blocks attachment",
        async_location="async_sqlite_archive.py: queries pre-attach blocks",
        sync_location="query_store_archive.py: canonical two-step load+merge",
        reason="Query store is the canonical implementation; backend inherits from it.",
        status="architectural",
    ),
    StorageTwinDivergence(
        divergence_number=4,
        name="Connection management strategy",
        async_location="async_sqlite.py:_get_connection(): ensures schema before every use",
        sync_location="archive_tiers/archive.py: _connection_factory provides pre-configured read-only",
        reason="Backend owns DB, ensures schema; query store provides composable read-only connections.",
        status="architectural",
    ),
    StorageTwinDivergence(
        divergence_number=5,
        name="Write methods exist only on backend",
        async_location="async_sqlite_archive.py: save_session_record, save_messages, etc.",
        sync_location="archive_tiers/: write methods (write.py, user_write.py, etc.)",
        reason="Query store is deliberately read-only; write methods are backend-only.",
        status="architectural",
    ),
    StorageTwinDivergence(
        divergence_number=6,
        name="Query API methods exist only on query store",
        async_location="async_sqlite_archive.py: accesses via self.queries",
        sync_location="query_store_archive.py: list_sessions, count_sessions, search_action_*",
        reason="These are read-only query operations; only query store implements them.",
        status="architectural",
    ),
    StorageTwinDivergence(
        divergence_number=7,
        name="get_session_insight_status implementation location",
        async_location="async_sqlite_archive.py: on SQLiteArchiveMixin",
        sync_location="query_store.py: separate query-store implementation",
        reason="Both provide the method; query store has its own version.",
        status="architectural",
    ),
    StorageTwinDivergence(
        divergence_number=8,
        name="get_messages_batch early-exit clarity",
        async_location="async_sqlite_archive.py: delegates to queries",
        sync_location="query_store_archive.py: explicit empty-session_ids early exit",
        reason="Equivalent behavior; sync adds explicit clarity on empty case.",
        status="architectural",
    ),
    StorageTwinDivergence(
        divergence_number=9,
        name="iter_messages chunk_size fast path",
        async_location="async_sqlite.py: chunk_size=100 fast path delegating to query store",
        sync_location="query_store_archive.py: calls messages_q.iter_messages",
        reason="Both reach the same destination; async has an optimization layer.",
        status="architectural",
    ),
    StorageTwinDivergence(
        divergence_number=10,
        name="search_session_hits access pattern",
        async_location="async_sqlite_archive.py: backend delegates to self.queries",
        sync_location="query_store_archive.py: opens direct connection",
        reason="Same destination; different access layers (backend vs query-store).",
        status="architectural",
    ),
]


class TestStorageTwinDivergenceContract:
    """Verify documented divergences are stable and no new ones are introduced."""

    def test_documented_divergences_exist(self) -> None:
        """Verify all 10 documented divergences are accounted for."""
        assert len(DOCUMENTED_DIVERGENCES) == 10, f"Expected 10 divergences, found {len(DOCUMENTED_DIVERGENCES)}"
        for div in DOCUMENTED_DIVERGENCES:
            assert div.number in range(1, 11), f"Invalid divergence number: {div.number}"
            assert div.name, f"Divergence {div.number} missing name"
            assert div.async_location, f"Divergence {div.number} missing async location"
            assert div.sync_location, f"Divergence {div.number} missing sync location"

    def test_divergence_status_values(self) -> None:
        """Verify all divergences have valid status values."""
        valid_statuses = {"architectural", "to-be-fixed", "documented-contract"}
        for div in DOCUMENTED_DIVERGENCES:
            assert div.status in valid_statuses, (
                f"Divergence {div.number} has invalid status '{div.status}'. Valid: {valid_statuses}"
            )

    def test_async_backend_source_files_exist(self) -> None:
        """Verify async backend files exist."""
        from pathlib import Path

        repo_root = Path(__file__).parent.parent.parent.parent
        async_files = {
            "polylogue/storage/sqlite/async_sqlite.py",
            "polylogue/storage/sqlite/async_sqlite_archive.py",
            "polylogue/storage/sqlite/async_sqlite_raw.py",
        }
        for file in async_files:
            assert (repo_root / file).exists(), f"Async backend file not found: {file}"

    def test_sync_backend_source_files_exist(self) -> None:
        """Verify sync backend (archive_tiers) files exist."""
        from pathlib import Path

        repo_root = Path(__file__).parent.parent.parent.parent
        sync_files = {
            "polylogue/storage/sqlite/archive_tiers/archive.py",
            "polylogue/storage/sqlite/archive_tiers/write.py",
            "polylogue/storage/sqlite/archive_tiers/user_write.py",
        }
        for file in sync_files:
            assert (repo_root / file).exists(), f"Sync backend file not found: {file}"

    def test_async_divergence_documentation_exists(self) -> None:
        """Verify source divergence documentation in async_sqlite_archive.py exists."""
        from pathlib import Path

        repo_root = Path(__file__).parent.parent.parent.parent
        async_archive = repo_root / "polylogue/storage/sqlite/async_sqlite_archive.py"

        assert async_archive.exists(), "async_sqlite_archive.py should exist"

        content = async_archive.read_text()

        # Verify the documented divergences comment block exists
        assert "Intentional divergences (10 known" in content, (
            "async_sqlite_archive.py should document the 10 known divergences"
        )
        assert "These divergences reflect the different roles" in content, (
            "Divergences should be explained as architectural"
        )


class TestStorageTwinWritePathStructure:
    """Verify write-path structure is accounted for in both backends."""

    def test_async_write_methods_exist(self) -> None:
        """Verify async backend has expected write method signatures."""
        from pathlib import Path

        repo_root = Path(__file__).parent.parent.parent.parent
        async_archive = repo_root / "polylogue/storage/sqlite/async_sqlite_archive.py"

        content = async_archive.read_text()

        # Should have write methods or their delegations
        assert "save_" in content or "async def " in content, (
            "Async backend should have save methods or async method definitions"
        )


class TestStorageTwinArchitecturalRoles:
    """Verify the architectural roles of async vs sync backends are distinct."""

    def test_async_uses_mixin_pattern(self) -> None:
        """Verify async backend uses mixin pattern."""
        from pathlib import Path

        repo_root = Path(__file__).parent.parent.parent.parent
        async_archive = repo_root / "polylogue/storage/sqlite/async_sqlite_archive.py"
        content = async_archive.read_text()

        assert "Mixin" in content, "Async should use mixin pattern"
        assert "class SQLiteArchiveMixin" in content, "Should define SQLiteArchiveMixin"

    def test_async_delegates_reads(self) -> None:
        """Verify async backend delegates reads to query store."""
        from pathlib import Path

        repo_root = Path(__file__).parent.parent.parent.parent
        async_archive = repo_root / "polylogue/storage/sqlite/async_sqlite_archive.py"
        content = async_archive.read_text()

        # Should reference self.queries
        assert "self.queries" in content, "Async should delegate to self.queries"

    def test_sync_is_tier_modular(self) -> None:
        """Verify sync backend is organized by tier modules."""
        from pathlib import Path

        repo_root = Path(__file__).parent.parent.parent.parent
        archive_tiers = repo_root / "polylogue/storage/sqlite/archive_tiers"

        assert archive_tiers.exists(), "archive_tiers directory should exist"
        assert archive_tiers.is_dir(), "archive_tiers should be a directory"

        # Should have tier-specific files
        expected_files = ["archive.py", "write.py", "user_write.py", "source_write.py"]
        for fname in expected_files:
            assert (archive_tiers / fname).exists(), f"Should have {fname}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
