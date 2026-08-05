"""Source-row selection helpers in ``maintenance/rebuild_index.py`` (polylogue-ogn1).

Covers the CodeRabbit findings on PR #3076's rebuild-index coordination that
were still genuinely present against current source:

- ``missing_index_raw_ids`` must treat every source row as missing when
  ``index.db`` does not exist yet (fresh archive, or one just reset via
  ``ops reset --index``), not silently return an empty selection -- a fresh
  index has nothing indexed, so ``--only-missing`` must select the full
  source set, matching ``all_index_rebuild_raw_ids``.
- ``validate_rebuild_index_request`` (the shared service's own validation,
  not merely the CLI's) rejects ``max_blob_mb`` without ``raw_ids``/
  ``only_missing``, and rejects a partial selection asking to promote.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.maintenance.archive_verification import REINDEX_CANARY_ACCEPTANCE_CHECKS
from polylogue.maintenance.rebuild_index import (
    RebuildIndexRequest,
    all_index_rebuild_raw_ids,
    missing_index_raw_ids,
    validate_rebuild_index_request,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _seed_source_raw_session(source_db: Path, *, raw_id: str, native_id: str, acquired_at_ms: int) -> None:
    with sqlite3.connect(source_db) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms
            )
            VALUES (?, 'codex-session', ?, ?, zeroblob(32), 0, ?)
            """,
            (raw_id, native_id, f"/tmp/{native_id}.jsonl", acquired_at_ms),
        )


class TestMissingIndexRawIds:
    def test_returns_every_raw_id_when_index_tier_does_not_exist_yet(self, tmp_path: Path) -> None:
        """polylogue-ogn1 finding #6: a fresh/lost index has nothing indexed.

        Before this fix, ``missing_index_raw_ids`` short-circuited to ``[]``
        whenever ``index.db`` was absent -- which made ``--only-missing``
        rebuild nothing on a fresh archive or right after
        ``ops reset --index``, exactly the case it exists to handle.
        """
        source_db = tmp_path / "source.db"
        initialize_archive_database(source_db, ArchiveTier.SOURCE)
        _seed_source_raw_session(source_db, raw_id="raw-1", native_id="n1", acquired_at_ms=1000)
        _seed_source_raw_session(source_db, raw_id="raw-2", native_id="n2", acquired_at_ms=2000)

        assert not (tmp_path / "index.db").exists()
        assert missing_index_raw_ids(tmp_path) == all_index_rebuild_raw_ids(tmp_path) == ["raw-1", "raw-2"]

    def test_returns_empty_when_source_tier_does_not_exist(self, tmp_path: Path) -> None:
        assert missing_index_raw_ids(tmp_path) == []

    def test_excludes_raw_ids_already_materialized_in_an_existing_index(self, tmp_path: Path) -> None:
        source_db = tmp_path / "source.db"
        index_db = tmp_path / "index.db"
        initialize_archive_database(source_db, ArchiveTier.SOURCE)
        initialize_archive_database(index_db, ArchiveTier.INDEX)
        _seed_source_raw_session(source_db, raw_id="raw-1", native_id="n1", acquired_at_ms=1000)
        _seed_source_raw_session(source_db, raw_id="raw-2", native_id="n2", acquired_at_ms=2000)
        with sqlite3.connect(index_db) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                """
                INSERT INTO sessions (
                    native_id, origin, raw_id, title, content_hash, created_at_ms, updated_at_ms
                )
                VALUES ('n1', 'codex-session', 'raw-1', 'Session n1', zeroblob(32), 1000, 1000)
                """
            )

        # index.db exists and already has raw-1 materialized -- only raw-2 is missing.
        assert missing_index_raw_ids(tmp_path) == ["raw-2"]


class TestValidateRebuildIndexRequestSharedService:
    """polylogue-ogn1 findings #1/#7: enforced by the shared service, not just the CLI."""

    def test_rejects_max_blob_mb_without_raw_ids_or_only_missing(self, tmp_path: Path) -> None:
        request = RebuildIndexRequest(archive_root=tmp_path, max_blob_mb=10.0)
        with pytest.raises(ValueError, match="--max-blob-mb requires --only-missing or --raw-id"):
            validate_rebuild_index_request(request)

    def test_accepts_max_blob_mb_with_only_missing(self, tmp_path: Path) -> None:
        request = RebuildIndexRequest(archive_root=tmp_path, only_missing=True, max_blob_mb=10.0, promote=False)
        validate_rebuild_index_request(request)  # must not raise

    def test_rejects_partial_selection_that_still_asks_to_promote(self, tmp_path: Path) -> None:
        request = RebuildIndexRequest(archive_root=tmp_path, raw_ids=("raw-1",), promote=True)
        with pytest.raises(ValueError, match="require --no-promote"):
            validate_rebuild_index_request(request)

    def test_rejects_caller_supplied_acceptance_profile_for_promotion(self, tmp_path: Path) -> None:
        request = RebuildIndexRequest(
            archive_root=tmp_path,
            promote=True,
            candidate_acceptance_checks=REINDEX_CANARY_ACCEPTANCE_CHECKS,
        )
        with pytest.raises(ValueError, match="require --no-promote"):
            validate_rebuild_index_request(request)

    def test_allows_caller_supplied_canary_profile_without_promotion(self, tmp_path: Path) -> None:
        request = RebuildIndexRequest(
            archive_root=tmp_path,
            promote=False,
            candidate_acceptance_checks=REINDEX_CANARY_ACCEPTANCE_CHECKS,
        )
        validate_rebuild_index_request(request)
