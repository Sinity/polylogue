"""FTS search must guard empty and special queries.

The legacy lexical path escaped/normalized the FTS5 MATCH expression before
running it. The native ``ArchiveStore`` search surface must do the same, otherwise
an empty query raises ``fts5: syntax error near ""`` and a hyphenated token like
``drive-file-1`` raises ``no such column: file`` because FTS5 parses ``-`` as a
column operator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.storage_records import SessionBuilder, db_setup


def _seed(workspace_env: dict[str, Path]) -> Path:
    db_path = db_setup(workspace_env)
    (
        SessionBuilder(db_path, "conv-search-guard")
        .provider("claude-code")
        .add_message(message_id="msg-1", text="reference to drive-file-1 attachment")
        .save()
    )
    return workspace_env["archive_root"]


def test_empty_query_returns_no_hits(workspace_env: dict[str, Path]) -> None:
    archive_root = _seed(workspace_env)
    with ArchiveStore.open_existing(archive_root) as archive:
        assert archive.search_summaries("") == []
        assert archive.search_summaries("   ") == []
        assert archive.count_search_sessions("") == 0


def test_hyphenated_token_does_not_raise(workspace_env: dict[str, Path]) -> None:
    archive_root = _seed(workspace_env)
    with ArchiveStore.open_existing(archive_root) as archive:
        # Must not raise "no such column: file".
        hits = archive.search_summaries("drive-file-1")
        # And the count path stays consistent.
        assert archive.count_search_sessions("drive-file-1") >= 0
        assert isinstance(hits, list)


@pytest.mark.parametrize("query", ["*", "AND", "NOT", '"unterminated', "col:val"])
def test_special_tokens_do_not_raise(workspace_env: dict[str, Path], query: str) -> None:
    archive_root = _seed(workspace_env)
    with ArchiveStore.open_existing(archive_root) as archive:
        assert isinstance(archive.search_summaries(query), list)
        assert isinstance(archive.count_search_sessions(query), int)
