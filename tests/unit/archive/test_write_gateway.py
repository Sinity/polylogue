from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.write_gateway import ArchiveWriteGateway, WriteOperation
from polylogue.storage.sqlite.connection import open_connection


def test_write_gateway_commits_effects_on_caller_owned_connection(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    with open_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        result = ArchiveWriteGateway(db_path).commit_write_sync(
            WriteOperation.INGEST,
            {
                "_connection": conn,
                "changed_conversation_ids": (),
            },
        )

        assert result.operation is WriteOperation.INGEST
        assert result.status == "committed"
        assert conn.execute("SELECT 1").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_write_gateway_async_commit_uses_same_local_effects_path(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    with open_connection(db_path) as conn:
        result = await ArchiveWriteGateway(db_path).commit_write(
            WriteOperation.INGEST,
            {
                "_connection": conn,
                "changed_conversation_ids": (),
            },
        )

        assert result.operation is WriteOperation.INGEST
        assert result.status == "committed"
