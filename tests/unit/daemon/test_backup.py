"""Backup verification tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.daemon.backup import backup_archive
from polylogue.storage.sqlite.connection import open_read_connection
from tests.infra.storage_records import ConversationBuilder, db_setup


@pytest.mark.contract
def test_backup_archive_copy_can_be_opened_and_queried(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    db_path = db_setup(workspace_env)
    ConversationBuilder(db_path, "backup-conv").provider("claude-code").add_message(
        role="user",
        text="backup restore smoke",
    ).save()

    result = backup_archive(output_dir=tmp_path / "backups")

    assert result.ok
    assert result.output_path is not None
    backup_path = Path(result.output_path)
    assert backup_path.exists()
    with open_read_connection(backup_path) as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        assert integrity == "ok"
        conversation_count = conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE conversation_id = ?",
            ("backup-conv",),
        ).fetchone()[0]
        message_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            ("backup-conv",),
        ).fetchone()[0]

    assert conversation_count == 1
    assert message_count == 1
