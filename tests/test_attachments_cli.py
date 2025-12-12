from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from polylogue.cli.attachments import run_attachments_cli
from polylogue.commands import CommandEnv
from polylogue.db import open_connection
from polylogue.ui import UI


def _seed_attachment(
    db_path: Path,
    *,
    provider: str,
    conversation_id: str,
    branch_id: str,
    message_id: str,
    timestamp: str,
    attachment_name: str,
    attachment_path: Path,
    size_bytes: int = 100,
    content_hash: str = "hash-1",
) -> None:
    with open_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO conversations(provider, conversation_id, slug, title)
            VALUES (?, ?, ?, ?)
            """,
            (provider, conversation_id, conversation_id, conversation_id),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO branches(provider, conversation_id, branch_id, is_current)
            VALUES (?, ?, ?, 1)
            """,
            (provider, conversation_id, branch_id),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO messages(provider, conversation_id, branch_id, message_id, position, timestamp, role)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (provider, conversation_id, branch_id, message_id, 0, timestamp, "user"),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO attachments(
                provider, conversation_id, branch_id, message_id,
                attachment_name, attachment_path, size_bytes, content_hash, mime_type, text_bytes, ocr_used
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                provider,
                conversation_id,
                branch_id,
                message_id,
                attachment_name,
                str(attachment_path),
                size_bytes,
                content_hash,
                "text/plain",
                0,
                0,
            ),
        )


def test_attachments_stats_filters_provider_and_time(state_env, tmp_path, capsys):
    env = CommandEnv(ui=UI(plain=True))
    db_path = env.database.resolve_path()
    assert db_path is not None

    a = tmp_path / "a.txt"
    a.write_text("a", encoding="utf-8")
    b = tmp_path / "b.txt"
    b.write_text("b", encoding="utf-8")

    _seed_attachment(
        db_path,
        provider="codex",
        conversation_id="conv-1",
        branch_id="main",
        message_id="m1",
        timestamp="2024-01-01T00:00:00Z",
        attachment_name="a.txt",
        attachment_path=a.resolve(),
        size_bytes=10,
        content_hash="hash-a",
    )
    _seed_attachment(
        db_path,
        provider="claude-code",
        conversation_id="conv-2",
        branch_id="main",
        message_id="m2",
        timestamp="2024-02-01T00:00:00Z",
        attachment_name="b.txt",
        attachment_path=b.resolve(),
        size_bytes=20,
        content_hash="hash-b",
    )

    args = SimpleNamespace(
        attachments_cmd="stats",
        from_index=True,
        json=True,
        json_lines=False,
        dir=None,
        provider="claude-code",
        since="2024-01-15",
        until=None,
        ext=None,
        hash=True,
        sort="size",
        limit=10,
        csv=None,
        clean_orphans=False,
        dry_run=False,
    )
    run_attachments_cli(args, env)
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    assert payload["top"][0]["provider"] == "claude-code"


def test_attachments_clean_orphans_dry_run_and_apply(state_env, tmp_path, capsys):
    env = CommandEnv(ui=UI(plain=True))
    db_path = env.database.resolve_path()
    assert db_path is not None

    archive_root = tmp_path / "archive"
    attachment_dir = archive_root / "codex" / "conv-1" / "attachments"
    attachment_dir.mkdir(parents=True)
    referenced = attachment_dir / "keep.txt"
    referenced.write_text("keep", encoding="utf-8")
    orphan = attachment_dir / "orphan.txt"
    orphan.write_text("orphan", encoding="utf-8")

    _seed_attachment(
        db_path,
        provider="codex",
        conversation_id="conv-1",
        branch_id="main",
        message_id="m1",
        timestamp="2024-01-01T00:00:00Z",
        attachment_name="keep.txt",
        attachment_path=referenced.resolve(),
        size_bytes=4,
        content_hash="hash-keep",
    )

    args = SimpleNamespace(
        attachments_cmd="stats",
        from_index=True,
        json=True,
        json_lines=False,
        dir=archive_root,
        provider="codex",
        since=None,
        until=None,
        ext=None,
        hash=False,
        sort="size",
        limit=10,
        csv=None,
        clean_orphans=True,
        dry_run=True,
    )
    run_attachments_cli(args, env)
    payload = json.loads(capsys.readouterr().out)
    assert payload["orphans"]["count"] == 1
    assert payload["orphans"]["removed"] == 0
    assert orphan.exists()

    args.dry_run = False
    run_attachments_cli(args, env)
    payload = json.loads(capsys.readouterr().out)
    assert payload["orphans"]["count"] == 1
    assert payload["orphans"]["removed"] == 1
    assert not orphan.exists()
