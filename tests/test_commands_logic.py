from __future__ import annotations

import json
import sqlite3
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from polylogue.commands import CommandEnv, RenderOptions, SyncOptions, render_command, sync_command
from polylogue.cli import run_prune_cli
from polylogue import commands as cmd_module, util


class DummyConsole:
    def __init__(self):
        self.messages = []

    def print(self, *args, **kwargs):  # pragma: no cover - debug helper
        self.messages.append((args, kwargs))


@dataclass
class DummyUI:
    plain: bool = True
    console: DummyConsole = field(default_factory=DummyConsole)

    def summary(self, *args, **kwargs):  # pragma: no cover - unused in tests
        pass

    def banner(self, *args, **kwargs):  # pragma: no cover - unused in tests
        pass


def test_render_command_persists_state(tmp_path, state_env, monkeypatch):
    records = []
    monkeypatch.setattr(cmd_module, "add_run", lambda record: records.append(record))
    src = tmp_path / "sample.json"
    payload = {
        "id": "conv-1",
        "title": "Sample Chat",
        "chunkedPrompt": {
            "chunks": [
                {"role": "user", "text": "Hello"},
                {"role": "model", "text": "Hi!"},
            ]
        },
    }
    src.write_text(json.dumps(payload), encoding="utf-8")
    out_dir = tmp_path / "out"

    options = RenderOptions(
        inputs=[src],
        output_dir=out_dir,
        collapse_threshold=16,
        download_attachments=False,
        dry_run=False,
        force=False,
        html=False,
        html_theme="light",
        diff=False,
    )
    result = render_command(options, CommandEnv(ui=DummyUI()))

    assert len(result.files) == 1
    md_path = out_dir / "sample" / "conversation.md"
    assert md_path.exists()
    assert result.files[0].slug == "sample"
    state_data = json.loads((state_env / "state.json").read_text(encoding="utf-8"))
    conv_state = state_data["conversations"]["render"]["conv-1"]
    assert conv_state["collapseThreshold"] == 16
    assert conv_state["dirty"] is False
    assert records and records[0].get("duration") is not None
    assert records[0]["duration"] >= 0

    db_path = state_env / "index.sqlite"
    assert db_path.exists()
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT title FROM conversations WHERE provider = ? AND conversation_id = ?",
            ("render", "conv-1"),
        ).fetchone()
        assert row and row[0] == "Sample Chat"
        fts_row = conn.execute(
            "SELECT content FROM conversations_fts WHERE provider = ? AND conversation_id = ?",
            ("render", "conv-1"),
        ).fetchone()
        assert fts_row and "Hello" in fts_row[0]
    finally:
        conn.close()
    # rerun should skip due to unchanged content
    second = render_command(options, CommandEnv(ui=DummyUI()))
    assert second.files == []


def test_sync_command_with_stub_drive(tmp_path, monkeypatch, state_env):
    records = []
    monkeypatch.setattr(cmd_module, "add_run", lambda record: records.append(record))
    class StubDrive:
        def __init__(self, ui):
            self.ui = ui

        def resolve_folder_id(self, folder_name, folder_id):  # noqa: ARG002
            return "folder-123"

        def list_chats(self, folder_name, folder_id):  # noqa: ARG002
            return [
                {
                    "id": "drive-1",
                    "name": "Drive Sample",
                    "modifiedTime": "2024-01-01T00:00:00Z",
                }
            ]

        def download_chat_bytes(self, file_id):  # noqa: ARG002
            payload = {
                "chunkedPrompt": {
                    "chunks": [
                        {"role": "user", "text": "Question"},
                        {"role": "model", "text": "Answer"},
                    ]
                }
            }
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(cmd_module, "DriveClient", StubDrive)

    out_dir = tmp_path / "drive"
    options = SyncOptions(
        folder_name="AI Studio",
        folder_id=None,
        output_dir=out_dir,
        collapse_threshold=12,
        download_attachments=False,
        dry_run=False,
        force=False,
        prune=False,
        since=None,
        until=None,
        name_filter=None,
        html=False,
        html_theme="light",
        diff=False,
    )
    result = sync_command(options, CommandEnv(ui=DummyUI()))

    assert result.count == 1
    expected_output = out_dir / util.sanitize_filename('Drive Sample') / "conversation.md"
    assert result.items[0].output == expected_output
    assert result.items[0].slug == util.sanitize_filename('Drive Sample')
    assert expected_output.exists()
    state_data = json.loads((state_env / "state.json").read_text(encoding="utf-8"))
    drive_state = state_data["conversations"]['drive-sync']['drive-1']
    assert drive_state['slug'] == util.sanitize_filename('Drive Sample')
    assert drive_state['outputPath'] == str(expected_output)
    assert records and records[0].get("duration") is not None
    assert records[0]["duration"] >= 0

    db_path = state_env / "index.sqlite"
    assert db_path.exists()
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT title FROM conversations WHERE provider = ? AND conversation_id = ?",
            ("drive-sync", "drive-1"),
        ).fetchone()
        assert row and row[0] == "Drive Sample"
    finally:
        conn.close()


def test_run_prune_cli_removes_legacy(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    legacy_md = root / "old.md"
    legacy_md.write_text("legacy", encoding="utf-8")
    legacy_html = root / "old.html"
    legacy_html.write_text("legacy", encoding="utf-8")
    attachment_dir = root / "old_attachments"
    attachment_dir.mkdir()
    (attachment_dir / "file.txt").write_text("data", encoding="utf-8")

    ui = DummyUI()
    env = CommandEnv(ui=ui)
    args = Namespace(dirs=[root], dry_run=False)

    run_prune_cli(args, env)

    assert not legacy_md.exists()
    assert not legacy_html.exists()
    assert not attachment_dir.exists()
