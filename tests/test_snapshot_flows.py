from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

from polylogue import util
from polylogue.commands import CommandEnv, RenderOptions, SyncOptions, render_command, sync_command
from polylogue.local_sync import sync_codex_sessions


class DummyUI:
    plain = True

    def __init__(self):
        self.console = self

    def print(self, *_args, **_kwargs):  # pragma: no cover - diagnostics only
        pass

    def summary(self, *_args, **_kwargs):  # pragma: no cover
        pass

    def banner(self, *_args, **_kwargs):  # pragma: no cover
        pass

    @contextmanager
    def progress(self, *_args, **_kwargs):  # pragma: no cover - unused in tests
        """Dummy progress context manager that yields None."""
        yield None


def _open_conversation_row(db_path: Path, provider: str, conversation_id: str):
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT slug, title FROM conversations WHERE provider = ? AND conversation_id = ?",
            (provider, conversation_id),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _read_state_entry(state_home: Path, provider: str, conversation_id: str) -> dict:
    db_path = state_home / "polylogue.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT metadata_json FROM conversations WHERE provider = ? AND conversation_id = ?",
            (provider, conversation_id),
        ).fetchone()
        if not row or not row["metadata_json"]:
            return {}
        return json.loads(row["metadata_json"])
    finally:
        conn.close()


@pytest.fixture
def fake_diff(monkeypatch):
    def _fake(old_path, new_path, suffix=".diff.txt"):
        path = new_path.with_suffix(new_path.suffix + suffix)
        path.write_text("diff", encoding="utf-8")
        return path

    monkeypatch.setattr(util, "write_delta_diff", _fake, raising=False)
    return _fake


def test_render_snapshot_with_html_and_diff(tmp_path, state_env, fake_diff):
    src = tmp_path / "render.json"
    payload = {
        "id": "conv-render",
        "title": "Snapshot Render",
        "chunkedPrompt": {
            "chunks": [
                {"role": "user", "text": "Hello", "timestamp": "2024-01-01T00:00:00Z"},
                {"role": "model", "text": "Hi there", "timestamp": "2024-01-01T00:01:00Z"},
            ]
        },
        "metadata": {"createTime": "2024-01-01T00:00:00Z"},
    }
    src.write_text(json.dumps(payload), encoding="utf-8")
    out_dir = tmp_path / "out"

    options = RenderOptions(
        inputs=[src],
        output_dir=out_dir,
        collapse_threshold=12,
        download_attachments=False,
        dry_run=False,
        force=False,
        html=True,
        html_theme="dark",
        diff=True,
    )

    env = CommandEnv(ui=DummyUI())
    first = render_command(options, env)
    assert first.files
    output_md = out_dir / "render" / "conversation.md"
    assert output_md.exists()
    html_path = output_md.with_name("conversation.html")
    assert html_path.exists()

    render_state = _read_state_entry(state_env, "render", "conv-render")
    assert render_state["outputPath"] == str(output_md)
    assert render_state["htmlPath"] == str(html_path)

    row = _open_conversation_row(state_env / "polylogue.db", "render", "conv-render")
    assert row and row["slug"] == "render"

    # mutate payload to force diff on rerender
    payload["chunkedPrompt"]["chunks"][1]["text"] = "Hi there v2"
    src.write_text(json.dumps(payload), encoding="utf-8")

    second = render_command(options, env)
    assert second.files[0].diff is not None
    assert second.files[0].diff.read_text(encoding="utf-8") == "diff"


def test_sync_snapshot_updates_state_and_diff(tmp_path, state_env, monkeypatch, fake_diff):
    class SnapshotDrive:
        def __init__(self, ui):
            self.ui = ui
            self.invocations = 0

        def resolve_folder_id(self, *_args, **_kwargs):
            return "folder-demo"

        def list_chats(self, *_args, **_kwargs):
            return [
                {
                    "id": "drive-snapshot",
                    "name": "Drive Snapshot",
                    "modifiedTime": "2024-01-02T00:00:00Z",
                }
            ]

        def download_chat_bytes(self, file_id, *_args, **_kwargs):  # noqa: ARG002
            self.invocations += 1
            suffix = "" if self.invocations == 1 else " updated"
            payload = {
                "chunkedPrompt": {
                    "chunks": [
                        {
                            "role": "user",
                            "text": "Sync question",
                            "timestamp": "2024-01-02T00:00:00Z",
                        },
                        {
                            "role": "model",
                            "text": f"Sync answer{suffix}",
                            "timestamp": "2024-01-02T00:01:00Z",
                        },
                    ]
                }
            }
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr("polylogue.commands.DriveClient", SnapshotDrive)

    options = SyncOptions(
        folder_name="Demo",
        folder_id=None,
        output_dir=tmp_path / "drive",
        collapse_threshold=16,
        download_attachments=False,
        dry_run=False,
        force=False,
        prune=False,
        since=None,
        until=None,
        name_filter=None,
        html=True,
        html_theme="dark",
        diff=True,
    )

    env = CommandEnv(ui=DummyUI())
    first = sync_command(options, env)
    assert first.count == 1

    drive_entry = _read_state_entry(state_env, "drive-sync", "drive-snapshot")
    expected_slug = util.sanitize_filename("Drive Snapshot")
    assert drive_entry["slug"] == expected_slug

    row = _open_conversation_row(state_env / "polylogue.db", "drive-sync", "drive-snapshot")
    assert row and row["title"] == "Drive Snapshot"

    second = sync_command(options, env)
    assert second.items[0].diff is not None
    assert second.items[0].diff.read_text(encoding="utf-8") == "diff"


def _write_codex_session(base_dir: Path, name: str, entries: list[dict]) -> Path:
    session_dir = base_dir / "sessions" / "2025" / "01" / "01"
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry))
            handle.write("\n")
    return path


def test_local_sync_snapshot_watch_mode(tmp_path, state_env, fake_diff):
    base_dir = tmp_path / "codex"
    entries = [
        {"type": "session_meta", "payload": {"session_id": "watch"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "text", "text": "ask"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "reply"}],
            },
        },
    ]
    session_path = _write_codex_session(base_dir, "watch", entries)

    env = CommandEnv(ui=DummyUI())
    out_dir = tmp_path / "codex-out"

    first = sync_codex_sessions(
        base_dir=base_dir / "sessions",
        output_dir=out_dir,
        collapse_threshold=10,
        html=True,
        html_theme="dark",
        force=False,
        prune=False,
        diff=True,
        sessions=[session_path],
        registrar=env.registrar,
    )

    assert first.written
    convo_dir = out_dir / first.written[0].slug
    assert (convo_dir / "conversation.md").exists()
    assert (convo_dir / "conversation.html").exists()

    conn = sqlite3.connect(state_env / "polylogue.db")
    try:
        providers = {
            row[0]
            for row in conn.execute("SELECT DISTINCT provider FROM conversations").fetchall()
        }
    finally:
        conn.close()
    assert "codex" in providers

    # append another assistant turn to trigger diff on resync
    entries.append(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "reply v2"}],
            },
        }
    )
    session_path.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")

    second = sync_codex_sessions(
        base_dir=base_dir / "sessions",
        output_dir=out_dir,
        collapse_threshold=10,
        html=True,
        html_theme="dark",
        force=False,
        prune=False,
        diff=True,
        sessions=[session_path],
        registrar=env.registrar,
    )

    assert second.written[0].diff_path is not None
    assert second.written[0].diff_path.read_text(encoding="utf-8") == "diff"
