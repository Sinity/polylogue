from __future__ import annotations

import json
from contextlib import contextmanager

from polylogue import util
from polylogue.commands import CommandEnv, RenderOptions, SyncOptions, render_command, sync_command
from polylogue.document_store import read_existing_document


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
    def progress(self, *_args, **_kwargs):  # pragma: no cover
        class DummyProgressTracker:
            def advance(self, *args, **kwargs):
                pass

            def update(self, *args, **kwargs):
                pass

        yield DummyProgressTracker()


def test_render_meta_in_frontmatter_and_runs(tmp_path, state_env):
    src = tmp_path / "render.json"
    payload = {
        "id": "conv-render-meta",
        "title": "Snapshot Render Meta",
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
        html=False,
        html_theme=None,
        diff=False,
        meta={"suite": "tests"},
    )

    env = CommandEnv(ui=DummyUI())
    render_command(options, env)
    output_md = out_dir / "render" / "conversation.md"
    existing = read_existing_document(output_md)
    assert existing is not None
    assert existing.metadata["polylogue"]["cliMeta"] == {"suite": "tests"}

    runs = util.load_runs()
    assert runs
    last = runs[-1]
    assert last["cmd"] == "render"
    assert last["meta"] == {"suite": "tests"}


def test_sync_drive_meta_in_frontmatter(tmp_path, state_env, monkeypatch):
    class SnapshotDrive:
        def __init__(self, ui):
            self.ui = ui

        def resolve_folder_id(self, *_args, **_kwargs):
            return "folder-demo"

        def list_chats(self, *_args, **_kwargs):
            return [
                {
                    "id": "drive-meta",
                    "name": "Drive Meta",
                    "modifiedTime": "2024-01-02T00:00:00Z",
                }
            ]

        def download_chat_bytes(self, *_args, **_kwargs):
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
                            "text": "Sync answer",
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
        html=False,
        html_theme="dark",
        diff=False,
        meta={"from": "tests"},
    )

    env = CommandEnv(ui=DummyUI())
    result = sync_command(options, env)
    assert result.count == 1
    existing = read_existing_document(result.items[0].output)
    assert existing is not None
    assert existing.metadata["polylogue"]["cliMeta"] == {"from": "tests"}

