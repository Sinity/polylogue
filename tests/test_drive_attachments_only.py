from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from polylogue import util
from polylogue.cli.sync import _run_sync_drive
from polylogue.commands import CommandEnv


class DummyConsole:
    def print(self, *_args, **_kwargs):
        return None


class DummyUI:
    plain = True

    def __init__(self):
        self.console = DummyConsole()


def test_sync_drive_attachments_only_retries_failed_items(tmp_path: Path, state_env, monkeypatch):
    out_dir = tmp_path / "archive"
    out_dir.mkdir(parents=True, exist_ok=True)

    util.add_run(
        {
            "cmd": "sync drive",
            "provider": "drive",
            "out": str(out_dir),
            "failedAttachments": [
                {
                    "id": "chat-1",
                    "name": "Chat One",
                    "slug": "chat-one",
                    "attachments": [
                        {"id": "att-1", "filename": "a.txt", "path": "attachments/a.txt"},
                    ],
                }
            ],
        }
    )
    run_id = util.load_runs(limit=1)[0]["id"]

    class StubDrive:
        def __init__(self):
            self.ui = DummyUI()

        def attachment_meta(self, file_id):  # noqa: ARG002
            return {"name": "a.txt", "modifiedTime": "2024-01-01T00:00:00Z"}

        def download_attachment(self, file_id, path: Path) -> bool:  # noqa: ARG002
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ok", encoding="utf-8")
            return True

        def touch_mtime(self, path: Path, _mt):  # noqa: ARG002
            return None

    captured = []
    monkeypatch.setattr("polylogue.cli.sync.add_run", lambda payload: captured.append(payload))

    env = CommandEnv(ui=DummyUI())
    env.drive = StubDrive()

    args = SimpleNamespace(
        json=False,
        list_only=False,
        attachments_only=True,
        resume_from=run_id,
        out=None,
        links_only=False,
        offline=False,
        drive_retries=None,
        drive_retry_base=None,
        dry_run=False,
        force=False,
        folder_name="Google AI Studio",
        folder_id=None,
        since=None,
        until=None,
        name_filter=None,
    )
    _run_sync_drive(args, env)

    assert (out_dir / "chat-one" / "attachments" / "a.txt").exists()
    assert captured
    assert captured[0]["attachmentsOnly"] is True
    assert captured[0]["attachmentDownloads"] == 1
