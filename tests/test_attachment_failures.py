from pathlib import Path

import pytest

from polylogue.pipeline import collect_attachments


class FakeDrive:
    def __init__(self, fail_ids):
        self.fail_ids = set(fail_ids)

    def attachment_meta(self, att_id):
        return {"name": att_id, "modifiedTime": "2024-01-01T00:00:00Z"}

    def download_attachment(self, att_id, path: Path) -> bool:
        if att_id in self.fail_ids:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
        return True

    def touch_mtime(self, path: Path, _mt):
        return None


def test_collect_attachments_raises_on_failure(tmp_path: Path):
    drive = FakeDrive(fail_ids={"bad"})
    chunks = [
        {
            "messages": [
                {
                    "author": "user",
                    "content": {
                        "parts": [
                            {"text": "see https://drive.google.com/file/d/bad"},
                            {"text": "and https://drive.google.com/file/d/good"},
                        ]
                    },
                }
            ]
        }
    ]
    with pytest.raises(RuntimeError, match="Failed to download attachments: bad"):
        collect_attachments(drive, chunks, tmp_path / "conv.md", force=False, dry_run=False)
