from __future__ import annotations

from pathlib import Path

from polylogue.pipeline import ChatContext, build_document_from_chunks


class FakeDrive:
    def __init__(self):
        self.downloaded = []

    def attachment_meta(self, file_id: str):
        return {"name": f"{file_id}.txt", "modifiedTime": "2024-01-01T00:00:00Z"}

    def download_attachment(self, file_id: str, path: Path) -> bool:
        path.write_text(f"payload-{file_id}", encoding="utf-8")
        self.downloaded.append((file_id, path))
        return True

    def touch_mtime(self, path: Path, iso_time: str) -> None:  # pragma: no cover - noop
        path.touch(exist_ok=True)


def _chunk_with_drive_id(file_id: str) -> dict:
    return {"driveDocument": {"id": file_id}, "role": "assistant", "text": f"see {file_id}"}


def test_build_document_downloads_attachments(tmp_path):
    chunks = [_chunk_with_drive_id("abc123"), _chunk_with_drive_id("xyz789")]
    context = ChatContext(
        title="Sample Conversation",
        chat_id="conv-1",
        modified_time="2024-01-01T00:00:00Z",
        created_time="2024-01-01T00:00:00Z",
        run_settings=None,
        citations=None,
        source_mime=None,
    )
    md_path = tmp_path / "conversation.md"
    drive = FakeDrive()

    document = build_document_from_chunks(
        chunks,
        context,
        md_path,
        collapse_threshold=10,
        download_attachments=True,
        drive=drive,
        force=False,
        dry_run=False,
    )

    attachments_dir = md_path.parent / "attachments"
    assert attachments_dir.exists()
    stored = sorted(p.name for p in attachments_dir.iterdir())
    assert stored == ["abc123.txt", "xyz789.txt"]
    assert len(document.attachments) == 2
    assert all(att.local_path is not None for att in document.attachments)


def test_build_document_remote_links_without_download(tmp_path):
    chunks = [_chunk_with_drive_id("abc123")]
    context = ChatContext(
        title="Remote Conversation",
        chat_id="conv-remote",
        modified_time=None,
        created_time=None,
        run_settings=None,
        citations=None,
        source_mime=None,
    )
    md_path = tmp_path / "remote.md"

    document = build_document_from_chunks(
        chunks,
        context,
        md_path,
        collapse_threshold=5,
        download_attachments=False,
        drive=None,
        force=False,
        dry_run=False,
    )

    assert not (md_path.parent / "attachments").exists()
    assert document.attachments  # remote attachments captured
    first = document.attachments[0]
    assert first.remote is True
    assert "abc123" in first.link
