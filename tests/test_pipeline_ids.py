from __future__ import annotations

from polylogue.assets import asset_path
from polylogue.importers.base import ParsedAttachment
from polylogue.pipeline.ids import attachment_content_id


def test_attachment_content_id_moves_file_into_assets(tmp_path):
    archive_root = tmp_path / "archive"
    uploads = tmp_path / "uploads"
    archive_root.mkdir()
    uploads.mkdir()
    source_file = uploads / "note.txt"
    source_file.write_text("hello world", encoding="utf-8")

    attachment = ParsedAttachment(
        provider_attachment_id="file-1",
        message_provider_id="msg-1",
        name="note.txt",
        mime_type="text/plain",
        size_bytes=11,
        path=str(source_file),
        provider_meta={},
    )

    # attachment_content_id now returns (digest, updated_meta, updated_path) without mutation
    digest, updated_meta, updated_path = attachment_content_id(
        "chatgpt", attachment, archive_root=archive_root
    )
    target = asset_path(archive_root, digest)

    assert digest
    assert updated_path == str(target)  # returned path, not mutated attachment.path
    assert updated_meta is not None and "sha256" in updated_meta
    assert not source_file.exists()
    assert target.exists()
