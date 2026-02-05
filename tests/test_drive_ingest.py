from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

from polylogue.config import Source
from polylogue.sources import DriveFile, iter_drive_conversations


@dataclass
class StubDriveClient:
    payload: dict

    def resolve_folder_id(self, folder_ref: str) -> str:
        return "folder-1"

    def iter_json_files(self, folder_id: str):
        yield DriveFile(
            file_id="file-1",
            name="chat.json",
            mime_type="application/json",
            modified_time=None,
            size_bytes=None,
        )

    def download_json_payload(self, file_id: str, *, name: str):
        return self.payload

    def download_to_path(self, file_id, dest):
        raise AssertionError("download_to_path should not be called when download_assets=False")


def test_drive_ingest_chunked_prompt_no_download(tmp_path):
    payload = {
        "title": "Drive Chat",
        "chunkedPrompt": {
            "chunks": [
                {"role": "user", "text": "Hi"},
                {
                    "role": "model",
                    "text": "Hello",
                    "driveDocument": {"id": "att-1", "name": "doc.txt"},
                },
            ]
        },
    }
    source = Source(name="gemini", folder="Google AI Studio")
    client = StubDriveClient(payload=payload)

    conversations = list(
        iter_drive_conversations(
            source=source,
            archive_root=tmp_path,
            client=client,
            download_assets=False,
        )
    )
    assert len(conversations) == 1
    convo = conversations[0]
    assert [msg.role for msg in convo.messages] == ["user", "assistant"]
    assert len(convo.attachments) == 1
    attachment = convo.attachments[0]
    assert attachment.provider_attachment_id == "att-1"
    assert attachment.message_provider_id == "chunk-2"
    assert attachment.path is None


class TestDriveDownloadFailureTracking:
    """Tests for tracking Drive download failures."""

    def test_download_failure_tracked_in_result(self):
        """Failed downloads should be tracked in the result, not silently ignored.

        This test SHOULD FAIL until failure tracking is implemented.
        """
        from polylogue.sources import download_drive_files
        from polylogue.sources.drive_client import DriveFile

        # Mock the drive client to fail on specific files
        mock_client = MagicMock()
        mock_client.iter_json_files.return_value = [
            DriveFile(file_id="file1", name="good.json", mime_type="application/json", modified_time=None, size_bytes=100),
            DriveFile(file_id="file2", name="bad.json", mime_type="application/json", modified_time=None, size_bytes=100),
            DriveFile(file_id="file3", name="also_good.json", mime_type="application/json", modified_time=None, size_bytes=100),
        ]

        def mock_download(file_id, dest):
            if file_id == "file2":
                raise OSError("Download failed")
            dest.write_text('{"test": true}')

        mock_client.download_to_path.side_effect = mock_download

        result = download_drive_files(mock_client, "folder123", Path("/tmp/test"))

        # Result should track failures
        assert hasattr(result, "failed_files") or "failed" in result
        assert len(result.failed_files) >= 1
        assert any("bad.json" in str(f) for f in result.failed_files)

    def test_download_continues_after_single_failure(self):
        """Download should continue processing other files after one fails."""
        from polylogue.sources import download_drive_files
        from polylogue.sources.drive_client import DriveFile

        mock_client = MagicMock()
        mock_client.iter_json_files.return_value = [
            DriveFile(file_id="f1", name="first.json", mime_type="application/json", modified_time=None, size_bytes=100),
            DriveFile(file_id="f2", name="fails.json", mime_type="application/json", modified_time=None, size_bytes=100),
            DriveFile(file_id="f3", name="third.json", mime_type="application/json", modified_time=None, size_bytes=100),
        ]

        download_count = [0]

        def mock_download(file_id, dest):
            if file_id == "f2":
                raise OSError("Failed")
            download_count[0] += 1
            dest.write_text('{}')

        mock_client.download_to_path.side_effect = mock_download

        download_drive_files(mock_client, "folder", Path("/tmp/test"))

        # Should have attempted all 3, succeeded on 2
        assert download_count[0] == 2
