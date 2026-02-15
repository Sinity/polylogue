"""Drive operations tests — file download, attachment processing, conversation iteration, JSON payload parsing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from polylogue.config import Source
from polylogue.sources import (
    DriveClient,
    DriveFile,
    download_drive_files,
    iter_drive_conversations,
)
from polylogue.sources.drive import DriveDownloadResult, _apply_drive_attachments
from polylogue.sources.parsers.base import ParsedAttachment, ParsedConversation

# ============================================================================
# Test Data Tables (Module-level constants for parametrization)
# ============================================================================

DOWNLOAD_DRIVE_FILES_CASES = [
    (1, 1, 0, "single_file_success"),
    (2, 2, 0, "multiple_files_success"),
    (2, 1, 1, "one_failure_mixed"),
    (3, 2, 1, "mixed_with_failures"),
    (0, 0, 0, "empty_folder"),
]


# ============================================================================
# Tests for download_drive_files
# ============================================================================


class TestDownloadDriveFiles:
    """Tests for download_drive_files function."""

    def test_successful_single_file_download(self, tmp_path):
        """Download single file successfully."""
        mock_client = MagicMock(spec=DriveClient)
        mock_client.iter_json_files.return_value = [
            DriveFile(
                file_id="f1",
                name="test.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
        ]
        mock_client.download_to_path.return_value = None

        result = download_drive_files(mock_client, "folder-1", tmp_path)

        assert result.total_files == 1
        assert len(result.downloaded_files) == 1
        assert len(result.failed_files) == 0
        assert result.downloaded_files[0].name == "test.json"

    def test_successful_multiple_files_download(self, tmp_path):
        """Download multiple files successfully."""
        mock_client = MagicMock(spec=DriveClient)
        mock_client.iter_json_files.return_value = [
            DriveFile(
                file_id="f1",
                name="chat1.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
            DriveFile(
                file_id="f2",
                name="chat2.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=200,
            ),
        ]
        mock_client.download_to_path.return_value = None

        result = download_drive_files(mock_client, "folder-1", tmp_path)

        assert result.total_files == 2
        assert len(result.downloaded_files) == 2
        assert len(result.failed_files) == 0

    def test_download_with_failure(self, tmp_path):
        """Handle download failure gracefully."""
        mock_client = MagicMock(spec=DriveClient)
        mock_client.iter_json_files.return_value = [
            DriveFile(
                file_id="f1",
                name="good.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
            DriveFile(
                file_id="f2",
                name="bad.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
        ]

        def mock_download(file_id, dest):
            if file_id == "f2":
                raise OSError("Network error")

        mock_client.download_to_path.side_effect = mock_download

        result = download_drive_files(mock_client, "folder-1", tmp_path)

        assert result.total_files == 2
        assert len(result.downloaded_files) == 1
        assert len(result.failed_files) == 1
        assert result.failed_files[0]["file_id"] == "f2"
        assert "error" in result.failed_files[0]

    def test_download_continues_after_failure(self, tmp_path):
        """Download should continue after a single file fails."""
        mock_client = MagicMock(spec=DriveClient)
        download_calls = []

        mock_client.iter_json_files.return_value = [
            DriveFile(
                file_id="f1",
                name="first.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
            DriveFile(
                file_id="f2",
                name="fails.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
            DriveFile(
                file_id="f3",
                name="third.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
        ]

        def mock_download(file_id, dest):
            download_calls.append(file_id)
            if file_id == "f2":
                raise Exception("Failed")

        mock_client.download_to_path.side_effect = mock_download

        result = download_drive_files(mock_client, "folder-1", tmp_path)

        assert len(download_calls) == 3
        assert result.total_files == 3
        assert len(result.downloaded_files) == 2
        assert len(result.failed_files) == 1

    def test_empty_folder(self, tmp_path):
        """Empty folder should return empty result."""
        mock_client = MagicMock(spec=DriveClient)
        mock_client.iter_json_files.return_value = []

        result = download_drive_files(mock_client, "folder-1", tmp_path)

        assert result.total_files == 0
        assert len(result.downloaded_files) == 0
        assert len(result.failed_files) == 0

    def test_result_type_is_data_class(self, tmp_path):
        """Result should be a DriveDownloadResult instance."""
        mock_client = MagicMock(spec=DriveClient)
        mock_client.iter_json_files.return_value = []

        result = download_drive_files(mock_client, "folder-1", tmp_path)

        assert isinstance(result, DriveDownloadResult)
        assert hasattr(result, "downloaded_files")
        assert hasattr(result, "failed_files")
        assert hasattr(result, "total_files")

    def test_file_count_accuracy(self, tmp_path):
        """total_files should match sum of downloaded and failed."""
        mock_client = MagicMock(spec=DriveClient)
        mock_client.iter_json_files.return_value = [
            DriveFile(
                file_id="f1",
                name="a.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
            DriveFile(
                file_id="f2",
                name="b.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
            DriveFile(
                file_id="f3",
                name="c.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
        ]

        def mock_download(file_id, dest):
            if file_id == "f2":
                raise Exception("Failed")

        mock_client.download_to_path.side_effect = mock_download

        result = download_drive_files(mock_client, "folder-1", tmp_path)

        assert result.total_files == len(result.downloaded_files) + len(result.failed_files)
        assert result.total_files == 3

    def test_all_files_fail(self, tmp_path):
        """All files failing should be reported."""
        mock_client = MagicMock(spec=DriveClient)
        mock_client.iter_json_files.return_value = [
            DriveFile(
                file_id="f1",
                name="fail1.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
            DriveFile(
                file_id="f2",
                name="fail2.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
        ]

        mock_client.download_to_path.side_effect = Exception("All fail")

        result = download_drive_files(mock_client, "folder-1", tmp_path)

        assert result.total_files == 2
        assert len(result.downloaded_files) == 0
        assert len(result.failed_files) == 2

    def test_failed_file_error_message_preserved(self, tmp_path):
        """Failed file should include the actual error message."""
        mock_client = MagicMock(spec=DriveClient)
        error_msg = "Permission denied: file is private"

        mock_client.iter_json_files.return_value = [
            DriveFile(
                file_id="f1",
                name="private.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
        ]
        mock_client.download_to_path.side_effect = PermissionError(error_msg)

        result = download_drive_files(mock_client, "folder-1", tmp_path)

        assert len(result.failed_files) == 1
        assert error_msg in result.failed_files[0]["error"]


# ============================================================================
# Tests for _apply_drive_attachments
# ============================================================================


class TestApplyDriveAttachments:
    """Tests for _apply_drive_attachments function."""

    def test_download_disabled_skips_processing(self, tmp_path):
        """download_assets=False should skip attachment processing."""
        convo = ParsedConversation(
            provider_name="test",
            provider_conversation_id="test-1",
            messages=[],
            attachments=[
                ParsedAttachment(
                    provider_attachment_id="attach-1",
                    message_provider_id="msg-1",
                    name="test.pdf",
                )
            ],
        )

        mock_client = MagicMock(spec=DriveClient)
        _apply_drive_attachments(
            convo=convo,
            client=mock_client,
            archive_root=tmp_path,
            download_assets=False,
        )

        mock_client.download_to_path.assert_not_called()

    def test_no_provider_attachment_id_skips(self, tmp_path):
        """Attachment without provider_attachment_id should be skipped."""
        convo = ParsedConversation(
            provider_name="test",
            provider_conversation_id="test-1",
            messages=[],
            attachments=[
                ParsedAttachment(
                    provider_attachment_id="test-id",
                    message_provider_id="msg-1",
                    name="test.pdf",
                )
            ],
        )
        convo.attachments[0].provider_attachment_id = None

        mock_client = MagicMock(spec=DriveClient)
        _apply_drive_attachments(
            convo=convo,
            client=mock_client,
            archive_root=tmp_path,
            download_assets=True,
        )

        mock_client.download_to_path.assert_not_called()

    def test_successful_attachment_download(self, tmp_path):
        """Successful download should update attachment fields."""
        convo = ParsedConversation(
            provider_name="test",
            provider_conversation_id="test-1",
            messages=[],
            attachments=[
                ParsedAttachment(
                    provider_attachment_id="attach-1",
                    message_provider_id="msg-1",
                    name=None,
                    mime_type=None,
                    size_bytes=None,
                    path=None,
                    provider_meta=None,
                )
            ],
        )

        mock_client = MagicMock(spec=DriveClient)
        mock_file = DriveFile(
            file_id="attach-1",
            name="document.pdf",
            mime_type="application/pdf",
            modified_time=None,
            size_bytes=5000,
        )
        mock_client.download_to_path.return_value = mock_file

        _apply_drive_attachments(
            convo=convo,
            client=mock_client,
            archive_root=tmp_path,
            download_assets=True,
        )

        att = convo.attachments[0]
        assert att.name == "document.pdf"
        assert att.mime_type == "application/pdf"
        assert att.size_bytes == 5000
        assert att.path is not None
        assert att.provider_meta is not None
        assert "drive_id" in att.provider_meta

    def test_partial_attachment_fields_filled(self, tmp_path):
        """Download should only fill empty fields, not overwrite existing."""
        convo = ParsedConversation(
            provider_name="test",
            provider_conversation_id="test-1",
            messages=[],
            attachments=[
                ParsedAttachment(
                    provider_attachment_id="attach-1",
                    message_provider_id="msg-1",
                    name="original_name.txt",
                    mime_type="text/plain",
                    size_bytes=1000,
                    path=None,
                    provider_meta=None,
                )
            ],
        )

        mock_client = MagicMock(spec=DriveClient)
        mock_file = DriveFile(
            file_id="attach-1",
            name="document.pdf",
            mime_type="application/pdf",
            modified_time=None,
            size_bytes=5000,
        )
        mock_client.download_to_path.return_value = mock_file

        _apply_drive_attachments(
            convo=convo,
            client=mock_client,
            archive_root=tmp_path,
            download_assets=True,
        )

        att = convo.attachments[0]
        assert att.name == "original_name.txt"
        assert att.mime_type == "text/plain"
        assert att.size_bytes == 1000

    def test_multiple_attachments_processed(self, tmp_path):
        """Multiple attachments should all be processed."""
        attachments = [
            ParsedAttachment(
                provider_attachment_id="attach-1",
                message_provider_id="msg-1",
                name=None,
            ),
            ParsedAttachment(
                provider_attachment_id="attach-2",
                message_provider_id="msg-2",
                name=None,
            ),
        ]
        convo = ParsedConversation(
            provider_name="test",
            provider_conversation_id="test-1",
            messages=[],
            attachments=attachments,
        )

        mock_client = MagicMock(spec=DriveClient)

        def mock_download(file_id, dest):
            if file_id == "attach-1":
                return DriveFile(
                    file_id=file_id,
                    name="doc1.pdf",
                    mime_type="application/pdf",
                    modified_time=None,
                    size_bytes=1000,
                )
            elif file_id == "attach-2":
                return DriveFile(
                    file_id=file_id,
                    name="doc2.pdf",
                    mime_type="application/pdf",
                    modified_time=None,
                    size_bytes=2000,
                )

        mock_client.download_to_path.side_effect = mock_download

        _apply_drive_attachments(
            convo=convo,
            client=mock_client,
            archive_root=tmp_path,
            download_assets=True,
        )

        assert convo.attachments[0].name == "doc1.pdf"
        assert convo.attachments[1].name == "doc2.pdf"


# ============================================================================
# Tests for iter_drive_conversations
# ============================================================================


class TestIterDriveConversations:
    """Tests for iter_drive_conversations function."""

    def test_no_folder_returns_empty(self, tmp_path):
        """If source.folder is None or empty, should return empty."""
        source = Source(name="test", path="/some/path")
        result = list(
            iter_drive_conversations(
                source=source,
                archive_root=tmp_path,
                download_assets=False,
            )
        )
        assert result == []

    def test_initializes_cursor_state(self, tmp_path):
        """cursor_state should be initialized with file_count."""
        source = Source(name="test", folder="Google AI Studio")
        cursor_state = {}

        mock_client = MagicMock(spec=DriveClient)
        mock_client.resolve_folder_id.return_value = "folder-1"
        mock_client.iter_json_files.return_value = []

        list(
            iter_drive_conversations(
                source=source,
                archive_root=tmp_path,
                client=mock_client,
                cursor_state=cursor_state,
                download_assets=False,
            )
        )

        assert "file_count" in cursor_state
        assert cursor_state["file_count"] == 0

    def test_tracks_file_count_in_cursor(self, tmp_path):
        """cursor_state should track file count as files are processed."""
        source = Source(name="test", folder="Google AI Studio")
        cursor_state = {}

        mock_client = MagicMock(spec=DriveClient)
        mock_client.resolve_folder_id.return_value = "folder-1"
        mock_client.iter_json_files.return_value = [
            DriveFile(
                file_id="f1",
                name="chat1.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
            DriveFile(
                file_id="f2",
                name="chat2.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
        ]
        mock_client.download_json_payload.return_value = {
            "title": "Test",
            "messages": [],
        }

        list(
            iter_drive_conversations(
                source=source,
                archive_root=tmp_path,
                client=mock_client,
                cursor_state=cursor_state,
                download_assets=False,
            )
        )

        assert cursor_state["file_count"] == 2

    def test_tracks_latest_mtime_in_cursor(self, tmp_path):
        """cursor_state should track latest modification time."""
        source = Source(name="test", folder="Google AI Studio")
        cursor_state = {}

        mock_client = MagicMock(spec=DriveClient)
        mock_client.resolve_folder_id.return_value = "folder-1"
        mock_client.iter_json_files.return_value = [
            DriveFile(
                file_id="f1",
                name="chat1.json",
                mime_type="application/json",
                modified_time="2024-01-10T10:00:00Z",
                size_bytes=100,
            ),
            DriveFile(
                file_id="f2",
                name="chat2.json",
                mime_type="application/json",
                modified_time="2024-01-15T10:00:00Z",
                size_bytes=100,
            ),
        ]
        mock_client.download_json_payload.return_value = {
            "title": "Test",
            "messages": [],
        }

        list(
            iter_drive_conversations(
                source=source,
                archive_root=tmp_path,
                client=mock_client,
                cursor_state=cursor_state,
                download_assets=False,
            )
        )

        assert "latest_mtime" in cursor_state
        assert cursor_state["latest_mtime"] > 0
        assert "latest_file_id" in cursor_state
        assert cursor_state["latest_file_id"] == "f2"
        assert cursor_state["latest_file_name"] == "chat2.json"

    def test_handles_download_error_continues(self, tmp_path):
        """Download error should be tracked but iteration should continue."""
        source = Source(name="test", folder="Google AI Studio")
        cursor_state = {}

        mock_client = MagicMock(spec=DriveClient)
        mock_client.resolve_folder_id.return_value = "folder-1"
        mock_client.iter_json_files.return_value = [
            DriveFile(
                file_id="f1",
                name="good.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
            DriveFile(
                file_id="f2",
                name="bad.json",
                mime_type="application/json",
                modified_time=None,
                size_bytes=100,
            ),
        ]

        def mock_download(file_id, *, name):
            if file_id == "f2":
                raise Exception("Download failed")
            return {"title": "Good", "messages": []}

        mock_client.download_json_payload.side_effect = mock_download

        list(
            iter_drive_conversations(
                source=source,
                archive_root=tmp_path,
                client=mock_client,
                cursor_state=cursor_state,
                download_assets=False,
            )
        )

        assert "error_count" in cursor_state
        assert cursor_state["error_count"] >= 1
        assert "latest_error" in cursor_state
        assert "latest_error_file" in cursor_state

    def test_creates_client_if_not_provided(self, tmp_path, mock_drive_credentials):
        """Should create DriveClient if not provided."""
        source = Source(name="test", folder="Google AI Studio")

        with patch(
            "polylogue.sources.drive.DriveClient"
        ) as mock_drive_client_class:
            mock_instance = MagicMock(spec=DriveClient)
            mock_drive_client_class.return_value = mock_instance
            mock_instance.resolve_folder_id.return_value = "folder-1"
            mock_instance.iter_json_files.return_value = []

            list(
                iter_drive_conversations(
                    source=source,
                    archive_root=tmp_path,
                    download_assets=False,
                )
            )

            mock_drive_client_class.assert_called_once()

    def test_uses_provided_client(self, tmp_path):
        """Should use provided client instead of creating new one."""
        source = Source(name="test", folder="Google AI Studio")

        mock_client = MagicMock(spec=DriveClient)
        mock_client.resolve_folder_id.return_value = "folder-1"
        mock_client.iter_json_files.return_value = []

        with patch(
            "polylogue.sources.drive.DriveClient"
        ) as mock_drive_client_class:
            list(
                iter_drive_conversations(
                    source=source,
                    archive_root=tmp_path,
                    client=mock_client,
                    download_assets=False,
                )
            )

            mock_drive_client_class.assert_not_called()


# ============================================================================
# Parametrized Tests for download_json_payload
# ============================================================================


class TestDownloadJsonPayload:
    """Tests for DriveClient.download_json_payload method."""

    @pytest.mark.parametrize(
        "content,expect_type,expect_len,filename,desc",
        [
            (b'{"a": 1}\n{"b": 2}\n{"c": 3}\n', list, 3, "data.jsonl", "jsonl_file"),
            (b'{"msg": "hello"}\n{"msg": "world"}\n', list, 2, "data.jsonl.txt", "jsonl_txt_file"),
            (b'{"x": 1}\n{"y": 2}\n', list, 2, "data.ndjson", "ndjson_file"),
            (b'{"title": "Test", "content": "Data"}', dict, None, "data.json", "json_file"),
            (b'{"valid": 1}\n{invalid json}\n{"valid": 2}\n', list, 2, "data.jsonl", "jsonl_skip_invalid"),
            (b'{"a": 1}\n\n\n{"b": 2}\n   \n', list, 2, "data.jsonl", "jsonl_skip_empty_lines"),
            (b'{"text": "line1\\nline2"}\n{"text": "another"}\n', list, 2, "data.jsonl", "jsonl_embedded_newlines"),
            (b'{"id": 1}\ninvalid\n{"id": 2}\n{broken\n{"id": 3}\n', list, 3, "data.jsonl", "jsonl_mixed_valid_invalid"),
            (b'{}', dict, None, "data.json", "json_empty_object"),
            (b'[{"a": 1}, {"b": 2}]', list, 2, "data.json", "json_array"),
            (b'\n  \n\t\n   \n', list, 0, "data.jsonl", "jsonl_only_empty_lines"),
        ]
    )
    def test_download_json_payload(
        self, mock_drive_credentials, mock_drive_service, content, expect_type, expect_len, filename, desc
    ):
        """Test JSON payload parsing with various formats."""
        client = DriveClient(
            credentials_path=mock_drive_credentials["credentials_path"],
            token_path=mock_drive_credentials["token_path"],
        )
        client._service = mock_drive_service["service"]

        with patch.object(client, "download_bytes", return_value=content):
            result = client.download_json_payload("file-1", name=filename)

        assert isinstance(result, expect_type), f"Failed for {desc}"
        if expect_len is not None:
            assert len(result) == expect_len, f"Failed for {desc}"

    def test_case_insensitive_extension_matching(self, mock_drive_credentials, mock_drive_service):
        """Extension matching should be case-insensitive."""
        client = DriveClient(
            credentials_path=mock_drive_credentials["credentials_path"],
            token_path=mock_drive_credentials["token_path"],
        )
        client._service = mock_drive_service["service"]

        jsonl_content = b'{"a": 1}\n{"b": 2}\n'

        with patch.object(client, "download_bytes", return_value=jsonl_content):
            result = client.download_json_payload("file-1", name="DATA.JSONL")

        assert isinstance(result, list)
        assert len(result) == 2

    def test_json_fallback_on_decode_error(self, mock_drive_credentials, mock_drive_service):
        """JSON decode should fall back to UTF-8 replacement on error."""
        client = DriveClient(
            credentials_path=mock_drive_credentials["credentials_path"],
            token_path=mock_drive_credentials["token_path"],
        )
        client._service = mock_drive_service["service"]

        json_content = b'{"title": "Test"}'

        with patch.object(client, "download_bytes", return_value=json_content):
            result = client.download_json_payload("file-1", name="test.json")

        assert isinstance(result, dict)
        assert result["title"] == "Test"
