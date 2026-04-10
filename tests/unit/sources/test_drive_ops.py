"""Focused contracts for Drive ingestion helpers and attachment handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

from polylogue.config import Source
from polylogue.sources import DriveFile, download_drive_files, iter_drive_conversations
from polylogue.sources.drive import _apply_drive_attachments
from polylogue.sources.parsers.base import ParsedAttachment, ParsedConversation


def _attachment(
    provider_attachment_id: str | None,
    *,
    name: str | None = None,
    mime_type: str | None = None,
    size_bytes: int | None = None,
    provider_meta: dict[str, object] | None = None,
) -> ParsedAttachment:
    return ParsedAttachment(
        provider_attachment_id=provider_attachment_id,
        message_provider_id="msg-1",
        name=name,
        mime_type=mime_type,
        size_bytes=size_bytes,
        path=None,
        provider_meta=provider_meta,
    )


def _conversation(*attachments: ParsedAttachment) -> ParsedConversation:
    return ParsedConversation(
        provider_name="gemini",
        provider_conversation_id="conv-1",
        messages=[],
        attachments=list(attachments),
    )


@dataclass
class _DriveConversationClient:
    files: list[DriveFile]
    payloads: dict[str, object]
    payload_failures: dict[str, Exception] | None = None
    attachment_meta: dict[str, DriveFile] | None = None

    def __post_init__(self) -> None:
        self.payload_failures = self.payload_failures or {}
        self.attachment_meta = self.attachment_meta or {}
        self.download_to_path_calls: list[tuple[str, Path]] = []

    def resolve_folder_id(self, folder_ref: str) -> str:
        return f"folder:{folder_ref}"

    def iter_json_files(self, folder_id: str):
        yield from self.files

    def download_json_payload(self, file_id: str, *, name: str):
        if file_id in self.payload_failures:
            raise self.payload_failures[file_id]
        return self.payloads[file_id]

    def download_to_path(self, file_id: str, dest: Path) -> DriveFile:
        self.download_to_path_calls.append((file_id, dest))
        return self.attachment_meta.get(
            file_id,
            DriveFile(
                file_id=file_id,
                name=dest.name,
                mime_type="application/octet-stream",
                modified_time=None,
                size_bytes=None,
            ),
        )


def test_download_drive_files_contract(tmp_path: Path) -> None:
    client = MagicMock()
    client.iter_json_files.return_value = [
        DriveFile("good", "session", "application/json", None, None),
        DriveFile("bad", "broken.jsonl", "application/json", None, None),
    ]

    def download(file_id: str, dest: Path) -> None:
        if file_id == "bad":
            raise PermissionError("denied")
        dest.write_bytes(b'{"id":"good"}')

    client.download_to_path.side_effect = download

    result = download_drive_files(client, "folder-1", tmp_path)

    assert result.total_files == 2
    assert [path.name for path in result.downloaded_files] == ["session.json"]
    assert result.downloaded_files[0].read_bytes() == b'{"id":"good"}'
    assert result.failed_files == [{"file_id": "bad", "name": "broken.jsonl", "error": "denied"}]


def test_apply_drive_attachments_contract(tmp_path: Path) -> None:
    missing_attachment = _attachment("placeholder")
    missing_attachment.provider_attachment_id = None
    conversation = _conversation(
        _attachment("keep-meta", name="keep.txt", mime_type="text/plain", size_bytes=12),
        _attachment("fill-meta"),
        missing_attachment,
    )
    client = MagicMock()
    client.download_to_path.side_effect = [
        DriveFile("keep-meta", "remote.bin", "application/octet-stream", None, 99),
        DriveFile("fill-meta", "filled.pdf", "application/pdf", None, 2048),
    ]

    _apply_drive_attachments(
        convo=conversation,
        client=client,
        archive_root=tmp_path,
        download_assets=True,
    )

    keep, filled, missing = conversation.attachments

    assert keep.name == "keep.txt"
    assert keep.mime_type == "text/plain"
    assert keep.size_bytes == 12
    assert keep.provider_meta == {
        "drive_id": "keep-meta",
        "name": "keep.txt",
        "mime_type": "text/plain",
    }

    assert filled.name == "filled.pdf"
    assert filled.mime_type == "application/pdf"
    assert filled.size_bytes == 2048
    assert filled.path is not None
    assert filled.provider_meta == {
        "drive_id": "fill-meta",
        "name": "filled.pdf",
        "mime_type": "application/pdf",
    }

    assert missing.path is None
    assert missing.provider_meta is None
    assert [call.args[0] for call in client.download_to_path.call_args_list] == ["keep-meta", "fill-meta"]


def test_apply_drive_attachments_skips_inline_and_external_media(tmp_path: Path) -> None:
    conversation = _conversation(
        _attachment(
            "inline-file-1",
            mime_type="text/plain",
            provider_meta={"attachment_kind": "inline_file"},
        ),
        _attachment(
            "youtube-video-1",
            mime_type="video/youtube",
            provider_meta={"attachment_kind": "youtube_video"},
        ),
    )
    client = MagicMock()

    _apply_drive_attachments(
        convo=conversation,
        client=client,
        archive_root=tmp_path,
        download_assets=True,
    )

    client.download_to_path.assert_not_called()
    assert all(attachment.path is None for attachment in conversation.attachments)


def test_iter_drive_conversations_returns_empty_without_folder(tmp_path: Path) -> None:
    assert (
        list(
            iter_drive_conversations(
                source=Source(name="gemini", path=tmp_path),
                archive_root=tmp_path,
                download_assets=False,
            )
        )
        == []
    )


def test_iter_drive_conversations_tracks_cursor_and_attachment_downloads(tmp_path: Path) -> None:
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
    client = _DriveConversationClient(
        files=[
            DriveFile("file-1", "chat.json", "application/json", "2025-01-01T00:00:00Z", 10),
            DriveFile("file-2", "newer.json", "application/json", "2025-01-01T00:05:00Z", 10),
        ],
        payloads={"file-1": payload, "file-2": payload},
        attachment_meta={
            "att-1": DriveFile("att-1", "doc.txt", "text/plain", None, 7),
        },
    )
    cursor_state: dict[str, object] = {}

    conversations = list(
        iter_drive_conversations(
            source=Source(name="gemini", folder="Google AI Studio"),
            archive_root=tmp_path,
            client=client,
            cursor_state=cursor_state,
            download_assets=True,
        )
    )

    assert len(conversations) == 2
    assert cursor_state["file_count"] == 2
    assert cursor_state["latest_file_id"] == "file-2"
    assert cursor_state["latest_file_name"] == "newer.json"
    assert conversations[0].attachments[0].provider_attachment_id == "att-1"
    assert conversations[0].attachments[0].path is not None
    assert [file_id for file_id, _ in client.download_to_path_calls] == ["att-1", "att-1"]


def test_iter_drive_conversations_tracks_payload_failures_and_continues(tmp_path: Path) -> None:
    good_payload = {"chunkedPrompt": {"chunks": [{"role": "user", "text": "ok"}]}}
    client = _DriveConversationClient(
        files=[
            DriveFile("good", "good.json", "application/json", None, None),
            DriveFile("bad", "bad.json", "application/json", None, None),
        ],
        payloads={"good": good_payload},
        payload_failures={"bad": RuntimeError("download failed")},
    )
    cursor_state: dict[str, object] = {}

    conversations = list(
        iter_drive_conversations(
            source=Source(name="gemini", folder="Google AI Studio"),
            archive_root=tmp_path,
            client=client,
            cursor_state=cursor_state,
            download_assets=False,
        )
    )

    assert len(conversations) == 1
    assert cursor_state["error_count"] == 1
    assert cursor_state["latest_error_file"] == "bad.json"
    assert "download failed" in str(cursor_state["latest_error"])
    assert client.download_to_path_calls == []


def test_iter_drive_conversations_uses_injected_client_without_recreating(tmp_path: Path) -> None:
    client = _DriveConversationClient(files=[], payloads={})

    with patch("polylogue.sources.drive.build_drive_source_client") as drive_client_factory:
        list(
            iter_drive_conversations(
                source=Source(name="gemini", folder="Google AI Studio"),
                archive_root=tmp_path,
                client=client,
                download_assets=False,
            )
        )

    drive_client_factory.assert_not_called()
