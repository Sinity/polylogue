from __future__ import annotations

from dataclasses import dataclass

from polylogue.config import Source
from polylogue.drive_client import DriveFile
from polylogue.drive_ingest import iter_drive_conversations


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
