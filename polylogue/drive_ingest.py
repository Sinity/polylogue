from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .assets import asset_path
from .config import Profile, Source
from .drive_client import DriveClient, drive_link
from .source_ingest import ParsedConversation, parse_drive_payload


@dataclass
class DriveIngestResult:
    conversations: int
    attachments: int


def _apply_attachment_policy(
    *,
    convo: ParsedConversation,
    policy: str,
    client: DriveClient,
    archive_root: Path,
    download_assets: bool,
) -> None:
    if policy == "skip":
        convo.attachments = []
        return
    for attachment in convo.attachments:
        if not attachment.provider_attachment_id:
            continue
        if policy == "link":
            attachment.path = drive_link(attachment.provider_attachment_id)
            continue
        if policy != "download":
            continue
        if not download_assets:
            continue
        dest = asset_path(archive_root, attachment.provider_attachment_id)
        meta = client.download_to_path(attachment.provider_attachment_id, dest)
        attachment.path = str(dest)
        attachment.name = attachment.name or meta.name
        attachment.mime_type = attachment.mime_type or meta.mime_type
        attachment.size_bytes = attachment.size_bytes or meta.size_bytes
        meta_payload = dict(attachment.provider_meta or {})
        meta_payload.update(
            {
                "drive_id": attachment.provider_attachment_id,
                "name": attachment.name,
                "mime_type": attachment.mime_type,
            }
        )
        attachment.provider_meta = meta_payload


def iter_drive_conversations(
    *,
    source: Source,
    profile: Profile,
    archive_root: Path,
    ui: Optional[object] = None,
    client: Optional[DriveClient] = None,
    download_assets: bool = True,
) -> Iterable[ParsedConversation]:
    if not source.folder:
        return
    drive_client = client or DriveClient(ui=ui)
    folder_id = drive_client.resolve_folder_id(source.folder)
    for file_meta in drive_client.iter_json_files(folder_id):
        try:
            payload = drive_client.download_json_payload(file_meta.file_id, name=file_meta.name)
        except Exception:
            continue
        conversations = parse_drive_payload(source.name, payload, file_meta.file_id)
        for convo in conversations:
            _apply_attachment_policy(
                convo=convo,
                policy=profile.attachments,
                client=drive_client,
                archive_root=archive_root,
                download_assets=download_assets,
            )
            yield convo


__all__ = ["iter_drive_conversations", "DriveIngestResult"]
