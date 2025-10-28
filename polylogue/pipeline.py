from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from .drive_client import DriveClient
from .render import (
    AttachmentInfo,
    MarkdownDocument,
    build_markdown_from_chunks,
    per_chunk_remote_links,
    remote_attachment_info,
)
from .util import sanitize_filename

PerChunkLink = Tuple[str, Union[Path, str]]
PerIndexLinks = Dict[int, List[PerChunkLink]]


@dataclass
class ChatContext:
    title: str
    chat_id: Optional[str]
    modified_time: Optional[str]
    created_time: Optional[str]
    run_settings: Optional[Any]
    citations: Optional[Any]
    source_mime: Optional[str]


def build_document_from_chunks(
    chunks: List[Dict],
    context: ChatContext,
    md_path: Path,
    *,
    collapse_threshold: int,
    download_attachments: bool,
    drive: Optional[DriveClient],
    force: bool,
    dry_run: bool,
) -> MarkdownDocument:
    if download_attachments and drive is not None:
        per_index_links, attachments = collect_attachments(
            drive,
            chunks,
            md_path,
            force=force,
            dry_run=dry_run,
        )
    else:
        per_index_links = cast(PerIndexLinks, per_chunk_remote_links(chunks))
        attachments = remote_attachment_info(per_index_links)

    return build_markdown_from_chunks(
        chunks,
        per_index_links,
        context.title,
        context.chat_id,
        context.modified_time,
        context.created_time,
        run_settings=context.run_settings,
        citations=context.citations,
        source_mime=context.source_mime,
        collapse_threshold=collapse_threshold,
        attachments=attachments,
    )


def collect_attachments(
    drive: DriveClient,
    chunks: List[Dict],
    md_path: Path,
    *,
    force: bool,
    dry_run: bool,
) -> Tuple[PerIndexLinks, List[AttachmentInfo]]:
    per_index_links: PerIndexLinks = {}
    attachments: List[AttachmentInfo] = []
    seen: set[Tuple[str, str]] = set()
    markdown_root = md_path.parent
    attachments_dir = markdown_root / "attachments"
    if not dry_run:
        attachments_dir.mkdir(parents=True, exist_ok=True)
    for idx, chunk in enumerate(chunks):
        ids = collect_remote_ids(chunk)
        if not ids:
            continue
        links: List[PerChunkLink] = []
        per_index_links[idx] = links
        for att_id in ids:
            meta_att = drive.attachment_meta(att_id)
            if not meta_att:
                continue
            fname = sanitize_filename(meta_att.get("name", att_id))
            local_path = attachments_dir / fname
            if not dry_run:
                if force or not local_path.exists():
                    ok = drive.download_attachment(att_id, local_path)
                    if not ok:
                        continue
                drive.touch_mtime(local_path, meta_att.get("modifiedTime"))
            try:
                rel = local_path.relative_to(markdown_root)
            except Exception:
                rel = local_path
            links.append((fname, rel))
            size = None
            if not dry_run and local_path.exists():
                try:
                    size = local_path.stat().st_size
                except OSError:
                    size = None
            key = (fname, att_id)
            if key not in seen:
                seen.add(key)
                attachments.append(
                    AttachmentInfo(
                        name=fname,
                        link=str(rel),
                        local_path=None if dry_run else rel,
                        size_bytes=size,
                        remote=False,
                    )
                )
    return per_index_links, attachments


def collect_remote_ids(chunk: Dict) -> List[str]:
    from .render import extract_drive_ids

    ids = extract_drive_ids(chunk)
    # Preserve order but drop duplicates for this chunk.
    seen: set[str] = set()
    unique: List[str] = []
    for item in ids:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique
