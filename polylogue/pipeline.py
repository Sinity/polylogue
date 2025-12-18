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
from .branching import MessageRecord
from .util import sanitize_filename, parse_rfc3339_to_epoch

PerChunkLink = Tuple[str, Union[Path, str]]
PerIndexLinks = Dict[int, List[PerChunkLink]]


class AttachmentDownloadError(RuntimeError):
    def __init__(
        self,
        failed_ids: List[str],
        *,
        per_index_links: PerIndexLinks,
        attachments: List[AttachmentInfo],
    ) -> None:
        missing = ", ".join(failed_ids)
        super().__init__(f"Failed to download attachments: {missing}")
        self.failed_ids = list(failed_ids)
        self.per_index_links = per_index_links
        self.attachments = attachments


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
    collapse_thresholds: Optional[Dict[str, int]] = None,
    download_attachments: bool,
    drive: Optional[DriveClient],
    force: bool,
    dry_run: bool,
) -> MarkdownDocument:
    per_index_links, attachments = prepare_render_assets(
        chunks,
        md_path=md_path,
        download_attachments=download_attachments,
        drive=drive,
        force=force,
        dry_run=dry_run,
    )

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
        collapse_thresholds=collapse_thresholds,
        attachments=attachments,
    )


def prepare_render_assets(
    chunks: List[Dict],
    *,
    md_path: Path,
    download_attachments: bool,
    drive: Optional[DriveClient],
    force: bool,
    dry_run: bool,
) -> Tuple[PerIndexLinks, List[AttachmentInfo]]:
    if download_attachments:
        if drive is None:
            raise ValueError(
                "Drive client required when download_attachments is enabled. "
                "Initialize a DriveClient and pass it to the function, or set download_attachments=False."
            )
        return collect_attachments(
            drive,
            chunks,
            md_path,
            force=force,
            dry_run=dry_run,
        )

    per_index_links = cast(PerIndexLinks, per_chunk_remote_links(chunks))
    attachments = remote_attachment_info(per_index_links)
    return per_index_links, attachments


def collect_attachments(
    drive: DriveClient,
    chunks: List[Dict],
    md_path: Path,
    *,
    force: bool,
    dry_run: bool,
) -> Tuple[PerIndexLinks, List[AttachmentInfo]]:
    collector = _AttachmentCollector(md_path.parent, dry_run=dry_run)
    failures: List[str] = []
    for idx, chunk in enumerate(chunks):
        ids = collect_remote_ids(chunk)
        if not ids:
            continue
        links: List[PerChunkLink] = []
        collector.per_index_links[idx] = links
        for att_id in ids:
            meta_att = drive.attachment_meta(att_id)
            if not meta_att:
                continue
            fname = sanitize_filename(meta_att.get("name", att_id))
            local_path = collector.attachments_dir / fname
            if not dry_run:
                remote_epoch = parse_rfc3339_to_epoch(meta_att.get("modifiedTime"))
                needs_download = force or not local_path.exists()
                if not needs_download and remote_epoch is not None:
                    try:
                        local_epoch = int(local_path.stat().st_mtime)
                    except OSError:
                        needs_download = True
                    else:
                        if local_epoch != int(remote_epoch):
                            needs_download = True
                if needs_download:
                    ok = drive.download_attachment(att_id, local_path)
                    if not ok:
                        failures.append(att_id)
                        collector.add_local(idx, fname, att_id, local_path)
                        continue
                drive.touch_mtime(local_path, meta_att.get("modifiedTime"))
            collector.add_local(idx, fname, att_id, local_path)
    if failures and not dry_run:
        raise AttachmentDownloadError(
            failures,
            per_index_links=collector.per_index_links,
            attachments=collector.attachments,
        )
    return collector.per_index_links, collector.attachments


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


def _normalize_chunk_role(role: Optional[str]) -> str:
    if not role:
        return "model"
    role = role.lower()
    if role in {"user", "model", "assistant", "tool", "system"}:
        if role == "assistant":
            return "model"
        return role
    if role in {"function", "tool_call"}:
        return "tool"
    return "model"


def build_message_records_from_chunks(
    chunks: List[Dict],
    *,
    provider: str,
    conversation_id: Optional[str],
    slug: str,
    per_chunk_links: PerIndexLinks,
) -> List[MessageRecord]:
    records: List[MessageRecord] = []
    seen_ids: set[str] = set()

    for idx, chunk in enumerate(chunks):
        text = chunk.get("text") or ""
        message_id = (
            chunk.get("messageId")
            or chunk.get("id")
            or f"{slug}-msg-{idx:04d}"
        )
        while message_id in seen_ids:
            message_id = f"{message_id}-{idx}"
        seen_ids.add(message_id)

        parent_id = chunk.get("parentId") or chunk.get("branchParent")
        if not parent_id and records:
            parent_id = records[-1].message_id

        token_count = int(chunk.get("tokenCount", 0) or 0)
        word_count = len(text.split()) if isinstance(text, str) else 0
        links = list(per_chunk_links.get(idx, []))

        record = MessageRecord(
            message_id=message_id,
            parent_id=parent_id,
            role=_normalize_chunk_role(chunk.get("role")),
            text=text if isinstance(text, str) else str(text),
            token_count=token_count,
            word_count=word_count,
            timestamp=chunk.get("timestamp"),
            attachments=len(links),
            chunk=chunk,
            links=links,
            metadata={
                "source": provider,
                "conversation": conversation_id,
            },
        )
        records.append(record)

    return records


class _AttachmentCollector:
    def __init__(self, markdown_root: Path, *, dry_run: bool) -> None:
        self.markdown_root = markdown_root
        self.attachments_dir = markdown_root / "attachments"
        self.dry_run = dry_run
        self.per_index_links: PerIndexLinks = {}
        self.attachments: List[AttachmentInfo] = []
        self._seen: set[Tuple[str, str]] = set()
        if not dry_run:
            self.attachments_dir.mkdir(parents=True, exist_ok=True)

    def add_local(self, idx: int, name: str, att_id: str, local_path: Path) -> None:
        try:
            rel = local_path.relative_to(self.markdown_root)
        except Exception:
            rel = local_path
        self.per_index_links.setdefault(idx, []).append((name, rel))
        if (name, att_id) in self._seen:
            return
        self._seen.add((name, att_id))
        size = None
        if not self.dry_run and local_path.exists():
            try:
                size = local_path.stat().st_size
            except OSError:
                size = None
        self.attachments.append(
            AttachmentInfo(
                name=name,
                link=str(rel),
                local_path=None if self.dry_run else rel,
                size_bytes=size,
                remote=False,
            )
        )
