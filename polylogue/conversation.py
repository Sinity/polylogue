from __future__ import annotations

import os
import shutil
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .branching import BranchPlan, BranchInfo, MessageRecord, build_branch_plan
from .importers.base import ImportResult
from .render import AttachmentInfo, MarkdownDocument, build_markdown_from_chunks
from .repository import ConversationRepository
from .services.conversation_registrar import ConversationRegistrar
from .services.attachment_text import (
    AttachmentText,
    MAX_BYTES_DEFAULT,
    MAX_CHARS_DEFAULT,
    extract_attachment_text,
)


def _compute_branch_stats(
    plan: BranchPlan,
    records_by_id: Dict[str, MessageRecord],
) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = {}
    for branch_id, info in plan.branches.items():
        branch_records = [records_by_id[mid] for mid in info.message_ids if mid in records_by_id]
        if not branch_records:
            stats[branch_id] = {
                "message_count": 0,
                "token_count": 0,
                "word_count": 0,
                "first_message_id": None,
                "last_message_id": None,
                "first_timestamp": None,
                "last_timestamp": None,
            }
            continue
        stats[branch_id] = {
            "message_count": len(branch_records),
            "token_count": sum(rec.token_count for rec in branch_records),
            "word_count": sum(rec.word_count for rec in branch_records),
            "first_message_id": branch_records[0].message_id,
            "last_message_id": branch_records[-1].message_id,
            "first_timestamp": branch_records[0].timestamp,
            "last_timestamp": branch_records[-1].timestamp,
        }
    return stats


def _branch_map_snippet(branch_dir: Path, branch_ids: List[str]) -> Optional[str]:
    if not branch_ids:
        return None
    lines: List[str] = []
    for branch_id in sorted(branch_ids):
        branch_file = f"./branches/{branch_id}/{branch_id}.md"
        overlay_file = f"./branches/{branch_id}/overlay.md"
        line = f"- [{branch_id}]({branch_file}) (overlay: [overlay.md]({overlay_file}))"
        lines.append(line)
    if not lines:
        return None
    return "## Branch map\n\n" + "\n".join(lines) + "\n"


def _links_mapping(records: List[MessageRecord]) -> Dict[int, List[Tuple[str, object]]]:
    mapping: Dict[int, List[Tuple[str, object]]] = {}
    for idx, record in enumerate(records):
        if record.links:
            mapping[idx] = list(record.links)
    return mapping


def _adjust_links_for_base(
    mapping: Dict[int, List[Tuple[str, object]]],
    base_path: Path,
    markdown_dir: Path,
) -> Dict[int, List[Tuple[str, object]]]:
    adjusted: Dict[int, List[Tuple[str, object]]] = {}
    for idx, links in mapping.items():
        adjusted_links: List[Tuple[str, object]] = []
        for name, link in links:
            if isinstance(link, Path):
                try:
                    if link.is_absolute():
                        rel = link
                    else:
                        rel = Path(os.path.relpath((markdown_dir / link).resolve(), base_path.resolve()))
                    adjusted_links.append((name, rel))
                except Exception:
                    adjusted_links.append((name, link))
            else:
                adjusted_links.append((name, link))
        adjusted[idx] = adjusted_links
    return adjusted


def _hash_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _attachment_rows(
    *,
    attachments: List[AttachmentInfo],
    plan: BranchPlan,
    records_by_id: Dict[str, MessageRecord],
    conversation_dir: Path,
    attachment_ocr: bool,
    max_bytes: int = MAX_BYTES_DEFAULT,
    max_chars: int = MAX_CHARS_DEFAULT,
) -> List[Dict[str, object]]:
    if not attachments:
        return []

    info_by_key: Dict[Tuple[str, str], AttachmentInfo] = {}
    info_by_name: Dict[str, AttachmentInfo] = {}
    for info in attachments:
        if info.remote or info.local_path is None:
            continue
        key = (info.name, str(info.link))
        info_by_key[key] = info
        info_by_name.setdefault(info.name, info)

    rows: List[Dict[str, object]] = []
    seen: Set[Tuple[str, str, str]] = set()

    for branch_id, info in plan.branches.items():
        for message_id in info.message_ids:
            record = records_by_id.get(message_id)
            if record is None or not record.links:
                continue
            for name, link in record.links:
                key = (name, str(link))
                att = info_by_key.get(key) or info_by_name.get(name)
                if att is None or att.local_path is None:
                    continue
                dedup_key = (branch_id, message_id, name)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                attachment_path = (conversation_dir / att.local_path).resolve()
                if not attachment_path.exists():
                    continue

                try:
                    text = extract_attachment_text(
                        attachment_path,
                        ocr=attachment_ocr,
                        max_bytes=max_bytes,
                        max_chars=max_chars,
                    )
                except Exception:
                    text = AttachmentText(
                        text=None,
                        mime=None,
                        truncated=False,
                        ocr_used=False,
                        size_bytes=attachment_path.stat().st_size if attachment_path.exists() else 0,
                    )

                size_bytes = att.size_bytes
                if size_bytes is None:
                    try:
                        size_bytes = attachment_path.stat().st_size
                    except OSError:
                        size_bytes = None

                rows.append(
                    {
                        "branch_id": branch_id,
                        "message_id": message_id,
                        "attachment_name": name,
                        "attachment_path": str(attachment_path),
                        "size_bytes": size_bytes,
                        "content_hash": _hash_file(attachment_path),
                        "mime_type": text.mime,
                        "text_content": text.text,
                        "text_bytes": len(text.text) if text.text else 0,
                        "truncated": text.truncated,
                        "ocr_used": text.ocr_used,
                    }
                )

    return rows


def _render_markdown_document(
    records: List[MessageRecord],
    *,
    title: str,
    source_file_id: Optional[str],
    modified_time: Optional[str],
    created_time: Optional[str],
    run_settings: Optional[Dict[str, object]],
    source_mime: Optional[str],
    source_size: Optional[int],
    collapse_threshold: int,
    collapse_thresholds: Optional[Dict[str, int]] = None,
    extra_yaml: Optional[Dict[str, object]],
    attachments: Optional[List[AttachmentInfo]],
    per_chunk_links: Dict[int, List[Tuple[str, object]]],
    citations: Optional[List[Any]] = None,
) -> MarkdownDocument:
    chunks = [rec.chunk for rec in records]
    return build_markdown_from_chunks(
        chunks,
        per_chunk_links,
        title=title,
        source_file_id=source_file_id,
        modified_time=modified_time,
        created_time=created_time,
        run_settings=run_settings,
        citations=citations,
        source_mime=source_mime,
        source_size=source_size,
        collapse_threshold=collapse_threshold,
        collapse_thresholds=collapse_thresholds,
        extra_yaml=extra_yaml,
        attachments=attachments,
    )


def _common_prefix(plan: BranchPlan) -> List[str]:
    if not plan.branches:
        return []
    paths = [info.message_ids for info in plan.branches.values() if info.message_ids]
    if not paths:
        return []
    min_length = min(len(path) for path in paths)
    prefix: List[str] = []
    for idx in range(min_length):
        candidate = paths[0][idx]
        if all(path[idx] == candidate for path in paths):
            prefix.append(candidate)
        else:
            break
    return prefix


def process_conversation(
    *,
    provider: str,
    conversation_id: str,
    slug: str,
    title: str,
    message_records: List[MessageRecord],
    attachments: List[AttachmentInfo],
    canonical_leaf_id: Optional[str],
    collapse_threshold: int,
    collapse_thresholds: Optional[Dict[str, int]] = None,
    html: bool,
    html_theme: str,
    output_dir: Path,
    extra_yaml: Optional[Dict[str, object]],
    extra_state: Optional[Dict[str, object]],
    source_file_id: Optional[str],
    modified_time: Optional[str],
    created_time: Optional[str],
    run_settings: Optional[Dict[str, object]],
    source_mime: Optional[str],
    source_size: Optional[int],
    attachment_policy,
    force: bool,
    allow_dirty: bool = False,
    attachment_ocr: bool = False,
    registrar: Optional[ConversationRegistrar] = None,
    repository: Optional[ConversationRepository] = None,
    citations: Optional[List[Any]] = None,
) -> ImportResult:
    """Process a conversation and write to database only.

    Database-first architecture: This function writes conversation data
    to the SQLite database. Markdown files are generated separately by
    the DatabaseRenderer.

    Args:
        provider: Provider name (chatgpt, claude, etc.)
        conversation_id: Provider's conversation ID
        slug: URL-safe conversation identifier
        title: Conversation title
        message_records: List of message records
        attachments: List of attachment info
        canonical_leaf_id: ID of the canonical leaf message
        collapse_threshold: Threshold for collapsing long content
        collapse_thresholds: Per-type collapse thresholds
        html: Whether to generate HTML (ignored, kept for compatibility)
        html_theme: HTML theme (ignored, kept for compatibility)
        output_dir: Output directory (used for slug path only)
        extra_yaml: Extra YAML metadata
        extra_state: Extra state metadata
        source_file_id: Source file ID
        modified_time: Modification timestamp
        created_time: Creation timestamp
        run_settings: Run settings
        source_mime: Source MIME type
        source_size: Source size in bytes
        attachment_policy: Attachment policy (ignored)
        force: Force flag (ignored)
        allow_dirty: Allow dirty flag (ignored)
        attachment_ocr: Whether to OCR attachments
        registrar: Conversation registrar
        repository: Conversation repository (ignored)
        citations: Citation list

    Returns:
        ImportResult with database status
    """
    if registrar is None:
        raise ValueError("ConversationRegistrar instance required")

    # Build conversation structure
    records_by_id = {record.message_id: record for record in message_records}
    plan = build_branch_plan(message_records, canonical_leaf_id=canonical_leaf_id)
    branch_stats = _compute_branch_stats(plan, records_by_id)
    attachment_bytes = sum(att.size_bytes or 0 for att in attachments)

    # Write to database via registrar
    registrar.record_branch_plan(
        provider=provider,
        conversation_id=conversation_id,
        slug=slug,
        title=title,
        plan=plan,
        branch_stats=branch_stats,
        records_by_id=records_by_id,
        attachment_bytes=attachment_bytes,
    )

    # Create minimal attachment rows (no file access needed for DB-only)
    attachment_rows = []
    for att in attachments:
        if att.local_path or att.size_bytes:
            attachment_rows.append({
                "branch_id": plan.canonical_branch_id,
                "message_id": None,  # Could be extracted from message_records if needed
                "attachment_name": att.name,
                "attachment_path": str(att.local_path) if att.local_path else None,
                "size_bytes": att.size_bytes,
                "content_hash": None,
                "mime_type": None,
                "text_content": None,
                "text_bytes": None,
                "truncated": False,
                "ocr_used": False,
            })

    registrar.record_attachments(
        provider=provider,
        conversation_id=conversation_id,
        attachments=attachment_rows,
    )

    # Return minimal result indicating DB-only write
    return ImportResult(
        markdown_path=output_dir / slug / "conversation.md",  # Path that would be created
        html_path=None,
        attachments_dir=output_dir / slug / "attachments" if attachments else None,
        document=None,  # No document generated in DB-only mode
        slug=slug,
        diff_path=None,
        skipped=False,
        skip_reason=None,
        dirty=False,
        content_hash=None,
        branch_count=len(plan.branches),
        canonical_branch_id=plan.canonical_branch_id,
        branch_directories=None,
    )
