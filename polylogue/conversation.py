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
    if registrar is None:
        raise ValueError("ConversationRegistrar instance required")

    if repository is None:
        repository = ConversationRepository()

    write_branch_tree = True
    write_full_branch_docs = True
    write_overlay_docs = True

    records_by_id = {record.message_id: record for record in message_records}
    plan = build_branch_plan(message_records, canonical_leaf_id=canonical_leaf_id)
    branch_stats = _compute_branch_stats(plan, records_by_id)

    attachment_bytes = sum(att.size_bytes or 0 for att in attachments)

    canonical_records = [records_by_id[mid] for mid in plan.messages_for_branch(plan.canonical_branch_id) if mid in records_by_id]
    canonical_links = _links_mapping(canonical_records)
    canonical_document = _render_markdown_document(
        canonical_records,
        title=title,
        source_file_id=source_file_id,
        modified_time=modified_time,
        created_time=created_time,
        run_settings=run_settings,
        source_mime=source_mime,
        source_size=source_size,
        collapse_threshold=collapse_threshold,
        collapse_thresholds=collapse_thresholds,
        extra_yaml=extra_yaml,
        attachments=attachments,
        per_chunk_links=canonical_links,
        citations=citations,
    )

    branch_map = _branch_map_snippet(output_dir / slug / "branches", plan.branch_ids)
    if branch_map and "Branch map" not in canonical_document.body:
        canonical_document.body = branch_map + "\n" + canonical_document.body

    persist_result = repository.persist(
        provider=provider,
        conversation_id=conversation_id,
        title=title,
        document=canonical_document,
        output_dir=output_dir,
        collapse_threshold=collapse_threshold,
        attachments=attachments,
        updated_at=modified_time,
        created_at=created_time,
        html=html,
        html_theme=html_theme,
        attachment_policy=attachment_policy,
        extra_state=extra_state,
        slug_hint=slug,
        id_hint=conversation_id[:8] if conversation_id else None,
        force=force,
        allow_dirty=allow_dirty,
        registrar=registrar,
    )

    conversation_dir: Path = persist_result.markdown_path.parent
    branch_directories: List[Path] = []
    if not persist_result.skipped and persist_result.markdown_path:
        branches_dir = conversation_dir / "branches"
        attachments_dir = persist_result.attachments_dir

        canonical_path = persist_result.markdown_path
        canonical_path.write_text(
            (persist_result.document or canonical_document).to_markdown(),
            encoding="utf-8",
        )

        common_path = conversation_dir / "conversation.common.md"
        if not write_branch_tree:
            if common_path.exists():
                try:
                    common_path.unlink()
                except OSError:
                    pass
            if branches_dir.exists():
                try:
                    shutil.rmtree(branches_dir)
                except OSError:
                    pass
        else:
            branches_dir.mkdir(parents=True, exist_ok=True)

            common_ids = _common_prefix(plan)
            common_records = [records_by_id[mid] for mid in common_ids if mid in records_by_id]
            if common_records:
                common_links = _links_mapping(common_records)
                common_doc = _render_markdown_document(
                    common_records,
                    title=f"{title} — Common Prefix",
                    source_file_id=source_file_id,
                    modified_time=modified_time,
                    created_time=created_time,
                    run_settings=run_settings,
                    source_mime=source_mime,
                    source_size=source_size,
                    collapse_threshold=collapse_threshold,
                    collapse_thresholds=collapse_thresholds,
                    extra_yaml=extra_yaml,
                    attachments=None,
                    per_chunk_links=_adjust_links_for_base(common_links, conversation_dir, persist_result.markdown_path.parent),
                )
                common_path.write_text(common_doc.to_markdown(), encoding="utf-8")
            elif common_path.exists():
                try:
                    common_path.unlink()
                except OSError:
                    pass

            for branch_id, info in plan.branches.items():
                branch_records = [records_by_id[mid] for mid in info.message_ids if mid in records_by_id]
                branch_dir = branches_dir / branch_id
                branch_dir.mkdir(parents=True, exist_ok=True)
                branch_directories.append(branch_dir)

                branch_links = _links_mapping(branch_records)
                branch_doc_path = branch_dir / f"{branch_id}.md"
                if write_full_branch_docs and branch_records:
                    branch_doc = _render_markdown_document(
                        branch_records,
                        title=f"{title} — {branch_id}",
                        source_file_id=source_file_id,
                        modified_time=modified_time,
                        created_time=created_time,
                        run_settings=run_settings,
                        source_mime=source_mime,
                        source_size=source_size,
                        collapse_threshold=collapse_threshold,
                        collapse_thresholds=collapse_thresholds,
                        extra_yaml=extra_yaml,
                        attachments=None,
                        per_chunk_links=_adjust_links_for_base(
                            branch_links, branch_dir, persist_result.markdown_path.parent
                        ),
                    )
                    branch_doc_path.write_text(branch_doc.to_markdown(), encoding="utf-8")
                elif branch_doc_path.exists():
                    try:
                        branch_doc_path.unlink()
                    except OSError:
                        pass

                overlay_records = branch_records[info.divergence_index :] if info.divergence_index else branch_records
                overlay_path = branch_dir / "overlay.md"
                if write_overlay_docs and overlay_records:
                    overlay_links = _links_mapping(overlay_records)
                    overlay_doc = _render_markdown_document(
                        overlay_records,
                        title=f"{title} — {branch_id} Overlay",
                        source_file_id=source_file_id,
                        modified_time=modified_time,
                        created_time=created_time,
                        run_settings=run_settings,
                        source_mime=source_mime,
                        source_size=source_size,
                        collapse_threshold=collapse_threshold,
                        collapse_thresholds=collapse_thresholds,
                        extra_yaml=extra_yaml,
                        attachments=None,
                        per_chunk_links=_adjust_links_for_base(
                            overlay_links, branch_dir, persist_result.markdown_path.parent
                        ),
                    )
                    overlay_path.write_text(overlay_doc.to_markdown(), encoding="utf-8")
                elif overlay_path.exists():
                    try:
                        overlay_path.unlink()
                    except OSError:
                        pass

                if attachments_dir and attachments_dir.exists():
                    (branch_dir / "attachments").mkdir(exist_ok=True)

            existing_branch_dirs = [path for path in branches_dir.iterdir() if path.is_dir()]
            active_branch_names = {branch_id for branch_id in plan.branches}
            for path in existing_branch_dirs:
                if path.name not in active_branch_names:
                    try:
                        shutil.rmtree(path)
                    except OSError:
                        continue

    registrar.record_branch_plan(
        provider=provider,
        conversation_id=conversation_id,
        slug=persist_result.slug,
        plan=plan,
        branch_stats=branch_stats,
        records_by_id=records_by_id,
        attachment_bytes=attachment_bytes,
    )
    attachment_rows = _attachment_rows(
        attachments=attachments,
        plan=plan,
        records_by_id=records_by_id,
        conversation_dir=conversation_dir,
        attachment_ocr=attachment_ocr,
    )
    registrar.record_attachments(
        provider=provider,
        conversation_id=conversation_id,
        attachments=attachment_rows,
    )

    return ImportResult(
        markdown_path=persist_result.markdown_path,
        html_path=persist_result.html_path,
        attachments_dir=persist_result.attachments_dir,
        document=persist_result.document or canonical_document,
        slug=persist_result.slug,
        diff_path=None,
        skipped=persist_result.skipped,
        skip_reason=persist_result.skip_reason,
        dirty=persist_result.dirty,
        content_hash=persist_result.content_hash,
        branch_count=len(plan.branches),
        canonical_branch_id=plan.canonical_branch_id,
        branch_directories=branch_directories if branch_directories else None,
    )
