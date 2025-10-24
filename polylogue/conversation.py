from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .branching import BranchPlan, BranchInfo, MessageRecord, build_branch_plan
from .db import open_connection, replace_branches, replace_messages, upsert_conversation
from .importers.base import ImportResult
from .render import AttachmentInfo, MarkdownDocument, build_markdown_from_chunks
from .document_store import persist_document


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
                    rel = link if link.is_absolute() else link
                    if not rel.is_absolute():
                        rel = Path(os.path.relpath((markdown_dir / link).resolve(), base_path.resolve()))
                    adjusted_links.append((name, rel))
                except Exception:
                    adjusted_links.append((name, link))
            else:
                adjusted_links.append((name, link))
        adjusted[idx] = adjusted_links
    return adjusted


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
    extra_yaml: Optional[Dict[str, object]],
    attachments: Optional[List[AttachmentInfo]],
    per_chunk_links: Dict[int, List[Tuple[str, object]]],
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
        citations=None,
        source_mime=source_mime,
        source_size=source_size,
        collapse_threshold=collapse_threshold,
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
) -> ImportResult:
    records_by_id = {record.message_id: record for record in message_records}
    plan = build_branch_plan(message_records, canonical_leaf_id=canonical_leaf_id)
    branch_stats = _compute_branch_stats(plan, records_by_id)

    canonical_stats = branch_stats.get(plan.canonical_branch_id, {})
    total_tokens = canonical_stats.get("token_count", 0)
    total_words = canonical_stats.get("word_count", 0)

    with open_connection() as conn:
        upsert_conversation(
            conn,
            provider=provider,
            conversation_id=conversation_id,
            slug=slug,
            title=title,
            current_branch=plan.canonical_branch_id,
            last_updated=modified_time,
            content_hash=None,
            token_count=total_tokens,
            word_count=total_words,
            attachment_bytes=sum(att.size_bytes or 0 for att in attachments),
            dirty=False,
            metadata=None,
        )
        branch_payloads: List[Dict[str, object]] = []
        for branch_id, info in plan.branches.items():
            stats = branch_stats.get(branch_id, {})
            branch_payloads.append(
                {
                    "branch_id": branch_id,
                    "parent_branch_id": info.parent_branch_id,
                    "is_canonical": info.is_canonical,
                    "depth": info.depth,
                    "message_count": stats.get("message_count", 0),
                    "token_count": stats.get("token_count", 0),
                    "word_count": stats.get("word_count", 0),
                    "first_message_id": stats.get("first_message_id"),
                    "last_message_id": stats.get("last_message_id"),
                    "first_timestamp": stats.get("first_timestamp"),
                    "last_timestamp": stats.get("last_timestamp"),
                    "metadata": {"divergence_index": info.divergence_index},
                }
            )
        replace_branches(
            conn,
            provider=provider,
            conversation_id=conversation_id,
            branches=branch_payloads,
        )
        for branch_id, info in plan.branches.items():
            branch_records = [records_by_id[mid] for mid in info.message_ids if mid in records_by_id]
            messages = [
                {
                    "message_id": record.message_id,
                    "parent_id": record.parent_id,
                    "role": record.role,
                    "position": idx,
                    "timestamp": record.timestamp,
                    "token_count": record.token_count,
                    "word_count": record.word_count,
                    "attachment_count": record.attachments,
                    "body": record.text,
                    "metadata": record.metadata,
                }
                for idx, record in enumerate(branch_records)
            ]
            replace_messages(
                conn,
                provider=provider,
                conversation_id=conversation_id,
                branch_id=branch_id,
                messages=messages,
            )
        conn.commit()

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
        extra_yaml=extra_yaml,
        attachments=attachments,
        per_chunk_links=canonical_links,
    )

    persist_result = persist_document(
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
    )

    branch_directories: List[Path] = []
    if not persist_result.skipped and persist_result.markdown_path:
        conversation_dir = persist_result.markdown_path.parent / persist_result.slug
        branches_dir = conversation_dir / "branches"
        attachments_dir = persist_result.attachments_dir
        conversation_dir.mkdir(parents=True, exist_ok=True)
        branches_dir.mkdir(parents=True, exist_ok=True)

        canonical_path = conversation_dir / "conversation.md"
        canonical_path.write_text(persist_result.document.to_markdown() if persist_result.document else canonical_document.to_markdown(), encoding="utf-8")

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
                extra_yaml=extra_yaml,
                attachments=None,
                per_chunk_links=_adjust_links_for_base(common_links, conversation_dir, persist_result.markdown_path.parent),
            )
            (conversation_dir / "conversation.common.md").write_text(common_doc.to_markdown(), encoding="utf-8")

        for branch_id, info in plan.branches.items():
            branch_records = [records_by_id[mid] for mid in info.message_ids if mid in records_by_id]
            branch_dir = branches_dir / branch_id
            branch_dir.mkdir(parents=True, exist_ok=True)
            branch_directories.append(branch_dir)

            branch_links = _links_mapping(branch_records)
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
                extra_yaml=extra_yaml,
                attachments=None,
                per_chunk_links=_adjust_links_for_base(branch_links, branch_dir, persist_result.markdown_path.parent),
            )
            (branch_dir / f"{branch_id}.md").write_text(branch_doc.to_markdown(), encoding="utf-8")

            overlay_records = branch_records[info.divergence_index :] if info.divergence_index else branch_records
            if overlay_records:
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
                    extra_yaml=extra_yaml,
                    attachments=None,
                    per_chunk_links=_adjust_links_for_base(overlay_links, branch_dir, persist_result.markdown_path.parent),
                )
                (branch_dir / "overlay.md").write_text(overlay_doc.to_markdown(), encoding="utf-8")

            if attachments_dir and attachments_dir.exists():
                # Ensure attachments dir is preserved; branches can reference relative paths.
                (branch_dir / "attachments").mkdir(exist_ok=True)

    return ImportResult(
        markdown_path=persist_result.markdown_path,
        html_path=persist_result.html_path,
        attachments_dir=persist_result.attachments_dir,
        document=persist_result.document or canonical_document,
        diff_path=None,
        skipped=persist_result.skipped,
        skip_reason=persist_result.skip_reason,
        dirty=persist_result.dirty,
        content_hash=persist_result.content_hash,
        branch_count=len(plan.branches),
        canonical_branch_id=plan.canonical_branch_id,
        branch_directories=branch_directories if branch_directories else None,
    )
