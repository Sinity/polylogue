from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:  # pragma: no cover - optional dependency
    import frontmatter  # type: ignore
except Exception:  # pragma: no cover
    frontmatter = None  # type: ignore

from .render import AttachmentInfo, MarkdownDocument, build_markdown_from_chunks
from .util import (
    assign_conversation_slug,
    conversation_is_current,
    current_utc_timestamp,
    get_conversation_state,
    sanitize_filename,
    update_conversation_state,
)
from .index import update_index


@dataclass
class ExistingDocument:
    metadata: Dict[str, Any]
    body: str
    body_hash: str


@dataclass
class DocumentPersistenceResult:
    markdown_path: Path
    html_path: Optional[Path]
    attachments_dir: Optional[Path]
    document: Optional[MarkdownDocument]
    slug: str
    skipped: bool
    skip_reason: Optional[str]
    dirty: bool
    content_hash: Optional[str]


def _parse_front_matter(text: str) -> Tuple[Dict[str, Any], str]:
    if frontmatter is not None:  # pragma: no branch
        post = frontmatter.loads(text)
        return dict(post.metadata), post.content
    if not text.startswith("---\n"):
        return {}, text
    # Minimal YAML parser for key: value pairs.
    lines = text.splitlines()
    meta_lines: List[str] = []
    i = 1
    while i < len(lines):
        if lines[i].strip() == "---":
            i += 1
            break
        meta_lines.append(lines[i])
        i += 1
    body = "\n".join(lines[i:])
    metadata: Dict[str, Any] = {}
    for raw in meta_lines:
        parts = raw.split(":", 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        value = parts[1].strip()
        if value.startswith("\"") and value.endswith("\""):
            metadata[key] = value[1:-1]
        elif value.lower() in {"true", "false"}:
            metadata[key] = value.lower() == "true"
        else:
            try:
                metadata[key] = json.loads(value)
            except Exception:
                metadata[key] = value
    return metadata, body


def _dump_front_matter(metadata: Dict[str, Any], body: str) -> str:
    if frontmatter is not None:
        post = frontmatter.Post(body, **metadata)
        return frontmatter.dumps(post)
    header_lines = ["---"]
    for key, value in metadata.items():
        if isinstance(value, (dict, list)):
            encoded = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            encoded = "true" if value else "false"
        else:
            encoded = json.dumps(value)
        header_lines.append(f"{key}: {encoded}")
    header_lines.append("---\n")
    return "\n".join(header_lines) + body


def read_existing_document(path: Path) -> Optional[ExistingDocument]:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    metadata, body = _parse_front_matter(text)
    body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return ExistingDocument(metadata=metadata, body=body, body_hash=body_hash)


def _write_markdown(path: Path, metadata: Dict[str, Any], body: str) -> None:
    payload = _dump_front_matter(metadata, body)
    path.write_text(payload, encoding="utf-8")


def _default_attachment_policy() -> Dict[str, Any]:
    from .importers.utils import CHAR_THRESHOLD, LINE_THRESHOLD, PREVIEW_LINES

    return {
        "previewLines": PREVIEW_LINES,
        "lineThreshold": LINE_THRESHOLD,
        "charThreshold": CHAR_THRESHOLD,
    }


def _attachment_policy_with_counts(
    base_policy: Dict[str, Any],
    attachments: List[AttachmentInfo],
) -> Dict[str, Any]:
    extracted = sum(1 for info in attachments if not info.remote and info.local_path is not None)
    enriched = dict(base_policy)
    enriched["extractedCount"] = extracted
    return enriched


def prepare_document(
    *,
    chunks: List[Dict[str, Any]],
    per_chunk_links: Dict[int, List[Tuple[str, Union[Path, str]]]],
    title: str,
    collapse_threshold: int,
    attachments: Optional[List[AttachmentInfo]] = None,
    source_file_id: Optional[str] = None,
    modified_time: Optional[str] = None,
    created_time: Optional[str] = None,
    run_settings: Optional[Dict[str, Any]] = None,
    citations: Optional[List[Any]] = None,
    source_mime: Optional[str] = None,
    source_size: Optional[int] = None,
    extra_yaml: Optional[Dict[str, Any]] = None,
) -> MarkdownDocument:
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
        extra_yaml=extra_yaml,
        attachments=attachments,
    )


def _ensure_polylogue_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(metadata)
    polylogue = data.get("polylogue")
    if not isinstance(polylogue, dict):
        polylogue = {}
    data["polylogue"] = polylogue
    return data


def persist_document(
    *,
    provider: Optional[str],
    conversation_id: Optional[str],
    title: str,
    document: MarkdownDocument,
    output_dir: Path,
    collapse_threshold: int,
    attachments: List[AttachmentInfo],
    updated_at: Optional[str],
    created_at: Optional[str],
    html: bool,
    html_theme: Optional[str],
    attachment_policy: Optional[Dict[str, Any]] = None,
    extra_state: Optional[Dict[str, Any]] = None,
    slug_hint: Optional[str] = None,
    id_hint: Optional[str] = None,
    force: bool = False,
) -> DocumentPersistenceResult:
    if slug_hint:
        slug = slug_hint
    elif provider and conversation_id:
        slug = assign_conversation_slug(provider, conversation_id, title, id_hint=id_hint)
    else:
        base = sanitize_filename(slug_hint or title)
        slug = base or "document"

    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"{slug}.md"
    attachments_dir = output_dir / f"{slug}_attachments"
    html_path = markdown_path.with_suffix(".html") if html else None

    state_entry: Optional[Dict[str, Any]] = None
    if provider and conversation_id:
        state_entry = get_conversation_state(provider, conversation_id)

    existing = read_existing_document(markdown_path)
    stored_hash = (state_entry or {}).get("contentHash")
    existing_dirty = False
    if existing and stored_hash and existing.body_hash != stored_hash:
        existing_dirty = True

    policy = _attachment_policy_with_counts(
        attachment_policy or _default_attachment_policy(),
        attachments,
    )

    content_hash = hashlib.sha256(document.body.encode("utf-8")).hexdigest()

    if provider and conversation_id and existing_dirty and state_entry and not force:
        remote_same = True
        if updated_at and state_entry.get("lastUpdated") and state_entry["lastUpdated"] != updated_at:
            remote_same = False
        if state_entry.get("contentHash") and state_entry["contentHash"] != content_hash:
            remote_same = False
        if remote_same and existing:
            dirty_meta = _ensure_polylogue_metadata(existing.metadata)
            polylogue = dirty_meta.get("polylogue", {})
            polylogue.update(
                {
                    "title": polylogue.get("title") or title,
                    "slug": polylogue.get("slug") or slug,
                    "provider": provider,
                    "conversationId": conversation_id,
                    "collapseThreshold": collapse_threshold,
                    "attachmentPolicy": policy,
                    "updatedAt": updated_at,
                    "createdAt": created_at,
                    "html": html,
                    "attachmentsDir": polylogue.get("attachmentsDir") or (str(attachments_dir) if attachments else None),
                    "htmlPath": polylogue.get("htmlPath"),
                    "contentHash": state_entry.get("contentHash") or content_hash,
                    "localHash": existing.body_hash,
                    "dirty": True,
                }
            )
            dirty_meta["polylogue"] = polylogue
            _write_markdown(markdown_path, dirty_meta, existing.body)
            html_state_path = state_entry.get("htmlPath")
            html_existing_path = Path(html_state_path) if isinstance(html_state_path, str) else None
            attach_state_path = state_entry.get("attachmentsDir")
            attachments_existing_path = (
                Path(attach_state_path) if isinstance(attach_state_path, str) else None
            )
            update_conversation_state(
                provider,
                conversation_id,
                slug=slug,
                title=title,
                lastUpdated=updated_at,
                contentHash=state_entry.get("contentHash") or content_hash,
                collapseThreshold=collapse_threshold,
                attachmentPolicy=policy,
                outputPath=str(markdown_path),
                htmlPath=html_state_path,
                attachmentsDir=attach_state_path,
                html=state_entry.get("html", html),
                dirty=True,
                localHash=existing.body_hash,
            )
            return DocumentPersistenceResult(
                markdown_path=markdown_path,
                html_path=html_existing_path,
                attachments_dir=attachments_existing_path,
                document=None,
                slug=slug,
                skipped=True,
                skip_reason="dirty-local",
                dirty=True,
                content_hash=state_entry.get("contentHash") or content_hash,
            )

    can_skip = False
    if provider and conversation_id and not force:
        can_skip = conversation_is_current(
            provider,
            conversation_id,
            updated_at=updated_at,
            content_hash=content_hash,
            output_path=markdown_path,
            collapse_threshold=collapse_threshold,
            attachment_policy=policy,
            html=html,
            dirty=existing_dirty,
        )

    if can_skip:
        html_existing: Optional[Path] = None
        attachments_existing: Optional[Path] = None
        if state_entry:
            html_entry = state_entry.get("htmlPath")
            if isinstance(html_entry, str):
                html_existing = Path(html_entry)
            attach_entry = state_entry.get("attachmentsDir")
            if isinstance(attach_entry, str):
                attachments_existing = Path(attach_entry)
        if html_existing is None and html:
            html_existing = html_path
        if attachments_existing is None and attachments:
            attachments_existing = attachments_dir
        return DocumentPersistenceResult(
            markdown_path=markdown_path,
            html_path=html_existing,
            attachments_dir=attachments_existing,
            document=None,
            slug=slug,
            skipped=True,
            skip_reason="up-to-date",
            dirty=existing_dirty,
            content_hash=content_hash,
        )

    metadata = _ensure_polylogue_metadata(document.metadata)
    polylogue_meta = metadata.get("polylogue", {})
    polylogue_meta.update(
        {
            "title": title,
            "slug": slug,
            "collapseThreshold": collapse_threshold,
            "attachmentPolicy": policy,
            "updatedAt": updated_at,
            "createdAt": created_at,
            "html": html,
            "contentHash": content_hash,
            "lastImported": current_utc_timestamp(),
            "attachmentsDir": str(attachments_dir) if attachments else None,
            "dirty": False,
        }
    )
    if html_path and html:
        polylogue_meta["htmlPath"] = str(html_path)
    if provider:
        polylogue_meta["provider"] = provider
    if conversation_id:
        polylogue_meta["conversationId"] = conversation_id
    if extra_state:
        for key, value in extra_state.items():
            if value is not None:
                polylogue_meta[key] = value
    metadata["polylogue"] = polylogue_meta
    document.metadata = metadata

    markdown_path.write_text(document.to_markdown(), encoding="utf-8")

    if html:
        from .html import write_html

        write_html(document, html_path, html_theme)
    else:
        html_path = None

    if not attachments:
        try:
            attachments_dir.rmdir()
        except OSError:
            pass

    if provider and conversation_id:
        state_payload = {
            "slug": slug,
            "title": title,
            "lastUpdated": updated_at,
            "lastImported": polylogue_meta.get("lastImported"),
            "contentHash": content_hash,
            "collapseThreshold": collapse_threshold,
            "attachmentPolicy": policy,
            "outputPath": str(markdown_path),
            "htmlPath": str(html_path) if html_path else None,
            "attachmentsDir": str(attachments_dir) if attachments else None,
            "html": html,
            "dirty": False,
        }
        if extra_state:
            state_payload.update({k: v for k, v in extra_state.items() if v is not None})
        update_conversation_state(provider, conversation_id, **state_payload)

        try:  # pragma: no cover - indexing failures shouldn't abort writes
            update_index(
                provider=provider,
                conversation_id=conversation_id,
                slug=slug,
                path=markdown_path,
                document=document,
                metadata=polylogue_meta,
            )
        except Exception:
            pass

    return DocumentPersistenceResult(
        markdown_path=markdown_path,
        html_path=html_path,
        attachments_dir=attachments_dir if attachments else None,
        document=document,
        slug=slug,
        skipped=False,
        skip_reason=None,
        dirty=False,
        content_hash=content_hash,
    )
