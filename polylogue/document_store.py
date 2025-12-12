from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import frontmatter
from .version import POLYLOGUE_VERSION, SCHEMA_VERSION


@dataclass
class DocumentMetadata:
    """Metadata for document persistence operations."""

    provider: Optional[str] = None
    conversation_id: Optional[str] = None
    title: str = ""
    updated_at: Optional[str] = None
    created_at: Optional[str] = None
    slug_hint: Optional[str] = None
    id_hint: Optional[str] = None
    extra_state: Optional[Dict[str, Any]] = field(default_factory=lambda: None)


@dataclass
class PersistenceOptions:
    """Options controlling document persistence behavior."""

    collapse_threshold: int = 25
    html: bool = False
    html_theme: Optional[str] = None
    attachment_policy: Optional[Dict[str, Any]] = field(default_factory=lambda: None)
    force: bool = False
    allow_dirty: bool = False

from .render import AttachmentInfo, MarkdownDocument, build_markdown_from_chunks
from .services.attachments import AttachmentManager
from .util import (
    assign_conversation_slug,
    conversation_is_current,
    current_utc_timestamp,
    slugify_title,
    sanitize_filename,
)
from .index import update_index
from .services.conversation_registrar import ConversationRegistrar


@dataclass
class ExistingDocument:
    metadata: Dict[str, Any]
    body: str
    body_hash: str
    content_hash: str


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
    post = frontmatter.loads(text)
    return dict(post.metadata), post.content


def _dump_front_matter(metadata: Dict[str, Any], body: str) -> str:
    post = frontmatter.Post(body, **metadata)
    return frontmatter.dumps(post)


def _metadata_without_content_hash(metadata: Dict[str, Any]) -> Dict[str, Any]:
    try:
        serialised = json.loads(json.dumps(metadata, default=str))
    except Exception:
        serialised = json.loads(json.dumps({}, default=str))
    polylogue = serialised.get("polylogue")
    if isinstance(polylogue, dict):
        polylogue.pop("contentHash", None)
        polylogue.pop("lastImported", None)
        polylogue.pop("dirty", None)
    return serialised


def _compute_content_hash(body: str, metadata: Dict[str, Any]) -> str:
    # Normalise body so YAML/frontmatter round-trips don't change the hash
    # solely due to trailing newlines/whitespace.
    body = body.rstrip()
    payload_metadata = _metadata_without_content_hash(metadata)
    try:
        metadata_str = json.dumps(payload_metadata, sort_keys=True, ensure_ascii=False)
    except Exception:
        metadata_str = "{}"
    material = body + "\n" + metadata_str
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def read_existing_document(path: Path) -> Optional[ExistingDocument]:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    metadata, body = _parse_front_matter(text)
    body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    content_hash = _compute_content_hash(body, metadata)
    return ExistingDocument(metadata=metadata, body=body, body_hash=body_hash, content_hash=content_hash)


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


def _prune_empty_values(payload: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        cleaned[key] = value
    return cleaned


def persist_document(
    *,
    document: MarkdownDocument,
    output_dir: Path,
    attachments: List[AttachmentInfo],
    registrar: ConversationRegistrar,
    metadata: Optional[DocumentMetadata] = None,
    options: Optional[PersistenceOptions] = None,
    provider: Optional[str] = None,
    conversation_id: Optional[str] = None,
    title: Optional[str] = None,
    updated_at: Optional[str] = None,
    created_at: Optional[str] = None,
    slug_hint: Optional[str] = None,
    id_hint: Optional[str] = None,
    extra_state: Optional[Dict[str, Any]] = None,
    collapse_threshold: Optional[int] = None,
    html: Optional[bool] = None,
    html_theme: Optional[str] = None,
    attachment_policy: Optional[Dict[str, Any]] = None,
    force: Optional[bool] = None,
    allow_dirty: Optional[bool] = None,
) -> DocumentPersistenceResult:
    # registrar is a required parameter (not Optional), so no None check needed
    metadata = metadata or DocumentMetadata(
        provider=provider,
        conversation_id=conversation_id,
        title=title or "",
        updated_at=updated_at,
        created_at=created_at,
        slug_hint=slug_hint,
        id_hint=id_hint,
        extra_state=extra_state,
    )
    options = options or PersistenceOptions(
        collapse_threshold=collapse_threshold or PersistenceOptions().collapse_threshold,
        html=bool(html) if html is not None else False,
        html_theme=html_theme,
        attachment_policy=attachment_policy,
        force=bool(force),
        allow_dirty=bool(allow_dirty),
    )
    # Extract metadata fields for readability
    provider = metadata.provider
    conversation_id = metadata.conversation_id
    title = metadata.title
    updated_at = metadata.updated_at
    created_at = metadata.created_at
    slug_hint = metadata.slug_hint
    id_hint = metadata.id_hint
    extra_state = metadata.extra_state

    # Extract options fields
    collapse_threshold = options.collapse_threshold
    html = options.html
    html_theme = options.html_theme
    attachment_policy = options.attachment_policy
    force = options.force
    allow_dirty = options.allow_dirty

    if slug_hint:
        slug = slug_hint
    elif provider and conversation_id:
        slug = assign_conversation_slug(provider, conversation_id, title, id_hint=id_hint)
    else:
        base = slugify_title(slug_hint or title)
        if not base:
            base = sanitize_filename(slug_hint or title)
            base = base.replace(" ", "-")
        slug = base or "document"

    output_dir.mkdir(parents=True, exist_ok=True)
    conversation_dir = output_dir / slug
    conversation_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = conversation_dir / "conversation.md"
    attachments_dir = conversation_dir / "attachments"
    html_path = conversation_dir / "conversation.html" if html else None

    state_entry: Optional[Dict[str, Any]] = None
    if provider and conversation_id:
        state_entry = registrar.get_state(provider, conversation_id)

    existing = read_existing_document(markdown_path)
    stored_hash = (state_entry or {}).get("contentHash")

    policy = _attachment_policy_with_counts(
        attachment_policy or _default_attachment_policy(),
        attachments,
    )

    doc_metadata = _ensure_polylogue_metadata(document.metadata)
    polylogue_meta = doc_metadata.get("polylogue", {})
    polylogue_meta.update(
        {
            "title": title,
            "slug": slug,
            "collapseThreshold": collapse_threshold,
            "attachmentPolicy": policy,
            "updatedAt": updated_at,
            "createdAt": created_at,
            "html": html,
            "lastImported": current_utc_timestamp(),
            "dirty": False,
            "polylogueVersion": POLYLOGUE_VERSION,
            "schemaVersion": SCHEMA_VERSION,
        }
    )
    expected_local_paths = AttachmentManager.expected_paths(markdown_path, attachments)
    if expected_local_paths:
        polylogue_meta["attachmentsDir"] = str(attachments_dir)
    else:
        polylogue_meta.pop("attachmentsDir", None)
    if html and html_path:
        polylogue_meta["htmlPath"] = str(html_path)
    else:
        polylogue_meta.pop("htmlPath", None)
    if provider:
        polylogue_meta["provider"] = provider
    if conversation_id:
        polylogue_meta["conversationId"] = conversation_id
    if extra_state:
        for key, value in extra_state.items():
            if value is not None:
                polylogue_meta[key] = value
    polylogue_meta = _prune_empty_values(polylogue_meta)
    doc_metadata["polylogue"] = polylogue_meta

    content_hash = _compute_content_hash(document.body, doc_metadata)
    polylogue_meta["contentHash"] = content_hash
    document.metadata = doc_metadata

    existing_dirty = False
    if existing and stored_hash:
        existing_dirty = existing.content_hash != stored_hash

    # Skip rewrite when content hash matches and not forcing/overwriting
    if existing and existing.content_hash == content_hash and not force and not allow_dirty:
        return DocumentPersistenceResult(
            markdown_path=markdown_path,
            html_path=html_path if html else None,
            attachments_dir=attachments_dir if expected_local_paths else None,
            document=document,
            slug=slug,
            skipped=True,
            skip_reason="up-to-date",
            dirty=existing_dirty,
            content_hash=content_hash,
        )

    # Protect user edits: if file is dirty and we're not explicitly allowing overwrite
    if provider and conversation_id and existing_dirty and state_entry:
        if force and not allow_dirty:
            # --force alone not enough to overwrite edited files
            raise ValueError(
                f"File {markdown_path} has local edits. "
                "Use --allow-dirty with --force to overwrite, or omit --force to preserve edits."
            )
        if not force:
            remote_same = True
            if updated_at and state_entry.get("lastUpdated") and state_entry["lastUpdated"] != updated_at:
                remote_same = False
            if state_entry.get("contentHash") and state_entry["contentHash"] != content_hash:
                remote_same = False
            if remote_same and existing:
                dirty_doc_metadata = _ensure_polylogue_metadata(existing.metadata)
                polylogue = dirty_doc_metadata.get("polylogue", {})
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
                        "attachmentsDir": polylogue.get("attachmentsDir") or (str(attachments_dir) if expected_local_paths else None),
                        "htmlPath": polylogue.get("htmlPath"),
                        "contentHash": state_entry.get("contentHash") or content_hash,
                        "localHash": existing.body_hash,
                        "dirty": True,
                    }
                )
                polylogue = _prune_empty_values(polylogue)
                dirty_doc_metadata["polylogue"] = polylogue
                _write_markdown(markdown_path, dirty_doc_metadata, existing.body)
                html_state_path = state_entry.get("htmlPath")
                html_existing_path = Path(html_state_path) if isinstance(html_state_path, str) else None
                attach_state_path = state_entry.get("attachmentsDir")
                attachments_existing_path = (
                    Path(attach_state_path) if isinstance(attach_state_path, str) else None
                )
                dirty_payload = {
                    "slug": slug,
                    "title": title,
                    "lastUpdated": updated_at,
                    "lastImported": polylogue_meta.get("lastImported"),
                    "contentHash": state_entry.get("contentHash") or content_hash,
                    "collapseThreshold": collapse_threshold,
                    "attachmentPolicy": policy,
                    "outputPath": str(markdown_path),
                    "htmlPath": html_state_path,
                    "attachmentsDir": attach_state_path,
                    "html": state_entry.get("html", html),
                    "dirty": True,
                    "localHash": existing.body_hash,
                }
                registrar.mark_dirty(provider, conversation_id, dirty_payload)
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
            entry=state_entry,
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
        if attachments_existing is None and expected_local_paths and attachments_dir.exists():
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

    rendered_markdown = document.to_markdown()

    if expected_local_paths:
        attachments_dir.mkdir(parents=True, exist_ok=True)
    AttachmentManager.reconcile(attachments_dir, expected_local_paths)

    markdown_path.write_text(rendered_markdown, encoding="utf-8")

    if html:
        from .html import write_html

        target_html = html_path or (conversation_dir / "conversation.html")
        write_html(document, target_html, html_theme)
        html_path = target_html
    else:
        html_file = conversation_dir / "conversation.html"
        if html_file.exists():
            try:
                html_file.unlink()
            except OSError:
                pass
        html_path = None

    if provider and conversation_id:
        attachments_dir_value = attachments_dir if (expected_local_paths and attachments_dir.exists()) else None
        stats = document.stats or {}
        total_tokens = int(stats.get("totalTokensApprox", 0) or 0)
        total_words = int(stats.get("totalWordsApprox", 0) or 0)
        attachment_bytes = sum(att.size_bytes or 0 for att in attachments)

        registrar.record_document(
            provider=provider,
            conversation_id=conversation_id,
            slug=slug,
            title=title,
            content_hash=content_hash,
            collapse_threshold=collapse_threshold,
            attachment_policy=policy,
            markdown_path=markdown_path,
            html_path=html_path,
            html_enabled=html,
            attachments_dir=attachments_dir_value,
            updated_at=updated_at,
            created_at=created_at,
            last_imported=polylogue_meta.get("lastImported"),
            attachment_bytes=attachment_bytes,
            tokens=total_tokens,
            words=total_words,
            dirty=False,
            extra_state=extra_state,
        )

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
        attachments_dir=attachments_dir if expected_local_paths else None,
        document=document,
        slug=slug,
        skipped=False,
        skip_reason=None,
        dirty=False,
        content_hash=content_hash,
    )
