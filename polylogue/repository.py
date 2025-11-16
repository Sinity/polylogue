from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .document_store import DocumentPersistenceResult, persist_document
from .services.conversation_registrar import ConversationRegistrar
from .render import AttachmentInfo, MarkdownDocument


class ConversationRepository:
    """Facade over document persistence so callers share a common entry point."""

    def persist(
        self,
        *,
        provider: Optional[str],
        conversation_id: Optional[str],
        title: str,
        document: MarkdownDocument,
        output_dir: Path,
        collapse_threshold: int,
        attachments: list[AttachmentInfo],
        updated_at: Optional[str],
        created_at: Optional[str],
        html: bool,
        html_theme: Optional[str],
        attachment_policy: Optional[Dict[str, object]] = None,
        extra_state: Optional[Dict[str, object]] = None,
        slug_hint: Optional[str] = None,
        id_hint: Optional[str] = None,
        force: bool = False,
        allow_dirty: bool = False,
        registrar: ConversationRegistrar = None,
    ) -> DocumentPersistenceResult:
        if registrar is None:
            raise ValueError("ConversationRegistrar instance required")
        return persist_document(
            provider=provider,
            conversation_id=conversation_id,
            title=title,
            document=document,
            output_dir=output_dir,
            collapse_threshold=collapse_threshold,
            attachments=attachments,
            updated_at=updated_at,
            created_at=created_at,
            html=html,
            html_theme=html_theme,
            attachment_policy=attachment_policy,
            extra_state=extra_state,
            slug_hint=slug_hint,
            id_hint=id_hint,
            force=force,
            allow_dirty=allow_dirty,
            registrar=registrar,
        )
