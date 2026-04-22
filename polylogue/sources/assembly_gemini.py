"""Gemini provider assembly for deterministic display labels."""

from __future__ import annotations

import re
from pathlib import Path

from polylogue.lib.roles import Role

from .assembly import SidecarData
from .parsers.base import ParsedAttachment, ParsedConversation, ParsedMessage

_DISPLAY_LABEL_LIMIT = 96
_ID_CHARS = re.compile(r"[A-Za-z0-9_.:-]+")
_HEX_ID = re.compile(r"[0-9a-f]{12,}", re.IGNORECASE)
_UUIDISH = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)


class GeminiAssemblySpec:
    """Gemini assembly using message and attachment evidence for display labels."""

    def discover_sidecars(self, source_paths: list[Path]) -> SidecarData:
        """Gemini display-label enrichment does not require sidecar files."""
        del source_paths
        return {}

    def enrich_conversation(
        self,
        conv: ParsedConversation,
        sidecar_data: SidecarData,
    ) -> ParsedConversation:
        """Add a display label for Gemini conversations whose imported title is weak."""
        del sidecar_data
        if not _weak_title(conv.title, conv.provider_conversation_id):
            return conv

        label, source = _derive_display_label(conv.messages, conv.attachments)
        if label is None or source is None:
            return conv

        provider_meta = dict(conv.provider_meta or {})
        provider_meta["display_label"] = label
        provider_meta["display_label_source"] = source
        provider_meta.setdefault("display_label_reason", "weak-title")
        return conv.model_copy(update={"provider_meta": provider_meta})


def _compact_text(value: object) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _truncate_label(text: str, *, limit: int = _DISPLAY_LABEL_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _looks_like_identifier(value: str) -> bool:
    token = value.strip()
    if not token or any(char.isspace() for char in token):
        return False
    if _UUIDISH.fullmatch(token) or _HEX_ID.fullmatch(token):
        return True
    if len(token) < 12 or _ID_CHARS.fullmatch(token) is None:
        return False
    digit_count = sum(char.isdigit() for char in token)
    separator_count = sum(char in "-_.:" for char in token)
    return digit_count >= 4 or separator_count >= 2


def _weak_title(title: str | None, provider_conversation_id: str) -> bool:
    text = _compact_text(title)
    if text is None:
        return True
    if text == provider_conversation_id or text == f"gemini:{provider_conversation_id}":
        return True
    return _looks_like_identifier(text)


def _first_user_prompt(messages: list[ParsedMessage]) -> str | None:
    for message in messages:
        if message.role is not Role.USER:
            continue
        text = _compact_text(message.text)
        if text:
            return text
    return None


def _first_attachment_name(attachments: list[ParsedAttachment]) -> str | None:
    for attachment in attachments:
        text = _compact_text(attachment.name)
        if text:
            return text
        if attachment.provider_meta:
            for key in ("name", "title"):
                text = _compact_text(attachment.provider_meta.get(key))
                if text:
                    return text
    return None


def _derive_display_label(
    messages: list[ParsedMessage],
    attachments: list[ParsedAttachment],
) -> tuple[str | None, str | None]:
    prompt = _first_user_prompt(messages)
    attachment_name = _first_attachment_name(attachments)
    if prompt and attachment_name:
        return _truncate_label(f"{attachment_name}: {prompt}"), "attachment-name:first-user-message"
    if prompt:
        return _truncate_label(prompt), "first-user-message"
    if attachment_name:
        return _truncate_label(f"Attachment: {attachment_name}"), "attachment-name"
    return None, None


__all__ = [
    "GeminiAssemblySpec",
]
