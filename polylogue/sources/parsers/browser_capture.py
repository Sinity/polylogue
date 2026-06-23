"""Parser for Polylogue browser-capture envelopes."""

from __future__ import annotations

from polylogue.browser_capture.models import BrowserCaptureEnvelope, looks_like_browser_capture
from polylogue.core.enums import Provider
from polylogue.sources.parsers.base_models import ParsedAttachment, ParsedMessage, ParsedSession


def _legacy_native_id(provider: Provider, provider_session_id: str | None) -> str | None:
    """Recover native ids from old browser-extension synthetic session ids."""
    if not provider_session_id:
        return None
    provider_prefix = provider.value
    synthetic_prefix = f"{provider_prefix}:"
    if provider_session_id.startswith(synthetic_prefix):
        parts = provider_session_id.split(":")
        if len(parts) == 3 and parts[1] and "/" not in parts[1]:
            return parts[1]
    hyphen_prefix = f"{provider_prefix}-"
    if provider_session_id.startswith(hyphen_prefix):
        import re

        match = re.search(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            provider_session_id,
        )
        if match:
            return match.group(0)
    return provider_session_id


def looks_like(payload: object) -> bool:
    """Return whether a payload is a browser-capture envelope."""
    return looks_like_browser_capture(payload)


def parse(payload: object, fallback_id: str) -> ParsedSession:
    """Parse a browser-capture envelope into the canonical parser contract."""
    envelope = BrowserCaptureEnvelope.model_validate(payload)
    provider = envelope.session.provider if envelope.session.provider is not Provider.UNKNOWN else Provider.UNKNOWN
    provider_session_id = _legacy_native_id(provider, envelope.session.provider_session_id) or fallback_id
    if envelope.session.provider is Provider.CHATGPT and envelope.raw_provider_payload:
        from polylogue.sources.parsers.chatgpt import parse as parse_chatgpt

        return parse_chatgpt(envelope.raw_provider_payload, provider_session_id)
    if envelope.session.provider is Provider.CLAUDE_AI and envelope.raw_provider_payload:
        from polylogue.sources.parsers.claude.ai_parser import parse_ai as parse_claude_ai

        return parse_claude_ai(envelope.raw_provider_payload, provider_session_id)

    seen_turns: set[str] = set()
    messages: list[ParsedMessage] = []
    attachments: list[ParsedAttachment] = []
    message_position = 0

    for turn in envelope.session.turns:
        if turn.provider_turn_id in seen_turns:
            continue
        seen_turns.add(turn.provider_turn_id)
        messages.append(
            ParsedMessage(
                provider_message_id=turn.provider_turn_id,
                role=turn.role,
                text=turn.text,
                timestamp=turn.timestamp,
                parent_message_provider_id=turn.parent_turn_id,
                position=message_position,
                variant_index=0,
                is_active_path=True,
                model_name=envelope.session.model,
            )
        )
        message_position += 1
        for attachment in turn.attachments:
            attachments.append(
                ParsedAttachment(
                    provider_attachment_id=attachment.provider_attachment_id,
                    message_provider_id=attachment.message_provider_id or turn.provider_turn_id,
                    name=attachment.name,
                    mime_type=attachment.mime_type,
                    size_bytes=attachment.size_bytes,
                    path=None,
                    source_url=attachment.url if attachment.url else None,
                    upload_origin="url" if attachment.url else "oauth",
                )
            )

    for attachment in envelope.session.attachments:
        attachments.append(
            ParsedAttachment(
                provider_attachment_id=attachment.provider_attachment_id,
                message_provider_id=attachment.message_provider_id,
                name=attachment.name,
                mime_type=attachment.mime_type,
                size_bytes=attachment.size_bytes,
                path=None,
                source_url=attachment.url if attachment.url else None,
                upload_origin="url" if attachment.url else "oauth",
            )
        )

    active_leaf_message_provider_id = messages[-1].provider_message_id if messages else None
    if active_leaf_message_provider_id is not None:
        messages = [
            message.model_copy(
                update={"is_active_leaf": message.provider_message_id == active_leaf_message_provider_id}
            )
            for message in messages
        ]
    return ParsedSession(
        source_name=provider,
        provider_session_id=provider_session_id,
        title=envelope.session.title or envelope.provenance.page_title or provider_session_id,
        created_at=envelope.session.created_at,
        updated_at=envelope.session.updated_at or envelope.provenance.captured_at,
        messages=messages,
        active_leaf_message_provider_id=active_leaf_message_provider_id,
        attachments=attachments,
    )


__all__ = ["looks_like", "parse"]
