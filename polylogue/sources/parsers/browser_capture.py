"""Parser for Polylogue browser-capture envelopes."""

from __future__ import annotations

from polylogue.browser_capture.models import BrowserCaptureEnvelope, looks_like_browser_capture
from polylogue.sources.parsers.base_models import ParsedAttachment, ParsedMessage, ParsedSession
from polylogue.types import Provider


def looks_like(payload: object) -> bool:
    """Return whether a payload is a browser-capture envelope."""
    return looks_like_browser_capture(payload)


def _provider_meta(envelope: BrowserCaptureEnvelope) -> dict[str, object]:
    meta: dict[str, object] = {
        "source": envelope.source,
        "capture_id": envelope.capture_id,
        "browser_capture": True,
        "provenance": envelope.provenance.model_dump(mode="json", exclude_none=True),
    }
    if envelope.session.model:
        meta["model"] = envelope.session.model
    if envelope.session.provider_meta:
        meta["session"] = envelope.session.provider_meta
    if envelope.provider_meta:
        meta["capture"] = envelope.provider_meta
    return meta


def parse(payload: object, fallback_id: str) -> ParsedSession:
    """Parse a browser-capture envelope into the canonical parser contract."""
    envelope = BrowserCaptureEnvelope.model_validate(payload)
    seen_turns: set[str] = set()
    messages: list[ParsedMessage] = []
    attachments: list[ParsedAttachment] = []
    message_position = 0

    for index, turn in enumerate(envelope.session.turns):
        if turn.provider_turn_id in seen_turns:
            continue
        seen_turns.add(turn.provider_turn_id)
        provider_meta: dict[str, object] = {
            "browser_capture": True,
            "ordinal": turn.ordinal if turn.ordinal else index,
        }
        if turn.provider_meta:
            provider_meta.update(turn.provider_meta)
        messages.append(
            ParsedMessage(
                provider_message_id=turn.provider_turn_id,
                role=turn.role,
                text=turn.text,
                timestamp=turn.timestamp,
                provider_meta=provider_meta,
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
                    provider_meta={**attachment.provider_meta, "url": attachment.url}
                    if attachment.url
                    else attachment.provider_meta or None,
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
                provider_meta={**attachment.provider_meta, "url": attachment.url}
                if attachment.url
                else attachment.provider_meta or None,
                upload_origin="url" if attachment.url else "oauth",
            )
        )

    provider = envelope.session.provider if envelope.session.provider is not Provider.UNKNOWN else Provider.UNKNOWN
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
        provider_session_id=envelope.session.provider_session_id or fallback_id,
        title=envelope.session.title or envelope.provenance.page_title or fallback_id,
        created_at=envelope.session.created_at,
        updated_at=envelope.session.updated_at or envelope.provenance.captured_at,
        messages=messages,
        active_leaf_message_provider_id=active_leaf_message_provider_id,
        attachments=attachments,
        provider_meta=_provider_meta(envelope),
    )


__all__ = ["looks_like", "parse"]
