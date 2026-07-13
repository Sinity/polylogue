"""Parser for Polylogue browser-capture envelopes."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from typing import TypeGuard

from polylogue.archive.ingest_flags import (
    COMPACT_BROWSER_CAPTURE_INGEST_FLAG,
    DOM_FALLBACK_INGEST_FLAG,
    NATIVE_BROWSER_CAPTURE_INGEST_FLAG,
    TEMPORARY_CHAT_INGEST_FLAG,
)
from polylogue.browser_capture.identity import legacy_browser_capture_native_id
from polylogue.browser_capture.models import (
    BrowserCaptureAttachment,
    BrowserCaptureEnvelope,
    looks_like_browser_capture,
)
from polylogue.core.enums import Provider, SessionKind
from polylogue.sources.parsers.base_models import (
    ParsedAttachment,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)


def _legacy_native_id(provider: Provider, provider_session_id: str | None) -> str | None:
    return legacy_browser_capture_native_id(provider, provider_session_id)


def looks_like(payload: object) -> bool:
    """Return whether a payload is a browser-capture envelope."""
    return looks_like_browser_capture(payload)


def _has_chatgpt_native_payload(payload: object) -> TypeGuard[Mapping[str, object]]:
    return (
        isinstance(payload, dict)
        and payload.get("polylogue_bridge_projection") != "chatgpt-native-compact-v1"
        and isinstance(payload.get("mapping"), dict)
    )


def _is_compact_native_capture(envelope: BrowserCaptureEnvelope) -> bool:
    return (
        envelope.provider_meta.get("capture_fidelity") == "native_compact"
        or envelope.session.provider_meta.get("capture_fidelity") == "native_compact"
    )


def _has_claude_ai_native_payload(payload: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(payload, dict) and isinstance(payload.get("chat_messages"), list)


def _ingest_flags_for_browser_capture(envelope: BrowserCaptureEnvelope, provider_session_id: str) -> list[str]:
    session_kind = envelope.session.session_kind
    legacy_session_kind = envelope.session.provider_meta.get("session_kind")
    if session_kind == "temporary" or provider_session_id.startswith("temporary:"):
        return [TEMPORARY_CHAT_INGEST_FLAG]
    if legacy_session_kind == "temporary":
        return [TEMPORARY_CHAT_INGEST_FLAG]
    return []


def _session_kind_for_browser_capture(envelope: BrowserCaptureEnvelope, provider_session_id: str) -> SessionKind:
    legacy_session_kind = envelope.session.provider_meta.get("session_kind")
    if (
        envelope.session.session_kind == "temporary"
        or provider_session_id.startswith("temporary:")
        or legacy_session_kind == "temporary"
    ):
        return SessionKind.TEMPORARY
    return SessionKind.STANDARD


def _apply_browser_capture_session_kind(
    session: ParsedSession,
    envelope: BrowserCaptureEnvelope,
    provider_session_id: str,
    *,
    has_native_payload: bool,
) -> ParsedSession:
    session_kind = _session_kind_for_browser_capture(envelope, provider_session_id)
    capture_flags = [NATIVE_BROWSER_CAPTURE_INGEST_FLAG] if has_native_payload else []
    ingest_flags = list(
        dict.fromkeys(
            [
                *session.ingest_flags,
                *_ingest_flags_for_browser_capture(envelope, provider_session_id),
                *capture_flags,
            ]
        )
    )
    return session.model_copy(update={"session_kind": session_kind, "ingest_flags": ingest_flags})


def _decode_base64_payload(value: object) -> bytes | None:
    if not isinstance(value, str) or not value:
        return None
    data = value
    if value.startswith("data:") and ";base64," in value:
        _, data = value.split(";base64,", 1)
    try:
        return base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error):
        return None


def _browser_capture_attachment_inline_bytes(attachment: BrowserCaptureAttachment) -> bytes | None:
    extracted_content = attachment.extracted_content
    if isinstance(extracted_content, str):
        return extracted_content.encode("utf-8")

    provider_meta = attachment.provider_meta
    if isinstance(provider_meta, Mapping):
        meta_extracted = provider_meta.get("extracted_content")
        if isinstance(meta_extracted, str):
            return meta_extracted.encode("utf-8")

    for value in (
        attachment.inline_base64,
        attachment.content_base64,
        attachment.data,
    ):
        decoded = _decode_base64_payload(value)
        if decoded is not None:
            return decoded

    if isinstance(provider_meta, Mapping):
        for key in ("inline_base64", "content_base64", "base64", "base64_data", "data"):
            decoded = _decode_base64_payload(provider_meta.get(key))
            if decoded is not None:
                return decoded

    return None


def _browser_capture_parsed_attachment(
    attachment: BrowserCaptureAttachment,
    *,
    message_provider_id: str | None,
) -> ParsedAttachment:
    inline_bytes = _browser_capture_attachment_inline_bytes(attachment)
    size_bytes = attachment.size_bytes
    if inline_bytes is not None and size_bytes is None:
        size_bytes = len(inline_bytes)
    url = attachment.url
    return ParsedAttachment(
        provider_attachment_id=attachment.provider_attachment_id,
        message_provider_id=message_provider_id,
        name=attachment.name,
        mime_type=attachment.mime_type,
        size_bytes=size_bytes,
        path=None,
        source_url=url if url else None,
        upload_origin="url" if url else "paste" if inline_bytes is not None else "oauth",
        inline_bytes=inline_bytes,
    )


def _merge_envelope_attachments(parsed: ParsedSession, envelope: BrowserCaptureEnvelope) -> ParsedSession:
    """Fold envelope attachments into a native-payload-delegated session.

    When a capture carries a provider-native payload the session structure is
    parsed from that payload, but the envelope may still carry attachments the
    extension acquired itself (e.g. sandbox/file-service bytes fetched through
    the authenticated page). Without this merge those bytes are silently
    dropped. Envelope rows win on id collision when they carry inline bytes —
    the native payload never does.
    """

    envelope_attachments = [
        _browser_capture_parsed_attachment(attachment, message_provider_id=attachment.message_provider_id)
        for attachment in (
            *(a for turn in envelope.session.turns for a in turn.attachments),
            *envelope.session.attachments,
        )
    ]
    if not envelope_attachments:
        return parsed
    merged: dict[str, ParsedAttachment] = {a.provider_attachment_id: a for a in parsed.attachments}
    for candidate in envelope_attachments:
        existing = merged.get(candidate.provider_attachment_id)
        if existing is None:
            merged[candidate.provider_attachment_id] = candidate
        elif candidate.inline_bytes is not None:
            merged[candidate.provider_attachment_id] = candidate.model_copy(
                update={
                    "name": candidate.name or existing.name,
                    "mime_type": candidate.mime_type or existing.mime_type,
                    "message_provider_id": candidate.message_provider_id or existing.message_provider_id,
                    "source_url": candidate.source_url or existing.source_url,
                    "attachment_kind": existing.attachment_kind,
                }
            )
    return parsed.model_copy(update={"attachments": list(merged.values())})


def _capture_interruption_session_events(envelope: BrowserCaptureEnvelope) -> list[ParsedSessionEvent]:
    """Turn a source-declared capture interruption into a session event.

    Distinct from the ``capture_gap`` event recorded when a lower-precedence
    DOM capture is skipped during write (an artifact of merge precedence): this
    is a positive, source-reported claim that observation stopped for a bounded
    interval, so it is preserved as its own ``source_outage`` event even when
    this capture otherwise wins outright.
    """

    interruption = envelope.provenance.capture_interruption
    if interruption is None:
        return []
    summary = (
        f"{envelope.provenance.adapter_name} reported no observation of this session "
        f"from {interruption.started_at} to {interruption.ended_at}: {interruption.reason}."
    )
    return [
        ParsedSessionEvent(
            event_type="source_outage",
            timestamp=interruption.ended_at,
            payload={
                "summary": summary,
                "started_at": interruption.started_at,
                "ended_at": interruption.ended_at,
                "reason": interruption.reason,
            },
        )
    ]


def _merge_envelope_session_events(parsed: ParsedSession, envelope: BrowserCaptureEnvelope) -> ParsedSession:
    events = _capture_interruption_session_events(envelope)
    if not events:
        return parsed
    return parsed.model_copy(update={"session_events": [*parsed.session_events, *events]})


def parse(payload: object, fallback_id: str) -> ParsedSession:
    """Parse a browser-capture envelope into the canonical parser contract."""
    envelope = BrowserCaptureEnvelope.model_validate(payload)
    provider = envelope.session.provider if envelope.session.provider is not Provider.UNKNOWN else Provider.UNKNOWN
    provider_session_id = _legacy_native_id(provider, envelope.session.provider_session_id) or fallback_id
    raw_provider_payload = envelope.raw_provider_payload
    if envelope.session.provider is Provider.CHATGPT and _has_chatgpt_native_payload(raw_provider_payload):
        from polylogue.sources.parsers.chatgpt import parse as parse_chatgpt

        return _merge_envelope_session_events(
            _apply_browser_capture_session_kind(
                _merge_envelope_attachments(parse_chatgpt(raw_provider_payload, provider_session_id), envelope),
                envelope,
                provider_session_id,
                has_native_payload=True,
            ),
            envelope,
        )
    if envelope.session.provider is Provider.CLAUDE_AI and _has_claude_ai_native_payload(raw_provider_payload):
        from polylogue.sources.parsers.claude.ai_parser import parse_ai as parse_claude_ai

        return _merge_envelope_session_events(
            _apply_browser_capture_session_kind(
                _merge_envelope_attachments(parse_claude_ai(raw_provider_payload, provider_session_id), envelope),
                envelope,
                provider_session_id,
                has_native_payload=True,
            ),
            envelope,
        )

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
                _browser_capture_parsed_attachment(
                    attachment,
                    message_provider_id=attachment.message_provider_id or turn.provider_turn_id,
                )
            )

    for attachment in envelope.session.attachments:
        attachments.append(
            _browser_capture_parsed_attachment(
                attachment,
                message_provider_id=attachment.message_provider_id,
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
        session_kind=_session_kind_for_browser_capture(envelope, provider_session_id),
        created_at=envelope.session.created_at,
        updated_at=envelope.session.updated_at,
        messages=messages,
        active_leaf_message_provider_id=active_leaf_message_provider_id,
        attachments=attachments,
        session_events=_capture_interruption_session_events(envelope),
        ingest_flags=[
            *dict.fromkeys(
                [
                    *_ingest_flags_for_browser_capture(envelope, provider_session_id),
                    COMPACT_BROWSER_CAPTURE_INGEST_FLAG
                    if _is_compact_native_capture(envelope)
                    else DOM_FALLBACK_INGEST_FLAG,
                ]
            )
        ],
    )


__all__ = [
    "DOM_FALLBACK_INGEST_FLAG",
    "NATIVE_BROWSER_CAPTURE_INGEST_FLAG",
    "TEMPORARY_CHAT_INGEST_FLAG",
    "looks_like",
    "parse",
]
