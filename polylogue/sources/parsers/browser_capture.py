"""Parser for Polylogue browser-capture envelopes."""

from __future__ import annotations

import base64
import binascii
import math
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
from polylogue.core.timestamps import parse_timestamp
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
            continue
        # The native row remains authoritative for provider identity and file
        # metadata. The browser projection contributes acquired bytes and can
        # fill omissions, but must not replace native size/origin/file IDs.
        merged[candidate.provider_attachment_id] = existing.model_copy(
            update={
                "message_provider_id": existing.message_provider_id or candidate.message_provider_id,
                "name": existing.name or candidate.name,
                "mime_type": existing.mime_type or candidate.mime_type,
                "size_bytes": existing.size_bytes if existing.size_bytes is not None else candidate.size_bytes,
                "provider_file_id": existing.provider_file_id or candidate.provider_file_id,
                "provider_drive_id": existing.provider_drive_id or candidate.provider_drive_id,
                "upload_origin": existing.upload_origin or candidate.upload_origin,
                "attachment_kind": existing.attachment_kind or candidate.attachment_kind,
                "source_url": existing.source_url or candidate.source_url,
                "caption": existing.caption or candidate.caption,
                "inline_bytes": candidate.inline_bytes or existing.inline_bytes,
            }
        )
    return parsed.model_copy(update={"attachments": list(merged.values())})


def _merge_envelope_native_metadata(parsed: ParsedSession, envelope: BrowserCaptureEnvelope) -> ParsedSession:
    """Use browser-envelope fields only when the native payload omitted them.

    The envelope fields are projections of the same provider response/page and
    must never create a second conversation identity or replace richer native
    values. They are useful for current Claude responses that omit optional
    title/model/timestamp fields from the embedded payload.
    """

    updates: dict[str, object] = {}
    fallback_titles = {None, "", parsed.provider_session_id, envelope.session.provider_session_id}
    if parsed.title in fallback_titles and envelope.session.title:
        updates["title"] = envelope.session.title
    if parsed.created_at is None and envelope.session.created_at is not None:
        updates["created_at"] = envelope.session.created_at
    if parsed.updated_at is None and envelope.session.updated_at is not None:
        updates["updated_at"] = envelope.session.updated_at

    envelope_model = envelope.session.model
    if envelope_model:
        messages = [
            message if message.model_name is not None else message.model_copy(update={"model_name": envelope_model})
            for message in parsed.messages
        ]
        if messages != parsed.messages:
            updates["messages"] = messages
        models_used = list(parsed.models_used)
        if envelope_model not in models_used:
            models_used.append(envelope_model)
            updates["models_used"] = models_used
        if not any(
            event.event_type == "model_configuration" and event.source_message_provider_id is None
            for event in parsed.session_events
        ):
            updates["session_events"] = [
                *parsed.session_events,
                ParsedSessionEvent(
                    event_type="model_configuration",
                    timestamp=envelope.session.updated_at or envelope.session.created_at,
                    payload={"model": envelope_model},
                ),
            ]
    return parsed.model_copy(update=updates) if updates else parsed


def _claude_fallback_turn_payload(turn: object) -> dict[str, object]:
    from polylogue.browser_capture.models import BrowserCaptureTurn

    assert isinstance(turn, BrowserCaptureTurn)
    raw: dict[str, object] = dict(turn.provider_meta)
    # Typed envelope identity wins over any provider_meta echo.
    raw.update(
        {
            "uuid": turn.provider_turn_id,
            "sender": turn.role.value,
            "text": turn.text,
            "created_at": turn.timestamp,
            "parent_message_uuid": turn.parent_turn_id,
            "ordinal": turn.ordinal,
        }
    )
    if turn.attachments:
        raw["attachments"] = [
            {
                "id": attachment.provider_attachment_id,
                "name": attachment.name,
                "mime_type": attachment.mime_type,
                "size_bytes": attachment.size_bytes,
            }
            for attachment in turn.attachments
        ]
    return raw


def _parse_claude_fallback_envelope(
    envelope: BrowserCaptureEnvelope,
    provider_session_id: str,
) -> ParsedSession:
    from polylogue.sources.parsers.claude.common import (
        _first_identity_field,
        _message_model_effort,
        _thinking_configuration,
        normalize_chat_messages,
        normalize_timestamp,
    )

    created_at = normalize_timestamp(envelope.session.created_at) if envelope.session.created_at else None
    updated_at = normalize_timestamp(envelope.session.updated_at) if envelope.session.updated_at else None
    active_leaf_message_provider_id = _first_identity_field(
        envelope.session.provider_meta,
        "current_leaf_message_uuid",
        "current_leaf_message_id",
        "active_leaf_message_uuid",
        "active_leaf_message_id",
        "current_message_uuid",
        "current_message_id",
        "current_node",
    )
    normalized = normalize_chat_messages(
        [_claude_fallback_turn_payload(turn) for turn in envelope.session.turns],
        session_model=envelope.session.model,
        session_effort=_message_model_effort(envelope.session.provider_meta),
        session_thinking_configuration=_thinking_configuration(envelope.session.provider_meta),
        session_created_at=created_at,
        session_updated_at=updated_at,
        active_leaf_message_provider_id=active_leaf_message_provider_id,
    )

    attachments = [
        _browser_capture_parsed_attachment(
            attachment,
            message_provider_id=attachment.message_provider_id or turn.provider_turn_id,
        )
        for turn in envelope.session.turns
        for attachment in turn.attachments
    ]
    attachments.extend(
        _browser_capture_parsed_attachment(
            attachment,
            message_provider_id=attachment.message_provider_id,
        )
        for attachment in envelope.session.attachments
    )
    fidelity_flag = (
        COMPACT_BROWSER_CAPTURE_INGEST_FLAG if _is_compact_native_capture(envelope) else DOM_FALLBACK_INGEST_FLAG
    )
    return ParsedSession(
        source_name=Provider.CLAUDE_AI,
        provider_session_id=provider_session_id,
        title=envelope.session.title or envelope.provenance.page_title or provider_session_id,
        session_kind=_session_kind_for_browser_capture(envelope, provider_session_id),
        created_at=created_at,
        updated_at=updated_at,
        messages=normalized.messages,
        active_leaf_message_provider_id=normalized.active_leaf_message_provider_id,
        attachments=attachments,
        session_events=[*normalized.session_events, *_capture_session_events(envelope)],
        reported_duration_ms=normalized.reported_duration_ms,
        models_used=normalized.models_used,
        ingest_flags=list(
            dict.fromkeys(
                [
                    *normalized.ingest_flags,
                    *_ingest_flags_for_browser_capture(envelope, provider_session_id),
                    fidelity_flag,
                ]
            )
        ),
    )


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


def _non_negative_milliseconds(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        return None
    return round(numeric)


def _bounded_string(value: object, *, max_length: int) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or len(normalized) > max_length:
        return None
    return normalized


def _capture_generation_session_events(envelope: BrowserCaptureEnvelope) -> list[ParsedSessionEvent]:
    """Project typed, bounded live lifecycle observations from the extension.

    These observations describe what the browser actually exposed while a
    generation was running. They intentionally remain distinct from the
    provider-native terminal event emitted by the ChatGPT parser: DOM wall
    time and the provider's reported reasoning duration are different
    measurements, and retaining both makes their provenance queryable.
    """

    raw_observations: list[object] = []
    for provider_meta in (envelope.provider_meta, envelope.session.provider_meta):
        candidate = provider_meta.get("generation_observations")
        if isinstance(candidate, list):
            raw_observations.extend(candidate)

    events: list[ParsedSessionEvent] = []
    seen_observation_ids: set[str] = set()
    for raw_observation in raw_observations:
        if not isinstance(raw_observation, Mapping):
            continue
        observation_id = _bounded_string(raw_observation.get("observation_id"), max_length=512)
        state = _bounded_string(raw_observation.get("state"), max_length=32)
        observed_at = _bounded_string(raw_observation.get("observed_at"), max_length=80)
        evidence_source = _bounded_string(raw_observation.get("evidence_source"), max_length=80)
        fidelity = _bounded_string(raw_observation.get("fidelity"), max_length=32)
        duration_semantics = _bounded_string(raw_observation.get("duration_semantics"), max_length=80)
        if (
            observation_id is None
            or observation_id in seen_observation_ids
            or state not in {"started", "in_progress", "completed"}
            or observed_at is None
            or parse_timestamp(observed_at) is None
            or evidence_source is None
            or fidelity not in {"observed", "inferred"}
            or duration_semantics is None
        ):
            continue

        payload: dict[str, object] = {
            "observation_id": observation_id,
            "state": state,
            "evidence_source": evidence_source,
            "fidelity": fidelity,
            "duration_semantics": duration_semantics,
        }
        malformed_duration = False
        for key in ("displayed_elapsed_ms", "wall_elapsed_ms"):
            raw_duration = raw_observation.get(key)
            if raw_duration is None:
                continue
            duration = _non_negative_milliseconds(raw_duration)
            if duration is None:
                malformed_duration = True
                break
            payload[key] = duration
        if malformed_duration:
            continue
        for key, max_length in (("raw_label", 512), ("trigger", 80)):
            raw_value = raw_observation.get(key)
            if raw_value is None:
                continue
            value = _bounded_string(raw_value, max_length=max_length)
            if value is None:
                continue
            payload[key] = value

        turn_provider_id = _bounded_string(raw_observation.get("turn_provider_id"), max_length=256)
        events.append(
            ParsedSessionEvent(
                event_type="generation_lifecycle",
                timestamp=observed_at,
                source_message_provider_id=turn_provider_id,
                payload=payload,
            )
        )
        seen_observation_ids.add(observation_id)
    return events


def _capture_session_events(envelope: BrowserCaptureEnvelope) -> list[ParsedSessionEvent]:
    return [
        *_capture_interruption_session_events(envelope),
        *_capture_generation_session_events(envelope),
    ]


def _merge_envelope_session_events(parsed: ParsedSession, envelope: BrowserCaptureEnvelope) -> ParsedSession:
    events = _capture_session_events(envelope)
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
                _merge_envelope_attachments(
                    _merge_envelope_native_metadata(
                        parse_claude_ai(raw_provider_payload, provider_session_id),
                        envelope,
                    ),
                    envelope,
                ),
                envelope,
                provider_session_id,
                has_native_payload=True,
            ),
            envelope,
        )

    if envelope.session.provider is Provider.CLAUDE_AI:
        return _parse_claude_fallback_envelope(envelope, provider_session_id)

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
        session_events=_capture_session_events(envelope),
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
    "COMPACT_BROWSER_CAPTURE_INGEST_FLAG",
    "DOM_FALLBACK_INGEST_FLAG",
    "NATIVE_BROWSER_CAPTURE_INGEST_FLAG",
    "TEMPORARY_CHAT_INGEST_FLAG",
    "looks_like",
    "parse",
]
