"""Parser for Polylogue browser-capture envelopes."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, BinaryIO, cast

from polylogue.archive.ingest_flags import (
    DOM_FALLBACK_INGEST_FLAG,
    NATIVE_BROWSER_CAPTURE_INGEST_FLAG,
    TEMPORARY_CHAT_INGEST_FLAG,
)
from polylogue.browser_capture.identity import legacy_browser_capture_native_id
from polylogue.browser_capture.models import (
    BrowserCaptureAttachment,
    BrowserCaptureBlock,
    BrowserCaptureEnvelope,
    BrowserCaptureTurn,
    SpilledCarrier,
    _CanonicalNativeTurnWitness,
    has_chatgpt_native_payload,
    has_claude_ai_native_payload,
    has_grok_native_payload,
    looks_like_browser_capture,
    validate_capture_envelope,
)
from polylogue.core.enums import BlockType, Provider, Role, SessionKind, TitleSource, ToolOutcome
from polylogue.core.hashing import hash_bytes
from polylogue.core.message_owner import MessageOwnerAmbiguityError, MessageOwnerCoordinate
from polylogue.core.timestamps import parse_timestamp
from polylogue.pipeline.ids import _message_owner_coordinate
from polylogue.sources.detection_projection import DetectorProjection
from polylogue.sources.parsers.base import parser_admission
from polylogue.sources.parsers.base_models import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)
from polylogue.sources.parsers.base_support import decode_attachment_base64, derive_attachment_provenance
from polylogue.sources.tool_result_reasons import unknown_reason

if TYPE_CHECKING:
    from typing import Protocol

    from polylogue.sources.prepared_message_sink import ScratchSessionSpill

    class _NativeReadInto(Protocol):
        def readinto(self, buffer: bytearray, /) -> int | None: ...


class NativeCaptureIdentityMismatchError(ValueError):
    """Native content cannot be composed with another session's capture evidence."""

    def __init__(self, expected_session_id: str, actual_session_id: str) -> None:
        self.expected_session_id = expected_session_id
        self.actual_session_id = actual_session_id
        super().__init__("native provider session identity disagrees with the capture envelope")


def _require_matching_native_identity(parsed: ParsedSession, provider_session_id: str) -> ParsedSession:
    observed = legacy_browser_capture_native_id(parsed.source_name, parsed.provider_session_id)
    if (observed or parsed.provider_session_id) != provider_session_id:
        raise NativeCaptureIdentityMismatchError(provider_session_id, parsed.provider_session_id)
    return parsed


def parse_native_payload(
    provider: Provider, payload: object, native_id: str, *, prepared_attachment_ownership: bool = False
) -> ParsedSession:
    """Validate original native content through its ordinary provider owner.

    This model route retains the ordinary parser's allocation contract. File
    preparation can use those same parsers' supported scratch-backed routes;
    this function is not a scalar-independent tokenizer.
    """
    if provider is Provider.CODEX:
        from polylogue.sources.parsers.codex import is_supported_session_stream
        from polylogue.sources.parsers.codex import parse as parse_codex

        if not isinstance(payload, list) or not is_supported_session_stream(payload):
            raise ValueError("Codex native capture requires a supported record stream")
        parsed = parse_codex(payload, native_id)
    elif provider is Provider.CHATGPT and has_chatgpt_native_payload(payload):
        from polylogue.sources.parsers.chatgpt import parse as parse_chatgpt

        def occurrence(attachment: ParsedAttachment, position: int) -> None:
            attachment.owner_coordinate = MessageOwnerCoordinate(position=position)

        parsed = parse_chatgpt(
            payload, native_id, attachment_occurrence=occurrence if prepared_attachment_ownership else None
        )
    elif provider is Provider.CLAUDE_AI and has_claude_ai_native_payload(payload):
        from polylogue.sources.parsers.claude.ai_parser import parse_ai

        parsed = parse_ai(payload, native_id)
    elif provider is Provider.GROK and has_grok_native_payload(payload):
        from polylogue.sources.parsers.grok import parse_native_bundle

        sessions = parse_native_bundle(payload, native_id)
        if len(sessions) != 1:
            raise ValueError("Grok native capture requires one supported conversation")
        parsed = sessions[0]
    else:
        raise ValueError("capture does not contain supported original native content")
    return _require_matching_native_identity(parsed, native_id)


def _parsed_blocks_for_turn(turn: BrowserCaptureTurn) -> list[ParsedContentBlock]:
    """Convert a turn's typed capture blocks into the parser contract's blocks.

    ``BrowserCaptureBlock`` mirrors ``ParsedContentBlock`` field-for-field
    (polylogue-ah21) so this is a direct projection, not a reconstruction --
    the capture adapter is the thing that decided ``type``/``tool_id``/
    ``tool_input``/``is_error``, not this parser. ``text``/``tool_name``/
    ``tool_id``/``tool_input``/``media_type``/``is_error``/``exit_code`` all
    land on real typed columns. ``metadata`` does not -- the ``blocks`` table
    has no metadata column and the write path only reads a ``language`` key
    back out of it (``storage/sqlite/archive_tiers/write.py:
    _block_language``), so anything the capture extension attaches here would
    be silently dropped at write time (bd polylogue-9x22). See
    ``_block_metadata_evidence_events`` below, which routes it through
    ``session_events`` instead -- the capture extension's own wire protocol
    has no fixed vocabulary for this field, so (unlike other polylogue-9x22
    sites) the whole dict is carried verbatim rather than picking known keys.
    """

    return [
        ParsedContentBlock(
            type=block.type,
            text=block.text,
            tool_name=block.tool_name,
            tool_id=block.tool_id,
            tool_input=block.tool_input,
            media_type=block.media_type,
            metadata=block.metadata,
            is_error=block.is_error,
            exit_code=block.exit_code,
            signature=block.signature,
            tool_outcome=block.tool_outcome,
            file_edit=block.file_edit,
            web_constructs=block.web_constructs,
            # The capture adapter's ``is_error``/``exit_code`` are already
            # typed, so a result that reaches here without one is a page the
            # extension read no outcome from.
            outcome_unknown_reason=(
                block.outcome_unknown_reason
                if block.outcome_unknown_reason is not None
                else (
                    unknown_reason(is_error=block.is_error, exit_code=block.exit_code)
                    if block.type is BlockType.TOOL_RESULT and block.tool_outcome in (None, ToolOutcome.UNKNOWN)
                    else None
                )
            ),
        )
        for block in turn.blocks
    ]


def _block_metadata_evidence_events(
    blocks: list[BrowserCaptureBlock],
    *,
    source_message_provider_id: str,
    timestamp: str | None,
) -> list[ParsedSessionEvent]:
    """Carry ``BrowserCaptureBlock.metadata`` into session_events, verbatim.

    One event per block with non-empty metadata; ``block_index`` lets a
    reader re-associate the event with its block in ``turn.blocks`` order
    (same shape as ``claude/common.py``'s ``claude_ai_web_tool_evidence``).
    """

    events: list[ParsedSessionEvent] = []
    for block_index, block in enumerate(blocks):
        if not block.metadata:
            continue
        events.append(
            ParsedSessionEvent(
                event_type="browser_capture_block_metadata",
                timestamp=timestamp,
                source_message_provider_id=source_message_provider_id,
                payload={"block_index": block_index, **dict(block.metadata)},
            )
        )
    return events


def _claude_raw_content_segments(turn: BrowserCaptureTurn) -> list[dict[str, object]] | None:
    """Project a turn's typed blocks into Anthropic-API-shaped raw segments.

    ``_parse_claude_fallback_envelope`` builds a synthetic raw record per turn
    and feeds it through ``normalize_chat_messages`` / ``_claude_content_blocks``
    -- the same machinery a genuine Claude web export uses to build typed
    tool_use/tool_result/thinking blocks from a message's ``content`` list.
    Without this projection a captured tool turn's blocks would be silently
    dropped by the fallback path even though the turn carries real structure.
    Returns ``None`` when the turn has no typed blocks, so the caller falls
    back to the legacy text-only raw shape untouched.
    """

    if not turn.blocks:
        return None
    segments: list[dict[str, object]] = []
    for block in turn.blocks:
        if block.type is BlockType.TOOL_USE:
            segments.append(
                {
                    "type": "tool_use",
                    "id": block.tool_id,
                    "name": block.tool_name,
                    "input": block.tool_input or {},
                }
            )
        elif block.type is BlockType.TOOL_RESULT:
            segments.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.tool_id,
                    "content": block.text,
                    "is_error": block.is_error,
                }
            )
        elif block.type is BlockType.THINKING:
            segments.append({"type": "thinking", "thinking": block.text or ""})
        elif block.type is BlockType.CODE:
            segments.append({"type": "code", "text": block.text or "", **(block.metadata or {})})
        elif block.type in (BlockType.IMAGE, BlockType.DOCUMENT):
            segments.append(
                {
                    "type": block.type.value,
                    "media_type": block.media_type,
                    **(block.metadata or {}),
                }
            )
        else:
            segments.append({"type": "text", "text": block.text or ""})
    return segments


def looks_like(payload: object) -> bool:
    """Return whether a payload is a browser-capture envelope."""
    return looks_like_browser_capture(payload)


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


def _browser_capture_attachment_content(attachment: BrowserCaptureAttachment) -> bytes | SpilledCarrier | None:
    """The attachment's bytes, or the blob a streamed decode already put them in."""
    if (spilled := attachment.spilled_carrier("content_base64")) is not None:
        return spilled
    if attachment.content_base64 is not None:
        return decode_attachment_base64(attachment.content_base64)

    if (spilled := attachment.spilled_carrier("inline_base64")) is not None:
        return spilled
    if attachment.inline_base64 is not None:
        return decode_attachment_base64(attachment.inline_base64, field_name="inline_base64")

    if (spilled := attachment.spilled_carrier("data")) is not None:
        return spilled
    if attachment.data is not None:
        return decode_attachment_base64(attachment.data, field_name="data")

    provider_meta = attachment.provider_meta
    if isinstance(provider_meta, Mapping):
        for key in ("content_base64", "inline_base64"):
            if key in provider_meta and provider_meta[key] is not None:
                return decode_attachment_base64(provider_meta[key], field_name=key)

    if isinstance(provider_meta, Mapping):
        for key in ("inline_base64", "content_base64", "base64", "base64_data", "data"):
            if key in {"inline_base64", "content_base64"} or provider_meta.get(key) is None:
                continue
            try:
                return decode_attachment_base64(provider_meta[key], field_name=key)
            except ValueError:
                continue

    extracted_content = attachment.extracted_content
    if isinstance(extracted_content, str):
        return extracted_content.encode("utf-8")

    if isinstance(provider_meta, Mapping):
        meta_extracted = provider_meta.get("extracted_content")
        if isinstance(meta_extracted, str):
            return meta_extracted.encode("utf-8")

    return None


def _browser_capture_parsed_attachment(
    attachment: BrowserCaptureAttachment,
    *,
    message_provider_id: str | None,
    role: Role | None = None,
) -> ParsedAttachment:
    content = _browser_capture_attachment_content(attachment)
    inline_bytes = content if isinstance(content, bytes) else None
    precomputed_blob = (content.blob_hash, content.size_bytes) if isinstance(content, SpilledCarrier) else None
    size_bytes = attachment.size_bytes
    if size_bytes is None:
        if inline_bytes is not None:
            size_bytes = len(inline_bytes)
        elif precomputed_blob is not None:
            size_bytes = precomputed_blob[1]
    url = attachment.url
    direction, producer_ref = derive_attachment_provenance(role, message_provider_id)
    return ParsedAttachment(
        provider_attachment_id=attachment.provider_attachment_id,
        message_provider_id=message_provider_id,
        attachment_kind=attachment.attachment_kind,
        name=attachment.name,
        mime_type=attachment.mime_type,
        size_bytes=size_bytes,
        path=None,
        source_url=url if url else None,
        upload_origin="url" if url else "paste" if content is not None else "oauth",
        direction=direction,
        producer_ref=producer_ref,
        inline_bytes=inline_bytes,
        precomputed_blob=precomputed_blob,
    )


def _attachment_content_hash(attachment: ParsedAttachment) -> str | None:
    if attachment.inline_bytes is not None:
        return hash_bytes(attachment.inline_bytes)
    if attachment.precomputed_blob is not None:
        return attachment.precomputed_blob[0]
    return None


def _is_claude_envelope_attachment_id(provider_attachment_id: str) -> bool:
    """Identify the Claude extension's id-less attachment projection.

    Claude's native response and the browser envelope expose the same source
    attachment through different routes.  The extension has no provider id for
    ``attachments[]`` records, so it deliberately uses this namespace for its
    stable projection id.  Keep this check local to the browser merge: an
    ``att-*`` id from another provider may be a real provider identity.
    """

    return provider_attachment_id.startswith("claude-attachment:")


def _claude_attachment_cross_route_match(
    native: ParsedAttachment,
    envelope: ParsedAttachment,
) -> bool:
    """Return whether two differently identified rows are one source file.

    A browser envelope row is only reconciled with a native row when the
    envelope carries Claude's known synthetic id and the rows share their
    owner and descriptors.  Byte evidence is required whenever available;
    if one side has not acquired bytes yet, the declared size must still
    agree.  Requiring one-to-one matching at the caller keeps two legitimate
    same-name uploads from being attributed to an arbitrary native row.
    """

    claude_native_file_id = (
        envelope.provider_file_id
        if envelope.provider_file_id
        else envelope.provider_attachment_id.removeprefix("claude-file:")
        if envelope.provider_attachment_id.startswith("claude-file:")
        else None
    )
    if not (_is_claude_envelope_attachment_id(envelope.provider_attachment_id) or claude_native_file_id):
        return False
    if claude_native_file_id and native.provider_attachment_id != claude_native_file_id:
        return False
    if native.message_provider_id != envelope.message_provider_id:
        return False
    if not native.name or native.name != envelope.name:
        return False
    if native.mime_type and envelope.mime_type and native.mime_type != envelope.mime_type:
        return False
    if native.size_bytes is not None and envelope.size_bytes is not None and native.size_bytes != envelope.size_bytes:
        return False

    if native.inline_bytes is not None and envelope.inline_bytes is not None:
        return native.inline_bytes == envelope.inline_bytes
    native_hash = _attachment_content_hash(native)
    envelope_hash = _attachment_content_hash(envelope)
    if native_hash is not None and envelope_hash is not None:
        return native_hash == envelope_hash
    if claude_native_file_id:
        # An exact provider file identity needs no acquired-byte witness.
        return True
    # With only one byte carrier, a declared size is the minimum evidence that
    # this is the same source object.  Two metadata-only rows remain distinct.
    return (
        (native_hash is not None or envelope_hash is not None)
        and native.size_bytes is not None
        and envelope.size_bytes == native.size_bytes
    )


def _merge_prepared_native_attachments(parsed: ParsedSession, envelope: BrowserCaptureEnvelope) -> ParsedSession | None:
    """Join the canonical plan by its exact original attachment occurrence."""
    rows = envelope.session.attachments
    if not any("native_attachment_ordinal" in row.provider_meta for row in rows):
        return None
    merged = list(parsed.attachments)
    seen: set[int] = set()
    for row in rows:
        ordinal = row.provider_meta.get("native_attachment_ordinal")
        if type(ordinal) is not int or not 0 <= ordinal < len(merged) or ordinal in seen:
            raise MessageOwnerAmbiguityError("capture asset lacks an exact canonical occurrence")
        seen.add(ordinal)
        native = parsed.attachments[ordinal]
        if (row.provider_attachment_id, row.message_provider_id) != (
            native.provider_attachment_id,
            native.message_provider_id,
        ):
            raise MessageOwnerAmbiguityError("capture asset occurrence disagrees with native content")
        turn_ordinal = row.provider_meta.get("native_turn_ordinal")
        coordinate = native.owner_coordinate
        role = None
        if turn_ordinal is not None:
            if type(turn_ordinal) is not int or not 0 <= turn_ordinal < len(parsed.messages):
                raise MessageOwnerAmbiguityError("capture asset lacks a canonical message occurrence")
            message = parsed.messages[turn_ordinal]
            if message.provider_message_id != native.message_provider_id:
                raise MessageOwnerAmbiguityError("capture asset message disagrees with native content")
            raw_position = row.provider_meta.get("native_raw_position")
            if parsed.source_name is Provider.CHATGPT:
                if (
                    type(raw_position) is not int
                    or native.owner_coordinate is None
                    or raw_position != native.owner_coordinate.position
                    or raw_position != message.position
                ):
                    raise MessageOwnerAmbiguityError("capture asset raw occurrence disagrees with native content")
            elif native.owner_coordinate is not None and native.owner_coordinate != _message_owner_coordinate(
                message, turn_ordinal
            ):
                raise MessageOwnerAmbiguityError("capture asset owner disagrees with native content")
            coordinate = _message_owner_coordinate(message, turn_ordinal)
            role = message.role
        if turn_ordinal is None and parsed.source_name is Provider.CHATGPT:
            raw_position = row.provider_meta.get("native_raw_position")
            if native.owner_coordinate is None or raw_position != native.owner_coordinate.position:
                raise MessageOwnerAmbiguityError("capture orphan asset raw occurrence disagrees with native content")
            if any(message.position == raw_position for message in parsed.messages):
                raise MessageOwnerAmbiguityError("capture asset omitted its canonical owner")
            coordinate = None
        if (row.name, row.mime_type, row.attachment_kind, row.url) != (
            native.name,
            native.mime_type,
            native.attachment_kind,
            native.source_url,
        ):
            raise MessageOwnerAmbiguityError("capture asset descriptor disagrees with native content")
        candidate = _browser_capture_parsed_attachment(row, message_provider_id=native.message_provider_id, role=role)
        merged[ordinal] = native.model_copy(
            update={
                "owner_coordinate": coordinate,
                "inline_bytes": candidate.inline_bytes if candidate.inline_bytes is not None else native.inline_bytes,
                "precomputed_blob": candidate.precomputed_blob
                if candidate.precomputed_blob is not None
                else native.precomputed_blob,
            }
        )
    if len(seen) != len(parsed.attachments):
        raise MessageOwnerAmbiguityError("capture asset plan does not cover canonical attachments")
    return parsed.model_copy(update={"attachments": merged})


def _merge_envelope_attachments(parsed: ParsedSession, envelope: BrowserCaptureEnvelope) -> ParsedSession:
    """Fold envelope attachments into a native-payload-delegated session.

    When a capture carries a provider-native payload the session structure is
    parsed from that payload, but the envelope may still carry attachments the
    extension acquired itself (e.g. sandbox/file-service bytes fetched through
    the authenticated page). Without this merge those bytes are silently
    dropped. Envelope rows win on id collision when they carry inline bytes —
    the native payload never does.
    """

    prepared = _merge_prepared_native_attachments(parsed, envelope)
    if prepared is not None:
        return prepared
    envelope_attachments = []
    for turn in envelope.session.turns:
        for attachment in turn.attachments:
            provider_id = attachment.message_provider_id or turn.provider_turn_id or None
            role = turn.role
            owner_coordinate = None
            if provider_id is None:
                if "ordinal" not in turn.model_fields_set or not 0 <= turn.ordinal < len(parsed.messages):
                    raise MessageOwnerAmbiguityError("capture attachment lacks a witnessed native message")
                native_message = parsed.messages[turn.ordinal]
                if native_message.role != turn.role or native_message.text != turn.text:
                    raise MessageOwnerAmbiguityError("capture attachment turn disagrees with its native message")
                provider_id = native_message.provider_message_id or None
                role = native_message.role
                owner_coordinate = _message_owner_coordinate(native_message, turn.ordinal)
            candidate = _browser_capture_parsed_attachment(attachment, message_provider_id=provider_id, role=role)
            candidate.owner_coordinate = owner_coordinate
            envelope_attachments.append(candidate)
    parsed_roles = {
        message.provider_message_id: message.role for message in parsed.messages if message.provider_message_id
    }
    envelope_attachments.extend(
        _browser_capture_parsed_attachment(
            attachment,
            message_provider_id=attachment.message_provider_id,
            role=parsed_roles.get(attachment.message_provider_id) if attachment.message_provider_id else None,
        )
        for attachment in envelope.session.attachments
    )
    if not envelope_attachments:
        return parsed
    merged: dict[str, ParsedAttachment] = {a.provider_attachment_id: a for a in parsed.attachments}
    # Keep cross-route matching one-to-one. Equal-name uploads can have equal
    # bytes too, so descriptor/byte evidence may be ambiguous; stable source
    # order still lets us retain one provider row per upload. Remembering the
    # synthetic id also folds a repeated envelope projection into that row.
    matched_native_ids: set[str] = set()
    cross_route_matches: dict[str, str] = {}
    # Descriptor index, so an unmatched candidate costs its own cohort rather
    # than a rescan of every merged row. Browser-captured content is
    # untrusted, and the previous full rescan made N same-descriptor rows
    # cost O(N^2) full-byte compares at parse time.
    #
    # The key is the pair ``_claude_attachment_cross_route_match`` requires to
    # be equal outright. ``size_bytes`` is deliberately NOT in the key: that
    # predicate only requires size equality when BOTH sides declare one, so
    # keying on it would drop legitimate matches where one side has no
    # declared size. It stays a predicate check inside the cohort.
    cohort: dict[tuple[str | None, str], list[str]] = defaultdict(list)
    for provider_attachment_id, native in merged.items():
        if native.name:
            cohort[(native.message_provider_id, native.name)].append(provider_attachment_id)
    for candidate in envelope_attachments:
        existing = merged.get(candidate.provider_attachment_id)
        if existing is None:
            matched_id = cross_route_matches.get(candidate.provider_attachment_id)
            if matched_id is not None and _claude_attachment_cross_route_match(merged[matched_id], candidate):
                existing = merged[matched_id]
            else:
                cross_route_ids = [
                    provider_attachment_id
                    for provider_attachment_id in (
                        cohort.get((candidate.message_provider_id, candidate.name), ()) if candidate.name else ()
                    )
                    if provider_attachment_id not in matched_native_ids
                    and _claude_attachment_cross_route_match(merged[provider_attachment_id], candidate)
                ]
                if cross_route_ids:
                    matched_id = cross_route_ids[0]
                    cross_route_matches[candidate.provider_attachment_id] = matched_id
                    matched_native_ids.add(matched_id)
                    existing = merged[matched_id]
                else:
                    merged[candidate.provider_attachment_id] = candidate
                    if candidate.name:
                        cohort[(candidate.message_provider_id, candidate.name)].append(candidate.provider_attachment_id)
                    continue
        # The native row remains authoritative for provider identity and file
        # metadata. The browser projection contributes acquired bytes and can
        # fill omissions, but must not replace native size/origin/file IDs.
        candidate_has_bytes = candidate.inline_bytes is not None or candidate.precomputed_blob is not None
        merged[existing.provider_attachment_id] = existing.model_copy(
            update={
                "message_provider_id": existing.message_provider_id or candidate.message_provider_id,
                "owner_coordinate": existing.owner_coordinate or candidate.owner_coordinate,
                "name": existing.name or candidate.name,
                "mime_type": existing.mime_type or candidate.mime_type,
                "size_bytes": existing.size_bytes if existing.size_bytes is not None else candidate.size_bytes,
                "provider_file_id": existing.provider_file_id or candidate.provider_file_id,
                "provider_drive_id": existing.provider_drive_id or candidate.provider_drive_id,
                "upload_origin": existing.upload_origin or candidate.upload_origin,
                "direction": existing.direction or candidate.direction,
                "producer_ref": existing.producer_ref or candidate.producer_ref,
                "attachment_kind": existing.attachment_kind or candidate.attachment_kind,
                "source_url": existing.source_url or candidate.source_url,
                "caption": existing.caption or candidate.caption,
                "inline_bytes": candidate.inline_bytes if candidate_has_bytes else existing.inline_bytes,
                "precomputed_blob": (candidate.precomputed_blob if candidate_has_bytes else existing.precomputed_blob),
            }
        )
    return parsed.model_copy(update={"attachments": list(merged.values())})


def _trusted_envelope_title(envelope: BrowserCaptureEnvelope) -> str | None:
    """Return only a provider-authored browser title.

    Current producers declare the source explicitly. Older v1 envelopes are
    accepted conservatively only when their title differs from both fallback
    values the extension is known to synthesize.
    """
    title = envelope.session.title
    if not title:
        return None
    if envelope.session.title_source == "provider":
        return title
    if envelope.session.title_source in {"page", "session-id"}:
        return None
    if title in {envelope.provenance.page_title, envelope.session.provider_session_id}:
        return None
    return title


def _merge_envelope_title(parsed: ParsedSession, envelope: BrowserCaptureEnvelope) -> ParsedSession:
    """Fill only a missing native title from trusted envelope evidence."""
    trusted_title = _trusted_envelope_title(envelope)
    fallback_titles = {None, "", parsed.provider_session_id, envelope.session.provider_session_id}
    if parsed.title not in fallback_titles or trusted_title is None:
        return parsed
    return parsed.model_copy(update={"title": trusted_title, "title_source": TitleSource.ORIGIN})


def _merge_envelope_native_metadata(parsed: ParsedSession, envelope: BrowserCaptureEnvelope) -> ParsedSession:
    """Use browser-envelope fields only when the native payload omitted them.

    The envelope fields are projections of the same provider response/page and
    must never create a second conversation identity or replace richer native
    values. They are useful for current Claude responses that omit optional
    title/model/timestamp fields from the embedded payload.
    """

    parsed = _merge_envelope_title(parsed, envelope)
    updates: dict[str, object] = {}
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
    # A captured turn's typed blocks (tool_use/tool_result/thinking/...)
    # project into the same "content" segment list a genuine Claude web
    # export's chat_messages carry, so normalize_chat_messages builds real
    # ParsedContentBlock rows instead of leaving the turn text-only
    # (polylogue-ah21). `text` above stays as the rendering.
    content_segments = _claude_raw_content_segments(turn)
    if content_segments is not None:
        raw["content"] = content_segments
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
            role=turn.role,
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
    fidelity_flag = DOM_FALLBACK_INGEST_FLAG
    trusted_title = _trusted_envelope_title(envelope)
    return ParsedSession(
        source_name=Provider.CLAUDE_AI,
        provider_session_id=provider_session_id,
        title=envelope.session.title or envelope.provenance.page_title or provider_session_id,
        title_source=TitleSource.ORIGIN if trusted_title is not None else None,
        session_kind=_session_kind_for_browser_capture(envelope, provider_session_id),
        created_at=created_at,
        updated_at=updated_at,
        messages=list(normalized.messages),
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


class _NativeWorkReader:
    """Non-owning borrowed reader that reports completed physical reads."""

    def __init__(self, handle: BinaryIO, progress: Callable[[], None]) -> None:
        self._handle = handle
        self._progress = progress

    @property
    def name(self) -> str:
        return str(self._handle.name)

    def fileno(self) -> int:
        return self._handle.fileno()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._handle.seek(offset, whence)

    def read(self, size: int = -1) -> bytes:
        result = self._handle.read(size)
        if result:
            self._progress()
        return result

    def readinto(self, buffer: bytearray) -> int | None:
        count = cast("_NativeReadInto", self._handle).readinto(buffer)
        if count:
            self._progress()
        return count


def parse_native_member_streams(
    provider: Provider,
    members: Mapping[str, BinaryIO],
    native_id: str,
    spill: ScratchSessionSpill,
    *,
    progress: Callable[[], None] | None = None,
) -> ParsedSession:
    """Use ordinary native normalization with the existing scratch collections.

    Callers borrow verified immutable raw members for this entire operation.
    Individual records, scalar values and native root metadata still
    materialize; this does not establish scalar-independent preparation.
    """
    import json
    import os

    import ijson

    from polylogue.sources.decoder_json import (
        _root_envelope_without,
        iter_root_array_items,
        normalize_ijson_stdlib_numbers,
    )
    from polylogue.sources.parsers import chatgpt, grok
    from polylogue.sources.parsers.claude.ai_parser import (
        looks_like_claude_memories,
        looks_like_claude_project,
        parse_ai,
        parse_ai_stream,
    )
    from polylogue.sources.prepared_message_sink import (
        ClaudeAttachmentScratch,
        ClaudeChatEvidence,
        ScratchSessionSpill,
        read_chatgpt_mapping_object,
    )

    if not isinstance(spill, ScratchSessionSpill):
        raise TypeError("native preparation requires its existing scratch owner")
    store = spill.store
    if progress is not None:
        members = {name: cast(BinaryIO, _NativeWorkReader(handle, progress)) for name, handle in members.items()}
    conversation = members["conversation"]
    conversation.seek(0)
    if provider is Provider.CHATGPT:
        extracted = read_chatgpt_mapping_object(
            conversation, store.conn, require_source_header=False, progress=progress
        )
        if extracted is None:
            raise ValueError("native ChatGPT raw lacks a mapping")
        envelope, mapping = extracted
        parsed = chatgpt.parse({**envelope, "mapping": mapping.shallow_view()}, native_id, spill=spill)
        for ordinal, message in enumerate(parsed.messages):
            if progress is not None:
                progress()
            if message.position is None:
                raise ValueError("canonical ChatGPT message lacks original mapping position")
            spill.set_record_origin(ordinal, str(message.position))
    elif provider is Provider.CLAUDE_AI:
        claude_extracted = _root_envelope_without(
            conversation,
            frozenset({"chat_messages", "attachments", "files"}),
            frozenset(),
            optional=frozenset({"attachments", "files"}),
        )
        if claude_extracted is None or claude_extracted[1]["chat_messages"] != 1:
            raise ValueError("native Claude raw lacks its message array")
        claude_envelope, arrays = claude_extracted
        # Keep the ordinary parser's internal route precedence even when
        # unrelated root metadata accompanies the declared message array.
        routing_header = {**claude_envelope, "chat_messages": []}
        if looks_like_claude_memories(routing_header) or looks_like_claude_project(routing_header):
            parsed = parse_ai(routing_header, native_id)
        else:
            evidence = ClaudeChatEvidence(store.conn)
            attachment_rows = ClaudeAttachmentScratch(store.conn)

            def attachments() -> Iterator[object]:
                for key in ("attachments", "files"):
                    if arrays[key]:
                        conversation.seek(0)
                        for item in iter_root_array_items(conversation, key):
                            if progress is not None:
                                progress()
                            yield item

            # Conversation attachments and chat messages use independent
            # cursors over the same immutable inode, never another revision.
            with open(conversation.name, "rb") as message_reader:
                original_stat = os.fstat(conversation.fileno())
                opened_stat = os.fstat(message_reader.fileno())
                if (original_stat.st_dev, original_stat.st_ino) != (opened_stat.st_dev, opened_stat.st_ino):
                    raise ValueError("native raw inode changed during preparation")
                try:
                    observed_messages = (
                        cast(BinaryIO, _NativeWorkReader(message_reader, progress))
                        if progress is not None
                        else message_reader
                    )

                    def message_rows() -> Iterator[object]:
                        for item in ijson.items(observed_messages, "chat_messages.item"):
                            if progress is not None:
                                progress()
                            yield normalize_ijson_stdlib_numbers(item)

                    parsed = parse_ai_stream(
                        claude_envelope,
                        message_rows(),
                        native_id,
                        conversation_attachments=attachments(),
                        evidence_store=evidence,
                        graph_connection=store.conn,
                        messages=store.new_sink(),
                        session_events=store.new_event_sink(),
                        attachment_rows=attachment_rows,
                        attachments=store.new_attachment_sink(),
                    )
                finally:
                    evidence.close()
                    attachment_rows.close()
    elif provider is Provider.GROK:
        original_conversation = json.load(conversation)
        responses = members["responses"]
        responses.seek(0)
        first = next(ijson.parse(responses), None)
        responses.seek(0)
        prefix = "item" if first is not None and first[1] == "start_array" else "responses.item"
        response_nodes = None
        if "response_nodes" in members:
            members["response_nodes"].seek(0)
            response_nodes = json.load(members["response_nodes"])

        def response_rows() -> Iterator[object]:
            for item in ijson.items(responses, prefix):
                if progress is not None:
                    progress()
                yield normalize_ijson_stdlib_numbers(item)

        parsed = grok.parse_native_response_stream(
            grok._native_conversation({"conversation": original_conversation}),
            response_rows(),
            native_id,
            response_nodes=response_nodes,
            spill=spill,
        )
    else:
        raise ValueError("unsupported native member preparation provider")
    return _require_matching_native_identity(parsed, native_id)


def _turn_needs_native_content_witness(turn: object) -> bool:
    if not isinstance(turn, Mapping):
        return False
    text = turn.get("text")
    return (
        (text is None or isinstance(text, str) and not text.strip())
        and not turn.get("attachments")
        and not turn.get("blocks")
    )


@parser_admission("browser_capture")
def parse(payload: object, fallback_id: str) -> ParsedSession:
    """Parse a browser-capture envelope into the canonical parser contract."""
    witnessed_native = None
    if isinstance(payload, Mapping) and isinstance(raw_session := payload.get("session"), Mapping):
        raw_turns = raw_session.get("turns")
        if isinstance(raw_turns, list) and any(_turn_needs_native_content_witness(turn) for turn in raw_turns):
            native_provider = Provider.from_string(str(raw_session.get("provider") or ""))
            declared_id = raw_session.get("provider_session_id")
            if not isinstance(declared_id, str):
                raise ValueError("native state-only turns require a declared conversation")
            native_id = legacy_browser_capture_native_id(native_provider, declared_id) or fallback_id
            witnessed_native = parse_native_payload(native_provider, payload.get("raw_provider_payload"), native_id)
    envelope = validate_capture_envelope(
        payload,
        native_witness=_CanonicalNativeTurnWitness(witnessed_native.messages) if witnessed_native is not None else None,
    )
    provider = envelope.session.provider if envelope.session.provider is not Provider.UNKNOWN else Provider.UNKNOWN
    provider_session_id = (
        legacy_browser_capture_native_id(provider, envelope.session.provider_session_id) or fallback_id
    )
    raw_provider_payload = envelope.raw_provider_payload

    def native_session() -> ParsedSession:
        prepared_attachment_ownership = provider is Provider.CHATGPT and any(
            "native_attachment_ordinal" in row.provider_meta for row in envelope.session.attachments
        )
        if prepared_attachment_ownership:
            return parse_native_payload(
                provider, raw_provider_payload, provider_session_id, prepared_attachment_ownership=True
            )
        return (
            witnessed_native
            if witnessed_native is not None
            else parse_native_payload(provider, raw_provider_payload, provider_session_id)
        )

    if provider is Provider.CODEX and raw_provider_payload is not None:
        return _merge_envelope_session_events(
            _apply_browser_capture_session_kind(
                _merge_envelope_attachments(
                    native_session(),
                    envelope,
                ),
                envelope,
                provider_session_id,
                has_native_payload=True,
            ),
            envelope,
        )
    if envelope.session.provider is Provider.CHATGPT and has_chatgpt_native_payload(raw_provider_payload):
        return _merge_envelope_session_events(
            _apply_browser_capture_session_kind(
                _merge_envelope_attachments(
                    _merge_envelope_title(
                        native_session(),
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
    if envelope.session.provider is Provider.CLAUDE_AI and has_claude_ai_native_payload(raw_provider_payload):
        return _merge_envelope_session_events(
            _apply_browser_capture_session_kind(
                _merge_envelope_attachments(
                    _merge_envelope_native_metadata(
                        native_session(),
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

    if provider is Provider.GROK and has_grok_native_payload(raw_provider_payload):
        return _merge_envelope_session_events(
            _apply_browser_capture_session_kind(
                _merge_envelope_attachments(native_session(), envelope),
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
    block_metadata_events: list[ParsedSessionEvent] = []
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
                blocks=_parsed_blocks_for_turn(turn),
            )
        )
        block_metadata_events.extend(
            _block_metadata_evidence_events(
                turn.blocks,
                source_message_provider_id=turn.provider_turn_id,
                timestamp=turn.timestamp,
            )
        )
        message_position += 1
        for attachment in turn.attachments:
            attachments.append(
                _browser_capture_parsed_attachment(
                    attachment,
                    message_provider_id=attachment.message_provider_id or turn.provider_turn_id,
                    role=turn.role,
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
    trusted_title = _trusted_envelope_title(envelope)
    return ParsedSession(
        source_name=provider,
        provider_session_id=provider_session_id,
        title=envelope.session.title or envelope.provenance.page_title or provider_session_id,
        title_source=TitleSource.ORIGIN if trusted_title is not None else None,
        session_kind=_session_kind_for_browser_capture(envelope, provider_session_id),
        created_at=envelope.session.created_at,
        updated_at=envelope.session.updated_at,
        messages=messages,
        active_leaf_message_provider_id=active_leaf_message_provider_id,
        attachments=attachments,
        session_events=[*_capture_session_events(envelope), *block_metadata_events],
        ingest_flags=[
            *dict.fromkeys(
                [
                    *_ingest_flags_for_browser_capture(envelope, provider_session_id),
                    DOM_FALLBACK_INGEST_FLAG,
                ]
            )
        ],
    )


__all__ = [
    "NativeCaptureIdentityMismatchError",
    "DOM_FALLBACK_INGEST_FLAG",
    "NATIVE_BROWSER_CAPTURE_INGEST_FLAG",
    "TEMPORARY_CHAT_INGEST_FLAG",
    "looks_like",
    "parse",
]


def detection_projection() -> DetectorProjection:
    """Preserve the capture discriminator and declared provider, consuming all payload bytes."""
    return DetectorProjection(
        fields={
            "polylogue_capture_kind": DetectorProjection(),
            "schema_version": DetectorProjection(),
            "session": DetectorProjection(fields={"provider": DetectorProjection()}),
            "provenance": DetectorProjection(),
        }
    )
