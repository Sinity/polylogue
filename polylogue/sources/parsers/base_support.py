"""Shared parser extraction helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from functools import wraps
from typing import Any, TypeVar

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, MessageType, ToolResultUnknownReason, WebConstructType
from polylogue.core.hashing import hash_text

from .base_models import (
    AdmissionDisposition,
    AdmissionOutcome,
    AdmissionRefusalReason,
    AdmissionUnit,
    AdmissionUnknownReason,
    ParseAccounting,
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
    ParsedWebConstruct,
)

_SessionParser = TypeVar("_SessionParser", bound=Callable[..., ParsedSession])


class AdmissionLedger:
    """Small mutable builder for the immutable parsed-session admission proof."""

    def __init__(self) -> None:
        self._expected: dict[AdmissionUnit, int] = {}
        self._outcomes: list[AdmissionOutcome] = []

    def expect(self, unit: AdmissionUnit, count: int) -> None:
        if count < 0:
            raise ValueError("admission denominators cannot be negative")
        self._expected[unit] = self._expected.get(unit, 0) + count

    def next_ordinal(self, unit: AdmissionUnit) -> int:
        return sum(1 for outcome in self._outcomes if outcome.unit is unit)

    def _record(
        self,
        unit: AdmissionUnit,
        ordinal: int,
        key: str,
        disposition: AdmissionDisposition,
        reason: AdmissionUnknownReason | AdmissionRefusalReason | None = None,
    ) -> AdmissionOutcome:
        outcome = AdmissionOutcome(
            unit=unit,
            ordinal=ordinal,
            key=key,
            disposition=disposition,
            reason=reason,
        )
        self._outcomes.append(outcome)
        return outcome

    def materialized(self, unit: AdmissionUnit, ordinal: int, key: str) -> AdmissionOutcome:
        return self._record(unit, ordinal, key, AdmissionDisposition.MATERIALIZED)

    def unknown(
        self,
        unit: AdmissionUnit,
        ordinal: int,
        key: str,
        reason: AdmissionUnknownReason = AdmissionUnknownReason.UNRECOGNIZED_TYPE,
    ) -> AdmissionOutcome:
        return self._record(unit, ordinal, key, AdmissionDisposition.TYPED_UNKNOWN, reason)

    def refusal(
        self,
        unit: AdmissionUnit,
        ordinal: int,
        key: str,
        reason: AdmissionRefusalReason,
    ) -> AdmissionOutcome:
        return self._record(unit, ordinal, key, AdmissionDisposition.TYPED_REFUSAL, reason)

    def close(self) -> ParseAccounting:
        accounting = ParseAccounting(expected=dict(self._expected), outcomes=list(self._outcomes))
        accounting.assert_conserved()
        return accounting


def _unknown_wire_type(value: object) -> str | None:
    """Return a deliberately future-shaped wire type, if one is visible.

    This is intentionally narrow.  Admission must not classify ordinary
    provider metadata as unknown merely because it contains a ``type`` field;
    the parser-specific lowering remains authoritative for known shapes.
    """
    if isinstance(value, dict):
        for key in ("type", "content_type", "kind", "record_type"):
            candidate = value.get(key)
            if isinstance(candidate, str) and (
                candidate.startswith(("future_", "unknown_", "unsupported_"))
                or candidate in {"future", "unknown", "unsupported"}
            ):
                return candidate
        for child in value.values():
            found = _unknown_wire_type(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _unknown_wire_type(child)
            if found is not None:
                return found
    return None


def _observe_wire_types(payload: Iterator[Any], observed: list[str | None]) -> Iterator[Any]:
    """Yield a one-pass payload through, recording each record's wire type."""
    for item in payload:
        observed.append(_unknown_wire_type(item))
        yield item


def parser_admission(provider: str) -> Callable[[_SessionParser], _SessionParser]:
    """Put a common conservation boundary around every session parser.

    Provider implementations are still responsible for their known wire
    shapes.  This boundary handles the class-killer case: a future-shaped
    outer envelope that the implementation does not recognize must become a
    typed session event, never vanish beside otherwise-valid content.
    """

    def decorate(parser: _SessionParser) -> _SessionParser:
        @wraps(parser)
        def wrapped(*args: Any, **kwargs: Any) -> ParsedSession:
            payload = args[0] if args else kwargs.get("payload")
            observed: list[str | None] = []
            if isinstance(payload, Iterator):
                # A one-pass payload can only be classified while the parser
                # pulls it; re-reading it afterwards would see an exhausted
                # iterator and undercount every record.
                instrumented = _observe_wire_types(payload, observed)
                if args:
                    args = (instrumented, *args[1:])
                else:
                    kwargs = {**kwargs, "payload": instrumented}
                session = parser(*args, **kwargs)
            else:
                raw_items = (
                    list(payload)
                    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes))
                    else [payload]
                )
                observed = [_unknown_wire_type(item) for item in raw_items]
                session = parser(*args, **kwargs)
            unknowns = [
                (index, wire_type) for index, wire_type in enumerate(observed, start=1) if wire_type is not None
            ]

            existing_types = {
                str(event.payload.get("wire_type"))
                for event in session.session_events
                if event.payload.get("wire_type") is not None
            }
            events = list(session.session_events)
            for index, wire_type in unknowns:
                if wire_type in existing_types:
                    continue
                events.append(
                    ParsedSessionEvent(
                        event_type=f"{provider}_unknown_input",
                        payload={"source_index": index, "wire_type": wire_type},
                    )
                )
                existing_types.add(wire_type)

            accounting = session.unit_accounting
            if accounting is None:
                ledger = AdmissionLedger()
                ledger.expect(AdmissionUnit.OUTER_RECORD, len(observed))
                for index, observed_type in enumerate(observed):
                    if observed_type is None:
                        ledger.materialized(AdmissionUnit.OUTER_RECORD, index, "parsed")
                    else:
                        ledger.unknown(AdmissionUnit.OUTER_RECORD, index, observed_type)
                accounting = ledger.close()
            accounting.assert_conserved()
            return session.model_copy(update={"session_events": events, "unit_accounting": accounting})

        return wrapped  # type: ignore[return-value]

    return decorate


def typed_unknown(
    unit: AdmissionUnit,
    ordinal: int,
    key: str,
    *,
    ledger: AdmissionLedger | None = None,
    reason: AdmissionUnknownReason = AdmissionUnknownReason.UNRECOGNIZED_TYPE,
) -> AdmissionOutcome:
    """Record an unknown through the one shared parser admission combinator."""
    if ledger is None:
        ledger = AdmissionLedger()
        ledger.expect(unit, 1)
    return ledger.unknown(unit, ordinal, key, reason)


def typed_unknown_block(
    value: object,
    *,
    wire_type: str | None = None,
    ledger: AdmissionLedger | None = None,
    ordinal: int = 0,
) -> ParsedContentBlock:
    """Retain an unrecognized structured segment without guessing its meaning."""
    observed_type = wire_type
    if observed_type is None and isinstance(value, dict):
        raw_type = value.get("type") or value.get("content_type")
        observed_type = raw_type if isinstance(raw_type, str) and raw_type else "unknown"
    observed_type = observed_type or "unknown"
    typed_unknown(
        AdmissionUnit.BLOCK,
        ordinal,
        observed_type,
        ledger=ledger,
        reason=AdmissionUnknownReason.UNRECOGNIZED_TYPE,
    )
    return ParsedContentBlock(
        type=BlockType.DOCUMENT,
        metadata={
            "admission_disposition": AdmissionDisposition.TYPED_UNKNOWN.value,
            "unknown_reason": AdmissionUnknownReason.UNRECOGNIZED_TYPE.value,
            "wire_type": observed_type,
        },
    )


def human_authored_override(
    role: Role,
    message_type: MessageType,
    material_origin: MaterialOrigin,
) -> MaterialOrigin:
    """Positive-evidence override for a plain user turn (polylogue-gzgyl).

    ``classify_material_origin`` (``archive/message/artifacts.py``) stopped
    falling through role=user/MESSAGE rows to ``HUMAN_AUTHORED`` (PR #2502,
    correct for agent runtimes where a user-shaped row can be
    generated/relayed context, not a real human turn). Consumer chat-export
    origins have no such ambiguity: a plain user message IS positive human
    evidence there, so this mirrors the compensating override Codex
    (``_codex_material_origin``) and Claude Code
    (``_claude_code_user_turn_origin``) already carry, for parsers whose
    genuine user-turn shape has no agent/subagent complexity to exclude.
    Callers pass the *already-classified* origin so this stays a pure
    positive-evidence bump, never a replacement for ``classify_material_origin``.
    """
    if material_origin is MaterialOrigin.UNKNOWN and role is Role.USER and message_type is MessageType.MESSAGE:
        return MaterialOrigin.HUMAN_AUTHORED
    return material_origin


def text_blocks_prose(blocks: Sequence[ParsedContentBlock]) -> str | None:
    """Join only TEXT-type block text, in position order, with ``\\n``.

    This is the parse-time twin of
    ``polylogue.storage.embeddings.materialization.message_prose_sql``
    (``block_types=("text",)``, ``separator="'\\n'"``), keeping the
    classifier input aligned with the persisted TEXT blocks.

    Anything that folds THINKING/TOOL_USE/TOOL_RESULT segments into the
    same string handed to ``classify_text_message_type`` (e.g. a combined
    "full record text" built before blocks are split apart) can see markers
    that are not part of the message's authored prose. Build the input from
    the message's own already-split content blocks instead.
    Callers that classify a message's runtime-artifact type from its text
    must build that text from the message's own already-split
    ``ParsedContentBlock`` list via this helper, not from a separately
    reconstructed "all segment types" string.
    """
    parts = [block.text for block in blocks if block.type is BlockType.TEXT and block.text]
    return "\n".join(parts) if parts else None


def synthetic_message_id(
    *,
    role: Role,
    text: str | None,
    timestamp: str | None,
    namespace: str = "",
    kind: str = "",
) -> str:
    """Build a reorder-stable id for a message with no provider id.

    This is reserved for parser-produced rows that are inherently synthetic,
    such as an exported summary or a transcript section. Native-id fallback
    paths must pass an empty string instead, so ``pipeline.ids`` can use its
    role/timestamp/text comparison anchor.
    """
    seed = "\x1f".join((namespace, str(role), timestamp or "", text or "", kind))
    return f"synthetic-{hash_text(seed)[:24]}"


def fill_linear_parent_chain(messages: Sequence[ParsedMessage]) -> list[ParsedMessage]:
    """Backfill linear parent evidence for a strictly linear message list.

    Gemini CLI, Grok, and AI Studio Drive's non-branch path never assert an
    explicit reply-to edge, because their session shape is a plain ordered
    turn sequence with no fork/retry concept at the message level -- there is
    no ``variant_index>0`` row in any of them today (ksgg finding B). Leaving
    ``parent_message_provider_id`` at ``None`` for every message makes
    position-order the ONLY way to reconstruct the conversation shape for
    these paths, unlike Claude Code / ChatGPT where a real parent chain is
    carried end to end.

    This fills the trivial, unambiguous case -- chaining each message to the
    previous message on the same active path -- without fabricating branch
    structure: a message that already carries real parent evidence (e.g.
    ``parsers/drive.py``'s explicit Gemini branch chunks) is left untouched,
    and only a message with ``parent_message_provider_id is None`` is
    chained to the nearest preceding *active-path* message. A session's
    first message (and any message with no preceding active-path message)
    keeps ``parent_message_provider_id=None`` -- there is nothing to chain
    it to.
    """
    filled: list[ParsedMessage] = []
    previous_active_message: ParsedMessage | None = None
    for message in messages:
        if message.parent_message_provider_id is None and previous_active_message is not None:
            if previous_active_message.provider_message_id:
                message = message.model_copy(
                    update={"parent_message_provider_id": previous_active_message.provider_message_id}
                )
            elif previous_active_message.position is not None:
                message = message.model_copy(update={"parent_message_position": previous_active_message.position})
        filled.append(message)
        if message.is_active_path is not False:
            previous_active_message = message
    return filled


def content_blocks_from_segments(
    content: object,
    *,
    admission: AdmissionLedger | None = None,
    lower_transport_text: bool = False,
) -> list[ParsedContentBlock]:
    """Convert raw API content (str, list, dict) to ParsedContentBlock list.

    Provider parsers normally lower transport-only text items themselves. A
    parser that needs the shared converter to retain those items can opt in;
    this keeps the default behavior for providers whose overlay has richer
    transport-item handling.
    """
    if isinstance(content, str):
        if admission is not None:
            part_ordinal = admission.next_ordinal(AdmissionUnit.PART)
            admission.expect(AdmissionUnit.PART, 1)
            admission.materialized(AdmissionUnit.PART, part_ordinal, "text")
            admission.expect(AdmissionUnit.BLOCK, 1 if content else 0)
            if content:
                admission.materialized(AdmissionUnit.BLOCK, admission.next_ordinal(AdmissionUnit.BLOCK), "text")
        return [ParsedContentBlock(type=BlockType.TEXT, text=content)] if content else []
    if not isinstance(content, list):
        return []
    part_offset = admission.next_ordinal(AdmissionUnit.PART) if admission is not None else 0
    if admission is not None:
        admission.expect(AdmissionUnit.PART, len(content))
    blocks: list[ParsedContentBlock] = []
    known_types = {
        "thinking",
        "tool_use",
        "tool_result",
        "text",
        "image",
        "document",
        "token_budget",
        "voice_note",
        "code",
        "input_text",
        "output_text",
        "input_image",
    }
    for part_ordinal, seg in enumerate(content):
        if admission is not None:
            if isinstance(seg, str):
                admission.materialized(AdmissionUnit.PART, part_offset + part_ordinal, "text")
            elif (
                isinstance(seg, dict)
                and isinstance(seg.get("type", "text"), str)
                and seg.get("type", "text") in known_types
            ):
                admission.materialized(AdmissionUnit.PART, part_offset + part_ordinal, str(seg.get("type", "text")))
            else:
                admission.unknown(
                    AdmissionUnit.PART,
                    part_offset + part_ordinal,
                    "unsupported",
                    AdmissionUnknownReason.UNSUPPORTED_SHAPE,
                )
        if isinstance(seg, str):
            if seg:
                blocks.append(ParsedContentBlock(type=BlockType.TEXT, text=seg))
            continue
        if not isinstance(seg, dict):
            continue
        seg_type = seg.get("type", "text")
        if seg_type == "text":
            text = seg.get("text") or ""
            if text:
                blocks.append(ParsedContentBlock(type=BlockType.TEXT, text=str(text)))
        elif seg_type == "thinking":
            text = seg.get("thinking") or seg.get("text") or ""
            signature = seg.get("signature")
            # polylogue-vf9x: since roughly 2026-06 the wire ships thinking
            # blocks with an empty `thinking` body and only a `signature` --
            # the reasoning genuinely occurred but its text is not on the
            # wire (verified against raw ~/.claude/projects JSONL: Feb-2026
            # sessions carry non-empty text, Jul-2026 sessions are 100%
            # empty-body/signature-only). Previously this `if text:` guard
            # dropped the block outright, silently zeroing thinking_count
            # and making the archive look like reasoning stopped -- record
            # the block regardless so the fact that the model reasoned here
            # (and the signature, for provenance) survives even without text.
            blocks.append(
                ParsedContentBlock(
                    type=BlockType.THINKING,
                    text=text or None,
                    signature=signature if isinstance(signature, str) and signature else None,
                )
            )
        elif seg_type == "tool_use":
            tool_name = seg.get("name")
            tool_id = seg.get("id")
            tool_input = seg.get("input") if isinstance(seg.get("input"), dict) else None
            if tool_name or tool_id or tool_input:
                blocks.append(
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name=tool_name,
                        tool_id=tool_id,
                        tool_input=tool_input,
                    )
                )
        elif seg_type == "tool_result":
            result_content = seg.get("content")
            result_text = None
            knowledge_constructs: list[ParsedWebConstruct] = []
            if isinstance(result_content, str):
                result_text = result_content
            elif isinstance(result_content, list):
                text_parts = [
                    block.get("text", "")
                    for block in result_content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                result_text = "\n".join(part for part in text_parts if part) or None
                # polylogue-zocm: web_search tool_result content also carries
                # retrieved-source entries the provider read but did not
                # necessarily cite in the answer text -- {type: knowledge,
                # title, url, metadata: {site_domain, site_name,
                # favicon_url}, text} (1,514 measured). Distinct from the
                # answer-text `citations` anchors ``_citation_construct``
                # (claude/common.py) projects as CONTENT_REFERENCE:
                # SEARCH_RESULT here means "retrieved", not "cited", so
                # "sources read" stays queryable separately from "sources
                # cited" instead of being flattened together.
                for rank, block in enumerate(result_content):
                    if not isinstance(block, dict) or block.get("type") != "knowledge":
                        continue
                    knowledge_title = block.get("title")
                    knowledge_url = block.get("url")
                    knowledge_text = block.get("text")
                    knowledge_metadata = block.get("metadata")
                    site_name = (
                        knowledge_metadata.get("site_name") or knowledge_metadata.get("site_domain")
                        if isinstance(knowledge_metadata, dict)
                        else None
                    )
                    knowledge_constructs.append(
                        ParsedWebConstruct(
                            construct_type=WebConstructType.SEARCH_RESULT,
                            provider_key="web_search_knowledge",
                            title=knowledge_title if isinstance(knowledge_title, str) else None,
                            url=knowledge_url if isinstance(knowledge_url, str) else None,
                            text=knowledge_text if isinstance(knowledge_text, str) else None,
                            group_title=site_name if isinstance(site_name, str) else None,
                            rank=rank,
                        )
                    )
            raw_is_error = seg.get("is_error")
            is_error = raw_is_error if isinstance(raw_is_error, bool) else None
            # polylogue-2qx.4 / polylogue-cuxz.8: this is the shared
            # Anthropic-protocol tool_result segment shape (Claude Code,
            # Claude common, Codex). When the segment itself carries no
            # boolean ``is_error`` the provider structurally emitted nothing
            # for this record -- NOT_REPORTED, not a bare unknown. Origin-
            # specific overlays (e.g. Claude Code's own toolUseResult
            # verdicts) may resolve or override this afterward.
            outcome_unknown_reason = None if is_error is not None else ToolResultUnknownReason.NOT_REPORTED.value
            blocks.append(
                ParsedContentBlock(
                    type=BlockType.TOOL_RESULT,
                    tool_id=seg.get("tool_use_id"),
                    text=result_text,
                    is_error=is_error,
                    outcome_unknown_reason=outcome_unknown_reason,
                    web_constructs=knowledge_constructs,
                )
            )
        elif seg_type in ("image", "document"):
            block_type = BlockType.from_string(seg_type)
            # bd polylogue-9x22: this ``metadata`` dict is never persisted --
            # the ``blocks`` table has no metadata column and the write path
            # only reads a ``language`` key back out of it -- but unlike
            # every other polylogue-9x22 site, it is NOT routed to
            # session_events here. The Anthropic-protocol image/document
            # segment shape's remaining keys after `type`/`media_type` are
            # dominated by `source` (the inline base64 payload itself, or a
            # file/url reference already captured by the attachment
            # pipeline) -- verbatim-copying this dict the way the other
            # sites do would duplicate large binary/attachment data into a
            # durable evidence table meant for small JSON payloads. Re-audit
            # with real corpus evidence if a genuinely small, non-blob,
            # non-attachment-duplicate field is ever found on these segments.
            blocks.append(
                ParsedContentBlock(
                    type=block_type,
                    media_type=seg.get("media_type"),
                )
            )
        elif seg_type == "token_budget":
            remaining = seg.get("remaining")
            if remaining is not None:
                blocks.append(
                    ParsedContentBlock(
                        type=BlockType.TEXT,
                        text=f"[Claude token budget remaining: {remaining}]",
                        web_constructs=[
                            ParsedWebConstruct(
                                construct_type=WebConstructType.TOKEN_BUDGET,
                                provider_key="token_budget",
                                text=str(remaining),
                            )
                        ],
                    )
                )
        elif seg_type == "voice_note":
            text = seg.get("text") or ""
            title = seg.get("title")
            if text or title:
                blocks.append(
                    ParsedContentBlock(
                        type=BlockType.TEXT,
                        text=str(text or title),
                        web_constructs=[
                            ParsedWebConstruct(
                                construct_type=WebConstructType.VOICE_NOTE,
                                provider_key="voice_note",
                                title=str(title) if title else None,
                                text=str(text) if text else None,
                            )
                        ],
                    )
                )
        elif seg_type == "code":
            text = seg.get("text") or seg.get("code") or ""
            if text:
                metadata: dict[str, object] | None = None
                language = seg.get("language")
                if isinstance(language, str) and language:
                    metadata = {"language": language}
                blocks.append(ParsedContentBlock(type=BlockType.CODE, text=str(text), metadata=metadata))
        elif seg_type in ("input_text", "output_text", "input_image", "image"):
            # Provider transport items are lowered by the owning parser:
            # Codex supplies text through ``extract_codex_text`` and bounded
            # image evidence through ``_codex_inline_image_blocks``.
            if lower_transport_text and seg_type in {"input_text", "output_text"}:
                text = seg.get("text") or ""
                if text:
                    blocks.append(ParsedContentBlock(type=BlockType.TEXT, text=str(text)))
            continue
        else:
            # Unknown structured content is still an input unit. Preserve its
            # existence as an opaque DOCUMENT block with a typed disposition;
            # callers must never infer prose from a future wire shape.
            unknown_block = typed_unknown_block(seg, wire_type=str(seg_type))
            if isinstance(seg.get("text"), str) and seg["text"]:
                unknown_block = unknown_block.model_copy(update={"type": BlockType.TEXT, "text": seg["text"]})
            blocks.append(unknown_block)
    if admission is not None:
        admission.expect(AdmissionUnit.BLOCK, len(blocks))
        for block in blocks:
            block_ordinal = admission.next_ordinal(AdmissionUnit.BLOCK)
            if (
                block.metadata
                and block.metadata.get("admission_disposition") == AdmissionDisposition.TYPED_UNKNOWN.value
            ):
                admission.unknown(
                    AdmissionUnit.BLOCK,
                    block_ordinal,
                    str(block.metadata.get("wire_type") or "unknown"),
                )
            else:
                admission.materialized(AdmissionUnit.BLOCK, block_ordinal, block.type.value)
    return blocks


def _make_attachment_id(seed: str) -> str:
    return f"att-{hash_text(seed)[:12]}"


def attachment_from_meta(meta: object, message_id: str | None) -> ParsedAttachment | None:
    if not isinstance(meta, dict):
        return None
    attachment_id = (
        meta.get("id") or meta.get("file_id") or meta.get("fileId") or meta.get("uuid") or meta.get("file_uuid")
    )
    name = meta.get("name") or meta.get("filename") or meta.get("file_name")
    mime_type = meta.get("mimeType") or meta.get("mime_type") or meta.get("content_type") or meta.get("file_type")
    if not attachment_id:
        if not name:
            return None
        # polylogue-hith: identity must be a property of the attachment
        # itself, not of its position in whichever bucket ("attachments" vs
        # "files") or order the export happened to walk them in. `index` used
        # to be part of this seed, so re-ordering an export's attachment list
        # -- observed happening between vintages of the SAME conversation --
        # minted a different id for the same physical attachment even though
        # nothing about it changed. `mime_type` is included instead: it is
        # read directly from the export's own metadata (never lazily
        # acquired the way `size`/inline bytes can be), so it adds real
        # disambiguation without reintroducing acquisition-state instability.
        # Two un-identified attachments sharing both name and mime_type on
        # the same message are genuinely indistinguishable from the metadata
        # available here; collapsing them to one identity is the honest
        # outcome of that, not a regression -- see polylogue-hith for the
        # full trade-off discussion (a real-id-havingness axis is a separate,
        # unfixed failure mode filed as a follow-up there).
        seed = f"{message_id or 'msg'}:{name}:{mime_type or ''}"
        attachment_id = _make_attachment_id(seed)
    size_raw = meta.get("size") or meta.get("size_bytes") or meta.get("sizeBytes") or meta.get("file_size")
    size_bytes = None
    if isinstance(size_raw, (int, str)):
        try:
            size_bytes = int(size_raw)
        except ValueError:
            size_bytes = None
    inline_bytes = None
    extracted_content = meta.get("extracted_content")
    if isinstance(extracted_content, str):
        inline_bytes = extracted_content.encode("utf-8")
        if size_bytes is None:
            size_bytes = len(inline_bytes)
    # #1252: promote native identifiers when present. claude-code/codex
    # attachments arrive via OAuth-authenticated session/export.
    file_id_raw = meta.get("file_id") or meta.get("fileId") or meta.get("file_uuid")
    drive_id_raw = meta.get("drive_id") or meta.get("driveId")
    return ParsedAttachment(
        provider_attachment_id=str(attachment_id),
        message_provider_id=message_id,
        name=name,
        mime_type=mime_type if isinstance(mime_type, str) else None,
        size_bytes=size_bytes,
        path=None,
        provider_file_id=str(file_id_raw) if isinstance(file_id_raw, str) and file_id_raw else None,
        provider_drive_id=str(drive_id_raw) if isinstance(drive_id_raw, str) and drive_id_raw else None,
        upload_origin="oauth",
        inline_bytes=inline_bytes,
    )


def extract_messages_from_list(items: Sequence[object]) -> list[ParsedMessage]:
    messages: list[ParsedMessage] = []
    for _idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue

        message_val = item.get("message")
        payload = message_val if isinstance(message_val, dict) else item

        role = Role.normalize(
            str(
                payload.get("role")
                or item.get("role")
                or payload.get("sender")
                or item.get("sender")
                or payload.get("author")
                or item.get("author")
                or "unknown"
            )
        )

        timestamp = (
            item.get("timestamp")
            or payload.get("timestamp")
            or payload.get("created_at")
            or item.get("created_at")
            or payload.get("create_time")
            or item.get("create_time")
        )

        text = None
        content_blocks: list[ParsedContentBlock] = []
        text_val = payload.get("text")
        if text_val is not None and isinstance(text_val, str):
            text = text_val
            if text:
                content_blocks = [ParsedContentBlock(type=BlockType.TEXT, text=text)]
        else:
            content = payload.get("content")
            if isinstance(content, str):
                text = content
                if text:
                    content_blocks = [ParsedContentBlock(type=BlockType.TEXT, text=text)]
            elif isinstance(content, dict):
                parts = content.get("parts")
                if isinstance(parts, list):
                    texts: list[str] = []
                    for part in parts:
                        if isinstance(part, str) and part:
                            texts.append(part)
                            content_blocks.append(ParsedContentBlock(type=BlockType.TEXT, text=part))
                        elif isinstance(part, dict):
                            part_text = part.get("text")
                            if isinstance(part_text, str) and part_text:
                                texts.append(part_text)
                                content_blocks.append(ParsedContentBlock(type=BlockType.TEXT, text=part_text))
                    text = "\n".join(texts) or None
                else:
                    text_dict_val = content.get("text")
                    if text_dict_val is not None and isinstance(text_dict_val, str):
                        text = text_dict_val
                        if text:
                            content_blocks = [ParsedContentBlock(type=BlockType.TEXT, text=text)]
            elif isinstance(content, list):
                content_blocks = content_blocks_from_segments(content)
                text = "\n".join(block.text for block in content_blocks if block.text) or None

        if text:
            # polylogue-slshy: no positional fallback -- empty id lets
            # _message_revision_match_id's content-derived anchor fallback run
            # instead of a position-derived string that would change identity
            # when array order shifts across re-acquisitions.
            msg_id = str(payload.get("id") or payload.get("uuid") or item.get("uuid") or item.get("id") or "")
            messages.append(
                ParsedMessage(
                    provider_message_id=msg_id,
                    role=role,
                    text=text,
                    timestamp=str(timestamp) if timestamp is not None else None,
                    blocks=content_blocks,
                )
            )
    return messages


def mark_last_occurrence_as_active_leaf(messages: list[ParsedMessage]) -> list[ParsedMessage]:
    """Flag exactly one message as ``is_active_leaf``: the LAST message in
    the list, by position -- never by comparing ``provider_message_id``.

    ``provider_message_id`` is not guaranteed unique across a flat message
    list assembled by concatenating streaming chunks or markdown sections --
    retries/variants/regenerations can legitimately reuse the same native id
    at more than one position (bd polylogue-2hwl). Comparing every message's
    id against ``messages[-1].provider_message_id`` (the naive approach)
    flags EVERY matching position, not just the true leaf, which then lets
    more than one ``is_active_leaf=True`` message reach MCP payloads and
    archive_query message output -- an invariant violation (at most one
    active leaf per session). Matching by exact list position instead of by
    id equality alone keeps the flag unique regardless of duplicate ids.
    """
    if not messages:
        return messages
    leaf_index = len(messages) - 1
    return [
        message.model_copy(update={"is_active_leaf": index == leaf_index}) for index, message in enumerate(messages)
    ]
