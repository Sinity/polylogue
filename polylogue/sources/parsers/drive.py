from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable, Iterable, MutableSequence, Sequence
from contextlib import closing
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import cast

from pydantic import ValidationError

from polylogue.archive.message.artifacts import classify_block_message_type, classify_material_origin
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.core.enums import Provider, TitleSource
from polylogue.core.json import JSONDocument, json_document
from polylogue.core.message_owner import MessageOwnerCoordinate
from polylogue.core.timestamps import parse_timestamp
from polylogue.logging import get_logger
from polylogue.sources.detection_projection import DetectorProjection
from polylogue.sources.providers.gemini import GeminiMessage

from .base import (
    ParsedAttachment,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
    human_authored_override,
    parser_admission,
)
from .base_models import upgrade_chat_export_user_authorship
from .base_support import AdmissionObserver, _unknown_wire_type
from .drive_support import (
    TimestampBounds,
    extract_text_from_chunk,
)
from .drive_support import (
    _attachment_from_doc as _attachment_from_doc_impl,
)
from .drive_support import (
    _collect_drive_docs as _collect_drive_docs_impl,
)
from .drive_support import (
    attachment_block_payloads as _attachment_block_payloads,
)
from .drive_support import (
    chunk_timestamp as _chunk_timestamp,
)
from .drive_support import (
    collect_chunk_attachments as _collect_chunk_attachments,
)
from .drive_support import (
    parsed_blocks_from_meta as _parsed_blocks_from_meta,
)
from .drive_support import (
    session_events_from_meta_blocks as _session_events_from_meta_blocks,
)
from .drive_support import (
    viewport_block_payload as _viewport_block_payload,
)

_logger = get_logger(__name__)

_CHUNK_CONTENT_KEYS = frozenset(
    {
        "text",
        "parts",
        "executableCode",
        "codeExecutionResult",
        "driveDocument",
        "driveImage",
        "driveAudio",
        "driveVideo",
        "inlineFile",
        "inlineImage",
        "youtubeVideo",
        "errorMessage",
        "grounding",
        "isThought",
    }
)


_EPOCH_FLOOR = datetime.min.replace(tzinfo=timezone.utc)


def _sort_instant(value: str | None) -> datetime:
    """Chain-ordering key: parsed instant, unparseable/missing floor to the start.

    Sorting the raw strings put '+02:00' offsets after 'Z' instants and
    reversed a persisted parent chain; position stays the tiebreak so
    unparseable values keep their original order.
    """
    parsed = parse_timestamp(value) if isinstance(value, str) and value else None
    return parsed if parsed is not None else _EPOCH_FLOOR


def _collect_drive_docs(payload: object) -> list[JSONDocument | str]:
    return _collect_drive_docs_impl(payload)


def _attachment_from_doc(doc: JSONDocument | str, message_id: str | None) -> ParsedAttachment | None:
    return _attachment_from_doc_impl(doc, message_id)


def _gemini_content_block_payloads(message: GeminiMessage, text: str | None) -> list[JSONDocument]:
    content_block_payloads = [
        block_payload
        for content_block in message.extract_content_blocks()
        if (block_payload := _viewport_block_payload(content_block)) is not None
    ]
    if content_block_payloads:
        return content_block_payloads
    if not text:
        return []
    return [{"type": "thinking" if message.isThought else "text", "text": text}]


def _fallback_gemini_content_blocks(chunk_obj: JSONDocument, text: str | None) -> list[JSONDocument]:
    fallback_content_blocks: list[JSONDocument] = []
    if text:
        block_type = "thinking" if chunk_obj.get("isThought") else "text"
        fallback_content_blocks.append({"type": block_type, "text": text})

    exec_code = chunk_obj.get("executableCode")
    if isinstance(exec_code, dict) and exec_code:
        code = exec_code.get("code")
        if isinstance(code, str) and code:
            fallback_content_blocks.append({"type": "code", "text": code})

    exec_result = chunk_obj.get("codeExecutionResult")
    if isinstance(exec_result, dict) and exec_result:
        output = exec_result.get("output")
        outcome = exec_result.get("outcome")
        # ``outcome`` is the execution's structural verdict (OUTCOME_OK /
        # OUTCOME_FAILED / ...). Carry it as metadata, not only rendered into
        # the text, so the block's outcome is read from the field.
        outcome_metadata: JSONDocument = (
            {"metadata": {"outcome": outcome}} if isinstance(outcome, str) and outcome else {}
        )
        if isinstance(output, str) and output:
            fallback_content_blocks.append({"type": "tool_result", "text": output, **outcome_metadata})
        elif isinstance(outcome, str) and outcome:
            fallback_content_blocks.append({"type": "tool_result", "text": f"[{outcome}]", **outcome_metadata})

    return fallback_content_blocks


def _append_attachment_blocks(
    content_blocks: list[JSONDocument], chunk_attachments: list[ParsedAttachment]
) -> list[JSONDocument]:
    # Attachment block payloads carry only JSON-valued fields (name/ids/mime);
    # cast bridges the DriveDocMetadata (dict[str, object]) alias to JSONDocument.
    return content_blocks + cast("list[JSONDocument]", _attachment_block_payloads(chunk_attachments))


def _string_field(payload: JSONDocument, *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _non_negative_int_field(payload: JSONDocument, *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int):
            return value if value >= 0 else None
        if isinstance(value, float):
            return int(value) if math.isfinite(value) and value.is_integer() and value >= 0 else None
        if isinstance(value, str):
            try:
                parsed = int(value)
            except ValueError:
                try:
                    number = Decimal(value)
                    if (
                        not number.is_finite()
                        or not math.isfinite(float(number))
                        or number != number.to_integral_value()
                    ):
                        continue
                    parsed = int(number)
                except (InvalidOperation, ValueError, OverflowError):
                    continue
            return parsed if parsed >= 0 else None
    return None


def _usage_fields(chunk_obj: JSONDocument, *, role: Role) -> dict[str, int | None]:
    token_count = _non_negative_int_field(chunk_obj, "tokenCount", "token_count")
    if token_count is None:
        return {"input_tokens": None, "output_tokens": None}
    if role is Role.USER:
        return {"input_tokens": token_count, "output_tokens": 0}
    return {"input_tokens": 0, "output_tokens": token_count}


def _branch_parent_message_provider_id(chunk_obj: JSONDocument) -> str | None:
    """Return only a branch parent's same-session message identity."""
    branch_parent_obj = json_document(chunk_obj.get("branchParent"))
    # Drive's promptId names a parent prompt/session, never a local message.
    return _string_field(branch_parent_obj, "id", "messageId")


def _branch_parent_session_provider_id(chunk_obj: JSONDocument) -> str | None:
    """Return a branch parent's source-asserted prompt/session identity."""
    return _string_field(json_document(chunk_obj.get("branchParent")), "promptId")


def _branch_child_provider_id(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return _string_field(json_document(value), "id", "messageId")


def _text_key(value: str) -> bytes:
    return value.encode("utf-8", "surrogatepass")


class _ChunkOrder:
    """One chunked prompt's branch declarations and per-message ordering rows.

    Branch evidence and the linear parent chain need every chunk. Both live
    in SQLite, so Python holds one chunk at a time whatever the prompt's
    length; the object parser runs the same rows over an in-memory
    connection.
    """

    _TABLES = ("drive_branch_candidate", "drive_prompt_child", "drive_unresolved", "drive_prompt_parent", "drive_chain")

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        conn.execute(
            "CREATE TABLE drive_branch_candidate (child_id BLOB NOT NULL, parent_id BLOB NOT NULL, "
            "PRIMARY KEY (child_id, parent_id)) WITHOUT ROWID"
        )
        conn.execute("CREATE TABLE drive_prompt_child (child_id BLOB PRIMARY KEY) WITHOUT ROWID")
        conn.execute("CREATE TABLE drive_unresolved (message_id BLOB PRIMARY KEY) WITHOUT ROWID")
        conn.execute("CREATE TABLE drive_prompt_parent (parent_id BLOB PRIMARY KEY) WITHOUT ROWID")
        # (chain instant, position, provider id, explicit parent) per message.
        conn.execute(
            "CREATE TABLE drive_chain (position INTEGER PRIMARY KEY, instant INTEGER NOT NULL, "
            "message_id BLOB NOT NULL, explicit_parent INTEGER NOT NULL)"
        )

    def record_branch_declarations(self, chunks: Iterable[object]) -> None:
        """Record every ``branchChildren`` declaration.

        A child declared under a prompt (``promptId``) names another
        session, never a local message. A child declared under more than one
        parent's ``branchChildren`` has no real single-parent evidence: it is
        left unresolved AND kept out of the later linear gap-fill (bd
        polylogue-ksgg), which cannot otherwise tell "None because ambiguous"
        from "None because no branch data existed at all".
        """
        for chunk in chunks:
            chunk_obj = json_document(chunk)
            branch_children = chunk_obj.get("branchChildren")
            if not isinstance(branch_children, list):
                continue
            parent_id = _string_field(chunk_obj, "id")
            for child in branch_children:
                prompt_id = _string_field(json_document(child), "promptId")
                if prompt_id is not None:
                    self._conn.execute("INSERT OR IGNORE INTO drive_prompt_child VALUES (?)", (_text_key(prompt_id),))
                child_id = _branch_child_provider_id(child)
                if parent_id is not None and child_id is not None:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO drive_branch_candidate VALUES (?, ?)",
                        (_text_key(child_id), _text_key(parent_id)),
                    )

    def _candidates(self, message_id: str) -> tuple[bytes | None, int]:
        parent, count = self._conn.execute(
            "SELECT MIN(parent_id), COUNT(*) FROM drive_branch_candidate WHERE child_id = ?", (_text_key(message_id),)
        ).fetchone()
        return parent, count

    def _is_prompt_child(self, message_id: str) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM drive_prompt_child WHERE child_id = ?", (_text_key(message_id),)
            ).fetchone()
            is not None
        )

    def branch_parent(self, message_id: str) -> str | None:
        """The one parent that declares ``message_id`` among its branch children."""
        parent, count = self._candidates(message_id)
        if count != 1 or parent is None or self._is_prompt_child(message_id):
            return None
        return parent.decode("utf-8", "surrogatepass")

    def mark_unresolved(self, message_id: str) -> None:
        """Exclude a message whose branch parent names a prompt from the gap-fill."""
        self._conn.execute("INSERT OR IGNORE INTO drive_unresolved VALUES (?)", (_text_key(message_id),))

    def add_prompt_parent(self, prompt_id: str) -> None:
        self._conn.execute("INSERT OR IGNORE INTO drive_prompt_parent VALUES (?)", (_text_key(prompt_id),))

    def only_prompt_parent(self) -> str | None:
        rows = self._conn.execute("SELECT parent_id FROM drive_prompt_parent LIMIT 2").fetchall()
        return rows[0][0].decode("utf-8", "surrogatepass") if len(rows) == 1 else None

    def add_chain_row(self, instant: datetime, position: int, message_id: str, explicit_parent: bool) -> None:
        self._conn.execute(
            "INSERT INTO drive_chain VALUES (?, ?, ?, ?)",
            (position, (instant - _EPOCH_FLOOR) // timedelta(microseconds=1), _text_key(message_id), explicit_parent),
        )

    def last_message_id(self) -> str | None:
        row = self._conn.execute("SELECT message_id FROM drive_chain ORDER BY position DESC LIMIT 1").fetchone()
        return row[0].decode("utf-8", "surrogatepass") if row is not None else None

    def gap_fillable(self, message_id: str) -> bool:
        """Whether no branch declaration keeps the linear fill off ``message_id``."""
        if self._candidates(message_id)[1] > 1 or self._is_prompt_child(message_id):
            return False
        return (
            self._conn.execute(
                "SELECT 1 FROM drive_unresolved WHERE message_id = ?", (_text_key(message_id),)
            ).fetchone()
            is None
        )

    def chronological(self) -> Iterable[tuple[int, str, bool]]:
        """``(position, message id, explicit parent)`` by chain instant, then position."""
        for position, message_id, explicit_parent in self._conn.execute(
            "SELECT position, message_id, explicit_parent FROM drive_chain ORDER BY instant, position"
        ):
            yield position, message_id.decode("utf-8", "surrogatepass"), bool(explicit_parent)

    def close(self) -> None:
        for table in self._TABLES:
            self._conn.execute(f"DROP TABLE IF EXISTS {table}")


def _instruction_text(payload: JSONDocument) -> str | None:
    instruction = payload.get("systemInstruction")
    if isinstance(instruction, str) and instruction:
        return instruction
    instruction_obj = json_document(instruction)
    direct = _string_field(instruction_obj, "text", "content")
    if direct is not None:
        return direct
    parts = instruction_obj.get("parts")
    if not isinstance(parts, list):
        return None
    text_parts = [text for part in parts if (text := _string_field(json_document(part), "text", "content")) is not None]
    return "\n".join(text_parts) or None


def _model_config_event(
    run_settings: JSONDocument,
    *,
    timestamp: str | None,
) -> ParsedSessionEvent | None:
    if not run_settings:
        return None
    event_payload: dict[str, object] = {"runSettings": dict(run_settings)}
    model_name = _string_field(run_settings, "model", "modelName", "model_name")
    if model_name is not None:
        event_payload["model"] = model_name
    return ParsedSessionEvent(
        event_type="model_config",
        timestamp=timestamp,
        payload=event_payload,
    )


def _citation_events(payload: JSONDocument) -> list[ParsedSessionEvent]:
    """Retain the export envelope's grounding citations as session evidence.

    One event per citation, in list order. AI Studio appends to this list as
    the conversation grows, so a later export of the same session is ``[A, B]``
    after ``[A]``. One event holding the whole list changed identity on every
    append, which made revision membership classify an ordinary grown session
    as ambiguous; per-citation events keep ``A`` identical across revisions.
    """
    citations = payload.get("citations")
    if not isinstance(citations, list):
        return []
    return [
        ParsedSessionEvent(event_type="gemini_citation", payload={"ordinal": ordinal, "citation": citation})
        for ordinal, citation in enumerate(citations)
    ]


def _pending_drafts(pending_inputs: object) -> list[dict[str, object]]:
    """Extract non-blank ``chunkedPrompt.pendingInputs`` entries.

    AI Studio's Drive-synced JSON carries the operator's not-yet-submitted
    textbox content here -- draft prompts that never became a chunk and are
    otherwise unrecoverable once overwritten (polylogue-o4j2). Entries with
    blank/whitespace-only text are the overwhelmingly common case (the
    textbox was empty when synced) and carry no evidence, so they are
    skipped rather than kept as near-100%-empty noise.

    Deliberately returned as plain dicts for ``ParsedSession.pending_drafts``,
    NOT ``ParsedSessionEvent``s: a draft is mutable CURRENT state (the
    operator edits the same textbox in place, and the entry disappears
    entirely once submitted), not an append-only historical fact.
    ``session_events`` feeds ``session_revision_projection``'s
    message/attachment/event comparison axes (polylogue-aggz Invariant 1),
    which assume every axis only ever grows between two acquisitions of the
    same session; a mutable, disappearing item there reproduces the exact
    defect class polylogue-bu1i (acquisition state in identity) and
    polylogue-nuec (provider-remeasurement in identity) were fixed for --
    edits would compare as disjoint forks, and submission would shrink the
    event axis while the message axis grows, both misclassifying revision
    membership. ``pending_drafts`` stays outside every identity/hash
    computation in ``pipeline/ids.py`` (see ``sessions.pending_drafts_json``).
    """
    if not isinstance(pending_inputs, list):
        return []
    drafts: list[dict[str, object]] = []
    for entry in pending_inputs:
        entry_obj = json_document(entry)
        text = entry_obj.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        draft: dict[str, object] = {"text": text}
        role_val = _string_field(entry_obj, "role")
        if role_val is not None:
            draft["role"] = role_val
        token_count = _non_negative_int_field(entry_obj, "tokenCount", "token_count")
        if token_count is not None:
            draft["token_count"] = token_count
        drafts.append(draft)
    return drafts


def _delivery_status(chunk_obj: JSONDocument) -> str | None:
    if _string_field(chunk_obj, "errorMessage", "error_message") is not None:
        return "error"
    return _string_field(chunk_obj, "finishReason", "finish_reason")


def _gemini_usage_event(
    chunk_obj: JSONDocument,
    *,
    role: Role,
    message_id: str,
    timestamp: str | None,
) -> ParsedSessionEvent | None:
    token_count = _non_negative_int_field(chunk_obj, "tokenCount", "token_count")
    finish_reason = _string_field(chunk_obj, "finishReason", "finish_reason")
    if token_count is None and finish_reason is None:
        return None

    usage: dict[str, int] = {}
    if token_count is not None:
        if role is Role.USER:
            usage["input_tokens"] = token_count
        else:
            usage["output_tokens"] = token_count

    payload: dict[str, object] = {"type": "token_count"}
    if usage:
        payload["last_token_usage"] = usage
    if finish_reason is not None:
        payload["finish_reason"] = finish_reason
    model_name = _string_field(chunk_obj, "model", "modelName", "model_name")
    if model_name is not None:
        payload["model"] = model_name
    return ParsedSessionEvent(
        event_type="token_count",
        timestamp=timestamp,
        source_message_provider_id=message_id,
        payload=payload,
    )


def _payload_chunks(payload: JSONDocument) -> Sequence[object]:
    prompt = json_document(payload.get("chunkedPrompt"))
    if prompt:
        prompt_chunks = prompt.get("chunks")
        return prompt_chunks if isinstance(prompt_chunks, list) else []
    payload_chunks = payload.get("chunks")
    return payload_chunks if isinstance(payload_chunks, list) else ()


@parser_admission("drive")
def parse_chunked_prompt(provider: Provider | str, payload: JSONDocument, fallback_id: str) -> ParsedSession:
    chunks = _payload_chunks(payload)
    return _parse_chunked_records(provider, payload, lambda: chunks, fallback_id)


def parse_chunked_prompt_stream(
    provider: Provider | str,
    envelope: JSONDocument,
    chunks: Callable[[], Iterable[object]],
    fallback_id: str,
    *,
    messages: MutableSequence[ParsedMessage],
    session_events: MutableSequence[ParsedSessionEvent],
    attachments: MutableSequence[ParsedAttachment],
    scratch: sqlite3.Connection,
    record_stream: bool = False,
) -> ParsedSession:
    """Lower a proved chunked prompt without retaining its chunk array.

    ``envelope`` holds the document's fields except the selected chunk list,
    which ``chunks`` re-reads for each pass; per-message ordering and branch
    rows go to ``scratch``. Admission runs over a stub that
    carries the document's first future wire type, so accounting and the
    typed unknown event match ``parse_chunked_prompt`` on the whole document.
    Bare record streams instead carry the actual per-chunk fold dispositions.
    """
    future_type = envelope.get("__admission_future_type")
    payload = {key: value for key, value in envelope.items() if key != "__admission_future_type"}
    session = _parse_chunked_records(
        provider,
        payload,
        chunks,
        fallback_id,
        messages=messages,
        session_events=session_events,
        attachments=attachments,
        scratch=scratch,
        record_stream=record_stream,
    )
    if record_stream:
        return session
    admission_stub: JSONDocument = {"chunks": []}
    if isinstance(future_type, str):
        admission_stub["type"] = future_type
    admitted = parse_chunked_prompt(provider, admission_stub, fallback_id)
    session_events.extend(admitted.session_events)
    return session.model_copy(update={"unit_accounting": admitted.unit_accounting})


def _parse_chunked_records(
    provider: Provider | str,
    payload: JSONDocument,
    chunk_records: Callable[[], Iterable[object]],
    fallback_id: str,
    *,
    messages: MutableSequence[ParsedMessage] | None = None,
    session_events: MutableSequence[ParsedSessionEvent] | None = None,
    attachments: MutableSequence[ParsedAttachment] | None = None,
    scratch: sqlite3.Connection | None = None,
    record_stream: bool = False,
) -> ParsedSession:
    """Normalize chunks read once per pass into the supplied message rows.

    Branch evidence and the linear parent chain need every chunk, so this
    keeps one small ordering row per message in ``scratch`` (an in-memory
    database by default) and rewrites only the rows whose parent or leaf
    flag the whole-prompt passes change.
    """
    if scratch is None:
        with closing(sqlite3.connect(":memory:")) as memory:
            return _parse_chunked_records(
                provider,
                payload,
                chunk_records,
                fallback_id,
                messages=messages,
                session_events=session_events,
                attachments=attachments,
                scratch=memory,
                record_stream=record_stream,
            )
    observer = AdmissionObserver(record_stream=True) if record_stream else None
    order = _ChunkOrder(scratch)
    runtime_provider = Provider.from_string(provider)
    run_settings = json_document(payload.get("runSettings"))
    default_model_name = _string_field(run_settings, "model", "modelName", "model_name")
    prompt = json_document(payload.get("chunkedPrompt"))

    # Fallback timestamp from session metadata
    create_time = payload.get("createTime")
    default_timestamp = str(create_time) if create_time else None

    message_rows: MutableSequence[ParsedMessage] = messages if messages is not None else []
    event_rows: MutableSequence[ParsedSessionEvent] = session_events if session_events is not None else []
    attachment_rows: MutableSequence[ParsedAttachment] = attachments if attachments is not None else []
    observed_timestamps = TimestampBounds()
    models_used: set[str] = set()
    if default_model_name is not None:
        models_used.add(default_model_name)
    if model_event := _model_config_event(run_settings, timestamp=default_timestamp):
        event_rows.append(model_event)
    event_rows.extend(_citation_events(payload))
    order.record_branch_declarations(chunk_records())
    message_position = 0
    for _idx, chunk in enumerate(chunk_records(), start=1):
        if isinstance(chunk, str):
            chunk_obj: JSONDocument = {"text": chunk}
        elif isinstance(chunk, dict):
            chunk_obj = chunk
        else:
            if observer is not None:
                observer.observe(chunk, lowered=False)
            continue
        text = extract_text_from_chunk(chunk_obj)
        # Role is required - skip chunks without one
        role_val = chunk_obj.get("role") or chunk_obj.get("author")
        if not isinstance(role_val, str) or not role_val:
            if observer is not None:
                observer.observe(chunk, lowered=False, malformed=_unknown_wire_type(chunk) is None)
            continue
        role = Role.normalize(role_val)
        msg_id = str(chunk_obj.get("id") or "")
        prompt_parent_id = _branch_parent_session_provider_id(chunk_obj)
        if prompt_parent_id is not None:
            order.add_prompt_parent(prompt_parent_id)
            if msg_id:
                order.mark_unresolved(msg_id)
        message_timestamp = _chunk_timestamp(chunk_obj, default_timestamp)
        model_name = _string_field(chunk_obj, "model", "modelName", "model_name") or default_model_name
        if model_name is not None:
            models_used.add(model_name)
        usage_fields = _usage_fields(chunk_obj, role=role)
        if usage_event := _gemini_usage_event(
            chunk_obj,
            role=role,
            message_id=msg_id,
            timestamp=message_timestamp,
        ):
            event_rows.append(usage_event)
        chunk_attachments = _collect_chunk_attachments(chunk_obj, msg_id, role=role)
        # Attachment identifiers describe the attachment, not the message
        # that owns it.  Using them as the message's stable owner key makes a
        # metadata-only move between same-timestamp idless chunks look like
        # no change at all: the same attachment recreates the same key under
        # its new chunk.  Keep ownership on the chunk's own provider id or
        # physical coordinate instead; ``message_owner_resolution`` then
        # supplies the content discriminator for duplicate idless turns and
        # fails closed when those turns are genuinely indistinguishable.
        owner_coordinate = MessageOwnerCoordinate(
            stable_key=None,
            position=message_position,
            variant_index=0,
        )
        chunk_attachments = [
            attachment.model_copy(
                update={
                    "message_position": message_position,
                    "message_variant_index": 0,
                    "owner_coordinate": owner_coordinate,
                }
            )
            for attachment in chunk_attachments
        ]
        observed_timestamps.observe(message_timestamp)
        used_typed_model = False

        # Try to parse via the rich GeminiMessage typed model for structured extraction.
        try:
            gemini_message = GeminiMessage.model_validate(chunk_obj)
            used_typed_model = True
            content_block_payloads = _gemini_content_block_payloads(gemini_message, text)
        except ValidationError:
            # Only a chunk the typed model rejects takes the fallback; a
            # defect in the typed extraction must surface, not silently drop
            # structured blocks and change the content hash (polylogue-hu24g).
            content_block_payloads = _fallback_gemini_content_blocks(chunk_obj, text)

        if chunk_attachments and not used_typed_model:
            content_block_payloads = _append_attachment_blocks(content_block_payloads, chunk_attachments)

        if not text and not chunk_attachments and not content_block_payloads:
            if observer is not None:
                observer.observe(chunk, lowered=False)
            continue

        event_rows.extend(
            _session_events_from_meta_blocks(
                content_block_payloads,
                source_message_provider_id=msg_id,
                timestamp=message_timestamp,
            )
        )

        message_blocks = _parsed_blocks_from_meta(content_block_payloads)
        # A block-derived type (e.g. TOOL_RESULT for a codeExecutionResult
        # outcome) must be resolved BEFORE classify_material_origin runs --
        # otherwise the origin gets computed against an assumed plain
        # MESSAGE type and a genuine tool-result turn is misclassified as
        # ASSISTANT_AUTHORED instead of TOOL_RESULT (caught by
        # test_ai_studio_normalizes_identity_authorship_config_blocks_artifacts_usage_and_status).
        resolved_message_type = (
            classify_block_message_type(tuple(block.type for block in message_blocks)) or MessageType.MESSAGE
        )
        parent_message_provider_id = _branch_parent_message_provider_id(chunk_obj) or order.branch_parent(msg_id)
        message_rows.append(
            upgrade_chat_export_user_authorship(
                runtime_provider,
                ParsedMessage(
                    provider_message_id=msg_id,
                    role=role,
                    text=text,
                    timestamp=message_timestamp,
                    blocks=message_blocks,
                    message_type=resolved_message_type,
                    position=message_position,
                    variant_index=0,
                    is_active_path=True,
                    is_active_leaf=False,
                    parent_message_provider_id=parent_message_provider_id,
                    owner_coordinate=owner_coordinate,
                    input_tokens=usage_fields["input_tokens"],
                    output_tokens=usage_fields["output_tokens"],
                    model_name=model_name,
                    duration_ms=_non_negative_int_field(chunk_obj, "durationMs", "duration_ms", "elapsed_ms"),
                    delivery_status=_delivery_status(chunk_obj),
                    end_turn=(
                        True
                        if _string_field(chunk_obj, "finishReason", "finish_reason", "errorMessage", "error_message")
                        is not None
                        else None
                    ),
                    # polylogue-gzgyl: AI-Studio/Drive has no agent/subagent
                    # artifact ambiguity for a plain user turn -- positive-
                    # evidence override for the shared classify_material_origin
                    # no-fallthrough (#2502).
                    material_origin=human_authored_override(
                        role,
                        resolved_message_type,
                        classify_material_origin(
                            role=role,
                            message_type=resolved_message_type,
                            text=text,
                            block_types=tuple(block.type for block in message_blocks),
                        ),
                    ),
                ),
            )
        )
        order.add_chain_row(
            _sort_instant(message_timestamp), message_position, msg_id, parent_message_provider_id is not None
        )
        message_position += 1
        attachment_rows.extend(chunk_attachments)
        if observer is not None:
            observer.observe(chunk, lowered=True)

    title_val = payload.get("title")
    title_source: TitleSource | None = TitleSource.ORIGIN
    if not title_val:
        title_val = payload.get("displayName")
        # polylogue-5dfu: None (not TitleSource.UNKNOWN) when neither field
        # carries a title -- NULL already means "no title evidence" here.
        title_source = TitleSource.ORIGIN if title_val else None
    title = str(title_val) if title_val else fallback_id
    create_time_str = (
        str(payload.get("createTime"))
        if payload.get("createTime")
        else (observed_timestamps.earliest[1] if observed_timestamps.earliest is not None else None)
    )
    update_time_str = (
        str(payload.get("updateTime"))
        if payload.get("updateTime")
        else (observed_timestamps.latest[1] if observed_timestamps.latest is not None else None)
    )
    pending_drafts = _pending_drafts(prompt.get("pendingInputs"))
    active_leaf_message_provider_id = order.last_message_id()
    if active_leaf_message_provider_id is not None:
        message_rows[-1] = message_rows[-1].model_copy(update={"is_active_leaf": True})
    # bd polylogue-ksgg: real Gemini branch evidence (``_branch_parent_message_provider_id``
    # / ``branch_child_parents`` above) already sets ``parent_message_provider_id``
    # for messages that carry it; most AI Studio Drive sessions have none
    # (0% parented measured) because they're a plain linear chat with no
    # branch. Only fill the remaining gap -- chain to the previous message on
    # the active path -- without touching real branch evidence already set.
    # A genuinely ambiguous branch child (more than one declared parent, see
    # _branch_child_parent_map) must keep parent_message_provider_id=None --
    # the linear fill cannot tell that apart from "no branch data at all" on
    # its own, so those specific ids are excluded from the gap-fill.
    # Prompt-grain branch evidence is also explicitly unresolved at message
    # grain: no local message id is guessed.
    # chunkedPrompt's array order is not guaranteed chronological (a Drive
    # payload can legitimately list chunks in a different order than they
    # occurred -- ai-studio-drive normalization laws require native facts,
    # including the reconstructed parent chain, to be order-independent).
    # The linear fill chains by order, so it runs over a temporally-sorted
    # view, not the raw chunk-input order. Every Drive message is on the
    # active path, so each one is the next message's chain predecessor.
    previous: tuple[int, str] | None = None
    for position, message_id, explicit_parent in order.chronological():
        if previous is not None and not explicit_parent and order.gap_fillable(message_id):
            update: dict[str, object] = (
                {"parent_message_provider_id": previous[1]} if previous[1] else {"parent_message_position": previous[0]}
            )
            message_rows[position] = message_rows[position].model_copy(update=update)
        previous = (position, message_id)
    parent_session_provider_id = order.only_prompt_parent()
    order.close()
    session = ParsedSession(
        source_name=runtime_provider,
        provider_session_id=str(payload.get("id") or fallback_id),
        title=title,
        title_source=title_source,
        created_at=create_time_str,
        updated_at=update_time_str,
        messages=message_rows if isinstance(message_rows, list) else [],
        session_events=event_rows if isinstance(event_rows, list) else [],
        active_leaf_message_provider_id=active_leaf_message_provider_id,
        attachments=attachment_rows if isinstance(attachment_rows, list) else [],
        instructions_text=_instruction_text(payload),
        models_used=sorted(models_used),
        # polylogue-o4j2: pendingInputs draft(s), kept off session_events on
        # purpose -- see _pending_drafts' docstring for why (mutable current
        # state must not enter session_revision_projection's comparison
        # axes).
        pending_drafts=pending_drafts,
        parent_session_provider_id=parent_session_provider_id,
    )
    if not (isinstance(message_rows, list) and isinstance(event_rows, list) and isinstance(attachment_rows, list)):
        session = session.model_copy(
            update={"messages": message_rows, "session_events": event_rows, "attachments": attachment_rows}
        )
    return observer.apply(session, runtime_provider.value) if observer is not None else session


def looks_like_chunk(payload: object) -> bool:
    """Return whether a record has the minimum AI Studio chunk wire shape."""
    chunk = json_document(payload)
    role = chunk.get("role") or chunk.get("author")
    return isinstance(role, str) and bool(role.strip()) and any(key in chunk for key in _CHUNK_CONTENT_KEYS)


def _looks_like_chunks(value: object) -> bool:
    """Return whether ``value`` is a non-empty list of genuine chunks.

    An empty (or absent, or wrong-typed) ``chunks`` list is never sufficient
    evidence on its own (polylogue-mvcbi) -- both the named ``chunkedPrompt``
    envelope and the bare top-level ``chunks`` key require at least one
    chunk carrying role/content evidence.
    """
    if not isinstance(value, list) or not value:
        return False
    return all(looks_like_chunk(item) for item in value)


def has_chunk_container(payload: object) -> bool:
    """Return whether an explicitly selected payload carries a chunk list.

    This is intentionally more tolerant than :func:`looks_like`: explicit
    provider routes should still salvage valid chunks from partially malformed
    exports, while auto-detection must not claim generic ``chunks`` records.
    """
    record = json_document(payload)
    prompt = json_document(record.get("chunkedPrompt"))
    return isinstance(prompt.get("chunks"), list) or isinstance(record.get("chunks"), list)


def looks_like(payload: object) -> bool:
    """Return True if payload looks like a Drive / Gemini chunkedPrompt export.

    Called from ``dispatch._looks_like_gemini_mapping`` -- that is this
    detector's sole auto-detection call site (polylogue-zkmi); it is not
    dead code despite the differently-named wrapper.

    Tightened (polylogue-mvcbi, sibling fix to #3428/#3537): a bare
    ``chunkedPrompt`` envelope with a present-but-empty (or wrong-typed)
    ``chunks`` list used to be accepted on the strength of the key's mere
    presence (``allow_empty=True``) -- the same guess-instead-of-verify
    shape closed for Claude Code's bare ``type`` check and claude.ai's bare
    ``chat_messages`` check. This now requires at least one genuine chunk
    with role/content evidence (see ``looks_like_chunk``), matching the
    bare top-level ``chunks`` case below.
    """
    record = json_document(payload)
    prompt = json_document(record.get("chunkedPrompt"))
    if prompt and _looks_like_chunks(prompt.get("chunks")):
        return True
    # Older exports expose ``chunks`` at the document top level.
    return _looks_like_chunks(record.get("chunks"))


def detection_projection() -> DetectorProjection:
    """Validate every chunk under both supported container placements."""
    fields: dict[str, DetectorProjection | None] = dict.fromkeys(_CHUNK_CONTENT_KEYS)
    fields.update(role=DetectorProjection(), author=DetectorProjection())
    chunks = DetectorProjection(
        item=DetectorProjection(fields=fields), array_fold="all", array_predicate=looks_like_chunk
    )
    return DetectorProjection(fields={"chunks": chunks, "chunkedPrompt": DetectorProjection(fields={"chunks": chunks})})
