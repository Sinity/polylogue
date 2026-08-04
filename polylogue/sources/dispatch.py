"""Provider detection and payload lowering for source parsing."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from polylogue.core.binary_signatures import detect_binary_signature
from polylogue.core.enums import Provider, TitleSource
from polylogue.core.json import JSONDocument, JSONValue, is_json_document, is_json_value, normalize_json_decimal
from polylogue.core.payload_coercion import optional_string
from polylogue.logging import get_logger

from .decoders import _decode_json_bytes, _iter_json_stream
from .parsers import (
    antigravity,
    beads,
    browser_capture,
    chatgpt,
    chatgpt_codex_sidecar,
    claude,
    codex,
    drive,
    grok,
    hermes_spans,
    hermes_state,
    hermes_verification,
    local_agent,
)
from .parsers.base import (
    ParsedMessage,
    ParsedSession,
    extract_messages_from_list,
    mark_last_occurrence_as_active_leaf,
)
from .parsers.claude import code_parser as claude_code_parser
from .parsers.claude.code_parser import apply_tool_result_sidecars

if TYPE_CHECKING:
    from polylogue.schemas.packages import SchemaResolution
    from polylogue.sources.live.tool_result_sidecars import SidecarJoinResult, ToolResultIndexAccumulator

logger = get_logger(__name__)

BUNDLE_PROVIDERS = frozenset({Provider.CHATGPT, Provider.CLAUDE_AI, Provider.CLAUDE_DESIGN})
GROUP_PROVIDERS = frozenset(
    {Provider.CLAUDE_CODE, Provider.CODEX, Provider.GEMINI, Provider.DRIVE, Provider.BEADS, Provider.HERMES}
)
STREAM_RECORD_PROVIDERS = frozenset({Provider.CLAUDE_CODE, Provider.CODEX, Provider.BEADS, Provider.HERMES})
DRIVE_LIKE_PROVIDERS = frozenset({Provider.GEMINI, Provider.DRIVE})
# The explicit record-shape branch order below remains the production
# implementation during the OriginSpec migration.  OriginSpec validates its
# declared detector tightness against this projection so a new declaration
# cannot silently contradict a stronger existing detector.
RECORD_DETECTOR_PROVIDER_ORDER = (
    Provider.GEMINI_CLI,
    Provider.HERMES,
    Provider.ANTIGRAVITY,
    Provider.BEADS,
    Provider.CODEX,
    Provider.CLAUDE_CODE,
    Provider.CHATGPT,
    Provider.CLAUDE_AI,
    Provider.CLAUDE_DESIGN,
    Provider.GROK,
    Provider.GEMINI,
)
_MAX_PARSE_DEPTH = 10

PayloadRecord: TypeAlias = JSONDocument
PayloadSequence: TypeAlias = list[JSONValue]
LoweredPayloadMode: TypeAlias = Literal[
    "bundle_record",
    "browser_capture",
    "chatgpt_codex_task",
    "chunked_prompt",
    "claude_code_multiway",
    "generic_messages",
    "grouped_records",
    "local_artifact_document",
    "local_agent_document",
    "single_record",
]


@dataclass(frozen=True, slots=True)
class LoweredPayloadSpec:
    provider: Provider
    fallback_id: str
    mode: LoweredPayloadMode
    payload: PayloadRecord | PayloadSequence
    source_path: str | None = None
    # bd polylogue-jc4q: true when ``fallback_id`` is a composite already
    # anchored to a resume/fork/usage-limit boundary carryover fragment's
    # own identity -- see ``code_parser.py``'s ``trust_fallback_id`` for why
    # this must be an explicit signal rather than an inferred one. Claude
    # Code's multi-session-per-file case (``mode="claude_code_multiway"``)
    # resolves this per group internally in
    # ``_claude_code_multiway_parse``/``_claude_code_new_group_identity``
    # instead of through this field, which stays relevant only for the
    # always-single-group ``mode="grouped_records"`` fallback (a dict-shaped
    # Claude Code payload, or any other grouped provider).
    trust_fallback_id: bool = False


def _payload_record(value: object) -> PayloadRecord | None:
    normalized = normalize_json_decimal(value)
    return normalized if is_json_document(normalized) else None


def _payload_sequence(value: object) -> PayloadSequence | None:
    if not isinstance(value, list):
        return None
    payloads: list[JSONValue] = []
    for item in value:
        normalized = normalize_json_decimal(item)
        if not is_json_value(normalized):
            return None
        payloads.append(normalized)
    return payloads


def _single_document_record(value: object) -> PayloadRecord | None:
    """Resolve a single JSON document, unwrapping a one-element sequence.

    Document-style providers (gemini-cli, hermes, antigravity) store one JSON
    object per file. The full-ingest path passes parsed payloads as a list
    (``list(_iter_json_stream(...))``), so a one-record file arrives here as a
    single-element list rather than a bare dict. ``_payload_record`` returns
    ``None`` for a list, which previously made these branches yield no sessions
    and marked the file as a permanent parse failure (perpetual retry).
    """
    record = _payload_record(value)
    if record is not None:
        return record
    sequence = _payload_sequence(value)
    if sequence is not None and len(sequence) == 1:
        return _payload_record(sequence[0])
    return None


def _record_messages(record: PayloadRecord) -> list[JSONValue] | None:
    messages = record.get("messages")
    return messages if isinstance(messages, list) else None


def _record_sessions(record: PayloadRecord) -> list[JSONValue] | None:
    sessions = record.get("sessions")
    return sessions if isinstance(sessions, list) else None


def is_jsonl_source_path(source_path: str | None) -> bool:
    """Return whether a path is a JSONL/NDJSON source path."""
    normalized_path = (source_path or "").lower()
    return normalized_path.endswith((".jsonl", ".jsonl.txt", ".ndjson")) or any(
        marker in normalized_path for marker in (".jsonl.", ".ndjson.")
    )


def is_stream_record_provider(source_path: str | None, provider: str | Provider | None) -> bool:
    """Return whether a source/provider pair should use stream-record parsing."""
    if provider is None:
        return False
    if not is_jsonl_source_path(source_path):
        return False
    return Provider.from_string(provider) in STREAM_RECORD_PROVIDERS


def _looks_like_gemini_mapping(record: PayloadRecord) -> bool:
    """Detect the Drive/Gemini chunked-prompt shape (polylogue-zkmi).

    Despite the name, this is not a separate Gemini-specific check layered
    on top of an unused Drive detector: ``drive.py`` owns the single
    structural detector (``looks_like``) for the ``chunkedPrompt``/``chunks``
    shape shared by both wire families, and this function is that detector's
    sole call site in auto-detection. The result is intentionally always
    surfaced as ``Provider.GEMINI`` here, never ``Provider.DRIVE``:
    ``Provider.GEMINI`` and ``Provider.DRIVE`` are a non-injective fiber over
    the same ``Origin.AISTUDIO_DRIVE`` (see ``core/sources.py``'s
    ``_PROVIDER_TO_ORIGIN``/``provider_from_origin`` notes), and ``GEMINI``
    is the documented canonical member of that fiber, so auto-detection has
    no shape-based reason to distinguish them. ``Provider.DRIVE`` remains a
    reachable value elsewhere -- pre-existing raw rows and explicit source
    configs, see ``revision_backfill._PATH_INDEPENDENT_PARSE_PROVIDERS`` and
    ``live/batch_support._large_non_jsonl_path_can_stream`` -- it is simply
    never *produced* by this detector.
    """
    return drive.looks_like(record)


def _detect_provider_from_record_evidence(record: PayloadRecord) -> tuple[Provider | None, str]:
    """Detect a provider from a single record, tagging the deciding evidence.

    Sole implementation of the record-level detector order; ``_detect_provider_from_record``
    below is a thin wrapper that discards the evidence label, so this is the
    single source of truth -- there is no separate "logging" copy of the
    decision tree that can silently drift from the real one (observability
    gap fix, mission item 1).
    """
    if browser_capture.looks_like(record):
        session = record.get("session")
        provider = session.get("provider") if isinstance(session, dict) else None
        return Provider.from_string(provider if isinstance(provider, str) else None), "browser_capture.looks_like"
    # Local-agent JSON session documents share enough generic message keys with
    # Claude Code that they must be recognized before broader validators.
    if local_agent.looks_like_gemini_cli(record):
        return Provider.GEMINI_CLI, "local_agent.looks_like_gemini_cli"
    if hermes_state.looks_like_state_db_payload(record):
        return Provider.HERMES, "hermes_state.looks_like_state_db_payload"
    if hermes_verification.looks_like_verification_evidence_db_payload(record):
        return Provider.HERMES, "hermes_verification.looks_like_verification_evidence_db_payload"
    if hermes_spans.looks_like_atif_payload(record):
        return Provider.HERMES, "hermes_spans.looks_like_atif_payload"
    if hermes_spans.looks_like_atof_payload(record):
        return Provider.HERMES, "hermes_spans.looks_like_atof_payload"
    if local_agent.looks_like_hermes(record):
        return Provider.HERMES, "local_agent.looks_like_hermes"
    if antigravity.looks_like_markdown_export(record):
        return Provider.ANTIGRAVITY, "antigravity.looks_like_markdown_export"
    if antigravity.looks_like_brain_metadata(record, None):
        return Provider.ANTIGRAVITY, "antigravity.looks_like_brain_metadata"
    if beads.looks_like(record):
        return Provider.BEADS, "beads.looks_like"
    # Specific type-level checks first (Codex uses Pydantic validation;
    # Claude Code uses a dict-key/type shape check, not Pydantic, despite
    # ClaudeCodeRecord existing as a separate typed parse-time model), then
    # weaker dict-key checks (ChatGPT, Claude AI, Gemini).
    if codex.looks_like([dict(record)]):
        return Provider.CODEX, "codex.looks_like (pydantic record validation)"
    if claude.looks_like_code([dict(record)]):
        return Provider.CLAUDE_CODE, "claude.looks_like_code (envelope marker, #3428)"
    # A single record here may be an intentionally partial ChatGPT fragment
    # (e.g. one line of a streamed JSONL sniff), not a whole assembled
    # export document, so this uses the fragment-level check rather than
    # ``chatgpt.looks_like``'s document-identity requirements (polylogue-t0ta).
    if chatgpt.looks_like_fragment(record):
        return Provider.CHATGPT, "chatgpt.looks_like_fragment (mapping node shape)"
    # A ChatGPT shared-page (chatgpt.com/share/<id>) stream decode has no
    # "mapping" key at all -- a flattened top-level "messages" list instead
    # (polylogue-4zqh3). Checked here, ahead of the generic "has a messages
    # list" fallback used elsewhere in dispatch, which would otherwise
    # silently accept-then-drop it (no payload-asserted "id" field).
    if chatgpt.looks_like_shared_decode(record):
        return Provider.CHATGPT, "chatgpt.looks_like_shared_decode (shared-page stream decode)"
    # Claude Design (bd polylogue-tbun) checked before the general claude.ai
    # detector: its shape (messages + project, no chat_messages, camelCase
    # contentBlocks) is a distinct, tighter product signature -- not a
    # claude.ai variant.
    if claude.looks_like_claude_design(record):
        return Provider.CLAUDE_DESIGN, "claude.looks_like_claude_design"
    if claude.looks_like_claude_memories(record):
        return Provider.CLAUDE_AI, "claude.looks_like_claude_memories"
    if claude.looks_like_ai(record):
        return Provider.CLAUDE_AI, "claude.looks_like_ai (non-empty plausible chat_messages)"
    if grok.looks_like_export(record):
        return Provider.GROK, "grok.looks_like_export"
    if _looks_like_gemini_mapping(record):
        return Provider.GEMINI, "drive.looks_like (chunkedPrompt/chunks)"
    return None, "no detector matched (single record)"


def _detect_provider_from_record(record: PayloadRecord) -> Provider | None:
    return _detect_provider_from_record_evidence(record)[0]


def _detect_provider_from_sequence_evidence(payloads: PayloadSequence) -> tuple[Provider | None, str]:
    """Detect a provider from a payload sequence, tagging the deciding evidence.

    Sole implementation of the sequence-level detector order; see
    ``_detect_provider_from_record_evidence`` for the equivalent single-record
    rationale.
    """
    if not payloads:
        return None, "empty sequence"

    first_record = _payload_record(payloads[0])
    if first_record is not None:
        # A one-document Gemini CLI JSON file reaches detection as a
        # one-element sequence on the stream path. Preserve the established
        # Claude-Code-before-Codex sequence ordering while restoring this
        # stronger local-session discriminator ahead of weaker family shapes.
        # Gemini CLI's ``.jsonl`` chat-log checkpoint format is a genuinely
        # different multi-line shape: a session-open stub record (no
        # ``messages`` key -- see ``local_agent.looks_like_gemini_cli``)
        # followed by one JSON object per turn/event, so it reaches here as
        # a many-element sequence whose bare ``sessionId`` field otherwise
        # collides with Claude Code's own ``_STRONG_SESSION_KEYS`` (#3428
        # sibling gap, polylogue-hs3y). Trust the stub shape at any sequence
        # length; keep the ``messages``-embedded shape restricted to the
        # single-document case above, unchanged.
        if local_agent.looks_like_gemini_cli(first_record) and (
            len(payloads) == 1 or not isinstance(first_record.get("messages"), list)
        ):
            return Provider.GEMINI_CLI, "local_agent.looks_like_gemini_cli (stub record)"
        if browser_capture.looks_like(first_record):
            provider, evidence = _detect_provider_from_record_evidence(first_record)
            return provider, f"sequence[0] browser_capture.looks_like -> {evidence}"
        if hermes_spans.looks_like_atof_payload(first_record):
            return Provider.HERMES, "hermes_spans.looks_like_atof_payload (sequence[0])"
        if beads.looks_like(first_record):
            return Provider.BEADS, "beads.looks_like (sequence[0])"
        # The first record of a *sequence* is a whole assembled document
        # (e.g. one conversation from a ChatGPT bundle array), not a
        # partial per-line fragment, so this uses the strict document-level
        # check rather than ``looks_like_fragment`` (polylogue-t0ta).
        if chatgpt.looks_like(first_record):
            return Provider.CHATGPT, "chatgpt.looks_like (sequence[0] whole-document)"
        if isinstance(first_record.get("chat_messages"), list):
            return Provider.CLAUDE_AI, "sequence[0] chat_messages dict-key present"
        # Claude AI account memory export (memories.json, bd polylogue-zng9)
        # arrives as a bare top-level JSON array of one-per-account records.
        if claude.looks_like_claude_memories(first_record):
            return Provider.CLAUDE_AI, "claude.looks_like_claude_memories (sequence[0])"
        if claude.looks_like_claude_design(first_record):
            return Provider.CLAUDE_DESIGN, "claude.looks_like_claude_design (sequence[0])"
        if grok.looks_like_export(first_record):
            return Provider.GROK, "grok.looks_like_export (sequence[0])"
        if _looks_like_gemini_mapping(first_record):
            return Provider.GEMINI, "drive.looks_like (sequence[0])"

    if claude.looks_like_code(payloads):
        return Provider.CLAUDE_CODE, "claude.looks_like_code (record stream envelope markers)"
    if codex.looks_like(payloads):
        return Provider.CODEX, "codex.looks_like (pydantic record stream validation)"
    return None, "no detector matched (sequence)"


def _detect_provider_from_sequence(payloads: PayloadSequence) -> Provider | None:
    return _detect_provider_from_sequence_evidence(payloads)[0]


def detect_provider_evidence(payload: object, path: object | None = None) -> tuple[Provider | None, str]:
    """Infer provider from payload shape, plus the evidence that decided it.

    ``detect_provider`` is a thin wrapper over this function that discards the
    evidence label for existing call sites; use this variant wherever the
    deciding rule needs to be surfaced (acquisition logging, ``import
    explain``-style diagnostics).
    """
    del path

    if record := _payload_record(payload):
        return _detect_provider_from_record_evidence(record)
    payloads = _payload_sequence(payload)
    if payloads is not None:
        return _detect_provider_from_sequence_evidence(payloads)
    return None, "payload is not a JSON document or sequence"


def detect_provider(payload: object, path: object | None = None) -> Provider | None:
    """Infer provider from payload shape. Path is accepted for surface compatibility."""
    return detect_provider_evidence(payload, path)[0]


def detect_provider_from_raw_bytes_evidence(
    raw_bytes: bytes,
    stream_name: str,
    fallback_provider: Provider,
    *,
    truncated_tail_ok: bool = False,
) -> tuple[Provider, str]:
    """Detect a provider from a raw byte prefix, plus the evidence that decided it.

    Sole implementation of the raw-bytes detection path; ``_detect_provider_from_raw_bytes``
    is a thin wrapper that discards the evidence label. Used directly by the
    production per-file acquisition log
    (``source_acquisition_components.read_plain_source_file``) and by the
    unclaimed-file sweep's real shape check (``sources.live.acquisition_log.
    default_file_claim_check``) so both surfaces report the same rule a real
    ingest would have used, not an approximation.
    """
    # polylogue-hbtj2: a recognized binary format (SQLite being the concrete
    # miscapture the audit found) must be positively refused here, at the
    # single shared raw-bytes detection chokepoint, rather than relying on
    # the incidental ``JSONDecodeError`` a binary blob happens to raise
    # further down. This still returns ``fallback_provider`` (detection
    # never invents an origin the caller didn't already suspect), but with
    # explicit evidence identifying the refusal, instead of the generic
    # "stream decode error" text a raised exception would otherwise produce.
    if (signature := detect_binary_signature(raw_bytes)) is not None:
        return (
            fallback_provider,
            f"{signature.name}-shaped binary payload; refused as session content, used fallback_provider",
        )
    jsonl_like = _is_jsonl_stream_name(stream_name)
    text = None if jsonl_like else _decode_json_bytes(raw_bytes)
    if text is not None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        else:
            detected, evidence = detect_provider_evidence(payload)
            if detected is not None:
                return detected, evidence

    stream_bytes = _trim_jsonl_detection_prefix(raw_bytes, stream_name) if truncated_tail_ok else raw_bytes
    if not stream_bytes:
        return fallback_provider, "empty byte stream after trimming; used fallback_provider"
    try:
        stream = _iter_json_stream(BytesIO(stream_bytes), stream_name)
        payloads = list(islice(stream, 32)) if jsonl_like else list(stream)
    except Exception as exc:
        # JSON-stream detection commonly fails on payloads that the
        # record-shape detectors above already handled (or that simply do
        # not parse as a record stream — e.g. ChatGPT bundles read via
        # `devtools pipeline-probe`). Emit a structured WARNING rather
        # than a Rich traceback so default invocations do not look
        # broken when the fallback is the intended path.
        logger.warning(
            "provider_detection_stream_fallback",
            stream_name=stream_name,
            fallback_provider=fallback_provider.value,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return fallback_provider, f"stream decode error ({type(exc).__name__}: {exc}); used fallback_provider"

    detected, evidence = detect_provider_evidence(payloads)
    if detected is None:
        return fallback_provider, f"{evidence}; used fallback_provider"
    return detected, evidence


def _detect_provider_from_raw_bytes(
    raw_bytes: bytes,
    stream_name: str,
    fallback_provider: Provider,
    *,
    truncated_tail_ok: bool = False,
) -> Provider:
    return detect_provider_from_raw_bytes_evidence(
        raw_bytes,
        stream_name,
        fallback_provider,
        truncated_tail_ok=truncated_tail_ok,
    )[0]


def _is_jsonl_stream_name(stream_name: str) -> bool:
    return stream_name.lower().endswith((".jsonl", ".jsonl.txt", ".ndjson"))


def _trim_jsonl_detection_prefix(raw_bytes: bytes, stream_name: str) -> bytes:
    if not _is_jsonl_stream_name(stream_name):
        return raw_bytes
    if raw_bytes.endswith((b"\n", b"\r")):
        return raw_bytes
    newline_at = raw_bytes.rfind(b"\n")
    return raw_bytes[: newline_at + 1] if newline_at >= 0 else b""


def _schema_guided_payload(
    provider: Provider,
    payload: object,
    schema_resolution: SchemaResolution | None,
) -> object:
    """Apply schema-derived structural hints before provider-specific lowering."""
    if schema_resolution is None:
        return payload
    if schema_resolution.element_kind not in {"session_record_stream", "subagent_session_stream"}:
        return payload
    if provider not in {Provider.CLAUDE_CODE, Provider.CODEX}:
        return payload

    record = _payload_record(payload)
    if record is None:
        return payload

    messages = _record_messages(record)
    if messages is not None:
        return messages
    return [record]


def _looks_like_chunked_session(payload: object) -> bool:
    record = _payload_record(payload)
    return record is not None and drive.has_chunk_container(record)


def _looks_like_chunked_session_list(payloads: PayloadSequence) -> bool:
    return bool(payloads) and all(_looks_like_chunked_session(item) for item in payloads)


def _single_record_spec(provider: Provider, payload: PayloadRecord, fallback_id: str) -> LoweredPayloadSpec:
    return LoweredPayloadSpec(
        provider=provider,
        fallback_id=fallback_id,
        mode="single_record",
        payload=payload,
    )


def _chunked_prompt_spec(
    provider: Provider,
    payload: PayloadRecord | PayloadSequence,
    fallback_id: str,
) -> LoweredPayloadSpec:
    return LoweredPayloadSpec(
        provider=provider,
        fallback_id=fallback_id,
        mode="chunked_prompt",
        payload=payload,
    )


def _generic_messages_spec(
    provider: Provider,
    payload: PayloadRecord,
    fallback_id: str,
) -> LoweredPayloadSpec:
    return LoweredPayloadSpec(
        provider=provider,
        fallback_id=fallback_id,
        mode="generic_messages",
        payload=payload,
    )


def _local_agent_document_spec(
    provider: Provider,
    payload: PayloadRecord,
    fallback_id: str,
) -> LoweredPayloadSpec:
    return LoweredPayloadSpec(
        provider=provider,
        fallback_id=fallback_id,
        mode="local_agent_document",
        payload=payload,
    )


def _local_artifact_document_spec(
    provider: Provider,
    payload: PayloadRecord,
    fallback_id: str,
    *,
    source_path: str | None,
) -> LoweredPayloadSpec:
    return LoweredPayloadSpec(
        provider=provider,
        fallback_id=fallback_id,
        mode="local_artifact_document",
        payload=payload,
        source_path=source_path,
    )


def _grouped_records_spec(
    provider: Provider,
    payload: PayloadRecord | PayloadSequence,
    fallback_id: str,
    *,
    source_path: str | None = None,
    trust_fallback_id: bool = False,
) -> LoweredPayloadSpec:
    return LoweredPayloadSpec(
        provider=provider,
        fallback_id=fallback_id,
        mode="grouped_records",
        payload=payload,
        source_path=source_path,
        trust_fallback_id=trust_fallback_id,
    )


def _join_claude_code_sidecars(payloads: PayloadSequence, source_path: str | None) -> SidecarJoinResult | None:
    """Join ``tool-results/`` sidecar content for a Claude Code JSONL payload, if any.

    Deferred import: ``polylogue.sources.live`` (package ``__init__``) pulls in
    ``batch.py``/``watcher.py``, which import back ``from
    polylogue.sources.dispatch import ...`` -- a module-level import here would
    be circular whenever ``dispatch`` is the first module imported. Calling
    this only at parse time (long after both modules are fully loaded) avoids
    it without restructuring either package.

    Returns ``None`` (a no-op for ``parse_code``/``parse_code_stream``) when
    there is no ``source_path`` to derive a directory from; the join itself is
    cheap when the directory doesn't exist (a single ``is_dir()`` stat).
    """
    if source_path is None:
        return None
    from polylogue.sources.live.tool_result_sidecars import (
        join_tool_result_sidecars_session_scoped,
        resolve_tool_results_dir,
    )

    tool_results_dir = resolve_tool_results_dir(source_path)
    if tool_results_dir is None:
        return None
    return join_tool_result_sidecars_session_scoped(payloads, tool_results_dir, source_path)


#  bd polylogue-t5lg: rank a chunk's own resolved title the same way
# ``_parse_code_records`` ranks candidates within a single pass (custom-title
# > ai-title > agent-name > first-human-message heuristic > raw id), so that
# merging streamed chunks of one huge session can never *regress* a stronger
# title found in one chunk in favor of a weaker one that happened to resolve
# in an earlier chunk. ``title_source``/``title_confidence`` alone collapse
# custom-title and ai-title into the same (ORIGIN, 1.0) pair, so a small
# ref-prefix tie-break preserves the exact within-chunk ordering across
# chunks too.
_CLAUDE_CODE_TITLE_REF_PRIORITY: dict[str, int] = {
    "claude-agent-name:": 0,
    "claude-ai-title:": 1,
    "claude-custom-title:": 2,
}


def _claude_code_title_rank(session: ParsedSession) -> tuple[int, float, int]:
    # polylogue-5dfu: no TitleSource.UNKNOWN entry -- a None title_source
    # (no title evidence) and a session.title_source that has no listed
    # entry both fall through to the same rank-0 default via .get().
    source_rank = (
        {
            TitleSource.HEURISTIC: 1,
            TitleSource.ORIGIN: 2,
        }.get(session.title_source, 0)
        if session.title_source is not None
        else 0
    )
    confidence = session.title_confidence or 0.0
    ref_bonus = 0
    if session.title_ref:
        for prefix, bonus in _CLAUDE_CODE_TITLE_REF_PRIORITY.items():
            if session.title_ref.startswith(prefix):
                ref_bonus = bonus
                break
    return (source_rank, confidence, ref_bonus)


def merge_parsed_session_chunks(sessions: Iterable[ParsedSession]) -> list[ParsedSession]:
    """Merge repeated provider-native sessions produced by streaming chunks."""

    merged: dict[str, ParsedSession] = {}
    for session in sessions:
        existing = merged.get(session.provider_session_id)
        if existing is None:
            merged[session.provider_session_id] = session
            continue

        messages = [*existing.messages, *session.messages]
        active_leaf_message_provider_id = messages[-1].provider_message_id if messages else None
        messages = [message.model_copy(update={"position": position}) for position, message in enumerate(messages)]
        # bd polylogue-2hwl: flag the active leaf by POSITION (the true last
        # message), never by comparing provider_message_id -- retries and
        # regenerated variants can legitimately reuse the same native id at
        # more than one position across merged chunks, and an id-equality
        # comparison flags every one of them, not just the real leaf.
        messages = mark_last_occurrence_as_active_leaf(messages)

        reported_cost_usd: float | None
        if existing.reported_cost_usd is None and session.reported_cost_usd is None:
            reported_cost_usd = None
        else:
            reported_cost_usd = (existing.reported_cost_usd or 0.0) + (session.reported_cost_usd or 0.0)

        reported_duration_ms: int | None
        if existing.reported_duration_ms is None and session.reported_duration_ms is None:
            reported_duration_ms = None
        else:
            reported_duration_ms = (existing.reported_duration_ms or 0) + (session.reported_duration_ms or 0)

        created_values = [value for value in (existing.created_at, session.created_at) if value]
        updated_values = [value for value in (existing.updated_at, session.updated_at) if value]
        # bd polylogue-t5lg: pick the chunk with the stronger title EVIDENCE
        # (title_source/title_confidence, ranked identically to the
        # single-pass parser's own precedence), not merely "whichever chunk
        # resolved a non-raw-id title first". The old rule froze the first
        # chunk's title (even a weak heuristic guess) permanently once it was
        # non-UUID, silently discarding a stronger ai-title/custom-title
        # sidecar record that only appeared in a later chunk of the same
        # huge streamed session -- see the chunk-merge regression this bead
        # measured on the live archive. All four title fields move together
        # so title_source/title_ref/title_confidence never point at a
        # different chunk's evidence than the title text they describe.
        title_winner = existing if _claude_code_title_rank(existing) >= _claude_code_title_rank(session) else session
        merged[session.provider_session_id] = existing.model_copy(
            update={
                "title": title_winner.title,
                "title_source": title_winner.title_source,
                "title_ref": title_winner.title_ref,
                "title_confidence": title_winner.title_confidence,
                "created_at": min(created_values) if created_values else None,
                "parent_session_provider_id": (
                    existing.parent_session_provider_id or session.parent_session_provider_id
                ),
                "branch_type": existing.branch_type or session.branch_type,
                "updated_at": max(updated_values) if updated_values else None,
                "messages": messages,
                "active_leaf_message_provider_id": active_leaf_message_provider_id,
                "attachments": [*existing.attachments, *session.attachments],
                "session_events": [*existing.session_events, *session.session_events],
                "reported_cost_usd": reported_cost_usd,
                "reported_duration_ms": reported_duration_ms,
                "models_used": sorted({*existing.models_used, *session.models_used}),
                "working_directories": sorted({*existing.working_directories, *session.working_directories}),
                "git_branch": existing.git_branch or session.git_branch,
                "ingest_flags": sorted({*existing.ingest_flags, *session.ingest_flags}),
            }
        )
    # bd polylogue-taj0o Stage 2: Claude Code streaming used to be chunked
    # into contiguous-run parses and glued back together here, needing a
    # trailing `claude.reconcile_code_session_chunks` pass to re-fold the
    # summary session_events (background completions, delegation progress,
    # parse coverage, session_kind) each chunk had recomputed locally.
    # `_claude_code_multiway_parse` now keys one ``_SessionAccumulator`` per
    # session id across the whole record stream directly, so a Claude Code
    # session is finalized exactly once and never reaches this function at
    # all -- the generic merge above (title-evidence ranking, active-leaf
    # recomputation, etc.) stays for any other provider that still needs to
    # fold two same-identity ``ParsedSession``s together.
    return list(merged.values())


def _claude_code_new_group_identity(
    group_id: str,
    record: PayloadRecord | None,
    *,
    fallback_id: str,
    is_agent_fallback: bool,
    is_first_group: bool,
    primary_started: bool,
    primary_uuids: set[str],
) -> tuple[str, bool, bool]:
    """Resolve a newly-encountered Claude Code group's identity.

    Returns ``(group_fallback_id, trust_fallback_id, provisional)``.
    ``provisional`` is true only when the decision genuinely cannot be made
    yet (this group's own sessionId differs from ``fallback_id`` and the
    primary group hasn't started -- it might be a boundary carryover opening
    the file, or ``fallback_id`` might never appear in the file at all, in
    which case every such group is independent). The caller must defer
    finalizing ``fallback_id``/``trust_fallback_id`` on those accumulators
    until the whole stream has been walked -- see
    ``_claude_code_multiway_parse``.

    Mirrors the identity rules the former ``_claude_code_grouped_record_specs``
    (eager) and ``_claude_code_stream_sessions`` (streaming) each implemented
    separately (bd polylogue-jc4q), collapsed onto ONE canonical primary
    definition: the group whose own ``sessionId`` equals ``fallback_id`` --
    streaming's former rule, not eager's former max-by-record-count rule.
    Eager's rule could pick the wrong group as primary for a small main
    session sharing a file with a huge subagent transcript (the "three-way
    duplication" finding, bd polylogue-taj0o notes); fallback_id-match is
    also the only one of the two that a genuine single forward pass can
    decide without full materialization, since it depends only on the
    caller's own already-known identity for this file, not on record counts
    that aren't known until the file ends.
    """
    if is_agent_fallback:
        # Subagent/self-compaction files have their own identity scheme
        # (code_parser.py's is_agent branch) that carryover detection below
        # does not apply to: the first-encountered group carries the
        # caller's fallback_id, every other group keeps its own bare
        # content id.
        if is_first_group:
            return fallback_id, False, False
        return group_id, False, False

    if group_id == fallback_id:
        return fallback_id, False, False

    if not primary_started:
        # Ambiguous until the whole stream is known: either a boundary
        # carryover opening the file (if the primary group shows up later)
        # or a genuinely independent session (if it never does).
        return group_id, False, True

    # Primary content has already streamed -- decide by the
    # parentUuid-chains-into-primary signal (the mid-file shape: a `/exit`
    # sent right after a "usage limit reached" notice gets re-stamped with
    # the ancestor's id even though it is chained straight off this file's
    # own preceding record).
    root_parent_uuid = optional_string(record.get("parentUuid")) if record is not None else None
    if root_parent_uuid is not None and root_parent_uuid in primary_uuids:
        return f"{group_id}:{fallback_id}", True, False
    return group_id, False, False


def _claude_code_multiway_parse(
    payloads: Iterable[object],
    fallback_id: str,
    *,
    source_path: str | None = None,
) -> Iterator[ParsedSession]:
    """Walk a Claude Code record stream exactly once, routing every record to
    a per-session ``_SessionAccumulator`` keyed by its own ``sessionId``, and
    finalize each accumulator only once the stream is exhausted
    (bd polylogue-taj0o Stage 2).

    Replaces two former shortcuts that each re-implemented a slice of this:
    eager grouping (``_claude_code_grouped_record_specs``, which
    materialized a ``dict[str, list]`` of every record up front, then parsed
    each group's full list independently) and streaming's contiguous-run
    chunker (``_claude_code_stream_sessions``, which fed each contiguous run
    of one sessionId to its own ``_parse_code_records`` call and glued the
    per-chunk results back together after the fact via
    ``merge_parsed_session_chunks``/``reconcile_code_session_chunks``, since
    each chunk's summary session_events -- background completions,
    delegation-progress ticks, parse coverage, session_kind -- were computed
    per chunk instead of per whole session). Keying one persistent
    ``_SessionAccumulator`` per session id instead means ``_fold_code_record``
    folds every record for that session into the SAME accumulator no matter
    how many times its sessionId reappears in the file, so
    ``_finalize_code_session`` runs its post-loop summary logic exactly once,
    seeing the whole session -- there is nothing left to reconcile. This
    also means genuinely interleaved records (not just contiguous runs) are
    now handled correctly by construction, not merely approximated.

    ``payloads`` may be a materialized list (the eager path) or a true
    one-pass iterator (the streaming path, multi-GiB raw JSONL); ``for item
    in payloads`` calls ``next()`` exactly once per record either way, so
    this single function serves both callers -- mirroring
    ``polylogue/sources/parsers/codex.py``'s ``parse``/``parse_stream``
    pattern of one shared walk-and-fold implementation.

    Identity/carryover (bd polylogue-jc4q): see
    ``_claude_code_new_group_identity`` for the per-group decision rule.
    Records with no ``sessionId`` at all queue in a prefix buffer and are
    folded into whichever group is encountered first, exactly like both
    former implementations.

    Sidecar join (polylogue-wjgf): built per accumulator (one
    ``ToolResultIndexAccumulator`` per session id, observing every record
    folded into that session as it streams past) rather than per chunk --
    the former streaming chunker built and joined a fresh index per
    contiguous run, which under-joined a session split across more than one
    run. Session-scoped (``join_session_scoped``): a subagent transcript
    only matches sidecars it owns; the root/parent transcript builds a
    session-wide union index from sibling files on disk.
    """
    is_agent_fallback = fallback_id.startswith("agent-")

    tool_results_dir: Path | None = None
    if source_path is not None:
        from polylogue.sources.live.tool_result_sidecars import resolve_tool_results_dir

        tool_results_dir = resolve_tool_results_dir(source_path)
        if tool_results_dir is not None and not tool_results_dir.is_dir():
            tool_results_dir = None

    def new_sidecar_accumulator() -> ToolResultIndexAccumulator:
        from polylogue.sources.live.tool_result_sidecars import ToolResultIndexAccumulator

        return ToolResultIndexAccumulator()

    sidecar_accumulators: dict[str, ToolResultIndexAccumulator] | None = {} if tool_results_dir is not None else None

    accumulators: dict[str, claude_code_parser._SessionAccumulator] = {}
    group_order: list[str] = []
    provisional_groups: set[str] = set()
    pending_prefix: list[object] = []
    current_group_id: str | None = None
    primary_started = False
    primary_uuids: set[str] = set()

    def new_accumulator(group_fallback_id: str, trust: bool) -> claude_code_parser._SessionAccumulator:
        return claude_code_parser._SessionAccumulator(
            fallback_id=group_fallback_id,
            trust_fallback_id=trust,
            is_agent=group_fallback_id.startswith("agent-"),
            is_acompact=group_fallback_id.startswith("agent-acompact-"),
        )

    def fold_into(group_id: str, index: int, item: object) -> None:
        if sidecar_accumulators is not None:
            sidecar_accumulators[group_id].observe(item)
        record = _payload_record(item)
        if record is not None and not is_agent_fallback and group_id == fallback_id:
            uuid = optional_string(record.get("uuid"))
            if uuid is not None:
                primary_uuids.add(uuid)
        if isinstance(item, dict):
            claude_code_parser._fold_code_record(accumulators[group_id], index, item)

    record_index = 0
    for item in payloads:
        record_index += 1
        record = _payload_record(item)
        session_id = optional_string(record.get("sessionId")) if record is not None else None

        if session_id is None:
            if current_group_id is None:
                pending_prefix.append(item)
                continue
            fold_into(current_group_id, record_index, item)
            continue

        if session_id not in accumulators:
            group_fallback_id, trust, provisional = _claude_code_new_group_identity(
                session_id,
                record,
                fallback_id=fallback_id,
                is_agent_fallback=is_agent_fallback,
                is_first_group=not accumulators,
                primary_started=primary_started,
                primary_uuids=primary_uuids,
            )
            accumulators[session_id] = new_accumulator(group_fallback_id, trust)
            group_order.append(session_id)
            if sidecar_accumulators is not None:
                sidecar_accumulators[session_id] = new_sidecar_accumulator()
            if provisional:
                provisional_groups.add(session_id)
            if not is_agent_fallback and session_id == fallback_id:
                primary_started = True
            prefix_index = record_index - len(pending_prefix)
            for prefix_item in pending_prefix:
                prefix_index += 1
                fold_into(session_id, prefix_index, prefix_item)
            pending_prefix = []

        current_group_id = session_id
        fold_into(session_id, record_index, item)

    if not accumulators:
        # No sessionId ever appeared -- either a genuinely empty stream or
        # one that only ever produced prefix records (e.g. a lone
        # ``summary`` row). Both former implementations still emit exactly
        # one (possibly zero-message) session for this fallback_id.
        accumulators[fallback_id] = new_accumulator(fallback_id, False)
        group_order.append(fallback_id)
        if sidecar_accumulators is not None:
            sidecar_accumulators[fallback_id] = new_sidecar_accumulator()
        for index, prefix_item in enumerate(pending_prefix, start=1):
            fold_into(fallback_id, index, prefix_item)

    # Provisional groups (non-agent, own sessionId != fallback_id, first
    # encountered before the primary group had started) can only be
    # resolved now that the whole stream is known: a carryover fragment of
    # the primary if the primary showed up anywhere in the file, otherwise
    # a genuinely independent session with no primary to anchor against.
    # Folding never reads ``fallback_id``/``trust_fallback_id`` off the
    # accumulator (only ``is_agent``/``is_acompact``, already correctly
    # False for every provisional group -- neither a bare sessionId nor
    # ``f"{group_id}:{fallback_id}"`` can start with "agent-" when
    # ``fallback_id`` itself doesn't), so mutating them here is safe.
    for group_id in provisional_groups:
        acc = accumulators[group_id]
        if primary_started:
            acc.fallback_id = f"{group_id}:{fallback_id}"
            acc.trust_fallback_id = True
        else:
            acc.fallback_id = group_id
            acc.trust_fallback_id = False

    for group_id in group_order:
        session = claude_code_parser._finalize_code_session(accumulators[group_id])
        if sidecar_accumulators is not None:
            # sidecar_accumulators is only set when tool_results_dir resolved, which itself
            # only happens when source_path is not None.
            assert tool_results_dir is not None
            assert source_path is not None
            join_result = sidecar_accumulators[group_id].join_session_scoped(tool_results_dir, source_path)
            session = apply_tool_result_sidecars(session, join_result)
        yield session


def _bundle_record_specs(
    provider: Provider,
    payloads: PayloadSequence,
    fallback_id: str,
) -> list[LoweredPayloadSpec]:
    return [
        LoweredPayloadSpec(
            provider=provider,
            fallback_id=f"{fallback_id}-{index}",
            mode="bundle_record",
            payload=record,
        )
        for index, item in enumerate(payloads)
        if (record := _payload_record(item)) is not None
    ]


#: Below this many candidate records, a zero-match bundle is unremarkable --
#: even a genuine format change might only affect a small shard, and
#: warning on tiny/edge-case payloads would be noise the next real drift
#: warning gets lost in.
_CHATGPT_BUNDLE_DRIFT_MIN_CANDIDATES = 5


def _looks_like_chatgpt_mapping_candidate(record: PayloadRecord) -> bool:
    """True if ``record`` carries a non-empty ``mapping`` dict.

    This is deliberately looser than ``chatgpt.looks_like_fragment`` (which
    also validates every node's shape): it is the "near miss" test used to
    decide whether a zero-match bundle is drift-worth-a-warning or routine
    sibling-file noise. ChatGPT's real metadata siblings
    (``message_feedback.json``, ``shared_conversations.json``,
    ``user_settings.json``, ``user.json``) never carry a ``mapping`` key at
    all, so they never count as candidates here regardless of list length --
    only records that got as far as having *a* ``mapping`` dict, but whose
    node shapes then failed validation, count. That is what an export
    format change to the conversation-tree shape itself would look like,
    as opposed to a large-but-irrelevant sibling array.
    """
    mapping = record.get("mapping")
    return isinstance(mapping, dict) and bool(mapping)


def _chatgpt_bundle_record_specs(
    payloads: PayloadSequence,
    fallback_id: str,
) -> list[LoweredPayloadSpec]:
    """Lower a ChatGPT bundle-shaped JSON array into per-conversation specs.

    A ChatGPT GDPR/Takeout export ZIP legitimately contains sibling arrays
    that are shaped like a bundle (a top-level JSON list) but are NOT
    conversation records -- ``message_feedback.json``,
    ``shared_conversations.json``, ``user_settings.json``, and (once an
    export is large enough that OpenAI shards it) the numbered
    ``conversations-NNN.json`` shard files sit alongside those siblings in
    the same ZIP, indistinguishable from each other by filename alone
    (dispatch's detection is shape-based, not filename-based -- see
    ``sources/dispatch.py`` module docstring). Filtering every candidate
    record through ``chatgpt.looks_like_fragment`` here, rather than
    admitting every list item and letting ``chatgpt.parse`` silently emit
    an empty/near-empty session for a non-conversation item, does two
    things: it gives non-conversation siblings a distinguishable
    "did not match the shape" reason instead of an opaque downstream
    "produced no sessions" parse outcome, and it lets this function detect
    the one failure mode a per-row parse-error can never distinguish from
    routine sibling-file noise -- every real conversation record in a
    shard failing the shape check at once, which is what an upstream
    export-format change (e.g. OpenAI renaming the ``mapping`` key) would
    look like. When that happens for a payload large enough to plausibly
    BE a conversation shard rather than a small sibling array, log a
    warning so the drop is visible in daemon logs rather than requiring an
    operator to already suspect drift and query
    ``raw_sessions.detection_warnings_json`` to find it (polylogue-iwv7).
    """
    matched: list[LoweredPayloadSpec] = []
    candidates = 0
    for index, item in enumerate(payloads):
        record = _payload_record(item)
        if record is None:
            continue
        # codex.json (bd polylogue-2m2e): Codex Cloud tasks delivered inside
        # the ChatGPT export, a completely different shape from a conversation
        # fragment (no "mapping" key). Checked first so these never fall
        # through to the mapping-candidate/near-miss accounting below.
        if chatgpt_codex_sidecar.looks_like(record):
            matched.append(
                LoweredPayloadSpec(
                    provider=Provider.CHATGPT,
                    fallback_id=f"{fallback_id}-{index}",
                    mode="chatgpt_codex_task",
                    payload=record,
                )
            )
            continue
        if _looks_like_chatgpt_mapping_candidate(record):
            candidates += 1
        if not chatgpt.looks_like_fragment(record):
            continue
        matched.append(
            LoweredPayloadSpec(
                provider=Provider.CHATGPT,
                fallback_id=f"{fallback_id}-{index}",
                mode="bundle_record",
                payload=record,
            )
        )
    if not matched and candidates >= _CHATGPT_BUNDLE_DRIFT_MIN_CANDIDATES:
        logger.warning(
            "ChatGPT bundle payload %r: none of %d candidate records matched the "
            "conversation-fragment shape (chatgpt.looks_like_fragment); if this array is a "
            "conversations shard rather than a metadata sibling file, the ChatGPT export "
            "shape may have changed and conversations are being silently dropped",
            fallback_id,
            candidates,
        )
    return matched


def _lower_bundle_payload(
    provider: Provider,
    shaped_payload: object,
    fallback_id: str,
) -> list[LoweredPayloadSpec]:
    payloads = _payload_sequence(shaped_payload)
    if payloads is not None:
        if provider is Provider.CHATGPT:
            return _chatgpt_bundle_record_specs(payloads, fallback_id)
        return _bundle_record_specs(provider, payloads, fallback_id)
    record = _payload_record(shaped_payload)
    if record is None:
        return []
    # codex.json (bd polylogue-2m2e): reached here when the file-level walk
    # already unpacked the top-level array into one dict per item (the
    # ordinary per-.json-file path -- see source_parsing.py/emitter.py), so
    # this function sees a single task record rather than the whole list.
    # Without this check the record fell straight into ``_single_record_spec``
    # with provider=CHATGPT and no shape validation at all, and
    # ``chatgpt.parse`` silently produced a zero-message session for it (no
    # "mapping" key) that write_parsed_session_to_archive then dropped --
    # "unparsed" with no visible error.
    if provider is Provider.CHATGPT and chatgpt_codex_sidecar.looks_like(record):
        return [
            LoweredPayloadSpec(
                provider=Provider.CHATGPT,
                fallback_id=fallback_id,
                mode="chatgpt_codex_task",
                payload=record,
            )
        ]
    return [_single_record_spec(provider, record, fallback_id)]


def _lower_grouped_payload(
    provider: Provider,
    shaped_payload: object,
    fallback_id: str,
    *,
    source_path: str | None = None,
) -> list[LoweredPayloadSpec]:
    payloads = _payload_sequence(shaped_payload)
    if payloads is not None:
        if provider is Provider.CLAUDE_CODE:
            return [
                LoweredPayloadSpec(
                    provider=Provider.CLAUDE_CODE,
                    fallback_id=fallback_id,
                    mode="claude_code_multiway",
                    payload=payloads,
                    source_path=source_path,
                )
            ]
        return [_grouped_records_spec(provider, payloads, fallback_id, source_path=source_path)]

    record = _payload_record(shaped_payload)
    if record is None:
        return []

    messages = _record_messages(record)
    grouped_payload: PayloadSequence = messages if messages is not None else [record]
    return [_grouped_records_spec(provider, grouped_payload, fallback_id, source_path=source_path)]


def _lower_drive_like_payload(
    provider: Provider,
    shaped_payload: object,
    fallback_id: str,
    *,
    depth: int,
    schema_resolution: SchemaResolution | None,
) -> list[LoweredPayloadSpec]:
    payloads = _payload_sequence(shaped_payload)
    if payloads is not None:
        if (
            payloads
            and any(drive.looks_like_chunk(item) for item in payloads)
            and not any(_looks_like_chunked_session(item) for item in payloads)
        ):
            return [_chunked_prompt_spec(provider, payloads, fallback_id)]
        if _looks_like_chunked_session_list(payloads):
            # A single-element wrapper (a lone session document decoded through
            # a list-shaped stream reader, e.g. any caller that materializes a
            # plain JSON document via a generic JSON-stream iterator) must keep
            # the bare ``fallback_id`` -- suffixing it with a spurious ``-0``
            # only for THIS branch (while the sibling fallthrough loop below
            # already special-cases ``len(payloads) == 1``) diverges a
            # one-item wrapped payload's session identity from the identical
            # bare-document payload's identity, purely based on which decode
            # path incidentally wrapped it in a list (polylogue-z1c6).
            nested_specs: list[LoweredPayloadSpec] = []
            for index, item in enumerate(payloads):
                nested_specs.extend(
                    _lower_payload_specs(
                        provider,
                        item,
                        fallback_id if len(payloads) == 1 else f"{fallback_id}-{index}",
                        depth=depth + 1,
                        schema_resolution=schema_resolution,
                    )
                )
            return nested_specs
        # Drive exports and full-ingest streams can add one or more list/document
        # wrappers around session records. Recurse through those containers, but
        # never reinterpret arbitrary records as raw chunks: that would revive
        # the loose ``chunks`` detector this route is meant to replace.
        nested_specs = []
        for index, item in enumerate(payloads):
            nested_specs.extend(
                _lower_payload_specs(
                    provider,
                    item,
                    fallback_id if len(payloads) == 1 else f"{fallback_id}-{index}",
                    depth=depth + 1,
                    schema_resolution=schema_resolution,
                )
            )
        return nested_specs

    record = _payload_record(shaped_payload)
    if record is None:
        return []
    if local_agent.looks_like_gemini_cli(record):
        return [_local_agent_document_spec(Provider.GEMINI_CLI, record, fallback_id)]
    if _record_messages(record) is not None:
        return [_generic_messages_spec(provider, record, fallback_id)]
    # This handles one already-lowered record, not a whole document/list, so
    # it uses the fragment-level check rather than requiring document-identity
    # fields (polylogue-t0ta) -- consistent with ``_detect_provider_from_record``.
    if chatgpt.looks_like_fragment(record):
        return [_single_record_spec(Provider.CHATGPT, record, fallback_id)]
    if _looks_like_chunked_session(record):
        return [_chunked_prompt_spec(provider, record, fallback_id)]
    return []


def _lower_grok_export_payload(payload: object, fallback_id: str) -> list[LoweredPayloadSpec]:
    """Unwrap a Grok account-data export document into per-conversation specs.

    Unlike ``BUNDLE_PROVIDERS`` (ChatGPT/Claude AI), whose bundle payload is
    already list-shaped at the point dispatch sees it, a Grok export document
    is a single JSON object wrapping its conversations under a
    ``"conversations"`` key -- one physical file, N logical sessions. Grok is
    a document-style provider like gemini-cli/hermes/antigravity (one JSON
    object per file), so ``_single_document_record`` unwraps the one-element
    list the full-ingest stream path wraps a lone document in.
    """
    record = _single_document_record(payload)
    if record is None:
        return []
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return []
    specs: list[LoweredPayloadSpec] = []
    for index, item in enumerate(conversations):
        item_record = _payload_record(item)
        # A malformed entry (missing "conversation"/"responses", wrong types)
        # is skipped rather than admitted as a zero-message phantom session --
        # matching how _bundle_record_specs silently drops non-record bundle
        # entries for ChatGPT/Claude AI rather than emitting an empty session
        # per malformed item.
        if item_record is None or not grok.looks_like_conversation(item_record):
            continue
        specs.append(
            _single_record_spec(
                Provider.GROK,
                item_record,
                fallback_id if len(conversations) == 1 else f"{fallback_id}-{index}",
            )
        )
    return specs


def _lower_fallback_payload(
    provider: Provider,
    shaped_payload: object,
    fallback_id: str,
) -> list[LoweredPayloadSpec]:
    record = _payload_record(shaped_payload)
    if record is None:
        return []
    if _record_messages(record) is not None:
        return [_generic_messages_spec(provider, record, fallback_id)]
    # Same rationale as the fallback branch in ``_lower_drive_like_payload``
    # above: one record, not a whole document, so fragment-level detection.
    if chatgpt.looks_like_fragment(record):
        return [_single_record_spec(Provider.CHATGPT, record, fallback_id)]
    if _looks_like_chunked_session(record):
        return [_chunked_prompt_spec(provider, record, fallback_id)]
    return []


def _lower_payload_specs(
    provider: str | Provider,
    payload: object,
    fallback_id: str,
    *,
    depth: int = 0,
    schema_resolution: SchemaResolution | None = None,
    source_path: str | None = None,
) -> list[LoweredPayloadSpec]:
    runtime_provider = Provider.from_string(provider)
    if depth > _MAX_PARSE_DEPTH:
        logger.warning("Recursion depth exceeded parsing %s (provider=%s)", fallback_id, provider)
        return []

    shaped_payload = _schema_guided_payload(runtime_provider, payload, schema_resolution)
    record = _payload_record(shaped_payload)
    if record is not None and browser_capture.looks_like(record):
        provider = _detect_provider_from_record(record) or runtime_provider
        return [
            LoweredPayloadSpec(
                provider=provider,
                fallback_id=fallback_id,
                mode="browser_capture",
                payload=record,
            )
        ]
    sequence = _payload_sequence(shaped_payload)
    if sequence:
        browser_capture_specs: list[LoweredPayloadSpec] = []
        for index, item in enumerate(sequence):
            item_record = _payload_record(item)
            if item_record is None or not browser_capture.looks_like(item_record):
                browser_capture_specs = []
                break
            provider = _detect_provider_from_record(item_record) or runtime_provider
            browser_capture_specs.append(
                LoweredPayloadSpec(
                    provider=provider,
                    fallback_id=fallback_id if len(sequence) == 1 else f"{fallback_id}-{index}",
                    mode="browser_capture",
                    payload=item_record,
                )
            )
        if browser_capture_specs:
            return browser_capture_specs
    if record is not None and (sessions := _record_sessions(record)):
        lowered_specs: list[LoweredPayloadSpec] = []
        for index, item in enumerate(sessions):
            lowered_specs.extend(
                _lower_payload_specs(
                    runtime_provider,
                    item,
                    f"{fallback_id}-{index}",
                    depth=depth + 1,
                    schema_resolution=schema_resolution,
                )
            )
        return lowered_specs

    if runtime_provider in BUNDLE_PROVIDERS:
        return _lower_bundle_payload(runtime_provider, shaped_payload, fallback_id)
    if runtime_provider in {Provider.CLAUDE_CODE, Provider.CODEX}:
        return _lower_grouped_payload(runtime_provider, shaped_payload, fallback_id, source_path=source_path)
    if runtime_provider is Provider.BEADS:
        payloads = _payload_sequence(shaped_payload)
        if payloads is not None and all(
            (record := _payload_record(item)) is not None and beads.looks_like(record) for item in payloads
        ):
            return [_grouped_records_spec(runtime_provider, payloads, fallback_id, source_path=source_path)]
        record = _single_document_record(shaped_payload)
        if record is not None and beads.looks_like(record):
            return [_grouped_records_spec(runtime_provider, [record], fallback_id, source_path=source_path)]
        return []
    if runtime_provider is Provider.GEMINI_CLI:
        record = _single_document_record(shaped_payload)
        if record is not None and local_agent.looks_like_gemini_cli(record):
            return [_local_agent_document_spec(runtime_provider, record, fallback_id)]
        return []
    if runtime_provider in DRIVE_LIKE_PROVIDERS:
        return _lower_drive_like_payload(
            runtime_provider,
            shaped_payload,
            fallback_id,
            depth=depth,
            schema_resolution=schema_resolution,
        )
    if runtime_provider is Provider.HERMES:
        payloads = _payload_sequence(shaped_payload)
        if (
            payloads is not None
            and payloads
            and all(
                (event := _payload_record(item)) is not None and hermes_spans.looks_like_atof_payload(event)
                for item in payloads
            )
        ):
            return [_grouped_records_spec(runtime_provider, payloads, fallback_id, source_path=source_path)]
        record = _single_document_record(shaped_payload)
        if record is not None and hermes_state.looks_like_state_db_payload(record):
            return [_local_artifact_document_spec(runtime_provider, record, fallback_id, source_path=source_path)]
        if record is not None and hermes_verification.looks_like_verification_evidence_db_payload(record):
            return [_local_artifact_document_spec(runtime_provider, record, fallback_id, source_path=source_path)]
        if record is not None and hermes_spans.looks_like_atif_payload(record):
            return [_local_artifact_document_spec(runtime_provider, record, fallback_id, source_path=source_path)]
        if record is not None and local_agent.looks_like_hermes(record):
            return [_local_agent_document_spec(runtime_provider, record, fallback_id)]
        return []
    if runtime_provider is Provider.ANTIGRAVITY:
        record = _single_document_record(shaped_payload)
        if record is not None and (
            antigravity.looks_like_markdown_export(record) or antigravity.looks_like_brain_metadata(record, source_path)
        ):
            return [
                _local_artifact_document_spec(
                    runtime_provider,
                    record,
                    fallback_id,
                    source_path=source_path,
                )
            ]
        return []
    if runtime_provider is Provider.GROK:
        return _lower_grok_export_payload(shaped_payload, fallback_id)
    return _lower_fallback_payload(runtime_provider, shaped_payload, fallback_id)


def _generic_messages_session(
    provider: Provider,
    payload: PayloadRecord,
    fallback_id: str,
) -> ParsedSession | None:
    """Parse the last-resort "unknown provider, but shaped like messages" bucket.

    polylogue-b508: of every branch in ``_lower_payload_specs``, this is the
    one with no provider-specific identity handling at all -- every other
    branch routes to a parser (chatgpt/claude/codex/drive/...) that derives
    identity from provider-native evidence. Here there is none, so the
    payload itself must assert its own ``id``. Falling back to
    ``fallback_id`` -- a filename stem or scratch value the *source
    discovery walk* invented, never something the provider asserted -- is
    exactly the "session identity derived from a filename stem" pathology
    this bead exists to make unrepresentable: a JSON sidecar that merely
    happens to contain a ``messages`` list must not become a session of its
    own. Refuse to parse (return ``None``) rather than synthesize an
    identity.
    """
    messages_payload = _record_messages(payload)
    if messages_payload is None:
        return None

    # A blank id is not an assertion. ``optional_string`` returns ``""`` for an
    # empty value rather than ``None``, so an ``"id": ""`` or whitespace-only
    # field would otherwise satisfy "the provider asserted an identity" and
    # produce a session keyed on nothing -- the same pathology as a
    # filename-stem identity, arriving through the guard meant to stop it.
    asserted_id = optional_string(payload.get("id"))
    session_id = asserted_id.strip() if asserted_id is not None else None
    if not session_id:
        return None

    messages = extract_messages_from_list(messages_payload)
    title = optional_string(payload.get("title")) or optional_string(payload.get("name")) or fallback_id
    created_at = optional_string(
        payload.get("created_at") or payload.get("create_time") or payload.get("created") or payload.get("createdAt")
    )
    updated_at = optional_string(
        payload.get("updated_at")
        or payload.get("update_time")
        or payload.get("updated")
        or payload.get("updatedAt")
        or payload.get("modified")
    )
    return ParsedSession(
        source_name=provider,
        provider_session_id=session_id,
        title=title,
        created_at=created_at,
        updated_at=updated_at,
        messages=messages,
    )


def _parse_lowered_spec(spec: LoweredPayloadSpec) -> list[ParsedSession]:
    if spec.mode == "browser_capture":
        record = _payload_record(spec.payload)
        return [browser_capture.parse(record, spec.fallback_id)] if record is not None else []

    if spec.mode == "chatgpt_codex_task":
        record = _payload_record(spec.payload)
        return [chatgpt_codex_sidecar.parse_codex_task(record, spec.fallback_id)] if record is not None else []

    if spec.mode == "claude_code_multiway":
        payloads = _payload_sequence(spec.payload)
        if payloads is None:
            return []
        return list(_claude_code_multiway_parse(payloads, spec.fallback_id, source_path=spec.source_path))

    if spec.provider is Provider.CHATGPT:
        record = _payload_record(spec.payload)
        return [chatgpt.parse(record, spec.fallback_id)] if record is not None else []

    if spec.provider is Provider.CLAUDE_AI:
        record = _payload_record(spec.payload)
        return [claude.parse_ai(record, spec.fallback_id)] if record is not None else []

    if spec.provider is Provider.CLAUDE_DESIGN:
        record = _payload_record(spec.payload)
        return [claude.parse_design(record, spec.fallback_id)] if record is not None else []

    if spec.provider is Provider.GROK:
        record = _payload_record(spec.payload)
        return [grok.parse_conversation(record, spec.fallback_id)] if record is not None else []

    if spec.provider is Provider.CLAUDE_CODE:
        payloads = _payload_sequence(spec.payload)
        if payloads is None:
            return []
        return [
            claude.parse_code(
                payloads,
                spec.fallback_id,
                tool_result_sidecars=_join_claude_code_sidecars(payloads, spec.source_path),
                trust_fallback_id=spec.trust_fallback_id,
            )
        ]

    if spec.provider is Provider.CODEX:
        payloads = _payload_sequence(spec.payload)
        return [codex.parse(payloads, spec.fallback_id)] if payloads is not None else []

    if spec.provider is Provider.BEADS:
        payloads = _payload_sequence(spec.payload)
        return beads.parse(payloads, spec.fallback_id, source_path=spec.source_path) if payloads is not None else []

    if spec.provider is Provider.HERMES and spec.mode == "grouped_records":
        payloads = _payload_sequence(spec.payload)
        return (
            hermes_spans.parse_atof_stream(
                payloads,
                spec.fallback_id,
                profile_root=Path(spec.source_path).parent if spec.source_path else None,
            )
            if payloads is not None
            else []
        )

    if spec.mode == "local_agent_document":
        record = _payload_record(spec.payload)
        if record is None:
            return []
        if spec.provider is Provider.GEMINI_CLI:
            return [local_agent.parse_gemini_cli(record, spec.fallback_id)]
        if spec.provider is Provider.HERMES:
            return [local_agent.parse_hermes(record, spec.fallback_id)]
        return []

    if spec.mode == "local_artifact_document":
        record = _payload_record(spec.payload)
        if record is None:
            return []
        if spec.provider is Provider.HERMES and hermes_state.looks_like_state_db_payload(record):
            return hermes_state.parse_state_db_payload(record, spec.fallback_id)
        if spec.provider is Provider.HERMES and hermes_verification.looks_like_verification_evidence_db_payload(record):
            return hermes_verification.parse_verification_evidence_db_payload(
                record,
                spec.fallback_id,
                profile_root=Path(spec.source_path).parent if spec.source_path else None,
            )
        if spec.provider is Provider.HERMES and hermes_spans.looks_like_atif_payload(record):
            return hermes_spans.parse_atif_document(
                record,
                spec.fallback_id,
                profile_root=Path(spec.source_path).parent if spec.source_path else None,
            )
        if spec.provider is Provider.ANTIGRAVITY:
            if antigravity.looks_like_markdown_export(record):
                return [antigravity.parse_markdown_export_payload(record, spec.fallback_id)]
            source_path = Path(spec.source_path) if spec.source_path is not None else Path(f"{spec.fallback_id}.md")
            return [antigravity.parse_brain_metadata(record, source_path, spec.fallback_id)]
        return []

    if spec.mode == "chunked_prompt":
        record = _payload_record(spec.payload)
        payload: JSONDocument = record if record is not None else {"chunks": _payload_sequence(spec.payload) or []}
        return [drive.parse_chunked_prompt(spec.provider, payload, spec.fallback_id)]

    if spec.mode == "generic_messages":
        record = _payload_record(spec.payload)
        generic = _generic_messages_session(spec.provider, record, spec.fallback_id) if record is not None else None
        return [generic] if generic is not None else []

    return []


def message_carries_authored_content(message: ParsedMessage) -> bool:
    """A message counts as positive conversational evidence when it carries
    any real text or content block. A message row that exists structurally
    (a provider_message_id, a role) but has neither -- e.g. a generic
    unrecognized-record fallback that manufactures a placeholder message --
    is not evidence of a conversation."""
    if message.text is not None and message.text.strip():
        return True
    return bool(message.blocks)


def require_positive_conversational_evidence(
    sessions: list[ParsedSession],
    *,
    provider: str | Provider,
    source_path: str | None,
) -> list[ParsedSession]:
    """polylogue-9ykn: a session requires positive evidence of a conversation
    -- at minimum one message carrying authored content -- or it is refused
    loudly rather than written.

    Deliberately NOT folded into ``parse_payload``/``parse_stream_payload``
    themselves: those two functions are pure provider-routing dispatch, and
    a large "law" test surface (``tests/unit/sources/test_source_laws.py``
    and friends) monkeypatches the underlying provider parsers with
    zero-message stubs specifically to pin *routing* behavior (which parser
    got called, with what arguments, how many times) independent of parsed
    content -- folding a content gate into the dispatch functions silently
    broke 20+ of those tests by deleting the stubbed sessions before the
    test could observe them. Instead, every real production write path
    calls this filter explicitly right after it gets a ``parse_payload``/
    ``parse_stream_payload`` result back: ``pipeline/services/
    ingest_worker.py`` (subprocess decode/parse worker),
    ``sources/live/batch.py`` (in-process daemon full-ingest convergence,
    which already treats an empty session list as a recorded, bounded
    ``mark_raw_parse_failed`` outcome -- this filter reuses that existing
    "refused loudly" mechanism rather than inventing a new one),
    ``sources/live/append_ingest.py`` (incremental append), and
    ``sources/revision_backfill.py`` (offline replay/rebuild, alongside its
    own OriginSpec/``classify_artifact`` path-and-shape gate from
    polylogue-6mpy -- this filter catches the sibling case where the shape
    is recognized but the parsed *content* still carries no message).

    Measured against the live archive (2026-07-31, read-only query against
    ``index.db``/``source.db``): every verified zero-message
    ``claude-code-session`` row was one of a handful of artifact classes --
    ``agent-*.meta`` sidecars (4,945, already refused by
    ``classify_artifact``'s path rule but pre-dating it), JSONL files
    containing only non-conversational envelope records
    (file-history-snapshot/progress/bridge-session/custom-title/agent-name,
    228+ rows), and ``tool-results/*.json`` sidecars mis-dispatched as
    sessions -- plus 47 ``claude-ai-export`` conversations with a real title
    but ``chat_messages: []``. polylogue-ne6k's own investigation concluded
    no "genuinely empty but legitimate" session construct exists in this
    corpus: every zero-message row was retained only because the prior
    repair predicate could not distinguish it from one, not because it was
    verified worth keeping. This filter stops the population from growing;
    the existing rows are purged by the already-planned index rebuild
    (polylogue-x1gd), not by this change.

    Checking message *content*, not just message *count*, also closes a
    narrower sibling gap found while testing this bead: an unrecognized
    single-record document (no ``mapping``/``messages``/envelope markers at
    all) previously fell through Claude Code's generic single-document
    lowering into a one-message session whose sole message had an empty
    ``text`` and no blocks -- structurally "has a message" but zero actual
    conversational evidence.
    """
    kept: list[ParsedSession] = []
    for session in sessions:
        if any(message_carries_authored_content(message) for message in session.messages):
            kept.append(session)
            continue
        logger.warning(
            "polylogue-9ykn: refusing session %s (%s, source_path=%s) -- "
            "no messages, no positive conversational evidence",
            session.provider_session_id,
            Provider.from_string(provider),
            source_path,
        )
    return kept


def parse_payload(
    provider: str | Provider,
    payload: object,
    fallback_id: str,
    _depth: int = 0,
    *,
    schema_resolution: SchemaResolution | None = None,
    source_path: str | None = None,
) -> list[ParsedSession]:
    """Dispatch parsed payload to the appropriate provider parser.

    Pure routing: returns whatever the selected provider parser reports,
    including a zero-message session. Production write paths must apply
    ``require_positive_conversational_evidence`` to the result themselves
    (see that function's docstring for why it is not applied here).
    """
    lowered_specs = _lower_payload_specs(
        provider,
        payload,
        fallback_id,
        depth=_depth,
        schema_resolution=schema_resolution,
        source_path=source_path,
    )
    sessions: list[ParsedSession] = []
    for spec in lowered_specs:
        sessions.extend(_parse_lowered_spec(spec))
    return sessions


def parse_stream_payload(
    provider: str | Provider,
    payloads: Iterable[object],
    fallback_id: str,
    *,
    source_path: str | None = None,
) -> list[ParsedSession]:
    """Parse a grouped record stream.

    Pure routing, same contract as ``parse_payload`` -- see
    ``require_positive_conversational_evidence``'s docstring.
    """
    runtime_provider = Provider.from_string(provider)
    if runtime_provider is Provider.CLAUDE_CODE:
        return list(_claude_code_multiway_parse(payloads, fallback_id, source_path=source_path))
    if runtime_provider is Provider.CODEX:
        return [codex.parse_stream(payloads, fallback_id)]
    if runtime_provider is Provider.BEADS:
        return beads.parse(payloads, fallback_id, source_path=source_path)
    if runtime_provider is Provider.HERMES:
        return hermes_spans.parse_atof_stream(
            payloads,
            fallback_id,
            profile_root=Path(source_path).parent if source_path else None,
        )
    raise ValueError(f"provider {runtime_provider} does not support stream parsing")


__all__ = [
    "GROUP_PROVIDERS",
    "STREAM_RECORD_PROVIDERS",
    "LoweredPayloadSpec",
    "_detect_provider_from_raw_bytes",
    "detect_provider",
    "detect_provider_evidence",
    "detect_provider_from_raw_bytes_evidence",
    "is_jsonl_source_path",
    "is_stream_record_provider",
    "parse_payload",
    "parse_stream_payload",
]
