"""Provider detection and payload lowering for source parsing."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from polylogue.core.enums import Provider
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
from .parsers.base import ParsedSession, extract_messages_from_list
from .parsers.claude.code_parser import apply_tool_result_sidecars

if TYPE_CHECKING:
    from polylogue.schemas.packages import SchemaResolution
    from polylogue.sources.live.tool_result_sidecars import SidecarJoinResult

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
_NO_LOOKAHEAD = object()

PayloadRecord: TypeAlias = JSONDocument
PayloadSequence: TypeAlias = list[JSONValue]
LoweredPayloadMode: TypeAlias = Literal[
    "bundle_record",
    "browser_capture",
    "chatgpt_codex_task",
    "chunked_prompt",
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


def _detect_provider_from_record(record: PayloadRecord) -> Provider | None:
    if browser_capture.looks_like(record):
        session = record.get("session")
        provider = session.get("provider") if isinstance(session, dict) else None
        return Provider.from_string(provider if isinstance(provider, str) else None)
    # Local-agent JSON session documents share enough generic message keys with
    # Claude Code that they must be recognized before broader validators.
    if local_agent.looks_like_gemini_cli(record):
        return Provider.GEMINI_CLI
    if hermes_state.looks_like_state_db_payload(record):
        return Provider.HERMES
    if hermes_verification.looks_like_verification_evidence_db_payload(record):
        return Provider.HERMES
    if hermes_spans.looks_like_atif_payload(record):
        return Provider.HERMES
    if hermes_spans.looks_like_atof_payload(record):
        return Provider.HERMES
    if local_agent.looks_like_hermes(record):
        return Provider.HERMES
    if antigravity.looks_like_markdown_export(record):
        return Provider.ANTIGRAVITY
    if antigravity.looks_like_brain_metadata(record, None):
        return Provider.ANTIGRAVITY
    if beads.looks_like(record):
        return Provider.BEADS
    # Specific type-level checks first (Codex uses Pydantic validation;
    # Claude Code uses a dict-key/type shape check, not Pydantic, despite
    # ClaudeCodeRecord existing as a separate typed parse-time model), then
    # weaker dict-key checks (ChatGPT, Claude AI, Gemini).
    if codex.looks_like([dict(record)]):
        return Provider.CODEX
    if claude.looks_like_code([dict(record)]):
        return Provider.CLAUDE_CODE
    # A single record here may be an intentionally partial ChatGPT fragment
    # (e.g. one line of a streamed JSONL sniff), not a whole assembled
    # export document, so this uses the fragment-level check rather than
    # ``chatgpt.looks_like``'s document-identity requirements (polylogue-t0ta).
    if chatgpt.looks_like_fragment(record):
        return Provider.CHATGPT
    # Claude Design (bd polylogue-tbun) checked before the general claude.ai
    # detector: its shape (messages + project, no chat_messages, camelCase
    # contentBlocks) is a distinct, tighter product signature -- not a
    # claude.ai variant.
    if claude.looks_like_claude_design(record):
        return Provider.CLAUDE_DESIGN
    if claude.looks_like_claude_memories(record):
        return Provider.CLAUDE_AI
    if claude.looks_like_ai(record):
        return Provider.CLAUDE_AI
    if grok.looks_like_export(record):
        return Provider.GROK
    if _looks_like_gemini_mapping(record):
        return Provider.GEMINI
    return None


def _detect_provider_from_sequence(payloads: PayloadSequence) -> Provider | None:
    if not payloads:
        return None

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
            return Provider.GEMINI_CLI
        if browser_capture.looks_like(first_record):
            return _detect_provider_from_record(first_record)
        if hermes_spans.looks_like_atof_payload(first_record):
            return Provider.HERMES
        if beads.looks_like(first_record):
            return Provider.BEADS
        # The first record of a *sequence* is a whole assembled document
        # (e.g. one conversation from a ChatGPT bundle array), not a
        # partial per-line fragment, so this uses the strict document-level
        # check rather than ``looks_like_fragment`` (polylogue-t0ta).
        if chatgpt.looks_like(first_record):
            return Provider.CHATGPT
        if isinstance(first_record.get("chat_messages"), list):
            return Provider.CLAUDE_AI
        # Claude AI account memory export (memories.json, bd polylogue-zng9)
        # arrives as a bare top-level JSON array of one-per-account records.
        if claude.looks_like_claude_memories(first_record):
            return Provider.CLAUDE_AI
        if claude.looks_like_claude_design(first_record):
            return Provider.CLAUDE_DESIGN
        if grok.looks_like_export(first_record):
            return Provider.GROK
        if _looks_like_gemini_mapping(first_record):
            return Provider.GEMINI

    if claude.looks_like_code(payloads):
        return Provider.CLAUDE_CODE
    if codex.looks_like(payloads):
        return Provider.CODEX
    return None


def detect_provider(payload: object, path: object | None = None) -> Provider | None:
    """Infer provider from payload shape. Path is accepted for surface compatibility."""
    del path

    if record := _payload_record(payload):
        return _detect_provider_from_record(record)
    payloads = _payload_sequence(payload)
    return _detect_provider_from_sequence(payloads) if payloads is not None else None


def _detect_provider_from_raw_bytes(
    raw_bytes: bytes,
    stream_name: str,
    fallback_provider: Provider,
    *,
    truncated_tail_ok: bool = False,
) -> Provider:
    jsonl_like = _is_jsonl_stream_name(stream_name)
    text = None if jsonl_like else _decode_json_bytes(raw_bytes)
    if text is not None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        else:
            detected = detect_provider(payload)
            if detected is not None:
                return detected

    stream_bytes = _trim_jsonl_detection_prefix(raw_bytes, stream_name) if truncated_tail_ok else raw_bytes
    if not stream_bytes:
        return fallback_provider
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
        return fallback_provider

    return detect_provider(payloads) or fallback_provider


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
) -> LoweredPayloadSpec:
    return LoweredPayloadSpec(
        provider=provider,
        fallback_id=fallback_id,
        mode="grouped_records",
        payload=payload,
        source_path=source_path,
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


def _claude_code_grouped_record_specs(
    payloads: PayloadSequence,
    fallback_id: str,
    *,
    source_path: str | None = None,
) -> list[LoweredPayloadSpec]:
    """Split concatenated Claude Code JSONL aggregates into session streams."""
    current_session_id: str | None = None
    groups: dict[str, PayloadSequence] = {}
    pending_prefix: PayloadSequence = []

    for payload in payloads:
        record = _payload_record(payload)
        session_id = optional_string(record.get("sessionId")) if record is not None else None
        if session_id is None:
            if current_session_id is None:
                pending_prefix.append(payload)
            else:
                groups.setdefault(current_session_id, []).append(payload)
            continue

        if current_session_id is None:
            groups.setdefault(session_id, []).extend(pending_prefix)
            pending_prefix = []

        current_session_id = session_id
        groups.setdefault(session_id, []).append(payload)

    if len(groups) <= 1:
        return [_grouped_records_spec(Provider.CLAUDE_CODE, payloads, fallback_id, source_path=source_path)]
    return [
        _grouped_records_spec(
            Provider.CLAUDE_CODE,
            group_payloads,
            fallback_id if index == 0 else group_id,
            source_path=source_path,
        )
        for index, (group_id, group_payloads) in enumerate(groups.items())
    ]


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
        if active_leaf_message_provider_id is not None:
            messages = [
                message.model_copy(
                    update={
                        "position": position,
                        "is_active_leaf": message.provider_message_id == active_leaf_message_provider_id,
                    }
                )
                for position, message in enumerate(messages)
            ]

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
        merged[session.provider_session_id] = existing.model_copy(
            update={
                "title": existing.title if existing.title != existing.provider_session_id else session.title,
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
    sessions = list(merged.values())
    return [
        claude.reconcile_code_session_chunks(session) if session.source_name is Provider.CLAUDE_CODE else session
        for session in sessions
    ]


def _claude_code_stream_sessions(
    payloads: Iterable[object],
    fallback_id: str,
    *,
    source_path: str | None = None,
) -> Iterator[ParsedSession]:
    """Parse Claude Code JSONL records without materializing the full stream.

    The eager ``parse_payload`` path preserves the strongest non-contiguous
    grouping semantics for already-materialized payloads. Raw JSONL ingest and
    repair, however, can be multi-GiB; for that path we split only on contiguous
    ``sessionId`` changes and feed each group to the provider parser as an
    iterator. Per-session record-index and UUID continuation state retains no
    raw payload bytes; its size is proportional to unique record identifiers and
    makes an interleaved stream semantically identical to eager grouping.

    Sidecar join (polylogue-wjgf): the raw payload is never materialized here,
    so ``join_tool_result_sidecars`` (which needs the full ``tool_use_id``
    index) can't run against it directly. Instead each group's records are
    teed through a ``ToolResultIndexAccumulator`` as they stream past
    (``observe_tool_result_stream``), and the join runs against the resulting
    index only after the group's iterator is fully consumed -- no raw record
    retention, same memory bound as the rest of this path.
    """
    tool_results_dir = None
    if source_path is not None:
        from polylogue.sources.live.tool_result_sidecars import resolve_tool_results_dir

        tool_results_dir = resolve_tool_results_dir(source_path)
        if tool_results_dir is not None and not tool_results_dir.is_dir():
            tool_results_dir = None

    iterator = iter(payloads)
    lookahead: object = _NO_LOOKAHEAD
    pending_prefix: list[object] = []
    first_group = True
    record_counts_by_session: dict[str, int] = {}
    seen_record_uuids_by_session: dict[str, set[str]] = {}

    def next_item() -> object:
        nonlocal lookahead
        if lookahead is not _NO_LOOKAHEAD:
            item = lookahead
            lookahead = _NO_LOOKAHEAD
            return item
        return next(iterator)

    def parse_group(
        records: Iterator[object],
        group_fallback_id: str,
        *,
        record_index_start: int = 0,
        seen_record_uuids: set[str] | None = None,
    ) -> ParsedSession:
        if tool_results_dir is None:
            return claude.parse_code_stream(
                records,
                group_fallback_id,
                record_index_start=record_index_start,
                seen_record_uuids=seen_record_uuids,
            )
        from polylogue.sources.live.tool_result_sidecars import (
            ToolResultIndexAccumulator,
            observe_tool_result_stream,
        )

        accumulator = ToolResultIndexAccumulator()
        session = claude.parse_code_stream(
            observe_tool_result_stream(records, accumulator),
            group_fallback_id,
            record_index_start=record_index_start,
            seen_record_uuids=seen_record_uuids,
        )
        assert source_path is not None  # tool_results_dir is only set when source_path is
        return apply_tool_result_sidecars(session, accumulator.join_session_scoped(tool_results_dir, source_path))

    while True:
        try:
            first = next_item()
        except StopIteration:
            if pending_prefix:
                yield parse_group(iter(pending_prefix), fallback_id)
            return

        first_record = _payload_record(first)
        first_session_id = optional_string(first_record.get("sessionId")) if first_record is not None else None
        if first_session_id is None:
            pending_prefix.append(first)
            continue

        group_session_id = first_session_id
        group_fallback_id = fallback_id if first_group else group_session_id
        first_group = False
        prefix = pending_prefix
        pending_prefix = []

        group_record_count = 0

        def group_records(
            prefix: list[object] = prefix,
            first: object = first,
            group_session_id: str = group_session_id,
        ) -> Iterator[object]:
            nonlocal group_record_count, lookahead
            for prefix_item in prefix:
                group_record_count += 1
                yield prefix_item
            group_record_count += 1
            yield first
            for item in iterator:
                record = _payload_record(item)
                session_id = optional_string(record.get("sessionId")) if record is not None else None
                if session_id is not None and session_id != group_session_id:
                    lookahead = item
                    return
                group_record_count += 1
                yield item

        record_index_start = record_counts_by_session.get(group_session_id, 0)
        seen_record_uuids = seen_record_uuids_by_session.setdefault(group_session_id, set())
        session = parse_group(
            group_records(),
            group_fallback_id,
            record_index_start=record_index_start,
            seen_record_uuids=seen_record_uuids,
        )
        record_counts_by_session[group_session_id] = record_index_start + group_record_count
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
            return _claude_code_grouped_record_specs(payloads, fallback_id, source_path=source_path)
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
                payloads, spec.fallback_id, tool_result_sidecars=_join_claude_code_sidecars(payloads, spec.source_path)
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


def parse_payload(
    provider: str | Provider,
    payload: object,
    fallback_id: str,
    _depth: int = 0,
    *,
    schema_resolution: SchemaResolution | None = None,
    source_path: str | None = None,
) -> list[ParsedSession]:
    """Dispatch parsed payload to the appropriate provider parser."""
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
    """Parse a grouped record stream."""
    runtime_provider = Provider.from_string(provider)
    if runtime_provider is Provider.CLAUDE_CODE:
        return merge_parsed_session_chunks(_claude_code_stream_sessions(payloads, fallback_id, source_path=source_path))
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


def parse_drive_payload(
    provider: str | Provider,
    payload: object,
    fallback_id: str,
    _depth: int = 0,
) -> list[ParsedSession]:
    """Adapter for Drive/Gemini payload parsing."""
    runtime_provider = Provider.from_string(provider)
    if _depth > _MAX_PARSE_DEPTH:
        logger.warning("Recursion depth exceeded parsing drive payload %s", fallback_id)
        return []

    payloads = _payload_sequence(payload)
    if payloads is not None:
        if payloads and all(isinstance(item, str) or _payload_record(item) is not None for item in payloads):
            first_record = _payload_record(payloads[0]) if payloads else None
            if first_record is None or "role" in first_record or "text" in first_record:
                spec = _chunked_prompt_spec(runtime_provider, payloads, fallback_id)
                return _parse_lowered_spec(spec)

        nested_sessions: list[ParsedSession] = []
        for index, item in enumerate(payloads):
            if _looks_like_chunked_session(item):
                nested_sessions.extend(
                    parse_drive_payload(
                        runtime_provider,
                        item,
                        f"{fallback_id}-{index}",
                        _depth + 1,
                    )
                )
                continue
            detected = detect_provider(item) or runtime_provider
            nested_sessions.extend(
                parse_payload(
                    detected,
                    item,
                    f"{fallback_id}-{index}",
                    _depth + 1,
                )
            )
        return nested_sessions

    record = _payload_record(payload)
    if record is None:
        return []
    if "chunkedPrompt" in record or "chunks" in record:
        spec = _chunked_prompt_spec(runtime_provider, record, fallback_id)
        return _parse_lowered_spec(spec)

    detected = detect_provider(record) or runtime_provider
    return parse_payload(detected, record, fallback_id, _depth + 1)


__all__ = [
    "GROUP_PROVIDERS",
    "STREAM_RECORD_PROVIDERS",
    "LoweredPayloadSpec",
    "_detect_provider_from_raw_bytes",
    "detect_provider",
    "is_jsonl_source_path",
    "is_stream_record_provider",
    "parse_drive_payload",
    "parse_payload",
    "parse_stream_payload",
]
