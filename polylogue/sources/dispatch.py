"""Provider detection and payload lowering for source parsing."""

from __future__ import annotations

import json
import sys
from builtins import BaseExceptionGroup
from collections.abc import Callable, Generator, Iterable, Iterator, MutableSequence, Sequence
from contextlib import ExitStack, closing
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import IO, TYPE_CHECKING, Literal, TypeAlias, cast

from polylogue.browser_capture.models import BrowserCaptureEnvelope, has_chatgpt_native_payload
from polylogue.core.binary_signatures import detect_binary_signature
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import Provider, TitleSource
from polylogue.core.json import (
    JSONDocument,
    JSONValue,
    is_json_value,
    json_document_or_none,
    normalize_json_decimal,
)
from polylogue.core.payload_coercion import optional_string
from polylogue.core.provider_identity import profile_root_for_artifact
from polylogue.core.timestamp_authority import timestamp_millis
from polylogue.logging import WARNING, emit, get_logger

from .chunk_positions import ChunkPositions
from .decoder_json import DecodedRecordSequence
from .detection import DetectionMode
from .detection_projection import DetectionReadMapping
from .origin_specs import detector_registry
from .parsers import (
    antigravity,
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
    otel_genai,
)
from .parsers.base import (
    ParsedAttachment,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
    extract_messages_from_list,
    mark_last_occurrence_as_active_leaf,
)
from .parsers.base_models import upgrade_chat_export_user_authorship
from .parsers.base_support import (
    AdmissionObserver,
    admit_parsed_sessions,
    claude_code_unknown_wire_type,
    codex_unknown_wire_type,
    hermes_unknown_wire_type,
    iter_messages_from_list,
)
from .parsers.claude import code_parser as claude_code_parser
from .parsers.claude.code_parser import apply_tool_result_sidecars
from .parsers.claude.stream_scratch import ClaudeStreamScratch, SqliteStringSet
from .sidecar_evidence import SidecarResolver

if TYPE_CHECKING:
    from polylogue.schemas.packages import SchemaResolution
    from polylogue.sources.live.tool_result_sidecars import SidecarJoinResult, ToolResultIndexAccumulator

logger = get_logger(__name__)

BUNDLE_PROVIDERS = frozenset({Provider.CHATGPT, Provider.CLAUDE_AI, Provider.CLAUDE_DESIGN})
GROUP_PROVIDERS = frozenset({Provider.CLAUDE_CODE, Provider.CODEX, Provider.GEMINI, Provider.DRIVE, Provider.HERMES})
STREAM_RECORD_PROVIDERS = frozenset({Provider.CLAUDE_CODE, Provider.CODEX, Provider.HERMES})
DRIVE_LIKE_PROVIDERS = frozenset({Provider.GEMINI, Provider.DRIVE})

PayloadRecord: TypeAlias = JSONDocument
PayloadSequence: TypeAlias = Sequence[JSONValue]
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


@dataclass(frozen=True, slots=True)
class _PayloadLoweringRequest:
    provider: str | Provider
    payload: object
    fallback_id: str
    schema_resolution: SchemaResolution | None = None
    source_path: str | None = None


@dataclass(frozen=True, slots=True)
class ChatGPTLoweredDocument:
    """One ChatGPT document after the production source-lowering route.

    This is intentionally a source-side projection: verification may inspect
    the acquired mapping independently, but it must not invent another
    envelope vocabulary or document splitter.
    """

    document_id: str
    mapping: PayloadRecord
    artifact_class: str


def _payload_record(value: object) -> PayloadRecord | None:
    if isinstance(value, DetectionReadMapping):
        return value.json_record()
    return json_document_or_none(value)


def _payload_sequence(value: object) -> PayloadSequence | None:
    if isinstance(value, DecodedRecordSequence):
        return value
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
    object per file. Intake passes a repeatable decoded record sequence, so
    a one-record file arrives here as a one-element sequence rather than a
    bare dict. ``_payload_record`` returns
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
    reachable value elsewhere -- explicit source configs and retained raw
    rows -- it is simply never *produced* by this detector.
    """
    return drive.looks_like(record)


def _sequence_has_record(payload: object, predicate: Callable[[PayloadRecord], bool]) -> bool:
    if not isinstance(payload, (list, DecodedRecordSequence)):
        return False
    return any((record := _payload_record(item)) is not None and predicate(record) for item in payload)


def _looks_like_browser_capture_record(payload: object) -> bool:
    record = _payload_record(payload)
    return record is not None and browser_capture.looks_like(record)


def _browser_capture_provider(payload: object) -> Provider | None:
    record = _payload_record(payload)
    session = record.get("session") if record is not None else None
    provider = session.get("provider") if isinstance(session, dict) else None
    return Provider.from_string(provider if isinstance(provider, str) else None)


def _declared_capture_provider(record: PayloadRecord) -> Provider | None:
    """The provider a browser-capture envelope declares; the envelope owns it.

    Location binding for captures is enforced at acquisition, where the
    location is known; here ``runtime_provider`` is only a parser hint (a
    mixed capture sequence's first element), so each envelope keeps its own
    declared provider.
    """
    provider = _browser_capture_provider(record)
    return None if provider in (None, Provider.UNKNOWN) else provider


def _looks_like_browser_capture_sequence(payload: object) -> bool:
    return _sequence_has_record(payload, browser_capture.looks_like)


def _browser_capture_sequence_provider(payload: object) -> Provider | None:
    if not isinstance(payload, (list, DecodedRecordSequence)):
        return None
    for item in payload:
        record = _payload_record(item)
        if record is not None and browser_capture.looks_like(record):
            return _browser_capture_provider(record)
    return None


def _looks_like_gemini_cli_record(payload: object) -> bool:
    record = _payload_record(payload)
    return record is not None and local_agent.looks_like_gemini_cli(record)


def _looks_like_gemini_cli_sequence_document(payload: object) -> bool:
    return _sequence_has_record(payload, local_agent.looks_like_gemini_cli)


def _looks_like_hermes_state_record(payload: object) -> bool:
    record = _payload_record(payload)
    return record is not None and hermes_state.looks_like_state_db_payload(record)


def _looks_like_hermes_verification_record(payload: object) -> bool:
    record = _payload_record(payload)
    return record is not None and hermes_verification.looks_like_verification_evidence_db_payload(record)


def _looks_like_hermes_atif_record(payload: object) -> bool:
    record = _payload_record(payload)
    return record is not None and hermes_spans.looks_like_atif_payload(record)


def _looks_like_hermes_atof_record(payload: object) -> bool:
    record = _payload_record(payload)
    return record is not None and hermes_spans.looks_like_atof_payload(record)


def _looks_like_hermes_local_agent_record(payload: object) -> bool:
    record = _payload_record(payload)
    return record is not None and local_agent.looks_like_hermes(record)


def _looks_like_hermes_atof_sequence(payload: object) -> bool:
    return _sequence_has_record(payload, hermes_spans.looks_like_atof_payload)


def _looks_like_antigravity_markdown_record(payload: object) -> bool:
    record = _payload_record(payload)
    return record is not None and antigravity.looks_like_markdown_export(record)


def _looks_like_codex_record(payload: object) -> bool:
    record = _payload_record(payload)
    return record is not None and codex.looks_like([record])


def _looks_like_codex_stream(payload: object) -> bool:
    return isinstance(payload, (list, DecodedRecordSequence)) and codex.looks_like(payload)


def _looks_like_claude_code_record(payload: object) -> bool:
    record = _payload_record(payload)
    return record is not None and claude.looks_like_code([record])


def _looks_like_claude_code_stream(payload: object) -> bool:
    return isinstance(payload, (list, DecodedRecordSequence)) and claude.looks_like_code(payload)


def _looks_like_chatgpt_fragment_record(payload: object) -> bool:
    record = _payload_record(payload)
    return record is not None and chatgpt.looks_like_fragment(record)


def _looks_like_chatgpt_shared_decode_record(payload: object) -> bool:
    record = _payload_record(payload)
    return record is not None and chatgpt.looks_like_shared_decode(record)


def _looks_like_chatgpt_sequence_document(payload: object) -> bool:
    return _sequence_has_record(payload, chatgpt.looks_like)


def _looks_like_claude_design_record(payload: object) -> bool:
    record = _payload_record(payload)
    return record is not None and claude.looks_like_claude_design(record)


def _looks_like_claude_design_sequence(payload: object) -> bool:
    return _sequence_has_record(payload, claude.looks_like_claude_design)


def _looks_like_claude_memories_record(payload: object) -> bool:
    record = _payload_record(payload)
    return record is not None and claude.looks_like_claude_memories(record)


def _looks_like_claude_memories_sequence(payload: object) -> bool:
    return _sequence_has_record(payload, claude.looks_like_claude_memories)


def _looks_like_claude_project_record(payload: object) -> bool:
    record = _payload_record(payload)
    return record is not None and claude.looks_like_claude_project(record)


def _looks_like_claude_project_sequence(payload: object) -> bool:
    return _sequence_has_record(payload, claude.looks_like_claude_project)


def _looks_like_claude_ai_record(payload: object) -> bool:
    record = _payload_record(payload)
    return record is not None and claude.looks_like_ai(record)


def _looks_like_claude_ai_sequence(payload: object) -> bool:
    return _sequence_has_record(payload, claude.looks_like_ai)


def _looks_like_grok_native_record(payload: object) -> bool:
    record = _payload_record(payload)
    return record is not None and grok.looks_like_native_bundle(record)


def _looks_like_grok_native_sequence(payload: object) -> bool:
    return _sequence_has_record(payload, grok.looks_like_native_bundle)


def _looks_like_grok_record(payload: object) -> bool:
    record = _payload_record(payload)
    return record is not None and grok.looks_like_export(record)


def _looks_like_otel_genai_record(payload: object) -> bool:
    """Recognize an OTLP-JSON trace export carrying GenAI attributes."""
    return otel_genai.looks_like(payload)


def _looks_like_grok_sequence(payload: object) -> bool:
    return _sequence_has_record(payload, grok.looks_like_export)


def _looks_like_gemini_mapping_record(payload: object) -> bool:
    record = _payload_record(payload)
    return record is not None and _looks_like_gemini_mapping(record)


def _looks_like_gemini_mapping_sequence(payload: object) -> bool:
    return _sequence_has_record(payload, _looks_like_gemini_mapping)


class ForeignOriginContentError(ValueError):
    """Content at a location bound to one origin carries another origin's shape.

    A source location admits only its own origin's material. A Codex rollout
    in Claude Code's project directory, or a Gemini CLI prompt log that
    happens to look like Claude Code records, is refused -- never reparsed as
    the origin its shape suggests, and never silently skipped.
    """

    code = "foreign_origin_content"

    def __init__(self, *, expected: Provider, found: Provider, evidence: str) -> None:
        super().__init__(
            f"content at a {expected.value} location has {found.value} shape ({evidence}); "
            "refused: a location admits only its own origin"
        )
        self.expected = expected
        self.found = found
        self.evidence = evidence

    def __reduce__(self) -> tuple[object, ...]:
        # Parse workers run in subprocesses; the refusal must survive pickling.
        return (_rebuild_foreign_origin_error, (self.expected.value, self.found.value, self.evidence))


def _rebuild_foreign_origin_error(expected: str, found: str, evidence: str) -> ForeignOriginContentError:
    return ForeignOriginContentError(expected=Provider(expected), found=Provider(found), evidence=evidence)


def bound_location_provider(expected: Provider | str | None) -> Provider | None:
    """The origin a location binds, or ``None`` for a classifying location.

    Only locations without a single owning origin -- the operator's import
    inbox and browser-capture envelopes, which declare their provider -- run
    shape classification. Every other location binds its origin, and shape
    detection there only validates.
    """
    if expected is None:
        return None
    provider = Provider.from_string(expected)
    return None if provider is Provider.UNKNOWN else provider


def detect_provider_evidence(
    payload: object,
    path: object | None = None,
    *,
    expected: Provider | str | None = None,
) -> tuple[Provider | None, str]:
    """Infer provider from payload shape, plus the evidence that decided it.

    With ``expected`` naming a bound location origin, the result is either
    that origin or ``None``; a payload carrying another origin's shape raises
    :class:`ForeignOriginContentError`. ``detect_provider`` is a thin wrapper
    that discards the evidence label.
    """
    del path
    provider, evidence = _classify_provider_evidence(payload)
    bound = bound_location_provider(expected)
    if bound is not None and provider is not None and not same_origin(provider, bound):
        raise ForeignOriginContentError(expected=bound, found=provider, evidence=evidence)
    return provider, evidence


def same_origin(left: Provider, right: Provider) -> bool:
    """Whether two provider wires name the same archive origin.

    Wires are not origins: ``drive`` and ``gemini`` both denote AI Studio on
    Drive, so a Drive location validating a ``gemini``-shaped prompt is its
    own origin, not foreign content.
    """
    if left is right:
        return True
    from polylogue.core.sources import origin_from_provider

    return origin_from_provider(left) is origin_from_provider(right)


def _validate_sequence_document_origins(payloads: PayloadSequence, expected: Provider) -> None:
    """Validate every complete document using the same tightness-ordered registry.

    Fragment-only records do not declare a document origin. Browser envelopes
    carry their own declared provider and remain a legitimate mixed bundle.
    """
    if expected is Provider.UNKNOWN:
        return
    for item in payloads:
        check_compute_cancelled()
        record = _payload_record(item)
        if record is None or browser_capture.looks_like(record):
            continue
        found, evidence = detector_registry().detect(DetectionMode.SEQUENCE_DOCUMENT, [record])
        if found is not None and not same_origin(found, expected):
            raise ForeignOriginContentError(expected=expected, found=found, evidence=evidence or "complete document")


def _classify_provider_evidence(payload: object) -> tuple[Provider | None, str]:
    if record := _payload_record(payload):
        provider, evidence = detector_registry().detect(DetectionMode.RECORD, record)
        return provider, evidence or "no detector matched (single record)"
    payloads = _payload_sequence(payload)
    if payloads is not None:
        if not payloads:
            return None, "empty sequence"
        provider, evidence = detector_registry().detect(DetectionMode.SEQUENCE_DOCUMENT, payloads)
        if evidence is not None:
            if provider is not None:
                _validate_sequence_document_origins(payloads, provider)
            return provider, evidence
        provider, evidence = detector_registry().detect(DetectionMode.SEQUENCE_RECORD_STREAM, payloads)
        return provider, evidence or "no detector matched (sequence)"
    return None, "payload is not a JSON document or sequence"


def detect_provider(
    payload: object,
    path: object | None = None,
    *,
    expected: Provider | str | None = None,
) -> Provider | None:
    """Infer provider from payload shape, validated against a bound location origin."""
    return detect_provider_evidence(payload, path, expected=expected)[0]


def detect_provider_from_stream_evidence(
    handle: IO[bytes],
    *,
    expected: Provider | str | None = None,
    check_stop: Callable[[], None] | None = None,
) -> tuple[Provider | None, str]:
    """Detect from complete acquired input under the declaration-owned registry."""
    provider, evidence = detector_registry().detect_stream(handle, check_stop=check_stop)
    evidence = evidence or "no detector matched (complete acquired stream)"
    bound = bound_location_provider(expected)
    if bound is not None and provider is not None and not same_origin(provider, bound):
        raise ForeignOriginContentError(expected=bound, found=provider, evidence=evidence)
    return provider, evidence


def _detect_record_stream_evidence(
    handle: IO[bytes],
    *,
    check_stop: Callable[[], None],
) -> tuple[Provider | None, str | None]:
    """Classify every decoded record; ``(None, None)`` names an empty stream.

    A malformed record anywhere raises after the records before it were
    seen, so a late bad line still denies provider proof.
    """
    from .detection_projection import iter_decoded_jsonl_records

    seen = False

    def records() -> Iterator[object]:
        nonlocal seen
        for record in iter_decoded_jsonl_records(handle, check_stop=check_stop):
            seen = True
            yield record

    detected, evidence = detector_registry().detect_record_stream(records(), check_stop=check_stop)
    if not seen:
        return None, None
    return detected, evidence or "no detector matched (complete record stream)"


def detect_provider_from_raw_stream_evidence(
    handle: IO[bytes],
    stream_name: str,
    fallback_provider: Provider,
    *,
    truncated_tail_ok: bool = False,
    check_stop: Callable[[], None] | None = None,
) -> tuple[Provider, str]:
    """Classify the complete input, preserving the caller's syntax evidence."""
    import ijson

    callback_failure: BaseException | None = None

    def checkpoint() -> None:
        nonlocal callback_failure
        if check_stop is not None:
            try:
                check_stop()
            except BaseException as exc:
                callback_failure = exc
                raise

    checkpoint()
    position = handle.tell()
    signature = detect_binary_signature(handle.read(32))
    handle.seek(position)
    if signature is not None:
        return fallback_provider, f"{signature.name}-shaped payload; refused as session content, used fallback_provider"
    try:
        if is_jsonl_source_path(stream_name):
            # A record stream is classified record by record through the
            # parser's own JSONL decoder: one pass, one record held at a time,
            # never projected as a single JSON document.
            if truncated_tail_ok:
                from .live.batch_support import jsonl_parse_input_of_handle

                with jsonl_parse_input_of_handle(handle, check_stop=checkpoint) as accepted:
                    detected, evidence = _detect_record_stream_evidence(accepted, check_stop=checkpoint)
            else:
                detected, evidence = _detect_record_stream_evidence(handle, check_stop=checkpoint)
            if evidence is None:
                return fallback_provider, "empty accepted record stream; used fallback_provider"
        else:
            detected, evidence = detect_provider_from_stream_evidence(handle, check_stop=checkpoint)
    except (ijson.JSONError, UnicodeError, json.JSONDecodeError) as exc:
        if callback_failure is not None:
            raise callback_failure from None
        return fallback_provider, f"stream decode error ({type(exc).__name__}: {exc}); used fallback_provider"
    finally:
        handle.seek(position)
    return (detected, evidence) if detected is not None else (fallback_provider, f"{evidence}; used fallback_provider")


def detect_provider_from_raw_bytes_evidence(
    raw_bytes: bytes,
    stream_name: str,
    fallback_provider: Provider,
    *,
    truncated_tail_ok: bool = False,
) -> tuple[Provider, str]:
    """Use the same complete input owner for an already acquired byte unit."""
    return detect_provider_from_raw_stream_evidence(
        BytesIO(raw_bytes),
        stream_name,
        fallback_provider,
        truncated_tail_ok=truncated_tail_ok,
    )


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
    *,
    source_path: str | None = None,
) -> LoweredPayloadSpec:
    return LoweredPayloadSpec(
        provider=provider,
        fallback_id=fallback_id,
        mode="local_agent_document",
        payload=payload,
        source_path=source_path,
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


def _default_sidecar_resolver() -> SidecarResolver:
    """Acquisition-time resolution, the routing default.

    Deferred import: ``polylogue.sources.live`` (package ``__init__``) pulls in
    ``batch.py``/``watcher.py``, which import back ``from
    polylogue.sources.dispatch import ...`` -- a module-level import here would
    be circular whenever ``dispatch`` is the first module imported. Calling
    this only at parse time (long after both modules are fully loaded) avoids
    it without restructuring either package.

    A derivation route must pass its own ``RetainedSidecarResolver``: this one
    reads the source tree, which is the input during acquisition and gone
    during a later reparse (polylogue-cq1ql).
    """
    from polylogue.sources.live.sidecar_resolution import FilesystemSidecarResolver

    return FilesystemSidecarResolver()


def _join_claude_code_sidecars(
    payloads: PayloadSequence,
    source_path: str | None,
    resolver: SidecarResolver,
) -> SidecarJoinResult | None:
    """Join ``tool-results/`` sidecar content for a Claude Code JSONL payload, if any.

    Returns ``None`` (a no-op for ``parse_code``/``parse_code_stream``) when
    there is no ``source_path`` to derive a scope coordinate from, and an
    empty result when ``resolver`` has no evidence for that scope.
    """
    if source_path is None:
        return None
    from polylogue.sources.live.tool_result_sidecars import join_tool_result_sidecars_session_scoped

    scope = resolver.claude_code_scope(source_path)
    if not scope.available:
        return None
    return join_tool_result_sidecars_session_scoped(payloads, scope, source_path)


# Declared precedence over title EVIDENCE, most authoritative tier first.
# Merging streamed chunks of one session must never regress a stronger title
# found in a later chunk in favor of a weaker one that resolved earlier, so a
# chunk's title is chosen by where its ``(title_source, title_ref)`` evidence
# sits in this order rather than by a score.
#
# Entries within one tier are equally authoritative: a tie keeps the chunk
# already held. A ``None`` prefix matches any ref for that source and applies
# only when no tier names a prefix the ref actually carries.
_TITLE_EVIDENCE_PRECEDENCE: tuple[tuple[tuple[TitleSource, str | None], ...], ...] = (
    # An explicit user rename outranks the provider's own computed title.
    ((TitleSource.ORIGIN, "claude-custom-title:"),),
    ((TitleSource.ORIGIN, "claude-ai-title:"),),
    # Provider-curated titles with no weaker-tier prefix (Claude AI's own
    # title/name, Codex's thread name).
    ((TitleSource.ORIGIN, None),),
    # Provider-assigned labels rather than titles of the content itself.
    (
        (TitleSource.ORIGIN, "claude-agent-name:"),
        (TitleSource.ORIGIN, "codex-history:"),
    ),
    ((TitleSource.ORIGIN, "codex-state-db:"),),
    ((TitleSource.ORIGIN, "codex-thread-title-hook-event:"),),
    # Parser heuristics (first human message, prompt-echo demotions) rank
    # below every provider signal and do not order among themselves.
    ((TitleSource.HEURISTIC, None),),
)


def _title_evidence_rank(session: ParsedSession) -> int:
    """Rank a chunk's title evidence; higher wins, 0 means no evidence.

    A NULL ``title_source`` ranks 0: a raw-id fallback title carries no
    evidence to prefer.
    """
    source = session.title_source
    if source is None:
        return 0
    ref = session.title_ref or ""
    tiers = _TITLE_EVIDENCE_PRECEDENCE
    for index, tier in enumerate(tiers):
        for tier_source, prefix in tier:
            if prefix is not None and tier_source is source and ref.startswith(prefix):
                return len(tiers) - index
    for index, tier in enumerate(tiers):
        for tier_source, prefix in tier:
            if prefix is None and tier_source is source:
                return len(tiers) - index
    return 0


def _later_chunk_title_winner(existing: ParsedSession, later: ParsedSession) -> ParsedSession:
    """Resolve title evidence across two consecutive chunks of one stream.

    Stronger evidence wins. Between equal provider records the later chunk
    wins, as the whole-file parse keeps the latest rename or ai-title record;
    a first-human-message heuristic (and absent evidence) keeps the earlier
    chunk, whose first message it names.
    """
    existing_rank = _title_evidence_rank(existing)
    later_rank = _title_evidence_rank(later)
    if existing_rank != later_rank:
        return existing if existing_rank > later_rank else later
    if existing.title_source is TitleSource.ORIGIN:
        return later
    return existing


def merge_parsed_session_chunks(sessions: Iterable[ParsedSession]) -> list[ParsedSession]:
    """Merge repeated provider-native sessions produced by streaming chunks."""

    def merge_claude_count_summary_events(
        events: list[ParsedSessionEvent],
        *,
        event_type: str,
        payload_keys: tuple[str, ...],
        timestamp: str | None,
        keep_empty_keys: bool,
    ) -> list[ParsedSessionEvent]:
        """Reduce chunk-local rows of one count-summary event into a single one.

        A count summary describes the complete input.  Keeping chunk rows would
        make event identity, position, and totals depend on the stream
        schedule.  Other event types remain untouched and retain their merge
        order.
        """
        summaries = [event for event in events if event.event_type == event_type]
        if not summaries:
            return events

        def count_map(key: str) -> dict[str, int]:
            totals: dict[str, int] = {}
            for event in summaries:
                values = event.payload.get(key, {})
                if not isinstance(values, dict):
                    continue
                for name, count in values.items():
                    if isinstance(name, str) and isinstance(count, int) and not isinstance(count, bool):
                        totals[name] = totals.get(name, 0) + count
            return dict(sorted(totals.items()))

        payload: dict[str, object] = {}
        for key in payload_keys:
            totals = count_map(key)
            if totals or keep_empty_keys:
                payload[key] = totals
        reduced = ParsedSessionEvent(event_type=event_type, timestamp=timestamp, payload=payload)
        return [event for event in events if event.event_type != event_type] + [reduced]

    def merge_claude_session_summaries(
        events: list[ParsedSessionEvent],
        *,
        timestamp: str | None,
    ) -> list[ParsedSessionEvent]:
        """Reduce both of the Claude Code parser's per-session count summaries."""
        events = merge_claude_count_summary_events(
            events,
            event_type="claude_session_environment",
            payload_keys=("entrypoints", "cli_versions", "permission_modes", "prompt_sources"),
            timestamp=timestamp,
            keep_empty_keys=False,
        )
        return merge_claude_count_summary_events(
            events,
            event_type="claude_parse_coverage",
            payload_keys=("sidecar_seen", "sidecar_persisted", "empty_dropped_by_record_type"),
            timestamp=timestamp,
            keep_empty_keys=True,
        )

    merged: dict[str, ParsedSession] = {}
    for session in sessions:
        existing = merged.get(session.provider_session_id)
        if existing is None:
            merged[session.provider_session_id] = session
            continue

        chunks = []
        offset = 0
        for chunk in (existing, session):
            positions = ChunkPositions(chunk.messages, offset)
            chunks.append(
                chunk.model_copy(
                    update={
                        "messages": [
                            positions.message(message, ordinal) for ordinal, message in enumerate(chunk.messages)
                        ],
                        "attachments": [positions.attachment(attachment) for attachment in chunk.attachments],
                        "session_events": [positions.event(event) for event in chunk.session_events],
                    }
                )
            )
            offset += len(chunk.messages)
        existing, session = chunks
        messages = [*existing.messages, *session.messages]
        active_leaf_message_provider_id = messages[-1].provider_message_id if messages else None
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

        def chronological(values: list[str], *, newest: bool) -> str | None:
            if not values:
                return None
            parseable = [(value, timestamp_millis(value)) for value in values]
            valid = [(value, millis) for value, millis in parseable if millis is not None]
            if valid:
                return (max if newest else min)(valid, key=lambda item: item[1])[0]
            # Preserve the old deterministic behavior only when every producer
            # value is malformed; valid evidence must never be ordered lexically.
            return (max if newest else min)(values)

        # bd polylogue-t5lg: pick the chunk with the stronger title EVIDENCE
        # (_TITLE_EVIDENCE_PRECEDENCE), not merely "whichever chunk resolved a
        # non-raw-id title first" -- that rule froze the first chunk's title,
        # even a weak heuristic guess, once it was non-UUID, discarding a
        # stronger ai-title/custom-title sidecar record that only appeared in
        # a later chunk of the same streamed session. All three title fields
        # move together so title_source/title_ref never point at a different
        # chunk's evidence than the title text they describe.
        title_winner = _later_chunk_title_winner(existing, session)
        # A branch point names a message inside the parent, so it is only
        # carried forward from a chunk that asserts the parent that wins.
        parent_winner = existing if existing.parent_session_provider_id else session
        branch_point_provider_message_id = next(
            (
                chunk.branch_point_provider_message_id
                for chunk in (existing, session)
                if chunk.branch_point_provider_message_id
                and chunk.parent_session_provider_id == parent_winner.parent_session_provider_id
            ),
            None,
        )
        session_events = [*existing.session_events, *session.session_events]
        if existing.source_name is Provider.CLAUDE_CODE:
            session_events = merge_claude_session_summaries(
                session_events, timestamp=chronological(updated_values, newest=True)
            )
        merged[session.provider_session_id] = existing.model_copy(
            update={
                "title": title_winner.title,
                "title_source": title_winner.title_source,
                "title_ref": title_winner.title_ref,
                "created_at": chronological(created_values, newest=False),
                "parent_session_provider_id": parent_winner.parent_session_provider_id,
                "branch_point_provider_message_id": branch_point_provider_message_id,
                "provider_session_aliases": sorted(
                    {*existing.provider_session_aliases, *session.provider_session_aliases}
                ),
                "branch_type": existing.branch_type or session.branch_type,
                "updated_at": chronological(updated_values, newest=True),
                "messages": messages,
                "active_leaf_message_provider_id": active_leaf_message_provider_id,
                "attachments": [*existing.attachments, *session.attachments],
                "session_events": session_events,
                "reported_cost_usd": reported_cost_usd,
                "reported_duration_ms": reported_duration_ms,
                "models_used": sorted({*existing.models_used, *session.models_used}),
                # Claude Code leads with its relocated cwd; a sorted union
                # would put the stale original back in front of it.
                "working_directories": (
                    claude_code_parser.order_working_directories(
                        {*existing.working_directories, *session.working_directories},
                        claude_code_parser.relocated_cwds_of(session_events),
                    )
                    if existing.source_name is Provider.CLAUDE_CODE
                    else sorted({*existing.working_directories, *session.working_directories})
                ),
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
    primary_uuids: set[str] | SqliteStringSet,
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
    sidecar_resolver: SidecarResolver | None = None,
    message_sink_factory: Callable[[], MutableSequence[ParsedMessage]] | None = None,
    event_sink_factory: Callable[[], MutableSequence[ParsedSessionEvent]] | None = None,
    attachment_sink_factory: Callable[[], MutableSequence[ParsedAttachment]] | None = None,
) -> Iterator[ParsedSession]:
    if message_sink_factory is None:
        yield from _claude_code_multiway_parse_inner(
            payloads,
            fallback_id,
            source_path=source_path,
            sidecar_resolver=sidecar_resolver,
            attachment_sink_factory=attachment_sink_factory,
        )
    else:
        with ClaudeStreamScratch() as scratch, ExitStack() as sidecar_stack:
            yield from _claude_code_multiway_parse_inner(
                payloads,
                fallback_id,
                source_path=source_path,
                sidecar_resolver=sidecar_resolver,
                message_sink_factory=message_sink_factory,
                event_sink_factory=event_sink_factory,
                attachment_sink_factory=attachment_sink_factory,
                scratch=scratch,
                sidecar_stack=sidecar_stack,
            )


def _claude_code_multiway_parse_inner(
    payloads: Iterable[object],
    fallback_id: str,
    *,
    source_path: str | None = None,
    sidecar_resolver: SidecarResolver | None = None,
    message_sink_factory: Callable[[], MutableSequence[ParsedMessage]] | None = None,
    event_sink_factory: Callable[[], MutableSequence[ParsedSessionEvent]] | None = None,
    attachment_sink_factory: Callable[[], MutableSequence[ParsedAttachment]] | None = None,
    scratch: ClaudeStreamScratch | None = None,
    sidecar_stack: ExitStack | None = None,
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
    session-wide union index from the scope's sibling transcripts.

    The scope is resolved once, before the walk, through ``sidecar_resolver``
    (polylogue-cq1ql) -- the source tree during acquisition, retained bytes
    during derivation -- so the stream itself never reaches back to a path.
    """
    is_agent_fallback = fallback_id.startswith("agent-")

    resolver = sidecar_resolver if sidecar_resolver is not None else _default_sidecar_resolver()
    sidecar_scope = resolver.claude_code_scope(source_path) if source_path is not None else None
    if sidecar_scope is not None and not sidecar_scope.available:
        sidecar_scope = None

    def new_sidecar_accumulator() -> ToolResultIndexAccumulator:
        from polylogue.sources.live.tool_result_sidecars import ToolResultIndexAccumulator

        accumulator = ToolResultIndexAccumulator(disk_backed=scratch is not None)
        return sidecar_stack.enter_context(accumulator) if sidecar_stack is not None else accumulator

    sidecar_accumulators: dict[str, ToolResultIndexAccumulator] | None = {} if sidecar_scope is not None else None

    accumulators: dict[str, claude_code_parser._SessionAccumulator] = {}
    # One admission observer per session group: an outer record belongs to
    # exactly the session it folds into, so each session's ledger proves its
    # own records rather than the whole multi-session stream.
    observers: dict[str, AdmissionObserver] = {}
    group_order: list[str] = []
    provisional_groups: set[str] = set()
    pending_prefix: list[tuple[object, PayloadRecord | None]] = []
    current_group_id: str | None = None
    primary_started = False
    primary_uuids: set[str] | SqliteStringSet = scratch.string_set("primary") if scratch is not None else set()

    def new_accumulator(group_fallback_id: str, trust: bool) -> claude_code_parser._SessionAccumulator:
        acc = claude_code_parser._SessionAccumulator(
            fallback_id=group_fallback_id,
            trust_fallback_id=trust,
            is_agent=group_fallback_id.startswith("agent-"),
            is_acompact=group_fallback_id.startswith("agent-acompact-"),
        )
        if message_sink_factory is not None:
            acc.messages = message_sink_factory()
        if event_sink_factory is not None:
            acc.session_events = event_sink_factory()
        if attachment_sink_factory is not None:
            acc.attachments = attachment_sink_factory()
        if scratch is not None:
            acc.scratch = scratch
            acc.scratch_scope = group_fallback_id
            acc.seen_uuids = scratch.string_set(f"seen:{group_fallback_id}")
            acc.background_notifications = scratch.notifications(group_fallback_id)
            acc.delegation_progress = scratch.mapped(f"delegation:{group_fallback_id}")
        return acc

    def fold_into(group_id: str, index: int, item: object, record: PayloadRecord | None) -> None:
        # ``record`` is the caller's already-coerced view of ``item``. Coercing
        # walks the whole decoded record, so it happens once per record here,
        # not once per read of a field.
        observer = observers.get(group_id)
        if observer is None:
            observer = observers[group_id] = AdmissionObserver(claude_code_unknown_wire_type, record_stream=True)
        if sidecar_accumulators is not None:
            sidecar_accumulators[group_id].observe(item)
        if record is not None and not is_agent_fallback and group_id == fallback_id:
            uuid = optional_string(record.get("uuid"))
            if uuid is not None:
                primary_uuids.add(uuid)
        # The fold is the record's admission owner: it says whether the record
        # was lowered (a dict without a string ``type`` leaves nothing), and a
        # non-object record is refused.
        lowered = isinstance(item, dict) and claude_code_parser._fold_code_record(accumulators[group_id], index, item)
        observer.observe(item, source_index=index, lowered=lowered)

    record_index = 0
    for item in payloads:
        record_index += 1
        record = _payload_record(item)
        session_id = optional_string(record.get("sessionId")) if record is not None else None

        if session_id is None:
            if current_group_id is None:
                if scratch is None:
                    pending_prefix.append((item, record))
                else:
                    scratch.add_prefix(record_index, item)
                continue
            fold_into(current_group_id, record_index, item, record)
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
            if scratch is None:
                prefix_index = record_index - len(pending_prefix)
                for prefix_item, prefix_record in pending_prefix:
                    prefix_index += 1
                    fold_into(session_id, prefix_index, prefix_item, prefix_record)
            else:
                prefix_index = record_index - scratch.prefix_count()
                for _, prefix_item in scratch.iter_prefix():
                    prefix_index += 1
                    fold_into(session_id, prefix_index, prefix_item, _payload_record(prefix_item))
                scratch.clear_prefix()
            pending_prefix = []

        current_group_id = session_id
        fold_into(session_id, record_index, item, record)

    if not accumulators:
        # No sessionId ever appeared -- either a genuinely empty stream or
        # one that only ever produced prefix records (e.g. a lone
        # ``summary`` row). Both former implementations still emit exactly
        # one (possibly zero-message) session for this fallback_id.
        accumulators[fallback_id] = new_accumulator(fallback_id, False)
        group_order.append(fallback_id)
        if sidecar_accumulators is not None:
            sidecar_accumulators[fallback_id] = new_sidecar_accumulator()
        if scratch is None:
            for index, (prefix_item, prefix_record) in enumerate(pending_prefix, start=1):
                fold_into(fallback_id, index, prefix_item, prefix_record)
        else:
            for index, prefix_item in scratch.iter_prefix():
                fold_into(fallback_id, index, prefix_item, _payload_record(prefix_item))

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
        session = observers.setdefault(group_id, AdmissionObserver(claude_code_unknown_wire_type)).apply(
            session, "claude_code"
        )
        if sidecar_accumulators is not None:
            # sidecar_accumulators is only set when the scope resolved, which
            # itself only happens when source_path is not None.
            assert sidecar_scope is not None
            assert source_path is not None
            join_result = sidecar_accumulators[group_id].join_session_scoped(sidecar_scope, source_path)
            session = apply_tool_result_sidecars(session, join_result)
        yield session


def _bundle_record_specs(
    provider: Provider,
    payloads: PayloadSequence,
    fallback_id: str,
) -> Generator[LoweredPayloadSpec, None, None]:
    for index, item in enumerate(payloads):
        check_compute_cancelled()
        record = _payload_record(item)
        if record is not None:
            yield LoweredPayloadSpec(
                provider=provider, fallback_id=f"{fallback_id}-{index}", mode="bundle_record", payload=record
            )


#: Below this many candidate records, a zero-match bundle is unremarkable --
#: even a genuine format change might only affect a small shard, and
#: warning on tiny/edge-case payloads would be noise the next real drift
#: warning gets lost in.
_CHATGPT_BUNDLE_DRIFT_MIN_CANDIDATES = 5


def _looks_like_chatgpt_mapping_candidate(record: PayloadRecord) -> bool:
    """True if ``record`` looks like a ChatGPT conversation record.

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
    if isinstance(mapping, dict) and bool(mapping):
        return True
    # A renamed or restructured tree key would leave no ``mapping`` at all
    # and so never count, which is exactly the drift this warning exists for
    # (polylogue-axkgy). The conversation envelope around the tree -- a
    # ``current_node`` pointer beside ``create_time`` and ``title`` -- still
    # marks the record as a conversation, and none of the export's sibling
    # arrays carries that combination.
    return all(key in record for key in _CHATGPT_CONVERSATION_ENVELOPE_KEYS)


#: Conversation-level fields that surround the message tree in a ChatGPT
#: export record, independent of what the tree key itself is called.
_CHATGPT_CONVERSATION_ENVELOPE_KEYS = ("current_node", "create_time", "title")


def _chatgpt_bundle_record_specs(
    payloads: PayloadSequence,
    fallback_id: str,
) -> Generator[LoweredPayloadSpec, None, None]:
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
    matched = 0
    candidates = 0
    rejected_candidates = 0
    for index, item in enumerate(payloads):
        record = _payload_record(item)
        if record is None:
            continue
        # codex.json (bd polylogue-2m2e): Codex Cloud tasks delivered inside
        # the ChatGPT export, a completely different shape from a conversation
        # fragment (no "mapping" key). Checked first so these never fall
        # through to the mapping-candidate/near-miss accounting below.
        if chatgpt_codex_sidecar.looks_like(record):
            matched += 1
            yield LoweredPayloadSpec(
                provider=Provider.CHATGPT,
                fallback_id=f"{fallback_id}-{index}",
                mode="chatgpt_codex_task",
                payload=record,
            )
            continue
        if _looks_like_chatgpt_mapping_candidate(record):
            candidates += 1
        if not chatgpt.looks_like_fragment(record):
            if _looks_like_chatgpt_mapping_candidate(record):
                rejected_candidates += 1
            continue
        matched += 1
        yield LoweredPayloadSpec(
            provider=Provider.CHATGPT,
            fallback_id=f"{fallback_id}-{index}",
            mode="bundle_record",
            payload=record,
        )
    if rejected_candidates and (matched or candidates >= _CHATGPT_BUNDLE_DRIFT_MIN_CANDIDATES):
        logger.warning(
            "ChatGPT bundle payload %r: rejected %d of %d candidate records after "
            "matching %d conversation fragments; rejected siblings received typed "
            "drift accounting and must not be treated as successfully parsed",
            fallback_id,
            rejected_candidates,
            candidates,
            matched,
        )


def _lower_bundle_payload(
    provider: Provider,
    shaped_payload: object,
    fallback_id: str,
) -> Iterable[LoweredPayloadSpec]:
    payloads = _payload_sequence(shaped_payload)
    if payloads is not None:
        if provider is Provider.CHATGPT:
            return _chatgpt_bundle_record_specs(payloads, fallback_id)
        return _bundle_record_specs(provider, payloads, fallback_id)
    record = _payload_record(shaped_payload)
    if record is None:
        return []
    if provider is Provider.CHATGPT and isinstance(record.get("conversations"), list):
        return _chatgpt_bundle_record_specs(cast(PayloadSequence, record["conversations"]), fallback_id)
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


def is_drive_chunk_sequence(records: Iterable[object]) -> bool:
    """Select bare chunks with the ordinary lowering and Browser precedence."""
    has_chunk = False
    has_container = False
    all_browser_captures = True
    for record in records:
        has_chunk = has_chunk or drive.looks_like_chunk(record)
        has_container = has_container or _looks_like_chunked_session(record)
        all_browser_captures = all_browser_captures and _looks_like_browser_capture_record(record)
    return has_chunk and not has_container and not all_browser_captures


def _lower_drive_like_payload(
    provider: Provider,
    shaped_payload: object,
    fallback_id: str,
    *,
    source_path: str | None,
    schema_resolution: SchemaResolution | None,
) -> Iterable[LoweredPayloadSpec | _PayloadLoweringRequest]:
    payloads = _payload_sequence(shaped_payload)
    if payloads is not None:
        if is_drive_chunk_sequence(payloads):
            return [_chunked_prompt_spec(provider, payloads, fallback_id)]
        # Both wrapper branches preserve the original singleton identity and
        # ordinal fallback. Their children are drained rather than retained.
        return (
            _PayloadLoweringRequest(
                provider,
                item,
                fallback_id if len(payloads) == 1 else f"{fallback_id}-{index}",
                source_path=source_path,
                schema_resolution=schema_resolution,
            )
            for index, item in enumerate(payloads)
        )

    record = _payload_record(shaped_payload)
    if record is None:
        return []
    # A bare document is validated by the same registry as a wrapped one: a
    # Drive location admits only its own origin, never a Gemini CLI document.
    _validate_sequence_document_origins((record,), provider)
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


def _lower_grok_export_payload(payload: object, fallback_id: str) -> Iterator[LoweredPayloadSpec]:
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
        return
    if grok.looks_like_native_bundle(record):
        yield _single_record_spec(Provider.GROK, record, fallback_id)
        return
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return
    for index, item in enumerate(conversations):
        item_record = _payload_record(item)
        # A malformed entry (missing "conversation"/"responses", wrong types)
        # is skipped rather than admitted as a zero-message phantom session --
        # matching how _bundle_record_specs silently drops non-record bundle
        # entries for ChatGPT/Claude AI rather than emitting an empty session
        # per malformed item.
        if item_record is None or not grok.looks_like_conversation(item_record):
            continue
        yield _single_record_spec(
            Provider.GROK,
            item_record,
            fallback_id if len(conversations) == 1 else f"{fallback_id}-{index}",
        )


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
    schema_resolution: SchemaResolution | None = None,
    source_path: str | None = None,
) -> list[LoweredPayloadSpec]:
    """Collect the canonical lowering for explicit in-process callers."""
    return list(
        iter_lowered_payload_specs(
            provider,
            payload,
            fallback_id,
            schema_resolution=schema_resolution,
            source_path=source_path,
        )
    )


def iter_lowered_payload_specs(
    provider: str | Provider,
    payload: object,
    fallback_id: str,
    *,
    schema_resolution: SchemaResolution | None = None,
    source_path: str | None = None,
) -> Generator[LoweredPayloadSpec, None, None]:
    """Lower in canonical depth-first order retaining only active ancestry."""
    stack: list[tuple[Iterator[LoweredPayloadSpec | _PayloadLoweringRequest], int | None, object]] = [
        (iter((_PayloadLoweringRequest(provider, payload, fallback_id, schema_resolution, source_path),)), None, None)
    ]
    active: set[int] = set()
    try:
        while stack:
            children, identity, _original = stack[-1]
            try:
                item = next(children)
            except StopIteration:
                stack.pop()
                if identity is not None:
                    active.remove(identity)
                continue
            if isinstance(item, LoweredPayloadSpec):
                yield item
                continue
            identity = id(item.payload)
            if identity in active:
                raise ValueError("cyclic payload cannot be lowered as source JSON")
            active.add(identity)
            stack.append(
                (
                    iter(
                        _lower_payload_specs_step(
                            item.provider,
                            item.payload,
                            item.fallback_id,
                            schema_resolution=item.schema_resolution,
                            source_path=item.source_path,
                        )
                    ),
                    identity,
                    item.payload,
                )
            )
    finally:
        primary = sys.exception()
        failures: list[BaseException] = []
        for children, _identity, _original in reversed(stack):
            close = getattr(children, "close", None)
            if close is not None:
                try:
                    close()
                except BaseException as failure:
                    failures.append(failure)
        if failures:
            if primary is not None:
                failures.insert(0, primary)
            raise BaseExceptionGroup("payload lowering and original iterator close failed", failures) from None


def _lower_payload_specs_step(
    provider: str | Provider,
    payload: object,
    fallback_id: str,
    *,
    schema_resolution: SchemaResolution | None = None,
    source_path: str | None = None,
) -> Iterable[LoweredPayloadSpec | _PayloadLoweringRequest]:
    runtime_provider = Provider.from_string(provider)
    if runtime_provider is Provider.BEADS:
        return []

    shaped_payload = _schema_guided_payload(runtime_provider, payload, schema_resolution)
    record = _payload_record(shaped_payload)
    if runtime_provider is Provider.CHATGPT and record is not None and chatgpt.looks_like_shared_decode(record):
        return [
            LoweredPayloadSpec(
                provider=Provider.CHATGPT,
                fallback_id=fallback_id,
                mode="bundle_record",
                payload=record,
            )
        ]
    if record is not None and browser_capture.looks_like(record):
        provider = _declared_capture_provider(record) or runtime_provider
        if provider is Provider.BEADS:
            return []
        return [
            LoweredPayloadSpec(
                provider=provider,
                fallback_id=fallback_id,
                mode="browser_capture",
                payload=record,
            )
        ]
    sequence = _payload_sequence(shaped_payload)
    if sequence is not None:
        _validate_sequence_document_origins(sequence, runtime_provider)
    if runtime_provider is Provider.CHATGPT and sequence is not None and len(sequence) == 1:
        shared_record = _payload_record(sequence[0])
        if shared_record is not None and chatgpt.looks_like_shared_decode(shared_record):
            return [
                _PayloadLoweringRequest(
                    runtime_provider,
                    shared_record,
                    fallback_id,
                    schema_resolution=schema_resolution,
                    source_path=source_path,
                )
            ]
    if sequence and all(
        (item_record := _payload_record(item)) is not None and browser_capture.looks_like(item_record)
        for item in sequence
    ):

        def browser_specs() -> Iterator[LoweredPayloadSpec]:
            for index, item in enumerate(sequence):
                item_record = _payload_record(item)
                assert item_record is not None
                selected_provider = _declared_capture_provider(item_record) or runtime_provider
                yield LoweredPayloadSpec(
                    provider=selected_provider,
                    fallback_id=fallback_id if len(sequence) == 1 else f"{fallback_id}-{index}",
                    mode="browser_capture",
                    payload=item_record,
                )

        return browser_specs()
    if record is not None and (sessions := _record_sessions(record)):
        return (
            _PayloadLoweringRequest(
                runtime_provider,
                item,
                f"{fallback_id}-{index}",
                schema_resolution=schema_resolution,
            )
            for index, item in enumerate(sessions)
        )

    if runtime_provider in BUNDLE_PROVIDERS:
        return _lower_bundle_payload(runtime_provider, shaped_payload, fallback_id)
    if runtime_provider in {Provider.CLAUDE_CODE, Provider.CODEX}:
        return _lower_grouped_payload(runtime_provider, shaped_payload, fallback_id, source_path=source_path)
    if runtime_provider is Provider.GEMINI_CLI:
        if sequence is not None and any(
            (item_record := _payload_record(item)) is not None
            and local_agent.looks_like_gemini_cli(item_record)
            and isinstance(item_record.get("messages"), list)
            for item in sequence
        ):
            return (
                _local_agent_document_spec(
                    runtime_provider,
                    item_record,
                    fallback_id if len(sequence) == 1 else f"{fallback_id}-{index}",
                    source_path=source_path,
                )
                for index, item in enumerate(sequence)
                if (item_record := _payload_record(item)) is not None
                and local_agent.looks_like_gemini_cli(item_record)
                and isinstance(item_record.get("messages"), list)
            )
        record = _single_document_record(shaped_payload)
        if record is not None and local_agent.looks_like_gemini_cli(record):
            return [_local_agent_document_spec(runtime_provider, record, fallback_id, source_path=source_path)]
        # polylogue-8u1p: Gemini CLI's second on-disk shape is a ``.jsonl``
        # checkpoint *log* -- a session-open stub line followed by one record
        # per turn/event and ``{"$set": ...}`` envelope patches. It is not a
        # single document, so the branch above yielded no specs at all and the
        # file never became a queryable session. Folding the log back into the
        # document it is a log of keeps one parser and one identity rule for
        # both shapes; the single-document path above is untouched.
        stream = _payload_sequence(shaped_payload)
        if stream is not None:
            folded = local_agent.fold_gemini_cli_checkpoint_stream(stream)
            if folded is not None:
                return [_local_agent_document_spec(runtime_provider, folded, fallback_id, source_path=source_path)]
        return []
    if runtime_provider in DRIVE_LIKE_PROVIDERS:
        return _lower_drive_like_payload(
            runtime_provider,
            shaped_payload,
            fallback_id,
            source_path=source_path,
            schema_resolution=schema_resolution,
        )
    if runtime_provider is Provider.HERMES:
        payloads = _payload_sequence(shaped_payload)
        if (
            payloads is not None
            and payloads
            and any(
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
            return [_local_agent_document_spec(runtime_provider, record, fallback_id, source_path=source_path)]
        return []
    if runtime_provider is Provider.ANTIGRAVITY:
        record = _single_document_record(shaped_payload)
        if record is not None and antigravity.looks_like_markdown_export(record):
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
        if sequence is not None:
            return (
                spec
                for index, item in enumerate(sequence)
                if (item_record := _payload_record(item)) is not None
                and (grok.looks_like_native_bundle(item_record) or grok.looks_like_export(item_record))
                for spec in _lower_grok_export_payload(
                    item_record, fallback_id if len(sequence) == 1 else f"{fallback_id}-{index}"
                )
            )
        return _lower_grok_export_payload(shaped_payload, fallback_id)
    if runtime_provider is Provider.OTEL_GENAI:
        record = _single_document_record(shaped_payload)
        if record is None or not otel_genai.looks_like(record):
            return []
        return [
            LoweredPayloadSpec(
                provider=Provider.OTEL_GENAI,
                fallback_id=fallback_id,
                mode="single_record",
                payload=record,
                source_path=source_path,
            )
        ]
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
    asserted_id = optional_string(payload.get("id"))
    if asserted_id is None or not asserted_id.strip():
        return None

    messages = extract_messages_from_list(messages_payload)
    return _generic_messages_session_from_messages(provider, payload, fallback_id, messages)


def _generic_messages_session_from_messages(
    provider: Provider,
    payload: PayloadRecord,
    fallback_id: str,
    messages: MutableSequence[ParsedMessage],
) -> ParsedSession | None:
    # A blank id is not an assertion. ``optional_string`` returns ``""`` for an
    # empty value rather than ``None``, so an ``"id": ""`` or whitespace-only
    # field would otherwise satisfy "the provider asserted an identity" and
    # produce a session keyed on nothing -- the same pathology as a
    # filename-stem identity, arriving through the guard meant to stop it.
    asserted_id = optional_string(payload.get("id"))
    session_id = asserted_id.strip() if asserted_id is not None else None
    if not session_id:
        return None

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
    session = ParsedSession(
        source_name=provider,
        provider_session_id=session_id,
        title=title,
        created_at=created_at,
        updated_at=updated_at,
        messages=messages if isinstance(messages, list) else [],
    )
    return session if isinstance(messages, list) else session.model_copy(update={"messages": messages})


def parse_generic_messages_stream(
    provider: Provider,
    envelope: PayloadRecord,
    records: Iterable[object],
    fallback_id: str,
    *,
    message_sink: MutableSequence[ParsedMessage],
) -> ParsedSession | None:
    """Use the generic object parser's field and message rules with disk-backed rows."""
    asserted_id = envelope.get("id")
    if not isinstance(asserted_id, str) or not asserted_id.strip():
        return None
    message_sink.extend(
        upgrade_chat_export_user_authorship(provider, message) for message in iter_messages_from_list(records)
    )
    return _generic_messages_session_from_messages(provider, envelope, fallback_id, message_sink)


def _parse_lowered_spec(
    spec: LoweredPayloadSpec, resolver: SidecarResolver, *, profile_identity: str | None = None
) -> list[ParsedSession]:
    """Parse, account for, and admit one lowered spec for publication.

    Every production route passes here, including the ones that reach an
    undecorated entry point (Hermes state/ATIF/verification, Antigravity
    markdown, Codex streams); their single-session results get the same
    outer-record ledger the decorated leaf parsers attach. This applies the
    shared positive-evidence rule once after accounting, so undecorated leaf
    parsers do not need a second publication check.
    """
    sessions = _parse_lowered_spec_unadmitted(spec, resolver, profile_identity=profile_identity)
    admitted = admit_parsed_sessions(
        spec.provider.value.replace("-", "_"),
        spec.payload,
        sessions,
        # The one grouped record sequence whose sessions carry no parser
        # ledger: the ATOF stream parser's own recognizer settles each record.
        recognizes=hermes_spans.looks_like_atof_payload
        if spec.provider is Provider.HERMES and spec.mode == "grouped_records"
        else None,
    )
    return admit_parsed_sessions_for_publication(
        admitted,
        provider=spec.provider,
        source_path=spec.source_path,
    )


def _parse_lowered_spec_unadmitted(
    spec: LoweredPayloadSpec, resolver: SidecarResolver, *, profile_identity: str | None = None
) -> list[ParsedSession]:
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
        return list(
            _claude_code_multiway_parse(
                payloads,
                spec.fallback_id,
                source_path=spec.source_path,
                sidecar_resolver=resolver,
            )
        )

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

    if spec.provider is Provider.OTEL_GENAI:
        record = _payload_record(spec.payload)
        return otel_genai.parse(record, spec.fallback_id) if record is not None else []

    if spec.provider is Provider.CLAUDE_CODE:
        payloads = _payload_sequence(spec.payload)
        if payloads is None:
            return []
        return [
            claude.parse_code(
                payloads,
                spec.fallback_id,
                tool_result_sidecars=_join_claude_code_sidecars(payloads, spec.source_path, resolver),
                trust_fallback_id=spec.trust_fallback_id,
            )
        ]

    if spec.provider is Provider.CODEX:
        payloads = _payload_sequence(spec.payload)
        return [codex.parse(payloads, spec.fallback_id)] if payloads is not None else []

    if spec.provider is Provider.HERMES and spec.mode == "grouped_records":
        payloads = _payload_sequence(spec.payload)
        return (
            hermes_spans.parse_atof_stream(
                payloads,
                spec.fallback_id,
                profile_identity=profile_identity,
                profile_root=(profile_root_for_artifact(Path(spec.source_path)) if spec.source_path else None),
            )
            if payloads is not None
            else []
        )

    if spec.mode == "local_agent_document":
        record = _payload_record(spec.payload)
        if record is None:
            return []
        if spec.provider is Provider.GEMINI_CLI:
            return [
                local_agent.parse_gemini_cli(
                    record,
                    spec.fallback_id,
                    source_path=spec.source_path,
                    sidecar_resolver=resolver,
                )
            ]
        if spec.provider is Provider.HERMES:
            return [
                local_agent.parse_hermes(
                    record, spec.fallback_id, source_path=spec.source_path, profile_identity=profile_identity
                )
            ]
        return []

    if spec.mode == "local_artifact_document":
        record = _payload_record(spec.payload)
        if record is None:
            return []
        if spec.provider is Provider.HERMES and hermes_state.looks_like_state_db_payload(record):
            return hermes_state.parse_state_db_payload(
                record, spec.fallback_id, source_path=spec.source_path, profile_identity=profile_identity
            )
        if spec.provider is Provider.HERMES and hermes_verification.looks_like_verification_evidence_db_payload(record):
            return hermes_verification.parse_verification_evidence_db_payload(
                record,
                spec.fallback_id,
                profile_identity=profile_identity,
                profile_root=(profile_root_for_artifact(Path(spec.source_path)) if spec.source_path else None),
                source_path=spec.source_path,
            )
        if spec.provider is Provider.HERMES and hermes_spans.looks_like_atif_payload(record):
            return hermes_spans.parse_atif_document(
                record,
                spec.fallback_id,
                profile_identity=profile_identity,
                profile_root=(profile_root_for_artifact(Path(spec.source_path)) if spec.source_path else None),
            )
        if spec.provider is Provider.ANTIGRAVITY and antigravity.looks_like_markdown_export(record):
            return [antigravity.parse_markdown_export_payload(record, spec.fallback_id)]
        return []

    if spec.mode == "chunked_prompt":
        record = _payload_record(spec.payload)
        if record is not None:
            payload = record
        else:
            chunks = _payload_sequence(spec.payload)
            return [
                drive._parse_chunked_records(
                    spec.provider, {}, lambda: chunks or (), spec.fallback_id, record_stream=True
                )
            ]
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


def admit_parsed_sessions_for_publication(
    sessions: list[ParsedSession],
    *,
    provider: str | Provider,
    source_path: str | None,
) -> list[ParsedSession]:
    """polylogue-9ykn: a session requires authored content or, for OTel GenAI,
    retained span evidence. Other empty sessions are refused before writing.

    This is the one positive-evidence admission call for production parsed
    sessions. Dispatch applies it after provider accounting; retained
    preparation and replay apply it before preparing or publishing their
    sessions. Low-level parser functions remain useful for parser-only laws.

    Checking message *content*, not only message *count*, also refuses an
    unrecognized single-record document that Claude Code's generic lowering
    turns into one message with empty text and no blocks.
    """
    kept: list[ParsedSession] = []
    for session in sessions:
        if any(message_carries_authored_content(message) for message in session.messages):
            kept.append(session)
            continue
        if session.source_name is Provider.OTEL_GENAI and any(
            event.event_type == "otel_span_evidence" for event in session.session_events
        ):
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
    *,
    schema_resolution: SchemaResolution | None = None,
    source_path: str | None = None,
    profile_identity: str | None = None,
    sidecar_resolver: SidecarResolver | None = None,
) -> list[ParsedSession]:
    """Dispatch parsed payload to the appropriate provider parser.

    Dispatches, records provider accounting and applies the shared positive-
    evidence admission rule before returning sessions for publication.

    ``sidecar_resolver`` decides where an overflowed tool output is read from
    (polylogue-cq1ql). It defaults to acquisition-time filesystem resolution;
    a route that derives from retained bytes rather than from the source tree
    must pass a ``RetainedSidecarResolver``, or a transcript whose original
    tree is gone reparses with only its truncated previews.
    """
    return list(
        iter_parsed_payload(
            provider,
            payload,
            fallback_id,
            schema_resolution=schema_resolution,
            source_path=source_path,
            profile_identity=profile_identity,
            sidecar_resolver=sidecar_resolver,
        )
    )


def iter_parsed_payload(
    provider: str | Provider,
    payload: object,
    fallback_id: str,
    *,
    schema_resolution: SchemaResolution | None = None,
    source_path: str | None = None,
    profile_identity: str | None = None,
    sidecar_resolver: SidecarResolver | None = None,
    message_sink_factory: Callable[[], MutableSequence[ParsedMessage]] | None = None,
    event_sink_factory: Callable[[], MutableSequence[ParsedSessionEvent]] | None = None,
    attachment_sink_factory: Callable[[], MutableSequence[ParsedAttachment]] | None = None,
) -> Generator[ParsedSession, None, None]:
    """Drain the canonical lowering without retaining its complete output cohort."""
    resolver = sidecar_resolver if sidecar_resolver is not None else _default_sidecar_resolver()
    specs = iter_lowered_payload_specs(
        provider,
        payload,
        fallback_id,
        schema_resolution=schema_resolution,
        source_path=source_path,
    )
    try:
        for spec in specs:
            if (
                message_sink_factory is not None
                or event_sink_factory is not None
                or attachment_sink_factory is not None
            ) and (
                spec.mode == "claude_code_multiway"
                or spec.mode == "grouped_records"
                and spec.provider in STREAM_RECORD_PROVIDERS
            ):
                payloads = _payload_sequence(spec.payload)
                if payloads is not None:
                    yield from iter_parsed_stream(
                        spec.provider,
                        payloads,
                        spec.fallback_id,
                        source_path=spec.source_path,
                        profile_identity=profile_identity,
                        sidecar_resolver=resolver,
                        message_sink_factory=message_sink_factory,
                        event_sink_factory=event_sink_factory,
                        attachment_sink_factory=attachment_sink_factory,
                    )
            else:
                yield from _parse_lowered_spec(spec, resolver, profile_identity=profile_identity)
    finally:
        primary = sys.exception()
        try:
            specs.close()
        except BaseException as cleanup:
            if primary is not None:
                raise BaseExceptionGroup(
                    "payload parser and original lowering close failed", [primary, cleanup]
                ) from None
            raise


class BundleCandidateDrift:
    """ChatGPT bundle-candidate accounting across one container's members.

    A member that looks like a conversation but fails the fragment shape is
    a refused candidate; ``emit`` reports them once the container is done,
    under the same threshold as the collecting bundle lowering.
    """

    def __init__(self) -> None:
        self.candidates = 0
        self.rejected = 0
        self.matched = 0

    def observe_streamed_conversation(self) -> None:
        """Count a member proved a conversation fragment and parsed from scratch."""
        self.candidates += 1
        self.matched += 1

    def emit(self, provider: Provider, fallback_id: str) -> None:
        if (
            provider is Provider.CHATGPT
            and self.rejected
            and (self.matched or self.candidates >= _CHATGPT_BUNDLE_DRIFT_MIN_CANDIDATES)
        ):
            emit(
                "sources.chatgpt_bundle_candidate_rejected",
                level=WARNING,
                source_id=fallback_id,
                provider=Provider.CHATGPT.value,
                refused=self.rejected,
                rows=self.candidates,
                succeeded=self.matched,
            )


def bundle_member_sessions(
    provider: Provider,
    record: JSONValue,
    fallback_id: str,
    index: int,
    *,
    count: int,
    all_browser_captures: bool,
    drift: BundleCandidateDrift,
    source_path: str | None = None,
    profile_identity: str | None = None,
    sidecar_resolver: SidecarResolver | None = None,
) -> list[ParsedSession]:
    """Parse one decoded bundle member through the ordinary lowering rules."""
    resolver = sidecar_resolver if sidecar_resolver is not None else _default_sidecar_resolver()
    if count == 1:
        # Singleton arrays have special shared-page and browser lowering.
        return parse_payload(
            provider,
            [record],
            fallback_id,
            source_path=source_path,
            profile_identity=profile_identity,
            sidecar_resolver=resolver,
        )
    if all_browser_captures:
        return parse_payload(
            provider,
            record,
            f"{fallback_id}-{index}",
            source_path=source_path,
            profile_identity=profile_identity,
            sidecar_resolver=resolver,
        )
    # Reuse the same bundle normalization, including ChatGPT fragment
    # rejection and Codex-task detection, before correcting the local
    # one-item suffix to the original array position.
    specs = _lower_bundle_payload(provider, [record], fallback_id)
    if provider is Provider.CHATGPT:
        shaped = _payload_record(record)
        if shaped is not None and _looks_like_chatgpt_mapping_candidate(shaped):
            drift.candidates += 1
            if not chatgpt.looks_like_fragment(shaped):
                drift.rejected += 1
    sessions: list[ParsedSession] = []
    for spec in specs:
        if provider is Provider.CHATGPT:
            drift.matched += 1
        sessions.extend(
            _parse_lowered_spec(
                replace(spec, fallback_id=f"{fallback_id}-{index}"), resolver, profile_identity=profile_identity
            )
        )
    return sessions


def _chatgpt_parsed_session_id(record: PayloadRecord, fallback_id: str) -> str:
    """The ``provider_session_id`` ``chatgpt.parse`` gives this record.

    The census keys a document by the session the parser materializes, so an
    id-less record takes the same supplied fallback id the parser does.
    """
    return str(record.get("id") or record.get("uuid") or record.get("conversation_id") or fallback_id)


def _lower_shared_chatgpt_document(record: PayloadRecord, fallback_id: str) -> ChatGPTLoweredDocument | None:
    if not chatgpt.looks_like_shared_decode(record):
        return None
    return ChatGPTLoweredDocument(
        _chatgpt_parsed_session_id(record, fallback_id),
        cast(PayloadRecord, chatgpt.shared_decode_mapping(record)),
        "shared_page_decode",
    )


def _validation_error_locations(exc: Exception) -> str | None:
    """Where an envelope failed validation, never what it contained.

    A validation error's rendering quotes the offending input, which here is
    the operator's captured conversation; only field locations and error
    kinds may reach a log.
    """
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return None
    try:
        details = errors(include_input=False, include_url=False, include_context=False)
    except Exception:
        return None
    return "; ".join(
        f"{'.'.join(str(part) for part in detail.get('loc', ()))}: {detail.get('type', 'invalid')}"
        for detail in details[:8]
    )[:512]


def lower_chatgpt_documents(payload: object, fallback_id: str) -> list[ChatGPTLoweredDocument]:
    """Lower direct, bundled, and browser-capture ChatGPT payloads.

    The returned mapping is the source document used by the independent
    conservation census. Browser-capture fallback turns are projected into
    the same provider-message shape used by their parser, while native
    capture payloads retain the provider mapping verbatim.
    """
    documents: list[ChatGPTLoweredDocument] = []
    shared_record = _payload_record(payload)
    if shared_record is not None:
        shared_document = _lower_shared_chatgpt_document(shared_record, fallback_id)
        if shared_document is not None:
            return [shared_document]
    for spec in _lower_payload_specs(Provider.CHATGPT, payload, fallback_id):
        if spec.provider is not Provider.CHATGPT and spec.mode != "browser_capture":
            continue
        artifact_class = (
            "browser_capture_envelope"
            if spec.mode == "browser_capture"
            else ("bundle" if spec.mode == "bundle_record" else "direct_export")
        )
        record = _payload_record(spec.payload)
        if record is None:
            continue
        shared_document = _lower_shared_chatgpt_document(record, spec.fallback_id)
        if shared_document is not None:
            documents.append(shared_document)
            continue
        if spec.mode == "browser_capture":
            try:
                envelope = BrowserCaptureEnvelope.model_validate(record)
            except Exception as exc:
                # The parser refuses this envelope too, so it contributes no
                # document; but the census must not undercount silently.
                emit(
                    "sources.census.browser_capture_envelope_invalid",
                    level=WARNING,
                    outcome="degraded",
                    reason="envelope_invalid",
                    error_type=type(exc).__name__,
                    error_detail=_validation_error_locations(exc),
                )
                continue
            native = envelope.raw_provider_payload
            # The census must lower exactly what the parser materializes. A
            # `chatgpt-native-compact-v1` bridge projection carries a
            # `mapping` the extension synthesized, and
            # `browser_capture.parse` deliberately ignores it in favour of
            # `session.turns`; censusing that untrusted mapping reported its
            # never-materialized message ids as conservation drops, and an
            # empty compact mapping reported a session with zero content
            # units. Ask the parser's own predicate instead of re-deriving it.
            if envelope.session.provider is Provider.CHATGPT and has_chatgpt_native_payload(native):
                conversation_id = optional_string(native.get("id")) or optional_string(native.get("uuid"))
                conversation_id = (
                    conversation_id
                    or optional_string(native.get("conversation_id"))
                    or envelope.session.provider_session_id
                )
                documents.append(
                    ChatGPTLoweredDocument(str(conversation_id), cast(PayloadRecord, native["mapping"]), artifact_class)
                )
                continue
            mapping: dict[str, object] = {}
            for turn in envelope.session.turns:
                mapping[turn.provider_turn_id] = {
                    "id": turn.provider_turn_id,
                    "message": {
                        "id": turn.provider_turn_id,
                        "author": {"role": turn.role.value},
                        "content": {"content_type": "text", "parts": [turn.text]}
                        if turn.text
                        else {
                            "content_type": "blocks",
                            "blocks": [block.model_dump(mode="json") for block in turn.blocks],
                        },
                    },
                }
            documents.append(
                ChatGPTLoweredDocument(
                    envelope.session.provider_session_id, cast(PayloadRecord, mapping), artifact_class
                )
            )
            continue
        if isinstance(record.get("mapping"), dict):
            documents.append(
                ChatGPTLoweredDocument(
                    _chatgpt_parsed_session_id(record, spec.fallback_id),
                    cast(PayloadRecord, record["mapping"]),
                    artifact_class,
                )
            )
    return documents


def chatgpt_rejected_mapping_candidates(payload: object) -> list[str | None]:
    """Conversation-shaped bundle records the ChatGPT bundle lowering rejects.

    :func:`lower_chatgpt_documents` returns only admitted conversations, so a
    census built from it alone cannot see a rejected sibling of a valid one.
    This applies the bundle lowering's own near-miss test and returns each
    rejected record's conversation id, or ``None`` when it names none.
    """
    records = _payload_sequence(payload)
    if records is None:
        record = _payload_record(payload)
        conversations = record.get("conversations") if record is not None else None
        if not isinstance(conversations, list):
            return []
        records = conversations
    rejected: list[str | None] = []
    for item in records:
        record = _payload_record(item)
        if record is None or chatgpt_codex_sidecar.looks_like(record):
            continue
        if _looks_like_chatgpt_mapping_candidate(record) and not chatgpt.looks_like_fragment(record):
            rejected.append(
                next(
                    (
                        str(record[key])
                        for key in ("id", "uuid", "conversation_id")
                        if isinstance(record.get(key), str) and record[key]
                    ),
                    None,
                )
            )
    return rejected


def parse_stream_payload(
    provider: str | Provider,
    payloads: Iterable[object],
    fallback_id: str,
    *,
    source_path: str | None = None,
    profile_identity: str | None = None,
    sidecar_resolver: SidecarResolver | None = None,
    message_sink_factory: Callable[[], MutableSequence[ParsedMessage]] | None = None,
    event_sink_factory: Callable[[], MutableSequence[ParsedSessionEvent]] | None = None,
    attachment_sink_factory: Callable[[], MutableSequence[ParsedAttachment]] | None = None,
) -> list[ParsedSession]:
    """Parse a grouped record stream.

    The shared positive-evidence admission rule runs once per lowered
    provider result; see ``parse_payload`` for ``sidecar_resolver``.
    """
    return list(
        iter_parsed_stream(
            provider,
            payloads,
            fallback_id,
            source_path=source_path,
            profile_identity=profile_identity,
            sidecar_resolver=sidecar_resolver,
            message_sink_factory=message_sink_factory,
            event_sink_factory=event_sink_factory,
            attachment_sink_factory=attachment_sink_factory,
        )
    )


def iter_parsed_stream(
    provider: str | Provider,
    payloads: Iterable[object],
    fallback_id: str,
    *,
    source_path: str | None = None,
    profile_identity: str | None = None,
    sidecar_resolver: SidecarResolver | None = None,
    message_sink_factory: Callable[[], MutableSequence[ParsedMessage]] | None = None,
    event_sink_factory: Callable[[], MutableSequence[ParsedSessionEvent]] | None = None,
    attachment_sink_factory: Callable[[], MutableSequence[ParsedAttachment]] | None = None,
) -> Generator[ParsedSession, None, None]:
    """Parse a grouped record stream.

    The shared positive-evidence admission rule runs once per lowered
    provider result; see ``parse_payload`` for ``sidecar_resolver``.
    """
    runtime_provider = Provider.from_string(provider)
    if runtime_provider is Provider.CLAUDE_CODE:
        yield from _claude_code_multiway_parse(
            payloads,
            fallback_id,
            source_path=source_path,
            sidecar_resolver=sidecar_resolver,
            message_sink_factory=message_sink_factory,
            event_sink_factory=event_sink_factory,
            attachment_sink_factory=attachment_sink_factory,
        )
        return
    if runtime_provider is Provider.CODEX:
        observer = AdmissionObserver(codex_unknown_wire_type)
        stream = observer.observing(payloads)
        session = codex.parse_stream(
            stream,
            fallback_id,
            message_sink=message_sink_factory() if message_sink_factory is not None else None,
            event_sink=event_sink_factory() if event_sink_factory is not None else None,
        )
        observer.drain(stream)
        yield observer.apply(session, "codex")
        return
    if runtime_provider is Provider.HERMES:
        observer = AdmissionObserver(hermes_unknown_wire_type, record_stream=True)
        parsed = False

        def admitted(records: Iterable[object]) -> Iterator[object]:
            # The parser's own recognition decides: a known-kind record it
            # skips (no uuid, say) is refused, not counted as materialized.
            # A record the parser never pulled has no disposition at all.
            for item in records:
                if parsed:
                    observer.observe(item)
                    continue
                record = _payload_record(item)
                recognized = record is not None and hermes_spans.looks_like_atof_payload(record)
                observer.observe(
                    item,
                    lowered=recognized,
                    malformed=record is not None and not recognized and hermes_unknown_wire_type(record) is None,
                )
                yield item

        stream = admitted(payloads)
        with closing(
            hermes_spans.iter_atof_sessions(
                stream,
                fallback_id,
                profile_root=profile_root_for_artifact(Path(source_path)) if source_path else None,
                profile_identity=profile_identity,
                new_events=event_sink_factory or list,
            )
        ) as sessions:
            for session in sessions:
                parsed = True
                AdmissionObserver.drain(stream)
                yield observer.apply(session, "hermes")
        parsed = True
        AdmissionObserver.drain(stream)
        return
    raise ValueError(f"provider {runtime_provider} does not support stream parsing")


__all__ = [
    "iter_parsed_payload",
    "iter_parsed_stream",
    "GROUP_PROVIDERS",
    "chatgpt_rejected_mapping_candidates",
    "STREAM_RECORD_PROVIDERS",
    "LoweredPayloadSpec",
    "ChatGPTLoweredDocument",
    "detect_provider",
    "ForeignOriginContentError",
    "bound_location_provider",
    "same_origin",
    "detect_provider_evidence",
    "detect_provider_from_raw_bytes_evidence",
    "detect_provider_from_raw_stream_evidence",
    "is_jsonl_source_path",
    "is_stream_record_provider",
    "parse_payload",
    "BundleCandidateDrift",
    "bundle_member_sessions",
    "lower_chatgpt_documents",
    "parse_stream_payload",
]
