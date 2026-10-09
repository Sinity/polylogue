"""Sealed, source-bound preparation for JSON and JSONL session captures."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import sys
import uuid
from builtins import BaseExceptionGroup
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, closing, contextmanager, suppress
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import partial
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal, cast, overload

import ijson

from polylogue.archive.artifact_taxonomy import (
    ArtifactClassification,
    ArtifactKind,
    ArtifactStreamClassification,
    classify_artifact_stream,
    fact_path_admits_session_content,
    strong_path_classification,
)
from polylogue.core.compute import DaemonBackpressureError, DaemonOperationCancelled
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import BlockType, Provider
from polylogue.core.identity_law import session_id as archive_session_id
from polylogue.core.json import JSONValue, is_json_value
from polylogue.core.prepared_file import PreparedFileSeal, VerificationCancelledError, file_digest
from polylogue.core.provider_identity import profile_root_for_artifact
from polylogue.core.raw_coordinates import CapturedZipMemberCoordinate
from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind
from polylogue.core.sources import origin_from_provider
from polylogue.core.work_progress import reports_work_progress, stable_productive_identity
from polylogue.logging import WARNING, emit
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.acquisition_boundary import bound_profile_identity, open_bound_path
from polylogue.sources.decoder_json import (
    DecodedRecordSequence,
    JsonlDecodeError,
    PartialJsonStreamError,
    _json_subtree,
    _root_envelope_without,
    _skip_json_subtree,
    claude_ai_object_envelope,
    claude_design_object_envelope,
    drive_chunked_prompt_envelope,
    generic_message_object_envelope,
    grok_export_item_count,
    hermes_snapshot_envelope,
    iter_container_member_files,
    iter_grok_export_events,
    iter_json_container_records,
    iter_root_array_items,
    json_record_container,
    normalize_ijson_stdlib_numbers,
    scan_container_members,
    spill_member_arrays,
    spill_otlp_spans,
)
from polylogue.sources.decoders import owned_json_records
from polylogue.sources.dispatch import (
    BUNDLE_PROVIDERS,
    BundleCandidateDrift,
    admit_parsed_sessions_for_publication,
    bundle_member_sessions,
    is_drive_chunk_sequence,
    is_jsonl_source_path,
    iter_parsed_payload,
    iter_parsed_stream,
    parse_generic_messages_stream,
)
from polylogue.sources.origin_specs import path_declaration_refuses_session
from polylogue.sources.parsers import (
    browser_capture,
    chatgpt,
    codex_state,
    drive,
    grok,
    hermes_spans,
    hermes_state,
    hermes_verification,
    local_agent,
    otel_genai,
)
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.base_support import (
    AdmissionObserver,
    _unknown_wire_type,
    admit_parsed_sessions,
    hermes_unknown_wire_type,
    otel_genai_unknown_wire_type,
)
from polylogue.sources.parsers.claude.ai_parser import parse_ai_stream, parse_design_stream
from polylogue.sources.parsers.hermes_identity import CapturedHermesProfile
from polylogue.sources.prepared_message_sink import (
    ChatGPTNodeMapping,
    ClaudeAttachmentScratch,
    ClaudeChatEvidence,
    GeminiToolOutputIndex,
    ScratchSessionSpill,
    SqliteAttachmentSink,
    SqliteMessageSink,
    SqliteMessageStore,
    SqliteSessionEventSink,
    _prepared_ordinal_rows,
    _prepared_reader,
    discard_decoded_sessions,
    read_chatgpt_mapping_object,
)
from polylogue.sources.sidecar_evidence import RetainedSidecarScope, SidecarResolver
from polylogue.sources.streamed_json_output import write_streamed_json
from polylogue.sources.value_bounds import ValueBoundRefusedError
from polylogue.storage.blob_publication import (
    ArchiveBlobPublisher,
    BlobPublicationSourceRead,
    PreparedBlobPublicationClaim,
    RetainedAttachmentSourceRead,
    _prepared_claim_from_record,
)
from polylogue.storage.blob_store import PreparedBlob
from polylogue.storage.io_phase_metrics import connection_cursor
from polylogue.storage.materials import PreparedMaterial
from polylogue.storage.sqlite.archive_tiers.write import (
    PreparedSessionWrite,
    append_session_to_shard,
    prepare_session_shard,
)
from polylogue.storage.sqlite.session_shard import (
    SessionShard,
    SessionShardBuilder,
    discard_session_shard,
    open_session_shard,
)

if TYPE_CHECKING:
    from polylogue.schemas.retained_validation import RetainedValidationVerdict
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation


_ARTIFACT_VERSION = 8


class _SourceChangedDuringPreparationError(ValueError):
    """The input revision changed while a worker was preparing it."""


def _gemini_cli_envelope(handle: BinaryIO) -> dict[str, JSONValue] | None:
    """Read parser-visible root fields without constructing the transcript."""
    events = iter(ijson.parse(handle))
    if next(events, None) != ("", "start_map", None):
        return None
    envelope: dict[str, JSONValue] = {}
    message_arrays = 0
    for prefix, event, value in events:
        check_compute_cancelled()
        if prefix == "" and event == "end_map":
            if next(events, None) is not None:
                return None
            break
        if prefix != "" or event != "map_key":
            return None
        key = str(value)
        field_prefix, field_event, field_value = next(events)
        if field_prefix != key:
            return None
        if key == "messages":
            if field_event != "start_array":
                return None
            message_arrays += 1
            _skip_json_subtree(events, field_event)
        elif field_event in {"start_map", "start_array"}:
            if key not in {"directories", "memoryScratchpad"}:
                return None
            envelope[key] = cast(JSONValue, _json_subtree(events, field_event, field_value))
        else:
            envelope[key] = cast(JSONValue, normalize_ijson_stdlib_numbers(field_value))
    envelope["messages"] = []
    return envelope if message_arrays == 1 else None


def _append_gemini_raw_message(conn: sqlite3.Connection, ordinal: int, item: object) -> None:
    normalized = normalize_ijson_stdlib_numbers(item)
    conn.execute(
        "INSERT INTO gemini_raw_message VALUES (?, ?)",
        (ordinal, json.dumps(normalized, ensure_ascii=False)),
    )


class DecodeFailure(StrEnum):
    """Which JSON decode boundary refused a source's bytes."""

    #: A JSON document or JSON stream did not decode.
    DOCUMENT = "document"
    #: A complete (newline-terminated or final) JSONL record did not decode.
    JSONL_RECORD = "jsonl_record"


class PreparedDecodeError(ValueError):
    """A worker's decode failure, raised again by the writer with its kind.

    A failed preparation records its decode kind in the sealed carrier. The
    writer reconstructs that typed refusal without parsing retained bytes.
    """

    def __init__(self, kind: DecodeFailure, detail: str) -> None:
        self.kind = kind
        super().__init__(detail)


def classify_decode_failure(error: BaseException) -> DecodeFailure | None:
    """Name the decode boundary ``error`` came from, or ``None`` if it is not a decode failure."""
    if isinstance(error, PreparedDecodeError):
        return error.kind
    if isinstance(error, JsonlDecodeError):
        return DecodeFailure.JSONL_RECORD
    # ``ijson.JSONError`` is the streamed decoders' refusal (a truncated or
    # malformed document read incrementally); it is the same verdict on the
    # bytes as ``json.JSONDecodeError``.
    if isinstance(error, PartialJsonStreamError):
        # The decoder wraps every mid-stream exception after it has yielded
        # records, including backend and source I/O failures. The wrapper is
        # terminal evidence only when its explicit cause says the bytes did
        # not decode; an OSError or parser assertion must remain retryable.
        return (
            DecodeFailure.DOCUMENT
            if isinstance(error.cause, (json.JSONDecodeError, UnicodeDecodeError, ijson.JSONError))
            else None
        )
    if isinstance(error, (json.JSONDecodeError, UnicodeDecodeError, ijson.JSONError)):
        return DecodeFailure.DOCUMENT
    return None


def terminal_decode_evidence(error: BaseException, *, provider: Provider) -> RawFailureEvidenceKind | None:
    """The terminal evidence a decode failure of retained bytes earns, if any.

    Every decode failure is terminal. The evidence is bound to immutable
    captured bytes, so re-reading them can only fail the same way; a source
    that later completes or changes is a new observation. An
    unknown-provider capture is ``terminal_unknown_json_decode``. For a known
    provider, a complete JSONL record or a JSON document that does not decode
    is ``terminal_corrupt_input``, which canonical ingest also reports
    (``classify_decode_exception``). An unterminated JSONL tail never reaches
    here: it is excluded from the parsed prefix. Live intake and retained
    replay both decide from this one rule, so a rebuild refuses exactly the
    bytes live intake refused, and neither re-selects them.
    """
    from polylogue.core.raw_failure_evidence import RetainedRawDecodeRefusalError

    if isinstance(error, RetainedRawDecodeRefusalError):
        return error.kind
    if classify_decode_failure(error) is None:
        return None
    if provider is Provider.UNKNOWN:
        return RawFailureEvidenceKind.TERMINAL_UNKNOWN_JSON_DECODE
    return RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT


@contextmanager
def source_snapshot(source: Path, directory: Path) -> Iterator[tuple[Path, str, CapturedHermesProfile]]:
    """Copy one revision of ``source`` into private scratch and name its digest.

    Everything that decides how a revision is interpreted -- provider
    sampling, the JSONL frontier, the parse and the seal -- reads the copy,
    whose digest was taken from the very bytes written into it. A source
    replaced or rewritten while it is prepared therefore cannot pair one
    revision's digest with another revision's interpretation; the writer
    rejects the carrier when its own capture hashes differently. The copy
    keeps the source's file name, which decides JSON and JSONL handling.
    """
    holder = directory / f"source-{uuid.uuid4().hex}"
    holder.mkdir(mode=0o700, parents=True)
    snapshot = holder / source.name
    try:
        digest = hashlib.sha256()
        with open_bound_path(source, None) as reader, snapshot.open("xb") as writer:
            profile = bound_profile_identity(reader)
            if profile is None:
                raise OSError("source snapshot has no bound declared namespace")
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                check_compute_cancelled()
                digest.update(chunk)
                writer.write(chunk)
        os.chmod(snapshot, 0o400)
        yield snapshot, digest.hexdigest(), profile
    finally:
        with suppress(FileNotFoundError):
            shutil.rmtree(holder)


class _PreparedPrefixInput:
    """Borrow a proven complete JSONL prefix without holding whole lines."""

    def __init__(self, handle: BinaryIO, prefix_size: int) -> None:
        if prefix_size < 0:
            raise ValueError("JSONL prefix size must be non-negative")
        self.handle = handle
        self.remaining = prefix_size

    def read(self, size: int = -1) -> bytes:
        if not self.remaining or size == 0:
            return b""
        check_compute_cancelled()
        requested = min(self.remaining, size if size >= 0 else 64 * 1024)
        chunk = self.handle.read(requested)
        if not chunk:
            raise OSError("sealed source ended before its prepared JSONL prefix")
        self.remaining -= len(chunk)
        return chunk


def _hermes_atif_envelope(handle: BinaryIO) -> tuple[dict[str, JSONValue], bool] | None:
    """Prove one Hermes ATIF trajectory and report whether it has subagents.

    Hermes lowering tries ATOF event lists, state and verification exports
    before ATIF, so documents those detectors claim stay on that route. ATIF
    lowering has no admission ledger, so no future wire type is carried.
    """
    result = _root_envelope_without(handle, frozenset({"steps", "subagent_trajectories"}), frozenset({"atof_version"}))
    if result is None:
        return None
    envelope, arrays = result
    envelope.pop("__admission_future_type", None)
    if arrays["steps"] != 1 or arrays["subagent_trajectories"] > 1:
        return None
    witness: dict[str, JSONValue] = {**envelope, "steps": []}
    if (
        not hermes_spans.looks_like_atif_payload(witness)
        or hermes_state.looks_like_state_db_payload(witness)
        or hermes_verification.looks_like_verification_evidence_db_payload(witness)
    ):
        return None
    return envelope, arrays["subagent_trajectories"] == 1


def _spill_atif_subagents(handle: BinaryIO, conn: sqlite3.Connection) -> bool:
    """Spill each ATIF subagent entry and its steps into scratch, one step at a time.

    Returns ``False`` for an entry that repeats ``steps``; the scratch tables
    are then dropped and the document stays on the object parser.
    """
    conn.execute(
        "CREATE TABLE atif_subagent (ordinal INTEGER PRIMARY KEY, fields_json TEXT NOT NULL, step_count INTEGER)"
    )
    conn.execute(
        "CREATE TABLE atif_subagent_step (subagent INTEGER NOT NULL, ordinal INTEGER NOT NULL, "
        "step_json TEXT NOT NULL, PRIMARY KEY (subagent, ordinal)) WITHOUT ROWID"
    )

    def on_member(index: int, fields: JSONValue, count: int | None) -> None:
        conn.execute("INSERT INTO atif_subagent VALUES (?, ?, ?)", (index, json.dumps(fields), count))

    def on_step(index: int, ordinal: int, step: JSONValue) -> None:
        conn.execute("INSERT INTO atif_subagent_step VALUES (?, ?, ?)", (index, ordinal, json.dumps(step)))

    if spill_member_arrays(handle, "subagent_trajectories", "steps", on_member=on_member, on_nested_item=on_step):
        return True
    _drop_atif_subagents(conn)
    return False


def _drop_atif_subagents(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE atif_subagent")
    conn.execute("DROP TABLE atif_subagent_step")


def _atif_subagent_steps(conn: sqlite3.Connection, subagent: int) -> Iterator[JSONValue]:
    for (step_json,) in conn.execute(
        "SELECT step_json FROM atif_subagent_step WHERE subagent = ? ORDER BY ordinal", (subagent,)
    ):
        yield cast(JSONValue, json.loads(step_json))


def _atif_subagents(conn: sqlite3.Connection) -> Iterator[hermes_spans.AtifSubagent]:
    """Rebuild spilled subagent entries one at a time; steps stay in scratch."""
    for ordinal, fields_json, step_count in conn.execute(
        "SELECT ordinal, fields_json, step_count FROM atif_subagent ORDER BY ordinal"
    ):
        fields = json.loads(fields_json)
        yield hermes_spans.AtifSubagent(
            fields if isinstance(fields, dict) else {},
            step_count,
            partial(_atif_subagent_steps, conn, ordinal),
        )


def _otlp_envelope(handle: BinaryIO) -> tuple[dict[str, JSONValue], str] | None:
    """Prove one OTLP-JSON export and name the root span array the parser reads.

    Returns every root field but the span arrays. Session wrappers and
    browser captures, which the lowering routes elsewhere, stay there.
    """
    result = _root_envelope_without(handle, frozenset({"resourceSpans", "resource_spans"}), frozenset({"sessions"}))
    if result is None:
        return None
    envelope, arrays = result
    envelope.pop("__admission_future_type", None)
    if browser_capture.looks_like(envelope):
        return None
    if arrays["resourceSpans"]:
        return envelope, "resourceSpans"
    if arrays["resource_spans"]:
        return envelope, "resource_spans"
    return None


def _index_otlp_spans(
    handle: BinaryIO, root_key: str, conn: sqlite3.Connection
) -> tuple[otel_genai.OtelSpanIndex, str | None] | None:
    """Spill an OTLP export's spans to scratch, then index them in document order.

    A span's resource identity and scope schema URL may follow it in the
    document, so spans are joined to both only after the walk. Returns the
    index and the first unknown span ``kind`` the admission scan of the
    whole document would report, or ``None``, with no tables left behind,
    when the walk refuses the document.
    """
    conn.execute(
        "CREATE TABLE otlp_resource (resource INTEGER PRIMARY KEY, resource_id TEXT NOT NULL, scope_field TEXT)"
    )
    conn.execute(
        "CREATE TABLE otlp_scope (resource INTEGER NOT NULL, scope_field TEXT NOT NULL, scope INTEGER NOT NULL, "
        "schema_url TEXT NOT NULL, PRIMARY KEY (resource, scope_field, scope)) WITHOUT ROWID"
    )
    conn.execute(
        "CREATE TABLE otlp_span (resource INTEGER NOT NULL, scope_field TEXT NOT NULL, scope INTEGER NOT NULL, "
        "span INTEGER NOT NULL, span_json TEXT NOT NULL, PRIMARY KEY (resource, scope_field, scope, span)) WITHOUT ROWID"
    )
    # The admission scan reads every object span's ``kind``, identified or
    # not, in the scope array the parser reads.
    conn.execute(
        "CREATE TABLE otlp_unknown_kind (resource INTEGER NOT NULL, scope_field TEXT NOT NULL, "
        "scope INTEGER NOT NULL, span INTEGER NOT NULL, kind TEXT NOT NULL, "
        "PRIMARY KEY (resource, scope_field, scope, span)) WITHOUT ROWID"
    )

    def on_resource(resource: int, fields: dict[str, object], scope_field: str | None) -> None:
        conn.execute(
            "INSERT INTO otlp_resource VALUES (?, ?, ?)",
            (resource, json.dumps(otel_genai.resource_id_for(fields)), scope_field),
        )

    def on_scope(resource: int, scope_field: str, scope: int, fields: dict[str, object]) -> None:
        conn.execute(
            "INSERT INTO otlp_scope VALUES (?, ?, ?, ?)",
            (resource, scope_field, scope, json.dumps(otel_genai.scope_schema_url(fields))),
        )

    def on_span(resource: int, scope_field: str, scope: int, span_ordinal: int, span: dict[str, object]) -> None:
        unknown_kind = otel_genai_unknown_wire_type(
            {"resourceSpans": [{"scopeSpans": [{"spans": [{"kind": span.get("kind")}]}]}]}
        )
        if unknown_kind is not None:
            conn.execute(
                "INSERT INTO otlp_unknown_kind VALUES (?, ?, ?, ?, ?)",
                (resource, scope_field, scope, span_ordinal, json.dumps(unknown_kind)),
            )
        if otel_genai.has_span_identity(span):
            conn.execute(
                "INSERT INTO otlp_span VALUES (?, ?, ?, ?, ?)",
                (resource, scope_field, scope, span_ordinal, json.dumps(span)),
            )

    result: tuple[otel_genai.OtelSpanIndex, str | None] | None = None
    if spill_otlp_spans(handle, root_key, on_resource=on_resource, on_scope=on_scope, on_span=on_span):
        unknown_row = conn.execute(
            "SELECT u.kind FROM otlp_unknown_kind u "
            "JOIN otlp_resource r ON r.resource = u.resource AND r.scope_field = u.scope_field "
            "ORDER BY u.resource, u.scope, u.span LIMIT 1"
        ).fetchone()
        index = otel_genai.OtelSpanIndex(conn)
        result = (index, json.loads(unknown_row[0]) if unknown_row is not None else None)
        for resource_id_json, schema_url_json, span_json in conn.execute(
            "SELECT r.resource_id, c.schema_url, s.span_json FROM otlp_span s "
            "JOIN otlp_resource r ON r.resource = s.resource AND r.scope_field = s.scope_field "
            "JOIN otlp_scope c ON c.resource = s.resource AND c.scope_field = s.scope_field AND c.scope = s.scope "
            "ORDER BY s.resource, s.scope, s.span"
        ):
            index.add(json.loads(resource_id_json), json.loads(span_json), json.loads(schema_url_json))
    for table in ("otlp_resource", "otlp_scope", "otlp_span", "otlp_unknown_kind"):
        conn.execute(f"DROP TABLE {table}")
    return result


#: The member fields ``browser_capture.looks_like`` reads.
_BROWSER_CAPTURE_SHAPE_KEYS = frozenset({"polylogue_capture_kind", "schema_version", "session", "provenance"})

#: Parser-only ChatGPT scratch, which never reaches a sealed artifact.
_CHATGPT_PARSER_SCRATCH_TABLES = (
    "chatgpt_node",
    "chatgpt_child",
    "chatgpt_sibling",
    "chatgpt_entry",
    "scratch_string_set",
    "scratch_string_map",
)


def _streamed_bundle_member(
    provider: Provider,
    member_path: Path,
    store: SqliteMessageStore,
    fallback_id: str,
    *,
    all_browser_captures: bool,
) -> tuple[bool, list[ParsedSession]]:
    """Parse one bundle member through its single-object streaming route.

    ``member_path`` holds one object member of a record container. When the
    single-object probe proves the member is a conversation the bundle
    lowering hands to that provider's parser -- a ChatGPT mapping, a
    claude.ai ``chat_messages`` chat, or a Claude Design ``messages`` chat --
    its messages, events and attachments go to ``store`` from the first
    record, and ``(True, sessions)`` is returned. Otherwise, or when the
    probe meets a value SQLite cannot store (the collecting lowering stores
    only the fields it reads, so it decides that refusal), every scratch
    write is rolled back and ``(False, [])`` asks for the collecting route.
    """
    if all_browser_captures or provider not in BUNDLE_PROVIDERS:
        return False, []
    conn = store.conn
    conn.execute("SAVEPOINT bundle_member")
    try:
        session = _stream_member_session(provider, member_path, store, fallback_id)
    except ValueBoundRefusedError:
        session = None
        streamed = False
    except BaseException:
        conn.execute("ROLLBACK TO bundle_member")
        conn.execute("RELEASE bundle_member")
        raise
    else:
        streamed = session is not None
    if not streamed:
        conn.execute("ROLLBACK TO bundle_member")
    conn.execute("RELEASE bundle_member")
    return streamed, [session] if session is not None else []


def _stream_member_session(
    provider: Provider, member_path: Path, store: SqliteMessageStore, fallback_id: str
) -> ParsedSession | None:
    if provider is Provider.CHATGPT:
        with member_path.open("rb") as handle:
            read_result = read_chatgpt_mapping_object(handle, store.conn)
        if read_result is None:
            return None
        envelope, mapping = read_result
        shallow = mapping.shallow_view()
        if (
            # A mapping-carrying capture envelope keeps its own lowering, and
            # the fragment shape is what the bundle lowering admits.
            browser_capture.looks_like({**envelope, "mapping": {}})
            or not mapping.children_are_all_strings()
            or not chatgpt._mapping_nodes_are_valid(shallow)
            or not chatgpt._mapping_node_shape_is_plausible(shallow)
        ):
            return None
        session = chatgpt.parse({**envelope, "mapping": shallow}, fallback_id, spill=ScratchSessionSpill(store))
        source_attachments: object = session.attachments
        if not isinstance(source_attachments, SqliteAttachmentSink):
            attachments = store.new_attachment_sink()
            attachments.extend(session.attachments)
            session = session.model_copy(update={"attachments": attachments})
        source_events: object = session.session_events
        if not isinstance(source_events, SqliteSessionEventSink):
            events = store.new_event_sink()
            events.extend(session.session_events)
            session = session.model_copy(update={"session_events": events})
        for table in _CHATGPT_PARSER_SCRATCH_TABLES:
            store.conn.execute(f"DROP TABLE IF EXISTS {table}")
        return session
    if provider is Provider.CLAUDE_AI:
        with member_path.open("rb") as handle:
            claude_object = claude_ai_object_envelope(handle)
        if claude_object is None:
            return None
        return _stream_claude_ai_object(member_path, *claude_object, store, fallback_id)
    with member_path.open("rb") as handle:
        design_envelope = claude_design_object_envelope(handle)
    if design_envelope is None:
        return None
    with member_path.open("rb") as handle:
        return parse_design_stream(
            design_envelope,
            (normalize_ijson_stdlib_numbers(item) for item in ijson.items(handle, "messages.item")),
            fallback_id,
            message_sink=store.new_sink(),
            event_sink=store.new_event_sink(),
            attachment_sink=store.new_attachment_sink(),
        )


def _stream_claude_ai_object(
    path: Path,
    envelope: dict[str, JSONValue],
    attachment_arrays: tuple[str, ...],
    store: SqliteMessageStore,
    fallback_id: str,
) -> ParsedSession:
    """Lower a proved claude.ai conversation with its evidence, graph and attachments in scratch."""

    def conversation_attachments() -> Iterator[JSONValue]:
        for key in attachment_arrays:
            with path.open("rb") as handle:
                yield from iter_root_array_items(handle, key)

    evidence_store = ClaudeChatEvidence(store.conn)
    attachment_rows = ClaudeAttachmentScratch(store.conn)
    with path.open("rb") as handle:
        session = parse_ai_stream(
            envelope,
            (normalize_ijson_stdlib_numbers(item) for item in ijson.items(handle, "chat_messages.item")),
            fallback_id,
            conversation_attachments=conversation_attachments(),
            evidence_store=evidence_store,
            graph_connection=store.conn,
            messages=store.new_sink(),
            session_events=store.new_event_sink(),
            attachment_rows=attachment_rows,
            attachments=store.new_attachment_sink(),
        )
    evidence_store.close()
    attachment_rows.close()
    return session


@dataclass(frozen=True, slots=True)
class _PreparedCodexThreads:
    artifact: PreparedJsonl

    def __iter__(self) -> Generator[codex_state.CodexThreadRecord, None, None]:
        with closing(self.artifact._iter_codex_records("prepared_codex_thread")) as rows:
            for row in rows:
                yield codex_state.CodexThreadRecord(
                    thread_id=cast(str, row["thread_id"]),
                    title=cast(str, row["title"]),
                    cwd=cast(str, row["cwd"]),
                    created_at_ms=cast(int, row["created_at_ms"]),
                    updated_at_ms=cast(int, row["updated_at_ms"]),
                    source=cast(str, row["source"]),
                    model=cast(str | None, row["model"]),
                    agent_nickname=cast(str | None, row["agent_nickname"]),
                    agent_role=cast(str | None, row["agent_role"]),
                    archived=cast(bool, row["archived"]),
                )


@dataclass(frozen=True, slots=True)
class _PreparedCodexSpawnEdges:
    artifact: PreparedJsonl

    def __iter__(self) -> Generator[codex_state.CodexSpawnEdge, None, None]:
        with closing(self.artifact._iter_codex_records("prepared_codex_spawn")) as rows:
            for row in rows:
                yield codex_state.CodexSpawnEdge(
                    parent_thread_id=cast(str, row["parent_thread_id"]),
                    child_thread_id=cast(str, row["child_thread_id"]),
                    status=cast(str, row["status"]),
                )


@dataclass(slots=True)
class _ArtifactBlobPublication:
    publisher: ArchiveBlobPublisher | None = None
    seal: PreparedIndexMutation | None = None
    started: bool = False
    retired: bool = False
    phase: Literal["attachments", "sidecars", "materials", "complete"] = "attachments"
    attachment_after: tuple[int, int] = (-1, -1)
    sidecar_after: tuple[int, str] = (-1, "")
    page: tuple[PreparedBlobPublicationClaim, ...] = ()
    prepared: bool = False
    material_page: tuple[PreparedMaterial, ...] = ()
    material_after: int = -1
    next_material_after: int = -1
    next_attachment_after: tuple[int, int] = (-1, -1)
    next_sidecar_after: tuple[int, str] = (-1, "")


@dataclass(slots=True)
class _ArtifactThreadProjection:
    seal: PreparedIndexMutation | None = None
    projection: PreparedThreadStateProjection | None = None
    index_available: bool | None = None
    applied: bool = False
    raw_id: str | None = None
    blob_hash: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedJsonl:
    """A disposable artifact tied to one retained JSON source revision."""

    blob_hash: str | None
    sessions_path: Path | None
    shard_path: Path | None
    error: str | None = None
    deferred: bool = False
    enrichment_digest: str | None = None
    enrichment_index_path: str | None = None
    sessions_seal: PreparedFileSeal | None = None
    shard_seal: PreparedFileSeal | None = None
    prepared_writes: tuple[PreparedSessionWrite, ...] = ()
    parsed_prefix_size: int | None = None
    resolved_provider: Provider | None = None
    validation_verdict: RetainedValidationVerdict | None = field(default=None, compare=False, repr=False)
    parser_stage_artifact: PreparedJsonl | None = field(default=None, compare=False, repr=False)
    positive_evidence_filtered: bool = False
    attempt_directory: Path | None = None
    #: For a terminal failure that never read bytes into a blob (a worker
    #: lost on this file), the source's (size, mtime_ns, inode) when the
    #: failing preparation began. Publication must not apply the failure to a
    #: capture of any other revision.
    failed_observation: tuple[int, int, int] | None = None
    #: For a terminal failure, which decode boundary refused the bytes, or
    #: ``None`` when the failure was not a decode failure.
    decode_failure: DecodeFailure | None = None
    missing_profile_identity: bool = False
    retained_zip_membership_unproved: bool = False
    unsupported_shape: bool = False
    captured_profile_key: str | None = None
    codex_state_kind: str | None = None
    codex_state_text_chars: int = codex_state.CODEX_STATE_MAX_TEXT_CHARS
    publication_publisher: ArchiveBlobPublisher | None = None
    _blob_publication: _ArtifactBlobPublication = field(
        default_factory=_ArtifactBlobPublication, compare=False, repr=False
    )
    _thread_projection: _ArtifactThreadProjection = field(
        default_factory=_ArtifactThreadProjection, compare=False, repr=False
    )

    def prepare_thread_projection(
        self,
        seal: PreparedIndexMutation,
        *,
        source_read: SessionSourceRead,
        raw_id: str,
        blob_hash: str,
        observed_at_ms: int,
        observation_order: int,
        source_path: str,
    ) -> None:
        from polylogue.sources.codex_state_projection import codex_state_source_scope, prepare_thread_state_projection

        state = self._thread_projection
        if state.index_available is not None:
            if state.seal is not seal or state.raw_id != raw_id or state.blob_hash != blob_hash:
                raise RuntimeError("thread projection cannot acquire a different original witness")
            if state.index_available and state.projection is None:
                raise RuntimeError("thread projection preparation failed and requires original cleanup")
            return
        if self.blob_hash != blob_hash:
            raise ValueError("thread projection differs from its exact prepared source bytes")
        snapshot = self.codex_state_snapshot
        if snapshot is None:
            raise ValueError("prepared thread-state snapshot is absent")
        state.seal = seal
        state.raw_id = raw_id
        state.blob_hash = blob_hash
        state.index_available = seal.has_tier_capability("index")
        if not state.index_available:
            return
        directory = self.sessions_path.parent if self.sessions_path is not None else self.attempt_directory
        if directory is None:
            raise RuntimeError("prepared thread projection has no owned artifact directory")
        # Leave normal capability unprepared on any failure, so a later
        # publication cannot interpret that failure as absent Index.
        state.projection = prepare_thread_state_projection(
            seal,
            snapshot,
            directory=directory,
            raw_id=raw_id,
            blob_hash=blob_hash,
            observed_at_ms=observed_at_ms,
            observation_order=observation_order,
            source_scope=codex_state_source_scope(source_path),
            source_read=source_read,
        )

    def apply_thread_projection(self, seal: PreparedIndexMutation, connection: sqlite3.Connection) -> bool:
        state = self._thread_projection
        if state.seal is not seal or state.index_available is None:
            raise RuntimeError("thread projection has no matching original preparation")
        if not state.index_available:
            return False
        if state.projection is None or state.applied:
            raise RuntimeError("thread projection is unavailable or already applied")
        written = state.projection.apply(connection)
        state.applied = True
        return written

    @classmethod
    def from_sessions(
        cls,
        sessions: Iterable[ParsedSession],
        *,
        blob_hash: str,
        artifact_directory: Path,
        publication_publisher: ArchiveBlobPublisher | None,
        publication_source_read: BlobPublicationSourceRead | None = None,
        classification: ArtifactStreamClassification | None = None,
        enrichment_digest: str | None = None,
        enrichment_index_path: str | None = None,
        parsed_prefix_size: int | None = None,
        resolved_provider: Provider | None = None,
        captured_profile_key: str | None = None,
        preparation_dependency: Callable[[], tuple[str | None, str | None]] | None = None,
    ) -> PreparedJsonl:
        """Seal already-admitted parser output on the canonical paged carrier.

        The caller owns the private directory through physical preparation
        completion, including failures. No parser or alternate attachment
        representation is introduced at this boundary.
        """
        from polylogue.core.compute import capture_compute_bridge
        from polylogue.core.sql_settlement import retain_native_sql_lifetimes
        from polylogue.storage.sqlite.connection_profile import native_sql_children

        sessions_path = artifact_directory / f"sessions-{uuid.uuid4().hex}.db"
        shard_path = artifact_directory / f"shard-{uuid.uuid4().hex}.db"
        store: SqliteMessageStore | None = None
        builder: SessionShardBuilder | None = None

        provisional = cls(blob_hash, sessions_path, shard_path, attempt_directory=artifact_directory)
        try:
            with capture_compute_bridge()(), retain_native_sql_lifetimes(artifact_directory):
                try:
                    store = SqliteMessageStore(sessions_path)
                    builder = SessionShardBuilder(shard_path)

                    def lowered_sessions() -> Iterator[ParsedSession]:
                        assert builder is not None and store is not None
                        for session in sessions:
                            check_compute_cancelled()
                            # Keep parser fields unchanged. The same artifact
                            # retains a separate canonical writer operand.
                            messages = store.new_sink()
                            messages.extend(session.messages)
                            retained = session.model_copy(update={"messages": messages})
                            append_session_to_shard(builder, retained)
                            yield retained

                    _write_artifact(
                        store,
                        blob_hash,
                        lowered_sessions(),
                        enrichment_digest=enrichment_digest,
                        enrichment_index_path=enrichment_index_path,
                    )
                    if preparation_dependency is not None:
                        enrichment_digest, enrichment_index_path = preparation_dependency()
                        store.conn.execute(
                            "UPDATE artifact_seal SET enrichment_digest=?, enrichment_index_path=?",
                            (enrichment_digest, enrichment_index_path),
                        )
                    if classification is not None:
                        record_prepared_classification(store.conn, classification)
                    if publication_publisher is not None:
                        _prepare_attachment_publications(
                            store, publication_publisher, artifact_directory, source_read=publication_source_read
                        )
                        _prepare_sidecar_publications(store, publication_publisher, artifact_directory)
                    builder.seal()
                    builder = None
                    store.conn.commit()
                    store.close()
                    store = None
                    return cls.seal(
                        blob_hash,
                        sessions_path,
                        shard_path,
                        enrichment_digest=enrichment_digest,
                        enrichment_index_path=enrichment_index_path,
                        parsed_prefix_size=parsed_prefix_size,
                        resolved_provider=resolved_provider,
                        captured_profile_key=captured_profile_key,
                        positive_evidence_filtered=True,
                        attempt_directory=artifact_directory,
                        publication_publisher=publication_publisher,
                    )
                except BaseException as primary:
                    failures: list[BaseException] = [primary]
                    for close in (
                        builder.abandon
                        if builder is not None
                        and not any(
                            owner.close_required and not owner._settled for owner in native_sql_children(builder)
                        )
                        else None,
                        store.close
                        if store is not None and (not store._sql_owner.close_required or store._sql_owner._settled)
                        else None,
                    ):
                        if close is not None:
                            try:
                                close()
                            except BaseException as cleanup:
                                failures.append(cleanup)
                    if len(failures) > 1:
                        raise BaseExceptionGroup("canonical session preparation cleanup failed", failures) from None
                    raise
        except BaseException as primary:
            try:
                provisional.discard()
            except BaseException as cleanup:
                raise BaseExceptionGroup("canonical carrier and cleanup failed", [primary, cleanup]) from None
            raise

    @classmethod
    def seal(
        cls,
        blob_hash: str,
        sessions_path: Path,
        shard_path: Path,
        *,
        enrichment_digest: str | None = None,
        enrichment_index_path: str | None = None,
        parsed_prefix_size: int | None = None,
        resolved_provider: Provider | None = None,
        captured_profile_key: str | None = None,
        positive_evidence_filtered: bool = False,
        attempt_directory: Path | None = None,
        codex_state_kind: str | None = None,
        codex_state_text_chars: int = codex_state.CODEX_STATE_MAX_TEXT_CHARS,
        publication_publisher: ArchiveBlobPublisher | None = None,
    ) -> PreparedJsonl:
        """Take custody only after both SQLite writers have closed."""
        return cls(
            blob_hash,
            sessions_path,
            shard_path,
            enrichment_digest=enrichment_digest,
            enrichment_index_path=enrichment_index_path,
            sessions_seal=PreparedFileSeal.capture(sessions_path),
            shard_seal=PreparedFileSeal.capture(shard_path),
            parsed_prefix_size=parsed_prefix_size,
            resolved_provider=resolved_provider,
            captured_profile_key=captured_profile_key,
            positive_evidence_filtered=positive_evidence_filtered,
            attempt_directory=attempt_directory,
            codex_state_kind=codex_state_kind,
            codex_state_text_chars=codex_state_text_chars,
            publication_publisher=publication_publisher,
        )

    def verify_files(self, *, full: bool, stop: Callable[[], bool] | None = None) -> None:
        """Scan bytes before admission; recheck inode identity at publication.

        ``stop`` is polled between digest chunks; when it returns true the
        scan raises :class:`polylogue.core.prepared_file.VerificationCancelledError`.
        """
        if (
            self.sessions_path is None
            or self.shard_path is None
            or self.sessions_seal is None
            or self.shard_seal is None
        ):
            raise ValueError("JSONL preparation lacks closed-file seals")
        self.sessions_seal.verify(self.sessions_path, full=full, stop=stop)
        self.shard_seal.verify(self.shard_path, full=full, stop=stop)

    def discard(self) -> None:
        if self._thread_projection.projection is not None:
            self._thread_projection.projection.close()
        state = self._blob_publication
        if (state.page or state.material_page) and state.seal is not None:
            from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError, native_sql_children

            pending = tuple(
                owner for owner in native_sql_children(state.seal) if owner.close_required and not owner._settled
            )
            if pending:
                raise NativeConnectionSettlementError(
                    pending[0], RuntimeError("artifact Blob continuation requires its original native drain")
                )
        if (state.page or state.material_page) and state.publisher is not None:
            state.publisher.discard_pending()
        state.page = ()
        state.material_page = ()
        state.retired = True
        failures: list[BaseException] = []
        if self.parser_stage_artifact is not None:
            try:
                self.parser_stage_artifact.discard()
            except BaseException as exc:
                failures.append(exc)
            object.__setattr__(self, "parser_stage_artifact", None)
        for prepared in self.prepared_writes:
            try:
                prepared.close()
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise BaseExceptionGroup("prepared write cleanup failed", failures)
        from polylogue.storage.sqlite.connection_profile import (
            NativeConnectionSettlementError,
            retained_native_sql_owners_for_lifetime,
        )

        for dependency in (self.sessions_path, self.shard_path, self.attempt_directory):
            pending = retained_native_sql_owners_for_lifetime(dependency) if dependency is not None else ()
            if pending:
                raise NativeConnectionSettlementError(
                    pending[0], RuntimeError("artifact cleanup requires native drain")
                )
        if self.sessions_path is not None:
            discard_decoded_sessions(self.sessions_path)
        if self.attempt_directory is not None:
            try:
                shutil.rmtree(self.attempt_directory)
            except FileNotFoundError:
                pass
            except OSError:
                emit(
                    "live.parse_prefetch.cleanup_blocked",
                    level=WARNING,
                    outcome="degraded",
                    reason="sealed attempt scratch removal failed",
                )
                raise
            return
        if self.sessions_path is not None:
            self.sessions_path.unlink(missing_ok=True)
            self.sessions_path.with_name(self.sessions_path.name + "-journal").unlink(missing_ok=True)
        if self.shard_path is not None:
            discard_session_shard(self.shard_path)

    def iter_provider_session_ids(self) -> Iterator[str]:
        """Read the complete original native-ID scope without hydrating transcripts."""
        if self.sessions_path is None:
            raise RuntimeError("prepared cohort has no header artifact")
        self.verify_files(full=False)
        for (metadata_json,) in _prepared_ordinal_rows(
            self.sessions_path,
            table="prepared_session",
            ordinal="ordinal",
            session=None,
            columns="metadata_json",
        ):
            check_compute_cancelled()
            if not isinstance(metadata_json, str):
                raise ValueError("prepared session metadata is not text")
            identity = json.loads(metadata_json)["provider_session_id"]
            if identity is not None and not isinstance(identity, str):
                raise ValueError("prepared provider identity is not text")
            if identity:
                yield identity

    def iter_sessions(self) -> Generator[ParsedSession]:
        if self.sessions_path is None or self.blob_hash is None:
            raise RuntimeError(self.error or "JSONL preparation has no sealed artifact")
        if self.shard_path is None:
            raise ValueError("JSONL preparation has no row shard")
        self.verify_files(full=False)
        shard = open_session_shard(self.shard_path)
        with _prepared_reader(self.sessions_path) as conn:
            seal = conn.execute(
                "SELECT version, source_hash, session_count, enrichment_digest, enrichment_index_path "
                "FROM artifact_seal"
            ).fetchall()
            if seal != [
                (
                    _ARTIFACT_VERSION,
                    self.blob_hash,
                    len(shard.sessions),
                    self.enrichment_digest,
                    self.enrichment_index_path,
                )
            ]:
                raise ValueError("JSONL preparation seal or source dependency changed")
            session_count = conn.execute("SELECT COUNT(*) FROM prepared_session").fetchone()[0]
            if session_count != len(shard.sessions):
                raise ValueError("JSONL preparation session count disagrees with row shard")
        for row in _prepared_ordinal_rows(
            self.sessions_path,
            table="prepared_session",
            ordinal="ordinal",
            session=None,
            columns=(
                "ordinal, session_id, metadata_json, message_ordinal, message_count, event_ordinal, event_count, "
                "attachment_ordinal, attachment_count, "
                "(SELECT COUNT(*) FROM prepared_message WHERE session_ordinal = prepared_session.message_ordinal), "
                "(SELECT COUNT(*) FROM prepared_event WHERE session_ordinal = prepared_session.event_ordinal), "
                "(SELECT COUNT(*) FROM prepared_attachment WHERE session_ordinal = prepared_session.attachment_ordinal)"
            ),
        ):
            if not isinstance(row[0], int):
                raise ValueError("prepared ordinal is not an integer")
            ordinal = row[0]
            session_id = str(row[1])
            metadata_json = str(row[2])
            message_ordinal, message_count, event_ordinal, event_count, attachment_ordinal, attachment_count = (
                int(cast(int, value)) for value in row[3:9]
            )
            physical_count, physical_events, physical_attachments = (int(cast(int, value)) for value in row[9:])
            shard_entry = shard.sessions[ordinal]
            if shard_entry.session_id != session_id:
                raise ValueError("JSONL preparation session identity disagrees with row shard")
            metadata = json.loads(metadata_json)
            if physical_count != message_count or physical_count != shard_entry.message_row_count:
                raise ValueError("JSONL preparation message count disagrees with row shard")
            if physical_events != event_count:
                raise ValueError("JSONL preparation event count changed")
            if physical_attachments != attachment_count:
                raise ValueError("JSONL preparation attachment count changed")
            metadata["messages"] = []
            metadata["session_events"] = []
            metadata["attachments"] = []
            _restore_spilled_accounting(metadata, self.sessions_path)
            session = ParsedSession.model_validate(metadata)
            yield session.model_copy(
                update={
                    "messages": SqliteMessageSink(self.sessions_path, message_ordinal, count=message_count),
                    "session_events": SqliteSessionEventSink(self.sessions_path, event_ordinal, count=event_count),
                    "attachments": SqliteAttachmentSink(self.sessions_path, attachment_ordinal, count=attachment_count),
                }
            )

    @property
    def codex_state_snapshot(self) -> codex_state.CodexStateSnapshot | None:
        if self.codex_state_kind != "thread_state":
            return None
        return codex_state.CodexStateSnapshot(
            threads=_PreparedCodexThreads(self),
            spawn_edges=_PreparedCodexSpawnEdges(self),
        )

    def _iter_codex_records(
        self, table: Literal["prepared_codex_thread", "prepared_codex_spawn"]
    ) -> Generator[dict[str, object], None, None]:
        from polylogue.sources.prepared_message_sink import _prepared_reader

        if self.sessions_path is None or self.codex_state_kind != "thread_state":
            raise ValueError("prepared artifact has no thread state")
        self.verify_files(full=False)
        ordinal = -1
        while True:
            check_compute_cancelled()
            with _prepared_reader(self.sessions_path) as conn:
                with connection_cursor(conn, "SELECT kind FROM prepared_codex_state") as cursor:
                    kind = cursor.fetchall()
                if kind != [("thread_state",)]:
                    raise ValueError("prepared state kind changed")
                with connection_cursor(
                    conn,
                    f"SELECT ordinal, metadata_json FROM {table} WHERE ordinal > ? ORDER BY ordinal LIMIT 256",
                    (ordinal,),
                ) as cursor:
                    rows = cursor.fetchall()
            if not rows:
                return
            ordinal = int(rows[-1][0])
            for _ordinal, metadata in rows:
                check_compute_cancelled()
                yield json.loads(metadata)

    def _codex_state_material_page(
        self,
        after: int,
    ) -> tuple[tuple[int, str, str, str, int, PreparedMaterial | None], ...]:
        """Return one closed carrier page with its exact ordinal coordinates."""
        from polylogue.storage.materials import _prepared_material_from_record

        if self.sessions_path is None or self.codex_state_kind not in {"goals", "memories"}:
            raise ValueError("prepared artifact has no state material")
        if self.publication_publisher is None:
            raise ValueError("prepared material has no captured publisher owner")
        self.verify_files(full=False)
        check_compute_cancelled()
        with _prepared_reader(self.sessions_path) as connection:
            with connection_cursor(connection, "SELECT kind FROM prepared_codex_state") as cursor:
                kind = cursor.fetchall()
            if kind != [(self.codex_state_kind,)]:
                raise ValueError("prepared state kind changed")
            with connection_cursor(
                connection,
                "SELECT ordinal, thread_id, item_id, part_kind, byte_size, prepared_json FROM prepared_codex_state_part "
                "WHERE ordinal > ? ORDER BY ordinal LIMIT 256",
                (after,),
            ) as cursor:
                rows = cursor.fetchall()
        return tuple(
            (
                int(ordinal),
                str(thread_id),
                str(item_id),
                str(part_kind),
                int(byte_size),
                _prepared_material_from_record(prepared_json, self.publication_publisher)
                if prepared_json is not None
                else None,
            )
            for ordinal, thread_id, item_id, part_kind, byte_size, prepared_json in rows
        )

    def iter_codex_state_material(self) -> Generator[tuple[str, str, str, int, PreparedMaterial | None], None, None]:
        """Read complete pre-encoded state parts through closed bounded pages."""
        ordinal = -1
        while True:
            rows = self._codex_state_material_page(ordinal)
            if not rows:
                return
            ordinal = rows[-1][0]
            for _ordinal, thread_id, item_id, part_kind, byte_size, material in rows:
                check_compute_cancelled()
                yield thread_id, item_id, part_kind, byte_size, material

    def iter_attachment_claims(
        self, *, session_ordinal: int | None = None, after: tuple[int, int] = (-1, -1)
    ) -> Generator[tuple[int, int, PreparedBlobPublicationClaim], None, None]:
        """Read exact captured claims through closed bounded artifact pages."""
        if self.sessions_path is None:
            raise ValueError("prepared artifact has no attachment carrier")
        self.verify_files(full=False)
        while True:
            check_compute_cancelled()
            with _prepared_reader(self.sessions_path) as connection:
                predicate = "" if session_ordinal is None else "AND session_ordinal=? "
                parameters = after if session_ordinal is None else (*after, session_ordinal)
                with connection_cursor(
                    connection,
                    "SELECT session_ordinal, attachment_ordinal, claim_json FROM prepared_attachment_publication "
                    "WHERE (session_ordinal, attachment_ordinal) > (?, ?) "
                    + predicate
                    + "ORDER BY session_ordinal, attachment_ordinal LIMIT 256",
                    parameters,
                ) as cursor:
                    rows = cursor.fetchall()
            if not rows:
                return
            if self.publication_publisher is None:
                raise ValueError("prepared attachments have no captured publisher")
            after = (int(rows[-1][0]), int(rows[-1][1]))
            for session_ordinal, attachment_ordinal, encoded in rows:
                check_compute_cancelled()
                yield (
                    int(session_ordinal),
                    int(attachment_ordinal),
                    _prepared_claim_from_record(str(encoded), self.publication_publisher),
                )

    def attachment_blobs(
        self, *, source_read: BlobPublicationSourceRead, session_id: str
    ) -> Mapping[object, tuple[bytes | None, int, str]]:
        """Borrow actual ordinary or selected Source evidence for the sealed attachment view."""
        if self.sessions_path is None:
            raise ValueError("prepared artifact has no attachment carrier")
        self.verify_files(full=False)
        with (
            _prepared_reader(self.sessions_path) as connection,
            connection_cursor(
                connection, "SELECT attachment_ordinal FROM prepared_session WHERE session_id=? LIMIT 2", (session_id,)
            ) as cursor,
        ):
            rows = cursor.fetchall()
        if len(rows) != 1:
            raise KeyError(session_id)
        return _PreparedAttachmentBlobs(self, source_read, int(rows[0][0]))

    def resident_attachment_blobs(
        self, *, source_read: RetainedAttachmentSourceRead, session_id: str, raw_id: str
    ) -> Mapping[object, tuple[bytes | None, int, str]]:
        """Bind acquired claims to current durable Raw evidence, not a consumed reservation."""
        if self.sessions_path is None:
            raise ValueError("prepared artifact has no attachment carrier")
        self.verify_files(full=False)
        with (
            _prepared_reader(self.sessions_path) as connection,
            connection_cursor(
                connection, "SELECT attachment_ordinal FROM prepared_session WHERE session_id=? LIMIT 2", (session_id,)
            ) as cursor,
        ):
            rows = cursor.fetchall()
        if len(rows) != 1:
            raise KeyError(session_id)
        return _ResidentAttachmentBlobs(self, source_read, int(rows[0][0]), raw_id)

    def iter_attachment_refs(
        self,
        *,
        source_path: str,
        acquired_at_ms: int,
        source_read: BlobPublicationSourceRead,
        before_input: Callable[[PreparedBlobPublicationClaim], object] | None = None,
    ) -> Iterator[ArchiveSourceBlobRef]:
        from polylogue.core.identity_law import attachment_acquisition_coordinate
        from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveSourceBlobRef

        if self.sessions_path is None:
            raise ValueError("prepared artifact has no attachment carrier")
        for session_ordinal, attachment_ordinal, claim in self.iter_attachment_claims():
            blob_hash = bytes.fromhex(claim.receipt.blob_hash)
            if not source_read.publication_blob_is_excised(blob_hash):
                assert self.publication_publisher is not None
                if before_input is not None:
                    before_input(claim)
                self.publication_publisher.validate_published_claim(source_read, claim, source_path=source_path)
                with (
                    _prepared_reader(self.sessions_path) as connection,
                    connection_cursor(
                        connection,
                        "SELECT json_extract(attachment_json,'$.provider_file_id'),"
                        "json_extract(attachment_json,'$.provider_attachment_id') FROM prepared_attachment "
                        "WHERE session_ordinal=? AND attachment_ordinal=?",
                        (session_ordinal, attachment_ordinal),
                    ) as cursor,
                ):
                    coordinate = cursor.fetchone()
                if coordinate is None:
                    raise ValueError("prepared attachment coordinate disappeared")
                provider_file_id, provider_attachment_id = coordinate
                if (provider_file_id is not None and not isinstance(provider_file_id, str)) or not isinstance(
                    provider_attachment_id, str
                ):
                    raise ValueError("prepared attachment coordinate is not a declared provider identity")
                yield ArchiveSourceBlobRef(
                    blob_hash=blob_hash,
                    ref_type="attachment",
                    source_path=attachment_acquisition_coordinate(provider_file_id, provider_attachment_id),
                    size_bytes=claim.receipt.size_bytes,
                    acquired_at_ms=acquired_at_ms,
                    publication_receipt_id=claim.receipt.publication_id,
                )

    def iter_sidecar_claims(
        self, *, session_ordinal: int | None = None, after: tuple[int, str] = (-1, "")
    ) -> Generator[tuple[int, str, PreparedBlobPublicationClaim, bool], None, None]:
        if self.sessions_path is None or self.publication_publisher is None:
            return
        self.verify_files(full=False)
        while True:
            check_compute_cancelled()
            with _prepared_reader(self.sessions_path) as connection:
                predicate = "" if session_ordinal is None else "AND session_ordinal=? "
                parameters = after if session_ordinal is None else (*after, session_ordinal)
                with connection_cursor(
                    connection,
                    "SELECT session_ordinal,tool_use_id,claim_json,already_present FROM prepared_sidecar_publication "
                    "WHERE claim_json IS NOT NULL AND (session_ordinal,tool_use_id)>(?,?) "
                    + predicate
                    + "ORDER BY session_ordinal,tool_use_id LIMIT 256",
                    parameters,
                ) as cursor:
                    rows = cursor.fetchall()
            if not rows:
                return
            after = int(rows[-1][0]), str(rows[-1][1])
            for ordinal, tool_use_id, encoded, present in rows:
                yield (
                    int(ordinal),
                    str(tool_use_id),
                    _prepared_claim_from_record(str(encoded), self.publication_publisher),
                    bool(present),
                )

    def publish_blobs(self, *, reference_seal: PreparedIndexMutation | None = None) -> None:
        """Reserve closed pages and retry only their original pending publication."""
        publisher = self.publication_publisher
        state = self._blob_publication
        if state.retired:
            raise RuntimeError("artifact Blob publication was terminally retired")
        if state.started and (state.publisher is not publisher or state.seal is not reference_seal):
            raise RuntimeError("artifact Blob continuation requires its original publisher and seal")
        if not state.started:
            state.publisher = publisher
            state.seal = reference_seal
            state.started = True

        def flush_page() -> None:
            assert publisher is not None
            if not state.prepared:
                raise RuntimeError("artifact Blob page did not complete its original reservation preparation")
            if state.material_page:
                from polylogue.storage.materials import flush_material_publication_page

                flush_material_publication_page(state.material_page, publisher, reference_seal=reference_seal)
                state.material_page = ()
                state.prepared = False
                state.material_after = state.next_material_after
                return
            if reference_seal is None:
                publisher.flush()
            else:
                from polylogue.core.stage_admission import admit_stage_write
                from polylogue.core.write_lease import current_write_lease

                if current_write_lease() is not None:
                    raise RuntimeError("prepared Blob reservations require lease-free preparation")
                admit_stage_write(
                    "prepared-artifact-blob-reservations",
                    partial(publisher.flush, reference_seal=reference_seal),
                )
            for completed in state.page:
                publisher.forget_completed_claim(completed)
            state.page = ()
            state.prepared = False
            state.attachment_after = state.next_attachment_after
            state.sidecar_after = state.next_sidecar_after

        # No carrier iterator or SQL cursor survives the admission boundary.
        # An interrupted flush keeps this exact page; reentry does not queue
        # it again or prepare a second reservation on its consumed tape.
        if state.page or state.material_page:
            if reference_seal is not None:
                if not state.prepared or publisher is None:
                    raise RuntimeError("artifact Blob page did not complete its original reservation preparation")
                # Creator cleanup precedes a new admission gate: a failed
                # physical child can still hold the original gate's claim.
                # This settles only an authoritative accepted reservation;
                # unaccepted or uncertain attempts cannot resume exposure.
                publisher.settle_prepared_flush(reference_seal=reference_seal)
            flush_page()
        while state.phase != "complete":
            check_compute_cancelled()
            if state.phase == "attachments":
                with closing(self.iter_attachment_claims(after=state.attachment_after)) as attachment_claims:
                    attachment_rows = tuple(islice(attachment_claims, 256))
                if not attachment_rows:
                    state.phase = "sidecars"
                    continue
                state.page = tuple(row[2] for row in attachment_rows)
                state.next_attachment_after = attachment_rows[-1][0], attachment_rows[-1][1]
            elif state.phase == "sidecars":
                with closing(self.iter_sidecar_claims(after=state.sidecar_after)) as sidecar_claims:
                    sidecar_rows = tuple(islice(sidecar_claims, 256))
                if not sidecar_rows:
                    state.phase = "materials" if self.codex_state_kind in {"goals", "memories"} else "complete"
                    continue
                state.page = tuple(row[2] for row in sidecar_rows)
                state.next_sidecar_after = sidecar_rows[-1][0], sidecar_rows[-1][1]
            else:
                from polylogue.storage.materials import prepare_material_publication_page

                material_rows = self._codex_state_material_page(state.material_after)
                if not material_rows:
                    state.phase = "complete"
                    continue
                state.next_material_after = material_rows[-1][0]
                state.material_page = tuple(
                    row[5] for row in material_rows if row[5] is not None and row[5].blob is not None
                )
                if not state.material_page:
                    state.material_after = state.next_material_after
                    continue
                material_publisher = prepare_material_publication_page(
                    state.material_page, reference_seal=reference_seal
                )
                if material_publisher is not publisher:
                    raise RuntimeError("artifact material page lost its original publisher")
                state.prepared = True
                flush_page()
                continue
            assert publisher is not None
            for claim in state.page:
                publisher.queue_prepared(
                    PreparedBlob(claim.receipt.blob_hash, claim.receipt.size_bytes, claim.prepared_path), claim=claim
                )
            if reference_seal is not None:
                from polylogue.core.write_lease import current_write_lease

                if current_write_lease() is not None:
                    raise RuntimeError("prepared Blob reservations require lease-free preparation")
                publisher.prepare_flush(reference_seal=reference_seal)
            state.prepared = True
            flush_page()

    def stream_classification(self) -> ArtifactStreamClassification | None:
        """Read the complete original-input proof from this sealed artifact."""
        if self.sessions_path is None:
            return None
        self.verify_files(full=False)
        with _prepared_reader(self.sessions_path) as connection:
            row = connection.execute(
                "SELECT provider, kind, parse_as_session, schema_eligible, priority, reason, "
                "proved_non_session, record_count FROM prepared_classification"
            ).fetchone()
        if row is None:
            return None
        return ArtifactStreamClassification(
            ArtifactClassification(
                provider=Provider.from_string(row[0]),
                kind=ArtifactKind(row[1]),
                parse_as_session=bool(row[2]),
                schema_eligible=bool(row[3]),
                default_priority=int(row[4]),
                reason=str(row[5]),
            ),
            proved_non_session=bool(row[6]),
            record_count=int(row[7]),
        )

    def session_sequence(self) -> PreparedSessionSequence:
        """Expose a sealed cohort without retaining its parsed sessions in Python."""
        if self.sessions_path is None:
            raise RuntimeError(self.error or "JSONL preparation has no sealed artifact")
        self.verify_files(full=False)
        with _prepared_reader(self.sessions_path) as conn:
            seal = conn.execute(
                "SELECT version, source_hash, session_count, enrichment_digest, enrichment_index_path "
                "FROM artifact_seal"
            ).fetchall()
            if len(seal) != 1 or (
                seal[0][0],
                seal[0][1],
                seal[0][3],
                seal[0][4],
            ) != (
                _ARTIFACT_VERSION,
                self.blob_hash,
                self.enrichment_digest,
                self.enrichment_index_path,
            ):
                raise ValueError("JSONL preparation seal or source dependency changed")
            count = int(seal[0][2])
            actual_count = int(conn.execute("SELECT COUNT(*) FROM prepared_session").fetchone()[0])
        if actual_count != count:
            raise ValueError("JSONL preparation session count changed")
        return PreparedSessionSequence(self, count)

    def session_by_id(self, session_id: str, *, _shard: SessionShard | None = None) -> ParsedSession:
        """Read exactly one identity; ambiguous parser outputs remain in the ordinal cohort."""
        if self.sessions_path is None:
            raise RuntimeError(self.error or "JSONL preparation has no sealed artifact")
        self.verify_files(full=False)
        with _prepared_reader(self.sessions_path) as conn:
            rows = conn.execute(
                "SELECT ordinal FROM prepared_session WHERE session_id = ? LIMIT 2", (session_id,)
            ).fetchall()
        if len(rows) != 1:
            raise KeyError(session_id)
        return self._session_by_ordinal(int(rows[0][0]), _shard=_shard)

    def _session_by_ordinal(self, ordinal: int, *, _shard: SessionShard | None = None) -> ParsedSession:
        """Read the original parser output and shard range at the same ordinal."""
        if self.sessions_path is None or self.blob_hash is None or self.shard_path is None:
            raise RuntimeError(self.error or "JSONL preparation has no sealed artifact")
        self.verify_files(full=False)
        shard = _shard if _shard is not None else open_session_shard(self.shard_path)
        with _prepared_reader(self.sessions_path) as conn:
            seal = conn.execute(
                "SELECT version, source_hash, session_count, enrichment_digest, enrichment_index_path "
                "FROM artifact_seal"
            ).fetchall()
            if seal != [
                (
                    _ARTIFACT_VERSION,
                    self.blob_hash,
                    len(shard.sessions),
                    self.enrichment_digest,
                    self.enrichment_index_path,
                )
            ]:
                raise ValueError("JSONL preparation seal or source dependency changed")
            rows = conn.execute(
                "SELECT session_id, metadata_json, message_ordinal, message_count, event_ordinal, event_count, attachment_ordinal, attachment_count "
                "FROM prepared_session WHERE ordinal = ?",
                (ordinal,),
            ).fetchall()
            if len(rows) != 1:
                raise IndexError(ordinal)
            (
                stored_id,
                metadata_json,
                message_ordinal,
                message_count,
                event_ordinal,
                event_count,
                attachment_ordinal,
                attachment_count,
            ) = rows[0]
            shard_entry = shard.sessions[ordinal]
            if shard_entry.session_id != stored_id:
                raise ValueError("JSONL preparation session identity disagrees with row shard")
            physical_count = conn.execute(
                "SELECT COUNT(*) FROM prepared_message WHERE session_ordinal = ?", (message_ordinal,)
            ).fetchone()[0]
            if physical_count != message_count or physical_count != shard_entry.message_row_count:
                raise ValueError("JSONL preparation message count disagrees with row shard")
            physical_events = conn.execute(
                "SELECT COUNT(*) FROM prepared_event WHERE session_ordinal = ?", (event_ordinal,)
            ).fetchone()[0]
            if physical_events != event_count:
                raise ValueError("JSONL preparation event count changed")
            physical_attachments = conn.execute(
                "SELECT COUNT(*) FROM prepared_attachment WHERE session_ordinal = ?", (attachment_ordinal,)
            ).fetchone()[0]
            if physical_attachments != attachment_count:
                raise ValueError("JSONL preparation attachment count changed")
            metadata = json.loads(metadata_json)
            metadata["messages"] = []
            metadata["session_events"] = []
            metadata["attachments"] = []
            _restore_spilled_accounting(metadata, self.sessions_path)
            session = ParsedSession.model_validate(metadata)
            return session.model_copy(
                update={
                    "messages": SqliteMessageSink(self.sessions_path, message_ordinal, count=message_count),
                    "session_events": SqliteSessionEventSink(self.sessions_path, event_ordinal, count=event_count),
                    "attachments": SqliteAttachmentSink(self.sessions_path, attachment_ordinal, count=attachment_count),
                }
            )


def complete_thread_projection_cohort(
    artifacts: Iterable[PreparedJsonl], seal: PreparedIndexMutation, *, source_read: SessionSourceRead
) -> None:
    """Bind selected state artifacts to one complete original graph postimage."""
    from polylogue.sources.codex_state_projection import prepare_thread_state_cohort

    projections: list[PreparedThreadStateProjection] = []
    for artifact in artifacts:
        if artifact.codex_state_kind != "thread_state" or artifact.error is not None:
            continue
        state = artifact._thread_projection
        if state.seal is not seal or state.index_available is None or state.applied:
            raise RuntimeError("thread cohort differs from its original prepared artifact binding")
        if not state.index_available:
            continue
        if state.projection is None or state.projection.cohort is not None:
            raise RuntimeError("thread cohort input is unavailable or already completed")
        projections.append(state.projection)
    prepare_thread_state_cohort(tuple(projections), source_read=source_read)


class _PreparedAttachmentBlobs(Mapping[object, tuple[bytes | None, int, str]]):
    """A writer-local view, never a transferred SQL handle or population map."""

    def __init__(self, artifact: PreparedJsonl, source_read: BlobPublicationSourceRead, ordinal: int) -> None:
        self.artifact = artifact
        self.source_read = source_read
        self.ordinal = ordinal

    def __len__(self) -> int:
        return sum(1 for _key in self)

    def __iter__(self) -> Iterator[object]:
        from polylogue.sources.prepared_message_sink import _decode_attachment

        if self.artifact.sessions_path is None:
            raise ValueError("prepared artifact has no attachment carrier")
        self.artifact.verify_files(full=False)
        with (
            _prepared_reader(self.artifact.sessions_path) as connection,
            connection_cursor(
                connection,
                "SELECT a.attachment_ordinal,a.attachment_json,p.claim_json "
                "FROM prepared_attachment a LEFT JOIN prepared_attachment_publication p "
                "ON p.session_ordinal=a.session_ordinal AND p.attachment_ordinal=a.attachment_ordinal "
                "WHERE a.session_ordinal=? ORDER BY a.attachment_ordinal",
                (self.ordinal,),
            ) as cursor,
        ):
            while rows := cursor.fetchmany(256):
                check_compute_cancelled()
                for ordinal, attachment_json, claim_json in rows:
                    if claim_json is None:
                        attachment = _decode_attachment(
                            str(attachment_json), self.artifact.sessions_path, self.ordinal, int(ordinal)
                        )
                        if attachment.precomputed_blob is None or not self.source_read.publication_blob_is_excised(
                            bytes.fromhex(attachment.precomputed_blob[0])
                        ):
                            continue
                    yield str(self.artifact.sessions_path), self.ordinal, int(ordinal)

    def __getitem__(self, key: object) -> tuple[bytes | None, int, str]:
        if (
            not isinstance(key, tuple)
            or len(key) != 3
            or key[0] != str(self.artifact.sessions_path)
            or not isinstance(key[1], int)
            or isinstance(key[1], bool)
            or key[1] != self.ordinal
            or not isinstance(key[2], int)
            or isinstance(key[2], bool)
        ):
            raise KeyError(key)
        self.artifact.verify_files(full=False)
        if self.artifact.sessions_path is None or self.artifact.publication_publisher is None:
            raise ValueError("prepared attachments have no captured publisher")
        with (
            _prepared_reader(self.artifact.sessions_path) as connection,
            connection_cursor(
                connection,
                "SELECT claim_json FROM prepared_attachment_publication WHERE session_ordinal = ? AND attachment_ordinal = ?",
                (key[1], key[2]),
            ) as cursor,
        ):
            row = cursor.fetchone()
        if row is None:
            # An originally excised precomputed hash was deliberately not
            # captured or reserved. Its retained attachment and same Source
            # witness still carry the exact unavailable result.
            from polylogue.sources.prepared_message_sink import _decode_attachment

            with (
                _prepared_reader(self.artifact.sessions_path) as connection,
                connection_cursor(
                    connection,
                    "SELECT attachment_json FROM prepared_attachment WHERE session_ordinal=? AND attachment_ordinal=?",
                    (key[1], key[2]),
                ) as cursor,
            ):
                attachment_row = cursor.fetchone()
            if attachment_row is None:
                raise KeyError(key)
            attachment = _decode_attachment(str(attachment_row[0]), self.artifact.sessions_path, key[1], key[2])
            if attachment.precomputed_blob is None:
                raise KeyError(key)
            expected_hash, expected_size = attachment.precomputed_blob
            if not self.source_read.publication_blob_is_excised(bytes.fromhex(expected_hash)):
                raise KeyError(key)
            return None, expected_size, "unavailable"
        claim = _prepared_claim_from_record(str(row[0]), self.artifact.publication_publisher)
        blob_hash = bytes.fromhex(claim.receipt.blob_hash)
        if self.source_read.publication_blob_is_excised(blob_hash):
            return None, claim.receipt.size_bytes, "unavailable"
        self._validate_claim(key, claim)
        return blob_hash, claim.receipt.size_bytes, "acquired"

    def _validate_claim(self, key: tuple[object, ...], claim: PreparedBlobPublicationClaim) -> None:
        claim.publisher.validate_published_claim(self.source_read, claim, source_path="")


class _ResidentAttachmentBlobs(_PreparedAttachmentBlobs):
    """Current Raw-bound durable proof on the same borrowed artifact and Source reader."""

    def __init__(
        self, artifact: PreparedJsonl, source_read: RetainedAttachmentSourceRead, ordinal: int, raw_id: str
    ) -> None:
        super().__init__(artifact, source_read, ordinal)
        self.retained_read = source_read
        self.raw_id = raw_id

    def _validate_claim(self, key: tuple[object, ...], claim: PreparedBlobPublicationClaim) -> None:
        from polylogue.core.identity_law import attachment_acquisition_coordinate

        if self.artifact.sessions_path is None:
            raise ValueError("resident attachment carrier is absent")
        if self.retained_read.publication_source_path() != claim.publisher.source_db_path.resolve():
            raise ValueError("resident attachment belongs to another Source database")
        with (
            _prepared_reader(self.artifact.sessions_path) as connection,
            connection_cursor(
                connection,
                "SELECT json_extract(attachment_json,'$.provider_file_id'),"
                "json_extract(attachment_json,'$.provider_attachment_id') FROM prepared_attachment "
                "WHERE session_ordinal=? AND attachment_ordinal=?",
                (key[1], key[2]),
            ) as cursor,
        ):
            coordinate = cursor.fetchone()
        if coordinate is None:
            raise ValueError("resident attachment coordinate is absent")
        file_id, attachment_id = coordinate
        if (file_id is not None and not isinstance(file_id, str)) or not isinstance(attachment_id, str):
            raise ValueError("resident attachment coordinate is not a declared provider identity")
        if self.artifact.blob_hash is None:
            raise ValueError("resident attachment lacks its original Raw payload binding")
        self.retained_read.retained_attachment_reference(
            self.raw_id,
            bytes.fromhex(self.artifact.blob_hash),
            attachment_acquisition_coordinate(file_id, attachment_id),
            bytes.fromhex(claim.receipt.blob_hash),
            claim.receipt.size_bytes,
        )
        from polylogue.core.storage_faults import ArchiveStorageFaultError, StorageFaultKind

        try:
            info = claim.publisher._store.blob_path(claim.receipt.blob_hash).lstat()
        except OSError as failure:
            raise ArchiveStorageFaultError(StorageFaultKind.EVICTED, failure) from failure
        if not stat.S_ISREG(info.st_mode) or info.st_size != claim.receipt.size_bytes:
            raise ArchiveStorageFaultError(
                StorageFaultKind.EVICTED, FileNotFoundError("resident attachment bytes are absent or changed")
            )


class PreparedSidecarLocators(Mapping[str, Mapping[str, str]]):
    """Read the captured sidecar claim for one session without a tool map."""

    def __init__(self, artifact: PreparedJsonl, ordinal: int, source_read: BlobPublicationSourceRead) -> None:
        self.artifact = artifact
        self.ordinal = ordinal
        self.source_read = source_read

    def __iter__(self) -> Iterator[str]:
        for _ordinal, tool_use_id, _claim, _present in self.artifact.iter_sidecar_claims(session_ordinal=self.ordinal):
            yield tool_use_id

    def __len__(self) -> int:
        if self.artifact.sessions_path is None:
            return 0
        self.artifact.verify_files(full=False)
        with _prepared_reader(self.artifact.sessions_path) as connection:
            with connection_cursor(
                connection,
                "SELECT COUNT(*) FROM prepared_sidecar_publication WHERE session_ordinal=? AND claim_json IS NOT NULL",
                (self.ordinal,),
            ) as cursor:
                row = cursor.fetchone()
            assert row is not None
            return int(row[0])

    def __getitem__(self, tool_use_id: str) -> Mapping[str, str]:
        if self.artifact.sessions_path is None or self.artifact.publication_publisher is None:
            raise KeyError(tool_use_id)
        self.artifact.verify_files(full=False)
        with (
            _prepared_reader(self.artifact.sessions_path) as connection,
            connection_cursor(
                connection,
                "SELECT claim_json FROM prepared_sidecar_publication WHERE session_ordinal=? AND tool_use_id=?",
                (self.ordinal, tool_use_id),
            ) as cursor,
        ):
            row = cursor.fetchone()
        if row is None or row[0] is None:
            raise KeyError(tool_use_id)
        claim = _prepared_claim_from_record(str(row[0]), self.artifact.publication_publisher)
        if self.source_read.publication_blob_is_excised(bytes.fromhex(claim.receipt.blob_hash)):
            return {"blob_refusal": "content_excised"}
        claim.publisher.validate_published_claim(self.source_read, claim, source_path="")
        return {"blob_hash": claim.receipt.blob_hash}

    def publication_counts(self) -> dict[str, int]:
        counts = {
            "sidecar_blob_bytes_new": 0,
            "sidecar_blob_bytes_dedup": 0,
            "sidecar_blobs_written": 0,
            "sidecar_blobs_refused_excised": 0,
        }
        for _ordinal, tool_use_id, claim, present in self.artifact.iter_sidecar_claims(session_ordinal=self.ordinal):
            if self[tool_use_id].get("blob_refusal"):
                counts["sidecar_blobs_refused_excised"] += 1
            else:
                counts["sidecar_blobs_written"] += 1
                counts["sidecar_blob_bytes_dedup" if present else "sidecar_blob_bytes_new"] += claim.receipt.size_bytes
        return counts


class PreparedSessionSequence(Sequence[ParsedSession]):
    """A reusable session view over the sealed worker artifact."""

    def __init__(self, artifact: PreparedJsonl, count: int) -> None:
        self.artifact = artifact
        self._count = count
        if artifact.shard_path is None:
            raise RuntimeError(artifact.error or "JSONL preparation has no row shard")
        self._shard = open_session_shard(artifact.shard_path)
        if len(self._shard.sessions) != count:
            raise ValueError("JSONL preparation session count disagrees with row shard")

    def __len__(self) -> int:
        return self._count

    def by_session_id(self, session_id: str) -> ParsedSession:
        return self.artifact.session_by_id(session_id, _shard=self._shard)

    def iter_provider_session_ids(self) -> Iterator[str]:
        """Read the complete original native-ID scope without hydrating transcripts."""
        return self.artifact.iter_provider_session_ids()

    def iter_session_ids(self) -> Iterator[str]:
        """Stream every original output identity, including repeated identities."""
        if self.artifact.sessions_path is None:
            raise RuntimeError(self.artifact.error or "JSONL preparation has no sealed artifact")
        self.artifact.verify_files(full=False)
        after = -1
        while True:
            check_compute_cancelled()
            with _prepared_reader(self.artifact.sessions_path) as connection:
                rows = connection.execute(
                    "SELECT ordinal, session_id FROM prepared_session WHERE ordinal > ? ORDER BY ordinal LIMIT 512",
                    (after,),
                ).fetchall()
            if not rows:
                return
            after = int(rows[-1][0])
            for _ordinal, session_id in rows:
                yield str(session_id)

    def __iter__(self) -> Iterator[ParsedSession]:
        return self.artifact.iter_sessions()

    @overload
    def __getitem__(self, index: int) -> ParsedSession: ...

    @overload
    def __getitem__(self, index: slice) -> list[ParsedSession]: ...

    def __getitem__(self, index: int | slice) -> ParsedSession | list[ParsedSession]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._count))]
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError(index)
        if self.artifact.sessions_path is None:
            raise RuntimeError(self.artifact.error or "JSONL preparation has no sealed artifact")
        self.artifact.verify_files(full=False)
        return self.artifact._session_by_ordinal(index, _shard=self._shard)


def record_prepared_classification(conn: sqlite3.Connection, taxonomy: ArtifactStreamClassification) -> None:
    """Seal the complete original-input classification beside the artifact's sessions."""
    classification = taxonomy.classification
    conn.execute(
        "INSERT INTO prepared_classification VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            classification.provider.value,
            classification.kind.value,
            int(classification.parse_as_session),
            int(classification.schema_eligible),
            classification.default_priority,
            classification.reason,
            int(taxonomy.proved_non_session),
            taxonomy.record_count,
        ),
    )


def _write_artifact(
    store: SqliteMessageStore,
    source_hash: str,
    sessions: Iterable[ParsedSession],
    *,
    enrichment_digest: str | None,
    enrichment_index_path: str | None,
    codex_state_path: Path | None = None,
    codex_state_kind: str | None = None,
    codex_state_text_chars: int = codex_state.CODEX_STATE_MAX_TEXT_CHARS,
    codex_semantic_source_path: str | None = None,
    codex_material_store: ArchiveBlobPublisher | None = None,
    codex_material_directory: Path | None = None,
) -> None:
    conn = store.conn
    try:
        _create_artifact_tables(conn)
        count = 0
        for ordinal, session in enumerate(sessions):
            _append_artifact_session(store, ordinal, session)
            count += 1
        if codex_state_kind is not None:
            if codex_state_path is None:
                raise ValueError("prepared state requires retained export path")
            conn.execute("INSERT INTO prepared_codex_state VALUES (?)", (codex_state_kind,))
            if codex_state_kind == "thread_state":
                from dataclasses import asdict

                thread_ordinal = 0
                spawn_ordinal = 0
                for record in codex_state.iter_codex_state_records(codex_state_path, immutable=True):
                    check_compute_cancelled()
                    if isinstance(record, codex_state.CodexThreadRecord):
                        conn.execute(
                            "INSERT INTO prepared_codex_thread VALUES (?, ?)",
                            (thread_ordinal, json.dumps(asdict(record), ensure_ascii=False)),
                        )
                        thread_ordinal += 1
                    else:
                        conn.execute(
                            "INSERT INTO prepared_codex_spawn VALUES (?, ?)",
                            (spawn_ordinal, json.dumps(asdict(record), ensure_ascii=False)),
                        )
                        spawn_ordinal += 1
            elif codex_state_kind in {"goals", "memories"}:
                from polylogue.sources.codex_state_evidence import _encode_state_payload, codex_material_coordinate
                from polylogue.storage.materials import _prepared_material_record, prepare_material

                if (
                    codex_semantic_source_path is None
                    or codex_material_store is None
                    or codex_material_directory is None
                ):
                    raise ValueError("state material requires captured scope and owned blob staging")

                for ordinal, part in enumerate(
                    codex_state.iter_codex_state_parts(
                        codex_state_path,
                        state_kind=cast(Literal["goals", "memories"], codex_state_kind),
                        text_chars=codex_state_text_chars,
                        immutable=True,
                    )
                ):
                    check_compute_cancelled()
                    encoded = _encode_state_payload(part.payload)
                    prepared_record = None
                    if part.part_kind != "invalid":
                        material_kind = codex_state_kind if part.part_kind == "record" else f"{codex_state_kind}-text"
                        source_uri, referrer_ref = codex_material_coordinate(
                            codex_semantic_source_path,
                            part.thread_id,
                            material_kind,
                            part.item_id,
                        )
                        prepared = prepare_material(
                            blob_store=codex_material_store,
                            staging_directory=codex_material_directory,
                            source_uri=source_uri,
                            referrer_ref=referrer_ref,
                            payload=encoded,
                            media_type="application/json",
                            filename=f"{material_kind}-{part.item_id}.json",
                            privacy_classification="private",
                        )
                        prepared_record = _prepared_material_record(prepared)
                    conn.execute(
                        "INSERT INTO prepared_codex_state_part VALUES (?, ?, ?, ?, ?, ?)",
                        (ordinal, part.thread_id, part.item_id, part.part_kind, len(encoded), prepared_record),
                    )
            else:
                raise ValueError("unsupported prepared Codex state kind")
        _seal_artifact(conn, source_hash, count, enrichment_digest, enrichment_index_path)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def _create_artifact_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE prepared_sidecar_publication (session_ordinal INTEGER NOT NULL, tool_use_id TEXT NOT NULL, "
        "claim_json TEXT, captured_text TEXT, already_present INTEGER NOT NULL DEFAULT 0, "
        "PRIMARY KEY(session_ordinal,tool_use_id)) WITHOUT ROWID"
    )
    conn.execute(
        "CREATE TABLE prepared_attachment_publication (session_ordinal INTEGER NOT NULL, "
        "attachment_ordinal INTEGER NOT NULL, claim_json TEXT NOT NULL, "
        "PRIMARY KEY(session_ordinal, attachment_ordinal)) WITHOUT ROWID"
    )
    conn.execute(
        "CREATE TABLE prepared_classification (provider TEXT NOT NULL, kind TEXT NOT NULL, "
        "parse_as_session INTEGER NOT NULL, schema_eligible INTEGER NOT NULL, priority INTEGER NOT NULL, "
        "reason TEXT NOT NULL, proved_non_session INTEGER NOT NULL, record_count INTEGER NOT NULL)"
    )
    conn.execute("CREATE TABLE prepared_codex_state (kind TEXT NOT NULL)")
    conn.execute("CREATE TABLE prepared_codex_thread (ordinal INTEGER PRIMARY KEY, metadata_json TEXT NOT NULL)")
    conn.execute("CREATE TABLE prepared_codex_spawn (ordinal INTEGER PRIMARY KEY, metadata_json TEXT NOT NULL)")
    conn.execute(
        "CREATE TABLE prepared_codex_state_part (ordinal INTEGER PRIMARY KEY, thread_id TEXT NOT NULL, item_id TEXT NOT NULL, part_kind TEXT NOT NULL, byte_size INTEGER NOT NULL, prepared_json TEXT)"
    )
    conn.execute(
        "CREATE TABLE prepared_session (ordinal INTEGER PRIMARY KEY, session_id TEXT NOT NULL, metadata_json TEXT NOT NULL, message_ordinal INTEGER NOT NULL, message_count INTEGER NOT NULL, event_ordinal INTEGER NOT NULL, event_count INTEGER NOT NULL, attachment_ordinal INTEGER NOT NULL, attachment_count INTEGER NOT NULL)"
    )
    conn.execute("CREATE INDEX prepared_session_identity ON prepared_session(session_id, ordinal)")
    conn.execute(
        "CREATE TABLE artifact_seal (version INTEGER NOT NULL, source_hash TEXT NOT NULL, "
        "session_count INTEGER NOT NULL, enrichment_digest TEXT, enrichment_index_path TEXT)"
    )


def _artifact_source_hash(store: SqliteMessageStore) -> str:
    """The sealed retained input these publications belong to."""
    row = store.conn.execute("SELECT source_hash FROM artifact_seal").fetchone()
    if row is None or not row[0]:
        raise ValueError("publication claims require the artifact's sealed source identity")
    return str(row[0])


def _prepare_attachment_publications(
    store: SqliteMessageStore,
    publisher: ArchiveBlobPublisher,
    directory: Path,
    *,
    source_read: BlobPublicationSourceRead | None = None,
) -> None:
    """Seal attachment claims on the existing artifact before writer admission."""
    from polylogue.sources.prepared_message_sink import _decode_attachment
    from polylogue.storage.blob_publication import AdoptedBlobEvictedError, _prepared_claim_record

    source_hash = _artifact_source_hash(store)
    after = (-1, -1)
    while True:
        check_compute_cancelled()
        rows = store.conn.execute(
            "SELECT session_ordinal, attachment_ordinal, attachment_json FROM prepared_attachment "
            "WHERE (session_ordinal, attachment_ordinal) > (?, ?) "
            "ORDER BY session_ordinal, attachment_ordinal LIMIT 256",
            after,
        ).fetchall()
        if not rows:
            break
        for session_ordinal, attachment_ordinal, attachment_json in rows:
            check_compute_cancelled()
            attachment = _decode_attachment(attachment_json, store.path, session_ordinal, attachment_ordinal)
            if attachment.inline_bytes is not None:
                blob = publisher.prepare_from_bytes(attachment.inline_bytes, staging_directory=directory)
            elif attachment.precomputed_blob is not None:
                expected_hash, expected_size = attachment.precomputed_blob
                # A refused original hash needs no private byte capture. The
                # same witnessed Source evidence still governs its unavailable
                # attachment row when the session is prepared and published.
                if source_read is not None:
                    if source_read.publication_source_path() != publisher.source_db_path.resolve():
                        raise ValueError("attachment preparation belongs to another Source database")
                    if source_read.publication_blob_is_excised(bytes.fromhex(expected_hash)):
                        continue
                try:
                    blob = publisher.prepare_from_path(
                        publisher.blob_path(expected_hash),
                        staging_directory=directory,
                        heartbeat=check_compute_cancelled,
                    )
                except FileNotFoundError as failure:
                    # Absent bytes are a storage fault only once the excision
                    # ledger says they were not removed on purpose; excised
                    # bytes leave the attachment to its unavailable outcome.
                    if publisher.excised_now(expected_hash):
                        continue
                    raise AdoptedBlobEvictedError((expected_hash,)) from failure
                if (blob.hash_hex, blob.size_bytes) != (expected_hash, expected_size):
                    publisher.discard_prepared(blob)
                    raise ValueError("prepared attachment disagrees with its retained blob")
            else:
                continue
            try:
                claim = publisher.prepare_claim(
                    blob,
                    coordinate=f"{source_hash}/attachment/{int(session_ordinal)}/{int(attachment_ordinal)}",
                )
                store.conn.execute(
                    "INSERT INTO prepared_attachment_publication VALUES (?, ?, ?)",
                    (session_ordinal, attachment_ordinal, _prepared_claim_record(claim)),
                )
            except BaseException as primary:
                try:
                    publisher.discard_prepared(blob)
                except BaseException as cleanup:
                    primary.add_note(f"attachment preparation cleanup failed: {cleanup!r}")
                raise
        after = (int(rows[-1][0]), int(rows[-1][1]))
    store.conn.commit()


def _prepare_sidecar_publications(store: SqliteMessageStore, publisher: ArchiveBlobPublisher, directory: Path) -> None:
    """Capture matched tool-result bytes on the existing sealed artifact."""
    import unicodedata

    from polylogue.pipeline.ids import SIDECAR_BLOB_EVENT_TYPES
    from polylogue.storage.blob_publication import _prepared_claim_record

    source_hash = _artifact_source_hash(store)
    event_marks = ",".join("?" for _ in SIDECAR_BLOB_EVENT_TYPES)
    store.conn.execute(
        "INSERT OR IGNORE INTO prepared_sidecar_publication(session_ordinal,tool_use_id) "
        "SELECT s.ordinal,json_extract(e.event_json,'$.payload.tool_use_id') "
        "FROM prepared_session s JOIN prepared_event e ON e.session_ordinal=s.event_ordinal "
        f"WHERE e.event_type IN ({event_marks}) "
        "AND json_extract(e.event_json,'$.payload.acquisition_status')='matched' "
        "AND json_extract(e.event_json,'$.payload.content_replaced') "
        "AND json_type(e.event_json,'$.payload.tool_use_id')='text'",
        tuple(SIDECAR_BLOB_EVENT_TYPES),
    )
    # Resolve the parser's final matching block once in parser order. The
    # exact sidecar row is the disk-backed owner of this temporary evidence.
    blocks = store.conn.execute(
        "SELECT s.ordinal,p.tool_use_id,json_extract(b.value,'$.text') "
        "FROM prepared_session s JOIN prepared_message m ON m.session_ordinal=s.message_ordinal "
        "JOIN json_each(m.message_json,'$.blocks') b "
        "JOIN prepared_sidecar_publication p ON p.session_ordinal=s.ordinal "
        "AND p.tool_use_id=json_extract(b.value,'$.tool_id') "
        "WHERE json_extract(b.value,'$.type')='tool_result' AND json_type(b.value,'$.text')='text' "
        "ORDER BY s.ordinal,m.message_ordinal,CAST(b.key AS INTEGER)"
    )
    try:
        while True:
            check_compute_cancelled()
            row = blocks.fetchone()
            if row is None:
                break
            ordinal, tool_use_id, text = row
            store.conn.execute(
                "UPDATE prepared_sidecar_publication SET captured_text=? WHERE session_ordinal=? AND tool_use_id=?",
                (text, ordinal, tool_use_id),
            )
            del row, text
    finally:
        blocks.close()
    after = (-1, "")
    while True:
        check_compute_cancelled()
        rows = store.conn.execute(
            "SELECT session_ordinal,tool_use_id FROM prepared_sidecar_publication "
            "WHERE captured_text IS NOT NULL AND (session_ordinal,tool_use_id)>(?,?) ORDER BY session_ordinal,tool_use_id LIMIT 128",
            after,
        ).fetchall()
        if not rows:
            break
        for ordinal, tool_use_id in rows:
            check_compute_cancelled()
            # The page carries only coordinates. Variable-size prose is read
            # and released one value at a time, never accumulated by row count.
            text_row = store.conn.execute(
                "SELECT captured_text FROM prepared_sidecar_publication WHERE session_ordinal=? AND tool_use_id=?",
                (ordinal, tool_use_id),
            ).fetchone()
            if text_row is None or text_row[0] is None:
                raise ValueError("captured sidecar content disappeared during preparation")
            text = text_row[0]
            blob = publisher.prepare_from_bytes(
                unicodedata.normalize("NFC", str(text)).encode("utf-8"), staging_directory=directory
            )
            del text_row, text
            claim = publisher.prepare_claim(blob, coordinate=f"{source_hash}/sidecar/{int(ordinal)}/{tool_use_id}")
            store.conn.execute(
                "UPDATE prepared_sidecar_publication SET claim_json=?,already_present=?,captured_text=NULL "
                "WHERE session_ordinal=? AND tool_use_id=?",
                (_prepared_claim_record(claim), int(publisher.exists(blob.hash_hex)), ordinal, tool_use_id),
            )
        after = int(rows[-1][0]), str(rows[-1][1])
    store.conn.commit()


def _restore_spilled_accounting(metadata: dict[str, object], sessions_path: Path) -> None:
    accounting = metadata.get("unit_accounting")
    if not isinstance(accounting, dict):
        return
    outcomes = accounting.get("outcomes")
    if not isinstance(outcomes, dict) or "$polylogue_spilled_outcomes" not in outcomes:
        return
    from polylogue.sources.parsers.base_models import ParseAccounting

    metadata["unit_accounting"] = ParseAccounting.from_prepared_payload(accounting, sessions_path)


def _append_artifact_session(store: SqliteMessageStore, ordinal: int, session: ParsedSession) -> None:
    conn = store.conn
    source_messages: object = session.messages
    messages: SqliteMessageSink
    if isinstance(source_messages, SqliteMessageSink) and source_messages.path == store.path:
        messages = source_messages
    else:
        messages = store.new_sink()
        messages.extend(session.messages)
    messages.normalized_messages(session.session_events, origin=origin_from_provider(session.source_name))
    source_events: object = session.session_events
    events: SqliteSessionEventSink
    if isinstance(source_events, SqliteSessionEventSink) and source_events.path == store.path:
        events = source_events
    else:
        events = store.new_event_sink()
        events.extend(session.session_events)
    source_attachments: object = session.attachments
    attachments: SqliteAttachmentSink
    if isinstance(source_attachments, SqliteAttachmentSink) and source_attachments.path == store.path:
        attachments = source_attachments
    else:
        attachments = store.new_attachment_sink()
        attachments.extend(session.attachments)
    metadata = session.model_dump(mode="json", exclude={"messages", "session_events", "attachments"})
    metadata["content_hash"] = session.content_hash
    metadata["enrichment_evidence_key"] = session.enrichment_evidence_key
    metadata["unit_accounting"] = (
        session.unit_accounting.to_prepared_payload() if session.unit_accounting is not None else None
    )
    metadata["provider_session_aliases"] = session.provider_session_aliases
    metadata["created_at_provenance"] = session.created_at_provenance
    metadata["updated_at_provenance"] = session.updated_at_provenance
    session_id = archive_session_id(origin_from_provider(session.source_name).value, session.provider_session_id)
    conn.execute(
        "INSERT INTO prepared_session VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            ordinal,
            session_id,
            json.dumps(metadata, ensure_ascii=False),
            messages.session_ordinal,
            len(messages),
            events.session_ordinal,
            len(events),
            attachments.session_ordinal,
            len(attachments),
        ),
    )


def _seal_artifact(
    conn: sqlite3.Connection,
    source_hash: str,
    count: int,
    enrichment_digest: str | None,
    enrichment_index_path: str | None,
) -> None:
    # Retain only the original and normalized operands of admitted sessions.
    # The normalized identity mapping lets canonical consumers borrow an
    # already normalized operand without deriving it a second time.
    conn.execute(
        "DELETE FROM prepared_message_normalization WHERE original_ordinal NOT IN "
        "(SELECT message_ordinal FROM prepared_session UNION "
        "SELECT normalized_ordinal FROM prepared_message_normalization WHERE original_ordinal IN "
        "(SELECT message_ordinal FROM prepared_session))"
    )
    conn.execute(
        "DELETE FROM prepared_message WHERE session_ordinal NOT IN "
        "(SELECT message_ordinal FROM prepared_session UNION "
        "SELECT normalized_ordinal FROM prepared_message_normalization)"
    )
    conn.execute(
        "INSERT INTO artifact_seal VALUES (?, ?, ?, ?, ?)",
        (_ARTIFACT_VERSION, source_hash, count, enrichment_digest, enrichment_index_path),
    )


def _prepare_codex_state_blob(
    source: Path,
    directory: Path,
    *,
    state_kind: str,
    source_hash: str,
    semantic_source_path: str,
    enrichment_digest: str | None = None,
    enrichment_index_path: str | None = None,
    attempt_directory: Path | None = None,
    text_chars: int = codex_state.CODEX_STATE_MAX_TEXT_CHARS,
    publication_publisher: ArchiveBlobPublisher | None = None,
    captured_profile_key: str | None = None,
) -> PreparedJsonl:
    """Seal the non-session branch of the existing canonical artifact."""
    sessions_path = directory / f"prepared-{uuid.uuid4().hex}.db"
    shard_path: Path | None = None
    store: SqliteMessageStore | None = None
    sealed = False
    from polylogue.core.sql_settlement import retain_native_sql_lifetimes

    with retain_native_sql_lifetimes(directory):
        try:
            staging_root = next((parent for parent in directory.parents if parent.name == ".staging"), None)
            if staging_root is None:
                raise ValueError("state material preparation requires archive-owned blob staging")
            material_store = publication_publisher or ArchiveBlobPublisher(
                staging_root.parent.parent / "source.db", staging_root.parent
            )
            if material_store.root.resolve() != staging_root.parent.resolve():
                raise ValueError("prepared state publisher belongs to another blob root")
            store = SqliteMessageStore(sessions_path)
            _write_artifact(
                store,
                source_hash,
                (),
                enrichment_digest=enrichment_digest,
                enrichment_index_path=enrichment_index_path,
                codex_state_path=source,
                codex_state_kind=state_kind,
                codex_state_text_chars=text_chars,
                codex_semantic_source_path=semantic_source_path,
                codex_material_store=material_store,
                codex_material_directory=directory,
            )
            shard_path = prepare_session_shard(directory, ()).path
            if file_digest(source) != source_hash:
                raise _SourceChangedDuringPreparationError("retained state changed during preparation")
            store.close()
            store = None
            artifact = PreparedJsonl.seal(
                source_hash,
                sessions_path,
                shard_path,
                enrichment_digest=enrichment_digest,
                enrichment_index_path=enrichment_index_path,
                resolved_provider=Provider.CODEX,
                positive_evidence_filtered=True,
                attempt_directory=attempt_directory,
                codex_state_kind=state_kind,
                codex_state_text_chars=text_chars,
                publication_publisher=material_store,
                captured_profile_key=captured_profile_key,
            )
            sealed = True
            return artifact
        finally:
            primary = sys.exception()
            if store is not None:
                try:
                    store.close()
                except BaseException as cleanup:
                    if primary is not None:
                        raise BaseExceptionGroup(
                            "state artifact preparation and physical close failed", [primary, cleanup]
                        ) from None
                    raise
            if not sealed:
                sessions_path.unlink(missing_ok=True)
                if shard_path is not None:
                    discard_session_shard(shard_path)


def _empty_parsed_sessions() -> Generator[ParsedSession, None, None]:
    yield from ()


def _finalize_prepared_cohort(
    original: PreparedJsonl,
    finalize: Callable[[PreparedSessionSequence], Iterable[ParsedSession]],
    *,
    artifact_directory: Path,
    publication_publisher: ArchiveBlobPublisher | None,
    publication_source_read: BlobPublicationSourceRead | None,
    preparation_dependency: Callable[[], tuple[str | None, str | None]] | None,
    preserve_parser_stage: bool = True,
) -> PreparedJsonl:
    """Retain the full original parse through one bounded cohort interpretation."""
    if original.blob_hash is None:
        raise RuntimeError("cohort finalization requires the original acquired-byte binding")

    def selected() -> Generator[ParsedSession, None, None]:
        finalized: Iterable[ParsedSession] | None = None
        output: Iterator[ParsedSession] | None = None
        try:
            cohort = original.session_sequence()
            finalized = finalize(cohort)
            output = iter(finalized)
            for session in output:
                check_compute_cancelled()
                session.content_hash = session_content_hash(session)
                yield session
        finally:
            primary = sys.exception()
            failures: list[BaseException] = []
            closed: set[int] = set()
            for owned in (output, finalized):
                if owned is None or id(owned) in closed:
                    continue
                closed.add(id(owned))
                close = getattr(owned, "close", None)
                if close is not None:
                    try:
                        close()
                    except BaseException as cleanup:
                        failures.append(cleanup)
            if failures:
                if primary is not None:
                    failures.insert(0, primary)
                if len(failures) == 1:
                    raise failures[0]
                raise BaseExceptionGroup("cohort interpretation and iterator close failed", failures) from None

    with closing(selected()) as sessions:
        result = PreparedJsonl.from_sessions(
            sessions,
            blob_hash=original.blob_hash,
            artifact_directory=artifact_directory,
            publication_publisher=publication_publisher,
            publication_source_read=publication_source_read,
            classification=original.stream_classification(),
            enrichment_digest=original.enrichment_digest,
            enrichment_index_path=original.enrichment_index_path,
            parsed_prefix_size=original.parsed_prefix_size,
            resolved_provider=original.resolved_provider,
            captured_profile_key=original.captured_profile_key,
            preparation_dependency=preparation_dependency,
        )
    # Preserve the neutral parser-stage carrier for exact per-revision
    # checkpoints. The finalized carrier owns its shared attempt directory;
    # the child owns only its sealed file paths and is discarded with parent.
    if not preserve_parser_stage:
        return result
    return replace(result, parser_stage_artifact=replace(original, attempt_directory=None))


def _prepared_jsonl_productive_identity(
    blob_path: str,
    source_path: str,
    provider_value: str,
    fallback_id: str,
    *,
    is_stream: bool,
    profile_identity: str | None = None,
    shard_directory: str,
    sidecar_resolver: SidecarResolver | None = None,
    prepare_sessions: Callable[[PreparedSessionSequence], Iterable[ParsedSession]] | None = None,
    prepare_session: Callable[[ParsedSession], ParsedSession] | None = None,
    preparation_dependency: Callable[[], tuple[str | None, str | None]] | None = None,
    parse_prefix_size: int | None = None,
    attempt_directory: Path | None = None,
    source_sha256: str | None = None,
    strict_jsonl_records: bool = False,
    publication_publisher: ArchiveBlobPublisher | None = None,
    publication_source_read: BlobPublicationSourceRead | None = None,
    progress_identity: str | None = None,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None = None,
) -> str | None:
    """Bind parser progress to retained bytes and source operands, never scratch paths."""
    has_prepare_sessions = prepare_sessions is not None
    has_prepare_session = prepare_session is not None
    del blob_path, shard_directory, prepare_sessions, prepare_session, preparation_dependency
    del attempt_directory, publication_publisher, publication_source_read
    if progress_identity is not None:
        return progress_identity
    sidecar_signature: tuple[object, ...] | None = None
    if sidecar_resolver is not None and provider_value == "claude-code":
        scope = sidecar_resolver.claude_code_scope(source_path)
        sidecar_signature = (scope.scope_key, scope.available, scope.witness)
    zip_coordinate_identity = None
    if captured_zip_coordinate is not None:
        zip_coordinate_identity = (
            "captured-zip-coordinate",
            captured_zip_coordinate.canonical_container,
            captured_zip_coordinate.declared_container,
            captured_zip_coordinate.member_name,
            captured_zip_coordinate.entry_ordinal,
            captured_zip_coordinate.split_index,
            captured_zip_coordinate.addressing_mode.value,
            captured_zip_coordinate.container_blob_hash,
            captured_zip_coordinate.decoder_fingerprint,
            captured_zip_coordinate.profile_namespace,
        )
    recipe = (
        "prepared-jsonl",
        provider_value,
        source_sha256,
        source_path,
        zip_coordinate_identity,
        fallback_id,
        is_stream,
        profile_identity,
        parse_prefix_size,
        strict_jsonl_records,
        sidecar_signature,
        has_prepare_sessions,
        has_prepare_session,
    )
    return stable_productive_identity(recipe) if source_sha256 is not None else None


@reports_work_progress("source_preparation", productive_identity=_prepared_jsonl_productive_identity)
def prepare_jsonl_blob(
    blob_path: str,
    source_path: str,
    provider_value: str,
    fallback_id: str,
    *,
    is_stream: bool,
    profile_identity: str | None = None,
    shard_directory: str,
    sidecar_resolver: SidecarResolver | None = None,
    prepare_sessions: Callable[[PreparedSessionSequence], Iterable[ParsedSession]] | None = None,
    prepare_session: Callable[[ParsedSession], ParsedSession] | None = None,
    preparation_dependency: Callable[[], tuple[str | None, str | None]] | None = None,
    parse_prefix_size: int | None = None,
    attempt_directory: Path | None = None,
    source_sha256: str | None = None,
    strict_jsonl_records: bool = False,
    progress_identity: str | None = None,
    publication_publisher: ArchiveBlobPublisher | None = None,
    publication_source_read: BlobPublicationSourceRead | None = None,
    captured_zip_coordinate: CapturedZipMemberCoordinate | None = None,
) -> PreparedJsonl:
    """Parse and seal one source without transferring a parsed tree over IPC.

    At an OriginSpec ``fact`` path, streamed candidacy lets records carrying a
    provider session envelope reach the full parser; the accepted-session rule
    still owns recovery, and input that yields no accepted session keeps its
    original fact classification.

    Every sealed session is admitted: the positive-conversational-evidence
    rule (``admit_parsed_sessions_for_publication``) runs here, before any
    ``prepare_session``/``prepare_sessions`` callback, on every provider
    branch. A caller consuming the artifact does not apply it again.

    ``source_sha256`` is the digest of ``blob_path`` when the caller already
    read it to decide how to prepare it; the seal then binds that decision's
    bytes. ``strict_jsonl_records`` refuses a complete JSONL record that does
    not decode for every provider, as live acquisition does, instead of only
    for ``Provider.UNKNOWN``.
    """
    from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError
    from polylogue.storage.sqlite.reference_seal import ReferenceSealError

    directory = Path(shard_directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    artifact_directory = attempt_directory if attempt_directory is not None else directory
    if attempt_directory is not None:
        artifact_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if artifact_directory.parent != directory:
            raise ValueError("prepared attempt directory must be a direct child of the shard directory")
    sessions_path = artifact_directory / f"prepared-{uuid.uuid4().hex}.db"
    member_path = artifact_directory / f"member-{uuid.uuid4().hex}.json"
    shard_path: Path | None = None
    store: SqliteMessageStore | None = None
    shard_builder: SessionShardBuilder | None = None
    sealed = False
    source = Path(blob_path)
    before_hash: str | None = None
    try:
        provider = Provider.from_string(provider_value)
        if provider is Provider.CODEX and not is_stream and source_path.lower().endswith((".sqlite", ".db")):
            state_kind = codex_state.classify_codex_sqlite_path(source, immutable=True)
            if state_kind in codex_state.IN_SCOPE_KINDS:
                before_hash = source_sha256 if source_sha256 is not None else file_digest(source)
                enrichment_digest, enrichment_index_path = (
                    preparation_dependency() if preparation_dependency is not None else (None, None)
                )
                return _prepare_codex_state_blob(
                    source,
                    artifact_directory,
                    state_kind=state_kind,
                    source_hash=before_hash,
                    semantic_source_path=source_path,
                    enrichment_digest=enrichment_digest,
                    enrichment_index_path=enrichment_index_path,
                    attempt_directory=attempt_directory,
                    publication_publisher=publication_publisher,
                    captured_profile_key=profile_identity,
                )
        store = SqliteMessageStore(sessions_path)
        before_hash = source_sha256 if source_sha256 is not None else file_digest(source)
        from polylogue.sources.live.batch_support import jsonl_parse_input_of_handle

        jsonl_wire = is_stream or is_jsonl_source_path(source_path)
        raw_only = strong_path_classification(source_path, provider=provider)
        if raw_only is not None and path_declaration_refuses_session(provider, source_path):
            # A raw-only rule is terminal: its bytes are evidence whatever
            # they hold (an export's PNG asset, an index table), so they are
            # never decoded to decide it, and no envelope probe reads them.
            taxonomy = ArtifactStreamClassification(raw_only, True, 0)
        else:
            with source.open("rb") as classification_source, ExitStack() as classification_lifetime:
                classification_input = (
                    classification_lifetime.enter_context(
                        jsonl_parse_input_of_handle(classification_source, check_stop=check_compute_cancelled)
                    )
                    if jsonl_wire
                    else classification_source
                )
                taxonomy = classify_artifact_stream(
                    classification_input,
                    provider=provider,
                    source_path=source_path,
                    wire_format="jsonl" if jsonl_wire else "json",
                    check_stop=check_compute_cancelled,
                )
        input_admitted = not taxonomy.proved_non_session
        record_container: str | None = None
        stream_prefix: str | None = None
        bundle_count = 0
        bundle_record_stream = False
        bundle_browser_captures = True
        generic_envelope: dict[str, JSONValue] | None = None
        hermes_envelope: dict[str, JSONValue] | None = None
        design_envelope: dict[str, JSONValue] | None = None
        claude_ai_envelope: dict[str, JSONValue] | None = None
        claude_ai_arrays: tuple[str, ...] = ()
        drive_chunked: tuple[dict[str, JSONValue], str] | None = None
        drive_record_stream = False
        drive_root_array = False
        atif: tuple[dict[str, JSONValue], bool] | None = None
        otel: tuple[dict[str, JSONValue], str, tuple[otel_genai.OtelSpanIndex, str | None]] | None = None
        chatgpt_envelope: dict[str, object] | None = None
        chatgpt_mapping: ChatGPTNodeMapping | None = None
        gemini_envelope: dict[str, JSONValue] | None = None
        gemini_sidecar_scope: RetainedSidecarScope | None = None
        grok_count: int | None = None
        if input_admitted and not is_stream and provider is Provider.CHATGPT and not jsonl_wire:
            with source.open("rb") as handle:
                read_result = read_chatgpt_mapping_object(handle, store.conn)
            if (
                read_result is not None
                # Detector precedence: a browser-capture envelope may also
                # carry a valid ``mapping``; the capture route owns it.
                and not browser_capture.looks_like({**read_result[0], "mapping": {}})
                and read_result[1].children_are_all_strings()
                and chatgpt._mapping_nodes_are_valid(read_result[1].shallow_view())
            ):
                chatgpt_envelope, chatgpt_mapping = read_result
            else:
                for table in _CHATGPT_PARSER_SCRATCH_TABLES:
                    store.conn.execute(f"DROP TABLE IF EXISTS {table}")
        if input_admitted and not is_stream and provider is Provider.GEMINI_CLI and not jsonl_wire:
            store.conn.execute(
                "CREATE TABLE gemini_raw_message (ordinal INTEGER PRIMARY KEY, message_json TEXT NOT NULL)"
            )
            future_wire_type = False
            with source.open("rb") as handle:
                for ordinal, item in enumerate(ijson.items(handle, "messages.item")):
                    check_compute_cancelled()
                    if not future_wire_type and _unknown_wire_type(item) is not None:
                        future_wire_type = True
                    _append_gemini_raw_message(store.conn, ordinal, item)
            with source.open("rb") as handle:
                gemini_envelope = _gemini_cli_envelope(handle)
            if (
                gemini_envelope is None
                or future_wire_type
                or _unknown_wire_type(gemini_envelope) is not None
                or not local_agent.looks_like_gemini_cli(gemini_envelope)
            ):
                gemini_envelope = None
                store.conn.execute("DROP TABLE gemini_raw_message")
            elif sidecar_resolver is not None:
                session_id = gemini_envelope.get("sessionId")
                if isinstance(session_id, str):
                    gemini_sidecar_scope = sidecar_resolver.gemini_cli_scope(source_path, session_id)
        if (
            input_admitted
            and not is_stream
            and provider is Provider.GEMINI_CLI
            and Path(source_path).name.lower().endswith(".jsonl")
        ):
            store.conn.execute(
                "CREATE TABLE gemini_raw_message (ordinal INTEGER PRIMARY KEY, message_json TEXT NOT NULL)"
            )
            checkpoint_ordinal = 0

            def append_checkpoint_message(item: JSONValue) -> None:
                nonlocal checkpoint_ordinal
                check_compute_cancelled()
                assert store is not None
                _append_gemini_raw_message(store.conn, checkpoint_ordinal, item)
                checkpoint_ordinal += 1

            def replace_checkpoint_messages(items: Iterable[JSONValue]) -> None:
                nonlocal checkpoint_ordinal
                assert store is not None
                store.conn.execute("DELETE FROM gemini_raw_message")
                checkpoint_ordinal = 0
                for item in items:
                    append_checkpoint_message(item)

            checkpoint_header_admitted = False

            def observe_checkpoint_records(records: Iterable[JSONValue]) -> Iterator[JSONValue]:
                nonlocal checkpoint_header_admitted
                for ordinal, record in enumerate(records):
                    check_compute_cancelled()
                    if ordinal == 0:
                        checkpoint_header_admitted = local_agent.is_gemini_cli_checkpoint_stream([record])
                    yield record

            with source.open("rb") as handle:
                record_input = (
                    _PreparedPrefixInput(handle, parse_prefix_size) if parse_prefix_size is not None else handle
                )
                with owned_json_records(
                    record_input, Path(source_path).name, fail_on_decode_error=strict_jsonl_records
                ) as checkpoint_records:
                    gemini_envelope = local_agent.fold_gemini_cli_checkpoint_records(
                        observe_checkpoint_records(checkpoint_records),
                        append_message=append_checkpoint_message,
                        replace_messages=replace_checkpoint_messages,
                    )
            if gemini_envelope is None:
                store.conn.execute("DROP TABLE gemini_raw_message")
                if checkpoint_header_admitted:
                    # The canonical fold refused this entire claimed checkpoint;
                    # no collecting parser can turn it into a different session.
                    input_admitted = False
            else:
                gemini_envelope["messages"] = []
                if sidecar_resolver is not None:
                    session_id = gemini_envelope.get("sessionId")
                    if isinstance(session_id, str):
                        gemini_sidecar_scope = sidecar_resolver.gemini_cli_scope(source_path, session_id)
        # Cohort callbacks may inspect or rewrite the entire parse result.
        # The direct worker route can publish independent bundle members.
        if input_admitted and not is_stream and provider is Provider.HERMES and not jsonl_wire:
            with source.open("rb") as handle:
                hermes_envelope = hermes_snapshot_envelope(handle)
            if hermes_envelope is not None and (
                hermes_state.looks_like_state_db_payload(hermes_envelope)
                or hermes_verification.looks_like_verification_evidence_db_payload(hermes_envelope)
                or hermes_spans.looks_like_atif_payload(hermes_envelope)
            ):
                hermes_envelope = None
        if input_admitted and not is_stream and provider is Provider.GROK and not jsonl_wire:
            store.conn.execute(
                "CREATE TABLE grok_member_valid (ordinal INTEGER PRIMARY KEY, valid INTEGER NOT NULL, future_type TEXT)"
            )
            grok_probe_conn = store.conn

            def record_grok_member(index: int, valid: bool, future_type: str | None) -> None:
                grok_probe_conn.execute(
                    "INSERT INTO grok_member_valid VALUES (?, ?, ?)", (index, int(valid), future_type)
                )

            with source.open("rb") as handle:
                grok_count = grok_export_item_count(handle, on_item=record_grok_member, detect=False)
            if grok_count is None:
                store.conn.execute("DROP TABLE grok_member_valid")
        if input_admitted and not is_stream and provider in BUNDLE_PROVIDERS and is_jsonl_source_path(source_path):
            bundle_record_stream = True
            with (
                source.open("rb") as handle,
                owned_json_records(
                    handle, Path(source_path).name, fail_on_decode_error=strict_jsonl_records
                ) as records,
            ):
                for record in records:
                    check_compute_cancelled()
                    bundle_count += 1
                    bundle_browser_captures = bundle_browser_captures and browser_capture.looks_like(record)
                    del record
        if input_admitted and not is_stream and provider in BUNDLE_PROVIDERS and not jsonl_wire:
            with source.open("rb") as handle:
                record_container = json_record_container(handle)
            if record_container is not None:

                def observe_bundle_member(index: int, shape: JSONValue, witness: JSONValue | None) -> None:
                    nonlocal bundle_browser_captures
                    bundle_browser_captures = bundle_browser_captures and browser_capture.looks_like(shape)

                with source.open("rb") as handle:
                    scanned = scan_container_members(
                        handle,
                        record_container,
                        shape_keys=_BROWSER_CAPTURE_SHAPE_KEYS,
                        witnesses=0,
                        on_member=observe_bundle_member,
                    )
                if scanned is not None:
                    stream_prefix = record_container
                    bundle_count = scanned
        if input_admitted and not is_stream and provider in {Provider.DRIVE, Provider.GEMINI, Provider.UNKNOWN}:
            with source.open("rb") as handle:
                candidate = generic_message_object_envelope(handle)
            asserted_id = candidate.get("id") if candidate is not None else None
            if candidate is not None and isinstance(asserted_id, str) and asserted_id.strip():
                generic_envelope = candidate
        if (
            input_admitted
            and not is_stream
            and provider is Provider.CLAUDE_DESIGN
            and not jsonl_wire
            and record_container is None
        ):
            with source.open("rb") as handle:
                design_envelope = claude_design_object_envelope(handle)
        if (
            input_admitted
            and not is_stream
            and provider is Provider.CLAUDE_AI
            and not jsonl_wire
            and record_container is None
        ):
            with source.open("rb") as handle:
                claude_ai_object = claude_ai_object_envelope(handle)
            if claude_ai_object is not None:
                claude_ai_envelope, claude_ai_arrays = claude_ai_object
        if (
            input_admitted
            and not is_stream
            and provider in {Provider.DRIVE, Provider.GEMINI}
            and not jsonl_wire
            and generic_envelope is None
        ):
            with source.open("rb") as handle:
                drive_chunked = drive_chunked_prompt_envelope(handle)
        if (
            input_admitted
            and not is_stream
            and provider in {Provider.DRIVE, Provider.GEMINI}
            and not jsonl_wire
            and parse_prefix_size is None
            and not jsonl_wire
        ):
            with source.open("rb") as handle:
                drive_root_array = json_record_container(handle) == "item"
        if (
            input_admitted
            and not is_stream
            and provider in {Provider.DRIVE, Provider.GEMINI}
            and (jsonl_wire or drive_root_array)
        ):
            drive_future_type: str | None = None

            def observe_drive_records() -> Iterator[JSONValue]:
                nonlocal drive_future_type
                with source.open("rb") as handle, ExitStack() as record_owners:
                    record_input = (
                        _PreparedPrefixInput(handle, parse_prefix_size) if parse_prefix_size is not None else handle
                    )
                    records = (
                        (normalize_ijson_stdlib_numbers(item) for item in ijson.items(handle, "item"))
                        if drive_root_array
                        else record_owners.enter_context(
                            owned_json_records(
                                record_input, Path(source_path).name, fail_on_decode_error=strict_jsonl_records
                            )
                        )
                    )
                    for record in records:
                        check_compute_cancelled()
                        if drive_future_type is None:
                            drive_future_type = _unknown_wire_type(record)
                        if not is_json_value(record):
                            raise ValueError("Drive decoded record is not JSON")
                        yield record

            if is_drive_chunk_sequence(observe_drive_records()):
                drive_envelope: dict[str, JSONValue] = {"chunks": []}
                if drive_future_type is not None:
                    drive_envelope["__admission_future_type"] = drive_future_type
                drive_chunked = (drive_envelope, "chunks")
                drive_record_stream = True
        if (
            input_admitted
            and not is_stream
            and provider is Provider.HERMES
            and not jsonl_wire
            and hermes_envelope is None
        ):
            with source.open("rb") as handle:
                atif = _hermes_atif_envelope(handle)
            if atif is not None and atif[1]:
                with source.open("rb") as handle:
                    if not _spill_atif_subagents(handle, store.conn):
                        atif = None
        if input_admitted and not is_stream and provider is Provider.OTEL_GENAI and not jsonl_wire:
            with source.open("rb") as handle:
                otlp = _otlp_envelope(handle)
            if otlp is not None:
                with source.open("rb") as handle:
                    otel_spilled = _index_otlp_spans(handle, otlp[1], store.conn)
                if otel_spilled is not None:
                    otel = (*otlp, otel_spilled)
        if gemini_envelope is not None:
            _create_artifact_tables(store.conn)
            shard_builder = SessionShardBuilder(artifact_directory / f"shard-{uuid.uuid4().hex}.db")
            gemini_admitted = input_admitted
            gemini_session = None
            if gemini_admitted:
                index = (
                    GeminiToolOutputIndex(store.conn)
                    if gemini_sidecar_scope is not None and gemini_sidecar_scope.available
                    else None
                )
                admission_document = dict(gemini_envelope)

                def gemini_records() -> Iterator[object]:
                    # Observe only the final rows after checkpoint replacement
                    # patches, sharing each decode with the message parser.
                    assert store is not None
                    future_type: str | None = None
                    with closing(
                        store.conn.execute("SELECT message_json FROM gemini_raw_message ORDER BY ordinal")
                    ) as rows:
                        for (encoded,) in rows:
                            check_compute_cancelled()
                            record = json.loads(encoded)
                            if future_type is None:
                                future_type = _unknown_wire_type(record)
                                if future_type is not None:
                                    admission_document["messages"] = [{"type": future_type}]
                            if index is not None:
                                index.observe(record)
                            yield record

                gemini_session = local_agent.parse_gemini_cli_records(
                    gemini_envelope,
                    gemini_records(),
                    fallback_id,
                    messages=store.new_sink(),
                    session_events=store.new_event_sink(),
                )
                admission = AdmissionObserver()
                admission.observe_input(admission_document)
                gemini_session = admission.apply(gemini_session, "gemini_cli")
                if index is not None and gemini_sidecar_scope is not None:
                    for outcome in index.join(gemini_sidecar_scope):
                        gemini_session.session_events.append(local_agent.gemini_sidecar_event(outcome))
                    for position in range(len(gemini_session.messages)):
                        message = gemini_session.messages[position]
                        updated_blocks = [
                            local_agent.replace_gemini_tool_result_output(block, replacement)
                            if block.type is BlockType.TOOL_RESULT
                            and block.tool_id is not None
                            and block.media_type != local_agent.TOOL_RESULT_DISPLAY_MEDIA_TYPE
                            and (replacement := index.replacement_for(block.tool_id)) is not None
                            else block
                            for block in message.blocks
                        ]
                        if updated_blocks != message.blocks:
                            gemini_session.messages[position] = message.model_copy(update={"blocks": updated_blocks})
                    index.close()
            store.conn.execute("DROP TABLE gemini_raw_message")
            if gemini_session is not None and admit_parsed_sessions_for_publication(
                [gemini_session], provider=provider, source_path=source_path
            ):
                if prepare_session is not None and prepare_sessions is None:
                    gemini_session = prepare_session(gemini_session)
            else:
                gemini_session = None
            session_count = 0
            if gemini_session is not None:
                gemini_session.content_hash = session_content_hash(gemini_session)
                append_session_to_shard(shard_builder, gemini_session)
                _append_artifact_session(store, session_count, gemini_session)
                session_count += 1
            after_hash = file_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency()
                if preparation_dependency is not None and prepare_sessions is None
                else (None, None)
            )
            _seal_artifact(store.conn, after_hash, session_count, enrichment_digest, enrichment_index_path)
            store.conn.commit()
            shard_path = shard_builder.seal().path
            shard_builder = None
        elif chatgpt_envelope is not None:
            assert chatgpt_mapping is not None
            _create_artifact_tables(store.conn)
            shard_builder = SessionShardBuilder(artifact_directory / f"shard-{uuid.uuid4().hex}.db")
            chatgpt_admitted = input_admitted
            # The parser reads the mapping node by node from scratch, and
            # keeps its normalized messages, attachments and events there.
            session: ParsedSession | None = (
                chatgpt.parse(
                    {**chatgpt_envelope, "mapping": chatgpt_mapping.shallow_view()},
                    f"{fallback_id}-0",
                    spill=ScratchSessionSpill(store),
                )
                if chatgpt_admitted
                else None
            )
            if session is not None and not admit_parsed_sessions_for_publication(
                [session], provider=provider, source_path=source_path
            ):
                session = None
            if session is not None:
                connection = store.conn
                connection.execute("SAVEPOINT chatgpt_prepared_sidecars")
                next_attachment = store._next_attachment_ordinal
                next_event = store._next_event_ordinal
                try:
                    source_attachments: object = session.attachments
                    if isinstance(source_attachments, SqliteAttachmentSink):
                        attachments = source_attachments
                    else:
                        attachments = store.new_attachment_sink()
                        attachments.extend(session.attachments)
                    source_events: object = session.session_events
                    if isinstance(source_events, SqliteSessionEventSink):
                        events = source_events
                    else:
                        events = store.new_event_sink()
                        events.extend(session.session_events)
                    session = session.model_copy(update={"attachments": attachments, "session_events": events})
                    if prepare_session is not None and prepare_sessions is None:
                        session = prepare_session(session)
                except BaseException:
                    connection.execute("ROLLBACK TO chatgpt_prepared_sidecars")
                    connection.execute("RELEASE chatgpt_prepared_sidecars")
                    store._next_attachment_ordinal = next_attachment
                    store._next_event_ordinal = next_event
                    raise
                if session is None:
                    connection.execute("ROLLBACK TO chatgpt_prepared_sidecars")
                    store._next_attachment_ordinal = next_attachment
                    store._next_event_ordinal = next_event
                connection.execute("RELEASE chatgpt_prepared_sidecars")
            session_count = 0
            if session is not None:
                session.content_hash = session_content_hash(session)
                append_session_to_shard(shard_builder, session)
                _append_artifact_session(store, session_count, session)
                session_count += 1
            else:
                store.conn.execute("DELETE FROM prepared_message")
                store.conn.execute("DELETE FROM prepared_event")
                store.conn.execute("DELETE FROM prepared_attachment")
            if session_count:
                store.conn.execute(
                    "DELETE FROM prepared_message WHERE session_ordinal NOT IN "
                    "(SELECT message_ordinal FROM prepared_session UNION SELECT normalized_ordinal FROM prepared_message_normalization)"
                )
                store.conn.execute(
                    "DELETE FROM prepared_event WHERE session_ordinal NOT IN "
                    "(SELECT event_ordinal FROM prepared_session)"
                )
                store.conn.execute(
                    "DELETE FROM prepared_attachment WHERE session_ordinal NOT IN "
                    "(SELECT attachment_ordinal FROM prepared_session)"
                )
            # Parser-only scratch never reaches the sealed artifact.
            for table in _CHATGPT_PARSER_SCRATCH_TABLES:
                store.conn.execute(f"DROP TABLE IF EXISTS {table}")
            after_hash = file_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency()
                if preparation_dependency is not None and prepare_sessions is None
                else (None, None)
            )
            _seal_artifact(store.conn, after_hash, session_count, enrichment_digest, enrichment_index_path)
            store.conn.commit()
            shard_path = shard_builder.seal().path
            shard_builder = None
        elif hermes_envelope is not None:
            _create_artifact_tables(store.conn)
            shard_builder = SessionShardBuilder(artifact_directory / f"shard-{uuid.uuid4().hex}.db")
            hermes_admitted = input_admitted
            session = None
            if hermes_admitted:
                with source.open("rb") as handle:
                    session = local_agent.parse_hermes_snapshot_stream(
                        hermes_envelope,
                        (normalize_ijson_stdlib_numbers(item) for item in ijson.items(handle, "messages.item")),
                        fallback_id,
                        messages=store.new_sink(),
                        session_events=store.new_event_sink(),
                        source_path=source_path,
                        profile_identity=profile_identity,
                    )
            session_count = 0
            if session is not None and admit_parsed_sessions_for_publication(
                [session], provider=provider, source_path=source_path
            ):
                if prepare_session is not None and prepare_sessions is None:
                    session = prepare_session(session)
            else:
                session = None
            if session is not None:
                session.content_hash = session_content_hash(session)
                append_session_to_shard(shard_builder, session)
                _append_artifact_session(store, session_count, session)
                session_count += 1
            after_hash = file_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency()
                if preparation_dependency is not None and prepare_sessions is None
                else (None, None)
            )
            _seal_artifact(store.conn, after_hash, session_count, enrichment_digest, enrichment_index_path)
            store.conn.commit()
            shard_path = shard_builder.seal().path
            shard_builder = None
        elif generic_envelope is not None:
            _create_artifact_tables(store.conn)
            shard_builder = SessionShardBuilder(artifact_directory / f"shard-{uuid.uuid4().hex}.db")
            generic_admitted = input_admitted
            session = None
            if generic_admitted:
                with source.open("rb") as handle:
                    session = parse_generic_messages_stream(
                        provider,
                        generic_envelope,
                        (normalize_ijson_stdlib_numbers(item) for item in ijson.items(handle, "messages.item")),
                        fallback_id,
                        message_sink=store.new_sink(),
                    )
            session_count = 0
            if session is not None and admit_parsed_sessions_for_publication(
                [session], provider=provider, source_path=source_path
            ):
                if prepare_session is not None and prepare_sessions is None:
                    session = prepare_session(session)
            else:
                session = None
            if session is not None:
                session.content_hash = session_content_hash(session)
                append_session_to_shard(shard_builder, session)
                _append_artifact_session(store, session_count, session)
                session_count += 1
            after_hash = file_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency()
                if preparation_dependency is not None and prepare_sessions is None
                else (None, None)
            )
            _seal_artifact(store.conn, after_hash, session_count, enrichment_digest, enrichment_index_path)
            store.conn.commit()
            shard_path = shard_builder.seal().path
            shard_builder = None
        elif design_envelope is not None:
            _create_artifact_tables(store.conn)
            shard_builder = SessionShardBuilder(artifact_directory / f"shard-{uuid.uuid4().hex}.db")
            design_admitted = input_admitted
            session = None
            if design_admitted:
                with source.open("rb") as handle:
                    session = parse_design_stream(
                        design_envelope,
                        (normalize_ijson_stdlib_numbers(item) for item in ijson.items(handle, "messages.item")),
                        fallback_id,
                        message_sink=store.new_sink(),
                        event_sink=store.new_event_sink(),
                        attachment_sink=store.new_attachment_sink(),
                    )
            session_count = 0
            if session is not None and admit_parsed_sessions_for_publication(
                [session], provider=provider, source_path=source_path
            ):
                if prepare_session is not None and prepare_sessions is None:
                    session = prepare_session(session)
            else:
                session = None
            if session is not None:
                session.content_hash = session_content_hash(session)
                append_session_to_shard(shard_builder, session)
                _append_artifact_session(store, session_count, session)
                session_count += 1
            after_hash = file_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency()
                if preparation_dependency is not None and prepare_sessions is None
                else (None, None)
            )
            _seal_artifact(store.conn, after_hash, session_count, enrichment_digest, enrichment_index_path)
            store.conn.commit()
            shard_path = shard_builder.seal().path
            shard_builder = None
        elif claude_ai_envelope is not None:
            _create_artifact_tables(store.conn)
            shard_builder = SessionShardBuilder(artifact_directory / f"shard-{uuid.uuid4().hex}.db")
            claude_ai_admitted = input_admitted
            # The collecting route parses this document as a one-item bundle,
            # so its fallback identity carries that suffix.
            session = (
                _stream_claude_ai_object(source, claude_ai_envelope, claude_ai_arrays, store, f"{fallback_id}-0")
                if claude_ai_admitted
                else None
            )
            session_count = 0
            if session is not None and admit_parsed_sessions_for_publication(
                [session], provider=provider, source_path=source_path
            ):
                if prepare_session is not None and prepare_sessions is None:
                    session = prepare_session(session)
            else:
                session = None
            if session is not None:
                session.content_hash = session_content_hash(session)
                append_session_to_shard(shard_builder, session)
                _append_artifact_session(store, session_count, session)
                session_count += 1
            for table, column in (
                ("prepared_message", "message_ordinal"),
                ("prepared_event", "event_ordinal"),
                ("prepared_attachment", "attachment_ordinal"),
            ):
                store.conn.execute(
                    f"DELETE FROM {table} WHERE session_ordinal NOT IN (SELECT {column} FROM prepared_session"
                    + (
                        " UNION SELECT normalized_ordinal FROM prepared_message_normalization"
                        if table == "prepared_message"
                        else ""
                    )
                    + ")"
                )
            after_hash = file_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency()
                if preparation_dependency is not None and prepare_sessions is None
                else (None, None)
            )
            _seal_artifact(store.conn, after_hash, session_count, enrichment_digest, enrichment_index_path)
            store.conn.commit()
            shard_path = shard_builder.seal().path
            shard_builder = None
        elif drive_chunked is not None:
            drive_envelope, chunk_prefix = drive_chunked
            _create_artifact_tables(store.conn)
            shard_builder = SessionShardBuilder(artifact_directory / f"shard-{uuid.uuid4().hex}.db")

            def drive_chunks() -> Iterator[object]:
                with source.open("rb") as handle, ExitStack() as record_owners:
                    if drive_record_stream:
                        record_input = (
                            _PreparedPrefixInput(handle, parse_prefix_size) if parse_prefix_size is not None else handle
                        )
                        records = (
                            (normalize_ijson_stdlib_numbers(item) for item in ijson.items(handle, "item"))
                            if drive_root_array
                            else record_owners.enter_context(
                                owned_json_records(
                                    record_input, Path(source_path).name, fail_on_decode_error=strict_jsonl_records
                                )
                            )
                        )
                        for item in records:
                            check_compute_cancelled()
                            yield item
                    else:
                        for item in ijson.items(handle, f"{chunk_prefix}.item"):
                            check_compute_cancelled()
                            yield normalize_ijson_stdlib_numbers(item)

            drive_admitted = input_admitted
            session = None
            if drive_admitted:
                session = drive.parse_chunked_prompt_stream(
                    provider,
                    drive_envelope,
                    drive_chunks,
                    fallback_id,
                    messages=store.new_sink(),
                    session_events=store.new_event_sink(),
                    attachments=store.new_attachment_sink(),
                    scratch=store.conn,
                    record_stream=drive_record_stream,
                )
            session_count = 0
            if session is not None and admit_parsed_sessions_for_publication(
                [session], provider=provider, source_path=source_path
            ):
                if prepare_session is not None and prepare_sessions is None:
                    session = prepare_session(session)
            else:
                session = None
            if session is not None:
                session.content_hash = session_content_hash(session)
                append_session_to_shard(shard_builder, session)
                _append_artifact_session(store, session_count, session)
                session_count += 1
            for table, column in (
                ("prepared_message", "message_ordinal"),
                ("prepared_event", "event_ordinal"),
                ("prepared_attachment", "attachment_ordinal"),
            ):
                store.conn.execute(
                    f"DELETE FROM {table} WHERE session_ordinal NOT IN (SELECT {column} FROM prepared_session"
                    + (
                        " UNION SELECT normalized_ordinal FROM prepared_message_normalization"
                        if table == "prepared_message"
                        else ""
                    )
                    + ")"
                )
            after_hash = file_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency()
                if preparation_dependency is not None and prepare_sessions is None
                else (None, None)
            )
            _seal_artifact(store.conn, after_hash, session_count, enrichment_digest, enrichment_index_path)
            store.conn.commit()
            shard_path = shard_builder.seal().path
            shard_builder = None
        elif atif is not None:
            atif_envelope, atif_has_subagents = atif
            _create_artifact_tables(store.conn)
            shard_builder = SessionShardBuilder(artifact_directory / f"shard-{uuid.uuid4().hex}.db")

            def atif_steps() -> Iterator[JSONValue]:
                with source.open("rb") as handle:
                    for item in ijson.items(handle, "steps.item"):
                        check_compute_cancelled()
                        yield cast(JSONValue, normalize_ijson_stdlib_numbers(item))

            atif_admitted = input_admitted
            atif_sessions = _empty_parsed_sessions()
            steps: Iterable[JSONValue] = ()
            unknown_steps: list[JSONValue] = []
            if atif_admitted:

                def observed_steps() -> Iterator[JSONValue]:
                    for step in atif_steps():
                        if not unknown_steps and isinstance(step, dict):
                            discriminators: dict[str, JSONValue] = {
                                key: step[key] for key in ("type", "kind") if key in step
                            }
                            if hermes_unknown_wire_type({"steps": [discriminators]}) is not None:
                                unknown_steps.append(discriminators)
                        yield step

                @contextmanager
                def retained_children(children: Iterable[ParsedSession]) -> Iterator[Iterable[ParsedSession]]:
                    child_directory = artifact_directory / f"atif-children-{uuid.uuid4().hex}"
                    child_directory.mkdir()

                    def admitted_children() -> Iterator[ParsedSession]:
                        for child in children:
                            yield from admit_parsed_sessions(
                                "hermes", {**atif_envelope, "steps": unknown_steps}, [child]
                            )

                    child_artifact = PreparedJsonl.from_sessions(
                        admitted_children(),
                        blob_hash=before_hash,
                        artifact_directory=child_directory,
                        publication_publisher=None,
                        classification=taxonomy,
                        parsed_prefix_size=parse_prefix_size,
                        resolved_provider=provider,
                        captured_profile_key=profile_identity,
                    )
                    try:
                        yield child_artifact.session_sequence()
                    finally:
                        child_artifact.discard()

                steps = observed_steps()
                atif_sessions = hermes_spans.iter_atif_sessions(
                    atif_envelope,
                    steps,
                    _atif_subagents(store.conn) if atif_has_subagents else (),
                    fallback_id,
                    profile_root=profile_root_for_artifact(Path(source_path)),
                    profile_identity=profile_identity,
                    new_events=store.new_event_sink,
                    retain_children=retained_children,
                )
            session_count = 0
            with closing(atif_sessions) as selected_atif_sessions:
                for session in selected_atif_sessions:
                    for _ in steps:  # The parser still owes any unconsumed original steps a scan.
                        pass
                    [session] = admit_parsed_sessions("hermes", {**atif_envelope, "steps": unknown_steps}, [session])
                    if not admit_parsed_sessions_for_publication([session], provider=provider, source_path=source_path):
                        continue
                    if prepare_session is not None and prepare_sessions is None:
                        session = prepare_session(session)
                    session.content_hash = session_content_hash(session)
                    append_session_to_shard(shard_builder, session)
                    _append_artifact_session(store, session_count, session)
                    session_count += 1
            if atif_has_subagents:
                _drop_atif_subagents(store.conn)
            for table, column in (
                ("prepared_message", "message_ordinal"),
                ("prepared_event", "event_ordinal"),
                ("prepared_attachment", "attachment_ordinal"),
            ):
                store.conn.execute(
                    f"DELETE FROM {table} WHERE session_ordinal NOT IN (SELECT {column} FROM prepared_session"
                    + (
                        " UNION SELECT normalized_ordinal FROM prepared_message_normalization"
                        if table == "prepared_message"
                        else ""
                    )
                    + ")"
                )
            after_hash = file_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency()
                if preparation_dependency is not None and prepare_sessions is None
                else (None, None)
            )
            _seal_artifact(store.conn, after_hash, session_count, enrichment_digest, enrichment_index_path)
            store.conn.commit()
            shard_path = shard_builder.seal().path
            shard_builder = None
        elif otel is not None:
            otel_envelope, otel_root_key, (otel_index, otel_unknown_kind) = otel
            # The dispatch route proves the whole document against the OTLP
            # discriminator scan, which reads only each span's ``kind``; the
            # spill recorded the first unknown one, so this one-record
            # document carries the same conservation proof.
            otel_admission_payload: dict[str, JSONValue] = (
                {otel_root_key: [{"scopeSpans": [{"spans": [{"kind": otel_unknown_kind}]}]}]}
                if otel_unknown_kind is not None
                else {}
            )
            _create_artifact_tables(store.conn)
            shard_builder = SessionShardBuilder(artifact_directory / f"shard-{uuid.uuid4().hex}.db")
            # Taxonomy decides a declared OTLP path by its rule and root
            # markers, so the witness carries the root fields, not the spans.
            otel_admitted = input_admitted and otel_index.normalizable
            session_count = 0
            for session in (
                otel_index.sessions(new_messages=store.new_sink, new_events=store.new_event_sink)
                if otel_admitted
                else ()
            ):
                session = admit_parsed_sessions(provider.value.replace("-", "_"), otel_admission_payload, [session])[0]
                if not admit_parsed_sessions_for_publication([session], provider=provider, source_path=source_path):
                    continue
                if prepare_session is not None and prepare_sessions is None:
                    session = prepare_session(session)
                session.content_hash = session_content_hash(session)
                append_session_to_shard(shard_builder, session)
                _append_artifact_session(store, session_count, session)
                session_count += 1
            otel_index.close()
            for table, column in (
                ("prepared_message", "message_ordinal"),
                ("prepared_event", "event_ordinal"),
                ("prepared_attachment", "attachment_ordinal"),
            ):
                store.conn.execute(
                    f"DELETE FROM {table} WHERE session_ordinal NOT IN (SELECT {column} FROM prepared_session"
                    + (
                        " UNION SELECT normalized_ordinal FROM prepared_message_normalization"
                        if table == "prepared_message"
                        else ""
                    )
                    + ")"
                )
            after_hash = file_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency()
                if preparation_dependency is not None and prepare_sessions is None
                else (None, None)
            )
            _seal_artifact(store.conn, after_hash, session_count, enrichment_digest, enrichment_index_path)
            store.conn.commit()
            shard_path = shard_builder.seal().path
            shard_builder = None
        elif grok_count is not None:
            _create_artifact_tables(store.conn)
            shard_builder = SessionShardBuilder(artifact_directory / f"shard-{uuid.uuid4().hex}.db")
            grok_admitted = input_admitted
            grok_member_conn = store.conn

            def include_grok_member(index: int) -> bool:
                row = grok_member_conn.execute(
                    "SELECT valid FROM grok_member_valid WHERE ordinal = ?", (index,)
                ).fetchone()
                if row is None:
                    raise _SourceChangedDuringPreparationError("Grok member changed during preparation")
                return bool(row[0])

            session_count = 0
            member_index = -1
            member_conversation: dict[str, object] | None = None
            member_responses = False
            member_messages: SqliteMessageSink | None = None
            with source.open("rb") as handle:
                for event, value in (
                    iter_grok_export_events(handle, include_item=include_grok_member) if grok_admitted else ()
                ):
                    if event == "begin":
                        member_index += 1
                        member_conversation = None
                        member_responses = False
                        member_messages = store.new_sink() if include_grok_member(member_index) else None
                    elif event == "conversation" and isinstance(value, dict):
                        member_conversation = value
                    elif event == "responses":
                        member_responses = True
                    elif event == "response" and member_messages is not None:
                        grok.append_conversation_response(member_messages, value)
                    elif event == "end" and member_conversation is not None and member_responses:
                        assert member_messages is not None
                        session = grok.finish_conversation(
                            member_conversation,
                            fallback_id if grok_count == 1 else f"{fallback_id}-{member_index}",
                            member_messages,
                        )
                        # Admit this outer record through the parser's own
                        # wrapper, over a stub carrying the member's first
                        # future wire type, without reloading its responses.
                        future_row = grok_member_conn.execute(
                            "SELECT future_type FROM grok_member_valid WHERE ordinal = ?", (member_index,)
                        ).fetchone()
                        admission_stub: dict[str, object] = {"conversation": {}, "responses": []}
                        if future_row is not None and future_row[0] is not None:
                            admission_stub["type"] = future_row[0]
                        admitted = grok.parse_conversation(admission_stub, session.provider_session_id)
                        session = session.model_copy(
                            update={
                                "session_events": [*session.session_events, *admitted.session_events],
                                "unit_accounting": admitted.unit_accounting,
                            }
                        )
                        if not admit_parsed_sessions_for_publication(
                            [session], provider=provider, source_path=source_path
                        ):
                            continue
                        if prepare_session is not None and prepare_sessions is None:
                            session = prepare_session(session)
                        session.content_hash = session_content_hash(session)
                        append_session_to_shard(shard_builder, session)
                        _append_artifact_session(store, session_count, session)
                        session_count += 1
            if grok_admitted and member_index + 1 != grok_count:
                raise _SourceChangedDuringPreparationError("Grok conversation count changed during preparation")
            store.conn.execute("DROP TABLE grok_member_valid")
            after_hash = file_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency()
                if preparation_dependency is not None and prepare_sessions is None
                else (None, None)
            )
            _seal_artifact(store.conn, after_hash, session_count, enrichment_digest, enrichment_index_path)
            store.conn.commit()
            shard_path = shard_builder.seal().path
            shard_builder = None
        elif stream_prefix is not None or bundle_record_stream:
            _create_artifact_tables(store.conn)
            shard_builder = SessionShardBuilder(artifact_directory / f"shard-{uuid.uuid4().hex}.db")
            bundle_admitted = input_admitted
            drift = BundleCandidateDrift()
            session_count = 0
            member_count = 0

            def original_bundle_members(handle: BinaryIO) -> Iterator[int | None]:
                if stream_prefix is not None:
                    yield from iter_container_member_files(handle, stream_prefix, member_path)
                    return
                with owned_json_records(
                    handle, Path(source_path).name, fail_on_decode_error=strict_jsonl_records
                ) as records:
                    for index, record in enumerate(records):
                        check_compute_cancelled()
                        # Reuse the canonical member parser over one original
                        # decoded record, rather than retaining the input cohort.
                        if not isinstance(record, dict):
                            yield None
                            continue
                        write_streamed_json(record, member_path, member_format=True)
                        del record
                        yield index

            with source.open("rb") as handle:
                for bundle_index in original_bundle_members(handle) if bundle_admitted else ():
                    member_count += 1
                    if bundle_index is None:
                        # No bundle lowering reads a non-object member.
                        continue
                    streamed, member_sessions = _streamed_bundle_member(
                        provider,
                        member_path,
                        store,
                        f"{fallback_id}-{bundle_index}",
                        all_browser_captures=bundle_browser_captures,
                    )
                    if streamed:
                        if provider is Provider.CHATGPT:
                            drift.observe_streamed_conversation()
                    else:
                        with member_path.open("rb") as member_handle:
                            (record,) = iter_json_container_records(member_handle, "")
                        member_sessions = bundle_member_sessions(
                            provider,
                            record,
                            fallback_id,
                            bundle_index,
                            count=bundle_count,
                            all_browser_captures=bundle_browser_captures,
                            drift=drift,
                            source_path=source_path,
                            sidecar_resolver=sidecar_resolver,
                        )
                        del record
                    for session in member_sessions:
                        if not admit_parsed_sessions_for_publication(
                            [session], provider=provider, source_path=source_path
                        ):
                            continue
                        if prepare_session is not None and prepare_sessions is None:
                            session = prepare_session(session)
                        session.content_hash = session_content_hash(session)
                        append_session_to_shard(shard_builder, session)
                        _append_artifact_session(store, session_count, session)
                        session_count += 1
            if bundle_admitted and member_count != bundle_count:
                raise _SourceChangedDuringPreparationError("bundle member count changed during preparation")
            drift.emit(provider, fallback_id)
            for table, column in (
                ("prepared_message", "message_ordinal"),
                ("prepared_event", "event_ordinal"),
                ("prepared_attachment", "attachment_ordinal"),
            ):
                store.conn.execute(
                    f"DELETE FROM {table} WHERE session_ordinal NOT IN (SELECT {column} FROM prepared_session"
                    + (
                        " UNION SELECT normalized_ordinal FROM prepared_message_normalization"
                        if table == "prepared_message"
                        else ""
                    )
                    + ")"
                )
            after_hash = file_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency()
                if preparation_dependency is not None and prepare_sessions is None
                else (None, None)
            )
            _seal_artifact(store.conn, after_hash, session_count, enrichment_digest, enrichment_index_path)
            store.conn.commit()
            shard_path = shard_builder.seal().path
            shard_builder = None
        else:
            _create_artifact_tables(store.conn)
            shard_builder = SessionShardBuilder(artifact_directory / f"shard-{uuid.uuid4().hex}.db")
            session_count = 0
            with ExitStack() as decoded_inputs:
                handle = decoded_inputs.enter_context(source.open("rb"))
                record_input = (
                    _PreparedPrefixInput(handle, parse_prefix_size) if parse_prefix_size is not None else handle
                )
                records = (
                    decoded_inputs.enter_context(
                        owned_json_records(
                            record_input,
                            Path(source_path).name,
                            fail_on_decode_error=strict_jsonl_records or provider is Provider.UNKNOWN,
                        )
                    )
                    if input_admitted
                    else iter(())
                )
                if not input_admitted:
                    sessions = _empty_parsed_sessions()
                elif is_stream:
                    sessions = iter_parsed_stream(
                        provider,
                        records,
                        fallback_id,
                        source_path=source_path,
                        profile_identity=profile_identity,
                        message_sink_factory=store.new_sink,
                        event_sink_factory=store.new_event_sink,
                        attachment_sink_factory=store.new_attachment_sink,
                        sidecar_resolver=sidecar_resolver,
                    )
                else:
                    cohort = (
                        records
                        if isinstance(records, DecodedRecordSequence)
                        else decoded_inputs.enter_context(closing(DecodedRecordSequence(records)))
                    )
                    sessions = iter_parsed_payload(
                        provider,
                        cohort,
                        fallback_id,
                        source_path=source_path,
                        profile_identity=profile_identity,
                        sidecar_resolver=sidecar_resolver,
                        message_sink_factory=store.new_sink,
                        event_sink_factory=store.new_event_sink,
                        attachment_sink_factory=store.new_attachment_sink,
                    )
                with closing(sessions) as selected_sessions:
                    for session in selected_sessions:
                        check_compute_cancelled()
                        if not admit_parsed_sessions_for_publication(
                            [session], provider=provider, source_path=source_path
                        ):
                            continue
                        if prepare_session is not None and prepare_sessions is None:
                            session = prepare_session(session)
                        session.content_hash = session_content_hash(session)
                        append_session_to_shard(shard_builder, session)
                        _append_artifact_session(store, session_count, session)
                        session_count += 1
                        del session
            after_hash = file_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency()
                if preparation_dependency is not None and prepare_sessions is None
                else (None, None)
            )
            _seal_artifact(store.conn, after_hash, session_count, enrichment_digest, enrichment_index_path)
            store.conn.commit()
            shard_path = shard_builder.seal().path
            shard_builder = None
        if fact_path_admits_session_content(source_path, provider=provider):
            explicit = strong_path_classification(source_path, provider=provider)
            accepted_count = store.conn.execute("SELECT COUNT(*) FROM prepared_session").fetchone()[0]
            if explicit is not None and not explicit.parse_as_session and not accepted_count:
                taxonomy = ArtifactStreamClassification(explicit, True, taxonomy.record_count)
        record_prepared_classification(store.conn, taxonomy)
        if publication_publisher is not None and prepare_sessions is None:
            _prepare_attachment_publications(
                store, publication_publisher, artifact_directory, source_read=publication_source_read
            )
            _prepare_sidecar_publications(store, publication_publisher, artifact_directory)
        check_compute_cancelled()
        store.conn.commit()
        store.close()
        store = None
        result = PreparedJsonl.seal(
            after_hash,
            sessions_path,
            shard_path,
            enrichment_digest=enrichment_digest,
            enrichment_index_path=enrichment_index_path,
            parsed_prefix_size=parse_prefix_size,
            resolved_provider=provider,
            captured_profile_key=profile_identity,
            # Every branch above admits its sessions before sealing.
            positive_evidence_filtered=True,
            attempt_directory=attempt_directory,
            publication_publisher=publication_publisher,
        )
        if prepare_sessions is not None:
            result = _finalize_prepared_cohort(
                result,
                prepare_sessions,
                artifact_directory=artifact_directory,
                publication_publisher=publication_publisher,
                publication_source_read=publication_source_read,
                preparation_dependency=preparation_dependency,
            )
            if file_digest(source) != before_hash:
                source_change = _SourceChangedDuringPreparationError("blob changed during cohort preparation")
                try:
                    result.discard()
                except BaseException as cleanup:
                    raise BaseExceptionGroup(
                        "cohort input changed and cleanup failed", [source_change, cleanup]
                    ) from None
                raise source_change
        sealed = True
        return result
    except (
        BaseExceptionGroup,
        DaemonBackpressureError,
        ReferenceSealError,
        NativeConnectionSettlementError,
    ):
        raise
    except VerificationCancelledError as exc:
        raise DaemonOperationCancelled("artifact preparation byte verification cancelled") from exc
    except Exception as exc:
        if shard_builder is not None:
            shard_builder.abandon()
        if shard_path is not None:
            discard_session_shard(shard_path)
        if isinstance(exc, DaemonOperationCancelled):
            raise
        retryable = isinstance(exc, (OSError, sqlite3.OperationalError, _SourceChangedDuringPreparationError))
        error_hash: str | None = None
        if before_hash is not None:
            try:
                after_error_hash = file_digest(source)
            except OSError:
                retryable = True
            else:
                if after_error_hash == before_hash:
                    error_hash = before_hash
                else:
                    retryable = True
        return PreparedJsonl(
            error_hash,
            None,
            None,
            f"{type(exc).__name__}: {exc}"[:500],
            deferred=retryable,
            # Only a failure bound to hashed bytes speaks for a capture. Its
            # provider token parsed before the source was hashed.
            resolved_provider=Provider.from_string(provider_value) if error_hash is not None else None,
            decode_failure=None if retryable else classify_decode_failure(exc),
            captured_profile_key=profile_identity,
        )
    finally:
        primary = sys.exception()
        if store is not None:
            try:
                store.close()
            except BaseException as cleanup:
                if primary is not None:
                    raise BaseExceptionGroup(
                        "artifact preparation and physical close failed", [primary, cleanup]
                    ) from None
                raise
        member_path.unlink(missing_ok=True)
        if not sealed:
            sessions_path.unlink(missing_ok=True)


if TYPE_CHECKING:
    from polylogue.sources.codex_state_projection import PreparedThreadStateProjection
    from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveSourceBlobRef
    from polylogue.storage.sqlite.archive_tiers.write import SessionSourceRead
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
