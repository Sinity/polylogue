"""Sealed, source-bound preparation for JSON and JSONL session captures."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from contextlib import closing, contextmanager, suppress
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from itertools import islice
from pathlib import Path
from typing import BinaryIO, cast, overload
from urllib.parse import quote

import ijson

from polylogue.core.enums import BlockType, Provider
from polylogue.core.identity_law import session_id as archive_session_id
from polylogue.core.json import JSONValue
from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind
from polylogue.core.sources import origin_from_provider
from polylogue.logging import WARNING, emit
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.decoder_json import (
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
    grok_taxonomy_witness,
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
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import (
    BUNDLE_PROVIDERS,
    BundleCandidateDrift,
    bundle_member_sessions,
    parse_generic_messages_stream,
    parse_payload,
    parse_stream_payload,
    require_positive_conversational_evidence,
)
from polylogue.sources.parsers import (
    browser_capture,
    chatgpt,
    drive,
    grok,
    hermes_identity,
    hermes_spans,
    hermes_state,
    hermes_verification,
    local_agent,
    otel_genai,
)
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.base_support import (
    _unknown_wire_type,
    admit_parsed_sessions,
    hermes_unknown_wire_type,
    otel_genai_unknown_wire_type,
)
from polylogue.sources.parsers.claude.ai_parser import parse_ai_stream, parse_design_stream
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
    discard_decoded_sessions,
    read_chatgpt_mapping_object,
)
from polylogue.sources.sidecar_evidence import RetainedSidecarScope, SidecarResolver
from polylogue.sources.value_bounds import ValueBoundRefusedError
from polylogue.storage.sqlite.archive_tiers.write import (
    PreparedSessionWrite,
    append_session_to_shard,
    prepare_session_shard,
)
from polylogue.storage.sqlite.archive_tiers.write_shard import (
    SessionShard,
    SessionShardBuilder,
    discard_session_shard,
    open_session_shard,
)

_ARTIFACT_VERSION = 3


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


class VerificationCancelledError(Exception):
    """A digest pass stopped at a chunk boundary because its caller was cancelled."""


class DecodeFailure(StrEnum):
    """Which JSON decode boundary refused a source's bytes."""

    #: A JSON document or JSON stream did not decode.
    DOCUMENT = "document"
    #: A complete (newline-terminated or final) JSONL record did not decode.
    JSONL_RECORD = "jsonl_record"


class PreparedDecodeError(ValueError):
    """A worker's decode failure, raised again by the writer with its kind.

    An exception does not survive the worker boundary with its type (the
    carrier is pickled, and ``JsonlDecodeError`` cannot be rebuilt from its
    message), so the carrier names the kind and the writer raises this.
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

    Every decode failure is terminal. The evidence is bound to the raw, and a
    raw is its content hash, so re-reading the same bytes can only fail the
    same way; a source that later completes or changes is a new raw. An
    unknown-provider capture is ``terminal_unknown_json_decode``. For a known
    provider, a complete JSONL record or a JSON document that does not decode
    is ``terminal_corrupt_input``, which canonical ingest also reports
    (``classify_decode_exception``). An unterminated JSONL tail never reaches
    here: it is excluded from the parsed prefix. Live intake and retained
    replay both decide from this one rule, so a rebuild refuses exactly the
    bytes live intake refused, and neither re-selects them.
    """
    if classify_decode_failure(error) is None:
        return None
    if provider is Provider.UNKNOWN:
        return RawFailureEvidenceKind.TERMINAL_UNKNOWN_JSON_DECODE
    return RawFailureEvidenceKind.TERMINAL_CORRUPT_INPUT


def _source_digest(path: Path, *, stop: Callable[[], bool] | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            if stop is not None and stop():
                raise VerificationCancelledError(str(path))
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def source_snapshot(source: Path, directory: Path) -> Iterator[tuple[Path, str]]:
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
    holder.mkdir(parents=True)
    snapshot = holder / source.name
    try:
        digest = hashlib.sha256()
        with source.open("rb") as reader, snapshot.open("xb") as writer:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                digest.update(chunk)
                writer.write(chunk)
        os.chmod(snapshot, 0o400)
        yield snapshot, digest.hexdigest()
    finally:
        with suppress(FileNotFoundError):
            shutil.rmtree(holder)


def _iter_prefix_lines(handle: BinaryIO, prefix_size: int) -> Iterator[bytes]:
    """Expose only the proven complete JSONL prefix to the record decoder."""
    if prefix_size < 0:
        raise ValueError("JSONL prefix size must be non-negative")
    remaining = prefix_size
    while remaining:
        line = handle.readline(remaining)
        if not line:
            raise OSError("sealed source ended before its prepared JSONL prefix")
        remaining -= len(line)
        yield line


@dataclass(frozen=True, slots=True)
class PreparedFileSeal:
    """Closed scratch-file bytes and the exact inode handed to publication."""

    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def capture(cls, path: Path) -> PreparedFileSeal:
        # No supported writer exists after the SQLite owner closes. Read-only
        # permissions make accidental edits fail; the stat pair rejects a
        # replacement or concurrent mutation during the digest pass.
        os.chmod(path, 0o400)
        before = path.stat()
        digest = _source_digest(path)
        after = path.stat()
        if _file_identity(before) != _file_identity(after):
            raise ValueError(f"prepared file changed while sealing: {path}")
        return cls(digest, *_file_identity(after))

    def verify(self, path: Path, *, full: bool, stop: Callable[[], bool] | None = None) -> None:
        before = path.stat()
        if _file_identity(before) != self.identity:
            raise ValueError(f"prepared file identity changed: {path}")
        if full:
            if _source_digest(path, stop=stop) != self.sha256:
                raise ValueError(f"prepared file content changed: {path}")
            after = path.stat()
            if _file_identity(after) != self.identity:
                raise ValueError(f"prepared file changed during verification: {path}")

    @property
    def identity(self) -> tuple[int, int, int, int, int]:
        return self.device, self.inode, self.size, self.mtime_ns, self.ctime_ns


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


def _atif_subagent_witness(conn: sqlite3.Connection) -> list[JSONValue]:
    """The first 64 subagent entries, each with at most its first 64 steps."""
    witness: list[JSONValue] = []
    for ordinal, fields_json, step_count in conn.execute(
        "SELECT ordinal, fields_json, step_count FROM atif_subagent ORDER BY ordinal LIMIT 64"
    ):
        fields = json.loads(fields_json)
        if step_count is not None:
            fields["steps"] = list(islice(_atif_subagent_steps(conn, ordinal), 64))
        witness.append(fields)
    return witness


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


def _classify_grok_witness(source: Path, classify: Callable[[JSONValue], bool]) -> bool:
    with source.open("rb") as handle:
        witness = grok_taxonomy_witness(handle)
    return classify(witness)


def _file_identity(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


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
        positive_evidence_filtered: bool = False,
        attempt_directory: Path | None = None,
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
            positive_evidence_filtered=positive_evidence_filtered,
            attempt_directory=attempt_directory,
        )

    def verify_files(self, *, full: bool, stop: Callable[[], bool] | None = None) -> None:
        """Scan bytes before admission; recheck inode identity at publication.

        ``stop`` is polled between digest chunks; when it returns true the
        scan raises ``VerificationCancelledError``.
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
        for prepared in self.prepared_writes:
            prepared.close()
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
            return
        if self.sessions_path is not None:
            self.sessions_path.unlink(missing_ok=True)
            self.sessions_path.with_name(self.sessions_path.name + "-journal").unlink(missing_ok=True)
        if self.shard_path is not None:
            discard_session_shard(self.shard_path)

    def iter_sessions(self) -> Generator[ParsedSession, None, None]:
        if self.sessions_path is None or self.blob_hash is None:
            raise RuntimeError(self.error or "JSONL preparation has no sealed artifact")
        if self.shard_path is None:
            raise ValueError("JSONL preparation has no row shard")
        self.verify_files(full=False)
        shard = open_session_shard(self.shard_path)
        uri = f"file:{quote(str(self.sessions_path))}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
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
            shard_by_id = shard.by_session_id()
            for (
                _ordinal,
                session_id,
                metadata_json,
                message_ordinal,
                message_count,
                event_ordinal,
                event_count,
                attachment_ordinal,
                attachment_count,
            ) in conn.execute(
                "SELECT ordinal, session_id, metadata_json, message_ordinal, message_count, event_ordinal, event_count, attachment_ordinal, attachment_count "
                "FROM prepared_session ORDER BY ordinal"
            ):
                try:
                    shard_entry = shard_by_id[session_id]
                except KeyError as exc:
                    raise ValueError("JSONL preparation session is absent from row shard") from exc
                metadata = json.loads(metadata_json)
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
                metadata["messages"] = []
                metadata["session_events"] = []
                metadata["attachments"] = []
                session = ParsedSession.model_validate(metadata)
                yield session.model_copy(
                    update={
                        "messages": SqliteMessageSink(self.sessions_path, message_ordinal, count=message_count),
                        "session_events": SqliteSessionEventSink(self.sessions_path, event_ordinal, count=event_count),
                        "attachments": SqliteAttachmentSink(
                            self.sessions_path, attachment_ordinal, count=attachment_count
                        ),
                    }
                )

    def session_sequence(self) -> PreparedSessionSequence:
        """Expose a sealed cohort without retaining its parsed sessions in Python."""
        if self.sessions_path is None:
            raise RuntimeError(self.error or "JSONL preparation has no sealed artifact")
        self.verify_files(full=False)
        uri = f"file:{quote(str(self.sessions_path))}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
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
        """Read one sealed session through the artifact's unique identity index."""
        if self.sessions_path is None or self.blob_hash is None or self.shard_path is None:
            raise RuntimeError(self.error or "JSONL preparation has no sealed artifact")
        self.verify_files(full=False)
        shard = _shard if _shard is not None else open_session_shard(self.shard_path)
        uri = f"file:{quote(str(self.sessions_path))}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
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
                "FROM prepared_session WHERE session_id = ? LIMIT 2",
                (session_id,),
            ).fetchall()
            if len(rows) != 1:
                raise KeyError(session_id)
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
            try:
                shard_entry = shard.by_session_id()[stored_id]
            except KeyError as exc:
                raise ValueError("JSONL preparation session is absent from row shard") from exc
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
            session = ParsedSession.model_validate(metadata)
            return session.model_copy(
                update={
                    "messages": SqliteMessageSink(self.sessions_path, message_ordinal, count=message_count),
                    "session_events": SqliteSessionEventSink(self.sessions_path, event_ordinal, count=event_count),
                    "attachments": SqliteAttachmentSink(self.sessions_path, attachment_ordinal, count=attachment_count),
                }
            )


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

    def iter_session_ids(self) -> Iterator[str]:
        """Stream canonical archive IDs from the artifact's unique SQLite index."""
        if self.artifact.sessions_path is None:
            raise RuntimeError(self.artifact.error or "JSONL preparation has no sealed artifact")
        self.artifact.verify_files(full=False)
        uri = f"file:{quote(str(self.artifact.sessions_path))}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            for (session_id,) in conn.execute("SELECT session_id FROM prepared_session ORDER BY session_id"):
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
        return next(islice(self.artifact.iter_sessions(), index, index + 1))


def _write_artifact(
    store: SqliteMessageStore,
    source_hash: str,
    sessions: Iterable[ParsedSession],
    *,
    enrichment_digest: str | None,
    enrichment_index_path: str | None,
) -> None:
    conn = store.conn
    try:
        _create_artifact_tables(conn)
        count = 0
        for ordinal, session in enumerate(sessions):
            _append_artifact_session(store, ordinal, session)
            count += 1
        _seal_artifact(conn, source_hash, count, enrichment_digest, enrichment_index_path)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def _create_artifact_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE prepared_session (ordinal INTEGER PRIMARY KEY, session_id TEXT NOT NULL UNIQUE, metadata_json TEXT NOT NULL, message_ordinal INTEGER NOT NULL, message_count INTEGER NOT NULL, event_ordinal INTEGER NOT NULL, event_count INTEGER NOT NULL, attachment_ordinal INTEGER NOT NULL, attachment_count INTEGER NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE artifact_seal (version INTEGER NOT NULL, source_hash TEXT NOT NULL, "
        "session_count INTEGER NOT NULL, enrichment_digest TEXT, enrichment_index_path TEXT)"
    )


def _append_artifact_session(store: SqliteMessageStore, ordinal: int, session: ParsedSession) -> None:
    conn = store.conn
    source_messages: object = session.messages
    messages: SqliteMessageSink
    if isinstance(source_messages, SqliteMessageSink) and source_messages.path == store.path:
        messages = source_messages
    else:
        messages = store.new_sink()
        messages.extend(session.messages)
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
        session.unit_accounting.model_dump(mode="json") if session.unit_accounting is not None else None
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
    conn.execute(
        "INSERT INTO artifact_seal VALUES (?, ?, ?, ?, ?)",
        (_ARTIFACT_VERSION, source_hash, count, enrichment_digest, enrichment_index_path),
    )


def prepare_jsonl_blob(
    blob_path: str,
    source_path: str,
    provider_value: str,
    fallback_id: str,
    *,
    is_stream: bool,
    shard_directory: str,
    sidecar_resolver: SidecarResolver | None = None,
    prepare_sessions: Callable[[list[ParsedSession]], list[ParsedSession]] | None = None,
    prepare_session: Callable[[ParsedSession], ParsedSession] | None = None,
    preparation_dependency: Callable[[], tuple[str | None, str | None]] | None = None,
    parse_prefix_size: int | None = None,
    prepare_records: Callable[[Iterable[JSONValue]], Iterable[JSONValue]] | None = None,
    classify_grok_export: Callable[[JSONValue], bool] | None = None,
    classify_generic_object: Callable[[dict[str, JSONValue], Sequence[JSONValue]], bool] | None = None,
    classify_hermes_object: Callable[[dict[str, JSONValue], Sequence[JSONValue]], bool] | None = None,
    classify_chatgpt_object: Callable[[dict[str, object]], bool] | None = None,
    classify_claude_design_object: Callable[[dict[str, JSONValue], Sequence[JSONValue]], bool] | None = None,
    classify_claude_ai_object: Callable[[dict[str, JSONValue], Sequence[JSONValue]], bool] | None = None,
    classify_drive_chunked_object: Callable[[dict[str, JSONValue]], bool] | None = None,
    classify_hermes_atif_object: Callable[[dict[str, JSONValue]], bool] | None = None,
    classify_gemini_object: Callable[[dict[str, JSONValue], Sequence[JSONValue]], bool] | None = None,
    classify_otel_object: Callable[[dict[str, JSONValue]], bool] | None = None,
    classify_bundle_members: Callable[[Sequence[JSONValue]], bool] | None = None,
    attempt_directory: Path | None = None,
    source_sha256: str | None = None,
    strict_jsonl_records: bool = False,
) -> PreparedJsonl:
    """Parse and seal one source without transferring a parsed tree over IPC.

    Every sealed session is admitted: the positive-conversational-evidence
    rule (``require_positive_conversational_evidence``) runs here, before any
    ``prepare_session``/``prepare_sessions`` callback, on every provider
    branch. A caller consuming the artifact does not apply it again.

    ``source_sha256`` is the digest of ``blob_path`` when the caller already
    read it to decide how to prepare it; the seal then binds that decision's
    bytes. ``strict_jsonl_records`` refuses a complete JSONL record that does
    not decode for every provider, as live acquisition does, instead of only
    for ``Provider.UNKNOWN``.
    """
    directory = Path(shard_directory)
    directory.mkdir(parents=True, exist_ok=True)
    artifact_directory = attempt_directory if attempt_directory is not None else directory
    if attempt_directory is not None:
        artifact_directory.mkdir(parents=True, exist_ok=True)
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
        store = SqliteMessageStore(sessions_path)
        before_hash = source_sha256 if source_sha256 is not None else _source_digest(source)
        record_container: str | None = None
        stream_prefix: str | None = None
        bundle_count = 0
        bundle_browser_captures = True
        bundle_witnesses: list[JSONValue] = []
        generic_envelope: dict[str, JSONValue] | None = None
        hermes_envelope: dict[str, JSONValue] | None = None
        design_envelope: dict[str, JSONValue] | None = None
        claude_ai_envelope: dict[str, JSONValue] | None = None
        claude_ai_arrays: tuple[str, ...] = ()
        drive_chunked: tuple[dict[str, JSONValue], str] | None = None
        atif: tuple[dict[str, JSONValue], bool] | None = None
        otel: tuple[dict[str, JSONValue], str, tuple[otel_genai.OtelSpanIndex, str | None]] | None = None
        chatgpt_envelope: dict[str, object] | None = None
        chatgpt_mapping: ChatGPTNodeMapping | None = None
        gemini_envelope: dict[str, JSONValue] | None = None
        gemini_sidecar_scope: RetainedSidecarScope | None = None
        grok_count: int | None = None
        if not is_stream and provider is Provider.CHATGPT and Path(source_path).name.lower().endswith(".json"):
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
        if (
            not is_stream
            and provider is Provider.GEMINI_CLI
            and (prepare_sessions is None or classify_gemini_object is not None)
            and (prepare_records is None or classify_gemini_object is not None)
            and Path(source_path).name.lower().endswith(".json")
        ):
            store.conn.execute(
                "CREATE TABLE gemini_raw_message (ordinal INTEGER PRIMARY KEY, message_json TEXT NOT NULL)"
            )
            future_wire_type = False
            with source.open("rb") as handle:
                for ordinal, item in enumerate(ijson.items(handle, "messages.item")):
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
        # Cohort callbacks may inspect or rewrite the entire parse result.
        # The direct worker route can publish independent bundle members.
        if (
            not is_stream
            and provider is Provider.HERMES
            and (prepare_sessions is None or classify_hermes_object is not None)
            and (prepare_records is None or classify_hermes_object is not None)
            and Path(source_path).name.lower().endswith(".json")
        ):
            with source.open("rb") as handle:
                hermes_envelope = hermes_snapshot_envelope(handle)
            if hermes_envelope is not None and (
                hermes_state.looks_like_state_db_payload(hermes_envelope)
                or hermes_verification.looks_like_verification_evidence_db_payload(hermes_envelope)
                or hermes_spans.looks_like_atif_payload(hermes_envelope)
            ):
                hermes_envelope = None
        if (
            not is_stream
            and provider is Provider.GROK
            and (prepare_sessions is None or classify_grok_export is not None)
            and (prepare_records is None or classify_grok_export is not None)
            and Path(source_path).name.lower().endswith(".json")
        ):
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
        if (
            not is_stream
            and provider in BUNDLE_PROVIDERS
            and prepare_sessions is None
            and (prepare_records is None or classify_bundle_members is not None)
            and Path(source_path).name.lower().endswith(".json")
        ):
            with source.open("rb") as handle:
                record_container = json_record_container(handle)
            if record_container is not None:

                def observe_bundle_member(index: int, shape: JSONValue, witness: JSONValue | None) -> None:
                    nonlocal bundle_browser_captures
                    bundle_browser_captures = bundle_browser_captures and browser_capture.looks_like(shape)
                    if witness is not None:
                        bundle_witnesses.append(witness)

                with source.open("rb") as handle:
                    scanned = scan_container_members(
                        handle,
                        record_container,
                        shape_keys=_BROWSER_CAPTURE_SHAPE_KEYS,
                        witnesses=64 if classify_bundle_members is not None else 0,
                        on_member=observe_bundle_member,
                    )
                if scanned is not None:
                    stream_prefix = record_container
                    bundle_count = scanned
        if (
            not is_stream
            and provider in {Provider.DRIVE, Provider.GEMINI, Provider.UNKNOWN}
            and (prepare_sessions is None or classify_generic_object is not None)
            and (prepare_records is None or classify_generic_object is not None)
            and Path(source_path).name.lower().endswith(".json")
        ):
            with source.open("rb") as handle:
                candidate = generic_message_object_envelope(handle)
            asserted_id = candidate.get("id") if candidate is not None else None
            if candidate is not None and isinstance(asserted_id, str) and asserted_id.strip():
                generic_envelope = candidate
        if (
            not is_stream
            and provider is Provider.CLAUDE_DESIGN
            and (prepare_sessions is None or classify_claude_design_object is not None)
            and (prepare_records is None or classify_claude_design_object is not None)
            and Path(source_path).name.lower().endswith(".json")
            and record_container is None
        ):
            with source.open("rb") as handle:
                design_envelope = claude_design_object_envelope(handle)
        if (
            not is_stream
            and provider is Provider.CLAUDE_AI
            and (prepare_sessions is None or classify_claude_ai_object is not None)
            and (prepare_records is None or classify_claude_ai_object is not None)
            and Path(source_path).name.lower().endswith(".json")
            and record_container is None
        ):
            with source.open("rb") as handle:
                claude_ai_object = claude_ai_object_envelope(handle)
            if claude_ai_object is not None:
                claude_ai_envelope, claude_ai_arrays = claude_ai_object
        if (
            not is_stream
            and provider in {Provider.DRIVE, Provider.GEMINI}
            and (prepare_sessions is None or classify_drive_chunked_object is not None)
            and (prepare_records is None or classify_drive_chunked_object is not None)
            and Path(source_path).name.lower().endswith(".json")
            and generic_envelope is None
        ):
            with source.open("rb") as handle:
                drive_chunked = drive_chunked_prompt_envelope(handle)
        if (
            not is_stream
            and provider is Provider.HERMES
            and (prepare_sessions is None or classify_hermes_atif_object is not None)
            and (prepare_records is None or classify_hermes_atif_object is not None)
            and Path(source_path).name.lower().endswith(".json")
            and hermes_envelope is None
        ):
            with source.open("rb") as handle:
                atif = _hermes_atif_envelope(handle)
            if atif is not None and atif[1]:
                with source.open("rb") as handle:
                    if not _spill_atif_subagents(handle, store.conn):
                        atif = None
        if (
            not is_stream
            and provider is Provider.OTEL_GENAI
            and (prepare_sessions is None or classify_otel_object is not None)
            and (prepare_records is None or classify_otel_object is not None)
            and Path(source_path).name.lower().endswith(".json")
        ):
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
            sample = tuple(
                json.loads(row[0])
                for row in store.conn.execute("SELECT message_json FROM gemini_raw_message ORDER BY ordinal LIMIT 64")
            )
            gemini_admitted = (
                classify_gemini_object(gemini_envelope, sample) if classify_gemini_object is not None else True
            )
            gemini_session = None
            if gemini_admitted:
                gemini_records = (
                    json.loads(row[0])
                    for row in store.conn.execute("SELECT message_json FROM gemini_raw_message ORDER BY ordinal")
                )
                gemini_session = local_agent.parse_gemini_cli_records(
                    gemini_envelope,
                    gemini_records,
                    fallback_id,
                    messages=store.new_sink(),
                    session_events=store.new_event_sink(),
                )
                admitted = local_agent.parse_gemini_cli(gemini_envelope, fallback_id)
                gemini_session = gemini_session.model_copy(update={"unit_accounting": admitted.unit_accounting})
                if gemini_sidecar_scope is not None and gemini_sidecar_scope.available:
                    index = GeminiToolOutputIndex(store.conn)
                    for row in store.conn.execute("SELECT message_json FROM gemini_raw_message ORDER BY ordinal"):
                        index.observe(json.loads(row[0]))
                    for outcome in index.join(gemini_sidecar_scope):
                        gemini_session.session_events.append(local_agent.gemini_sidecar_event(outcome))
                    for position in range(len(gemini_session.messages)):
                        message = gemini_session.messages[position]
                        updated_blocks = [
                            block.model_copy(update={"text": replacement})
                            if block.type is BlockType.TOOL_RESULT
                            and block.tool_id is not None
                            and (replacement := index.replacement_for(block.tool_id)) is not None
                            else block
                            for block in message.blocks
                        ]
                        if updated_blocks != message.blocks:
                            gemini_session.messages[position] = message.model_copy(update={"blocks": updated_blocks})
                    index.close()
            store.conn.execute("DROP TABLE gemini_raw_message")
            if gemini_session is not None and require_positive_conversational_evidence(
                [gemini_session], provider=provider, source_path=source_path
            ):
                if prepare_sessions is not None:
                    selected = prepare_sessions([gemini_session])
                    if len(selected) > 1:
                        raise ValueError("Gemini CLI finalizer expanded one session")
                    gemini_session = selected[0] if selected else None
                elif prepare_session is not None:
                    gemini_session = prepare_session(gemini_session)
            else:
                gemini_session = None
            session_count = 0
            if gemini_session is not None:
                gemini_session.content_hash = session_content_hash(gemini_session)
                append_session_to_shard(shard_builder, gemini_session)
                _append_artifact_session(store, session_count, gemini_session)
                session_count += 1
            after_hash = _source_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency() if preparation_dependency is not None else (None, None)
            )
            _seal_artifact(store.conn, after_hash, session_count, enrichment_digest, enrichment_index_path)
            store.conn.commit()
            shard_path = shard_builder.seal().path
            shard_builder = None
        elif chatgpt_envelope is not None:
            assert chatgpt_mapping is not None
            _create_artifact_tables(store.conn)
            shard_builder = SessionShardBuilder(artifact_directory / f"shard-{uuid.uuid4().hex}.db")
            chatgpt_admitted = (
                classify_chatgpt_object(chatgpt_envelope) if classify_chatgpt_object is not None else True
            )
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
            if session is not None and not require_positive_conversational_evidence(
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
                    if prepare_sessions is not None:
                        selected = prepare_sessions([session])
                        if len(selected) > 1:
                            raise ValueError("ChatGPT object finalizer expanded one session")
                        session = selected[0] if selected else None
                    elif prepare_session is not None:
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
                    "(SELECT message_ordinal FROM prepared_session)"
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
            after_hash = _source_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency() if preparation_dependency is not None else (None, None)
            )
            _seal_artifact(store.conn, after_hash, session_count, enrichment_digest, enrichment_index_path)
            store.conn.commit()
            shard_path = shard_builder.seal().path
            shard_builder = None
        elif hermes_envelope is not None:
            _create_artifact_tables(store.conn)
            shard_builder = SessionShardBuilder(artifact_directory / f"shard-{uuid.uuid4().hex}.db")
            hermes_admitted = True
            if classify_hermes_object is not None:
                with source.open("rb") as handle:
                    sample = tuple(
                        islice(
                            (
                                cast(JSONValue, normalize_ijson_stdlib_numbers(item))
                                for item in ijson.items(handle, "messages.item")
                            ),
                            64,
                        )
                    )
                hermes_admitted = classify_hermes_object(hermes_envelope, sample)
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
                    )
            session_count = 0
            if session is not None and require_positive_conversational_evidence(
                [session], provider=provider, source_path=source_path
            ):
                if prepare_sessions is not None:
                    selected = prepare_sessions([session])
                    if len(selected) > 1:
                        raise ValueError("Hermes snapshot finalizer expanded one session")
                    session = selected[0] if selected else None
                elif prepare_session is not None:
                    session = prepare_session(session)
            else:
                session = None
            if session is not None:
                session.content_hash = session_content_hash(session)
                append_session_to_shard(shard_builder, session)
                _append_artifact_session(store, session_count, session)
                session_count += 1
            after_hash = _source_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency() if preparation_dependency is not None else (None, None)
            )
            _seal_artifact(store.conn, after_hash, session_count, enrichment_digest, enrichment_index_path)
            store.conn.commit()
            shard_path = shard_builder.seal().path
            shard_builder = None
        elif generic_envelope is not None:
            _create_artifact_tables(store.conn)
            shard_builder = SessionShardBuilder(artifact_directory / f"shard-{uuid.uuid4().hex}.db")
            generic_admitted = True
            if classify_generic_object is not None:
                with source.open("rb") as handle:
                    sample = tuple(
                        islice(
                            (
                                cast(JSONValue, normalize_ijson_stdlib_numbers(item))
                                for item in ijson.items(handle, "messages.item")
                            ),
                            64,
                        )
                    )
                generic_admitted = classify_generic_object(generic_envelope, sample)
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
            if session is not None and require_positive_conversational_evidence(
                [session], provider=provider, source_path=source_path
            ):
                if prepare_sessions is not None:
                    selected = prepare_sessions([session])
                    if len(selected) > 1:
                        raise ValueError("generic object finalizer expanded one session")
                    session = selected[0] if selected else None
                elif prepare_session is not None:
                    session = prepare_session(session)
            else:
                session = None
            if session is not None:
                session.content_hash = session_content_hash(session)
                append_session_to_shard(shard_builder, session)
                _append_artifact_session(store, session_count, session)
                session_count += 1
            after_hash = _source_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency() if preparation_dependency is not None else (None, None)
            )
            _seal_artifact(store.conn, after_hash, session_count, enrichment_digest, enrichment_index_path)
            store.conn.commit()
            shard_path = shard_builder.seal().path
            shard_builder = None
        elif design_envelope is not None:
            _create_artifact_tables(store.conn)
            shard_builder = SessionShardBuilder(artifact_directory / f"shard-{uuid.uuid4().hex}.db")
            design_admitted = True
            if classify_claude_design_object is not None:
                with source.open("rb") as handle:
                    sample = tuple(
                        islice(
                            (
                                cast(JSONValue, normalize_ijson_stdlib_numbers(item))
                                for item in ijson.items(handle, "messages.item")
                            ),
                            64,
                        )
                    )
                design_admitted = classify_claude_design_object(design_envelope, sample)
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
            if session is not None and require_positive_conversational_evidence(
                [session], provider=provider, source_path=source_path
            ):
                if prepare_sessions is not None:
                    selected = prepare_sessions([session])
                    if len(selected) > 1:
                        raise ValueError("Claude Design object finalizer expanded one session")
                    session = selected[0] if selected else None
                elif prepare_session is not None:
                    session = prepare_session(session)
            else:
                session = None
            if session is not None:
                session.content_hash = session_content_hash(session)
                append_session_to_shard(shard_builder, session)
                _append_artifact_session(store, session_count, session)
                session_count += 1
            after_hash = _source_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency() if preparation_dependency is not None else (None, None)
            )
            _seal_artifact(store.conn, after_hash, session_count, enrichment_digest, enrichment_index_path)
            store.conn.commit()
            shard_path = shard_builder.seal().path
            shard_builder = None
        elif claude_ai_envelope is not None:
            _create_artifact_tables(store.conn)
            shard_builder = SessionShardBuilder(artifact_directory / f"shard-{uuid.uuid4().hex}.db")
            claude_ai_admitted = True
            if classify_claude_ai_object is not None:
                with source.open("rb") as handle:
                    sample = tuple(
                        islice(
                            (
                                cast(JSONValue, normalize_ijson_stdlib_numbers(item))
                                for item in ijson.items(handle, "chat_messages.item")
                            ),
                            64,
                        )
                    )
                witness = dict(claude_ai_envelope)
                for key in claude_ai_arrays:
                    with source.open("rb") as handle:
                        witness[key] = list(islice(iter_root_array_items(handle, key), 64))
                claude_ai_admitted = classify_claude_ai_object(witness, sample)
            # The collecting route parses this document as a one-item bundle,
            # so its fallback identity carries that suffix.
            session = (
                _stream_claude_ai_object(source, claude_ai_envelope, claude_ai_arrays, store, f"{fallback_id}-0")
                if claude_ai_admitted
                else None
            )
            session_count = 0
            if session is not None and require_positive_conversational_evidence(
                [session], provider=provider, source_path=source_path
            ):
                if prepare_sessions is not None:
                    selected = prepare_sessions([session])
                    if len(selected) > 1:
                        raise ValueError("Claude AI object finalizer expanded one session")
                    session = selected[0] if selected else None
                elif prepare_session is not None:
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
                    f"DELETE FROM {table} WHERE session_ordinal NOT IN (SELECT {column} FROM prepared_session)"
                )
            after_hash = _source_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency() if preparation_dependency is not None else (None, None)
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
                with source.open("rb") as handle:
                    for item in ijson.items(handle, f"{chunk_prefix}.item"):
                        yield normalize_ijson_stdlib_numbers(item)

            drive_admitted = True
            if classify_drive_chunked_object is not None:
                chunk_sample: list[JSONValue] = [cast(JSONValue, item) for item in islice(drive_chunks(), 64)]
                witness = {key: value for key, value in drive_envelope.items() if not key.startswith("__")}
                if chunk_prefix == "chunks":
                    witness["chunks"] = chunk_sample
                else:
                    prompt = witness.get("chunkedPrompt")
                    witness["chunkedPrompt"] = {**(prompt if isinstance(prompt, dict) else {}), "chunks": chunk_sample}
                drive_admitted = classify_drive_chunked_object(witness)
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
                )
            session_count = 0
            if session is not None and require_positive_conversational_evidence(
                [session], provider=provider, source_path=source_path
            ):
                if prepare_sessions is not None:
                    selected = prepare_sessions([session])
                    if len(selected) > 1:
                        raise ValueError("chunked prompt finalizer expanded one session")
                    session = selected[0] if selected else None
                elif prepare_session is not None:
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
                    f"DELETE FROM {table} WHERE session_ordinal NOT IN (SELECT {column} FROM prepared_session)"
                )
            after_hash = _source_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency() if preparation_dependency is not None else (None, None)
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
                        yield cast(JSONValue, normalize_ijson_stdlib_numbers(item))

            atif_admitted = True
            if classify_hermes_atif_object is not None:
                atif_witness: dict[str, JSONValue] = {**atif_envelope, "steps": list(islice(atif_steps(), 64))}
                if atif_has_subagents:
                    atif_witness["subagent_trajectories"] = _atif_subagent_witness(store.conn)
                atif_admitted = classify_hermes_atif_object(atif_witness)
            atif_sessions: list[ParsedSession] = []
            if atif_admitted:
                # The dispatch route proves the whole document against the
                # Hermes discriminator scan, which reads only the envelope's
                # and each step's ``type``/``kind``. The steps are observed
                # as the parser consumes them, so the streamed carrier gets the
                # same conservation proof without holding the document.
                unknown_steps: list[JSONValue] = []

                def observed_steps() -> Iterator[JSONValue]:
                    for step in atif_steps():
                        if not unknown_steps and isinstance(step, dict):
                            discriminators: dict[str, JSONValue] = {
                                key: step[key] for key in ("type", "kind") if key in step
                            }
                            if hermes_unknown_wire_type({"steps": [discriminators]}) is not None:
                                unknown_steps.append(discriminators)
                        yield step

                steps = observed_steps()
                atif_sessions = hermes_spans.parse_atif_stream(
                    atif_envelope,
                    steps,
                    _atif_subagents(store.conn) if atif_has_subagents else (),
                    fallback_id,
                    profile_root=hermes_identity.profile_root_for_artifact(Path(source_path)),
                    new_events=store.new_event_sink,
                )
                if atif_sessions:
                    for _ in steps:  # a parser that stopped early still owes the rest a scan
                        pass
                    atif_sessions = admit_parsed_sessions(
                        "hermes", {**atif_envelope, "steps": unknown_steps}, atif_sessions
                    )
            if atif_has_subagents:
                _drop_atif_subagents(store.conn)
            atif_sessions = require_positive_conversational_evidence(
                atif_sessions, provider=provider, source_path=source_path
            )
            if prepare_sessions is not None:
                atif_sessions = prepare_sessions(atif_sessions)
            elif prepare_session is not None:
                atif_sessions = [prepare_session(session) for session in atif_sessions]
            session_count = 0
            for session in atif_sessions:
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
                    f"DELETE FROM {table} WHERE session_ordinal NOT IN (SELECT {column} FROM prepared_session)"
                )
            after_hash = _source_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency() if preparation_dependency is not None else (None, None)
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
            otel_admitted = otel_index.normalizable and (
                classify_otel_object is None or classify_otel_object({**otel_envelope, otel_root_key: []})
            )
            session_count = 0
            for session in (
                otel_index.sessions(new_messages=store.new_sink, new_events=store.new_event_sink)
                if otel_admitted
                else ()
            ):
                session = admit_parsed_sessions(provider.value.replace("-", "_"), otel_admission_payload, [session])[0]
                if not require_positive_conversational_evidence([session], provider=provider, source_path=source_path):
                    continue
                if prepare_sessions is not None:
                    selected = prepare_sessions([session])
                    if len(selected) > 1:
                        raise ValueError("OTLP per-session finalizer expanded one session")
                    if not selected:
                        continue
                    session = selected[0]
                elif prepare_session is not None:
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
                    f"DELETE FROM {table} WHERE session_ordinal NOT IN (SELECT {column} FROM prepared_session)"
                )
            after_hash = _source_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency() if preparation_dependency is not None else (None, None)
            )
            _seal_artifact(store.conn, after_hash, session_count, enrichment_digest, enrichment_index_path)
            store.conn.commit()
            shard_path = shard_builder.seal().path
            shard_builder = None
        elif grok_count is not None:
            _create_artifact_tables(store.conn)
            shard_builder = SessionShardBuilder(artifact_directory / f"shard-{uuid.uuid4().hex}.db")
            grok_admitted = (
                _classify_grok_witness(source, classify_grok_export) if classify_grok_export is not None else True
            )
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
                        if not require_positive_conversational_evidence(
                            [session], provider=provider, source_path=source_path
                        ):
                            continue
                        if prepare_sessions is not None:
                            selected = prepare_sessions([session])
                            if len(selected) > 1:
                                raise ValueError("Grok per-member finalizer expanded one session")
                            if not selected:
                                continue
                            session = selected[0]
                        elif prepare_session is not None:
                            session = prepare_session(session)
                        session.content_hash = session_content_hash(session)
                        append_session_to_shard(shard_builder, session)
                        _append_artifact_session(store, session_count, session)
                        session_count += 1
            if grok_admitted and member_index + 1 != grok_count:
                raise _SourceChangedDuringPreparationError("Grok conversation count changed during preparation")
            store.conn.execute("DROP TABLE grok_member_valid")
            after_hash = _source_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency() if preparation_dependency is not None else (None, None)
            )
            _seal_artifact(store.conn, after_hash, session_count, enrichment_digest, enrichment_index_path)
            store.conn.commit()
            shard_path = shard_builder.seal().path
            shard_builder = None
        elif stream_prefix is not None:
            _create_artifact_tables(store.conn)
            shard_builder = SessionShardBuilder(artifact_directory / f"shard-{uuid.uuid4().hex}.db")
            bundle_admitted = classify_bundle_members(bundle_witnesses) if classify_bundle_members is not None else True
            drift = BundleCandidateDrift()
            session_count = 0
            member_count = 0
            with source.open("rb") as handle:
                for bundle_index in (
                    iter_container_member_files(handle, stream_prefix, member_path) if bundle_admitted else ()
                ):
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
                        if not require_positive_conversational_evidence(
                            [session], provider=provider, source_path=source_path
                        ):
                            continue
                        if prepare_session is not None:
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
                    f"DELETE FROM {table} WHERE session_ordinal NOT IN (SELECT {column} FROM prepared_session)"
                )
            after_hash = _source_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            enrichment_digest, enrichment_index_path = (
                preparation_dependency() if preparation_dependency is not None else (None, None)
            )
            _seal_artifact(store.conn, after_hash, session_count, enrichment_digest, enrichment_index_path)
            store.conn.commit()
            shard_path = shard_builder.seal().path
            shard_builder = None
        else:
            with source.open("rb") as handle:
                record_input = (
                    _iter_prefix_lines(handle, parse_prefix_size) if parse_prefix_size is not None else handle
                )
                records = _iter_json_stream(
                    record_input,  # type: ignore[arg-type]
                    Path(source_path).name,
                    fail_on_decode_error=strict_jsonl_records or provider is Provider.UNKNOWN,
                )
                if prepare_records is not None:
                    records = prepare_records(records)
                if is_stream:
                    sessions = parse_stream_payload(
                        provider,
                        records,
                        fallback_id,
                        source_path=source_path,
                        message_sink_factory=store.new_sink,
                        event_sink_factory=store.new_event_sink,
                        sidecar_resolver=sidecar_resolver,
                    )
                else:
                    sessions = parse_payload(
                        provider,
                        list(records),
                        fallback_id,
                        source_path=source_path,
                        sidecar_resolver=sidecar_resolver,
                    )
            after_hash = _source_digest(source)
            if before_hash != after_hash:
                raise _SourceChangedDuringPreparationError("blob changed during worker preparation")
            sessions = require_positive_conversational_evidence(sessions, provider=provider, source_path=source_path)
            if prepare_sessions is not None:
                sessions = prepare_sessions(sessions)
            elif prepare_session is not None:
                sessions = [prepare_session(session) for session in sessions]
            for session in sessions:
                session.content_hash = session_content_hash(session)
            shard_path = prepare_session_shard(artifact_directory, sessions).path
            enrichment_digest, enrichment_index_path = (
                preparation_dependency() if preparation_dependency is not None else (None, None)
            )
            _write_artifact(
                store,
                after_hash,
                sessions,
                enrichment_digest=enrichment_digest,
                enrichment_index_path=enrichment_index_path,
            )
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
            # Every branch above admits its sessions before sealing.
            positive_evidence_filtered=True,
            attempt_directory=attempt_directory,
        )
        sealed = True
        return result
    except Exception as exc:
        if shard_builder is not None:
            shard_builder.abandon()
        if shard_path is not None:
            discard_session_shard(shard_path)
        retryable = isinstance(exc, (OSError, sqlite3.OperationalError, _SourceChangedDuringPreparationError))
        error_hash: str | None = None
        if before_hash is not None:
            try:
                after_error_hash = _source_digest(source)
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
        )
    finally:
        if store is not None:
            store.close()
        member_path.unlink(missing_ok=True)
        if not sealed:
            sessions_path.unlink(missing_ok=True)
