"""Sealed, source-bound preparation for JSON and JSONL session captures."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import BinaryIO, cast, overload
from urllib.parse import quote

import ijson

from polylogue.core.enums import BlockType, Provider
from polylogue.core.identity_law import session_id as archive_session_id
from polylogue.core.json import JSONValue
from polylogue.core.sources import origin_from_provider
from polylogue.logging import WARNING, emit
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.decoder_json import (
    _json_subtree,
    _skip_json_subtree,
    claude_design_object_envelope,
    generic_message_object_envelope,
    grok_export_item_count,
    hermes_snapshot_envelope,
    iter_grok_export_events,
    iter_json_container_records,
    json_record_container,
    normalize_ijson_stdlib_numbers,
)
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import (
    BUNDLE_PROVIDERS,
    iter_bundle_record_sessions,
    parse_generic_messages_stream,
    parse_payload,
    parse_stream_payload,
    require_positive_conversational_evidence,
)
from polylogue.sources.parsers import (
    browser_capture,
    chatgpt,
    grok,
    hermes_spans,
    hermes_state,
    hermes_verification,
    local_agent,
)
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.base_support import _unknown_wire_type
from polylogue.sources.parsers.claude.ai_parser import parse_design_stream
from polylogue.sources.prepared_message_sink import (
    ChatGPTNodeMapping,
    GeminiToolOutputIndex,
    ScratchSessionSpill,
    SqliteAttachmentSink,
    SqliteMessageSink,
    SqliteMessageStore,
    SqliteSessionEventSink,
    read_chatgpt_mapping_object,
)
from polylogue.sources.sidecar_evidence import RetainedSidecarScope, SidecarResolver
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


def _source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

    def verify(self, path: Path, *, full: bool) -> None:
        before = path.stat()
        if _file_identity(before) != self.identity:
            raise ValueError(f"prepared file identity changed: {path}")
        if full:
            if _source_digest(path) != self.sha256:
                raise ValueError(f"prepared file content changed: {path}")
            after = path.stat()
            if _file_identity(after) != self.identity:
                raise ValueError(f"prepared file changed during verification: {path}")

    @property
    def identity(self) -> tuple[int, int, int, int, int]:
        return self.device, self.inode, self.size, self.mtime_ns, self.ctime_ns


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

    def verify_files(self, *, full: bool) -> None:
        """Scan bytes before admission; recheck inode identity at publication."""
        if (
            self.sessions_path is None
            or self.shard_path is None
            or self.sessions_seal is None
            or self.shard_seal is None
        ):
            raise ValueError("JSONL preparation lacks closed-file seals")
        self.sessions_seal.verify(self.sessions_path, full=full)
        self.shard_seal.verify(self.shard_path, full=full)

    def discard(self) -> None:
        for prepared in self.prepared_writes:
            prepared.close()
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
    classify_grok_export: Callable[[int, bool], bool] | None = None,
    classify_generic_object: Callable[[dict[str, JSONValue], Sequence[JSONValue]], bool] | None = None,
    classify_hermes_object: Callable[[dict[str, JSONValue], Sequence[JSONValue]], bool] | None = None,
    classify_chatgpt_object: Callable[[dict[str, object]], bool] | None = None,
    classify_claude_design_object: Callable[[dict[str, JSONValue], Sequence[JSONValue]], bool] | None = None,
    classify_gemini_object: Callable[[dict[str, JSONValue], Sequence[JSONValue]], bool] | None = None,
    attempt_directory: Path | None = None,
) -> PreparedJsonl:
    """Parse and seal one source without transferring a parsed tree over IPC."""
    directory = Path(shard_directory)
    directory.mkdir(parents=True, exist_ok=True)
    artifact_directory = attempt_directory if attempt_directory is not None else directory
    if attempt_directory is not None:
        artifact_directory.mkdir(parents=True, exist_ok=True)
        if artifact_directory.parent != directory:
            raise ValueError("prepared attempt directory must be a direct child of the shard directory")
    sessions_path = artifact_directory / f"prepared-{uuid.uuid4().hex}.db"
    shard_path: Path | None = None
    store: SqliteMessageStore | None = None
    shard_builder: SessionShardBuilder | None = None
    sealed = False
    source = Path(blob_path)
    before_hash: str | None = None
    try:
        provider = Provider.from_string(provider_value)
        store = SqliteMessageStore(sessions_path)
        before_hash = _source_digest(source)
        stream_prefix: str | None = None
        generic_envelope: dict[str, JSONValue] | None = None
        hermes_envelope: dict[str, JSONValue] | None = None
        design_envelope: dict[str, JSONValue] | None = None
        chatgpt_envelope: dict[str, object] | None = None
        chatgpt_mapping: ChatGPTNodeMapping | None = None
        gemini_envelope: dict[str, JSONValue] | None = None
        gemini_sidecar_scope: RetainedSidecarScope | None = None
        grok_count: int | None = None
        grok_positive_marker = False
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
                store.conn.execute("DROP TABLE chatgpt_node")
                store.conn.execute("DROP TABLE chatgpt_child")
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
            store.conn.execute("CREATE TABLE grok_member_valid (ordinal INTEGER PRIMARY KEY, valid INTEGER NOT NULL)")
            grok_probe_conn = store.conn

            def record_grok_member(index: int, valid: bool) -> None:
                grok_probe_conn.execute("INSERT INTO grok_member_valid VALUES (?, ?)", (index, int(valid)))

            def record_grok_marker(found: bool) -> None:
                nonlocal grok_positive_marker
                grok_positive_marker = found

            with source.open("rb") as handle:
                grok_count = grok_export_item_count(
                    handle,
                    on_item=record_grok_member,
                    on_positive_marker=record_grok_marker if classify_grok_export is not None else None,
                )
            if grok_count is None:
                store.conn.execute("DROP TABLE grok_member_valid")
        if (
            not is_stream
            and provider in BUNDLE_PROVIDERS
            and prepare_sessions is None
            and Path(source_path).name.lower().endswith(".json")
        ):
            with source.open("rb") as handle:
                stream_prefix = json_record_container(handle)
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
            and stream_prefix is None
        ):
            with source.open("rb") as handle:
                design_envelope = claude_design_object_envelope(handle)
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
            for table in ("chatgpt_node", "chatgpt_child", "chatgpt_entry"):
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
        elif grok_count is not None:
            _create_artifact_tables(store.conn)
            shard_builder = SessionShardBuilder(directory / f"shard-{uuid.uuid4().hex}.db")
            grok_admitted = (
                classify_grok_export(grok_count, grok_positive_marker) if classify_grok_export is not None else True
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
                        # The event probe leaves future wire types on the
                        # ordinary parser path. Admit this known outer record
                        # through the same wrapper without reloading responses.
                        admitted = grok.parse_conversation(
                            {"conversation": {}, "responses": []}, session.provider_session_id
                        )
                        session = session.model_copy(update={"unit_accounting": admitted.unit_accounting})
                        if prepare_sessions is not None:
                            selected = prepare_sessions([session])
                            if len(selected) > 1:
                                raise ValueError("Grok per-member finalizer expanded one session")
                            if not selected:
                                continue
                            session = selected[0]
                        elif prepare_session is not None:
                            if not require_positive_conversational_evidence(
                                [session], provider=provider, source_path=source_path
                            ):
                                continue
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

            def bundle_records() -> Iterator[JSONValue]:
                with source.open("rb") as handle:
                    records: Iterable[JSONValue] = iter_json_container_records(handle, stream_prefix)
                    if prepare_records is not None:
                        records = prepare_records(records)
                    yield from records

            count = 0
            all_browser_captures = True
            for record in bundle_records():
                count += 1
                all_browser_captures = all_browser_captures and (
                    isinstance(record, dict) and browser_capture.looks_like(record)
                )
            _create_artifact_tables(store.conn)
            shard_builder = SessionShardBuilder(artifact_directory / f"shard-{uuid.uuid4().hex}.db")
            session_count = 0
            for session in iter_bundle_record_sessions(
                provider,
                bundle_records(),
                fallback_id,
                count=count,
                all_browser_captures=all_browser_captures,
                source_path=source_path,
                sidecar_resolver=sidecar_resolver,
            ):
                if not require_positive_conversational_evidence([session], provider=provider, source_path=source_path):
                    continue
                if prepare_session is not None:
                    session = prepare_session(session)
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
        else:
            with source.open("rb") as handle:
                record_input = (
                    _iter_prefix_lines(handle, parse_prefix_size) if parse_prefix_size is not None else handle
                )
                records = _iter_json_stream(
                    record_input,  # type: ignore[arg-type]
                    Path(source_path).name,
                    fail_on_decode_error=provider is Provider.UNKNOWN,
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
            if prepare_sessions is not None:
                sessions = prepare_sessions(sessions)
            elif prepare_session is not None:
                sessions = [
                    prepare_session(session)
                    for session in sessions
                    if require_positive_conversational_evidence([session], provider=provider, source_path=source_path)
                ]
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
            positive_evidence_filtered=stream_prefix is not None
            or chatgpt_envelope is not None
            or gemini_envelope is not None
            or generic_envelope is not None
            or hermes_envelope is not None
            or design_envelope is not None
            or (grok_count is not None and classify_grok_export is not None)
            or (prepare_sessions is None and prepare_session is not None),
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
        return PreparedJsonl(error_hash, None, None, f"{type(exc).__name__}: {exc}"[:500], deferred=retryable)
    finally:
        if store is not None:
            store.close()
        if not sealed:
            sessions_path.unlink(missing_ok=True)
