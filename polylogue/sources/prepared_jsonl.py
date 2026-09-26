"""Sealed, source-bound preparation for JSON and JSONL session captures."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO
from urllib.parse import quote

from polylogue.core.enums import Provider
from polylogue.core.identity_law import session_id as archive_session_id
from polylogue.core.json import JSONValue
from polylogue.core.sources import origin_from_provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import parse_payload, parse_stream_payload
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.prepared_message_sink import (
    SqliteMessageSink,
    SqliteMessageStore,
    SqliteSessionEventSink,
)
from polylogue.sources.sidecar_evidence import SidecarResolver
from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionWrite, prepare_session_shard
from polylogue.storage.sqlite.archive_tiers.write_shard import discard_session_shard, open_session_shard

_ARTIFACT_VERSION = 2


class _SourceChangedDuringPreparationError(ValueError):
    """The input revision changed while a worker was preparing it."""


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
            ) in conn.execute(
                "SELECT ordinal, session_id, metadata_json, message_ordinal, message_count, event_ordinal, event_count "
                "FROM prepared_session ORDER BY ordinal"
            ):
                if session_id not in shard_by_id:
                    raise ValueError("JSONL preparation session is absent from row shard")
                metadata = json.loads(metadata_json)
                physical_count = conn.execute(
                    "SELECT COUNT(*) FROM prepared_message WHERE session_ordinal = ?", (message_ordinal,)
                ).fetchone()[0]
                if physical_count != message_count or physical_count != shard_by_id[session_id].message_row_count:
                    raise ValueError("JSONL preparation message count disagrees with row shard")
                physical_events = conn.execute(
                    "SELECT COUNT(*) FROM prepared_event WHERE session_ordinal = ?", (event_ordinal,)
                ).fetchone()[0]
                if physical_events != event_count:
                    raise ValueError("JSONL preparation event count changed")
                for attachment in metadata.get("attachments", []):
                    encoded = attachment.pop("_prepared_inline_bytes", None)
                    if encoded is not None:
                        attachment["inline_bytes"] = base64.b64decode(encoded, validate=True)
                metadata["messages"] = []
                metadata["session_events"] = []
                session = ParsedSession.model_validate(metadata)
                yield session.model_copy(
                    update={
                        "messages": SqliteMessageSink(self.sessions_path, message_ordinal, count=message_count),
                        "session_events": SqliteSessionEventSink(self.sessions_path, event_ordinal, count=event_count),
                    }
                )

    def load_sessions(self) -> list[ParsedSession]:
        """Compatibility adapter for publication callers that consume a cohort."""
        return list(self.iter_sessions())


def _write_artifact(
    store: SqliteMessageStore,
    source_hash: str,
    sessions: list[ParsedSession],
    *,
    enrichment_digest: str | None,
    enrichment_index_path: str | None,
) -> None:
    conn = store.conn
    try:
        conn.execute(
            "CREATE TABLE prepared_session (ordinal INTEGER PRIMARY KEY, session_id TEXT NOT NULL UNIQUE, metadata_json TEXT NOT NULL, message_ordinal INTEGER NOT NULL, message_count INTEGER NOT NULL, event_ordinal INTEGER NOT NULL, event_count INTEGER NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE artifact_seal (version INTEGER NOT NULL, source_hash TEXT NOT NULL, "
            "session_count INTEGER NOT NULL, enrichment_digest TEXT, enrichment_index_path TEXT)"
        )
        for ordinal, session in enumerate(sessions):
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
            metadata = session.model_dump(mode="json", exclude={"messages", "session_events"})
            metadata["content_hash"] = session.content_hash
            metadata["unit_accounting"] = (
                session.unit_accounting.model_dump(mode="json") if session.unit_accounting is not None else None
            )
            metadata["provider_session_aliases"] = session.provider_session_aliases
            metadata["created_at_provenance"] = session.created_at_provenance
            metadata["updated_at_provenance"] = session.updated_at_provenance
            metadata["attachments"] = [
                {
                    **attachment,
                    "message_position": source.message_position,
                    "message_variant_index": source.message_variant_index,
                    "owner_coordinate": (
                        asdict(source.owner_coordinate) if source.owner_coordinate is not None else None
                    ),
                    "precomputed_blob": source.precomputed_blob,
                    "_prepared_inline_bytes": (
                        base64.b64encode(source.inline_bytes).decode("ascii")
                        if source.inline_bytes is not None
                        else None
                    ),
                }
                for attachment, source in zip(metadata["attachments"], session.attachments, strict=True)
            ]
            session_id = archive_session_id(
                origin_from_provider(session.source_name).value, session.provider_session_id
            )
            conn.execute(
                "INSERT INTO prepared_session VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ordinal,
                    session_id,
                    json.dumps(metadata, ensure_ascii=False),
                    messages.session_ordinal,
                    len(messages),
                    events.session_ordinal,
                    len(events),
                ),
            )
        conn.execute(
            "INSERT INTO artifact_seal VALUES (?, ?, ?, ?, ?)",
            (_ARTIFACT_VERSION, source_hash, len(sessions), enrichment_digest, enrichment_index_path),
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


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
    preparation_dependency: Callable[[], tuple[str | None, str | None]] | None = None,
    parse_prefix_size: int | None = None,
    prepare_records: Callable[[Iterable[JSONValue]], Iterable[JSONValue]] | None = None,
) -> PreparedJsonl:
    """Parse and seal one source without transferring a parsed tree over IPC."""
    directory = Path(shard_directory)
    directory.mkdir(parents=True, exist_ok=True)
    sessions_path = directory / f"prepared-{uuid.uuid4().hex}.db"
    shard_path: Path | None = None
    store: SqliteMessageStore | None = None
    sealed = False
    source = Path(blob_path)
    before_hash: str | None = None
    try:
        provider = Provider.from_string(provider_value)
        store = SqliteMessageStore(sessions_path)
        before_hash = _source_digest(source)
        with source.open("rb") as handle:
            record_input = _iter_prefix_lines(handle, parse_prefix_size) if parse_prefix_size is not None else handle
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
        for session in sessions:
            session.content_hash = session_content_hash(session)
        shard_path = prepare_session_shard(directory, sessions).path
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
        )
        sealed = True
        return result
    except Exception as exc:
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
