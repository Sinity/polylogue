"""Bounded worker-side composition of retained JSONL revision artifacts."""

from __future__ import annotations

import hashlib
import pickle
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import Provider
from polylogue.core.iterator_lifetime import settled_iterator
from polylogue.core.message_owner import MessageOwnerCoordinate
from polylogue.pipeline.ids import message_content_identity, session_content_hash
from polylogue.sources.chunk_positions import ChunkPositions
from polylogue.sources.dispatch import merge_parsed_session_chunks
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.sources.parsers.claude.code_parser import order_working_directories, relocated_cwds_of
from polylogue.sources.prepared_jsonl import PreparedJsonl, _write_artifact
from polylogue.sources.prepared_message_sink import (
    SqliteAttachmentSink,
    SqliteMessageStore,
    SqliteSessionEventSink,
    _prepared_reader,
)
from polylogue.storage.blob_publication import BlobPublicationSourceRead, RetainedAttachmentSourceRead
from polylogue.storage.io_phase_metrics import connection_cursor
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_shard

_CLAUDE_SUMMARIES: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("claude_session_environment", ("entrypoints", "cli_versions", "permission_modes", "prompt_sources"), False),
    ("claude_parse_coverage", ("sidecar_seen", "sidecar_persisted", "empty_dropped_by_record_type"), True),
)


def prepared_cohort_source_hash(ordered: Sequence[tuple[str, PreparedJsonl]]) -> str:
    """Hash the exact accepted raw order and each artifact's source revision."""
    if not ordered:
        raise ValueError("retained cohort is empty")
    digest = hashlib.sha256(b"polylogue-prepared-cohort-v1\x00")
    for raw_id, artifact in ordered:
        if not raw_id or artifact.blob_hash is None:
            raise ValueError("retained cohort has an unbound raw dependency")
        for value in (raw_id, artifact.blob_hash):
            encoded = value.encode("utf-8", errors="surrogatepass")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _count_claude_summary(
    store: SqliteMessageStore,
    event: ParsedSessionEvent,
    payload_keys: tuple[str, ...],
) -> None:
    for payload_key in payload_keys:
        values = event.payload.get(payload_key, {})
        if not isinstance(values, dict):
            continue
        for name, count in values.items():
            if not isinstance(name, str) or not isinstance(count, int) or isinstance(count, bool):
                continue
            name_key = pickle.dumps(name, protocol=pickle.HIGHEST_PROTOCOL)
            row = store.conn.execute(
                "SELECT total FROM temp.prepared_merge_count WHERE event_type = ? AND payload_key = ? AND name = ?",
                (event.event_type, payload_key, name_key),
            ).fetchone()
            if row is None:
                store.conn.execute(
                    "INSERT INTO temp.prepared_merge_count VALUES (?, ?, ?, ?)",
                    (event.event_type, payload_key, name_key, str(count)),
                )
            else:
                store.conn.execute(
                    "UPDATE temp.prepared_merge_count SET total = ? "
                    "WHERE event_type = ? AND payload_key = ? AND name = ?",
                    (str(int(row[0]) + count), event.event_type, payload_key, name_key),
                )


def _append_claude_summaries(
    store: SqliteMessageStore,
    events: SqliteSessionEventSink,
    *,
    timestamp: str | None,
    seen: set[str],
) -> None:
    for event_type, payload_keys, keep_empty_keys in _CLAUDE_SUMMARIES:
        if event_type not in seen:
            continue
        payload: dict[str, object] = {}
        for payload_key in payload_keys:
            totals = {
                pickle.loads(name): int(total)
                for name, total in store.conn.execute(
                    "SELECT name, total FROM temp.prepared_merge_count WHERE event_type = ? AND payload_key = ?",
                    (event_type, payload_key),
                )
            }
            if totals or keep_empty_keys:
                payload[payload_key] = dict(sorted(totals.items()))
        events.append(ParsedSessionEvent(event_type=event_type, timestamp=timestamp, payload=payload))


def _single_prepared_session(artifact: PreparedJsonl) -> ParsedSession:
    with closing(artifact.iter_sessions()) as iterator:
        session = next(iterator, None)
        if session is None or next(iterator, None) is not None:
            raise ValueError("each retained revision must prepare exactly one session")
    return session


class _CohortOwnerOccurrences:
    """Bind acquired original owners to semantic digest occurrences."""

    def __init__(self, ordered: Sequence[tuple[str, PreparedJsonl]], store: SqliteMessageStore) -> None:
        self.conn = store.conn
        self.conn.execute(
            "CREATE TEMP TABLE cohort_owner_source (stable TEXT NOT NULL, raw_id TEXT NOT NULL, count INTEGER NOT NULL, PRIMARY KEY(stable,raw_id)) WITHOUT ROWID"
        )
        self.conn.execute(
            "CREATE TEMP TABLE cohort_owner_count (digest TEXT PRIMARY KEY, count INTEGER NOT NULL) WITHOUT ROWID"
        )
        self.conn.execute(
            "CREATE TEMP TABLE cohort_owner_placed (position INTEGER NOT NULL, variant INTEGER NOT NULL, stable TEXT NOT NULL, PRIMARY KEY(position,variant)) WITHOUT ROWID"
        )
        for raw_id, artifact in ordered:
            session = _single_prepared_session(artifact)
            with settled_iterator(session.messages) as messages:
                for message in messages:
                    check_compute_cancelled()
                    owner = message.owner_coordinate
                    if owner is not None and owner.stable_key is not None:
                        self.conn.execute(
                            "INSERT INTO cohort_owner_source VALUES (?,?,1) ON CONFLICT(stable,raw_id) DO UPDATE SET count=count+1",
                            (owner.stable_key, raw_id),
                        )

    def message(self, raw_id: str, message: ParsedMessage) -> ParsedMessage:
        digest = message_content_identity(message)
        row = self.conn.execute("SELECT count FROM cohort_owner_count WHERE digest=?", (digest,)).fetchone()
        occurrence = int(row[0]) if row is not None else 0
        self.conn.execute(
            "INSERT INTO cohort_owner_count VALUES (?,1) ON CONFLICT(digest) DO UPDATE SET count=count+1", (digest,)
        )
        owner = message.owner_coordinate
        if owner is None or owner.stable_key is None or owner.physical_key is None:
            return message
        row = self.conn.execute(
            "SELECT COUNT(*),MIN(count),MAX(count),SUM(raw_id=?) FROM cohort_owner_source WHERE stable=?",
            (raw_id, owner.stable_key),
        ).fetchone()
        assert row is not None
        if int(row[0]) < 2 or row[1] != 1 or row[2] != 1:
            return message
        if row[3] != 1:
            raise ValueError("cohort owner lacks its exact acquired origin")
        stable = f"content-occurrence:{digest}:{occurrence}"
        self.conn.execute("INSERT INTO cohort_owner_placed VALUES (?,?,?)", (*owner.physical_key, stable))
        return message.model_copy(update={"owner_coordinate": replace(owner, stable_key=stable)})

    def owner(self, owner: MessageOwnerCoordinate | None) -> MessageOwnerCoordinate | None:
        if owner is None or owner.physical_key is None:
            return owner
        row = self.conn.execute(
            "SELECT stable FROM cohort_owner_placed WHERE position=? AND variant=?", owner.physical_key
        ).fetchone()
        return owner if row is None else replace(owner, stable_key=str(row[0]))


def _merge_into_store(
    ordered: Sequence[tuple[str, PreparedJsonl]], merged: ParsedSession, store: SqliteMessageStore
) -> ParsedSession:
    messages = store.new_sink()
    events = store.new_event_sink()
    owner_occurrences = _CohortOwnerOccurrences(ordered, store)
    claude_summaries = {event_type: payload_keys for event_type, payload_keys, _ in _CLAUDE_SUMMARIES}
    seen: set[str] = set()
    if merged.source_name is Provider.CLAUDE_CODE:
        store.conn.execute(
            "CREATE TEMP TABLE prepared_merge_count ("
            "event_type TEXT NOT NULL, payload_key TEXT NOT NULL, name BLOB NOT NULL, total TEXT NOT NULL, "
            "PRIMARY KEY (event_type, payload_key, name)) WITHOUT ROWID"
        )
    attachments = store.new_attachment_sink()
    store.conn.execute(
        "CREATE TABLE prepared_attachment_origin (session_ordinal INTEGER NOT NULL, "
        "attachment_ordinal INTEGER NOT NULL, raw_id TEXT NOT NULL, original_session_ordinal INTEGER NOT NULL, "
        "original_attachment_ordinal INTEGER NOT NULL, PRIMARY KEY(session_ordinal,attachment_ordinal)) WITHOUT ROWID"
    )
    store.conn.execute(
        "CREATE TABLE prepared_attachment_origin_artifact (ordinal INTEGER PRIMARY KEY, raw_id TEXT NOT NULL UNIQUE, "
        "source_hash TEXT NOT NULL, sessions_path TEXT NOT NULL, sessions_hash TEXT NOT NULL, shard_hash TEXT NOT NULL)"
    )
    for raw_ordinal, (raw_id, artifact) in enumerate(ordered):
        artifact.verify_files(full=False)
        if artifact.sessions_seal is None or artifact.shard_seal is None or artifact.sessions_path is None:
            raise ValueError("attachment origin lacks original file seals")
        store.conn.execute(
            "INSERT INTO prepared_attachment_origin_artifact VALUES (?, ?, ?, ?, ?, ?)",
            (
                raw_ordinal,
                raw_id,
                artifact.blob_hash,
                str(artifact.sessions_path),
                artifact.sessions_seal.sha256,
                artifact.shard_seal.sha256,
            ),
        )
        check_compute_cancelled()
        session = _single_prepared_session(artifact)
        positions = ChunkPositions(session.messages, len(messages), conn=store.conn)
        with settled_iterator(session.messages) as _original_messages:
            for ordinal, message in enumerate(_original_messages):
                check_compute_cancelled()
                messages.append(owner_occurrences.message(raw_id, positions.message(message, ordinal)))
        original_attachments = session.attachments
        if (
            not isinstance(original_attachments, SqliteAttachmentSink)
            or original_attachments.path != artifact.sessions_path
        ):
            raise ValueError("aggregate attachments lack their original sealed row carrier")
        with settled_iterator(original_attachments) as _original_attachments:
            for original_ordinal, attachment in enumerate(_original_attachments):
                check_compute_cancelled()
                aggregate_ordinal = len(attachments)
                placed = positions.attachment(attachment)
                attachments.append(
                    placed.model_copy(update={"owner_coordinate": owner_occurrences.owner(placed.owner_coordinate)})
                )
                store.conn.execute(
                    "INSERT INTO prepared_attachment_origin VALUES (?, ?, ?, ?, ?)",
                    (
                        attachments.session_ordinal,
                        aggregate_ordinal,
                        raw_id,
                        original_attachments.session_ordinal,
                        original_ordinal,
                    ),
                )
        with settled_iterator(session.session_events) as _original_events:
            for event in _original_events:
                check_compute_cancelled()
                event = positions.event(event)
                event = event.model_copy(update={"owner_coordinate": owner_occurrences.owner(event.owner_coordinate)})
                if merged.source_name is Provider.CLAUDE_CODE and event.event_type in claude_summaries:
                    seen.add(event.event_type)
                    _count_claude_summary(store, event, claude_summaries[event.event_type])
                else:
                    events.append(event)
    active_leaf_id = messages[-1].provider_message_id if messages else None
    if messages:
        messages[-1] = messages[-1].model_copy(update={"is_active_leaf": True})
    if merged.source_name is Provider.CLAUDE_CODE:
        _append_claude_summaries(store, events, timestamp=merged.updated_at, seen=seen)

    return merged.model_copy(
        update={
            "messages": messages,
            "attachments": attachments,
            "session_events": events,
            "active_leaf_message_provider_id": active_leaf_id,
        }
    )


def prepare_retained_cohort_artifact(ordered: Sequence[tuple[str, PreparedJsonl]], directory: Path) -> PreparedJsonl:
    """Seal one bounded full-replay artifact from accepted raw revisions."""
    if len(ordered) < 2:
        raise ValueError("retained cohort needs at least two revisions")
    source_hash = prepared_cohort_source_hash(ordered)
    merged_metadata: ParsedSession | None = None
    identity: tuple[Provider, str] | None = None
    # Metadata merges without events, but Claude Code orders its working
    # directories by the relocation events; carry those in record order.
    relocated_cwds: list[str] = []
    for _, artifact in ordered:
        check_compute_cancelled()
        session = _single_prepared_session(artifact)
        candidate = (session.source_name, session.provider_session_id)
        if identity is not None and candidate != identity:
            raise ValueError("retained revisions disagree on provider-native session identity")
        identity = candidate
        if session.source_name is Provider.CLAUDE_CODE:
            relocated_cwds.extend(relocated_cwds_of(session.session_events))
        metadata_only = session.model_copy(update={"messages": [], "session_events": [], "attachments": []})
        merged_metadata = (
            metadata_only
            if merged_metadata is None
            else merge_parsed_session_chunks([merged_metadata, metadata_only])[0]
        )
    assert merged_metadata is not None
    if merged_metadata.source_name is Provider.CLAUDE_CODE:
        merged_metadata = merged_metadata.model_copy(
            update={
                "working_directories": order_working_directories(merged_metadata.working_directories, relocated_cwds)
            }
        )

    directory.mkdir(parents=True, exist_ok=True)
    store_path = directory / f"prepared-cohort-{uuid.uuid4().hex}.db"
    store: SqliteMessageStore | None = None
    shard_path: Path | None = None
    sealed = False
    try:
        store = SqliteMessageStore(store_path)
        merged = _merge_into_store(ordered, merged_metadata, store)
        merged.content_hash = session_content_hash(merged)
        shard_path = prepare_session_shard(directory, [merged]).path
        _write_artifact(store, source_hash, [merged], enrichment_digest=None, enrichment_index_path=None)
        store.close()
        store = None
        artifact = PreparedJsonl.seal(source_hash, store_path, shard_path)
        sealed = True
        return artifact
    finally:
        if store is not None:
            store.close()
        if not sealed:
            PreparedJsonl(source_hash, store_path, shard_path).discard()


def aggregate_attachment_blobs(
    aggregate: PreparedJsonl,
    *,
    source_read: BlobPublicationSourceRead,
    session_id: str,
    original_artifacts: Mapping[str, PreparedJsonl],
    accepted_raw_ids: Sequence[str],
) -> Mapping[object, tuple[bytes | None, int, str]]:
    """Borrow original live publication claims through sealed aggregate origins."""
    raw_ids, ordinal = _aggregate_attachment_binding(
        aggregate,
        source_read=source_read,
        session_id=session_id,
        original_artifacts=original_artifacts,
        accepted_raw_ids=accepted_raw_ids,
    )
    return _AggregateAttachmentBlobs(aggregate, source_read, session_id, original_artifacts, raw_ids, ordinal)


def aggregate_resident_attachment_blobs(
    aggregate: PreparedJsonl,
    *,
    source_read: RetainedAttachmentSourceRead,
    session_id: str,
    original_artifacts: Mapping[str, PreparedJsonl],
    accepted_raw_ids: Sequence[str],
) -> Mapping[object, tuple[bytes | None, int, str]]:
    """Bind each sealed origin to its actual current Raw's durable attachment reference."""
    raw_ids, ordinal = _aggregate_attachment_binding(
        aggregate,
        source_read=source_read,
        session_id=session_id,
        original_artifacts=original_artifacts,
        accepted_raw_ids=accepted_raw_ids,
    )
    return _AggregateResidentAttachmentBlobs(aggregate, source_read, session_id, original_artifacts, raw_ids, ordinal)


def _aggregate_attachment_binding(
    aggregate: PreparedJsonl,
    *,
    source_read: BlobPublicationSourceRead,
    session_id: str,
    original_artifacts: Mapping[str, PreparedJsonl],
    accepted_raw_ids: Sequence[str],
) -> tuple[frozenset[str], int]:
    """Borrow original acquired claims through the aggregate's exact sealed row origins."""
    if aggregate.sessions_path is None:
        raise ValueError("aggregate has no sealed attachment carrier")
    aggregate.verify_files(full=False)
    try:
        ordered = tuple((raw_id, original_artifacts[raw_id]) for raw_id in accepted_raw_ids)
    except KeyError as missing:
        raise ValueError("aggregate attachment origin lacks an accepted original artifact") from missing
    if len(set(accepted_raw_ids)) != len(accepted_raw_ids) or aggregate.blob_hash != prepared_cohort_source_hash(
        ordered
    ):
        raise ValueError("aggregate attachment origins differ from accepted Raw order")
    with _prepared_reader(aggregate.sessions_path) as connection:
        with connection_cursor(
            connection,
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('prepared_attachment_origin','prepared_attachment_origin_artifact') ORDER BY name",
        ) as cursor:
            names = cursor.fetchall()
        if names != [("prepared_attachment_origin",), ("prepared_attachment_origin_artifact",)]:
            raise ValueError("aggregate attachment origin representation is absent")
        with connection_cursor(
            connection,
            "SELECT ordinal,raw_id,source_hash,sessions_path,sessions_hash,shard_hash "
            "FROM prepared_attachment_origin_artifact ORDER BY ordinal",
        ) as cursor:
            count = 0
            while rows := cursor.fetchmany(256):
                check_compute_cancelled()
                for ordinal, raw_id, source_hash, path, sessions_hash, shard_hash in rows:
                    if count >= len(ordered) or ordinal != count or raw_id != ordered[count][0]:
                        raise ValueError("aggregate attachment origin is outside accepted Raw order")
                    original = ordered[count][1]
                    original.verify_files(full=False)
                    if (
                        original.sessions_seal is None
                        or original.shard_seal is None
                        or (source_hash, path, sessions_hash, shard_hash)
                        != (
                            original.blob_hash,
                            str(original.sessions_path),
                            original.sessions_seal.sha256,
                            original.shard_seal.sha256,
                        )
                    ):
                        raise ValueError("aggregate attachment origin differs from original postimage")
                    count += 1
            if count != len(ordered):
                raise ValueError("aggregate attachment origin inventory is incomplete")
        with connection_cursor(
            connection, "SELECT attachment_ordinal FROM prepared_session WHERE session_id=? LIMIT 2", (session_id,)
        ) as cursor:
            rows = cursor.fetchall()
        if len(rows) != 1:
            raise ValueError("aggregate attachment session is ambiguous or absent")
    return frozenset(accepted_raw_ids), int(rows[0][0])


class _AggregateAttachmentBlobs(Mapping[object, tuple[bytes | None, int, str]]):
    """A bounded borrowed view; original artifacts and Source custody stay with the caller."""

    def __init__(
        self,
        aggregate: PreparedJsonl,
        source_read: BlobPublicationSourceRead,
        session_id: str,
        originals: Mapping[str, PreparedJsonl],
        raw_ids: frozenset[str],
        ordinal: int,
    ) -> None:
        self.aggregate = aggregate
        self.source_read = source_read
        self.session_id = session_id
        self.originals = originals
        self.raw_ids = raw_ids
        self.ordinal = ordinal

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __iter__(self) -> Iterator[object]:
        self.aggregate.verify_files(full=False)
        if self.aggregate.sessions_path is None:
            raise ValueError("aggregate attachment carrier is absent")
        after = -1
        while True:
            with (
                _prepared_reader(self.aggregate.sessions_path) as connection,
                connection_cursor(
                    connection,
                    "SELECT attachment_ordinal FROM prepared_attachment "
                    "WHERE session_ordinal=? AND attachment_ordinal>? ORDER BY attachment_ordinal LIMIT 256",
                    (self.ordinal, after),
                ) as cursor,
            ):
                rows = cursor.fetchall()
            if not rows:
                return
            for (ordinal,) in rows:
                check_compute_cancelled()
                key = (str(self.aggregate.sessions_path), self.ordinal, int(ordinal))
                try:
                    self[key]
                except KeyError:
                    continue
                yield key
            after = int(rows[-1][0])

    def __getitem__(self, key: object) -> tuple[bytes | None, int, str]:
        if (
            not isinstance(key, tuple)
            or len(key) != 3
            or key[0] != str(self.aggregate.sessions_path)
            or not isinstance(key[1], int)
            or isinstance(key[1], bool)
            or key[1] != self.ordinal
            or not isinstance(key[2], int)
            or isinstance(key[2], bool)
        ):
            raise KeyError(key)
        self.aggregate.verify_files(full=False)
        if self.aggregate.sessions_path is None:
            raise ValueError("aggregate attachment carrier is absent")
        with (
            _prepared_reader(self.aggregate.sessions_path) as connection,
            connection_cursor(
                connection,
                "SELECT o.raw_id,o.original_session_ordinal,o.original_attachment_ordinal, r.source_hash,r.sessions_path,r.sessions_hash,r.shard_hash "
                "FROM prepared_attachment a LEFT JOIN prepared_attachment_origin o "
                "ON o.session_ordinal=a.session_ordinal AND o.attachment_ordinal=a.attachment_ordinal "
                "LEFT JOIN prepared_attachment_origin_artifact r ON r.raw_id=o.raw_id WHERE a.session_ordinal=? AND a.attachment_ordinal=?",
                (key[1], key[2]),
            ) as cursor,
        ):
            row = cursor.fetchone()
        if row is None:
            raise KeyError(key)
        raw_id, session_ordinal, attachment_ordinal, source_hash, path, sessions_hash, shard_hash = row
        if raw_id not in self.raw_ids or session_ordinal is None or attachment_ordinal is None:
            raise ValueError("aggregate attachment row has no accepted original Raw origin")
        try:
            original = self.originals[raw_id]
        except KeyError as missing:
            raise ValueError("aggregate attachment origin lost its accepted original artifact") from missing
        original.verify_files(full=False)
        if (
            original.sessions_seal is None
            or original.shard_seal is None
            or (source_hash, path, sessions_hash, shard_hash)
            != (
                original.blob_hash,
                str(original.sessions_path),
                original.sessions_seal.sha256,
                original.shard_seal.sha256,
            )
        ):
            raise ValueError("aggregate attachment origin differs from original postimage")
        if original.sessions_path is None:
            raise ValueError("aggregate attachment origin carrier is absent")
        with (
            _prepared_reader(original.sessions_path) as connection,
            connection_cursor(
                connection,
                "SELECT 1 FROM prepared_attachment a JOIN prepared_session s "
                "ON s.attachment_ordinal=a.session_ordinal WHERE s.session_id=? "
                "AND a.session_ordinal=? AND a.attachment_ordinal=? LIMIT 2",
                (self.session_id, session_ordinal, attachment_ordinal),
            ) as cursor,
        ):
            original_rows = cursor.fetchall()
        if len(original_rows) != 1:
            raise ValueError("aggregate attachment origin is missing or belongs to another session")
        original_key = (str(original.sessions_path), int(session_ordinal), int(attachment_ordinal))
        return self._original_attachment_blobs(original, raw_id)[original_key]

    def _original_attachment_blobs(
        self, original: PreparedJsonl, raw_id: str
    ) -> Mapping[object, tuple[bytes | None, int, str]]:
        return original.attachment_blobs(source_read=self.source_read, session_id=self.session_id)


class _AggregateResidentAttachmentBlobs(_AggregateAttachmentBlobs):
    def __init__(
        self,
        aggregate: PreparedJsonl,
        source_read: RetainedAttachmentSourceRead,
        session_id: str,
        originals: Mapping[str, PreparedJsonl],
        raw_ids: frozenset[str],
        ordinal: int,
    ) -> None:
        super().__init__(aggregate, source_read, session_id, originals, raw_ids, ordinal)
        self.retained_read = source_read

    def _original_attachment_blobs(
        self, original: PreparedJsonl, raw_id: str
    ) -> Mapping[object, tuple[bytes | None, int, str]]:
        return original.resident_attachment_blobs(
            source_read=self.retained_read, session_id=self.session_id, raw_id=raw_id
        )


__all__ = [
    "aggregate_attachment_blobs",
    "aggregate_resident_attachment_blobs",
    "prepare_retained_cohort_artifact",
    "prepared_cohort_source_hash",
]
