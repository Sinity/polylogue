"""Bounded worker-side composition of retained JSONL revision artifacts."""

from __future__ import annotations

import hashlib
import pickle
import uuid
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.dispatch import merge_parsed_session_chunks
from polylogue.sources.parsers.base import ParsedSession, ParsedSessionEvent
from polylogue.sources.prepared_jsonl import PreparedJsonl, _write_artifact
from polylogue.sources.prepared_message_sink import SqliteMessageStore, SqliteSessionEventSink
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


def _merge_into_store(
    ordered: Sequence[tuple[str, PreparedJsonl]], merged: ParsedSession, store: SqliteMessageStore
) -> ParsedSession:
    messages = store.new_sink()
    events = store.new_event_sink()
    claude_summaries = {event_type: payload_keys for event_type, payload_keys, _ in _CLAUDE_SUMMARIES}
    seen: set[str] = set()
    if merged.source_name is Provider.CLAUDE_CODE:
        store.conn.execute(
            "CREATE TEMP TABLE prepared_merge_count ("
            "event_type TEXT NOT NULL, payload_key TEXT NOT NULL, name BLOB NOT NULL, total TEXT NOT NULL, "
            "PRIMARY KEY (event_type, payload_key, name)) WITHOUT ROWID"
        )
    for _, artifact in ordered:
        session = _single_prepared_session(artifact)
        for message in session.messages:
            messages.append(message.model_copy(update={"position": len(messages), "is_active_leaf": False}))
        for event in session.session_events:
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
    attachments = []
    identity: tuple[Provider, str] | None = None
    for _, artifact in ordered:
        session = _single_prepared_session(artifact)
        candidate = (session.source_name, session.provider_session_id)
        if identity is not None and candidate != identity:
            raise ValueError("retained revisions disagree on provider-native session identity")
        identity = candidate
        attachments.extend(session.attachments)
        metadata_only = session.model_copy(update={"messages": [], "session_events": [], "attachments": []})
        merged_metadata = (
            metadata_only
            if merged_metadata is None
            else merge_parsed_session_chunks([merged_metadata, metadata_only])[0]
        )
    assert merged_metadata is not None
    merged_metadata = merged_metadata.model_copy(update={"attachments": attachments})

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


__all__ = ["prepare_retained_cohort_artifact", "prepared_cohort_source_hash"]
