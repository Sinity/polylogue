"""Memory-release helpers for large ingest result payloads."""

from __future__ import annotations

from builtins import BaseExceptionGroup

from polylogue.pipeline.services.ingest_worker import IngestRecordResult, SessionWritePayload
from polylogue.sources.prepared_message_sink import SqliteMessageSink, SqliteSessionEventSink

INGEST_RELEASE_BLOB_MB_THRESHOLD = 16.0
INGEST_RELEASE_MESSAGE_THRESHOLD = 1_000
INGEST_RELEASE_ROW_THRESHOLD = 10_000


def _session_payload_row_count(cdata: SessionWritePayload) -> int:
    return cdata.message_count + cdata.attachment_count


def ingest_result_needs_memory_release(ir: IngestRecordResult) -> bool:
    if ir.serialized_size_bytes is not None:
        return ir.serialized_size_bytes >= int(INGEST_RELEASE_BLOB_MB_THRESHOLD * 1024 * 1024)
    total_messages = sum(cdata.message_count for cdata in ir.sessions)
    if total_messages >= INGEST_RELEASE_MESSAGE_THRESHOLD:
        return True
    total_rows = sum(_session_payload_row_count(cdata) for cdata in ir.sessions)
    return total_rows >= INGEST_RELEASE_ROW_THRESHOLD


def discard_session_data_payload(cdata: SessionWritePayload) -> None:
    if cdata.prepared_write is not None:
        cdata.prepared_write.close()
        cdata.prepared_write = None
    cdata.prepared_rows = None
    if not isinstance(cdata.parsed_session.messages, SqliteMessageSink):
        cdata.parsed_session.messages.clear()
    cdata.parsed_session.attachments.clear()
    if not isinstance(cdata.parsed_session.session_events, SqliteSessionEventSink):
        cdata.parsed_session.session_events.clear()


def discard_ingest_result_payload(ir: IngestRecordResult) -> None:
    failures: list[BaseException] = []
    for payload in ir.sessions:
        if payload.prepared_write is not None:
            try:
                payload.prepared_write.close()
            except BaseException as failure:
                failures.append(failure)
            else:
                payload.prepared_write = None
    if failures:
        # The artifact contains the files those native carriers still borrow.
        # Keep it on this result until the physical worker's existing cleanup
        # retry closes every carrier. Independent carrier closes above still
        # receive their own attempt before this dependent deletion is deferred.
        raise BaseExceptionGroup("ingest preparation cleanup remains unsettled", failures)
    if ir.prepared_artifact is not None:
        ir.prepared_artifact.discard()
        ir.prepared_artifact = None
    ir.sessions.clear()


__all__ = [
    "INGEST_RELEASE_BLOB_MB_THRESHOLD",
    "INGEST_RELEASE_MESSAGE_THRESHOLD",
    "discard_session_data_payload",
    "discard_ingest_result_payload",
    "ingest_result_needs_memory_release",
]
